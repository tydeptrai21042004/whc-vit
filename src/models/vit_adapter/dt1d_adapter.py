"""Canonical DT1D-Adapter for ViT patch-token grids.

The public proposal is frozen to the same architecture used by the CNN code:

    architecture            = R124-P2-G16-Axis-LearnedGate
    axes                    = H + W
    group size              = 16
    symmetric support       = {1, 2, 4}
    channel detail          = normalized psi4
    weighted shift p        = 2
    shift lambda            = learned independently for H and W
    lambda init / bound     = 0 / [-0.5, 0.5]
    joint H/W L1 cap        = 1
    residual gate           = learned, initialized at 0.01
    pointwise mixing        = off
    padding                 = replicate
    effective kernel        = 13
    depthwise conv calls    = 2

ViT patch tokens are reshaped to a 2-D grid, filtered, and restored to token
order. Prefix/class tokens are never filtered.

The constructor keeps a small set of explicit switches only for reviewer
ablations. Normal experiment configs must use the frozen defaults above.
"""
from __future__ import annotations

import math
from math import gcd
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_VALID_OFFSETS = (1, 2, 4)
_GATE_MODES = {"learned", "fixed"}
_LAMBDA_MODES = {"learned", "fixed", "off"}
_LAMBDA_SCOPES = {"block", "axis"}
_SHIFT_NORMALIZATIONS = {"mean", "paper"}


def _parse_offsets(value: str | Sequence[int] | None) -> Tuple[int, ...]:
    if value is None:
        return _VALID_OFFSETS
    if isinstance(value, str):
        raw = [
            int(v)
            for v in value.replace(";", ",").replace(" ", ",").split(",")
            if v.strip()
        ]
    else:
        raw = [int(v) for v in value]
    unknown = sorted(set(raw) - set(_VALID_OFFSETS))
    if unknown:
        raise ValueError(
            f"DT1D-Adapter supports offsets {_VALID_OFFSETS}; got unsupported {unknown}"
        )
    offsets = tuple(v for v in _VALID_OFFSETS if v in set(raw))
    if not offsets:
        raise ValueError(f"active_offsets must contain at least one of {_VALID_OFFSETS}")
    return offsets


class DT1DAdapter(nn.Module):
    """Weighted axial DT1D operator for BCHW feature maps."""

    method_name = "DT1D-Adapter"
    proposal_name = "DT1D-Adapter"
    architecture_name = "R124-P2-G16-Axis-LearnedGate"

    def __init__(
        self,
        C: int,
        *,
        axis: str = "hw",
        group_size: int = 16,
        residual_scale: float = 1.0,
        gate_init: float = 0.01,
        gate_mode: str = "learned",
        padding_mode: str = "replicate",
        contrast_split: int = 8,
        detail_components: str = "offset4",
        active_offsets: str | Sequence[int] | None = "1,2,4",
        use_pointwise: bool = False,
        pointwise_ratio: int = 32,
        pointwise_groups: int = 4,
        use_bn: bool = False,
        cache_kernel: bool = False,
        project_l1: bool = True,
        shift_p: int = 2,
        shift_lambda_mode: str = "learned",
        shift_lambda_scope: str = "axis",
        shift_lambda_init: float = 0.0,
        shift_lambda_max: float = 0.5,
        shift_normalization: str = "mean",
    ) -> None:
        super().__init__()
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        if axis not in {"h", "w", "hw"}:
            raise ValueError(f"axis must be h, w, or hw, got {axis!r}")
        if padding_mode not in {"reflect", "replicate", "zeros", "constant"}:
            raise ValueError(f"Unsupported padding_mode={padding_mode!r}")
        if detail_components not in {"offset4", "none"}:
            raise ValueError("detail_components must be 'offset4' or 'none'")
        if gate_mode not in _GATE_MODES:
            raise ValueError(f"gate_mode must be one of {_GATE_MODES}")
        if shift_lambda_mode not in _LAMBDA_MODES:
            raise ValueError(f"shift_lambda_mode must be one of {_LAMBDA_MODES}")
        if shift_lambda_scope not in _LAMBDA_SCOPES:
            raise ValueError(f"shift_lambda_scope must be one of {_LAMBDA_SCOPES}")
        if shift_normalization not in _SHIFT_NORMALIZATIONS:
            raise ValueError(
                f"shift_normalization must be one of {_SHIFT_NORMALIZATIONS}"
            )

        offsets = _parse_offsets(active_offsets)
        p = int(shift_p)
        if p <= 0:
            raise ValueError(f"shift_p must be positive, got {shift_p!r}")
        lambda_max = float(shift_lambda_max)
        lambda_init = float(shift_lambda_init)
        if not math.isfinite(lambda_max) or lambda_max <= 0:
            raise ValueError("shift_lambda_max must be finite and > 0")
        if abs(lambda_init) > lambda_max + 1e-12:
            raise ValueError(
                f"|shift_lambda_init| must be <= {lambda_max}, got {lambda_init}"
            )
        if shift_lambda_mode == "off" and abs(lambda_init) > 1e-12:
            raise ValueError("shift_lambda_init must be 0 when shift_lambda_mode='off'")

        self.C = int(C)
        self.axis = str(axis)
        self.axis_names = tuple(a for a in ("h", "w") if a in self.axis)
        self.num_axes = len(self.axis_names)
        self.group_size = max(1, int(group_size))
        self.num_groups = math.ceil(self.C / self.group_size)
        self.residual_scale = float(residual_scale)
        self.padding_mode = "constant" if padding_mode == "zeros" else padding_mode
        self.contrast_split = max(1, int(contrast_split))
        self.detail_components = str(detail_components)
        self.active_offsets = offsets
        self.base_offsets = (0,) + self.active_offsets
        self.gate_mode = str(gate_mode)
        self.cache_kernel = bool(cache_kernel)
        self.project_l1 = bool(project_l1)
        self.use_pointwise = bool(use_pointwise)
        self.shift_p = p
        self.shift_lambda_mode = str(shift_lambda_mode)
        self.shift_lambda_scope = str(shift_lambda_scope)
        self.shift_lambda_max = lambda_max
        self.shift_normalization = str(shift_normalization)

        self.base_coefficients = nn.Parameter(
            torch.zeros(self.num_axes, self.num_groups, len(self.base_offsets))
        )
        with torch.no_grad():
            self.base_coefficients[..., 0].fill_(1.0 / max(1, self.num_axes))

        detail_count = 1 if self.detail_components == "offset4" else 0
        self.detail_coefficients = nn.Parameter(
            torch.zeros(self.num_axes, self.num_groups, detail_count)
        )
        contrast, valid = self._make_channel_contrast()
        self.register_buffer("channel_contrast", contrast, persistent=True)
        self.register_buffer("valid_contrast_group", valid, persistent=True)
        psi4 = self._normalized_psi4()
        self.register_buffer("psi4", psi4, persistent=True)

        if self.gate_mode == "learned":
            self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        else:
            self.register_buffer(
                "gate", torch.tensor(float(gate_init)), persistent=True
            )

        scalar_count = 1 if self.shift_lambda_scope == "block" else self.num_axes
        if self.shift_lambda_mode == "learned":
            ratio = max(-0.999999, min(0.999999, lambda_init / lambda_max))
            theta_init = math.atanh(ratio)
            self.shift_theta = nn.Parameter(torch.full((scalar_count,), theta_init))
        else:
            fixed = lambda_init if self.shift_lambda_mode == "fixed" else 0.0
            self.register_buffer(
                "shift_lambda_fixed",
                torch.full((scalar_count,), fixed, dtype=torch.float32),
                persistent=True,
            )

        if self.use_pointwise:
            hidden = max(1, self.C // max(1, int(pointwise_ratio)))
            groups = max(1, min(int(pointwise_groups), self.C, hidden))
            groups = gcd(groups, self.C)
            groups = gcd(groups, hidden) or 1
            self.pointwise = nn.Sequential(
                nn.Conv2d(self.C, hidden, 1, groups=groups, bias=False),
                nn.BatchNorm2d(hidden) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, self.C, 1, groups=groups, bias=False),
                nn.BatchNorm2d(self.C) if use_bn else nn.Identity(),
            )
        else:
            self.pointwise = nn.Identity()

        self.base_radius = max(max(self.active_offsets), 4 if detail_count else 0)
        self.base_kernel_size = 2 * self.base_radius + 1
        self.weighting_active = not (
            self.shift_lambda_mode == "off"
            or (
                self.shift_lambda_mode == "fixed"
                and abs(lambda_init) <= 1e-12
            )
        )
        self.effective_radius = self.base_radius + (
            self.shift_p if self.weighting_active else 0
        )
        self.effective_kernel_size = 2 * self.effective_radius + 1

        self.register_buffer("_cached_kernels", torch.empty(0), persistent=False)
        self.is_dt1d_adapter = True
        self.is_canonical_dt1d_adapter = (
            self.axis == "hw"
            and self.group_size == 16
            and self.active_offsets == (1, 2, 4)
            and self.detail_components == "offset4"
            and self.shift_p == 2
            and self.shift_lambda_mode == "learned"
            and self.shift_lambda_scope == "axis"
            and abs(lambda_init) <= 1e-12
            and abs(self.shift_lambda_max - 0.5) <= 1e-12
            and self.shift_normalization == "mean"
            and self.project_l1
            and self.gate_mode == "learned"
            and abs(float(gate_init) - 0.01) <= 1e-12
            and not self.use_pointwise
            and self.padding_mode == "replicate"
        )
        self.implementation = "dt1d_r124_p2_g16_axis_learnedgate"

    def _make_channel_contrast(self) -> Tuple[torch.Tensor, torch.Tensor]:
        contrast = torch.zeros(self.C, dtype=torch.float32)
        valid = torch.zeros(self.num_groups, dtype=torch.float32)
        start = 0
        for group in range(self.num_groups):
            n = min(self.group_size, self.C - start)
            n1 = min(self.contrast_split, max(1, n // 2))
            n2 = n - n1
            if n1 > 0 and n2 > 0:
                pos = math.sqrt(n2 / (n1 * (n1 + n2)))
                neg = -math.sqrt(n1 / (n2 * (n1 + n2)))
                contrast[start : start + n1] = pos
                contrast[start + n1 : start + n] = neg
                valid[group] = 1.0
            start += n
        return contrast, valid

    @staticmethod
    def _normalized_psi4() -> torch.Tensor:
        atom = torch.zeros(9, dtype=torch.float64)
        atom[0] = 1.0
        atom[4] = -2.0
        atom[8] = 1.0
        return (atom / torch.linalg.vector_norm(atom)).float()

    def _group_index(self, device: torch.device) -> torch.Tensor:
        return (
            torch.arange(self.C, device=device) // self.group_size
        ).clamp_max(self.num_groups - 1)

    def _build_base_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        group_idx = self._group_index(device)
        beta = self.base_coefficients.to(device=device, dtype=dtype)[:, group_idx, :]
        kernel = torch.zeros(
            self.num_axes,
            self.C,
            self.base_kernel_size,
            device=device,
            dtype=dtype,
        )
        center = self.base_radius
        kernel[..., center] = beta[..., 0]
        for coefficient, offset in enumerate(self.active_offsets, start=1):
            kernel[..., center - offset] = beta[..., coefficient]
            kernel[..., center + offset] = beta[..., coefficient]

        if self.detail_coefficients.shape[-1]:
            valid = self.valid_contrast_group.to(device=device, dtype=dtype)
            eta = self.detail_coefficients.to(device=device, dtype=dtype)
            eta = eta * valid.view(1, self.num_groups, 1)
            eta_channel = eta[:, group_idx, 0]
            contrast = self.channel_contrast.to(device=device, dtype=dtype)
            psi4 = self.psi4.to(device=device, dtype=dtype)
            kernel = kernel + (
                eta_channel * contrast.view(1, self.C)
            ).unsqueeze(-1) * psi4.view(1, 1, -1)
        return kernel

    def shift_lambda(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.shift_lambda_mode == "learned":
            value = self.shift_lambda_max * torch.tanh(
                self.shift_theta.to(device=device, dtype=dtype)
            )
        else:
            value = self.shift_lambda_fixed.to(device=device, dtype=dtype)
        if self.shift_lambda_scope == "block":
            value = value.expand(self.num_axes)
        return value

    @staticmethod
    def _shift_pair(base: torch.Tensor, p: int, scale: float) -> torch.Tensor:
        k = int(base.shape[-1])
        out = base.new_zeros(*base.shape[:-1], k + 2 * p)
        out[..., :k] += base
        out[..., 2 * p : 2 * p + k] += base
        return out * float(scale)

    def build_unprojected_kernels(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        base = self._build_base_kernel(device, dtype)
        if not self.weighting_active:
            return base.unsqueeze(2)
        p = self.shift_p
        centered = F.pad(base, (p, p))
        scale = 0.5 if self.shift_normalization == "mean" else 1.0
        shifted = self._shift_pair(base, p, scale)
        lam = self.shift_lambda(device, dtype).view(self.num_axes, 1, 1)
        return ((1.0 - lam) * centered + lam * shifted).unsqueeze(2)

    def build_kernels(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        project: bool | None = None,
    ) -> torch.Tensor:
        if project is None:
            project = self.project_l1
        kernel = self.build_unprojected_kernels(device, dtype)
        if project:
            joint_l1 = kernel.abs().sum(dim=-1).sum(dim=0).squeeze(-1)
            scale = torch.maximum(joint_l1, torch.ones_like(joint_l1))
            kernel = kernel / scale.view(1, self.C, 1, 1)
        if int(kernel.shape[-1]) != self.effective_kernel_size:
            raise RuntimeError(
                f"Expected K{self.effective_kernel_size}, got {tuple(kernel.shape)}"
            )
        return kernel

    def _pad(self, x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        if pad_h == 0 and pad_w == 0:
            return x
        if self.padding_mode == "constant":
            return F.pad(
                x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0.0
            )
        mode = self.padding_mode
        h, w = x.shape[-2:]
        if mode == "reflect" and (
            (pad_h >= h and pad_h > 0) or (pad_w >= w and pad_w > 0)
        ):
            mode = "replicate"
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=mode)

    def _conv_axis(
        self, x: torch.Tensor, axis_name: str, kernel: torch.Tensor
    ) -> torch.Tensor:
        k = int(kernel.shape[-1])
        radius = k // 2
        if axis_name == "h":
            x = self._pad(x, radius, 0)
            return F.conv2d(x, kernel.view(self.C, 1, k, 1), groups=self.C)
        if axis_name == "w":
            x = self._pad(x, 0, radius)
            return F.conv2d(x, kernel.view(self.C, 1, 1, k), groups=self.C)
        raise ValueError(f"Unknown axis {axis_name!r}")

    @torch.no_grad()
    def prepare_for_inference(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        ref = self.base_coefficients
        device = ref.device if device is None else device
        dtype = ref.dtype if dtype is None else dtype
        self._cached_kernels = self.build_kernels(device, dtype).detach()

    def clear_inference_cache(self) -> None:
        self._cached_kernels = torch.empty(0, device=self.base_coefficients.device)

    def train(self, mode: bool = True):
        if mode:
            self.clear_inference_cache()
        return super().train(mode)

    def _kernels_for(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache_kernel and not self.training:
            if (
                self._cached_kernels.numel()
                and self._cached_kernels.device == x.device
                and self._cached_kernels.dtype == x.dtype
            ):
                return self._cached_kernels
            kernels = self.build_kernels(x.device, x.dtype)
            self._cached_kernels = kernels.detach()
            return kernels
        return self.build_kernels(x.device, x.dtype)

    def parameter_count_breakdown(self) -> Dict[str, int]:
        base = self.base_coefficients.numel()
        detail = self.detail_coefficients.numel()
        gate = self.gate.numel() if isinstance(self.gate, nn.Parameter) else 0
        shift = self.shift_theta.numel() if hasattr(self, "shift_theta") else 0
        pointwise = sum(p.numel() for p in self.pointwise.parameters())
        return {
            "shared_base": int(base),
            "channel_correction": int(detail),
            "shift_weight": int(shift),
            "learned_gate": int(gate),
            "pointwise": int(pointwise),
            "total": int(base + detail + shift + gate + pointwise),
            "base_kernel_size": int(self.base_kernel_size),
            "effective_kernel_size": int(self.effective_kernel_size),
        }

    @property
    def convolution_calls_per_forward(self) -> int:
        return self.num_axes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DT1DAdapter expects BCHW input, got {tuple(x.shape)}")
        if x.shape[1] != self.C:
            raise ValueError(
                f"Channel mismatch: adapter C={self.C}, input C={x.shape[1]}"
            )
        kernels = self._kernels_for(x)
        response = torch.zeros_like(x)
        for axis_index, axis_name in enumerate(self.axis_names):
            response = response + self._conv_axis(
                x, axis_name, kernels[axis_index]
            )
        response = self.pointwise(response)
        gate = self.gate.to(dtype=x.dtype, device=x.device)
        return x + self.residual_scale * gate * response

    def extra_repr(self) -> str:
        return (
            f"C={self.C}, architecture={self.architecture_name}, axis={self.axis}, "
            f"group={self.group_size}, offsets={self.active_offsets}, "
            f"baseK={self.base_kernel_size}, p={self.shift_p}, "
            f"effectiveK={self.effective_kernel_size}, "
            f"lambda={self.shift_lambda_mode}-{self.shift_lambda_scope}, "
            f"lambda_max={self.shift_lambda_max}, project_l1={self.project_l1}, "
            f"gate={self.gate_mode}, pointwise={self.use_pointwise}, "
            f"padding={self.padding_mode}"
        )


class DT1DTokenAdapter(nn.Module):
    """Apply DT1D-Adapter to patch tokens while preserving prefix tokens."""

    def __init__(
        self,
        embed_dim: int,
        grid_size: Tuple[int, int],
        num_prefix_tokens: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.grid_size = (int(grid_size[0]), int(grid_size[1]))
        self.num_prefix_tokens = int(num_prefix_tokens)
        if min(self.grid_size) <= 0:
            raise ValueError(f"Invalid grid_size={grid_size}")
        if self.num_prefix_tokens < 0:
            raise ValueError("num_prefix_tokens must be non-negative")
        self.spatial_adapter = DT1DAdapter(C=self.embed_dim, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"DT1DTokenAdapter expects BND tokens, got {tuple(x.shape)}"
            )
        batch, tokens, dim = x.shape
        if dim != self.embed_dim:
            raise ValueError(f"Embedding mismatch: got {dim}, expected {self.embed_dim}")
        gh, gw = self.grid_size
        expected = self.num_prefix_tokens + gh * gw
        if tokens != expected:
            raise ValueError(
                f"N={tokens}, expected {expected} for grid {gh}x{gw} with "
                f"{self.num_prefix_tokens} prefix token(s)"
            )
        prefix = x[:, : self.num_prefix_tokens, :]
        patches = x[:, self.num_prefix_tokens :, :]
        fmap = patches.transpose(1, 2).reshape(batch, dim, gh, gw)
        fmap = self.spatial_adapter(fmap)
        patches = fmap.reshape(batch, dim, gh * gw).transpose(1, 2)
        return (
            torch.cat((prefix, patches), dim=1)
            if self.num_prefix_tokens
            else patches
        )

    def parameter_count_breakdown(self) -> Dict[str, int]:
        return self.spatial_adapter.parameter_count_breakdown()


__all__ = ["DT1DAdapter", "DT1DTokenAdapter"]
