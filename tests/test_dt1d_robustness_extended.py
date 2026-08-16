from __future__ import annotations

import io
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from proposal_contract import load_spec, proposal_fingerprint
from src.models.vit_adapter.dt1d_adapter import DT1DAdapter, DT1DTokenAdapter


@pytest.mark.parametrize("channels", [1, 2, 15, 16, 17, 24, 31, 32, 64, 768])
def test_canonical_parameter_count_formula_for_edge_channel_counts(channels):
    m = DT1DAdapter(channels)
    groups = math.ceil(channels / 16)
    expected = 10 * groups + 3
    assert sum(p.numel() for p in m.parameters()) == expected
    assert m.parameter_count_breakdown()["total"] == expected


def test_singleton_group_disables_detail_without_nan():
    m = DT1DAdapter(1)
    assert m.valid_contrast_group.tolist() == [0.0]
    assert torch.count_nonzero(m.channel_contrast).item() == 0
    with torch.no_grad():
        m.detail_coefficients.fill_(1000.0)
    k = m.build_kernels(torch.device("cpu"), torch.float32)
    assert torch.isfinite(k).all()


@pytest.mark.parametrize("channels", [17, 24, 31, 33])
def test_partial_groups_keep_balanced_zero_mean_contrast(channels):
    m = DT1DAdapter(channels)
    for group in range(m.num_groups):
        start = group * m.group_size
        end = min(channels, start + m.group_size)
        chunk = m.channel_contrast[start:end]
        if len(chunk) >= 2:
            assert m.valid_contrast_group[group].item() == 1.0
            assert abs(float(chunk.sum())) < 1e-6
            assert torch.count_nonzero(chunk).item() == len(chunk)
        else:
            assert m.valid_contrast_group[group].item() == 0.0


def test_weighted_shift_preserves_dc_sum_before_projection():
    torch.manual_seed(201)
    m = DT1DAdapter(16, project_l1=False)
    with torch.no_grad():
        m.base_coefficients.normal_(0.0, 0.2)
        m.detail_coefficients.normal_(0.0, 0.2)
        m.shift_theta[:] = torch.tensor([0.8, -0.7])
    base = m._build_base_kernel(torch.device("cpu"), torch.float32)
    shifted = m.build_unprojected_kernels(torch.device("cpu"), torch.float32).squeeze(2)
    assert torch.allclose(base.sum(-1), shifted.sum(-1), atol=1e-6, rtol=1e-6)


def test_extreme_trainable_values_still_produce_finite_l1_capped_kernels():
    m = DT1DAdapter(33)
    with torch.no_grad():
        m.base_coefficients.uniform_(-1e4, 1e4)
        m.detail_coefficients.uniform_(-1e4, 1e4)
        m.shift_theta[:] = torch.tensor([1e4, -1e4])
    k = m.build_kernels(torch.device("cpu"), torch.float32)
    mass = k.squeeze(2).abs().sum(-1).sum(0)
    assert torch.isfinite(k).all()
    assert torch.all(mass <= 1.00001)


@pytest.mark.parametrize("shape", [(1, 16, 1, 1), (1, 16, 1, 3), (1, 16, 3, 1), (2, 16, 2, 2)])
def test_tiny_spatial_inputs_are_supported(shape):
    m = DT1DAdapter(16)
    x = torch.randn(*shape)
    y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_float64_forward_backward_is_finite():
    torch.manual_seed(202)
    m = DT1DAdapter(16).double().train()
    x = torch.randn(1, 16, 5, 6, dtype=torch.float64, requires_grad=True)
    y = m(x)
    y.square().mean().backward()
    assert y.dtype == torch.float64
    assert torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in m.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_exactly_two_depthwise_convolutions_per_spatial_forward(monkeypatch):
    m = DT1DAdapter(8)
    x = torch.randn(1, 8, 7, 9)
    real_conv2d = F.conv2d
    calls = []

    def counted(*args, **kwargs):
        calls.append((args[1].shape, kwargs.get("groups")))
        return real_conv2d(*args, **kwargs)

    monkeypatch.setattr(F, "conv2d", counted)
    y = m(x)
    assert y.shape == x.shape
    assert len(calls) == 2
    assert all(groups == 8 for _, groups in calls)


def test_checkpoint_load_invalidates_cached_kernel_and_uses_new_weights():
    torch.manual_seed(203)
    cached = DT1DAdapter(16, cache_kernel=True).eval()
    x = torch.randn(1, 16, 7, 7)
    _ = cached(x)
    assert cached._cached_kernels.numel() > 0

    fresh = DT1DAdapter(16).eval()
    with torch.no_grad():
        fresh.base_coefficients.normal_(0.0, 0.3)
        fresh.detail_coefficients.normal_(0.0, 0.2)
        fresh.shift_theta[:] = torch.tensor([0.4, -0.3])
        fresh.gate.fill_(0.07)
    expected = fresh(x)
    cached.load_state_dict(fresh.state_dict())
    assert cached._cached_kernels.numel() == 0
    got = cached(x)
    assert torch.allclose(got, expected, atol=1e-7, rtol=1e-6)


def test_state_dict_binary_roundtrip_preserves_output():
    torch.manual_seed(204)
    a = DT1DAdapter(16).eval()
    with torch.no_grad():
        for p in a.parameters():
            p.normal_(0.0, 0.05)
    x = torch.randn(1, 16, 5, 5)
    expected = a(x)
    stream = io.BytesIO()
    torch.save(a.state_dict(), stream)
    stream.seek(0)
    b = DT1DAdapter(16).eval()
    b.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))
    assert torch.allclose(b(x), expected, atol=1e-7, rtol=1e-6)


def test_token_adapter_preserves_multiple_prefix_tokens_exactly():
    m = DT1DTokenAdapter(embed_dim=8, grid_size=(2, 3), num_prefix_tokens=2)
    x = torch.randn(2, 8, 8)
    y = m(x)
    assert y.shape == x.shape
    assert torch.equal(y[:, :2], x[:, :2])


@pytest.mark.parametrize("bad_input, message", [
    (torch.randn(2, 8, 8, 1), "BND"),
    (torch.randn(2, 7, 7), "Embedding mismatch"),
    (torch.randn(2, 6, 8), "expected"),
])
def test_token_adapter_rejects_invalid_token_shapes(bad_input, message):
    m = DT1DTokenAdapter(embed_dim=8, grid_size=(2, 3), num_prefix_tokens=1)
    with pytest.raises(ValueError, match=message):
        m(bad_input)


def _adapter_cfg(hidden=16):
    d = SimpleNamespace(
        AXIS="hw", GROUP_SIZE=16, ACTIVE_OFFSETS="1,2,4", DETAIL_COMPONENTS="offset4",
        CONTRAST_SPLIT=8, PROJECT_L1=True, GATE_MODE="learned", GATE_INIT=0.01,
        RESIDUAL_SCALE=1.0, PADDING="replicate", USE_POINTWISE=False,
        POINTWISE_RATIO=32, POINTWISE_GROUPS=4, USE_BN=False, CACHE_KERNEL=False,
        SHIFT_P=2, SHIFT_LAMBDA_MODE="learned", SHIFT_LAMBDA_SCOPE="axis",
        SHIFT_LAMBDA_INIT=0.0, SHIFT_LAMBDA_MAX=0.5, SHIFT_NORMALIZATION="mean",
    )
    return SimpleNamespace(NAME="DT1D", DT1D=d)


def test_dt1d_integrates_into_actual_transformer_block_and_backpropagates():
    pytest.importorskip("ml_collections")
    from src.models.vit_adapter.vit import ADPT_Block

    config = SimpleNamespace(
        hidden_size=16,
        transformer={"num_heads": 4, "mlp_dim": 32, "attention_dropout_rate": 0.0, "dropout_rate": 0.0},
    )
    block = ADPT_Block(config, vis=False, adapter_config=_adapter_cfg(), grid_size=(2, 2)).train()
    x = torch.randn(2, 5, 16, requires_grad=True)
    y, weights = block(x)
    assert y.shape == x.shape
    assert weights is None
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    s = block.token_adapter.spatial_adapter
    assert s.shift_theta.grad is not None and s.shift_theta.grad.abs().sum() > 0
    assert s.gate.grad is not None and s.gate.grad.abs() > 0


def test_runtime_model_matches_machine_readable_proposal_contract():
    spec = load_spec()
    m = DT1DAdapter(32)
    assert spec["proposal"] == m.proposal_name
    assert spec["architecture"] == m.architecture_name
    assert tuple(spec["axes"]) == m.axis_names
    assert tuple(spec["active_offsets"]) == m.active_offsets
    assert spec["group_size"] == m.group_size
    assert spec["base_kernel_size"] == m.base_kernel_size
    assert spec["effective_kernel_size"] == m.effective_kernel_size
    assert spec["depthwise_convolution_calls"] == m.convolution_calls_per_forward
    assert proposal_fingerprint() == "c5a9104df8cb882b75d6176c3730a18d0dd61c2645f4000e654b62e561f14093"


@pytest.mark.parametrize("kwargs", [
    {"shift_lambda_init": float("nan")},
    {"shift_lambda_init": float("inf")},
    {"shift_lambda_max": float("nan")},
    {"shift_lambda_max": float("inf")},
])
def test_nonfinite_shift_hyperparameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DT1DAdapter(16, **kwargs)
