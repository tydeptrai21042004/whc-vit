#!/usr/bin/env python3
"""Dependency-light structural validation for canonical DT1D-Adapter on ViT-B/16."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models.vit_adapter.dt1d_adapter import DT1DAdapter, DT1DTokenAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    dim, grid, blocks = 768, (14, 14), 12
    token_adapter = DT1DTokenAdapter(dim, grid)
    m = token_adapter.spatial_adapter

    torch.manual_seed(0)
    with torch.no_grad():
        m.base_coefficients.normal_(0, 0.1)
        m.detail_coefficients.normal_(0, 0.1)
    k = m.build_kernels(torch.device("cpu"), torch.float32)
    joint = k.squeeze(2).abs().sum(-1).sum(0)

    x = torch.randn(2, 197, dim, requires_grad=True)
    y = token_adapter(x)
    y.square().mean().backward()

    per_block = sum(p.numel() for p in token_adapter.parameters())
    result = {
        "proposal": m.proposal_name,
        "architecture": m.architecture_name,
        "base_support": list(m.active_offsets),
        "shift_p": m.shift_p,
        "base_kernel_size": m.base_kernel_size,
        "effective_kernel_size": m.effective_kernel_size,
        "gate_mode": m.gate_mode,
        "gate_value": float(m.gate.detach()),
        "gate_is_trainable": isinstance(m.gate, torch.nn.Parameter),
        "lambda_mode": m.shift_lambda_mode,
        "lambda_scope": m.shift_lambda_scope,
        "lambda_init": m.shift_lambda(torch.device("cpu"), torch.float32).detach().tolist(),
        "joint_l1_max": float(joint.max().detach()),
        "lambda_gradient_abs_sum": float(m.shift_theta.grad.abs().sum()),
        "gate_gradient_abs": float(m.gate.grad.abs()),
        "class_token_preserved": bool(torch.allclose(y[:, :1], x[:, :1])),
        "adapter_params_per_block": per_block,
        "adapter_params_12_blocks": per_block * blocks,
        "axial_macs_224": blocks * 2 * dim * grid[0] * grid[1] * m.effective_kernel_size,
        "convolution_calls_per_block": m.convolution_calls_per_forward,
    }

    assert result["proposal"] == "DT1D-Adapter"
    assert result["architecture"] == "R124-P2-G16-Axis-LearnedGate"
    assert result["base_support"] == [1, 2, 4]
    assert result["shift_p"] == 2
    assert result["effective_kernel_size"] == 13
    assert result["gate_mode"] == "learned"
    assert result["gate_is_trainable"] is True
    assert result["lambda_mode"] == "learned"
    assert result["lambda_scope"] == "axis"
    assert result["joint_l1_max"] <= 1.000001
    assert result["lambda_gradient_abs_sum"] > 0
    assert result["gate_gradient_abs"] > 0
    assert result["class_token_preserved"]

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
