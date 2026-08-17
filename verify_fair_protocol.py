#!/usr/bin/env python3
"""Dependency-light structural checks for the fair ViT comparison protocol."""
from pathlib import Path
from types import SimpleNamespace
import yaml

import run_fair_vit_comparison as fair
from src.models.vit_adapter.dt1d_adapter import DT1DAdapter


def main():
    assert fair.METHOD_ORDER == ("dt1d", "vpt", "pfeiffer", "full", "linear")
    assert {"vtab-dtd", "vtab-eurosat"}.issubset(fair.DATASETS)

    model = DT1DAdapter(32)
    assert model.architecture_name == "R124-P2-G16-Axis-LearnedGate"
    assert model.active_offsets == (1, 2, 4)
    assert model.shift_p == 2
    assert model.shift_lambda_mode == "learned"
    assert model.shift_lambda_scope == "axis"
    assert model.gate_mode == "learned"
    assert model.project_l1 is True
    assert model.use_pointwise is False

    root = Path(__file__).resolve().parent
    proposal_configs = [
        root / "configs/finetune/flowers_dt1d.yaml",
        root / "configs/vtab/caltech101_dt1d.yaml",
        root / "configs/vtab/dtd_dt1d.yaml",
        root / "configs/vtab/eurosat_dt1d.yaml",
    ]
    for path in proposal_configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        d = cfg["MODEL"]["ADAPTER"]["DT1D"]
        assert cfg["MODEL"]["ADAPTER"]["NAME"] == "DT1D"
        assert d["ACTIVE_OFFSETS"] == "1,2,4"
        assert d["SHIFT_P"] == 2
        assert d["SHIFT_LAMBDA_MODE"] == "learned"
        assert d["SHIFT_LAMBDA_SCOPE"] == "axis"
        assert d["GATE_MODE"] == "learned"
        assert d["GATE_INIT"] == 0.01
        assert d["PROJECT_L1"] is True
        assert d["USE_POINTWISE"] is False

    args = SimpleNamespace(
        dataset="vtab-dtd",
        data_path="/tmp/data",
        model_root="/tmp/models",
        output_root="/tmp/out",
        resolution=224,
        num_workers=0,
        weight_decay=1e-4,
        warmup_epoch=1,
        epochs=10,
        patience=20,
        log_every=10,
        vpt_tokens=10,
        pfeiffer_reduction=16,
        lr_grid=None,
    )
    for method in fair.METHOD_ORDER:
        setattr(args, f"{method}_lr_grid", None)

    assert fair.resolve_final_seeds(None, "table") == [0, 1, 2]
    assert fair.resolve_final_seeds("0,1,2,3", "table") == [0, 1, 2, 3]
    assert fair.resolve_final_seeds(None, "figure") == [0]
    assert fair.resolve_final_seeds("7", "figure") == [7]

    grids = {m: fair.resolve_method_lr_grid(args, m, 32) for m in fair.METHOD_ORDER}
    assert {len(x) for x in grids.values()} == {10}
    for method in fair.METHOD_ORDER:
        cfg = fair.make_config(args, 32, method, grids[method][0], "tune")
        assert cfg["DATA"]["NO_TEST"] is True
        assert cfg["SOLVER"]["TOTAL_EPOCH"] == 10

    dt = fair.method_fragment("dt1d", 10, 16)["MODEL"]["ADAPTER"]["DT1D"]
    assert dt["SHIFT_P"] == 2
    assert dt["SHIFT_LAMBDA_SCOPE"] == "axis"
    assert dt["GATE_MODE"] == "learned"
    print("FAIR PROTOCOL PASS")


if __name__ == "__main__":
    main()
