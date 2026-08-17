from pathlib import Path
from types import SimpleNamespace


def _args(tmp_path, dataset="vtab-dtd"):
    import run_fair_vit_comparison as fair
    kw = dict(
        dataset=dataset,
        data_path=str(tmp_path / "data"),
        model_root=str(tmp_path / "weights"),
        output_root=str(tmp_path / "out"),
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
    for m in fair.METHOD_ORDER:
        kw[f"{m}_lr_grid"] = None
    return SimpleNamespace(**kw)


def test_exact_method_set_preserves_all_real_baselines():
    import run_fair_vit_comparison as fair
    assert fair.METHOD_ORDER == ("dt1d", "vpt", "pfeiffer", "full", "linear")
    assert fair.DISPLAY["vpt"] == "VPT"
    assert fair.DISPLAY["pfeiffer"] == "Pfeiffer Adapter"
    assert fair.DISPLAY["full"] == "Full fine-tuning"
    assert fair.DISPLAY["linear"] == "Linear probing"


def test_support_datasets_are_first_class_not_runtime_patches():
    import run_fair_vit_comparison as fair
    assert fair.DATASETS["vtab-dtd"]["classes"] == 47
    assert fair.DATASETS["vtab-eurosat"]["classes"] == 10
    assert fair.DATASETS["vtab-dtd"]["vpt_tokens"] == 10
    assert fair.DATASETS["vtab-eurosat"]["vpt_tokens"] == 10


def test_every_method_gets_same_ten_trial_budget(tmp_path):
    import run_fair_vit_comparison as fair
    args = _args(tmp_path)
    grids = {m: fair.resolve_method_lr_grid(args, m, 32) for m in fair.METHOD_ORDER}
    assert {len(g) for g in grids.values()} == {10}
    assert fair.method_optimizer("vpt") == "sgd"
    assert fair.method_optimizer("linear") == "sgd"
    assert fair.method_optimizer("full") == "adamw"
    assert fair.method_optimizer("dt1d") == "adamw"
    assert fair.method_optimizer("pfeiffer") == "adamw"


def test_dt1d_generated_fragment_is_canonical(tmp_path):
    import run_fair_vit_comparison as fair
    d = fair.method_fragment("dt1d", 10, 16)["MODEL"]["ADAPTER"]["DT1D"]
    assert d["GROUP_SIZE"] == 16
    assert d["SHIFT_P"] == 2
    assert d["SHIFT_LAMBDA_MODE"] == "learned"
    assert d["SHIFT_LAMBDA_SCOPE"] == "axis"
    assert d["GATE_MODE"] == "learned"
    assert d["GATE_INIT"] == 0.01
    assert d["PROJECT_L1"] is True
    assert d["USE_POINTWISE"] is False


def test_tuning_configs_disable_test(tmp_path):
    import run_fair_vit_comparison as fair
    args = _args(tmp_path)
    for method in fair.METHOD_ORDER:
        lr = fair.resolve_method_lr_grid(args, method, 32)[0]
        cfg = fair.make_config(args, 32, method, lr, "tune")
        assert cfg["DATA"]["NO_TEST"] is True
        assert cfg["SOLVER"]["TOTAL_EPOCH"] == 10
        assert cfg["SOLVER"]["WEIGHT_DECAY"] == 1e-4


def test_final_configs_enable_test_once_after_tuning(tmp_path):
    import run_fair_vit_comparison as fair
    args = _args(tmp_path)
    cfg = fair.make_config(args, 32, "dt1d", 1e-3, "final")
    assert cfg["DATA"]["NO_TEST"] is False


def test_lr_selection_uses_validation_only():
    import run_fair_vit_comparison as fair
    records = [
        (1e-4, {"best_val_top1": 0.60, "test_top1": 0.99}),
        (1e-3, {"best_val_top1": 0.80, "test_top1": 0.10}),
        (5e-3, {"best_val_top1": 0.70, "test_top1": 1.00}),
    ]
    lr, val = fair.select_lr(records)
    assert lr == 1e-3
    assert val == 0.80


def test_vpt_source_files_are_untouched():
    import run_fair_vit_comparison as fair
    audit = fair.verify_vpt_source_fidelity(Path(__file__).resolve().parents[1])
    assert audit["status"] == "PASS"


def test_publication_seed_modes_are_fail_closed():
    import run_fair_vit_comparison as fair
    assert fair.resolve_final_seeds(None, "table") == [0, 1, 2]
    assert fair.resolve_final_seeds("0,1,2,3", "table") == [0, 1, 2, 3]
    assert fair.resolve_final_seeds(None, "figure") == [0]
    assert fair.resolve_final_seeds("7", "figure") == [7]
    import pytest
    with pytest.raises(SystemExit): fair.resolve_final_seeds("0,1", "table")
    with pytest.raises(SystemExit): fair.resolve_final_seeds("0,1", "figure")
    with pytest.raises(SystemExit): fair.resolve_final_seeds("0,0,1", "table")

def test_result_csv_names_preserve_support_run_contract():
    import run_fair_vit_comparison as fair
    assert fair.result_csv_name("vtab-eurosat", "table", [0,1,2]).endswith("_fair_three_seed.csv")
    assert fair.result_csv_name("vtab-eurosat", "figure", [0]).endswith("_fair_single_seed.csv")
