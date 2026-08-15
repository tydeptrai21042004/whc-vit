from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_no_legacy_proposal_files_or_aliases():
    forbidden_files = [
        ROOT / "src/models/vit_adapter/whc_compact_dt1d_adapter.py",
        ROOT / "validate_whc_p2_vit.py",
        ROOT / "run_whc_p2_vit_ablation.py",
        ROOT / "configs/ablations/whc_p2_fixed_gate_vit.yaml",
        ROOT / "configs/finetune/flowers_whc_dt1d.yaml",
        ROOT / "configs/vtab/caltech101_whc_dt1d.yaml",
        ROOT / "tune_vtab.py",
        ROOT / "tune_fgvc.py",
    ]
    assert not [p for p in forbidden_files if p.exists()]


def test_no_whc_or_previous_dt1d_method_names_in_owned_code():
    paths = [
        ROOT / "run_fair_vit_comparison.py",
        ROOT / "train.py",
        ROOT / "src/configs/config.py",
        ROOT / "src/models/vit_adapter/dt1d_adapter.py",
        ROOT / "src/models/vit_adapter/vit.py",
        ROOT / "src/models/vit_adapter/adapter_block.py",
    ]
    text = "\n".join(p.read_text(encoding="utf-8") for p in paths)
    lowered = text.lower()
    assert "whc_dt1d" not in lowered
    assert "whc-compact" not in lowered
    assert "previous dt1d" not in lowered


def test_canonical_proposal_configs_exist_for_all_four_datasets():
    expected = {
        "configs/finetune/flowers_dt1d.yaml": "OxfordFlowers",
        "configs/vtab/caltech101_dt1d.yaml": "vtab-caltech101",
        "configs/vtab/dtd_dt1d.yaml": "vtab-dtd",
        "configs/vtab/eurosat_dt1d.yaml": "vtab-eurosat",
    }
    for rel, dataset in expected.items():
        data = yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))
        assert data["DATA"]["NAME"] == dataset
        assert data["MODEL"]["ADAPTER"]["NAME"] == "DT1D"
        d = data["MODEL"]["ADAPTER"]["DT1D"]
        assert d["ACTIVE_OFFSETS"] == "1,2,4"
        assert d["SHIFT_P"] == 2
        assert d["SHIFT_LAMBDA_SCOPE"] == "axis"
        assert d["GATE_MODE"] == "learned"


def test_baseline_source_files_are_preserved():
    for rel in [
        "src/models/vit_prompt/vit.py",
        "src/models/vit_prompt/vit_ablations.py",
        "src/models/build_vit_backbone.py",
        "src/models/build_model.py",
        "src/models/vit_adapter/adapter_block.py",
        "configs/base-prompt.yaml",
        "configs/base-linear.yaml",
        "configs/base-finetune.yaml",
    ]:
        assert (ROOT / rel).is_file(), rel


def test_support_run_cells_exist():
    assert (ROOT / "kaggle_cells/V01_vtab_dtd_vitb16_fair.sh").is_file()
    assert (ROOT / "kaggle_cells/V02_vtab_eurosat_vitb16_fair.sh").is_file()


def test_documentation_surface_is_minimal_and_current():
    docs = sorted(p.name for p in ROOT.glob("*.md"))
    assert docs == [
        "FAIR_COMPARISON.md",
        "README.md",
        "SUPPORT_RUNS.md",
        "VPT_SOURCE_FIDELITY.md",
        "VTAB_SETUP.md",
    ]
    text = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in docs).lower()
    assert "whc_dt1d" not in text
    assert "previous dt1d" not in text


def test_dt1d_direct_configs_do_not_carry_pfeiffer_only_fields():
    paths = [
        ROOT / "configs/finetune/flowers_dt1d.yaml",
        ROOT / "configs/vtab/caltech101_dt1d.yaml",
        ROOT / "configs/vtab/dtd_dt1d.yaml",
        ROOT / "configs/vtab/eurosat_dt1d.yaml",
    ]
    for path in paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        adapter = cfg["MODEL"]["ADAPTER"]
        assert set(adapter) == {"NAME", "DT1D"}


def test_support_cells_use_canonical_method_set_without_runtime_source_patching():
    expected = "dt1d,vpt,pfeiffer,full,linear"
    cells = {
        "V01_vtab_dtd_vitb16_fair.sh": "vtab-dtd",
        "V02_vtab_eurosat_vitb16_fair.sh": "vtab-eurosat",
    }
    for name, dataset in cells.items():
        text = (ROOT / "kaggle_cells" / name).read_text(encoding="utf-8")
        assert f'DATASET="{dataset}"' in text
        assert f'METHODS="{expected}"' in text
        assert "PYPATCH" not in text
        assert "validate_dt1d_vit.py" in text
        assert "verify_fair_protocol.py" in text
        assert "SESSION_STATUS.json" in text


def test_all_kaggle_cells_have_valid_shell_syntax():
    import subprocess
    cells = sorted((ROOT / "kaggle_cells").glob("*.sh"))
    assert len(cells) == 4
    for path in cells:
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_ablation_manifest_changes_only_dt1d_component_keys():
    path = ROOT / "configs/ablations/dt1d_components_vit.yaml"
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert manifest["proposal"] == "DT1D-Adapter"
    assert manifest["architecture"] == "R124-P2-G16-Axis-LearnedGate"
    variants = manifest["variants"]
    assert len(variants) == 15
    assert "dt1d_final" in variants and variants["dt1d_final"] == {}
    for name, overrides in variants.items():
        for key in overrides:
            assert key.startswith("MODEL.ADAPTER.DT1D."), (name, key)
