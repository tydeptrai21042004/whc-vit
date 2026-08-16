# DT1D-Adapter for Vision Transformers

This repository contains the ViT implementation and fair comparison protocol for a **single proposal method**:

**DT1D-Adapter — R124-P2-G16-Axis-LearnedGate**

The proposal reshapes ViT patch tokens to a 2D grid, applies a compact two-axis DT1D spatial adapter, and restores the token sequence without modifying prefix/class tokens.

## Canonical proposal

| Property | Value |
|---|---|
| Public method key | `dt1d` |
| Adapter name | `DT1D` |
| Axes | H + W |
| Group size | 16 |
| Base support | `{1,2,4}` |
| Detail term | normalized `psi4` / offset-4 detail |
| Shift | `p=2` |
| Shift coefficient | learned |
| Coefficient scope | one coefficient per axis |
| Coefficient init / bound | `0`, bounded by `±0.5` |
| Joint H/W L1 projection | enabled |
| Residual gate | learned, initialized at `0.01` |
| Pointwise mixing | disabled |
| Padding | replicate |
| Effective axial kernel | 13 |
| Axial depthwise calls | 2 |
| Eval kernel cache | enabled in paper configs; execution-only optimization |

The canonical implementation is `src/models/vit_adapter/dt1d_adapter.py`. Main paper experiments must not alter the architecture above. Component changes belong only in `configs/ablations/dt1d_components_vit.yaml`.


## Cross-repository proposal lock and release provenance

`proposal_spec.json` is byte-for-byte the same method contract shipped by `tydeptrai21042004/whc-dt1d`. The model implementation remains separate for ViT token integration, but the frozen operator settings are checked against this contract.

```bash
python proposal_fingerprint.py
python proposal_fingerprint.py --compare /path/to/whc-dt1d
python check_environment.py
```

Final run summaries and fair-protocol manifests record the repository version, Git commit when available, and the proposal-contract SHA256. `CITATION.cff`, `codemeta.json`, `.zenodo.json`, and `environment.yml` are included for a versioned archival release. Eval-time fused-kernel caching does not change the proposal, parameter count, or numerical operator; it only avoids rebuilding an unchanged effective kernel on repeated inference calls.

## Preserved baselines

The proposal cleanup does **not** remove the paper baselines:

- VPT
- Pfeiffer Adapter
- Full fine-tuning
- Linear probing

`run_fair_vit_comparison.py` exposes exactly:

```text
dt1d,vpt,pfeiffer,full,linear
```

The original VPT implementation/build files and base VPT/linear/full configurations are hash-checked before a fair run. See `VPT_SOURCE_FIDELITY.md`.

## Fair comparison protocol

For each dataset and batch size:

1. every method receives exactly 10 validation-only LR trials;
2. the same dataset split, ViT-B/16 checkpoint, resolution, epoch budget, scheduler family, weight decay, tune seed and final seeds are used;
3. optimizer family/LR scale remain method-faithful (VPT/Linear use SGD+momentum; Full FT uses AdamW; DT1D/Pfeiffer use AdamW);
4. `DATA.NO_TEST=True` during hyperparameter tuning;
5. the selected LR is chosen from validation results only;
6. final runs use seeds `0,1,2` and evaluate test once after restoring the best-validation checkpoint.

See `FAIR_COMPARISON.md` for details.

## Self-contained DT1D configs

```text
configs/finetune/flowers_dt1d.yaml
configs/vtab/caltech101_dt1d.yaml
configs/vtab/dtd_dt1d.yaml
configs/vtab/eurosat_dt1d.yaml
configs/ablations/dt1d_components_vit.yaml
```

These direct configs expose the same canonical proposal fields as the fair runner. The fair paper numbers should be produced by `run_fair_vit_comparison.py`, not by comparing hand-selected learning rates from separate historical runs.

## Supporting Kaggle runs

```text
kaggle_cells/flowers102_vitb16_fair.sh
kaggle_cells/vtab_caltech101_vitb16_fair.sh
kaggle_cells/V01_vtab_dtd_vitb16_fair.sh
kaggle_cells/V02_vtab_eurosat_vitb16_fair.sh
```

The V01 and V02 cells correspond to the supplied DTD and EuroSAT supporting experiment definitions. They now use the canonical `dt1d` method directly; no runtime source patching is required. See `SUPPORT_RUNS.md`.

## Validate before training

```bash
python validate_dt1d_vit.py
python verify_vpt_original.py
python verify_fair_protocol.py
python -m pytest -q tests
```

## Dry-run a fair comparison

```bash
python run_fair_vit_comparison.py \
  --dataset vtab-dtd \
  --data-path /path/to/tfds \
  --model-root /path/to/vit_weights \
  --output-root /tmp/dt1d_dtd \
  --batch-sizes 32 \
  --methods dt1d,vpt,pfeiffer,full,linear \
  --epochs 10 \
  --seeds 0,1,2 \
  --tune-seed 42 \
  --dry-run \
  --gpus cpu
```

## Reviewer ablations

```bash
python run_dt1d_vit_ablation.py --dry-run
```

Ablation rows are reviewer controls, not additional proposal methods. They isolate shift weighting, coefficient sharing, axis choice, support, L1 projection, residual gate, detail term and optional pointwise mixing.

## Dataset setup

See `VTAB_SETUP.md`. The fair runner directly registers Caltech101, DTD and EuroSAT; the Kaggle cells download the exact required TFDS dataset before training.
