# Supporting ViT Runs

Two supplied standalone experiment definitions have been integrated into the repository as first-class Kaggle cells.

## V01 — VTAB-DTD

```text
kaggle_cells/V01_vtab_dtd_vitb16_fair.sh
```

- dataset: VTAB-DTD
- backbone: ViT-B/16
- resolution: 224
- batch size: 32
- methods: `dt1d,vpt,pfeiffer,full,linear`
- tuning: 10 LR candidates per method on seed 42
- final seeds: 0,1,2
- test rule: once at the best-validation checkpoint
- output archive: `V01_results.zip`

## V02 — VTAB-EuroSAT

```text
kaggle_cells/V02_vtab_eurosat_vitb16_fair.sh
```

- dataset: VTAB-EuroSAT
- backbone: ViT-B/16
- resolution: 224
- batch size: 32
- methods: `dt1d,vpt,pfeiffer,full,linear`
- tuning: 10 LR candidates per method on seed 42
- final seeds: 0,1,2
- test rule: once at the best-validation checkpoint
- output archive: `V02_results.zip`

## Important

The supplied materials define and launch the supporting runs; they do not contain completed numeric result tables. Therefore this repository stores the corrected reproducible run cells and result-packaging logic, but does not invent accuracy numbers. Once a session completes, its cell writes `SESSION_STATUS.json`, run metadata/aggregates and a compact result ZIP.
