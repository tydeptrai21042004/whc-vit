# Fair ViT Comparison Protocol

The purpose of `run_fair_vit_comparison.py` is to compare **DT1D-Adapter** with VPT, Pfeiffer Adapter, Full fine-tuning and Linear probing without using the test set for model selection.

## Shared budget

Every method receives the same:

- dataset and train/validation/test split;
- ViT-B/16 pretrained checkpoint;
- input resolution;
- batch size within a comparison;
- 10-epoch paper budget;
- cosine scheduler family;
- weight decay `1e-4`;
- tune seed `42`;
- table mode: at least three final seeds (default `0,1,2`);
- figure mode: exactly one representative final seed (default `0`);
- number of LR candidates: **10**;
- best-validation checkpoint rule;
- one final test evaluation per final seed.

## Method-faithful optimization

Equal comparison does not require forcing all algorithms to use the wrong optimizer. The runner preserves the optimizer family used by the corresponding implementation while matching the number of validation trials.

| Method | Optimizer | LR interpretation |
|---|---|---|
| DT1D-Adapter | AdamW | effective adapter LR grid |
| VPT | SGD + momentum | original nominal LR scaled by `batch_size/256` |
| Pfeiffer Adapter | AdamW | effective adapter LR grid |
| Full fine-tuning | AdamW | source-style nominal LR scaled by `batch_size/256` |
| Linear probing | SGD + momentum | original nominal LR scaled by `batch_size/256` |

Each row receives 10 candidate LRs. `verify_fair_protocol.py` fails if the candidate counts differ.

## Selection rule

During tuning:

```text
DATA.NO_TEST = True
```

For each method, validation performance selects the LR. Final seeds are then trained under the selected method configuration, the best-validation checkpoint is restored, and test is evaluated exactly once for that seed.

The primary manuscript metric should therefore be **Test Acc1 at best validation**, summarized as mean ± sample standard deviation over at least three final seeds. A training-based figure is explicitly single-seed and must not display a multi-seed standard deviation.

## Proposal identity

Main experiments expose one proposal key only:

```text
dt1d = DT1D-Adapter (R124-P2-G16-Axis-LearnedGate)
```

Component variants in `configs/ablations/dt1d_components_vit.yaml` are reviewer controls and must not be presented as competing proposal methods.

## Registered paper datasets

- `flowers102`
- `vtab-caltech101`
- `vtab-dtd`
- `vtab-eurosat`

DTD and EuroSAT are registered in source code directly; supporting Kaggle runs no longer modify `run_fair_vit_comparison.py` at runtime.

## Provenance and inference timing

Each final run records the repository version, Git commit when available, and proposal-contract SHA256. DT1D paper configs enable fused-kernel caching in evaluation mode so latency measurement does not repeatedly rebuild an unchanged kernel. This cache is execution-only and does not alter the mathematical proposal or predictions.
