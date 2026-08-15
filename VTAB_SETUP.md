# VTAB Dataset Setup

The current paper runs use VTAB-style 1k adaptation splits for Caltech101, DTD and EuroSAT. `run_fair_vit_comparison.py` keeps the adaptation train/validation split separated from the official test set.

## Required TFDS datasets

```python
import tensorflow_datasets as tfds

DATA_DIR = "/path/to/tfds"

for spec in (
    "caltech101:3.*.*",
    "dtd:3.*.*",
    "eurosat/rgb:2.*.*",
):
    builder = tfds.builder(spec, data_dir=DATA_DIR)
    builder.download_and_prepare()
    print("ready:", builder.info.full_name)
```

For the supplied support sessions:

| Session | Dataset | TFDS source | Classes | Batch |
|---|---|---|---:|---:|
| V01 | VTAB-DTD | `dtd:3.*.*` | 47 | 32 |
| V02 | VTAB-EuroSAT | `eurosat/rgb:2.*.*` | 10 | 32 |

Both use ViT-B/16 at 224×224, VPT length 10, tune seed 42 and final seeds 0/1/2.

## Pretrained ViT checkpoint

The Kaggle cells use the original ViT-B/16 224 checkpoint:

```text
ViT-B_16-224.npz
```

The cells validate the downloaded NPZ before launching training.

## Split rule

For VTAB datasets the paper protocol is:

```text
train800 -> training
val200   -> validation / LR and checkpoint selection
official test -> evaluated only after selection
```

Do not merge validation into training for the reported fair comparison because the test result must correspond to a validation-selected configuration under the same rule for all methods.
