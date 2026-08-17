#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="${REPO_URL:-https://github.com/tydeptrai21042004/whc-vit.git}"
REPO_COMMIT="${REPO_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/whc-vit-fair-flowers"
DOWNLOAD_ROOT="$WORKDIR/flowers_download"
MODEL_ROOT="$WORKDIR/vit_weights"
OUTPUT_ROOT="$WORKDIR/flowers102_vitb16_fair"
RESULT_ZIP="$WORKDIR/flowers102_vitb16_fair.zip"

rm -rf "$REPO_DIR"
git clone --depth 1 "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
if [[ -n "$REPO_COMMIT" ]]; then
  git fetch --depth 1 origin "$REPO_COMMIT"
  git checkout --detach "$REPO_COMMIT"
fi

echo "SOURCE COMMIT: $(git rev-parse HEAD)"

# Keep Kaggle's CUDA-enabled torch/torchvision; install only missing support packages.
python -m pip install -q --upgrade-strategy only-if-needed \
  scipy scikit-learn pandas Pillow fvcore iopath yacs simplejson termcolor \
  tabulate tqdm ml-collections 'timm>=1.0.0,<2' PyYAML

python validate_dt1d_vit.py
python verify_vpt_original.py
python verify_fair_protocol.py
python -m pytest -q tests/test_dt1d_token_adapter.py tests/test_fair_protocol.py tests/test_repository_contracts.py

mkdir -p "$DOWNLOAD_ROOT" "$MODEL_ROOT"
WEIGHT_FILE="$MODEL_ROOT/ViT-B_16-224.npz"
if [[ ! -s "$WEIGHT_FILE" ]]; then
  curl -L --fail --retry 5 --retry-delay 5 \
    "https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz" \
    -o "$WEIGHT_FILE"
fi
python - "$WEIGHT_FILE" <<'PY'
import sys, numpy as np
z=np.load(sys.argv[1])
assert z.files, "invalid ViT checkpoint"
print("ViT checkpoint tensors:", len(z.files))
PY

FLOWERS_PATH="$(python - "$DOWNLOAD_ROOT" <<'PY'
import json, sys
from pathlib import Path
from torchvision.datasets import Flowers102
root=Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
datasets={s: Flowers102(root=str(root), split=s, download=True) for s in ('train','val','test')}
image_dir=root/'flowers-102'/'jpg'
assert image_dir.exists(), image_dir
for split, ds in datasets.items():
    mapping={Path(str(p)).name:int(y) for p,y in zip(ds._image_files, ds._labels)}
    (image_dir/f'{split}.json').write_text(json.dumps(mapping, indent=2), encoding='utf-8')
    print(f'{split}: {len(mapping)}', file=sys.stderr)
print(image_dir)
PY
)"

test -f "$FLOWERS_PATH/train.json"
test -f "$FLOWERS_PATH/val.json"
test -f "$FLOWERS_PATH/test.json"

# Fairness rule: same data/splits/backbone/batch/resolution/epochs/scheduler/WD,
# same number of LR-tuning trials, same tuning seed and final seeds. Optimizer
# and LR scale remain method-faithful: original VPT/Linear use SGD+momentum and
# the original nominal-LR * batch_size/256 rule; Full FT uses AdamW; DT1D and
# Pfeiffer use AdamW. Test is disabled during tuning and is evaluated once per
# final seed after restoring that seed's best-validation checkpoint.
python run_fair_vit_comparison.py \
  --dataset flowers102 \
  --data-path "$FLOWERS_PATH" \
  --model-root "$MODEL_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --batch-sizes 32,16 \
  --methods dt1d,vpt,pfeiffer,full,linear \
  --epochs 10 \
  --warmup-epoch 1 \
  --vpt-tokens 5 \
  --gpus 0,1

rm -f "$RESULT_ZIP"
cd "$WORKDIR"
zip -qr "$RESULT_ZIP" "$(basename "$OUTPUT_ROOT")"
echo "RESULT ZIP: $RESULT_ZIP"
ls -lh "$RESULT_ZIP"
