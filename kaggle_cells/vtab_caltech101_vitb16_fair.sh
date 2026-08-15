#!/usr/bin/env bash
set -Eeuo pipefail

REPO_URL="${REPO_URL:-https://github.com/tydeptrai21042004/DT1D-vit.git}"
REPO_COMMIT="${REPO_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/DT1D-vit-fair-vtab"
DATA_DIR="$WORKDIR/vtab_data"
MODEL_ROOT="$WORKDIR/vit_weights"
OUTPUT_ROOT="$WORKDIR/vtab_caltech101_vitb16_fair"
RESULT_ZIP="$WORKDIR/vtab_caltech101_vitb16_fair.zip"

rm -rf "$REPO_DIR"
git clone --depth 1 "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
if [[ -n "$REPO_COMMIT" ]]; then
  git fetch --depth 1 origin "$REPO_COMMIT"
  git checkout --detach "$REPO_COMMIT"
fi

echo "SOURCE COMMIT: $(git rev-parse HEAD)"

python -m pip install -q --upgrade-strategy only-if-needed \
  scipy scikit-learn pandas Pillow fvcore iopath yacs simplejson termcolor \
  tabulate tqdm ml-collections 'timm>=1.0.0,<2' PyYAML tensorflow-datasets
if ! python - <<'PY'
import tensorflow
print(tensorflow.__version__)
PY
then
  python -m pip install -q 'tensorflow>=2.16,<2.20'
fi

python validate_dt1d_vit.py
python verify_vpt_original.py
python verify_fair_protocol.py
python -m pytest -q tests/test_dt1d_token_adapter.py tests/test_fair_protocol.py tests/test_repository_contracts.py

mkdir -p "$DATA_DIR" "$MODEL_ROOT"
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

python - "$DATA_DIR" <<'PY'
import sys, tensorflow_datasets as tfds
root=sys.argv[1]
builder=tfds.builder('caltech101:3.*.*', data_dir=root)
builder.download_and_prepare()
print('VTAB Caltech101 prepared:', root)
PY

# train.py keeps train800/val200/test separated. The fair runner also preserves
# the original VPT optimizer/LR convention (SGD+momentum with nominal LR scaled
# by batch_size/256) while giving every method the same number of tuning trials.
# Test is disabled during tuning and evaluated only after best-val restoration.
python run_fair_vit_comparison.py \
  --dataset vtab-caltech101 \
  --data-path "$DATA_DIR" \
  --model-root "$MODEL_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --batch-sizes 32 \
  --methods dt1d,vpt,pfeiffer,full,linear \
  --epochs 10 \
  --warmup-epoch 1 \
  --vpt-tokens 10 \
  --gpus 0,1

rm -f "$RESULT_ZIP"
cd "$WORKDIR"
zip -qr "$RESULT_ZIP" "$(basename "$OUTPUT_ROOT")"
echo "RESULT ZIP: $RESULT_ZIP"
ls -lh "$RESULT_ZIP"
