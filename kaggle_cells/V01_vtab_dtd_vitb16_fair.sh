#!/usr/bin/env bash
set -Eeuo pipefail

# V01: VTAB-DTD / ViT-B/16 / BS32
# Methods: dt1d,vpt,pfeiffer,full,linear
# Fair protocol: 10 validation-only LR candidates per method on tune seed 42,
# followed by final seeds 0/1/2 and one test evaluation at the best-val checkpoint.

SESSION_ID="V01"
DATASET="vtab-dtd"
BATCH_SIZE="32"
METHODS="dt1d,vpt,pfeiffer,full,linear"
VPT_TOKENS="10"
REPO_URL="https://github.com/tydeptrai21042004/DT1D-vit.git"
REPO_COMMIT="${DT1D_VIT_COMMIT:-}"
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/DT1D-vit-$SESSION_ID"
DATA_ROOT="$WORKDIR/data_$SESSION_ID"
MODEL_ROOT="$WORKDIR/models_$SESSION_ID"
OUTPUT_ROOT="$WORKDIR/vit_$SESSION_ID"
RESULT_ZIP="$WORKDIR/${SESSION_ID}_results.zip"
DEADLINE_EPOCH="$(( $(date +%s) + 715*60 ))"
export SESSION_ID DATASET BATCH_SIZE METHODS VPT_TOKENS REPO_DIR DATA_ROOT MODEL_ROOT OUTPUT_ROOT RESULT_ZIP DEADLINE_EPOCH

pack_results() {
  [[ -d "$OUTPUT_ROOT" ]] || return 0
  python - <<'PYZIP'
import os, zipfile
from pathlib import Path
root=Path(os.environ['OUTPUT_ROOT']); dst=Path(os.environ['RESULT_ZIP'])
if dst.exists(): dst.unlink()
with zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() not in {'.pth','.pt','.ckpt'}:
            z.write(p,p.relative_to(root.parent))
print('RESULT_ZIP =', dst)
PYZIP
}
trap 'rc=$?; trap - EXIT; [[ -f "$RESULT_ZIP" ]] || pack_results || true; exit $rc' EXIT

command -v git >/dev/null
command -v python >/dev/null
rm -rf "$REPO_DIR"
git clone --depth 1 "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"
if [[ -n "$REPO_COMMIT" ]]; then
  git fetch --depth 1 origin "$REPO_COMMIT"
  git checkout --detach "$REPO_COMMIT"
fi
SOURCE_COMMIT="$(git rev-parse HEAD)"; export SOURCE_COMMIT
echo "SOURCE_COMMIT=$SOURCE_COMMIT"

python -m pip install -q --upgrade-strategy only-if-needed   scipy scikit-learn pandas Pillow fvcore iopath yacs simplejson termcolor   tabulate tqdm ml-collections 'timm>=1.0.0,<2' PyYAML tensorflow-datasets six
if ! python - <<'PYTF'
import tensorflow
print('tensorflow:', tensorflow.__version__)
PYTF
then
  python -m pip install -q 'tensorflow>=2.16,<2.20'
fi

python validate_dt1d_vit.py
python verify_vpt_original.py
python verify_fair_protocol.py
python -m pytest -q tests/test_dt1d_token_adapter.py tests/test_fair_protocol.py tests/test_repository_contracts.py
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator.'
print('GPU count =', torch.cuda.device_count())
for i in range(torch.cuda.device_count()): print(i, torch.cuda.get_device_name(i))
PYGPU

# Restore a prior resumable result archive if attached.
rm -rf "$OUTPUT_ROOT"
python - <<'PYRESTORE'
import json, os, shutil, zipfile
from pathlib import Path
sid=os.environ['SESSION_ID']; out=Path(os.environ['OUTPUT_ROOT']); inp=Path('/kaggle/input')
zs=list(inp.rglob(f'{sid}_results.zip')) if inp.exists() else []
def score(z):
    complete=0; summaries=0
    try:
        with zipfile.ZipFile(z) as f:
            names=f.namelist(); summaries=sum(n.endswith('run_summary.json') or n.endswith('_fair_three_seed.csv') for n in names)
            for n in names:
                if n.endswith('SESSION_STATUS.json'):
                    try: complete=max(complete,int(bool(json.loads(f.read(n)).get('complete',False))))
                    except Exception: pass
    except Exception: pass
    return complete, summaries, z.stat().st_mtime
if zs:
    z=max(zs,key=score); tmp=Path('/kaggle/working')/f'_restore_{sid}'; shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir()
    with zipfile.ZipFile(z) as f: f.extractall(tmp)
    found=list(tmp.rglob(f'vit_{sid}'))
    if len(found)==1: shutil.move(str(found[0]),str(out)); print('RESTORED',z)
    shutil.rmtree(tmp,ignore_errors=True)
PYRESTORE
mkdir -p "$OUTPUT_ROOT" "$DATA_ROOT" "$MODEL_ROOT"

WEIGHT_FILE="$MODEL_ROOT/ViT-B_16-224.npz"
if [[ ! -s "$WEIGHT_FILE" ]]; then
  tmp="${WEIGHT_FILE}.part"; rm -f "$tmp"
  curl -L --fail --retry 5 --retry-delay 5 --connect-timeout 30     'https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz' -o "$tmp"
  mv "$tmp" "$WEIGHT_FILE"
fi
python - "$WEIGHT_FILE" <<'PYWEIGHT'
import sys,numpy as np
z=np.load(sys.argv[1]); assert len(z.files)==200, len(z.files); print('ViT weight tensors=',len(z.files))
PYWEIGHT

DATA_PATH="$DATA_ROOT/tfds"; mkdir -p "$DATA_PATH"; export DATA_PATH
python - <<'PYDATA'
import os, tensorflow_datasets as tfds
spec='dtd:3.*.*'
b=tfds.builder(spec,data_dir=os.environ['DATA_PATH']); b.download_and_prepare(); print('TFDS READY',b.info.full_name)
PYDATA

PREFLIGHT="$WORKDIR/_preflight_$SESSION_ID"; rm -rf "$PREFLIGHT"
python run_fair_vit_comparison.py   --dataset "$DATASET" --data-path "$DATA_PATH" --model-root "$MODEL_ROOT"   --output-root "$PREFLIGHT" --batch-sizes "$BATCH_SIZE" --methods "$METHODS"   --epochs 10 --resolution 224 --seeds 0,1,2 --tune-seed 42   --weight-decay 1e-4 --warmup-epoch 1 --patience 20   --vpt-tokens "$VPT_TOKENS" --allow-boundary-best --gpus cpu --dry-run
rm -rf "$PREFLIGHT"

LOG="$OUTPUT_ROOT/session.log"; mkdir -p "$OUTPUT_ROOT"; set +e
setsid python run_fair_vit_comparison.py   --dataset "$DATASET" --data-path "$DATA_PATH" --model-root "$MODEL_ROOT"   --output-root "$OUTPUT_ROOT" --batch-sizes "$BATCH_SIZE" --methods "$METHODS"   --epochs 10 --resolution 224 --seeds 0,1,2 --tune-seed 42   --weight-decay 1e-4 --warmup-epoch 1 --patience 20   --vpt-tokens "$VPT_TOKENS" --allow-boundary-best --gpus auto >"$LOG" 2>&1 &
RUN_PID=$!
while kill -0 "$RUN_PID" 2>/dev/null; do
  if (( $(date +%s) >= DEADLINE_EPOCH )); then
    echo 'TIME CAP reached; terminating process group' | tee -a "$LOG"
    kill -TERM -- "-$RUN_PID" 2>/dev/null || true; sleep 30; kill -KILL -- "-$RUN_PID" 2>/dev/null || true; break
  fi
  sleep 10
done
wait "$RUN_PID"; RUN_RC=$?; set -e

python - <<'PYSTATUS'
import json, os, pandas as pd
from pathlib import Path
out=Path(os.environ['OUTPUT_ROOT']); expected={x for x in os.environ['METHODS'].split(',') if x}
bs=int(os.environ['BATCH_SIZE']); cp=out/'aggregated'/f"{os.environ['DATASET'].replace('-','_')}_fair_three_seed.csv"
actual=set(); complete=False
if cp.is_file():
    df=pd.read_csv(cp); df=df[df['batch_size'].astype(int)==bs]; actual=set(df['method_key'].astype(str)); complete=(actual==expected and len(df)==len(expected))
status={'session':os.environ['SESSION_ID'],'family':'vit','dataset':os.environ['DATASET'],'methods':sorted(expected),'batch_size':bs,'actual_methods':sorted(actual),'complete':bool(complete),'source_commit':os.environ.get('SOURCE_COMMIT')}
(out/'SESSION_STATUS.json').write_text(json.dumps(status,indent=2)); print(json.dumps(status,indent=2))
PYSTATUS

{ echo "session=$SESSION_ID"; echo "dataset=$DATASET"; echo "methods=$METHODS"; echo "commit=$SOURCE_COMMIT"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true; } > "$OUTPUT_ROOT/run_environment.txt"
pack_results; trap - EXIT
if [[ "$RUN_RC" -eq 0 ]] && python - <<'PYCHECK'
import json, os, sys
from pathlib import Path
p=Path(os.environ['OUTPUT_ROOT'])/'SESSION_STATUS.json'
sys.exit(0 if p.is_file() and json.loads(p.read_text()).get('complete') else 1)
PYCHECK
then
  echo "SESSION COMPLETE: $SESSION_ID"
else
  echo "SESSION INCOMPLETE/TIME-CAPPED: $SESSION_ID -- attach its ZIP and rerun the same cell."
fi
exit 0
