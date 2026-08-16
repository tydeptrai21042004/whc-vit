#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python proposal_fingerprint.py
python validate_dt1d_vit.py
python verify_fair_protocol.py
python verify_vpt_original.py
python -m compileall -q train.py launch.py run_fair_vit_comparison.py run_three_seeds.py aggregate_three_seeds.py run_dt1d_vit_ablation.py proposal_contract.py proposal_fingerprint.py src tests
python -m pytest -q tests
while IFS= read -r -d '' file; do bash -n "$file"; done < <(find . -maxdepth 5 -type f -name '*.sh' -print0)
rm -rf .pytest_cache
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
echo "ViT release validation passed."
