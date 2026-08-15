#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# For VTAB-Caltech101 additionally run:
# python -m pip install -r requirements-vtab.txt
