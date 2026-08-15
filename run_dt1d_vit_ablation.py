#!/usr/bin/env python3
"""Run reviewer DT1D-Adapter component ablations through train.py."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _encode_opt(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    # YACS literal-decodes numeric-looking strings; quote them explicitly.
    if text.replace(".", "", 1).isdigit() or "," in text:
        return json.dumps(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-config", default="configs/finetune/flowers_dt1d.yaml")
    ap.add_argument("--manifest", default="configs/ablations/dt1d_components_vit.yaml")
    ap.add_argument("--output-root", default="outputs/dt1d_vit_ablation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default="all", help="all or comma-separated names")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    manifest = yaml.safe_load((root / args.manifest).read_text())
    variants = manifest["variants"]
    selected = list(variants) if args.variants == "all" else [x.strip() for x in args.variants.split(",") if x.strip()]
    unknown = [x for x in selected if x not in variants]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}")

    out_root = (root / args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    plan = []

    for name in selected:
        out = out_root / name
        opts = ["SEED", str(args.seed), "RUN_N_TIMES", "1", "OUTPUT_DIR", str(out)]
        for key, value in variants[name].items():
            opts += [key, _encode_opt(value)]
        cmd = [
            sys.executable,
            str(root / "train.py"),
            "--config-file",
            str(root / args.base_config),
            *opts,
        ]
        plan.append({"variant": name, "cmd": cmd, "output": str(out)})

    (out_root / "execution_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Planned {len(plan)} run(s).")
    for i, item in enumerate(plan, 1):
        print(f"{i:02d}. {item['variant']}")

    if args.dry_run:
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for i, item in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] RUN {item['variant']}", flush=True)
        subprocess.run(item["cmd"], cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
