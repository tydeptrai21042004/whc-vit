#!/usr/bin/env python3
"""Aggregate seed 0/1/2 summaries with protocol-consistency checks."""
import argparse
import json
import math
from pathlib import Path


def mean_std(values):
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-15
    return a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="output directory containing seed0/seed1/seed2 run summaries")
    args = p.parse_args()
    files = sorted(Path(args.root).rglob("run_summary.json"))
    rows = []
    for f in files:
        data = json.loads(f.read_text())
        if data.get("seed") in (0, 1, 2):
            rows.append((f, data))
    by_seed = {int(d["seed"]): (f, d) for f, d in rows}
    missing = [s for s in (0, 1, 2) if s not in by_seed]
    if missing:
        raise SystemExit(f"Missing seed summaries: {missing}")

    # Refuse to combine seeds produced by different training protocols.
    consistency_fields = [
        "dataset", "feature", "batch_size", "crop_size", "total_epoch",
        "optimizer", "base_lr", "weight_decay", "scheduler", "warmup_epoch",
        "transfer_type", "adapter_name", "protocol", "trainable_parameters",
        "total_parameters",
    ]
    ref = by_seed[0][1]
    mismatches = []
    for seed in (1, 2):
        cur = by_seed[seed][1]
        for field in consistency_fields:
            if field in ref or field in cur:
                if field not in ref or field not in cur or not _same(ref.get(field), cur.get(field)):
                    mismatches.append(
                        f"seed{seed} field {field}: seed0={ref.get(field)!r}, seed{seed}={cur.get(field)!r}"
                    )
    if mismatches:
        raise SystemExit("Refusing to aggregate inconsistent seed runs:\n" + "\n".join(mismatches))

    vals = []
    for seed in (0, 1, 2):
        value = by_seed[seed][1].get("test_top1")
        if value is None:
            raise SystemExit(f"seed{seed} has no final test result")
        vals.append(100.0 * float(value))
    mean, std = mean_std(vals)
    print("seeds: 0,1,2")
    print("test Acc@1 (%): " + ", ".join(f"{v:.3f}" for v in vals))
    print(f"mean ± std: {mean:.3f} ± {std:.3f}")
    if "base_lr" in ref:
        print(
            "protocol: "
            f"optimizer={ref.get('optimizer')}, lr={ref.get('base_lr')}, "
            f"wd={ref.get('weight_decay')}, epochs={ref.get('total_epoch')}, "
            f"batch={ref.get('batch_size')}"
        )


if __name__ == "__main__":
    main()
