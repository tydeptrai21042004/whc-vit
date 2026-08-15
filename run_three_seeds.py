#!/usr/bin/env python3
"""Run one config with the paper seeds 0, 1, and 2."""
import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.replace(";", ",").split(",") if s.strip()]
    if len(seeds) != 3:
        raise SystemExit("Paper runs require exactly three seeds (default: 0,1,2).")
    for seed in seeds:
        cmd = [sys.executable, "train.py", "--config-file", args.config_file, "SEED", str(seed)] + args.opts
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
