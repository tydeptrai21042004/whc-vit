#!/usr/bin/env python3
"""Verify that the VPT baseline matches the supplied original source contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_fair_vit_comparison as fair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--tokens', type=int, default=5)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    fidelity = fair.verify_vpt_source_fidelity(root)
    eff = fair.source_lr_grid('vpt', args.batch_size)
    report = {
        'status': 'PASS',
        'source_hashes': fidelity['hashes'],
        'vpt_structure': {
            'type': 'shallow prompt tuning',
            'location': 'prepend',
            'initiation': 'random',
            'project': -1,
            'deep': False,
            'vit_pool_type': 'original',
            'dropout': 0.0,
            'tokens_requested': args.tokens,
        },
        'optimizer': 'sgd',
        'momentum': 0.9,
        'source_default_weight_decay': 1e-4,
        'source_lr_rule': 'nominal_lr * batch_size / 256',
        'batch_size': args.batch_size,
        'nominal_lr_candidates': list(fair.VPT_NOMINAL_LRS),
        'effective_lr_candidates': eff,
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
