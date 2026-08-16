#!/usr/bin/env python3
"""Print or compare the frozen DT1D-Adapter proposal fingerprint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from proposal_contract import load_spec, proposal_fingerprint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare", default=None, help="Other repository root or proposal_spec.json")
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    payload = {
        "proposal": load_spec(here)["proposal"],
        "architecture": load_spec(here)["architecture"],
        "fingerprint_sha256": proposal_fingerprint(here),
    }
    if args.compare:
        other = Path(args.compare).expanduser().resolve()
        other_root = other.parent if other.is_file() else other
        payload["other_fingerprint_sha256"] = proposal_fingerprint(other_root)
        payload["same_proposal_contract"] = payload["fingerprint_sha256"] == payload["other_fingerprint_sha256"]
        print(json.dumps(payload, indent=2))
        return 0 if payload["same_proposal_contract"] else 2
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
