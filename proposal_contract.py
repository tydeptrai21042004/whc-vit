#!/usr/bin/env python3
"""Shared proposal-contract helpers for DT1D-Adapter repositories.

This module contains no model implementation.  It only fingerprints the frozen
proposal specification and records release/runtime provenance.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SPEC_PATH = ROOT / "proposal_spec.json"


def load_spec(root: Path | None = None) -> dict[str, Any]:
    root = ROOT if root is None else Path(root)
    return json.loads((root / "proposal_spec.json").read_text(encoding="utf-8"))


def canonical_spec_bytes(root: Path | None = None) -> bytes:
    return json.dumps(load_spec(root), sort_keys=True, separators=(",", ":")).encode("utf-8")


def proposal_fingerprint(root: Path | None = None) -> str:
    return hashlib.sha256(canonical_spec_bytes(root)).hexdigest()


def repo_version(root: Path | None = None) -> str:
    root = ROOT if root is None else Path(root)
    path = root / "VERSION"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "unknown"


def git_commit(root: Path | None = None) -> str:
    root = ROOT if root is None else Path(root)
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else "unavailable"


def runtime_metadata(root: Path | None = None) -> dict[str, Any]:
    root = ROOT if root is None else Path(root)
    spec = load_spec(root)
    return {
        "repo_version": repo_version(root),
        "git_commit": git_commit(root),
        "proposal": spec["proposal"],
        "proposal_architecture": spec["architecture"],
        "proposal_fingerprint_sha256": proposal_fingerprint(root),
    }
