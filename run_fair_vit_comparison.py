#!/usr/bin/env python3
"""Run a reviewer-safe, equal-budget ViT comparison with method-faithful optimization.

Fairness here means the methods share the experimental protocol that should be
shared: dataset/splits, pretrained backbone, resolution, batch size, epoch
budget, scheduler family, weight decay, tuning seed, final seeds, checkpoint
selection rule, and number of hyperparameter trials.  Optimizer family and
learning-rate scale are allowed to follow the original method implementation.

This is important for VPT.  The original VPT source uses SGD with momentum and
large *nominal* LR candidates that are scaled by batch_size/256 in its tuning
scripts.  Forcing VPT to use AdamW with a tiny LR is not a faithful baseline and
can cause severe under-training.

Protocol:
1. Each method receives exactly the same NUMBER of validation-only LR trials.
2. VPT/Linear use their source-faithful SGD profiles; Full FT uses AdamW;
   DT1D-Adapter/Pfeiffer use AdamW profiles appropriate to their implementations.
3. DATA.NO_TEST=True during tuning, so test data cannot select hyperparameters.
4. Final seeds are 0/1/2 and test is evaluated once after restoring each seed's
   best-validation checkpoint.
5. The VPT implementation files are hash-checked against the user-supplied
   original VPT repository before any experiment starts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

from proposal_contract import runtime_metadata

METHOD_ORDER = ("dt1d", "vpt", "pfeiffer", "full", "linear")
DISPLAY = {
    "dt1d": "DT1D-Adapter",
    "vpt": "VPT",
    "pfeiffer": "Pfeiffer Adapter",
    "full": "Full fine-tuning",
    "linear": "Linear probing",
}
DATASETS = {
    "flowers102": {
        "name": "OxfordFlowers",
        "classes": 102,
        "default_batches": (32, 16),
        "vpt_tokens": 5,
        "protocol": "official-train->official-val->official-test@best-val",
    },
    "vtab-caltech101": {
        "name": "vtab-caltech101",
        "classes": 102,
        "default_batches": (32,),
        "vpt_tokens": 10,
        "protocol": "train800->val200->official-test@best-val",
    },
    "vtab-dtd": {
        "name": "vtab-dtd",
        "classes": 47,
        "default_batches": (32,),
        "vpt_tokens": 10,
        "protocol": "train800->val200->official-test@best-val",
    },
    "vtab-eurosat": {
        "name": "vtab-eurosat",
        "classes": 10,
        "default_batches": (32,),
        "vpt_tokens": 10,
        "protocol": "train800->val200->official-test@best-val",
    },
}


# SHA256 hashes from the user-supplied original vpt-main.zip.  These files are
# the VPT model/build implementation used by this repository and must stay
# byte-identical for the VPT baseline to be called source-faithful.
VPT_SOURCE_HASHES = {
    "src/models/vit_prompt/vit.py": "06613eabb95b3b488c46957334bf4cf0d6fbe875afff9785eb1787d09581af1a",
    "src/models/vit_prompt/vit_ablations.py": "a49b2a6d49ad48e390d540eb1af7ad1dd7877fd165bd7d4a93a561b5be133291",
    "src/models/build_vit_backbone.py": "9d8d33c89852013854c46fd950724929cebee7e9c51a367827f856d61d8eb576",
    "src/models/build_model.py": "4712b2a1b10e13c1fea079cc5d19cbcd93f761971f429f7005e0a5c1486cf1cc",
    "configs/base-prompt.yaml": "5f2a75839468b730f89fa0774051a52673197f09bc1de1bcf9b3f0d9ed1a3f37",
    "configs/prompt/flowers.yaml": "b654b0f65255607718478c1a5b366c210d6b0fdfe2d610613cdc63da1b27baf5",
    "configs/base-linear.yaml": "5f437c22d78c59ea90e2967f724053f1825a972b1fa91f6abd1d671bb4c6daba",
    "configs/base-finetune.yaml": "cb774eb2c3f80e1396a1581db51719196a74e48bbd9b6f50887efcfb19c15adf",
}

# Original VPT/linear tuning scripts express a nominal LR and then apply
#     effective_lr = nominal_lr / 256 * batch_size.
# The VTAB VPT source includes these 10 prompt candidates.  The FGVC prompt
# script has the same set except 0.05; we include 0.05 so every method receives
# an equal 10-candidate tuning budget and because it is present in the supplied
# original VTAB VPT tuner.
VPT_NOMINAL_LRS = (50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.5, 0.25, 0.1, 0.05)
LINEAR_NOMINAL_LRS = (50.0, 25.0, 10.0, 5.0, 2.5, 1.0, 0.5, 0.25, 0.1, 0.05)

# Full fine-tuning in the original VPT code uses AdamW and nominal LR values
# {1e-4, 5e-4, 1e-3, 5e-3}, also scaled by batch_size/256.  We densify only
# within/near that source range to give the same 10 LR trials as every method.
FULL_NOMINAL_LRS = (1e-4, 2e-4, 5e-4, 7.5e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3.5e-3, 5e-3)

# DT1D-Adapter/Pfeiffer are not VPT source baselines. Their AdamW grids are effective
# LRs and deliberately contain the values used in the manuscript experiments.
ADAMW_ADAPTER_LRS = (1e-5, 2.5e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2.5e-3, 5e-3, 1e-2)


def verify_vpt_source_fidelity(repo_root: Path) -> dict:
    observed = {}
    for rel, expected in VPT_SOURCE_HASHES.items():
        path = repo_root / rel
        if not path.is_file():
            raise RuntimeError(f"Missing original VPT implementation file: {rel}")
        got = sha256_file(path)
        observed[rel] = got
        if got != expected:
            raise RuntimeError(
                "VPT source fidelity check failed. Refusing to benchmark a modified "
                f"VPT implementation: {rel} expected={expected} got={got}"
            )
    return {"status": "PASS", "hashes": observed}


def method_optimizer(method: str) -> str:
    # Matches the supplied original VPT code for VPT, Linear and Full FT.
    if method in {"vpt", "linear"}:
        return "sgd"
    return "adamw"


def source_lr_grid(method: str, batch_size: int) -> list[float]:
    scale = float(batch_size) / 256.0
    if method == "vpt":
        return sorted({float(x) * scale for x in VPT_NOMINAL_LRS})
    if method == "linear":
        return sorted({float(x) * scale for x in LINEAR_NOMINAL_LRS})
    if method == "full":
        return sorted({float(x) * scale for x in FULL_NOMINAL_LRS})
    if method in {"dt1d", "pfeiffer"}:
        return sorted({float(x) for x in ADAMW_ADAPTER_LRS})
    raise ValueError(method)


def resolve_method_lr_grid(args, method: str, batch_size: int) -> list[float]:
    # A per-method override is allowed for a documented rerun, but every method
    # must still have the same number of candidates (enforced by the audit).
    attr = f"{method}_lr_grid"
    override = getattr(args, attr, None)
    if override:
        vals = parse_csv_numbers(override, float)
    elif args.lr_grid:
        vals = parse_csv_numbers(args.lr_grid, float)
    else:
        vals = source_lr_grid(method, batch_size)
    vals = sorted(set(float(x) for x in vals))
    if not vals or any(x <= 0 for x in vals):
        raise SystemExit(f"Invalid LR grid for {method}: {vals}")
    return vals


def parse_csv_numbers(text: str, typ=float) -> List:
    return [typ(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]


def resolve_final_seeds(seed_text: str | None, result_mode: str) -> List[int]:
    """Fail-closed publication seed policy for quantitative tables and figures."""
    if result_mode not in {"table", "figure"}:
        raise ValueError(f"Unknown result_mode={result_mode!r}")
    default = [0, 1, 2] if result_mode == "table" else [0]
    seeds = default if seed_text is None else parse_csv_numbers(seed_text, int)
    if not seeds or len(seeds) != len(set(seeds)):
        raise SystemExit("Final seeds must be non-empty and unique")
    if result_mode == "table" and len(seeds) < 3:
        raise SystemExit("Table mode requires at least three independent final seeds")
    if result_mode == "figure" and len(seeds) != 1:
        raise SystemExit("Figure mode requires exactly one representative final seed")
    return [int(x) for x in seeds]


def result_csv_name(dataset: str, result_mode: str, seeds: List[int]) -> str:
    stem = dataset.replace("-", "_")
    if result_mode == "figure":
        return f"{stem}_fair_single_seed.csv"
    if seeds == [0, 1, 2]:
        return f"{stem}_fair_three_seed.csv"
    return f"{stem}_fair_table_{len(seeds)}seed.csv"


def deep_merge(base: dict, update: dict) -> dict:
    out = json.loads(json.dumps(base))
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def canonical_active_offsets(value) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [x.strip() for x in value.replace(';', ',').split(',') if x.strip()]
        return tuple(int(x) for x in parts)
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    raise TypeError(f"Unsupported ACTIVE_OFFSETS type: {type(value).__name__}: {value!r}")


def validate_dt1d_default() -> dict:
    """Fail closed unless repository defaults match the canonical proposal."""
    from src.configs.config import get_cfg

    d = get_cfg().MODEL.ADAPTER.DT1D
    observed = {
        "offsets": canonical_active_offsets(d.ACTIVE_OFFSETS),
        "axis": str(d.AXIS).lower(),
        "group_size": int(d.GROUP_SIZE),
        "detail_components": str(d.DETAIL_COMPONENTS).lower(),
        "project_l1": bool(d.PROJECT_L1),
        "gate_mode": str(d.GATE_MODE).lower(),
        "gate_init": float(d.GATE_INIT),
        "use_pointwise": bool(d.USE_POINTWISE),
        "padding": str(d.PADDING).lower(),
        "shift_p": int(d.SHIFT_P),
        "lambda_mode": str(d.SHIFT_LAMBDA_MODE).lower(),
        "lambda_scope": str(d.SHIFT_LAMBDA_SCOPE).lower(),
        "lambda_init": float(d.SHIFT_LAMBDA_INIT),
        "lambda_max": float(d.SHIFT_LAMBDA_MAX),
        "shift_normalization": str(d.SHIFT_NORMALIZATION).lower(),
    }
    expected = {
        "offsets": (1, 2, 4),
        "axis": "hw",
        "group_size": 16,
        "detail_components": "offset4",
        "project_l1": True,
        "gate_mode": "learned",
        "gate_init": 0.01,
        "use_pointwise": False,
        "padding": "replicate",
        "shift_p": 2,
        "lambda_mode": "learned",
        "lambda_scope": "axis",
        "lambda_init": 0.0,
        "lambda_max": 0.5,
        "shift_normalization": "mean",
    }
    if observed != expected:
        raise SystemExit(f"Unexpected DT1D-Adapter defaults: {observed}; expected {expected}")
    return observed

def preflight_yacs_merge(paths: Iterable[Path], label: str) -> None:
    """Merge every generated YAML through the real YACS schema before GPU work."""
    from src.configs.config import get_cfg

    paths = list(paths)
    failures = []
    for path in paths:
        try:
            cfg = get_cfg()
            cfg.merge_from_file(str(path))
            cfg.freeze()
        except Exception as exc:
            failures.append((str(path), repr(exc)))
    if failures:
        detail = '\n'.join(f"  - {p}: {e}" for p, e in failures[:20])
        raise SystemExit(
            f"PRE-FLIGHT YACS MERGE FAILED ({len(failures)}/{len(paths)}) for {label}:\n{detail}"
        )
    print(f"PRE-FLIGHT YACS MERGE: PASS ({len(paths)}/{len(paths)} configs) [{label}]", flush=True)


def method_fragment(method: str, vpt_tokens: int, pfeiffer_reduction: int) -> dict:
    if method == "dt1d":
        return {
            "MODEL": {
                "TRANSFER_TYPE": "adapter",
                "ADAPTER": {
                    "NAME": "DT1D",
                    "DT1D": {
                        "AXIS": "hw",
                        "GROUP_SIZE": 16,
                        # ACTIVE_OFFSETS inherits the canonical repository tuple (1,2,4).
                        "DETAIL_COMPONENTS": "offset4",
                        "CONTRAST_SPLIT": 8,
                        "PROJECT_L1": True,
                        "GATE_MODE": "learned",
                        "GATE_INIT": 0.01,
                        "RESIDUAL_SCALE": 1.0,
                        "PADDING": "replicate",
                        "USE_POINTWISE": False,
                        "CACHE_KERNEL": True,
                        "SHIFT_P": 2,
                        "SHIFT_LAMBDA_MODE": "learned",
                        "SHIFT_LAMBDA_SCOPE": "axis",
                        "SHIFT_LAMBDA_INIT": 0.0,
                        "SHIFT_LAMBDA_MAX": 0.5,
                        "SHIFT_NORMALIZATION": "mean",
                    },
                },
            }
        }
    if method == "full":
        return {"MODEL": {"TRANSFER_TYPE": "end2end", "ADAPTER": {"NAME": "none"}}}
    if method == "linear":
        return {"MODEL": {"TRANSFER_TYPE": "linear", "ADAPTER": {"NAME": "none"}}}
    if method == "vpt":
        return {
            "MODEL": {
                "TRANSFER_TYPE": "prompt",
                "ADAPTER": {"NAME": "none"},
                "PROMPT": {
                    "NUM_TOKENS": int(vpt_tokens),
                    "LOCATION": "prepend",
                    "INITIATION": "random",
                    "PROJECT": -1,
                    "DEEP": False,
                    "NUM_DEEP_LAYERS": None,
                    "DEEP_SHARED": False,
                    "VIT_POOL_TYPE": "original",
                    "DROPOUT": 0.0,
                },
            }
        }
    if method == "pfeiffer":
        return {
            "MODEL": {
                "TRANSFER_TYPE": "adapter",
                "ADAPTER": {
                    "NAME": "Pfeiffer",
                    "STYLE": "Pfeiffer",
                    "REDUCTION_FACTOR": int(pfeiffer_reduction),
                },
            }
        }
    raise ValueError(f"Unknown method: {method}")

def common_config(args, batch_size: int, output_dir: Path, no_test: bool) -> dict:
    ds = DATASETS[args.dataset]
    return {
        "NUM_GPUS": 1,
        "NUM_SHARDS": 1,
        "RUN_N_TIMES": 1,
        "OUTPUT_DIR": str(output_dir),
        "MODEL": {
            "TYPE": "vit",
            "MODEL_ROOT": str(Path(args.model_root).resolve()),
            "MLP_NUM": 0,
            "SAVE_CKPT": False,
        },
        "DATA": {
            "NAME": ds["name"],
            "DATAPATH": str(Path(args.data_path).resolve()),
            "NUMBER_CLASSES": ds["classes"],
            "MULTILABEL": False,
            "FEATURE": "sup_vitb16_224",
            "BATCH_SIZE": int(batch_size),
            "CROPSIZE": int(args.resolution),
            "NUM_WORKERS": int(args.num_workers),
            "NO_TEST": bool(no_test),
        },
        "SOLVER": {
            # Shared training budget. Optimizer family is added by make_config
            # from the method's source-faithful profile.
            "WEIGHT_DECAY": float(args.weight_decay),
            "SCHEDULER": "cosine",
            "WARMUP_EPOCH": int(args.warmup_epoch),
            "TOTAL_EPOCH": int(args.epochs),
            "PATIENCE": int(args.patience),
            "LOSS": "softmax",
            "LOG_EVERY_N": int(args.log_every),
        },
    }


def method_signature_from_dict(cfg: dict) -> str:
    model = cfg["MODEL"]
    transfer = str(model["TRANSFER_TYPE"])
    adapter_cfg = model.get("ADAPTER", {})
    adapter = str(adapter_cfg.get("NAME", "none"))
    sig = {"transfer_type": transfer, "adapter_name": adapter}
    if transfer == "prompt":
        p = model.get("PROMPT", {})
        sig["prompt"] = {
            "num_tokens": int(p.get("NUM_TOKENS", 5)),
            "location": str(p.get("LOCATION", "prepend")),
            "initiation": str(p.get("INITIATION", "random")),
            "project": int(p.get("PROJECT", -1)),
            "deep": bool(p.get("DEEP", False)),
            "deep_shared": bool(p.get("DEEP_SHARED", False)),
            "vit_pool_type": str(p.get("VIT_POOL_TYPE", "original")),
            "dropout": float(p.get("DROPOUT", 0.0)),
        }
    if adapter.lower() == "pfeiffer":
        sig["pfeiffer"] = {
            "reduction_factor": int(adapter_cfg.get("REDUCTION_FACTOR", 8)),
            "style": str(adapter_cfg.get("STYLE", "Pfeiffer")),
        }
    if adapter.lower() == "dt1d":
        d = adapter_cfg.get("DT1D", {})
        sig["dt1d"] = {
            "architecture": "R124-P2-G16-Axis-LearnedGate",
            "axis": str(d.get("AXIS", "hw")),
            "group_size": int(d.get("GROUP_SIZE", 16)),
            "active_offsets": list(canonical_active_offsets(d.get("ACTIVE_OFFSETS", "1,2,4"))),
            "detail_components": str(d.get("DETAIL_COMPONENTS", "offset4")),
            "project_l1": bool(d.get("PROJECT_L1", True)),
            "gate_mode": str(d.get("GATE_MODE", "learned")),
            "gate_init": float(d.get("GATE_INIT", 0.01)),
            "residual_scale": float(d.get("RESIDUAL_SCALE", 1.0)),
            "padding": str(d.get("PADDING", "replicate")),
            "use_pointwise": bool(d.get("USE_POINTWISE", False)),
            "shift_p": int(d.get("SHIFT_P", 2)),
            "lambda_mode": str(d.get("SHIFT_LAMBDA_MODE", "learned")),
            "lambda_scope": str(d.get("SHIFT_LAMBDA_SCOPE", "axis")),
            "lambda_init": float(d.get("SHIFT_LAMBDA_INIT", 0.0)),
            "lambda_max": float(d.get("SHIFT_LAMBDA_MAX", 0.5)),
            "shift_normalization": str(d.get("SHIFT_NORMALIZATION", "mean")),
        }
    return json.dumps(sig, sort_keys=True, separators=(",", ":"))


def make_config(args, batch_size: int, method: str, lr: float, phase: str) -> dict:
    no_test = phase == "tune"
    root = Path(args.output_root).resolve() / phase / f"bs{batch_size}" / method
    cfg = common_config(args, batch_size, root, no_test=no_test)
    cfg = deep_merge(cfg, method_fragment(method, args.vpt_tokens, args.pfeiffer_reduction))
    cfg["SOLVER"]["OPTIMIZER"] = method_optimizer(method)
    cfg["SOLVER"]["MOMENTUM"] = 0.9
    cfg["SOLVER"]["BASE_LR"] = float(lr)
    return cfg


def write_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def stable_float(v: float) -> str:
    return str(float(v))


def expected_summary(cfg: dict, seed: int) -> Path:
    return (
        Path(cfg["OUTPUT_DIR"])
        / cfg["DATA"]["NAME"]
        / cfg["DATA"]["FEATURE"]
        / f"lr{stable_float(cfg['SOLVER']['BASE_LR'])}_wd{stable_float(cfg['SOLVER']['WEIGHT_DECAY'])}"
        / f"seed{seed}"
        / "run_summary.json"
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_tuning_configs(
    configs: Dict[Tuple[int, str, float], dict],
    methods: Iterable[str],
    lr_grids: Dict[Tuple[int, str], List[float]],
) -> dict:
    """Verify equal tuning *budget* and shared experimental protocol.

    Optimizer family and numerical LR range are method-specific by design so
    source-faithful baselines are not crippled. Fairness is enforced through
    equal candidate counts plus identical data/training/evaluation budgets.
    """
    methods = tuple(methods)
    batches = sorted({key[0] for key in configs})
    shared_fields = (
        ("DATA", "NAME"), ("DATA", "DATAPATH"), ("DATA", "NUMBER_CLASSES"),
        ("DATA", "FEATURE"), ("DATA", "BATCH_SIZE"), ("DATA", "CROPSIZE"),
        ("SOLVER", "WEIGHT_DECAY"), ("SOLVER", "SCHEDULER"),
        ("SOLVER", "WARMUP_EPOCH"), ("SOLVER", "TOTAL_EPOCH"),
        ("SOLVER", "PATIENCE"),
    )
    audit = {
        "status": "PASS",
        "fairness_definition": (
            "same data/splits/backbone/resolution/batch/epochs/scheduler/WD/"
            "tuning-seed/final-seeds/selection rule and equal LR-trial count; "
            "method-native optimizer and LR scale are allowed"
        ),
        "batches": {},
        "allowed_method_differences": [
            "MODEL.TRANSFER_TYPE", "MODEL.ADAPTER", "MODEL.PROMPT",
            "SOLVER.OPTIMIZER", "SOLVER.MOMENTUM", "SOLVER.BASE_LR",
        ],
    }
    for bs in batches:
        counts = {m: len(lr_grids[(bs, m)]) for m in methods}
        if len(set(counts.values())) != 1:
            raise AssertionError(f"Unequal tuning budgets at batch {bs}: {counts}")
        trial_count = next(iter(counts.values()))
        batch_audit = {"trials_per_method": trial_count, "methods": {}}
        ref_lr = lr_grids[(bs, methods[0])][0]
        reference = configs[(bs, methods[0], ref_lr)]
        for method in methods:
            expected_optimizer = method_optimizer(method)
            seen = []
            for lr in lr_grids[(bs, method)]:
                cfg = configs[(bs, method, lr)]
                if cfg["DATA"]["NO_TEST"] is not True:
                    raise AssertionError(
                        f"Tuning config leaked test data: bs={bs} method={method} lr={lr}"
                    )
                if cfg["SOLVER"]["OPTIMIZER"].lower() != expected_optimizer:
                    raise AssertionError(
                        f"Optimizer mismatch for {method}: "
                        f"{cfg['SOLVER']['OPTIMIZER']} != {expected_optimizer}"
                    )
                for section, key in shared_fields:
                    expected = bs if key == "BATCH_SIZE" else reference[section][key]
                    if cfg[section][key] != expected:
                        raise AssertionError(
                            f"Fairness mismatch {section}.{key}: method={method}, lr={lr}, "
                            f"got {cfg[section][key]!r}, expected {expected!r}"
                        )
                if float(cfg["SOLVER"]["BASE_LR"]) != float(lr):
                    raise AssertionError("Generated LR does not match method candidate grid")
                seen.append(float(lr))
            if seen != list(lr_grids[(bs, method)]):
                raise AssertionError(f"Method {method} did not receive its complete LR grid")
            batch_audit["methods"][method] = {
                "optimizer": expected_optimizer,
                "lr_candidates": seen,
                "source_profile": method in {"vpt", "linear", "full"},
            }
        audit["batches"][str(bs)] = batch_audit
    return audit

def _assert_resume_compatible(summary_path: Path, cfg: dict, seed: int) -> None:
    data = read_summary(summary_path)
    expected = {
        "seed": int(seed),
        "dataset": cfg["DATA"]["NAME"],
        "feature": cfg["DATA"]["FEATURE"],
        "batch_size": int(cfg["DATA"]["BATCH_SIZE"]),
        "crop_size": int(cfg["DATA"]["CROPSIZE"]),
        "total_epoch": int(cfg["SOLVER"]["TOTAL_EPOCH"]),
        "optimizer": str(cfg["SOLVER"]["OPTIMIZER"]),
        "momentum": float(cfg["SOLVER"].get("MOMENTUM", 0.9)),
        "base_lr": float(cfg["SOLVER"]["BASE_LR"]),
        "weight_decay": float(cfg["SOLVER"]["WEIGHT_DECAY"]),
        "scheduler": str(cfg["SOLVER"]["SCHEDULER"]),
        "warmup_epoch": int(cfg["SOLVER"]["WARMUP_EPOCH"]),
        "transfer_type": str(cfg["MODEL"]["TRANSFER_TYPE"]),
        "adapter_name": str(cfg["MODEL"].get("ADAPTER", {}).get("NAME", "none")),
        "method_signature": method_signature_from_dict(cfg),
        "prompt_num_tokens": int(cfg["MODEL"].get("PROMPT", {}).get("NUM_TOKENS", 5)),
        "pfeiffer_reduction_factor": int(cfg["MODEL"].get("ADAPTER", {}).get("REDUCTION_FACTOR", 8)),
        "data_path": str(cfg["DATA"]["DATAPATH"]),
        "model_root": str(cfg["MODEL"]["MODEL_ROOT"]),
        "data_no_test": bool(cfg["DATA"]["NO_TEST"]),
    }
    mismatches = {}
    for key, value in expected.items():
        got = data.get(key, "<missing>")
        if isinstance(value, float) and got != "<missing>":
            equal = abs(float(got) - value) <= 1e-15
        else:
            equal = got == value
        if not equal:
            mismatches[key] = {"expected": value, "found": got}
    if mismatches:
        raise RuntimeError(
            "Refusing to reuse a stale/incompatible run_summary.json. "
            f"Use a fresh --output-root or remove the stale run. File: {summary_path}; "
            f"mismatches={json.dumps(mismatches, sort_keys=True)}"
        )


def run_one(repo_root: Path, cfg_path: Path, cfg: dict, seed: int, gpu: str, log_path: Path) -> Path:
    summary = expected_summary(cfg, seed)
    if summary.exists():
        try:
            _assert_resume_compatible(summary, cfg, seed)
        except Exception as exc:
            # Never reuse stale results, but do not make the whole notebook fail.
            # Only the incompatible seed directory is removed and recomputed.
            stale_dir = summary.parent
            print(f"[RERUN] incompatible completed run: {summary}\n  reason: {exc}", flush=True)
            shutil.rmtree(stale_dir, ignore_errors=True)
        else:
            print(f"[SKIP] compatible completed run: {summary}", flush=True)
            return summary
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if gpu != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [sys.executable, "train.py", "--config-file", str(cfg_path), "SEED", str(seed)]
    print(f"[RUN gpu={gpu}] {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=repo_root, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        except Exception:
            pass
        raise RuntimeError(f"Run failed ({proc.returncode}): {cfg_path}\n{tail}")
    if not summary.exists():
        raise RuntimeError(f"Run finished but summary is missing: {summary}")
    return summary


def read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_summary(summary: dict, cfg: dict, seed: int, tuning: bool) -> None:
    required = ["best_epoch", "best_val_top1", "seed", "optimizer", "base_lr", "total_epoch", "batch_size"]
    missing = [k for k in required if k not in summary]
    if missing:
        raise AssertionError(f"Summary missing fields {missing}")
    if int(summary["seed"]) != int(seed):
        raise AssertionError("Seed mismatch in run summary")
    if str(summary["optimizer"]).lower() != str(cfg["SOLVER"]["OPTIMIZER"]).lower():
        raise AssertionError(
            f"Optimizer mismatch: summary={summary['optimizer']} "
            f"config={cfg['SOLVER']['OPTIMIZER']}"
        )
    if str(cfg["SOLVER"]["OPTIMIZER"]).lower() == "sgd":
        if abs(float(summary.get("momentum", 0.9)) - float(cfg["SOLVER"].get("MOMENTUM", 0.9))) > 1e-15:
            raise AssertionError("Momentum mismatch in run summary")
    if abs(float(summary["base_lr"]) - float(cfg["SOLVER"]["BASE_LR"])) > 1e-15:
        raise AssertionError("Learning-rate mismatch in run summary")
    if abs(float(summary.get("weight_decay", cfg["SOLVER"]["WEIGHT_DECAY"])) -
           float(cfg["SOLVER"]["WEIGHT_DECAY"])) > 1e-15:
        raise AssertionError("Weight-decay mismatch in run summary")
    if int(summary["total_epoch"]) != int(cfg["SOLVER"]["TOTAL_EPOCH"]):
        raise AssertionError("Epoch-budget mismatch")
    if int(summary["batch_size"]) != int(cfg["DATA"]["BATCH_SIZE"]):
        raise AssertionError("Batch-size mismatch")
    if tuning:
        if summary.get("test_top1") is not None:
            raise AssertionError("Tuning run evaluated the test set; refusing to continue")
        if not bool(summary.get("data_no_test")):
            raise AssertionError("Tuning summary does not prove DATA.NO_TEST=True")
    else:
        if summary.get("test_top1") is None:
            raise AssertionError("Final run did not produce a test result")
        if bool(summary.get("data_no_test")):
            raise AssertionError("Final run incorrectly disabled the test set")


def select_lr(records: List[Tuple[float, dict]]) -> Tuple[float, float]:
    # Validation only.  Smaller LR breaks exact ties for stability.
    ranked = sorted(records, key=lambda x: (-float(x[1]["best_val_top1"]), float(x[0])))
    best_lr, best = ranked[0]
    return float(best_lr), float(best["best_val_top1"])


def aggregate_final(rows: List[dict]) -> dict:
    tests = [100.0 * float(r["test_top1"]) for r in rows]
    vals = [100.0 * float(r["best_val_top1"]) for r in rows]
    epochs = [int(r["best_epoch"]) for r in rows]
    return {
        "test_mean": statistics.mean(tests),
        "test_std": statistics.stdev(tests) if len(tests) > 1 else 0.0,
        "best_val_mean": statistics.mean(vals),
        "best_val_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "best_epoch_mean": statistics.mean(epochs),
        "test_by_seed": tests,
        "best_epoch_by_seed": epochs,
        "trainable_parameters": int(rows[0]["trainable_parameters"]),
        "total_parameters": int(rows[0]["total_parameters"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--batch-sizes", default=None, help="e.g. 32 or 32,16")
    ap.add_argument("--methods", default=",".join(METHOD_ORDER))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--result-mode", choices=("table", "figure"), default="table", help="table: >=3 final seeds; figure: exactly 1 representative final seed")
    ap.add_argument("--seeds", default=None, help="Final seeds. Defaults: table=0,1,2; figure=0")
    ap.add_argument("--tune-seed", type=int, default=42)
    ap.add_argument(
        "--lr-grid", default=None,
        help=(
            "Optional shared *effective* LR grid. Leave unset for the recommended "
            "source-faithful per-method grids."
        ),
    )
    for _m in METHOD_ORDER:
        ap.add_argument(
            f"--{_m}-lr-grid", dest=f"{_m}_lr_grid", default=None,
            help=f"Optional effective LR override for {_m}; must keep equal candidate count.",
        )
    ap.add_argument(
        "--fixed-lr", type=float, default=None,
        help="Skip LR tuning and use one identical effective LR for every method (not source-faithful VPT tuning).",
    )
    ap.add_argument(
        "--allow-boundary-best", action="store_true",
        help=(
            "Allow final runs when a method selects the edge of its LR grid. "
            "Default aborts first so the method grid can be shifted while keeping the same trial count."
        ),
    )
    ap.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="Common fixed WD. 1e-4 is the original VPT base-prompt default and is shared by every method.",
    )
    ap.add_argument("--warmup-epoch", type=int, default=1)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--vpt-tokens", type=int, default=None,
        help=(
            "VPT prompt length. Default=5 from the supplied original VPT source. "
            "Use an override only when a paper/table explicitly documents another value."
        ),
    )
    ap.add_argument("--pfeiffer-reduction", type=int, default=16)
    ap.add_argument("--gpus", default="auto", help="auto, cpu, or comma list such as 0,1")
    ap.add_argument("--dry-run", action="store_true", help="Generate and audit configs only")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    # Validate the one proposal before generating any benchmark config.
    dt1d_defaults = validate_dt1d_default()
    print("DT1D-ADAPTER DEFAULTS:", dt1d_defaults, flush=True)
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Fail closed if the VPT implementation has drifted from the original ZIP.
    vpt_fidelity = verify_vpt_source_fidelity(repo_root)

    ds = DATASETS[args.dataset]
    batches = list(ds["default_batches"]) if args.batch_sizes is None else parse_csv_numbers(args.batch_sizes, int)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHOD_ORDER]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    final_seeds = resolve_final_seeds(args.seeds, args.result_mode)
    args.vpt_tokens = int(args.vpt_tokens if args.vpt_tokens is not None else ds["vpt_tokens"])

    if args.fixed_lr is not None and args.fixed_lr <= 0:
        raise SystemExit("--fixed-lr must be positive")

    lr_grids: Dict[Tuple[int, str], List[float]] = {}
    if args.fixed_lr is None:
        for bs in batches:
            for method in methods:
                lr_grids[(bs, method)] = resolve_method_lr_grid(args, method, bs)
        counts = {(bs, m): len(v) for (bs, m), v in lr_grids.items()}
        if len(set(counts.values())) != 1:
            raise SystemExit(
                "Fair comparison requires the same number of LR candidates for every method. "
                f"Got {counts}"
            )

    if args.gpus == "cpu":
        gpu_ids = ["cpu"]
    elif args.gpus == "auto":
        try:
            import torch
            n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            n = 0
        gpu_ids = [str(i) for i in range(n)] or ["cpu"]
    else:
        gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]

    config_root = out_root / "protocol_configs"
    logs_root = out_root / "logs"

    tuning_configs: Dict[Tuple[int, str, float], dict] = {}
    tuning_paths: Dict[Tuple[int, str, float], Path] = {}
    if args.fixed_lr is None:
        for bs in batches:
            for method in methods:
                for lr in lr_grids[(bs, method)]:
                    cfg = make_config(args, bs, method, lr, "tune")
                    path = config_root / "tune" / f"bs{bs}" / f"{method}_lr{lr:.12g}.yaml"
                    write_yaml(path, cfg)
                    tuning_configs[(bs, method, lr)] = cfg
                    tuning_paths[(bs, method, lr)] = path
        audit = audit_tuning_configs(tuning_configs, methods, lr_grids)
        preflight_yacs_merge(tuning_paths.values(), "tuning")
    else:
        audit = {
            "status": "PASS",
            "mode": "fixed-identical-effective-lr",
            "fixed_lr": args.fixed_lr,
            "warning": "This mode does not reproduce the original VPT LR tuning policy.",
        }

    audit.update({
        **runtime_metadata(repo_root),
        "dataset": args.dataset,
        "protocol": ds["protocol"],
        "methods": methods,
        "batch_sizes": batches,
        "result_mode": args.result_mode,
        "seed_policy": "at_least_three" if args.result_mode == "table" else "exactly_one",
        "final_seeds": final_seeds,
        "final_seed_count": len(final_seeds),
        "tune_seed": None if args.fixed_lr is not None else args.tune_seed,
        "method_optimizers": {m: method_optimizer(m) for m in methods},
        "weight_decay": args.weight_decay,
        "scheduler": "cosine",
        "warmup_epoch": args.warmup_epoch,
        "epochs": args.epochs,
        "resolution": args.resolution,
        "vpt_tokens": args.vpt_tokens,
        "vpt_source_fidelity": vpt_fidelity,
        "lr_policy": (
            "fixed-identical" if args.fixed_lr is not None
            else "source-faithful method-specific scales with equal candidate counts"
        ),
        "lr_grids": {
            f"bs{bs}/{m}": lr_grids[(bs, m)]
            for bs in batches for m in methods
        } if args.fixed_lr is None else {},
        "test_policy": "disabled during tuning; once per final seed after best-validation checkpoint restore",
        "head_policy": "same deterministic classifier-head initialization for the same seed across methods",
        "data_rng_policy": "explicit seed-derived DataLoader generator/workers shared across methods",
        "vpt_protocol_note": (
            "VPT model/build code is byte-identical to the supplied original source; optimizer is SGD+momentum; "
            "LR candidates follow original batch-size scaling. The paper comparison keeps train/val/test separate "
            "instead of the original VTAB final train800+val200 retraining step."
        ),
    })
    (out_root / "fairness_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))

    if args.dry_run:
        print("DRY RUN: source fidelity, configs, equal tuning budget, and fairness audit passed.")
        return
    if gpu_ids == ["cpu"]:
        raise SystemExit("Training requires GPU. Use --dry-run for CPU-only validation.")

    selected: Dict[Tuple[int, str], dict] = {}
    if args.fixed_lr is None:
        jobs = []
        for bs in batches:
            for method in methods:
                for lr in lr_grids[(bs, method)]:
                    cfg = tuning_configs[(bs, method, lr)]
                    cfg_path = tuning_paths[(bs, method, lr)]
                    log = logs_root / "tune" / f"bs{bs}_{method}_lr{lr:.12g}_seed{args.tune_seed}.log"
                    jobs.append((cfg_path, cfg, args.tune_seed, log))

        tuning_failures = []
        if len(gpu_ids) == 1:
            for j in jobs:
                try:
                    run_one(repo_root, j[0], j[1], j[2], gpu_ids[0], j[3])
                except Exception as exc:
                    tuning_failures.append({"config": str(j[0]), "error": str(exc)})
                    print(f"[TUNE CANDIDATE FAILED] {j[0]}: {exc}", file=sys.stderr, flush=True)
        else:
            with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
                fs = {}
                for i, j in enumerate(jobs):
                    fut = ex.submit(run_one, repo_root, j[0], j[1], j[2], gpu_ids[i % len(gpu_ids)], j[3])
                    fs[fut] = j
                for f in as_completed(fs):
                    j = fs[f]
                    try:
                        f.result()
                    except Exception as exc:
                        tuning_failures.append({"config": str(j[0]), "error": str(exc)})
                        print(f"[TUNE CANDIDATE FAILED] {j[0]}: {exc}", file=sys.stderr, flush=True)

        if tuning_failures:
            (out_root / "tuning_failures.json").write_text(
                json.dumps(tuning_failures, indent=2), encoding="utf-8"
            )

        for bs in batches:
            for method in methods:
                grid = lr_grids[(bs, method)]
                records = []
                failed = []
                for lr in grid:
                    cfg = tuning_configs[(bs, method, lr)]
                    path = expected_summary(cfg, args.tune_seed)
                    if not path.exists():
                        failed.append(float(lr))
                        continue
                    summary = read_summary(path)
                    try:
                        validate_summary(summary, cfg, args.tune_seed, tuning=True)
                    except Exception:
                        failed.append(float(lr))
                        continue
                    records.append((lr, summary))
                if not records:
                    raise SystemExit(
                        f"NO VALID TUNING RESULT: BS={bs} method={method}. "
                        f"All LR candidates failed: {grid}. See tuning logs/failures."
                    )
                best_lr, best_val = select_lr(records)
                if not args.allow_boundary_best and len(grid) > 1 and (best_lr == min(grid) or best_lr == max(grid)):
                    raise SystemExit(
                        f"TUNING GRID TOO NARROW: BS={bs} method={method} selected boundary LR={best_lr}. "
                        f"No final test runs were started. Shift/expand --{method}-lr-grid while keeping "
                        f"exactly {len(grid)} candidates so every method retains the same tuning budget."
                    )
                selected[(bs, method)] = {
                    "lr": best_lr,
                    "optimizer": method_optimizer(method),
                    "tune_best_val": best_val,
                    "attempted_lr_grid": list(grid),
                    "failed_lr_candidates": failed,
                }
    else:
        for bs in batches:
            for method in methods:
                selected[(bs, method)] = {
                    "lr": float(args.fixed_lr),
                    "optimizer": method_optimizer(method),
                    "tune_best_val": None,
                    "attempted_lr_grid": [float(args.fixed_lr)],
                    "failed_lr_candidates": [],
                }

    final_configs = {}
    final_paths = {}
    for bs in batches:
        for method in methods:
            lr = selected[(bs, method)]["lr"]
            cfg = make_config(args, bs, method, lr, "final")
            if cfg["DATA"]["NO_TEST"] is not False:
                raise AssertionError("Final config must enable test evaluation")
            path = config_root / "final" / f"bs{bs}" / f"{method}.yaml"
            write_yaml(path, cfg)
            final_configs[(bs, method)] = cfg
            final_paths[(bs, method)] = path

    preflight_yacs_merge(final_paths.values(), "final")

    final_jobs = []
    for bs in batches:
        for method in methods:
            for seed in final_seeds:
                cfg = final_configs[(bs, method)]
                path = final_paths[(bs, method)]
                log = logs_root / "final" / f"bs{bs}_{method}_seed{seed}.log"
                final_jobs.append((bs, method, seed, path, cfg, log))
    if len(gpu_ids) == 1:
        for _, _, seed, path, cfg, log in final_jobs:
            run_one(repo_root, path, cfg, seed, gpu_ids[0], log)
    else:
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
            fs = []
            for i, (_, _, seed, path, cfg, log) in enumerate(final_jobs):
                fs.append(ex.submit(run_one, repo_root, path, cfg, seed, gpu_ids[i % len(gpu_ids)], log))
            for f in as_completed(fs):
                f.result()

    final_table = []
    for bs in batches:
        for method in methods:
            cfg = final_configs[(bs, method)]
            summaries = []
            for seed in final_seeds:
                pth = expected_summary(cfg, seed)
                sm = read_summary(pth)
                validate_summary(sm, cfg, seed, tuning=False)
                summaries.append(sm)
            agg = aggregate_final(summaries)
            final_table.append({
                "batch_size": bs,
                "method": DISPLAY[method],
                "method_key": method,
                "optimizer": method_optimizer(method),
                "selected_lr": selected[(bs, method)]["lr"],
                "tune_best_val": selected[(bs, method)]["tune_best_val"],
                **agg,
            })

    result_dir = out_root / "aggregated"
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / result_csv_name(args.dataset, args.result_mode, final_seeds)
    fields = [
        "batch_size", "method", "method_key", "optimizer", "selected_lr", "tune_best_val",
        "trainable_parameters", "total_parameters", "best_val_mean", "best_val_std",
        "test_mean", "test_std", "best_epoch_mean", "test_by_seed", "best_epoch_by_seed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in final_table:
            out = dict(row)
            out["test_by_seed"] = json.dumps(out["test_by_seed"])
            out["best_epoch_by_seed"] = json.dumps(out["best_epoch_by_seed"])
            w.writerow(out)

    manifest = {
        **runtime_metadata(repo_root),
        "result_mode": args.result_mode,
        "seed_policy": "at_least_three" if args.result_mode == "table" else "exactly_one",
        "final_seeds": final_seeds,
        "fairness_audit": audit,
        "selected_hyperparameters": {
            f"bs{bs}/{method}": info for (bs, method), info in selected.items()
        },
        "results": final_table,
        "config_sha256": {
            str(path.relative_to(out_root)): sha256_file(path)
            for path in sorted(config_root.rglob("*.yaml"))
        },
    }
    (result_dir / "fair_protocol_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    heading = "FINAL FAIR TABLE RESULTS" if args.result_mode == "table" else "FINAL FAIR SINGLE-SEED FIGURE RESULTS"
    print("\n" + heading)
    print("=" * len(heading))
    for row in final_table:
        metric = (
            f"test={row['test_mean']:.3f} ± {row['test_std']:.3f}"
            if args.result_mode == "table"
            else f"test={row['test_mean']:.3f} (seed {final_seeds[0]})"
        )
        print(
            f"BS={row['batch_size']:>3}  {row['method']:<22} "
            f"opt={row['optimizer']:<5} lr={row['selected_lr']:<10g} {metric}"
        )
    print(f"\nCSV: {csv_path}")
    print(f"Manifest: {result_dir / 'fair_protocol_manifest.json'}")


if __name__ == "__main__":
    main()
