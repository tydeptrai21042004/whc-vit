#!/usr/bin/env python3
"""
major actions here: fine-tune the features and evaluate different settings
"""
import os
import json
import torch
import warnings

import numpy as np
import random

from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer
from src.models.build_model import build_model
from src.utils.file_io import PathManager

from launch import default_argument_parser, logging_train_setup
warnings.filterwarnings("ignore")

import os
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Simple single-node init for Colab
    cfg.DIST_INIT_PATH = "env://"

    # Setup output dir: OUTPUT_DIR / DATA.NAME / FEATURE / lr_wd / run1
    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}"
    )
    run_name = f"seed{cfg.SEED}" if cfg.SEED is not None else "run1"
    output_path = os.path.join(output_dir, output_folder, run_name)

    # Make dirs (no multi-run logic, just reuse run1)
    PathManager.mkdirs(output_path)
    cfg.OUTPUT_DIR = output_path

    cfg.freeze()
    return cfg



def _canonical_active_offsets(value):
    if isinstance(value, str):
        return [int(x.strip()) for x in value.replace(';', ',').split(',') if x.strip()]
    if isinstance(value, (tuple, list)):
        return [int(x) for x in value]
    raise TypeError(f"Unsupported ACTIVE_OFFSETS value: {value!r}")


def _method_signature(cfg):
    """Canonical architecture-only signature for safe resume/auditing."""
    transfer = str(cfg.MODEL.TRANSFER_TYPE)
    adapter = str(cfg.MODEL.ADAPTER.NAME)
    sig = {"transfer_type": transfer, "adapter_name": adapter}
    if transfer == "prompt":
        sig["prompt"] = {
            "num_tokens": int(cfg.MODEL.PROMPT.NUM_TOKENS),
            "location": str(cfg.MODEL.PROMPT.LOCATION),
            "initiation": str(cfg.MODEL.PROMPT.INITIATION),
            "project": int(cfg.MODEL.PROMPT.PROJECT),
            "deep": bool(cfg.MODEL.PROMPT.DEEP),
            "deep_shared": bool(cfg.MODEL.PROMPT.DEEP_SHARED),
            "vit_pool_type": str(cfg.MODEL.PROMPT.VIT_POOL_TYPE),
            "dropout": float(cfg.MODEL.PROMPT.DROPOUT),
        }
    if adapter.lower() == "pfeiffer":
        sig["pfeiffer"] = {
            "reduction_factor": int(cfg.MODEL.ADAPTER.REDUCTION_FACTOR),
            "style": str(cfg.MODEL.ADAPTER.STYLE),
        }
    if adapter.lower() == "dt1d":
        d = cfg.MODEL.ADAPTER.DT1D
        sig["dt1d"] = {
            "architecture": "R124-P2-G16-Axis-LearnedGate",
            "axis": str(d.AXIS),
            "group_size": int(d.GROUP_SIZE),
            "active_offsets": _canonical_active_offsets(d.ACTIVE_OFFSETS),
            "detail_components": str(d.DETAIL_COMPONENTS),
            "project_l1": bool(d.PROJECT_L1),
            "gate_mode": str(d.GATE_MODE),
            "gate_init": float(d.GATE_INIT),
            "residual_scale": float(d.RESIDUAL_SCALE),
            "padding": str(d.PADDING),
            "use_pointwise": bool(d.USE_POINTWISE),
            "shift_p": int(d.SHIFT_P),
            "shift_lambda_mode": str(d.SHIFT_LAMBDA_MODE),
            "shift_lambda_scope": str(d.SHIFT_LAMBDA_SCOPE),
            "shift_lambda_init": float(d.SHIFT_LAMBDA_INIT),
            "shift_lambda_max": float(d.SHIFT_LAMBDA_MAX),
            "shift_normalization": str(d.SHIFT_NORMALIZATION),
        }
    return json.dumps(sig, sort_keys=True, separators=(",", ":"))


def get_loaders(cfg, logger):
    logger.info("Loading training data...")
    # Keep validation data completely separate from optimization.  In
    # particular, VTAB uses train800 for optimization and val200 only for
    # checkpoint/model selection.  The official test split is evaluated once
    # after restoring the best-validation checkpoint.
    train_loader = data_loader.construct_train_loader(cfg)

    logger.info("Loading validation data...")
    val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        logger.info("...no test data is constructed")
        test_loader = None
    else:
        test_loader = data_loader.construct_test_loader(cfg)
    return train_loader,  val_loader, test_loader


def train(cfg, args):
    # clear up residual cache from previous runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # main training / eval actions here

    # Fix all RNGs before model construction.  DataLoader instances also use
    # their own seed-derived generators (see src/data/loader.py), so the same
    # paper seed gives the same sample order/augmentations for every method.
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(cfg.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # setup training env including loggers
    logging_train_setup(args, cfg)
    logger = logging.get_logger("visual_prompt")

    train_loader, val_loader, test_loader = get_loaders(cfg, logger)
    logger.info("Constructing models...")
    model, cur_device = build_model(cfg)

    # Model-specific modules consume different amounts of RNG during
    # initialization.  Reset the training RNG stream after construction so
    # that the data/augmentation randomness does not depend on which method is
    # being compared.
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(cfg.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.SEED)

    logger.info("Setting up Evalutator...")
    evaluator = Evaluator()
    logger.info("Setting up Trainer...")
    trainer = Trainer(cfg, model, evaluator, cur_device)

    summary = None
    if train_loader:
        summary = trainer.train_classifier(train_loader, val_loader, test_loader)
    else:
        print("No train loader presented. Exit")

    if cfg.SOLVER.TOTAL_EPOCH == 0 and test_loader is not None:
        trainer.eval_classifier(test_loader, "test", 0)

    torch.save(evaluator.results, os.path.join(cfg.OUTPUT_DIR, "eval_results.pth"))
    if summary is not None:
        summary = dict(summary)
        summary.update({
            "seed": cfg.SEED,
            "dataset": cfg.DATA.NAME,
            "feature": cfg.DATA.FEATURE,
            "batch_size": int(cfg.DATA.BATCH_SIZE),
            "crop_size": int(cfg.DATA.CROPSIZE),
            "total_epoch": int(cfg.SOLVER.TOTAL_EPOCH),
            "optimizer": str(cfg.SOLVER.OPTIMIZER),
            "momentum": float(cfg.SOLVER.MOMENTUM),
            "base_lr": float(cfg.SOLVER.BASE_LR),
            "weight_decay": float(cfg.SOLVER.WEIGHT_DECAY),
            "scheduler": str(cfg.SOLVER.SCHEDULER),
            "warmup_epoch": int(cfg.SOLVER.WARMUP_EPOCH),
            "transfer_type": str(cfg.MODEL.TRANSFER_TYPE),
            "adapter_name": str(cfg.MODEL.ADAPTER.NAME),
            "method_signature": _method_signature(cfg),
            "prompt_num_tokens": int(cfg.MODEL.PROMPT.NUM_TOKENS),
            "pfeiffer_reduction_factor": int(cfg.MODEL.ADAPTER.REDUCTION_FACTOR),
            "dt1d_architecture": "R124-P2-G16-Axis-LearnedGate",
            "dt1d_group_size": int(cfg.MODEL.ADAPTER.DT1D.GROUP_SIZE),
            "dt1d_active_offsets": _canonical_active_offsets(cfg.MODEL.ADAPTER.DT1D.ACTIVE_OFFSETS),
            "dt1d_shift_p": int(cfg.MODEL.ADAPTER.DT1D.SHIFT_P),
            "dt1d_gate_mode": str(cfg.MODEL.ADAPTER.DT1D.GATE_MODE),
            "dt1d_gate_init": float(cfg.MODEL.ADAPTER.DT1D.GATE_INIT),
            "dt1d_lambda_mode": str(cfg.MODEL.ADAPTER.DT1D.SHIFT_LAMBDA_MODE),
            "dt1d_lambda_scope": str(cfg.MODEL.ADAPTER.DT1D.SHIFT_LAMBDA_SCOPE),
            "data_path": str(cfg.DATA.DATAPATH),
            "model_root": str(cfg.MODEL.MODEL_ROOT),
            "data_no_test": bool(cfg.DATA.NO_TEST),
            "protocol": (
                "train800->val200->test@best-val"
                if cfg.DATA.NAME.startswith("vtab-")
                else "official-train->official-val->official-test@best-val"
            ),
            "total_parameters": int(sum(p.numel() for p in model.parameters())),
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        })
        with open(os.path.join(cfg.OUTPUT_DIR, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def main(args):
    """main function to call from workflow"""

    # set up cfg and args
    cfg = setup(args)

    # Perform training.
    train(cfg, args)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
