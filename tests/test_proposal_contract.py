from __future__ import annotations
import json
from pathlib import Path
import yaml
from src.models.vit_adapter.dt1d_adapter import DT1DAdapter
from proposal_contract import proposal_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_proposal_matches_frozen_cross_repo_contract():
    spec=json.loads((ROOT/'proposal_spec.json').read_text(encoding='utf-8'))
    m=DT1DAdapter(64)
    assert m.proposal_name==spec['proposal']
    assert m.architecture_name==spec['architecture']
    assert list(m.axis_names)==spec['axes']
    assert m.group_size==spec['group_size']
    assert list(m.active_offsets)==spec['active_offsets']
    assert m.shift_p==spec['weighted_shift']['p']
    assert m.shift_lambda_mode==spec['weighted_shift']['lambda_mode']
    assert m.shift_lambda_scope==spec['weighted_shift']['lambda_scope']
    assert m.shift_lambda_max==spec['weighted_shift']['lambda_max_abs']
    assert m.shift_normalization==spec['weighted_shift']['normalization']
    assert m.project_l1 is spec['projection']['joint_hw_l1']
    assert m.gate_mode==spec['residual']['gate_mode']
    assert m.padding_mode==spec['padding']
    assert m.base_kernel_size==spec['base_kernel_size']
    assert m.effective_kernel_size==spec['effective_kernel_size']
    assert m.convolution_calls_per_forward==spec['depthwise_convolution_calls']
    assert len(proposal_fingerprint(ROOT))==64


def test_canonical_paper_configs_enable_eval_kernel_cache_only_as_runtime_optimization():
    for rel in ['configs/finetune/flowers_dt1d.yaml','configs/vtab/caltech101_dt1d.yaml','configs/vtab/dtd_dt1d.yaml','configs/vtab/eurosat_dt1d.yaml']:
        cfg=yaml.safe_load((ROOT/rel).read_text(encoding='utf-8'))
        assert cfg['MODEL']['ADAPTER']['DT1D']['CACHE_KERNEL'] is True
