#!/usr/bin/env python3
"""ViT adapter blocks with final DT1D token-adapter support."""
from functools import partial

import torch
import torch.nn as nn

try:
    from timm.layers import Mlp, DropPath
except ImportError:  # timm <= 0.4.x
    from timm.models.layers import Mlp, DropPath
from timm.models.vision_transformer import Block  # base ViT block

# project logger (avoid clobbering stdlib `logging`)
from ...utils import logging as vlogging
logger = vlogging.get_logger("visual_prompt")

from .dt1d_adapter import DT1DTokenAdapter


def build_adapter(name, embed_dim, grid_size, cfg):
    """Build a token-space adapter for a ViT block."""
    name = (name or "").lower()
    if name != "dt1d":
        return None
    d = cfg.ADAPTER.DT1D
    return DT1DTokenAdapter(
        embed_dim=embed_dim,
        grid_size=grid_size,
        axis=d.AXIS,
        group_size=d.GROUP_SIZE,
        active_offsets=d.ACTIVE_OFFSETS,
        detail_components=d.DETAIL_COMPONENTS,
        contrast_split=d.CONTRAST_SPLIT,
        project_l1=d.PROJECT_L1,
        gate_mode=d.GATE_MODE,
        gate_init=d.GATE_INIT,
        residual_scale=d.RESIDUAL_SCALE,
        padding_mode=d.PADDING,
        use_pointwise=d.USE_POINTWISE,
        pointwise_ratio=d.POINTWISE_RATIO,
        pointwise_groups=d.POINTWISE_GROUPS,
        use_bn=d.USE_BN,
        cache_kernel=d.CACHE_KERNEL,
        shift_p=d.SHIFT_P,
        shift_lambda_mode=d.SHIFT_LAMBDA_MODE,
        shift_lambda_scope=d.SHIFT_LAMBDA_SCOPE,
        shift_lambda_init=d.SHIFT_LAMBDA_INIT,
        shift_lambda_max=d.SHIFT_LAMBDA_MAX,
        shift_normalization=d.SHIFT_NORMALIZATION,
    )



class Pfeiffer_Block(Block):
    """
    A ViT block with a parallel Pfeiffer-style MLP adapter branch.
    Keeps the standard ViT residual ordering:
      x = x + DropPath( Attn( Norm1(x) ) )
      x = x + DropPath( MLP( Norm2(x) ) [+ Adapter] )
    """

    def __init__(self, adapter_config, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.adapter_config = adapter_config
        if adapter_config.STYLE != "Pfeiffer":
            raise ValueError("Only Pfeiffer adapter style is supported here.")

        red_factor = int(adapter_config.REDUCTION_FACTOR)
        red = max(1, dim // red_factor)
        self.adapter_downsample = nn.Linear(dim, red)
        self.adapter_act_fn = act_layer()
        self.adapter_upsample = nn.Linear(red, dim)

        # Zero-init so the adapter path is identity-safe at start
        nn.init.zeros_(self.adapter_downsample.weight)
        nn.init.zeros_(self.adapter_downsample.bias)
        nn.init.zeros_(self.adapter_upsample.weight)
        nn.init.zeros_(self.adapter_upsample.bias)

    def forward(self, x):
        # 1) MHSA residual
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # 2) MLP residual (+ Pfeiffer adapter in parallel)
        u = self.mlp(self.norm2(x))
        a = self.adapter_upsample(self.adapter_act_fn(self.adapter_downsample(self.norm2(x))))
        x = x + self.drop_path(u + a)
        return x
