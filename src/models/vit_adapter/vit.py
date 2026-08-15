#!/usr/bin/env python3
"""
ViT with Pfeiffer or canonical DT1D-Adapter token adaptation.
"""
import copy
import numpy as np
import torch
import torch.nn as nn

from scipy import ndimage
from torch.nn import Linear, LayerNorm

# ViT backbone bits & ACT2FN
from ..vit_backbones.vit import *
from ...utils import logging
logger = logging.get_logger("visual_prompt")

from .dt1d_adapter import DT1DTokenAdapter


class ADPT_Block(nn.Module):
    """
    One transformer block with an optional adapter.

    Supported:
      - adapter_config.NAME == "Pfeiffer": classic down->act->up MLP adapter
      - adapter_config.NAME == "DT1D": canonical DT1D-Adapter proposal
    """
    def __init__(self, config, vis, adapter_config, grid_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

        self.adapter_config = adapter_config
        self.token_adapter = None

        name = str(getattr(adapter_config, "NAME", "")).lower()

        if name == "pfeiffer":
            red = int(adapter_config.REDUCTION_FACTOR)
            self.adapter_downsample = nn.Linear(self.hidden_size, self.hidden_size // red)
            self.adapter_upsample   = nn.Linear(self.hidden_size // red, self.hidden_size)
            self.adapter_act_fn     = ACT2FN["gelu"]

            # Near-identity, but non-degenerate: the zero up-projection makes
            # the initial residual exactly zero, while a Xavier-initialized
            # down-projection allows the branch to start learning immediately.
            nn.init.xavier_uniform_(self.adapter_downsample.weight)
            nn.init.zeros_(self.adapter_downsample.bias)
            nn.init.zeros_(self.adapter_upsample.weight)
            nn.init.zeros_(self.adapter_upsample.bias)

        elif name == "dt1d":
            if grid_size is None:
                raise ValueError("grid_size (H, W) is required for DT1D token adaptation")
            d = adapter_config.DT1D
            self.token_adapter = DT1DTokenAdapter(
                embed_dim=self.hidden_size,
                grid_size=(int(grid_size[0]), int(grid_size[1])),
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
        elif name in ("", "none", "null"):
            pass  # no adapter
        else:
            raise ValueError(f"Unknown adapter NAME='{adapter_config.NAME}'")

    def forward(self, x):
        # Standard ViT block
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)

        # Adapter path(s)
        name = str(getattr(self.adapter_config, "NAME", "")).lower()
        if name == "pfeiffer":
            adpt = self.adapter_downsample(x)
            adpt = self.adapter_act_fn(adpt)
            adpt = self.adapter_upsample(adpt)
            x = x + adpt
        elif name == "dt1d" and self.token_adapter is not None:
            # Token adapters contain their own residual injection rule.
            x = self.token_adapter(x)

        # MLP residual
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # attention qkv + out
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight   = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight   = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias   = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias   = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight);  self.attn.query.bias.copy_(query_bias)
            self.attn.key.weight.copy_(key_weight);      self.attn.key.bias.copy_(key_bias)
            self.attn.value.weight.copy_(value_weight);  self.attn.value.bias.copy_(value_bias)
            self.attn.out.weight.copy_(out_weight);      self.attn.out.bias.copy_(out_bias)

            # MLP
            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0   = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1   = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()
            self.ffn.fc1.weight.copy_(mlp_weight_0); self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1); self.ffn.fc2.bias.copy_(mlp_bias_1)

            # norms
            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class ADPT_Encoder(nn.Module):
    def __init__(self, config, vis, adapter_cfg, grid_size):
        super().__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

        num_layers = config.transformer["num_layers"]
        for _ in range(num_layers):
            self.layer.append(copy.deepcopy(ADPT_Block(config, vis, adapter_cfg, grid_size=grid_size)))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


def _infer_grid_size(img_size, config, embeddings):
    """
    Try several reliable ways to get (H, W) of patch grid, for patch-token adaptation.
    """
    # 1) If embeddings/patch embed exposes grid_size (common in ViT/timm)
    pe = getattr(embeddings, "patch_embeddings", None)
    gs = getattr(pe, "grid_size", None)
    if gs is not None:
        # can be tuple or torch.Size
        if isinstance(gs, (tuple, list)):
            return int(gs[0]), int(gs[1])
        if hasattr(gs, "__iter__"):
            gs = list(gs)
            return int(gs[0]), int(gs[1])

    # 2) From config.patches['size']
    if hasattr(config, "patches") and isinstance(config.patches, dict) and "size" in config.patches:
        ph, pw = config.patches["size"]
        return int(img_size // ph), int(img_size // pw)

    # 3) From position embeddings length (square grid assumption)
    pos = getattr(embeddings, "position_embeddings", None)
    if pos is not None:
        ntok = pos.shape[1]
        n_patches = ntok - 1  # minus cls token
        g = int(np.sqrt(n_patches))
        return g, g

    # Fallback (typical ViT-B/16)
    return int(img_size // 16), int(img_size // 16)


class ADPT_Transformer(nn.Module):
    def __init__(self, config, img_size, vis, adapter_cfg):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        grid_size = _infer_grid_size(img_size, config, self.embeddings)
        self.encoder = ADPT_Encoder(config, vis, adapter_cfg, grid_size)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class ADPT_VisionTransformer(nn.Module):
    def __init__(self, model_type, img_size=224, num_classes=21843, vis=False, adapter_cfg=None):
        super().__init__()
        config = CONFIGS[model_type]
        self.num_classes = num_classes
        self.classifier = config.classifier

        self.transformer = ADPT_Transformer(config, img_size, vis, adapter_cfg)
        self.head = Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])
        return (logits, attn_weights) if vis else logits

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            # position embedding (resize if needed)
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s", posemb.size(), posemb_new.size())
                ntok_new = posemb_new.size(1)
                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                logger.info("load_pretrained: grid-size from %s to %s", gs_old, gs_new)
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # per-block weights
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            # hybrid stem (if used)
            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias   = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)
                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)
