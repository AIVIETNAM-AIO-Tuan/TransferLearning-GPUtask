# -*- coding: utf-8 -*-
"""Tháp ảnh CLIP định dạng OpenAI (CheXzero, PubMedCLIP — cả hai là ViT-B/32 224, state dict kiểu `clip.load`).

Lớp mô hình chép từ model.py của rajpurkarlab/CheXzero @ 5c341db0fe0db2f663a2c136c7120a8b303f1d11 (= mã CLIP của OpenAI, giấy phép MIT): LayerNorm tính
fp32, QuickGELU, ResidualAttentionBlock (nn.MultiheadAttention, khuôn LND), Transformer, VisualTransformer — chỉ phần ảnh. Cấu hình suy từ hình tensor
như build_model của họ. Trọng số tệp gốc trộn fp16 / fp32 (convert_weights của CLIP) -> nạp vào tham số fp32 (đổi kiểu chính xác); dưới autocast fp16
của mã CheXFound, phép nhân lại chạy fp16 trên đúng các giá trị fp16 gốc.

Đặc trưng: n khối cuối, MỌI token qua ln_post (quy ước của ta — CLIP gốc chỉ áp ln_post lên CLS); vector toàn ảnh = ln_post(CLS) TRƯỚC phép chiếu `proj`
(đường chính thức: VisualTransformer.forward = ln_post(x[:, 0]) @ proj). Chỉ cú pháp / API có ở Python 3.9 + torch 2.0."""
from collections import OrderedDict

import torch
from torch import nn

from .base import EncoderWrapper

PREFIX = 'visual.'


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


# ==== phần của 89 ====
def visual_config(sd):
    '''Cấu hình tháp ảnh suy từ state dict định dạng OpenAI (như build_model của CLIP / CheXzero).'''
    width = int(sd[PREFIX + 'conv1.weight'].shape[0])
    patch = int(sd[PREFIX + 'conv1.weight'].shape[-1])
    layers = len({k.split('.')[3] for k in sd if k.startswith(PREFIX + 'transformer.resblocks.') and k.endswith('.attn.in_proj_weight')})
    grid = round((sd[PREFIX + 'positional_embedding'].shape[0] - 1) ** 0.5)
    return dict(input_resolution=patch * grid, patch_size=patch, width=width, layers=layers, heads=width // 64,
                output_dim=int(sd[PREFIX + 'proj'].shape[1]))


def build_visual(sd):
    '''VisualTransformer fp32 + nạp CHẶT các khoá `visual.*` (bỏ tiền tố). -> (mô hình eval, cấu hình).'''
    cfg = visual_config(sd)
    vis = VisualTransformer(**cfg)
    vsd = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}
    want = set(vis.state_dict())
    assert set(vsd) == want, f'khoa visual.* khac mo hinh: thieu {sorted(want - set(vsd))[:5]} / thua {sorted(set(vsd) - want)[:5]} -> DUNG'
    vis.load_state_dict({k: v.float() for k, v in vsd.items()}, strict=True)
    return vis.float().eval(), cfg


class OpenAIClipWrapper(EncoderWrapper):
    '''Lớp bọc chung: lớp con đặt self.visual (VisualTransformer đã nạp) + self.spec + self.nap.'''
    n_prefix = 1

    def _blocks(self, x, n):
        v = self.visual
        x = v.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = v.ln_pre(x + v.positional_embedding.to(x.dtype))
        x = x.permute(1, 0, 2)
        blocks = v.transformer.resblocks
        outs = []
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i >= len(blocks) - n:
                outs.append(x)
        return [v.ln_post(h.permute(1, 0, 2)) for h in outs]
