# -*- coding: utf-8 -*-
"""Tạo / chạy 4 loại đầu đọc của Ver2 (PLAN_89 §3) trên đặc trưng của lớp bọc: blocks = 4 tầng cuối x (token patch [B, T, D], vector toàn ảnh [B, D]).

    lin1   Linear-1    LinearClassifier của CheXFound trên vector toàn ảnh TẦNG CUỐI (use_n_blocks=1, không avgpool) — đầu thêm ở 89
    lin4   Linear-4    LinearClassifier của CheXFound trên 4 vector nối lại (mặc định của họ) — = đầu 'lin' của 86
    mldec  ML-Decoder  đầu của 85f trên token patch tầng cuối (d 512, 8 head, FFN 1024, dropout 0,1) — = đầu 'ml' của 86
    glori  GLoRI       mã công khai của CheXFound trên token patch 4 tầng cuối + nối vector toàn ảnh (decoder 768, cat_cls) — = đầu 'glori' của 86

build_head / run_head giữ cách làm của vindr_protocol/heads.py (86): khởi tạo CHỈ phụ thuộc hạt của đầu (RNG riêng)."""
import torch

from ...heads.glori import GLoRI
from ...heads.linear import LinearClassifier
from ...heads.mldecoder import MLDecoder


def build_head(kind, seed, dim, num_classes, device, protocol):
    '''Đầu mới; khởi tạo CHỈ phụ thuộc `seed` (RNG riêng, không động RNG chung).'''
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind == 'lin1':
            m = LinearClassifier(dim, use_n_blocks=1, use_avgpool=False, num_classes=num_classes)
        elif kind == 'lin4':
            m = LinearClassifier(protocol.n_last_blocks * dim, use_n_blocks=protocol.n_last_blocks, use_avgpool=False, num_classes=num_classes)
        elif kind == 'glori':
            m = GLoRI(num_classes=num_classes, decoder_embedding=protocol.glori_dim, initial_num_features=protocol.n_last_blocks * dim,
                      use_n_blocks=protocol.n_last_blocks, cat_cls=True)
        elif kind == 'mldec':
            m = MLDecoder(num_classes, dim, d=protocol.mldecoder_dim, nhead=8, ff=1024, drop=0.1)
        else:
            raise ValueError(kind)
    return m.to(device)


def run_head(kind, model, blocks, last_patch):
    '''-> logit [B, C]. Linear / GLoRI nhận list các view như mã gốc ([blocks]); ML-Decoder nhận patch tầng cuối (float32).'''
    return model(last_patch.float())[0] if kind == 'mldec' else model([blocks])
