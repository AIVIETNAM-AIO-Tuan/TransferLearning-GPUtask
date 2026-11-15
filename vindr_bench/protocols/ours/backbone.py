# -*- coding: utf-8 -*-
"""Backbone đông cứng của Ver2 = LỚP BỌC chung (encoders/base.py, cùng lớp bọc với Ver1 — PLAN_89 §1 "Backbone") chạy dưới autocast của lượt.

features(x) -> (blocks, patch tầng cuối): blocks = 4 tầng cuối x (token patch [B, T, D], vector toàn ảnh [B, D]) — cùng khuôn với features của
vindr_protocol (86) nên vòng học / đầu đọc chép từ 86 dùng nguyên. Ảnh vào: chuẩn hoá ImageNet của biến đổi mã CheXFound (như Ver1); lớp bọc tự đổi
sang chuẩn hoá gốc và từ chối kích thước khác độ phân giải gốc."""
from dataclasses import dataclass

import torch

from ... import encoders
from ...encoders.spec import IMAGENET_MEAN, IMAGENET_STD

N_BLOCKS = 4


@dataclass(frozen=True)
class InputSpec:
    '''Ảnh vào của lớp bọc (cho data.loaders.build_transforms): độ phân giải gốc của encoder + chuẩn hoá ImageNet (biến đổi của mã CheXFound).'''
    name: str
    image_size: int
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD


@dataclass
class Backbone:
    enc: object                   # lớp bọc đã nạp trọng số, eval(), trên `device`
    device: str
    amp: object                   # dtype autocast của backbone trên GPU (bf16); CPU: không autocast

    @property
    def name(self):
        return self.enc.spec.name

    @property
    def spec(self):
        return InputSpec(self.enc.spec.name, self.enc.spec.image_size)

    @property
    def dim(self):
        return self.enc.spec.embed_dim

    @torch.no_grad()
    def features(self, x, autocast=True):
        '''ảnh [B, 3, S, S] (chuẩn hoá ImageNet) -> (blocks, patch tầng cuối). autocast=False: fp32 (chỉ cho cổng số học).'''
        if autocast and self.device == 'cuda':
            with torch.autocast('cuda', dtype=self.amp):
                blocks = self.enc.get_intermediate_layers(x, N_BLOCKS)
        else:                                       # CPU (bộ kiểm) hoặc đường đối chiếu fp32: không autocast
            blocks = self.enc.get_intermediate_layers(x, N_BLOCKS)
        return blocks, blocks[-1][0]


def load(encoder, device, amp, **kw):
    '''encoders.build (cổng nạp trọng số: từng tensor = tệp) -> eval() trên `device`. Cờ requires_grad giữ như lớp bọc dựng (đặc trưng tính dưới
       no_grad; không đổi đường tính của mô hình).'''
    enc = encoders.build(encoder, **kw).to(device).eval()
    return Backbone(enc, device, amp)
