# -*- coding: utf-8 -*-
"""Lớp bọc encoder — giao diện CHUNG giữa một backbone và mọi đầu đọc (PLAN_89 §2).

"Lớp bọc" = một lớp mỏng quanh mô hình gốc: KHÔNG đổi trọng số hay phép tính của mô hình, chỉ (1) đổi chuẩn hoá ảnh từ chuẩn hoá của bên gọi sang
chuẩn hoá gốc của mô hình, (2) chạy mô hình ở độ phân giải gốc, (3) trả đặc trưng theo một khuôn duy nhất cho mọi encoder:

    get_intermediate_layers(x, n, return_class_token=True) -> tuple n phần tử, mỗi phần tử (patch [B, T, D], vector toàn ảnh [B, D])

của n tầng cuối, MỌI token đã qua norm cuối của mô hình (= get_intermediate_layers(norm=True) của DINOv2 — giao diện duy nhất mà linear_glori.py
của CheXFound gọi). Vector toàn ảnh = CLS hoặc TB token patch tuỳ spec.pool (kiểu gộp gốc của mô hình).

Chỉ dùng cú pháp / API có ở Python 3.9 + torch 2.0 (môi trường thư viện của CheXFound)."""
import torch
import torch.nn as nn

from .spec import IMAGENET_MEAN, IMAGENET_STD


class EncoderWrapper(nn.Module):
    '''Lớp cha. Lớp con đặt self.spec, self.nap (bản ghi cổng nạp trọng số) và hiện thực _blocks(x_goc, n) -> list n tensor [B, tiền tố + T, D] của n
       tầng cuối, đã qua norm cuối, token tiền tố (CLS / register) đứng trước; self.n_prefix = số token tiền tố.
       input_mean / input_std: chuẩn hoá mà BÊN GỌI đã áp lên ảnh (mặc định ImageNet = biến đổi của mã CheXFound).'''
    n_prefix = 1

    def __init__(self, spec, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD):
        super().__init__()
        self.spec = spec
        self.nap = None
        # x_goc = ((x * std_vao + mean_vao) - mean_goc) / std_goc = x * scale + shift
        self.register_buffer('scale', torch.tensor([i / r for i, r in zip(input_std, spec.std)]).view(1, 3, 1, 1))
        self.register_buffer('shift', torch.tensor([(m - rm) / r for m, rm, r in zip(input_mean, spec.mean, spec.std)]).view(1, 3, 1, 1))

    def to_native(self, x):
        return x * self.scale.to(x.dtype) + self.shift.to(x.dtype)

    def _blocks(self, x, n):
        raise NotImplementedError

    def get_intermediate_layers(self, x, n_last_blocks, return_class_token=True):
        s = self.spec.image_size
        assert x.dim() == 4 and x.shape[1] == 3 and x.shape[-2] == s and x.shape[-1] == s, \
            f'{self.spec.name}: anh {tuple(x.shape)} khac do phan giai goc {s} (khong noi suy vi tri) -> DUNG'
        outs = []
        for h in self._blocks(self.to_native(x), int(n_last_blocks)):
            patch = h[:, self.n_prefix:]
            assert patch.shape[1] == self.spec.n_patches and patch.shape[2] == self.spec.embed_dim, (self.spec.name, tuple(h.shape))
            outs.append((patch, h[:, 0] if self.spec.pool == 'cls' else patch.mean(1)))
        return tuple(outs) if return_class_token else tuple(p for p, _ in outs)

    def forward(self, x):
        '''Vector toàn ảnh của tầng cuối.'''
        return self.get_intermediate_layers(x, 1)[0][1]
