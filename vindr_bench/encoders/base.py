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

# Backbone đông cứng chạy theo KHÚC tối đa FORWARD_CHUNK ảnh (0.2.1, 13/11): lô 256 của lượt học -> 4 khúc 64 = lô mỗi GPU trong README CheXFound
# (4 GPU × 64). Lý do đo được: 89v1_raddino (A100 80 GB) — transformers 4.36.2 chỉ có attention eager cho Dinov2, ma trận attention 256 × 12 × 1.370²
# cần ~43 GiB tạm mỗi tầng (softmax fp32 21,48 GiB) -> Linear-4 OOM. Mỗi ảnh tính độc lập (ViT ở eval không trộn ảnh) nên phép tính không đổi;
# bit có thể lệch ở mức làm tròn do hình ma trận khác (cùng loại lệch giữa lô học 256 và lô chấm 32 của mã họ).
FORWARD_CHUNK = 64


class EncoderWrapper(nn.Module):
    '''Lớp cha. Lớp con đặt self.spec, self.nap (bản ghi cổng nạp trọng số) và hiện thực _blocks(x_goc, n) -> list n tensor [B, tiền tố + T, D] của n
       tầng cuối, đã qua norm cuối, token tiền tố (CLS / register) đứng trước; self.n_prefix = số token tiền tố.
       input_mean / input_std: chuẩn hoá mà BÊN GỌI đã áp lên ảnh (mặc định ImageNet = biến đổi của mã CheXFound).'''
    n_prefix = 1

    def __init__(self, spec, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD):
        super().__init__()
        self.spec = spec
        self.nap = None
        self.forward_chunk = FORWARD_CHUNK
        # x_goc = ((x * std_vao + mean_vao) - mean_goc) / std_goc = x * scale + shift
        self.register_buffer('scale', torch.tensor([i / r for i, r in zip(input_std, spec.std)]).view(1, 3, 1, 1))
        self.register_buffer('shift', torch.tensor([(m - rm) / r for m, rm, r in zip(input_mean, spec.mean, spec.std)]).view(1, 3, 1, 1))

    def to_native(self, x):
        return x * self.scale.to(x.dtype) + self.shift.to(x.dtype)

    def _blocks(self, x, n):
        raise NotImplementedError

    def _blocks_chunked(self, x, n):
        '''_blocks theo khúc <= self.forward_chunk ảnh (đổi chuẩn hoá từng khúc). Không tính gradient: cấp sẵn tensor kết quả cả lô rồi chép từng khúc
           (chỉ giữ một khúc trung gian); có gradient (không dùng trong giao thức): nối bằng torch.cat. -> list n tensor [B, tiền tố + T, D].'''
        B, c = x.shape[0], int(self.forward_chunk)
        if B <= c:
            return self._blocks(self.to_native(x), n)
        if torch.is_grad_enabled():
            parts = [self._blocks(self.to_native(x[i:i + c]), n) for i in range(0, B, c)]
            return [torch.cat([p[k] for p in parts]) for k in range(len(parts[0]))]
        outs = None
        for i in range(0, B, c):
            hs = self._blocks(self.to_native(x[i:i + c]), n)
            if outs is None:
                outs = [h.new_empty((B,) + tuple(h.shape[1:])) for h in hs]
            for o, h in zip(outs, hs):
                o[i:i + h.shape[0]] = h
            del hs
        return outs

    def get_intermediate_layers(self, x, n_last_blocks, return_class_token=True):
        s = self.spec.image_size
        assert x.dim() == 4 and x.shape[1] == 3 and x.shape[-2] == s and x.shape[-1] == s, \
            f'{self.spec.name}: anh {tuple(x.shape)} khac do phan giai goc {s} (khong noi suy vi tri) -> DUNG'
        outs = []
        for h in self._blocks_chunked(x, int(n_last_blocks)):
            patch, vec = self._patch_and_vector(h)
            assert patch.shape[1] == self.spec.n_patches and patch.shape[2] == self.spec.embed_dim, (self.spec.name, tuple(h.shape))
            outs.append((patch, vec))
        return tuple(outs) if return_class_token else tuple(p for p, _ in outs)

    def _patch_and_vector(self, h):
        '''h [B, tiền tố + T, D] (đầu ra _blocks) -> (token patch, vector toàn ảnh) theo spec.pool: 'cls' = token đầu; 'mean_patch' = TB token patch
           (đã qua norm cuối). Lớp con đổi khi đường chính thức khác (EVA-X 'norm_of_mean': _blocks trả token THÔ, norm áp sau).'''
        patch = h[:, self.n_prefix:]
        return patch, (h[:, 0] if self.spec.pool == 'cls' else patch.mean(1))

    def forward(self, x):
        '''Vector toàn ảnh của tầng cuối.'''
        return self.get_intermediate_layers(x, 1)[0][1]
