# -*- coding: utf-8 -*-
"""Hình học hộp: ảnh gốc -> khung ảnh của mô hình; IoU.

`hop_sang_letterbox` và `iou` chép NGUYÊN VĂN từ experiments/78_chi_anh/khoi/khoi_vung.py (tests/test_region_loc.py so cây cú pháp với tệp nguồn).

Khung của mô hình: ảnh trong kho 518 là ảnh letterbox 1022 (28c) thu về 518, nên vùng ảnh thật trong khung 518 = hình chữ nhật letterbox 1022 nhân
518 / 1022 (78g, hàm vẽ: `v * SIZE / 1022 for v in hop_sang_letterbox(0, 0, W, H, W, H, 1022)`). Toạ độ chuẩn hoá [0, 1] của khung vì vậy bằng toạ độ
letterbox 1022 chia 1022 — đúng khung mà hộp (cx, cy, w, h) của luồng vùng dùng."""
import numpy as np

from .config import LETTERBOX


# ==== NGUYÊN VĂN khoi_vung.py (78) ====
def hop_sang_letterbox(x1, y1, x2, y2, W, H, SIZE):
    '''Hop toa do ANH GOC (W x H) -> toa do anh letterbox SIZE x SIZE, CUNG HINH HOC voi `_letterbox` cua 28c: scale theo
       canh DAI s = SIZE / max(W, H), nw = max(1, round(W s)), nh = max(1, round(H s)), dan CAN GIUA tai
       ((SIZE - nw) // 2, (SIZE - nh) // 2). Nhan mang numpy (vecto hoa). test_78f so voi chinh ham 28c tren anh that gia.'''
    W = np.asarray(W, 'float64'); H = np.asarray(H, 'float64')
    s = SIZE / np.maximum(W, H)
    nw = np.maximum(1, np.round(W * s)); nh = np.maximum(1, np.round(H * s))
    ox = (SIZE - nw) // 2; oy = (SIZE - nh) // 2
    fx, fy = nw / W, nh / H
    return (ox + np.asarray(x1) * fx, oy + np.asarray(y1) * fy, ox + np.asarray(x2) * fx, oy + np.asarray(y2) * fy)


def iou(a, b):
    '''IoU cua cac hop [n, 4] (x1, y1, x2, y2) tung cap.'''
    a = np.asarray(a, 'float64'); b = np.asarray(b, 'float64')
    iw = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    gi = iw * ih
    dt = lambda h: np.clip(h[:, 2] - h[:, 0], 0, None) * np.clip(h[:, 3] - h[:, 1], 0, None)
    return gi / np.maximum(dt(a) + dt(b) - gi, 1e-9)


# ==== của gói ====
def to_frame(x1, y1, x2, y2, W, H):
    '''Hộp toạ độ ẢNH GỐC (W x H điểm ảnh) -> [n, 4] (x1, y1, x2, y2) chuẩn hoá [0, 1] trong khung ảnh của mô hình; cắt về [0, 1].'''
    p = np.stack([np.asarray(v, 'float64') for v in hop_sang_letterbox(x1, y1, x2, y2, W, H, LETTERBOX)], 1) / float(LETTERBOX)
    return np.clip(p, 0.0, 1.0)


def cxcywh_to_xyxy(b):
    '''[..., 4] (cx, cy, w, h) -> [..., 4] (x1, y1, x2, y2), numpy float64.'''
    b = np.asarray(b, 'float64')
    return np.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2, b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], -1)


def letterbox_224_error(x1, y1, x2, y2, W, H, coords_224):
    '''Độ lệch lớn nhất (px, khung 224) giữa letterbox 224 của hộp gốc và toạ độ 224 ghi trong tệp ImaGenome, từng hộp — cổng "đúng W / H + đúng hình học"
       (cùng phép kiểm của 78f C3).'''
    p = np.stack([np.asarray(v, 'float64') for v in hop_sang_letterbox(x1, y1, x2, y2, W, H, 224)], 1)
    return np.abs(p - np.asarray(coords_224, 'float64')).max(1)
