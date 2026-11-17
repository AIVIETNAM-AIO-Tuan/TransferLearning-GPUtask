# -*- coding: utf-8 -*-
"""Bản đồ trên lưới token (37 x 37, thứ tự theo HÀNG: chỉ số y * lưới + x — thứ tự token patch của Dinov2, như b_src.mat_na_hop) và hai số đo định vị.

  box_mask      hộp chuẩn (hợp của các hộp) -> mặt nạ ô có TÂM nằm trong hộp; hộp nhỏ hơn một ô -> lấy ô chứa tâm hộp
  paint         điểm từng vùng -> bản đồ: mỗi ô nhận điểm LỚN NHẤT trong các vùng có hộp dự đoán phủ tâm ô (vùng lồng nhau: phổi chứa các phần phổi)
  attention_map hàng attention của một query lớp trên [1369 patch ; 36 token vùng] -> bản đồ patch chuẩn hoá lại + phần attention rơi vào token vùng
  hit           "trúng điểm": tỉ lệ các ô đạt giá trị lớn nhất nằm trong mặt nạ (một ô lớn nhất -> 0 / 1; nhiều ô bằng nhau -> tỉ lệ)
  energy        "khối lượng trong hộp": tổng bản đồ trong mặt nạ / tổng bản đồ
Mức may rủi của cả hai số đo = tỉ lệ diện tích mặt nạ (area)."""
import numpy as np


def _centres(grid):
    return (np.arange(grid, dtype='float64') + 0.5) / grid


def box_mask(boxes, grid):
    '''boxes [m, 4] (x1, y1, x2, y2) chuẩn hoá -> bool [grid, grid] (hàng = y).'''
    c = _centres(grid)
    out = np.zeros((grid, grid), bool)
    for x1, y1, x2, y2 in np.asarray(boxes, 'float64').reshape(-1, 4):
        mx, my = (c >= x1) & (c <= x2), (c >= y1) & (c <= y2)
        if not mx.any():
            mx = np.zeros(grid, bool); mx[min(grid - 1, max(0, int((x1 + x2) / 2 * grid)))] = True
        if not my.any():
            my = np.zeros(grid, bool); my[min(grid - 1, max(0, int((y1 + y2) / 2 * grid)))] = True
        out |= my[:, None] & mx[None, :]
    return out


def paint(boxes, scores, grid):
    '''boxes [R, 4] xyxy chuẩn hoá (hộp DỰ ĐOÁN của từng vùng), scores [R] >= 0 -> float [grid, grid]; ô không vùng nào phủ = 0.'''
    c = _centres(grid)
    b = np.asarray(boxes, 'float64'); s = np.asarray(scores, 'float64')
    assert b.shape == (len(s), 4) and bool((s >= 0).all()), (b.shape, s.shape)
    mx = (c[None, :] >= b[:, 0:1]) & (c[None, :] <= b[:, 2:3])          # [R, grid]
    my = (c[None, :] >= b[:, 1:2]) & (c[None, :] <= b[:, 3:4])
    cover = my[:, :, None] & mx[:, None, :]                              # [R, grid, grid]
    return np.where(cover, s[:, None, None], 0.0).max(0)


def attention_map(row, grid):
    '''row [grid * grid + R] >= 0, tổng ~1 -> (bản đồ patch [grid, grid] chuẩn hoá tổng 1, phần attention ở R token vùng).'''
    r = np.asarray(row, 'float64')
    n = grid * grid
    assert r.ndim == 1 and r.size >= n and bool((r >= 0).all()), r.shape
    p, tot = r[:n], float(r.sum())
    return (p / max(float(p.sum()), 1e-12)).reshape(grid, grid), (float(r[n:].sum()) / max(tot, 1e-12))


def hit(m, mask):
    m = np.asarray(m, 'float64')
    top = m >= m.max() - 1e-12
    return float((top & mask).sum() / top.sum())


def energy(m, mask):
    m = np.asarray(m, 'float64')
    s = float(m.sum())
    return float((m * mask).sum() / s) if s > 0 else float('nan')


def area(mask):
    return float(np.asarray(mask, bool).mean())
