# -*- coding: utf-8 -*-
"""Đọc ảnh Shenzhen (PLAN_88 §4.3 (1), §4.7). Chỉ numpy / PIL — không import mã CheXFound (lớp ShenzhenL ở shenzhen_dataset.py).

`decode_png` = `Shenzhen.get_image_data` của họ (chexfound/data/datasets/shenzhen.py:44–51 @ 7e288d0) với ĐÚNG MỘT sửa: `convert('L')` trước `np.array`.
Lý do: 635 / 662 PNG Shenzhen là ảnh palette (mode P); `np.array` trên ảnh P trả CHỈ SỐ palette, không phải mức xám. 334 ảnh có palette ĐẢO
(chỉ số i -> xám 255 − i; 254 bình thường / 80 lao) -> đọc kiểu của họ ra âm bản, lệch theo nhãn (kiểm 10/11). 27 ảnh RGB (ba kênh bằng nhau) và ảnh
palette thuận cho kết quả trùng từng điểm ảnh với đường của họ (`palette_audit` trên tệp thật: 328 trùng + 334 bù)."""
import collections

import numpy as np
from PIL import Image


def decode_png(path):
    '''Ảnh Shenzhen -> PIL RGB uint8: convert('L') (áp palette -> mức xám thật) rồi min-max -> uint8 -> RGB đúng như mã họ.'''
    img = Image.open(path).convert('L')
    # phần dưới chép nguyên văn shenzhen.py:46–50 (rescale to 0-255 and 8-bit depth)
    img = np.array(img)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype("uint8")
    img = Image.fromarray(img).convert(mode="RGB")
    return img


def decode_png_source(path):
    '''= get_image_data của họ NGUYÊN VĂN (không convert('L')) — CHỈ để kiểm / thống kê sai khác; không dùng để học.'''
    img = Image.open(path)
    img = np.array(img)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype("uint8")
    img = Image.fromarray(img).convert(mode="RGB")
    return img


def compare_decodes(path):
    '''Một ảnh: đường sửa so với đường của họ -> 'trung' (từng điểm ảnh) | 'bu' (tổng hai ảnh ∈ {254, 255} mọi điểm: palette đảo) | 'khac';
       kèm mode gốc của tệp.'''
    with Image.open(path) as im:
        mode = im.mode
    a = np.asarray(decode_png(path), dtype=np.int16)
    b = np.asarray(decode_png_source(path), dtype=np.int16)
    if a.shape == b.shape and np.array_equal(a, b):
        return 'trung', mode
    s = a + b
    if a.shape == b.shape and int(s.min()) >= 254 and int(s.max()) <= 255:
        return 'bu', mode
    return 'khac', mode


def palette_audit(paths, labels=None):
    '''Thống kê sai khác giải mã trên nhiều ảnh (đọc mỗi tệp 2 lần; ~0,3 s / ảnh Shenzhen). labels (0 / 1, cùng thứ tự) -> đếm 'bu' theo nhãn.
       -> dict(n, trung, bu, khac, mode {mode: số}, bu_theo_nhan {nhãn: số}, ten_khac [tên tệp]).'''
    import os
    out = collections.Counter()
    modes, by_label, odd = collections.Counter(), collections.Counter(), []
    for i, p in enumerate(paths):
        kind, mode = compare_decodes(p)
        out[kind] += 1
        modes[mode] += 1
        if kind == 'bu' and labels is not None:
            by_label[int(labels[i])] += 1
        if kind == 'khac':
            odd.append(os.path.basename(p))
    return dict(n=len(paths), trung=out['trung'], bu=out['bu'], khac=out['khac'], mode=dict(modes), bu_theo_nhan=dict(by_label), ten_khac=odd)
