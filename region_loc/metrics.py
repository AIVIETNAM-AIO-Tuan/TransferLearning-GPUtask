# -*- coding: utf-8 -*-
"""Gộp số đo theo khoá (vùng giải phẫu / dấu hiệu) + bootstrap theo BỆNH NHÂN.

Mỗi đơn vị (một hộp của C1, một cặp (ảnh, dấu hiệu) của C2) có: giá trị, khoá, bệnh nhân. Hai cách gộp báo cạnh nhau:
  pooled = trung bình mọi đơn vị (cách của nhật ký huấn luyện: `luong_iou`)
  macro  = trung bình của trung bình từng khoá
Bootstrap rút lại bệnh nhân (có hoàn lại). Lần rút mà một khoá không còn đơn vị nào bị LOẠI và được đếm (tập khoá cố định — không để nanmean đổi số khoá;
bộ nhớ dự án "bootstrap trên tập lớp cố định"). Giá trị NaN (vd bản đồ toàn 0) bị bỏ khỏi cả tổng lẫn số đếm và được đếm riêng."""
import numpy as np


def summarize(values, keys, patients, n_boot, seed, key_names=None):
    '''values [n] float (NaN = bỏ), keys [n] int, patients [n] -> dict(pooled, macro, ci_pooled, ci_macro, per_key{khoá: (trung bình, n)}, n, n_nan,
       n_patients, n_boot_used, n_boot_dropped). CI95 = phân vị 2,5 / 97,5 của các lần rút hợp lệ.'''
    v = np.asarray(values, 'float64'); k = np.asarray(keys); p = np.asarray(patients)
    assert v.shape == k.shape == p.shape and v.ndim == 1 and v.size > 0, (v.shape, k.shape, p.shape)
    ok = np.isfinite(v)
    n_nan = int((~ok).sum())
    v, k, p = v[ok], k[ok], p[ok]
    assert v.size > 0, 'moi gia tri deu NaN -> DUNG'
    uk, ki = np.unique(k, return_inverse=True)
    up, pi = np.unique(p, return_inverse=True)
    S = np.zeros((len(up), len(uk))); N = np.zeros((len(up), len(uk)))
    np.add.at(S, (pi, ki), v); np.add.at(N, (pi, ki), 1.0)
    per = S.sum(0) / N.sum(0)
    out = dict(pooled=float(S.sum() / N.sum()), macro=float(per.mean()), n=int(v.size), n_nan=n_nan, n_patients=int(len(up)),
               per_key={(key_names[int(a)] if key_names is not None else int(a)): (float(m), int(c)) for a, m, c in zip(uk, per, N.sum(0))})
    rng = np.random.default_rng(seed)
    pooled, macro, dropped = [], [], 0
    for _ in range(int(n_boot)):
        w = np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up)).astype('float64')
        s, c = w @ S, w @ N
        if (c <= 0).any():
            dropped += 1
            continue
        pooled.append(s.sum() / c.sum()); macro.append((s / c).mean())
    out.update(n_boot_used=len(pooled), n_boot_dropped=dropped,
               ci_pooled=tuple(float(x) for x in np.percentile(pooled, [2.5, 97.5])) if pooled else None,
               ci_macro=tuple(float(x) for x in np.percentile(macro, [2.5, 97.5])) if macro else None)
    return out


def share_at(values, thr):
    v = np.asarray(values, 'float64')
    v = v[np.isfinite(v)]
    return float((v >= thr).mean()) if v.size else float('nan')


def fmt(s, digits=3):
    '''Một dòng: pooled [CI] | macro [CI] | n.'''
    f = lambda x: 'nan' if x is None or not np.isfinite(x) else f'{x:.{digits}f}'
    ci = lambda c: '[--]' if c is None else f'[{f(c[0])}; {f(c[1])}]'
    return (f"gop {f(s['pooled'])} {ci(s['ci_pooled'])} | TB theo khoa {f(s['macro'])} {ci(s['ci_macro'])} | n {s['n']} / {s['n_patients']} benh nhan"
            + (f" | NaN {s['n_nan']}" if s['n_nan'] else '') + (f" | bo {s['n_boot_dropped']} lan rut thieu khoa" if s['n_boot_dropped'] else ''))
