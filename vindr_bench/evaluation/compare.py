# -*- coding: utf-8 -*-
"""So nhiều lượt trên CÙNG tập test (PLAN_89 §7.2–§7.4): điểm + bootstrap ghép cặp theo ảnh + khoảng đồng thời max-T.

    set_stats   một bộ dự đoán [N, 28]: điểm 4 panel (AP bậc thang, AUPRC hình thang, AUROC) + mAP21 / mAP7 của TỪNG lần rút (cùng dãy rút cho mọi bộ:
                trọng số metrics.bootstrap_weights(N, 2.000, hạt 8502)) + AP21 trên hai nửa test (hạt 8501)
    make_cell   một Ô = (phiên bản, kho, encoder, đầu, nhánh): Ver1 = 1 bộ; Ver2 = 5 bộ (5 hạt) -> số chính = TRUNG BÌNH metric của các hạt (mỗi lần rút
                tính từng hạt rồi lấy TB, không gộp ảnh của 5 hạt), SD mẫu giữa hạt, ensemble (TB xác suất) báo kèm
    diff        hiệu ghép cặp A - B trên cùng lần rút (hai ô phải cùng nhãn test theo thứ tự) -> hiệu điểm, CI95 danh nghĩa, hiệu trên hai nửa, nhãn theo
                dạng luật §7.4 (hiệu >= 0,01 VÀ cận dưới > 0 VÀ dương ở cả hai nửa)
    max_t       khoảng đồng thời 95 % cho một HỌ phép so (§7.3): T_b = max_c |d_cb - d_c| / SE_c, q = phân vị 95 % -> d_c ± q · SE_c

Lần rút thiếu ca dương ở bất kỳ lớp nào của panel 21 / 7 lớp bị LOẠI (NaN) và đếm — tập lớp cố định, không để nanmean đổi số lớp."""
import hashlib

import numpy as np

from . import metrics as M
from . import panels as P

MARGIN = 0.01                      # ngưỡng hiệu của luật §7.4
_W = {}


def weights(n_images, n_boot=P.N_BOOT, seed=P.BOOT_SEED):
    k = (int(n_images), int(n_boot), int(seed))
    if k not in _W:
        _W[k] = M.bootstrap_weights(*k)
    return _W[k]


def target_key(target):
    '''Dấu của ma trận nhãn test THEO THỨ TỰ — hai bộ cùng dấu mới ghép cặp theo ảnh được.'''
    t = np.ascontiguousarray(np.asarray(target), dtype=np.int8)
    return hashlib.md5(t.tobytes()).hexdigest()[:12] + f'_{t.shape[0]}x{t.shape[1]}'


def draws_ap(preds, target, n_boot=P.N_BOOT, seed=P.BOOT_SEED):
    '''AP bậc thang từng lớp ở từng lần rút -> [n_boot, C] (NaN = lớp không có ca dương trong lần rút).'''
    pre = M.sort_once(target, preds)
    return np.stack([M.ap_weighted(pre, w) for w in weights(len(target), n_boot, seed)])


def set_stats(preds, target, cols, n_boot=P.N_BOOT, seed=P.BOOT_SEED, half_seed=P.HALF_SEED):
    '''Một bộ dự đoán -> dict(diem {panel: {ap, auprc_thang, auroc}}, ap21_lop, m21 / m7 [n_boot] (NaN = lần rút bị loại), nua [AP21 nửa 0, nửa 1]).'''
    preds, target = np.asarray(preds, dtype=np.float32), np.asarray(target)
    assert preds.shape == target.shape and np.isfinite(preds).all(), 'du doan / nhan KHONG khop hoac khong huu han -> DUNG'
    pts = {p: P.point(preds, target, cols[p]) for p in P.PANELS}
    D = draws_ap(preds, target, n_boot, seed)
    m = {p: np.where(np.isfinite(D[:, cols[p]]).all(1), np.nan_to_num(D[:, cols[p]]).mean(1), np.nan) for p in ('p21', 'p7')}
    half = M.halves(len(target), half_seed)
    return dict(diem={p: {k: float(pts[p][k]) for k in ('ap', 'auprc_thang', 'auroc')} for p in P.PANELS},
                ap21_lop=[float(v) for v in pts['p21']['ap_lop']], m21=m['p21'], m7=m['p7'],
                nua=[float(P.point(preds[half == h], target[half == h], cols['p21'])['ap']) for h in (0, 1)])


def _job(a):
    return set_stats(*a)


def many_stats(jobs, workers=1):
    '''jobs = [(preds, target, cols, n_boot, seed, half_seed)] -> [set_stats]. workers > 1: chia tiến trình (mỗi tiến trình tự sinh lại cùng dãy rút).'''
    if workers > 1 and len(jobs) > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(_job, jobs, chunksize=1))
        except Exception:                           # không tạo được tiến trình con (môi trường hạn chế) -> chạy lần lượt, kết quả như nhau
            pass
    return [_job(j) for j in jobs]


def _ci(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float('nan')] * 2


def make_cell(stats, ensemble=None):
    '''stats = [set_stats của từng hạt] (Ver1: 1). ensemble = điểm của TB xác suất (dict panel) hoặc None. -> Ô.'''
    k = len(stats)
    mean = lambda f: float(np.mean([f(s) for s in stats]))
    sd = lambda f: float(np.std([f(s) for s in stats], ddof=1)) if k > 1 else None
    diem = {p: {m: mean(lambda s: s['diem'][p][m]) for m in ('ap', 'auprc_thang', 'auroc')} for p in P.PANELS}
    sds = {p: {m: sd(lambda s: s['diem'][p][m]) for m in ('ap', 'auprc_thang', 'auroc')} for p in P.PANELS}
    m21, m7 = np.mean([s['m21'] for s in stats], 0), np.mean([s['m7'] for s in stats], 0)       # NaN ở một hạt -> NaN (cùng nhãn nên cùng lần rút bị loại)
    return dict(so_hat=k, diem=diem, sd=sds, tung_hat=[s['diem']['p21']['ap'] for s in stats], m21=m21, m7=m7, ci21=_ci(m21), ci7=_ci(m7),
                loai21=int(np.isnan(m21).sum()), loai7=int(np.isnan(m7).sum()), nua=[mean(lambda s: s['nua'][h]) for h in (0, 1)],
                ap21_lop=[float(v) for v in np.mean([s['ap21_lop'] for s in stats], 0)], ensemble=ensemble)


def verdict(d, lo, hi, nua, margin=MARGIN):
    '''Dạng luật §7.4 trên một hiệu A - B: 'hon' | 'kem' | 'khong phan biet'.'''
    if d >= margin and lo > 0 and all(v > 0 for v in nua): return 'hon'
    if d <= -margin and hi < 0 and all(v < 0 for v in nua): return 'kem'
    return 'khong phan biet'


def diff(a, b):
    '''Hiệu ghép cặp ô A - ô B trên mAP21 (cùng lần rút). -> dict(d, ci (danh nghĩa), nua, rut [n_boot], se, ket_luan (danh nghĩa)).'''
    d = a['diem']['p21']['ap'] - b['diem']['p21']['ap']
    r = a['m21'] - b['m21']
    ci = _ci(r)
    nua = [a['nua'][h] - b['nua'][h] for h in (0, 1)]
    ok = r[np.isfinite(r)]
    return dict(d=float(d), ci=ci, nua=nua, rut=r, se=float(ok.std(ddof=1)) if len(ok) > 1 else float('nan'), ket_luan=verdict(d, ci[0], ci[1], nua))


def max_t(diffs, level=95.0):
    '''Khoảng đồng thời cho một họ phép so (list kết quả diff). -> (q, [dict(d, ci_dong_thoi, ket_luan)]) theo đúng thứ tự vào.'''
    if not diffs:
        return float('nan'), []
    R = np.stack([x['rut'] for x in diffs])
    R = R[:, np.isfinite(R).all(0)]
    d = np.array([x['d'] for x in diffs])
    se = R.std(1, ddof=1)
    q = float(np.percentile(np.max(np.abs(R - d[:, None]) / se[:, None], 0), level))
    out = []
    for x, dc, s in zip(diffs, d, se):
        lo, hi = float(dc - q * s), float(dc + q * s)
        out.append(dict(d=float(dc), ci_dong_thoi=[lo, hi], ket_luan=verdict(dc, lo, hi, x['nua'])))
    return q, out


def spearman(x, y):
    '''Tương quan hạng (không hiệu chỉnh hoà) giữa hai dãy số cùng độ dài; < 3 phần tử -> NaN.'''
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float('nan')
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])
