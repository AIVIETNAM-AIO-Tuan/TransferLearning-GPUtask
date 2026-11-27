# -*- coding: utf-8 -*-
"""Chỉ số 28 nhãn theo từng lần rút bootstrap (PLAN_94 §2) — dùng lại vindr_bench.evaluation (metrics / panels / compare), chỉ thêm:

    column_draws   AP bậc thang + AUROC TỪNG cột ở TỪNG lần rút, véc-tơ hoá theo lần rút — cùng phép tính với metrics.ap_weighted / auroc_weighted
                   (sắp xếp một lần mỗi cột bằng metrics.sort_once; tests/test_vindr28.py kiểm trùng từng lần rút, kể cả hoà, trọng số 0, cột thiếu dương)
    set_stats28    một bộ dự đoán [N, 28] -> điểm 4 panel (panels.point) + ap_b / ap_d / auroc của 2.000 lần rút + m21 / m7 (như compare.set_stats,
                   CHỈ cho cổng) + hai nửa test
    cache          bộ nhớ đệm .npz (không pickle) cho resume / chạy lại

Cách gộp (Tuấn 27/11 chọn B + D; không bỏ lần rút nào):
    ap_b   TB AP trên các nhãn CÓ ca dương trong lần rút (= metrics.tm0103_macro — ngữ nghĩa torchmetrics 0.10.3, như mã CheXFound)
    ap_d   TB AP trên tập CỐ ĐỊNH các nhãn có ca dương trên CẢ test (27 nhãn VinDr); nhãn vắng trong lần rút -> AP = 0
    auroc  TB AUROC trên 28 nhãn; nhãn thiếu ca dương (hoặc ca âm) trong lần rút -> 0, vẫn chia 28 (= tm0103_macro). B và D trùng ở AUROC.
Số điểm trên cả test: AP28 = AP27 (Edema 0 ca dương), AUROC28 có Edema = 0 (panels.point trên p28)."""
import hashlib
import json
import time

import numpy as np

from vindr_bench.evaluation import compare as CP
from vindr_bench.evaluation import metrics as M
from vindr_bench.evaluation import panels as P

from . import config as C

VERSION = 'v28_1'                                     # đổi cách tính -> đổi chuỗi: bộ nhớ đệm cũ tự bị bỏ qua
assert (P.N_BOOT, P.BOOT_SEED, P.HALF_SEED) == (C.N_BOOT, C.BOOT_SEED, C.HALF_SEED), 'tham so bootstrap cua vindr_bench KHAC config -> DUNG'


def column_draws(target, preds, W, chunk=C.CHUNK):
    '''-> (AP [B, L], AUROC [B, L]) cho B lần rút (W [B, N]: số lần có mặt của từng ảnh). AP: NaN khi lần rút không có ca dương của cột, 1 khi toàn dương;
       AUROC: NaN khi thiếu ca dương hoặc ca âm. Cột không có ca dương trên CẢ tập -> NaN ở mọi lần rút. = metrics.ap_weighted / auroc_weighted từng hàng.'''
    W = np.asarray(W, dtype=np.float64)
    pre = M.sort_once(target, preds)
    B = W.shape[0]
    AP, AU = np.full((B, len(pre)), np.nan), np.full((B, len(pre)), np.nan)
    for j, x in enumerate(pre):
        if x is None:
            continue
        r, yy, li = x
        for b0 in range(0, B, chunk):
            Ws = W[b0:b0 + chunk][:, r]
            n = Ws.shape[0]
            tp = np.cumsum(Ws * yy, axis=1)[:, li]
            tot = np.cumsum(Ws, axis=1)[:, li]
            cn = np.cumsum(Ws * (1 - yy), axis=1)[:, li]
            last, tl, nl = tp[:, -1], tot[:, -1], cn[:, -1]
            with np.errstate(divide='ignore', invalid='ignore'):
                pr = np.divide(tp, tot, out=np.zeros_like(tp), where=tot > 0)
                rec = tp / last[:, None]
                ap = (np.diff(np.concatenate([np.zeros((n, 1)), rec], axis=1), axis=1) * pr).sum(1)
                pg = np.diff(np.concatenate([np.zeros((n, 1)), tp], axis=1), axis=1)
                ng = np.diff(np.concatenate([np.zeros((n, 1)), cn], axis=1), axis=1)
                au = (pg * (nl[:, None] - cn + 0.5 * ng)).sum(1) / (last * nl)
            ap = np.where(last > 0, np.where(last >= tl, 1.0, ap), np.nan)
            AP[b0:b0 + n, j] = ap
            AU[b0:b0 + n, j] = np.where((last > 0) & (nl > 0), au, np.nan)
    return AP, AU


_WT = {}


def _weights_t(W):
    '''W [B, N] (số lần có mặt, nguyên) -> W.T [N, B] int32 liền bộ nhớ (lấy hàng nhanh); nhớ theo id + hình (cùng dãy rút cho mọi bộ).'''
    k = (id(W), W.shape)
    if k not in _WT:
        assert np.array_equal(W, np.round(W)) and W.min() >= 0, 'trong so bootstrap KHONG phai so nguyen khong am -> DUNG'
        _WT.clear()
        _WT[k] = (W, np.ascontiguousarray(W.T).astype(np.int32))
    return _WT[k][1]


def column_draws_fast(target, preds, W):
    '''Như column_draws (cùng định nghĩa, cùng kết quả tới sai số làm tròn ~1e-16), nhanh hơn ~10 lần: chỉ nhóm điểm hoà CÓ ca dương góp vào AP / AUROC, nên
       chỉ cần tổng trọng số tại cuối các nhóm đó và nhóm liền trước (np.add.reduceat trên W.T theo thứ tự điểm) — trọng số nguyên nên cộng dồn chính xác.'''
    W = np.asarray(W, dtype=np.float64)
    WT = _weights_t(W)
    N, B = WT.shape
    pre = M.sort_once(target, preds)
    AP, AU = np.full((B, len(pre)), np.nan), np.full((B, len(pre)), np.nan)
    for j, x in enumerate(pre):
        if x is None:
            continue
        r, yy, li = x
        npos = np.cumsum(yy)[li]
        g = np.flatnonzero(np.diff(np.r_[0.0, npos]) > 0)                    # nhóm có ít nhất một ca dương
        end, prev = li[g], np.where(g > 0, li[np.maximum(g - 1, 0)], -1)
        bnd = np.unique(np.r_[end, prev[prev >= 0], N - 1])                  # vị trí (theo thứ tự điểm) cần tổng dồn
        Wr = WT[r]
        tot = np.cumsum(np.add.reduceat(Wr, np.r_[0, bnd[:-1] + 1], axis=0), axis=0)         # [m, B] tổng trọng số tới bnd
        pos = np.flatnonzero(yy)
        tpc = np.cumsum(Wr[pos], axis=0)                                     # [P, B] tổng trọng số ca dương theo thứ tự
        k = np.searchsorted(pos, bnd, side='right')
        tp = np.where((k > 0)[:, None], tpc[np.maximum(k - 1, 0)], 0)         # [m, B] tổng ca dương tới bnd
        ie, ip = np.searchsorted(bnd, end), np.searchsorted(bnd, np.maximum(prev, 0))
        has_prev = (prev >= 0)[:, None]
        te, tp_e = tot[ie].astype(np.float64), tp[ie].astype(np.float64)
        tq, tp_q = np.where(has_prev, tot[ip], 0).astype(np.float64), np.where(has_prev, tp[ip], 0).astype(np.float64)
        last, tl = tp[-1].astype(np.float64), tot[-1].astype(np.float64)
        nl = tl - last
        with np.errstate(divide='ignore', invalid='ignore'):
            pr = np.divide(tp_e, te, out=np.zeros_like(tp_e), where=te > 0)
            ap = ((tp_e / last - tp_q / last) * pr).sum(0)
            cn_e, cn_q = te - tp_e, tq - tp_q
            au = ((tp_e - tp_q) * (nl - cn_e + 0.5 * (cn_e - cn_q))).sum(0) / (last * nl)
        AP[:, j] = np.where(last > 0, np.where(last >= tl, 1.0, ap), np.nan)
        AU[:, j] = np.where((last > 0) & (nl > 0), au, np.nan)
    return AP, AU


def macro(AP, AU, cols):
    '''Gộp theo cách B / D (docstring của module). cols = panels.panel_columns. -> (ap_b [B], ap_d [B], auroc [B]).'''
    A = AP[:, cols['p27']]
    k = np.isfinite(A).sum(1)
    ap_b = np.where(k > 0, np.nan_to_num(A).sum(1) / np.maximum(k, 1), np.nan)
    ap_d = np.nan_to_num(A).mean(1)
    auroc = np.where(np.isfinite(AU), AU, 0.0).mean(1)
    return ap_b, ap_d, auroc


def fixed_mean(AP, c):
    '''= compare.set_stats: TB trên tập cột cố định `c`; lần rút có cột NaN -> NaN (lần rút bị loại).'''
    D = AP[:, list(c)]
    return np.where(np.isfinite(D).all(1), np.nan_to_num(D).mean(1), np.nan)


def half_values(preds, target, cols, half):
    '''Số điểm trên hai nửa test cố định (hạt 8501) theo cùng ba cách. -> ({ap_b, ap_d, auroc: [nửa 0, nửa 1]}, [AP21 nửa 0, nửa 1] như set_stats).'''
    out, n21 = {s: [] for s in C.STATS}, []
    c27 = cols['p27']
    for h in (0, 1):
        m = half == h
        pt = P.point(preds[m], target[m], cols['p28'])
        a = np.asarray(pt['ap_lop'], dtype=np.float64)[c27]
        out['ap_b'].append(float(np.nanmean(a)) if np.isfinite(a).any() else float('nan'))
        out['ap_d'].append(float(np.nan_to_num(a).mean()))
        out['auroc'].append(float(pt['auroc']))
        n21.append(float(P.point(preds[m], target[m], cols['p21'])['ap']))
    return out, n21


def set_stats28(preds, target, cols, n_boot=C.N_BOOT, seed=C.BOOT_SEED, half_seed=C.HALF_SEED):
    '''Một bộ dự đoán test -> dict(diem {panel: {ap, auprc_thang, auroc}}, ap21_lop, ap28_lop, auroc28_lop, ap_b / ap_d / auroc [n_boot], m21 / m7 [n_boot]
       (NaN = lần rút bị loại, như compare.set_stats), nua {ba cách: [hai nửa]}, nua21).'''
    preds, target = np.asarray(preds, dtype=np.float32), np.asarray(target)
    assert preds.shape == target.shape and preds.shape[1] == len(cols['p28']) and np.isfinite(preds).all(), 'du doan / nhan KHONG khop hoac khong huu han -> DUNG'
    pts = {p: P.point(preds, target, cols[p]) for p in P.PANELS}
    AP, AU = column_draws_fast(target, preds, CP.weights(len(target), n_boot, seed))
    ap_b, ap_d, auroc = macro(AP, AU, cols)
    nua, nua21 = half_values(preds, target, cols, M.halves(len(target), half_seed))
    return dict(diem={p: {k: float(pts[p][k]) for k in ('ap', 'auprc_thang', 'auroc')} for p in P.PANELS},
                ap21_lop=[float(v) for v in pts['p21']['ap_lop']], ap28_lop=[float(v) for v in pts['p28']['ap_lop']],
                auroc28_lop=[float(v) for v in pts['p28']['auroc_lop']], ap_b=ap_b, ap_d=ap_d, auroc=auroc,
                m21=fixed_mean(AP, cols['p21']), m7=fixed_mean(AP, cols['p7']), nua=nua, nua21=nua21)


def stats_key(preds, target, cols, n_boot=C.N_BOOT, seed=C.BOOT_SEED, half_seed=C.HALF_SEED):
    '''Khoá bộ nhớ đệm 28 nhãn: md5(dự đoán float32 + nhãn int8 + cột panel + tham số bootstrap + VERSION).'''
    h = hashlib.md5()
    h.update(np.ascontiguousarray(np.asarray(preds), dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(np.asarray(target), dtype=np.int8).tobytes())
    h.update(json.dumps(dict(c={p: [int(j) for j in cols[p]] for p in sorted(cols)}, n=int(n_boot), s=int(seed), hs=int(half_seed), v=VERSION),
                        sort_keys=True).encode())
    return h.hexdigest()


ARRAYS = ('ap_b', 'ap_d', 'auroc', 'm21', 'm7')
F32 = ('ap_b', 'ap_d', 'auroc')                       # 0.2.0: chỉ ba đại lượng 28 nhãn lưu float32; m21 / m7 (CHỈ cho cổng so với 89t / 92) giữ float64 — cổng
                                                      # 1e-9 của 94b so CI21 dựng lại từ m21 với JSON 92 (float32 làm lệch ~1e-8)


def stored(st):
    '''Bản lưu được của set_stats28: ba đại lượng 28 nhãn -> float32 (cùng một cách cho lần tính mới và lần nạp lại, nên kết quả không phụ thuộc resume).'''
    return dict(st, **{k: (np.asarray(st[k], dtype=np.float32).astype(np.float64) if k in F32 else np.asarray(st[k], dtype=np.float64)) for k in ARRAYS})


def save_cache(fh, cache):
    '''{khoá: set_stats28 (+ 'cong')} -> .npz (KHÔNG pickle) vào tệp đã mở `fh` (np.savez với đường dẫn tự thêm đuôi .npz).'''
    keys = sorted(cache)
    meta = dict(v=VERSION, keys=keys, idx={k: {n: v for n, v in cache[k].items() if n not in ARRAYS} for k in keys})
    arr = {f'{n}_{i}': np.asarray(cache[k][n], dtype=np.float32 if n in F32 else np.float64) for i, k in enumerate(keys) for n in ARRAYS}
    np.savez_compressed(fh, chi_muc=np.frombuffer(json.dumps(meta).encode('utf-8'), dtype=np.uint8), **arr)


def load_cache(path):
    '''Ngược save_cache. Tệp của VERSION khác -> {} (tính lại).'''
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z['chi_muc'].tobytes().decode('utf-8'))
        if meta.get('v') != VERSION:
            return {}
        return {k: dict(meta['idx'][k], **{n: z[f'{n}_{i}'].astype(np.float64) for n in ARRAYS}) for i, k in enumerate(meta['keys'])}


def _job(a):
    return set_stats28(*a)


def many(jobs, workers=1, log=None, on_progress=None):
    '''jobs = [(preds, target, cols)] -> [set_stats28] (đúng thứ tự). workers > 1: tiến trình con 'spawn' (như compare.many_stats). on_progress(i, st) gọi
       sau mỗi bộ xong (cổng / bộ nhớ đệm / resume ở bên gọi). log: tiến độ ~10 % một lần + thời gian còn lại ước.'''
    n, out, t0 = len(jobs), [None] * len(jobs), time.time()
    step = max(1, n // 10)

    def tick(done):
        if log is not None and (done % step == 0 or done == n):
            el = time.time() - t0
            log(f'  28 nhan: {done}/{n} bo | {el / 60:.1f} phut | con ~{el / done * (n - done) / 60:.1f} phut')

    if workers > 1 and n > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ex:
                fut = {ex.submit(_job, j): i for i, j in enumerate(jobs)}
                for k, f in enumerate(as_completed(fut), 1):
                    i = fut[f]
                    out[i] = f.result()
                    if on_progress is not None:
                        on_progress(i, out[i])
                    tick(k)
            return out
        except (OSError, RuntimeError, ImportError) as e:
            if log is not None:
                log(f'  ! tien trinh con loi ({type(e).__name__}: {str(e)[:120]}) -> chay lan luot')
            out, t0 = [None] * n, time.time()
    for i, j in enumerate(jobs):
        out[i] = _job(j)
        if on_progress is not None:
            on_progress(i, out[i])
        tick(i + 1)
    return out


def gate_89(st, ref):
    '''Cổng 94a: set_stats28 (tính mới) so với set_stats của 89t (bộ nhớ đệm, cùng bộ dự đoán). -> dict(diem_trung, nua21_trung, ap21_lop_trung, lech_m21,
       nan_trung). diem / hai nửa / AP từng lớp: cùng hàm panels.point -> phải TRÙNG; m21 từng lần rút: véc-tơ hoá so với vòng lặp -> lệch <= TOL_89.'''
    a, b = np.asarray(st['m21'], dtype=np.float64), np.asarray(ref['m21'], dtype=np.float64)
    nan_ok = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    ok = np.isfinite(a) & np.isfinite(b)
    lech = float(np.max(np.abs(a[ok] - b[ok]))) if ok.any() else float('nan')
    same = all(st['diem'][p][m] == ref['diem'][p][m] for p in P.PANELS for m in ('ap', 'auprc_thang', 'auroc'))
    return dict(diem_trung=bool(same), nua21_trung=bool(list(st['nua21']) == list(ref['nua'])), ap21_lop_trung=bool(list(st['ap21_lop']) == list(ref['ap21_lop'])),
                lech_m21=lech, nan_trung=nan_ok, dat=bool(same and nan_ok and lech <= C.TOL_89 and list(st['nua21']) == list(ref['nua'])))
