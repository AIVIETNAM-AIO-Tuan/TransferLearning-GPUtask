"""Phép đo của bảng VinDr (= 85d / 85g, test_87 kiểm trùng từng bit với các hàm gốc):

- AP bậc thang (= average_precision_score của sklearn, KHÔNG nội suy) -> mAP = TB macro trên lớp hợp lệ; số của luật.
- AUROC (roc_auc_score); AUPRC hình thang (= sklearn auc(recall, precision) của precision_recall_curve) — báo kèm, khác AP.
- Bản có trọng số (ảnh i xuất hiện w[i] lần) cho bootstrap theo ảnh: sắp xếp MỘT lần mỗi cột, mỗi lần rút chỉ đổi trọng số.
- Bootstrap trên TẬP LỚP CỐ ĐỊNH: lần rút làm một lớp thiếu ca dương / toàn dương -> loại cả lần rút (in số lần + tên lớp)."""
import collections

import numpy as np
from sklearn.metrics import auc as _auc_trapezoid
from sklearn.metrics import precision_recall_curve, roc_auc_score


def ap_cols(Yc, Sc):
    '''AP bậc thang từng cột; cột không có ca dương / toàn dương -> NaN. (= ap_cols của C0 bảng 78a và bước z_c5 của 78z, nguyên văn.)'''
    o = np.full(Yc.shape[1], np.nan)
    for j in range(Yc.shape[1]):
        y, s = Yc[:, j], Sc[:, j]; n1 = y.sum()
        if n1 == 0 or n1 == len(y): continue
        r = np.argsort(-s, kind='stable'); yy, ss = y[r], s[r]
        tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
        last = np.r_[np.diff(ss) != 0, True]; tp, fp = tp[last], fp[last]
        pr, rc = tp / (tp + fp), tp / n1
        o[j] = float((np.diff(np.r_[0.0, rc]) * pr).sum())
    return o


def auroc_cols(Y, S):
    '''AUROC từng cột (= _auc_cols_d của 85d).'''
    return np.array([roc_auc_score(Y[:, j], S[:, j]) if 0 < Y[:, j].sum() < len(Y) else np.nan for j in range(Y.shape[1])])


def auprc_cols(Y, S):
    '''AUPRC hình thang từng cột (= _auprc_cols_d của 85d); không dương / toàn dương -> NaN.'''
    o = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        if 0 < Y[:, j].sum() < len(Y):
            _p, _r, _ = precision_recall_curve(Y[:, j], S[:, j]); o[j] = float(_auc_trapezoid(_r, _p))
    return o


def sort_once(Y, P):
    '''Sắp xếp MỘT lần mỗi cột cho bootstrap (= _thu_tu_d của 82d / 85d / 85g).'''
    out = []
    for j in range(Y.shape[1]):
        y = Y[:, j].astype('float64')
        if y.sum() == 0 or y.sum() == len(y): out.append(None); continue
        r = np.argsort(-P[:, j], kind='stable'); ss = P[r, j]
        out.append((r, y[r], np.flatnonzero(np.r_[np.diff(ss) != 0, True])))
    return out


def ap_weighted(pre, w):
    '''AP từng cột khi ảnh i có mặt w[i] lần (= _ap_w_d; w = 1 -> ap_cols).'''
    o = np.full(len(pre), np.nan)
    for j, x in enumerate(pre):
        if x is None: continue
        r, yy, li = x; ws = w[r]
        tp = np.cumsum(ws * yy)[li]; tong = np.cumsum(ws)[li]
        if tp[-1] <= 0 or tp[-1] >= tong[-1]: continue
        pr = np.divide(tp, tong, out=np.zeros_like(tp), where=tong > 0)
        o[j] = float((np.diff(np.r_[0.0, tp / tp[-1]]) * pr).sum())
    return o


def auroc_weighted(pre, w):
    '''AUROC từng cột có trọng số (= _auc_w_d; hoà -> 0,5 như roc_auc_score sample_weight).'''
    o = np.full(len(pre), np.nan)
    for j, x in enumerate(pre):
        if x is None: continue
        r, yy, li = x; ws = w[r]
        cp = np.cumsum(ws * yy)[li]; cn = np.cumsum(ws * (1 - yy))[li]
        if cp[-1] <= 0 or cn[-1] <= 0: continue
        pg, ng = np.diff(np.r_[0.0, cp]), np.diff(np.r_[0.0, cn])
        o[j] = float((pg * (cn[-1] - cn + 0.5 * ng)).sum() / (cp[-1] * cn[-1]))
    return o


def auprc_weighted(pre, w):
    '''AUPRC hình thang từng cột có trọng số (= _auprc_w_d = sklearn auc(recall, precision) của precision_recall_curve(sample_weight=w)).'''
    o = np.full(len(pre), np.nan)
    for j, x in enumerate(pre):
        if x is None: continue
        r, yy, li = x; ws = w[r]
        tp = np.cumsum(ws * yy)[li]; tong = np.cumsum(ws)[li]
        if tp[-1] <= 0 or tp[-1] >= tong[-1]: continue
        giu = (tong > 0) & np.r_[True, np.diff(tong) > 0]
        rc, pr = np.r_[0.0, tp[giu] / tp[-1]], np.r_[1.0, tp[giu] / tong[giu]]
        o[j] = float((np.diff(rc) * (pr[1:] + pr[:-1])).sum() / 2)
    return o


def halves(n, seed):
    '''Chia hai nửa theo ảnh (= NUA_G / NUA_D của 85g / 85d): nửa A = n // 2 ảnh rút không lặp bằng hạt `seed` -> mảng 0 / 1.'''
    a = np.random.default_rng(seed).choice(n, size=n // 2, replace=False)
    return np.where(np.isin(np.arange(n), a), 0, 1)


def summary(Y, S, half):
    '''mAP / mAUROC / mAUPRC (TB lớp hợp lệ) toàn tập + hai nửa + từng lớp (= _so_g của 85g; khoá như JSON 85g).'''
    o = {}
    for ten, m in (('ap', ap_cols), ('auc', auroc_cols), ('auprc', auprc_cols)):
        a, aA, aB = m(Y, S), m(Y[half == 0], S[half == 0]), m(Y[half == 1], S[half == 1])
        o.update({ten: float(np.nanmean(a)), f'{ten}_A': float(np.nanmean(aA)), f'{ten}_B': float(np.nanmean(aB)),
                  f'{ten}_lop': [None if not np.isfinite(x) else float(x) for x in a]})
    return o


def mean_ap(Y, S):
    '''mAP (TB lớp hợp lệ) — dùng cho điểm từng đầu đơn.'''
    return float(np.nanmean(ap_cols(Y, S)))


def bootstrap_class_ap(Y, scores, class_names, seed, n_boot):
    '''Bootstrap theo ảnh: mỗi lần rút n ảnh có lặp (hạt `seed`, = 85g / 85d); AP TỪNG LỚP có trọng số của mọi điểm trong `scores`
       ({khoá: [n, lớp]}). Lần rút làm một lớp không hữu hạn -> loại. Điều kiện loại CHỈ phụ thuộc nhãn + trọng số (lớp không có / toàn ca
       dương), không phụ thuộc điểm -> cùng hạt cho cùng tập lần rút hợp lệ với 85g / 85d dù tập khoá / phép đo khác.
       -> ({khoá: mảng [lần hợp lệ, lớp]}, số lần bị loại, Counter tên lớp làm loại).'''
    n = Y.shape[0]
    pre = {k: sort_once(Y, S) for k, S in scores.items()}
    draws, skipped, by_class = {k: [] for k in pre}, 0, collections.Counter()
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype('float64')
        m = {k: ap_weighted(v, w) for k, v in pre.items()}
        thieu = sorted({class_names[j] for x in m.values() for j in np.flatnonzero(~np.isfinite(x))})
        if thieu:
            skipped += 1; by_class.update(thieu); continue
        for k, x in m.items(): draws[k].append(x)
    return {k: np.array(v) for k, v in draws.items()}, skipped, by_class


def ci95(v):
    '''{ci: [2,5 %; 97,5 %], sd (ddof = 0, như 85g), n} của các giá trị bootstrap.'''
    v = np.asarray(v, 'float64')
    return dict(ci=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], sd=float(np.std(v)), n=int(len(v)))


def mean_sd(v):
    '''TB + SD MẪU (ddof = 1) — điểm đầu đơn / qua seed.'''
    v = [float(x) for x in v]
    return dict(tb=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else float('nan'), n=len(v))
