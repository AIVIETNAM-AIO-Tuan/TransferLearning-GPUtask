# -*- coding: utf-8 -*-
"""Chỉ số (PLAN_89 §7.1): AP / AUROC theo ngữ nghĩa torchmetrics 0.10.3 (bản CheXFound ghim), AUPRC HÌNH THANG báo kèm, và bản có trọng số cho bootstrap.

Chép NGUYÊN VĂN phần thân của vindr_protocol/metrics.py @ 0.8.2 (module 86; giá trị vàng ở test_vindr_protocol: trùng sklearn / torchmetrics) — tests/test_ver1.py
so mã với tệp nguồn; thêm ở 89: tm_scores (torchmetrics 0.10.3 thật, = cổng lần chấm của 86) và halves (hai nửa test, hạt 8501, như 85g / 87b).

Tên gọi: "AUPRC" trong bài CheXFound = torchmetrics MultilabelAveragePrecision = AP BẬC THANG (cột AP ở đây). AUPRC hình thang = diện tích dưới đường PR
nối thẳng các điểm (sklearn auc) — cách tính khác, số khác."""
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score


def ap_auroc_tm0103(labels, probs):
    '''AP từng lớp (lớp KHÔNG có ca dương -> NaN, BỎ khỏi macro); AUROC từng lớp (lớp không có ca dương HOẶC toàn dương -> 0, VẪN tính vào
       macro). -> (ap_lop, ap_macro, auroc_lop, auroc_macro).'''
    labels, probs = np.asarray(labels), np.asarray(probs, dtype=np.float64)
    ap, auroc = np.full(labels.shape[1], np.nan), np.zeros(labels.shape[1])
    for j in range(labels.shape[1]):
        n_pos = int(labels[:, j].sum())
        if n_pos > 0: ap[j] = average_precision_score(labels[:, j], probs[:, j])
        if 0 < n_pos < len(labels): auroc[j] = roc_auc_score(labels[:, j], probs[:, j])
    return ap, (float(np.nanmean(ap)) if np.isfinite(ap).any() else float('nan')), auroc, float(auroc.mean())


def auprc_trapezoid(labels, probs):
    '''AUPRC HÌNH THANG từng lớp = sklearn auc(recall, precision) của precision_recall_curve — CÙNG định nghĩa _auprc_cols_d của 85d / 85g;
       lớp không có ca dương HOẶC toàn dương -> NaN, bỏ khỏi macro. -> (auprc_lop, auprc_macro).'''
    labels, probs = np.asarray(labels), np.asarray(probs, dtype=np.float64)
    out = np.full(labels.shape[1], np.nan)
    for j in range(labels.shape[1]):
        if 0 < labels[:, j].sum() < len(labels):
            precision, recall, _ = precision_recall_curve(labels[:, j], probs[:, j])
            out[j] = float(auc(recall, precision))
    return out, (float(np.nanmean(out)) if np.isfinite(out).any() else float('nan'))


# ---- bootstrap theo ảnh bằng TRỌNG SỐ (ảnh i có mặt w[i] lần): sắp xếp MỘT lần mỗi cột — cách tính của _thu_tu_d / _ap_w_d / _auc_w_d /
# _auprc_w_d ở 85j (cùng 85g / 78a); w = 1 / trọng số nguyên -> trùng sklearn trên dữ liệu nhân bản (test_vindr_protocol). Dùng cho ô hiệu chỉnh
# với bài (paper_calibration). Khác 85j đúng một chỗ, theo ngữ nghĩa torchmetrics 0.10.3 (= ap_auroc_tm0103): cột TOÀN DƯƠNG có AP = 1.
def sort_once(labels, probs):
    '''Mỗi cột: thứ tự giảm dần của xác suất (ổn định), nhãn theo thứ tự đó, chỉ số cuối mỗi nhóm điểm hoà. Cột không có ca dương trên CẢ
       tập -> None.'''
    labels, probs = np.asarray(labels, dtype=np.float64), np.asarray(probs, dtype=np.float64)
    out = []
    for j in range(labels.shape[1]):
        y = labels[:, j]
        if y.sum() == 0:
            out.append(None); continue
        r = np.argsort(-probs[:, j], kind='stable')
        s = probs[r, j]
        out.append((r, y[r], np.flatnonzero(np.r_[np.diff(s) != 0, True])))
    return out


def ap_weighted(pre, w):
    '''AP bậc thang từng cột khi ảnh i có mặt w[i] lần; cột không có ca dương trong lần rút -> NaN; toàn dương -> 1 (= sklearn / torchmetrics).'''
    o = np.full(len(pre), np.nan)
    for j, x in enumerate(pre):
        if x is None: continue
        r, yy, li = x
        ws = w[r]
        tp, tot = np.cumsum(ws * yy)[li], np.cumsum(ws)[li]
        if tp[-1] <= 0: continue
        if tp[-1] >= tot[-1]:
            o[j] = 1.0; continue
        pr = np.divide(tp, tot, out=np.zeros_like(tp), where=tot > 0)
        o[j] = float((np.diff(np.r_[0.0, tp / tp[-1]]) * pr).sum())
    return o


def auroc_weighted(pre, w):
    '''AUROC từng cột khi ảnh i có mặt w[i] lần (hoà -> 0,5, = roc_auc_score sample_weight); cột thiếu ca dương hoặc ca âm trong lần rút -> NaN.'''
    o = np.full(len(pre), np.nan)
    for j, x in enumerate(pre):
        if x is None: continue
        r, yy, li = x
        ws = w[r]
        cp, cn = np.cumsum(ws * yy)[li], np.cumsum(ws * (1 - yy))[li]
        if cp[-1] <= 0 or cn[-1] <= 0: continue
        pg, ng = np.diff(np.r_[0.0, cp]), np.diff(np.r_[0.0, cn])
        o[j] = float((pg * (cn[-1] - cn + 0.5 * ng)).sum() / (cp[-1] * cn[-1]))
    return o


def auprc_trapezoid_weighted(pre, w):
    '''AUPRC hình thang từng cột khi ảnh i có mặt w[i] lần (= precision_recall_curve(sample_weight=w) + auc; bỏ ngưỡng chỉ gồm ảnh trọng số 0,
       thêm điểm (recall 0, precision 1)); cột thiếu ca dương / toàn dương -> NaN.'''
    o = np.full(len(pre), np.nan)
    for j, x in enumerate(pre):
        if x is None: continue
        r, yy, li = x
        ws = w[r]
        tp, tot = np.cumsum(ws * yy)[li], np.cumsum(ws)[li]
        if tp[-1] <= 0 or tp[-1] >= tot[-1]: continue
        keep = (tot > 0) & np.r_[True, np.diff(tot) > 0]
        rc, pr = np.r_[0.0, tp[keep] / tp[-1]], np.r_[1.0, tp[keep] / tot[keep]]
        o[j] = float((np.diff(rc) * (pr[1:] + pr[:-1])).sum() / 2)
    return o


def tm0103_macro(ap_cols, auroc_cols):
    '''Macro theo ngữ nghĩa torchmetrics 0.10.3 từ giá trị từng cột: AP bỏ NaN; AUROC NaN (thiếu ca dương / ca âm) -> 0 và VẪN tính.'''
    return float(np.nanmean(ap_cols)), float(np.where(np.isfinite(auroc_cols), auroc_cols, 0.0).mean())


def bootstrap_weights(n_images, n_boot, seed):
    '''n_boot vector trọng số: mỗi lần rút n_images ảnh CÓ HOÀN LẠI (np.random.default_rng(seed)) -> số lần có mặt của từng ảnh [n_boot, n].'''
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n_images, n_images), minlength=n_images).astype(np.float64) for _ in range(n_boot)])


def tm_scores(preds, target):
    '''Macro AP / AUROC bằng torchmetrics 0.10.3 (MultilabelAveragePrecision / MultilabelAUROC average='macro', như build_metric của họ). -> (ap, auroc).'''
    import warnings

    import torch
    p = torch.as_tensor(np.asarray(preds), dtype=torch.float32)
    t = torch.as_tensor(np.asarray(target), dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import torchmetrics
        from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
        assert torchmetrics.__version__ == '0.10.3', f'torchmetrics {torchmetrics.__version__} != 0.10.3 (ban CheXFound ghim) -> DUNG'
        ap = MultilabelAveragePrecision(num_labels=p.shape[1], average='macro')(p, t)
        auroc = MultilabelAUROC(num_labels=p.shape[1], average='macro')(p, t)
    return float(ap), float(auroc)


def halves(n_images, seed):
    '''Hai nửa test cố định: nửa 0 = np.random.default_rng(seed).choice(n, n // 2, replace=False), còn lại nửa 1 (= 85g / 87b). -> mảng 0 / 1.'''
    h = np.ones(n_images, dtype=np.int64)
    h[np.random.default_rng(seed).choice(n_images, size=n_images // 2, replace=False)] = 0
    return h
