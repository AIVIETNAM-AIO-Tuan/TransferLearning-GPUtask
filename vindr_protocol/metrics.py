"""Chỉ số theo ngữ nghĩa torchmetrics 0.10.3 — bản CheXFound ghim (PLAN_86 F8, §6.1; giá trị vàng ở test_vindr_protocol)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


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
