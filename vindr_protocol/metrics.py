"""Chỉ số: AP / AUROC theo ngữ nghĩa torchmetrics 0.10.3 — bản CheXFound ghim (PLAN_86 F8, §6.1; giá trị vàng ở test_vindr_protocol) — và
AUPRC HÌNH THANG báo kèm như 85d / 85g (Tuấn 09/11: "đo thêm AUPRC").

Lưu ý tên gọi: "AUPRC" trong bài CheXFound = torchmetrics MultilabelAveragePrecision (mã của họ: MetricType.MULTILABEL_AP) = AP BẬC THANG
(cột AP ở đây). AUPRC hình thang = diện tích dưới đường PR nối thẳng các điểm (sklearn auc) — cách tính khác, số khác."""
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
