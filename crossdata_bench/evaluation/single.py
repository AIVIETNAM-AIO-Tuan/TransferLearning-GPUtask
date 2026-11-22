# -*- coding: utf-8 -*-
"""Chấm điểm (không bootstrap) cho ô báo cáo của notebook lượt (PLAN_93 §5): AP BẬC THANG (= "AUPRC" của bài CheXFound, sklearn average_precision) + AUPRC
HÌNH THANG (sklearn auc trên đường PR) + AUROC — dùng nguyên vindr_bench.evaluation.metrics (ngữ nghĩa torchmetrics 0.10.3). Một nhãn: số của cột duy nhất;
nhiều nhãn (CheXpert, sau): macro trên tập lớp cố định truyền vào.

Hai nửa test: theo CỤM (người NIH ở RSNA; ảnh ở Shenzhen) — cụm sắp tăng, nửa 0 = default_rng(HALF_SEED).choice(số cụm, số cụm // 2) (= cách của
vindr_bench.evaluation.metrics.halves, áp lên cụm thay vì ảnh)."""
import numpy as np

from .. import config


def point(y, p, cols=None):
    '''y, p [N, C]; cols = chỉ số lớp tính macro (None: mọi lớp). -> dict(ap, auprc_thang, auroc) (macro trên cols).'''
    from vindr_bench.evaluation.metrics import ap_auroc_tm0103, auprc_trapezoid
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    assert y.ndim == 2 and y.shape == p.shape, (y.shape, p.shape)
    cols = list(range(y.shape[1])) if cols is None else list(cols)
    ap, _, auroc, _ = ap_auroc_tm0103(y[:, cols], p[:, cols])
    tr, _ = auprc_trapezoid(y[:, cols], p[:, cols])
    return dict(ap=float(np.nanmean(ap)), auprc_thang=float(np.nanmean(tr)), auroc=float(np.mean(auroc)))


def halves(groups, seed=config.HALF_SEED):
    '''Mã cụm từng ảnh -> mảng 0 / 1 (nửa của ảnh); mọi ảnh của một cụm cùng nửa.'''
    g = np.asarray(groups).astype(str)
    u = np.array(sorted(set(g.tolist())))
    first = set(u[np.random.default_rng(seed).choice(len(u), size=len(u) // 2, replace=False)].tolist())
    return np.array([0 if x in first else 1 for x in g], dtype=np.int64)
