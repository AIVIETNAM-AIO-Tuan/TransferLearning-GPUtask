# -*- coding: utf-8 -*-
"""Ô 28 nhãn có khuôn của vindr_bench.evaluation.compare.make_cell (PLAN_94 §3): panel 'p21' của mỗi ô mang số của MỘT đại lượng 28 nhãn (ap_b, ap_d
hoặc auroc), nên mọi họ phép so đã có — vindr_bench.notebook.summary.compare / crop_test / ablation (89t) và frozen_probe.scoring.family_pair /
descriptive (92) — chạy lại NGUYÊN VẸN (cùng định nghĩa họ, cùng max-T, cùng dạng luật) trên số 28 nhãn. Số in / chữ "mAP21" của các hàm đó KHÔNG dùng:
notebook của 94 tắt log của chúng và tự in bảng.

Ô của đại lượng `stat`:
    diem['p21'] = {ap: số điểm của stat (AP28 với ap_b / ap_d — trùng nhau trên cả test; AUROC28 với auroc), auprc_thang: AUPRC hình thang 27 nhãn,
                   auroc: AUROC28}
    m21 = m7    = stat của từng lần rút (2.000)
    nua         = stat trên hai nửa test
    panel khác (p28 / p27 / p7) giữ nguyên số thật."""
import numpy as np

from vindr_bench.evaluation import compare as CP
from vindr_bench.evaluation import panels as P

from . import config as C

M3 = ('ap', 'auprc_thang', 'auroc')


def point_p21(diem, stat):
    '''diem {panel: {ap, auprc_thang, auroc}} -> bản sao với 'p21' mang số điểm của `stat`.'''
    d = {p: dict(v) for p, v in diem.items()}
    v = d['p28']['auroc'] if stat == 'auroc' else d['p28']['ap']
    d['p21'] = dict(ap=float(v), auprc_thang=float(d['p27']['auprc_thang']), auroc=float(d['p28']['auroc']))
    return d


def as_stats(st, stat):
    '''set_stats28 (hoặc bản nạp từ bộ nhớ đệm) -> dict khuôn compare.set_stats (diem, ap21_lop, m21, m7, nua) cho đại lượng `stat`.'''
    assert stat in C.STATS, stat
    arr = np.asarray(st[stat], dtype=np.float64)
    return dict(diem=point_p21(st['diem'], stat), ap21_lop=list(st['ap21_lop']), m21=arr, m7=arr, nua=[float(v) for v in st['nua'][stat]])


def ensemble_points(mean_preds, target, cols):
    '''Điểm 4 panel của TB xác suất các hạt (như summary.score) -> {panel: {ap, auprc_thang, auroc}}.'''
    return {p: {m: float(v) for m, v in P.point(mean_preds, target, cols[p]).items() if m in M3} for p in P.PANELS}


def cell(stats, stat, ensemble=None, **extra):
    '''[set_stats28 của từng hạt / lần rút nhãn, đúng thứ tự] -> ô của `stat` (compare.make_cell) + các trường thêm (lr, bien, nguon, …).'''
    c = CP.make_cell([as_stats(s, stat) for s in stats], None if ensemble is None else point_p21(ensemble, stat))
    c.update(extra)
    return c
