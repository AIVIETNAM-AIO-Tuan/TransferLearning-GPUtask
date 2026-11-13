# -*- coding: utf-8 -*-
"""Panel lớp và chấm MỘT bộ dự đoán test (PLAN_89 §7.1, §7.3) — chỉ mô tả; luật so B − X, khoảng đồng thời max-T, bảng gộp: bước 5.

    p28  28 cột, ngữ nghĩa torchmetrics 0.10.3 (AP lớp không dương = NaN bị bỏ; AUROC lớp đó = 0 vẫn tính) — cạnh Bảng IV CheXFound; bootstrap không loại
    p27  27 lớp có ca dương test — chỉ số điểm (~38,6 % lần rút thiếu >= 1 lớp)
    p21  21 lớp >= 10 ca dương test — số của luật; tập cố định: lần rút thiếu lớp bị LOẠI + đếm
    p7   7 dấu hiệu của bài RAD-DINO — tập cố định, như p21; cạnh Bảng 1a bài RAD-DINO (mốc tham khảo)

Mỗi panel: AP bậc thang (= "AUPRC" của CheXFound), AUPRC hình thang, AUROC — macro + từng lớp. Bootstrap: 2.000 lần rút ảnh có hoàn lại, hạt 8502
(metrics.bootstrap_weights — cùng dãy rút cho mọi hàng / đầu -> ghép cặp theo ảnh ở bước 5); hai nửa test hạt 8501."""
import numpy as np

from . import metrics as M

RADDINO7 = ('Lung Opacity', 'Cardiomegaly', 'Pleural thickening', 'Aortic enlargement', 'Pulmonary fibrosis', 'Tuberculosis', 'Pleural effusion')
N_BOOT, BOOT_SEED, HALF_SEED = 2000, 8502, 8501
PANELS = ('p28', 'p27', 'p21', 'p7')
# Bài CheXFound Bảng IV, VinDr-CXR (AUPRC = AP bậc thang torchmetrics; số bài = TB bootstrap 1.000 lần — suy luận PLAN_86 §11.6). Chỉ đặt cạnh.
PAPER_TABLE_IV = {
    'chexfound': {'lin4': dict(ap=0.298, ap_ci=(0.280, 0.317), auroc=0.838, auroc_ci=(0.776, 0.866)),
                  'glori': dict(ap=0.341, ap_ci=(0.319, 0.367), auroc=0.872, auroc_ci=(0.813, 0.898))},
    'raddino': {'lin4': dict(ap=0.102, ap_ci=(0.096, 0.107), auroc=0.477, auroc_ci=(0.441, 0.507)),
                'glori': dict(ap=0.155, ap_ci=(0.147, 0.163), auroc=0.680, auroc_ci=(0.630, 0.718))},
    'biomedclip': {'lin4': dict(ap=0.100, ap_ci=(0.094, 0.106), auroc=0.587, auroc_ci=(0.544, 0.618)),
                   'glori': dict(ap=0.094, ap_ci=(0.089, 0.102), auroc=0.537, auroc_ci=(0.497, 0.566))},
}
RADDINO_PAPER_7 = 0.528       # Bảng 1a bài RAD-DINO: macro AUPRC 7 dấu hiệu VinDr (checkpoint bài ≠ checkpoint công khai; cách tính AUPRC bài không nói)


def panel_columns(label_names, main_classes, target):
    '''-> {panel: [chỉ số cột]}. p27 = cột có ca dương trên CẢ tập test; p21 = lớp số chính (thứ tự nhãn); p7 ⊂ p21.'''
    names = list(label_names)
    pos = np.asarray(target).sum(0)
    cols = dict(p28=list(range(len(names))), p27=[j for j in range(len(names)) if pos[j] > 0],
                p21=[j for j, n in enumerate(names) if n in set(main_classes)], p7=[names.index(c) for c in RADDINO7])
    assert len(cols['p21']) == len(main_classes) and set(cols['p7']) <= set(cols['p21']) <= set(cols['p27']), 'panel lop KHONG long nhau -> DUNG'
    return cols


def point(preds, target, cols):
    '''Số điểm toàn tập test trên cột `cols` (hàm của gói; AP lớp không dương -> NaN bỏ; AUROC -> 0 giữ). -> dict macro + từng lớp.'''
    y, p = np.asarray(target)[:, cols], np.asarray(preds)[:, cols]
    ap, ap_m, au, au_m = M.ap_auroc_tm0103(y, p)
    tr, tr_m = M.auprc_trapezoid(y, p)
    return dict(ap=ap_m, auprc_thang=tr_m, auroc=au_m, ap_lop=[float(v) for v in ap], auprc_thang_lop=[float(v) for v in tr],
                auroc_lop=[float(v) for v in au])


def _ci(v):
    v = np.asarray(v, dtype=np.float64)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float('nan')] * 2


def bootstrap(preds, target, cols, n_boot=N_BOOT, seed=BOOT_SEED):
    '''Bootstrap theo ảnh (trọng số = số lần có mặt). p28: ngữ nghĩa torchmetrics (không loại). p21 / p7: lần rút thiếu ca dương (hoặc thiếu ca âm) ở bất kỳ
       lớp nào của panel -> LOẠI và đếm. -> {panel: dict(tb, ci) cho ap / auprc_thang / auroc + so_lan_loai}.'''
    target, preds = np.asarray(target), np.asarray(preds)
    W = M.bootstrap_weights(len(target), n_boot, seed)
    pre = M.sort_once(target, preds)
    acc = {p: dict(ap=[], auprc_thang=[], auroc=[], loai=0) for p in ('p28', 'p21', 'p7')}
    for w in W:
        ap, au, tr = M.ap_weighted(pre, w), M.auroc_weighted(pre, w), M.auprc_trapezoid_weighted(pre, w)
        a28, u28 = M.tm0103_macro(ap, au)
        acc['p28']['ap'].append(a28); acc['p28']['auroc'].append(u28); acc['p28']['auprc_thang'].append(float(np.nanmean(tr)))
        for p in ('p21', 'p7'):
            c = cols[p]
            if not (np.isfinite(ap[c]).all() and np.isfinite(au[c]).all() and np.isfinite(tr[c]).all()):
                acc[p]['loai'] += 1
                continue
            acc[p]['ap'].append(float(ap[c].mean())); acc[p]['auroc'].append(float(au[c].mean())); acc[p]['auprc_thang'].append(float(tr[c].mean()))
    out = {}
    for p, a in acc.items():
        out[p] = {m: dict(tb=float(np.mean(a[m])) if a[m] else float('nan'), ci=_ci(a[m])) for m in ('ap', 'auprc_thang', 'auroc')}
        out[p]['so_lan_loai'] = int(a['loai'])
        out[p]['so_lan'] = int(n_boot)
    return out


def score_predictions(preds, target, label_names, main_classes, n_boot=N_BOOT, seed=BOOT_SEED, half_seed=HALF_SEED):
    '''Chấm một bộ dự đoán test: p28 bằng torchmetrics 0.10.3 THẬT (+ đối chiếu hàm của gói), điểm p27 / p21 / p7, bootstrap p28 / p21 / p7, AP p21 trên
       hai nửa test. -> dict.'''
    target, preds = np.asarray(target), np.asarray(preds, dtype=np.float32)
    assert preds.shape == target.shape and preds.shape[1] == len(label_names) and np.isfinite(preds).all(), 'du doan / nhan KHONG khop -> DUNG'
    cols = panel_columns(label_names, main_classes, target)
    tm_ap, tm_auroc = M.tm_scores(preds, target)
    pts = {p: point(preds, target, cols[p]) for p in PANELS}
    half = M.halves(len(target), half_seed)
    nua = [point(preds[half == h], target[half == h], cols['p21'])['ap'] for h in (0, 1)]
    return dict(p28_tm=dict(ap=tm_ap, auroc=tm_auroc, lech_ap_goi=abs(tm_ap - pts['p28']['ap']), lech_auroc_goi=abs(tm_auroc - pts['p28']['auroc'])),
                diem=pts, bootstrap=bootstrap(preds, target, cols, n_boot, seed), ap21_hai_nua=nua, cot=cols,
                so_duong_test=[int(v) for v in target.sum(0)])
