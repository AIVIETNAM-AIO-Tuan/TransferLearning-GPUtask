# -*- coding: utf-8 -*-
"""Lựa chọn dựa trên val: lr tốt nhất (như evaluate_linear_classifiers của mã gốc) + chẩn đoán đường học (PLAN_86 §5.2 — chỉ mô tả).

best_run / learning_curve_summary / max_finite CHÉP NGUYÊN VĂN từ vindr_protocol/selection.py của 86 (tests/test_ver2.py so cây cú pháp). best_by
(mới ở 89): cùng luật so sánh cho tiêu chí bất kỳ — 'auroc' (= best_run, LUẬT) | 'ap' (nhánh mô tả, protocols/ours/config.CRITERIA)."""
import numpy as np

CURVES = {'auroc': 'val_auroc', 'ap': 'val_ap'}          # tiêu chí -> đường val của HeadRun (lần chấm cuối)


def best_run(runs):
    '''Đầu có AUROC val (lần chấm cuối) LỚN NHẤT; max_score = 0 + so sánh '>' nghiêm như mã gốc -> hoà giữ đầu ĐẦU TIÊN theo thứ tự (lr tăng
       dần); NaN không bao giờ được chọn. Không đầu nào > 0 -> None.'''
    best, score = None, 0.0
    for run in runs:
        v = run.val_auroc[-1] if run.val_auroc else float('nan')
        if v > score: best, score = run, v
    return best


def best_by(runs, criterion):
    '''Đầu có số đo val `criterion` (lần chấm cuối) LỚN NHẤT; cùng luật với best_run: max_score = 0 + so sánh '>' nghiêm -> hoà giữ đầu ĐẦU TIÊN
       (lr tăng dần); NaN không bao giờ được chọn. best_by(runs, 'auroc') là best_run(runs).'''
    best, score = None, 0.0
    for run in runs:
        curve = getattr(run, CURVES[criterion])
        v = curve[-1] if curve else float('nan')
        if v > score: best, score = run, v
    return best


def learning_curve_summary(val_auroc, train_loss=None, drop_threshold=0.002, gain_window=5):
    '''QUY TẮC CHẨN ĐOÁN ĐƯỜNG HỌC (PLAN_86 §5.2) — CHỈ MÔ TẢ: không chạy lại, không đổi E, không chọn epoch theo đường val.
       Trên đường AUROC val từng epoch: epoch đỉnh (đếm từ 1; hoà -> epoch sớm nhất), AUROC đỉnh, AUROC cuối, mức giảm sau đỉnh (đỉnh - cuối),
       cờ val_drop_after_peak (giảm > ngưỡng; +1e-12 như luật epoch), mức tăng `gain_window` epoch cuối (cuối - giá trị gain_window epoch trước;
       đường không dài hơn cửa sổ -> None) — cờ giảm KHÔNG thấy học thiếu (đường còn tăng có giảm = 0), mức tăng cuối mới thấy; loss train tại
       epoch đỉnh / cuối khi có (val giảm trong lúc loss train vẫn giảm mới củng cố giả thuyết học quá đà). Đường có NaN / Inf -> finite False,
       các số None (xử lý kỹ thuật, không phải chẩn đoán). Trên đường hữu hạn: val_drop_after_peak = NOT cổng "hội tụ" cũ (86a: hoi_tu).'''
    a = np.asarray(val_auroc, dtype=float)
    out = dict(n_epochs=int(len(a)), finite=bool(len(a) and np.isfinite(a).all()), drop_threshold=float(drop_threshold), gain_window=int(gain_window),
               peak_epoch=None, peak_auroc=None, last_auroc=None, drop_after_peak=None, val_drop_after_peak=None, gain_last=None,
               train_loss_at_peak=None, train_loss_last=None)
    if not out['finite']:
        return out
    k = int(np.argmax(a))
    drop = float(a[k] - a[-1])
    out.update(peak_epoch=k + 1, peak_auroc=float(a[k]), last_auroc=float(a[-1]), drop_after_peak=drop,
               val_drop_after_peak=bool(drop > drop_threshold + 1e-12), gain_last=(float(a[-1] - a[-1 - gain_window]) if len(a) > gain_window else None))
    if train_loss is not None and len(train_loss) == len(a):
        out.update(train_loss_at_peak=float(train_loss[k]), train_loss_last=float(train_loss[-1]))
    return out


def max_finite(values):
    v = [float(x) for x in values if np.isfinite(x)]
    return max(v) if v else float('nan')
