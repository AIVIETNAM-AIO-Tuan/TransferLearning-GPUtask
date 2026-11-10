"""Lựa chọn dựa trên val: lr tốt nhất (như evaluate_linear_classifiers của mã gốc), luật khoá epoch (PLAN_86 §5.1), chẩn đoán đường học
(§5.2 — chỉ mô tả, Tuấn duyệt 10/11; thay "kiểm hội tụ" cũ)."""
import numpy as np

from .config import HEAD_TYPES


def best_run(runs):
    '''Đầu có AUROC val (lần chấm cuối) LỚN NHẤT; max_score = 0 + so sánh '>' nghiêm như mã gốc -> hoà giữ đầu ĐẦU TIÊN theo thứ tự (lr tăng
       dần); NaN không bao giờ được chọn. Không đầu nào > 0 -> None.'''
    best, score = None, 0.0
    for run in runs:
        v = run.val_auroc[-1] if run.val_auroc else float('nan')
        if v > score: best, score = run, v
    return best


def epoch_rule(best, budgets, threshold):
    '''best = {(ngân sách, loại đầu): đầu tốt nhất}. Với MỌI loại đầu, AUROC val tốt nhất của ngân sách lớn - của ngân sách nhỏ <= ngưỡng ->
       ngân sách nhỏ; ngược lại ngân sách lớn (+1e-12: số thực đúng ngưỡng vẫn tính là <=). -> (E chốt, {loại: chênh}).'''
    e0, e1 = min(budgets), max(budgets)
    diff = {t: float(best[(e1, t)].val_auroc[-1] - best[(e0, t)].val_auroc[-1]) for t in HEAD_TYPES}
    return (e0 if all(v <= threshold + 1e-12 for v in diff.values()) else e1), diff


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
