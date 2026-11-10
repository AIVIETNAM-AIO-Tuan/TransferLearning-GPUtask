"""Lựa chọn dựa trên val: lr tốt nhất (như evaluate_linear_classifiers của mã gốc), luật khoá epoch (PLAN_86 §5.1), kiểm hội tụ (§5.2)."""
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


def converged(run, threshold):
    '''AUROC val ở lần chấm cuối >= cực đại qua các lần chấm - ngưỡng (đường hữu hạn).'''
    a = np.asarray(run.val_auroc, dtype=float)
    return bool(len(a) and np.isfinite(a).all() and a[-1] >= a.max() - threshold - 1e-12)


def max_finite(values):
    v = [float(x) for x in values if np.isfinite(x)]
    return max(v) if v else float('nan')
