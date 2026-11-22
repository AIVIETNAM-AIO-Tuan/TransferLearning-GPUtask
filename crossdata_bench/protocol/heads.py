# -*- coding: utf-8 -*-
"""Một nhãn (C = 1) trên vòng học của `89` (vindr_bench @ c81af62) — sửa ĐÚNG một chỗ hình, không động mã của gói đó.

Vấn đề (đọc mã 22/11): LinearClassifier (heads/linear.py) và GLoRI (heads/glori.py) — mã công khai CheXFound chép nguyên văn — kết thúc bằng
`.squeeze(-1)`. Với C >= 2 lệnh đó không làm gì; với C = 1 logit thành [B] thay vì [B, 1] -> BCE với nhãn [B, 1] báo lệch hình (hoặc phát tán sai), dự
đoán [N] làm hỏng phần chấm theo cột. ML-Decoder ra [B, 1] (không squeeze).

Sửa: trong lúc chạy (context manager), thay `vindr_bench.protocols.ours.training.run_head` (tên mà train_step / predict tra lúc gọi) bằng bản bọc: kết quả
một chiều -> unsqueeze(-1); kết quả hai chiều -> trả NGUYÊN đối tượng (C >= 2: không đổi gì, bộ kiểm so từng bit). Thoát khối (kể cả khi lỗi) -> trả hàm
gốc. Không có chỗ nào khác của pipeline gọi run_head (đọc mã: pipeline / gates / notebook.ver2 không gọi)."""
import contextlib

PATCH_TARGET = 'vindr_bench.protocols.ours.training.run_head'


def _wrap(orig):
    def run_head(kind, model, blocks, last_patch):
        out = orig(kind, model, blocks, last_patch)
        return out.unsqueeze(-1) if out.ndim == 1 else out
    run_head.__wrapped__ = orig
    run_head.crossdata_single_label = True
    return run_head


@contextlib.contextmanager
def keep_class_dim():
    '''Trong khối: mọi đầu đọc của vòng học 89 ra logit [B, C] kể cả khi C = 1.'''
    from vindr_bench.protocols.ours import training as T
    orig = T.run_head
    assert not getattr(orig, 'crossdata_single_label', False), 'run_head da duoc boc (khoi long nhau) -> DUNG'
    T.run_head = _wrap(orig)
    try:
        yield
    finally:
        T.run_head = orig


def is_patched():
    from vindr_bench.protocols.ours import training as T
    return bool(getattr(T.run_head, 'crossdata_single_label', False))
