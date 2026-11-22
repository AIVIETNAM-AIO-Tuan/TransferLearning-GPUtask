# -*- coding: utf-8 -*-
"""Shenzhen (lao) — bảng khoá / tách / nhãn từ tệp chia BenchmarkTransformers @ 68b7f442 (463 / 65 / 134; như CheXFound / CheXWorld). Đọc tệp chia bằng
external_benchmarks.splits của `88` (commit ghim; md5 từng tệp = external_benchmarks.config.SHENZHEN_SPLITS). Khoá = tên PNG bỏ đuôi (CHNCXR_####_<0|1>);
bộ không có mã người -> cụm = chính ảnh (giả định mỗi ảnh một người, như PLAN_88 §14.2). Thứ tự hàng: train, val, test; khoá sắp tăng trong mỗi tách."""
LABEL = 'Tuberculosis'


def table(lists):
    '''lists = external_benchmarks.splits.read_shenzhen_lists(...) ({tách: [(tên .png, nhãn)]}) -> DataFrame(khoa, tach, nhom, Tuberculosis).'''
    import pandas as pd
    rows = []
    for split in ('train', 'val', 'test'):
        for fname, lab in sorted(lists[split]):
            assert fname.endswith('.png'), fname
            rows.append((fname[:-4], split, fname[:-4], int(lab)))
    return pd.DataFrame(rows, columns=['khoa', 'tach', 'nhom', LABEL])
