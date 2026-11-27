# -*- coding: utf-8 -*-
"""vindr28 — chấm lại VinDr-CXR trên 28 nhãn (module 94, docs/PLAN_94_VINDR_28_NHAN.md; Tuấn 27/11: "tôi cần số trên 28 nhãn thôi", "làm theo cách B và D").

Chỉ CHẤM LẠI dự đoán test đã lưu của 89 (+ 86f) và 92 — không học lại đầu đọc, không trích đặc trưng. Dùng lại hàm chấm / họ phép so của vindr_bench
(nhánh vindr-bench) và frozen_probe (nhánh frozen-probe) ở commit ghim; gói này chỉ thêm:

    stats28      AP / AUROC từng nhãn theo TỪNG lần rút bootstrap (véc-tơ hoá; = metrics.ap_weighted / auroc_weighted) + gộp theo cách B / D
    remap        ô có khuôn compare.make_cell với panel p21 mang số 28 nhãn -> gọi lại nguyên các họ phép so của 89t / 92
    notebook     bench (94a: 89 + 86f) · it_nhan (94b: 92b / 92e / 92g)

KHÔNG chứa ID Drive, email, mã ảnh."""
__version__ = '0.1.0'
