# -*- coding: utf-8 -*-
"""noninf_family — khoảng nới chung cho họ 9 ô "hơn + không kém" (3) − (1b) của PLAN_90 §13.4 / §13.8 (module 90; Tuấn duyệt 23/11).

  config.py    hằng đặt trước: nguồn tệp, lần rút, δ, mức khoảng, ngưỡng "hơn"; check_deps = hằng TRÙNG frozen_probe / vindr_bench ở commit ghim
  cxrlt.py     khối CXR-LT: val (khối nhãn chuẩn, chỉ hàng val), gộp view, hai nửa, AP có trọng số, bootstrap theo bệnh nhân CÙNG dãy 78r C9 / C10
  vindr.py     ô VinDr: s43 từ JSON + lần rút của 92g; s42 từ dự đoán float16 của 92b (bootstrap của 92g) + cổng so với ô 92b
  family.py    khoảng nới mỗi khối (vindr_bench.evaluation.compare.max_t) + luật "không kém" / "hơn" + gộp họ 9 ô
  notebook.py  ô của notebook 90d_ho_9_o_khong_kem (Colab CPU)

Phụ thuộc (clone theo commit ghim trong notebook): vindr_bench (nhánh vindr-bench) và frozen_probe (nhánh frozen-probe). Không chứa ID Drive."""
__version__ = '0.1.0'
