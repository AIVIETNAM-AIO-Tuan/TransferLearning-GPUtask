# -*- coding: utf-8 -*-
"""Hằng số của phép kiểm ít nhãn VinDr cho ba cặp mới (PLAN_92 §13; Tuấn 21/11 "được dựng đi"). Giao thức = §11 NGUYÊN VẸN: mọi hằng của dữ liệu,
lần rút, đầu đọc, chấm, luật lấy từ config.py (KHÔNG chép lại). Chỉ khác: hàng (6 hàng = 3 cặp), RUN_TAG / notebook, không bản nhạy, đối chứng âm cho
mọi hàng mới, cổng tái lập trên cặp RAD-DINO của 92a / 92b. Luật đặt trước khoá ở đây khi dựng (21/11). KHÔNG chứa ID Drive."""
from . import config as C
from .encoders.rows import ENCODERS, PAIRS, ROWS as SPECS, TITLE

FOLDER = C.FOLDER
RUN_TAGS = dict(trich='r92d_trich_dac_trung_vindr_bb', it_nhan='r92e_it_nhan_vindr_bb')
NOTEBOOKS = dict(trich='92d_trich_dac_trung_vindr_bb', it_nhan='92e_it_nhan_vindr_bb')
SOURCE_RAD = dict(trich=C.RUN_TAGS['trich'], it_nhan=C.RUN_TAGS['it_nhan'])          # cặp RAD-DINO đã chạy (92a / 92b, gói 0.1.0)

# ---- hàng: mỗi cặp (CÓ, KHÔNG) trong CÙNG backbone ----
ROWS = tuple(r for e in ENCODERS for r in PAIRS[e])
CODE = {r: f"{TITLE[SPECS[r].encoder]} {'CO' if SPECS[r].hang == 'co' else 'KHONG'}" for r in ROWS}
TEXT = {r: (f"{TITLE[SPECS[r].encoder]} {'CO luong vung' if SPECS[r].hang == 'co' else 'KHONG luong vung'}"
            + (f" (ghi nguoc x{SPECS[r].g:g}, tran {SPECS[r].rho:g})" if SPECS[r].hang == 'co' else '')
            + f", s42, EMA ep{SPECS[r].ep_bb}" + (f" / lich {SPECS[r].epochs} epoch" if SPECS[r].epochs != 6 else '')) for r in ROWS}
PAIR_TEXT = {e: f"{TITLE[e]}: CO - KHONG luong vung (ghi nguoc x{SPECS[PAIRS[e][0]].g:g}; EMA ep{SPECS[PAIRS[e][0]].ep_bb})" for e in ENCODERS}
RAD_PAIR = C.PAIR                                     # (vt_b4v2, vt_khong_luong) của 92a: (3) − (0) RAD-DINO
RAD_TEXT = 'RAD-DINO: (3) B.4 v2 - (0) KHONG luong vung (ghi nguoc x5; EMA ep4)'
assert len(ROWS) == 6 and set(CODE) == set(TEXT) == set(ROWS) and not set(ROWS) & set(C.ROWS)

# ---- đặc trưng / đầu đọc / tỉ lệ / luật: = config.py ----
FEATURES, PRIMARY_FEATURE = C.FEATURES, C.PRIMARY_FEATURE
PROBES, FRACTIONS, N_DRAWS, FAMILY = C.PROBES, C.FRACTIONS, C.N_DRAWS, C.FAMILY
CONFIGS = tuple((C.PRIMARY_FEATURE, p) for p in C.PROBES) + ((C.SENSITIVITY['feature'], 'knn'),)
# = chính (TB token patch x {k-NN, logistic}) + CLS chỉ k-NN (mô tả, chỉ điểm; logistic trên CLS bỏ: gấp đôi thời gian CPU của 92e). KHÔNG bản nhạy.
NEG_ROWS = ROWS                                       # đối chứng âm (nhãn xáo, 100 %) cho MỌI hàng mới
# Cổng tái lập (gói 0.2.0 trên dữ liệu đã chạy): k-NN 1 / 10 / 100 % + logistic 1 % của cặp RAD-DINO tính lại từ tệp đặc trưng 92a phải TRÙNG ô của 92b
REPRO = (('knn', 0.01), ('knn', 0.10), ('knn', 1.0), ('logistic', 0.01))
REPRO_TOL = 1e-9
