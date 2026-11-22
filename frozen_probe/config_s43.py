# -*- coding: utf-8 -*-
"""Hằng số của phép kiểm ít nhãn VinDr ở SEED 43 cho (3), (1b), (0) (docs/PLAN_90_SO_SANH_GIAI_PHAU.md §13.8; Tuấn 22/11 "dựa vào plan đó dựng luôn").
Giao thức = PLAN_92 §11 NGUYÊN VẸN: mọi hằng của dữ liệu, lần rút, đầu đọc, chấm, dạng luật lấy từ config.py (KHÔNG chép lại). Chỉ khác: ba hàng s43 (lớp
bọc encoders/vt_s43 = lớp bọc s42 của vindr_bench, tệp s43), không bản nhạy, không CLS, đối chứng âm cho ba hàng, cổng tái lập trên cặp RAD-DINO s42 của
92a / 92b (= 92e), và ba bảng đọc ĐẶT TRƯỚC (khoá khi dựng 22/11, trước khi có số s43):

  (A) họ 6 phép (3) − (0) s43 theo luật PLAN_92 §11.3 (max-T 95 % trong họ; "hơn" = hiệu ≥ +0,01 VÀ cận dưới đồng thời > 0 VÀ dương ở hai nửa test) —
      lặp đúng phép đo s42 của 92b.
  (B) xác nhận seed 43 từng ô s42 "hơn" của (A) — Tuấn CHỌN B1 ngày 22/11, trước khi có số s43 (CONFIRM_RULE):
      B1 (LUẬT): ô s43 cũng "hơn" theo đúng luật §11.3 của (A) — áp lại đúng luật đã gọi ô đó "hơn" ở s42;
      B2 (chỉ in kèm, KHÔNG thuộc luật): cùng dấu VÀ CI95 riêng của ô (danh nghĩa) không chứa 0 — cách (a) duyệt 22/11 cho ba ô 78r CXR-LT.
  (C) phần VinDr của họ "hơn + không kém" (3) − (1b) (PLAN_90 §13.4; Tuấn duyệt δ 22/11): họ 9 ô = 3 ô CXR-LT (số quyết định, 78r tuyến tính, 78r k-NN) +
      6 ô VinDr. Hai khối dữ liệu độc lập (bệnh nhân CXR-LT / ảnh VinDr) -> khoảng đồng thời 95 % của cả họ = max-T TRONG từng khối ở mức 0,95^(số ô của
      khối / 9) (tích hai mức = 0,95). Khối VinDr (6 ô) ở mức 0,95^(6/9) = 96,64 %. "Không kém" = cận dưới > −δ ở MỌI ô; "hơn" = dạng luật §11.3 trên
      khoảng này. Khối CXR-LT (3 ô, mức 0,95^(3/9) = 98,30 %) tính ở bước sau (lần rút 78r s43 đã lưu; số quyết định cần tính lại lần rút).
      δ VinDr (PLAN_90 §13.4: "theo nửa độ rộng đồng thời đã đo ở 92b, đặt khi dựng 92"): nửa độ rộng khoảng đồng thời 92b (q = 2,64, họ 6 phép) × z(họ 9
      ô, Šidák 2,766) / 2,64, làm tròn LÊN tới 0,001 — cùng cách đã ra δ CXR-LT (nửa độ rộng danh nghĩa × 2,766 / 1,96).
KHÔNG chứa ID Drive."""
import math

from . import config as C
from .encoders.vt_s43 import ROWS, S43

FOLDER = C.FOLDER
RUN_TAGS = dict(trich='r92f_trich_dac_trung_vindr_s43', it_nhan='r92g_it_nhan_vindr_s43')
NOTEBOOKS = dict(trich='92f_trich_dac_trung_vindr_s43', it_nhan='92g_it_nhan_vindr_s43')
SOURCE_S42 = dict(trich=C.RUN_TAGS['trich'], it_nhan=C.RUN_TAGS['it_nhan'])            # 92a / 92b (s42, gói 0.1.0)

# ---- hàng ----
BASE = {r: S43[r]['base'] for r in ROWS}                                                # hàng s43 -> hàng s42 cùng vị trí của 92a
REF, NONINF_CONTROL, CONTROL = 'vt_b4v2_s43', 'vt_sau1_s43', 'vt_khong_luong_s43'       # (3), (1b), (0)
CODE = {r: f'{C.CODE[BASE[r]]} s43' for r in ROWS}
TEXT = {r: C.TEXT[BASE[r]].replace('s42', 's43') for r in ROWS}
CODE42 = {BASE[r]: f'{C.CODE[BASE[r]]} s42' for r in ROWS}
TEXT42 = {BASE[r]: C.TEXT[BASE[r]] for r in ROWS}
assert set(BASE.values()) == {'vt_b4v2', 'vt_sau1', 'vt_khong_luong'} and all('s43' in t and 's42' not in t for t in TEXT.values())

# ---- đặc trưng / đầu đọc / tỉ lệ / dạng luật: = config.py ----
FEATURES, PRIMARY_FEATURE = C.FEATURES, C.PRIMARY_FEATURE
PROBES, FRACTIONS, N_DRAWS, FAMILY = C.PROBES, C.FRACTIONS, C.N_DRAWS, C.FAMILY
CONFIGS = tuple((C.PRIMARY_FEATURE, p) for p in C.PROBES)                              # chỉ cấu hình chính; KHÔNG bản nhạy, KHÔNG CLS
NEG_ROWS = ROWS                                                                         # đối chứng âm (nhãn xáo, 100 %) cho cả ba hàng s43
# Cổng tái lập (= 92e): k-NN 1 / 10 / 100 % + logistic 1 % của cặp RAD-DINO s42 tính lại từ tệp 92a phải TRÙNG ô của 92b
REPRO = (('knn', 0.01), ('knn', 0.10), ('knn', 1.0), ('logistic', 0.01))
REPRO_TOL = 1e-9

# ---- (B) xác nhận seed 43 (Tuấn chọn B1 ngày 22/11, trước khi có số s43; B2 chỉ in kèm) ----
CONFIRM = ('B1', 'B2')
CONFIRM_RULE = 'B1'
CONFIRM_TEXT = dict(B1='o s43 cung "hon" theo luat 11.3 cua (A) (LUAT — Tuan chon 22/11)',
                    B2='cung dau VA CI95 rieng cua o (danh nghia) khong chua 0 (chi in kem, KHONG thuoc luat)')
assert CONFIRM_RULE in CONFIRM and CONFIRM[0] == CONFIRM_RULE

# ---- (C) phần VinDr của họ "hơn + không kém" (3) − (1b) (δ Tuấn duyệt 22/11) ----
NONINF_CELLS, NONINF_VINDR = 9, len(C.FAMILY)
NONINF_LEVEL = 100.0 * 0.95 ** (NONINF_VINDR / NONINF_CELLS)                            # 96,64 %: max-T trong khối VinDr
CXR_LEVEL = 100.0 * 0.95 ** ((NONINF_CELLS - NONINF_VINDR) / NONINF_CELLS)             # 98,30 %: khối CXR-LT (bước sau)
Q_92B = 2.64                                                                            # q của họ 6 phép 92b (in 20/11)
Z_SIDAK9 = 2.766                                                                        # z hai phía ở mức 0,95^(1/9) (Šidák 9 ô; = cách ra δ CXR-LT)
# khoảng đồng thời (3) − (0) của 92b (in 20/11, 3 chữ số; PLAN_92 §11.6) -> nửa độ rộng
CI_92B = {('knn', 0.01): (0.020, 0.030), ('knn', 0.10): (0.033, 0.057), ('knn', 1.0): (0.026, 0.058),
          ('logistic', 0.01): (0.016, 0.024), ('logistic', 0.10): (0.005, 0.020), ('logistic', 1.0): (-0.009, 0.024)}


def delta_from_92b(lo, hi):
    '''δ của một ô VinDr = nửa độ rộng khoảng đồng thời 92b × Z_SIDAK9 / Q_92B, làm tròn LÊN tới 0,001.'''
    return math.ceil(round((hi - lo) / 2 * Z_SIDAK9 / Q_92B * 1000, 6)) / 1000


DELTA = {('knn', 0.01): 0.006, ('knn', 0.10): 0.013, ('knn', 1.0): 0.017, ('logistic', 0.01): 0.005, ('logistic', 0.10): 0.008, ('logistic', 1.0): 0.018}
DELTA_CXR = dict(so_quyet_dinh=0.008, tuyen_tinh=0.006, knn=0.007)                    # PLAN_90 §13.8 (Tuấn duyệt 22/11) — khối CXR-LT, bước sau
assert set(DELTA) == set(C.FAMILY) == set(CI_92B) and all(DELTA[k] == delta_from_92b(*CI_92B[k]) for k in DELTA), 'DELTA KHAC cong thuc -> DUNG'
assert abs(NONINF_LEVEL - 96.638) < 1e-3 and abs(CXR_LEVEL - 98.305) < 1e-3
