# -*- coding: utf-8 -*-
"""Hằng số của phép tính khoảng nới chung cho họ 9 ô "hơn + không kém" (3) − (1b) (docs/PLAN_90_SO_SANH_GIAI_PHAU.md §13.4 / §13.8; Tuấn duyệt 23/11
"okay approve"; Event 2026-11-23 "Tuấn duyệt dựng phép tính khoảng nới chung …"). ĐẶT TRƯỚC khi tính khối CXR-LT.

  Họ 9 ô = 3 ô CXR-LT (số quyết định, 78r tuyến tính, 78r k-NN) + 6 ô VinDr (k-NN / logistic × 1 / 10 / 100 % nhãn). Hai bộ dữ liệu độc lập -> mỗi khối có
  khoảng nới riêng ở mức 0,95^(số ô của khối / 9); tích hai mức = 0,95 -> cả họ cùng đúng với xác suất 95 %.
  Khối CXR-LT: hiệu (3) − (1b) cùng seed; lần rút = bootstrap theo bệnh nhân 2.000 lần, hạt 22101 = ĐÚNG dãy của 78r C9 / C10 (lần rút 78r đọc từ JSON,
  số quyết định tính lại trên cùng trọng số; lần rút thiếu lớp bị loại như 78r).
  Seed 43 = luật đặt trước. Seed 42 = đọc SAU khi đã thấy số điểm (mô tả; không đổi kết luận s43).
KHÔNG chứa ID Drive."""
import math

RUN_TAG = 'r90d_ho_9_o_khong_kem'
NOTEBOOK = '90d_ho_9_o_khong_kem'
FOLDER = 'stage4_nb90'                         # thư mục ghi của module 90 (dưới cxrlt_2024) — = region_loc.config.FOLDER
OUT_PREFIX = 'ho_9_o_khong_kem'
SEEDS = (43, 42)                               # 43: luật đặt trước | 42: đọc sau khi thấy số (mô tả)
SEED_ROLE = {43: 'luat dat truoc', 42: 'doc SAU khi da thay so (mo ta)'}
ROW_TEXT = {'ref': '(3) luong vung trong encoder + ghi nguoc (B.4 v2, EMA ep4)', 'ctl': '(1b) luong vung sau encoder, 1 lop, chi doc (EMA ep4)'}
PAIR_TEXT = '(3) tru (1b), cung seed backbone'

# ---------------------------------------------------------------------------------------------------- CXR-LT
# dữ liệu val = khối nhãn chuẩn của 74b / 78a (experiments/72_loss_kien_truc/dung_74c.DU_LIEU), CHỈ giữ hàng val (không giữ nhãn test)
DATA_FILES = (('cxrlt_inventory.json', ()), ('cxrlt_index.parquet', ()), ('emb_index.parquet', ('embeddings',)))
N_VAL = 39293                                  # val CXR-LT (in ở 78r C9 và C10, 22/11)
# 78r s43: JSON C9 của notebook (3) s43 (xong sau cùng -> đủ 6 hàng + cặp s42 tính lại trên CÙNG lần rút)
R78_DIR = ('stage4_nb78r', 'r78r_vt_b4v2_s43')
R78_JSON_RE = r'cap_s43_\d{8}_\d{6}\.json'
R78_KEY = {43: ('b4v2_s43', 'sau1_s43'), 42: ('b4v2', 'sau1')}          # (3), (1b) trong JSON 78r
PV_DIR = {43: ('r78r_vt_b4v2_s43', 'r78r_vt_sau1_s43'), 42: ('r78r_vt_b4v2_s42', 'r78r_vt_sau1_s42')}   # stage4_nb78r: pv_<đầu>_s42.npy
HEADS_78R = ('lin', 'knn')
PV_FILE = 'pv_{}_s42.npy'                      # đầu đọc nhẹ hạt 42 ở cả hai seed backbone
# số quyết định: fus_attn_hr gộp trọng số ep1–10 (Phần 3 của 78x) — p_gopA của hai lượt; JSON C10 của lượt (1b) để làm cổng
GTS_DIR = {43: ('r78z_Jv3b_dp_ema_v2_s43_gts', 'r78z_Jv3b_dp_ema_vt_sau1_s43_gts'), 42: ('r78z_Jv3b_dp_ema_v2_s42_gts', 'r78z_Jv3b_dp_ema_vt_sau1_s42_gts')}
GTS_FOLDER = 'stage4_nb78z'
DECISION_HEAD = 'fus_attn_hr'
P_GOPA = 'p_gopA_fus_attn_hr_s42.npy'
C10_JSON_RE = {43: r'quyet_dinh_vt_sau1_s43_\d{8}_\d{6}\.json', 42: r'quyet_dinh_vt_sau1_\d{8}_\d{6}\.json'}
C10_ROW_KEYS = ('VT sau1 (luot nay)', 'B.4 v2 (ghi nguoc)')            # (1b), (3) trong ket_qua của JSON C10
# bootstrap = 78r C9 / C10
N_BOOT, BOOT_SEED, HALF_SEED = 2000, 22101, 7801
N_CHECK_DRAWS = 20                             # cổng: số lần rút đầu của 78r tính lại từ pv_ phải TRÙNG JSON
TOL = 1e-9
# số đã in 22/11 (4 chữ số) — cổng nhận diện (đúng tệp / đúng cặp), không phải luật
EXPECT_78R = {43: dict(knn=0.0123, lin=0.0056), 42: dict(knn=0.0056, lin=0.0011)}   # (3) − (1b), AP bậc thang toàn val gộp view
EXPECT_C10 = {43: (-0.0044, -0.0083, 0.0051), 42: (0.0011, -0.0015, 0.0058)}   # (1b) − (3), số quyết định: toàn val / nửa A / nửa B (output C10)
EXPECT_TOL = 6e-5                              # số in 4 chữ số -> lệch <= 5e-5 (+ sai số float)
EXPECT_KEPT = 1999                             # 78r: 1.999 / 2.000 lần rút giữ (loại 1 lần thiếu lớp)

# ---------------------------------------------------------------------------------------------------- luật
CELLS_CXR = ('so_quyet_dinh', 'tuyen_tinh', 'knn')
CELL_TEXT = dict(so_quyet_dinh='so quyet dinh (fus_attn_hr gop trong so ep1-10)', tuyen_tinh='78r tuyen tinh', knn='78r k-NN')
HEAD_OF = dict(tuyen_tinh='lin', knn='knn')
DELTA_CXR = dict(so_quyet_dinh=0.008, tuyen_tinh=0.006, knn=0.007)    # PLAN_90 §13.4 (Tuấn duyệt 22/11)
MARGIN_CXR = 0.003                             # "hơn" CXR-LT = ngưỡng §10.2 (Claude đặt 23/11 cho cả 3 ô; hiệu điểm đã biết đều ≥ 0,003 -> không quyết ô nào)
N_CELLS, N_CXR = 9, 3
CXR_LEVEL = 100.0 * 0.95 ** (N_CXR / N_CELLS)                       # 98,30 %
VINDR_LEVEL = 100.0 * 0.95 ** ((N_CELLS - N_CXR) / N_CELLS)         # 96,64 % (= frozen_probe.config_s43.NONINF_LEVEL)
assert abs(CXR_LEVEL - 98.305) < 1e-3 and abs(VINDR_LEVEL - 96.638) < 1e-3 and abs(CXR_LEVEL * VINDR_LEVEL / 1e4 - 0.95) < 1e-12
assert set(DELTA_CXR) == set(CELLS_CXR) and set(HEAD_OF) < set(CELLS_CXR)

# ---------------------------------------------------------------------------------------------------- VinDr
VINDR_FOLDER = 'stage4_nb92'
R92G_DIR = ('stage4_nb92', 'r92g_it_nhan_vindr_s43')
R92G_JSON_RE = r'it_nhan_vindr_s43_(\d{8}_\d{6}(?:_lan\d+)?)\.json'
R92G_M21 = 'it_nhan_vindr_s43_m21_{}.npz'
R92B_DIR = ('stage4_nb92', 'r92b_it_nhan_vindr')
R92B_JSON_RE = r'it_nhan_vindr_(\d{8}_\d{6})\.json'
R92B_PRED = 'it_nhan_vindr_du_doan_{}.npz'
R92A_DIR = ('stage4_nb92', 'r92a_trich_dac_trung_vindr')           # nhãn / tách test của s42 (= 92f; 92g đã kiểm trùng)
VINDR_ROWS = {43: ('vt_b4v2_s43', 'vt_sau1_s43'), 42: ('vt_b4v2', 'vt_sau1')}
VINDR_TOL = 1e-12


def check_deps(fp_config, fp_config_s43, vb_compare):
    '''Hằng của gói phải TRÙNG frozen_probe / vindr_bench ở commit ghim (một nguồn sự thật) -> AssertionError nếu lệch.'''
    S = fp_config_s43
    assert S.DELTA_CXR == DELTA_CXR, f'delta CXR-LT {DELTA_CXR} KHAC frozen_probe.config_s43 {S.DELTA_CXR} -> DUNG'
    assert math.isclose(S.CXR_LEVEL, CXR_LEVEL, abs_tol=1e-12) and math.isclose(S.NONINF_LEVEL, VINDR_LEVEL, abs_tol=1e-12), 'muc khoang noi KHAC config_s43 -> DUNG'
    assert (S.NONINF_CELLS, S.NONINF_VINDR) == (N_CELLS, N_CELLS - N_CXR), 'so o ho KHAC config_s43 -> DUNG'
    assert (S.REF, S.NONINF_CONTROL) == VINDR_ROWS[43] and tuple(S.BASE[r] for r in VINDR_ROWS[43]) == VINDR_ROWS[42], 'hang VinDr KHAC config_s43 -> DUNG'
    assert S.RUN_TAGS['it_nhan'] == R92G_DIR[1] and fp_config.RUN_TAGS['it_nhan'] == R92B_DIR[1] and fp_config.RUN_TAGS['trich'] == R92A_DIR[1], 'RUN_TAG 92 KHAC -> DUNG'
    assert fp_config.FOLDER == VINDR_FOLDER and fp_config.MARGIN == vb_compare.MARGIN, 'thu muc / nguong VinDr KHAC -> DUNG'
    assert len(fp_config.FAMILY) == N_CELLS - N_CXR, 'so o VinDr KHAC -> DUNG'
    return True
