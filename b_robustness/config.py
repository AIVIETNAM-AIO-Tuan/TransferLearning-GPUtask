"""Hằng số module 87 (PLAN_KIEM_THU §15–§16). Không chứa ID Drive.

Các số lấy từ nguồn đã chạy (test_87 đối chiếu lại): số quyết định đọc từ output ô p3_C6S của notebook gói 84 (s43: output 87a); cấu hình seed = nhanh_c.cau_hinh;
mốc B s42 / SOTA / bảng 85d, 85g = dung_85."""

# ---- gói nhanh_c (module 84, nhánh mồ côi nhanh-c): Phần 3 biến thể Bd (s44 _lai / s45 / s46) và B (s43; Tuấn duyệt 10/11, PLAN_KIEM_THU §17) ----
NHANH_C_BRANCH = 'nhanh-c'
NHANH_C_COMMIT = '628750a69c1d7c0373f7c19f05025f6b4bdc88f7'          # lượt s44 _lai chạy bằng commit này; 12 bước Phần 3 Bd trùng commit lượt s45 / s46,
NHANH_C_COMMIT_S45_S46 = 'e761f2075fb06dd09432b91135df6abbdddc92cc'  # 12 bước Phần 3 B trùng commit lượt s43 chạy lại (test_87 so qua git)
NHANH_C_COMMIT_S43 = 'dca686ae85ac0c0f192be63b8bce91df4318a98a'
NHANH_C_VERSION = 'nhanh_c_v1'
PART3_STEPS = {                                                        # = nhanh_c.ds_buoc(3, biến thể) bỏ PART3_SKIPPED (quyết định / tổng hợp của 84)
    'Bd': ('bd3_C0', 'z_C1', 'z_C2', 'bd_C2B', 'z_C2C', 'z_c3', 'z_C3B', 'z_C3C', 'z_C4', 'z_c5', 'z_C5B', 'p3_C6S'),
    'B': ('b3_C0', 'z_C1', 'z_C2', 'b_C2B', 'z_C2C', 'z_c3', 'z_C3B', 'z_C3C', 'z_C4', 'z_c5', 'z_C5B', 'p3_C6S'),
}
PART3_SKIPPED = {'Bd': ('bd3_TONG_HOP', 'p3_C9S'), 'B': ('b3_QUYET', 'p3_C9S')}
PART3_C0 = {'Bd': 'bd3_C0', 'B': 'b3_C0'}                               # bước có khối ghi Drive -> thay 3 dòng ghi (backbone_b.redirect_output)

# ---- backbone seed B mới (K3) ----
K3_SEEDS = (43, 44, 45, 46)
K3_VARIANT = {43: 'B', 44: 'Bd', 45: 'Bd', 46: 'Bd'}                   # biến thể gói nhanh_c của lượt backbone (B: ghép cặp C cùng seed, ep = epoch C đã chọn; Bd: khoá ep5)
ROW = {43: 'B_s43', 44: 'B_s44_lai', 45: 'B_s45', 46: 'B_s46'}        # tên hàng trong bảng / tên tệp p2_<hàng>_*
RUN_K3 = {s: 'r87c_' + ROW[s] for s in K3_SEEDS}
FOLDER_K3 = 'stage4_nb87c'
NOTEBOOK_K3 = {s: f'87c_k3_vindr_{ROW[s]}.ipynb' for s in K3_SEEDS}
DECISION_SCORE = {43: 0.2888, 44: 0.2874, 45: 0.2877, 46: 0.2903}      # fus_attn_hr gộp trọng số ep1-10, val gộp view
DECISION_TOL = 5e-4
RUN_NOTEBOOK_84 = {43: '84_nhanh_c_b_s43 (1).ipynb', 44: '84_nhanh_c_b_s44_lai.ipynb', 45: '84_nhanh_c_b_s45.ipynb', 46: '84_nhanh_c_b_s46.ipynb'}
RUN_COMMIT_84 = {43: NHANH_C_COMMIT_S43, 44: NHANH_C_COMMIT, 45: NHANH_C_COMMIT_S45_S46, 46: NHANH_C_COMMIT_S45_S46}
# nguồn số quyết định: s44 _lai / s45 / s46 = output ô p3_C6S của notebook gói 84; s43 = notebook lượt chạy lại KHÔNG in số ở p3_C6S ->
# số trên Drive (JSON _gts) do 87a đã chạy in (dòng B_s43, cột "so QD Drive") = Event 2026-11-09 "TỔNG HỢP"
DECISION_SOURCE = {43: '87a_k0_kiem_nguon.ipynb', 44: RUN_NOTEBOOK_84[44], 45: RUN_NOTEBOOK_84[45], 46: RUN_NOTEBOOK_84[46]}

# ---- mốc B s42 + SOTA (85f, lưới lr khoá 09/11) ----
ANCHOR = 'Jv3b_dp_ema_v2_skip7_s42'                                    # tên hàng 85f của B s42
ANCHOR_ROW = 'B_s42'
ANCHOR_DECISION = 0.2883
SOTA = ('raddino', 'chexfound', 'chexworld')
FOLDER_85F = 'stage4_nb85f'
RUN_85F = {b: f'r85f_{b}_luoi3' for b in (ANCHOR,) + SOTA}
NOTEBOOK_85F = {b: f'85f_p2_{b}' for b in (ANCHOR,) + SOTA}
TABLE_85G = ('stage4_nb85g', 'r85g_p2')
TABLE_85D = ('stage4_nb85d', 'r85d_p1_B')
VINDR_TEST = ('stage4_nb85b', 'r85b_vindr_test')
VINDR_TRAIN = {k: ('stage4_nb85e', f'r85e_vindr_train_p{k}') for k in (1, 2, 3)}

# ---- đầu đọc ----
P1_HEADS = ('mldec_hr', 'fus_attn_hr')                                 # đầu MIMIC chấm ở P1 (bảng 85d)
P1_ALL_HEADS = ('mldec_hr', 'fus_attn_hr', 'fus_tb_hr')                # đầu Phần 3 ghi ở P1
B_HEADS = ('ml', 'ml_patch', 'lin')                                    # đầu P2 của hàng B
SOTA_HEADS = ('ml', 'lin')
HEAD_SEEDS = (8520, 8521, 8522, 8523, 8524)                            # = SEED_85F: 5 lần học đầu
ML_GRID, ML_EPOCHS, ML_BATCH, ML_WD = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2), 15, 64, 1e-4          # lưới khoá (Tuấn 09/11)
LIN_GRID, LIN_EPOCHS, LIN_BATCH, LIN_WD = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1), 100, 1024, 1e-4

# ---- nhãn + thống kê (= 85g / 85d) ----
N_LABELS, N_MAIN, MIN_TEST_POSITIVES = 28, 21, 10
RADDINO7 = ('Lung Opacity', 'Cardiomegaly', 'Pleural thickening', 'Aortic enlargement', 'Pulmonary fibrosis', 'Tuberculosis', 'Pleural effusion')
RIB = 'Rib fracture'
HALF_SEED, BOOT_SEED, N_BOOT = 8501, 8502, 2000
RULE_THRESHOLD = 0.01

# ---- K0: nguồn từng backbone seed (= cau_hinh của gói 84; s42 = notebook 78x / 78z riêng) ----
K0_SOURCES = {
    'B_s42': dict(seed=42, variant='B (78x skip7 s42, notebook rieng)', k3=False, x=('stage4_nb78x', 'r78x_J_v3b_dp_ema_v2_skip7_s42'),
                  z=('stage4_nb78z', 'r78z_Jv3b_dp_ema_v2_skip7_s42'), gts=('stage4_nb78z', 'r78z_Jv3b_dp_ema_v2_skip7_s42_gts'), decision_nb=0.2883,
                  decision_source='78x_jca_v3b_dp_ema_v2_skip7_s42.ipynb (C6S; dung_85.SO_QD_85)'),
    'B_s43': dict(seed=43, variant='B (ghep cap C s43; chay lai 08/11)', k3=True, x=('stage4_nb84', 'r84x_B_s43'), z=('stage4_nb84', 'r84z_B_s43'),
                  gts=('stage4_nb84', 'r84z_B_s43_gts'), decision_nb=None,
                  decision_source='luot chay lai 08/11 khong in so o o C6S chuan; Event 09/11 "TONG HOP" ghi 0,2888 -> KHONG doi chieu'),
    **{ROW[s]: dict(seed=s, variant='Bd (doc lap, khoa ep5)', k3=True, x=('stage4_nb84', f'r84x_{ROW[s]}'), z=('stage4_nb84', f'r84z_{ROW[s]}'),
                    gts=('stage4_nb84', f'r84z_{ROW[s]}_gts'), decision_nb=DECISION_SCORE[s], decision_source=f'{RUN_NOTEBOOK_84[s]} (o p3_C6S)')
       for s in K3_SEEDS if K3_VARIANT[s] == 'Bd'},
}
# 87a đã chạy 10/11 bằng gói @ 8a3fd74 khi s43 còn k3=False (in sự thật để Tuấn quyết): s43 đủ P1 + P2, ep5, số Drive 0.2888 -> Tuấn duyệt đưa vào K3 (§17)
K0_BACKBONE_EPOCH = 5                                                  # backbone khoá ep5 (rs_ lưu ep đếm từ 0 = 4)
K0_PART2_EPOCHS = tuple(range(1, 11))

# ---- thư mục ghi của notebook 87 ----
RUN_87A, RUN_87B, RUN_87D = 'r87a_kiem_nguon', 'r87b_k5_thong_ke', 'r87d_k3_bang'
FOLDER_87A, FOLDER_87B, FOLDER_87D = 'stage4_nb87a', 'stage4_nb87b', 'stage4_nb87d'
