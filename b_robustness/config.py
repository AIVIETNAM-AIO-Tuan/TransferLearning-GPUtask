"""Hằng số module 87 (PLAN_KIEM_THU §15–§16). Không chứa ID Drive.

Các số lấy từ nguồn đã chạy (test_87 đối chiếu lại): số quyết định đọc từ output ô p3_C6S của notebook gói 84; cấu hình seed = nhanh_c.cau_hinh;
mốc B s42 / SOTA / bảng 85d, 85g = dung_85."""

# ---- gói nhanh_c (module 84, nhánh mồ côi nhanh-c): Phần 3 biến thể Bd ----
NHANH_C_BRANCH = 'nhanh-c'
NHANH_C_COMMIT = '628750a69c1d7c0373f7c19f05025f6b4bdc88f7'          # lượt s44 _lai chạy bằng commit này; 12 bước Phần 3 Bd trùng commit lượt s45 / s46
NHANH_C_COMMIT_S45_S46 = 'e761f2075fb06dd09432b91135df6abbdddc92cc'
NHANH_C_VERSION = 'nhanh_c_v1'
NHANH_C_VARIANT = 'Bd'
PART3_STEPS = ('bd3_C0', 'z_C1', 'z_C2', 'bd_C2B', 'z_C2C', 'z_c3', 'z_C3B', 'z_C3C', 'z_C4', 'z_c5', 'z_C5B', 'p3_C6S')   # bỏ bd3_TONG_HOP, p3_C9S

# ---- backbone seed B mới (K3) ----
K3_SEEDS = (44, 45, 46)
ROW = {44: 'B_s44_lai', 45: 'B_s45', 46: 'B_s46'}                      # tên hàng trong bảng / tên tệp p2_<hàng>_*
RUN_K3 = {s: 'r87c_' + ROW[s] for s in K3_SEEDS}
FOLDER_K3 = 'stage4_nb87c'
NOTEBOOK_K3 = {s: f'87c_k3_vindr_{ROW[s]}.ipynb' for s in K3_SEEDS}
DECISION_SCORE = {44: 0.2874, 45: 0.2877, 46: 0.2903}                  # fus_attn_hr gộp trọng số ep1-10, val gộp view (ô p3_C6S của notebook gói 84)
DECISION_TOL = 5e-4
RUN_NOTEBOOK_84 = {44: '84_nhanh_c_b_s44_lai.ipynb', 45: '84_nhanh_c_b_s45.ipynb', 46: '84_nhanh_c_b_s46.ipynb'}
RUN_COMMIT_84 = {44: NHANH_C_COMMIT, 45: NHANH_C_COMMIT_S45_S46, 46: NHANH_C_COMMIT_S45_S46}

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
    'B_s43': dict(seed=43, variant='B (ghep cap C s43; chay lai 08/11)', k3=False, x=('stage4_nb84', 'r84x_B_s43'), z=('stage4_nb84', 'r84z_B_s43'),
                  gts=('stage4_nb84', 'r84z_B_s43_gts'), decision_nb=None,
                  decision_source='luot chay lai 08/11 khong in so o o C6S chuan; Event 09/11 "TONG HOP" ghi 0,2888 -> KHONG doi chieu'),
    **{ROW[s]: dict(seed=s, variant='Bd (doc lap, khoa ep5)', k3=True, x=('stage4_nb84', f'r84x_{ROW[s]}'), z=('stage4_nb84', f'r84z_{ROW[s]}'),
                    gts=('stage4_nb84', f'r84z_{ROW[s]}_gts'), decision_nb=DECISION_SCORE[s], decision_source=f'{RUN_NOTEBOOK_84[s]} (o p3_C6S)')
       for s in K3_SEEDS},
}
K0_BACKBONE_EPOCH = 5                                                  # backbone khoá ep5 (rs_ lưu ep đếm từ 0 = 4)
K0_PART2_EPOCHS = tuple(range(1, 11))

# ---- thư mục ghi của notebook 87 ----
RUN_87A, RUN_87B, RUN_87D = 'r87a_kiem_nguon', 'r87b_k5_thong_ke', 'r87d_k3_bang'
FOLDER_87A, FOLDER_87B, FOLDER_87D = 'stage4_nb87a', 'stage4_nb87b', 'stage4_nb87d'
