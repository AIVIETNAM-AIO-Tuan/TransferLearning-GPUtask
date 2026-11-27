# -*- coding: utf-8 -*-
"""Hằng số của module 94 (docs/PLAN_94_VINDR_28_NHAN.md). KHÔNG chứa ID Drive."""
FOLDER = 'stage4_nb94'                                       # thư mục ghi của module 94 (dưới cxrlt_2024)
RUN_TAGS = dict(bench='r94a_vindr28_benchmark', it_nhan='r94b_vindr28_it_nhan')
NOTEBOOKS = dict(bench='94a_vindr28_benchmark', it_nhan='94b_vindr28_it_nhan')

# ---- đại lượng (PLAN_94 §2) ----
STATS = ('ap_b', 'ap_d', 'auroc')
STAT_TEXT = dict(ap_b='AP28 cach B (nhan vang trong lan rut bi BO khoi TB)', ap_d='AP28 cach D (nhan vang trong lan rut = 0, tap co dinh 27 nhan)',
                 auroc='AUROC28 (nhan vang = 0, chia 28; B = D)')
STAT_SHORT = dict(ap_b='AP28-B', ap_d='AP28-D', auroc='AUROC28')

# ---- bootstrap: CÙNG dãy rút của 89t / 92 (vindr_bench.evaluation.panels.N_BOOT / BOOT_SEED / HALF_SEED) — kiểm ở setup ----
N_BOOT, BOOT_SEED, HALF_SEED = 2000, 8502, 8501
CHUNK = 500                                                  # số lần rút mỗi khúc khi tính véc-tơ hoá (giới hạn bộ nhớ ~6 mảng [CHUNK, N] mỗi cột)

# ---- cổng ----
TOL_89 = 1e-12                                               # 94a: mAP21 từng lần rút so với bộ nhớ đệm của 89t (cùng mã, cùng dữ liệu)
FLAG_92, STOP_92 = 0.002, 0.01                               # 94b: lệch mAP21 / cận CI so với JSON của 92 (dự đoán float16): > FLAG gắn cờ, > STOP dừng
RESUME_EVERY = 0.10                                          # lưu tệp resume mỗi ~10 % số bộ dự đoán cần tính

# ---- nguồn 89 (đọc) ----
SRC_89T = ('stage4_nb89', 'r89t_tong_hop')                   # bộ nhớ đệm bootstrap của 89t (cổng)

# ---- nguồn 92 (đọc): (tên, thư mục, tiền tố tệp) — hàng đọc theo config của frozen_probe ----
SRC_92 = (('s42', ('stage4_nb92', 'r92b_it_nhan_vindr'), 'it_nhan_vindr'),
          ('bb', ('stage4_nb92', 'r92e_it_nhan_vindr_bb'), 'it_nhan_vindr_bb'),
          ('s43', ('stage4_nb92', 'r92g_it_nhan_vindr_s43'), 'it_nhan_vindr_s43'))
SRC_92A = ('stage4_nb92', 'r92a_trich_dac_trung_vindr')      # nhãn VinDr + manifest (vân tay = các lượt 89)

# ---- đọc kết quả (PLAN_94 §3) ----
READING = ('PHAN TICH SAU (PLAN_94 3): luat dat truoc cua 89 / 92 tinh tren 21 lop chinh; ban 28 nhan bao hieu + CI95 danh nghia + khoang DONG THOI max-T '
           '95 % trong dung ho phep so nhu ban 21 lop + hai nua; nhan "dang luat cu" (hieu >= 0,01, can duoi dong thoi > 0, cung dau o hai nua) CHI MO TA '
           '(nguong 0,01 dat cho mAP21). CI chua 0 = CHUA phat hien khac, khong phai "bang nhau".')
