# -*- coding: utf-8 -*-
"""Hằng số của phép kiểm ít nhãn (PLAN_92 §11). Luật đặt trước khoá ở đây khi dựng (19/11) — không đổi sau khi có số. KHÔNG chứa ID Drive."""
import math

FOLDER = 'stage4_nb92'                                   # thư mục ghi của module 92 (dưới cxrlt_2024)
RUN_TAGS = dict(trich='r92a_trich_dac_trung_vindr', it_nhan='r92b_it_nhan_vindr')
NOTEBOOKS = dict(trich='92a_trich_dac_trung_vindr', it_nhan='92b_it_nhan_vindr')
STORE = 'kcb'                                            # VinDr: kho đệm vuông, KHÔNG cân bằng histogram (như 89 Ver2 / Việc B)

# ---- hàng (tên lớp bọc của vindr_bench) ----
PAIR = ('vt_b4v2', 'vt_khong_luong')                     # (3) CÓ luồng vùng + ghi ngược, (0) KHÔNG luồng vùng — cùng seed 42, cùng thứ tự lô, EMA ep4
REF, CONTROL = PAIR
ROWS = PAIR + ('vt_mot_chieu', 'vt_sau4', 'vt_sau1', 'raddino', 'asg', 'b')
CODE = dict(vt_b4v2='(3)', vt_khong_luong='(0)', vt_mot_chieu='(2)', vt_sau4='(1a)', vt_sau1='(1b)', raddino='RAD-DINO', asg='ASG', b='ArrE')
TEXT = dict(vt_b4v2='(3) luong vung trong encoder + ghi nguoc (B.4 v2, s42, EMA ep4)',
            vt_khong_luong='(0) KHONG luong vung (B1 + drop-path + EMA, s42, EMA ep4)',
            vt_mot_chieu='(2) luong vung trong encoder, chi doc (s42, EMA ep4)',
            vt_sau4='(1a) luong vung sau encoder, 4 lop, chi doc (s42, EMA ep4)',
            vt_sau1='(1b) luong vung sau encoder, 1 lop, chi doc (s42, EMA ep4)',
            raddino='RAD-DINO goc (microsoft/rad-dino, chua tinh chinh)',
            asg='ASG (MICCAI 2024, ViT-B/16 224, trong so cong bo)',
            b='ArrE (cong thuc B, s42, ep5, co duong o cat)')
assert set(CODE) == set(TEXT) == set(ROWS) and len(ROWS) == len(set(ROWS))
DESCRIPTIVE_VS_CONTROL = ('vt_mot_chieu', 'vt_sau4', 'vt_sau1', 'raddino', 'asg', 'b')     # hiệu X − (0), mô tả
WRITE_BACK_PAIR = ('vt_b4v2', 'vt_mot_chieu')                                               # (3) − (2): phần của ghi ngược, mô tả

# ---- đặc trưng ----
FEATURES = ('mean_patch', 'glob')                        # TB token patch tầng cuối sau LayerNorm (= G của 78r) | vector toàn ảnh của lớp bọc (CLS)
PRIMARY_FEATURE = 'mean_patch'
FEATURE_TEXT = dict(mean_patch='TB token patch tang cuoi (sau LayerNorm cuoi; = G cua 78r)', glob='vector toan anh cua lop boc (CLS)')
EXTRACT_BATCH = 64                                       # = khúc backbone của lớp bọc (89 Ver2)

# ---- tỉ lệ nhãn + lần rút ----
POOL_SPLITS = ('train', 'val')                           # 15.000 ảnh train chính thức VinDr (train 13.500 + val 1.500 của 89; không bước nào cần val)
TEST_SPLIT = 'test'                                      # 3.000 ảnh test chính thức
FRACTIONS = (0.01, 0.10, 1.0)
FRACTION_CODE = {0.01: 'f1', 0.10: 'f10', 1.0: 'f100'}
FRACTION_TEXT = {0.01: '1 %', 0.10: '10 %', 1.0: '100 %'}
N_DRAWS = {0.01: 10, 0.10: 5, 1.0: 1}
DRAW_SEED = 9211

# ---- đầu đọc (không lr; không chọn gì trên val / test) ----
PROBES = ('knn', 'logistic')
PROBE_TEXT = dict(knn='k-NN', logistic='logistic')


def knn_k(n):
    '''k = round(sqrt(n)) -> 12 / 39 / 122 với 150 / 1.500 / 15.000 ảnh (đặt trước; không chọn theo dữ liệu).'''
    return max(1, int(round(math.sqrt(int(n)))))


LOGISTIC_C = 1.0                                         # ½‖w‖² + C·Σ loss (quy ước sklearn), không phạt hệ số chặn
LBFGS = dict(maxiter=20000, maxfun=40000, ftol=1e-12, gtol=1e-6, maxcor=20)   # trên mục tiêu CHIA n (cùng nghiệm); hội tụ = success hoặc |grad| <= GRAD_OK
GRAD_OK = 1e-5
SENSITIVITY = dict(logistic_C=(0.1, 10.0), knn_k=50, knn_k_fractions=(0.10, 1.0), feature='glob')   # chỉ cặp chính, mô tả
NEG_SEED = 9212                                          # đối chứng âm: nhãn học bị xáo (100 %, cặp chính)
NEG_MAX_ABOVE_CHANCE = 0.05                              # mAP21 nhãn xáo phải <= mức may rủi + 0,05 (bắt rò nhãn; mức thật ~0,45–0,55)

# ---- luật đặt trước (PLAN_92 §11.3; khoá 19/11) ----
MARGIN = 0.01                                            # = vindr_bench.evaluation.compare.MARGIN (dạng luật VinDr của PLAN_89 §7.4 / PLAN_90 §10.2)
FAMILY = tuple((p, f) for p in PROBES for f in FRACTIONS)                     # 6 phép (3) − (0), đặc trưng chính
LEVEL = 95.0
