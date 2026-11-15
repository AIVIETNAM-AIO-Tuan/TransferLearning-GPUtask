# -*- coding: utf-8 -*-
"""Hằng số giao thức Ver2 (PLAN_89 §1, §5.1 = giao thức `86b`, PLAN_86 §3–§4 theo mã CheXFound @ 7e288d0) — chép từ vindr_protocol/config.py của 86
(Protocol / MainRun / CurveDiagnostic / steps_per_epoch nguyên giá trị; tests/test_ver2.py so với tệp nguồn), thêm loại đầu Linear-1 và nhánh chọn lr
theo AP val (chỉ mô tả)."""
import math
from dataclasses import dataclass

CHEXFOUND_COMMIT = '7e288d06532f24abedb7d00781185c36727d9d69'
HEAD_KINDS = ('lin1', 'lin4', 'mldec', 'glori')          # cùng tên với Ver1 (protocols/chexfound_env/runner.HEADS)
TITLES = {'lin1': 'Linear-1', 'lin4': 'Linear-4', 'mldec': 'ML-Decoder', 'glori': 'GLoRI'}
# Ô hạt của từng loại đầu trong công thức head_seed: lin4 / mldec / glori giữ đúng chỉ số của 86 (HEAD_TYPES = ('lin', 'ml', 'glori')) -> cùng hạt khởi
# tạo / dropout với `86b`; Linear-1 (mới ở 89) nhận ô 3.
SEED_SLOT = {'lin4': 0, 'mldec': 1, 'glori': 2, 'lin1': 3}
EPOCHS_LOCKED = 20                                       # E1 = E2 = 20: luật PLAN_86 §5.1 trên chạy thử 86a (10/11), áp cho MỌI encoder
LR_GRID = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1)   # learning_rates mặc định của linear_glori.py
# Tiêu chí chọn lr trên val (lần chấm cuối pha 1). 'auroc' = LUẬT (PLAN_89 §5.1, như mã CheXFound: MULTILABEL_AUROC). 'ap' = nhánh MÔ TẢ (Claude
# 15/11 sau Ver1: 8 / 12 lượt Linear chọn lr ở biên, GLoRI `86f` hoà 0,0001 AUROC val nhưng lệch 0,10 mAP21 test): lr có AP val (macro, ngữ nghĩa
# torchmetrics = `multilabel_ap` của mã họ) cao nhất; KHÁC lr của luật thì học thêm hạt + pha 2 cho lr đó (đầu thêm, RNG đầu tách -> đầu của luật
# không đổi); trùng thì không thêm gì. Không vào luật §7.4.
CRITERIA = ('auroc', 'ap')
RULE = 'auroc'


@dataclass(frozen=True)
class Protocol:
    '''Điều kiện học CHUNG cho mọi backbone (PLAN_86 §4.3–4.4; F2, F5, F6, F10).'''
    lr_grid: tuple = LR_GRID
    batch_size: int = 256          # README CheXFound: 64 ảnh / GPU x 4 GPU
    loss_divisor: int = 4          # DDP lấy TB gradient của 4 GPU -> trên 1 GPU: BCE reduction='sum' / 4
    weight_decay: float = 1e-4     # AdamW (mặc định argparser của linear_glori.py)
    n_last_blocks: int = 4         # Linear-4 / GLoRI đọc 4 tầng cuối
    glori_dim: int = 768           # decoder_dim của GLoRI
    mldecoder_dim: int = 512       # đầu ML-Decoder của 85f
    data_seed: int = 8601          # thứ tự mẫu + tăng cường
    head_seed: int = 8602          # khởi tạo + dropout đầu đọc — TÁCH khỏi data_seed


@dataclass(frozen=True)
class CurveDiagnostic:
    '''Quy tắc chẩn đoán đường học (PLAN_86 §5.2): CHỈ MÔ TẢ, áp như nhau cho mọi encoder x đầu.'''
    drop_threshold: float = 0.002           # cờ val_drop_after_peak khi AUROC đỉnh - AUROC cuối vượt ngưỡng này
    gain_window: int = 5                    # mức tăng AUROC trong bấy nhiêu epoch cuối (thấy đường còn tăng)


@dataclass(frozen=True)
class MainRun:
    '''Lượt chính (PLAN_86 §4.4): dò lr (seed 0) -> thêm seed ở lr đã chọn -> pha 2 học tiếp trên train + val -> dự đoán test MỘT lần.'''
    epochs_phase1: int = EPOCHS_LOCKED      # trên train (13.500 ảnh)
    epochs_phase2: int = EPOCHS_LOCKED      # trên train + val (15.000 ảnh)
    n_seeds: int = 5                        # seed 0 = dò lr; 1 .. n-1 thêm ở lr đã chọn
    phase2_seed_offset: int = 500_000       # dropout pha 2 theo (hạt đầu + offset); khởi tạo pha 2 = trọng số pha 1


def steps_per_epoch(n_images, batch_size):
    '''= epoch_length của linear_glori.py: ceil(N / (64 x 4)).'''
    return math.ceil(n_images / batch_size)


def head_seed(protocol, budget, kind, lr_index, seed_index=0):
    '''Hạt riêng từng đầu: head_seed + 1.000.000 x chỉ số seed + 10.000 x ngân sách + 1.000 x ô loại đầu + chỉ số lr (= công thức của 86).'''
    return protocol.head_seed + 1_000_000 * int(seed_index) + 10_000 * int(budget) + 1_000 * SEED_SLOT[kind] + int(lr_index)
