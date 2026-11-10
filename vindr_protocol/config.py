"""Hằng số giao thức (theo mã CheXFound @ 7e288d0 — PLAN_86 §3, §4) và cấu hình theo backbone / lượt chạy."""
import math
from dataclasses import dataclass

CHEXFOUND_COMMIT = '7e288d06532f24abedb7d00781185c36727d9d69'
HEAD_TYPES = ('lin', 'ml', 'glori')                     # Linear (CheXFound) | ML-Decoder (của 85f) | GLoRI (CheXFound)
LR_GRID = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1)   # learning_rates mặc định của linear_glori.py


@dataclass(frozen=True)
class Protocol:
    '''Điều kiện học CHUNG cho mọi backbone (PLAN_86 §4.3–4.4; F2, F5, F6, F10).'''
    lr_grid: tuple = LR_GRID
    batch_size: int = 256          # README: 64 ảnh / GPU x 4 GPU
    loss_divisor: int = 4          # DDP lấy TB gradient của 4 GPU -> trên 1 GPU: BCE reduction='sum' / 4
    weight_decay: float = 1e-4     # AdamW (mặc định argparser của linear_glori.py)
    n_last_blocks: int = 4         # Linear / GLoRI đọc 4 tầng cuối
    glori_dim: int = 768           # decoder_dim của GLoRI
    mldecoder_dim: int = 512       # đầu ML-Decoder của 85f
    data_seed: int = 8601          # thứ tự mẫu + tăng cường
    head_seed: int = 8602          # khởi tạo + dropout đầu đọc — TÁCH khỏi data_seed


@dataclass(frozen=True)
class BackboneSpec:
    '''Ảnh vào + chuẩn hoá theo MÔ HÌNH (PLAN_86 §4.2).'''
    name: str
    image_size: int
    mean: tuple
    std: tuple


@dataclass(frozen=True)
class Pilot:
    '''Chạy thử pha 1 (86a; PLAN_86 §5.1).'''
    budgets: tuple = (10, 20)               # hai ngân sách epoch học song song
    epoch_rule_threshold: float = 0.002     # chênh AUROC val tối đa để chọn ngân sách nhỏ
    convergence_threshold: float = 0.002    # PLAN_86 §5.2
    solo_check_steps: int = 10              # số bước của phép kiểm "một đầu riêng = trong nhóm"
    solo_check_logit_tol: float = 1e-3      # ngưỡng lệch logit (GPU không tất định từng bit)
    check_lr: float = 1e-3                  # lr của các đầu kiểm


def steps_per_epoch(n_images, batch_size):
    '''= epoch_length của linear_glori.py: ceil(N / (64 x 4)).'''
    return math.ceil(n_images / batch_size)


def head_seed(protocol, budget, head_type, lr_index):
    '''Hạt riêng từng đầu (như 86a): head_seed + 10.000 x ngân sách + 1.000 x loại đầu + chỉ số lr.'''
    return protocol.head_seed + 10_000 * int(budget) + 1_000 * HEAD_TYPES.index(head_type) + int(lr_index)
