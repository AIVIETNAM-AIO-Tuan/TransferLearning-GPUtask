"""Hằng số giao thức (theo mã CheXFound @ 7e288d0 — PLAN_86 §3, §4) và cấu hình theo backbone / lượt chạy."""
import math
from dataclasses import dataclass

CHEXFOUND_COMMIT = '7e288d06532f24abedb7d00781185c36727d9d69'
HEAD_TYPES = ('lin', 'ml', 'glori')                     # Linear (CheXFound) | ML-Decoder (của 85f) | GLoRI (CheXFound)
EPOCHS_LOCKED = 20                                      # E1 = E2 = 20: luật §5.1 trên chạy thử 86a (10/11), áp cho MỌI encoder
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
class CurveDiagnostic:
    '''Quy tắc chẩn đoán đường học (PLAN_86 §5.2, Tuấn duyệt 10/11): CHỈ MÔ TẢ, áp như nhau cho mọi encoder x đầu.'''
    drop_threshold: float = 0.002           # cờ val_drop_after_peak khi AUROC đỉnh - AUROC cuối vượt ngưỡng này
    gain_window: int = 5                    # mức tăng AUROC trong bấy nhiêu epoch cuối (thấy đường còn tăng)


@dataclass(frozen=True)
class Pilot:
    '''Chạy thử pha 1 (86a; PLAN_86 §5.1).'''
    budgets: tuple = (10, 20)               # hai ngân sách epoch học song song
    epoch_rule_threshold: float = 0.002     # chênh AUROC val tối đa để chọn ngân sách nhỏ
    solo_check_steps: int = 10              # số bước của phép kiểm "một đầu riêng = trong nhóm"
    solo_check_logit_tol: float = 1e-3      # ngưỡng lệch logit (GPU không tất định từng bit)
    check_lr: float = 1e-3                  # lr của các đầu kiểm


@dataclass(frozen=True)
class MainRun:
    '''Lượt chính (86b; PLAN_86 §4.4): dò lr (seed 0) -> thêm seed ở lr đã chọn -> pha 2 học tiếp trên train + val -> dự đoán test MỘT lần.'''
    epochs_phase1: int = EPOCHS_LOCKED      # trên train (13.500 ảnh)
    epochs_phase2: int = EPOCHS_LOCKED      # trên train + val (15.000 ảnh)
    n_seeds: int = 5                        # seed 0 = dò lr; 1 .. n-1 thêm ở lr đã chọn
    phase2_seed_offset: int = 500_000       # dropout pha 2 theo (hạt đầu + offset); khởi tạo pha 2 = trọng số pha 1


PAPER_LR_GRID = LR_GRID[:9]                             # bài CheXFound §IV-A2: {1e-5 … 5e-3} = 9 giá trị đầu của lưới mã


@dataclass(frozen=True)
class PaperArms:
    '''Nhánh SÁT BÀI CheXFound (PLAN_86 §11.3) — học song song trong lượt chính (thêm đầu; RNG đầu tách -> đầu của lượt chính không đổi), CHỈ
       mô tả, không vào luật §4.5. Mỗi ngân sách E: E1 = E2 = E; CHỈ SỐ seed 0 — hạt thật = head_seed(E, loại, lr), đổi theo E, KHÔNG phải hạt 0
       của mã gốc (đính chính 12/11; mã gốc: torch.manual_seed(0) rồi khởi tạo tuần tự); chọn lr theo AUROC val như mã (lưới 13) và
       báo thêm lựa chọn trong lưới 9 của bài (khác nhau -> học pha 2 cả hai); dự đoán test ở mọi chế độ predict_modes:
       'eval' = đầu đọc eval() + autocast của lượt (sửa của 86) | 'train' = dropout BẬT như evaluate() của mã gốc (F7) |
       'train_fp16' = dropout bật + encoder autocast fp16 như mã gốc (F13) — gần cách chấm của họ nhất (đầu vẫn học trên đặc trưng của lượt).'''
    budgets: tuple = (10, 5)                # 10 = sát bài (mặc định mã; bài: 10 epoch cho bộ lớn) | 5 = độ nhạy ngân sách (≈ số bước khi lô 512)
    head_types: tuple = ('lin', 'glori')    # hai đầu của Bảng IV
    paper_grid: tuple = PAPER_LR_GRID
    predict_modes: tuple = ('eval', 'train', 'train_fp16')


def steps_per_epoch(n_images, batch_size):
    '''= epoch_length của linear_glori.py: ceil(N / (64 x 4)).'''
    return math.ceil(n_images / batch_size)


def head_seed(protocol, budget, head_type, lr_index, seed_index=0):
    '''Hạt riêng từng đầu: head_seed + 1.000.000 x chỉ số seed + 10.000 x ngân sách + 1.000 x loại đầu + chỉ số lr (seed 0 = như 86a).'''
    return (protocol.head_seed + 1_000_000 * int(seed_index) + 10_000 * int(budget) + 1_000 * HEAD_TYPES.index(head_type) + int(lr_index))
