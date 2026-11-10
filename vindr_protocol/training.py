"""Học đầu đọc (= vòng eval_linear của linear_glori.py): lượt học của một đầu (AdamW + cosine theo iteration, không warmup), bước học, dự
đoán, chấm val. Mỗi đầu một optimizer = một param group của optimizer chung trong mã gốc (AdamW cập nhật từng tham số độc lập).

RNG (PLAN_86 §4.4): khởi tạo theo hạt của đầu; dropout theo (hạt đầu, bước) -> thêm / bớt đầu khác không đổi đầu này."""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .heads import build_head, run_head
from .metrics import ap_auroc_tm0103, auprc_trapezoid

PREDICT_SEED_OFFSET = 10 ** 7          # dropout khi chấm ở chế độ train (chỉ để đo chênh lệch với mã gốc): hạt (đầu, 10^7 + lô)


@dataclass
class HeadRun:
    '''Một đầu đọc đang học + đường val của nó. tags: nhãn tự do (vd budget=E, seed_index=k) — vào tên đầu.'''
    head_type: str
    lr: float
    seed: int
    max_steps: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: object
    tags: dict = field(default_factory=dict)
    step: int = 0
    val_auroc: list = field(default_factory=list)          # macro 28 nhãn, ngữ nghĩa torchmetrics 0.10.3
    val_ap: list = field(default_factory=list)             # AP bậc thang (= "AUPRC" của bài CheXFound)
    val_auroc_main: list = field(default_factory=list)     # TB 21 lớp số chính
    val_auprc: list = field(default_factory=list)          # AUPRC hình thang (báo kèm như 85d / 85g)
    train_loss: list = field(default_factory=list)         # loss TB từng epoch
    loss_sum: object = 0.0                                 # cộng dồn trong epoch (tensor, không đồng bộ GPU)

    @property
    def active(self):
        return self.step < self.max_steps

    @property
    def name(self):
        return '|'.join([str(v) for v in self.tags.values()] + [self.head_type, f'{self.lr:g}'])


def new_run(head_type, lr, seed, max_steps, dim, num_classes, device, protocol, **tags):
    '''Đầu mới + AdamW(lr, weight_decay) + CosineAnnealingLR(max_steps, eta_min=0) bước theo iteration.'''
    model = build_head(head_type, seed, dim, num_classes, device, protocol)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=protocol.weight_decay)
    return HeadRun(head_type, float(lr), int(seed), int(max_steps), model, opt,
                   torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(max_steps), eta_min=0), dict(tags))


def step_seed(seed, step):
    return (int(seed) * 1_000_003 + int(step)) % 2 ** 31


def zero_nan_grads(model):
    '''= vòng "zero out nan grad" của mã gốc (tham số có grad chứa NaN -> grad = 0, tham số khác giữ nguyên), MỘT lần đồng bộ GPU / đầu.'''
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if grads:
        for g, has_nan in zip(grads, torch.stack([torch.isnan(g).any() for g in grads]).tolist()):
            if has_nan: g.zero_()


def train_step(run, blocks, last_patch, labels, loss_divisor):
    '''MỘT bước: dropout theo (hạt đầu, bước); BCE reduction='sum' / loss_divisor; xoá grad NaN; KHÔNG clip; optimizer rồi scheduler.
       -> loss (tensor).'''
    run.model.train()
    torch.manual_seed(step_seed(run.seed, run.step))                   # đặt RNG CPU + mọi GPU
    out = run_head(run.head_type, run.model, blocks, last_patch)
    loss = F.binary_cross_entropy_with_logits(out.float(), labels.float(), reduction='sum') / loss_divisor
    run.optimizer.zero_grad()
    loss.backward()
    zero_nan_grads(run.model)
    run.optimizer.step()
    run.scheduler.step()
    run.step += 1
    run.loss_sum = run.loss_sum + loss.detach()
    return loss.detach()


def close_epoch(run, steps):
    '''Ghi loss TB của epoch vừa xong rồi đặt lại bộ cộng dồn.'''
    run.train_loss.append(float(run.loss_sum) / steps)
    run.loss_sum = 0.0


def max_weight_diff(sd_a, sd_b):
    assert sd_a.keys() == sd_b.keys(), 'hai state_dict khac khoa'
    return max(float((sd_a[k].float() - sd_b[k].float()).abs().max()) for k in sd_a)


@torch.no_grad()
def predict(runs, backbone, loader, mode='eval'):
    '''Xác suất (sigmoid) của MỌI đầu trên các lô của `loader` (đúng thứ tự). mode 'eval' = SỬA của 86 (encoder + đầu đọc ở eval());
       'train' = như mã gốc (đầu đọc ở train(): dropout BẬT khi chấm) — chỉ để đo chênh lệch. Xong (kể cả khi lỗi): mọi đầu về train().'''
    for run in runs:
        if mode == 'eval': run.model.eval()
        else: run.model.train()
    out = [[] for _ in runs]
    try:
        for b, (x, _) in enumerate(loader):
            blocks, last = backbone.features(x.to(backbone.device, non_blocking=True))
            for k, run in enumerate(runs):
                if mode != 'eval': torch.manual_seed(step_seed(run.seed, PREDICT_SEED_OFFSET + b))
                out[k].append(torch.sigmoid(run_head(run.head_type, run.model, blocks, last).float()).cpu())
    finally:
        for run in runs: run.model.train()
    return [torch.cat(o).numpy().astype(np.float64) for o in out]


def evaluate(runs, backbone, loader, labels, main_idx):
    '''Thêm AUROC / AP (28 nhãn, torchmetrics) + AUROC TB 21 lớp số chính + AUPRC hình thang (lớp hợp lệ) vào đường val của từng đầu; dự đoán
       không hữu hạn (phân kỳ) -> NaN.'''
    for run, probs in zip(runs, predict(runs, backbone, loader)):
        if np.isfinite(probs).all():
            _, ap, auroc_cls, auroc = ap_auroc_tm0103(labels, probs)
            run.val_auroc.append(auroc); run.val_ap.append(ap); run.val_auroc_main.append(float(auroc_cls[main_idx].mean()))
            run.val_auprc.append(auprc_trapezoid(labels, probs)[1])
        else:
            for curve in (run.val_auroc, run.val_ap, run.val_auroc_main, run.val_auprc): curve.append(float('nan'))
