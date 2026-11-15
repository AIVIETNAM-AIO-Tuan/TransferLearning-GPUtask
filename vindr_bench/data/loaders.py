# -*- coding: utf-8 -*-
"""Ảnh VinDr cho vòng học của ta (Ver2): biến đổi của mã CheXFound (học: RandomResizedCrop 0,75–1 + lật ngang; đánh giá: Resize + CenterCrop; min-max
từng ảnh; chuẩn hoá) + tập ảnh + dòng mẫu vô hạn.

CHÉP NGUYÊN VĂN từ vindr_protocol/data.py của module 86 (build_transforms, VinDrImages, InfiniteStream, _loader, train_loader, eval_loader — đã
chạy ở 86a / 86b / 87; tests/test_ver2.py so cây cú pháp). RNG dữ liệu TÁCH khỏi RNG đầu đọc: tăng cường chỉ phụ thuộc (hạt, lượt hoán vị, ảnh).
`spec` của build_transforms = ảnh vào của LỚP BỌC (protocols/ours/backbone.InputSpec: độ phân giải gốc của encoder + chuẩn hoá ImageNet như biến
đổi của mã CheXFound; lớp bọc tự đổi sang chuẩn hoá gốc)."""
import os

import numpy as np
import torch
from PIL import Image

from .transforms import make_classification_eval_transform, make_classification_train_transform


def build_transforms(spec):
    '''-> (biến đổi học, biến đổi đánh giá) với kích thước / chuẩn hoá của backbone (PLAN_86 §4.2).'''
    return (make_classification_train_transform(crop_size=spec.image_size, mean=spec.mean, std=spec.std),
            make_classification_eval_transform(resize_size=spec.image_size, crop_size=spec.image_size, mean=spec.mean, std=spec.std))


class VinDrImages(torch.utils.data.Dataset):
    '''khoá = chỉ số ảnh (biến đổi đánh giá, tất định) hoặc (lượt hoán vị p, chỉ số ảnh) (học): tăng cường CHỈ phụ thuộc (seed, p, ảnh) —
       không phụ thuộc số worker, thứ tự, số đầu đọc (RNG dữ liệu TÁCH khỏi RNG đầu đọc, PLAN_86 §4.4). Ảnh 518 xám -> RGB như mã gốc.'''
    def __init__(self, keys, image_dir, transform, seed=None):
        self.keys, self.image_dir, self.transform, self.seed = list(keys), image_dir, transform, seed

    def __getitem__(self, key):
        p, j = key if isinstance(key, tuple) else (None, int(key))
        im = Image.open(os.path.join(self.image_dir, f'{self.keys[j]}.jpg')).convert('RGB')
        if p is None: return self.transform(im), j
        assert self.seed is not None, 'tap danh gia khong nhan khoa (p, anh)'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((self.seed * 1_000_003 + p * 100_003 + j) % 2 ** 31)
            return self.transform(im), j


class InfiniteStream(torch.utils.data.Sampler):
    '''Bước [step_from, step_to) của dòng mẫu vô hạn trên `indices` (như InfiniteSampler(shuffle) của mã gốc): các hoán vị NỐI TIẾP (hoán
       vị p sinh từ (seed, p)); bước b = vị trí [b x lô, (b + 1) x lô) -> mọi lô đủ, lô vắt qua ranh giới hoán vị. Phần tử = (p, chỉ số ảnh).'''
    def __init__(self, indices, seed, step_from, step_to, batch_size):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.seed, self.step_from, self.step_to, self.batch_size = int(seed), int(step_from), int(step_to), int(batch_size)

    def __len__(self):
        return max(0, self.step_to - self.step_from) * self.batch_size

    def __iter__(self):
        n, t0, t1 = len(self.indices), self.step_from * self.batch_size, self.step_to * self.batch_size
        if t1 <= t0: return
        for p in range(t0 // n, (t1 - 1) // n + 1):
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed * 10_007 + p)).numpy()
            for i in range(max(t0 - p * n, 0), min(t1 - p * n, n)):
                yield (p, int(self.indices[perm[i]]))


def _loader(dataset, sampler, batch_size, num_workers, pin_memory):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False, num_workers=num_workers,
                                       pin_memory=pin_memory)


def train_loader(dataset, indices, seed, step_from, step_to, batch_size, num_workers=0, pin_memory=False):
    '''Bước [step_from, step_to) của dòng mẫu học trên `indices` (pha 1: train; pha 2: train + val).'''
    return _loader(dataset, InfiniteStream(indices, seed, step_from, step_to, batch_size), batch_size, num_workers, pin_memory)


def eval_loader(dataset, indices, batch_size, num_workers=0, pin_memory=False):
    '''Ảnh `indices` theo đúng thứ tự (biến đổi đánh giá).'''
    return _loader(dataset, [int(i) for i in indices], batch_size, num_workers, pin_memory)
