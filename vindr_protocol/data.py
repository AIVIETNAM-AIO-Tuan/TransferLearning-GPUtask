"""Dữ liệu VinDr: nạp kho ảnh 518 + nhãn 28 lớp + tách từ Drive (= ô D của 85f), tập ảnh với biến đổi CheXFound, luồng mẫu vô hạn
như InfiniteSampler của mã gốc (PLAN_86 F14), bộ nạp học / đánh giá.

Không gọi Drive trực tiếp: nhận các hàm của khối đọc Drive chuẩn (kiem_ke, tai, xung_dot) làm tham số."""
import hashlib
import json
import os
import tarfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .chexfound import make_classification_eval_transform, make_classification_train_transform

# Kho ảnh của 85: 'cb' = cân bằng histogram (như MIMIC-CXR-JPG), 'kcb' = không cân bằng (phép kiểm tiền xử lý 85j). Kiểu ảnh từng hàng
# khoá theo val của 85j (PLAN_86 §4.1).
STORES = {
    'cb': dict(test=('stage4_nb85b', 'r85b_vindr_test'), train={k: ('stage4_nb85e', f'r85e_vindr_train_p{k}') for k in (1, 2, 3)}),
    'kcb': dict(test=('stage4_nb85b', 'r85b_vindr_test_khong_cb'),
                train={k: ('stage4_nb85e', f'r85e_vindr_train_p{k}_khong_cb') for k in (1, 2, 3)}),
}
N_LABELS, N_MAIN_CLASSES, MIN_TEST_POSITIVES, IMAGE_SIDE = 28, 21, 10, 518


def md5_keys(keys):
    '''md5 của danh sách khoá ảnh (= _md5k_p2 của 85: nối bằng xuống dòng).'''
    return hashlib.md5('\n'.join(keys).encode()).hexdigest()


@dataclass
class VinDrData:
    '''Thứ tự hàng: 15.000 ảnh train chính thức (sắp xếp) rồi 3.000 ảnh test. split: 'train' / 'val' (tách của 85e, hạt 8510) / 'test'.'''
    keys: list
    split: np.ndarray
    labels: np.ndarray            # [N, 28] float32 — train: đa số 3 bác sĩ; test: đồng thuận
    label_names: list
    main_classes: list            # 21 lớp số chính (>= 10 ca dương test)
    image_dir: str
    store: str
    sources: dict = field(default_factory=dict)   # tệp nguồn trên Drive: id, md5, số bản

    def indices(self, *splits):
        return np.flatnonzero(np.isin(self.split, splits))

    def summary(self):
        n = {t: int((self.split == t).sum()) for t in ('train', 'val', 'test')}
        return (f'VinDr (kho {self.store}): {len(self.keys):,} anh {IMAGE_SIDE} -> {self.image_dir} | train {n["train"]:,} / val {n["val"]:,} / '
                f'test {n["test"]:,} | {len(self.label_names)} nhan, {len(self.main_classes)} lop so chinh | md5 khoa / nhan khop 85b + 3 phan 85e')


def load_vindr(kiem_ke, tai, xung_dot, image_dir, store='cb', log=print):
    '''Nạp ảnh + nhãn VinDr QUA API (= ô D của 85f, cùng các cổng). kiem_ke / tai / xung_dot = hàm của khối đọc Drive chuẩn.'''
    du_test, du_train = STORES[store]['test'], STORES[store]['train']
    sources = {}

    def lay(du, kk, ten):
        v = kk.get(ten)
        assert v, f"thieu {ten} trong {'/'.join(du)} -> chay notebook nguon truoc. DUNG"
        if xung_dot(v): log(f"  ! {ten}: {len(v)} ban NOI DUNG KHAC -> dung ban MOI NHAT ({v[0]['mt']})")
        sources['/'.join(du) + '/' + ten] = dict(id=v[0]['id'], md5=v[0]['md5'], so_ban=len(v))
        return tai(v[0])

    # ---- test (85b): chỉ mục + nhãn đồng thuận + meta ----
    kk_te = kiem_ke(du_test, mau=r'(vd518_test\.tar|vd_test_chi_muc\.parquet|vd_test_nhan\.parquet|vd_test_meta\.json)')[1]
    cm_te = pd.read_parquet(lay(du_test, kk_te, 'vd_test_chi_muc.parquet'))
    nh_te = pd.read_parquet(lay(du_test, kk_te, 'vd_test_nhan.parquet'))
    with open(lay(du_test, kk_te, 'vd_test_meta.json'), encoding='utf-8') as f: mt_te = json.load(f)
    keys_te = cm_te.khoa.astype(str).tolist()
    names = list(mt_te['nhan'])
    assert (md5_keys(keys_te) == mt_te['md5_khoa'] and list(nh_te.khoa.astype(str)) == keys_te and cm_te.anh_ok.astype(bool).all()
            and len(names) == N_LABELS and list(nh_te.columns[1:]) == names), 'test VinDr (85b): chi muc / nhan / meta KHONG khop -> DUNG'
    # ---- train (85e p1-p3): chỉ mục từng phần + bảng nhãn (giống nhau ở 3 phần) + meta ----
    kk_tr, keys_part, mt_tr, nh_tr = {}, {}, {}, None
    for k in (1, 2, 3):
        kk_tr[k] = kiem_ke(du_train[k], mau=rf'(vd518_train_p{k}\.tar|vd_train_chi_muc_p{k}\.parquet|vd_train_nhan\.parquet|vd_train_meta_p{k}\.json)')[1]
        cm = pd.read_parquet(lay(du_train[k], kk_tr[k], f'vd_train_chi_muc_p{k}.parquet'))
        nh = pd.read_parquet(lay(du_train[k], kk_tr[k], 'vd_train_nhan.parquet'))
        with open(lay(du_train[k], kk_tr[k], f'vd_train_meta_p{k}.json'), encoding='utf-8') as f: mt_tr[k] = json.load(f)
        keys_part[k] = cm.khoa.astype(str).tolist()
        assert (md5_keys(keys_part[k]) == mt_tr[k]['md5_khoa'] and cm.anh_ok.astype(bool).all() and mt_tr[k]['phan'] == k
                and mt_tr[k]['nhan'] == names), f'85e phan {k}: chi muc / meta KHONG khop -> DUNG'
        if nh_tr is None: nh_tr = nh
        assert nh.equals(nh_tr), f'bang nhan train cua 85e phan {k} KHAC phan 1 -> DUNG'
    keys_tr = nh_tr.khoa.astype(str).tolist()
    hop = sum(keys_part.values(), [])
    assert sorted(hop) == keys_tr and len(set(hop)) == len(hop), 'anh train 3 phan KHONG roi nhau / KHONG phu bang nhan -> DUNG'
    assert (len({m['md5_nhan'] for m in mt_tr.values()}) == 1 and all(m['md5_khoa_train'] == md5_keys(keys_tr) for m in mt_tr.values())
            and list(nh_tr.columns[2:]) == names and set(nh_tr.tach) == {'train', 'val'}), 'meta / bang nhan train 85e KHONG khop -> DUNG'
    assert not set(keys_tr) & set(keys_te), 'anh train TRUNG test -> DUNG'
    # ---- thứ tự hàng, nhãn 28 lớp, tách; 21 lớp số chính = 85e, kiểm lại từ số ca dương test của 85b ----
    keys = keys_tr + keys_te
    split = np.concatenate([nh_tr.tach.astype(str).to_numpy(), np.full(len(keys_te), 'test')]).astype(object)
    labels = np.concatenate([nh_tr[names].to_numpy('float32'), nh_te[names].to_numpy('float32')])
    main = list(mt_tr[1]['lop_chinh_p2'])
    assert main == [t for t in names if mt_te['so_duong'][t] >= MIN_TEST_POSITIVES] and len(main) == N_MAIN_CLASSES, (
        'lop so chinh P2 (85e) KHAC nhan >= 10 ca duong test (85b) -> DUNG')
    # ---- giải nén ảnh 518 (test + 3 phần train) ----
    os.makedirs(image_dir, exist_ok=True)
    for du, kk, ten in [(du_test, kk_te, 'vd518_test.tar')] + [(du_train[k], kk_tr[k], f'vd518_train_p{k}.tar') for k in (1, 2, 3)]:
        with tarfile.open(lay(du, kk, ten)) as t: t.extractall(image_dir, filter='data')
    assert all(os.path.exists(os.path.join(image_dir, f'{k}.jpg')) for k in keys), 'thieu anh VinDr sau khi giai nen -> DUNG'
    assert all(Image.open(os.path.join(image_dir, f'{k}.jpg')).size == (IMAGE_SIDE, IMAGE_SIDE) for k in keys[:: max(1, len(keys) // 64)]), (
        f'anh khong phai {IMAGE_SIDE} -> DUNG')
    return VinDrData(keys=keys, split=split, labels=labels, label_names=names, main_classes=main, image_dir=image_dir, store=store, sources=sources)


# ---- biến đổi ảnh: CheXFound (học: RandomResizedCrop 0,75–1 + lật; đánh giá: Resize + CenterCrop; RescaleImage + chuẩn hoá theo mô hình) ----
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
