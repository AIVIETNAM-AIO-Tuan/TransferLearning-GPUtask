"""Dữ liệu VinDr của 85 (chỉ đọc): nhãn + meta test (85b) và kho ảnh 518 + nhãn train / test cho P2 (85e p1–p3 + 85b).

load_test_labels = phần nạp nhãn trong ô C1 của 85g; load_p2_data = ô D của 85f (cùng cổng, cùng thứ tự hàng). Chỉ in số đếm, không in id ảnh."""
import hashlib
import os
import tarfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from PIL import Image

from . import config
from .metrics import halves


def md5_keys(keys):
    '''md5 của danh sách khoá ảnh nối bằng xuống dòng (= _md5k_g / _md5k_p2 của 85).'''
    return hashlib.md5('\n'.join(keys).encode()).hexdigest()


@dataclass
class VinDrTest:
    '''3.000 ảnh test (85b): thứ tự khoá của chỉ mục, nhãn đồng thuận 28 lớp (float64: ap_cols khớp sklearn từng bit), meta, hai nửa.'''
    keys: list
    label_names: list
    Y: np.ndarray
    meta: dict
    main_classes: list            # >= 10 ca dương (21 lớp, luật P2)
    any_classes: list             # >= 1 ca dương
    half: np.ndarray              # 0 / 1 theo ảnh, hạt config.HALF_SEED (= 85g / 85d)

    def columns(self, classes):
        return [self.label_names.index(c) for c in classes]


def load_test_labels(reader):
    '''Nạp chỉ mục + nhãn + meta VinDr test của 85b QUA API (= ô C1 của 85g, phần nhãn) + chia hai nửa.'''
    du = config.VINDR_TEST
    kb = reader.inventory(du, r'(vd_test_chi_muc\.parquet|vd_test_nhan\.parquet|vd_test_meta\.json)')
    cm = pd.read_parquet(reader.fetch(du, kb, 'vd_test_chi_muc.parquet'))
    nh = pd.read_parquet(reader.fetch(du, kb, 'vd_test_nhan.parquet'))
    meta = reader.load_json(du, kb, 'vd_test_meta.json')
    keys, names = cm.khoa.astype(str).tolist(), list(meta['nhan'])
    assert list(nh.khoa.astype(str)) == keys and md5_keys(keys) == meta['md5_khoa'] and list(nh.columns[1:]) == names, (
        'nhan / chi muc VinDr test lech meta 85b -> DUNG')
    Y = nh[names].to_numpy('float64')
    main = [t for t in names if meta['so_duong'][t] >= config.MIN_TEST_POSITIVES]
    anyc = [t for t in names if meta['so_duong'][t] >= 1]
    assert len(main) == config.N_MAIN and set(config.RADDINO7) <= set(main), 'lop so chinh P2 KHAC 21 nhan >= 10 ca duong -> DUNG'
    return VinDrTest(keys=keys, label_names=names, Y=Y, meta=meta, main_classes=main, any_classes=anyc, half=halves(len(keys), config.HALF_SEED))


@dataclass
class P2Data:
    '''Thứ tự hàng P2 = 15.000 ảnh train chính thức (sắp xếp) rồi 3.000 ảnh test; split: train / val (tách của 85e) / test.'''
    keys: list
    keys_train: list
    keys_test: list
    split: np.ndarray
    labels: np.ndarray            # [N, 28] float32
    label_names: list
    main_classes: list            # 21 lớp số chính (85e, kiểm lại từ số ca dương test của 85b)
    image_dir: str
    sources: dict = field(default_factory=dict)

    def counts(self):
        return {t: int((self.split == t).sum()) for t in ('train', 'val', 'test')}


def load_p2_data(reader, image_dir, log=print):
    '''= ô D của 85f: nạp ảnh 518 + nhãn VinDr train (85e p1–p3) + test (85b) QUA API, giải nén vào `image_dir`, cùng mọi cổng.'''
    du_te, du_tr = config.VINDR_TEST, config.VINDR_TRAIN
    kb = reader.inventory(du_te, r'(vd518_test\.tar|vd_test_chi_muc\.parquet|vd_test_nhan\.parquet|vd_test_meta\.json)')
    cmt = pd.read_parquet(reader.fetch(du_te, kb, 'vd_test_chi_muc.parquet'))
    nht = pd.read_parquet(reader.fetch(du_te, kb, 'vd_test_nhan.parquet'))
    mtt = reader.load_json(du_te, kb, 'vd_test_meta.json')
    keys_te = cmt.khoa.astype(str).tolist()
    names = list(mtt['nhan'])
    assert (md5_keys(keys_te) == mtt['md5_khoa'] and list(nht.khoa.astype(str)) == keys_te and cmt.anh_ok.astype(bool).all()
            and len(names) == config.N_LABELS and list(nht.columns[1:]) == names), 'test VinDr (85b): chi muc / nhan / meta KHONG khop -> DUNG'
    ke, ks, mte, nhtr = {}, {}, {}, None
    for k in (1, 2, 3):
        ke[k] = reader.inventory(du_tr[k], rf'(vd518_train_p{k}\.tar|vd_train_chi_muc_p{k}\.parquet|vd_train_nhan\.parquet|vd_train_meta_p{k}\.json)')
        cm = pd.read_parquet(reader.fetch(du_tr[k], ke[k], f'vd_train_chi_muc_p{k}.parquet'))
        nh = pd.read_parquet(reader.fetch(du_tr[k], ke[k], 'vd_train_nhan.parquet'))
        mte[k] = reader.load_json(du_tr[k], ke[k], f'vd_train_meta_p{k}.json')
        ks[k] = cm.khoa.astype(str).tolist()
        assert (md5_keys(ks[k]) == mte[k]['md5_khoa'] and cm.anh_ok.astype(bool).all() and mte[k]['phan'] == k
                and mte[k]['nhan'] == names), f'85e phan {k}: chi muc / meta KHONG khop -> DUNG'
        if nhtr is None: nhtr = nh
        assert nh.equals(nhtr), f'bang nhan train cua 85e phan {k} KHAC phan 1 -> DUNG'
    keys_tr = nhtr.khoa.astype(str).tolist()
    hop = sum(ks.values(), [])
    assert sorted(hop) == keys_tr and len(set(hop)) == len(hop), 'anh train 3 phan KHONG roi nhau / KHONG phu bang nhan -> DUNG'
    assert (len({m['md5_nhan'] for m in mte.values()}) == 1 and all(m['md5_khoa_train'] == md5_keys(keys_tr) for m in mte.values())
            and list(nhtr.columns[2:]) == names and set(nhtr.tach) == {'train', 'val'}), 'meta / bang nhan train 85e KHONG khop -> DUNG'
    assert not set(keys_tr) & set(keys_te), 'anh train TRUNG test -> DUNG'
    keys = keys_tr + keys_te
    split = np.concatenate([nhtr.tach.astype(str).to_numpy(), np.full(len(keys_te), 'test')]).astype(object)
    Y = np.concatenate([nhtr[names].to_numpy('float32'), nht[names].to_numpy('float32')])
    main = list(mte[1]['lop_chinh_p2'])
    assert main == [t for t in names if mtt['so_duong'][t] >= config.MIN_TEST_POSITIVES] and len(main) == config.N_MAIN, (
        'lop so chinh P2 (85e) KHAC nhan >= 10 ca duong test (85b) -> DUNG')
    os.makedirs(image_dir, exist_ok=True)
    for du, kk, t in [(du_te, kb, 'vd518_test.tar')] + [(du_tr[k], ke[k], f'vd518_train_p{k}.tar') for k in (1, 2, 3)]:
        with tarfile.open(reader.fetch(du, kk, t)) as tt: tt.extractall(image_dir, filter='data')
    assert all(os.path.exists(os.path.join(image_dir, f'{k}.jpg')) for k in keys), 'thieu anh VinDr sau khi giai nen -> DUNG'
    assert all(Image.open(os.path.join(image_dir, f'{k}.jpg')).size == (518, 518) for k in keys[:: max(1, len(keys) // 64)]), 'anh khong phai 518 -> DUNG'
    data = P2Data(keys=keys, keys_train=keys_tr, keys_test=keys_te, split=split, labels=Y, label_names=names, main_classes=main,
                  image_dir=image_dir, sources=dict(reader.sources))
    n = data.counts()
    log(f'P2 VinDr: {len(keys):,} anh 518 -> {image_dir} | train {n["train"]:,} / val {n["val"]:,} / test {n["test"]:,} | '
        f'{len(names)} nhan, {len(main)} lop so chinh | md5 khoa / nhan khop 85b + 3 phan 85e')
    return data
