# -*- coding: utf-8 -*-
"""Nạp VinDr-CXR từ kho ảnh 518 của module 85 qua khối đọc Drive chuẩn: ảnh + nhãn 28 lớp + tách (PLAN_89 §1, §6).

Chép từ vindr_protocol/data.py của module 86 (load_vindr + cổng; đã chạy ở 86b / 86e / 86f / 86j), bỏ phần chọn kho theo 85j: giao thức 89 gọi kho
TƯỜNG MINH ('kcb' = đệm vuông không cân bằng — Ver1; Ver2 cả hai kho). Cổng giữ nguyên: md5 khoá ảnh / bảng nhãn khớp meta 85b + 3 phần 85e, ba phần train
rời nhau, train không trùng test, meta kho ghi đúng kiểu ảnh được yêu cầu, 21 lớp số chính = nhãn có >= 10 ca dương test.

Thứ tự hàng: 15.000 ảnh train chính thức (sắp xếp) rồi 3.000 ảnh test (thứ tự 85b). Nhãn train = đa số 3 bác sĩ; test = đồng thuận. Tách val = của
85e (hạt 8510). Mã ảnh VinDr (32 hex) KHÔNG in ra (luật PhysioNet)."""
import hashlib
import json
import os
import tarfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from PIL import Image

STORES = {
    'cb': dict(test=('stage4_nb85b', 'r85b_vindr_test'), train={k: ('stage4_nb85e', f'r85e_vindr_train_p{k}') for k in (1, 2, 3)}),
    'kcb': dict(test=('stage4_nb85b', 'r85b_vindr_test_khong_cb'),
                train={k: ('stage4_nb85e', f'r85e_vindr_train_p{k}_khong_cb') for k in (1, 2, 3)}),
}
STYLE_MARKS = {'cb': 'cv2.equalizeHist', 'kcb': 'KHONG equalizeHist'}     # trường anh.cach trong meta kho 85b / 85e
STORE_LABELS = {'cb': 'dem vuong, CAN BANG histogram (nhu MIMIC-CXR-JPG)', 'kcb': 'dem vuong, KHONG can bang histogram'}
VER1_STORE = 'kcb'                                                          # PLAN_89 §1: Ver1 = kho đệm vuông không cân bằng (= 86f)
N_LABELS, N_MAIN_CLASSES, MIN_TEST_POSITIVES, IMAGE_SIDE = 28, 21, 10, 518
SPLITS = ('train', 'val', 'test')


def store_style(meta):
    '''Kiểu ảnh ghi trong meta kho (anh.cach): 'kcb' / 'cb' / None (không rõ).'''
    cach = str((meta.get('anh') or {}).get('cach', ''))
    return 'kcb' if STYLE_MARKS['kcb'] in cach else ('cb' if STYLE_MARKS['cb'] in cach else None)


def md5_keys(keys):
    '''md5 của danh sách khoá ảnh (nối bằng xuống dòng) — cùng cách băm của 85 / 86.'''
    return hashlib.md5('\n'.join(keys).encode()).hexdigest()


def md5_labels(y):
    '''md5 của ma trận nhãn (int8, C-order) — cổng "nhãn trùng giữa các lượt" (PLAN_89 §8 mục 6).'''
    return hashlib.md5(np.ascontiguousarray(np.asarray(y), dtype=np.int8).tobytes()).hexdigest()


@dataclass
class VinDrData:
    keys: list
    split: np.ndarray
    labels: np.ndarray            # [N, 28] float32
    label_names: list
    main_classes: list            # 21 lớp số chính (>= 10 ca dương test)
    image_dir: str
    store: str
    sources: dict = field(default_factory=dict)   # tệp nguồn trên Drive: md5, số bản (ID Drive giữ ở đây để ghi JSON kết quả; KHÔNG in)

    def indices(self, *splits):
        return np.flatnonzero(np.isin(self.split, splits))

    def fingerprint(self):
        '''md5 khoá + nhãn từng tách + kho -> dict (so giữa các lượt; không chứa mã ảnh).'''
        out = dict(kho=self.store, nhan=list(self.label_names), lop_chinh=list(self.main_classes))
        for s in SPLITS:
            idx = self.indices(s)
            out[s] = dict(so_anh=int(len(idx)), md5_khoa=md5_keys([self.keys[i] for i in idx]), md5_nhan=md5_labels(self.labels[idx]))
        return out

    def summary(self):
        n = {t: int((self.split == t).sum()) for t in SPLITS}
        return (f'VinDr (kho {self.store} = {STORE_LABELS[self.store]}): {len(self.keys):,} anh {IMAGE_SIDE} | train {n["train"]:,} / val {n["val"]:,} / '
                f'test {n["test"]:,} | {len(self.label_names)} nhan, {len(self.main_classes)} lop so chinh | md5 khoa / nhan khop 85b + 3 phan 85e')


def _extract(tar_path, dest):
    with tarfile.open(tar_path) as t:
        try:
            t.extractall(dest, filter='data')
        except TypeError:                                   # Python cũ: extractall chưa có filter
            for m in t.getmembers():
                assert m.isfile() and not os.path.isabs(m.name) and '..' not in m.name.split('/'), f'tep la trong kho anh: {m.name[:40]} -> DUNG'
            t.extractall(dest)


def load_vindr(kiem_ke, tai, xung_dot, image_dir, store, log=print):
    '''Nạp ảnh + nhãn VinDr QUA API của khối đọc Drive chuẩn (kiem_ke / tai / xung_dot). store: 'kcb' | 'cb' (bắt buộc ghi rõ). -> VinDrData.'''
    assert store in STORES, f'kho {store!r} khong thuoc {tuple(STORES)} -> DUNG'
    du_test, du_train = STORES[store]['test'], STORES[store]['train']
    sources = {}

    def lay(du, kk, ten):
        v = kk.get(ten)
        assert v, f"thieu {ten} trong {'/'.join(du)} -> chay notebook nguon (85b / 85e) truoc. DUNG"
        if xung_dot(v): log(f"  ! {ten}: {len(v)} ban NOI DUNG KHAC -> dung ban MOI NHAT ({v[0]['mt']})")
        sources['/'.join(du) + '/' + ten] = dict(id=v[0]['id'], md5=v[0]['md5'], so_ban=len(v))
        return tai(v[0])

    kk_te = kiem_ke(du_test, mau=r'(vd518_test\.tar|vd_test_chi_muc\.parquet|vd_test_nhan\.parquet|vd_test_meta\.json)')[1]
    cm_te = pd.read_parquet(lay(du_test, kk_te, 'vd_test_chi_muc.parquet'))
    nh_te = pd.read_parquet(lay(du_test, kk_te, 'vd_test_nhan.parquet'))
    with open(lay(du_test, kk_te, 'vd_test_meta.json'), encoding='utf-8') as f: mt_te = json.load(f)
    keys_te = cm_te.khoa.astype(str).tolist()
    names = list(mt_te['nhan'])
    assert (md5_keys(keys_te) == mt_te['md5_khoa'] and list(nh_te.khoa.astype(str)) == keys_te and cm_te.anh_ok.astype(bool).all()
            and len(names) == N_LABELS and list(nh_te.columns[1:]) == names), 'test VinDr (85b): chi muc / nhan / meta KHONG khop -> DUNG'
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
    styles = [store_style(m) for m in [mt_te] + [mt_tr[k] for k in (1, 2, 3)]]
    assert styles == [store] * 4, f"kho '{store}' ({STORE_LABELS[store]}): meta kho ghi kieu anh {styles} (test, p1, p2, p3) -> DUNG"
    keys = keys_tr + keys_te
    split = np.concatenate([nh_tr.tach.astype(str).to_numpy(), np.full(len(keys_te), 'test')]).astype(object)
    labels = np.concatenate([nh_tr[names].to_numpy('float32'), nh_te[names].to_numpy('float32')])
    main = list(mt_tr[1]['lop_chinh_p2'])
    assert main == [t for t in names if mt_te['so_duong'][t] >= MIN_TEST_POSITIVES] and len(main) == N_MAIN_CLASSES, (
        'lop so chinh (85e) KHAC nhan >= 10 ca duong test (85b) -> DUNG')
    os.makedirs(image_dir, exist_ok=True)
    for du, kk, ten in [(du_test, kk_te, 'vd518_test.tar')] + [(du_train[k], kk_tr[k], f'vd518_train_p{k}.tar') for k in (1, 2, 3)]:
        _extract(lay(du, kk, ten), image_dir)
    assert all(os.path.exists(os.path.join(image_dir, f'{k}.jpg')) for k in keys), 'thieu anh VinDr sau khi giai nen -> DUNG'
    assert all(Image.open(os.path.join(image_dir, f'{k}.jpg')).size == (IMAGE_SIDE, IMAGE_SIDE) for k in keys[:: max(1, len(keys) // 64)]), (
        f'anh khong phai {IMAGE_SIDE} -> DUNG')
    return VinDrData(keys=keys, split=split, labels=labels, label_names=names, main_classes=main, image_dir=image_dir, store=store, sources=sources)
