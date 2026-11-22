# -*- coding: utf-8 -*-
"""Lớp dữ liệu của module 93 = GIAO DIỆN VinDrData của vindr_bench (data/vindr.py @ c81af62) để pipeline của `89` chạy nguyên văn:
pipeline.prepare / run / assemble chỉ đọc keys, indices(), labels, label_names, main_classes, image_dir, store, sources (đọc mã 22/11) — thêm ở đây:
dataset (khoá bộ), groups (mã cụm cho bootstrap / hai nửa: người NIH ở RSNA; chính ảnh ở Shenzhen).

Kho của một bộ (notebook 93k, thư mục stage4_nb93/r93k_kho_<bộ>): tar phẳng <khoá>.jpg (518 x 518 'L') + bảng parquet (khoa, tach, nhom, <nhãn>…, thông
tin header ảnh gốc) + meta JSON (vân tay, chuỗi chuyển ảnh, cổng). load_store kiểm: md5 khoá = meta; vân tay = config.FINGERPRINT; đủ ảnh 518; meta ghi
đúng chuỗi "KHONG equalizeHist" (= kho 'kcb' của 85)."""
import json
import os
import re
import tarfile
from dataclasses import dataclass, field

import numpy as np

from .. import config

STORE_FILES = dict(tar='x518_{name}.tar', bang='x518_{name}_bang.parquet', meta='x518_{name}_meta.json')
NO_EQ_MARK = 'KHONG equalizeHist'          # = vindr_bench.data.vindr.STYLE_MARKS['kcb'] — chữ trong meta 'cach'
IMAGE_SIDE = 518


def store_files(name):
    return {k: v.format(name=name) for k, v in STORE_FILES.items()}


def md5_keys(keys):
    '''= vindr_bench.data.vindr.md5_keys (khoá nối bằng xuống dòng).'''
    import hashlib
    return hashlib.md5('\n'.join(keys).encode()).hexdigest()


def md5_labels(y):
    '''= vindr_bench.data.vindr.md5_labels (int8, C-order).'''
    import hashlib
    return hashlib.md5(np.ascontiguousarray(np.asarray(y), dtype=np.int8).tobytes()).hexdigest()


def fingerprint_of(keys, split, labels):
    '''-> {tách: dict(so_anh, so_duong [theo cột], md5_khoa, md5_nhan)} (không chứa khoá ảnh).'''
    out = {}
    split, labels = np.asarray(split).astype(str), np.asarray(labels)
    for s in config.SPLITS:
        idx = np.flatnonzero(split == s)
        out[s] = dict(so_anh=int(len(idx)), so_duong=[int(v) for v in labels[idx].sum(0)], md5_khoa=md5_keys([keys[i] for i in idx]),
                      md5_nhan=md5_labels(labels[idx]))
    return out


def fingerprint_gate(name, fp):
    '''Vân tay của bảng / kho so với config.FINGERPRINT[name]. -> (trùng, {tách: [có, ghim]}).'''
    pinned = config.FINGERPRINT[name]
    diff = {s: [[fp[s]['md5_khoa'], fp[s]['md5_nhan']], list(pinned[s])] for s in config.SPLITS
            if (fp[s]['md5_khoa'], fp[s]['md5_nhan']) != tuple(pinned[s])}
    return not diff, diff


def check_table(name, df):
    '''Cổng bảng của một bộ (dùng ở 93k khi dựng và 93v khi nạp): cột, tách, nhãn nhị phân, số ảnh / số dương từng tách = DatasetSpec, khoá duy nhất, cụm
       (nhom) không vắt qua hai tách, vân tay = ghim. -> vân tay.'''
    sp = config.dataset(name)
    cols = ['khoa', 'tach', 'nhom', *sp.label_names]
    assert all(c in df.columns for c in cols), f'{name}: bang thieu cot {[c for c in cols if c not in df.columns]} -> DUNG'
    keys = df.khoa.astype(str).tolist()
    assert len(set(keys)) == len(keys), f'{name}: khoa anh trung -> DUNG'
    assert set(df.tach) <= set(config.SPLITS), f'{name}: tach la {sorted(set(df.tach) - set(config.SPLITS))} -> DUNG'
    y = df[list(sp.label_names)].to_numpy()
    assert set(np.unique(y).tolist()) <= {0, 1}, f'{name}: nhan khong nhi phan -> DUNG'
    per = df.groupby('nhom').tach.nunique()
    assert int(per.max()) == 1, f'{name}: {int((per > 1).sum())} cum (nhom) vat qua hai tach -> DUNG'
    fp = fingerprint_of(keys, df.tach.to_numpy(), y)
    sizes = tuple(fp[s]['so_anh'] for s in config.SPLITS)
    pos = tuple(fp[s]['so_duong'][0] for s in config.SPLITS)
    assert sizes == sp.sizes, f'{name}: so anh {sizes} != {sp.sizes} -> DUNG'
    assert len(sp.label_names) != 1 or pos == sp.positives, f'{name}: so duong {pos} != {sp.positives} -> DUNG'
    return fp


@dataclass
class StoreData:
    '''= giao diện VinDrData (vindr_bench.data.vindr) + dataset, groups.'''
    keys: list
    split: np.ndarray
    labels: np.ndarray            # [N, C] float32
    label_names: list
    main_classes: list
    image_dir: str
    store: str                    # = config.STORE_KEY ('kcb') — pipeline.assemble tra vindr.STORE_LABELS[store]
    dataset: str
    groups: np.ndarray            # mã cụm từng ảnh (chuỗi)
    sources: dict = field(default_factory=dict)

    def indices(self, *splits):
        return np.flatnonzero(np.isin(self.split, splits))

    def fingerprint(self):
        fp = fingerprint_of(self.keys, self.split, self.labels)
        fp.update(bo=self.dataset, kho=self.store, nhan=list(self.label_names))
        return fp

    def summary(self):
        sp = config.dataset(self.dataset)
        n = {s: len(self.indices(s)) for s in config.SPLITS}
        pos = {s: int(self.labels[self.indices(s)].sum()) for s in config.SPLITS}
        ng = {s: len(set(self.groups[self.indices(s)].tolist())) for s in config.SPLITS}
        return (f"{sp.title} (kho 518 dem vuong, KHONG can bang histogram): {len(self.keys):,} anh | " + ' / '.join(
            f"{s} {n[s]:,} ({pos[s]:,} duong" + (f", {ng[s]:,} nguoi" if sp.unit == 'nguoi' else '') + ')' for s in config.SPLITS)
            + f" | {len(self.label_names)} nhan | don vi bootstrap: {'nguoi' if sp.unit == 'nguoi' else 'anh'}")


def _extract(tar_path, dest):
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            assert m.isfile() and '/' not in m.name and re.fullmatch(r'[A-Za-z0-9_.-]+\.jpg', m.name), f'tep la trong kho anh: {m.name[:40]} -> DUNG'
        try:
            t.extractall(dest, filter='data')
        except TypeError:
            t.extractall(dest)


def load_store(kiem_ke, tai, xung_dot, image_dir, name, log=print):
    '''Kho của bộ `name` từ Drive (stage4_nb93/r93k_kho_<bộ>, khối đọc chuẩn) -> StoreData. Cổng: md5 khoá = meta; bảng (check_table: số ảnh, số dương,
       cụm không vắt tách, vân tay = ghim); meta ghi chuỗi KHONG equalizeHist; đủ ảnh 518 x 518 sau giải nén.'''
    import pandas as pd
    sp = config.dataset(name)
    du = (config.FOLDER, sp.store_tag)
    fn = store_files(name)
    kk = kiem_ke(du, mau='(' + '|'.join(re.escape(v) for v in fn.values()) + ')')[1]
    sources = {}

    def get(ten):
        v = kk.get(ten)
        assert v, f"thieu {ten} trong {'/'.join(du)} -> chay notebook kho 93k_kho_{name} truoc. DUNG"
        if xung_dot(v): log(f"  ! {ten}: {len(v)} ban NOI DUNG KHAC -> dung ban MOI NHAT ({v[0]['mt']})")
        sources['/'.join(du) + '/' + ten] = dict(id=v[0]['id'], md5=v[0]['md5'], so_ban=len(v))
        return tai(v[0])
    df = pd.read_parquet(get(fn['bang']))
    with open(get(fn['meta']), encoding='utf-8') as f:
        meta = json.load(f)
    keys = df.khoa.astype(str).tolist()
    assert md5_keys(keys) == meta['md5_khoa'] and meta['bo'] == name, f'{name}: bang / meta kho KHONG khop -> DUNG'
    assert NO_EQ_MARK in str((meta.get('anh') or {}).get('cach', '')), f'{name}: meta kho khong ghi chuoi KHONG can bang histogram -> DUNG'
    fp = check_table(name, df)
    same, diff = fingerprint_gate(name, fp)
    assert same, f'{name}: van tay kho KHAC ghim {diff} -> DUNG, gui Claude dong nay'
    os.makedirs(image_dir, exist_ok=True)
    _extract(get(fn['tar']), image_dir)
    from PIL import Image
    assert all(os.path.exists(os.path.join(image_dir, f'{k}.jpg')) for k in keys), f'{name}: thieu anh sau khi giai nen -> DUNG'
    assert all(Image.open(os.path.join(image_dir, f'{k}.jpg')).size == (IMAGE_SIDE, IMAGE_SIDE) for k in keys[:: max(1, len(keys) // 64)]), (
        f'{name}: anh khong phai {IMAGE_SIDE} -> DUNG')
    return StoreData(keys=keys, split=df.tach.astype(str).to_numpy().astype(object), labels=df[list(sp.label_names)].to_numpy('float32'),
                     label_names=list(sp.label_names), main_classes=list(sp.main_classes), image_dir=image_dir, store=config.STORE_KEY, dataset=name,
                     groups=df.nhom.astype(str).to_numpy(), sources=sources)
