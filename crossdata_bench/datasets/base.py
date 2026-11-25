# -*- coding: utf-8 -*-
"""Lớp dữ liệu của module 93 = GIAO DIỆN VinDrData của vindr_bench (data/vindr.py @ c81af62) để pipeline của `89` chạy nguyên văn:
pipeline.prepare / run / assemble chỉ đọc keys, indices(), labels, label_names, main_classes, image_dir, store, sources (đọc mã 22/11) — thêm ở đây:
dataset (khoá bộ), groups (mã cụm cho bootstrap / hai nửa: người NIH ở RSNA; chính ảnh ở Shenzhen).

Kho của một bộ (notebook 93k, thư mục stage4_nb93/r93k_kho_<bộ>): tar phẳng <khoá>.jpg (518 x 518 'L') + bảng parquet (khoa, tach, nhom, <nhãn>…, thông
tin header ảnh gốc) + meta JSON (vân tay, chuỗi chuyển ảnh, cổng). load_store kiểm: md5 khoá = meta; vân tay = config.FINGERPRINT; đủ ảnh 518; meta ghi
đúng chuỗi "KHONG equalizeHist" (= kho 'kcb' của 85).
0.5.0: kho CHIA PHẦN (CheXpert ~191.700 ảnh): x518_<bộ>_pKK.tar (SHARD ảnh liền nhau theo thứ tự bảng), meta['phan'] = [dict(ten, n, md5, md5_khoa)]; load_store
tải song song, md5 từng phần = meta = Drive, tên mục = khoá của phần; quota tải của một phần hết -> bản sao phía máy chủ (on_quota). Nhiều nhãn: check_table
so số dương theo nhãn ở tách có ghim; fingerprint_gate so tiền tố (config.FINGERPRINT_PREFIX) khi bộ chưa có vân tay đủ."""
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
    '''Vân tay của bảng / kho so với config.FINGERPRINT[name]. -> (trùng, {tách: [có, ghim]}). Bộ chưa có vân tay đủ (0.5.0: CheXpert) -> so md5 khoá với
       TIỀN TỐ ghim ở config.FINGERPRINT_PREFIX (tách không ghim: không so). Bộ không có cả hai -> lỗi.'''
    if name in config.FINGERPRINT:
        pinned = config.FINGERPRINT[name]
        diff = {s: [[fp[s]['md5_khoa'], fp[s]['md5_nhan']], list(pinned[s])] for s in config.SPLITS
                if (fp[s]['md5_khoa'], fp[s]['md5_nhan']) != tuple(pinned[s])}
        return not diff, diff
    assert name in config.FINGERPRINT_PREFIX, f'{name}: chua ghim van tay (config.FINGERPRINT / FINGERPRINT_PREFIX) -> DUNG'
    diff = {s: [fp[s]['md5_khoa'], f'{p}…'] for s, p in config.FINGERPRINT_PREFIX[name].items() if not fp[s]['md5_khoa'].startswith(p)}
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
    assert sizes == sp.sizes, f'{name}: so anh {sizes} != {sp.sizes} -> DUNG'
    if len(sp.label_names) == 1:
        pos = tuple(fp[s]['so_duong'][0] for s in config.SPLITS)
        assert pos == sp.positives, f'{name}: so duong {pos} != {sp.positives} -> DUNG'
    else:                                                       # nhiều nhãn (0.5.0): so theo nhãn ở tách có ghim; None = chưa ghim
        for s, want in zip(config.SPLITS, sp.positives):
            if want is not None:
                assert tuple(fp[s]['so_duong']) == tuple(want), f"{name}: so duong {s} theo nhan {fp[s]['so_duong']} != {list(want)} -> DUNG"
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
        multi = len(self.label_names) > 1
        pos = {s: (self.labels[self.indices(s)].sum(0).astype(int).tolist() if multi else int(self.labels[self.indices(s)].sum())) for s in config.SPLITS}
        ng = {s: len(set(self.groups[self.indices(s)].tolist())) for s in config.SPLITS}
        return (f"{sp.title} (kho 518 dem vuong, KHONG can bang histogram): {len(self.keys):,} anh | " + ' / '.join(
            f"{s} {n[s]:,} (" + (f"duong theo nhan {pos[s]}" if multi else f"{pos[s]:,} duong") + (f", {ng[s]:,} nguoi" if sp.unit == 'nguoi' else '') + ')'
            for s in config.SPLITS)
            + f" | {len(self.label_names)} nhan" + (f" {list(self.label_names)}" if multi else '') + f" | don vi bootstrap: {'nguoi' if sp.unit == 'nguoi' else 'anh'}")


def _extract(tar_path, dest, expect=None):
    '''Giải nén tar PHẲNG <khoá>.jpg. expect (0.5.0, kho chia phần) = danh sách khoá: tên mục phải ĐÚNG [<khoá>.jpg …] theo đúng thứ tự (write_tar ghi theo
       thứ tự khoá) — lệch -> dừng trước khi giải nén.'''
    with tarfile.open(tar_path) as t:
        ms = t.getmembers()
        for m in ms:
            assert m.isfile() and '/' not in m.name and re.fullmatch(r'[A-Za-z0-9_.-]+\.jpg', m.name), f'tep la trong kho anh: {m.name[:40]} -> DUNG'
        if expect is not None:
            assert [m.name for m in ms] == [f'{k}.jpg' for k in expect], f'{os.path.basename(tar_path)}: muc trong tar KHAC khoa cua phan (meta) -> DUNG'
        try:
            t.extractall(dest, filter='data')
        except TypeError:
            t.extractall(dest)


# ---- kho CHIA PHẦN (0.5.0, CheXpert ~191.700 ảnh): x518_<bộ>_pKK.tar, mỗi phần SHARD ảnh liền nhau theo thứ tự bảng; meta['phan'] = [dict(ten, n, md5,
# md5_khoa)], meta['co_phan'] = SHARD. Notebook kho ghi từng phần ngay khi xong (luu_resume của khối chuẩn) -> runtime CPU chết chỉ làm lại phần dở.
SHARD = 6000
SHARD_FILE = 'x518_{name}_p{k:02d}.tar'


def shard_name(name, k):
    return SHARD_FILE.format(name=name, k=k)


def shard_slices(n, size=SHARD):
    '''-> [(đầu, cuối)] các phần liền nhau phủ [0, n).'''
    assert n > 0 and size > 0
    return [(a, min(a + size, n)) for a in range(0, n, size)]


def _download(f, fetch, on_quota, log):
    '''Tải một tệp (fetch: dict(id, md5, size) -> bytes, đã kiểm md5 + số byte). Lỗi quota tải CỦA TỆP (downloadQuotaExceeded) và có on_quota -> tạo bản sao
       phía máy chủ (quota mới), tải bản sao, bỏ bản sao (memory: "bản sao có quota mới").'''
    try:
        return fetch(f)
    except Exception as e:
        msg = str(e)
        if on_quota is None or not ('downloadQuotaExceeded' in msg or 'download quota' in msg.lower()):
            raise
        log(f"  ! quota tai cua mot phan kho da het -> tao ban sao phia may chu roi tai ban sao")
        g, cleanup = on_quota(f)
        try:
            return fetch(g)
        finally:
            cleanup()


def _fetch_shards(kk, meta, keys, image_dir, du, sources, fetch, threads, on_quota, log):
    '''Tải song song MỌI phần (md5 = meta = Drive), kiểm tên mục = khoá của phần, giải nén vào image_dir. -> (số phần, GB, phút).'''
    import time
    from concurrent.futures import ThreadPoolExecutor
    size, phan = int(meta['co_phan']), meta['phan']
    sl = shard_slices(len(keys), size)
    assert len(phan) == len(sl) and [p['ten'] for p in phan] == [shard_name(meta['bo'], k) for k in range(len(sl))], (
        f"meta kho: danh sach phan KHONG khop so anh / co phan ({len(phan)} phan, can {len(sl)}) -> DUNG")
    jobs = []
    for p, (a, b) in zip(phan, sl):
        assert b - a == int(p['n']) and md5_keys(keys[a:b]) == p['md5_khoa'], f"{p['ten']}: khoa cua phan KHAC meta -> DUNG"
        v = [x for x in kk.get(p['ten'], []) if x['md5'] == p['md5']]
        assert v, f"thieu {p['ten']} dung md5 cua meta trong {'/'.join(du)} -> chay lai C4-C6 cua notebook kho. DUNG"
        sources['/'.join(du) + '/' + p['ten']] = dict(id=v[0]['id'], md5=v[0]['md5'], so_ban=len(kk[p['ten']]))
        jobs.append((p['ten'], keys[a:b], dict(id=v[0]['id'], md5=v[0]['md5'], size=v[0]['size'])))
    if fetch is None:
        from vindr_bench.data import crop_store as CS
        fetch = CS.download_bytes

    def one(job):
        ten, ks, f = job
        b = _download(f, fetch, on_quota, log)
        tmp = os.path.join(image_dir, f'.{ten}.part')
        with open(tmp, 'wb') as h:
            h.write(b)
        try:
            _extract(tmp, image_dir, expect=ks)
        finally:
            os.remove(tmp)
        return len(b)
    t0, tot = time.time(), 0
    with ThreadPoolExecutor(max(1, int(threads))) as pool:
        for i, n in enumerate(pool.map(one, jobs), 1):
            tot += n
            if i % 8 == 0 or i == len(jobs):
                log(f'  kho chia phan: {i} / {len(jobs)} phan ({tot / 1e9:.2f} GB) | {(time.time() - t0) / 60:.1f} phut')
    return len(jobs), tot / 1e9, (time.time() - t0) / 60


def load_store(kiem_ke, tai, xung_dot, image_dir, name, log=print, fetch=None, threads=6, on_quota=None):
    '''Kho của bộ `name` từ Drive (stage4_nb93/r93k_kho_<bộ>, khối đọc chuẩn) -> StoreData. Cổng: md5 khoá = meta; bảng (check_table: số ảnh, số dương,
       cụm không vắt tách, vân tay = ghim); meta ghi chuỗi KHONG equalizeHist; đủ ảnh 518 x 518 sau giải nén. Kho chia phần (meta['phan'], 0.5.0): tải song
       song `threads` phần (fetch mặc định = crop_store.download_bytes, md5 + số byte), tên mục = khoá của phần; meta['csv'] (nếu có) = md5 CSV ghim.'''
    import pandas as pd
    sp = config.dataset(name)
    du = (config.FOLDER, sp.store_tag)
    fn = store_files(name)
    kk = kiem_ke(du, mau='(' + '|'.join(re.escape(v) for v in fn.values()) + '|' + re.escape(f'x518_{name}_p') + r'\d{2}\.tar)')[1]
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
    if name == 'chexpert':                                      # 0.5.0: bảng dựng từ ba CSV md5 ghim (datasets.chexpert.CSV_PINS) — meta phải ghi đúng các md5 đó
        from . import chexpert as CX
        got = {r: (v or {}).get('md5') for r, v in (meta.get('csv') or {}).items()}
        assert got == {r: v[1] for r, v in CX.CSV_PINS.items()}, f'{name}: md5 CSV trong meta kho KHAC ban ghim {got} -> DUNG'
    os.makedirs(image_dir, exist_ok=True)
    if meta.get('phan'):
        n_p, gb, minutes = _fetch_shards(kk, meta, keys, image_dir, du, sources, fetch, threads, on_quota, log)
        log(f'  kho {name}: {n_p} phan, {gb:.2f} GB tai + giai nen trong {minutes:.1f} phut (DO); md5 tung phan = meta = Drive; muc trong tar = khoa')
    else:
        _extract(get(fn['tar']), image_dir)
    from PIL import Image
    assert all(os.path.exists(os.path.join(image_dir, f'{k}.jpg')) for k in keys), f'{name}: thieu anh sau khi giai nen -> DUNG'
    assert all(Image.open(os.path.join(image_dir, f'{k}.jpg')).size == (IMAGE_SIDE, IMAGE_SIDE) for k in keys[:: max(1, len(keys) // 64)]), (
        f'{name}: anh khong phai {IMAGE_SIDE} -> DUNG')
    return StoreData(keys=keys, split=df.tach.astype(str).to_numpy().astype(object), labels=df[list(sp.label_names)].to_numpy('float32'),
                     label_names=list(sp.label_names), main_classes=list(sp.main_classes), image_dir=image_dir, store=config.STORE_KEY, dataset=name,
                     groups=df.nhom.astype(str).to_numpy(), sources=sources)
