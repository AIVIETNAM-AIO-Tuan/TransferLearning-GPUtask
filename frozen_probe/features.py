# -*- coding: utf-8 -*-
"""Trích đặc trưng ĐÔNG CỨNG của mọi ảnh VinDr qua lớp bọc của vindr_bench (cùng đường ảnh -> đặc trưng của 89 Ver2: biến đổi đánh giá Resize +
CenterCrop của mã CheXFound trên kho 518, autocast bf16, lớp bọc tự đổi sang chuẩn hoá gốc) + tệp đặc trưng.

Mỗi ảnh hai vector (float16): mean_patch = TB token patch tầng cuối (lớp bọc trả token ĐÃ qua LayerNorm cuối -> = vector G của 78r); glob = vector toàn
ảnh của lớp bọc (CLS ở mọi hàng của phép kiểm). Thứ tự hàng = thứ tự ảnh của VinDrData (vân tay md5 khoá / nhãn ghim ở gói vindr_bench)."""
import hashlib
import io
import json
import time

import numpy as np

from . import config as C

LABEL_FILE = 'nhan_tach_vindr.npz'
MANIFEST_FILE = 'manifest_dac_trung.json'


def feature_file(row):
    return f'dac_trung_{row}.npz'


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def extract(bb, vd, batch_size=C.EXTRACT_BATCH, num_workers=0, log=print, first_batch_gate=None):
    '''bb = vindr_bench.protocols.ours.backbone.Backbone (đã nạp, eval), vd = VinDrData. -> dict(mean_patch, glob: float16 [N, D]; so_anh; phut;
       gpu_dinh_gib). first_batch_gate(peak_bytes) gọi sau lô đầu (cổng bộ nhớ) — trả False thì dừng.'''
    import torch
    from vindr_bench.data import loaders as L
    _, tf_eval = L.build_transforms(bb.spec)
    ds = L.VinDrImages(vd.keys, vd.image_dir, tf_eval)
    n, d = len(vd.keys), bb.dim
    loader = L.eval_loader(ds, range(n), batch_size, num_workers=num_workers, pin_memory=(bb.device == 'cuda'))
    mp = np.zeros((n, d), dtype=np.float16)
    gv = np.zeros((n, d), dtype=np.float16)
    seen = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    if bb.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for k, (x, j) in enumerate(loader):
        blocks, last = bb.features(x.to(bb.device, non_blocking=True))
        p = last.float().mean(1)
        v = blocks[-1][1].float()
        assert tuple(p.shape) == (len(j), d) and tuple(v.shape) == (len(j), d), f'hinh dac trung {tuple(p.shape)} / {tuple(v.shape)} SAI -> DUNG'
        assert bool(torch.isfinite(p).all()) and bool(torch.isfinite(v).all()), f'dac trung co NaN / inf (lo {k}) -> DUNG'
        jj = j.numpy()
        mp[jj] = p.cpu().numpy().astype(np.float16)
        gv[jj] = v.cpu().numpy().astype(np.float16)
        np.add.at(seen, jj, 1)
        if k == 0 and first_batch_gate is not None and bb.device == 'cuda':
            assert first_batch_gate(torch.cuda.max_memory_allocated()), 'cong bo nho GPU (lo dau) TRUOT -> DUNG'
    assert bool((seen == 1).all()), f'trich: {int((seen != 1).sum())} anh KHONG duoc ghi dung mot lan -> DUNG'
    assert np.isfinite(mp.astype(np.float32)).all() and np.isfinite(gv.astype(np.float32)).all(), 'dac trung float16 tran so -> DUNG'
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if bb.device == 'cuda' else 0.0
    return dict(mean_patch=mp, glob=gv, so_anh=n, phut=(time.time() - t0) / 60, gpu_dinh_gib=peak)


def feature_bytes(feats, row, spec_name):
    '''-> bytes của tệp .npz (KHÔNG nén; float16) cho một hàng.'''
    buf = io.BytesIO()
    np.savez(buf, mean_patch=feats['mean_patch'], glob=feats['glob'], hang=np.array(row), lop_boc=np.array(spec_name))
    return buf.getvalue()


def label_bytes(vd, fingerprint):
    '''-> bytes của tệp nhãn / tách (KHÔNG có mã ảnh): nhãn 28 lớp (int8), mã tách, tên nhãn, 21 lớp số chính, vân tay (JSON).'''
    from vindr_bench.data import vindr as V
    codes = np.array([V.SPLITS.index(s) for s in vd.split], dtype=np.int8)
    buf = io.BytesIO()
    np.savez(buf, nhan=np.asarray(vd.labels).astype(np.int8), tach=codes, ten_tach=np.array(V.SPLITS), ten_nhan=np.array(vd.label_names),
             lop_chinh=np.array(vd.main_classes), van_tay=np.array(json.dumps(fingerprint, sort_keys=True)))
    return buf.getvalue()


def load_labels(path):
    '''-> dict(nhan [N, 28] int8, tach [N] (chuỗi), ten_nhan, lop_chinh, van_tay (dict)).'''
    with np.load(path, allow_pickle=False) as z:
        names = [str(s) for s in z['ten_tach']]
        return dict(nhan=z['nhan'].astype(np.int8), tach=np.array([names[int(c)] for c in z['tach']]), ten_nhan=[str(s) for s in z['ten_nhan']],
                    lop_chinh=[str(s) for s in z['lop_chinh']], van_tay=json.loads(str(z['van_tay'])))


def load_features(path, row):
    '''-> dict(mean_patch, glob: float32 [N, D]). Hàng ghi trong tệp phải đúng `row`.'''
    with np.load(path, allow_pickle=False) as z:
        assert str(z['hang']) == row, f"tep {path}: hang {str(z['hang'])!r} != {row!r} -> DUNG"
        return {f: z[f].astype(np.float32) for f in C.FEATURES}
