# -*- coding: utf-8 -*-
"""Ảnh của phép thử: lấy ĐÚNG các ảnh cần từ kho 518 trên Drive (cxrlt_2024/cxrlt_518_ready/*.tar — kho mà mọi notebook 78 dùng) và đọc ảnh y như đường
chấm của 78 (`Image.open(...).convert('L')` -> ba kênh giống nhau -> ToTensor -> chuẩn hoá gốc của RAD-DINO).

Tệp .tar tải qua khối Drive chuẩn (API, kiểm md5), không mở trên mount; mỗi tệp tải xong chỉ rút các ảnh cần rồi xoá bản tải."""
import os
import tarfile

import numpy as np

from . import config as C


def fetch_images(drive, ids, out_dir, log=print):
    '''-> dict(n_can, n_co_san, n_lay, n_tar_doc). Thiếu ảnh sau khi đọc hết kho -> dừng.'''
    os.makedirs(out_dir, exist_ok=True)
    need = {d for d in ids if not os.path.exists(os.path.join(out_dir, d + '.jpg'))}
    n_can, n_san = len(set(ids)), len(set(ids)) - len(need)
    vs = drive['kiem_ke'](C.STORE_DIR)[1]
    tars = sorted(n for n in vs if n.endswith('.tar'))
    assert tars, f"khong thay tep .tar nao trong cxrlt_2024/{'/'.join(C.STORE_DIR)} -> DUNG"
    n_tar = 0
    for name in tars:
        if not need:
            break
        p = drive['tai'](vs[name][0])
        n_tar += 1
        with tarfile.open(p) as tar:
            for m in tar:
                b = os.path.basename(m.name)
                if m.isfile() and b.endswith('.jpg') and b[:-4] in need:
                    dst = os.path.join(out_dir, b)
                    with open(dst + '.tmp', 'wb') as f:
                        f.write(tar.extractfile(m).read())
                    os.replace(dst + '.tmp', dst)
                    need.discard(b[:-4])
        os.remove(p)
        log(f'  kho 518: {name} xong | con thieu {len(need)} / {n_can}')
    assert not need, f'kho 518 thieu {len(need)} anh can cho phep thu -> DUNG'
    return dict(n_can=n_can, n_co_san=n_san, n_lay=n_can - n_san, n_tar_doc=n_tar, n_tar=len(tars))


def load_batch(out_dir, ids, mean, std):
    '''-> tensor [n, 3, 518, 518] float32, chuẩn hoá (x / 255 - mean) / std trên kênh xám, ba kênh giống nhau (= eval_tf của 78 trên ảnh 'L').'''
    import torch
    from PIL import Image
    xs = []
    for d in ids:
        a = np.asarray(Image.open(os.path.join(out_dir, d + '.jpg')).convert('L'), 'float32')
        assert a.shape == (C.SIZE, C.SIZE), f'{d}: anh {a.shape} khac {C.SIZE} x {C.SIZE} -> DUNG'
        xs.append(a)
    x = (np.stack(xs) / 255.0 - float(mean)) / float(std)
    return torch.from_numpy(x.astype('float32'))[:, None].expand(-1, 3, -1, -1).contiguous()
