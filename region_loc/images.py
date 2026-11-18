# -*- coding: utf-8 -*-
"""Ảnh của phép thử: lấy ĐÚNG các ảnh cần từ kho 518 trên Drive (cxrlt_2024/cxrlt_518_ready/*.tar — kho mà mọi notebook 78 dùng) và đọc ảnh y như đường
chấm của 78 (`Image.open(...).convert('L')` -> ba kênh giống nhau -> ToTensor -> chuẩn hoá gốc của RAD-DINO).

Tải (0.2.0): nhiều luồng qua Drive API, mỗi luồng MỘT dịch vụ API riêng (httplib2 không an toàn đa luồng), tải nguyên tệp .tar vào RAM, kiểm md5 máy
chủ, thử lại 4 lần; cửa sổ trượt <= 2n tệp trong RAM — cách làm của ô C1K ở experiments/78_chi_anh/dung_78r_cat_vung.py (`_tai_tar_k`, `_cua_so_k`; đo
26/10: 6 luồng ~99 MB/s). Lượt 0.1.x tải tuần tự bằng `tai` của khối chuẩn: hơn 100 tệp mất ~45 phút (đo 18/11). Đường song song hỏng (vd không dựng
được dịch vụ API) -> ghi rõ rồi tải tuần tự phần còn lại bằng `tai` — đường đã chạy thật."""
import collections
import hashlib
import io
import itertools
import os
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config as C


def api_tar_bytes(ban, tl):
    '''MỘT tệp .tar qua Drive API -> bytes (md5 = máy chủ). tl = threading.local của luồng: mỗi luồng một dịch vụ; lỗi -> dịch vụ mới, chờ, thử lại (4 lần).'''
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    loi = None
    for lan in range(4):
        try:
            if getattr(tl, 'api', None) is None:
                tl.api = build('drive', 'v3', cache_discovery=False)
            bio = io.BytesIO()
            dl, xong = MediaIoBaseDownload(bio, tl.api.files().get_media(fileId=ban['id'], supportsAllDrives=True), chunksize=32 << 20), False
            while not xong:
                _, xong = dl.next_chunk()
            b = bio.getvalue()
            if not ban.get('md5') or hashlib.md5(b).hexdigest() == ban['md5']:
                return b
            loi = f'md5 ban tai ve lech Drive ({len(b)} byte)'
        except Exception as e:                                       # noqa: BLE001
            loi, tl.api = f'{type(e).__name__}: {str(e)[:200]}', None
        time.sleep(5 * (lan + 1))
    raise OSError(f"{ban.get('ten')}: tai QUA API hong 4 lan ({loi})")


def _window(items, fn, n):
    '''(mục, fn(mục)) ĐÚNG thứ tự items, n luồng; tối đa 2n việc đang chạy / chờ. Dừng sớm / lỗi -> huỷ việc chưa chạy, đợi việc đang chạy.'''
    pool, it, q = ThreadPoolExecutor(n), iter(items), collections.deque()
    try:
        for t in itertools.islice(it, 2 * n):
            q.append((t, pool.submit(fn, t)))
        while q:
            t, fu = q.popleft()
            for t2 in itertools.islice(it, 1):
                q.append((t2, pool.submit(fn, t2)))
            yield t, fu.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _extract(fileobj_or_path, need, out_dir):
    '''Rút các ảnh thuộc `need` (tập dicom_id, bị bớt dần) từ một tệp .tar -> số ảnh rút được.'''
    n = 0
    kw = dict(name=fileobj_or_path) if isinstance(fileobj_or_path, str) else dict(fileobj=fileobj_or_path)
    with tarfile.open(**kw) as tar:
        for m in tar:
            b = os.path.basename(m.name)
            if m.isfile() and b.endswith('.jpg') and b[:-4] in need:
                dst = os.path.join(out_dir, b)
                with open(dst + '.tmp', 'wb') as f:
                    f.write(tar.extractfile(m).read())
                os.replace(dst + '.tmp', dst)
                need.discard(b[:-4])
                n += 1
    return n


def fetch_images(drive, ids, out_dir, log=print, n_threads=C.N_THREADS, tar_bytes=api_tar_bytes):
    '''-> dict(n_can, n_co_san, n_lay, n_tar_doc, n_tar, mb, giay, cach). Thiếu ảnh sau khi đọc hết kho -> dừng. tar_bytes: hàm tải (bộ kiểm truyền hàm giả).'''
    os.makedirs(out_dir, exist_ok=True)
    need = {d for d in ids if not os.path.exists(os.path.join(out_dir, d + '.jpg'))}
    n_can, n_san = len(set(ids)), len(set(ids)) - len(need)
    vs = drive['kiem_ke'](C.STORE_DIR)[1]
    tars = sorted(n for n in vs if n.endswith('.tar'))
    assert tars, f"khong thay tep .tar nao trong cxrlt_2024/{'/'.join(C.STORE_DIR)} -> DUNG"
    t0, n_tar, n_byte, cach, xong = time.time(), 0, 0, 'song song', set()
    if need and n_threads and n_threads > 1:
        tl = threading.local()
        try:
            for name, b in _window(tars, lambda nm: tar_bytes(vs[nm][0], tl), int(n_threads)):
                n_tar += 1
                n_byte += len(b)
                _extract(io.BytesIO(b), need, out_dir)
                xong.add(name)
                del b
                if n_tar % 20 == 0 or not need:
                    log(f'  kho 518: {n_tar} / {len(tars)} tep .tar | con thieu {len(need)} / {n_can} | {n_byte / 2 ** 20 / max(time.time() - t0, 1e-9):.0f} MB/s')
                if not need:
                    break
        except Exception as e:                                       # noqa: BLE001  đường song song hỏng -> tuần tự bằng khối chuẩn (đã chạy thật 18/11)
            cach = f'song song hong ({type(e).__name__}: {str(e)[:120]}) -> tuan tu'
            log(f'  ! kho 518: {cach}')
    if need:
        if cach == 'song song':
            cach = 'tuan tu'
        for name in tars:
            if not need:
                break
            if name in xong:
                continue
            p = drive['tai'](vs[name][0])
            n_tar += 1
            n_byte += os.path.getsize(p)
            _extract(p, need, out_dir)
            os.remove(p)
            if n_tar % 20 == 0 or not need:
                log(f'  kho 518: {n_tar} / {len(tars)} tep .tar | con thieu {len(need)} / {n_can}')
    assert not need, f'kho 518 thieu {len(need)} anh can cho phep thu -> DUNG'
    return dict(n_can=n_can, n_co_san=n_san, n_lay=n_can - n_san, n_tar_doc=n_tar, n_tar=len(tars), mb=round(n_byte / 2 ** 20), giay=round(time.time() - t0), cach=cach)


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
