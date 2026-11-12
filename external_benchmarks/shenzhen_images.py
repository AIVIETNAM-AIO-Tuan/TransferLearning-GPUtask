# -*- coding: utf-8 -*-
"""Ảnh Shenzhen cho notebook (PLAN_88 §4.7, §14.20): 662 PNG gốc đọc THẲNG từ Drive MyDrive/MIMIC_CXR/Shenzhen (Tuấn đặt; không tar, không giải nén —
Tuấn 12/11). Trước khi tải: integrity.check_shenzhen — md5 + kích thước từng tệp qua Drive API = mốc của bản đã kiểm trên máy (shenzhen_manifest).
Tải từng tệp qua khối đọc Drive chuẩn (`tai` kiểm md5 bản tải về = md5 máy chủ, mà md5 máy chủ đã = mốc), đặt TÊN GỐC trong một thư mục (liên kết tới
bản tải về — tên bản tải về mang ID Drive; không tạo được liên kết -> chép). Đo cho 88a: giải mã PNG / bộ đệm uint8 (time_decode)."""
import hashlib
import os
import shutil
import time

from . import datasets


def _md5_file(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def png_manifest(image_dir, names):
    '''[(tên, md5, kích thước)] theo thứ tự tên + md5 của bảng "tên\\tmd5\\tkích thước\\n" (= cách tính config.SHENZHEN_PNG_TABLE_MD5).'''
    rows = [(n, _md5_file(os.path.join(image_dir, n)), os.path.getsize(os.path.join(image_dir, n))) for n in sorted(names)]
    text = ''.join(f'{n}\t{m}\t{s}\n' for n, m, s in rows)
    return rows, hashlib.md5(text.encode()).hexdigest()


def fetch_drive_pngs(chosen, tai, dest_dir, log=print):
    '''COLAB: chosen = integrity.check_shenzhen(...)['png']['chon'] (đã kiểm: đủ 662 tên, md5 + kích thước máy chủ = mốc) -> tải từng tệp bằng `tai`
       (khối đọc Drive chuẩn: md5 bản tải về = md5 máy chủ) -> dest_dir/<tên gốc>. Cổng: tập tên = mốc; kích thước bản tải về = mốc; thư mục đích không có
       tệp ngoài mốc. -> dict(thu_muc_anh, so_anh, kich_thuoc, giay_tai, lien_ket).'''
    from . import shenzhen_manifest as man
    assert set(chosen) == set(man.PNG), f'tap anh chon {len(chosen)} != moc {len(man.PNG)} -> DUNG (chay integrity.check_shenzhen truoc)'
    assert all(chosen[n]['md5'] == man.PNG[n][0] for n in chosen), 'md5 may chu cua anh chon khac moc -> DUNG'
    os.makedirs(dest_dir, exist_ok=True)
    t0, total, links = time.time(), 0, 0
    for i, n in enumerate(sorted(chosen)):
        src = os.path.abspath(tai(chosen[n]))
        assert os.path.getsize(src) == man.PNG[n][1], f'{n}: kich thuoc ban tai ve {os.path.getsize(src)} != moc {man.PNG[n][1]} -> DUNG'
        dst = os.path.join(dest_dir, n)
        if os.path.lexists(dst):
            os.remove(dst)
        try:
            os.symlink(src, dst)
            links += 1
        except OSError:
            shutil.copyfile(src, dst)
        total += man.PNG[n][1]
        if (i + 1) % 100 == 0:
            log(f'  {i + 1} / {len(chosen)} anh ({time.time() - t0:.0f} s)')
    extra = sorted(set(os.listdir(dest_dir)) - set(man.PNG))
    assert not extra, f'thu muc anh co tep ngoai moc: {extra[:5]} -> DUNG'
    return dict(thu_muc_anh=dest_dir, so_anh=len(chosen), kich_thuoc=total, giay_tai=time.time() - t0, lien_ket=links)


def time_decode(image_dir, names, n=48, seed=0, cache_dir=None):
    '''ĐO (88a): n ảnh chọn ngẫu nhiên cố định (hạt `seed`) trong tiến trình gọi (một lõi):
       - giải mã = datasets.decode_png (= đường ShenzhenL: convert('L') + min-max -> uint8 -> RGB) từ PNG;
       - bộ đệm: mảng uint8 [H, W] (mức xám SAU min-max — đúng ảnh ShenzhenL dựng ra) ghi .npy rồi đọc lại + Image.fromarray(...).convert('RGB');
         cổng: ảnh từ bộ đệm TRÙNG từng điểm ảnh với giải mã PNG.
       -> dict(n, giay_png [TB, trung vị, max], giay_bo_dem [...], mb_png_tb, mb_bo_dem_tb, trung_tung_diem_anh).'''
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    pick = [sorted(names)[i] for i in sorted(rng.choice(len(names), size=min(n, len(names)), replace=False))]
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(image_dir)), 'bo_dem_do_88a')
    os.makedirs(cache_dir, exist_ok=True)
    t_png, t_npy, sz_png, sz_npy, same = [], [], [], [], True
    for name in pick:
        p = os.path.join(image_dir, name)
        t0 = time.perf_counter()
        im = datasets.decode_png(p)
        t_png.append(time.perf_counter() - t0)
        gray = np.asarray(im)[:, :, 0].copy()                 # RGB từ L: ba kênh bằng nhau
        q = os.path.join(cache_dir, name[:-4] + '.npy')
        np.save(q, gray)
        t0 = time.perf_counter()
        im2 = Image.fromarray(np.load(q)).convert('RGB')
        t_npy.append(time.perf_counter() - t0)
        same &= np.array_equal(np.asarray(im), np.asarray(im2))
        sz_png.append(os.path.getsize(p) / 2 ** 20)
        sz_npy.append(os.path.getsize(q) / 2 ** 20)
        os.remove(q)
    st = lambda v: dict(tb=float(np.mean(v)), trung_vi=float(np.median(v)), max=float(np.max(v)))
    return dict(n=len(pick), hat=seed, giay_png=st(t_png), giay_bo_dem=st(t_npy), mb_png_tb=float(np.mean(sz_png)), mb_bo_dem_tb=float(np.mean(sz_npy)),
                trung_tung_diem_anh=bool(same))
