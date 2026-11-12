# -*- coding: utf-8 -*-
"""Ảnh Shenzhen trên Drive (PLAN_88 §4.7, §14.17): 662 PNG gốc — đúng các tệp đã kiểm palette 10–12/11 trên máy — gói thành MỘT tệp tar không nén,
thứ tự tên + siêu dữ liệu cố định (dựng lại từ cùng tệp -> cùng byte, cùng md5 ghim ở config). Tuấn tải tệp tar lên `cxrlt_2024/shenzhen/`; notebook
tải qua khối đọc Drive chuẩn (md5 máy chủ = ghim, `tai` kiểm md5 bản tải về), giải nén AN TOÀN: chỉ tệp thường, tên đúng `<thư mục>/<tên>` của 662 tên
trong tệp chia, không trùng / thiếu / thừa. Đo cho `88a`: giải mã PNG trong tiến trình chính (loader val / test của họ num_workers=0) và đọc bộ đệm
mảng uint8 (chỉ đo — có dùng bộ đệm cho `88c` hay không do Tuấn chọn sau `88a`)."""
import hashlib
import os
import re
import tarfile
import time

from . import config, datasets

TAR_DIR = 'shenzhen_png'                   # thư mục trong tar: shenzhen_png/<tên>.png
_NAME = re.compile(r'CHNCXR_\d{4}_[01]\.png')


def _md5_file(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def png_manifest(image_dir, names):
    '''[(tên, md5, kích thước)] theo thứ tự tên + md5 của bảng đó (dấu vân tay 662 tệp PNG, độc lập với cách gói).'''
    rows = [(n, _md5_file(os.path.join(image_dir, n)), os.path.getsize(os.path.join(image_dir, n))) for n in sorted(names)]
    text = ''.join(f'{n}\t{m}\t{s}\n' for n, m, s in rows)
    return rows, hashlib.md5(text.encode()).hexdigest()


def build_tar(image_dir, names, out_path):
    '''MÁY CLAUDE: gói các PNG `names` của image_dir thành tar (USTAR, không nén) theo thứ tự tên; mtime 0, mode 0644, uid / gid 0, không tên người dùng
       -> cùng tệp nguồn thì cùng byte. Cổng: đọc lại tar -> từng thành viên trùng tệp nguồn (md5). -> dict(md5, kich_thuoc, so_anh, md5_bang_png).'''
    names = sorted(names)
    assert all(_NAME.fullmatch(n) for n in names), 'ten anh la (khong phai CHNCXR_dddd_[01].png) -> DUNG'
    rows, man_md5 = png_manifest(image_dir, names)
    tmp = out_path + '.dang_ghi'
    with tarfile.open(tmp, 'w', format=tarfile.USTAR_FORMAT) as tf:
        for n, _, size in rows:
            ti = tarfile.TarInfo(f'{TAR_DIR}/{n}')
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = size, 0, 0o644, 0, 0, '', ''
            with open(os.path.join(image_dir, n), 'rb') as f:
                tf.addfile(ti, f)
    want = {f'{TAR_DIR}/{n}': m for n, m, _ in rows}
    with tarfile.open(tmp, 'r:') as tf:
        got = {m.name: hashlib.md5(tf.extractfile(m).read()).hexdigest() for m in tf.getmembers() if m.isfile()}
    assert got == want, 'tar doc lai KHAC tep nguon -> DUNG'
    os.replace(tmp, out_path)
    return dict(md5=_md5_file(out_path), kich_thuoc=os.path.getsize(out_path), so_anh=len(rows), md5_bang_png=man_md5)


def extract_tar(tar_path, dest_dir, names):
    '''Giải nén AN TOÀN vào dest_dir/TAR_DIR: mọi thành viên phải là tệp thường tên `TAR_DIR/<tên>` với <tên> thuộc `names` (tệp chia), không trùng;
       đủ đúng tập `names` (không thiếu / thừa); ghi bằng nội dung đọc từ tar (không dùng extract -> không có liên kết / đường dẫn lạ). -> thư mục ảnh.'''
    want = set(names)
    out_dir = os.path.join(dest_dir, TAR_DIR)
    os.makedirs(out_dir, exist_ok=True)
    seen = set()
    with tarfile.open(tar_path, 'r:') as tf:
        for m in tf:
            parts = m.name.split('/')
            assert m.isfile() and len(parts) == 2 and parts[0] == TAR_DIR and parts[1] in want and parts[1] not in seen, (
                f'thanh vien tar la: {m.name!r} (kieu {m.type!r}) -> DUNG')
            seen.add(parts[1])
            with tf.extractfile(m) as src, open(os.path.join(out_dir, parts[1]), 'wb') as dst:
                dst.write(src.read())
    assert seen == want, f'tar thieu {len(want - seen)} anh cua tep chia -> DUNG'
    extra = sorted(set(os.listdir(out_dir)) - want)
    assert not extra, f'thu muc anh co tep ngoai tep chia: {extra[:5]} -> DUNG'
    return out_dir


def fetch_shenzhen_images(kiem_ke, tai, xung_dot, dest_dir, names, log=print):
    '''COLAB: tệp tar ở cxrlt_2024/<config.SHENZHEN_DRIVE> qua khối đọc Drive chuẩn. Cổng: có tệp; md5 máy chủ của bản dùng = ghim (TRƯỚC khi tải);
       `tai` kiểm md5 bản tải về = md5 máy chủ; giải nén an toàn đúng 662 tên của tệp chia. -> dict(thu_muc_anh, md5, kich_thuoc, giay_tai, giay_giai_nen).'''
    kk = kiem_ke(config.SHENZHEN_DRIVE, mau=re.escape(config.SHENZHEN_TAR))[1]
    versions = kk.get(config.SHENZHEN_TAR)
    assert versions, (f"thieu {config.SHENZHEN_TAR} trong cxrlt_2024/{'/'.join(config.SHENZHEN_DRIVE)} -> Tuan tai tep tar len Drive "
                      '(PLAN_88 §14.17) -> DUNG')
    if xung_dot(versions):
        log(f'  ! {config.SHENZHEN_TAR}: {len(versions)} ban NOI DUNG KHAC -> chi dung ban co md5 = ghim')
    good = [v for v in versions if v['md5'] == config.SHENZHEN_TAR_MD5]
    assert good, f"{config.SHENZHEN_TAR}: md5 tren Drive {[v['md5'] for v in versions]} != ghim {config.SHENZHEN_TAR_MD5} -> DUNG"
    t0 = time.time()
    path = tai(good[0])
    t1 = time.time()
    assert os.path.getsize(path) == config.SHENZHEN_TAR_SIZE, f'{path}: kich thuoc {os.path.getsize(path)} != {config.SHENZHEN_TAR_SIZE} -> DUNG'
    image_dir = extract_tar(path, dest_dir, names)
    return dict(thu_muc_anh=image_dir, md5=good[0]['md5'], kich_thuoc=os.path.getsize(path), giay_tai=t1 - t0, giay_giai_nen=time.time() - t1)


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
