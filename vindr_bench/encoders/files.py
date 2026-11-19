# -*- coding: utf-8 -*-
"""Tệp trọng số nằm trên Drive của ta (CheXFound, MRM, CheXzero, CheXWorld, B): đường dẫn cục bộ TÊN CỐ ĐỊNH cho mọi tiến trình (notebook, torchrun,
đo đặc trưng) + hàm lấy tệp từ Drive qua khối đọc chuẩn (notebook truyền các hàm của khối vào — gói không gọi Drive, không chứa ID Drive).

Vị trí trong spec (`drive_dir`): 'cxrlt_2024/<...>' -> kiem_ke từ gốc cxrlt_2024; còn lại tính từ My Drive (vd 'MIMIC_CXR/MRM') -> _duong + _liet_ke
(như ô C10 của module 88). Mỗi tệp (tệp chính + `extra_files` cùng thư mục, vd cấu hình của B) chọn bản có md5 + kích thước = ghim (nhiều bản trùng
tên: chỉ nhận đúng bản đã ghim; không có -> dừng). Tên tệp tải về mang ID Drive (khối chuẩn) -> liên kết sang tên cố định dưới WEIGHTS_DIR (mã họ ghi đường
dẫn vào log)."""
import os
import shutil

WEIGHTS_DIR_ENV = 'VB_WEIGHTS_DIR'
DEFAULT_DIR = '/content/trong_so_89'


def weights_dir():
    return os.environ.get(WEIGHTS_DIR_ENV, DEFAULT_DIR)


def weights_path(spec, name=None):
    '''Đường dẫn cục bộ của tệp Drive `name` (mặc định: tệp trọng số chính) của spec (đặt bởi fetch; bộ kiểm trên máy: đặt biến VB_WEIGHTS_DIR).'''
    assert spec.drive_file, f'{spec.name}: khong co tep tren Drive (tai tu Hugging Face) -> DUNG'
    name = name or spec.drive_file
    assert name in [f for f, _, _ in spec.drive_files], f'{spec.name}: {name} khong thuoc tep Drive cua spec -> DUNG'
    p = os.path.join(weights_dir(), name)
    assert os.path.exists(p), f'{spec.name}: thieu {p} (C4 chua lay tep tu Drive / chua dat {WEIGHTS_DIR_ENV}) -> DUNG'
    return p


def _versions(spec, drive, name):
    parts = [p for p in spec.drive_dir.split('/') if p]
    if parts[0] == 'cxrlt_2024':
        return drive['kiem_ke'](tuple(parts[1:]))[1].get(name, [])
    folder = drive['_duong'](tuple(parts))
    assert folder, f"{spec.name}: khong thay thu muc My Drive/{spec.drive_dir} (ten thu muc la gia dinh tu chat -> kiem tren Drive) -> DUNG"
    out = [dict(id=f['id'], ten=f['name'], md5=f.get('md5Checksum'), size=int(f.get('size') or 0), mt=f.get('modifiedTime', ''))
           for f in drive['_liet_ke'](folder) if f['name'] == name]
    return sorted(out, key=lambda b: b['mt'], reverse=True)


def _fetch_one(spec, drive, name, md5, size, dest_dir, log):
    '''size None (0.11.0, 5 hàng Việc B): chỉ ghim md5; kích thước tệp tải về phải = kích thước Drive liệt kê cho bản đã ghim (ghi lại trong kết quả).'''
    vs = _versions(spec, drive, name)
    assert vs, f'{spec.name}: khong co {name} trong {spec.drive_dir} tren Drive -> DUNG'
    pin = [v for v in vs if v['md5'] == md5 and (size is None or int(v['size']) == int(size))]
    assert pin, (f"{spec.name}: {len(vs)} ban {name} tren Drive, KHONG ban nao co md5 / kich thuoc da ghim "
                 f"({[(v['md5'], v['size']) for v in vs]} != {md5} / {size}) -> DUNG")
    if len(vs) > 1:
        log(f'  ! {name}: {len(vs)} ban trung ten tren Drive -> dung ban co md5 da ghim')
    src = os.path.abspath(drive['tai'](pin[0]))
    dst = os.path.join(dest_dir, name)
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
    assert os.path.getsize(dst) == int(pin[0]['size']) and (size is None or os.path.getsize(dst) == int(size)), (
        f'{dst}: kich thuoc {os.path.getsize(dst)} khac ban ghim ({size}) / Drive ({pin[0]["size"]}) -> DUNG')
    return dict(tep=name, md5=pin[0]['md5'], byte=int(pin[0]['size']), so_ban=len(vs), ghim_kich_thuoc=size is not None)


def fetch(spec, drive, dest_dir=None, log=print):
    '''Lấy MỌI tệp Drive của spec (tệp chính + extra_files) từ Drive (khối đọc chuẩn) -> dest_dir/<tên>; md5 + kích thước = ghim; đặt VB_WEIGHTS_DIR cho
       mọi tiến trình con. -> dict của tệp chính (tep, md5, byte, so_ban, vi_tri) + 'phu' = [dict của từng tệp phụ].'''
    dest_dir = dest_dir or weights_dir()
    os.makedirs(dest_dir, exist_ok=True)
    got = [_fetch_one(spec, drive, n, m, z, dest_dir, log) for n, m, z in spec.drive_files]
    os.environ[WEIGHTS_DIR_ENV] = dest_dir
    return dict(got[0], vi_tri=spec.drive_dir, phu=got[1:])
