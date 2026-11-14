# -*- coding: utf-8 -*-
"""Tệp trọng số nằm trên Drive của ta (CheXFound, MRM, CheXzero, CheXWorld): đường dẫn cục bộ TÊN CỐ ĐỊNH cho mọi tiến trình (notebook, torchrun,
đo đặc trưng) + hàm lấy tệp từ Drive qua khối đọc chuẩn (notebook truyền các hàm của khối vào — gói không gọi Drive, không chứa ID Drive).

Vị trí trong spec (`drive_dir`): 'cxrlt_2024/<...>' -> kiem_ke từ gốc cxrlt_2024; còn lại tính từ My Drive (vd 'MIMIC_CXR/MRM') -> _duong + _liet_ke
(như ô C10 của module 88). Chọn bản có md5 = spec.file_md5 (nhiều bản trùng tên: chỉ nhận đúng bản đã ghim; không có -> dừng). Tên tệp tải về mang
ID Drive (khối chuẩn) -> liên kết sang tên cố định dưới WEIGHTS_DIR (mã họ ghi đường dẫn vào log)."""
import os
import shutil

WEIGHTS_DIR_ENV = 'VB_WEIGHTS_DIR'
DEFAULT_DIR = '/content/trong_so_89'


def weights_dir():
    return os.environ.get(WEIGHTS_DIR_ENV, DEFAULT_DIR)


def weights_path(spec):
    '''Đường dẫn cục bộ của tệp trọng số Drive của spec (đặt bởi fetch; bộ kiểm trên máy: đặt biến VB_WEIGHTS_DIR).'''
    assert spec.drive_file, f'{spec.name}: khong co tep tren Drive (tai tu Hugging Face) -> DUNG'
    p = os.path.join(weights_dir(), spec.drive_file)
    assert os.path.exists(p), f'{spec.name}: thieu {p} (C4 chua lay tep tu Drive / chua dat {WEIGHTS_DIR_ENV}) -> DUNG'
    return p


def _versions(spec, drive):
    parts = [p for p in spec.drive_dir.split('/') if p]
    if parts[0] == 'cxrlt_2024':
        return drive['kiem_ke'](tuple(parts[1:]))[1].get(spec.drive_file, [])
    folder = drive['_duong'](tuple(parts))
    assert folder, f"{spec.name}: khong thay thu muc My Drive/{spec.drive_dir} (ten thu muc la gia dinh tu chat -> kiem tren Drive) -> DUNG"
    out = [dict(id=f['id'], ten=f['name'], md5=f.get('md5Checksum'), size=int(f.get('size') or 0), mt=f.get('modifiedTime', ''))
           for f in drive['_liet_ke'](folder) if f['name'] == spec.drive_file]
    return sorted(out, key=lambda b: b['mt'], reverse=True)


def fetch(spec, drive, dest_dir=None, log=print):
    '''Lấy tệp trọng số của spec từ Drive (khối đọc chuẩn) -> dest_dir/<drive_file>; md5 + kích thước = ghim; đặt VB_WEIGHTS_DIR cho mọi tiến trình con.
       -> dict(tep, md5, byte, so_ban, vi_tri).'''
    dest_dir = dest_dir or weights_dir()
    vs = _versions(spec, drive)
    assert vs, f'{spec.name}: khong co {spec.drive_file} trong {spec.drive_dir} tren Drive -> DUNG'
    pin = [v for v in vs if v['md5'] == spec.file_md5 and int(v['size']) == int(spec.file_size)]
    assert pin, (f"{spec.name}: {len(vs)} ban {spec.drive_file} tren Drive, KHONG ban nao co md5 / kich thuoc da ghim "
                 f"({[(v['md5'], v['size']) for v in vs]} != {spec.file_md5} / {spec.file_size}) -> DUNG")
    if len(vs) > 1:
        log(f'  ! {spec.drive_file}: {len(vs)} ban trung ten tren Drive -> dung ban co md5 da ghim')
    src = os.path.abspath(drive['tai'](pin[0]))
    os.makedirs(dest_dir, exist_ok=True)
    dst = os.path.join(dest_dir, spec.drive_file)
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
    assert os.path.getsize(dst) == int(spec.file_size), f'{dst}: kich thuoc khac ban ghim -> DUNG'
    os.environ[WEIGHTS_DIR_ENV] = dest_dir
    return dict(tep=spec.drive_file, md5=pin[0]['md5'], byte=int(pin[0]['size']), so_ban=len(vs), vi_tri=spec.drive_dir)
