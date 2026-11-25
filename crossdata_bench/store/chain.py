# -*- coding: utf-8 -*-
"""Chuỗi kho 518 của module 93 = kho 'kcb' của 85 (đệm vuông, KHÔNG cân bằng histogram) — dùng NGUYÊN hàm của vindr_bench.data.crop_store @ c81af62
(to_u8 = _u8_khong_cb_85, chain(.., 'dem_vuong') = phần sau bước u8 của chuyen_85, save_jpeg = bước cuối; test_crop_store của 89 so từng bit với nguồn 85):

    DICOM (RSNA)  pixel -> trừ min / chia max / cắt uint8 -> đảo MONOCHROME1 -> JPEG q95 cv2 -> letterbox 1022 (cạnh DÀI, đệm viền 0) BICUBIC -> JPEG q90
                  -> 518 BICUBIC -> JPEG q95; không cửa sổ DICOM; header phải qua đúng các kiểm tra của 85 (crop_store.decode)
    PNG (Shenzhen) Image.convert('L') (áp palette -> mức xám thật; 334 / 662 ảnh Shenzhen có palette ĐẢO lệch theo nhãn — 88) -> cùng chuỗi từ bước trừ
                  min / chia max (= decode_png của 88: min-max -> uint8, rồi chuỗi 85)
    JPEG (CheXpert, 0.5.0) cùng hàm với PNG (Pillow mở JPEG, convert('L')); chỉ khác chữ 'cach' ghi trong meta

Một ảnh = một hàm thuần (bytes -> ảnh 518) để cổng mẫu chạy lại được và so từng bit."""
import io

import numpy as np
from PIL import Image

from vindr_bench.data import crop_store as CS

KINDS = ('png', 'dicom', 'jpg')
CACH = {
    'png': ('PNG -> convert(L) (ap palette) -> tru min / chia max / cat uint8 -> KHONG equalizeHist -> JPEG q95 cv2 -> letterbox 1022 (canh DAI, dem '
            'vien 0) BICUBIC -> JPEG q90 -> 518 BICUBIC -> JPEG q95 (= chuoi kho kcb cua 85)'),
    'dicom': ('DICOM pixel -> tru min / chia max / cat uint8 -> dao MONOCHROME1 -> KHONG equalizeHist -> JPEG q95 cv2 -> letterbox 1022 (canh DAI, dem '
              'vien 0) BICUBIC -> JPEG q90 -> 518 BICUBIC -> JPEG q95; KHONG cua so DICOM (= chuoi kho kcb cua 85)'),
    # 0.5.0 (CheXpert bản đầy đủ: JPEG 8 bit, 93j: mode L 902 / 902 ảnh val + test) — cùng hàm với PNG (Pillow mở JPEG), chỉ khác chữ ghi trong meta
    'jpg': ('JPEG goc -> convert(L) -> tru min / chia max / cat uint8 -> KHONG equalizeHist -> JPEG q95 cv2 -> letterbox 1022 (canh DAI, dem vien 0) '
            'BICUBIC -> JPEG q90 -> 518 BICUBIC -> JPEG q95 (= chuoi kho kcb cua 85)'),
}
SIDE_OUT = CS.SIDE_OUT


def png_u8(b):
    '''bytes PNG -> (lỗi | None, uint8 2D, thông tin: mode gốc, rộng, cao).'''
    try:
        with Image.open(io.BytesIO(b)) as im:
            mode, (w, h) = im.mode, im.size
            g = np.asarray(im.convert('L'))
        return None, CS.to_u8(g, False), dict(mode=mode, w=int(w), h=int(h))
    except Exception as e:
        return f'giai ma: {type(e).__name__}', None, None


def dicom_u8(b):
    '''bytes DICOM -> (lỗi | None, uint8 2D, thông tin header của 85: pi, ts, rows, cols, bits).'''
    err, a, info = CS.decode(b)
    if err:
        return err, None, None
    try:
        return None, CS.to_u8(a, info['pi'] == 'MONOCHROME1'), info
    except Exception as e:
        return f'u8: {type(e).__name__}', None, None


def to_u8(kind, b):
    assert kind in KINDS, kind
    return dicom_u8(b) if kind == 'dicom' else png_u8(b)            # 'png' và 'jpg' (0.5.0): cùng đường Pillow convert('L')


def to_518(u8):
    '''uint8 2D -> ảnh 'L' 518 x 518 (CHƯA nén JPEG q95 cuối — save_jpeg làm).'''
    return CS.chain(u8, 'dem_vuong')


def jpeg_roundtrip(im):
    '''Ảnh 'L' -> mảng uint8 sau JPEG q95 (= ảnh đọc lại từ kho).'''
    bio = io.BytesIO()
    im.save(bio, 'JPEG', quality=CS.JPEG_OUT)
    bio.seek(0)
    return np.asarray(Image.open(bio).convert('L'))


save_jpeg = CS.save_jpeg
