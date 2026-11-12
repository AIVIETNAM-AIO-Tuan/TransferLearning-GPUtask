# -*- coding: utf-8 -*-
"""Lớp dữ liệu `ShenzhenL` cho mã gốc CheXFound (PLAN_88 §4.3 (1)). IMPORT mã CheXFound ở mức module -> chỉ import trong tiến trình có repo CheXFound
(commit ghim) trên sys.path: tiến trình torchrun (shenzhen_launch) và bộ kiểm. Bộ điều phối (shenzhen_upstream) KHÔNG import tệp này.

ShenzhenL = lớp `Shenzhen` của họ (chexfound/data/datasets/shenzhen.py @ 7e288d0), kế thừa nguyên: đọc `<root>/<split>.csv` (cột `path`, `label`),
nhãn int64, độ dài từng tách assert 463 / 65 / 134, is_multilabel False, nhánh 'ood' trong root. Đổi ĐÚNG get_image_data -> datasets.decode_png
(convert('L') trước NumPy). Lớp ở mức module (không phải lớp cục bộ) -> pickle được khi DataLoader dùng tiến trình spawn.

Đăng ký tên 'ShenzhenL': bọc chexfound.data.loaders._parse_dataset_str (make_dataset của họ tra hàm này theo tên trong module lúc gọi). Phần
tách chuỗi / kiểm khoá / đổi split giao cho CHÍNH nhánh 'Shenzhen' của họ, chỉ thay lớp trả về."""
from chexfound.data.datasets.shenzhen import Shenzhen

from external_benchmarks.config import DATASET_NAME as NAME
from external_benchmarks.datasets import decode_png

SOURCE_NAME = 'Shenzhen'


class ShenzhenL(Shenzhen):
    '''= Shenzhen của họ; chỉ get_image_data: convert('L') trước np.array (palette -> mức xám thật).'''

    def get_image_data(self, index: int):
        return decode_png(self.labels.iloc[index]['path'])


def register_shenzhen():
    '''Bọc _parse_dataset_str: 'ShenzhenL:...' -> nhánh 'Shenzhen:...' của họ (cùng tách / kiểm khoá / đổi split) rồi trả ShenzhenL; tên khác -> hàm
       gốc. Cổng: root không chứa 'ood' (shenzhen.py:54 đổi sang chế độ 40 nhãn OOD). Gọi lại -> không bọc hai lần. Trả hàm gốc.'''
    import chexfound.data.loaders as loaders
    original = loaders._parse_dataset_str
    if getattr(original, '_shenzhen_l', False):
        return original

    def parse(dataset_str):
        tokens = dataset_str.split(':')
        if tokens[0] != NAME:
            return original(dataset_str)
        class_, kwargs = original(':'.join([SOURCE_NAME] + tokens[1:]))
        assert class_ is Shenzhen, f'nhanh Shenzhen cua ho tra {class_} -> DUNG'
        assert 'ood' not in kwargs.get('root', ''), f"root {kwargs.get('root')!r} chua 'ood' -> lop cua ho doi sang 40 nhan OOD -> DUNG"
        return ShenzhenL, kwargs

    parse._shenzhen_l = True
    loaders._parse_dataset_str = parse
    return original
