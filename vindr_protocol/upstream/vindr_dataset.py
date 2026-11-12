"""Lớp dữ liệu VinDr cho mã gốc CheXFound (PLAN_86 §12.3). Tác giả KHÔNG công bố lớp VinDr; lớp này chép khuôn lớp `CheXpert` của họ
(chexfound/data/datasets/chexpert.py @ 7e288d0): đọc `<root>/<split>.csv` (cột `Path` + tên nhãn), ảnh `Image.open(...).convert(mode="RGB")`, nhãn int64
(torchmetrics 0.10.3 nhận nhãn nguyên), is_multilabel True, Split TRAIN / VAL / TEST = 'train' / 'val' / 'test', __len__ kiểm số ảnh. Khác CheXpert:
không lọc ảnh nghiêng / không thay -1 (VinDr chỉ có 0 / 1); số ảnh từng tách đọc từ split_sizes.json (CheXpert ghi cứng trong lớp).

Không import gì của họ: lớp cha là VisionDataset của torchvision (lớp cha ExtendedVisionDataset của họ kế thừa nó; CheXpert tự viết __getitem__).
write_split_csvs: VinDrData (vindr_protocol.data.load_vindr: kho ảnh 518, nhãn 28 lớp, tách) -> ba CSV + split_sizes.json."""
import hashlib
import json
import os
from enum import Enum
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset

NAME = 'VinDr'
SIZES_FILE = 'split_sizes.json'
SPLITS = ('train', 'val', 'test')


class _Split(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class VinDr(VisionDataset):
    Split = Union[_Split]

    def __init__(self, *, split: '_Split', root: str, transforms: Optional[Callable] = None, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self.labels = pd.read_csv(os.path.join(self.root, self._split.value + '.csv'))
        with open(os.path.join(self.root, SIZES_FILE), encoding='utf-8') as f:
            sizes = json.load(f)
        self.class_names = list(sizes['class_names'])
        assert list(self.labels.columns) == ['Path'] + self.class_names, f'{self._split.value}.csv: cot khac Path + {len(self.class_names)} nhan'
        self.targets = self.labels[self.class_names].to_numpy()
        self._length = int(sizes[self._split.value])

    def get_image_data(self, index: int):
        img = Image.open(self.labels.iloc[index]['Path'])
        img = img.convert(mode='RGB')
        return img

    def get_target(self, index: int):
        return self.targets[index].astype(np.int64)

    def is_multilabel(self):
        return True

    def is_3d(self):
        return False

    @property
    def split(self) -> '_Split':
        return self._split

    def get_num_classes(self) -> int:
        return len(self.class_names)

    def __len__(self):
        assert len(self.labels) == self._length
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.get_image_data(index)
        target = self.get_target(index)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def parse_dataset_str(dataset_str: str):
    '''= nhánh của _parse_dataset_str (chexfound/data/loaders.py) cho tên 'VinDr': "VinDr:split=TRAIN:root=<thư mục CSV>" -> (VinDr, kwargs).'''
    tokens = dataset_str.split(':')
    assert tokens[0] == NAME, dataset_str
    kwargs = {}
    for token in tokens[1:]:
        key, value = token.split('=')
        assert key in ('root', 'extra', 'split')
        kwargs[key] = value
    if 'split' in kwargs:
        kwargs['split'] = _Split[kwargs['split']]
    return VinDr, kwargs


def dataset_str(split: str, root: str) -> str:
    '''Chuỗi dataset theo cú pháp của họ, vd VinDr:split=TRAIN:root=/content/vindr_csv.'''
    assert split in SPLITS and ':' not in root and '=' not in root, (split, root)
    return f'{NAME}:split={split.upper()}:root={root}'


def write_subset_csvs(src_root, dst_root, n_per_split):
    '''Bộ CSV NHỎ cho lượt chạy thử: n ảnh ĐẦU mỗi tách của bộ CSV đầy đủ (cùng cột, cùng đường dẫn ảnh). Thay cho --num-samples của họ:
       SystemicSamplerWrapper kế thừa Subset mà không gọi Subset.__init__ -> torch mới (Subset.__getitems__ đọc .dataset) lỗi (thử trên máy 12/11).'''
    os.makedirs(dst_root, exist_ok=True)
    with open(os.path.join(src_root, SIZES_FILE), encoding='utf-8') as f:
        sizes = json.load(f)
    out = dict(class_names=sizes['class_names'], store=sizes.get('store'), image_dir=sizes.get('image_dir'), tap_con_cua=src_root)
    for split in SPLITS:
        df = pd.read_csv(os.path.join(src_root, f'{split}.csv')).head(int(n_per_split[split]))
        df.to_csv(os.path.join(dst_root, f'{split}.csv'), index=False)
        out[split] = int(len(df))
    with open(os.path.join(dst_root, SIZES_FILE), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False)
    return out


def _md5_keys(keys):
    return hashlib.md5('\n'.join(keys).encode()).hexdigest()


def write_split_csvs(vindr, root):
    '''VinDrData -> <root>/{train,val,test}.csv (Path = ảnh 518 đã giải nén, 28 cột nhãn int theo thứ tự nhãn của 85) + split_sizes.json (số ảnh,
       tên nhãn, md5 khoá từng tách — cùng cách băm khoá của 85). Thứ tự hàng trong từng tách = vindr.indices(tách) (test = thứ tự 85b, như 86b).'''
    os.makedirs(root, exist_ok=True)
    names = list(vindr.label_names)
    assert len(set(names)) == len(names) and not any(n == 'Path' for n in names), 'ten nhan trung / trung cot Path -> DUNG'
    sizes = dict(class_names=names, store=vindr.store, image_dir=vindr.image_dir)
    for split in SPLITS:
        idx = vindr.indices(split)
        keys = [vindr.keys[i] for i in idx]
        paths = [os.path.join(vindr.image_dir, f'{k}.jpg') for k in keys]
        assert all(os.path.exists(p) for p in paths[:: max(1, len(paths) // 50)]), f'{split}: thieu anh -> DUNG'
        y = np.asarray(vindr.labels[idx])
        assert set(np.unique(y).tolist()) <= {0.0, 1.0}, f'{split}: nhan khong nhi phan -> DUNG'
        df = pd.DataFrame({'Path': paths})
        for j, name in enumerate(names):
            df[name] = y[:, j].astype(np.int64)
        df.to_csv(os.path.join(root, f'{split}.csv'), index=False)
        sizes[split] = int(len(idx))
        sizes[f'md5_{split}'] = _md5_keys(keys)
        sizes[f'chi_so_{split}'] = [int(i) for i in idx]
    with open(os.path.join(root, SIZES_FILE), 'w', encoding='utf-8') as f:
        json.dump(sizes, f, ensure_ascii=False)
    return {k: v for k, v in sizes.items() if not k.startswith('chi_so_')}
