# -*- coding: utf-8 -*-
"""RSNA Pneumonia (Kaggle stage 2) — bảng khoá / người / nhãn / tách (PLAN_93 §3; thuật toán chia = PLAN_88 §14.2, tỉ lệ 60 / 20 / 20 của §14.8).

- Khoá = mã Kaggle (`patientId` của Kaggle là mã ẢNH: 26.684 ảnh có nhãn = 11.452 người NIH — PLAN_88 §2.2).
- Nhãn = max cột `Target` của stage_2_train_labels.csv theo mã (md5 ghim ở external_benchmarks.rsna_manifest.CSV); 88 đã kiểm = nhãn chốt "Lung Opacity" của
  tệp nhãn RSNA ở 26.684 / 26.684 ảnh.
- Người = 8 số đầu tên ảnh NIH (= cột Patient ID của Data_Entry_2017 ở 100 % ảnh, PLAN_88 §14.1) qua tệp nối công khai của RSNA
  pneumonia-challenge-dataset-mappings_2018.json (md5 ghim; tải theo URL công khai lúc chạy, không chép vào gói).
- Chia: đơn vị = người; tầng = người có >= 1 ảnh dương hay không; trong MỖI tầng: mã người sắp tăng dần -> hoán vị numpy.random.RandomState(8810)
  (mới cho từng tầng) -> train = round(0,6 n), val = round(0,2 n), còn lại test. Thứ tự hàng của bảng: train, val, test; trong mỗi tách khoá sắp tăng.
3.000 ảnh test chính thức KHÔNG dùng (2.203 ảnh thuộc người có trong 26.684 ảnh — PLAN_88)."""
import hashlib
import json
import os
import re

import numpy as np

MAPPING_NAME = 'pneumonia-challenge-dataset-mappings_2018.json'
MAPPING_URL = 'https://s3.amazonaws.com/east1.public.rsna.org/AI/2018/' + MAPPING_NAME
MAPPING_MD5, MAPPING_SIZE = '9a80eed58d070bd8d3edbc79629a53c0', 17587143          # = mốc 88 (10/11); tải lại 22/11, trùng
MAPPING_N, MAPPING_PEOPLE = 30000, 12274
LABELS_CSV = 'stage_2_train_labels.csv'
IMAGE_FOLDER = 'stage_2_train_images'
N_IMAGES, N_PEOPLE = 26684, 11452
SPLIT_SEED, FRACTIONS = 8810, (0.6, 0.2)
_NIH = re.compile(r'(\d{8})_\d{3}\.png')


def _md5_file(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fetch_mapping(dest_dir, log=print):
    '''Tải tệp nối công khai của RSNA vào dest_dir (đã có + đúng md5 thì giữ). md5 + kích thước = ghim, không thì dừng. -> đường dẫn.'''
    import urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    p = os.path.join(dest_dir, MAPPING_NAME)
    if not (os.path.exists(p) and _md5_file(p) == MAPPING_MD5):
        req = urllib.request.Request(MAPPING_URL, headers={'User-Agent': 'Wget/1.21'})
        data = urllib.request.urlopen(req, timeout=300).read()
        with open(p, 'wb') as f:
            f.write(data)
        log(f'  tai {MAPPING_NAME}: {len(data):,} B')
    assert os.path.getsize(p) == MAPPING_SIZE and _md5_file(p) == MAPPING_MD5, f'{MAPPING_NAME}: md5 / kich thuoc KHAC ban ghim -> DUNG'
    return p


def read_mapping(path):
    '''-> {mã Kaggle: tên ảnh NIH}. Cổng: 30.000 mục, hai phía không trùng, tên NIH đúng dạng <8 số>_<3 số>.png, 12.274 người.'''
    with open(path, encoding='utf-8') as f:
        rows = json.load(f)
    m = {r['subset_img_id']: r['img_id'] for r in rows}
    assert len(rows) == len(m) == MAPPING_N and len(set(m.values())) == MAPPING_N, 'tep noi RSNA: so muc / trung ma -> DUNG'
    assert all(_NIH.fullmatch(v) for v in m.values()), 'tep noi RSNA: ten anh NIH sai dang -> DUNG'
    assert len({v[:8] for v in m.values()}) == MAPPING_PEOPLE, 'tep noi RSNA: so nguoi NIH != 12.274 -> DUNG'
    return m


def read_labels(csv_path, md5=None):
    '''stage_2_train_labels.csv -> {mã Kaggle: 0 / 1} (max Target theo mã). Cổng: md5 (nếu truyền), Target nhị phân, mọi dòng của một mã cùng Target,
       26.684 mã.'''
    import pandas as pd
    if md5 is not None:
        assert _md5_file(csv_path) == md5, f'{LABELS_CSV}: md5 KHAC ban ghim -> DUNG'
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ['patientId', 'x', 'y', 'width', 'height', 'Target'] and set(df.Target.unique()) <= {0, 1}, f'{LABELS_CSV}: cot / Target la -> DUNG'
    g = df.groupby('patientId').Target
    assert int((g.nunique() > 1).sum()) == 0, f'{LABELS_CSV}: mot ma co ca Target 0 va 1 -> DUNG'
    out = g.max().astype(int).to_dict()
    assert len(out) == N_IMAGES, f'{LABELS_CSV}: {len(out)} ma != {N_IMAGES} -> DUNG'
    return out


def split_people(person_positive, seed=SPLIT_SEED, fractions=FRACTIONS):
    '''{người: có ảnh dương (bool)} -> {người: tách}. MỖI tầng một RandomState(seed) MỚI (= mã chạy thử của 88 ngày 10/11, lấy lại từ nhật ký phiên:
       `ng[np.random.RandomState(8810).permutation(len(ng))]` trong vòng `for tang in (1, 0)`); thứ tự tầng vì thế không đổi kết quả.'''
    out = {}
    for stratum in (True, False):
        ids = np.array(sorted(p for p, v in person_positive.items() if bool(v) == stratum))
        ids = ids[np.random.RandomState(seed).permutation(len(ids))]
        n_tr, n_va = int(round(fractions[0] * len(ids))), int(round(fractions[1] * len(ids)))
        out.update({p: 'train' for p in ids[:n_tr]})
        out.update({p: 'val' for p in ids[n_tr:n_tr + n_va]})
        out.update({p: 'test' for p in ids[n_tr + n_va:]})
    return out


def table(mapping, labels):
    '''-> DataFrame(khoa, tach, nhom (mã người NIH 8 số), anh_nih, 'Lung Opacity') theo thứ tự train / val / test, khoá sắp tăng trong mỗi tách. Cổng: mọi
       mã có nhãn nằm trong tệp nối; 11.452 người.'''
    import pandas as pd
    keys = sorted(labels)
    miss = [k for k in keys if k not in mapping]
    assert not miss, f'{len(miss)} ma Kaggle co nhan KHONG co trong tep noi -> DUNG'
    person = {k: mapping[k][:8] for k in keys}
    pos = {}
    for k in keys:
        pos[person[k]] = pos.get(person[k], False) or labels[k] == 1
    assert len(pos) == N_PEOPLE, f'{len(pos)} nguoi != {N_PEOPLE} -> DUNG'
    sp = split_people(pos)
    df = pd.DataFrame(dict(khoa=keys, tach=[sp[person[k]] for k in keys], nhom=[person[k] for k in keys], anh_nih=[mapping[k] for k in keys],
                           **{'Lung Opacity': [int(labels[k]) for k in keys]}))
    order = {s: i for i, s in enumerate(('train', 'val', 'test'))}
    return df.assign(_o=df.tach.map(order)).sort_values(['_o', 'khoa'], kind='stable').drop(columns='_o').reset_index(drop=True)
