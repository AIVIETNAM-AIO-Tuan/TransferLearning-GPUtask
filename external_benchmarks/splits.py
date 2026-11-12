# -*- coding: utf-8 -*-
"""Danh sách chia (PLAN_88 §2.1, §6.2). Phần Shenzhen: tệp chia BenchmarkTransformers đã ghim (commit + md5; CheXFound IV-B dùng đúng 463 / 65 / 134)
-> CSV đúng khuôn lớp Shenzhen của họ (`<root>/{train,val,test}.csv`, cột `path` = đường dẫn ảnh đầy đủ, `label` = 0 / 1). Thứ tự hàng = thứ tự dòng
trong tệp của BenchmarkTransformers (GIẢ ĐỊNH: CheXFound không công bố CSV; thứ tự hàng đổi lô của luồng mẫu seed 0). Không sinh lại split.
Phần RSNA (5-fold hai điều kiện, test ngoài) thêm khi dựng (mã kiểm ở scratchpad 10–11/11)."""
import hashlib
import json
import os

from . import config

SHENZHEN_SPLIT_NAMES = ('train', 'val', 'test')
CSV_COLUMNS = ('path', 'label')            # lớp của họ đọc labels.iloc[i]['path'] và cột 'label' (root không chứa 'ood')
MANIFEST = 'split_manifest.json'


def _md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def download_shenzhen_lists(dest_dir, log=print):
    '''Tải 3 tệp chia theo COMMIT GHIM (raw.githubusercontent.com) vào dest_dir; tệp đã có thì giữ. md5 kiểm ở read_shenzhen_lists. -> dest_dir.'''
    import urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    for split, (name, _, _) in config.SHENZHEN_SPLITS.items():
        p = os.path.join(dest_dir, name)
        if os.path.exists(p):
            continue
        url = config.SHENZHEN_SPLIT_URL.format(commit=config.SHENZHEN_SPLIT_COMMIT, name=name)
        data = urllib.request.urlopen(url, timeout=120).read()
        with open(p, 'wb') as f:
            f.write(data)
        log(f'  tai {name}: {len(data)} B')
    return dest_dir


def read_shenzhen_lists(list_dir):
    '''3 tệp chia -> {tách: [(tên ảnh, nhãn int)]} theo thứ tự dòng. Cổng: md5 = bản ghim; số dòng 463 / 65 / 134; mỗi dòng "<tên>.png,<0|1>";
       nhãn = hậu tố tên (_0 bình thường / _1 lao); không trùng tên trong / giữa các tách; tổng 662.'''
    out = {}
    for split, (name, want_md5, want_n) in config.SHENZHEN_SPLITS.items():
        with open(os.path.join(list_dir, name), 'rb') as f:
            raw = f.read()
        assert _md5_bytes(raw) == want_md5, f'{name}: md5 {_md5_bytes(raw)} != {want_md5} (ban ghim) -> DUNG'
        rows = []
        for line in raw.decode('utf-8').splitlines():
            if not line.strip():
                continue
            fname, lab = line.strip().split(',')
            assert fname.endswith('.png') and lab in ('0', '1'), f'{name}: dong la {line!r} -> DUNG'
            assert fname[-6:-4] == f'_{lab}', f'{name}: nhan {lab} khac hau to ten {fname} -> DUNG'
            rows.append((fname, int(lab)))
        assert len(rows) == want_n, f'{name}: {len(rows)} dong != {want_n} -> DUNG'
        out[split] = rows
    names = [n for rows in out.values() for n, _ in rows]
    assert len(names) == len(set(names)) == config.SHENZHEN_N_IMAGES, f'ten trung / tong {len(set(names))} != {config.SHENZHEN_N_IMAGES} -> DUNG'
    return out


def _check_root(csv_root):
    '''root đi vào chuỗi dataset của họ (tách bằng ':' và '='; 'ood' trong root -> lớp đổi sang 40 nhãn OOD).'''
    assert ':' not in csv_root and '=' not in csv_root, f'{csv_root!r}: co ":" / "=" -> chuoi dataset cua ho tach sai (Windows: duong dan tuong doi) -> DUNG'
    assert 'ood' not in csv_root, f"{csv_root!r} chua 'ood' -> lop Shenzhen cua ho doi sang 40 nhan OOD -> DUNG"


def write_shenzhen_csvs(list_dir, image_dir, csv_root, metadata_csv=None):
    '''Tệp chia đã ghim -> <csv_root>/{train,val,test}.csv (path = image_dir/<tên>, label) + split_manifest.json. Cổng: read_shenzhen_lists; mọi ảnh có
       trong image_dir và thư mục có ĐÚNG 662 PNG; (metadata_csv: nhãn = 'findings' != 'normal' cho cả 662). Đường dẫn ảnh giữ kiểu của image_dir
       (tuyệt đối trên Colab; tương đối khi kiểm trên Windows). -> manifest dict.'''
    import pandas as pd
    _check_root(csv_root)
    lists = read_shenzhen_lists(list_dir)
    on_disk = sorted(f for f in os.listdir(image_dir) if f.lower().endswith('.png'))
    listed = sorted(n for rows in lists.values() for n, _ in rows)
    assert on_disk == listed, f'thu muc anh {len(on_disk)} PNG KHAC danh sach chia {len(listed)} -> DUNG'
    if metadata_csv is not None:
        meta = pd.read_csv(metadata_csv)
        lab_meta = dict(zip(meta['study_id'], (meta['findings'].astype(str).str.strip().str.lower() != 'normal').astype(int)))
        bad = [n for rows in lists.values() for n, lab in rows if lab_meta.get(n) != lab]
        assert not bad, f'{len(bad)} anh: nhan danh sach chia != findings trong metadata -> DUNG'
    os.makedirs(csv_root, exist_ok=True)
    man = dict(nguon='jlianglab/BenchmarkTransformers', commit=config.SHENZHEN_SPLIT_COMMIT, image_dir=image_dir, cot=list(CSV_COLUMNS),
               thu_tu='theo dong tep chia (gia dinh)', kiem_metadata=metadata_csv is not None)
    for split, rows in lists.items():
        df = pd.DataFrame({'path': [os.path.join(image_dir, n) for n, _ in rows], 'label': [lab for _, lab in rows]})
        p = os.path.join(csv_root, f'{split}.csv')
        df.to_csv(p, index=False)
        with open(p, 'rb') as f:
            csv_md5 = _md5_bytes(f.read())
        man[split] = dict(tep=config.SHENZHEN_SPLITS[split][0], md5_tep_chia=config.SHENZHEN_SPLITS[split][1], so_anh=len(rows),
                          so_duong=int(sum(lab for _, lab in rows)), md5_ten=_md5_bytes('\n'.join(n for n, _ in rows).encode()), md5_csv=csv_md5)
    with open(os.path.join(csv_root, MANIFEST), 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    return man


def read_shenzhen_csv(csv_root, split):
    '''-> (tên ảnh [N], nhãn int64 [N]) theo thứ tự hàng của CSV (= thứ tự mã họ đọc / chấm).'''
    import numpy as np
    import pandas as pd
    df = pd.read_csv(os.path.join(csv_root, f'{split}.csv'))
    assert list(df.columns) == list(CSV_COLUMNS), f'{split}.csv: cot {list(df.columns)} != {list(CSV_COLUMNS)} -> DUNG'
    return [os.path.basename(p) for p in df['path']], df['label'].to_numpy().astype(np.int64)


def dataset_str(split, csv_root, name=config.DATASET_NAME):
    '''Chuỗi dataset theo cú pháp của họ, vd ShenzhenL:split=TRAIN:root=/content/shenzhen_csv.'''
    assert split in SHENZHEN_SPLIT_NAMES, split
    _check_root(csv_root)
    return f'{name}:split={split.upper()}:root={csv_root}'
