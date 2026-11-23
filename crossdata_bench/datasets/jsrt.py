# -*- coding: utf-8 -*-
"""JSRT (nốt phổi đơn độc) — bảng khoá / tách / nhãn (PLAN_93 §3; Tuấn cho dựng 22/11, dựng 23/11). CHỈ MÔ TẢ (PLAN_93 §10 điểm 5: test ~50 ảnh).

- Nguồn: Drive MyDrive/MIMIC_CXR/JSRT (Tuấn 23/11: có sẵn trên Drive; tên thư mục = config.DATASETS['jsrt'].source, xác nhận ở lượt chạy đầu). Mốc md5 +
  kích thước TỪNG tệp: jsrt_manifest = bản `data/jsrt` của máy (247 PNG 2048 x 2048 kiểu palette 'P' + jsrt_metadata.csv; kiểm kê 23/11) — cùng cách
  external_benchmarks.integrity làm với Shenzhen; bản Drive khác từng byte -> C3 dừng. Tệp CÙNG TÊN khác nội dung ở thư mục khác (vd mặt nạ phổi đặt
  trùng tên ảnh): nếu có đúng bản trùng mốc thì dùng bản đó, các bản kia bỏ qua + đếm + in đường dẫn; không bản nào trùng mốc -> dừng như cũ.
- Kiểm kê 23/11 (chuỗi kho kcb thật): palette ĐẢO (chỉ số i -> mức xám 255 - i) ở 247 / 247 ảnh, cả hai nhãn như nhau -> convert('L') áp palette; sau chuỗi
  518 mọi ảnh "giữa sáng hơn hai bên"; tương quan với ảnh trung bình min 0,41, không ảnh âm. Sàng lọc lối tắt: thống kê ảnh đơn giản đoán nhãn AUROC 0,51 –
  0,67 (kích thước tệp PNG 0,67; phân vị 5 % mức xám 0,65) — ghi ở PLAN_93 §3 (bẫy đã biết).
- Nhãn từ TÊN tệp: JPCLN### = có nốt (154: ác 100 / lành 54), JPCNN### = không nốt (93); đối chiếu cột `state` của jsrt_metadata.csv (md5 ghim):
  malignant / benign <-> JPCLN, non-nodule <-> JPCNN — lệch thì dừng.
- Không có mã người -> mỗi ảnh một cụm (giả định mỗi ảnh một người, như Shenzhen).
- Chia MỘT lần, phân tầng theo nhãn (= thuật toán rsna.split_people, đơn vị ảnh): trong MỖI tầng (có nốt, rồi không nốt) tên sắp tăng -> hoán vị
  numpy.random.RandomState(SPLIT_SEED) MỚI cho từng tầng -> train = round(0,6 n), val = round(0,2 n), còn lại test. Thứ tự hàng: train, val, test; khoá
  sắp tăng trong mỗi tách. Số đếm + vân tay ghim ở config (tính trên máy 23/11)."""
import hashlib
import re

import numpy as np

LABEL = 'Nodule'
METADATA_CSV = 'jsrt_metadata.csv'
META_COLUMNS = ['study_id', 'subtlety', 'size', 'age', 'gender', 'x', 'y', 'state', 'position', 'diagnosis']
N_IMAGES, N_NODULE = 247, 154
SPLIT_SEED, FRACTIONS = 9301, (0.6, 0.2)
PNG_NAME = re.compile(r'JPC(LN|NN)\d{3}\.png')
STATE_OF = {'LN': ('malignant', 'benign'), 'NN': ('non-nodule',)}


def label_of(name):
    '''Tên PNG (hoặc khoá) -> 1 (JPCLN = có nốt) / 0 (JPCNN = không nốt).'''
    stem = name if name.endswith('.png') else name + '.png'
    m = PNG_NAME.fullmatch(stem)
    assert m, f'ten JSRT sai dang: {name!r} -> DUNG'
    return 1 if m.group(1) == 'LN' else 0


def _md5_file(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_metadata(csv_path, md5=None):
    '''jsrt_metadata.csv -> DataFrame. Cổng: md5 (nếu truyền); đúng cột; 247 dòng, tên duy nhất đúng dạng; 154 có nốt; cột state khớp nhãn theo tên.'''
    import pandas as pd
    if md5 is not None:
        assert _md5_file(csv_path) == md5, f'{METADATA_CSV}: md5 KHAC ban ghim -> DUNG'
    df = pd.read_csv(csv_path)
    assert list(df.columns) == META_COLUMNS, f'{METADATA_CSV}: cot la {list(df.columns)} -> DUNG'
    names = df.study_id.astype(str).tolist()
    assert len(names) == N_IMAGES and len(set(names)) == N_IMAGES and all(PNG_NAME.fullmatch(n) for n in names), (
        f'{METADATA_CSV}: {len(names)} dong / ten trung / sai dang -> DUNG')
    bad = [n for n, s in zip(names, df.state.astype(str)) if s not in STATE_OF[PNG_NAME.fullmatch(n).group(1)]]
    assert not bad, f'{METADATA_CSV}: {len(bad)} dong cot state KHONG khop nhan theo ten (vd {bad[:3]}) -> DUNG'
    assert sum(label_of(n) for n in names) == N_NODULE, f'{METADATA_CSV}: so anh co not != {N_NODULE} -> DUNG'
    return df


def split_images(labels, seed=SPLIT_SEED, fractions=FRACTIONS):
    '''{khoá: nhãn 0 / 1} -> {khoá: tách}. MỖI tầng (1 rồi 0) một RandomState(seed) MỚI; thứ tự tầng không đổi kết quả.'''
    out = {}
    for stratum in (1, 0):
        ids = np.array(sorted(k for k, v in labels.items() if int(v) == stratum))
        ids = ids[np.random.RandomState(seed).permutation(len(ids))]
        n_tr, n_va = int(round(fractions[0] * len(ids))), int(round(fractions[1] * len(ids)))
        out.update({k: 'train' for k in ids[:n_tr]})
        out.update({k: 'val' for k in ids[n_tr:n_tr + n_va]})
        out.update({k: 'test' for k in ids[n_tr + n_va:]})
    return out


def table(meta):
    '''meta = read_metadata(...) -> DataFrame(khoa, tach, nhom (= khoá), Nodule, do_kho (subtlety 1–5; 0 = không nốt), trang_thai) theo thứ tự train / val /
       test, khoá sắp tăng trong mỗi tách. do_kho / trang_thai chỉ để mô tả sau (không vào phép học).'''
    import pandas as pd
    keys = [n[:-4] for n in meta.study_id.astype(str)]
    labels = {k: label_of(k) for k in keys}
    sp = split_images(labels)
    sub = pd.to_numeric(meta.subtlety, errors='coerce').fillna(0).astype(int).tolist()
    df = pd.DataFrame(dict(khoa=keys, tach=[sp[k] for k in keys], nhom=keys, **{LABEL: [labels[k] for k in keys]}, do_kho=sub,
                           trang_thai=meta.state.astype(str).tolist()))
    order = {s: i for i, s in enumerate(('train', 'val', 'test'))}
    return df.assign(_o=df.tach.map(order)).sort_values(['_o', 'khoa'], kind='stable').drop(columns='_o').reset_index(drop=True)


def prefer_pinned(files, expected):
    '''Tệp cùng tên: nếu trong các bản có bản trùng mốc (md5, kích thước) -> giữ MỘT bản trùng mốc, bỏ các bản khác nội dung (trả đường dẫn để in); không bản
       nào trùng mốc -> giữ nguyên tất cả (compare_files của 88 sẽ báo xung đột / khác md5 -> dừng). -> (tệp giữ lại, [đường dẫn bản bỏ qua]).'''
    by = {}
    for f in files:
        by.setdefault(f['ten'], []).append(f)
    keep, dropped = [], []
    for n, v in by.items():
        good = [f for f in v if n in expected and (f['md5'], f['size']) == tuple(expected[n])]
        if good and any((f['md5'], f['size']) != (good[0]['md5'], good[0]['size']) for f in v):
            keep.append(good[0])
            dropped += sorted(f['duong'] for f in v if (f['md5'], f['size']) != (good[0]['md5'], good[0]['size']))
        else:
            keep += v
    return keep, sorted(dropped)


def check_jsrt(files):
    '''JSRT trên Drive vs mốc từng tệp (247 PNG + jsrt_metadata.csv) — cùng dạng kết quả với external_benchmarks.integrity.check_shenzhen (dùng được
       integrity.for_log / summary). files = integrity.walk_drive(...). Bản cùng tên khác nội dung khi đã có bản trùng mốc: bỏ qua (prefer_pinned), đếm ở
       trung_ten_bo_qua. -> dict(dat, png, metadata, tong_tep_drive, thua_ngoai_mau, so_thua_ngoai_mau, trung_ten_bo_qua, so_trung_ten_bo_qua).'''
    from external_benchmarks import integrity
    from . import jsrt_manifest as man
    png_files, drop_png = prefer_pinned([f for f in files if PNG_NAME.fullmatch(f['ten'])], man.PNG)
    meta_files, drop_meta = prefer_pinned([f for f in files if f['ten'] == man.META[0]], {man.META[0]: man.META[1:]})
    png = integrity.compare_files(png_files, man.PNG)
    meta = integrity.compare_files(meta_files, {man.META[0]: man.META[1:]})
    other = sorted(f['duong'] for f in files if not PNG_NAME.fullmatch(f['ten']) and f['ten'] != man.META[0])
    dropped = drop_png + drop_meta
    return dict(dat=png['dat'] and meta['dat'], png=png, metadata=meta, tong_tep_drive=len(files), thua_ngoai_mau=other[:20], so_thua_ngoai_mau=len(other),
                trung_ten_bo_qua=dropped[:20], so_trung_ten_bo_qua=len(dropped))
