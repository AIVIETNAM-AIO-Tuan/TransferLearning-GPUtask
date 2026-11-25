# -*- coding: utf-8 -*-
"""CheXpert bản ĐẦY ĐỦ (Drive MyDrive/MIMIC_CXR/CheXpert — Tuấn 24/11) — phần dữ liệu của notebook kiểm kê `93i_kiem_ke_chexpert` (PLAN_93 §3, §10 điểm 6).

Kiểm kê chỉ ĐỌC (Drive API) và in số tổng hợp; bảng đề xuất (tập học + test + tập lớp) là MẶC ĐỊNH của Claude theo PLAN_93 §10 điểm 6, Tuấn sửa được
trước khi dựng kho. Số "theo bài" (CheXpert, Irvin và cs. 2019): train 223.414 ảnh / 64.540 người, valid 234 / 200, test 668 / 500 (= 224.316 / 65.240) —
để đối chiếu, không phải cổng.

- Khoá ảnh = <tách CheXpert>_<patientNNNNN>_<studyN>_<viewN_frontal|lateral> lấy từ ĐUÔI đường dẫn — cùng một hàm cho cột `Path` của CSV, đường dẫn
  trên Drive và tên mục trong zip. Hai bố cục:
    (1) .../train|valid|test/patient…/study…/view….jpg  (cột Path của CSV: 'CheXpert-v1.0/train/…'; cây có thể bọc thêm thư mục)
    (2) .../<thư mục có "(train k)" | "(valid…" | "(test…">/patient…/study…/view….jpg — bản tải từng batch của Stanford AIMI giải nén (Drive thật 24/11:
        'CheXpert-v1.0 batch 2 (train 1)/patient00001/study1/view1_frontal.jpg', KHÔNG có thư mục train/); mục zip không có thư mục tách thì lấy tách
        từ tên zip (`hint` = split_hint(tên zip)).
  0.4.0 chỉ nhận (1) -> 223.414 ảnh train trên Drive thành "không đúng dạng" (lượt 93i 24/11, Event). (1) thử trước nên khoá của CSV không đổi.
  0.4.2: (1) nhận thêm thư mục `val/` = tách `valid` (bộ CheXlocalize của Stanford AIMI: `CheXpert/val/patient…`, `CheXpert/test/patient…` — README
  rajpurkarlab/cheXlocalize); khoá ghi `valid_…` như CheXpert -> val của CheXlocalize so được từng ảnh với valid của CheXpert. Đường dẫn CheXpert
  không có thư mục `val/` nên khoá của 93i không đổi.
  Khoá hợp lệ cho tar phẳng của kho (base._extract: [A-Za-z0-9_.-]+).
- CSV: đọc toàn bộ là chuỗi (ô trống giữ là ''), nhãn: '1' / '1.0' -> 1; '0', '-1' (không chắc), '' (trống) -> 0 (PLAN_93 §10 điểm 6: U và trống tính âm).
  Giá trị khác -> đếm ở `khac`, không đoán.
- Tập học (mặc định): chỉ ảnh THẲNG của CSV train; người sắp tăng -> hoán vị RandomState(SUBSET_SEED) -> lấy lần lượt từng người (mọi ảnh thẳng của người
  đó): người đầu vào val tới khi val >= N_VAL ảnh, rồi vào train tới khi tổng >= N_TOTAL ảnh. Không người nào ở hai tách.
- Test (mặc định): CSV test (bác sĩ gán) nếu có, không thì CSV valid — chỉ ảnh thẳng; nhóm bootstrap = người.
- Tập lớp của luật: lớp có >= MIN_TEST_POS ảnh dương ở test (đặt trước, đếm trên nhãn — không nhìn dự đoán); 5 nhãn cuộc thi báo kèm.
- 0.5.0 (Tuấn 25/11: "(A)" + 5 nhãn): bảng đề xuất / tập lớp ở trên KHÔNG dùng cho kho. Kho dùng table_chexfound (cuối tệp) = cách chia của CheXFound:
  train = mọi ảnh thẳng train.csv, val = ảnh thẳng valid.csv, test = ảnh thẳng test_labels.csv (CheXlocalize); 5 nhãn CHEXFOUND_LABELS; ba CSV md5 ghim."""
import collections
import re

import numpy as np

SOURCE = ('MIMIC_CXR', 'CheXpert')
LABELS = ('No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
          'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices')
COMPETITION = ('Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion')
META_COLUMNS = ('Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA')
PAPER = {'train': (223414, 64540), 'valid': (234, 200), 'test': (668, 500)}       # (ảnh, người) theo bài — đối chiếu
SPLITS_CX = ('train', 'valid', 'test')
IMAGE_RE = re.compile(r'(?:^|/)(train|valid|val|test)/(patient\d+)/(study\d+)/(view\d+_(?:frontal|lateral))\.jpg$')
SPLIT_ALIAS = {'val': 'valid'}                                     # 0.4.2: thư mục val/ của CheXlocalize = tách valid của CheXpert
TAIL_RE = re.compile(r'(?:^|/)(patient\d+)/(study\d+)/(view\d+_(?:frontal|lateral))\.jpg$')
HINT_RE = re.compile(r'\((train|valid|test)', re.I)              # 'CheXpert-v1.0 batch 2 (train 1)' -> train; '… batch 1 (validate & csv)' -> valid
SUBSET_SEED, N_TOTAL, N_VAL = 9302, 15000, 1500
MIN_TEST_POS = 10
ORDER = {'train': 0, 'val': 1, 'test': 2}


def split_hint(name):
    '''Tên thư mục batch / tên zip -> 'train' | 'valid' | 'test' | None (theo "(train …", "(valid…", "(test…").'''
    m = HINT_RE.search(str(name))
    return m.group(1).lower() if m else None


def key_of(path, hint=None):
    '''Đường dẫn (cột Path, đường trên Drive, tên mục zip) -> khoá, hoặc None nếu không đúng dạng ảnh CheXpert. Bố cục (1) trước; không khớp thì (2):
       tách theo thư mục NGAY TRƯỚC patient… (split_hint), không có thì `hint` (tách của cả zip), không có nữa -> None.'''
    p = str(path).replace('\\', '/')
    m = IMAGE_RE.search(p)
    if m:
        s, *rest = m.groups()
        return '_'.join([SPLIT_ALIAS.get(s, s)] + rest)
    m = TAIL_RE.search(p)
    if m is None:
        return None
    s = split_hint(p[:m.start(1)].rstrip('/').rpartition('/')[2]) or hint
    return None if s is None else '_'.join((s,) + m.groups())


def head_of(path):
    '''Phần đường dẫn TRƯỚC patient… (gồm '/' cuối; '' nếu patient ở gốc), hoặc None nếu không có đuôi patient/study/view.jpg.'''
    m = TAIL_RE.search(str(path).replace('\\', '/'))
    return None if m is None else str(path).replace('\\', '/')[:m.start(1)]


def parts(key):
    '''khoá -> (tách CheXpert, người, lần chụp, view).'''
    s, p, st, v = key.split('_', 3)
    return s, p, st, v


def is_frontal(key):
    return key.endswith('_frontal')


def label_value(v):
    '''Ô nhãn (chuỗi) -> '1' | '0' | '-1' | 'trong' | 'khac'.'''
    v = str(v).strip()
    if v == '':
        return 'trong'
    try:
        f = float(v)
    except ValueError:
        return 'khac'
    return {1.0: '1', 0.0: '0', -1.0: '-1'}.get(f, 'khac')


def binarize(col):
    '''Cột nhãn (chuỗi) -> int8: 1 khi giá trị 1; 0 khi 0 / -1 / trống / khác (khác đã đếm ở describe).'''
    return np.array([1 if label_value(v) == '1' else 0 for v in col], dtype=np.int8)


def read_csv(path):
    import pandas as pd
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def describe(df):
    '''Số tổng hợp của một CSV CheXpert (không khoá ảnh). -> dict.'''
    cols = list(df.columns)
    out = dict(dong=int(len(df)), cot=cols, co_path='Path' in cols, nhan_co=[c for c in LABELS if c in cols], nhan_thieu=[c for c in LABELS if c not in cols])
    if 'Path' not in cols:
        return out
    keys = [key_of(p) for p in df.Path]
    ok = [k for k in keys if k]
    out.update(path_sai_dang=int(len(keys) - len(ok)), tach_trong_path=dict(collections.Counter(parts(k)[0] for k in ok)),
               nguoi=len({parts(k)[1] for k in ok}), lan_chup=len({parts(k)[1:3] for k in ok}), khoa_trung=int(len(ok) - len(set(ok))),
               thang=int(sum(is_frontal(k) for k in ok)), nghieng=int(sum(not is_frontal(k) for k in ok)))
    if 'Frontal/Lateral' in cols:
        fl = [str(v) for v, k in zip(df['Frontal/Lateral'], keys) if k]
        out['cot_thang_nghieng'] = dict(collections.Counter(fl))
        out['cot_lech_ten_tep'] = int(sum((v == 'Frontal') != is_frontal(k) for v, k in zip(fl, ok)))
    if 'AP/PA' in cols:
        out['ap_pa_anh_thang'] = dict(collections.Counter(str(v) for v, k in zip(df['AP/PA'], keys) if k and is_frontal(k)))
    out['gia_tri_nhan'] = {c: dict(collections.Counter(label_value(v) for v in df[c])) for c in out['nhan_co']}
    return out


def role_of(desc, share=0.99):
    '''Vai của CSV theo tách trong cột Path: 'train' / 'valid' / 'test' khi >= 99 % dòng hợp lệ cùng một tách và đủ 14 cột nhãn; không thì None.'''
    if not desc.get('co_path') or desc['nhan_thieu']:
        return None
    t = desc.get('tach_trong_path') or {}
    n = sum(t.values())
    if not n:
        return None
    s, c = max(t.items(), key=lambda x: x[1])
    return s if c >= share * n else None


PREFERRED = {'train': ('train.csv',), 'valid': ('valid.csv',), 'test': ('test_labels.csv', 'test.csv', 'groundtruth.csv')}


def pick(cands):
    '''cands: [dict(ten, duong, vai, dong, ...)] -> {vai: bản chọn}. Ưu tiên tên chuẩn (PREFERRED), rồi đường ngắn nhất; các bản khác cùng vai in ra để
       Tuấn / Claude xem (vd train_cheXbert.csv).'''
    out = {}
    for role in SPLITS_CX:
        c = [x for x in cands if x.get('vai') == role]
        if c:
            pref = PREFERRED[role]
            out[role] = sorted(c, key=lambda x: (pref.index(x['ten']) if x['ten'] in pref else len(pref), len(x['duong']), x['duong']))[0]
    return out


def frontal_rows(df):
    '''-> (khoá, người) của ảnh THẲNG (theo tên tệp) có khoá hợp lệ, theo thứ tự CSV; khoá trùng giữ lần đầu.'''
    seen, rows = set(), []
    for i, p in enumerate(df.Path):
        k = key_of(p)
        if k and is_frontal(k) and k not in seen:
            seen.add(k)
            rows.append((i, k, parts(k)[1]))
    return rows


def select_people(df, seed=None, n_total=None, n_val=None):
    '''Tập học mặc định từ CSV train (None -> hằng của module LÚC GỌI). -> {người: 'train' | 'val'}.'''
    seed = SUBSET_SEED if seed is None else seed
    n_total = N_TOTAL if n_total is None else n_total
    n_val = N_VAL if n_val is None else n_val
    count = collections.Counter(p for _, _, p in frontal_rows(df))
    people = np.array(sorted(count))
    order = people[np.random.RandomState(seed).permutation(len(people))]
    out, cum = {}, 0
    for p in order:
        if cum >= n_total:
            break
        out[str(p)] = 'val' if cum < n_val else 'train'
        cum += count[p]
    return out


def table(train_df, test_df, test_role, seed=None, n_total=None, n_val=None):
    '''Bảng đề xuất: DataFrame(khoa, tach, nhom (= người), 14 nhãn, ap_pa, nguon) — train / val từ CSV train, test từ test_df (vai test_role), chỉ ảnh thẳng;
       thứ tự train, val, test, khoá sắp tăng trong mỗi tách.'''
    import pandas as pd
    who = select_people(train_df, seed, n_total, n_val)
    parts_ = []
    for df, role, split_of in ((train_df, 'train', lambda p: who.get(p)), (test_df, test_role, lambda p: 'test')):
        rows = [(i, k, p, split_of(p)) for i, k, p in frontal_rows(df) if split_of(p)]
        idx = [r[0] for r in rows]
        d = pd.DataFrame(dict(khoa=[r[1] for r in rows], tach=[r[3] for r in rows], nhom=[r[2] for r in rows]))
        for c in LABELS:
            d[c] = binarize(df[c].iloc[idx].tolist()) if rows else np.zeros(0, np.int8)
        d['ap_pa'] = df['AP/PA'].iloc[idx].astype(str).tolist() if 'AP/PA' in df.columns else ''
        d['nguon'] = role
        parts_.append(d)
    t = pd.concat(parts_, ignore_index=True)
    return t.assign(_o=t.tach.map(ORDER)).sort_values(['_o', 'khoa'], kind='stable').drop(columns='_o').reset_index(drop=True)


def positives(t):
    '''-> {tách: {nhãn: số ảnh dương}}.'''
    return {s: {c: int(t.loc[t.tach == s, c].sum()) for c in LABELS} for s in ORDER}


def class_set(t, min_pos=None):
    '''Tập lớp của luật: nhãn có >= min_pos ảnh dương ở test (thứ tự LABELS; None -> MIN_TEST_POS lúc gọi).'''
    min_pos = MIN_TEST_POS if min_pos is None else min_pos
    pos = positives(t)['test']
    return [c for c in LABELS if pos[c] >= min_pos]


# ---------------- 0.5.0: cách chia của CheXFound (Tuấn 25/11: "(A)" + 5 nhãn) ----------------
# CheXFound-main/chexfound/data/datasets/chexpert.py (đọc 25/11): đọc <tách>.csv, bỏ dòng có 'lateral' trong Path, fillna(0), giữ 5 cột, replace(-1, 0);
# số dòng ghi cứng train 191.027 / val 202 / test 518. Ở đây: train = train.csv, val = valid.csv, test = test_labels.csv của CheXlocalize (bộ test chính thức;
# 93h tải, 93j kiểm nhãn = groundtruth.csv GitHub). Lọc ảnh thẳng theo khoá (view…_frontal) VÀ đếm lại theo luật của CheXFound (Path không chứa 'lateral') —
# hai cách phải ra cùng số dòng, không thì dừng.
CHEXFOUND_LABELS = ('Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion')    # thứ tự cột của CheXFound
CHEXFOUND_SIZES = {'train': 191027, 'val': 202, 'test': 518}
# vai CSV -> (tên tệp, md5, số dòng): ĐO trên Drive — train / valid ở output 93i 0.4.1 (25/11), test ở output 93j lượt 2 (25/11). valid.csv có md5 TRÙNG
# val_labels.csv của CheXlocalize (93j).
CSV_PINS = {'train': ('train.csv', 'd8e53af9a37651cea91a569c3c138558', 223414),
            'valid': ('valid.csv', '7e8ccffcffa96a2b90e09c7b9335d4de', 234),
            'test': ('test_labels.csv', '6fd4befe8d38a7da732c96e62b320f82', 668)}
CSV_DIR = {'train': 'CheXpert-v1.0 batch 1 (validate & csv)', 'valid': 'CheXpert-v1.0 batch 1 (validate & csv)', 'test': 'CheXpert'}   # thư mục chứa CSV
TEST_POSITIVES = (154, 79, 29, 158, 105)       # ảnh thẳng test dương theo CHEXFOUND_LABELS (output 93j lượt 2: Cardiomegaly / Edema / Consolidation / Atelectasis / Pleural Effusion)
TEST_KEY_MD5_PREFIX = '196ee1bc9cca'           # md5 khoá ảnh thẳng test (khoá sắp tăng, base.md5_keys) — output 93j lượt 2
SPLIT_OF_ROLE = {'train': 'train', 'valid': 'val', 'test': 'test'}


def chexfound_frontal_count(df):
    '''Số dòng CheXFound giữ: Path KHÔNG chứa 'lateral' (đúng biểu thức của mã CheXFound).'''
    return int((~df['Path'].astype(str).str.contains('lateral')).sum())


def table_chexfound(frames, labels=CHEXFOUND_LABELS):
    '''frames = {'train': train.csv, 'valid': valid.csv, 'test': test_labels.csv} (DataFrame chuỗi, read_csv) -> bảng kho: khoa, tach (train / val / test),
       nhom (= người), 5 nhãn (int8: 1 khi giá trị 1; 0 / -1 / trống -> 0 như CheXFound), ap_pa, nguon (vai CSV); thứ tự train, val, test, khoá sắp tăng
       trong mỗi tách. Dừng khi: Path sai dạng, khoá trùng, số ảnh thẳng theo khoá != số dòng CheXFound giữ, nhãn có giá trị lạ (không thuộc 1 / 0 / -1 /
       trống — CheXFound giữ nguyên giá trị lạ, ở đây không đoán).'''
    import pandas as pd
    parts_, info = [], {}
    for role in ('train', 'valid', 'test'):
        df = frames[role]
        miss = [c for c in ('Path',) + tuple(labels) if c not in df.columns]
        assert not miss, f'{role}: CSV thieu cot {miss} -> DUNG'
        keys = [key_of(p) for p in df.Path]
        bad = sum(k is None for k in keys)
        assert bad == 0, f'{role}: {bad} dong Path sai dang -> DUNG'
        assert len(set(keys)) == len(keys), f'{role}: khoa trung trong CSV -> DUNG'
        assert all(parts(k)[0] == role for k in keys), f'{role}: Path chua tach khac {role} -> DUNG'
        rows = frontal_rows(df)
        n_cf = chexfound_frontal_count(df)
        assert len(rows) == n_cf, f'{role}: anh thang theo khoa {len(rows)} != so dong CheXFound giu (Path khong chua lateral) {n_cf} -> DUNG'
        idx = [r[0] for r in rows]
        odd = {c: int(sum(label_value(v) == 'khac' for v in df[c].iloc[idx])) for c in labels}
        assert not any(odd.values()), f'{role}: nhan co gia tri la {odd} -> DUNG'
        d = pd.DataFrame(dict(khoa=[r[1] for r in rows], tach=SPLIT_OF_ROLE[role], nhom=[r[2] for r in rows]))
        for c in labels:
            d[c] = binarize(df[c].iloc[idx].tolist())
        d['ap_pa'] = df['AP/PA'].iloc[idx].astype(str).tolist() if 'AP/PA' in df.columns else ''
        d['nguon'] = role
        parts_.append(d)
        info[role] = dict(dong=int(len(df)), thang=len(rows), chexfound_giu=n_cf, nguoi=int(d.nhom.nunique()),
                          gia_tri=dict((c, dict(collections.Counter(label_value(v) for v in df[c].iloc[idx]))) for c in labels))
    t = pd.concat(parts_, ignore_index=True)
    t = t.assign(_o=t.tach.map(ORDER)).sort_values(['_o', 'khoa'], kind='stable').drop(columns='_o').reset_index(drop=True)
    return t, info
