# -*- coding: utf-8 -*-
"""CheXlocalize (Stanford AIMI; Tuấn 25/11: MyDrive/MIMIC_CXR/CheXlocalize) — nguồn BỘ TEST CHÍNH THỨC của CheXpert cho cách chia theo CheXFound
(PLAN_93 §6 phiên 5; Event 2026-11-25 "CheXpert: Tuấn muốn chia theo CheXFound …"). Phần dữ liệu của notebook kiểm kê `93j_kiem_ke_chexlocalize` (chỉ đọc).

- Gói CheXpert của AIMI chỉ có train + valid (test giữ kín cho bảng xếp hạng — datasheet CheXpert, arXiv 2105.03020); ảnh test nằm trong CheXlocalize:
  `CheXpert/val/`, `CheXpert/val_labels.csv`, `CheXpert/test/`, `CheXpert/test_labels.csv` (README rajpurkarlab/cheXlocalize). Khoá ảnh = khoá CheXpert
  (datasets.chexpert.key_of; thư mục `val/` = tách `valid`), nên val của CheXlocalize so được từng ảnh với valid của MyDrive/MIMIC_CXR/CheXpert.
- Nhãn test tham chiếu: `groundtruth.csv` của rajpurkarlab/cheXpert-test-set-labels (đa số 5 bác sĩ, theo LẦN CHỤP, 14 nhãn 0 / 1), tải ở commit ghim và so
  md5 ghim (tải + đếm trên máy 25/11: 500 dòng = 500 lần chụp / 500 người, 33.692 byte). Ảnh test mang nhãn của lần chụp chứa nó.
- Số đối chiếu (không phải cổng cứng trừ khi ghi ở phán quyết): bài CheXpert — test 668 ảnh / 500 người / 500 lần chụp, valid 234 / 200; CheXFound
  (arXiv 2502.05142v2 mục IV-B + mã chexfound/data/datasets/chexpert.py) — 191.027 / 202 / 518 ảnh thẳng. 518 chưa kiểm trước 93j.
- Không in khoá ảnh; số đếm + khoảng mã người."""
import collections
import io
import re

import numpy as np

from . import chexpert as CX

SOURCE = ('MIMIC_CXR', 'CheXlocalize')
CHEXPERT_B1 = 'CheXpert-v1.0 batch 1 (validate & csv)'          # thư mục chứa train.csv / valid.csv / valid/ trong MyDrive/MIMIC_CXR/CheXpert (93i, 25/11)
PAPER = {'test': dict(anh=668, nguoi=500, lan_chup=500), 'valid': dict(anh=234, nguoi=200)}
CHEXFOUND = dict(train=191027, val=202, test=518)                  # ảnh thẳng
GT_REPO = 'rajpurkarlab/cheXpert-test-set-labels'
GT_SHA = 'a03eb3438ca55c28eb4379dd0dd09e3b38c0c414'                # đầu main 25/11; groundtruth.csv không đổi từ a930205df292 (21/10/2022)
GT_URL = f'https://raw.githubusercontent.com/{GT_REPO}/{GT_SHA}/groundtruth.csv'
GT_MD5 = 'f783258db67f6757003bfe8dfb0c376c'
# 0.4.3 — nguồn ảnh val / test (notebook 93h): bản Redivis của CheXlocalize (trang AIMI hiện tại) CHỈ có thư mục CheXlocalize/ (= bản trên Drive, 93j 25/11);
# thư mục CheXpert/ (val/, test/, val_labels.csv, test_labels.csv) chỉ còn ở trang AIMI CŨ trên Azure (ảnh chụp màn hình Tuấn 25/11: mức gốc CheXpert +
# CheXlocalize, 5,73 GB, 3.256 tệp) — nút Download cấp link SAS cho cả container.
AIMI_OLD = 'https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c'
AZ_PREFIX = 'CheXpert/'                                            # = tên thư mục con tạo trong MyDrive/MIMIC_CXR/CheXlocalize (giữ nguyên cấu trúc)
AZ_CSV = ('val_labels.csv', 'test_labels.csv')
AZ_FILES = 3256                                                    # "File count" của trang cũ — chỉ in để so, không phải cổng
STUDY_RE = re.compile(r'(?:^|/)test/(patient\d+)/(study\d+)/?$')
SPLIT_NAME = {'test': 'test', 'valid': 'val'}                      # tách CheXpert -> tách của bảng (config.SPLITS)


def fetch(url=None, timeout=60):
    '''Tải groundtruth.csv (commit ghim) -> bytes. UA chung, không thông tin người dùng.'''
    import urllib.request
    req = urllib.request.Request(url or GT_URL, headers={'User-Agent': 'crossdata-bench'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def read_groundtruth(data):
    '''bytes của groundtruth.csv -> dict(lan_chup {(người, lần chụp): {nhãn: '1' | '0' | '-1' | 'trong' | 'khac'}}, dong, sai_dang, trung, thieu_cot).'''
    import pandas as pd
    df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    miss = [c for c in ('Study',) + CX.LABELS if c not in df.columns]
    out, bad, dup = {}, 0, 0
    if not miss:
        for i, s in enumerate(df.Study):
            m = STUDY_RE.search(str(s).replace('\\', '/'))
            if m is None:
                bad += 1
                continue
            dup += m.groups() in out
            out[m.groups()] = {c: CX.label_value(df[c].iat[i]) for c in CX.LABELS}
    return dict(lan_chup=out, dong=int(len(df)), sai_dang=bad, trung=dup, thieu_cot=miss)


def _rows(df, split):
    '''-> [(chỉ số dòng, khoá)] của dòng có khoá thuộc tách `split`; khoá trùng giữ lần đầu.'''
    seen, out = set(), []
    for i, p in enumerate(df.Path):
        k = CX.key_of(p)
        if k and CX.parts(k)[0] == split and k not in seen:
            seen.add(k)
            out.append((i, k))
    return out


def study_of(key):
    _, p, st, _ = CX.parts(key)
    return p, st


def counts(df, split):
    '''Số ảnh / người / lần chụp / ảnh thẳng của tách `split` trong CSV (khoá không trùng).'''
    ks = [k for _, k in _rows(df, split)]
    return dict(anh=len(ks), nguoi=len({CX.parts(k)[1] for k in ks}), lan_chup=len({study_of(k) for k in ks}), thang=int(sum(CX.is_frontal(k) for k in ks)))


def compare_test(df, gt):
    '''CSV test (CheXlocalize) <-> groundtruth.csv theo LẦN CHỤP, 14 nhãn (giá trị chuẩn hoá: '1.0' = '1'). -> dict số đếm (không khoá ảnh).'''
    rows, g = _rows(df, 'test'), gt['lan_chup']
    labs = [c for c in CX.LABELS if c in df.columns]
    st_csv = {study_of(k) for _, k in rows}
    lech, khac, n_lech, n_ref = collections.Counter(), collections.Counter(), 0, 0
    for i, k in rows:
        vals = {c: CX.label_value(df[c].iat[i]) for c in labs}
        khac.update(c for c, v in vals.items() if v not in ('0', '1'))
        ref = g.get(study_of(k))
        if ref is None:
            continue
        n_ref += 1
        bad = [c for c in labs if vals[c] != ref[c]]
        n_lech += bool(bad)
        lech.update(bad)
    return dict(anh=len(rows), anh_co_tham_chieu=n_ref, lan_chup_csv=len(st_csv), lan_chup_gt=len(g), chi_o_csv=len(st_csv - set(g)), chi_o_gt=len(set(g) - st_csv),
                anh_lech=n_lech, lech_theo_nhan=dict(lech), gia_tri_khac_0_1=dict(khac), thieu_cot=[c for c in CX.LABELS if c not in labs])


def compare_val(lx_df, cx_df):
    '''val_labels.csv (CheXlocalize) <-> valid.csv (CheXpert) theo khoá ảnh, nhãn có ở cả hai (giá trị chuẩn hoá). -> dict số đếm.'''
    a, b = dict((k, i) for i, k in _rows(lx_df, 'valid')), dict((k, i) for i, k in _rows(cx_df, 'valid'))
    labs = [c for c in CX.LABELS if c in lx_df.columns and c in cx_df.columns]
    common = sorted(set(a) & set(b))
    lech, n_lech = collections.Counter(), 0
    for k in common:
        bad = [c for c in labs if CX.label_value(lx_df[c].iat[a[k]]) != CX.label_value(cx_df[c].iat[b[k]])]
        n_lech += bool(bad)
        lech.update(bad)
    return dict(chung=len(common), chi_o_chexlocalize=len(set(a) - set(b)), chi_o_chexpert=len(set(b) - set(a)), anh_lech=n_lech, lech_theo_nhan=dict(lech),
                nhan_so=len(labs))


def patients(df):
    '''Mã người (patientNNNNN) của mọi dòng có khoá hợp lệ.'''
    return {CX.parts(k)[1] for k in (CX.key_of(p) for p in df.Path) if k}


def image_table(df, split, anh):
    '''Bảng ảnh của một CSV, MỌI view (thẳng + nghiêng; cột `thang`): khoa, tach ('test' | 'val'), nhom (= người, đơn vị bootstrap), lan_chup, thang,
       14 nhãn (int8: 1 khi giá trị 1, còn lại 0 — test chính thức chỉ có 0 / 1, phán quyết kiểm), duong (từ gốc CheXlocalize), byte, md5 (Drive).'''
    import pandas as pd
    rows = _rows(df, split)
    idx = [i for i, _ in rows]
    ks = [k for _, k in rows]
    t = pd.DataFrame(dict(khoa=ks, tach=SPLIT_NAME[split], nhom=[CX.parts(k)[1] for k in ks], lan_chup=['_'.join(study_of(k)) for k in ks],
                          thang=[CX.is_frontal(k) for k in ks]))
    for c in CX.LABELS:
        t[c] = CX.binarize(df[c].iloc[idx].tolist()) if (rows and c in df.columns) else np.zeros(len(rows), np.int8)
    t['duong'] = [anh[k]['duong'] if k in anh else None for k in ks]
    t['byte'] = [anh[k]['size'] if k in anh else -1 for k in ks]
    t['md5'] = [anh[k]['md5'] if k in anh else None for k in ks]
    return t.sort_values('khoa', kind='stable').reset_index(drop=True)


def positives(t, frontal_only=True):
    '''-> {nhãn: số ảnh dương} (mặc định chỉ ảnh thẳng — như CheXFound).'''
    s = t[t.thang] if frontal_only else t
    return {c: int(s[c].sum()) for c in CX.LABELS}
