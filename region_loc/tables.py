# -*- coding: utf-8 -*-
"""Bảng hộp của hai phép thử + split CXR-LT.

Luật dữ liệu (PLAN_90 §11.2, luật chỉ-ảnh + ngoại lệ 17/11):
  - chỉ đọc toạ độ hộp + tên vùng / tên dấu hiệu; cột câu report của MS-CXR (`label_text`) KHÔNG BAO GIỜ được nạp (read_mscxr dùng usecols);
  - hàng của bệnh nhân val / test chỉ được dùng tới mức ĐẾM (cổng số đếm ghim) cho tới khi luật đọc khoá: bên gọi gọi keep(df, ('train',)) ngay sau
    cổng số đếm, trước mọi phép tính trên hộp.

`doc_ds_vung` chép NGUYÊN VĂN từ experiments/78_chi_anh/khoi/khoi_vung.py (thứ tự vùng của luồng = thứ tự tệp semantics của ImaGenome, như 78f)."""
import numpy as np
import pandas as pd

from . import config as C
from . import geometry as G

_BOX = ['original_x1', 'original_y1', 'original_x2', 'original_y2']
_BOX224 = ['x1', 'y1', 'x2', 'y2']


# ==== NGUYÊN VĂN khoi_vung.py (78) ====
def doc_ds_vung(txt):
    '''semantics/objects_detectable_by_bbox_pipeline_v1.txt -> danh sach ten vung (giu thu tu tep; moi dong "ten,").'''
    return [l.strip().rstrip(',').strip() for l in txt.splitlines() if l.strip()]


# ==== của gói ====
def region_names(path):
    names = doc_ds_vung(open(path, encoding='utf-8').read())
    assert len(names) == C.N_REGIONS and len(set(names)) == C.N_REGIONS, f'danh sach vung: {len(names)} ten (can {C.N_REGIONS}, khong trung) -> DUNG'
    return names


def _dicom(s):
    return s.astype(str).str.replace('.dcm', '', regex=False)


def read_gold(path):
    '''Tệp hộp gold ImaGenome -> dicom_id, bbox_name, hộp ảnh gốc (original_*), hộp khung 224 (x1..y2). Chỉ các cột hộp.'''
    df = pd.read_csv(path, usecols=['image_id', 'bbox_name'] + _BOX224 + _BOX)
    df.insert(0, 'dicom_id', _dicom(df.pop('image_id')))
    assert not df.duplicated(['dicom_id', 'bbox_name']).any(), 'gold: trung (anh, vung) -> DUNG'
    return df


def read_auto(path):
    '''Hộp TỰ ĐỘNG của ImaGenome cho đúng 1.000 ảnh gold (tệp tab; không có câu) -> dicom_id, bbox_name, hộp ảnh gốc.'''
    df = pd.read_csv(path, sep='\t', usecols=['image_id', 'bbox_name'] + _BOX)
    df.insert(0, 'dicom_id', _dicom(df.pop('image_id')))
    assert not df.duplicated(['dicom_id', 'bbox_name']).any(), 'hop tu dong: trung (anh, vung) -> DUNG'
    return df


def read_mscxr(path):
    '''Bảng MS-CXR, CHỈ các cột của config.MSCXR_COLUMNS: cột câu report không được pandas phân tích, không vào bộ nhớ của bảng.'''
    assert C.MSCXR_TEXT_COLUMN not in C.MSCXR_COLUMNS
    df = pd.read_csv(path, usecols=list(C.MSCXR_COLUMNS))
    assert C.MSCXR_TEXT_COLUMN not in df.columns and list(df.columns) == list(C.MSCXR_COLUMNS), list(df.columns)
    df['dicom_id'] = df.dicom_id.astype(str)
    la = sorted(set(df.category_name) - set(C.FINDINGS))
    assert not la, f'MS-CXR: ten dau hieu la {la} -> DUNG'
    assert bool(((df.w > 0) & (df.h > 0)).all()), 'MS-CXR: hop co canh <= 0 -> DUNG'
    return df


def read_sizes(path, ids):
    '''Metadata MIMIC-CXR-JPG -> DataFrame chỉ mục dicom_id, cột W (Columns), H (Rows) cho các ảnh `ids` (như 78f C3). Thiếu ảnh -> dừng.'''
    m = pd.read_csv(path, usecols=['dicom_id', 'Rows', 'Columns'])
    m = m[m.dicom_id.isin(set(ids))].drop_duplicates('dicom_id').set_index('dicom_id')
    assert len(m) == len(set(ids)), f'metadata thieu {len(set(ids)) - len(m)} anh -> DUNG'
    return pd.DataFrame({'W': m.Columns.astype('int64'), 'H': m.Rows.astype('int64')})


def read_order(path):
    '''Bảng ảnh của các notebook 78 (emb_index.parquet) -> dicom_id, subject_id, study_id, split.'''
    o = pd.read_parquet(path, columns=['dicom_id', 'subject_id', 'study_id', 'split'])
    o['dicom_id'] = o.dicom_id.astype(str)
    assert not o.dicom_id.duplicated().any() and set(C.SPLITS) <= set(o.split.unique()), (
        f'emb_index: dicom_id trung hoac thieu split {C.SPLITS} (co {sorted(o.split.unique())}) -> DUNG')
    return o


def attach_split(df, order, what):
    '''Gắn subject_id, study_id, split theo dicom_id. Ảnh không có trong bảng / bệnh nhân ở hai split -> dừng.'''
    o = order.set_index('dicom_id')[['subject_id', 'study_id', 'split']]
    out = df.join(o, on='dicom_id')
    assert out.split.notna().all(), f'{what}: {int(out.split.isna().sum())} hang khong co trong emb_index -> DUNG'
    assert int(out.groupby('subject_id').split.nunique().max()) == 1, f'{what}: benh nhan nam o hai split -> DUNG'
    return out


def counts(df):
    '''{split: (số ảnh, số bệnh nhân, số hàng)} — chỉ đếm, không đụng hộp.'''
    return {s: (int(g.dicom_id.nunique()), int(g.subject_id.nunique()), int(len(g))) for s, g in df.groupby('split')}


def check_counts(df, pinned, what):
    got = counts(df)
    assert got == {k: tuple(v) for k, v in pinned.items()}, f'{what}: so dem theo split {got} KHAC so ghim {pinned} -> sai tep / sai split -> DUNG'
    return got


def keep(df, splits):
    '''Chỉ giữ hàng của `splits` (bản sao mới). Gọi NGAY sau cổng số đếm.'''
    return df[df.split.isin(tuple(splits))].reset_index(drop=True).copy()


def gold_frame(gold, sizes, names):
    '''gold (đã keep) + kích thước ảnh -> thêm k (chỉ số vùng), g1..g4 (hộp chuẩn hoá khung mô hình). Cổng: mọi tên vùng thuộc danh sách; toạ độ 224 của
       tệp khớp letterbox 224 của (hộp gốc, W, H) — sai W / H hoặc sai hình học thì trượt.'''
    idx = {v: i for i, v in enumerate(names)}
    la = sorted(set(gold.bbox_name) - set(idx))
    assert not la, f'gold: ten vung khong co trong danh sach 36 vung {la} -> DUNG'
    g = gold.join(sizes, on='dicom_id')
    assert g.W.notna().all(), 'gold: thieu kich thuoc anh -> DUNG'
    err = G.letterbox_224_error(g.original_x1, g.original_y1, g.original_x2, g.original_y2, g.W, g.H, g[_BOX224].to_numpy())
    share, med = float((err <= C.TOL_224_PX).mean()), float(np.median(err))
    st = dict(share_224=share, err_224_median=med, err_224_p99=float(np.quantile(err, 0.99)), err_224_max=float(err.max()),
              share_1px=float((err <= 1.0).mean()), n_hop_lech=int((err > C.TOL_224_PX).sum()),
              n_anh_lech=int(g.dicom_id[err > C.TOL_224_PX].nunique()), n_anh_lech_1px=int(g.dicom_id[err > 1.0].nunique()), n_anh=int(g.dicom_id.nunique()))
    assert share >= C.MIN_SHARE_224 and med <= C.MAX_MEDIAN_224_PX, (
        f'gold: toa do 224 KHONG khop letterbox 224 cua (hop goc, W, H metadata): {share:.2%} hop lech <= {C.TOL_224_PX} px (can {C.MIN_SHARE_224:.0%}), '
        f"trung vi {med:.3f} px (can <= {C.MAX_MEDIAN_224_PX}); p99 {st['err_224_p99']:.2f}, lon nhat {st['err_224_max']:.2f} px; {st['n_anh_lech']} / "
        f"{st['n_anh']} anh co hop lech -> sai W / H hoac hinh hoc letterbox -> DUNG")
    f = G.to_frame(g.original_x1, g.original_y1, g.original_x2, g.original_y2, g.W, g.H)
    g = g.assign(k=g.bbox_name.map(idx).astype('int64'), g1=f[:, 0], g2=f[:, 1], g3=f[:, 2], g4=f[:, 3])
    return g, st


def auto_frame(auto, gold_framed, names):
    '''Hộp tự động (đã keep) -> cùng khung, ghép theo (ảnh, vùng) với gold đã có W / H. -> DataFrame (dicom_id, k, a1..a4).'''
    idx = {v: i for i, v in enumerate(names)}
    wh = gold_framed.drop_duplicates('dicom_id').set_index('dicom_id')[['W', 'H']]
    a = auto[auto.bbox_name.isin(idx)].join(wh, on='dicom_id')
    a = a[a.W.notna()]
    f = G.to_frame(a.original_x1, a.original_y1, a.original_x2, a.original_y2, a.W, a.H)
    return pd.DataFrame(dict(dicom_id=a.dicom_id.to_numpy(), k=a.bbox_name.map(idx).to_numpy('int64'), a1=f[:, 0], a2=f[:, 1], a3=f[:, 2], a4=f[:, 3]))


def mscxr_frame(ms):
    '''MS-CXR (đã keep) -> thêm j (chỉ số nhãn trong LAB40), b1..b4 (hộp chuẩn hoá khung mô hình). Hộp vượt biên ảnh > 2 px -> dừng.'''
    out = bool(((ms.x < -2) | (ms.y < -2) | (ms.x + ms.w > ms.image_width + 2) | (ms.y + ms.h > ms.image_height + 2)).any())
    assert not out, 'MS-CXR: co hop vuot bien anh > 2 px -> sai don vi toa do -> DUNG'
    f = G.to_frame(ms.x, ms.y, ms.x + ms.w, ms.y + ms.h, ms.image_width, ms.image_height)
    return ms.assign(j=ms.category_name.map({n: C.LAB40.index(n) for n in C.FINDINGS}).astype('int64'), b1=f[:, 0], b2=f[:, 1], b3=f[:, 2], b4=f[:, 3])
