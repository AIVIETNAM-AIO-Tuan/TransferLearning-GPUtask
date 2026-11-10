"""Backbone B của một seed (biến thể Bd của gói nhanh_c, module 84): Phần 3 = dựng lại backbone + luồng vùng đông cứng từ Phần 1 / 2 đã chạy,
gộp trọng số đầu MIMIC ep1–10 và cổng tái lập C6S (so dự đoán đã lưu của lượt gốc). KHÔNG viết lại: 12 bước đọc nguyên từ gói nhanh_c ghim
commit, sha256 = bảng BAM (như NC.buoc), chạy trong namespace notebook.

Khác NC.buoc DUY NHẤT ở bước bd3_C0: thay đúng 3 dòng RUN_TAG / RES57 / DRV_DUONG để kết quả ghi vào thư mục của 87c. Khối ghi Drive trong
bd3_C0 dò ID thư mục ghi NGAY LÚC KHỞI TẠO (DRV_ID, DRV_TAT_CA = _dich(DRV_DUONG)) -> không thể chạy nguyên bước rồi đặt lại DRV_DUONG.

Part3View: các tên của namespace Phần 3 mà phần sau (P1, trích token, học đầu) đọc — khai báo một chỗ, thiếu tên -> dừng; nối / trả hàng ảnh
ngoài vào các mảng theo hàng ORDER mà predict đọc (= _noi_hang_g / _tra_lai_g của 82c)."""
import contextlib
import hashlib
import os
import re

import numpy as np
import pandas as pd
import torch

from . import config

# dòng gốc của bd3_C0 (regex cả dòng) -> dòng thay (format run_tag, folder)
OUTPUT_LINES = (
    (r"^RUN_TAG = CAU_HINH_C\['run_gts'\].*$",
     "RUN_TAG = '{run_tag}'   # 87c (b_robustness.backbone_b): P1 + trich token VinDr + hoc lai dau, KHONG train; thu muc RIENG"),
    (r"^RES57 = os\.path\.join\(OUT_DIR, CAU_HINH_C\['tm'\], RUN_TAG\).*$",
     "RES57 = os.path.join(OUT_DIR, '{folder}', RUN_TAG)   # CHI de in; GHI qua khoi chuan (cuoi cell)"),
    (r"^DRV_DUONG = \(CAU_HINH_C\['tm'\], RUN_TAG\).*$",
     "DRV_DUONG = ('{folder}', RUN_TAG)   # 87c: thu muc CUA NO (KHONG ghi stage4_nb84); nguon doc van o stage4_nb84 qua CAU_HINH_C"),
)


def step_source(nhanh_c, name):
    '''Mã bước `name` của gói nhanh_c đã nạp (LF) + đường dẫn; sha256 KHÁC bảng BAM -> RuntimeError (như NC.buoc).'''
    p = os.path.join(os.path.dirname(os.path.abspath(nhanh_c.__file__)), 'buoc', name + '.py')
    with open(p, 'rb') as f:
        b = f.read().replace(b'\r\n', b'\n')
    if hashlib.sha256(b).hexdigest() != nhanh_c.BAM[name]:
        raise RuntimeError(f'nhanh_c: {p} KHAC ban sinh (sha256) -> tai lai goi dung commit ghim')
    return b.decode('utf-8'), p


def redirect_output(code, run_tag, folder):
    '''Thay đúng 3 dòng ghi của bd3_C0; mỗi dòng gốc phải khớp ĐÚNG MỘT lần, không thì dừng.'''
    for rx, new in OUTPUT_LINES:
        m = list(re.finditer(rx, code, flags=re.M))
        if len(m) != 1:
            raise RuntimeError(f'bd3_C0: dong {rx!r} khop {len(m)} lan (can 1) -> goi nhanh_c KHAC ban da kiem. DUNG')
        code = code[:m[0].start()] + new.format(run_tag=run_tag, folder=folder) + code[m[0].end():]
    return code


def part3_code(nhanh_c, seed, name):
    '''Mã sẽ chạy cho bước `name` của seed (bd3_C0: đã thay 3 dòng ghi).'''
    code, p = step_source(nhanh_c, name)
    if name == 'bd3_C0':
        code = redirect_output(code, config.RUN_K3[seed], config.FOLDER_K3)
    return code, p


def run_part3(ns, nhanh_c, seed, start=None, log=print):
    '''Chạy 12 bước Phần 3 biến thể Bd cho `seed` trong namespace `ns` (globals() của notebook) — như các ô Phần 3 của notebook gói 84.
       start = tên bước để chạy lại từ đó (sau khi một bước lỗi giữa chừng, cùng phiên).'''
    assert seed in config.K3_SEEDS, seed
    assert isinstance(ns, dict), 'ns phai la globals() cua notebook'
    steps = list(config.PART3_STEPS)
    assert list(nhanh_c.ds_buoc(3, bien_the=config.NHANH_C_VARIANT)) == steps + ['bd3_TONG_HOP', 'p3_C9S'], 'thu tu Phan 3 cua goi KHAC ban da kiem -> DUNG'
    cfg = nhanh_c.cau_hinh(seed, False, config.NHANH_C_VARIANT)
    assert cfg['run_x'] == f'r84x_{config.ROW[seed]}' and cfg['ep_co_dinh'] == config.K0_BACKBONE_EPOCH, cfg['run_x']
    cu = ns.get('CAU_HINH_C')
    if cu is not None and cu != cfg:
        raise ValueError('namespace dang mang cau hinh luot KHAC -> Restart session roi chay lai tu dau notebook')
    for name in steps[steps.index(start) if start else 0:]:
        code, p = part3_code(nhanh_c, seed, name)
        ns['CAU_HINH_C'] = cfg
        log(f'\n######## Phan 3 · buoc {name} (goi nhanh_c, seed {seed}, bien the {config.NHANH_C_VARIANT}) ########')
        exec(compile(code, p, 'exec'), ns)


# ---- các tên Phần 3 mà phần sau đọc (tính bằng symtable từ các ô C6V / T / H của 85f, 10/11) ----
P1_NAMES = ('C', 'CFG_FS', 'CFG_JV2J', 'DAU_QD_S', 'DAU_S', 'KQ_S', 'LAB40', 'LOCAL', 'MOD', 'SIZE', 'SOUP_A', 'TEP_S', 'THU_NHANH', '_IVA', '_KA', 'dev',
            'predict')
TOKEN_NAMES = ('AMP', 'C', 'CFG_FS', 'CFG_JV2J', 'DAU_QD_S', 'EP_BB', 'KQ_S', 'LUONG', 'MEAN', 'MOD', 'NGUON_BB', 'SIZE', 'SOUP_A', 'STD', 'THU_NHANH',
               '_IVA', '_KA', '_loader_eval', '_out_luong', 'backbone', 'dev', 'eval_tf', 'mldec_h_hr', 'predict', 'q_cua', 'shared_lo', 'upper_frozen')
HEAD_NAMES = ('DocVung', 'MLDecoder', 'mldec_h', 'mldec_h_hr', 'q_cua')
PART3_NAMES = tuple(sorted(set(P1_NAMES) | set(TOKEN_NAMES) | set(HEAD_NAMES)))
ROW_ARRAYS = ('ORDER', 'YM', 'PREV_ROW', 'ST_ID', 'VIEW_FUS', 'NEO', 'ST_FUS_T', 'VIEW_FUS_T', 'NEO_T')


class Part3View:
    '''Nhìn vào namespace Phần 3 (globals() của notebook sau run_part3). Thiếu tên -> dừng ngay.'''

    def __init__(self, ns, names=PART3_NAMES + ROW_ARRAYS):
        thieu = [n for n in names if n not in ns]
        assert not thieu, f'chua chay Phan 3 (thieu {thieu[:6]}) -> chay o Phan 3 truoc. DUNG'
        self.ns = ns

    def __getattr__(self, name):
        try:
            return self.__dict__['ns'][name]
        except KeyError:
            raise AttributeError(name) from None

    def decision_score(self):
        '''Số quyết định tính lại: đầu quyết định (fus_attn_hr), bộ gộp A, val gộp view.'''
        return float(self.ns['KQ_S'][(self.ns['DAU_QD_S'], self.ns['_KA'])]['toan'])

    def _append(self, keys, study, view, neo, y):
        '''Nối hàng vào cuối các mảng theo hàng ORDER (= _noi_hang_g của 82c) -> chỉ số hàng mới.'''
        ns = self.ns
        n0, keys = len(ns['ORDER']), [str(k) for k in keys]
        ns['ORDER'] = pd.concat([ns['ORDER'], pd.DataFrame(dict(dicom_id=keys, study_id=np.asarray(study, 'int64')))], ignore_index=True)
        ns['YM'] = np.concatenate([ns['YM'], np.asarray(y, ns['YM'].dtype).reshape(len(keys), ns['YM'].shape[1])], 0)
        ns['PREV_ROW'] = np.concatenate([ns['PREV_ROW'], np.full(len(keys), -1, ns['PREV_ROW'].dtype)])
        ns['ST_ID'] = np.concatenate([ns['ST_ID'], np.asarray(study, ns['ST_ID'].dtype)])
        ns['VIEW_FUS'] = np.concatenate([ns['VIEW_FUS'], np.asarray(view, ns['VIEW_FUS'].dtype)])
        ns['NEO'] = np.concatenate([ns['NEO'], np.asarray(neo, ns['NEO'].dtype)])
        ns['ST_FUS_T'], ns['VIEW_FUS_T'], ns['NEO_T'] = (torch.from_numpy(a).to(ns['dev']) for a in (ns['ST_ID'], ns['VIEW_FUS'], ns['NEO']))
        return np.arange(n0, len(ns['ORDER']))

    @contextlib.contextmanager
    def appended_rows(self):
        '''with view.appended_rows() as append: ... — mọi mảng theo hàng được trả về bản gốc khi ra (kể cả khi lỗi; = _tra_lai_g).'''
        goc = {k: self.ns[k] for k in ROW_ARRAYS}
        try:
            yield self._append
        finally:
            for k, v in goc.items(): self.ns[k] = v
        assert all(self.ns[k] is goc[k] for k in ROW_ARRAYS), 'tra lai mang KHONG du -> DUNG'

    def identity_rows(self, append, n_studies=48):
        '''Cổng đồng nhất: n_studies study val đầu (mọi ảnh của chúng, thứ tự _IVA) nối lại dưới study giả, giữ thứ tự tương đối id
           -> (hàng gốc, hàng nối). Dự đoán hai bộ hàng phải trùng (đường nối hàng đủ mảng).'''
        ns = self.ns
        st = ns['ST_ID']
        uv = list(dict.fromkeys(st[ns['_IVA']].tolist()))[:n_studies]
        suv = set(uv)
        r0 = np.array([r for r in ns['_IVA'] if st[r] in suv], 'int64')
        b1 = int(st.max()) + 1
        hg = {s: i for i, s in enumerate(sorted(uv))}
        rd = append(ns['ORDER'].dicom_id.to_numpy()[r0], [b1 + hg[s] for s in st[r0]], ns['VIEW_FUS'][r0], ns['NEO'][r0], ns['YM'][r0])
        return r0, rd, len(uv)

    def frontal_rows(self, append, keys):
        '''Ảnh ngoài: study giả riêng từng ảnh, view frontal, neo theo view (ảnh PA) -> chỉ số hàng mới.'''
        ns = self.ns
        n = len(keys)
        return append(keys, int(ns['ST_ID'].max()) + 1 + np.arange(n), np.full(n, int(ns['CFG_FS']['view_ma']['frontal'])),
                      np.array(['frontal'] * n) == ns['CFG_JV2J']['view_neo'], np.zeros((n, ns['C']), 'float32'))

    def head_ops(self):
        '''Lớp / hàm đầu đọc của Phần 3 (= các ô c3 / C2B / C3C của 78z) cho heads.train_p2_heads.'''
        return dict(MLDecoder=self.ns['MLDecoder'], DocVung=self.ns['DocVung'], mldec_h=self.ns['mldec_h'], mldec_h_hr=self.ns['mldec_h_hr'],
                    q_cua=self.ns['q_cua'])
