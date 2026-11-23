# -*- coding: utf-8 -*-
"""Khối CXR-LT của họ 9 ô: 3 ô (3) − (1b) cùng seed trên val CXR-LT (AP bậc thang macro 40 nhãn, gộp view), lần rút bootstrap theo bệnh nhân CÙNG dãy với
78r C9 / C10 (hạt 22101, 2.000 lần).

Chép NGUYÊN VĂN (tests/test_noninf_family.py kiểm trùng cây cú pháp / trùng số):
  ap_cols       <- experiments/72_loss_kien_truc/dung_74c.AP (khối chuẩn của 74b / 78a / 78r)
  sort_once     <- _thu_tu_r9 của 78r C9 (experiments/78_chi_anh/dung_78r_vt.C9_VT; = _thu_tu_10 của C10)
  ap_weighted   <- _ap_w_r9 của 78r C9 (= _ap_w10 của C10)
  merge_views   <- gop_r của 78r C6R / _gop_v của 78z C6S (TB dự đoán các ảnh cùng study, gán lại từng ảnh)
  halves        <- NUA_R của 78r / NUA_S của 78z (hai nửa theo BỆNH NHÂN, hạt 7801)
  val_arrays    <- khối nhãn chuẩn dung_74c.DU_LIEU, CHỈ giữ hàng val (không giữ nhãn test)
  draws         <- vòng bootstrap của 78r C9 (trọng số theo bệnh nhân; tập lớp cố định; lần rút thiếu lớp -> loại)"""
import numpy as np

from . import config as CF


def ap_cols(Yc, Sc):
    o = np.full(Yc.shape[1], np.nan)
    for j in range(Yc.shape[1]):
        y, s = Yc[:, j], Sc[:, j]; n1 = y.sum()
        if n1 == 0 or n1 == len(y): continue
        r = np.argsort(-s, kind='stable'); yy, ss = y[r], s[r]
        tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
        last = np.r_[np.diff(ss) != 0, True]; tp, fp = tp[last], fp[last]
        pr, rc = tp / (tp + fp), tp / n1
        o[j] = float((np.diff(np.r_[0.0, rc]) * pr).sum())
    return o


def sort_once(Y, G):
    '''moi lop: thu tu giam dan theo diem (stable), nhan da xep, mat na cuoi chuoi diem bang nhau - CUNG cach ap_cols (= C10 cua 78x).'''
    out = []
    for j in range(G.shape[1]):
        r = np.argsort(-G[:, j], kind='stable')
        out.append((r, Y[r, j].astype('float64'), np.r_[np.diff(G[r, j]) != 0, True]))
    return out


def ap_weighted(tt, w):
    '''AP tung lop voi trong so mau w (bootstrap: so lan benh nhan duoc rut); w = 1 -> TRUNG ap_cols. Lop thieu duong / am trong lan rut -> nan.'''
    o = np.full(len(tt), np.nan)
    for j, (r, yy, last) in enumerate(tt):
        ww = w[r]
        n1, nw = float((yy * ww).sum()), float(ww.sum())
        if n1 == 0 or n1 == nw: continue
        tp, fp = np.cumsum(yy * ww)[last], np.cumsum((1 - yy) * ww)[last]
        mau = tp + fp
        pr = np.divide(tp, mau, out=np.zeros_like(tp), where=mau > 0)
        o[j] = float((np.diff(np.r_[0.0, tp / n1]) * pr).sum())
    return o


class Val:
    '''Tập val CXR-LT theo thứ tự ORDER (= thứ tự hàng của mọi tệp pv_ / p_gopA): nhãn 40 lớp, bệnh nhân, study, gộp view, hai nửa, chỉ số bệnh nhân.'''

    def __init__(self, Y, sub, study, lab40, half_seed):
        self.Y, self.sub, self.study, self.lab40 = np.asarray(Y), np.asarray(sub), np.asarray(study), list(lab40)
        assert self.Y.ndim == 2 and self.Y.shape[1] == len(self.lab40) and len(self.sub) == len(self.study) == len(self.Y), 'Val: kich thuoc lech -> DUNG'
        assert set(np.unique(self.Y).tolist()) <= {0.0, 1.0}, 'nhan val khong phai 0 / 1 -> DUNG'
        self._us, self._inv = np.unique(self.study, return_inverse=True)
        su = np.unique(self.sub)
        A = np.random.default_rng(half_seed).choice(su, size=len(su) // 2, replace=False)
        self.half = np.where(np.isin(self.sub, A), 0, 1)
        assert not (set(self.sub[self.half == 0].tolist()) & set(self.sub[self.half == 1].tolist())), 'benh nhan o CA HAI nua -> DUNG'
        self.patients, self.pinv = np.unique(self.sub, return_inverse=True)
        n1 = self.Y.sum(0)
        self.classes = (n1 > 0) & (n1 < len(self.Y))           # tập lớp cố định (= ~isnan(ap_cols(Y, ·)) của 78r: chỉ phụ thuộc nhãn)

    def merge_views(self, Q):
        S = np.zeros((len(self._us), Q.shape[1]), 'float64'); np.add.at(S, self._inv, Q)
        return (S / np.bincount(self._inv, minlength=len(self._us))[:, None])[self._inv].astype('float32')

    def point(self, Q):
        '''Q = dự đoán ĐÃ gộp view -> dict(toan, nua_A, nua_B) (= _so3 của 78z / so_r của 78r, không tier).'''
        Y, h = self.Y, self.half
        return dict(toan=float(np.nanmean(ap_cols(Y, Q))), nua_A=float(np.nanmean(ap_cols(Y[h == 0], Q[h == 0]))),
                    nua_B=float(np.nanmean(ap_cols(Y[h == 1], Q[h == 1]))))

    def three(self, Q):
        '''Q đã gộp view -> (AP bậc thang, AUPRC hình thang, AUROC), macro trên các lớp có cả hai nhãn (= ba_so_r9 của 78r C9).'''
        from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
        ap = ap_cols(self.Y, Q)
        pr, au = [], []
        for j in range(self.Y.shape[1]):
            y = self.Y[:, j]
            if 0 < y.sum() < len(y):
                p_, r_, _ = precision_recall_curve(y, Q[:, j]); pr.append(float(auc(r_, p_))); au.append(float(roc_auc_score(y, Q[:, j])))
        return float(np.nanmean(ap)), float(np.mean(pr)), float(np.mean(au))

    def weights(self, n_boot, seed):
        '''Dãy trọng số bootstrap theo bệnh nhân = 78r C9 / C10: np.random.default_rng(seed); mỗi lần rút n bệnh nhân có hoàn lại.'''
        rng, n = np.random.default_rng(seed), len(self.patients)
        for _ in range(n_boot):
            yield np.bincount(rng.integers(0, n, n), minlength=n)[self.pinv].astype('float64')

    def kept(self, n_boot, seed):
        '''Chỉ số lần rút GIỮ (không lớp nào của tập cố định thiếu dương / thiếu âm) — chỉ phụ thuộc nhãn + trọng số.'''
        Yc = self.Y[:, self.classes].astype('float64')
        out = []
        for i, w in enumerate(self.weights(n_boot, seed)):
            n1, nw = w @ Yc, w.sum()
            if not ((n1 == 0) | (n1 == nw)).any():
                out.append(i)
        return out

    def draws(self, merged, n_boot, seed, stop=None):
        '''merged = {tên: dự đoán ĐÃ gộp view} -> ({tên: [n_boot] mAP macro trên tập lớp cố định, NaN = lần rút bị loại}, [chỉ số giữ]). stop: chỉ tính
           tới lần rút này (cổng nhanh). Cùng vòng với 78r C9: cùng trọng số mọi hàng; lần rút nào một lớp cố định ra NaN -> loại.'''
        tts = {k: sort_once(self.Y, v) for k, v in merged.items()}
        one = np.ones(len(self.Y))
        for k, tt in tts.items():
            assert np.allclose(ap_weighted(tt, one), ap_cols(self.Y, merged[k]), atol=1e-12, equal_nan=True), f'{k}: AP co trong so (w = 1) KHAC ap_cols -> DUNG'
        m = n_boot if stop is None else min(stop, n_boot)
        out, kept = {k: np.full(n_boot, np.nan) for k in tts}, []
        for i, w in enumerate(self.weights(n_boot, seed)):
            if i >= m:
                break
            ap = {k: ap_weighted(tt, w) for k, tt in tts.items()}
            nan0 = np.isnan(next(iter(ap.values())))
            assert all(np.array_equal(np.isnan(v), nan0) for v in ap.values()), 'lop thieu trong lan rut KHAC giua cac hang -> DUNG'
            if (nan0 & self.classes).any():
                continue
            kept.append(i)
            for k, v in ap.items():
                out[k][i] = float(np.mean(v[self.classes]))
        return out, kept


def val_arrays(inv, order, index):
    '''Khối nhãn chuẩn (dung_74c.DU_LIEU) CHỈ cho hàng val: inv = cxrlt_inventory.json (dict), order = emb_index.parquet, index = cxrlt_index.parquet (cột
       dicom_id + 40 nhãn). -> dict(lab40, Y [n_val, 40] float32, sub, study, i_va). Thứ tự = np.where(split == 'val') của ORDER (= I_VA của 78r / 78z).'''
    lab40 = list(inv['labels_task1'])
    split = order.split.to_numpy()
    i_va = np.where(split == 'val')[0]
    idx = index[['dicom_id'] + lab40]
    assert idx.dicom_id.is_unique, 'cxrlt_index: dicom_id trung -> DUNG'
    dic = order.dicom_id.to_numpy()[i_va]
    Y = idx.set_index('dicom_id').loc[dic].reset_index()[lab40].to_numpy('float32')
    sub = order.subject_id.to_numpy()[i_va]
    study = order.study_id.to_numpy().astype('int64')[i_va]
    return dict(lab40=lab40, Y=Y, sub=sub, study=study, i_va=i_va)


def r78_cell(js, seed, head):
    '''Ô 78r (tuyến tính / k-NN) từ JSON C9 của 78r s43: hiệu (3) − (1b) cùng seed + lần rút đầy đủ [n_boot] (NaN = lần rút bị loại).'''
    a, b = CF.R78_KEY[seed]
    key = f'{a} - {b}|{head}'
    assert key in js['chenh'] and key in js['rut'], f'JSON 78r thieu cap {key} -> DUNG'
    ch, rt = js['chenh'][key], js['rut'][key]
    rut = np.full(int(js['n_boot']), np.nan)
    rut[np.asarray(rt['giu'], dtype=np.int64)] = np.asarray(rt['hieu'], dtype=np.float64)
    return dict(key=key, d=float(ch['gop']), nua=[float(ch['nua_A']), float(ch['nua_B'])], rut=rut, giu=list(rt['giu']),
                d3=dict(ap=float(ch['gop']), auprc_thang=float(ch['auprc']), auroc=float(ch['auroc'])))


def decision_cell(val, P_ref, P_ctl, n_boot, seed):
    '''Ô số quyết định: P_* = p_gopA (chưa gộp view) của (3) / (1b) -> hiệu điểm + hai nửa + 3 số + lần rút (cùng dãy 78r).'''
    G = {'ref': val.merge_views(P_ref), 'ctl': val.merge_views(P_ctl)}
    pt = {k: val.point(v) for k, v in G.items()}
    t3 = {k: val.three(v) for k, v in G.items()}
    rows, kept = val.draws(G, n_boot, seed)
    return dict(d=pt['ref']['toan'] - pt['ctl']['toan'], nua=[pt['ref']['nua_A'] - pt['ctl']['nua_A'], pt['ref']['nua_B'] - pt['ctl']['nua_B']],
                rut=rows['ref'] - rows['ctl'], giu=kept, diem=pt, ba=t3,
                d3=dict(ap=t3['ref'][0] - t3['ctl'][0], auprc_thang=t3['ref'][1] - t3['ctl'][1], auroc=t3['ref'][2] - t3['ctl'][2]))
