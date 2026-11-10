"""Chuyển giao VinDr cho một backbone seed B sau Phần 3 (= các ô C6V, T, G của 85f, chỉ đổi tên hàng / tệp theo seed; test_87 kiểm trùng
từng bit trên CPU với ô gốc trên môi trường giả của test_85):

- predict_p1: P1 — 3.000 ảnh test VinDr qua bộ gộp trọng số A (đầu MIMIC 40 lớp, KHÔNG học lại); ảnh nối vào cuối ORDER (study giả riêng
  từng ảnh, view frontal, neo theo view) -> cùng hàm predict của C6S; cổng số quyết định + cổng đồng nhất. KHÔNG đọc nhãn VinDr.
- extract_tokens: trích token patch + vùng + [CLS ; TB patch] của 18.000 ảnh bằng backbone + luồng vùng đông cứng (đường shared_lo của C6S,
  gồm skip block 7); cổng: patch = upper_frozen, đầu MIMIC trên token trích = P1 cùng phiên.
- p2_record / manifest: nội dung JSON của ô G / M."""
import datetime
import gc
import hashlib
import os
import tarfile
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from PIL import Image

from . import config
from .vindr import md5_keys

TOL_IDENTITY, N_IDENTITY_STUDIES = 1e-4, 48
TOL_PATCH, TOL_P1_TOKENS, TOKEN_BATCH = 1e-4, 2e-3, 32


@dataclass
class P1Result:
    probs: dict                   # {đầu: [ảnh, 40] float32}
    keys: list                    # thứ tự ảnh test (85b)
    decision: float
    identity: dict                # lệch tối đa cổng đồng nhất theo đầu
    sources: dict = field(default_factory=dict)


def predict_p1(view, reader, row, decision_ref, tol_decision=config.DECISION_TOL, log=print):
    '''= ô C6V của 85f. view = backbone_b.Part3View; row = tên hàng (vd 'B_s45'); decision_ref = số quyết định đã biết của seed.'''
    ns = view.ns
    assert not ns['THU_NHANH'], 'THU_NHANH = True -> DUNG'
    qd = view.decision_score()
    log(f"P1 {row}: so quyet dinh tinh lai ({ns['DAU_QD_S']}, {ns['_KA']}, val gop view) {qd:.4f} | so da biet {decision_ref} (+-{tol_decision})")
    assert abs(qd - decision_ref) <= tol_decision, 'so quyet dinh tinh lai KHAC so da biet -> bo gop / backbone KHAC luot goc. DUNG'
    ns['MOD'].load_state_dict(ns['SOUP_A'])
    du = config.VINDR_TEST
    kk = reader.inventory(du, r'(vd518_test\.tar|vd_test_chi_muc\.parquet|vd_test_meta\.json)')
    thieu = [t for t in ('vd518_test.tar', 'vd_test_chi_muc.parquet', 'vd_test_meta.json') if not kk.get(t)]
    assert not thieu, f"thieu {thieu} trong stage4_nb85b/r85b_vindr_test -> chay 85b truoc. DUNG"
    cm = pd.read_parquet(reader.fetch(du, kk, 'vd_test_chi_muc.parquet'), columns=['khoa', 'anh_ok'])
    meta = reader.load_json(du, kk, 'vd_test_meta.json')
    keys = cm.khoa.astype(str).tolist()
    assert md5_keys(keys) == meta['md5_khoa'] and len(keys) == meta['n'] and cm.anh_ok.astype(bool).all(), (
        'chi muc VinDr KHAC meta cua 85b / co anh hong -> DUNG')
    assert not set(keys) & set(ns['ORDER'].dicom_id.astype(str)), 'khoa VinDr TRUNG dicom_id MIMIC -> DUNG'
    with tarfile.open(reader.fetch(du, kk, 'vd518_test.tar')) as t: t.extractall(ns['LOCAL'], filter='data')
    assert all(os.path.exists(os.path.join(ns['LOCAL'], f'{k}.jpg')) for k in keys), 'thieu anh VinDr sau khi giai nen -> DUNG'
    S = ns['SIZE']
    assert all(Image.open(os.path.join(ns['LOCAL'], f'{k}.jpg')).size == (S, S) for k in keys[:: max(1, len(keys) // 32)]), f'anh VinDr khong phai {S} x {S} -> DUNG'
    log(f"VinDr: {len(keys):,} anh 518 giai nen vao {ns['LOCAL']} (md5 khoa khop meta 85b)")
    heads = list(ns['DAU_S'])
    with view.appended_rows() as append:
        r0, rd, n_st = view.identity_rows(append, N_IDENTITY_STUDIES)
        p0, p1 = ns['predict'](r0, arms=heads), ns['predict'](rd, arms=heads)
        identity = {d: float(np.abs(p0[d] - p1[d]).max()) for d in heads}
        log(f'cong dong nhat ({n_st} study val / {len(r0)} anh noi lai duoi study gia): lech max '
            + ', '.join(f'{d} {v:.1e}' for d, v in identity.items()) + f' (<= {TOL_IDENTITY})')
        assert max(identity.values()) <= TOL_IDENTITY, 'du doan hang NOI KHAC hang goc -> duong noi hang thieu mang / sai. DUNG, bao Claude'
        rv = view.frontal_rows(append, keys)
        t0 = time.time()
        probs = ns['predict'](rv, arms=heads)
        log(f"VinDr: du doan {len(keys):,} anh ({(time.time() - t0) / 60:.1f} phut) | neo {bool(ns['CFG_JV2J']['view_neo'] == 'frontal')} | view frontal "
            f"(ma {ns['CFG_FS']['view_ma']['frontal']}) | TB xac suat " + ', '.join(f'{d} {float(probs[d].mean()):.3f}' for d in heads))
    C = ns['C']
    assert all(probs[d].shape == (len(keys), C) and np.isfinite(probs[d]).all() for d in heads), 'du doan VinDr sai hinh / NaN -> DUNG'
    return P1Result(probs=probs, keys=keys, decision=qd, identity=identity, sources=dict(reader.sources))


def p1_files(view, p1, row, decision_ref, seed, run_tag):
    '''Tệp của P1 (= phần ghi của ô C6V): p_vd_<đầu>_s<seed>.npy + du_doan_vd_<thời điểm>.json.'''
    ns = view.ns
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    files = {f'p_vd_{d}_s{seed}.npy': np.asarray(p1.probs[d]).astype('float32') for d in ns['DAU_S']}
    files[f'du_doan_vd_{ts}.json'] = dict(
        t=ts, bb=row, run=run_tag, so_quyet_dinh=p1.decision, moc=decision_ref, lech_dong=p1.identity, dau=list(ns['DAU_S']), lab40=list(ns['LAB40']),
        n_vd=len(p1.keys), md5_khoa_vd=md5_keys(p1.keys), nguon_gop=ns['TEP_S'], nguon_vd=p1.sources,
        ghi_chu='87c (= o C6V cua 85f): du doan VinDr test bang bo gop trong so A (val gop view = so quyet dinh). KHONG doc nhan VinDr; bang 87d cham.')
    return files


@torch.no_grad()
def _upper_stream_with_cls(bb, luong, h8, out_luong):
    '''= upper_luong_hr (khoi_doc_dong_cung, mo_ghi=True) nhưng GIỮ CLS: -> (layernorm [B, 1 + N, d] float32, r [B, R, d] float32).'''
    st = luong.bat_dau(h8.shape[0]); h = h8
    L = bb.encoder.layer
    for l in range(luong.tang[0], len(L)):
        h = out_luong(L[l](h))
        h = luong.buoc(l, h, st, True)
    return bb.layernorm(h).float(), luong.dau_ra(st)['r'].float()


@dataclass
class Tokens:
    tok: torch.Tensor             # [N, T, d] fp16 (patch), thứ tự hàng P2
    vung: torch.Tensor            # [N, R, d] fp16 (token vùng)
    glob: torch.Tensor            # [N, 2d] float32 ([CLS ; TB patch])
    gates: dict
    source: dict


def extract_tokens(view, data, p1, row, decision_ref, tol_decision=config.DECISION_TOL, log=print):
    '''= ô T của 85f. data = vindr.P2Data (ảnh đã giải nén vào kho ảnh của Phần 3); p1 = P1Result cùng phiên.'''
    ns = view.ns
    assert not ns['THU_NHANH'], 'THU_NHANH = True -> DUNG'
    qd = view.decision_score()
    log(f'87c {row}: so quyet dinh tinh lai ({ns["DAU_QD_S"]}, {ns["_KA"]}) {qd:.4f} | so da biet {decision_ref} (+-{tol_decision})')
    assert abs(qd - decision_ref) <= tol_decision, 'so quyet dinh KHAC so da biet -> backbone / bo gop KHAC luot goc. DUNG'
    MOD = ns['MOD']
    MOD.load_state_dict(ns['SOUP_A']); MOD.eval()
    assert list(p1.keys) == list(data.keys_test), 'thu tu anh test cua C6V KHAC o D -> DUNG'
    C = ns['C']
    p_p1 = np.asarray(p1.probs['mldec_hr']).astype('float32')
    assert p_p1.shape == (len(data.keys_test), C), p_p1.shape
    dev, backbone, LUONG = ns['dev'], ns['backbone'], ns['LUONG']
    N, T, D, R = len(data.keys), (ns['SIZE'] // 14) ** 2, int(backbone.config.hidden_size), int(LUONG.hop_tb.shape[0])
    need = N * (T + R) * D * 2
    gc.collect()
    if dev == 'cuda': torch.cuda.empty_cache()                                    # bộ nhớ đệm PyTorch (sau C6S / P1) KHÔNG tính là trống
    free = torch.cuda.mem_get_info()[0] if dev == 'cuda' else 1 << 62
    log(f'token P2: {N:,} anh x ({T} patch + {R} vung) x {D} fp16 = {need / 1e9:.1f} GB | GPU con trong {free / 1e9:.1f} GB')
    assert free >= need + 6e9, 'GPU khong du cho token P2 -> chay tren G4 (96 GB). DUNG'
    TOK = torch.empty((N, T, D), dtype=torch.float16, device=dev)
    VUNG = torch.empty((N, R, D), dtype=torch.float16, device=dev)
    GLOB = torch.empty((N, 2 * D), dtype=torch.float32, device=dev)
    done = np.zeros(N, bool)
    pm = np.full((len(data.keys_test), C), np.nan, 'float32')
    lech_patch = None
    shared_lo, upper_frozen, loader_eval, out_luong = ns['shared_lo'], ns['upper_frozen'], ns['_loader_eval'], ns['_out_luong']
    mldec_h_hr, q_cua = ns['mldec_h_hr'], ns['q_cua']
    with view.appended_rows() as append:
        r0, rd, _ = view.identity_rows(append, N_IDENTITY_STUDIES)
        q0, q1 = ns['predict'](r0, arms=['mldec_hr']), ns['predict'](rd, arms=['mldec_hr'])
        lech_dong = float(np.abs(q0['mldec_hr'] - q1['mldec_hr']).max())
        assert lech_dong <= TOL_IDENTITY, f'cong dong nhat lech {lech_dong:.1e} -> DUNG'
        # test TRƯỚC (cùng thứ tự + lô 32 như predict của P1), rồi train; vị trí trong rv -> hàng P2 (train trước, test sau)
        ntr, nte = len(data.keys_train), len(data.keys_test)
        rv = view.frontal_rows(append, list(data.keys_test) + list(data.keys_train))
        j = np.r_[ntr + np.arange(nte), np.arange(ntr)]
        t0 = time.time()
        with torch.no_grad():                                                     # đầu MIMIC (MOD) có tham số học -> KHÔNG đồ thị
            for tap, sl in (('test', slice(0, nte)), ('train', slice(nte, N))):
                for b in loader_eval(rv[sl], ns['eval_tf'], TOKEN_BATCH):
                    img, rows = b['image'].to(dev, non_blocking=True), b['row'].to(dev)
                    pos = b['pos'].numpy() + sl.start
                    h8 = shared_lo(img)
                    with torch.autocast('cuda', dtype=ns['AMP'], enabled=dev == 'cuda'):
                        LUONG.dat_neo(ns['NEO_T'][rows].float())
                        full, r = _upper_stream_with_cls(backbone, LUONG, h8, out_luong)
                    if lech_patch is None:                                        # cổng (a): patch = upper_frozen của C6S
                        lech_patch = float((upper_frozen(h8, rows) - full[:, 1:]).abs().max())
                        assert lech_patch <= TOL_PATCH, f'patch trich KHAC upper_frozen (lech {lech_patch:.1e}) -> DUNG'
                    jj = j[pos]
                    jt = torch.as_tensor(jj, device=dev)
                    TOK[jt] = full[:, 1:].half(); VUNG[jt] = r.half()
                    GLOB[jt] = torch.cat([full[:, 0], full[:, 1:].mean(1)], 1)
                    done[jj] = True
                    if tap == 'test':                                             # cổng (b): đầu MIMIC trên token trích (trước fp16)
                        pm[jj - ntr] = torch.sigmoid(mldec_h_hr(MOD['mldec_hr'], q_cua(MOD['mldec_hr']), full[:, 1:], r)[0].float()).cpu().numpy()
    assert done.all() and torch.isfinite(GLOB).all() and bool(torch.isfinite(TOK[:: max(1, N // 97)].float()).all()), 'thieu anh / token NaN -> DUNG'
    lech_p1 = float(np.abs(pm - p_p1).max())
    log(f'trich {N:,} anh ({(time.time() - t0) / 60:.1f} phut) | cong dong nhat {lech_dong:.1e} | patch - upper_frozen {lech_patch:.1e} | '
        f'dau MIMIC tren token trich - P1: lech max {lech_p1:.1e} (<= {TOL_P1_TOKENS})')
    assert lech_p1 <= TOL_P1_TOKENS, 'token trich + dau MIMIC KHAC du doan P1 -> duong trich KHAC P1. DUNG, bao Claude'
    source = dict(bb=row, nguon=f"78z gop trong so — backbone + luong vung DONG CUNG, rs_ ep{ns['EP_BB']} EMA", so_quyet_dinh=qd,
                  backbone=ns['NGUON_BB'], kich_thuoc=ns['SIZE'], chuan_hoa=dict(mean=ns['MEAN'], std=ns['STD']))
    gates = dict(dong_nhat=lech_dong, patch_upper_frozen=lech_patch, dau_mimic_p1=lech_p1, tol_p1=TOL_P1_TOKENS)
    return Tokens(tok=TOK, vung=VUNG, glob=GLOB, gates=gates, source=source)


def p2_files(data, tokens, result, row, run_tag):
    '''Tệp của P2 (= ô G của 85f): p2_<hàng>_<đầu>_val / _test.npy + p2_<hàng>_<thời điểm>.json. KHÔNG chấm test.'''
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    files = {}
    for d, kq in result['heads'].items():
        for tap in ('val', 'test'):
            files[f'p2_{row}_{d}_{tap}.npy'] = np.asarray(kq[tap]).astype('float32')
    files[f'p2_{row}_{ts}.json'] = dict(
        t=ts, bb=row, run=run_tag, nguon_bb=tokens.source, dau=list(result['heads']), nhan=data.label_names, lop_chinh=data.main_classes,
        md5_khoa=md5_keys(data.keys), md5_khoa_val=md5_keys([data.keys[i] for i in np.flatnonzero(data.split == 'val')]), md5_khoa_test=md5_keys(data.keys_test),
        hinh_token=list(tokens.tok.shape), hinh_vung=None if tokens.vung is None else list(tokens.vung.shape), dim_glob=int(tokens.glob.shape[1]),
        cau_hinh=result['config'], chon={d: k['chon'] for d, k in result['heads'].items()}, cong=tokens.gates, nguon=data.sources,
        ghi_chu='87c (= o G cua 85f): backbone DONG CUNG, dau hoc lai tren VinDr train; chon theo mAP val 21 lop; KHONG cham test (87d).')
    return files


def manifest(view, nhanh_c_commit, package_commit, seed, row, run_tag, tokens, result, step_sha256, notebook_84):
    '''Manifest của lượt 87c (K0 / K3; PLAN_KIEM_THU §15.5): gói + bước + nguồn backbone + số quyết định tính lại + cổng P2.'''
    ns = view.ns
    qd = view.decision_score()
    assert abs(qd - config.DECISION_SCORE[seed]) <= config.DECISION_TOL, 'so quyet dinh tinh lai KHAC manifest -> DUNG'
    return dict(module='87', muc='K3', notebook=config.NOTEBOOK_K3[seed], hang=row, seed=seed, bien_the=config.K3_VARIANT[seed], run_tag=run_tag,
                drv=[config.FOLDER_K3, run_tag], goi=dict(nhanh_c=dict(nhanh=config.NHANH_C_BRANCH, commit=nhanh_c_commit, commit_luot_goc=config.RUN_COMMIT_84[seed],
                                                                      buoc=list(config.PART3_STEPS[config.K3_VARIANT[seed]]), sha256_buoc=step_sha256),
                                                          b_robustness=package_commit),
                so_quyet_dinh_nb84=dict(gia_tri=config.DECISION_SCORE[seed], tep=notebook_84, nguon_so=config.DECISION_SOURCE[seed], dung_sai=config.DECISION_TOL),
                so_quyet_dinh_tinh_lai=qd,
                cau_hinh_c=ns['CAU_HINH_C'], nguon_bb=ns['NGUON_BB'], ep_bb=ns['EP_BB'], cong_p2=tokens.gates, dau_p2=list(result['heads']),
                dau_p1=list(ns['DAU_S']))


def sha256_steps(nhanh_c, seed):
    return {t: nhanh_c.BAM[t] for t in config.PART3_STEPS[config.K3_VARIANT[seed]]}


def md5_text(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()
