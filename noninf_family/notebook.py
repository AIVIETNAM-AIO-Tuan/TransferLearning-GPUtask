# -*- coding: utf-8 -*-
"""Ô của notebook `90d_ho_9_o_khong_kem` (Colab CPU; PLAN_90 §13.8, Tuấn duyệt 23/11). Chỉ in SỐ TỔNG HỢP (luật PhysioNet).

    C2 setup        khối Drive + phiên bản ba gói + hằng TRÙNG frozen_probe / vindr_bench + luật (in ra)
    C3 load_cxrlt   val CXR-LT (khối nhãn chuẩn, chỉ hàng val) + JSON 78r s43 + pv_ (cổng) + p_gopA + JSON C10 (cổng) ; CỔNG: chỉ số lần rút giữ = JSON 78r;
                    điểm 78r tính lại từ pv_ = JSON; 20 lần rút đầu = JSON; điểm + hai nửa số quyết định = JSON C10
    C4 cxrlt        số quyết định tính lại trên CÙNG lần rút -> khối CXR-LT (3 ô) ở 98,30 % cho s43 (luật) và s42 (đọc sau khi thấy số)
    C5 load_vindr   92g: JSON + lần rút (s43) ; 92b: JSON + dự đoán float16 (s42) ; 92a: nhãn test
    C6 vindr        khối VinDr s43 tính lại (CỔNG = 92g) ; khối VinDr s42 từ dự đoán 92b (CỔNG: ô k-NN = 92b tuyệt đối; logistic: in độ lệch)
    C7 summary      họ 9 ô s43 (luật) + s42 (mô tả): "không kém" mọi ô, ô "hơn" theo từng ô
    C8 sync         JSON + CSV -> dong_bo()"""
import csv
import io
import json
import os
import re
import time

import numpy as np

from . import __version__
from . import config as CF
from . import cxrlt as CX
from . import family as FA
from . import vindr as VI

DRIVE_NAMES = ('kiem_ke', 'tai', 'ghi', 'ghi_json', 'dong_bo', 'ten_tren_drive', 'GHI_LOC', 'RUN_TAG')
ROLE = ('ref', 'ctl')


def drive_api(ns):
    missing = [k for k in DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    return {k: ns[k] for k in DRIVE_NAMES}


def _sgn(x, nd=4):
    return '-' if x is None or x != x else f'{x:+.{nd}f}'


def _ci(c, nd=4):
    return f'[{_sgn(c[0], nd)}; {_sgn(c[1], nd)}]'


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda v: v.item() if hasattr(v, 'item') else (v.tolist() if hasattr(v, 'tolist') else str(v))))


def _newest(d, folder, pattern, what):
    '''-> (tên, bản) của tệp MỚI NHẤT theo tên (dấu thời gian trong tên) khớp `pattern` trong `folder`; nhiều bản cùng tên khác md5 -> bản mới nhất + in.'''
    kk = d['kiem_ke'](folder, mau=pattern)[1]
    assert kk, f'thieu {what} ({pattern}) trong {"/".join(folder)} -> DUNG'
    name = sorted(kk)[-1]
    return name, kk[name][0], len({b['md5'] for b in kk[name]}) > 1


def _load(d, folder, name, what):
    kk = d['kiem_ke'](folder, mau=re.escape(name))[1]
    assert name in kk, f'thieu {what} {name} trong {"/".join(folder)} -> DUNG'
    return d['tai'](kk[name][0]), kk[name][0]['md5']


def setup(ns, log=print):
    '''C2. -> CTX.'''
    import scipy
    import vindr_bench
    import frozen_probe
    from frozen_probe import config as C
    from frozen_probe import config_s43 as S
    from vindr_bench.evaluation import compare as CP
    import noninf_family
    d = drive_api(ns)
    assert d['RUN_TAG'] == CF.RUN_TAG, f"RUN_TAG {d['RUN_TAG']!r} != {CF.RUN_TAG!r} -> DUNG"
    CF.check_deps(C, S, CP)
    pk = dict(noninf_family=dict(phien_ban=__version__, sha=getattr(noninf_family, 'NF_SHA', None)),
              frozen_probe=dict(phien_ban=frozen_probe.__version__, sha=getattr(frozen_probe, 'FP_SHA', None)),
              vindr_bench=dict(phien_ban=vindr_bench.__version__, sha=getattr(vindr_bench, 'GOI_SHA', None)))
    log(f"90d | noninf_family {__version__} @ {(pk['noninf_family']['sha'] or '?')[:12]} + frozen_probe {pk['frozen_probe']['phien_ban']} @ "
        f"{(pk['frozen_probe']['sha'] or '?')[:12]} + vindr_bench {pk['vindr_bench']['phien_ban']} @ {(pk['vindr_bench']['sha'] or '?')[:12]} | numpy {np.__version__} | "
        f"scipy {scipy.__version__} | {os.cpu_count()} CPU")
    log(f"ho 9 o {CF.PAIR_TEXT}: 3 o CXR-LT ({', '.join(CF.CELL_TEXT[c] for c in CF.CELLS_CXR)}) + {len(C.FAMILY)} o VinDr (k-NN / logistic x 1 / 10 / 100 % "
        f"nhan); hai bo du lieu doc lap -> khoang noi rieng moi khoi: CXR-LT {CF.CXR_LEVEL:.2f} %, VinDr {CF.VINDR_LEVEL:.2f} % (tich = 95 %)")
    log(f"luat (dat truoc, s43): 'khong kem' = can duoi > -delta o MOI o (CXR-LT: " + ', '.join(f"{CF.CELL_TEXT[c]} {CF.DELTA_CXR[c]:g}" for c in CF.CELLS_CXR)
        + "; VinDr: " + ', '.join(f"{C.PROBE_TEXT[p]} {C.FRACTION_TEXT[f]} {S.DELTA[(p, f)]:g}" for p, f in C.FAMILY)
        + f"); 'hon' = hieu >= nguong (CXR-LT {CF.MARGIN_CXR:g}, VinDr {C.MARGIN:g}) VA can duoi > 0 VA duong o hai nua; doc 'hon' theo TUNG o")
    log(f"seed 43 = luat dat truoc | seed 42 = doc SAU khi da thay so (mo ta, khong doi ket luan s43)")
    return dict(drive=d, goi=pk, t=time.strftime('%Y%m%d_%H%M%S'), t0=time.time(), tep={})


def load_cxrlt(ctx, log=print):
    '''C3. -> dict(val, r78). Mọi cổng trượt -> AssertionError (dừng).'''
    import pandas as pd
    d, tep = ctx['drive'], ctx['tep']
    t0 = time.time()
    paths = {}
    for name, folder in CF.DATA_FILES:
        paths[name], tep[name] = _load(d, folder, name, 'tep du lieu CXR-LT')
    inv = json.load(open(paths['cxrlt_inventory.json'], encoding='utf-8'))
    order = pd.read_parquet(paths['emb_index.parquet'])
    index = pd.read_parquet(paths['cxrlt_index.parquet'], columns=['dicom_id'] + list(inv['labels_task1']))
    va = CX.val_arrays(inv, order, index)
    del order, index
    assert len(va['i_va']) == CF.N_VAL, f"val {len(va['i_va']):,} anh KHAC {CF.N_VAL:,} (78r / C10) -> DUNG"
    val = CX.Val(va['Y'], va['sub'], va['study'], va['lab40'], CF.HALF_SEED)
    log(f"val CXR-LT: {len(val.Y):,} anh, {len(val.patients):,} benh nhan, {len(np.unique(val.study)):,} study | {len(val.lab40)} nhan, tap lop co dinh "
        f"{int(val.classes.sum())} | hai nua theo benh nhan (hat {CF.HALF_SEED}): {int((val.half == 0).sum()):,} / {int((val.half == 1).sum()):,} anh")
    name, ban, xd = _newest(d, CF.R78_DIR, CF.R78_JSON_RE, 'JSON 78r s43')
    js = json.load(open(d['tai'](ban), encoding='utf-8'))
    tep['78r_json'] = dict(ten=name, md5=ban['md5'])
    assert js.get('du_s43') and int(js['n_boot']) == CF.N_BOOT and int(js['hat']) == CF.BOOT_SEED, f'JSON 78r {name}: chua du s43 / n_boot / hat khac -> DUNG'
    assert int(js['n_benh_nhan']) == len(val.patients), f"JSON 78r: {js['n_benh_nhan']} benh nhan KHAC val ({len(val.patients)}) -> DUNG"
    log(f"78r: {name}{' (! nhieu ban khac md5 -> ban moi nhat)' if xd else ''} | {js['n_boot']} lan rut, hat {js['hat']}, {js['n_benh_nhan']:,} benh nhan")
    kept = val.kept(CF.N_BOOT, CF.BOOT_SEED)
    gk = []
    for seed in CF.SEEDS:
        for head in CF.HEADS_78R:
            a, b = CF.R78_KEY[seed]
            gk.append(list(js['rut'][f'{a} - {b}|{head}']['giu']) == kept and list(js['rut_hang'][head]['giu']) == kept)
    log(f"CONG lan rut: sinh lai tu hat {CF.BOOT_SEED} tren val nay -> giu {len(kept):,} / {CF.N_BOOT:,} (loai {CF.N_BOOT - len(kept)}) | chi so giu = JSON 78r "
        f"(4 cap x dau): {'DAT' if all(gk) else 'TRUOT'}")
    assert all(gk) and len(kept) == CF.EXPECT_KEPT, 'lan rut sinh lai KHAC 78r -> khong ghep cap duoc, DUNG'
    gate_pv = []
    for seed in CF.SEEDS:
        a, b = CF.R78_KEY[seed]
        for head in CF.HEADS_78R:
            G = {}
            for role, run, rk in zip(ROLE, CF.PV_DIR[seed], (a, b)):
                p, tep[f'pv_{seed}_{role}_{head}'] = _load(d, ('stage4_nb78r', run), CF.PV_FILE.format(head), f'pv_ 78r s{seed}')
                P = np.load(p).astype('float32')
                assert P.shape == (len(val.Y), len(val.lab40)) and np.isfinite(P).all(), f'pv_ {run} {head}: hinh {P.shape} -> DUNG'
                G[role] = val.merge_views(P)
            pt = {r: val.point(G[r]) for r in ROLE}
            key = f'{a} - {b}|{head}'
            ch = js['chenh'][key]
            lech = max(abs(pt['ref']['toan'] - pt['ctl']['toan'] - ch['gop']), abs(pt['ref']['nua_A'] - pt['ctl']['nua_A'] - ch['nua_A']),
                       abs(pt['ref']['nua_B'] - pt['ctl']['nua_B'] - ch['nua_B']))
            for role, rk in zip(ROLE, (a, b)):
                kq = js['ket_qua'][rk][head]
                lech = max(lech, abs(pt[role]['toan'] - kq['gop']), abs(pt[role]['nua_A'] - kq['nua_A']), abs(pt[role]['nua_B'] - kq['nua_B']))
            rows, kept_s = val.draws(G, CF.N_BOOT, CF.BOOT_SEED, stop=CF.N_CHECK_DRAWS)
            hang = js['rut_hang'][head]['hang']
            n = len(kept_s)
            ld = max((max(abs(rows['ref'][i] - hang[a][j]), abs(rows['ctl'][i] - hang[b][j]), abs(rows['ref'][i] - rows['ctl'][i] - js['rut'][key]['hieu'][j]))
                      for j, i in enumerate(kept_s)), default=float('inf'))
            ok = lech <= CF.TOL and ld <= CF.TOL and kept_s == kept[:n] and abs(ch['gop'] - CF.EXPECT_78R[seed][head]) <= CF.EXPECT_TOL
            gate_pv.append(dict(seed=seed, dau=head, lech_diem=float(lech), lech_lan_rut=float(ld), so_lan_rut_kiem=n, hieu=float(ch['gop']), dat=bool(ok)))
    log('CONG 78r tinh lai tu pv_ (diem + hai nua tung hang + hieu = JSON; ' + f'{CF.N_CHECK_DRAWS} lan rut dau = JSON; nguong {CF.TOL:g}): ' + ' | '.join(
        f"s{g['seed']} {g['dau']} (3)-(1b) {_sgn(g['hieu'])} lech {g['lech_diem']:.1e} / {g['lech_lan_rut']:.1e}" for g in gate_pv)
        + f" -> {'DAT' if all(g['dat'] for g in gate_pv) else 'TRUOT'}")
    assert all(g['dat'] for g in gate_pv), 'diem / lan rut 78r tinh lai KHAC JSON -> du lieu val / cach tinh lech, DUNG'
    P = {}
    gate_c10 = []
    for seed in CF.SEEDS:
        P[seed] = {}
        for role, run in zip(ROLE, CF.GTS_DIR[seed]):
            p, tep[f'p_gopA_{seed}_{role}'] = _load(d, (CF.GTS_FOLDER, run), CF.P_GOPA, f'p_gopA s{seed}')
            P[seed][role] = np.load(p).astype('float32')
            assert P[seed][role].shape == (len(val.Y), len(val.lab40)) and np.isfinite(P[seed][role]).all(), f'p_gopA {run}: hinh {P[seed][role].shape} -> DUNG'
        pt = {r: val.point(val.merge_views(P[seed][r])) for r in ROLE}
        dd = [pt['ctl'][k] - pt['ref'][k] for k in ('toan', 'nua_A', 'nua_B')]            # chiều của C10: (1b) − (3)
        ok4 = all(abs(x - y) <= CF.EXPECT_TOL for x, y in zip(dd, CF.EXPECT_C10[seed]))
        kk = d['kiem_ke']((CF.GTS_FOLDER, CF.GTS_DIR[seed][1]), mau=CF.C10_JSON_RE[seed])[1]
        if kk:
            name = sorted(kk)[-1]
            cj = json.load(open(d['tai'](kk[name][0]), encoding='utf-8'))
            tep[f'c10_{seed}'] = dict(ten=name, md5=kk[name][0]['md5'])
            kq = cj['ket_qua']
            assert set(kq) == set(CF.C10_ROW_KEYS), f'JSON C10 {name}: hang {sorted(kq)} -> DUNG'
            lech = max(abs(pt[r][k] - kq[c][k]) for r, c in zip(('ctl', 'ref'), CF.C10_ROW_KEYS) for k in ('toan', 'nua_A', 'nua_B'))
            lech = max(lech, max(abs(x - cj['chenh'][k]) for x, k in zip(dd, ('toan', 'nua_A', 'nua_B'))))
            src, ok = f'JSON {name}', lech <= CF.TOL and ok4
        else:
            lech, src, ok = float('nan'), 'KHONG thay JSON C10 -> chi so voi so in 4 chu so', ok4
        gate_c10.append(dict(seed=seed, nguon=src, lech=lech, hieu_1b_tru_3=dd, dat=bool(ok)))
    log('CONG so quyet dinh (diem + hai nua tung hang tu p_gopA = C10): ' + ' | '.join(
        f"s{g['seed']} (1b)-(3) {_sgn(g['hieu_1b_tru_3'][0])} (nua A {_sgn(g['hieu_1b_tru_3'][1])} / B {_sgn(g['hieu_1b_tru_3'][2])}) {g['nguon']} lech "
        f"{g['lech']:.1e}" for g in gate_c10) + f" -> {'DAT' if all(g['dat'] for g in gate_c10) else 'TRUOT'}")
    assert all(g['dat'] for g in gate_c10), 'so quyet dinh tinh lai KHAC C10 -> DUNG'
    ctx.update(val=val, r78=js, kept=kept, p_gopA=P, cong=dict(lan_rut=dict(giu=len(kept), dat=True), r78_pv=gate_pv, c10=gate_c10))
    log(f"C3 xong | {(time.time() - t0) / 60:.1f} phut")
    return dict(val=val, r78=js)


def cxrlt(ctx, log=print):
    '''C4. Số quyết định trên CÙNG lần rút -> khối CXR-LT ở CXR_LEVEL cho s43 (luật) + s42 (mô tả). -> {seed: khối}.'''
    val, js = ctx['val'], ctx['r78']
    out, cells_all = {}, {}
    for seed in CF.SEEDS:
        t0 = time.time()
        dec = CX.decision_cell(val, ctx['p_gopA'][seed]['ref'], ctx['p_gopA'][seed]['ctl'], CF.N_BOOT, CF.BOOT_SEED)
        assert dec['giu'] == ctx['kept'], f's{seed}: lan rut giu cua so quyet dinh KHAC 78r -> DUNG'
        cells = dict(so_quyet_dinh=dec, tuyen_tinh=CX.r78_cell(js, seed, 'lin'), knn=CX.r78_cell(js, seed, 'knn'))
        blk = FA.cxr_block(cells)
        blk.update(seed=seed, vai=CF.SEED_ROLE[seed], phut=(time.time() - t0) / 60, diem_so_quyet_dinh=dec['diem'], ba_so_quyet_dinh=dec['ba'],
                   so_anh=len(val.Y), so_nhan=len(val.lab40))
        out[seed], cells_all[seed] = blk, cells
        _print_cxr(blk, log)
    ctx.update(cxr=out, cxr_cells=cells_all)
    return out


def _print_cxr(blk, log):
    log(f"\n=== Khoi CXR-LT, seed {blk['seed']} ({blk['vai']}): hieu = {CF.PAIR_TEXT}; val CXR-LT {blk['so_anh']:,} anh, gop view; AP bac thang macro "
        f"{blk['so_nhan']} nhan; khoang noi chung 3 o o muc {blk['muc']:.2f} % (q = {blk['q']:.2f}; {blk['so_lan_rut_giu']:,} lan rut theo benh nhan) | "
        f"{blk['phut']:.1f} phut ===")
    log(f"{'o':<50}{'d AP / AUPRC hinh thang / AUROC':<34}{'khoang noi':<22}{'CI95 rieng':<22}{'hai nua':<20}{'-delta':>8}  {'khong kem':<11}hon")
    for r in blk['phep_so']:
        d3 = r['d3']
        log(f"{CF.CELL_TEXT[r['o']]:<50}{(_sgn(d3['ap']) + ' / ' + _sgn(d3['auprc_thang']) + ' / ' + _sgn(d3['auroc'])):<34}{_ci(r['ci_khoi']):<22}"
            f"{_ci(r['ci_danh_nghia']):<22}{(_sgn(r['nua'][0]) + ' / ' + _sgn(r['nua'][1])):<20}{-r['delta']:>8.3f}  {('DAT' if r['khong_kem'] else 'TRUOT'):<11}"
            f"{r['ket_luan_hon']}")
    log(f"    khoi CXR-LT s{blk['seed']}: 'khong kem' o MOI o: {'DAT' if blk['khong_kem_moi_o'] else 'TRUOT'} (nguong 'hon' {blk['nguong_hon']:g})")


def load_vindr(ctx, log=print):
    '''C5. 92g (JSON + m21, s43) + 92b (JSON + dự đoán float16, s42) + nhãn test 92a.'''
    from frozen_probe import features as FT
    from frozen_probe.notebook import probe_bb as PB
    from vindr_bench.evaluation import panels as P
    from frozen_probe import config as C
    d, tep = ctx['drive'], ctx['tep']
    name, ban, xd = _newest(d, CF.R92G_DIR, CF.R92G_JSON_RE, 'JSON 92g')
    t = re.fullmatch(CF.R92G_JSON_RE, name).group(1)
    js92g = json.load(open(d['tai'](ban), encoding='utf-8'))
    tep['92g_json'] = dict(ten=name, md5=ban['md5'])
    pm, tep['92g_m21'] = _load(d, CF.R92G_DIR, CF.R92G_M21.format(t), 'lan rut 92g')
    with np.load(pm, allow_pickle=False) as z:
        m21 = {k: z[k].copy() for k in z.files}
    nameb, banb, xdb = _newest(d, CF.R92B_DIR, CF.R92B_JSON_RE, 'JSON 92b')
    tb = re.fullmatch(CF.R92B_JSON_RE, nameb).group(1)
    js92b = json.load(open(d['tai'](banb), encoding='utf-8'))
    tep['92b_json'] = dict(ten=nameb, md5=banb['md5'])
    assert js92g.get('tep_92b') == nameb, f"92g doc {js92g.get('tep_92b')} nhung JSON 92b moi nhat la {nameb} -> DUNG"
    pp, tep['92b_du_doan'] = _load(d, CF.R92B_DIR, CF.R92B_PRED.format(tb), 'du doan 92b')
    with np.load(pp, allow_pickle=False) as z:
        preds = {k: z[k].copy() for k in z.files if k.split('|')[0] in CF.VINDR_ROWS[42]}
    kk = d['kiem_ke'](CF.R92A_DIR)[1]
    assert FT.MANIFEST_FILE in kk and FT.LABEL_FILE in kk, f'thieu manifest / nhan 92a trong {"/".join(CF.R92A_DIR)} -> DUNG'
    man = json.load(open(d['tai'](kk[FT.MANIFEST_FILE][0]), encoding='utf-8'))
    lab = PB._labels(d, kk, man, '92a')
    test = np.flatnonzero(lab['tach'] == C.TEST_SPLIT)
    target = lab['nhan'][test]
    cols = P.panel_columns(lab['ten_nhan'], lab['lop_chinh'], target)
    log(f"92g: {name} (+ lan rut {CF.R92G_M21.format(t)}) | 92b: {nameb} (+ du doan {CF.R92B_PRED.format(tb)}, {len(preds)} bo cua hai hang s42) | nhan test 92a: "
        f"{len(test):,} anh x {target.shape[1]} nhan (van tay = cac luot 89: TRUNG)")
    ctx.update(js92g=js92g, m21_92g=m21, js92b=js92b, preds92b=preds, target=target, cols=cols)
    return dict(js92g=name, js92b=nameb)


def vindr(ctx, log=print, workers=None, n_boot=None):
    '''C6. Khối VinDr s43 tính lại (cổng = 92g) + khối VinDr s42 từ dự đoán 92b (cổng k-NN = 92b). -> {seed: khối}.'''
    t0 = time.time()
    ref, ctl = CF.VINDR_ROWS[43]
    c43 = VI.cells_92g(ctx['js92g'], ctx['m21_92g'], (ref, ctl))
    b43 = FA.vindr_block(c43, ref, ctl)
    ok, dev, err = FA.compare_92g(b43, ctx['js92g']['khong_kem_C'])
    log(f"CONG khoi VinDr s43 tinh lai tu lan rut 92g = Bang 3 cua 92g (q, khoang, CI, hai nua, tung lan rut, ket luan; nguong {CF.VINDR_TOL:g}): lech lon nhat "
        f"{dev:.1e} -> {'DAT' if ok else 'TRUOT ' + '; '.join(err[:3])}")
    assert ok, 'khoi VinDr s43 tinh lai KHAC 92g -> DUNG'
    workers = workers or max(1, min(4, os.cpu_count() or 1))          # Colab CPU = 2; trần 4 để máy nhiều lõi không đẻ quá nhiều tiến trình
    r42, c42 = CF.VINDR_ROWS[42]
    log(f"s42: bootstrap tu du doan 92b (float16) cho 2 hang x 6 phep ({workers} tien trinh) ...")
    cells42 = VI.cells_92b(ctx['preds92b'], ctx['target'], ctx['cols'], (r42, c42), workers=workers, log=log, n_boot=n_boot)
    chk = VI.check_92b(cells42, ctx['js92b'])
    kn = [c for c in chk if c['dau'] == 'knn']
    lg = [c for c in chk if c['dau'] != 'knn']
    log(f"CONG o s42 tinh lai = o 92b: k-NN {len(kn)} o lech lon nhat {max(c['lech_max'] for c in kn):.1e} (phai <= {CF.VINDR_TOL:g}) | logistic {len(lg)} o lech "
        f"lon nhat {max(c['lech_max'] for c in lg):.1e} (float16; in de doc) -> {'DAT' if all(c['dat'] for c in chk) else 'TRUOT'}")
    assert all(c['dat'] for c in chk), 'o k-NN s42 tinh lai KHAC 92b -> sai anh xa du doan / nhan, DUNG'
    b42 = FA.vindr_block(cells42, r42, c42)
    for seed, b in ((43, b43), (42, b42)):
        b.update(seed=seed, vai=CF.SEED_ROLE[seed], so_anh_test=int(len(ctx['target'])))
    b42['lech_logistic_vs_92b'] = float(max(c['lech_max'] for c in lg))
    b42['sat_nguong'] = sat_float16(b42)
    ctx.update(vindr_blk={43: b43, 42: b42}, cong_vindr=dict(s43=dict(dat=ok, lech=dev), s42=chk), phut_vindr=(time.time() - t0) / 60)
    for b in (b43, b42):
        _print_vindr(b, log)
    log(f"C6 xong | {(time.time() - t0) / 60:.1f} phut")
    return ctx['vindr_blk']


def sat_float16(b):
    '''Ô logistic s42 (tính từ dự đoán float16) mà kết luận có thể đổi trong phạm vi lệch so với 92b: lệch của cận dưới ước <= (1 + q) x lệch lớn nhất
       (hiệu lệch <= lệch; độ lệch chuẩn lần rút lệch <= ~lệch). -> [tên ô] (ước, không phải phép đo).'''
    from frozen_probe import config as C
    dev = b['lech_logistic_vs_92b']
    band = (1.0 + b['q']) * dev
    out = []
    for x in b['phep_so']:
        if x['dau'] == 'knn':
            continue
        lo = x['ci_khoi'][0]
        if abs(lo + x['delta']) <= band or abs(lo) <= band or abs(x['d']['ap'] - C.MARGIN) <= dev:
            out.append(f"{C.PROBE_TEXT[x['dau']]} {C.FRACTION_TEXT[x['ti_le']]}")
    return out


def _print_vindr(b, log):
    from frozen_probe import config as C
    log(f"\n=== Khoi VinDr, seed {b['seed']} ({b['vai']}): hieu = {CF.PAIR_TEXT}; test VinDr {b['so_anh_test']:,} anh; mAP21 = AP bac thang macro 21 lop; "
        f"khoang noi chung 6 o o muc {b['muc']:.2f} % (q = {b['q']:.2f}) ===")
    log(f"{'dau doc':<11}{'nhan':>7}  {'d AP / AUPRC hinh thang / AUROC':<34}{'khoang noi':<22}{'CI95 rieng':<22}{'hai nua':<20}{'-delta':>8}  {'khong kem':<11}hon")
    for x in b['phep_so']:
        d3 = x['d']
        log(f"{C.PROBE_TEXT[x['dau']]:<11}{C.FRACTION_TEXT[x['ti_le']]:>7}  {(_sgn(d3['ap'], 3) + ' / ' + _sgn(d3['auprc_thang'], 3) + ' / ' + _sgn(d3['auroc'], 3)):<34}"
            f"{_ci(x['ci_khoi'], 3):<22}{_ci(x['ci_danh_nghia'], 3):<22}{(_sgn(x['nua'][0], 3) + ' / ' + _sgn(x['nua'][1], 3)):<20}{-x['delta']:>8.3f}  "
            f"{('DAT' if x['khong_kem'] else 'TRUOT'):<11}{x['ket_luan_hon']}")
    extra = f" | o logistic tinh tu du doan float16: lech so voi 92b <= {b['lech_logistic_vs_92b']:.1e}" if 'lech_logistic_vs_92b' in b else ''
    sat = f" | sat nguong (trong pham vi lech float16): {b['sat_nguong']}" if b.get('sat_nguong') else ''
    log(f"    khoi VinDr s{b['seed']}: 'khong kem' o MOI o: {'DAT' if b['khong_kem_moi_o'] else 'TRUOT'}{extra}{sat}")


def summary(ctx, log=print):
    '''C7. Họ 9 ô từng seed.'''
    out = {}
    for seed in CF.SEEDS:
        out[seed] = FA.family(ctx['cxr'][seed], ctx['vindr_blk'][seed])
        f = out[seed]
        log(f"\n=== HO 9 O {CF.PAIR_TEXT}, seed {seed} ({CF.SEED_ROLE[seed]}): hai khoi doc lap, muc chung {f['muc_chung']:.2f} % ===")
        log(f"  'khong kem' o MOI o (3 CXR-LT + 6 VinDr): {'DAT' if f['khong_kem_ca_ho'] else 'TRUOT'}" + (f" | o truot: {f['o_truot_khong_kem']}" if f['o_truot_khong_kem'] else ''))
        log(f"  o 'hon' ({len(f['o_hon'])} / {f['so_o']}): {f['o_hon'] or '-'}" + (f" | o 'kem': {f['o_kem']}" if f['o_kem'] else ''))
    log("\n  doc THEO TUNG O; ket luan cua luat = seed 43; seed 42 la mo ta doc sau khi da thay so diem. Khoang chi do may rui cua anh / benh nhan val-test, "
        "KHONG do may rui khi huan luyen lai (seed).")
    ctx['ho'] = out
    return out


def _rows_csv(ctx):
    from frozen_probe import config as C
    out = []
    for seed in CF.SEEDS:
        for r in ctx['cxr'][seed]['phep_so']:
            out.append(dict(seed=seed, vai=CF.SEED_ROLE[seed], bo_du_lieu='CXR-LT val', o=CF.CELL_TEXT[r['o']], dieu_kien=f"{CF.ROW_TEXT['ref']} TRU {CF.ROW_TEXT['ctl']}",
                            chieu_phep_tru='(3) - (1b)', d_AP=r['d3']['ap'], d_AUPRC_hinh_thang=r['d3']['auprc_thang'], d_AUROC=r['d3']['auroc'],
                            khoang_noi_thap=r['ci_khoi'][0], khoang_noi_cao=r['ci_khoi'][1], muc=ctx['cxr'][seed]['muc'], delta=r['delta'], khong_kem=r['khong_kem'],
                            hon=r['ket_luan_hon']))
        for x in ctx['vindr_blk'][seed]['phep_so']:
            out.append(dict(seed=seed, vai=CF.SEED_ROLE[seed], bo_du_lieu='VinDr test', o=f"{C.PROBE_TEXT[x['dau']]} {C.FRACTION_TEXT[x['ti_le']]} nhan",
                            dieu_kien=f"{CF.ROW_TEXT['ref']} TRU {CF.ROW_TEXT['ctl']}", chieu_phep_tru='(3) - (1b)', d_AP=x['d']['ap'],
                            d_AUPRC_hinh_thang=x['d']['auprc_thang'], d_AUROC=x['d']['auroc'], khoang_noi_thap=x['ci_khoi'][0], khoang_noi_cao=x['ci_khoi'][1],
                            muc=ctx['vindr_blk'][seed]['muc'], delta=x['delta'], khong_kem=x['khong_kem'], hon=x['ket_luan_hon']))
    return out


def sync(ctx, log=print):
    '''C8. JSON + CSV -> dong_bo(). Chạy lại trong cùng phiên: tên tệp có hậu tố lần ghi.'''
    d = ctx['drive']
    ctx['lan_ghi'] = ctx.get('lan_ghi', 0) + 1
    t = ctx['t'] + ('' if ctx['lan_ghi'] == 1 else f"_lan{ctx['lan_ghi']}")
    cxr = {str(s): {k: v for k, v in b.items()} for s, b in ctx['cxr'].items()}
    rut_cxr = {str(s): {c: [None if v != v else float(v) for v in ctx['cxr_cells'][s][c]['rut']] for c in CF.CELLS_CXR} for s in CF.SEEDS}
    obj = _jsonable(dict(run=CF.RUN_TAG, t=t, goi=ctx['goi'], tep=ctx['tep'], cong=dict(ctx['cong'], vindr=ctx['cong_vindr']), khoi_cxr=cxr,
                         lan_rut_cxr=rut_cxr, khoi_vindr={str(s): b for s, b in ctx['vindr_blk'].items()}, ho={str(s): f for s, f in ctx['ho'].items()},
                         hang=CF.ROW_TEXT, vai_seed={str(s): CF.SEED_ROLE[s] for s in CF.SEEDS}, phut=(time.time() - ctx['t0']) / 60,
                         ghi_chu='PLAN_90 13.4 / 13.8 (Tuan duyet 23/11): khoang noi chung moi khoi (CXR-LT 98.30 %, VinDr 96.64 %, tich 95 %); seed 43 = luat, '
                                 'seed 42 = mo ta doc sau khi thay so'))
    names = [f'{CF.OUT_PREFIX}_{t}.json']
    d['ghi'](names[-1], lambda q: d['ghi_json'](q, obj))
    rows = _rows_csv(ctx)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    text = buf.getvalue()
    names.append(f'{CF.OUT_PREFIX}_bang_{t}.csv')

    def _csv(q):
        with open(q, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(text)
    d['ghi'](names[-1], _csv)
    du = d['dong_bo']()
    co = set(names) <= set(d['ten_tren_drive']())
    log(f"C8: da ghi {len(names)} tep (1 JSON, 1 CSV {len(rows)} hang) | dong_bo {'DU' if du else 'CHUA DU'} | tren Drive: {'CO du' if co else 'THIEU'}"
        + ('' if du and co else ' | KHONG tat runtime; chay lai C8') + f" | ca notebook {(time.time() - ctx['t0']) / 60:.1f} phut")
    return dict(dong_bo=du, du_tep=co, tep=names)
