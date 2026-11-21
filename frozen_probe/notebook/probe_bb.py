# -*- coding: utf-8 -*-
"""Ô của notebook `92e_it_nhan_vindr_bb` (CPU; PLAN_92 §13). Giao thức = 92b (probe.py) NGUYÊN VẸN: cùng lần rút, cùng hai đầu đọc (probe._probe), cùng
bootstrap / ô / dạng luật (vindr_bench.evaluation qua scoring); chỉ khác: sáu hàng của ba cặp, họ 6 phép cho TỪNG cặp, không bản nhạy, đối chứng âm cho
mọi hàng mới, cổng tái lập trên cặp RAD-DINO của 92a / 92b.

    C2 setup    khối Drive + phiên bản hai gói + luật (in ra)
    C3 load     92d: manifest + nhãn (vân tay = các lượt 89) + 6 tệp đặc trưng (md5 = manifest); 92a: nhãn (TRÙNG 92d) + 2 tệp cặp RAD-DINO; 92b: JSON kết quả
    C4 probes   lần rút (md5 = 92b) -> k-NN + logistic ở 1 / 10 / 100 % (6 hàng; + k-NN trên CLS, mô tả) + các ô tái lập của cặp RAD-DINO; CỔNG hội tụ;
                đối chứng âm (nhãn xáo, 100 %) mọi hàng mới
    C5 score    bootstrap theo ảnh -> CỔNG tái lập (ô RAD-DINO = 92b) -> Bảng 1 (6 hàng) · Bảng 2 (LUẬT: họ 6 phép CÓ − KHÔNG từng cặp, max-T) · Bảng 3
                (bốn backbone, mô tả; RAD-DINO từ 92b) · Bảng 4 (CLS k-NN, mô tả)
    C6 figures  hình họ 6 phép từng cặp (khoảng đồng thời)
    C7 sync     JSON + CSV + PNG + dự đoán (float16) -> dong_bo()

Chỉ in SỐ TỔNG HỢP (luật PhysioNet)."""
import csv
import io
import json
import os
import time

import numpy as np

from .. import __version__
from .. import config as C
from .. import config_bb as CB
from .. import features as FT
from .. import probes as PR
from .. import scoring as SC
from .. import subsets as S
from . import probe as NBP

RES_PREFIX = 'it_nhan_vindr_'                 # tên tệp JSON kết quả của 92b: it_nhan_vindr_<t>.json


def setup(ns, log=print):
    '''C2. -> CTX.'''
    import scipy
    import vindr_bench
    import frozen_probe
    drive = NBP.drive_api(ns)
    assert drive['RUN_TAG'] == CB.RUN_TAGS['it_nhan'], f"RUN_TAG {drive['RUN_TAG']!r} != {CB.RUN_TAGS['it_nhan']!r} -> DUNG"
    pk = dict(frozen_probe=dict(phien_ban=__version__, sha=getattr(frozen_probe, 'FP_SHA', None)),
              vindr_bench=dict(phien_ban=vindr_bench.__version__, sha=getattr(vindr_bench, 'GOI_SHA', None)))
    log(f"92e | frozen_probe {__version__} @ {(pk['frozen_probe']['sha'] or '?')[:12]} + vindr_bench {pk['vindr_bench']['phien_ban']} @ "
        f"{(pk['vindr_bench']['sha'] or '?')[:12]} | numpy {np.__version__} | scipy {scipy.__version__} | {os.cpu_count()} CPU")
    log(f"luat dat truoc (PLAN_92 13.3 = 11.3 cho TUNG cap): ho {len(C.FAMILY)} phep CO − KHONG x {{k-NN, logistic}} x {{1 %, 10 %, 100 %}} tren mAP21 (AP bac thang "
        f"macro 21 lop), dac trung {C.PRIMARY_FEATURE}; khoang dong thoi max-T {C.LEVEL:g} % TRONG ho cua cap; 'hon' = hieu >= +{C.MARGIN:g} VA can duoi dong thoi > 0 "
        f"VA duong o hai nua test; 'kem' doi xung; doc theo tung o cua tung backbone; khong gop backbone; ket luan duong can seed 43")
    log('cap: ' + ' | '.join(CB.PAIR_TEXT[e] for e in CB.ENCODERS))
    return dict(drive=drive, goi=pk, t=time.strftime('%Y%m%d_%H%M%S'))


def _labels(d, kk, man, src):
    from vindr_bench.data import vindr as V
    from vindr_bench.protocols.ours import gates as G
    p = d['tai'](kk[FT.LABEL_FILE][0])
    assert FT.md5_file(p) == man['tep_nhan']['md5'], f'{src}: tep nhan md5 KHAC manifest -> DUNG'
    lab = FT.load_labels(p)
    same, diff = G.data_gate(lab['van_tay'])
    ok = all(V.md5_labels(lab['nhan'][lab['tach'] == s]) == lab['van_tay'][s]['md5_nhan'] for s in V.SPLITS)
    assert same and ok, f'{src}: du lieu KHAC cac luot 89 ({diff}) / md5 nhan tinh lai KHAC van tay -> DUNG'
    return lab


def _feature(d, kk, man, row, n, src):
    name = FT.feature_file(row)
    assert name in kk and row in man['hang'], f'{src}: thieu tep dac trung {name} -> DUNG'
    p = d['tai'](kk[name][0])
    m5 = FT.md5_file(p)
    assert m5 == man['hang'][row]['md5'], f'{src}/{name}: md5 {m5[:12]} KHAC manifest -> DUNG'
    f = FT.load_features(p, row)
    for k, X in f.items():
        assert X.shape == (n, int(man['hang'][row]['chieu'])) and np.isfinite(X).all(), f'{name} / {k}: hinh {X.shape} / khong huu han -> DUNG'
    return f, m5


def load(ctx, log=print):
    '''C3. -> dict(nhan, dac_trung).'''
    from vindr_bench.data import vindr as V
    d = ctx['drive']
    srcd = (CB.FOLDER, CB.RUN_TAGS['trich'])
    kk = d['kiem_ke'](srcd)[1]
    assert FT.MANIFEST_FILE in kk and FT.LABEL_FILE in kk, f'chua co ket qua 92d trong {"/".join(srcd)} -> chay 92d truoc. DUNG'
    man = json.load(open(d['tai'](kk[FT.MANIFEST_FILE][0]), encoding='utf-8'))
    lab = _labels(d, kk, man, '92d')
    n = len(lab['nhan'])
    log(f"nhan 92d: {n:,} anh x {lab['nhan'].shape[1]} nhan, {len(lab['lop_chinh'])} lop so chinh | tach " + ' / '.join(
        f"{s} {int((lab['tach'] == s).sum()):,}" for s in V.SPLITS) + ' | van tay = cac luot 89: TRUNG | md5 nhan tinh lai = van tay: DAT')
    feats = {}
    for row in CB.ROWS:
        feats[row], m5 = _feature(d, kk, man, row, n, '92d')
        h = man['hang'][row]
        log(f"  {CB.CODE[row]:<18}{CB.TEXT[row]}: {n:,} x {feats[row][C.PRIMARY_FEATURE].shape[1]} | md5 {m5[:12]} = manifest | nap: {h['nap'][:48]} | lech pretrain "
            f"{(h.get('lech_pre') or {}).get('do')} | ghi nguoc VinDr {[round(t, 3) for t in (h.get('ghi_nguoc_vindr') or [])]}")
    srca = (C.FOLDER, CB.SOURCE_RAD['trich'])
    ka = d['kiem_ke'](srca)[1]
    assert FT.MANIFEST_FILE in ka and FT.LABEL_FILE in ka, f'thieu ket qua 92a trong {"/".join(srca)} (can cho cong tai lap) -> DUNG'
    mana = json.load(open(d['tai'](ka[FT.MANIFEST_FILE][0]), encoding='utf-8'))
    laba = _labels(d, ka, mana, '92a')
    same = all(np.array_equal(lab[k], laba[k]) for k in ('nhan', 'tach')) and lab['ten_nhan'] == laba['ten_nhan'] and lab['lop_chinh'] == laba['lop_chinh']
    assert same, 'nhan / tach cua 92d KHAC 92a (khong cung thu tu anh) -> DUNG'
    rad = {}
    for row in CB.RAD_PAIR:
        rad[row], m5 = _feature(d, ka, mana, row, n, '92a')
        log(f"  {C.CODE[row]:<18}{C.TEXT[row]} (92a, cho cong tai lap): md5 {m5[:12]} = manifest 92a")
    srcb = (C.FOLDER, CB.SOURCE_RAD['it_nhan'])
    kb = d['kiem_ke'](srcb)[1]
    js = sorted(t for t in kb if t.startswith(RES_PREFIX) and t.endswith('.json') and t[len(RES_PREFIX):-5].replace('_', '').isdigit())
    assert js, f'thieu JSON ket qua 92b trong {"/".join(srcb)} (can cho cong tai lap + bang bon backbone) -> DUNG'
    res92b = json.load(open(d['tai'](kb[js[-1]][0]), encoding='utf-8'))
    log(f"92a / 92b: nhan TRUNG 92d (cung thu tu anh) | 92b {js[-1]} (lan rut md5 {res92b['md5_lan_rut'][:12]}, goi {res92b['goi']['frozen_probe']['phien_ban']})")
    ctx.update(manifest=man, nhan=lab, dac_trung=feats, rad=rad, ket_qua_92b=res92b, tep_92b=js[-1])
    return dict(nhan=lab, dac_trung=feats)


def probes(ctx, log=print):
    '''C4. -> {(hàng, đặc trưng, cấu hình, tỉ lệ, lần rút): dự đoán test [m, 28] float32}.'''
    lab, feats = ctx['nhan'], ctx['dac_trung']
    Y = lab['nhan'].astype(np.float64)
    pool = np.flatnonzero(np.isin(lab['tach'], C.POOL_SPLITS))
    test = np.flatnonzero(lab['tach'] == C.TEST_SPLIT)
    dr = S.all_draws(pool)
    m5 = S.md5(dr)
    assert m5 == ctx['ket_qua_92b']['md5_lan_rut'], f'md5 lan rut {m5[:12]} KHAC 92b ({ctx["ket_qua_92b"]["md5_lan_rut"][:12]}) -> khong ghep cap voi cap RAD-DINO. DUNG'
    log(f"be hoc {len(pool):,} anh | test {len(test):,} | lan rut: " + ', '.join(f"{C.FRACTION_TEXT[f]} = {S.size(len(pool), f):,} anh x {C.N_DRAWS[f]} lan" for f in C.FRACTIONS)
        + f" | md5 chi so {m5[:12]} = 92b (CUNG chi so voi cap RAD-DINO) | hat {C.DRAW_SEED}")
    preds, conv, t0 = {}, [], time.time()
    jobs = [(row, feats[row], CB.CONFIGS, C.FRACTIONS) for row in CB.ROWS] + [(row, ctx['rad'][row], None, None) for row in CB.RAD_PAIR]
    for row, fr, cfgs, _ in jobs:
        t1 = time.time()
        todo = ([(fe, cfg, f) for fe, cfg in cfgs for f in C.FRACTIONS] if cfgs is not None
                else [(C.PRIMARY_FEATURE, cfg, f) for cfg, f in CB.REPRO])
        for feat in sorted({x[0] for x in todo}):
            X = fr[feat].astype(np.float64)
            std = PR.standardizer(X[pool])
            Xte = X[test]
            for fe, cfg, f in todo:
                if fe != feat:
                    continue
                for dd in range(C.N_DRAWS[f]):
                    idx = dr[(f, dd)]
                    pv, info = NBP._probe(cfg, f, X[idx], Y[idx], Xte, std)
                    preds[(row, feat, cfg, f, dd)] = pv
                    if cfg.startswith('logistic'):
                        conv.append(dict(hang=row, dac_trung=feat, cau_hinh=cfg, ti_le=f, lan_rut=dd, **info))
        name = CB.CODE.get(row) or C.CODE[row] + ' (tai lap)'
        log(f"  {name:<20}xong {sum(1 for k in preds if k[0] == row)} bo du doan | {(time.time() - t1) / 60:.1f} phut | tong {(time.time() - t0) / 60:.1f} phut", )
    bad = [c for c in conv if not c['hoi_tu']]
    log(f"logistic: {len(conv)} lan fit | so vong lon nhat {max(c['so_vong'] for c in conv)} | |grad| lon nhat {max(c['grad_max'] for c in conv):.1e} | "
        f"hoi tu {'DAT' if not bad else f'TRUOT {len(bad)} lan'} | tong {(time.time() - t0) / 60:.1f} phut")
    assert not bad, f'logistic KHONG hoi tu o {[(b["hang"], b["cau_hinh"], b["ti_le"], b["lan_rut"]) for b in bad][:5]} -> DUNG, gui Claude dong tren'
    ctx.update(du_doan=preds, hoi_tu=conv, ve=dict(be=pool, test=test, md5_lan_rut=m5))
    ctx['doi_chung_am'] = negative_control(ctx, log)
    return preds


def negative_control(ctx, log=print):
    '''Nhãn học XÁO (hạt NEG_SEED như 92b), 100 %, đặc trưng chính, MỌI hàng mới: mAP21 test phải về mức may rủi (bắt rò nhãn test).'''
    from vindr_bench.evaluation import panels as P
    lab, feats, ve = ctx['nhan'], ctx['dac_trung'], ctx['ve']
    Y = lab['nhan'].astype(np.float64)
    pool, test = ve['be'], ve['test']
    target = lab['nhan'][test]
    cols = P.panel_columns(lab['ten_nhan'], lab['lop_chinh'], target)
    chance = SC.chance_level(target, cols)
    Ysh = Y[pool][np.random.default_rng(C.NEG_SEED).permutation(len(pool))]
    out = []
    for row in CB.NEG_ROWS:
        X = feats[row][C.PRIMARY_FEATURE].astype(np.float64)
        std = PR.standardizer(X[pool])
        for cfg in C.PROBES:
            pv, _ = NBP._probe(cfg, 1.0, X[pool], Ysh, X[test], std)
            real = P.point(ctx['du_doan'][(row, C.PRIMARY_FEATURE, cfg, 1.0, 0)], target, cols['p21'])['ap']
            sh = P.point(pv, target, cols['p21'])['ap']
            out.append(dict(hang=row, dau=cfg, map21_xao=sh, map21_that=real, may_rui=chance, dat=bool(sh <= chance + C.NEG_MAX_ABOVE_CHANCE)))
    ok = all(o['dat'] for o in out)
    log(f"doi chung am (nhan hoc bi xao, 100 %, {len(CB.NEG_ROWS)} hang): muc may rui mAP21 {chance:.3f} | " + ' | '.join(
        f"{CB.CODE[o['hang']]} {C.PROBE_TEXT[o['dau']]} xao {o['map21_xao']:.3f} (that {o['map21_that']:.3f})" for o in out)
        + f" | nguong <= may rui + {C.NEG_MAX_ABOVE_CHANCE:g}: {'DAT' if ok else 'TRUOT'}")
    assert ok, 'doi chung am TRUOT (nhan xao van cho mAP cao -> nghi ro nhan) -> DUNG, gui Claude dong tren'
    return out


def _cell_key(k):
    return '|'.join(str(x) for x in k)


def repro_gate(cells, res92b, tol=CB.REPRO_TOL):
    '''Ô tái lập của cặp RAD-DINO (tính lại bằng gói này từ tệp 92a) so với ô 92b đã ghi: điểm (4 panel x 3 số), CI95 AP21, hai nửa, từng lần rút, AP
       từng lớp. -> (đạt, [dict]).'''
    def _flat(x, pre=''):
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                out.update(_flat(v, f'{pre}{k}.'))
            return out
        if isinstance(x, (list, tuple)):
            out = {}
            for i, v in enumerate(x):
                out.update(_flat(v, f'{pre}{i}.'))
            return out
        return {pre[:-1]: x}
    rows = []
    for row in CB.RAD_PAIR:
        for cfg, f in CB.REPRO:
            k = (row, C.PRIMARY_FEATURE, cfg, f)
            a, b = cells.get(k), (res92b.get('o') or {}).get(_cell_key(k))
            if a is None or b is None:
                rows.append(dict(o=_cell_key(k), dat=False, ly_do='thieu o'))
                continue
            fa = _flat({t: a[t] for t in ('so_hat', 'diem', 'ci21', 'nua', 'tung_hat', 'ap21_lop')})
            fb = _flat({t: b.get(t) for t in ('so_hat', 'diem', 'ci21', 'nua', 'tung_hat', 'ap21_lop')})
            def _d(x, y):
                x, y = float(x), float(y)
                return 0.0 if (x != x and y != y) else (float('inf') if (x != x) != (y != y) else abs(x - y))   # NaN cả hai phía = trùng
            lech = [_d(fa[t], fb[t]) for t in fa if t in fb and fa[t] is not None and fb[t] is not None]
            ok = set(fa) == set(fb) and bool(lech) and max(lech) <= tol
            rows.append(dict(o=_cell_key(k), dat=bool(ok), lech_max=max(lech) if lech else None, so_so=len(lech), ap21=a['diem']['p21']['ap'], ap21_92b=b['diem']['p21']['ap']))
    return all(r['dat'] for r in rows), rows


def score(ctx, log=print, n_boot=None, workers=None):
    '''C5. Bootstrap + ô + cổng tái lập + họ 6 phép từng cặp + bảng. -> dict kết quả.'''
    from vindr_bench.evaluation import compare as CP
    from vindr_bench.evaluation import panels as P
    lab, preds, ve = ctx['nhan'], ctx['du_doan'], ctx['ve']
    target = lab['nhan'][ve['test']]
    cols = P.panel_columns(lab['ten_nhan'], lab['lop_chinh'], target)
    n_boot = n_boot or P.N_BOOT
    groups = {}
    for (row, feat, cfg, f, d), pv in preds.items():
        groups.setdefault((row, feat, cfg, f), {})[d] = pv
    prim = sorted(k for k in groups if SC.is_primary(k[1], k[2]))
    jobs, keys = [], []
    for k in prim:
        for d in sorted(groups[k]):
            jobs.append((groups[k][d], target, cols, n_boot, P.BOOT_SEED, P.HALF_SEED))
            keys.append((k, d))
    t0 = time.time()
    workers = workers or max(1, os.cpu_count() or 1)
    log(f"bootstrap {n_boot} lan theo anh (hat {P.BOOT_SEED}; hai nua hat {P.HALF_SEED}) cho {len(jobs)} bo du doan CHINH ({workers} tien trinh); CLS k-NN chi diem")
    stats = CP.many_stats(jobs, workers, log=log)
    by = {}
    for (k, d), st in zip(keys, stats):
        by.setdefault(k, []).append((d, st))
    cells = {k: SC.cell_from_stats([st for _, st in sorted(v, key=lambda x: x[0])]) for k, v in by.items()}
    for k in groups:
        if k not in cells:
            cells[k] = SC.cell_point([groups[k][d] for d in sorted(groups[k])], target, cols)
    ok, rep = repro_gate(cells, ctx['ket_qua_92b'])
    log(f"CONG TAI LAP (cap RAD-DINO, {len(rep)} o tinh lai tu tep 92a bang goi {__version__} = o cua 92b {ctx['tep_92b']}; nguong {CB.REPRO_TOL:g}): "
        + ' | '.join(f"{r['o'].replace('|mean_patch|', ' ')} {('lech ' + format(r['lech_max'], '.1e')) if r.get('lech_max') is not None else r.get('ly_do')}" for r in rep)
        + f" -> {'DAT' if ok else 'TRUOT'}")
    assert ok, 'cong tai lap TRUOT: goi 0.2.0 KHONG cho lai o cua 92b tren cung dac trung -> DUNG, gui Claude dong tren'
    fam = {e: SC.family_pair(cells, *CB.PAIRS[e]) for e in CB.ENCODERS}
    ctx.update(o=cells, cot=cols, ho=fam, tai_lap=rep, phut_cham=(time.time() - t0) / 60)
    _print_tables(ctx, log)
    return dict(ho=fam, tai_lap=rep)


def _rad_family(res92b):
    return res92b.get('ho') or {}


def _print_tables(ctx, log):
    cells, fam = ctx['o'], ctx['ho']
    log(f"\n=== Bang 1 · SAU HANG, dac trung {C.PRIMARY_FEATURE} ({C.FEATURE_TEXT[C.PRIMARY_FEATURE]}), test VinDr {len(ctx['ve']['test']):,} anh; moi o = TB cac lan "
        f"rut; [..] = CI95 danh nghia theo anh; SD = giua lan rut ===")
    log(f"{'hang':<20}{'dau doc':<11}{'nhan':>7}{'lan':>4}  {'mAP21 [CI95]':<24}{'SD':>7}{'AUPRC21':>9}{'AUROC21':>9}  hai nua AP21")
    for row in CB.ROWS:
        log(f'  {CB.TEXT[row]}')
        for probe in C.PROBES:
            for f in C.FRACTIONS:
                c = cells.get((row, C.PRIMARY_FEATURE, probe, f))
                if c is None:
                    continue
                dd = c['diem']['p21']
                sd = (c.get('sd') or {}).get('p21', {}).get('ap')
                log(f"{CB.CODE[row]:<20}{C.PROBE_TEXT[probe]:<11}{C.FRACTION_TEXT[f]:>7}{c['so_hat']:>4}  {SC.fmt(dd['ap'], 4)} [{SC.fmt(c['ci21'][0])}, "
                    f"{SC.fmt(c['ci21'][1])}]{'':<3}{SC.fmt(sd, 4):>7}{SC.fmt(dd['auprc_thang'], 4):>9}{SC.fmt(dd['auroc'], 4):>9}  {SC.fmt(c['nua'][0])} / "
                    f"{SC.fmt(c['nua'][1])}")
    for e in CB.ENCODERS:
        fe = fam[e]
        log(f"\n=== Bang 2 · LUAT (PLAN_92 13.3) {CB.PAIR_TEXT[e]} | HO {fe['so_phep']} PHEP CO − KHONG | {fe['co']} / {fe['so_phep']} phep co du lieu"
            + (f" -> khoang DONG THOI max-T {C.LEVEL:g} %, q = {fe['q']:.2f}" if fe['day_du'] else ' -> HO CHUA DU: chi hieu DANH NGHIA, CHUA ap luat') + ' ===')
        log(f"{'dau doc':<11}{'nhan':>7}  {'Δ AP / AUPRC / AUROC':<26}{'dong thoi Δ AP21':<19}{'danh nghia':<19}{'hai nua':<17}{'lan rut duong':<15}ket luan")
        for p in fe['phep_so']:
            ci_s = p['ci_dong_thoi']
            log(f"{C.PROBE_TEXT[p['dau']]:<11}{C.FRACTION_TEXT[p['ti_le']]:>7}  {SC.t3(p['d']):<26}"
                + (f"[{SC.sgn(ci_s[0])}, {SC.sgn(ci_s[1])}]  " if ci_s else f"{'-':<19}") + f"[{SC.sgn(p['ci_danh_nghia'][0])}, {SC.sgn(p['ci_danh_nghia'][1])}]  "
                f"{SC.sgn(p['nua'][0])} / {SC.sgn(p['nua'][1])}  {p['so_lan_rut_duong']} / {len(p['tung_lan_rut'])}{'':<9}{p['ket_luan']}")
        log('    Δ tung lan rut (AP21): ' + ' | '.join(f"{C.PROBE_TEXT[p['dau']]} {C.FRACTION_TEXT[p['ti_le']]}: {' '.join(SC.sgn(v) for v in p['tung_lan_rut'])}"
                                                    for p in fe['phep_so'] if len(p['tung_lan_rut']) > 1))
    rad = _rad_family(ctx['ket_qua_92b'])
    log(f"\n=== Bang 3 · BON BACKBONE (MO TA; moi cap la mot phep so TRONG backbone, he so ghi nguoc / epoch doc khac nhau giua cap — khong so cheo): CO − KHONG, "
        f"Δ AP21 [khoang dong thoi max-T cua ho tung cap] ket luan ===")
    hd = [(p, f) for p in C.PROBES for f in C.FRACTIONS]
    log(f"{'cap':<62}" + ''.join(f"{C.PROBE_TEXT[p] + ' ' + C.FRACTION_TEXT[f]:<30}" for p, f in hd))
    lines = [(CB.RAD_TEXT + ' [92b]', rad)] + [(CB.PAIR_TEXT[e], fam[e]) for e in CB.ENCODERS]
    for name, fe in lines:
        by = {(p['dau'], p['ti_le']): p for p in (fe.get('phep_so') or [])}
        cells_txt = []
        for p, f in hd:
            x = by.get((p, f))
            if x is None:
                cells_txt.append(f"{'-':<30}")
                continue
            ci = x.get('ci_dong_thoi') or x.get('ci_danh_nghia') or [float('nan')] * 2
            cells_txt.append(f"{SC.sgn(x['d']['ap'])} [{SC.sgn(ci[0])}, {SC.sgn(ci[1])}] {str(x['ket_luan'])[:6]:<6}"[:29].ljust(30))
        log(f"{name:<62}" + ''.join(cells_txt))
    log(f"\n=== Bang 4 · MO TA: k-NN tren CLS (chi diem), CO − KHONG tung cap ===")
    for e in CB.ENCODERS:
        a, b = CB.PAIRS[e]
        parts = []
        for f in C.FRACTIONS:
            ca, cb = cells.get((a, 'glob', 'knn', f)), cells.get((b, 'glob', 'knn', f))
            if ca is not None and cb is not None:
                parts.append(f"{C.FRACTION_TEXT[f]} {SC.t3(SC.d3(ca, cb))}")
        log(f"{CB.PAIR_TEXT[e]:<62}" + ' | '.join(parts))
    log('    doc THEO TUNG O cua TUNG backbone; KHONG cau chung, khong gop backbone. Mot seed backbone: ket luan duong can seed 43 (ca hai hang).')


def figures(ctx, log=print, show=True):
    '''C6. -> {tên: Figure}: họ 6 phép từng cặp (hàm family_ci của vindr_bench).'''
    from vindr_bench.evaluation import figures as FG
    figs = {}
    base = (f"VinDr test {len(ctx['ve']['test']):,} ảnh, 21 lớp chính; đặc trưng đông cứng = trung bình token patch tầng cuối; mỗi ô trung bình các lần rút "
            f"(1 % ×{C.N_DRAWS[0.01]}, 10 % ×{C.N_DRAWS[0.10]}, 100 % ×1; cùng chỉ số mọi hàng). k-NN: k = round(√n); logistic: L2, C = 1, giải tới hội tụ; không "
            f"chọn siêu tham số theo dữ liệu chấm. Một seed backbone (s42).")
    for e in CB.ENCODERS:
        fe = ctx['ho'][e]
        fi = {}
        for p in fe['phep_so']:
            lo, hi = p['ci_dong_thoi'] if p['ci_dong_thoi'] else p['ci_danh_nghia']
            fi[(C.FRACTION_CODE[p['ti_le']], p['dau'])] = dict(d=p['d']['ap'], lo=lo, hi=hi, dat=p['ket_luan'] in ('hon', 'kem'))
        f = FG.family_ci(fi, [C.FRACTION_CODE[f] for f in C.FRACTIONS], list(C.PROBES), {C.FRACTION_CODE[f]: f'{C.FRACTION_TEXT[f]} nhãn' for f in C.FRACTIONS},
                         C.PROBE_TEXT, f'{CB.PAIR_TEXT[e]} trên AP 21 lớp (bậc thang): họ {len(C.FAMILY)} phép của luật', f'{CB.TITLE[e]} CÓ − KHÔNG · AP 21',
                         base + (f" Vạch = khoảng ĐỒNG THỜI max-T 95 % của họ (q = {fe['q']:.2f}".replace('.', ',') + '); chấm đặc = đạt dạng luật ("hơn" / "kém": '
                                 '|hiệu| ≥ 0,01, khoảng đồng thời không chứa 0, cùng dấu ở hai nửa test).' if fe['day_du'] else
                                 ' Họ CHƯA đủ: vạch = CI95 danh nghĩa, chưa áp luật.') + ' Đường đứt: ±0,01. Đọc theo từng ô; kết luận dương cần seed 43.',
                         margin=C.MARGIN, labels=('đạt dạng luật ("hơn" / "kém")', 'không phân biệt'))
        if f is not None:
            figs[f'ho_6_{e}'] = f
    ctx['hinh'] = figs
    log(f'{len(figs)} hinh: {sorted(figs)}')
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return figs


def _rows_csv(ctx):
    out = []
    order = list(CB.ROWS) + list(CB.RAD_PAIR)
    for (row, feat, cfg, f), c in sorted(ctx['o'].items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1], kv[0][2], kv[0][3])):
        dd = c['diem']['p21']
        out.append(dict(bang='1_tuyet_doi' if row in CB.ROWS else '0_tai_lap_92b', hang=row, dieu_kien=CB.TEXT.get(row) or C.TEXT[row], dac_trung=feat, dau_doc=cfg,
                        ti_le_nhan=f, so_lan_rut=c['so_hat'], chieu_phep_tru='', AP21=dd['ap'], AUPRC21_hinh_thang=dd['auprc_thang'], AUROC21=dd['auroc'],
                        ci_thap=(c.get('ci21') or [None])[0], ci_cao=(c.get('ci21') or [None, None])[1], chi_diem=bool(c.get('chi_diem')), ket_luan=''))
    for e in CB.ENCODERS:
        a, b = CB.PAIRS[e]
        for p in ctx['ho'][e]['phep_so']:
            out.append(dict(bang='2_ho_6_luat', hang=f'{a} - {b}', dieu_kien=f'{CB.TEXT[a]} TRU {CB.TEXT[b]}', dac_trung=C.PRIMARY_FEATURE, dau_doc=p['dau'],
                            ti_le_nhan=p['ti_le'], so_lan_rut=len(p['tung_lan_rut']), chieu_phep_tru=f'{CB.CODE[a]} - {CB.CODE[b]}', AP21=p['d']['ap'],
                            AUPRC21_hinh_thang=p['d']['auprc_thang'], AUROC21=p['d']['auroc'], ci_thap=(p['ci_dong_thoi'] or [None])[0],
                            ci_cao=(p['ci_dong_thoi'] or [None, None])[1], chi_diem=False, ket_luan=p['ket_luan']))
    return out


def sync(ctx, log=print):
    '''C7. Ghi JSON + CSV + PNG + dự đoán (float16) rồi dong_bo().'''
    d, t = ctx['drive'], ctx['t']
    names = []
    cells = {_cell_key(k): {kk: v for kk, v in c.items() if kk not in ('m21', 'm7')} for k, c in ctx['o'].items()}
    obj = NBP._jsonable(dict(run=CB.RUN_TAGS['it_nhan'], nguon=CB.RUN_TAGS['trich'], nguon_rad=CB.SOURCE_RAD, tep_92b=ctx['tep_92b'], t=t, goi=ctx['goi'],
                             manifest_92d=dict(t=ctx['manifest'].get('t'), goi=ctx['manifest'].get('goi')), hang={r: CB.TEXT[r] for r in CB.ROWS},
                             cap={e: list(p) for e, p in CB.PAIRS.items()}, be_hoc=len(ctx['ve']['be']), test=len(ctx['ve']['test']),
                             md5_lan_rut=ctx['ve']['md5_lan_rut'], ti_le=list(C.FRACTIONS), so_lan_rut={str(k): v for k, v in C.N_DRAWS.items()}, o=cells,
                             ho=ctx['ho'], ho_rad_92b=_rad_family(ctx['ket_qua_92b']), tai_lap=ctx['tai_lap'], doi_chung_am=ctx['doi_chung_am'], hoi_tu=ctx['hoi_tu'],
                             ghi_chu='PLAN_92 13: luat = ho 6 phep CO - KHONG cho TUNG cap (DINOv3, CheXFound x1, BiomedCLIP), mAP21 (AP bac thang macro 21 lop), '
                                     'max-T 95 % trong ho cua cap; doc theo tung o; CLS k-NN + bang bon backbone = mo ta'))
    names.append(f'it_nhan_vindr_bb_{t}.json')
    d['ghi'](names[-1], lambda q: d['ghi_json'](q, obj))
    rows = _rows_csv(ctx)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    text = buf.getvalue()
    names.append(f'it_nhan_vindr_bb_bang_{t}.csv')

    def _csv(q):
        with open(q, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(text)
    d['ghi'](names[-1], _csv)
    for name, fig in ctx.get('hinh', {}).items():
        names.append(f'it_nhan_vindr_bb_hinh_{name}_{t}.png')
        d['ghi'](names[-1], lambda q, fig=fig: fig.savefig(q, dpi=160, bbox_inches='tight'))
    pv = {f"{r}|{fe}|{cfg}|{C.FRACTION_CODE[f]}|{dd}": v.astype(np.float16) for (r, fe, cfg, f, dd), v in ctx['du_doan'].items()}
    names.append(f'it_nhan_vindr_bb_du_doan_{t}.npz')
    d['ghi'](names[-1], lambda q: np.savez(q, **pv))
    du = d['dong_bo']()
    co = set(names) <= set(d['ten_tren_drive']())
    log(f"C7: da ghi {len(names)} tep (1 JSON, 1 CSV {len(rows)} hang, {len(ctx.get('hinh', {}))} PNG, du doan {len(pv)} bo) | dong_bo {'DU' if du else 'CHUA DU'} | "
        f"tren Drive: {'CO du' if co else 'THIEU'}" + ('' if du and co else ' | KHONG tat runtime; chay lai C7'))
    return dict(dong_bo=du, du_tep=co, tep=names)
