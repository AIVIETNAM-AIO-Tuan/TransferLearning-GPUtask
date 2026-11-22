# -*- coding: utf-8 -*-
"""Ô của notebook `92g_it_nhan_vindr_s43` (CPU; PLAN_90 §13.8). Giao thức = 92b / 92e NGUYÊN VẸN: cùng lần rút (md5 = 92b), cùng hai đầu đọc
(probe._probe), cùng bootstrap / ô / dạng luật (vindr_bench.evaluation qua scoring), cùng cổng tái lập (probe_bb.repro_gate). Chỉ khác: ba hàng s43 và ba
bảng đọc đặt trước của config_s43 ((A) luật §11.3 cho (3) − (0) s43; (B) xác nhận seed 43 theo B1 — Tuấn chọn 22/11, B2 chỉ in kèm; (C) khối VinDr của họ "hơn + không
kém" (3) − (1b) với δ đã duyệt).

    C2 setup    khối Drive + phiên bản hai gói + luật / δ (in ra)
    C3 load     92f: manifest (cổng seed 43 từng hàng) + nhãn (vân tay = các lượt 89) + 3 tệp đặc trưng s43 (md5 = manifest); 92a: nhãn (TRÙNG 92f) + 2 tệp
                cặp RAD-DINO s42 (cổng tái lập); 92b: JSON (ô + họ s42)
    C4 probes   lần rút (md5 = 92b) -> k-NN + logistic ở 1 / 10 / 100 % (3 hàng s43) + ô tái lập; CỔNG hội tụ; đối chứng âm (nhãn xáo, 100 %) cả ba hàng
    C5 score    bootstrap theo ảnh -> CỔNG tái lập -> Bảng 1 (6 hàng: 3 s43 + 3 s42 của 92b) · Bảng 2 ((A) + (B)) · Bảng 3 ((C)) · Bảng 4 ((1b) − (0), mô tả)
    C6 sync     JSON (+ lần rút bootstrap của các hiệu) + CSV + m21 từng ô + dự đoán (float16) -> dong_bo(); chạy lại được trong cùng phiên (tên tệp mới)

Chỉ in SỐ TỔNG HỢP."""
import csv
import io
import json
import os
import time

import numpy as np

from .. import __version__
from .. import config as C
from .. import config_s43 as S
from .. import features as FT
from .. import probes as PR
from .. import scoring as SC
from .. import subsets as SS
from . import probe as NBP
from . import probe_bb as PB

RES_PREFIX = PB.RES_PREFIX                     # JSON kết quả 92b: it_nhan_vindr_<t>.json
OUT_PREFIX = 'it_nhan_vindr_s43'


def setup(ns, log=print):
    '''C2. -> CTX.'''
    import scipy
    import vindr_bench
    import frozen_probe
    drive = NBP.drive_api(ns)
    assert drive['RUN_TAG'] == S.RUN_TAGS['it_nhan'], f"RUN_TAG {drive['RUN_TAG']!r} != {S.RUN_TAGS['it_nhan']!r} -> DUNG"
    pk = dict(frozen_probe=dict(phien_ban=__version__, sha=getattr(frozen_probe, 'FP_SHA', None)),
              vindr_bench=dict(phien_ban=vindr_bench.__version__, sha=getattr(vindr_bench, 'GOI_SHA', None)))
    log(f"92g | frozen_probe {__version__} @ {(pk['frozen_probe']['sha'] or '?')[:12]} + vindr_bench {pk['vindr_bench']['phien_ban']} @ "
        f"{(pk['vindr_bench']['sha'] or '?')[:12]} | numpy {np.__version__} | scipy {scipy.__version__} | {os.cpu_count()} CPU")
    log('hang: ' + ' | '.join(S.TEXT[r] for r in S.ROWS) + f" | s42 cung vi tri lay tu 92b ({S.SOURCE_S42['it_nhan']})")
    log(f"(A) luat dat truoc PLAN_92 11.3 cho (3) - (0) s43: ho {len(C.FAMILY)} phep x {{k-NN, logistic}} x {{1 %, 10 %, 100 %}} tren mAP21 (AP bac thang macro 21 "
        f"lop), dac trung {C.PRIMARY_FEATURE}; max-T {C.LEVEL:g} % trong ho; 'hon' = hieu >= +{C.MARGIN:g} VA can duoi dong thoi > 0 VA duong o hai nua test")
    log(f'(B) xac nhan seed 43 o s42 "hon" — luat = {S.CONFIRM_RULE} (Tuan chon 22/11, truoc khi co so s43): ' + ' | '.join(f'{k}: {S.CONFIRM_TEXT[k]}' for k in S.CONFIRM))
    log(f"(C) khoi VinDr cua ho 'hon + khong kem' (3) - (1b) (PLAN_90 13.4; {S.NONINF_CELLS} o = 3 CXR-LT + {S.NONINF_VINDR} VinDr): max-T TRONG khoi VinDr o muc "
        f"0.95^({S.NONINF_VINDR}/{S.NONINF_CELLS}) = {S.NONINF_LEVEL:.2f} % (khoi CXR-LT {S.CXR_LEVEL:.2f} % — buoc sau); 'khong kem' = can duoi > -delta o MOI o; "
        f"delta (Tuan duyet 22/11, tu nua do rong dong thoi 92b x {S.Z_SIDAK9} / {S.Q_92B}): "
        + ', '.join(f"{C.PROBE_TEXT[p]} {C.FRACTION_TEXT[f]} {S.DELTA[(p, f)]:g}" for p, f in C.FAMILY))
    return dict(drive=drive, goi=pk, t=time.strftime('%Y%m%d_%H%M%S'))


def load(ctx, log=print):
    '''C3. -> dict(nhan, dac_trung).'''
    from vindr_bench.data import vindr as V
    d = ctx['drive']
    srcf = (S.FOLDER, S.RUN_TAGS['trich'])
    kk = d['kiem_ke'](srcf)[1]
    assert FT.MANIFEST_FILE in kk and FT.LABEL_FILE in kk, f'chua co ket qua 92f trong {"/".join(srcf)} -> chay 92f truoc. DUNG'
    man = json.load(open(d['tai'](kk[FT.MANIFEST_FILE][0]), encoding='utf-8'))
    assert man.get('run') == S.RUN_TAGS['trich'] and list(man.get('thu_tu_hang') or []) == list(S.ROWS), f"manifest 92f {man.get('run')} / {man.get('thu_tu_hang')} -> DUNG"
    bad = [r for r in S.ROWS if not ((man['hang'][r].get('cong_seed43') or {}).get('dat'))]
    assert not bad, f'manifest 92f: cong seed 43 KHONG dat o {bad} -> DUNG'
    lab = PB._labels(d, kk, man, '92f')
    n = len(lab['nhan'])
    log(f"nhan 92f: {n:,} anh x {lab['nhan'].shape[1]} nhan, {len(lab['lop_chinh'])} lop so chinh | tach " + ' / '.join(
        f"{s} {int((lab['tach'] == s).sum()):,}" for s in V.SPLITS) + ' | van tay = cac luot 89: TRUNG | md5 nhan tinh lai = van tay: DAT')
    feats = {}
    for row in S.ROWS:
        feats[row], m5 = PB._feature(d, kk, man, row, n, '92f')
        h = man['hang'][row]
        log(f"  {S.CODE[row]:<10}{S.TEXT[row]}: {n:,} x {feats[row][C.PRIMARY_FEATURE].shape[1]} | md5 {m5[:12]} = manifest | cong seed 43 "
            f"{h['cong_seed43']['run']} / {'/'.join(h['cong_seed43']['ghep_cap'])}: DAT | nap: {h['nap'][:60]}")
    srca = (C.FOLDER, S.SOURCE_S42['trich'])
    ka = d['kiem_ke'](srca)[1]
    assert FT.MANIFEST_FILE in ka and FT.LABEL_FILE in ka, f'thieu ket qua 92a trong {"/".join(srca)} (can cho cong tai lap) -> DUNG'
    mana = json.load(open(d['tai'](ka[FT.MANIFEST_FILE][0]), encoding='utf-8'))
    laba = PB._labels(d, ka, mana, '92a')
    same = all(np.array_equal(lab[k], laba[k]) for k in ('nhan', 'tach')) and lab['ten_nhan'] == laba['ten_nhan'] and lab['lop_chinh'] == laba['lop_chinh']
    assert same, 'nhan / tach cua 92f KHAC 92a (khong cung thu tu anh) -> DUNG'
    rad = {}
    for row in C.PAIR:
        rad[row], m5 = PB._feature(d, ka, mana, row, n, '92a')
        log(f"  {C.CODE[row]:<10}{C.TEXT[row]} (92a, cho cong tai lap): md5 {m5[:12]} = manifest 92a")
    srcb = (C.FOLDER, S.SOURCE_S42['it_nhan'])
    kb = d['kiem_ke'](srcb)[1]
    js = sorted(t for t in kb if t.startswith(RES_PREFIX) and t.endswith('.json') and t[len(RES_PREFIX):-5].replace('_', '').isdigit())
    assert js, f'thieu JSON ket qua 92b trong {"/".join(srcb)} (can cho cong tai lap + so s42) -> DUNG'
    res92b = json.load(open(d['tai'](kb[js[-1]][0]), encoding='utf-8'))
    miss = [_key(b, p, f) for b in S.BASE.values() for p, f in C.FAMILY if _key(b, p, f) not in (res92b.get('o') or {})]
    assert not miss and (res92b.get('ho') or {}).get('day_du'), f'92b thieu o s42 {miss[:3]} / ho chua du -> DUNG'
    log(f"92a / 92b: nhan TRUNG 92f (cung thu tu anh) | 92b {js[-1]} (lan rut md5 {res92b['md5_lan_rut'][:12]}, goi {res92b['goi']['frozen_probe']['phien_ban']}) | "
        f"du {len(S.BASE) * len(C.FAMILY)} o s42 + ho 6 phep s42")
    ctx.update(manifest=man, nhan=lab, dac_trung=feats, rad=rad, ket_qua_92b=res92b, tep_92b=js[-1])
    return dict(nhan=lab, dac_trung=feats)


def _key(row, probe, f):
    return PB._cell_key((row, C.PRIMARY_FEATURE, probe, f))


def probes(ctx, log=print):
    '''C4. -> {(hàng, đặc trưng, cấu hình, tỉ lệ, lần rút): dự đoán test [m, 28] float32}.'''
    lab, feats = ctx['nhan'], ctx['dac_trung']
    Y = lab['nhan'].astype(np.float64)
    pool = np.flatnonzero(np.isin(lab['tach'], C.POOL_SPLITS))
    test = np.flatnonzero(lab['tach'] == C.TEST_SPLIT)
    dr = SS.all_draws(pool)
    m5 = SS.md5(dr)
    assert m5 == ctx['ket_qua_92b']['md5_lan_rut'], f'md5 lan rut {m5[:12]} KHAC 92b ({ctx["ket_qua_92b"]["md5_lan_rut"][:12]}) -> khong ghep cap voi s42. DUNG'
    log(f"be hoc {len(pool):,} anh | test {len(test):,} | lan rut: " + ', '.join(f"{C.FRACTION_TEXT[f]} = {SS.size(len(pool), f):,} anh x {C.N_DRAWS[f]} lan"
                                                                            for f in C.FRACTIONS) + f" | md5 chi so {m5[:12]} = 92b | hat {C.DRAW_SEED}")
    preds, conv, t0 = {}, [], time.time()
    jobs = [(row, feats[row], [(fe, cfg, f) for fe, cfg in S.CONFIGS for f in C.FRACTIONS]) for row in S.ROWS]
    jobs += [(row, ctx['rad'][row], [(C.PRIMARY_FEATURE, cfg, f) for cfg, f in S.REPRO]) for row in C.PAIR]
    for row, fr, todo in jobs:
        t1 = time.time()
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
        name = S.CODE.get(row) or C.CODE[row] + ' s42 (tai lap)'
        log(f"  {name:<22}xong {sum(1 for k in preds if k[0] == row)} bo du doan | {(time.time() - t1) / 60:.1f} phut | tong {(time.time() - t0) / 60:.1f} phut")
    bad = [c for c in conv if not c['hoi_tu']]
    log(f"logistic: {len(conv)} lan fit | so vong lon nhat {max(c['so_vong'] for c in conv)} | |grad| lon nhat {max(c['grad_max'] for c in conv):.1e} | "
        f"hoi tu {'DAT' if not bad else f'TRUOT {len(bad)} lan'} | tong {(time.time() - t0) / 60:.1f} phut")
    assert not bad, f'logistic KHONG hoi tu o {[(b["hang"], b["cau_hinh"], b["ti_le"], b["lan_rut"]) for b in bad][:5]} -> DUNG, gui Claude dong tren'
    ctx.update(du_doan=preds, hoi_tu=conv, ve=dict(be=pool, test=test, md5_lan_rut=m5))
    ctx['doi_chung_am'] = negative_control(ctx, log)
    return preds


def negative_control(ctx, log=print):
    '''= probe_bb.negative_control cho ba hàng s43: nhãn học XÁO (hạt NEG_SEED), 100 %, đặc trưng chính -> mAP21 test phải về mức may rủi.'''
    from vindr_bench.evaluation import panels as P
    lab, feats, ve = ctx['nhan'], ctx['dac_trung'], ctx['ve']
    Y = lab['nhan'].astype(np.float64)
    pool, test = ve['be'], ve['test']
    target = lab['nhan'][test]
    cols = P.panel_columns(lab['ten_nhan'], lab['lop_chinh'], target)
    chance = SC.chance_level(target, cols)
    Ysh = Y[pool][np.random.default_rng(C.NEG_SEED).permutation(len(pool))]
    out = []
    for row in S.NEG_ROWS:
        X = feats[row][C.PRIMARY_FEATURE].astype(np.float64)
        std = PR.standardizer(X[pool])
        for cfg in C.PROBES:
            pv, _ = NBP._probe(cfg, 1.0, X[pool], Ysh, X[test], std)
            real = P.point(ctx['du_doan'][(row, C.PRIMARY_FEATURE, cfg, 1.0, 0)], target, cols['p21'])['ap']
            sh = P.point(pv, target, cols['p21'])['ap']
            out.append(dict(hang=row, dau=cfg, map21_xao=sh, map21_that=real, may_rui=chance, dat=bool(sh <= chance + C.NEG_MAX_ABOVE_CHANCE)))
    ok = all(o['dat'] for o in out)
    log(f"doi chung am (nhan hoc bi xao, 100 %, {len(S.NEG_ROWS)} hang): muc may rui mAP21 {chance:.3f} | " + ' | '.join(
        f"{S.CODE[o['hang']]} {C.PROBE_TEXT[o['dau']]} xao {o['map21_xao']:.3f} (that {o['map21_that']:.3f})" for o in out)
        + f" | nguong <= may rui + {C.NEG_MAX_ABOVE_CHANCE:g}: {'DAT' if ok else 'TRUOT'}")
    assert ok, 'doi chung am TRUOT (nhan xao van cho mAP cao -> nghi ro nhan) -> DUNG, gui Claude dong tren'
    return out


def _find(fam, probe, f):
    x = [p for p in (fam.get('phep_so') or []) if p['dau'] == probe and abs(float(p['ti_le']) - f) < 1e-12]
    assert len(x) == 1, f'ho: {len(x)} phep {probe} {f} -> DUNG'
    return x[0]


def _d42(res92b, a, b, probe, f):
    '''Hiệu ĐIỂM s42 (AP / AUPRC / AUROC trên p21) giữa hai hàng s42 từ ô của 92b.'''
    oa, ob = res92b['o'][_key(a, probe, f)], res92b['o'][_key(b, probe, f)]
    return {m: float(oa['diem']['p21'][m] - ob['diem']['p21'][m]) for m in ('ap', 'auprc_thang', 'auroc')}


def confirm(fam43, fam42):
    '''(A) + (B): từng ô (3) − (0) — kết luận luật §11.3 ở s42 (92b) và s43; xác nhận B1 / B2 cho ô s42 "hơn". -> list dict.'''
    out = []
    for probe, f in C.FAMILY:
        p, q = _find(fam43, probe, f), _find(fam42, probe, f)
        d43, d42, ci = p['d']['ap'], q['d']['ap'], p['ci_danh_nghia']
        target = q['ket_luan'] == 'hon'
        b1 = p['ket_luan'] == 'hon'
        b2 = bool(d43 * d42 > 0 and (ci[0] > 0 or ci[1] < 0))
        out.append(dict(dau=probe, ti_le=f, d42=q['d'], ket_luan42=q['ket_luan'], ci42=q['ci_dong_thoi'], d43=p['d'], ket_luan43=p['ket_luan'],
                        ci43=p['ci_dong_thoi'], ci43_danh_nghia=ci, nua43=p['nua'], thuoc_dien=target,
                        B1=('xac nhan' if b1 else 'KHONG xac nhan') if target else 'ngoai dien (s42 khong "hon")',
                        B2=('xac nhan' if b2 else 'KHONG xac nhan') if target else 'ngoai dien (s42 khong "hon")'))
    return out


def noninferiority(cells, res92b):
    '''(C): 6 ô (3) − (1b) s43 trên mAP21; max-T trong khối VinDr ở mức NONINF_LEVEL; "không kém" = cận dưới > −δ; "hơn" = dạng luật §11.3 trên khoảng
       này. Kèm hiệu điểm s42 (92b) và lần rút bootstrap (cho khối CXR-LT / gộp họ 9 ô sau). -> dict.'''
    from vindr_bench.evaluation import compare as CP
    items = []
    for probe, f in C.FAMILY:
        a, b = cells[(S.REF, C.PRIMARY_FEATURE, probe, f)], cells[(S.NONINF_CONTROL, C.PRIMARY_FEATURE, probe, f)]
        items.append((probe, f, a, b, CP.diff(a, b)))
    q, sims = CP.max_t([it[4] for it in items], level=S.NONINF_LEVEL)
    rec = []
    for (probe, f, a, b, r), s in zip(items, sims):
        lo, hi = s['ci_dong_thoi']
        de = S.DELTA[(probe, f)]
        pd = SC.per_draw(a, b)
        rec.append(dict(dau=probe, ti_le=f, d=SC.d3(a, b), ci_khoi=[lo, hi], ci_danh_nghia=r['ci'], nua=r['nua'], delta=de, khong_kem=bool(lo > -de),
                        ket_luan_hon=s['ket_luan'], d42=_d42(res92b, S.BASE[S.REF], S.BASE[S.NONINF_CONTROL], probe, f), tung_lan_rut=pd,
                        so_lan_rut_duong=int(sum(v > 0 for v in pd)), rut=[None if v != v else float(v) for v in r['rut']]))
    return dict(q=float(q), muc=S.NONINF_LEVEL, so_o_khoi=len(rec), so_o_ho=S.NONINF_CELLS, phep_so=rec, khong_kem_moi_o=bool(all(x['khong_kem'] for x in rec)))


def descriptive(cells, res92b):
    '''(1b) − (0) s43, CI95 danh nghĩa (mô tả) + hiệu điểm s42.'''
    from vindr_bench.evaluation import compare as CP
    out = []
    for probe, f in C.FAMILY:
        a, b = cells[(S.NONINF_CONTROL, C.PRIMARY_FEATURE, probe, f)], cells[(S.CONTROL, C.PRIMARY_FEATURE, probe, f)]
        r = CP.diff(a, b)
        out.append(dict(dau=probe, ti_le=f, d=SC.d3(a, b), ci_danh_nghia=r['ci'], nua=r['nua'], d42=_d42(res92b, S.BASE[S.NONINF_CONTROL], S.BASE[S.CONTROL], probe, f)))
    return out


def score(ctx, log=print, n_boot=None, workers=None):
    '''C5. Bootstrap + ô + cổng tái lập + (A) / (B) / (C) + bảng. -> dict kết quả.'''
    from vindr_bench.evaluation import compare as CP
    from vindr_bench.evaluation import panels as P
    lab, preds, ve = ctx['nhan'], ctx['du_doan'], ctx['ve']
    target = lab['nhan'][ve['test']]
    cols = P.panel_columns(lab['ten_nhan'], lab['lop_chinh'], target)
    n_boot = n_boot or P.N_BOOT
    groups = {}
    for (row, feat, cfg, f, d), pv in preds.items():
        groups.setdefault((row, feat, cfg, f), {})[d] = pv
    jobs, keys = [], []
    for k in sorted(groups):
        for d in sorted(groups[k]):
            jobs.append((groups[k][d], target, cols, n_boot, P.BOOT_SEED, P.HALF_SEED))
            keys.append((k, d))
    t0 = time.time()
    workers = workers or max(1, os.cpu_count() or 1)
    log(f"bootstrap {n_boot} lan theo anh (hat {P.BOOT_SEED}; hai nua hat {P.HALF_SEED}) cho {len(jobs)} bo du doan ({workers} tien trinh)")
    stats = CP.many_stats(jobs, workers, log=log)
    by = {}
    for (k, d), st in zip(keys, stats):
        by.setdefault(k, []).append((d, st))
    cells = {k: SC.cell_from_stats([st for _, st in sorted(v, key=lambda x: x[0])]) for k, v in by.items()}
    res = ctx['ket_qua_92b']
    ok, rep = PB.repro_gate(cells, res, tol=S.REPRO_TOL)
    log(f"CONG TAI LAP (cap RAD-DINO s42, {len(rep)} o tinh lai tu tep 92a bang goi {__version__} = o cua 92b {ctx['tep_92b']}; nguong {S.REPRO_TOL:g}): "
        + ' | '.join(f"{r['o'].replace('|mean_patch|', ' ')} {('lech ' + format(r['lech_max'], '.1e')) if r.get('lech_max') is not None else r.get('ly_do')}"
                     for r in rep) + f" -> {'DAT' if ok else 'TRUOT'}")
    assert ok, f'cong tai lap TRUOT: goi {__version__} KHONG cho lai o cua 92b tren cung dac trung -> DUNG, gui Claude dong tren'
    fam = SC.family_pair(cells, S.REF, S.CONTROL)
    assert fam['day_du'], 'ho (3) - (0) s43 chua du 6 phep -> DUNG'
    conf = confirm(fam, res['ho'])
    ni = noninferiority(cells, res)
    desc = descriptive(cells, res)
    ctx.update(o=cells, cot=cols, ho=fam, xac_nhan=conf, khong_kem=ni, mo_ta=desc, tai_lap=rep, phut_cham=(time.time() - t0) / 60)
    _print_tables(ctx, log)
    return dict(ho=fam, xac_nhan=conf, khong_kem=ni, mo_ta=desc, tai_lap=rep)


def _ci(c):
    return f"[{SC.sgn(c[0])}, {SC.sgn(c[1])}]" if c else '-'


def _print_tables(ctx, log):
    cells, res = ctx['o'], ctx['ket_qua_92b']
    log(f"\n=== Bang 1 · SAU HANG (3 s43 o day + 3 s42 cua 92b), dac trung {C.PRIMARY_FEATURE}, test VinDr {len(ctx['ve']['test']):,} anh; moi o = TB cac lan rut; "
        f"[..] = CI95 danh nghia theo anh ===")
    log(f"{'hang':<12}{'dau doc':<11}{'nhan':>7}  {'mAP21 [CI95]':<24}{'AUPRC21':>9}{'AUROC21':>9}  hai nua AP21")
    for r43 in S.ROWS:
        for row, txt, get in ((r43, S.TEXT[r43], lambda p, f, r=r43: cells.get((r, C.PRIMARY_FEATURE, p, f))),
                              (S.BASE[r43], S.TEXT42[S.BASE[r43]] + ' [92b]', lambda p, f, r=S.BASE[r43]: res['o'].get(_key(r, p, f)))):
            log(f'  {txt}')
            for probe in C.PROBES:
                for f in C.FRACTIONS:
                    c = get(probe, f)
                    if c is None:
                        continue
                    dd = c['diem']['p21']
                    code = S.CODE.get(row) or S.CODE42[row]
                    log(f"{code:<12}{C.PROBE_TEXT[probe]:<11}{C.FRACTION_TEXT[f]:>7}  {SC.fmt(dd['ap'], 4)} [{SC.fmt(c['ci21'][0])}, {SC.fmt(c['ci21'][1])}]{'':<3}"
                        f"{SC.fmt(dd['auprc_thang'], 4):>9}{SC.fmt(dd['auroc'], 4):>9}  {SC.fmt(c['nua'][0])} / {SC.fmt(c['nua'][1])}")
    fam = ctx['ho']
    log(f"\n=== Bang 2 · (A) LUAT PLAN_92 11.3, (3) - (0) (hieu = (3) tru (0), mAP21 = AP bac thang macro 21 lop; Δ AP / AUPRC hinh thang / AUROC): ho 6 phep s43, "
        f"max-T {C.LEVEL:g} %, q = {fam['q']:.2f} | cot s42 = 92b (q {res['ho']['q']:.2f}) | (B) xac nhan seed 43 cho o s42 'hon': LUAT = {S.CONFIRM_RULE}, cot B2 chi in kem ===")
    log(f"{'dau doc':<11}{'nhan':>7}  {'s42 Δ AP [dong thoi]':<27}{'s42':<17}{'s43 Δ AP / AUPRC / AUROC':<27}{'s43 dong thoi':<19}{'s43 danh nghia':<19}"
        f"{'s43 hai nua':<17}{'s43':<17}{'B1 (luat)':<16}B2 (kem)")
    for x in ctx['xac_nhan']:
        log(f"{C.PROBE_TEXT[x['dau']]:<11}{C.FRACTION_TEXT[x['ti_le']]:>7}  {SC.sgn(x['d42']['ap'])} {_ci(x['ci42']):<20}{x['ket_luan42']:<17}{SC.t3(x['d43']):<27}"
            f"{_ci(x['ci43']):<19}{_ci(x['ci43_danh_nghia']):<19}{SC.sgn(x['nua43'][0])} / {SC.sgn(x['nua43'][1])}  {x['ket_luan43']:<17}{x['B1']:<16}{x['B2']}")
    log('    Δ tung lan rut s43 (AP21): ' + ' | '.join(f"{C.PROBE_TEXT[p['dau']]} {C.FRACTION_TEXT[p['ti_le']]}: {' '.join(SC.sgn(v) for v in p['tung_lan_rut'])}"
                                                    for p in fam['phep_so'] if len(p['tung_lan_rut']) > 1))
    ni = ctx['khong_kem']
    log(f"\n=== Bang 3 · (C) KHOI VinDr cua ho 'hon + khong kem' (3) - (1b) (hieu = (3) tru (1b), mAP21): max-T trong khoi {ni['so_o_khoi']} o o muc "
        f"{ni['muc']:.2f} %, q = {ni['q']:.2f}; 'khong kem' = can duoi > -delta; 'hon' = dang luat 11.3 tren khoang nay; khoi CXR-LT 3 o: buoc sau ===")
    log(f"{'dau doc':<11}{'nhan':>7}  {'s42 Δ AP':<10}{'s43 Δ AP / AUPRC / AUROC':<27}{'khoang khoi':<19}{'danh nghia':<19}{'hai nua':<17}{'delta':>7}  "
        f"{'khong kem':<11}hon")
    for x in ni['phep_so']:
        log(f"{C.PROBE_TEXT[x['dau']]:<11}{C.FRACTION_TEXT[x['ti_le']]:>7}  {SC.sgn(x['d42']['ap']):<10}{SC.t3(x['d']):<27}{_ci(x['ci_khoi']):<19}"
            f"{_ci(x['ci_danh_nghia']):<19}{SC.sgn(x['nua'][0])} / {SC.sgn(x['nua'][1])}  {x['delta']:>7.3f}  {('DAT' if x['khong_kem'] else 'TRUOT'):<11}{x['ket_luan_hon']}")
    log(f"    khoi VinDr: 'khong kem' o MOI o: {'DAT' if ni['khong_kem_moi_o'] else 'TRUOT'} (ket luan ca ho can them khoi CXR-LT)")
    log(f"\n=== Bang 4 · MO TA: (1b) - (0) s43 (hieu = (1b) tru (0), mAP21; CI95 danh nghia) + hieu diem s42 (92b) ===")
    for x in ctx['mo_ta']:
        log(f"{C.PROBE_TEXT[x['dau']]:<11}{C.FRACTION_TEXT[x['ti_le']]:>7}  s42 {SC.sgn(x['d42']['ap'])} | s43 {SC.t3(x['d'])} {_ci(x['ci_danh_nghia'])} | hai nua "
            f"{SC.sgn(x['nua'][0])} / {SC.sgn(x['nua'][1])}")
    log(f'    doc THEO TUNG O; (B) ket luan theo {S.CONFIRM_RULE} (B2 chi in kem); (C) chua phai ket luan ca ho (thieu khoi CXR-LT).')


def _rows_csv(ctx):
    out = []
    for (row, feat, cfg, f), c in sorted(ctx['o'].items(), key=lambda kv: (str(kv[0][0]), kv[0][2], kv[0][3])):
        dd = c['diem']['p21']
        out.append(dict(bang='1_tuyet_doi' if row in S.ROWS else '0_tai_lap_92b', hang=row, dieu_kien=S.TEXT.get(row) or C.TEXT[row], dau_doc=cfg, ti_le_nhan=f,
                        so_lan_rut=c['so_hat'], chieu_phep_tru='', AP21=dd['ap'], AUPRC21_hinh_thang=dd['auprc_thang'], AUROC21=dd['auroc'],
                        ci_thap=c['ci21'][0], ci_cao=c['ci21'][1], ket_luan=''))
    for x in ctx['xac_nhan']:
        out.append(dict(bang='2_A_luat_3_tru_0_s43', hang=f'{S.REF} - {S.CONTROL}', dieu_kien=f'{S.TEXT[S.REF]} TRU {S.TEXT[S.CONTROL]}', dau_doc=x['dau'],
                        ti_le_nhan=x['ti_le'], so_lan_rut=C.N_DRAWS[x['ti_le']], chieu_phep_tru='(3) s43 - (0) s43', AP21=x['d43']['ap'],
                        AUPRC21_hinh_thang=x['d43']['auprc_thang'], AUROC21=x['d43']['auroc'], ci_thap=x['ci43'][0], ci_cao=x['ci43'][1],
                        ket_luan=f"{x['ket_luan43']} | s42 {x['ket_luan42']} | B1 {x['B1']} | B2 {x['B2']}"))
    for x in ctx['khong_kem']['phep_so']:
        out.append(dict(bang='3_C_khoi_vindr_3_tru_1b_s43', hang=f'{S.REF} - {S.NONINF_CONTROL}', dieu_kien=f'{S.TEXT[S.REF]} TRU {S.TEXT[S.NONINF_CONTROL]}',
                        dau_doc=x['dau'], ti_le_nhan=x['ti_le'], so_lan_rut=C.N_DRAWS[x['ti_le']], chieu_phep_tru='(3) s43 - (1b) s43', AP21=x['d']['ap'],
                        AUPRC21_hinh_thang=x['d']['auprc_thang'], AUROC21=x['d']['auroc'], ci_thap=x['ci_khoi'][0], ci_cao=x['ci_khoi'][1],
                        ket_luan=f"khong kem {'DAT' if x['khong_kem'] else 'TRUOT'} (delta {x['delta']:g}) | {x['ket_luan_hon']}"))
    for x in ctx['mo_ta']:
        out.append(dict(bang='4_mo_ta_1b_tru_0_s43', hang=f'{S.NONINF_CONTROL} - {S.CONTROL}', dieu_kien=f'{S.TEXT[S.NONINF_CONTROL]} TRU {S.TEXT[S.CONTROL]}',
                        dau_doc=x['dau'], ti_le_nhan=x['ti_le'], so_lan_rut=C.N_DRAWS[x['ti_le']], chieu_phep_tru='(1b) s43 - (0) s43', AP21=x['d']['ap'],
                        AUPRC21_hinh_thang=x['d']['auprc_thang'], AUROC21=x['d']['auroc'], ci_thap=x['ci_danh_nghia'][0], ci_cao=x['ci_danh_nghia'][1],
                        ket_luan='mo ta (CI95 danh nghia)'))
    return out


def sync(ctx, log=print):
    '''C6. Ghi JSON + CSV + m21 từng ô + dự đoán (float16) rồi dong_bo(). Chạy lại trong cùng phiên: tên tệp có hậu tố lần ghi (không trùng tên).'''
    d = ctx['drive']
    ctx['lan_ghi'] = ctx.get('lan_ghi', 0) + 1
    t = ctx['t'] + ('' if ctx['lan_ghi'] == 1 else f"_lan{ctx['lan_ghi']}")
    names = []
    cells = {PB._cell_key(k): {kk: v for kk, v in c.items() if kk not in ('m21', 'm7')} for k, c in ctx['o'].items()}
    obj = NBP._jsonable(dict(run=S.RUN_TAGS['it_nhan'], nguon=S.RUN_TAGS['trich'], nguon_s42=S.SOURCE_S42, tep_92b=ctx['tep_92b'], t=t, goi=ctx['goi'],
                             manifest_92f=dict(t=ctx['manifest'].get('t'), goi=ctx['manifest'].get('goi'),
                                               cong_seed43={r: ctx['manifest']['hang'][r].get('cong_seed43') for r in S.ROWS}),
                             hang={r: S.TEXT[r] for r in S.ROWS}, hang_s42=dict(S.BASE), be_hoc=len(ctx['ve']['be']), test=len(ctx['ve']['test']),
                             md5_lan_rut=ctx['ve']['md5_lan_rut'], ti_le=list(C.FRACTIONS), so_lan_rut={str(k): v for k, v in C.N_DRAWS.items()}, o=cells,
                             ho_A=ctx['ho'], ho_A_s42_92b=ctx['ket_qua_92b']['ho'], xac_nhan_B=ctx['xac_nhan'], cach_B=S.CONFIRM_TEXT, luat_B=S.CONFIRM_RULE, khong_kem_C=ctx['khong_kem'],
                             delta=[dict(dau=p, ti_le=f, delta=S.DELTA[(p, f)]) for p, f in C.FAMILY], mo_ta_1b_tru_0=ctx['mo_ta'], tai_lap=ctx['tai_lap'],
                             doi_chung_am=ctx['doi_chung_am'], hoi_tu=ctx['hoi_tu'],
                             ghi_chu='PLAN_90 13.8: (A) luat PLAN_92 11.3 cho (3) - (0) s43; (B) xac nhan seed 43 theo B1 (Tuan chon 22/11; B2 in kem); (C) khoi VinDr cua ho hon + '
                                     'khong kem (3) - (1b), max-T trong khoi o muc 0.95^(6/9), delta da duyet; khoi CXR-LT buoc sau'))
    names.append(f'{OUT_PREFIX}_{t}.json')
    d['ghi'](names[-1], lambda q: d['ghi_json'](q, obj))
    rows = _rows_csv(ctx)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    text = buf.getvalue()
    names.append(f'{OUT_PREFIX}_bang_{t}.csv')

    def _csv(q):
        with open(q, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(text)
    d['ghi'](names[-1], _csv)
    m21 = {PB._cell_key(k): np.asarray(c['m21'], dtype=np.float64) for k, c in ctx['o'].items()}
    names.append(f'{OUT_PREFIX}_m21_{t}.npz')
    d['ghi'](names[-1], lambda q: np.savez(q, **m21))
    pv = {f"{r}|{fe}|{cfg}|{C.FRACTION_CODE[f]}|{dd}": v.astype(np.float16) for (r, fe, cfg, f, dd), v in ctx['du_doan'].items()}
    names.append(f'{OUT_PREFIX}_du_doan_{t}.npz')
    d['ghi'](names[-1], lambda q: np.savez(q, **pv))
    du = d['dong_bo']()
    co = set(names) <= set(d['ten_tren_drive']())
    log(f"C6: da ghi {len(names)} tep (1 JSON, 1 CSV {len(rows)} hang, m21 {len(m21)} o, du doan {len(pv)} bo) | dong_bo {'DU' if du else 'CHUA DU'} | "
        f"tren Drive: {'CO du' if co else 'THIEU'}" + ('' if du and co else ' | KHONG tat runtime; chay lai C6'))
    return dict(dong_bo=du, du_tep=co, tep=names)
