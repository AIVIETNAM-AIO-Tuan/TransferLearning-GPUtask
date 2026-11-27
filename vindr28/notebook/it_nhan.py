# -*- coding: utf-8 -*-
"""Ô của notebook `94b_vindr28_it_nhan` (Colab CPU; PLAN_94 §4): chấm lại trên 28 nhãn dự đoán test của phép kiểm ít nhãn (0.2.0: bản FLOAT32 do 94c tính lại; không dùng tệp float16 của 92) — 92b (s42, 8 hàng),
92e (4 backbone: 3 cặp CÓ / KHÔNG luồng vùng), 92g (s43: (3), (1b), (0)) — đặc trưng chính (TB token patch), k-NN + logistic ở 1 / 10 / 100 % nhãn. Họ phép so
gọi lại ĐÚNG hàm của frozen_probe (scoring.family_pair / descriptive: họ 6 phép, max-T 95 %) trên ô 28 nhãn.

    C2 setup     khối Drive + phiên bản ba gói
    C3 load      nhãn VinDr của 92a (md5 = manifest, vân tay = các lượt 89) + dự đoán + JSON của 92b / 92e / 92g (bộ MỚI NHẤT có đủ cặp)
    C4 score     set_stats28 mọi bộ (resume ~10 %); CỔNG: mAP21 + CI95 chấm lại so với JSON của 92 (lệch > 1e-9 gắn cờ, > 1e-4 dừng)
    C5 tables    Bảng 3 (ô VinDr, 28 nhãn): hàng x {k-NN, logistic} x {1, 10, 100 %}; TB 2 seed cho (0), (1b), (3); bốn backbone CÓ / KHÔNG
    C6 families  họ 6 phép (3) − (0) và (3) − (1b) ở từng seed; mô tả (X − (0), (3) − (2), ASG − ArrE); hiệu TB 2 seed; bốn cặp CÓ − KHÔNG (họ 6 mỗi cặp)
    C7 sync      JSON + CSV + bộ nhớ đệm -> dong_bo()

Chỉ in SỐ TỔNG HỢP (luật PhysioNet)."""
import json
import os
import re
import time

import numpy as np

from .. import __version__
from .. import config as C
from .. import remap as R
from .. import stats28 as S
from . import common as U

FROZEN_PROBE_VERSION = '0.3.0'
VINDR_BENCH_VERSION = '0.12.0'


def _cfg():
    from frozen_probe import config as FC
    from frozen_probe import config_bb as CB
    from frozen_probe import config_s43 as CS
    return FC, CB, CS


def _rows():
    '''-> {nguồn: hàng đọc} (đặc trưng chính, k-NN + logistic).'''
    FC, CB, CS = _cfg()
    from frozen_probe.encoders.vt_s43 import ROWS as R43
    return dict(s42=tuple(FC.ROWS), bb=tuple(CB.ROWS), s43=tuple(R43))


def code(row):
    FC, CB, CS = _cfg()
    return FC.CODE.get(row) or CB.CODE.get(row) or CS.CODE.get(row) or row


def setup(ns, log=print):
    '''C2. -> CTX.'''
    import frozen_probe
    import vindr_bench
    import vindr28
    d = U.drive_api(ns, C.RUN_TAGS['it_nhan'])
    assert vindr_bench.__version__ == VINDR_BENCH_VERSION and frozen_probe.__version__ == FROZEN_PROBE_VERSION, (
        f'vindr_bench {vindr_bench.__version__} / frozen_probe {frozen_probe.__version__} KHAC ban ghim -> DUNG')
    goi = dict(vindr28=dict(phien_ban=__version__, sha=getattr(vindr28, 'V28_SHA', None)),
               vindr_bench=dict(phien_ban=vindr_bench.__version__, sha=getattr(vindr_bench, 'GOI_SHA', None)),
               frozen_probe=dict(phien_ban=frozen_probe.__version__, sha=getattr(frozen_probe, 'FP_SHA', None)))
    log(f"94b | vindr28 {__version__} @ {(goi['vindr28']['sha'] or '?')[:12]} + vindr_bench {vindr_bench.__version__} @ {(goi['vindr_bench']['sha'] or '?')[:12]} + "
        f"frozen_probe {frozen_probe.__version__} @ {(goi['frozen_probe']['sha'] or '?')[:12]} | {os.cpu_count()} CPU")
    log('28 nhan (PLAN_94 2): ' + ' | '.join(f'{C.STAT_SHORT[s]} = {C.STAT_TEXT[s]}' for s in C.STATS) + f'; bootstrap {C.N_BOOT} lan theo anh, hat {C.BOOT_SEED} (= 92)')
    log(C.READING)
    return dict(drive=d, goi=goi, t=time.strftime('%Y%m%d_%H%M%S'))


def _labels(ctx, log):
    from frozen_probe import features as FT
    from vindr_bench.data import vindr as V
    from vindr_bench.protocols.ours import gates as G
    d = ctx['drive']
    kk = d['kiem_ke'](C.SRC_92A)[1]
    assert FT.MANIFEST_FILE in kk and FT.LABEL_FILE in kk, f'chua co nhan cua 92a trong {"/".join(C.SRC_92A)} -> DUNG'
    man = json.load(open(d['tai'](kk[FT.MANIFEST_FILE][0]), encoding='utf-8'))
    p = d['tai'](kk[FT.LABEL_FILE][0])
    assert FT.md5_file(p) == man['tep_nhan']['md5'], 'tep nhan: md5 KHAC manifest -> DUNG'
    lab = FT.load_labels(p)
    same, diff = G.data_gate(lab['van_tay'])
    ok = all(V.md5_labels(lab['nhan'][lab['tach'] == s]) == lab['van_tay'][s]['md5_nhan'] for s in V.SPLITS)
    log(f"nhan 92a: {len(lab['nhan']):,} anh x {lab['nhan'].shape[1]} nhan, {len(lab['lop_chinh'])} lop so chinh | van tay = cac luot 89: "
        f"{'TRUNG' if same else f'KHAC {diff}'} | md5 nhan tinh lai = van tay: {'DAT' if ok else 'TRUOT'}")
    assert same and ok, 'nhan KHAC cac luot 89 -> DUNG'
    return lab


def load(ctx, log=print):
    '''C3. -> dict(nguon {tên: dict(du_doan {(hàng, đặc trưng, cấu hình, tỉ lệ, lần rút): [m, 28]}, json, t)}).'''
    FC, CB, CS = _cfg()
    d = ctx['drive']
    lab = _labels(ctx, log)
    test = np.flatnonzero(lab['tach'] == FC.TEST_SPLIT)
    target = lab['nhan'][test].astype(np.int8)
    rows = _rows()
    back = {v: k for k, v in FC.FRACTION_CODE.items()}
    src = {}
    for name, folder, prefix in C.SRC_92:
        t, ban = U.latest(d, folder, prefix, 'json')                     # JSON kết quả của 92 — CHỈ cho cổng
        assert ban is not None, f'{name}: chua co JSON ket qua trong {"/".join(folder)} -> DUNG'
        js = json.load(open(d['tai'](ban), encoding='utf-8'))
        f94 = (C.FOLDER, C.RUN_TAGS[f'du_doan_{name}'])                 # 0.2.0: dự đoán float32 do 94c tính lại (KHÔNG dùng bản float16 của 92)
        tp, bp = U.latest(d, f94, f'{C.PRED_FILE}_{name}', 'npz')
        assert bp is not None, f"{name}: chua co du doan float32 cua 94c trong {'/'.join(f94)} -> chay {C.NOTEBOOKS[f'du_doan_{name}']} truoc. DUNG"
        preds = {}
        with np.load(d['tai'](bp), allow_pickle=False) as z:
            for k in z.files:
                if k == 'chi_muc':
                    continue
                row, feat, cfg, fc, dr = k.split('|')
                if row in rows[name] and feat == FC.PRIMARY_FEATURE and cfg in FC.PROBES:
                    v = z[k]
                    assert v.dtype == np.float32 and v.shape == target.shape and np.isfinite(v).all(), f'{name} {k}: {v.dtype} {v.shape} / khong huu han -> DUNG'
                    preds[(row, feat, cfg, back[fc], int(dr))] = v
        have = sorted({k[0] for k in preds})
        miss = [r for r in rows[name] if r not in have]
        src[name] = dict(du_doan=preds, json=js, t=t, t_du_doan=tp, thu_muc='/'.join(f94))
        log(f"  {name:<4} du doan float32 {'/'.join(f94)} @ {tp}: {len(preds)} bo, {len(have)} hang | JSON 92 {'/'.join(folder)} @ {t}"
            + (f' | THIEU hang {miss}' if miss else ''))
    ctx.update(nhan=lab, target=target, nguon=src)
    return src


def _p21(st):
    return dict(diem=st['diem'], ap21_lop=list(st['ap21_lop']), m21=np.asarray(st['m21'], dtype=np.float64), m7=np.asarray(st['m7'], dtype=np.float64),
                nua=list(st['nua21']))


def score(ctx, log=print, workers=None):
    '''C4. -> {nguồn: {đại lượng | 'p21': {(hàng, đặc trưng, cấu hình, tỉ lệ): ô}}}.'''
    from vindr_bench.evaluation import compare as CP
    from vindr_bench.evaluation import panels as P
    lab, target = ctx['nhan'], ctx['target']
    cols = P.panel_columns(lab['ten_nhan'], lab['lop_chinh'], target)
    items = [(name, k, v) for name, s in ctx['nguon'].items() for k, v in sorted(s['du_doan'].items(), key=lambda kv: str(kv[0]))]
    jobs = [(v, target, cols) for _, _, v in items]
    keys = [S.stats_key(*j) for j in jobs]
    ctx['cache28'] = ctx.get('cache28') or U.load_caches(ctx['drive'], (C.FOLDER, C.RUN_TAGS['it_nhan']), log)
    stats = U.run_jobs(ctx, jobs, keys, ctx['cache28'], log=log, workers=workers)
    groups = {}
    for (name, k, _), st in zip(items, stats):
        groups.setdefault((name, k[:4]), []).append((k[4], st))
    cells = {name: {s: {} for s in C.STATS + ('p21',)} for name in ctx['nguon']}
    for (name, k4), v in groups.items():
        v.sort(key=lambda x: x[0])
        sts = [st for _, st in v]
        for s in C.STATS:
            cells[name][s][k4] = R.cell(sts, s)
        cells[name]['p21'][k4] = CP.make_cell([_p21(st) for st in sts])
    gate, worst = [], dict(diem=0.0, ci=0.0)
    for name, s in ctx['nguon'].items():
        ref = s['json'].get('o', {})
        for k4, c in cells[name]['p21'].items():
            r = ref.get('|'.join(str(x) for x in k4))
            if r is None:
                gate.append(dict(nguon=name, o=list(k4), co_ban_92=False))
                continue
            dp = abs(c['diem']['p21']['ap'] - r['diem']['p21']['ap'])
            dc = max(abs(a - b) for a, b in zip(c['ci21'], r['ci21']))
            worst['diem'], worst['ci'] = max(worst['diem'], dp), max(worst['ci'], dc)
            gate.append(dict(nguon=name, o=list(k4), co_ban_92=True, lech_diem=dp, lech_ci=dc, so_lan_rut=c['so_hat'], so_lan_rut_92=r['so_hat']))
    have = [g for g in gate if g['co_ban_92']]
    nrut = all(g['so_lan_rut'] == g['so_lan_rut_92'] for g in have)
    flag = [g for g in have if max(g['lech_diem'], g['lech_ci']) > C.FLAG_92]
    stop = [g for g in have if max(g['lech_diem'], g['lech_ci']) > C.STOP_92]
    log(f"CONG so voi JSON cua 92 (mAP21 cham lai tu du doan float32 cua 94c): {len(have)} / {len(gate)} o co ban 92 | so lan rut nhan trung: {'DAT' if nrut else 'TRUOT'} | lech "
        f"lon nhat: so diem {worst['diem']:.1e}, can CI95 {worst['ci']:.1e} | > {C.FLAG_92:g}: {len(flag)} o (gan co) | > {C.STOP_92:g}: {len(stop)} o -> "
        + ('DAT' if have and nrut and not stop else 'TRUOT'))
    for g in flag[:6]:
        log(f"  ! co: {g['nguon']} {code(g['o'][0])} {g['o'][2]} {g['o'][3]:g}: lech diem {g['lech_diem']:.4f}, CI {g['lech_ci']:.4f}")
    assert have and nrut and not stop, f'cong 92 TRUOT (lech > {C.STOP_92:g} -> nghi doc sai tep / khoa): {stop[:3]} -> DUNG, gui Claude dong tren'
    ctx.update(o28=cells, cot=cols, cong92=dict(so_o=len(gate), co_ban_92=len(have), lech_diem_lon_nhat=worst['diem'], lech_ci_lon_nhat=worst['ci'],
                                                gan_co=len(flag), chi_tiet=gate))
    log(f"{len(jobs)} bo -> o: " + ', '.join(f"{n} {len(cells[n]['ap_b'])}" for n in cells) + f" | tinh moi {ctx.get('phut_tinh', 0):.1f} phut")
    return cells


def _point(c):
    return c['diem']['p21']['ap']


def tables(ctx, log=print):
    '''C5. Bảng 3 (ô VinDr, 28 nhãn) + bốn backbone; số điểm = TB các lần rút nhãn (1 % x10, 10 % x5, 100 % x1).'''
    FC, CB, CS = _cfg()
    o, rows = ctx['o28'], []
    feat = FC.PRIMARY_FEATURE
    s43 = {CS.BASE[r]: r for r in _rows()['s43']}
    order = ('raddino', 'vt_khong_luong', 'vt_sau1', 'vt_sau4', 'vt_mot_chieu', 'vt_b4v2', 'b', 'asg')
    log('cach doc: AP28 (B = D tren ca test) / AUROC28; moi o = TB cac lan rut nhan; TB 2 seed = TB cua s42 (92b) va s43 (92g) cung vi tri; k-NN k = round(sqrt n), '
        'logistic C = 1. So diem, KHONG CI.')
    head = ''.join(f'{FC.PROBE_TEXT[p] + " " + FC.FRACTION_TEXT[f]:>17}' for p in FC.PROBES for f in FC.FRACTIONS)
    log(f"\n=== Bang 3 (VinDr, dac trung {feat}) — AP28 / AUROC28 ===\n{'hang':<26}{'seed':<7}{head}")
    for r in order:
        variants = [('s42', [('s42', r)])]
        if r in s43:
            variants = [('s42', [('s42', r)]), ('s43', [('s43', s43[r])]), ('TB 2', [('s42', r), ('s43', s43[r])])]
        for vname, members in variants:
            ks = [(n, (rr, feat, p, f)) for n, rr in members for p in FC.PROBES for f in FC.FRACTIONS]
            if not all(k in o[n]['ap_b'] for n, k in ks):
                continue
            line = f"{code(r)[:25]:<26}{vname:<7}"
            for p in FC.PROBES:
                for f in FC.FRACTIONS:
                    a = np.mean([_point(o[n]['ap_b'][(rr, feat, p, f)]) for n, rr in members])
                    u = np.mean([_point(o[n]['auroc'][(rr, feat, p, f)]) for n, rr in members])
                    line += f'{U.f3(a) + " / " + U.f3(u):>17}'
                    rows.append(dict(bang='3a', hang=r, ma=code(r), seed=vname, dau_doc=p, ti_le_nhan=f, AP28=float(a), AUROC28=float(u)))
            log(line)
    log(f"\n=== Bang 3 (b) bon backbone, CO / KHONG luong vung (s42, 92e; RAD-DINO = 92b) — AP28 / AUROC28 ===\n{'hang':<26}{'':<7}{head}")
    pairs = [('RAD-DINO', ('s42', FC.PAIR[0]), ('s42', FC.PAIR[1]))] + [(e, ('bb', CB.PAIRS[e][0]), ('bb', CB.PAIRS[e][1])) for e in CB.ENCODERS]
    for e, co, kh in pairs:
        for lab_, (n, rr) in (('CO', co), ('KHONG', kh)):
            if not all((rr, feat, p, f) in o[n]['ap_b'] for p in FC.PROBES for f in FC.FRACTIONS):
                continue
            line = f"{(e + ' ' + lab_)[:25]:<26}{'s42':<7}"
            for p in FC.PROBES:
                for f in FC.FRACTIONS:
                    a, u = _point(o[n]['ap_b'][(rr, feat, p, f)]), _point(o[n]['auroc'][(rr, feat, p, f)])
                    line += f'{U.f3(a) + " / " + U.f3(u):>17}'
                    rows.append(dict(bang='3b', hang=rr, ma=f'{e} {lab_}', seed='s42', dau_doc=p, ti_le_nhan=f, AP28=a, AUROC28=u))
            log(line)
    ctx['bang28'] = rows
    return rows


def _fam_print(name, fams, log, rows):
    FC, _, _ = _cfg()
    f0 = fams['ap_b']
    if not f0['phep_so']:
        log(f'--- {name}: chua co du lieu')
        return
    log(f"--- {name} | {f0['co']} / {f0['so_phep']} phep | " + ' | '.join(
        f"q {C.STAT_SHORT[s]} = {fams[s]['q']:.2f}" if fams[s]['q'] else f'{C.STAT_SHORT[s]}: ho chua du' for s in C.STATS))
    for i, p in enumerate(f0['phep_so']):
        log(f"    {FC.PROBE_TEXT[p['dau']]:<9}{FC.FRACTION_TEXT[p['ti_le']]:>7}  "
            + ''.join(f"{C.STAT_SHORT[s]} {U.s3(fams[s]['phep_so'][i]['d']['ap'])} {U.ci(fams[s]['phep_so'][i]['ci_dong_thoi'])}  " for s in C.STATS)
            + ' / '.join(fams[s]['phep_so'][i]['ket_luan'] for s in C.STATS))
    for s in C.STATS:
        labs = [p['ket_luan'] for p in fams[s]['phep_so']]
        log(f"    {C.STAT_SHORT[s]:<8}: " + ', '.join(f'{v} {labs.count(v)}' for v in sorted(set(labs))) + f" / {len(labs)} | khoang dong thoi > 0: "
            f"{sum(U.above(p['ci_dong_thoi']) == '>0' for p in fams[s]['phep_so'])}, < 0: {sum(U.above(p['ci_dong_thoi']) == '<0' for p in fams[s]['phep_so'])}")
        for p in fams[s]['phep_so']:
            rows.append(dict(bang=name, dai_luong=C.STAT_SHORT[s], dau_doc=p['dau'], ti_le_nhan=p['ti_le'], hieu=p['d']['ap'], ci_danh_nghia_thap=p['ci_danh_nghia'][0],
                             ci_danh_nghia_cao=p['ci_danh_nghia'][1], ci_dong_thoi_thap=(p['ci_dong_thoi'] or [None])[0],
                             ci_dong_thoi_cao=(p['ci_dong_thoi'] or [None, None])[1], dong_thoi=U.above(p['ci_dong_thoi']), dang_luat_cu_mo_ta=p['ket_luan'],
                             so_lan_rut_duong=p['so_lan_rut_duong']))


def _mean_diff(pairs):
    '''Hiệu TB 2 seed: pairs = [(ô a, ô b) từng seed] cùng lần rút bootstrap -> dict(d, ci danh nghĩa, tung_seed, nua).'''
    r = np.mean([a['m21'] - b['m21'] for a, b in pairs], 0)
    return dict(d=float(np.mean([_point(a) - _point(b) for a, b in pairs])), ci=U.pct_ci(r), tung_seed=[float(_point(a) - _point(b)) for a, b in pairs],
                nua=[float(np.mean([a['nua'][h] - b['nua'][h] for a, b in pairs])) for h in (0, 1)])


def families(ctx, log=print):
    '''C6. Họ 6 phép + mô tả + TB 2 seed + bốn backbone, cho từng đại lượng.'''
    from frozen_probe import scoring as SC
    from vindr_bench.evaluation import compare as CP
    FC, CB, CS = _cfg()
    o, rows, out = ctx['o28'], [], {}
    feat = FC.PRIMARY_FEATURE
    r43 = {CS.BASE[r]: r for r in _rows()['s43']}
    log(C.READING)
    log('\n=== HO 6 PHEP (k-NN, logistic x 1 / 10 / 100 %), khoang DONG THOI max-T 95 % trong ho; moi o: hieu [dong thoi]; dang luat cu B / D / AUROC ===')
    specs = [('(3) - (0) s42', 's42', FC.PAIR[0], FC.PAIR[1]), ('(3) - (0) s43', 's43', r43[FC.PAIR[0]], r43[FC.PAIR[1]]),
             ('(3) - (1b) s42', 's42', FC.PAIR[0], 'vt_sau1'), ('(3) - (1b) s43', 's43', r43[FC.PAIR[0]], r43['vt_sau1']),
             ('(3) - (2) s42', 's42', FC.PAIR[0], 'vt_mot_chieu'), ('(0) - RAD-DINO s42', 's42', FC.PAIR[1], 'raddino'), ('ASG - ArrE s42', 's42', 'asg', 'b')]
    specs += [(f"{e}: CO - KHONG s42", 'bb', CB.PAIRS[e][0], CB.PAIRS[e][1]) for e in CB.ENCODERS]
    for name, src, a, b in specs:
        fams = {s: SC.family_pair(o[src][s], a, b) for s in C.STATS}
        out[name] = fams
        _fam_print(name, fams, log, rows)
    log('\n=== HIEU TB 2 SEED (s42 + s43; cung lan rut bootstrap; CI95 DANH NGHIA; mo ta) ===')
    tb = {}
    for name, a, b in (('(3) - (0)', FC.PAIR[0], FC.PAIR[1]), ('(3) - (1b)', FC.PAIR[0], 'vt_sau1'), ('(1b) - (0)', 'vt_sau1', FC.PAIR[1])):
        for p in FC.PROBES:
            line = f'    {name:<12}{FC.PROBE_TEXT[p]:<9}'
            for f in FC.FRACTIONS:
                ks = [('s42', (a, feat, p, f), (b, feat, p, f)), ('s43', (r43[a], feat, p, f), (r43[b], feat, p, f))]
                if not all(k1 in o[n]['ap_b'] and k2 in o[n]['ap_b'] for n, k1, k2 in ks):
                    continue
                res = {s: _mean_diff([(o[n][s][k1], o[n][s][k2]) for n, k1, k2 in ks]) for s in C.STATS}
                tb[(name, p, f)] = res
                line += f"{FC.FRACTION_TEXT[f]:>6}: " + ' '.join(f"{C.STAT_SHORT[s]} {U.s3(res[s]['d'])} {U.ci(res[s]['ci'])}" for s in C.STATS) + '  '
                for s in C.STATS:
                    rows.append(dict(bang=f'TB2seed {name}', dai_luong=C.STAT_SHORT[s], dau_doc=p, ti_le_nhan=f, hieu=res[s]['d'], ci_danh_nghia_thap=res[s]['ci'][0],
                                     ci_danh_nghia_cao=res[s]['ci'][1], hieu_s42=res[s]['tung_seed'][0], hieu_s43=res[s]['tung_seed'][1]))
            log(line)
    desc = {s: SC.descriptive(o['s42'][s]) for s in C.STATS}
    log('\n=== MO TA s42 (CI95 danh nghia; X − (0) moi hang, (3) − (2)) — AP28-B ===')
    for h in desc['ap_b']['hieu']:
        log(f"    {code(h['hang_a']) + ' − ' + code(h['hang_b']):<22}{FC.PROBE_TEXT[h['dau']]:<9}{FC.FRACTION_TEXT[h['ti_le']]:>7}  {U.s3(h['d']['ap'])} {U.ci(h['ci_danh_nghia'])}")
    ctx.update(ho28=out, tb2=tb, mo_ta28=desc, hieu28=rows)
    return dict(ho=out, tb2=tb, mo_ta=desc)


def sync(ctx, log=print):
    '''C7. JSON + CSV + bộ nhớ đệm 28 nhãn -> dong_bo().'''
    d, t = ctx['drive'], ctx['t']
    names = []
    strip = lambda o: {'|'.join(str(x) for x in k): {kk: v for kk, v in c.items() if kk not in ('m21', 'm7')} for k, c in o.items()}
    obj = U.jsonable(dict(run=C.RUN_TAGS['it_nhan'], t=t, goi=ctx['goi'], cach_tinh={s: C.STAT_TEXT[s] for s in C.STATS}, cach_doc=C.READING,
                          nguon={n: dict(thu_muc=s['thu_muc'], t=s['t'], so_bo=len(s['du_doan'])) for n, s in ctx['nguon'].items()}, cong_92=ctx.get('cong92'),
                          o={n: {s: strip(c[s]) for s in C.STATS} for n, c in ctx['o28'].items()},
                          ho={k: v for k, v in (ctx.get('ho28') or {}).items()}, tb2={f'{k[0]}|{k[1]}|{k[2]}': v for k, v in (ctx.get('tb2') or {}).items()},
                          mo_ta=ctx.get('mo_ta28')))
    names.append(f'vindr28_it_nhan_{t}.json')
    d['ghi'](names[-1], lambda q: d['ghi_json'](q, obj))
    for tag, rows in (('bang', ctx.get('bang28') or []), ('hieu', ctx.get('hieu28') or [])):
        n = f'vindr28_it_nhan_{tag}_{t}.csv'
        if U.write_csv(d, n, rows):
            names.append(n)
    names.append(f'bo_dem_28_{t}.npz')
    U.write_cache(d, names[-1], ctx['cache28'])
    du = d['dong_bo']()
    co = set(names) <= set(d['ten_tren_drive']())
    log(f"C7: da ghi {len(names)} tep | dong_bo {'DU' if du else 'CHUA DU'} | tren Drive: {'CO du' if co else 'THIEU'}" + ('' if du and co else ' | KHONG tat runtime; chay lai C7'))
    return dict(dong_bo=du, du_tep=co, tep=names)
