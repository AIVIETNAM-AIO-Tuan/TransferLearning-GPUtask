# -*- coding: utf-8 -*-
"""Ô của notebook `94a_vindr28_benchmark` (Colab CPU; PLAN_94 §4): chấm lại trên 28 nhãn mọi bộ dự đoán test của 89 (Ver1, Ver2 ba kho, Việc B) + Linear-4
/ GLoRI CheXFound của 86f — dữ liệu gom bằng ĐÚNG hàm của 89t (vindr_bench.notebook.summary.collect), họ phép so gọi lại ĐÚNG hàm của 89t trên ô 28 nhãn.

    C2 setup      khối Drive + phiên bản hai gói + cách tính (in ra để đối chiếu)
    C3 collect    kiểm kê + tải dự đoán (summary.collect của 89t) + bộ nhớ đệm bootstrap của 89t (cổng) + bộ nhớ đệm / resume 28 nhãn
    C4 score      set_stats28 cho mọi bộ (resume ~10 % một lần); CỔNG so với 89t: điểm 4 panel trùng, mAP21 từng lần rút lệch <= 1e-12 -> ô của ba đại lượng
    C5 tables     Bảng 2 (28 nhãn): ma trận encoder x đầu mỗi điều kiện + 5 hàng Việc B; bảng dài (CSV)
    C6 families   họ ArrE − X (32 phép, ba điều kiện) + họ riêng ArrE − ASG + hai họ hình học + kho cân bằng − không + ArrE − RAD-DINO trên AUROC
    C7 ablation   Việc B: họ 16 phép (3) − X + báo kèm + cùng lr; (0) − RAD-DINO công khai; phép thử cắt giữa (Δ) của ArrE
    C8 chexfound  CheXFound / RAD-DINO / BiomedCLIP (Ver1 = mã của CheXFound) so với Bảng IV bài CheXFound — bốn cách tính, độ rộng CI (MÔ TẢ)
    C9 sync       JSON + CSV + bộ nhớ đệm 28 nhãn -> dong_bo()

Chỉ in SỐ TỔNG HỢP. Cách đọc: config.READING."""
import os
import time

import numpy as np

from .. import __version__
from .. import config as C
from .. import remap as R
from .. import stats28 as S
from . import common as U

VINDR_BENCH_VERSION = '0.12.0'                    # = 89t (gói @ c81af62a71dd): cùng hàm gom dữ liệu, chấm, họ phép so


def _S89():
    from vindr_bench.notebook import summary
    return summary


def setup(ns, log=print):
    '''C2. -> CTX.'''
    import vindr_bench
    import vindr28
    d = U.drive_api(ns, C.RUN_TAGS['bench'])
    assert vindr_bench.__version__ == VINDR_BENCH_VERSION, f'vindr_bench {vindr_bench.__version__} != {VINDR_BENCH_VERSION} (ban cua 89t) -> DUNG'
    goi = dict(vindr28=dict(phien_ban=__version__, sha=getattr(vindr28, 'V28_SHA', None)),
               vindr_bench=dict(phien_ban=vindr_bench.__version__, sha=getattr(vindr_bench, 'GOI_SHA', None)))
    log(f"94a | vindr28 {__version__} @ {(goi['vindr28']['sha'] or '?')[:12]} + vindr_bench {goi['vindr_bench']['phien_ban']} @ "
        f"{(goi['vindr_bench']['sha'] or '?')[:12]} | numpy {np.__version__} | {os.cpu_count()} CPU")
    log(f'28 nhan (PLAN_94 2): ' + ' | '.join(f'{C.STAT_SHORT[s]} = {C.STAT_TEXT[s]}' for s in C.STATS)
        + f'; so diem: AP28 = AP27 (Edema 0 ca duong test), AUROC28 co Edema = 0; bootstrap {C.N_BOOT} lan theo anh, hat {C.BOOT_SEED} (= 89t), khong bo lan rut nao')
    log(C.READING)
    return dict(drive=d, goi=goi, t=time.strftime('%Y%m%d_%H%M%S'))


def collect(ctx, log=print):
    '''C3. -> dl (summary.collect của 89t).'''
    S89 = _S89()
    dl = S89.collect(ctx, log=log)
    ctx['cache89'] = S89.load_cache(ctx, log=log)
    assert ctx['cache89'], 'chua doc duoc bo nho dem bootstrap cua 89t (stage4_nb89/r89t_tong_hop) -> chay 89t truoc. DUNG'
    ctx['cache28'] = U.load_caches(ctx['drive'], (C.FOLDER, C.RUN_TAGS['bench']), log)
    return dl


def score(ctx, log=print, workers=None):
    '''C4. -> {đại lượng: {(ver, kho, encoder, đầu, nhánh): ô}}.'''
    from vindr_bench.evaluation import compare as CP
    from vindr_bench.evaluation import panels as P
    dl = ctx['dl']
    cols = {}
    for s in dl['sets']:
        k = CP.target_key(s.target)
        if k not in cols:
            cols[k] = P.panel_columns(dl['nhan'], dl['lop_chinh'], s.target)
    jobs = [(s.preds, s.target, cols[CP.target_key(s.target)]) for s in dl['sets']]
    keys = [S.stats_key(*j) for j in jobs]
    k89 = [CP.stats_key(*j, P.N_BOOT, P.BOOT_SEED, P.HALF_SEED) for j in jobs]
    ref = ctx['cache89']

    def gate(i, st):
        r = ref.get(k89[i])
        return None if r is None else S.gate_89(st, r)

    stats = U.run_jobs(ctx, jobs, keys, ctx['cache28'], log=log, workers=workers, on_new=gate)
    g = [st.get('cong') for st in stats]
    have = [x for x in g if x is not None]
    bad = [dl['sets'][i].cell for i, x in enumerate(g) if x is not None and not x['dat']]
    miss = [dl['sets'][i].cell for i, x in enumerate(g) if x is None]
    lech = max((x['lech_m21'] for x in have if x['lech_m21'] == x['lech_m21']), default=float('nan'))
    log(f"CONG so voi 89t (cung bo du doan, cung day rut): {len(have)} / {len(g)} bo co ban 89t | diem 4 panel TRUNG: "
        f"{sum(x['diem_trung'] for x in have)} / {len(have)} | hai nua AP21 TRUNG: {sum(x['nua21_trung'] for x in have)} / {len(have)} | mAP21 tung lan rut: lech lon "
        f"nhat {lech:.1e} (nguong {C.TOL_89:g}), vi tri lan rut bi loai TRUNG: {sum(x['nan_trung'] for x in have)} / {len(have)} -> "
        + ('DAT' if have and not bad else 'TRUOT'))
    if miss:
        log(f'  ! {len(miss)} bo KHONG co trong bo nho dem cua 89t (khong cong duoc): {miss[:4]}')
    assert have and not bad, f'cong so voi 89t TRUOT o {bad[:4]} -> DUNG, gui Claude dong tren'
    ctx['cong89'] = dict(co_ban_89t=len(have), tong=len(g), truot=len(bad), thieu=[list(m) for m in miss], lech_m21_lon_nhat=lech)
    ctx['st28'] = stats
    groups = {}
    for s, st in zip(dl['sets'], stats):
        groups.setdefault(s.cell, []).append((s, st))
    cells = {stat: {} for stat in C.STATS}
    for key, items in groups.items():
        items.sort(key=lambda x: x[0].seed)
        s0 = items[0][0]
        c = cols[CP.target_key(s0.target)]
        ens = R.ensemble_points(np.mean([s.preds for s, _ in items], 0), s0.target, c) if len(items) > 1 else None
        for stat in C.STATS:
            cells[stat][key] = R.cell([st for _, st in items], stat, ens, lr=s0.lr, bien=s0.edge, nguon=s0.source, val_auroc=s0.val_auroc,
                                      khoa_nhan=CP.target_key(s0.target), hat=[s.seed for s, _ in items])
    ctx.update(o28=cells, cot=cols)
    log(f"{len(dl['sets'])} bo -> {len(cells['ap_b'])} o (phien ban, kho, encoder, dau, nhanh) x {len(C.STATS)} dai luong | tinh moi {ctx.get('phut_tinh', 0):.1f} phut")
    return cells


def _o(ctx, stat='ap_b'):
    return ctx['o28'][stat]


def tables(ctx, log=print):
    '''C5. Bảng 2 (28 nhãn) — số điểm (AP28 B = D trên cả test) + AUROC28; bảng dài (CSV).'''
    S89 = _S89()
    from vindr_bench.encoders import ABLATION_ROWS, ALL_ROWS, SPECS
    from vindr_bench.evaluation import collect as CL
    oa, ou = _o(ctx, 'ap_b'), _o(ctx, 'auroc')
    rows = []
    log('cach doc: AP28 = AP bac thang TB 27 nhan co ca duong test (Edema 0 ca duong -> bo); AUROC28 = TB 28 nhan, Edema = 0 (ngu nghia torchmetrics 0.10.3, '
        'nhu ma CheXFound); Ver2 = TB 5 hat dau doc; Ver1 = mot luot hat 0; * = lr o bien luoi. So diem, KHONG CI (CI cua hieu o C6 / C7).')
    for fam in S89.FAMS:
        encs = [e for e in ALL_ROWS if any((fam[0], fam[1], e, h, 'luat') in oa for h in S89.HEADS)]
        if not encs:
            continue
        log(f'\n=== {S89.FAM_TEXT[fam]} ===')
        log(f"{'hang (anh, vector)':<34}" + ''.join(f'{CL.TITLES[h] + " AP / AUROC":>24}' for h in S89.HEADS))
        for e in encs:
            line = f'{S89._row(e):<34}'
            for h in S89.HEADS:
                k = (fam[0], fam[1], e, h, 'luat')
                line += f"{(U.f3(oa[k]['diem']['p21']['ap']) + ' / ' + U.f3(ou[k]['diem']['p21']['ap']) + ('*' if oa[k]['bien'] else ' ')) if k in oa else '-':>24}"
            log(line)
    abl = [e for e in ABLATION_ROWS if any(('v2', 'kcb', e, h, 'luat') in oa for h in S89.HEADS)]
    if abl:
        log('\n=== Viec B (PLAN_90 10.2): 5 hang ablation, Ver2 kho dem vuong KHONG can bang, TB 5 hat dau doc — AP28 / AUROC28 ===')
        for e in abl:
            line = f'{SPECS[e].title[:34]:<34}'
            for h in S89.HEADS:
                k = ('v2', 'kcb', e, h, 'luat')
                line += f"{(U.f3(oa[k]['diem']['p21']['ap']) + ' / ' + U.f3(ou[k]['diem']['p21']['ap'])) if k in oa else '-':>24}"
            log(line)
    for key in sorted(oa, key=str):
        a, u, da = oa[key], ou[key], _o(ctx, 'ap_d')[key]
        ver, store, e, h, arm = key
        rows.append(dict(bang='diem', phien_ban=ver, kho=store, dieu_kien=S89.FAM_TEXT.get((ver, store), f'{ver} {store}'), encoder=e,
                         hang=SPECS[e].title if e in SPECS else e, dau=h, nhanh=arm, lr=a['lr'], lr_o_bien=a['bien'], so_hat=a['so_hat'],
                         AP28=a['diem']['p21']['ap'], AUROC28=u['diem']['p21']['ap'], AUPRC27_hinh_thang=a['diem']['p27']['auprc_thang'],
                         AP28_sd_giua_hat=a['sd']['p21']['ap'], AUROC28_sd_giua_hat=u['sd']['p21']['ap'],
                         AP28_ensemble=(a['ensemble'] or {}).get('p21', {}).get('ap'), AUROC28_ensemble=(u['ensemble'] or {}).get('p21', {}).get('ap'),
                         AP28B_nua_0=a['nua'][0], AP28B_nua_1=a['nua'][1], AP28D_nua_0=da['nua'][0], AP28D_nua_1=da['nua'][1],
                         AUROC28_nua_0=u['nua'][0], AUROC28_nua_1=u['nua'][1], mAP21_doi_chieu=_p21(ctx, key), nguon=a['nguon']))
    ctx['bang28'] = rows
    log(f'\n{len(rows)} dong bang dai (CSV o C9)')
    return rows


def _p21(ctx, key):
    '''mAP21 thật (21 lớp chính) của ô — chỉ để đối chiếu trong CSV (không vào bài).'''
    idx = [i for i, s in enumerate(ctx['dl']['sets']) if s.cell == key]
    return float(np.mean([ctx['st28'][i]['diem']['p21']['ap'] for i in idx])) if idx else float('nan')


def _cmp_rows(stat, fam, ho, rec, nominal):
    return [dict(bang=ho, dai_luong=C.STAT_SHORT[stat], dieu_kien=fam, dau=r['dau'], phep_tru=f"ArrE - {r['doi_chung']}", hieu=r['d'],
                 ci_danh_nghia_thap=(nominal.get(r['dau'] + '|b|' + r['doi_chung']) or {}).get('ci', [None, None])[0],
                 ci_danh_nghia_cao=(nominal.get(r['dau'] + '|b|' + r['doi_chung']) or {}).get('ci', [None, None])[1],
                 ci_dong_thoi_thap=r['ci_dong_thoi'][0], ci_dong_thoi_cao=r['ci_dong_thoi'][1], dong_thoi=U.above(r['ci_dong_thoi']),
                 dang_luat_cu_mo_ta=r['ket_luan']) for r in rec]


def families(ctx, log=print):
    '''C6. -> {đại lượng: kết quả summary.compare của 89t trên ô 28 nhãn}.'''
    S89 = _S89()
    from vindr_bench.encoders import SPECS
    from vindr_bench.evaluation import collect as CL
    res, rows = {}, []
    for stat in C.STATS:
        c = dict(ctx, o=ctx['o28'][stat])
        for k in ('so_sanh', 'cap'):
            c.pop(k, None)
        res[stat] = S89.compare(c, log=U.quiet)
    ctx['ho28'] = res
    log(C.READING)
    for fam in S89.RULE_FAMS:
        fs = S89.FAM_SHORT[fam]
        for ho, title in (('ho_tham_chieu', 'HO 32 PHEP ArrE − X (8 doi chung x 4 dau)'), ('ho_them', 'HO RIENG ArrE − ASG (4 dau; so hai he thong)')):
            if fs not in res['ap_b'][ho]:
                continue
            log(f"\n=== {title} | {S89.FAM_TEXT[fam]} ===")
            qs = ' | '.join(f"q {C.STAT_SHORT[s]} = {res[s][ho][fs]['q']:.2f}" for s in C.STATS)
            log(f'    khoang DONG THOI max-T 95 % trong ho: {qs}')
            log(f"    {'dau':<11}{'X':<14}" + ''.join(f"{C.STAT_SHORT[s] + ': hieu [dong thoi]':<34}" for s in C.STATS) + 'dang luat cu (mo ta) B / D / AUROC')
            recs = {s: res[s][ho][fs]['phep_so'] for s in C.STATS}
            for i, r in enumerate(recs['ap_b']):
                line = f"    {CL.TITLES[r['dau']]:<11}{SPECS[r['doi_chung']].title[:13]:<14}"
                for s in C.STATS:
                    x = recs[s][i]
                    line += f"{U.s3(x['d']) + ' ' + U.ci(x['ci_dong_thoi']):<34}"
                log(line + ' / '.join(recs[s][i]['ket_luan'] for s in C.STATS))
            for s in C.STATS:
                n = len(recs[s])
                pos = sum(U.above(x['ci_dong_thoi']) == '>0' for x in recs[s])
                neg = sum(U.above(x['ci_dong_thoi']) == '<0' for x in recs[s])
                lab = {v: sum(x['ket_luan'] == v for x in recs[s]) for v in ('hon', 'kem', 'khong phan biet')}
                log(f"    {C.STAT_SHORT[s]:<8}: khoang dong thoi > 0 o {pos} / {n}, < 0 o {neg} / {n} | dang luat cu: hon {lab['hon']}, kem {lab['kem']}, "
                    f"khong phan biet {lab['khong phan biet']}")
                rows += _cmp_rows(s, fs, ho, recs[s], res[s]['giua_encoder'].get(fs, {}))
    log('\n=== HAI HO HINH HOC (kho CAT GIUA − kho DEM VUONG, Ver2, cung khong can bang; (a) Δ, (b) Δ_ArrE − Δ_X, (c) ArrE − X tren kho cat giua) ===')
    for name in res['ap_b']['hinh_hoc']:
        hh = {s: res[s]['hinh_hoc'][name] for s in C.STATS}
        log(f"--- {hh['ap_b']['tieu_de']} | {hh['ap_b']['co']} / {hh['ap_b']['so_phep']} phep | "
            + ' | '.join(f"q {C.STAT_SHORT[s]} {hh[s]['q']:.2f}" if hh[s]['q'] else f'{C.STAT_SHORT[s]}: ho chua du' for s in C.STATS))
        for kind, kt in (('a', '(a) Δ = cat giua − dem vuong'), ('c', '(c) ArrE − X tren kho cat giua'), ('b', '(b) Δ_ArrE − Δ_X')):
            idx = [i for i, p in enumerate(hh['ap_b']['phep_so']) if p['dang'] == kind]
            if not idx:
                continue
            log(f'    {kt}:')
            for i in idx:
                p = hh['ap_b']['phep_so'][i]
                if kind == 'a' and p['encoder'] != 'b' and not any(U.above(hh[s]['phep_so'][i]['ci_dong_thoi']) == '<0' for s in C.STATS):
                    continue
                log(f"      {CL.TITLES[p['dau']]:<11}{SPECS[p['encoder']].title[:16]:<17}"
                    + ''.join(f"{C.STAT_SHORT[s]} {U.s3(hh[s]['phep_so'][i]['d'])} {U.ci(hh[s]['phep_so'][i]['ci_dong_thoi'])}  " for s in C.STATS)
                    + ' / '.join(hh[s]['phep_so'][i]['ket_luan'] for s in C.STATS))
            for s in C.STATS:
                labs = [hh[s]['phep_so'][i]['ket_luan'] for i in idx]
                log(f"      {C.STAT_SHORT[s]:<8} {kt.split(' ')[0]}: " + ', '.join(f'{v} {labs.count(v)}' for v in sorted(set(labs))) + f' / {len(idx)}')
            for s in C.STATS:
                for i in idx:
                    p = hh[s]['phep_so'][i]
                    rows.append(dict(bang=f'hinh_hoc_{name}', dai_luong=C.STAT_SHORT[s], dieu_kien='Ver2 cat giua vs dem vuong', dau=p['dau'],
                                     phep_tru=f"{kind}: {p['encoder']}", hieu=p['d'], ci_danh_nghia_thap=p['ci_danh_nghia'][0], ci_danh_nghia_cao=p['ci_danh_nghia'][1],
                                     ci_dong_thoi_thap=(p['ci_dong_thoi'] or [None])[0], ci_dong_thoi_cao=(p['ci_dong_thoi'] or [None, None])[1],
                                     dong_thoi=U.above(p['ci_dong_thoi']), dang_luat_cu_mo_ta=p['ket_luan']))
    log('\n=== Ver2: kho CO can bang − kho KHONG can bang (cung encoder, cung dau; hai luot hoc rieng; CI95 danh nghia; mo ta) ===')
    for e in ('b',) + tuple(x for x in SPECS if x != 'b'):
        rr = {s: [g for g in res[s]['giua_kho'] if g['encoder'] == e] for s in C.STATS}
        if not rr['ap_b']:
            continue
        if e != 'b':
            big = [g for g in rr['ap_b'] if abs(g['hieu']) > 0.01]
            if not big:
                continue
        log(f"    {SPECS[e].title[:16]:<17}" + '  '.join(f"{CL.TITLES[g['dau']]} {U.s3(g['hieu'])} {U.ci(g['ci'])} / AUROC {U.s3(u['hieu'])}"
                                                     for g, u in zip(rr['ap_b'], rr['auroc'])))
        for s in C.STATS:
            for g in rr[s]:
                rows.append(dict(bang='kho_cb_tru_kcb', dai_luong=C.STAT_SHORT[s], dieu_kien='Ver2 cb - kcb', dau=g['dau'], phep_tru=f"{e}: cb - kcb", hieu=g['hieu'],
                                 ci_danh_nghia_thap=g['ci'][0], ci_danh_nghia_cao=g['ci'][1]))
    log('    (encoder khac ArrE chi in khi |hieu AP28-B| > 0,01 o mot dau)')
    log('\n=== ArrE − RAD-DINO tren AUROC28 (CI95 danh nghia; Ver1 / Ver2 kho khong can bang) ===')
    for fam in S89.RULE_FAMS[:2]:
        fs = S89.FAM_SHORT[fam]
        ge = res['auroc']['giua_encoder'].get(fs, {})
        log(f"    {fs:<9}" + '  '.join(f"{CL.TITLES[h]} {U.s3(ge[k]['hieu'])} {U.ci(ge[k]['ci'])}" for h in S89.HEADS for k in [f'{h}|b|raddino'] if k in ge))
    ctx['hieu28'] = rows
    return res


def ablation(ctx, log=print):
    '''C7. Việc B (họ 16 phép (3) − X + báo kèm + cùng lr), (0) − RAD-DINO công khai, Δ cắt giữa của ArrE — trên ô 28 nhãn.'''
    import matplotlib.pyplot as plt
    S89 = _S89()
    from vindr_bench.encoders import SPECS
    from vindr_bench.evaluation import collect as CL
    from vindr_bench.evaluation import compare as CP
    ab, cg, rows = {}, {}, ctx.setdefault('hieu28', [])
    for stat in C.STATS:
        c = dict(ctx, o=ctx['o28'][stat], so_sanh=ctx['ho28'][stat])
        cg[stat] = S89.crop_test(c, log=U.quiet, show=False)
        plt.close('all')
        ab[stat] = S89.ablation(c, log=U.quiet, show=False)
        plt.close('all')
    ctx.update(ablation28=ab, cat_giua28=cg)
    h16 = {s: ab[s]['ho_16'] for s in C.STATS}
    if h16['ap_b'] and h16['ap_b']['phep_so']:
        log('=== VIEC B: HO 16 PHEP (3) − X (Ver2 kho dem vuong khong can bang, TB 5 hat; doc THEO TUNG DAU; khong ket luan ve vi tri) ===')
        log('    ' + ' | '.join(f"q {C.STAT_SHORT[s]} = {h16[s]['q']:.2f}" if h16[s]['q'] else f'{C.STAT_SHORT[s]}: ho chua du' for s in C.STATS))
        for i, p in enumerate(h16['ap_b']['phep_so']):
            log(f"    {'(3) − ' + p['ma']:<12}{CL.TITLES[p['dau']]:<11}"
                + ''.join(f"{C.STAT_SHORT[s]} {U.s3(h16[s]['phep_so'][i]['d']['ap'])} {U.ci(h16[s]['phep_so'][i]['ci_dong_thoi'])}  " for s in C.STATS)
                + ' / '.join(h16[s]['phep_so'][i]['ket_luan'] for s in C.STATS))
        for s in C.STATS:
            labs = [p['ket_luan'] for p in h16[s]['phep_so']]
            log(f"    {C.STAT_SHORT[s]:<8}: " + ', '.join(f'{v} {labs.count(v)}' for v in sorted(set(labs))) + f" / {len(labs)}"
                + f" | khoang dong thoi > 0: {sum(U.above(p['ci_dong_thoi']) == '>0' for p in h16[s]['phep_so'])}")
            for p in h16[s]['phep_so']:
                rows.append(dict(bang='viec_b_ho_16', dai_luong=C.STAT_SHORT[s], dieu_kien='Ver2 kcb, Viec B', dau=p['dau'], phep_tru=f"(3) - {p['ma']}", hieu=p['d']['ap'],
                                 ci_danh_nghia_thap=p['ci_danh_nghia'][0], ci_danh_nghia_cao=p['ci_danh_nghia'][1], ci_dong_thoi_thap=(p['ci_dong_thoi'] or [None])[0],
                                 ci_dong_thoi_cao=(p['ci_dong_thoi'] or [None, None])[1], dong_thoi=U.above(p['ci_dong_thoi']), dang_luat_cu_mo_ta=p['ket_luan']))
        log('    cung lr (mo ta; dat sau khi thay ket qua 19/11): hieu (3) − X AP28-B khi hai hang dung CUNG lr: '
            + ', '.join(f"{SPECS[t['doi_chung']].title.split(' ')[0]} {CL.TITLES[t['dau']]} {U.s3(t['d']['ap'])}" for t in ab['ap_b']['tach_lr'] if t['cach'] == 'cung_lr'))
        log('    bao kem (mo ta, CI95 danh nghia): ' + '; '.join(
            f"{SPECS[b['hang_a']].title.split(' ')[0]} − {SPECS[b['hang_b']].title.split(' ')[0]} {CL.TITLES[b['dau']]} {U.s3(b['d']['ap'])} {U.ci(b['ci_danh_nghia'])}"
            for b in ab['ap_b']['bao_kem']))
    o = {s: ctx['o28'][s] for s in C.STATS}
    log('\n=== (0) KHONG luong vung − RAD-DINO cong khai, dau doc co hoc (Ver2 kcb, TB 5 hat; CI95 danh nghia; mo ta) ===')
    rec0 = {}
    for h in S89.HEADS:
        k0, kr = ('v2', 'kcb', 'vt_khong_luong', h, 'luat'), ('v2', 'kcb', 'raddino', h, 'luat')
        if k0 not in o['ap_b'] or kr not in o['ap_b'] or o['ap_b'][k0]['khoa_nhan'] != o['ap_b'][kr]['khoa_nhan']:
            continue
        rr = {s: CP.diff(o[s][k0], o[s][kr]) for s in C.STATS}
        rec0[h] = {s: dict(d=rr[s]['d'], ci=rr[s]['ci'], nua=rr[s]['nua']) for s in C.STATS}
        log(f"    {CL.TITLES[h]:<11}" + '  '.join(f"{C.STAT_SHORT[s]} {U.s3(rr[s]['d'])} {U.ci(rr[s]['ci'])}" for s in C.STATS))
        for s in C.STATS:
            rows.append(dict(bang='khong_luong_tru_raddino', dai_luong=C.STAT_SHORT[s], dieu_kien='Ver2 kcb', dau=h, phep_tru='(0) - RAD-DINO cong khai', hieu=rr[s]['d'],
                             ci_danh_nghia_thap=rr[s]['ci'][0], ci_danh_nghia_cao=rr[s]['ci'][1]))
    ctx['khong_luong_raddino'] = rec0
    log('\n=== Δ cat giua − dem vuong cua ArrE (nhanh luat; CI95 danh nghia; khoang dong thoi o ho hinh hoc C6) ===')
    for i, dd in enumerate(cg['ap_b']['delta']):
        if dd['encoder'] != 'b':
            continue
        log(f"    {CL.TITLES[dd['dau']]:<11}lr {dd['lr_dem_vuong']:g} -> {dd['lr_cat_giua']:g}  "
            + '  '.join(f"{C.STAT_SHORT[s]} {U.s3(cg[s]['delta'][i]['d']['ap'])} {U.ci(cg[s]['delta'][i]['ci_danh_nghia'])}" for s in C.STATS))
    return dict(ablation=ab, cat_giua=cg, khong_luong_raddino=rec0)


def chexfound(ctx, log=print):
    '''C8 (MÔ TẢ). Ver1 (mã + môi trường của CheXFound) của CheXFound / RAD-DINO / BiomedCLIP so với Bảng IV bài CheXFound theo bốn cách tính AP và hai cách
       tính AUROC; độ rộng CI; vị trí số trong CI. Không vào luật, không vào bảng chính.'''
    from vindr_bench.evaluation import panels as P
    from vindr_bench.evaluation import collect as CL
    dl, out = ctx['dl'], []
    log('cach doc: bai CheXFound Bang IV in "AUPRC" (= AP bac thang) va AUROC kem CI95 bootstrap 1.000 mau; chu thich KHONG noi so in la TB bootstrap hay so '
        'diem (suy luan PLAN_86: TB bootstrap). Bon cach cua ta: (1) diem, bo Edema; (2) diem, Edema = 0 (chia 28); (3) TB bootstrap cach B; (4) TB bootstrap '
        'cach D; (5) TB cach D voi Edema = 0. Vi tri = (so - can duoi) / do rong CI cua chinh cach do.')
    for enc in ('chexfound', 'raddino', 'biomedclip'):
        for h in ('lin4', 'glori', 'lin1', 'mldec'):
            idx = [i for i, s in enumerate(dl['sets']) if (s.ver, s.store, s.encoder, s.head, s.arm) == ('v1', 'kcb', enc, h, 'luat')]
            if not idx:
                continue
            st = ctx['st28'][idx[0]]
            pub = P.PAPER_TABLE_IV.get(enc, {}).get(h)
            ap28 = np.asarray(st['ap28_lop'], dtype=np.float64)
            vals = dict(ap_diem=st['diem']['p28']['ap'], ap_diem_edema0=float(np.nan_to_num(ap28).sum() / len(ap28)),
                        ap_tb_b=float(np.mean(st['ap_b'])), ap_ci_b=U.pct_ci(st['ap_b']), ap_tb_d=float(np.mean(st['ap_d'])), ap_ci_d=U.pct_ci(st['ap_d']),
                        ap_tb_d_edema0=float(np.mean(st['ap_d']) * 27 / 28), ap_ci_d_edema0=[v * 27 / 28 for v in U.pct_ci(st['ap_d'])],
                        auroc_diem=st['diem']['p28']['auroc'], auroc_diem_bo_edema=st['diem']['p28']['auroc'] * 28 / 27,
                        auroc_tb=float(np.mean(st['auroc'])), auroc_ci=U.pct_ci(st['auroc']))
            pos = lambda v, c: (v - c[0]) / (c[1] - c[0]) if c[1] > c[0] else float('nan')
            rec = dict(encoder=enc, dau=h, **vals, bai=pub)
            if pub:
                rec.update(vi_tri_bai_ap=pos(pub['ap'], pub['ap_ci']), vi_tri_bai_auroc=pos(pub['auroc'], pub['auroc_ci']),
                           rong_bai_ap=pub['ap_ci'][1] - pub['ap_ci'][0], rong_bai_auroc=pub['auroc_ci'][1] - pub['auroc_ci'][0])
            rec.update(vi_tri_diem_b=pos(vals['ap_diem'], vals['ap_ci_b']), vi_tri_tb_b=pos(vals['ap_tb_b'], vals['ap_ci_b']),
                       vi_tri_tb_d=pos(vals['ap_tb_d'], vals['ap_ci_d']), vi_tri_auroc_diem=pos(vals['auroc_diem'], vals['auroc_ci']),
                       vi_tri_auroc_tb=pos(vals['auroc_tb'], vals['auroc_ci']), rong_b=vals['ap_ci_b'][1] - vals['ap_ci_b'][0],
                       rong_d=vals['ap_ci_d'][1] - vals['ap_ci_d'][0], rong_auroc=vals['auroc_ci'][1] - vals['auroc_ci'][0])
            out.append(rec)
            log(f"\n{enc} {CL.TITLES[h]} (Ver1, lr {dl['sets'][idx[0]].lr:g}):")
            log(f"    AP: (1) diem {U.f3(vals['ap_diem'])} | (2) Edema = 0 {U.f3(vals['ap_diem_edema0'])} | (3) TB B {U.f3(vals['ap_tb_b'])} {U.ci(vals['ap_ci_b'])} | (4) TB D "
                f"{U.f3(vals['ap_tb_d'])} {U.ci(vals['ap_ci_d'])} | (5) TB D Edema = 0 {U.f3(vals['ap_tb_d_edema0'])} {U.ci(vals['ap_ci_d_edema0'])}")
            log(f"    AUROC: diem {U.f3(vals['auroc_diem'])} (bo Edema {U.f3(vals['auroc_diem_bo_edema'])}) | TB {U.f3(vals['auroc_tb'])} {U.ci(vals['auroc_ci'])}")
            log(f"    vi tri trong CI cua ta: AP diem {U.f3(rec['vi_tri_diem_b'], 2)}, TB B {U.f3(rec['vi_tri_tb_b'], 2)}, TB D {U.f3(rec['vi_tri_tb_d'], 2)}; AUROC diem "
                f"{U.f3(rec['vi_tri_auroc_diem'], 2)}, TB {U.f3(rec['vi_tri_auroc_tb'], 2)} | do rong CI AP: B {U.f3(rec['rong_b'])}, D {U.f3(rec['rong_d'])}; AUROC "
                f"{U.f3(rec['rong_auroc'])}")
            if pub:
                log(f"    bai Bang IV: AP {U.f3(pub['ap'])} {U.ci(list(pub['ap_ci']))} (vi tri {U.f3(rec['vi_tri_bai_ap'], 2)}, rong {U.f3(rec['rong_bai_ap'])}) | AUROC "
                    f"{U.f3(pub['auroc'])} {U.ci(list(pub['auroc_ci']))} (vi tri {U.f3(rec['vi_tri_bai_auroc'], 2)}, rong {U.f3(rec['rong_bai_auroc'])})")
    ctx['chexfound28'] = out
    return out


def sync(ctx, log=print):
    '''C9. JSON + CSV + bộ nhớ đệm 28 nhãn -> dong_bo().'''
    d, t = ctx['drive'], ctx['t']
    names = []
    strip = lambda o: {'|'.join(str(x) for x in k): {kk: v for kk, v in c.items() if kk not in ('m21', 'm7')} for k, c in o.items()}
    obj = U.jsonable(dict(run=C.RUN_TAGS['bench'], t=t, goi=ctx['goi'], cach_tinh={s: C.STAT_TEXT[s] for s in C.STATS}, cach_doc=C.READING,
                          bootstrap=dict(n=C.N_BOOT, hat=C.BOOT_SEED, hat_nua=C.HALF_SEED), cong_89t=ctx.get('cong89'), thieu=ctx['dl']['thieu'],
                          o={s: strip(ctx['o28'][s]) for s in C.STATS}, ho=ctx.get('ho28'), ablation=ctx.get('ablation28'),
                          cat_giua={s: {k: v for k, v in x.items() if k != 'bang'} for s, x in (ctx.get('cat_giua28') or {}).items()},
                          khong_luong_raddino=ctx.get('khong_luong_raddino'), chexfound=ctx.get('chexfound28')))
    names.append(f'vindr28_benchmark_{t}.json')
    d['ghi'](names[-1], lambda q: d['ghi_json'](q, obj))
    for tag, rows in (('bang', ctx.get('bang28') or []), ('hieu', ctx.get('hieu28') or [])):
        n = f'vindr28_benchmark_{tag}_{t}.csv'
        if U.write_csv(d, n, rows):
            names.append(n)
    names.append(f'bo_dem_28_{t}.npz')
    U.write_cache(d, names[-1], ctx['cache28'])
    du = d['dong_bo']()
    co = set(names) <= set(d['ten_tren_drive']())
    log(f"C9: da ghi {len(names)} tep (JSON, {len(names) - 2} CSV, bo nho dem {len(ctx['cache28'])} bo) | dong_bo {'DU' if du else 'CHUA DU'} | tren Drive: "
        f"{'CO du' if co else 'THIEU'}" + ('' if du and co else ' | KHONG tat runtime; chay lai C9'))
    return dict(dong_bo=du, du_tep=co, tep=names)
