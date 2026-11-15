# -*- coding: utf-8 -*-
"""Ô của notebook tổng hợp `89t_tong_hop` (CPU): bảng, biểu đồ và phép so cho Ver1 + Ver2 từ kết quả ĐÃ CÓ trên Drive (PLAN_89 §7). Chạy được khi mới có
một phần kết quả: lượt thiếu được liệt kê, ô tương ứng để trống.

    C2 setup    hàm của khối Drive chuẩn
    C3 collect  kiểm kê + tải dự đoán test của mọi lượt Ver1 / Ver2 (+ Linear-4 / GLoRI của CheXFound từ `86f`, cổng PLAN_89 §4.3)
    C4 score    điểm 4 panel + bootstrap 2.000 lần theo ảnh (cùng dãy rút cho mọi lượt -> ghép cặp) -> ô (Ver2: TB 5 hạt)
    C5 tables   bảng từng điều kiện (ma trận encoder x đầu + bảng dài đủ điều kiện từng hàng) + Ver1 cạnh Ver2 (chỉ đặt cạnh, KHÔNG tính Ver2 - Ver1)
    C6 compare  hiệu ghép cặp: giữa encoder (từng đầu), giữa đầu đọc (từng encoder), giữa hai kho (Ver2), nhánh "theo AP" so với luật; họ so với hàng tham
                chiếu + khoảng đồng thời max-T khi có hàng B
    C7 figures  biểu đồ (evaluation/figures.py)
    C8 sync     ghi JSON + CSV + PNG, dong_bo()

Cách đọc (in kèm ở từng bảng): mAP21 = AP bậc thang macro 21 lớp (số của luật); AUPRC = hình thang; CI95 = bootstrap theo ảnh, DANH NGHĨA (chưa hiệu
chỉnh nhiều phép so) trừ họ max-T; Ver1 một hạt (sàng lọc); CI chứa 0 = chưa phát hiện khác biệt, không phải "bằng nhau"."""
import csv
import io
import json
import os
import time

import numpy as np

from .. import __version__
from ..encoders import ORDER, SPECS
from ..evaluation import collect as CL
from ..evaluation import compare as CP
from ..evaluation import figures as FG
from ..evaluation import panels as P

DRIVE_NAMES = ('kiem_ke', 'tai', 'xung_dot', 'ghi', 'ghi_json', 'dong_bo', 'ten_tren_drive', 'RUN_TAG', 'GHI_LOC')
RUN_TAG = 'r89t_tong_hop'
HEADS = CL.HEADS
FAMS = (('v1', 'kcb'), ('v2', 'kcb'), ('v2', 'cb'))
FAM_TEXT = {('v1', 'kcb'): 'Ver1 = ma + moi truong CheXFound (A100) | kho dem vuong KHONG can bang | luoi 9 lr | 1 luot hat 0 | dropout BAT khi cham',
            ('v2', 'kcb'): 'Ver2 = ma + moi truong cua ta (G4) | kho dem vuong KHONG can bang | luoi 13 lr | TB 5 hat | dau doc eval() khi cham',
            ('v2', 'cb'): 'Ver2 = ma + moi truong cua ta (G4) | kho dem vuong CO can bang histogram | luoi 13 lr | TB 5 hat | dau doc eval() khi cham'}
FAM_SHORT = {('v1', 'kcb'): 'Ver1 kcb', ('v2', 'kcb'): 'Ver2 kcb', ('v2', 'cb'): 'Ver2 cb'}
FAM_TITLE = {('v1', 'kcb'): 'Ver1 · đệm vuông, không cân bằng', ('v2', 'kcb'): 'Ver2 · đệm vuông, không cân bằng', ('v2', 'cb'): 'Ver2 · đệm vuông, có cân bằng'}
HEAD_PAIRS = (('lin4', 'lin1'), ('mldec', 'lin4'), ('glori', 'lin4'), ('glori', 'mldec'))
REFERENCE = 'b'                      # hàng tham chiếu của luật §7.4 (B - X); chưa có B -> chỉ có hiệu danh nghĩa


def _f(x, nd=3):
    return '  -  ' if x is None or x != x else f'{x:.{nd}f}'


def _d(x):
    return '   -  ' if x is None or x != x else f'{x:+.3f}'


def _row(enc):
    s = SPECS[enc]
    return f'{s.title} ({s.image_size}, {s.pool})'


def setup(ns, log=print):
    '''C2. -> CTX.'''
    missing = [k for k in DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    drive = {k: ns[k] for k in DRIVE_NAMES}
    assert drive['RUN_TAG'] == RUN_TAG, f"RUN_TAG {drive['RUN_TAG']!r} != {RUN_TAG!r} -> DUNG"
    import vindr_bench
    ctx = dict(drive=drive, t=time.strftime('%Y%m%d_%H%M%S'), goi=dict(phien_ban=__version__, sha=getattr(vindr_bench, 'GOI_SHA', None)), hinh={})
    log(f"tong hop 89 | goi vindr_bench {__version__} @ {(ctx['goi']['sha'] or '?')[:12]} | {os.cpu_count()} CPU | so cua luat: mAP21 = AP bac thang macro 21 lop; "
        f"bootstrap {P.N_BOOT} lan theo anh, hat {P.BOOT_SEED}; hai nua test hat {P.HALF_SEED}")
    return ctx


def collect(ctx, log=print):
    '''C3. Kiểm kê + tải. -> dict của evaluation.collect.collect.'''
    dl = CL.collect(ctx['drive'], [e for e in ORDER], CL.V2_STORES, log=log)
    ctx['dl'] = dl
    log('--- kiem ke luot (so bo du doan test da tai) ---')
    for k, n in dl['co'].items():
        log(f"  {'CO   ' if n else 'THIEU'} {k}: {n} bo")
    for m in dl['thieu']:
        if not m.startswith('Ver1 b:'):                                  # B chưa có Ver1 (notebook/ver1.NOT_VER1)
            log(f'  ! {m}')
    k = dl['kiem_86f']
    if k:
        log(f"cong dung lai 86f (PLAN_89 4.3) cho CheXFound {k['co']}: nhan test theo thu tu = luot 89: {'TRUNG' if k['nhan_test_trung_89'] else 'KHAC'} | luoi du 9 lr "
            f"cua bai: {k['luoi_du_9lr']} | GLoRI chia nua {k['glori_hai_nua']} (89: 5 + 4) | tham so dong lenh da so bang ma o commit ghim (Event 14/11)")
    keys = {}
    for s in dl['sets']:
        keys.setdefault(CP.target_key(s.target), set()).add((s.ver, s.store))
    ctx['khoa_nhan'] = {k: sorted(v) for k, v in keys.items()}
    log(f"{len(dl['sets'])} bo du doan | {len(dl['nhan'])} nhan, {len(dl['lop_chinh'])} lop so chinh | bo nhan test theo thu tu: {len(keys)} "
        + ('(mot bo -> moi luot ghep cap theo anh duoc)' if len(keys) == 1 else f'-> KHAC NHAU giua {ctx["khoa_nhan"]}: chi ghep cap cac luot cung bo nhan'))
    return dl


CACHE_RE = r'bo_dem_bootstrap_\d{8}_\d{6}\.npz'


def load_cache(ctx, log=print):
    '''Bộ nhớ đệm bootstrap MỚI NHẤT trên Drive (stage4_nb89/r89t_tong_hop; ghi ở C8) -> {khoá: set_stats}; chưa có / phiên bản cách tính khác -> {}.'''
    d = ctx['drive']
    kk = d['kiem_ke']((CL.FOLDER, RUN_TAG), mau=CACHE_RE)[1]
    if not kk:
        return {}
    name = sorted(kk)[-1]
    cache = CP.load_cache(d['tai'](kk[name][0]))
    log(f"bo nho dem bootstrap: {name} -> {len(cache)} bo" + ('' if cache else ' (phien ban cach tinh khac -> tinh lai)'))
    return cache


def score(ctx, log=print, n_boot=P.N_BOOT, workers=None):
    '''C4. -> {(ver, kho, encoder, đầu, nhánh): ô}. Bộ đã có trong bộ nhớ đệm (cùng dự đoán, nhãn, cột, tham số) không tính lại.'''
    dl, t0 = ctx['dl'], time.time()
    cols = {}
    for s in dl['sets']:
        k = CP.target_key(s.target)
        if k not in cols:
            cols[k] = P.panel_columns(dl['nhan'], dl['lop_chinh'], s.target)
    workers = workers or min(8, os.cpu_count() or 1)
    jobs = [(s.preds, s.target, cols[CP.target_key(s.target)], n_boot, P.BOOT_SEED, P.HALF_SEED) for s in dl['sets']]
    keys = [CP.stats_key(*j) for j in jobs]
    cache = ctx['bo_dem'] if 'bo_dem' in ctx else load_cache(ctx, log)
    todo = [i for i, k in enumerate(keys) if k not in cache]
    log(f"{len(jobs)} bo du doan | da co trong bo nho dem bootstrap (Drive): {len(jobs) - len(todo)} | can tinh: {len(todo)} "
        f"({workers} tien trinh; ~4–9 giay / bo / nhan CPU)")
    for i, st in zip(todo, CP.many_stats([jobs[i] for i in todo], workers, log=log)):
        cache[keys[i]] = st
    stats = [cache[k] for k in keys]
    ctx.update(bo_dem=cache, bo_dem_moi=len(todo))
    groups = {}
    for s, st in zip(dl['sets'], stats):
        groups.setdefault(s.cell, []).append((s, st))
    cells = {}
    for key, items in groups.items():
        items.sort(key=lambda x: x[0].seed)
        s0 = items[0][0]
        ens = None
        if len(items) > 1:
            mean = np.mean([s.preds for s, _ in items], 0)
            c = cols[CP.target_key(s0.target)]
            ens = {p: {m: float(v) for m, v in P.point(mean, s0.target, c[p]).items() if m in ('ap', 'auprc_thang', 'auroc')} for p in P.PANELS}
        cell = CP.make_cell([st for _, st in items], ens)
        cell.update(lr=s0.lr, bien=s0.edge, nguon=s0.source, val_auroc=s0.val_auroc, khoa_nhan=CP.target_key(s0.target), hat=[s.seed for s, _ in items])
        cells[key] = cell
    ctx.update(o=cells, n_boot=n_boot, cot=cols)
    log(f"{len(dl['sets'])} bo du doan -> {len(cells)} o (phien ban, kho, encoder, dau, nhanh) | bootstrap {n_boot} lan / bo, {workers} tien trinh | "
        f"lan rut bi loai (thieu lop): p21 toi da {max(c['loai21'] for c in cells.values())}, p7 toi da {max(c['loai7'] for c in cells.values())} | "
        f"{(time.time() - t0) / 60:.1f} phut")
    return cells


def _encs(ctx, fam, arm='luat'):
    return [e for e in ORDER if any((fam[0], fam[1], e, h, arm) in ctx['o'] for h in HEADS)]


def tables(ctx, log=print):
    '''C5. In bảng; -> list hàng (bảng dài) để ghi CSV.'''
    o, rows = ctx['o'], []
    for fam in FAMS:
        encs = _encs(ctx, fam)
        if not encs:
            log(f'\n=== {FAM_TEXT[fam]} === CHUA CO KET QUA')
            continue
        log(f'\n=== {FAM_TEXT[fam]} ===')
        log('mAP21 (AP bac thang macro 21 lop, diem tren toan test 3.000 anh; * = lr chon o BIEN luoi)')
        log(f"{'hang (anh, vector toan anh)':<34}" + ''.join(f'{CL.TITLES[h]:>13}' for h in HEADS))
        for e in encs:
            log(f'{_row(e):<34}' + ''.join(f"{(_f(o[(fam[0], fam[1], e, h, 'luat')]['diem']['p21']['ap']) + ('*' if o[(fam[0], fam[1], e, h, 'luat')]['bien'] else ' ')) if (fam[0], fam[1], e, h, 'luat') in o else '  -   ':>13}"
                                           for h in HEADS))
        log(f"\n{'hang':<28}{'dau':<11}{'nhanh':<6}{'lr':>8}  {'mAP21 [CI95 danh nghia]':<24}{'SD hat':>7}{'ens':>7}{'AUPRC21':>8}{'AUROC21':>8}{'AP28':>7}{'AUROC28':>8}"
            f"{'AP27':>7}{'AP7':>7}  {'hai nua AP21':<14}nguon")
        for e in encs:
            for h in HEADS:
                for arm in ('luat', 'ap'):
                    c = o.get((fam[0], fam[1], e, h, arm))
                    if c is None:
                        continue
                    d, sd, en = c['diem'], c['sd']['p21']['ap'], (c['ensemble'] or {}).get('p21', {}).get('ap')
                    log(f"{SPECS[e].title + ' ' + str(SPECS[e].image_size):<28}{CL.TITLES[h]:<11}{arm:<6}{c['lr']:>8g}{'*' if c['bien'] else ' '} "
                        f"{_f(d['p21']['ap'])} [{_f(c['ci21'][0])}, {_f(c['ci21'][1])}]  {_f(sd, 4) if sd is not None else '   -  ':>7}{_f(en):>7}"
                        f"{_f(d['p21']['auprc_thang']):>8}{_f(d['p21']['auroc']):>8}{_f(d['p28']['ap']):>7}{_f(d['p28']['auroc']):>8}{_f(d['p27']['ap']):>7}"
                        f"{_f(d['p7']['ap']):>7}  {_f(c['nua'][0])} / {_f(c['nua'][1])}  {c['nguon']}")
                    rows.append(dict(phien_ban=fam[0], kho=fam[1], dieu_kien=FAM_TEXT[fam], encoder=e, hang=_row(e), dau=h, nhanh=arm, lr=c['lr'], lr_o_bien=c['bien'],
                                     so_hat=c['so_hat'], mAP21=d['p21']['ap'], ci_thap=c['ci21'][0], ci_cao=c['ci21'][1], sd_giua_hat=sd, ensemble_mAP21=en,
                                     AUPRC21_hinh_thang=d['p21']['auprc_thang'], AUROC21=d['p21']['auroc'], AP28_torchmetrics=d['p28']['ap'],
                                     AUROC28_torchmetrics=d['p28']['auroc'], AP27=d['p27']['ap'], AUPRC27_hinh_thang=d['p27']['auprc_thang'], AP7=d['p7']['ap'],
                                     AUPRC7_hinh_thang=d['p7']['auprc_thang'], AP21_nua_0=c['nua'][0], AP21_nua_1=c['nua'][1], lan_rut_bi_loai=c['loai21'],
                                     nguon=c['nguon']))
    log('\n=== Ver1 canh Ver2 (mAP21; CHI DAT CANH — khac ma, moi truong, luoi lr, so hat, che do cham; KHONG tinh Ver2 - Ver1, PLAN_89 7.4) ===')
    side = {}
    for h in HEADS:
        log(f"--- {CL.TITLES[h]} --- {'hang':<28}" + ''.join(f'{FAM_SHORT[f]:>12}{"hang thu":>9}' for f in FAMS))
        val = {f: {e: ctx['o'][(f[0], f[1], e, h, 'luat')]['diem']['p21']['ap'] for e in ORDER if (f[0], f[1], e, h, 'luat') in ctx['o']} for f in FAMS}
        rank = {f: {e: 1 + sorted(v.values(), reverse=True).index(x) for e, x in v.items()} for f, v in val.items()}
        for e in [e for e in ORDER if any(e in val[f] for f in FAMS)]:
            log(f"{'':<4}{SPECS[e].title + ' ' + str(SPECS[e].image_size):<28}" + ''.join(f'{_f(val[f].get(e)):>12}{rank[f].get(e, "-"):>9}' for f in FAMS))
        sp = {}
        for a, b in ((FAMS[0], FAMS[1]), (FAMS[1], FAMS[2])):
            common = [e for e in val[a] if e in val[b]]
            sp[f'{FAM_SHORT[a]} ~ {FAM_SHORT[b]}'] = (CP.spearman([val[a][e] for e in common], [val[b][e] for e in common]), len(common))
        log(f"{'':<4}tuong quan hang (Spearman, mo ta): " + ' | '.join(f'{k}: {_f(v[0], 2)} ({v[1]} encoder)' for k, v in sp.items()))
        side[h] = dict(mAP21={FAM_SHORT[f]: val[f] for f in FAMS}, hang={FAM_SHORT[f]: rank[f] for f in FAMS}, spearman={k: v[0] for k, v in sp.items()})
    ctx.update(bang_dai=rows, canh_nhau=side)
    return rows


def _pair_ok(a, b):
    return a is not None and b is not None and a['khoa_nhan'] == b['khoa_nhan']


def compare(ctx, log=print, reference=REFERENCE):
    '''C6. Hiệu ghép cặp (CHỈ MÔ TẢ trừ họ max-T). -> dict.'''
    o, out = ctx['o'], dict(giua_encoder={}, ho_tham_chieu={}, giua_dau={}, giua_kho=[], nhanh_ap=[])
    cell = lambda fam, e, h, arm='luat': o.get((fam[0], fam[1], e, h, arm))
    log("cach doc: hieu = hang - cot tren mAP21, ghep cap theo anh (cung lan rut); [..] = CI95 DANH NGHIA; dau '+' / '-' = dat dang luat 7.4 mot cach danh "
        "nghia (|hieu| >= 0,01, CI khong chua 0, cung dau o hai nua); CI chua 0 = CHUA phat hien khac, khong phai 'bang nhau'. Ver1: mot hat.")
    for fam in FAMS:
        encs = _encs(ctx, fam)
        if len(encs) < 2:
            continue
        table = {}
        log(f'\n=== hieu giua encoder | {FAM_TEXT[fam]} ===')
        for h in HEADS:
            log(f"--- {CL.TITLES[h]}: hang - cot --- {'':<14}" + ''.join(f'{SPECS[b].title[:10]:>11}' for b in encs))
            for a in encs:
                line = f"{'':<4}{SPECS[a].title:<14}"
                for b in encs:
                    if a == b or not _pair_ok(cell(fam, a, h), cell(fam, b, h)):
                        line += f"{'·':>11}"
                        continue
                    r = CP.diff(cell(fam, a, h), cell(fam, b, h))
                    table[(h, a, b)] = r
                    line += f"{_d(r['d']) + {'hon': '+', 'kem': '-'}.get(r['ket_luan'], ' '):>11}"
                log(line)
        out['giua_encoder'][FAM_SHORT[fam]] = {f'{h}|{a}|{b}': dict(hieu=r['d'], ci=r['ci'], nua=r['nua'], dang_luat_danh_nghia=r['ket_luan']) for (h, a, b), r in table.items()}
        ctx.setdefault('cap', {})[fam] = table
        if reference in encs:
            comps = [(h, x) for h in HEADS for x in encs if x != reference and (h, reference, x) in table]
            q, sim = CP.max_t([table[(h, reference, x)] for h, x in comps])
            log(f'--- HO {len(comps)} phep so {SPECS[reference].title} - X (luat PLAN_89 7.4), khoang DONG THOI max-T 95 %: q = {q:.2f} ---')
            for (h, x), s in zip(comps, sim):
                r = table[(h, reference, x)]
                log(f"{'':<4}{CL.TITLES[h]:<11}{SPECS[reference].title} - {SPECS[x].title:<12} {_d(s['d'])} dong thoi [{_d(s['ci_dong_thoi'][0])}, {_d(s['ci_dong_thoi'][1])}] | danh nghia "
                    f"[{_d(r['ci'][0])}, {_d(r['ci'][1])}] | hai nua {_d(r['nua'][0])} / {_d(r['nua'][1])} -> {s['ket_luan']}")
            out['ho_tham_chieu'][FAM_SHORT[fam]] = dict(tham_chieu=reference, q=q, phep_so=[dict(dau=h, doi_chung=x, **s) for (h, x), s in zip(comps, sim)])
        else:
            log(f'--- chua co hang {reference.upper()} trong dieu kien nay -> CHUA ap luat 7.4 (ho B - X + max-T); cac hieu tren chi la danh nghia ---')
        log(f"--- hieu giua DAU DOC (cung encoder) --- {'hang':<16}" + ''.join(f"{CL.TITLES[a] + ' - ' + CL.TITLES[b]:>36}" for a, b in HEAD_PAIRS))
        hd = {}
        for e in encs:
            line = f"{'':<4}{SPECS[e].title:<16}"
            for a, b in HEAD_PAIRS:
                if not _pair_ok(cell(fam, e, a), cell(fam, e, b)):
                    line += f"{'-':>36}"
                    continue
                r = CP.diff(cell(fam, e, a), cell(fam, e, b))
                hd[f'{e}|{a}-{b}'] = dict(hieu=r['d'], ci=r['ci'], dang_luat_danh_nghia=r['ket_luan'])
                line += f"{_d(r['d']) + ' [' + _d(r['ci'][0]) + ', ' + _d(r['ci'][1]) + ']' + {'hon': '+', 'kem': '-'}.get(r['ket_luan'], ' '):>36}"
            log(line)
        out['giua_dau'][FAM_SHORT[fam]] = hd
    log('\n=== Ver2: kho CO can bang - kho KHONG can bang (cung encoder, cung dau; hai luot hoc rieng -> CI theo anh thieu bien thien huan luyen; mo ta) ===')
    for e in ORDER:
        line, any_ = f"{'':<4}{SPECS[e].title:<16}", False
        for h in HEADS:
            a, b = cell(('v2', 'cb'), e, h), cell(('v2', 'kcb'), e, h)
            if not _pair_ok(a, b):
                line += f"{CL.TITLES[h] + ': -':>40}"
                continue
            r, any_ = CP.diff(a, b), True
            out['giua_kho'].append(dict(encoder=e, dau=h, hieu=r['d'], ci=r['ci'], dang_luat_danh_nghia=r['ket_luan']))
            line += f"{CL.TITLES[h] + ' ' + _d(r['d']) + ' [' + _d(r['ci'][0]) + ', ' + _d(r['ci'][1]) + ']':>40}"
        if any_:
            log(line)
    if not out['giua_kho']:
        log('    chua co cap luot nao du hai kho (hoac hai kho khac thu tu anh)')
    log("\n=== Ver2: nhanh 'theo AP' (lr co AP val cao nhat) - LUAT (lr co AUROC val cao nhat), chi cac o hai lr KHAC nhau; mo ta, khong vao luat ===")
    for fam in FAMS[1:]:
        for e in ORDER:
            for h in HEADS:
                a, b = cell(fam, e, h, 'ap'), cell(fam, e, h)
                if not _pair_ok(a, b):
                    continue
                r = CP.diff(a, b)
                out['nhanh_ap'].append(dict(dieu_kien=FAM_SHORT[fam], encoder=e, dau=h, lr_ap=a['lr'], lr_luat=b['lr'], hieu=r['d'], ci=r['ci']))
                log(f"    {FAM_SHORT[fam]:<9}{SPECS[e].title:<14}{CL.TITLES[h]:<11} lr theo AP {a['lr']:g} (mAP21 {_f(a['diem']['p21']['ap'])}) - lr cua luat {b['lr']:g} "
                    f"(mAP21 {_f(b['diem']['p21']['ap'])}) = {_d(r['d'])} [{_d(r['ci'][0])}, {_d(r['ci'][1])}]")
    if not out['nhanh_ap']:
        log('    khong co o nao (chua co luot Ver2, hoac hai tieu chi chon cung lr o moi o)')
    ctx['so_sanh'] = out
    return out


def figures(ctx, log=print, show=True):
    '''C7. Vẽ + giữ Figure (ghi PNG ở C8). -> {tên: Figure}.'''
    o = ctx['o']
    flat = {k[:4]: v for k, v in o.items() if k[4] == 'luat'}
    conds = [f for f in FAMS if _encs(ctx, f)]
    encs = [e for e in ORDER if any(k[2] == e for k in flat)]
    order = sorted(encs, key=lambda e: -max(v['diem']['p21']['ap'] for k, v in flat.items() if k[2] == e))      # cùng thứ tự ở mọi ô hình
    et, ht = {e: SPECS[e].title for e in encs}, CL.TITLES
    figs = dict(cham_mAP21=FG.dots(flat, order, HEADS, conds, et, ht, FAM_TITLE), luoi_mAP21=FG.heat(flat, order, HEADS, conds, et, ht, FAM_TITLE))
    if FAMS[0] in conds and FAMS[1] in conds:
        figs['ver1_so_ver2'] = FG.versus(flat, order, HEADS, FAMS[0], FAMS[1], et, ht, FAM_TITLE)
    for fam, table in ctx.get('cap', {}).items():
        if table:
            figs[f'hieu_giua_encoder_{fam[0]}_{fam[1]}'] = FG.pairs(table, [e for e in order if e in _encs(ctx, fam)], HEADS, et, ht,
                                                                     f'Hiệu mAP 21 lớp giữa encoder (hàng − cột), ghép cặp theo ảnh — {FAM_TITLE[fam]}; số: ô đạt dạng luật (danh nghĩa)')
    for st in CL.V2_STORES:
        for metric, name in (('auc', 'AUROC'), ('ap', 'AP')):
            f = FG.lr_curves(ctx['dl']['duong_lr'], ctx['dl']['chon_v2'], order, HEADS, st, metric, et, ht,
                             f'Ver2 · kho {st}: {name} val (lần chấm cuối pha 1, hạt 0) theo lr — chỗ chọn lr')
            if f is not None:
                figs[f'do_lr_{metric}_{st}'] = f
    ctx['hinh'] = figs
    log(f'{len(figs)} hinh: {sorted(figs)}')
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return figs


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))


def sync(ctx, log=print):
    '''C8. Ghi tổng hợp (JSON), bảng dài (CSV), hình (PNG) rồi dong_bo().'''
    d, t = ctx['drive'], time.strftime('%Y%m%d_%H%M%S')
    names = []
    cells = {'|'.join(k): {kk: v for kk, v in c.items() if kk not in ('m21', 'm7')} for k, c in ctx['o'].items()}
    obj = _jsonable(dict(run=d['RUN_TAG'], t=t, goi=ctx['goi'], n_boot=ctx['n_boot'], hat_boot=P.BOOT_SEED, hat_nua=P.HALF_SEED, dieu_kien={FAM_SHORT[f]: FAM_TEXT[f] for f in FAMS},
                         co=ctx['dl']['co'], thieu=ctx['dl']['thieu'], kiem_86f=ctx['dl']['kiem_86f'], khoa_nhan={k: [list(x) for x in v] for k, v in ctx['khoa_nhan'].items()},
                         o=cells, canh_nhau=ctx.get('canh_nhau'), so_sanh=ctx.get('so_sanh'),
                         ghi_chu='mAP21 = AP bac thang macro 21 lop; CI danh nghia tru ho max-T; Ver2 = TB 5 hat; khong tinh Ver2 - Ver1 (PLAN_89 7.4)'))
    names.append(f'tong_hop_89_{t}.json')
    d['ghi'](names[-1], lambda q: d['ghi_json'](q, obj))
    rows = ctx.get('bang_dai') or []
    if rows:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        text = buf.getvalue()
        names.append(f'bang_dai_89_{t}.csv')

        def _csv(q):
            with open(q, 'w', encoding='utf-8-sig', newline='') as f:
                f.write(text)
        d['ghi'](names[-1], _csv)
    for name, fig in ctx.get('hinh', {}).items():
        names.append(f'hinh_{name}_{t}.png')
        d['ghi'](names[-1], lambda q, fig=fig: fig.savefig(q, dpi=160, bbox_inches='tight'))
    n_cache = 0
    if ctx.get('bo_dem_moi'):                                     # có bộ mới tính -> ghi bộ nhớ đệm đầy đủ (lần sau chỉ tính lượt mới)
        names.append(f'bo_dem_bootstrap_{t}.npz')
        d['ghi'](names[-1], lambda q: CP.save_cache(q, ctx['bo_dem']))
        n_cache = 1
    du = d['dong_bo']()
    on = d['ten_tren_drive']()
    co = set(names) <= on
    log(f"C8: da ghi {len(names)} tep (1 JSON, {'1 CSV, ' if rows else ''}{len(ctx.get('hinh', {}))} PNG"
        + (f", bo nho dem bootstrap {len(ctx['bo_dem'])} bo" if n_cache else ', bo nho dem khong doi') + f") | dong_bo {'DU' if du else 'CHUA DU'} | tren Drive: "
        f"{'CO du' if co else 'THIEU'}" + ('' if du and co else ' | KHONG tat runtime; chay lai C8'))
    return dict(dong_bo=du, du_tep=co, tep=names)
