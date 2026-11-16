# -*- coding: utf-8 -*-
"""Ô của notebook tổng hợp `89t_tong_hop` (CPU): bảng, biểu đồ và phép so cho Ver1 + Ver2 từ kết quả ĐÃ CÓ trên Drive (PLAN_89 §7). Chạy được khi mới có
một phần kết quả: lượt thiếu được liệt kê, ô tương ứng để trống.

    C2 setup    hàm của khối Drive chuẩn
    C3 collect  kiểm kê + tải dự đoán test của mọi lượt Ver1 / Ver2 (+ Linear-4 / GLoRI của CheXFound từ `86f`, cổng PLAN_89 §4.3)
    C4 score    điểm 4 panel + bootstrap 2.000 lần theo ảnh (cùng dãy rút cho mọi lượt -> ghép cặp) -> ô (Ver2: TB 5 hạt)
    C5 tables   bảng từng điều kiện (ma trận encoder x đầu + bảng dài đủ điều kiện từng hàng) + Ver1 cạnh Ver2 (chỉ đặt cạnh, KHÔNG tính Ver2 - Ver1)
    C6 compare  hiệu ghép cặp: giữa encoder (từng đầu), giữa đầu đọc (từng encoder), giữa hai kho (Ver2), nhánh "theo AP" so với luật; họ so với hàng tham
                chiếu + khoảng đồng thời max-T khi có hàng B; 0.9.0: hai họ HÌNH HỌC (kho cắt giữa − kho đệm vuông; 28 phép khoá 13/11 + 60 phép mở rộng
                16/11 — protocols/geometry.py), mỗi họ max-T riêng khi đủ phép
    C7 figures  biểu đồ 28 nhãn — AP bậc thang, AUROC, AUPRC hình thang, điểm không CI (evaluation/figures.py; Tuấn 15/11: hình chỉ cần 28 nhãn); kho cắt
                giữa: ô thứ tư của hình lưới + hình hiệu cắt giữa − đệm vuông (hình chấm giữ 3 điều kiện — màu); 0.9.1 (Tuấn 16/11): 7 dấu hiệu Bảng 1a bài
                RAD-DINO — TB 7 (AUPRC hình thang + AP bậc thang: chấm + lưới) và AUPRC từng dấu hiệu (mỗi điều kiện một hình)
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
from ..protocols import geometry as GEO

DRIVE_NAMES = ('kiem_ke', 'tai', 'xung_dot', 'ghi', 'ghi_json', 'dong_bo', 'ten_tren_drive', 'RUN_TAG', 'GHI_LOC')
RUN_TAG = 'r89t_tong_hop'
HEADS = CL.HEADS
GEO_FAM, PAD_FAM = ('v2', GEO.CROP_STORE), ('v2', GEO.PAD_STORE)      # phép thử hình học (PLAN_89 §5.3): cắt giữa so với đệm vuông, cùng kiểu histogram
FAMS = (('v1', 'kcb'), ('v2', 'kcb'), ('v2', 'cb'), GEO_FAM)
RULE_FAMS = FAMS[:3]                 # ba họ "B − X" 32 phép của luật §7.4; B − X trên kho cắt giữa thuộc họ hình học (phần (c))
FAM_TEXT = {('v1', 'kcb'): 'Ver1 = ma + moi truong CheXFound (A100) | kho dem vuong KHONG can bang | luoi 9 lr | 1 luot hat 0 | dropout BAT khi cham',
            ('v2', 'kcb'): 'Ver2 = ma + moi truong cua ta (G4) | kho dem vuong KHONG can bang | luoi 13 lr | TB 5 hat | dau doc eval() khi cham',
            ('v2', 'cb'): 'Ver2 = ma + moi truong cua ta (G4) | kho dem vuong CO can bang histogram | luoi 13 lr | TB 5 hat | dau doc eval() khi cham',
            GEO_FAM: ('Ver2 = ma + moi truong cua ta (G4) | kho CAT GIUA (co canh ngan + cat giua) KHONG can bang | luoi 13 lr | TB 5 hat | dau doc eval() '
                      'khi cham | chi 8 hang cua phep thu hinh hoc')}
FAM_SHORT = {('v1', 'kcb'): 'Ver1 kcb', ('v2', 'kcb'): 'Ver2 kcb', ('v2', 'cb'): 'Ver2 cb', GEO_FAM: 'Ver2 cg_kcb'}
FAM_TITLE = {('v1', 'kcb'): 'Ver1 · đệm vuông, không cân bằng', ('v2', 'kcb'): 'Ver2 · đệm vuông, không cân bằng', ('v2', 'cb'): 'Ver2 · đệm vuông, có cân bằng',
             GEO_FAM: 'Ver2 · cắt giữa, không cân bằng'}
HEAD_PAIRS = (('lin4', 'lin1'), ('mldec', 'lin4'), ('glori', 'lin4'), ('glori', 'mldec'))
REFERENCE = 'b'                      # hàng tham chiếu của luật §7.4 (B - X); chưa có B -> chỉ có hiệu danh nghĩa
assert REFERENCE == GEO.REFERENCE


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
    for m in dl['thieu']:                                                # 0.8.0: B có Ver1 -> lượt Ver1 của B chưa chạy cũng được liệt kê
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
        # 0.9.1: TỪNG dấu hiệu của 7 dấu hiệu RAD-DINO (điểm; TB các hạt như số macro) — tính thẳng từ dự đoán, KHÔNG vào bộ nhớ đệm bootstrap
        c7 = cols[CP.target_key(s0.target)]['p7']
        per7 = [P.point(s.preds, s.target, c7) for s, _ in items]
        cell['p7_lop'] = {m: [float(v) for v in np.mean([x[f'{m}_lop'] for x in per7], 0)] for m in ('ap', 'auprc_thang', 'auroc')}
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
        for a, b in ((FAMS[0], FAMS[1]), (FAMS[1], FAMS[2]), (PAD_FAM, GEO_FAM)):
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
        if fam == GEO_FAM:
            log(f'--- {SPECS[reference].title} - X tren kho cat giua: LUAT o phan (c) cua hai ho hinh hoc (cuoi C6), khong lap ho 32 phep o day ---')
        elif reference in encs:
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
    out['hinh_hoc'] = geometry(ctx, cell, log)
    ctx['so_sanh'] = out
    return out


def _per_seed(*pairs):
    '''Δ từng hạt (cùng chỉ số hạt ở mọi ô): pairs = [(dấu, ô)] -> [tổng dấu x AP21 của hạt k] cho các hạt có ở MỌI ô (hai lượt học riêng ghép theo hạt).'''
    common = sorted(set.intersection(*[set(c['hat']) for _, c in pairs]))
    return [float(sum(s * dict(zip(c['hat'], c['tung_hat']))[k] for s, c in pairs)) for k in common]


def geometry(ctx, cell, log=print):
    '''Hai họ phép thử hình học (PLAN_89 §5.3, §7.4; protocols/geometry.py): mỗi họ ĐỦ phép -> khoảng đồng thời max-T 95 % riêng của họ + luật từng dạng;
       chưa đủ -> chỉ hiệu danh nghĩa, "chưa áp luật". Δ = mAP21 kho cắt giữa − kho đệm vuông KHÔNG cân bằng (cùng encoder, cùng đầu, Ver2); kèm Δ từng hạt.'''
    out = {}
    log('\n=== HINH HOC (PLAN_89 5.3 / 7.4): Δ = mAP21(kho CAT GIUA) - mAP21(kho DEM VUONG, cung KHONG can bang), cung encoder, cung dau, Ver2 (hai luot hoc '
        'rieng, moi ben TB 5 hat -> CI theo anh THIEU bien thien huan luyen; Δ tung hat in kem). Luat: (a) Δ "te di nhieu" = Δ <= -0,01 VA can tren dong '
        'thoi < 0, "khong te di" = can duoi dong thoi > -0,01; (b) Δ_B - Δ_X < 0 voi can tren < 0 -> dem vuong nghieng ve B, > 0 voi can duoi > 0 -> nghieng '
        've X; (c) B - X tren kho cat giua theo luat ho 32 ===')
    ref = GEO.REFERENCE
    for name, fam in GEO.FAMILIES.items():
        rows, items, need = fam['rows'], [], 0
        for h in HEADS:                                                             # (a) Δ_E
            for e in rows:
                need += 1
                a1, a0 = cell(GEO_FAM, e, h), cell(PAD_FAM, e, h)
                if _pair_ok(a1, a0):
                    items.append(('a', h, e, CP.diff(a1, a0), CP.verdict_delta, _per_seed((1, a1), (-1, a0))))
        for h in HEADS:                                                             # (b) Δ_B − Δ_X
            for x in rows:
                if x == ref:
                    continue
                need += 1
                b1, b0, x1, x0 = cell(GEO_FAM, ref, h), cell(PAD_FAM, ref, h), cell(GEO_FAM, x, h), cell(PAD_FAM, x, h)
                if all(_pair_ok(b1, c) for c in (b0, x1, x0)):
                    items.append(('b', h, x, CP.diff_of_diffs(b1, b0, x1, x0), CP.verdict_interaction, _per_seed((1, b1), (-1, b0), (-1, x1), (1, x0))))
        for h in HEADS:                                                             # (c) B − X trên kho cắt giữa
            for x in rows:
                if x == ref:
                    continue
                need += 1
                b1, x1 = cell(GEO_FAM, ref, h), cell(GEO_FAM, x, h)
                if _pair_ok(b1, x1):
                    items.append(('c', h, x, CP.diff(b1, x1), CP.verdict, _per_seed((1, b1), (-1, x1))))
        full = bool(items) and len(items) == need
        log(f"\n--- {fam['title']} | {len(items)} / {need} phep so co du lieu"
            + (' -> khoang DONG THOI max-T 95 % cua ho' if full else ' -> HO CHUA DU: chi hieu DANH NGHIA, CHUA ap luat ---'))
        q, sim = CP.max_t([it[3] for it in items], rules=[it[4] for it in items]) if full else (float('nan'), [None] * len(items))
        if full:
            log(f'    q = {q:.2f} (ho {need} phep)')
        dang = {'a': '(a) Δ', 'b': f'(b) Δ_{SPECS[ref].title} - Δ_X', 'c': f'(c) {SPECS[ref].title} - X cat giua'}
        rec = []
        for (kind, h, x, r, rule, seeds), s in zip(items, sim):
            ci_s = s['ci_dong_thoi'] if s else None
            lab = s['ket_luan'] if s else f"danh nghia: {rule(r['d'], r['ci'][0], r['ci'][1], r['nua'])}"
            log(f"    {dang[kind]:<22}{CL.TITLES[h]:<11}{SPECS[x].title:<22}{_d(r['d'])}"
                + (f" dong thoi [{_d(ci_s[0])}, {_d(ci_s[1])}]" if ci_s else '') + f" | danh nghia [{_d(r['ci'][0])}, {_d(r['ci'][1])}] | hai nua "
                f"{_d(r['nua'][0])} / {_d(r['nua'][1])} | tung hat {' '.join(_d(v) for v in seeds)} -> {lab}")
            rec.append(dict(dang=kind, dau=h, encoder=x, d=r['d'], ci_danh_nghia=r['ci'], ci_dong_thoi=ci_s, ket_luan=lab, nua=r['nua'], tung_hat=seeds))
        out[name] = dict(tieu_de=fam['title'], so_phep=need, co=len(items), day_du=full, q=q if full else None, phep_so=rec)
    return out


def _note28(ctx):
    '''Câu ghi chú chung của hình 28 nhãn, dựng từ DỮ LIỆU (không chép tên lớp cố định): lớp nào không có ca dương ở test -> bỏ khỏi AP / AUPRC, AUROC = 0.'''
    names = list(ctx['dl']['nhan'])
    miss = sorted({tuple(names[j] for j in range(len(names)) if j not in set(c['p27'])) for c in ctx['cot'].values()})
    n_img = sorted({len(s.target) for s in ctx['dl']['sets']})
    n = len(names)
    if len(miss) == 1 and miss[0]:
        m, k = list(miss[0]), n - len(miss[0])
        cls = (f"{', '.join(m)} không có ca dương ở test -> AP và AUPRC là trung bình {k} nhãn có ca dương; AUROC tính {'lớp đó' if len(m) == 1 else 'các lớp đó'} "
               f"= 0 (AUROC {n} = {k}/{n} x AUROC {k} nhãn, thứ hạng không đổi).")
    elif len(miss) == 1:
        cls = 'mọi nhãn có ca dương ở test.'
    else:
        cls = f"lớp không có ca dương KHÁC NHAU giữa các bộ nhãn test ({miss}) -> so giữa các bộ đó phải đọc thận trọng."
    imgs = '/'.join(f'{v:,}'.replace(',', '.') for v in n_img)
    return (f'{n} nhãn, ngữ nghĩa torchmetrics 0.10.3 (như Bảng IV CheXFound): {cls} Điểm trên cả test {imgs} ảnh; Ver2 = trung bình 5 hạt, Ver1 = một lượt '
            f'hạt 0. Không vẽ khoảng tin cậy (bootstrap trên 28 nhãn không ổn vì lớp hiếm); CI và luật §7.4 ở bảng mAP21 (C5–C6).')


def _note7(ctx):
    '''Câu ghi chú của hình 7 dấu hiệu RAD-DINO, dựng từ DỮ LIỆU (số ca dương test của từng dấu hiệu) + cách tính + mốc bài (chỉ đặt cạnh).'''
    names = list(ctx['dl']['nhan'])
    t = np.asarray(ctx['dl']['sets'][0].target)
    pos = ', '.join(f'{n} {int(t[:, names.index(n)].sum())}' for n in P.RADDINO7)
    paper = f'{P.RADDINO_PAPER_7:.3f}'.replace('.', ',')
    return (f'7 dấu hiệu của Bảng 1a bài RAD-DINO (ca dương ở test: {pos}). AUPRC = diện tích dưới đường precision–recall nối thẳng các điểm (hình '
            f'thang); AP = bậc thang (= "AUPRC" của CheXFound / torchmetrics); TB 7 = trung bình 7 dấu hiệu. Bài RAD-DINO báo AUPRC macro 7 dấu hiệu {paper} '
            f'— checkpoint của bài khác checkpoint công khai và bài không nêu cách tính AUPRC: chỉ đặt cạnh, không so. Ver2 = trung bình 5 hạt (từng dấu hiệu '
            f'cũng vậy), Ver1 = một lượt hạt 0. Không vẽ khoảng tin cậy.')


def _expected(e, c):
    '''Hàng e có phải chạy ở điều kiện c không (kho cắt giữa: chỉ 8 hàng của phép thử hình học).'''
    return e in CL.STORE_ROWS.get(c[1], (e,)) if c[0] == 'v2' else True


def _gaps(order, flat, conds):
    '''-> câu liệt kê (encoder, điều kiện) CHƯA có lượt nào (chấm / ô trống trên hình), hoặc ''. Hàng không thuộc điều kiện (CheXzero ở kho cắt giữa) ghi riêng.'''
    g = [f'{SPECS[e].title} · {FAM_SHORT[c]}' for e in order for c in conds if _expected(e, c) and not any((c[0], c[1], e, h) in flat for h in HEADS)]
    skip = [SPECS[e].title for e in order if GEO_FAM in conds and not _expected(e, GEO_FAM)]
    return ((f" Chưa có lượt (chấm / ô trống): {'; '.join(g)}." if g else '')
            + (f" {', '.join(skip)}: không chạy kho cắt giữa (tiền xử lý gốc = đệm vuông, PLAN_89 §5.3)." if skip else ''))


def figures(ctx, log=print, show=True):
    '''C7. Vẽ + giữ Figure (ghi PNG ở C8). Số vẽ = panel 28 nhãn — AP bậc thang, AUROC, AUPRC hình thang (điểm, không CI; Tuấn 15/11); bảng + luật vẫn ở
       mAP21. -> {tên: Figure}.'''
    o = ctx['o']
    flat = {k[:4]: v for k, v in o.items() if k[4] == 'luat'}
    conds = [f for f in FAMS if _encs(ctx, f)]
    dot_conds = [f for f in conds if f in FG.COND]           # chấm: tối đa 3 điều kiện (màu chấm phải phân biệt MỌI cặp) -> kho cắt giữa ở lưới + hình hiệu
    encs = [e for e in ORDER if any(k[2] == e for k in flat)]
    order = sorted(encs, key=lambda e: -max(v['diem'][FG.PANEL]['ap'] for k, v in flat.items() if k[2] == e))      # cùng thứ tự ở mọi hình
    et, ht = {e: SPECS[e].title for e in encs}, CL.TITLES
    base = _note28(ctx)
    note = base + _gaps(order, flat, conds)
    figs = {}
    for m in FG.METRICS:
        figs[f'cham_28_{FG.SLUG[m]}'] = FG.dots(flat, order, HEADS, dot_conds, m, et, ht, FAM_TITLE, base + _gaps(order, flat, dot_conds)
                                                + (' Kho cắt giữa: ở hình lưới và hình hiệu cắt giữa − đệm vuông.' if GEO_FAM in conds else ''))
    for m in FG.METRICS:
        figs[f'luoi_28_{FG.SLUG[m]}'] = FG.heat(flat, order, HEADS, conds, m, et, ht, FAM_TITLE, note)
    # 0.9.1 (Tuấn 16/11 "add plots for AUPRC in 7 findings like RAD-DINO"): 7 dấu hiệu Bảng 1a bài RAD-DINO — TB 7 (AUPRC hình thang + AP bậc thang, vì bài
    # không nêu cách tính AUPRC) và TỪNG dấu hiệu (AUPRC hình thang), mỗi điều kiện một hình
    base7 = _note7(ctx)
    for m in ('auprc_thang', 'ap'):
        figs[f'cham_7_{FG.SLUG[m]}'] = FG.dots(flat, order, HEADS, dot_conds, m, et, ht, FAM_TITLE, base7 + _gaps(order, flat, dot_conds)
                                               + (' Kho cắt giữa: ở hình lưới.' if GEO_FAM in conds else ''), panel='p7')
        figs[f'luoi_7_{FG.SLUG[m]}'] = FG.heat(flat, order, HEADS, conds, m, et, ht, FAM_TITLE, base7 + _gaps(order, flat, conds), panel='p7')
    for fam in conds:
        f = FG.findings(flat, order, HEADS, fam, P.RADDINO7, 'auprc_thang', et, ht, FAM_TITLE, base7 + _gaps(order, flat, [fam]))
        if f is not None:
            figs[f'auprc_7_dau_hieu_{fam[0]}_{fam[1]}'] = f
    if GEO_FAM in conds and PAD_FAM in conds:
        f = FG.cond_diff(flat, [e for e in order if e in GEO.ROWS], HEADS, GEO_FAM, PAD_FAM, FG.METRICS, et, ht, 'cắt giữa', 'đệm vuông',
                         base + ' Cả hai bên: Ver2, kho KHÔNG cân bằng histogram, mỗi bên trung bình 5 hạt của một lượt học riêng. Hiệu điểm MÔ TẢ — không CI; '
                         'luật hai họ hình học (khoảng đồng thời max-T) ở C6 trên mAP21. Ba thứ đổi cùng lúc: viền đen, phần bị cắt ở hai đầu cạnh dài, cỡ '
                         'giải phẫu trong khung (PLAN_89 §5.3).' + _gaps([e for e in order if e in GEO.ROWS], flat, [GEO_FAM, PAD_FAM]))
        if f is not None:
            figs['hieu_cat_giua_tru_dem_vuong_28'] = f
    if FAMS[0] in conds and FAMS[1] in conds:
        only = [et[e] for e in order if any(k[:2] == FAMS[1] and k[2] == e for k in flat) and not any(k[:2] == FAMS[0] and k[2] == e for k in flat)]
        vs_note = base + (f" {', '.join(only)}: chưa có {FAM_SHORT[FAMS[0]]} nên không có trong hình." if only else '') + \
            ' Chỉ đặt cạnh (khác mã, môi trường, lưới lr, số hạt, chế độ chấm) — không tính Ver2 − Ver1 (PLAN_89 §7.4).'
        for m in FG.METRICS:
            f = FG.versus(flat, order, HEADS, FAMS[0], FAMS[1], m, et, ht, FAM_TITLE, vs_note)
            if f is not None:
                figs[f'ver1_so_ver2_28_{FG.SLUG[m]}'] = f
    for fam in conds:
        f = FG.ref_diff(flat, REFERENCE, order, HEADS, fam, FG.METRICS, et, ht, FAM_TITLE,
                        base + ' Hiệu điểm MÔ TẢ (cùng điều kiện, cùng đầu đọc; mỗi bên TB 5 hạt; "·" = không cùng bộ nhãn test hoặc chưa có lượt) — không có '
                        'CI, không phải luật; luật §7.4 (họ B − X, khoảng đồng thời max-T) ở C6 trên mAP21.')
        if f is not None:
            figs[f'hieu_{REFERENCE}_tru_x_28_{fam[0]}_{fam[1]}'] = f
    for st in CL.V2_STORES:
        for metric, name in (('auc', 'AUROC'), ('ap', 'AP')):
            f = FG.lr_curves(ctx['dl']['duong_lr'], ctx['dl']['chon_v2'], order, HEADS, st, metric, et, ht,
                             f'Ver2 · kho {st}: {name} val 28 nhãn (lần chấm cuối pha 1, hạt 0) theo lr — chỗ chọn lr')
            if f is not None:
                figs[f'do_lr_{metric}_{st}'] = f
    ctx['hinh'] = figs
    log(f'{len(figs)} hinh (28 nhan: AP bac thang, AUROC, AUPRC hinh thang; diem, khong CI): {sorted(figs)}')
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
