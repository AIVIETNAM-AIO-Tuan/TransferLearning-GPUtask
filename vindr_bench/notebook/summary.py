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
    C7b crop_test  0.9.2 (Tuấn 17/11): phép thử cắt giữa RIÊNG — ba bảng (Δ ba số đo ở nhánh luật; tách phần chọn lr: luật / cùng lr / theo AP; B − X trên
                từng kho) + ba hình (hiệu 21 lớp, ba cách chọn lr, B − X trước -> sau); chỉ mô tả, luật vẫn ở C6; tệp riêng cat_giua_* ở C8
    C7c ablation   0.12.0 (PLAN_90 §10.2 + §10.2b, Việc B): 5 hàng ablation vị trí mô-đun vùng, RIÊNG (không vào bảng 9 encoder) — năm hàng; HỌ 16 PHÉP
                (3) − X với khoảng đồng thời max-T 95 % + phán quyết TỪNG Ô, đọc theo TỪNG ĐẦU (Tuấn 19/11: không gộp 4 đầu, không kết luận về vị trí);
                báo kèm §10.2; tách phần chọn lr (luật / cùng lr / theo AP) + chỗ chọn lr (mô tả, đặt sau khi thấy kết quả); tệp riêng ablation_* ở C8
    C8 sync     ghi JSON + CSV + PNG (+ tệp cat_giua_* của C7b, ablation_* của C7c), dong_bo()

Cách đọc (in kèm ở từng bảng): mAP21 = AP bậc thang macro 21 lớp (số của luật); AUPRC = hình thang; CI95 = bootstrap theo ảnh, DANH NGHĨA (chưa hiệu
chỉnh nhiều phép so) trừ họ max-T; Ver1 một hạt (sàng lọc); CI chứa 0 = chưa phát hiện khác biệt, không phải "bằng nhau"."""
import csv
import io
import json
import os
import time

import numpy as np

from .. import __version__
from ..encoders import ABLATION_REF, ABLATION_ROWS, ABLATION_STORE, ALL_ROWS, EXTRA_ROWS, ORDER, SPECS
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


def _d4(x):
    return '   -   ' if x is None or x != x else f'{x:+.4f}'


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
    dl = CL.collect(ctx['drive'], [e for e in ALL_ROWS], CL.V2_STORES, log=log, ablation=ABLATION_ROWS)   # 0.12.0: + 5 hàng Việc B (ô RIÊNG: C7c)
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
    return [e for e in ALL_ROWS if any((fam[0], fam[1], e, h, arm) in ctx['o'] for h in HEADS)]


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
        val = {f: {e: ctx['o'][(f[0], f[1], e, h, 'luat')]['diem']['p21']['ap'] for e in ALL_ROWS if (f[0], f[1], e, h, 'luat') in ctx['o']} for f in FAMS}
        rank = {f: {e: 1 + sorted(v.values(), reverse=True).index(x) for e, x in v.items()} for f, v in val.items()}
        for e in [e for e in ALL_ROWS if any(e in val[f] for f in FAMS)]:
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
    o, out = ctx['o'], dict(giua_encoder={}, ho_tham_chieu={}, ho_them={}, giua_dau={}, giua_kho=[], nhanh_ap=[])
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
            comps = [(h, x) for h in HEADS for x in encs if x != reference and x in ORDER and (h, reference, x) in table]   # 9 hàng đã khoá (13/11)
            q, sim = CP.max_t([table[(h, reference, x)] for h, x in comps])
            log(f'--- HO {len(comps)} phep so {SPECS[reference].title} - X (luat PLAN_89 7.4), khoang DONG THOI max-T 95 %: q = {q:.2f} ---')
            for (h, x), s in zip(comps, sim):
                r = table[(h, reference, x)]
                log(f"{'':<4}{CL.TITLES[h]:<11}{SPECS[reference].title} - {SPECS[x].title:<12} {_d(s['d'])} dong thoi [{_d(s['ci_dong_thoi'][0])}, {_d(s['ci_dong_thoi'][1])}] | danh nghia "
                    f"[{_d(r['ci'][0])}, {_d(r['ci'][1])}] | hai nua {_d(r['nua'][0])} / {_d(r['nua'][1])} -> {s['ket_luan']}")
            out['ho_tham_chieu'][FAM_SHORT[fam]] = dict(tham_chieu=reference, q=q, phep_so=[dict(dau=h, doi_chung=x, **s) for (h, x), s in zip(comps, sim)])
            extra = [(h, x) for h in HEADS for x in encs if x in EXTRA_ROWS and (h, reference, x) in table]
            if extra:                                # 0.10.0 (PLAN_90 §2): hàng thêm sau khi giao thức khoá — họ RIÊNG, max-T riêng, cùng dạng luật 7.4
                qe, sime = CP.max_t([table[(h, reference, x)] for h, x in extra])
                log(f"--- HO RIENG {len(extra)} phep so {SPECS[reference].title} - hang them {[SPECS[x].title for x in EXTRA_ROWS if any(x == y for _, y in extra)]} "
                    f"(dat 17/11 truoc moi ket qua cua hang them; KHONG thuoc ho {len(comps)} phep tren; so HAI HE THONG), khoang DONG THOI max-T 95 %: q = {qe:.2f} ---")
                for (h, x), s in zip(extra, sime):
                    r = table[(h, reference, x)]
                    log(f"{'':<4}{CL.TITLES[h]:<11}{SPECS[reference].title} - {SPECS[x].title:<16} {_d(s['d'])} dong thoi [{_d(s['ci_dong_thoi'][0])}, {_d(s['ci_dong_thoi'][1])}] | "
                        f"danh nghia [{_d(r['ci'][0])}, {_d(r['ci'][1])}] | hai nua {_d(r['nua'][0])} / {_d(r['nua'][1])} -> {s['ket_luan']}")
                out['ho_them'][FAM_SHORT[fam]] = dict(tham_chieu=reference, q=qe, phep_so=[dict(dau=h, doi_chung=x, **s) for (h, x), s in zip(extra, sime)])
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
    for e in ALL_ROWS:
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
        for e in ALL_ROWS:
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
    encs = [e for e in ALL_ROWS if any(k[2] == e for k in flat)]
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


CROP_MODES = ('luat', 'cung_lr', 'ap')
CROP_MODE_TITLE = {'luat': 'lr của luật (AUROC val), chọn riêng ở mỗi kho', 'cung_lr': 'cùng một lr ở hai kho', 'ap': 'lr theo AP val, chọn riêng ở mỗi kho'}
CROP_MODE_TEXT = {'luat': 'luat', 'cung_lr': 'cung lr', 'ap': 'theo AP'}
M3 = ('ap', 'auprc_thang', 'auroc')


def _lr_key(lr):
    return float(f'{float(lr):.6g}')


def _arm_cells(o, fam, e, h):
    '''-> {lr: (nhánh, ô)} của các đầu ĐÃ HỌC XONG pha 2 ở (kho, encoder, đầu): nhánh luật, và nhánh "theo AP" khi lr của nó khác lr của luật.'''
    out = {}
    for arm in ('luat', 'ap'):
        c = o.get((fam[0], fam[1], e, h, arm))
        if c is not None:
            out.setdefault(_lr_key(c['lr']), (arm, c))
    return out


def _ap_cell(ctx, fam, e, h):
    '''Ô ở lr có AP val cao nhất: nhánh "theo AP" nếu có; lượt ghi rõ lr theo AP TRÙNG lr của luật -> ô nhánh luật; không rõ -> None (không đoán).'''
    c = ctx['o'].get((fam[0], fam[1], e, h, 'ap'))
    if c is not None:
        return 'ap', c
    ch = (ctx['dl']['chon_v2'].get((e, fam[1])) or {}).get(h)
    c = ctx['o'].get((fam[0], fam[1], e, h, 'luat'))
    return ('luat', c) if c is not None and ch and ch.get('ap_trung_luat') else None


def _d3(plus, minus, panel='p21'):
    '''Hiệu điểm ba số đo của một panel (AP bậc thang, AUPRC hình thang, AUROC): tổng các ô `plus` − tổng các ô `minus`.'''
    return {m: float(sum(c['diem'][panel][m] for c in plus) - sum(c['diem'][panel][m] for c in minus)) for m in M3}


def _t3(d):
    return f"{_d(d['ap'])} / {_d(d['auprc_thang'])} / {_d(d['auroc'])}"


def _cg_row(bang, e, h, cach, chieu, lr_pad, lr_crop, d3, r=None, ghi_chu=''):
    '''Một hàng của bảng dài phép thử cắt giữa (CSV): một hiệu + ba số đo + CI danh nghĩa của AP21 + điều kiện bằng chữ.'''
    ci, nua = (r['ci'], r['nua']) if r else ([None, None], [None, None])
    return dict(bang=bang, encoder=e, hang=_row(e), dau=h, cach_chon_lr=cach, chieu_phep_tru=chieu, lr_dem_vuong=lr_pad, lr_cat_giua=lr_crop, d_AP21=d3['ap'],
                ci_thap=ci[0], ci_cao=ci[1], d_AUPRC21_hinh_thang=d3['auprc_thang'], d_AUROC21=d3['auroc'], d_AP21_nua_0=nua[0], d_AP21_nua_1=nua[1], ghi_chu=ghi_chu)


def crop_test(ctx, log=print, show=True, reference=REFERENCE):
    '''C7b (0.9.2, Tuấn 17/11: bảng + hình của phép thử cắt giữa để RIÊNG, không trộn vào C5 / C6 / C7). Ver2, kho cắt giữa so với kho đệm vuông (cả hai
       không cân bằng), nhánh nào cũng TB các hạt; ba bảng + ba hình, chỉ MÔ TẢ (luật + khoảng đồng thời max-T vẫn ở cuối C6):
         Bảng 1  Δ = cắt giữa − đệm vuông ở nhánh luật: AP bậc thang / AUPRC hình thang / AUROC 21 lớp + CI95 danh nghĩa của Δ AP21 + Δ từng hạt
         Bảng 2  tách phần chọn lr (đặt SAU khi thấy kết quả 17/11): Δ theo luật, Δ ở CÙNG lr (lr có đầu học xong ở cả hai kho), Δ khi mỗi kho chọn lr theo AP val
         Bảng 3  B − X trên từng kho và thay đổi của nó (= Δ_B − Δ_X), nhánh luật + nhánh theo AP
       -> dict (ctx['cat_giua']); hình ở ctx['hinh_cat_giua'] (ghi tệp RIÊNG ở C8: cat_giua_*).'''
    o = ctx['o']
    cell = lambda fam, e, h, arm='luat': o.get((fam[0], fam[1], e, h, arm))
    rows = [e for e in ORDER if e in GEO.ROWS and any(_pair_ok(cell(GEO_FAM, e, h), cell(PAD_FAM, e, h)) for h in HEADS)]
    gaps = [SPECS[e].title for e in ORDER if e in GEO.ROWS and e not in rows]
    n_img = sorted({len(s.target) for s in ctx['dl']['sets'] if s.ver == 'v2' and s.store in (GEO.CROP_STORE, GEO.PAD_STORE)})
    cond = ('Ver2 = ma + moi truong cua ta (G4) | kho CAT GIUA (co canh ngan + cat giua) so voi kho DEM VUONG, ca hai KHONG can bang histogram | backbone dong '
            'cung, chuan hoa goc cua tung encoder | dau doc hoc lai tu dau o moi kho | moi ben TB cac hat | test '
            + '/'.join(f'{v:,}'.replace(',', '.') for v in n_img) + ' anh, 21 lop so chinh')
    out = dict(dieu_kien=cond, hang=rows, chua_du_cap=gaps, delta=[], tach_lr=[], b_tru_x=[], bang=[])
    ctx['cat_giua'], ctx['hinh_cat_giua'] = out, {}
    log(f'=== PHEP THU CAT GIUA — bang + hinh RIENG (PLAN_89 5.3) | {cond} ===')
    if gaps:
        log(f"    chua du cap (thieu luot o mot kho, hoac hai kho khac thu tu anh): {', '.join(gaps)}")
    if not rows:
        log('    CHUA CO hang nao du ca hai kho -> khong co bang / hinh')
        return out
    log('cach doc: Δ = kho CAT GIUA − kho DEM VUONG, cung encoder, cung dau. Ba so moi o: AP bac thang (so cua luat) / AUPRC hinh thang / AUROC, deu macro 21 lop. '
        '[..] = CI95 DANH NGHIA cua Δ AP21 (bootstrap ghep cap theo anh; co dieu kien tren dau da hoc + lr da chon — KHONG chua bien thien chon lr / huan luyen). '
        'CI chua 0 = CHUA phat hien khac. Luat (khoang dong thoi max-T): cuoi C6. Ba thu doi cung luc giua hai kho (vien den, phan bi cat, co giai phau) -> '
        'khong quy rieng cho vien den.')
    verdicts = {(p['dang'], p['dau'], p['encoder']): p for f in ((ctx.get('so_sanh') or {}).get('hinh_hoc') or {}).values() for p in f['phep_so']}

    log('\n--- Bang 1 · Δ nhanh LUAT (moi kho: lr co AUROC val cao nhat) ---')
    log(f"{'hang (anh, vector)':<30}{'dau':<11}{'lr dv -> cg':>16}  {'AP21 dv':>8}{'AP21 cg':>8}  {'Δ AP / AUPRC / AUROC':<26}{'CI95 Δ AP21':<19}{'hai nua':<17}"
        f"{'Δ tung hat':<38}luat (a) o C6")
    for e in rows:
        for h in HEADS:
            a1, a0 = cell(GEO_FAM, e, h), cell(PAD_FAM, e, h)
            if not _pair_ok(a1, a0):
                continue
            r, d3, seeds = CP.diff(a1, a0), _d3([a1], [a0]), _per_seed((1, a1), (-1, a0))
            v = verdicts.get(('a', h, e))
            lab = '(chay C6 truoc)' if v is None else v['ket_luan'] + (f" [{_d(v['ci_dong_thoi'][0])}, {_d(v['ci_dong_thoi'][1])}]" if v.get('ci_dong_thoi') else '')
            log(f"{_row(e):<30}{CL.TITLES[h]:<11}{a0['lr']:>7g} -> {a1['lr']:<6g}  {_f(a0['diem']['p21']['ap']):>8}{_f(a1['diem']['p21']['ap']):>8}  {_t3(d3):<26}"
                f"[{_d(r['ci'][0])}, {_d(r['ci'][1])}]  {_d(r['nua'][0])} / {_d(r['nua'][1])}  {' '.join(_d(s) for s in seeds):<38}{lab}")
            out['delta'].append(dict(encoder=e, dau=h, lr_dem_vuong=a0['lr'], lr_cat_giua=a1['lr'], AP21_dem_vuong=a0['diem']['p21']['ap'],
                                     AP21_cat_giua=a1['diem']['p21']['ap'], d=d3, ci_danh_nghia=r['ci'], nua=r['nua'], tung_hat=seeds,
                                     d_p7=_d3([a1], [a0], 'p7'), d_p28=_d3([a1], [a0], 'p28'), luat_a=None if v is None else v['ket_luan'],
                                     ci_dong_thoi=None if v is None else v.get('ci_dong_thoi')))
            out['bang'].append(_cg_row('1_delta_nhanh_luat', e, h, 'luat', 'cat giua - dem vuong', a0['lr'], a1['lr'], d3, r, ghi_chu=lab))

    log("\n--- Bang 2 · TACH PHAN CHON LR (MO TA; cach 'cung lr' va 'theo AP' dat SAU khi thay ket qua 17/11 — khong thuoc luat dat truoc) ---")
    log("    luat    = moi kho lay lr co AUROC val cao nhat (Bang 1) | cung lr = lr co dau hoc xong pha 2 o CA HAI kho (nhanh luat hoac nhanh 'theo AP' cua tung "
        "kho; khong co -> khong so duoc) | theo AP = moi kho lay lr co AP val cao nhat")
    log(f"{'hang':<16}{'dau':<11}{'cach':<9}{'lr dv -> cg':>16}  {'Δ AP / AUPRC / AUROC':<26}CI95 Δ AP21 (danh nghia)")
    vals = {}
    for e in rows:
        for h in HEADS:
            a1, a0 = cell(GEO_FAM, e, h), cell(PAD_FAM, e, h)
            if not _pair_ok(a1, a0):
                continue
            items = [('luat', a0, a1, '')]
            A1, A0 = _arm_cells(o, GEO_FAM, e, h), _arm_cells(o, PAD_FAM, e, h)
            same = [lr for lr in sorted(set(A1) & set(A0)) if _pair_ok(A1[lr][1], A0[lr][1])]
            items += [('cung_lr', A0[lr][1], A1[lr][1], f'nhanh dv {A0[lr][0]}, cg {A1[lr][0]}') for lr in same]
            p1, p0 = _ap_cell(ctx, GEO_FAM, e, h), _ap_cell(ctx, PAD_FAM, e, h)
            if p1 and p0 and _pair_ok(p1[1], p0[1]):
                items.append(('ap', p0[1], p1[1], f'nhanh dv {p0[0]}, cg {p1[0]}'))
            for k, (mode, c0, c1, why) in enumerate(items):
                r, d3 = CP.diff(c1, c0), _d3([c1], [c0])
                vals.setdefault((e, h, mode), []).append(d3['ap'])
                log(f"{(SPECS[e].title if k == 0 else ''):<16}{(CL.TITLES[h] if k == 0 else ''):<11}{CROP_MODE_TEXT[mode]:<9}{c0['lr']:>7g} -> {c1['lr']:<6g}  {_t3(d3):<26}"
                    f"[{_d(r['ci'][0])}, {_d(r['ci'][1])}]" + (f'  ({why})' if why else ''))
                out['tach_lr'].append(dict(encoder=e, dau=h, cach=mode, lr_dem_vuong=c0['lr'], lr_cat_giua=c1['lr'], d=d3, ci_danh_nghia=r['ci'], nua=r['nua'], nhanh=why))
                out['bang'].append(_cg_row('2_tach_chon_lr', e, h, mode, 'cat giua - dem vuong', c0['lr'], c1['lr'], d3, r, ghi_chu=why))
            if not same:
                log(f"{'':<27}{'cung lr':<9}{'-':>16}  khong co lr nao hoc xong o ca hai kho")

    ref_ok = reference in rows
    lead = {}
    if ref_ok:
        t = SPECS[reference].title
        log(f"\n--- Bang 3 · {t} − X tren TUNG kho va thay doi cua no (= Δ_{t} − Δ_X; am = khoang cach cua {t} co lai khi bo dem vuong) ---")
        log(f"    moi o ba so: AP / AUPRC / AUROC (21 lop); dv = kho dem vuong, cg = kho cat giua; thay doi = ({t} − X cg) − ({t} − X dv)")
        log(f"{'X':<14}{'dau':<11}{'cach':<9}{t + ' − X dv':<26}{t + ' − X cg':<26}{'hai nua AP21 cg':<17}{'thay doi':<26}CI95 thay doi AP21 (danh nghia)")
        for x in rows:
            if x == reference:
                continue
            for h in HEADS:
                quads = [('luat', cell(GEO_FAM, reference, h), cell(PAD_FAM, reference, h), cell(GEO_FAM, x, h), cell(PAD_FAM, x, h))]
                ap = [_ap_cell(ctx, f, e, h) for f, e in ((GEO_FAM, reference), (PAD_FAM, reference), (GEO_FAM, x), (PAD_FAM, x))]
                if all(ap):
                    quads.append(('ap',) + tuple(c for _, c in ap))
                for k, (mode, b1, b0, x1, x0) in enumerate(quads):
                    if any(c is None for c in (b1, b0, x1, x0)) or not all(_pair_ok(b1, c) for c in (b0, x1, x0)):
                        continue
                    r0, r1, rr = CP.diff(b0, x0), CP.diff(b1, x1), CP.diff_of_diffs(b1, b0, x1, x0)
                    d0, d1, dd = _d3([b0], [x0]), _d3([b1], [x1]), _d3([b1, x0], [b0, x1])
                    if mode == 'luat':
                        lead[(x, h)] = (d0['ap'], d1['ap'])
                    log(f"{(SPECS[x].title if k == 0 else ''):<14}{(CL.TITLES[h] if k == 0 else ''):<11}{CROP_MODE_TEXT[mode]:<9}{_t3(d0):<26}{_t3(d1):<26}"
                        f"{_d(r1['nua'][0])} / {_d(r1['nua'][1])}  {_t3(dd):<26}[{_d(rr['ci'][0])}, {_d(rr['ci'][1])}]")
                    out['b_tru_x'].append(dict(doi_chung=x, dau=h, cach=mode, dem_vuong=d0, ci_dem_vuong=r0['ci'], cat_giua=d1, ci_cat_giua=r1['ci'],
                                               nua_cat_giua=r1['nua'], thay_doi=dd, ci_thay_doi=rr['ci']))
                    for chieu, d3, r in ((f'{t} - X tren kho dem vuong', d0, r0), (f'{t} - X tren kho cat giua', d1, r1),
                                         (f'({t} - X cat giua) - ({t} - X dem vuong)', dd, rr)):
                        out['bang'].append(_cg_row('3_b_tru_x', x, h, mode, chieu, b0['lr'], b1['lr'], d3, r, ghi_chu=f'lr cua X: dv {x0["lr"]:g}, cg {x1["lr"]:g}'))
    else:
        log(f'\n--- Bang 3: chua co hang {reference.upper()} du ca hai kho -> khong co ---')

    et, ht = {e: SPECS[e].title for e in ALL_ROWS}, CL.TITLES
    flat = {k[:4]: v for k, v in o.items() if k[4] == 'luat'}
    base = (f"Ver2, kho không cân bằng histogram ở cả hai bên; backbone đông cứng, đầu đọc học lại từ đầu ở mỗi kho; mỗi bên trung bình các hạt; test "
            f"{'/'.join(f'{v:,}'.replace(',', '.') for v in n_img)} ảnh, 21 lớp chính. Điểm MÔ TẢ, không khoảng tin cậy ở hình — CI danh nghĩa ở bảng cùng ô, luật "
            'và khoảng đồng thời max-T ở C6. Ba thứ đổi cùng lúc giữa hai kho: viền đen, phần bị cắt ở hai đầu cạnh dài, cỡ giải phẫu trong khung (PLAN_89 §5.3).'
            + (f" Chưa đủ cặp: {', '.join(gaps)}." if gaps else ''))
    figs = {}
    f = FG.cond_diff(flat, rows, HEADS, GEO_FAM, PAD_FAM, FG.METRICS, et, ht, 'cắt giữa', 'đệm vuông', base + ' Nhánh luật: mỗi kho lấy lr có AUROC val cao nhất.',
                     panel='p21')
    if f is not None:
        figs['hieu_21'] = f
    f = FG.delta_modes(vals, rows, HEADS, CROP_MODES, et, ht, CROP_MODE_TITLE,
                       'Δ AP 21 lớp (bậc thang) cắt giữa − đệm vuông theo ba cách chọn lr — mỗi ô một đầu đọc, mỗi hàng một encoder', 'Δ AP 21',
                       base + ' Cùng lr: chỉ các lr có đầu học xong ở CẢ hai kho (không có -> không có chấm; hai lr chung -> hai chấm). Cách "cùng lr" và "theo AP" '
                       'đặt sau khi thấy kết quả (17/11) — không thuộc luật đặt trước. Đường đứt: ±0,01 (ngưỡng của luật).', margin=GEO.MARGIN)
    if f is not None:
        figs['tach_chon_lr'] = f
    if lead:
        t = SPECS[reference].title
        others = [x for x in rows if x != reference and any((x, h) in lead for h in HEADS)]
        f = FG.lead_shift({k: v[0] for k, v in lead.items()}, {k: v[1] for k, v in lead.items()}, others, HEADS, et, ht, ('kho đệm vuông', 'kho cắt giữa'),
                          f'{t} − X trên AP 21 lớp (bậc thang): kho đệm vuông -> kho cắt giữa — mỗi ô một đầu đọc, mỗi hàng một X', f'{t} − X · AP 21',
                          base + f' Nhánh luật. Chấm đặc nằm bên trái chấm rỗng = khoảng cách của {t} so với X co lại khi bỏ đệm vuông.')
        if f is not None:
            figs['b_tru_x'] = f
    ctx['hinh_cat_giua'] = figs
    log(f"\n{len(out['delta'])} o Δ, {len(out['tach_lr'])} dong tach lr, {len(out['b_tru_x'])} dong {SPECS[reference].title} − X | {len(figs)} hinh rieng: {sorted(figs)} "
        '(ghi tep cat_giua_* o C8)')
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return out


# ---- 0.12.0 · C7c: Việc B của PLAN_90 (§10.2 đặt trước + §10.2b cách đọc, Tuấn 19/11) — 5 hàng ablation vị trí mô-đun vùng, bảng + hình + tệp RIÊNG ----
ABL_FAM = ('v2', ABLATION_STORE)
ABL_X = ('vt_khong_luong', 'vt_sau4', 'vt_sau1', 'vt_mot_chieu')        # X của họ 16 phép (3) − X, thứ tự PLAN_90 §10.2: (0), (1a), (1b), (2)
ABL_PAIRS = (('vt_sau4', 'vt_khong_luong'), ('vt_sau1', 'vt_khong_luong'), ('vt_mot_chieu', 'vt_khong_luong'), ('vt_sau4', 'vt_sau1'),
             ('vt_sau4', 'vt_mot_chieu'))                                # báo kèm §10.2 (mô tả): (1a) − (0), (1b) − (0), (2) − (0), (1a) − (1b), (1a) − (2)
ABL_FAMILY_SIZE = len(ABL_X) * len(HEADS)                                # 16
assert set(ABL_X) | {ABLATION_REF} == set(ABLATION_ROWS) and ABLATION_REF not in ABL_X and ABL_FAMILY_SIZE == 16
assert all(a in ABLATION_ROWS and b in ABLATION_ROWS for a, b in ABL_PAIRS)
ABL_VERDICTS = ('hon', 'kem', 'khong phan biet')
ABL_MODE_TITLE = {'luat': 'lr của luật (AUROC val), chọn riêng ở mỗi hàng', 'cung_lr': 'cùng một lr ở hai hàng', 'ap': 'lr theo AP val, chọn riêng ở mỗi hàng'}
ABL_READING = ('cach doc (PLAN_90 10.2 dat truoc + 10.2b Tuan 19/11): 16 phan quyet TUNG O — "hon" = hieu >= +0,01 VA can duoi dong thoi > 0 VA duong o hai nua '
               'test; "kem" doi xung; con lai "khong phan biet". Bao THEO TUNG DAU DOC: KHONG gop 4 dau thanh cau chung, KHONG ket luan ve vi tri mo-dun vung '
               '(trong / sau encoder). Mot seed backbone (s42) = sang loc: ket luan duong can xac nhan seed 43 truoc khi thanh khang dinh trong bai. CI theo anh, '
               'co dieu kien tren dau da hoc + lr da chon (KHONG chua bien thien chon lr / huan luyen); CI chua 0 = CHUA phat hien khac.')


def _code(e):
    '''Mã của hàng ablation = phần đầu tên in: '(3)', '(2)', '(1a)', '(1b)', '(0)'.'''
    return SPECS[e].title.split(' ', 1)[0]


assert all(_code(e).startswith('(') and _code(e).endswith(')') for e in ABLATION_ROWS) and len({_code(e) for e in ABLATION_ROWS}) == len(ABLATION_ROWS)


def _abl_text(e):
    '''Điều kiện BẰNG CHỮ của một hàng ablation (tên in + mô-đun vùng lúc tinh chỉnh CXR-LT).'''
    return f'{SPECS[e].title} — {SPECS[e].arch}'


def _abl_row(bang, a, b, h, cach, lr_a, lr_b, d3, r=None, ci_dong_thoi=None, ket_luan='', ghi_chu=''):
    '''Một hàng của bảng dài hiệu của Việc B (CSV): hiệu a − b ở một đầu đọc + ba số đo + CI + điều kiện bằng chữ của HAI hàng + chiều phép trừ.'''
    ci, nua = (r['ci'], r['nua']) if r else ([None, None], [None, None])
    cs = ci_dong_thoi or [None, None]
    return dict(bang=bang, hang_a=a, dieu_kien_a=_abl_text(a), hang_b=b, dieu_kien_b=_abl_text(b), dau=h, cach_chon_lr=cach, chieu_phep_tru=f'{_code(a)} - {_code(b)}',
                lr_a=lr_a, lr_b=lr_b, d_AP21=d3['ap'], ci_dong_thoi_thap=cs[0], ci_dong_thoi_cao=cs[1], ci_danh_nghia_thap=ci[0], ci_danh_nghia_cao=ci[1],
                d_AUPRC21_hinh_thang=d3['auprc_thang'], d_AUROC21=d3['auroc'], d_AP21_nua_0=nua[0], d_AP21_nua_1=nua[1], ket_luan=ket_luan, ghi_chu=ghi_chu)


def _names(heads):
    return '[' + ', '.join(CL.TITLES[h] for h in heads) + ']' if heads else '—'


def ablation(ctx, log=print, show=True):
    '''C7c (0.12.0; PLAN_90 §10.2 + §10.2b). Năm hàng ablation của Việc B trên VinDr (Ver2, kho đệm vuông không cân bằng; backbone RAD-DINO tinh chỉnh trên
       CXR-LT, năm hàng chỉ khác mô-đun vùng lúc tinh chỉnh), để RIÊNG — không vào bảng / họ phép so của 9 encoder:
         Bảng 1  năm hàng: AP bậc thang / AUPRC hình thang / AUROC 21 lớp, lr, CI95 danh nghĩa, SD giữa hạt, hai nửa (nhánh luật + nhánh theo AP khi khác)
         Bảng 2  HỌ 16 PHÉP (3) − X (LUẬT đặt trước): khoảng đồng thời max-T 95 % khi đủ 16 phép + phán quyết TỪNG Ô; đọc theo TỪNG ĐẦU (§10.2b: không
                 gộp 4 đầu thành câu chung, không kết luận về vị trí); họ chưa đủ -> chỉ hiệu danh nghĩa, CHƯA áp luật
         Bảng 3  báo kèm §10.2 (mô tả, CI danh nghĩa): (1a) − (0), (1b) − (0), (2) − (0), (1a) − (1b), (1a) − (2)
         Bảng 4  tách phần chọn lr của (3) − X: luật / cùng lr / theo AP (mô tả; đặt SAU khi thấy kết quả 19/11)
         Bảng 5  chỗ chọn lr của từng hàng × đầu: lr của luật so với lr đứng thứ hai theo AUROC val và với lr theo AP val (mô tả)
       -> dict (ctx['ablation']); hình ở ctx['hinh_ablation'] (ghi tệp RIÊNG ở C8: ablation_*).'''
    o, fam, ref = ctx['o'], ABL_FAM, ABLATION_REF
    cell = lambda e, h, arm='luat': o.get((fam[0], fam[1], e, h, arm))
    rows = [e for e in ABLATION_ROWS if any(cell(e, h) is not None for h in HEADS)]
    gaps = [SPECS[e].title for e in ABLATION_ROWS if e not in rows]
    n_img = sorted({len(s.target) for s in ctx['dl']['sets'] if s.encoder in ABLATION_ROWS})
    cond = ('Ver2 = ma + moi truong cua ta (G4) | kho DEM VUONG, KHONG can bang histogram | luoi 13 lr | TB 5 hat dau doc | backbone DONG CUNG = RAD-DINO tinh '
            'chinh tren CXR-LT (mot seed backbone s42, trong so EMA epoch 4); nam hang chi khac MO-DUN VUNG luc tinh chinh | test '
            + '/'.join(f'{v:,}'.replace(',', '.') for v in n_img) + ' anh, 21 lop so chinh')
    out = dict(dieu_kien=cond, cach_doc=ABL_READING, hang=rows, hang_bang_chu={e: _abl_text(e) for e in rows}, chua_co=gaps, nam_hang=[], ho_16=None, bao_kem=[],
               tach_lr=[], chon_lr=[], bang_hieu=[])
    ctx['ablation'], ctx['hinh_ablation'] = out, {}
    log(f'=== VIEC B (PLAN_90 10.2) — 5 hang ablation vi tri mo-dun vung: bang + hinh + tep RIENG, KHONG vao bang 9 encoder | {cond} ===')
    if gaps:
        log(f"    chua co luot: {', '.join(gaps)}")
    if not rows:
        log('    CHUA CO hang nao -> khong co bang / hinh')
        return out
    log(ABL_READING)
    log('    moi o ba so: AP bac thang (so cua luat) / AUPRC hinh thang / AUROC, deu macro 21 lop; [..] = CI95 bootstrap ghep cap theo anh (cung lan rut moi hang)')

    log("\n--- Bang 1 · NAM HANG (nhanh 'luat' = lr co AUROC val cao nhat; nhanh 'ap' = lr co AP val cao nhat, chi co khi khac lr cua luat; * = lr o bien luoi) ---")
    for e in rows:
        log(f'  {_abl_text(e)}')
        log(f"    {'dau':<11}{'nhanh':<6}{'lr':>8}  {'mAP21 [CI95 danh nghia]':<26}{'SD hat':>8}{'ens':>7}{'AUPRC21':>9}{'AUROC21':>9}  {'hai nua AP21':<15}AUROC val luc chon")
        for h in HEADS:
            for arm in ('luat', 'ap'):
                c = cell(e, h, arm)
                if c is None:
                    continue
                d, sd, en = c['diem']['p21'], c['sd']['p21']['ap'], (c['ensemble'] or {}).get('p21', {}).get('ap')
                log(f"    {CL.TITLES[h]:<11}{arm:<6}{c['lr']:>8g}{'*' if c['bien'] else ' '} {_f(d['ap'], 4)} [{_f(c['ci21'][0])}, {_f(c['ci21'][1])}]"
                    f"{'':<4}{_f(sd, 4):>8}{_f(en):>7}{_f(d['auprc_thang'], 4):>9}{_f(d['auroc'], 4):>9}  {_f(c['nua'][0])} / {_f(c['nua'][1])}  {_f(c['val_auroc'], 4)}")
                out['nam_hang'].append(dict(hang=e, ma=_code(e), dieu_kien=_abl_text(e), dau=h, nhanh=arm, lr=c['lr'], lr_o_bien=c['bien'], so_hat=c['so_hat'],
                                            mAP21=d['ap'], ci_thap=c['ci21'][0], ci_cao=c['ci21'][1], sd_giua_hat=sd, ensemble_mAP21=en,
                                            AUPRC21_hinh_thang=d['auprc_thang'], AUROC21=d['auroc'], AP21_nua_0=c['nua'][0], AP21_nua_1=c['nua'][1],
                                            AUROC_val_luc_chon=c['val_auroc'], lan_rut_bi_loai=c['loai21'], nguon=c['nguon']))

    items = []
    for x in ABL_X:
        for h in HEADS:
            a, b = cell(ref, h), cell(x, h)
            if _pair_ok(a, b):
                items.append((x, h, CP.diff(a, b), _d3([a], [b]), _per_seed((1, a), (-1, b)), a, b))
    need = ABL_FAMILY_SIZE
    full = len(items) == need
    q, sim = CP.max_t([it[2] for it in items]) if full else (float('nan'), [None] * len(items))
    log(f"\n--- Bang 2 · HO {need} PHEP SO {_code(ref)} − X (LUAT dat truoc, PLAN_90 10.2) | {len(items)} / {need} phep co du lieu"
        + (f' -> khoang DONG THOI max-T 95 % cua ho, q = {q:.2f}' if full else ' -> HO CHUA DU: chi hieu DANH NGHIA, CHUA ap luat') + ' ---')
    log(f"    {'X':<8}{'dau':<11}{'lr (3) | X':>16}  {'Δ AP / AUPRC / AUROC':<26}{'dong thoi Δ AP21':<19}{'danh nghia':<19}{'hai nua':<17}{'Δ tung hat':<38}ket luan")
    rec, by = [], {}
    for (x, h, r, d3, seeds, a, b), s in zip(items, sim):
        ci_s = s['ci_dong_thoi'] if s else None
        lab = s['ket_luan'] if s else f"danh nghia: {r['ket_luan']}"
        by[(x, h)] = lab
        log(f"    {_code(x):<8}{CL.TITLES[h]:<11}{a['lr']:>7g} | {b['lr']:<6g}  {_t3(d3):<26}"
            + (f"[{_d(ci_s[0])}, {_d(ci_s[1])}]  " if ci_s else f"{'-':<19}") + f"[{_d(r['ci'][0])}, {_d(r['ci'][1])}]  {_d(r['nua'][0])} / {_d(r['nua'][1])}  "
            f"{' '.join(_d(v) for v in seeds):<38}{lab}")
        rec.append(dict(doi_chung=x, ma=_code(x), dau=h, lr_tham_chieu=a['lr'], lr_doi_chung=b['lr'], d=d3, ci_dong_thoi=ci_s, ci_danh_nghia=r['ci'], nua=r['nua'],
                        tung_hat=seeds, ket_luan=lab))
        out['bang_hieu'].append(_abl_row('2_ho_16_luat', ref, x, h, 'luat', a['lr'], b['lr'], d3, r, ci_s, lab))
    by_x, by_head = {}, {}
    if full:
        log(f'    --- doc THEO TUNG DAU (10.2b; KHONG gop thanh cau chung, KHONG ket luan ve vi tri mo-dun vung) ---')
        for x in ABL_X:
            by_x[x] = {v: [h for h in HEADS if by[(x, h)] == v] for v in ABL_VERDICTS}
            log(f"    {_code(ref)} so voi {SPECS[x].title:<26}: HON o {_names(by_x[x]['hon'])} | KEM o {_names(by_x[x]['kem'])} | khong phan biet o "
                f"{_names(by_x[x]['khong phan biet'])}")
        for h in HEADS:
            by_head[h] = {v: [x for x in ABL_X if by[(x, h)] == v] for v in ABL_VERDICTS}
            log(f"    {CL.TITLES[h]:<11}: {_code(ref)} HON {[_code(x) for x in by_head[h]['hon']] or '—'} | KEM {[_code(x) for x in by_head[h]['kem']] or '—'} | khong phan "
                f"biet {[_code(x) for x in by_head[h]['khong phan biet']] or '—'}")
    out['ho_16'] = dict(tham_chieu=ref, so_phep=need, co=len(items), day_du=full, q=q if full else None, phep_so=rec, theo_doi_chung=by_x, theo_dau=by_head)

    log('\n--- Bang 3 · BAO KEM (MO TA, PLAN_90 10.2; CI95 DANH NGHIA — khong thuoc ho 16 phep, khong hieu chinh nhieu phep so) ---')
    log(f"    {'hieu':<14}{'dau':<11}{'lr a | b':>16}  {'Δ AP / AUPRC / AUROC':<26}{'CI95 Δ AP21':<19}hai nua")
    for a_, b_ in ABL_PAIRS:
        for h in HEADS:
            a, b = cell(a_, h), cell(b_, h)
            if not _pair_ok(a, b):
                continue
            r, d3 = CP.diff(a, b), _d3([a], [b])
            log(f"    {_code(a_) + ' − ' + _code(b_):<14}{CL.TITLES[h]:<11}{a['lr']:>7g} | {b['lr']:<6g}  {_t3(d3):<26}[{_d(r['ci'][0])}, {_d(r['ci'][1])}]  "
                f"{_d(r['nua'][0])} / {_d(r['nua'][1])}")
            out['bao_kem'].append(dict(hang_a=a_, hang_b=b_, dau=h, lr_a=a['lr'], lr_b=b['lr'], d=d3, ci_danh_nghia=r['ci'], nua=r['nua']))
            out['bang_hieu'].append(_abl_row('3_bao_kem', a_, b_, h, 'luat', a['lr'], b['lr'], d3, r, ghi_chu='mo ta; CI danh nghia'))

    log(f"\n--- Bang 4 · TACH PHAN CHON LR cua {_code(ref)} − X (MO TA; cach 'cung lr' va 'theo AP' dat SAU khi thay ket qua 19/11 — khong thuoc luat dat truoc; "
        'nhanh luat van la so chinh) ---')
    log("    luat    = moi hang lay lr co AUROC val cao nhat (Bang 2) | cung lr = lr co dau hoc xong pha 2 o CA HAI hang (nhanh luat hoac nhanh 'theo AP' cua tung "
        "hang; khong co -> khong so duoc) | theo AP = moi hang lay lr co AP val cao nhat")
    log(f"    {'X':<8}{'dau':<11}{'cach':<9}{'lr (3) | X':>16}  {'Δ AP / AUPRC / AUROC':<26}CI95 Δ AP21 (danh nghia)")
    vals = {}
    for x in ABL_X:
        for h in HEADS:
            a, b = cell(ref, h), cell(x, h)
            if not _pair_ok(a, b):
                continue
            its = [('luat', a, b, '')]
            A, B = _arm_cells(o, fam, ref, h), _arm_cells(o, fam, x, h)
            same = [lr for lr in sorted(set(A) & set(B)) if _pair_ok(A[lr][1], B[lr][1])]
            its += [('cung_lr', A[lr][1], B[lr][1], f'nhanh {_code(ref)} {A[lr][0]}, {_code(x)} {B[lr][0]}') for lr in same]
            pa, pb = _ap_cell(ctx, fam, ref, h), _ap_cell(ctx, fam, x, h)
            if pa and pb and _pair_ok(pa[1], pb[1]):
                its.append(('ap', pa[1], pb[1], f'nhanh {_code(ref)} {pa[0]}, {_code(x)} {pb[0]}'))
            for k, (mode, ca, cb, why) in enumerate(its):
                r, d3 = CP.diff(ca, cb), _d3([ca], [cb])
                vals.setdefault((x, h, mode), []).append(d3['ap'])
                log(f"    {(_code(x) if k == 0 else ''):<8}{(CL.TITLES[h] if k == 0 else ''):<11}{CROP_MODE_TEXT[mode]:<9}{ca['lr']:>7g} | {cb['lr']:<6g}  {_t3(d3):<26}"
                    f"[{_d(r['ci'][0])}, {_d(r['ci'][1])}]" + (f'  ({why})' if why else ''))
                out['tach_lr'].append(dict(doi_chung=x, dau=h, cach=mode, lr_tham_chieu=ca['lr'], lr_doi_chung=cb['lr'], d=d3, ci_danh_nghia=r['ci'], nua=r['nua'],
                                           nhanh=why))
                out['bang_hieu'].append(_abl_row('4_tach_chon_lr', ref, x, h, mode, ca['lr'], cb['lr'], d3, r, ghi_chu=why or 'nhanh luat'))
            if not same:
                log(f"    {'':<19}{'cung lr':<9}{'-':>16}  khong co lr nao hoc xong o ca hai hang")

    log('\n--- Bang 5 · CHO CHON LR (MO TA; hat 0, lan cham val cuoi pha 1, 28 nhan): lr cua luat so voi lr dung thu hai theo AUROC val va voi lr theo AP val; AP21 '
        'test = TB 5 hat ---')
    log(f"    {'hang':<6}{'dau':<11}{'lr luat':>9}{'AUROC val':>10}  {'lr thu hai (AUROC val kem)':<28}{'lr theo AP (AUROC val kem)':<28}AP21 test: luat | theo AP | luat − theo AP")
    for e in rows:
        cv_e = ctx['dl']['duong_lr'].get((e, fam[1])) or {}
        for h in HEADS:
            c, cv = cell(e, h), [p for p in cv_e.get(h, []) if p[1] == p[1]]
            if c is None or len(cv) < 2:
                continue
            auc = {_lr_key(lr): a for lr, a, _ in cv}
            lr_rule = _lr_key(c['lr'])
            if lr_rule not in auc:
                continue
            a2, lr2 = max((a, lr) for lr, a in auc.items() if lr != lr_rule)
            pc = _ap_cell(ctx, fam, e, h)
            lr_ap = _lr_key(pc[1]['lr']) if pc else None
            d_test = None if not pc or pc[0] == 'luat' else c['diem']['p21']['ap'] - pc[1]['diem']['p21']['ap']
            gap_ap = None if lr_ap is None or lr_ap not in auc else auc[lr_rule] - auc[lr_ap]
            log(f"    {_code(e):<6}{CL.TITLES[h]:<11}{lr_rule:>9g}{_f(auc[lr_rule], 4):>10}  {f'{lr2:g} ({_d4(auc[lr_rule] - a2)})':<28}"
                + (f"{'= lr cua luat':<28}" if pc and pc[0] == 'luat' else f"{(f'{lr_ap:g} ({_d4(gap_ap)})' if lr_ap is not None else '-'):<28}")
                + f"{_f(c['diem']['p21']['ap'], 4)} | {_f(pc[1]['diem']['p21']['ap'], 4) if pc else '  -  '} | {_d4(d_test)}")
            if auc[lr_rule] < a2:
                log(f'    ! {_code(e)} {CL.TITLES[h]}: lr cua luat KHONG co AUROC val cao nhat tren duong do lr (lr {lr2:g} cao hon) -> kiem lai ban chon lr cua luot')
            out['chon_lr'].append(dict(hang=e, dau=h, lr_luat=lr_rule, luat_la_max=bool(auc[lr_rule] >= a2), auroc_val_luat=auc[lr_rule], lr_thu_hai=lr2,
                                       auroc_val_kem_thu_hai=auc[lr_rule] - a2,
                                       lr_theo_ap=lr_ap, auroc_val_kem_lr_ap=gap_ap, AP21_test_luat=c['diem']['p21']['ap'],
                                       AP21_test_theo_ap=pc[1]['diem']['p21']['ap'] if pc else None, AP21_test_luat_tru_ap=d_test))

    et, ht = {e: SPECS[e].title for e in ABLATION_ROWS}, CL.TITLES
    flat = {k[:4]: v for k, v in o.items() if k[4] == 'luat' and k[2] in ABLATION_ROWS}
    ftitle = {fam: 'Ver2 · đệm vuông, không cân bằng'}
    base = (f"Việc B (PLAN_90 §10.2): Ver2, kho đệm vuông không cân bằng histogram; backbone đông cứng = RAD-DINO tinh chỉnh trên CXR-LT (một seed backbone s42, "
            f"trọng số EMA epoch 4), năm hàng chỉ khác mô-đun vùng lúc tinh chỉnh; mỗi hàng trung bình 5 hạt đầu đọc; test "
            f"{'/'.join(f'{v:,}'.replace(',', '.') for v in n_img)} ảnh, 21 lớp chính."
            + (f" Chưa có lượt: {', '.join(gaps)}." if gaps else ''))
    figs = {}
    if items:
        fi = {}
        for (x, h, r, d3, seeds, a, b), s in zip(items, sim):
            lo, hi = s['ci_dong_thoi'] if s else r['ci']
            fi[(x, h)] = dict(d=r['d'], lo=lo, hi=hi, dat=bool(s) and s['ket_luan'] in ('hon', 'kem'))
        f = FG.family_ci(fi, [x for x in ABL_X if any((x, h) in fi for h in HEADS)], HEADS, et, ht,
                         f'{_code(ref)} − X trên AP 21 lớp (bậc thang): họ {need} phép của luật — mỗi ô một đầu đọc, mỗi hàng một X', f'{_code(ref)} − X · AP 21',
                         base + (f' Vạch = khoảng ĐỒNG THỜI max-T 95 % của họ {need} phép (q = {q:.2f}'.replace('.', ',') + '); chấm đặc = đạt dạng luật ("hơn" hoặc "kém": '
                                 '|hiệu| ≥ 0,01, khoảng đồng thời không chứa 0, cùng dấu ở hai nửa test).' if full else
                                 f' Họ CHƯA đủ {need} phép: vạch = CI95 DANH NGHĨA, chưa áp luật (mọi chấm rỗng).')
                         + ' Đường đứt: ±0,01. Đọc theo từng đầu đọc; không gộp thành câu chung, không kết luận về vị trí mô-đun vùng (Tuấn 19/11). Một seed backbone: '
                         'kết luận dương cần xác nhận seed 43.', margin=CP.MARGIN, labels=('đạt dạng luật ("hơn" / "kém")', 'không phân biệt'))
        if f is not None:
            figs['ho_16'] = f
    figs['luoi_21_ap'] = FG.heat(flat, rows, HEADS, [fam], 'ap', et, ht, ftitle, base + ' Nhánh luật: mỗi hàng lấy lr có AUROC val cao nhất. Điểm, không khoảng tin cậy '
                                 '(CI ở bảng cùng ô).', panel='p21')
    f = FG.ref_diff(flat, ref, rows, HEADS, fam, FG.METRICS, et, ht, ftitle, base + ' Nhánh luật. Hiệu điểm MÔ TẢ, không CI ở hình — khoảng đồng thời và phán quyết ở '
                    'Bảng 2 cùng ô (trên AP 21).', panel='p21')
    if f is not None:
        figs['hieu_21'] = f
    f = FG.delta_modes(vals, [x for x in ABL_X if x in rows], HEADS, CROP_MODES, et, ht, ABL_MODE_TITLE,
                       f'{_code(ref)} − X trên AP 21 lớp (bậc thang) theo ba cách chọn lr — mỗi ô một đầu đọc, mỗi hàng một X', f'{_code(ref)} − X · AP 21',
                       base + ' Cùng lr: chỉ các lr có đầu học xong ở CẢ hai hàng (không có -> không có chấm; hai lr chung -> hai chấm). Cách "cùng lr" và "theo AP" '
                       'đặt sau khi thấy kết quả (19/11) — không thuộc luật đặt trước; nhánh luật là số chính. Điểm, không khoảng tin cậy. Đường đứt: ±0,01.',
                       margin=CP.MARGIN)
    if f is not None:
        figs['tach_chon_lr'] = f
    for metric, name in (('auc', 'AUROC'), ('ap', 'AP')):
        f = FG.lr_curves(ctx['dl']['duong_lr'], ctx['dl']['chon_v2'], rows, HEADS, fam[1], metric, et, ht,
                         f'Việc B · {name} val 28 nhãn (lần chấm cuối pha 1, hạt 0) theo lr — chỗ chọn lr của từng hàng')
        if f is not None:
            figs[f'do_lr_{metric}'] = f
    ctx['hinh_ablation'] = figs
    log(f"\n{len(out['nam_hang'])} dong nam hang, {len(rec)} / {need} phep cua ho, {len(out['bao_kem'])} dong bao kem, {len(out['tach_lr'])} dong tach lr, "
        f"{len(out['chon_lr'])} dong cho chon lr | {len(figs)} hinh rieng: {sorted(figs)} (ghi tep ablation_* o C8)")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return out


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))


def sync(ctx, log=print):
    '''C8. Ghi tổng hợp (JSON), bảng dài (CSV), hình (PNG) rồi dong_bo().'''
    d, t = ctx['drive'], time.strftime('%Y%m%d_%H%M%S')
    names = []
    slim = lambda c: {kk: v for kk, v in c.items() if kk not in ('m21', 'm7')}
    cells = {'|'.join(k): slim(c) for k, c in ctx['o'].items() if k[2] not in ABLATION_ROWS}      # 0.12.0: ô của 5 hàng Việc B vào tệp RIÊNG ablation_*
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
    cg, n_cg = ctx.get('cat_giua') or {}, 0
    if cg.get('bang'):                                            # 0.9.2: phép thử cắt giữa (C7b) — tệp RIÊNG, tiền tố cat_giua_ (không trộn vào tệp tổng hợp)
        cg_obj = _jsonable(dict(run=d['RUN_TAG'], t=t, goi=ctx['goi'], n_boot=ctx['n_boot'], **{k: v for k, v in cg.items() if k != 'bang'},
                                ghi_chu='Δ = kho cat giua - kho dem vuong (Ver2, khong can bang); AP bac thang / AUPRC hinh thang / AUROC macro 21 lop; CI danh nghia '
                                        "theo anh; 'cung lr' va 'theo AP' la phan tich mo ta dat sau khi thay ket qua (17/11); luat o so_sanh.hinh_hoc cua tep tong hop"))
        names.append(f'cat_giua_89_{t}.json')
        d['ghi'](names[-1], lambda q: d['ghi_json'](q, cg_obj))
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(cg['bang'][0]))
        w.writeheader()
        w.writerows(cg['bang'])
        cg_text = buf.getvalue()
        names.append(f'cat_giua_bang_89_{t}.csv')

        def _cg_csv(q):
            with open(q, 'w', encoding='utf-8-sig', newline='') as f:
                f.write(cg_text)
        d['ghi'](names[-1], _cg_csv)
        for name, fig in ctx.get('hinh_cat_giua', {}).items():
            names.append(f'cat_giua_hinh_{name}_{t}.png')
            d['ghi'](names[-1], lambda q, fig=fig: fig.savefig(q, dpi=160, bbox_inches='tight'))
        n_cg = 2 + len(ctx.get('hinh_cat_giua', {}))
    ab, n_ab = ctx.get('ablation') or {}, 0
    if ab.get('nam_hang'):                                        # 0.12.0: Việc B (C7c) — tệp RIÊNG, tiền tố ablation_ (không trộn vào tệp tổng hợp / cat_giua_*)
        ab_obj = _jsonable(dict(run=d['RUN_TAG'], t=t, goi=ctx['goi'], n_boot=ctx['n_boot'], hat_boot=P.BOOT_SEED, hat_nua=P.HALF_SEED,
                                **{k: v for k, v in ab.items() if k != 'bang_hieu'},
                                o={'|'.join(k): slim(c) for k, c in ctx['o'].items() if k[2] in ABLATION_ROWS},
                                ghi_chu='PLAN_90 10.2 + 10.2b: ho 16 phep (3) - X, khoang dong thoi max-T; phan quyet TUNG O, doc theo TUNG DAU (khong gop, khong ket '
                                        "luan ve vi tri); mAP21 = AP bac thang macro 21 lop; AUPRC = hinh thang; 'bao kem', 'tach chon lr', 'cho chon lr' la MO TA "
                                        '(CI danh nghia; tach chon lr + cho chon lr dat sau khi thay ket qua 19/11); mot seed backbone'))
        names.append(f'ablation_89_{t}.json')
        d['ghi'](names[-1], lambda q: d['ghi_json'](q, ab_obj))
        n_ab = 1
        for stem, rows_ab in (('ablation_nam_hang_89', ab['nam_hang']), ('ablation_hieu_89', ab.get('bang_hieu') or [])):
            if not rows_ab:
                continue
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=list(rows_ab[0]))
            w.writeheader()
            w.writerows(rows_ab)
            names.append(f'{stem}_{t}.csv')

            def _ab_csv(q, text=buf.getvalue()):
                with open(q, 'w', encoding='utf-8-sig', newline='') as f:
                    f.write(text)
            d['ghi'](names[-1], _ab_csv)
            n_ab += 1
        for name, fig in ctx.get('hinh_ablation', {}).items():
            names.append(f'ablation_hinh_{name}_{t}.png')
            d['ghi'](names[-1], lambda q, fig=fig: fig.savefig(q, dpi=160, bbox_inches='tight'))
            n_ab += 1
    n_cache = 0
    if ctx.get('bo_dem_moi'):                                     # có bộ mới tính -> ghi bộ nhớ đệm đầy đủ (lần sau chỉ tính lượt mới)
        names.append(f'bo_dem_bootstrap_{t}.npz')
        d['ghi'](names[-1], lambda q: CP.save_cache(q, ctx['bo_dem']))
        n_cache = 1
    du = d['dong_bo']()
    on = d['ten_tren_drive']()
    co = set(names) <= on
    log(f"C8: da ghi {len(names)} tep (1 JSON, {'1 CSV, ' if rows else ''}{len(ctx.get('hinh', {}))} PNG"
        + (f"; phep thu cat giua RIENG {n_cg} tep cat_giua_* (1 JSON, 1 CSV, {n_cg - 2} PNG)" if n_cg else '; phep thu cat giua: khong co (C7b chua chay / chua du cap)')
        + (f"; Viec B RIENG {n_ab} tep ablation_* (1 JSON, {n_ab - 1 - len(ctx.get('hinh_ablation', {}))} CSV, {len(ctx.get('hinh_ablation', {}))} PNG)" if n_ab
           else '; Viec B: khong co (C7c chua chay / chua co luot)')
        + (f", bo nho dem bootstrap {len(ctx['bo_dem'])} bo" if n_cache else ', bo nho dem khong doi') + f") | dong_bo {'DU' if du else 'CHUA DU'} | tren Drive: "
        f"{'CO du' if co else 'THIEU'}" + ('' if du and co else ' | KHONG tat runtime; chay lai C8'))
    return dict(dong_bo=du, du_tep=co, tep=names)
