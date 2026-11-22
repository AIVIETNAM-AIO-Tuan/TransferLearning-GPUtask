# -*- coding: utf-8 -*-
"""Ô của notebook tổng hợp `93t_tong_hop` (CPU; PLAN_93 §5, §7) — khuôn của `89t`, nhưng MỖI BỘ DỮ LIỆU MỘT PHẦN RIÊNG (bảng, luật, hình; Tuấn 22/11).
Chạy được khi mới có một phần kết quả: lượt thiếu được liệt kê, họ phép so chưa đủ thì chỉ in hiệu danh nghĩa.

    C2  setup         hàm của khối Drive chuẩn + thông tin gói
    C3  collect       kiểm kê + tải mọi lượt 93v của mọi bộ + cổng từng lượt + cổng giữa các lượt của một bộ (summary/collect.py)
    C4  score         bootstrap 2.000 lần theo CỤM (RSNA: người NIH; Shenzhen: ảnh), cùng lần rút cho mọi hàng / đầu / hạt của một bộ -> ô (TB 5 hạt);
                      cổng: bootstrap trọng số 1 = điểm
    Phần của mỗi bộ (RSNA: C5–C7; Shenzhen: C8–C10):
        tables(bộ)    điều kiện + val cuối pha 1 (theo dõi) + ma trận test 3 số đo + bảng dài đủ điều kiện từng hàng + thứ hạng + cờ biên lưới
        rule(bộ)      điều khoản trần + họ 32 phép B − X (khoảng đồng thời max-T, phán quyết) + hiệu mô tả (giữa đầu đọc; nhánh theo AP − luật)
        figures(bộ)   8 hình: chấm + CI (AP / AUPRC / AUROC), lưới encoder × đầu, họ B − X, hiệu B − X (3 số đo), đường dò lr (AUROC / AP val)
    C11 side_by_side  đặt cạnh thứ hạng giữa các bộ (+ VinDr của 89t Ver2 kho đệm vuông không cân bằng nếu đọc được) — MÔ TẢ
    C12 sync          ghi JSON + CSV + PNG (tên tệp mang tên bộ) rồi dong_bo()"""
import csv
import io
import json
import os
import time

import numpy as np

from .. import __version__, config
from ..evaluation import bootstrap as BS
from ..evaluation import single as S
from ..summary import collect as SC
from ..summary import figures as FG3
from ..summary import tables as T

DRIVE_NAMES = ('kiem_ke', 'tai', 'xung_dot', 'ghi', 'ghi_json', 'dong_bo', 'ten_tren_drive', 'RUN_TAG', 'GHI_LOC')
NAMES = ('rsna', 'shenzhen')
SELF_CHECK_TOL = 1e-9


def _packages():
    import external_benchmarks
    import vindr_bench
    import crossdata_bench
    return {m.__name__: dict(phien_ban=m.__version__, sha=getattr(m, 'GOI_SHA', None)) for m in (crossdata_bench, vindr_bench, external_benchmarks)}


def setup(ns, log=print):
    '''C2. -> CTX.'''
    missing = [k for k in DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    drive = {k: ns[k] for k in DRIVE_NAMES}
    assert drive['RUN_TAG'] == config.SUMMARY_TAG, f"RUN_TAG {drive['RUN_TAG']!r} != {config.SUMMARY_TAG!r} -> DUNG"
    BS._check_margin()
    ctx = dict(drive=drive, t=time.strftime('%Y%m%d_%H%M%S'), goi=_packages(), bo={}, hinh={})
    log(f"tong hop 93 | goi crossdata_bench {__version__} | " + ', '.join(f"{k} {g['phien_ban']} @ {(g['sha'] or '?')[:12]}" for k, g in ctx['goi'].items())
        + f" | {os.cpu_count()} CPU | so cua luat: AP bac thang (RSNA, Shenzhen) | bootstrap {config.N_BOOT} lan theo cum, hat {config.BOOT_SEED}; hai nua "
        f"hat {config.HALF_SEED} | luat: hieu >= +{config.RULE_DELTA} VA can duoi dong thoi > 0 VA duong o hai nua; dieu khoan tran {config.CEILING}")
    return ctx


def collect(ctx, names=NAMES, log=print):
    '''C3. -> {bộ: Bundle}.'''
    t0 = time.time()
    log('--- kiem ke + tai cac luot 93v (moi bo) ---')
    dl = SC.collect(ctx['drive'], names, log=log)
    ctx['dl'] = dl
    for name, bd in dl.items():
        commits = sorted({r['goi_93'] or '?' for r in bd.runs.values()})
        log(f"{name}: {len(bd.runs)} / {len(config.TABLE_ORDER)} luot, {len(bd.sets)} bo du doan (moi bo = mot hat cua mot nhanh) | crossdata_bench cua luot: "
            f"{[c[:12] for c in commits]} | nhan / thu tu anh / ma cum test giong nhau o moi luot: DAT")
        for m in bd.missing:
            log(f'  ! {m}')
    log(f'C3: {(time.time() - t0) / 60:.1f} phut')
    return dl


def _progress(log, name, n):
    step, t0 = max(1, n // 10), time.time()

    def tick(i):
        if i % step == 0 or i == n:
            el = time.time() - t0
            log(f'  {name}: bootstrap {i}/{n} bo | {el / 60:.1f} phut | con ~{el / i * (n - i) / 60:.1f} phut')
    return tick


def score(ctx, log=print, n_boot=config.N_BOOT):
    '''C4. -> {bộ: dict(cells {(encoder, đầu, nhánh): ô}, …)}.'''
    from vindr_bench.evaluation import metrics as M
    t0 = time.time()
    for name, bd in ctx['dl'].items():
        if not bd.sets:
            log(f'{name}: chua co luot nao -> bo qua')
            continue
        sp, t1 = config.dataset(name), time.time()
        cols = list(range(len(sp.label_names)))                 # một nhãn: [0]
        W = BS.weights(bd.g_test, sp.unit, n_boot)
        half = S.halves(bd.g_test)
        # cổng: bootstrap bằng trọng số 1 (mọi ảnh một lần) = điểm của sklearn trên cả test — trên bộ đầu tiên của bộ dữ liệu
        s0 = bd.sets[0]
        one = BS.set_stats(s0.test, bd.y_test, cols, np.ones((1, len(bd.y_test))), half)
        lech = max(abs(float(one['rut'][m][0]) - one['diem'][m]) for m in BS.METRICS)
        assert lech <= SELF_CHECK_TOL, f'{name}: bootstrap trong so 1 KHAC diem (lech {lech:.1e}) -> DUNG'
        tick = _progress(log, name, len(bd.sets))
        stats = []
        for i, s in enumerate(bd.sets, 1):
            stats.append(BS.set_stats(s.test, bd.y_test, cols, W, half))
            tick(i)
        groups = {}
        for s, st in zip(bd.sets, stats):
            groups.setdefault((s.encoder, s.head, s.arm), []).append((s, st))
        cells = {}
        for key, items in groups.items():
            items.sort(key=lambda x: x[0].seed)
            s0 = items[0][0]
            ens = S.point(bd.y_test, np.mean([s.test for s, _ in items], 0), cols)
            cell = BS.make_cell([st for _, st in items], ens)
            vals = [S.point(bd.y_val, s.val, cols) for s, _ in items]
            cell.update(lr=s0.lr, bien=s0.edge, nguon=s0.source, val_auroc_hat0=s0.val_auroc, hat=[s.seed for s, _ in items],
                        val={m: dict(tb=float(np.mean([v[m] for v in vals])), sd=float(np.std([v[m] for v in vals], ddof=1)) if len(vals) > 1 else None)
                             for m in BS.METRICS})
            cells[key] = cell
        loai = max(c['loai'][m] for c in cells.values() for m in BS.METRICS)
        ctx['bo'][name] = dict(name=name, cells=cells, cot=cols, n_boot=n_boot, n_test=int(len(bd.y_test)), n_val=int(len(bd.y_val)),
                               so_duong_test=int(bd.y_test.sum()), so_cum_test=int(len(set(bd.g_test.tolist()))), nua_anh=[int((half == h).sum()) for h in (0, 1)],
                               nua_cum=[int(len(set(bd.g_test[half == h].tolist()))) for h in (0, 1)],
                               duong_lr={e: SC.curves(r['js']) for e, r in bd.runs.items()}, chon={e: r['js']['chon'] for e, r in bd.runs.items()},
                               thieu=list(bd.missing), lan_rut_loai_toi_da=loai, phut=(time.time() - t1) / 60,
                               luot={e: dict(tag=r['tag'], t=r['t'], crossdata_bench=r['goi_93'], ban_cu_bo_qua=len(r['cu'])) for e, r in bd.runs.items()},
                               cong={e: g for e, g in bd.gates.items()})
        log(f"{name}: {len(bd.sets)} bo -> {len(cells)} o (encoder, dau, nhanh) | bootstrap {n_boot} lan theo "
            f"{'nguoi NIH' if sp.unit == 'nguoi' else 'anh'} ({ctx['bo'][name]['so_cum_test']} cum) | cong trong so 1 = diem: lech {lech:.1e} DAT | lan rut bi "
            f"loai (thieu ca duong / am): toi da {loai} | {(time.time() - t1) / 60:.1f} phut")
    log(f'C4: {(time.time() - t0) / 60:.1f} phut')
    return ctx['bo']


def _bo(ctx, name):
    assert name in ctx['bo'], f'{name}: chua co o (chay C4 truoc / bo chua co luot) -> DUNG'
    return ctx['bo'][name]


def tables(ctx, name, log=print):
    '''Phần của một bộ — BẢNG. -> bảng dài (list dict).'''
    bo = _bo(ctx, name)
    log(f"=== {config.dataset(name).title} — dieu kien chung moi hang: {T.condition_text(bo)} ===")
    for m in bo['thieu']:
        log(f'  ! {m}')
    T.val_table(bo, log)
    T.matrices(bo, log)
    rows = T.long_table(bo, log)
    rk = T.ranks(bo, log)
    fl = T.edge_flags(bo, log)
    bo.update(bang_dai=rows, thu_hang=rk, co_bien=fl)
    return rows


def rule(ctx, name, log=print):
    '''Phần của một bộ — LUẬT: điều khoản trần + họ B − X + hiệu mô tả. -> họ phép so (dict).'''
    bo = _bo(ctx, name)
    sp = config.dataset(name)
    hk = T.heads()[0]
    encs = T.encoders(bo)
    comps = [e for e in config.TABLE_ORDER if e != config.REFERENCE]
    log(f"=== {sp.title}: HO {len(comps) * len(hk)} phep {config.ROW_TITLES[config.REFERENCE]} - X (luat PLAN_93 5; so cua luat = AP bac thang; 'B hon' = hieu "
        f">= +{config.RULE_DELTA} VA can duoi DONG THOI > 0 VA hieu > 0 o CA HAI nua test; 'B kem' doi xung) ===")
    tr = BS.ceiling(bo['cells'], config.REFERENCE, encs, hk)
    T.ceiling_text(tr, log)
    descriptive = tr['ap_dung'] or not sp.in_rule
    if not sp.in_rule:
        log('bo nay NGOAI luat (PLAN_93 5) -> chi mo ta')
    fam = BS.family(bo['cells'], config.REFERENCE, comps, hk, descriptive=descriptive)
    rows = T.family_table(bo, fam, log)
    out = dict(giua_dau={}, nhanh_ap=[])
    c = bo['cells']
    for e in encs:
        for a, b in T.HEAD_PAIRS:
            if (e, a, 'luat') in c and (e, b, 'luat') in c:
                out['giua_dau'][f'{e}|{a}-{b}'] = BS.pairwise(c[(e, a, 'luat')], c[(e, b, 'luat')])
        for h in hk:
            if (e, h, 'ap') in c and (e, h, 'luat') in c:
                out['nhanh_ap'].append(dict(encoder=e, dau=h, lr_ap=c[(e, h, 'ap')]['lr'], lr_luat=c[(e, h, 'luat')]['lr'],
                                            hieu=BS.pairwise(c[(e, h, 'ap')], c[(e, h, 'luat')])))
    T.descriptive_tables(bo, out, log)
    bo.update(tran=tr, ho=fam, ho_bang=rows, mo_ta=out)
    return fam


def _n(x):
    '''Số nguyên kiểu Việt: dấu chấm ngăn nghìn.'''
    return f'{int(x):,}'.replace(',', '.')


def _note(bo):
    sp = config.dataset(bo['name'])
    unit = 'người NIH' if sp.unit == 'nguoi' else 'ảnh'
    exp = '; '.join(f'{config.ROW_TITLES[e]} {v}' for e, v in config.EXPOSURE[bo['name']].items())
    tail = (f' Test {_n(bo["n_test"])} ảnh: khoảng rộng; điều khoản trần (đối chứng tốt nhất ≥ 0,98) -> luật chỉ mô tả ở bộ này.'
            if (bo.get('tran') or {}).get('ap_dung') else '')
    return (f'{sp.title}: test {_n(bo["n_test"])} ảnh, {_n(bo["so_duong_test"])} dương, {_n(bo["so_cum_test"])} cụm ({unit}). Giao thức 89 Ver2, kho 518 '
            f'đệm vuông không cân bằng histogram, backbone đông cứng; số = TB 5 hạt đầu đọc; CI95 danh nghĩa = bootstrap {_n(bo["n_boot"])} lần theo {unit}. '
            f'Tiếp xúc tiền huấn luyện (không nhãn) với ảnh của bộ: {exp}.{tail}')


def figures(ctx, name, log=print, show=True):
    '''Phần của một bộ — HÌNH. -> {tên: Figure}.'''
    bo = _bo(ctx, name)
    if 'ho' not in bo:
        rule(ctx, name, log=lambda *a, **k: None)
    note = _note(bo)
    figs = {}
    for m in BS.METRICS:
        figs[f'cham_ci_{FG3.SLUG[m]}'] = FG3.dots_ci(bo, m, note)
    figs['luoi_3_so_do'] = FG3.heat3(bo, note)
    f = FG3.family(bo, bo['ho'], note + ' Vạch = khoảng đồng thời 95 % (max-T) của cả họ; đường đứt = ±0,01 (ngưỡng của luật).')
    if f is not None:
        figs['ho_b_tru_x'] = f
    f = FG3.diff_heat(bo, note)
    if f is not None:
        figs['hieu_b_tru_x_3_so_do'] = f
    for metric in ('auc', 'ap'):
        f = FG3.lr_curves(bo, metric, f'{config.dataset(name).title}: val {_n(bo["n_val"])} ảnh; đường = hạt 0 của lần dò lr (pha 1); lưới 13 lr giữ như 89.')
        if f is not None:
            figs[f'do_lr_{metric}'] = f
    ctx['hinh'][name] = figs
    log(f"{config.dataset(name).title}: {len(figs)} hinh: {sorted(figs)}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return figs


def _vindr(ctx, log):
    '''Bảng VinDr của 89t (Ver2, kho đệm vuông không cân bằng; mAP21 = AP bậc thang macro 21 lớp) — chỉ để đặt cạnh THỨ HẠNG. -> dict | None.'''
    v, d = config.VINDR_SUMMARY, ctx['drive']
    try:
        kk = d['kiem_ke'](v['folder'], mau=v['pattern'])[1]
    except Exception as e:                                         # thư mục của 89 không đọc được -> bỏ qua, không lỗi
        log(f'  (VinDr 89t: khong kiem ke duoc {"/".join(v["folder"])}: {type(e).__name__}) -> bo qua')
        return None
    if not kk:
        log(f'  (VinDr 89t: chua co tong_hop_89_*.json trong {"/".join(v["folder"])}) -> bo qua')
        return None
    name = sorted(kk)[-1]
    with open(d['tai'](kk[name][0]), encoding='utf-8') as f:
        o = json.load(f).get('o') or {}
    hk = T.heads()[0]
    ap = {(e, h): float(o[k]['diem'][v['panel']]['ap']) for e in config.TABLE_ORDER for h in hk for k in [v['cell'].format(enc=e, head=h)] if k in o}
    return dict(tep=name, ap=ap) if ap else None


def side_by_side(ctx, log=print):
    '''C11. Thứ hạng của 9 hàng ở từng đầu, đặt cạnh giữa các bộ + tương quan hạng Spearman (MÔ TẢ; số đo khác nhau giữa bộ — chỉ so thứ hạng).'''
    from vindr_bench.evaluation import compare as CP
    hk, ht = T.heads()
    cols = {name: {(e, h): c['diem']['ap'] for (e, h, arm), c in bo['cells'].items() if arm == 'luat'} for name, bo in ctx['bo'].items()}
    title = {name: config.dataset(name).title.split(' (')[0] for name in cols}
    vd = _vindr(ctx, log)
    if vd:
        cols['vindr_89'] = vd['ap']
        title['vindr_89'] = 'VinDr (89, mAP21)'
    log("=== DAT CANH cac bo (MO TA): so cua luat tung bo — RSNA / Shenzhen: AP bac thang mot nhan; VinDr (89t): mAP21 = AP bac thang macro 21 lop — so KHAC "
        "nhau giua bo -> chi doc THU HANG (1 = cao nhat); cung giao thuc 89 Ver2, cung kieu kho dem vuong khong can bang ===")
    out = {}
    for h in hk:
        ranks = {}
        for n, v in cols.items():
            hv = {e: x for (e, hh), x in v.items() if hh == h}
            order = sorted(hv, key=lambda e: -hv[e])
            ranks[n] = {e: order.index(e) + 1 for e in hv}
        log(f"--- {ht[h]} --- {'hang':<14}" + ''.join(f'{title[n]:>22}' for n in cols))
        for e in config.TABLE_ORDER:
            log(f"{'':<4}{config.ROW_TITLES[e]:<14}" + ''.join((f"{T.f3(cols[n][(e, h)])} (hang {ranks[n][e]})" if (e, h) in cols[n] else '-').rjust(22) for n in cols))
        names = list(cols)
        sp = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                common = [e for e in config.TABLE_ORDER if (e, h) in cols[a] and (e, h) in cols[b]]
                sp[f'{a}~{b}'] = (CP.spearman([cols[a][(e, h)] for e in common], [cols[b][(e, h)] for e in common]), len(common))
        log(f"{'':<4}tuong quan hang Spearman (mo ta, 9 hang): " + ' | '.join(f'{title[k.split("~")[0]]} ~ {title[k.split("~")[1]]}: {T.f3(v[0], 2)} ({v[1]})'
                                                                       for k, v in sp.items()))
        out[h] = dict(hang=ranks, spearman={k: v[0] for k, v in sp.items()})
    ctx['canh_nhau'] = dict(bo=list(cols), vindr=(vd or {}).get('tep'), theo_dau=out)
    return ctx['canh_nhau']


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda v: v.item() if hasattr(v, 'item') else (v.tolist() if hasattr(v, 'tolist') else str(v))))


def _slim(c):
    return {k: v for k, v in c.items() if k != 'rut'}


def _csv_writer(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    text = buf.getvalue()

    def write(q):
        with open(q, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(text)
    return write


def sync(ctx, log=print):
    '''C12. Ghi tổng hợp (JSON), bảng dài + họ B − X mỗi bộ (CSV), hình mỗi bộ (PNG) rồi dong_bo().'''
    d, t = ctx['drive'], time.strftime('%Y%m%d_%H%M%S')
    names = []
    per = {}
    for name, bo in ctx['bo'].items():
        per[name] = dict(dieu_kien=T.condition_text(bo), thieu=bo['thieu'], cong=bo['cong'], luot=bo['luot'], n_test=bo['n_test'], n_val=bo['n_val'],
                         so_duong_test=bo['so_duong_test'], so_cum_test=bo['so_cum_test'], nua_anh=bo['nua_anh'], nua_cum=bo['nua_cum'], n_boot=bo['n_boot'],
                         lan_rut_loai_toi_da=bo['lan_rut_loai_toi_da'], o={'|'.join(k): _slim(c) for k, c in bo['cells'].items()}, tran=bo.get('tran'),
                         ho=bo.get('ho'), mo_ta=bo.get('mo_ta'), thu_hang=bo.get('thu_hang'), co_bien=bo.get('co_bien'), tiep_xuc=config.EXPOSURE[name],
                         duong_lr={e: v for e, v in bo['duong_lr'].items()})
    obj = _jsonable(dict(run=d['RUN_TAG'], t=t, goi=ctx['goi'], n_boot=config.N_BOOT, hat_boot=config.BOOT_SEED, hat_nua=config.HALF_SEED,
                         luat=dict(nguong=config.RULE_DELTA, tran=config.CEILING, so_cua_luat='AP bac thang'), bo=per, canh_nhau=ctx.get('canh_nhau'),
                         ghi_chu='PLAN_93 5: so chinh = TB 5 hat; CI danh nghia tru khoang dong thoi max-T cua ho B - X; bootstrap theo cum (RSNA nguoi NIH, '
                                 'Shenzhen anh); AP = bac thang, AUPRC = hinh thang; Shenzhen: dieu khoan tran -> mo ta'))
    names.append(f'tong_hop_93_{t}.json')
    d['ghi'](names[-1], lambda q: d['ghi_json'](q, obj))
    n_csv = 0
    for name, bo in ctx['bo'].items():
        for stem, rows in ((f'bang_dai_93_{name}', bo.get('bang_dai')), (f'ho_b_tru_x_93_{name}', bo.get('ho_bang'))):
            if rows:
                names.append(f'{stem}_{t}.csv')
                d['ghi'](names[-1], _csv_writer(rows))
                n_csv += 1
    n_png = 0
    for name, figs in ctx.get('hinh', {}).items():
        for fn, fig in figs.items():
            names.append(f'hinh_93_{name}_{fn}_{t}.png')
            d['ghi'](names[-1], lambda q, fig=fig: fig.savefig(q, dpi=160, bbox_inches='tight'))
            n_png += 1
    du = d['dong_bo']()
    on = d['ten_tren_drive']()
    co = set(names) <= on
    log(f"C12: da ghi {len(names)} tep (1 JSON, {n_csv} CSV, {n_png} PNG) | dong_bo {'DU' if du else 'CHUA DU'} | tren Drive: {'CO du' if co else 'THIEU'}"
        + ('' if du and co else ' | KHONG tat runtime; chay lai C12'))
    return dict(dong_bo=du, du_tep=co, tep=names)
