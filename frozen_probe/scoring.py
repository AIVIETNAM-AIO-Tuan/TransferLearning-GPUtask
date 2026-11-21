# -*- coding: utf-8 -*-
"""Chấm + luật của phép kiểm ít nhãn (PLAN_92 §11.2–11.3) bằng hàm của vindr_bench.evaluation (KHÔNG viết lại): mỗi bộ dự đoán test -> set_stats
(AP bậc thang / AUPRC hình thang / AUROC trên 4 panel + mAP21 của 2.000 lần rút bootstrap theo ảnh, hạt 8502 + hai nửa test, hạt 8501); ô = make_cell
của các LẦN RÚT (vai trò như các hạt của 89 Ver2: số = TB metric của các lần rút, mỗi lần rút bootstrap tính riêng rồi lấy TB); hiệu ghép cặp, max-T,
dạng luật = compare.diff / max_t / verdict.

Khoá ô: (hàng, đặc trưng, cấu hình đầu đọc, tỉ lệ). Cấu hình: 'knn' (k = round(√n)), 'logistic' (C = 1) — chính; 'knn_k50', 'logistic_C0.1',
'logistic_C10' — nhạy (chỉ cặp chính)."""
import numpy as np

from . import config as C

PRIMARY_CFGS = C.PROBES
SENS_CFGS = tuple(f'logistic_C{c:g}' for c in C.SENSITIVITY['logistic_C']) + (f"knn_k{C.SENSITIVITY['knn_k']}",)
CFG_TEXT = dict(knn='k-NN (k = round(sqrt n))', logistic='logistic L2 (C = 1)', **{f'logistic_C{c:g}': f'logistic L2 (C = {c:g})' for c in C.SENSITIVITY['logistic_C']},
                **{f"knn_k{C.SENSITIVITY['knn_k']}": f"k-NN (k = {C.SENSITIVITY['knn_k']}, nhu 78r)"})


def fmt(x, nd=3):
    return '  -  ' if x is None or x != x else f'{x:.{nd}f}'


def sgn(x, nd=3):
    return '   -  ' if x is None or x != x else f'{x:+.{nd}f}'


def configs_for(row, fraction):
    '''Danh sách (đặc trưng, cấu hình) cần chạy cho một hàng ở một tỉ lệ: chính cho mọi hàng; nhạy chỉ cho cặp chính.'''
    out = [(C.PRIMARY_FEATURE, p) for p in PRIMARY_CFGS]
    if row in C.PAIR:
        out += [(C.PRIMARY_FEATURE, f'logistic_C{c:g}') for c in C.SENSITIVITY['logistic_C']]
        if fraction in C.SENSITIVITY['knn_k_fractions']:
            out.append((C.PRIMARY_FEATURE, f"knn_k{C.SENSITIVITY['knn_k']}"))
        out += [(C.SENSITIVITY['feature'], p) for p in PRIMARY_CFGS]
    return out


def is_primary(feature, cfg):
    return feature == C.PRIMARY_FEATURE and cfg in PRIMARY_CFGS


def cell_from_stats(stats):
    '''[set_stats của từng lần rút, theo thứ tự lần rút] -> ô (compare.make_cell: TB các lần rút; SD giữa lần rút; tung_hat = AP21 từng lần rút).'''
    from vindr_bench.evaluation import compare as CP
    return CP.make_cell(stats)


def cell_point(preds_list, target, cols):
    '''Ô CHỈ điểm (không bootstrap) cho bản nhạy: TB các lần rút của P.point trên 4 panel + AP21 từng lần rút.'''
    from vindr_bench.evaluation import panels as P
    pts = [{p: P.point(x, target, cols[p]) for p in P.PANELS} for x in preds_list]
    diem = {p: {m: float(np.mean([q[p][m] for q in pts])) for m in ('ap', 'auprc_thang', 'auroc')} for p in P.PANELS}
    return dict(so_hat=len(pts), diem=diem, tung_hat=[float(q['p21']['ap']) for q in pts], chi_diem=True)


def d3(a, b, panel='p21'):
    return {m: float(a['diem'][panel][m] - b['diem'][panel][m]) for m in ('ap', 'auprc_thang', 'auroc')}


def t3(d):
    return f"{sgn(d['ap'])} / {sgn(d['auprc_thang'])} / {sgn(d['auroc'])}"


def per_draw(a, b):
    '''Δ AP21 từng lần rút (cùng chỉ số lần rút ở hai hàng) -> list.'''
    assert len(a['tung_hat']) == len(b['tung_hat']), 'hai o khac so lan rut -> DUNG'
    return [float(x - y) for x, y in zip(a['tung_hat'], b['tung_hat'])]


def family(cells):
    '''Họ 6 phép (3) − (0) (đặc trưng chính; PLAN_92 §11.3) -> dict(q, phep_so=[...], day_du). Thiếu ô nào -> họ CHƯA đủ: chỉ hiệu danh nghĩa, chưa áp luật.'''
    return family_pair(cells, C.REF, C.CONTROL)


def family_pair(cells, ref, control):
    '''Họ 6 phép ref − control (đặc trưng chính; luật §11.3 — PLAN_92 §13.3 áp cho TỪNG cặp CÓ − KHÔNG của cùng backbone) -> như family().'''
    from vindr_bench.evaluation import compare as CP
    assert CP.MARGIN == C.MARGIN, 'nguong luat cua goi khac config -> DUNG'
    items = []
    for probe, f in C.FAMILY:
        a, b = cells.get((ref, C.PRIMARY_FEATURE, probe, f)), cells.get((control, C.PRIMARY_FEATURE, probe, f))
        if a is not None and b is not None:
            items.append((probe, f, a, b, CP.diff(a, b)))
    full = len(items) == len(C.FAMILY)
    q, sims = CP.max_t([it[4] for it in items], level=C.LEVEL) if full else (float('nan'), [None] * len(items))
    rec = []
    for (probe, f, a, b, r), s in zip(items, sims):
        pd = per_draw(a, b)
        rec.append(dict(dau=probe, ti_le=f, d=d3(a, b), ci_dong_thoi=s['ci_dong_thoi'] if s else None, ci_danh_nghia=r['ci'], nua=r['nua'], tung_lan_rut=pd,
                        so_lan_rut_duong=int(sum(v > 0 for v in pd)), ket_luan=s['ket_luan'] if s else f"danh nghia: {r['ket_luan']}"))
    return dict(q=q if full else None, day_du=full, co=len(items), so_phep=len(C.FAMILY), phep_so=rec)


def descriptive(cells):
    '''Hiệu mô tả (CI95 danh nghĩa), đặc trưng chính: X − (0) cho mọi hàng mô tả, (3) − (2) (ghi ngược), và Δ(1 %) − Δ(100 %) của cặp chính.'''
    from vindr_bench.evaluation import compare as CP
    out = dict(hieu=[], xu_huong=[])
    pairs = [(x, C.CONTROL) for x in C.DESCRIPTIVE_VS_CONTROL] + [C.WRITE_BACK_PAIR]
    for a_, b_ in pairs:
        for probe in C.PROBES:
            for f in C.FRACTIONS:
                a, b = cells.get((a_, C.PRIMARY_FEATURE, probe, f)), cells.get((b_, C.PRIMARY_FEATURE, probe, f))
                if a is None or b is None:
                    continue
                r = CP.diff(a, b)
                out['hieu'].append(dict(hang_a=a_, hang_b=b_, dau=probe, ti_le=f, d=d3(a, b), ci_danh_nghia=r['ci'], nua=r['nua']))
    lo, hi = min(C.FRACTIONS), max(C.FRACTIONS)
    for probe in C.PROBES:
        k = lambda row, f: cells.get((row, C.PRIMARY_FEATURE, probe, f))
        a1, b1, a0, b0 = k(C.REF, lo), k(C.CONTROL, lo), k(C.REF, hi), k(C.CONTROL, hi)
        if all(c is not None for c in (a1, b1, a0, b0)):
            r = CP.diff_of_diffs(a1, b1, a0, b0)
            out['xu_huong'].append(dict(dau=probe, d=r['d'], ci_danh_nghia=r['ci'], nua=r['nua']))
    return out


def sensitivity(cells):
    '''Bản nhạy của cặp chính (CHỈ điểm): Δ (3) − (0) với từng cấu hình / đặc trưng phụ, cạnh Δ của cấu hình chính cùng tỉ lệ.'''
    out = []
    for f in C.FRACTIONS:
        for feat, cfg in configs_for(C.REF, f):
            if is_primary(feat, cfg):
                continue
            a, b = cells.get((C.REF, feat, cfg, f)), cells.get((C.CONTROL, feat, cfg, f))
            if a is None or b is None:
                continue
            base_cfg = 'knn' if cfg.startswith('knn') else 'logistic'
            a0, b0 = cells.get((C.REF, C.PRIMARY_FEATURE, base_cfg, f)), cells.get((C.CONTROL, C.PRIMARY_FEATURE, base_cfg, f))
            out.append(dict(dac_trung=feat, cau_hinh=cfg, ti_le=f, d=d3(a, b), d_chinh=d3(a0, b0) if a0 is not None and b0 is not None else None,
                            tung_lan_rut=per_draw(a, b)))
    return out


def chance_level(target, cols):
    '''Mức may rủi của mAP21 = TB tỉ lệ dương test trên 21 lớp chính (AP kỳ vọng của điểm ngẫu nhiên).'''
    t = np.asarray(target)[:, cols['p21']]
    return float(t.mean(0).mean())
