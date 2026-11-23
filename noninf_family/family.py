# -*- coding: utf-8 -*-
"""Khoảng nới chung từng khối + gộp họ 9 ô.

  cxr_block    3 ô CXR-LT -> vindr_bench.evaluation.compare.max_t ở mức CXR_LEVEL (CÙNG hàm của khối VinDr); "không kém" = cận dưới > −δ; "hơn" = dạng luật
               compare.verdict với ngưỡng MARGIN_CXR (hiệu ≥ ngưỡng VÀ cận dưới > 0 VÀ dương ở hai nửa)
  vindr_block  6 ô VinDr (đặc trưng chính, k-NN / logistic × 1 / 10 / 100 %) = frozen_probe.notebook.probe_s43.noninferiority nhưng tên hàng là tham số (dùng
               cho s43 = cổng tái lập 92g, và s42 = đọc sau khi thấy số)
  family       gộp hai khối: "không kém cả họ" = MỌI ô của cả hai khối; hai bộ dữ liệu độc lập -> cùng đúng với xác suất CXR_LEVEL × VINDR_LEVEL = 95 %"""
import numpy as np

from . import config as CF


def _ci95(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float('nan')] * 2


def cxr_block(cells, level=CF.CXR_LEVEL, margin=CF.MARGIN_CXR, deltas=CF.DELTA_CXR, order=CF.CELLS_CXR):
    '''cells = {tên ô: dict(d, rut [n_boot], nua [2], d3)} -> dict(q, muc, so_lan_rut_giu, phep_so, khong_kem_moi_o).'''
    from vindr_bench.evaluation import compare as CP
    rule = lambda d, lo, hi, nua: CP.verdict(d, lo, hi, nua, margin=margin)
    R = np.stack([np.asarray(cells[c]['rut'], dtype=np.float64) for c in order])
    q, sims = CP.max_t([cells[c] for c in order], level=level, rules=[rule] * len(order))
    rows = []
    for c, s in zip(order, sims):
        lo, hi = s['ci_dong_thoi']
        rows.append(dict(o=c, d=float(s['d']), d3=cells[c]['d3'], ci_khoi=[lo, hi], ci_danh_nghia=_ci95(cells[c]['rut']), nua=list(cells[c]['nua']),
                         delta=float(deltas[c]), khong_kem=bool(lo > -deltas[c]), ket_luan_hon=s['ket_luan']))
    return dict(q=float(q), muc=float(level), nguong_hon=float(margin), so_lan_rut_giu=int(np.isfinite(R).all(0).sum()), phep_so=rows,
                khong_kem_moi_o=bool(all(r['khong_kem'] for r in rows)))


def vindr_block(cells, ref, ctl, level=CF.VINDR_LEVEL):
    '''= probe_s43.noninferiority với tên hàng là tham số. cells = {(hàng, đặc trưng, đầu, tỉ lệ): ô của vindr_bench (diem, nua, m21, tung_hat)}.'''
    from frozen_probe import config as C
    from frozen_probe import config_s43 as S
    from frozen_probe import scoring as SC
    from vindr_bench.evaluation import compare as CP
    items = []
    for probe, f in C.FAMILY:
        a, b = cells[(ref, C.PRIMARY_FEATURE, probe, f)], cells[(ctl, C.PRIMARY_FEATURE, probe, f)]
        items.append((probe, f, a, b, CP.diff(a, b)))
    q, sims = CP.max_t([it[4] for it in items], level=level)
    rec = []
    for (probe, f, a, b, r), s in zip(items, sims):
        lo, hi = s['ci_dong_thoi']
        de = S.DELTA[(probe, f)]
        pd = SC.per_draw(a, b)
        rec.append(dict(dau=probe, ti_le=f, d=SC.d3(a, b), ci_khoi=[lo, hi], ci_danh_nghia=r['ci'], nua=r['nua'], delta=de, khong_kem=bool(lo > -de),
                        ket_luan_hon=s['ket_luan'], tung_lan_rut=pd, so_lan_rut_duong=int(sum(v > 0 for v in pd)), rut=[None if v != v else float(v) for v in r['rut']]))
    return dict(q=float(q), muc=float(level), so_o_khoi=len(rec), phep_so=rec, khong_kem_moi_o=bool(all(x['khong_kem'] for x in rec)))


def compare_92g(mine, js):
    '''Khối VinDr s43 tính lại (vindr_block) so với khong_kem_C của JSON 92g: q, khoảng, kết luận, hiệu, hai nửa, lần rút -> (đạt, lệch lớn nhất, [lỗi]).'''
    err, dev = [], 0.0
    if not np.isclose(mine['q'], js['q'], rtol=0, atol=CF.VINDR_TOL):
        err.append(f"q {mine['q']} != {js['q']}")
    dev = max(dev, abs(mine['q'] - js['q']))
    if len(mine['phep_so']) != len(js['phep_so']):
        return False, float('inf'), ['so o khac']
    for a, b in zip(mine['phep_so'], js['phep_so']):
        if (a['dau'], float(a['ti_le'])) != (b['dau'], float(b['ti_le'])):
            err.append(f"thu tu o {a['dau']} {a['ti_le']} != {b['dau']} {b['ti_le']}"); continue
        num = [*zip(a['ci_khoi'], b['ci_khoi']), *zip(a['ci_danh_nghia'], b['ci_danh_nghia']), *zip(a['nua'], b['nua'])]
        num += [(a['d'][m], b['d'][m]) for m in ('ap', 'auprc_thang', 'auroc')] + list(zip(a['tung_lan_rut'], b['tung_lan_rut']))
        ra, rb = [x for x in a['rut'] if x is not None], [x for x in b['rut'] if x is not None]
        num += list(zip(ra, rb)) if len(ra) == len(rb) else [(0.0, float('inf'))]
        dd = max(abs(float(x) - float(y)) for x, y in num)
        dev = max(dev, dd)
        if dd > CF.VINDR_TOL or a['khong_kem'] != b['khong_kem'] or a['ket_luan_hon'] != b['ket_luan_hon'] or float(a['delta']) != float(b['delta']):
            err.append(f"{a['dau']} {a['ti_le']}: lech {dd:.1e} / ket luan {a['khong_kem']},{a['ket_luan_hon']} vs {b['khong_kem']},{b['ket_luan_hon']}")
    return not err, float(dev), err


def family(cxr, vindr):
    '''Gộp hai khối của CÙNG seed -> dict(khong_kem_ca_ho, o_hon [...], so_o, muc_chung). Tên ô = tên mô tả (không mã).'''
    from frozen_probe import config as C
    tc = lambda r: f"CXR-LT {CF.CELL_TEXT[r['o']]}"
    tv = lambda r: f"VinDr {C.PROBE_TEXT[r['dau']]} {C.FRACTION_TEXT[r['ti_le']]} nhan"
    hon = [tc(r) for r in cxr['phep_so'] if r['ket_luan_hon'] == 'hon'] + [tv(r) for r in vindr['phep_so'] if r['ket_luan_hon'] == 'hon']
    kem = [tc(r) for r in cxr['phep_so'] if r['ket_luan_hon'] == 'kem'] + [tv(r) for r in vindr['phep_so'] if r['ket_luan_hon'] == 'kem']
    truot = [tc(r) for r in cxr['phep_so'] if not r['khong_kem']] + [tv(r) for r in vindr['phep_so'] if not r['khong_kem']]
    n = len(cxr['phep_so']) + len(vindr['phep_so'])
    assert n == CF.N_CELLS, f'ho {n} o != {CF.N_CELLS} -> DUNG'
    return dict(so_o=n, muc_chung=float(cxr['muc'] * vindr['muc'] / 100.0), khong_kem_ca_ho=bool(cxr['khong_kem_moi_o'] and vindr['khong_kem_moi_o']),
                o_truot_khong_kem=truot, o_hon=hon, o_kem=kem)
