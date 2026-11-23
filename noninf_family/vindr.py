# -*- coding: utf-8 -*-
"""Ô VinDr của họ 9 ô (đặc trưng chính, k-NN / logistic × 1 / 10 / 100 % nhãn, mAP21 = AP bậc thang macro 21 lớp).

  cells_92g    s43: ô của 92g = JSON (o) + lần rút bootstrap đã lưu (m21) -> đủ để tính lại khối VinDr s43 (cổng: = khong_kem_C của 92g)
  cells_92b    s42: 92b (gói 0.1.0) KHÔNG lưu lần rút -> tính lại từ dự đoán test đã lưu (float16) bằng ĐÚNG bootstrap của 92g (compare.many_stats + make_cell)
  check_92b    ô s42 tính lại so với ô 92b (điểm 4 panel × 3 số, CI95, hai nửa, từng lần rút, AP từng lớp): k-NN = trung bình nhãn k láng giềng (bội của
               1/k) -> float16 giữ nguyên thứ tự -> PHẢI trùng tuyệt đối; logistic -> in độ lệch (float16 có thể gộp điểm gần nhau)"""
import numpy as np

from . import config as CF


def _key(k):
    return '|'.join(str(x) for x in k)


def cells_92g(js, m21, rows):
    '''js = JSON 92g, m21 = {khoá ô: [n_boot]} -> {(hàng, đặc trưng, đầu, tỉ lệ): ô} cho các hàng `rows` (đặc trưng chính, 6 phép).'''
    from frozen_probe import config as C
    out = {}
    for row in rows:
        for probe, f in C.FAMILY:
            k = (row, C.PRIMARY_FEATURE, probe, f)
            kk = _key(k)
            assert kk in js['o'] and kk in m21, f'92g thieu o {kk} -> DUNG'
            c = dict(js['o'][kk])
            c['m21'] = np.asarray(m21[kk], dtype=np.float64)
            assert len(c['m21']) == len(m21[kk]) and c['so_hat'] == len(c['tung_hat']), f'o {kk}: lan rut / so hat lech -> DUNG'
            out[k] = c
    return out


def cells_92b(preds, target, cols, rows, workers=1, log=None, n_boot=None):
    '''preds = {'hàng|đặc trưng|đầu|f<tỉ lệ>|lần rút': [m, 28] float16} của 92b -> {(hàng, đặc trưng, đầu, tỉ lệ): ô} (đặc trưng chính, 6 phép, hàng `rows`).'''
    from frozen_probe import config as C
    from frozen_probe import scoring as SC
    from vindr_bench.evaluation import compare as CP
    from vindr_bench.evaluation import panels as P
    inv = {v: k for k, v in C.FRACTION_CODE.items()}
    groups = {}
    for name, v in preds.items():
        r, fe, cfg, fc, dd = name.split('|')
        if r in rows and fe == C.PRIMARY_FEATURE and cfg in C.PROBES:
            groups.setdefault((r, fe, cfg, inv[fc]), {})[int(dd)] = np.asarray(v, dtype=np.float32)
    want = {(r, C.PRIMARY_FEATURE, p, f) for r in rows for p, f in C.FAMILY}
    assert set(groups) == want, f'du doan 92b thieu / thua o: {sorted(want ^ set(groups))[:4]} -> DUNG'
    for k, g in groups.items():
        assert sorted(g) == list(range(C.N_DRAWS[k[3]])), f'{k}: lan rut {sorted(g)} != {C.N_DRAWS[k[3]]} -> DUNG'
    n_boot = n_boot or P.N_BOOT
    jobs, keys = [], []
    for k in sorted(groups):
        for d in sorted(groups[k]):
            jobs.append((groups[k][d], target, cols, n_boot, P.BOOT_SEED, P.HALF_SEED))
            keys.append((k, d))
    stats = CP.many_stats(jobs, workers, log=log)
    by = {}
    for (k, d), st in zip(keys, stats):
        by.setdefault(k, []).append((d, st))
    return {k: SC.cell_from_stats([st for _, st in sorted(v, key=lambda x: x[0])]) for k, v in by.items()}


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


def check_92b(cells, res92b):
    '''Ô tính lại so với ô 92b (= các trường của probe_bb.repro_gate). -> list dict(o, dau, lech_max, so_so, dat_knn) — đầu k-NN phải lệch <= VINDR_TOL.'''
    fields = ('so_hat', 'diem', 'ci21', 'nua', 'tung_hat', 'ap21_lop')
    rows = []
    for k in sorted(cells):
        b = (res92b.get('o') or {}).get(_key(k))
        assert b is not None, f'92b thieu o {_key(k)} -> DUNG'
        fa, fb = _flat({t: cells[k][t] for t in fields}), _flat({t: b.get(t) for t in fields})
        assert set(fa) == set(fb), f'{_key(k)}: truong o KHAC 92b -> DUNG'

        def _d(x, y):
            x, y = float(x), float(y)
            return 0.0 if (x != x and y != y) else (float('inf') if (x != x) != (y != y) else abs(x - y))
        lech = [_d(fa[t], fb[t]) for t in fa if fa[t] is not None and fb[t] is not None]
        m = max(lech) if lech else float('inf')
        rows.append(dict(o=_key(k), dau=k[2], ti_le=k[3], lech_max=float(m), so_so=len(lech), dat=bool(k[2] != 'knn' or m <= CF.VINDR_TOL),
                         ap21=float(cells[k]['diem']['p21']['ap']), ap21_92b=float(b['diem']['p21']['ap'])))
    return rows
