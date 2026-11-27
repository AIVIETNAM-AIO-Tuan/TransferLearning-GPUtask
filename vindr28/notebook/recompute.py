# -*- coding: utf-8 -*-
"""Ô của ba notebook `94c_du_doan_<nguồn>` (Colab CPU; PLAN_94 §7b; Tuấn 27/11 "tính lại đi"): tính lại dự đoán test của phép kiểm ít nhãn `92` và LƯU
FLOAT32 — `92` chấm trên dự đoán float32 nhưng chỉ lưu bản float16, nên chấm lại từ tệp đã lưu lệch tới 0,0046 ở mAP21 (lượt float16 của 94b).

    nguồn  s42 (đặc trưng 92a, 8 hàng; đối chiếu JSON 92b) · bb (92d, 6 hàng; JSON 92e) · s43 (92f, 3 hàng; JSON 92g)

    C2 setup    khối Drive + phiên bản ba gói
    C3 load     nhãn + đặc trưng đã trích của nguồn (md5 = manifest; nhãn = 92a, cùng thứ tự ảnh) + JSON kết quả của 92 + tệp resume (hàng đã xong)
    C4 probes   ĐÚNG đầu đọc của 92 (frozen_probe.notebook.probe._probe; chuẩn hoá theo bể học; cùng chỉ số lần rút, md5 = 92b): đặc trưng chính x k-NN /
                logistic x 1 / 10 / 100 % x mọi lần rút nhãn -> dự đoán float32; CỔNG TỪNG HÀNG khi xong hàng: điểm 21 lớp (AP / AUPRC / AUROC, TB các lần rút)
                trùng JSON của 92 (> 1e-9 gắn cờ, > 1e-4 dừng) + logistic hội tụ; resume sau mỗi hàng
    C5 sync     tệp dự đoán float32 + JSON (cổng, hội tụ, thời gian) -> dong_bo()

Chỉ in SỐ TỔNG HỢP (luật PhysioNet)."""
import io
import json
import os
import time

import numpy as np

from .. import __version__
from .. import config as C
from . import common as U
from . import it_nhan as NI

TAG = dict(s42='92a', bb='92d', s43='92f')


def setup(ns, src, log=print):
    '''C2. -> CTX.'''
    import frozen_probe
    import vindr_bench
    import vindr28
    assert src in TAG, src
    d = U.drive_api(ns, C.RUN_TAGS[f'du_doan_{src}'])
    assert vindr_bench.__version__ == NI.VINDR_BENCH_VERSION and frozen_probe.__version__ == NI.FROZEN_PROBE_VERSION, 'phien ban goi KHAC ban ghim -> DUNG'
    goi = dict(vindr28=dict(phien_ban=__version__, sha=getattr(vindr28, 'V28_SHA', None)),
               vindr_bench=dict(phien_ban=vindr_bench.__version__, sha=getattr(vindr_bench, 'GOI_SHA', None)),
               frozen_probe=dict(phien_ban=frozen_probe.__version__, sha=getattr(frozen_probe, 'FP_SHA', None)))
    log(f"94c {src} | vindr28 {__version__} @ {(goi['vindr28']['sha'] or '?')[:12]} + vindr_bench {vindr_bench.__version__} @ "
        f"{(goi['vindr_bench']['sha'] or '?')[:12]} + frozen_probe {frozen_probe.__version__} @ {(goi['frozen_probe']['sha'] or '?')[:12]} | numpy {np.__version__} | "
        f"{os.cpu_count()} CPU")
    log(f'tinh lai du doan cua {dict(s42="92b", bb="92e", s43="92g")[src]} bang DUNG dau doc cua 92, luu float32; cong tung hang: diem 21 lop trung JSON cua 92 '
        f'(> {C.FLAG_92:g} gan co, > {C.STOP_92:g} dung)')
    return dict(drive=d, goi=goi, t=time.strftime('%Y%m%d_%H%M%S'), src=src)


def _resume_load(d):
    kk = d['kiem_ke']((C.FOLDER, d['RUN_TAG']), mau=U.re.escape(d['TEN_RESUME']))[1]
    if d['TEN_RESUME'] not in kk:
        return {}, {}
    with np.load(d['tai'](kk[d['TEN_RESUME']][0]), allow_pickle=False) as z:
        meta = json.loads(z['chi_muc'].tobytes().decode('utf-8'))
        preds = {tuple(_parse(k)): z[k].astype(np.float32) for k in z.files if k != 'chi_muc'}
    return preds, meta


def _key(row, feat, cfg, f, dd):
    from frozen_probe import config as FC
    return f'{row}|{feat}|{cfg}|{FC.FRACTION_CODE[f]}|{dd}'


def _parse(k):
    from frozen_probe import config as FC
    back = {v: kk for kk, v in FC.FRACTION_CODE.items()}
    row, feat, cfg, fc, dd = k.split('|')
    return row, feat, cfg, back[fc], int(dd)


def _pack(preds, meta):
    buf = io.BytesIO()
    np.savez(buf, chi_muc=np.frombuffer(json.dumps(U.jsonable(meta)).encode('utf-8'), dtype=np.uint8),
             **{_key(*k): np.asarray(v, dtype=np.float32) for k, v in sorted(preds.items(), key=lambda kv: str(kv[0]))})
    return buf.getvalue()


def load(ctx, log=print):
    '''C3. -> dict(nhan, dac_trung, json_92).'''
    from frozen_probe.notebook import probe_bb as PB
    d, src = ctx['drive'], ctx['src']
    kk = d['kiem_ke'](C.SRC_TRICH[src])[1]
    from frozen_probe import features as FT
    assert FT.MANIFEST_FILE in kk and FT.LABEL_FILE in kk, f'chua co ket qua {TAG[src]} trong {"/".join(C.SRC_TRICH[src])} -> DUNG'
    man = json.load(open(d['tai'](kk[FT.MANIFEST_FILE][0]), encoding='utf-8'))
    lab = PB._labels(d, kk, man, TAG[src])
    if src != 's42':
        ka = d['kiem_ke'](C.SRC_TRICH['s42'])[1]
        laba = PB._labels(d, ka, json.load(open(d['tai'](ka[FT.MANIFEST_FILE][0]), encoding='utf-8')), '92a')
        same = all(np.array_equal(lab[k], laba[k]) for k in ('nhan', 'tach')) and lab['ten_nhan'] == laba['ten_nhan'] and lab['lop_chinh'] == laba['lop_chinh']
        assert same, f'nhan / tach cua {TAG[src]} KHAC 92a (khong cung thu tu anh) -> DUNG'
    n = len(lab['nhan'])
    rows = NI._rows()[src]
    feats = {}
    for row in rows:
        f, m5 = PB._feature(d, kk, man, row, n, TAG[src])
        feats[row] = f
        log(f"  {NI.code(row):<20}{n:,} x {f['mean_patch'].shape[1]} | md5 {m5[:12]} = manifest {TAG[src]}")
    name, folder, prefix = next(x for x in C.SRC_92 if x[0] == src)
    t, ban = U.latest(d, folder, prefix, 'json')
    assert ban is not None, f'chua co JSON ket qua trong {"/".join(folder)} -> DUNG'
    js = json.load(open(d['tai'](ban), encoding='utf-8'))
    preds, meta = _resume_load(d)
    log(f"nhan {TAG[src]}: {n:,} anh x {lab['nhan'].shape[1]} nhan | JSON {'/'.join(folder)} @ {t} ({len(js.get('o', {}))} o, md5 lan rut {js['md5_lan_rut'][:12]}) | "
        f"resume: {len(meta.get('xong', []))} hang da xong")
    ctx.update(nhan=lab, dac_trung=feats, json92=js, json92_t=t, rows=rows, du_doan=preds, meta=dict(meta) or dict(xong=[], cong=[], hoi_tu=[], phut={}))
    return dict(nhan=lab, json92_t=t)


def probes(ctx, log=print):
    '''C4. -> {(hàng, đặc trưng, cấu hình, tỉ lệ, lần rút): dự đoán test [m, 28] float32}.'''
    from frozen_probe import config as FC
    from frozen_probe import subsets as SS
    from frozen_probe import probes as PR
    from frozen_probe.notebook import probe as NBP
    from vindr_bench.evaluation import panels as P
    d, lab, meta, preds = ctx['drive'], ctx['nhan'], ctx['meta'], ctx['du_doan']
    Y = lab['nhan'].astype(np.float64)
    pool = np.flatnonzero(np.isin(lab['tach'], FC.POOL_SPLITS))
    test = np.flatnonzero(lab['tach'] == FC.TEST_SPLIT)
    target = lab['nhan'][test]
    cols = P.panel_columns(lab['ten_nhan'], lab['lop_chinh'], target)
    dr = SS.all_draws(pool)
    m5 = SS.md5(dr)
    assert m5 == ctx['json92']['md5_lan_rut'], f"md5 lan rut {m5[:12]} KHAC JSON cua 92 ({ctx['json92']['md5_lan_rut'][:12]}) -> DUNG"
    feat = FC.PRIMARY_FEATURE
    log(f"be hoc {len(pool):,} anh | test {len(test):,} | lan rut md5 {m5[:12]} = 92 | {len(ctx['rows'])} hang x {len(FC.PROBES)} dau x "
        f"{sum(FC.N_DRAWS.values())} lan rut = {len(ctx['rows']) * len(FC.PROBES) * sum(FC.N_DRAWS.values())} bo")
    t0 = time.time()
    ref = ctx['json92'].get('o', {})
    for row in ctx['rows']:
        if row in meta['xong']:
            log(f'  {NI.code(row):<20}da xong o lan truoc (resume) -> bo qua')
            continue
        t1 = time.time()
        X = ctx['dac_trung'][row][feat].astype(np.float64)
        std = PR.standardizer(X[pool])
        Xte = X[test]
        conv = []
        for cfg in FC.PROBES:
            for f in FC.FRACTIONS:
                for dd in range(FC.N_DRAWS[f]):
                    idx = dr[(f, dd)]
                    pv, info = NBP._probe(cfg, f, X[idx], Y[idx], Xte, std)
                    preds[(row, feat, cfg, f, dd)] = np.asarray(pv, dtype=np.float32)
                    if cfg.startswith('logistic'):
                        conv.append(dict(hang=row, cau_hinh=cfg, ti_le=f, lan_rut=dd, so_vong=info['so_vong'], grad_max=info['grad_max'], hoi_tu=bool(info['hoi_tu'])))
        bad = [c for c in conv if not c['hoi_tu']]
        assert not bad, f'{row}: logistic KHONG hoi tu o {bad[:3]} -> DUNG, gui Claude dong tren'
        worst, rec = 0.0, []
        for cfg in FC.PROBES:
            for f in FC.FRACTIONS:
                pts = [P.point(preds[(row, feat, cfg, f, dd)], target, cols['p21']) for dd in range(FC.N_DRAWS[f])]
                mine = {m: float(np.mean([q[m] for q in pts])) for m in ('ap', 'auprc_thang', 'auroc')}
                r = ref.get('|'.join(str(x) for x in (row, feat, cfg, f)))
                assert r is not None, f'{row} {cfg} {f}: JSON cua 92 KHONG co o nay -> DUNG'
                lech = max(abs(mine[m] - r['diem']['p21'][m]) for m in mine)
                worst = max(worst, lech)
                rec.append(dict(hang=row, cau_hinh=cfg, ti_le=f, lech=lech, so_lan_rut=len(pts), so_lan_rut_92=r['so_hat']))
        assert all(x['so_lan_rut'] == x['so_lan_rut_92'] for x in rec), f'{row}: so lan rut KHAC 92 -> DUNG'
        assert worst <= C.STOP_92, f'{row}: diem 21 lop lech {worst:.2e} > {C.STOP_92:g} so voi JSON cua 92 -> du doan tinh lai KHAC 92. DUNG, gui Claude dong tren'
        meta['xong'].append(row)
        meta['cong'] += rec
        meta['hoi_tu'] += conv
        meta['phut'][row] = (time.time() - t1) / 60
        U.save_resume_bytes(d, _pack(preds, meta))
        log(f"  {NI.code(row):<20}{sum(1 for k in preds if k[0] == row)} bo | {(time.time() - t1) / 60:.1f} phut | logistic hoi tu {len(conv)} / {len(conv)} | cong: diem 21 "
            f"lop lech lon nhat {worst:.1e} -> {'TRUNG' if worst <= C.FLAG_92 else 'GAN CO (lech nho)'} | tong {(time.time() - t0) / 60:.1f} phut")
    w = max((x['lech'] for x in meta['cong']), default=float('nan'))
    log(f"xong {len(meta['xong'])} / {len(ctx['rows'])} hang | {len(preds)} bo du doan float32 | cong toan bo: lech lon nhat {w:.1e} | "
        f"{sum(x['lech'] > C.FLAG_92 for x in meta['cong'])} o > {C.FLAG_92:g}")
    return preds


def sync(ctx, log=print):
    '''C5. Tệp dự đoán float32 + JSON -> dong_bo().'''
    d, t, src = ctx['drive'], ctx['t'], ctx['src']
    assert len(ctx['meta']['xong']) == len(ctx['rows']), f"moi xong {ctx['meta']['xong']} / {list(ctx['rows'])} -> chay lai C4"
    names = [f'{C.PRED_FILE}_{src}_{t}.npz', f'tinh_lai_{src}_{t}.json']
    data = _pack(ctx['du_doan'], ctx['meta'])

    def w(q):
        with open(q, 'wb') as fh:
            fh.write(data)
    d['ghi'](names[0], w)
    obj = U.jsonable(dict(run=d['RUN_TAG'], t=t, goi=ctx['goi'], nguon=src, json_92=ctx['json92_t'], hang=list(ctx['rows']), so_bo=len(ctx['du_doan']),
                          cong=ctx['meta']['cong'], phut=ctx['meta']['phut'], hoi_tu=dict(so_fit=len(ctx['meta']['hoi_tu']),
                          so_vong_lon_nhat=max((c['so_vong'] for c in ctx['meta']['hoi_tu']), default=None),
                          grad_lon_nhat=max((c['grad_max'] for c in ctx['meta']['hoi_tu']), default=None))))
    d['ghi'](names[1], lambda q: d['ghi_json'](q, obj))
    du = d['dong_bo']()
    co = set(names) <= set(d['ten_tren_drive']())
    log(f"C5: da ghi {names} ({len(data) / 2 ** 20:.0f} MB) | dong_bo {'DU' if du else 'CHUA DU'} | tren Drive: {'CO du' if co else 'THIEU'}"
        + ('' if du and co else ' | KHONG tat runtime; chay lai C5'))
    return dict(dong_bo=du, du_tep=co, tep=names)
