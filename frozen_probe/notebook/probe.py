# -*- coding: utf-8 -*-
"""Ô của notebook `92b_it_nhan_vindr` (CPU; PLAN_92 §11). Đọc tệp đặc trưng của 92a trên Drive; KHÔNG GPU, không học backbone.

    C2 setup    khối Drive + phiên bản hai gói + thư viện
    C3 load     manifest + tệp nhãn (vân tay = các lượt 89) + 8 tệp đặc trưng (md5 = manifest; hữu hạn; đúng hình); thiếu hàng mô tả -> liệt kê;
                thiếu cặp chính -> dừng
    C4 probes   lần rút (md5 in ra; CÙNG chỉ số mọi hàng) -> k-NN + logistic ở 1 / 10 / 100 % -> dự đoán test; CỔNG: logistic hội tụ mọi lần fit;
                đối chứng âm (nhãn học bị xáo) về mức may rủi
    C5 score    bootstrap theo ảnh (bộ chính) -> ô = TB các lần rút -> Bảng 1 (tám hàng) · Bảng 2 (LUẬT: họ 6 phép (3) − (0), max-T) · Bảng 3 (mô tả) ·
                Bảng 4 (nhạy)
    C6 figures  hình họ 6 phép (khoảng đồng thời), lưới mAP21 tám hàng, hiệu (3) − X
    C7 sync     JSON + CSV + PNG + dự đoán (float16) -> dong_bo()

Chỉ in SỐ TỔNG HỢP (luật PhysioNet)."""
import csv
import io
import json
import os
import time

import numpy as np

from .. import __version__
from .. import config as C
from .. import features as FT
from .. import probes as PR
from .. import scoring as SC
from .. import subsets as S

DRIVE_NAMES = ('kiem_ke', 'tai', 'ghi', 'ghi_json', 'dong_bo', 'ten_tren_drive', 'GHI_LOC', 'RUN_TAG')


def drive_api(ns):
    missing = [k for k in DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    return {k: ns[k] for k in DRIVE_NAMES}


def setup(ns, log=print):
    '''C2. -> CTX.'''
    import scipy
    import vindr_bench
    import frozen_probe
    drive = drive_api(ns)
    assert drive['RUN_TAG'] == C.RUN_TAGS['it_nhan'], f"RUN_TAG {drive['RUN_TAG']!r} != {C.RUN_TAGS['it_nhan']!r} -> DUNG"
    pk = dict(frozen_probe=dict(phien_ban=__version__, sha=getattr(frozen_probe, 'FP_SHA', None)),
              vindr_bench=dict(phien_ban=vindr_bench.__version__, sha=getattr(vindr_bench, 'GOI_SHA', None)))
    log(f"92b | frozen_probe {__version__} @ {(pk['frozen_probe']['sha'] or '?')[:12]} + vindr_bench {pk['vindr_bench']['phien_ban']} @ "
        f"{(pk['vindr_bench']['sha'] or '?')[:12]} | numpy {np.__version__} | scipy {scipy.__version__} | {os.cpu_count()} CPU")
    log(f"luat dat truoc (PLAN_92 11.3): ho {len(C.FAMILY)} phep {C.CODE[C.REF]} − {C.CODE[C.CONTROL]} x {{k-NN, logistic}} x {{1 %, 10 %, 100 %}} tren mAP21 (AP bac thang "
        f"macro 21 lop), dac trung {C.PRIMARY_FEATURE}; khoang dong thoi max-T {C.LEVEL:g} %; 'hon' = hieu >= +{C.MARGIN:g} VA can duoi dong thoi > 0 VA duong o "
        f"hai nua test; 'kem' doi xung; doc THEO TUNG O, khong cau chung; ket luan duong can seed 43")
    return dict(drive=drive, goi=pk, t=time.strftime('%Y%m%d_%H%M%S'))


def load(ctx, log=print):
    '''C3. -> dict(nhan, dac_trung {hàng: {đặc trưng: [N, D] float32}}).'''
    from vindr_bench.data import vindr as V
    from vindr_bench.protocols.ours import gates as G
    d = ctx['drive']
    src = (C.FOLDER, C.RUN_TAGS['trich'])
    kk = d['kiem_ke'](src)[1]
    assert FT.MANIFEST_FILE in kk and FT.LABEL_FILE in kk, f'chua co ket qua 92a trong {"/".join(src)} -> chay 92a truoc. DUNG'
    man = json.load(open(d['tai'](kk[FT.MANIFEST_FILE][0]), encoding='utf-8'))
    p = d['tai'](kk[FT.LABEL_FILE][0])
    assert FT.md5_file(p) == man['tep_nhan']['md5'], 'tep nhan: md5 KHAC manifest -> DUNG'
    lab = FT.load_labels(p)
    same, diff = G.data_gate(lab['van_tay'])
    n = len(lab['nhan'])
    re_md5 = {s: V.md5_labels(lab['nhan'][lab['tach'] == s]) for s in V.SPLITS}
    ok_lab = all(re_md5[s] == lab['van_tay'][s]['md5_nhan'] for s in V.SPLITS)
    log(f"nhan: {n:,} anh x {lab['nhan'].shape[1]} nhan, {len(lab['lop_chinh'])} lop so chinh | tach " + ' / '.join(f"{s} {int((lab['tach'] == s).sum()):,}" for s in V.SPLITS)
        + f" | van tay = cac luot 89: {'TRUNG' if same else f'KHAC {diff}'} | md5 nhan tinh lai tu tep = van tay: {'DAT' if ok_lab else 'TRUOT'}")
    assert same and ok_lab, 'du lieu KHAC cac luot 89 -> DUNG'
    feats, thieu = {}, []
    for row in C.ROWS:
        name = FT.feature_file(row)
        if name not in kk or row not in man['hang']:
            thieu.append(row)
            continue
        p = d['tai'](kk[name][0])
        m5 = FT.md5_file(p)
        assert m5 == man['hang'][row]['md5'], f'{name}: md5 {m5[:12]} KHAC manifest -> DUNG'
        f = FT.load_features(p, row)
        for k, X in f.items():
            assert X.shape == (n, int(man['hang'][row]['chieu'])) and np.isfinite(X).all(), f'{name} / {k}: hinh {X.shape} / khong huu han -> DUNG'
        feats[row] = f
        log(f"  {C.CODE[row]:<9}{C.TEXT[row]}: {n:,} x {X.shape[1]} | md5 {m5[:12]} = manifest | anh vao {man['hang'][row]['anh']} | "
            f"nap: {man['hang'][row]['nap'][:60]}")
    assert all(r in feats for r in C.PAIR), f'thieu cap chinh {[r for r in C.PAIR if r not in feats]} -> DUNG'
    if thieu:
        log(f'  ! chua co tep dac trung cua: {thieu} (hang mo ta) -> bang bo trong cac hang nay')
    ctx.update(manifest=man, nhan=lab, dac_trung=feats, thieu=thieu)
    return dict(nhan=lab, dac_trung=feats)


def _probe(cfg, f, Xtr, Ytr, Xte, std=None):
    '''Một đầu đọc một lần rút -> (dự đoán test [m, L], bản ghi hội tụ | None).'''
    if cfg.startswith('knn'):
        k = C.SENSITIVITY['knn_k'] if cfg != 'knn' else C.knn_k(len(Xtr))
        return PR.knn_scores(Xtr, Ytr, Xte, k), dict(k=k)
    c_reg = C.LOGISTIC_C if cfg == 'logistic' else float(cfg.split('_C')[1])
    mu, sd = std
    m = PR.fit_logistic((Xtr - mu) / sd, Ytr, c_reg)
    return PR.logistic_scores(m, (Xte - mu) / sd), dict(C=c_reg, so_vong=m['so_vong'], grad_max=m['grad_max'], hoi_tu=m['hoi_tu'],
                                                       so_lop_hoc=int(len(m['lop_hoc'])), thong_bao=m['thong_bao'])


def probes(ctx, log=print):
    '''C4. -> {(hàng, đặc trưng, cấu hình, tỉ lệ, lần rút): dự đoán test [m, 28] float32}.'''
    lab, feats = ctx['nhan'], ctx['dac_trung']
    Y = lab['nhan'].astype(np.float64)
    pool = np.flatnonzero(np.isin(lab['tach'], C.POOL_SPLITS))
    test = np.flatnonzero(lab['tach'] == C.TEST_SPLIT)
    dr = S.all_draws(pool)
    log(f"be hoc {len(pool):,} anh ({'+'.join(C.POOL_SPLITS)}) | test {len(test):,} | lan rut: " + ', '.join(
        f"{C.FRACTION_TEXT[f]} = {S.size(len(pool), f):,} anh x {C.N_DRAWS[f]} lan (k-NN k = {C.knn_k(S.size(len(pool), f))})" for f in C.FRACTIONS)
        + f" | md5 chi so {S.md5(dr)[:12]} (CUNG cho moi hang) | hat {C.DRAW_SEED}")
    preds, conv, t0 = {}, [], time.time()
    for row in [r for r in C.ROWS if r in feats]:
        t1 = time.time()
        for feat in sorted({fe for f in C.FRACTIONS for fe, _ in SC.configs_for(row, f)}):
            X = feats[row][feat].astype(np.float64)
            std = PR.standardizer(X[pool])
            Xte = X[test]
            for f in C.FRACTIONS:
                for fe, cfg in SC.configs_for(row, f):
                    if fe != feat:
                        continue
                    for d in range(C.N_DRAWS[f]):
                        idx = dr[(f, d)]
                        pv, info = _probe(cfg, f, X[idx], Y[idx], Xte, std)
                        preds[(row, feat, cfg, f, d)] = pv
                        if cfg.startswith('logistic'):
                            conv.append(dict(hang=row, dac_trung=feat, cau_hinh=cfg, ti_le=f, lan_rut=d, **info))
        log(f"  {C.CODE[row]:<9}xong {sum(1 for k in preds if k[0] == row)} bo du doan | {(time.time() - t1) / 60:.1f} phut")
    bad = [c for c in conv if not c['hoi_tu']]
    log(f"logistic: {len(conv)} lan fit | so vong lon nhat {max(c['so_vong'] for c in conv)} | |grad| lon nhat {max(c['grad_max'] for c in conv):.1e} | "
        f"hoi tu {'DAT' if not bad else f'TRUOT {len(bad)} lan'} | tong {(time.time() - t0) / 60:.1f} phut")
    assert not bad, f'logistic KHONG hoi tu o {[(b["hang"], b["cau_hinh"], b["ti_le"], b["lan_rut"]) for b in bad][:5]} -> DUNG, gui Claude dong tren'
    ctx.update(du_doan=preds, hoi_tu=conv, ve=dict(be=pool, test=test, md5_lan_rut=S.md5(dr)))
    ctx['doi_chung_am'] = negative_control(ctx, log)
    return preds


def negative_control(ctx, log=print):
    '''Nhãn học bị XÁO (hoán vị hàng trong bể, hạt cố định), 100 %, cặp chính, đặc trưng chính: mAP21 test phải về mức may rủi (bắt rò nhãn test).'''
    from vindr_bench.evaluation import panels as P
    lab, feats, ve = ctx['nhan'], ctx['dac_trung'], ctx['ve']
    Y = lab['nhan'].astype(np.float64)
    pool, test = ve['be'], ve['test']
    target = lab['nhan'][test]
    cols = P.panel_columns(lab['ten_nhan'], lab['lop_chinh'], target)
    chance = SC.chance_level(target, cols)
    Ysh = Y[pool][np.random.default_rng(C.NEG_SEED).permutation(len(pool))]
    out = []
    for row in C.PAIR:
        X = feats[row][C.PRIMARY_FEATURE].astype(np.float64)
        std = PR.standardizer(X[pool])
        for cfg in C.PROBES:
            pv, _ = _probe(cfg, 1.0, X[pool], Ysh, X[test], std)
            real = P.point(ctx['du_doan'][(row, C.PRIMARY_FEATURE, cfg, 1.0, 0)], target, cols['p21'])['ap']
            sh = P.point(pv, target, cols['p21'])['ap']
            out.append(dict(hang=row, dau=cfg, map21_xao=sh, map21_that=real, may_rui=chance, dat=bool(sh <= chance + C.NEG_MAX_ABOVE_CHANCE)))
    ok = all(o['dat'] for o in out)
    log(f"doi chung am (nhan hoc bi xao, 100 %, cap chinh): muc may rui mAP21 {chance:.3f} | " + ' | '.join(
        f"{C.CODE[o['hang']]} {C.PROBE_TEXT[o['dau']]} xao {o['map21_xao']:.3f} (that {o['map21_that']:.3f})" for o in out)
        + f" | nguong <= may rui + {C.NEG_MAX_ABOVE_CHANCE:g}: {'DAT' if ok else 'TRUOT'}")
    assert ok, 'doi chung am TRUOT (nhan xao van cho mAP cao -> nghi ro nhan) -> DUNG, gui Claude dong tren'
    return out


def score(ctx, log=print, n_boot=None, workers=None):
    '''C5. Bootstrap (bộ chính) + ô + họ 6 phép + mô tả + nhạy; in bốn bảng. -> dict kết quả.'''
    from vindr_bench.evaluation import compare as CP
    from vindr_bench.evaluation import panels as P
    lab, preds, ve = ctx['nhan'], ctx['du_doan'], ctx['ve']
    target = lab['nhan'][ve['test']]
    cols = P.panel_columns(lab['ten_nhan'], lab['lop_chinh'], target)
    n_boot = n_boot or P.N_BOOT
    groups = {}
    for (row, feat, cfg, f, d), pv in preds.items():
        groups.setdefault((row, feat, cfg, f), {})[d] = pv
    prim = sorted(k for k in groups if SC.is_primary(k[1], k[2]))
    jobs, keys = [], []
    for k in prim:
        for d in sorted(groups[k]):
            jobs.append((groups[k][d], target, cols, n_boot, P.BOOT_SEED, P.HALF_SEED))
            keys.append((k, d))
    t0 = time.time()
    workers = workers or max(1, os.cpu_count() or 1)
    log(f"bootstrap {n_boot} lan theo anh (hat {P.BOOT_SEED}; hai nua hat {P.HALF_SEED}) cho {len(jobs)} bo du doan CHINH ({workers} tien trinh); bo nhay chi diem")
    stats = CP.many_stats(jobs, workers, log=log)
    by = {}
    for (k, d), st in zip(keys, stats):
        by.setdefault(k, []).append((d, st))
    cells = {k: SC.cell_from_stats([st for _, st in sorted(v, key=lambda x: x[0])]) for k, v in by.items()}
    for k in groups:
        if k not in cells:
            cells[k] = SC.cell_point([groups[k][d] for d in sorted(groups[k])], target, cols)
    fam, desc, sens = SC.family(cells), SC.descriptive(cells), SC.sensitivity(cells)
    ctx.update(o=cells, cot=cols, ho=fam, mo_ta=desc, nhay=sens, phut_cham=(time.time() - t0) / 60)
    _print_tables(ctx, log)
    return dict(ho=fam, mo_ta=desc, nhay=sens)


def _print_tables(ctx, log):
    cells, fam, desc, sens = ctx['o'], ctx['ho'], ctx['mo_ta'], ctx['nhay']
    n_pool = len(ctx['ve']['be'])
    log(f"\n=== Bang 1 · TAM HANG, dac trung {C.PRIMARY_FEATURE} ({C.FEATURE_TEXT[C.PRIMARY_FEATURE]}), test VinDr {len(ctx['ve']['test']):,} anh; moi o = TB cac lan "
        f"rut; [..] = CI95 danh nghia theo anh; SD = giua lan rut ===")
    log(f"{'hang':<10}{'dau doc':<11}{'nhan':>7}{'lan':>4}  {'mAP21 [CI95]':<24}{'SD':>7}{'AUPRC21':>9}{'AUROC21':>9}  hai nua AP21")
    for row in [r for r in C.ROWS if r in ctx['dac_trung']]:
        log(f'  {C.TEXT[row]}')
        for probe in C.PROBES:
            for f in C.FRACTIONS:
                c = cells.get((row, C.PRIMARY_FEATURE, probe, f))
                if c is None:
                    continue
                dd = c['diem']['p21']
                sd = (c.get('sd') or {}).get('p21', {}).get('ap')
                log(f"{C.CODE[row]:<10}{C.PROBE_TEXT[probe]:<11}{C.FRACTION_TEXT[f]:>7}{c['so_hat']:>4}  {SC.fmt(dd['ap'], 4)} [{SC.fmt(c['ci21'][0])}, "
                    f"{SC.fmt(c['ci21'][1])}]{'':<3}{SC.fmt(sd, 4):>7}{SC.fmt(dd['auprc_thang'], 4):>9}{SC.fmt(dd['auroc'], 4):>9}  {SC.fmt(c['nua'][0])} / "
                    f"{SC.fmt(c['nua'][1])}")
    log(f"\n=== Bang 2 · LUAT (PLAN_92 11.3): HO {fam['so_phep']} PHEP {C.CODE[C.REF]} − {C.CODE[C.CONTROL]} | {fam['co']} / {fam['so_phep']} phep co du lieu"
        + (f" -> khoang DONG THOI max-T {C.LEVEL:g} %, q = {fam['q']:.2f}" if fam['day_du'] else ' -> HO CHUA DU: chi hieu DANH NGHIA, CHUA ap luat') + ' ===')
    log(f"    moi o ba so: AP bac thang (so cua luat) / AUPRC hinh thang / AUROC, macro 21 lop; be hoc {n_pool:,} anh; hai hang dung CUNG chi so lan rut")
    log(f"{'dau doc':<11}{'nhan':>7}  {'Δ AP / AUPRC / AUROC':<26}{'dong thoi Δ AP21':<19}{'danh nghia':<19}{'hai nua':<17}{'lan rut duong':<15}ket luan")
    for p in fam['phep_so']:
        ci_s = p['ci_dong_thoi']
        log(f"{C.PROBE_TEXT[p['dau']]:<11}{C.FRACTION_TEXT[p['ti_le']]:>7}  {SC.t3(p['d']):<26}"
            + (f"[{SC.sgn(ci_s[0])}, {SC.sgn(ci_s[1])}]  " if ci_s else f"{'-':<19}") + f"[{SC.sgn(p['ci_danh_nghia'][0])}, {SC.sgn(p['ci_danh_nghia'][1])}]  "
            f"{SC.sgn(p['nua'][0])} / {SC.sgn(p['nua'][1])}  {p['so_lan_rut_duong']} / {len(p['tung_lan_rut'])}{'':<9}{p['ket_luan']}")
    log('    Δ tung lan rut (AP21): ' + ' | '.join(f"{C.PROBE_TEXT[p['dau']]} {C.FRACTION_TEXT[p['ti_le']]}: {' '.join(SC.sgn(v) for v in p['tung_lan_rut'])}"
                                                for p in fam['phep_so'] if len(p['tung_lan_rut']) > 1))
    log('    doc THEO TUNG O (dau doc x ti le nhan); KHONG cau chung. Mot seed backbone: ket luan duong can seed 43 (ca hai hang).')
    log(f"\n=== Bang 3 · MO TA (CI95 danh nghia; KHONG thuoc ho 6 phep), dac trung {C.PRIMARY_FEATURE} ===")
    log(f"{'hieu':<20}{'dau doc':<11}{'nhan':>7}  {'Δ AP / AUPRC / AUROC':<26}{'CI95 Δ AP21':<19}hai nua")
    for h in desc['hieu']:
        log(f"{C.CODE[h['hang_a']] + ' − ' + C.CODE[h['hang_b']]:<20}{C.PROBE_TEXT[h['dau']]:<11}{C.FRACTION_TEXT[h['ti_le']]:>7}  {SC.t3(h['d']):<26}"
            f"[{SC.sgn(h['ci_danh_nghia'][0])}, {SC.sgn(h['ci_danh_nghia'][1])}]  {SC.sgn(h['nua'][0])} / {SC.sgn(h['nua'][1])}")
    for t in desc['xu_huong']:
        log(f"xu huong {C.PROBE_TEXT[t['dau']]}: Δ({C.FRACTION_TEXT[min(C.FRACTIONS)]}) − Δ({C.FRACTION_TEXT[max(C.FRACTIONS)]}) cua {C.CODE[C.REF]} − "
            f"{C.CODE[C.CONTROL]} = {SC.sgn(t['d'])} [{SC.sgn(t['ci_danh_nghia'][0])}, {SC.sgn(t['ci_danh_nghia'][1])}]")
    log(f"\n=== Bang 4 · NHAY (chi diem; cap chinh {C.CODE[C.REF]} − {C.CODE[C.CONTROL]}): cau hinh / dac trung phu canh cau hinh chinh cung ti le ===")
    for s in sens:
        log(f"{SC.CFG_TEXT[s['cau_hinh']] + ' | ' + s['dac_trung']:<46}{C.FRACTION_TEXT[s['ti_le']]:>7}  Δ {SC.t3(s['d'])}  | cau hinh chinh "
            + (SC.t3(s['d_chinh']) if s['d_chinh'] else '-'))


def figures(ctx, log=print, show=True):
    '''C6. -> {tên: Figure}. Hình dùng hàm của vindr_bench.evaluation.figures (bảng màu đã kiểm).'''
    from vindr_bench.evaluation import figures as FG
    cells, fam = ctx['o'], ctx['ho']
    rows = [r for r in C.ROWS if r in ctx['dac_trung']]
    base = (f"VinDr test {len(ctx['ve']['test']):,} ảnh, 21 lớp chính; đặc trưng đông cứng = trung bình token patch tầng cuối; mỗi ô trung bình các lần rút "
            f"(1 % ×{C.N_DRAWS[0.01]}, 10 % ×{C.N_DRAWS[0.10]}, 100 % ×1; cùng chỉ số mọi hàng). k-NN: k = round(√n); logistic: L2, C = 1, giải tới hội tụ; không "
            f"chọn siêu tham số theo dữ liệu chấm. Một seed backbone (s42).")
    figs = {}
    fi = {}
    for p in fam['phep_so']:
        lo, hi = p['ci_dong_thoi'] if p['ci_dong_thoi'] else p['ci_danh_nghia']
        fi[(C.FRACTION_CODE[p['ti_le']], p['dau'])] = dict(d=p['d']['ap'], lo=lo, hi=hi, dat=p['ket_luan'] in ('hon', 'kem'))
    f = FG.family_ci(fi, [C.FRACTION_CODE[f] for f in C.FRACTIONS], list(C.PROBES), {C.FRACTION_CODE[f]: f'{C.FRACTION_TEXT[f]} nhãn' for f in C.FRACTIONS},
                     C.PROBE_TEXT, f'{C.CODE[C.REF]} − {C.CODE[C.CONTROL]} trên AP 21 lớp (bậc thang): họ {len(C.FAMILY)} phép của luật — mỗi ô một đầu đọc, mỗi hàng một '
                     f'tỉ lệ nhãn', f'{C.CODE[C.REF]} − {C.CODE[C.CONTROL]} · AP 21',
                     base + (f" Vạch = khoảng ĐỒNG THỜI max-T 95 % của họ (q = {fam['q']:.2f}".replace('.', ',') + '); chấm đặc = đạt dạng luật ("hơn" / "kém": '
                             '|hiệu| ≥ 0,01, khoảng đồng thời không chứa 0, cùng dấu ở hai nửa test).' if fam['day_du'] else
                             ' Họ CHƯA đủ: vạch = CI95 danh nghĩa, chưa áp luật.') + ' Đường đứt: ±0,01. Đọc theo từng ô; kết luận dương cần seed 43.',
                     margin=C.MARGIN, labels=('đạt dạng luật ("hơn" / "kém")', 'không phân biệt'))
    if f is not None:
        figs['ho_6'] = f
    cond = ('ti_le_nhan', C.PRIMARY_FEATURE)
    heads = [f'{p}|{C.FRACTION_CODE[fr]}' for p in C.PROBES for fr in C.FRACTIONS]
    flat = {(cond[0], cond[1], r, f'{p}|{C.FRACTION_CODE[fr]}'): cells[(r, C.PRIMARY_FEATURE, p, fr)]
            for r in rows for p in C.PROBES for fr in C.FRACTIONS if (r, C.PRIMARY_FEATURE, p, fr) in cells}
    ht = {f'{p}|{C.FRACTION_CODE[fr]}': f'{C.PROBE_TEXT[p]} {C.FRACTION_TEXT[fr]}' for p in C.PROBES for fr in C.FRACTIONS}
    et = {r: f'{C.CODE[r]}' for r in rows}
    figs['luoi_21_ap'] = FG.heat(flat, rows, heads, [cond], 'ap', et, ht, {cond: 'mAP 21 lớp theo tỉ lệ nhãn'},
                                 base + ' Hàng: ' + '; '.join(C.TEXT[r] for r in rows) + '. Điểm, không khoảng tin cậy (CI ở Bảng 1).', panel='p21')
    f = FG.ref_diff(flat, C.REF, rows, heads, cond, FG.METRICS, et, ht, {cond: 'theo tỉ lệ nhãn'},
                    base + ' Hiệu điểm MÔ TẢ — luật + khoảng đồng thời ở Bảng 2 (chỉ cặp chính).', panel='p21')
    if f is not None:
        figs['hieu_21'] = f
    ctx['hinh'] = figs
    log(f'{len(figs)} hinh: {sorted(figs)}')
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return figs


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda v: v.item() if hasattr(v, 'item') else (v.tolist() if hasattr(v, 'tolist') else str(v))))


def _rows_csv(ctx):
    out = []
    for (row, feat, cfg, f), c in sorted(ctx['o'].items(), key=lambda kv: (C.ROWS.index(kv[0][0]), kv[0][1], kv[0][2], kv[0][3])):
        dd = c['diem']['p21']
        out.append(dict(bang='1_tuyet_doi', hang=row, dieu_kien=C.TEXT[row], dac_trung=feat, dau_doc=cfg, ti_le_nhan=f, so_lan_rut=c['so_hat'], chieu_phep_tru='',
                        AP21=dd['ap'], AUPRC21_hinh_thang=dd['auprc_thang'], AUROC21=dd['auroc'], ci_thap=(c.get('ci21') or [None])[0],
                        ci_cao=(c.get('ci21') or [None, None])[1], chi_diem=bool(c.get('chi_diem')), ket_luan=''))
    for p in ctx['ho']['phep_so']:
        out.append(dict(bang='2_ho_6_luat', hang=f'{C.REF} - {C.CONTROL}', dieu_kien=f'{C.TEXT[C.REF]} TRU {C.TEXT[C.CONTROL]}', dac_trung=C.PRIMARY_FEATURE,
                        dau_doc=p['dau'], ti_le_nhan=p['ti_le'], so_lan_rut=len(p['tung_lan_rut']), chieu_phep_tru=f'{C.CODE[C.REF]} - {C.CODE[C.CONTROL]}',
                        AP21=p['d']['ap'], AUPRC21_hinh_thang=p['d']['auprc_thang'], AUROC21=p['d']['auroc'],
                        ci_thap=(p['ci_dong_thoi'] or [None])[0], ci_cao=(p['ci_dong_thoi'] or [None, None])[1], chi_diem=False, ket_luan=p['ket_luan']))
    for h in ctx['mo_ta']['hieu']:
        out.append(dict(bang='3_mo_ta', hang=f"{h['hang_a']} - {h['hang_b']}", dieu_kien=f"{C.TEXT[h['hang_a']]} TRU {C.TEXT[h['hang_b']]}", dac_trung=C.PRIMARY_FEATURE,
                        dau_doc=h['dau'], ti_le_nhan=h['ti_le'], so_lan_rut='', chieu_phep_tru=f"{C.CODE[h['hang_a']]} - {C.CODE[h['hang_b']]}", AP21=h['d']['ap'],
                        AUPRC21_hinh_thang=h['d']['auprc_thang'], AUROC21=h['d']['auroc'], ci_thap=h['ci_danh_nghia'][0], ci_cao=h['ci_danh_nghia'][1],
                        chi_diem=False, ket_luan='mo ta; CI danh nghia'))
    return out


def sync(ctx, log=print):
    '''C7. Ghi JSON + CSV + PNG + dự đoán (float16) rồi dong_bo().'''
    d, t = ctx['drive'], ctx['t']
    names = []
    cells = {'|'.join(str(x) for x in k): {kk: v for kk, v in c.items() if kk not in ('m21', 'm7')} for k, c in ctx['o'].items()}
    obj = _jsonable(dict(run=C.RUN_TAGS['it_nhan'], nguon=C.RUN_TAGS['trich'], t=t, goi=ctx['goi'], manifest_92a=dict(t=ctx['manifest'].get('t'),
                         goi=ctx['manifest'].get('goi')), hang={r: C.TEXT[r] for r in C.ROWS}, thieu=ctx['thieu'], be_hoc=len(ctx['ve']['be']),
                         test=len(ctx['ve']['test']), md5_lan_rut=ctx['ve']['md5_lan_rut'], ti_le=list(C.FRACTIONS), so_lan_rut={str(k): v for k, v in C.N_DRAWS.items()},
                         o=cells, ho=ctx['ho'], mo_ta=ctx['mo_ta'], nhay=ctx['nhay'], doi_chung_am=ctx['doi_chung_am'], hoi_tu=ctx['hoi_tu'],
                         ghi_chu='PLAN_92 11: luat = ho 6 phep (3) - (0), mAP21 (AP bac thang macro 21 lop), max-T 95 %; doc theo tung o; bao kem / nhay = mo ta'))
    names.append(f'it_nhan_vindr_{t}.json')
    d['ghi'](names[-1], lambda q: d['ghi_json'](q, obj))
    rows = _rows_csv(ctx)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    text = buf.getvalue()
    names.append(f'it_nhan_vindr_bang_{t}.csv')

    def _csv(q):
        with open(q, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(text)
    d['ghi'](names[-1], _csv)
    for name, fig in ctx.get('hinh', {}).items():
        names.append(f'it_nhan_vindr_hinh_{name}_{t}.png')
        d['ghi'](names[-1], lambda q, fig=fig: fig.savefig(q, dpi=160, bbox_inches='tight'))
    pv = {f"{r}|{fe}|{cfg}|{C.FRACTION_CODE[f]}|{dd}": v.astype(np.float16) for (r, fe, cfg, f, dd), v in ctx['du_doan'].items()}
    names.append(f'it_nhan_vindr_du_doan_{t}.npz')
    d['ghi'](names[-1], lambda q: np.savez(q, **pv))
    du = d['dong_bo']()
    co = set(names) <= set(d['ten_tren_drive']())
    log(f"C7: da ghi {len(names)} tep (1 JSON, 1 CSV {len(rows)} hang, {len(ctx.get('hinh', {}))} PNG, du doan {len(pv)} bo) | dong_bo {'DU' if du else 'CHUA DU'} | "
        f"tren Drive: {'CO du' if co else 'THIEU'}" + ('' if du and co else ' | KHONG tat runtime; chay lai C7'))
    return dict(dong_bo=du, du_tep=co, tep=names)
