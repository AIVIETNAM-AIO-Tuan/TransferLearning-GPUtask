# -*- coding: utf-8 -*-
"""Bootstrap ghép cặp + họ phép so B − X cho bảng gộp `93t` (PLAN_93 §5). Mọi phép tính số đo dùng NGUYÊN hàm của vindr_bench @ c81af62 (đã kiểm trùng
sklearn ở `86` / `89`): metrics.sort_once / ap_weighted / auprc_trapezoid_weighted / auroc_weighted (bootstrap bằng trọng số), compare.max_t (khoảng
đồng thời) và compare.verdict (dạng luật §7.4 của PLAN_89 = §5 của PLAN_93).

    weights     trọng số W [n_boot, N] (ảnh i có mặt W[b, i] lần ở lần rút b). Đơn vị 'anh' (Shenzhen): = metrics.bootstrap_weights(N, 2.000, hạt 8502)
                — đúng cách của `89`. Đơn vị 'nguoi' (RSNA): rút CỤM người NIH có hoàn lại (cụm sắp tăng theo mã; default_rng(8502).integers mỗi lần)
                -> ảnh mang số lần cụm của nó có mặt. Cùng một W cho MỌI hàng / đầu / hạt của một bộ -> hiệu ghép cặp theo cụm.
    set_stats   một bộ dự đoán [N, C] của MỘT hạt: điểm (AP bậc thang, AUPRC hình thang, AUROC; macro trên `cols`) + giá trị của TỪNG lần rút + AP hai nửa
    make_cell   một Ô = (encoder, đầu, nhánh) của một bộ: TB các hạt (mỗi lần rút tính từng hạt rồi lấy TB — như `89`), SD mẫu giữa hạt, ensemble báo kèm
    diff        hiệu ghép cặp A − B trên cùng lần rút -> hiệu điểm, CI95 danh nghĩa, (số của luật) hai nửa + nhãn dạng luật danh nghĩa
    family      họ phép so B − X (8 đối chứng × 4 đầu = 32) -> khoảng đồng thời max-T 95 % + phán quyết; điều khoản trần -> chỉ mô tả
    ceiling     điều khoản trần (đặt trước): đối chứng tốt nhất >= 0,98 ở số của luật -> luật ở bộ đó thành mô tả

Lần rút thiếu ca dương (hoặc thiếu ca âm) ở bất kỳ lớp nào của tập lớp cố định -> NaN, BỊ LOẠI và được đếm (không để nanmean đổi số lớp)."""
import numpy as np

from .. import config
from . import single as S

METRICS = ('ap', 'auprc_thang', 'auroc')
RULE_METRIC = 'ap'                         # số của luật ở RSNA / Shenzhen (PLAN_93 §5)
VERDICT_TEXT = {'hon': 'B hon', 'kem': 'B kem', 'khong phan biet': 'khong phan biet', 'mo ta': 'MO TA (dieu khoan tran)'}


def _check_margin():
    from vindr_bench.evaluation import compare as CP
    assert abs(CP.MARGIN - config.RULE_DELTA) < 1e-12, f'nguong luat cua vindr_bench ({CP.MARGIN}) KHAC PLAN_93 ({config.RULE_DELTA}) -> DUNG'
    return CP


def weights(groups, unit, n_boot=config.N_BOOT, seed=config.BOOT_SEED):
    '''groups [N] = mã cụm từng ảnh (đơn vị 'anh': mã ảnh). -> W [n_boot, N] float64.'''
    from vindr_bench.evaluation import metrics as M
    g = np.asarray(groups).astype(str)
    if unit == 'anh':
        return M.bootstrap_weights(len(g), n_boot, seed)
    assert unit == 'nguoi', f'don vi bootstrap {unit!r} khong biet -> DUNG'
    u, inv = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(seed)
    W = np.empty((n_boot, len(g)), dtype=np.float64)
    for b in range(n_boot):
        W[b] = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(np.float64)[inv]
    return W


def set_stats(preds, target, cols, W, half):
    '''preds / target [N, C]; cols = tập lớp cố định (một nhãn: [0]); W = weights(...); half = mảng 0 / 1 (S.halves). -> dict(diem, rut, nua).'''
    from vindr_bench.evaluation import metrics as M
    preds, target = np.asarray(preds, dtype=np.float32), np.asarray(target)
    assert preds.ndim == 2 and preds.shape == target.shape and np.isfinite(preds).all(), 'du doan / nhan KHONG khop hoac khong huu han -> DUNG'
    assert W.shape[1] == len(target) and half.shape == (len(target),), 'trong so / hai nua KHONG cung so anh -> DUNG'
    cols = [int(j) for j in cols]
    pre = M.sort_once(target[:, cols], preds[:, cols])
    D = {m: np.empty((len(W), len(cols)), dtype=np.float64) for m in METRICS}
    for b, w in enumerate(W):
        D['ap'][b] = M.ap_weighted(pre, w)
        D['auprc_thang'][b] = M.auprc_trapezoid_weighted(pre, w)
        D['auroc'][b] = M.auroc_weighted(pre, w)
    rut = {m: np.where(np.isfinite(D[m]).all(1), np.nan_to_num(D[m]).mean(1), np.nan) for m in METRICS}
    return dict(diem=S.point(target, preds, cols), rut=rut,
                nua=[S.point(target[half == h], preds[half == h], cols)[RULE_METRIC] for h in (0, 1)])


def ci95(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float('nan')] * 2


def make_cell(stats, ensemble=None):
    '''stats = [set_stats của từng hạt]; ensemble = điểm của TB xác suất các hạt (dict METRICS) | None. -> Ô.'''
    k = len(stats)
    assert k >= 1
    diem = {m: float(np.mean([s['diem'][m] for s in stats])) for m in METRICS}
    sd = {m: (float(np.std([s['diem'][m] for s in stats], ddof=1)) if k > 1 else None) for m in METRICS}
    rut = {m: np.mean([s['rut'][m] for s in stats], 0) for m in METRICS}        # NaN ở một hạt -> NaN (cùng nhãn, cùng W nên cùng lần rút bị loại)
    return dict(so_hat=k, diem=diem, sd=sd, tung_hat={m: [float(s['diem'][m]) for s in stats] for m in METRICS}, rut=rut,
                ci={m: ci95(rut[m]) for m in METRICS}, loai={m: int(np.isnan(rut[m]).sum()) for m in METRICS},
                nua=[float(np.mean([s['nua'][h] for s in stats])) for h in (0, 1)], ensemble=ensemble)


def diff(a, b, metric=RULE_METRIC):
    '''Hiệu ghép cặp Ô A − Ô B trên cùng lần rút. -> dict(d, ci (danh nghĩa), rut, se, nua (chỉ số của luật), ket_luan (danh nghĩa, chỉ số của luật)).'''
    CP = _check_margin()
    d = float(a['diem'][metric] - b['diem'][metric])
    r = a['rut'][metric] - b['rut'][metric]
    ok = r[np.isfinite(r)]
    out = dict(d=d, ci=ci95(r), rut=r, se=float(ok.std(ddof=1)) if len(ok) > 1 else float('nan'), nua=None, ket_luan=None)
    if metric == RULE_METRIC:
        out['nua'] = [float(a['nua'][h] - b['nua'][h]) for h in (0, 1)]
        out['ket_luan'] = CP.verdict(d, out['ci'][0], out['ci'][1], out['nua'])
    return out


def ceiling(cells, ref, encoders, heads, threshold=config.CEILING):
    '''cells[(encoder, đầu, 'luat')] = Ô. -> dict(ap_dung, tot_nhat, theo_dau, hai_cach_doc_trung). Cách đọc ĐẶT TRƯỚC (PLAN_93 §5): MỘT ngưỡng cho cả bộ =
       ô đối chứng (X ≠ ref, mọi đầu) có số của luật cao nhất; cách đọc "theo từng đầu" in kèm để thấy hai cách có cùng kết quả không.'''
    vals = {(e, h): cells[(e, h, 'luat')]['diem'][RULE_METRIC] for e in encoders if e != ref for h in heads if (e, h, 'luat') in cells}
    if not vals:
        return dict(ap_dung=False, tot_nhat=None, theo_dau={}, hai_cach_doc_trung=True, nguong=threshold)
    best = max(vals, key=lambda k: vals[k])
    per = {}
    for h in heads:
        hv = {e: v for (e, hh), v in vals.items() if hh == h}
        if hv:
            e = max(hv, key=lambda k: hv[k])
            per[h] = dict(encoder=e, gia_tri=float(hv[e]), dat_nguong=bool(hv[e] >= threshold))
    applies = bool(vals[best] >= threshold)
    return dict(ap_dung=applies, nguong=threshold, tot_nhat=dict(encoder=best[0], dau=best[1], gia_tri=float(vals[best])), theo_dau=per,
                hai_cach_doc_trung=all(p['dat_nguong'] == applies for p in per.values()))


def family(cells, ref, comparators, heads, descriptive=False, metric=RULE_METRIC):
    '''Họ phép so ref − X (mọi X ∈ comparators, mọi đầu; ô của luật). -> dict(q, phep_so [..], du, so_phep, co). descriptive=True (điều khoản trần / bộ
       ngoài luật): khoảng đồng thời vẫn tính, phán quyết = 'mo ta'. Họ chưa đủ (thiếu ô) -> chỉ hiệu danh nghĩa, chưa áp luật.'''
    CP = _check_margin()
    need = [(h, x) for h in heads for x in comparators]
    have = [(h, x) for h, x in need if (ref, h, 'luat') in cells and (x, h, 'luat') in cells]
    diffs = [diff(cells[(ref, h, 'luat')], cells[(x, h, 'luat')], metric) for h, x in have]
    flat = [(h, x) for (h, x), r in zip(have, diffs) if not (np.isfinite(r['se']) and r['se'] > 0)]
    assert not flat, f'phep so SUY BIEN (sai so chuan 0 / khong huu han — hai o trung du doan?): {flat} -> DUNG, bao Claude'
    full = bool(have) and len(have) == len(need)
    q, sim = CP.max_t(diffs) if full else (float('nan'), [None] * len(diffs))
    rows = []
    for (h, x), r, s in zip(have, diffs, sim):
        lab = ('mo ta' if descriptive else s['ket_luan']) if s else f"chua ap luat (danh nghia: {r['ket_luan']})"
        side = {m: diff(cells[(ref, h, 'luat')], cells[(x, h, 'luat')], m) for m in METRICS if m != metric}
        rows.append(dict(dau=h, doi_chung=x, d=r['d'], ci_danh_nghia=r['ci'], ci_dong_thoi=(s['ci_dong_thoi'] if s else None), nua=r['nua'], se=r['se'],
                         ket_luan=lab, kem={m: dict(d=v['d'], ci_danh_nghia=v['ci']) for m, v in side.items()}))
    return dict(q=(float(q) if full else None), du=full, so_phep=len(need), co=len(have), mo_ta=bool(descriptive), phep_so=rows,
                lan_rut_dung=(int(np.isfinite(np.stack([r['rut'] for r in diffs])).all(0).sum()) if diffs else 0))


def pairwise(a, b):
    '''Hiệu ghép cặp MÔ TẢ (giữa đầu đọc / nhánh theo AP − luật): ba số đo, CI danh nghĩa. -> {số đo: dict(d, ci)}.'''
    return {m: dict(d=v['d'], ci=v['ci']) for m in METRICS for v in [diff(a, b, m)]}
