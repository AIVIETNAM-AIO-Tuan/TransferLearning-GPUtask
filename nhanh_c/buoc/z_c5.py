# ==== nhanh_c buoc z_c5 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
from sklearn.metrics import average_precision_score
def ap_cols(Yc, Sc):
    o = np.full(Yc.shape[1], np.nan)
    for j in range(Yc.shape[1]):
        y, s = Yc[:, j], Sc[:, j]; n1 = y.sum()
        if n1 == 0 or n1 == len(y): continue
        r = np.argsort(-s, kind='stable'); yy, ss = y[r], s[r]
        tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
        last = np.r_[np.diff(ss) != 0, True]; tp, fp = tp[last], fp[last]
        pr, rc = tp / (tp + fp), tp / n1
        o[j] = float((np.diff(np.r_[0.0, rc]) * pr).sum())
    return o
tiers = lambda ap: {t: float(np.nanmean(ap[i])) for t, i in TIER.items()}
_rs6 = np.random.default_rng(6); _yy = (_rs6.random((3000, C)) < 0.2).astype('float32')
_ss6 = _rs6.random((3000, C)).astype('float32')
_dd6 = float(np.nanmax(np.abs(ap_cols(_yy, _ss6) -
        np.array([average_precision_score(_yy[:, j], _ss6[:, j]) for j in range(C)]))))
assert _dd6 < 1e-6, f'ap_cols lech sklearn {_dd6:.2e}'
print(f'cong 5: ap_cols khop sklearn toi {_dd6:.2e}')
