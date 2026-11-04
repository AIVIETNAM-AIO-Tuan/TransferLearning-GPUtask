# ==== nhanh_c buoc bg3_QUYET — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== QUYET DINH · BIEN THE Bg - B CUNG seed (luat dat TRUOC, Tuan duyet 04/11: Event 2026-11-04 "Tuan duyet thiet ke Bg s44 ..."; SPEC B.2ai) - SAU C6S, TRUOC C9S ====
# So = fus_attn_hr gop trong so ep1-10 (bo A, p_gopA), val gop view (CUNG _gop_v / _so3 cua C6S). Doi chung = B CUNG seed (CAU_HINH_C['quyet']
# ['gts'], CHI doc; cong: tinh lai = moc +- tol). LUAT: Bg - B >= nguong toan val VA > 0 o CA HAI nua (benh nhan, hat 7801) -> Bg CO ICH (xac
# nhan seed tiep theo ghep cap - Tuan quyet); khong -> Bg KHONG DAT (dung huong gradient tang thap, giu cong thuc B). Bao kem: CI95 bootstrap GHEP
# CAP theo benh nhan tren TAP LOP CO DINH (lan rut thieu lop -> LOAI, in so lan + ten lop; KHONG nanmean am tham), mldec_hr / fus_tb_hr, tier
# (tier cua C0 Phan 3). Ma tinh = g3_QUYET (Cg) NGUYEN VAN tru nhan luot / ten tep. KHONG doc nhan test.
_Q_QD = CAU_HINH_C['quyet']
assert CAU_HINH_C.get('bien_the') == 'Bg' and DAU_QD_S == _Q_QD['dau'] and all(d in _pvA for d in DAU_S), 'QUYET DINH: chay sau C6S cua bien the Bg -> DUNG'
_kk_qd = kiem_ke(tuple(_Q_QD['gts']), mau=r'p_gopA_[a-z_]+_s\d+\.npy')[1]
_QC_QD = {}
for _d_qd in DAU_S:
    _ds_qd = _kk_qd.get(f'p_gopA_{_d_qd}_s{SEED}.npy') or []
    assert _ds_qd, f"thieu p_gopA_{_d_qd}_s{SEED}.npy cua {CAU_HINH_C['nhan_moc']} trong {'/'.join(_Q_QD['gts'])} -> DUNG"
    if xung_dot(_ds_qd): print(f"  ! p_gopA_{_d_qd}_s{SEED}.npy: {len(_ds_qd)} ban NOI DUNG KHAC -> dung ban MOI NHAT ({_ds_qd[0]['mt']})")
    _QC_QD[_d_qd] = np.load(tai(_ds_qd[0])).astype('float32')
    assert _QC_QD[_d_qd].shape == np.asarray(_pvA[_d_qd]).shape and np.isfinite(_QC_QD[_d_qd]).all(), f'{CAU_HINH_C["nhan_moc"]} {_d_qd}: sai hinh -> DUNG'
KQ_QD = {(b, d): _so3(_gop_v(np.asarray(src[d], 'float32'))) for b, src in (('Bg', _pvA), ('B', _QC_QD)) for d in DAU_S}
assert abs(KQ_QD[('B', _Q_QD['dau'])]['toan'] - float(_Q_QD['moc_gop'])) < float(_Q_QD['tol_moc']), (
    f"cong: {CAU_HINH_C['nhan_moc']} {_Q_QD['dau']} gop tinh lai {KQ_QD[('B', _Q_QD['dau'])]['toan']:.4f} KHAC moc {_Q_QD['moc_gop']} -> DUNG")
assert abs(KQ_QD[('Bg', _Q_QD['dau'])]['toan'] - KQ_S[(_Q_QD['dau'], _KA)]['toan']) < 1e-9, 'QUYET DINH: so cua luot nay KHAC bang C6S -> DUNG'
CHENH_QD = {d: {f: KQ_QD[('Bg', d)][f] - KQ_QD[('B', d)][f] for f in ('toan', 'nua_A', 'nua_B')} for d in DAU_S}
_cq = CHENH_QD[_Q_QD['dau']]
PHAN_XU_QD = 'Bg CO ICH' if (_cq['toan'] >= float(_Q_QD['nguong']) and _cq['nua_A'] > 0 and _cq['nua_B'] > 0) else 'Bg KHONG DAT'


def _thu_tu_10(Y, G):
    '''moi lop: thu tu giam dan theo diem (stable), nhan da xep, mat na cuoi chuoi diem bang nhau - CUNG cach ap_cols.'''
    out = []
    for j in range(G.shape[1]):
        r = np.argsort(-G[:, j], kind='stable')
        out.append((r, Y[r, j].astype('float64'), np.r_[np.diff(G[r, j]) != 0, True]))
    return out


def _ap_w10(tt, w):
    '''AP tung lop voi trong so mau w (bootstrap: so lan benh nhan duoc rut); w = 1 -> TRUNG ap_cols. Lop khong duong / toan duong -> NaN.'''
    o = np.full(len(tt), np.nan)
    for j, (r, yy, last) in enumerate(tt):
        ww = w[r]
        n1, nw = float((yy * ww).sum()), float(ww.sum())
        if n1 == 0 or n1 == nw: continue
        tp, fp = np.cumsum(yy * ww)[last], np.cumsum((1 - yy) * ww)[last]
        mau = tp + fp
        pr = np.divide(tp, mau, out=np.zeros_like(tp), where=mau > 0)
        o[j] = float((np.diff(np.r_[0.0, tp / n1]) * pr).sum())
    return o


_G_QD = {b: _gop_v(np.asarray(src[_Q_QD['dau']], 'float32')) for b, src in (('Bg', _pvA), ('B', _QC_QD))}
_TT_QD = {b: _thu_tu_10(YV_S, v) for b, v in _G_QD.items()}
_a1_qd = {b: _ap_w10(_TT_QD[b], np.ones(len(YV_S))) for b in _G_QD}
assert all(np.allclose(_a1_qd[b], ap_cols(YV_S, _G_QD[b]), atol=1e-12, equal_nan=True) for b in _G_QD), 'AP co trong so (w = 1) KHAC ap_cols -> DUNG'
LOP_CD_QD = np.isfinite(_a1_qd['B'])
assert np.array_equal(np.isfinite(_a1_qd['Bg']), LOP_CD_QD), 'tap lop hop le KHAC giua hai luot (cung nhan) -> DUNG'
_bn_qd, _inv_qd = np.unique(SUB[_IVA], return_inverse=True)
_rng_qd, _t0_qd = np.random.default_rng(int(_Q_QD['hat'])), time.time()
_bs_qd, _thieu_qd = np.full(int(_Q_QD['n_boot']), np.nan), {}
for _i_qd in range(int(_Q_QD['n_boot'])):
    _w_qd = np.bincount(_rng_qd.integers(0, len(_bn_qd), len(_bn_qd)), minlength=len(_bn_qd))[_inv_qd].astype('float64')
    _a_qd = {b: _ap_w10(_TT_QD[b], _w_qd)[LOP_CD_QD] for b in _G_QD}
    _hl_qd = np.isfinite(_a_qd['B'])
    assert np.array_equal(np.isfinite(_a_qd['Bg']), _hl_qd), 'bootstrap: lop hop le KHAC giua hai luot -> DUNG'
    if not _hl_qd.all():                                    # lan rut THIEU lop -> LOAI khoi CI (dem + ten lop)
        for _j_qd in np.flatnonzero(LOP_CD_QD)[~_hl_qd].tolist(): _thieu_qd[_j_qd] = _thieu_qd.get(_j_qd, 0) + 1
        continue
    _bs_qd[_i_qd] = _a_qd['Bg'].mean() - _a_qd['B'].mean()
N_DUNG_QD, BS_QD = int(np.isfinite(_bs_qd).sum()), _bs_qd
CI_QD = (float(np.quantile(_bs_qd[np.isfinite(_bs_qd)], 0.025)), float(np.quantile(_bs_qd[np.isfinite(_bs_qd)], 0.975))) if N_DUNG_QD else (float('nan'),) * 2
print(f"\n=== QUYET DINH BIEN THE Bg s{SEED} - {CAU_HINH_C['nhan_moc']} (val {len(_IVA):,} anh, gop view; gop trong so ep1-10; hai nua theo benh nhan, "
      f"hat {HAT_NUA_S}) ===")
print(f"{'dau':<13}{'ban':<9}{'toan':>8}{'nua A':>8}{'nua B':>8}{'head':>8}{'medium':>8}{'tail':>8}")
for _d_qd in DAU_S:
    for _b_qd in ('Bg', 'B'):
        _r_qd = KQ_QD[(_b_qd, _d_qd)]
        print(f"{_d_qd:<13}{_b_qd + ' s' + str(SEED):<9}{_r_qd['toan']:>8.4f}{_r_qd['nua_A']:>8.4f}{_r_qd['nua_B']:>8.4f}"
              + ''.join(f"{_r_qd['tier'][t]:>8.4f}" for t in ('head', 'medium', 'tail')))
print(f"Bg - B ({_Q_QD['dau']}, QUYET DINH): toan {_cq['toan']:+.4f} | nua A {_cq['nua_A']:+.4f} | nua B {_cq['nua_B']:+.4f} | CI95 [{CI_QD[0]:+.4f}; "
      f"{CI_QD[1]:+.4f}] (bootstrap {int(_Q_QD['n_boot'])} lan, hat {_Q_QD['hat']}, {(time.time() - _t0_qd) / 60:.1f} phut; {N_DUNG_QD} lan du "
      f"{int(LOP_CD_QD.sum())} lop co dinh, loai {int(_Q_QD['n_boot']) - N_DUNG_QD}"
      + (' - lop: ' + ', '.join(f'{LAB40[j]} x{n}' for j, n in sorted(_thieu_qd.items())) if _thieu_qd else '') + ')'
      + ('' if N_DUNG_QD >= 0.95 * int(_Q_QD['n_boot']) else ' !! < 95% lan rut du lop -> CI KHONG dang tin (bao Claude); luat KHONG doi'))
print(f"LUAT (dat truoc, Tuan duyet 04/11): Bg - B >= +{_Q_QD['nguong']} toan val VA > 0 o CA HAI nua -> {PHAN_XU_QD}"
      + (' -> xac nhan seed tiep theo ghep cap (Tuan quyet)' if PHAN_XU_QD == 'Bg CO ICH' else ' -> dung huong gradient tang thap, giu cong thuc B'))
print('bao kem (KHONG vao luat): ' + ' | '.join(f"{d}: toan {CHENH_QD[d]['toan']:+.4f} (nua A {CHENH_QD[d]['nua_A']:+.4f} / nua B {CHENH_QD[d]['nua_B']:+.4f})"
                                                 for d in DAU_S if d != _Q_QD['dau']) + ' | 1 seed = SANG LOC; nhieu chay lai cung seed ~0.002')
_ts_qd = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
ghi(f'quyet_dinh_bg_s{SEED}_{_ts_qd}.json', lambda q: ghi_json(q, dict(
    t=_ts_qd, seed=SEED, dau=_Q_QD['dau'], nguong=_Q_QD['nguong'], ket_qua={f'{b} | {d}': v for (b, d), v in KQ_QD.items()}, chenh=CHENH_QD,
    ci95=list(CI_QD), n_boot=int(_Q_QD['n_boot']), hat=_Q_QD['hat'], n_rut_du_lop=N_DUNG_QD, lop_thieu={LAB40[j]: n for j, n in _thieu_qd.items()},
    phan_xu=PHAN_XU_QD, doi_chung=dict(gts=list(_Q_QD['gts']), moc_gop=_Q_QD['moc_gop'], nhan=CAU_HINH_C['nhan_moc']),
    luat=f"{_Q_QD['dau']} gop trong so ep1-10: Bg - B cung seed >= {_Q_QD['nguong']} toan val VA > 0 o ca hai nua (Event 2026-11-04)")))
print(f"da ghi quyet_dinh_bg_s{SEED}_{_ts_qd}.json (len Drive o C9S). KHONG doc nhan test.")
del _kk_qd, _QC_QD, _d_qd, _ds_qd, _cq, _G_QD, _TT_QD, _a1_qd, _bn_qd, _inv_qd, _rng_qd, _t0_qd, _bs_qd, _i_qd, _w_qd, _a_qd, _hl_qd, _b_qd, _r_qd
