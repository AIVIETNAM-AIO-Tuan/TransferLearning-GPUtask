# ==== nhanh_c buoc b3_QUYET — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== QUYET DINH · BIEN THE B: C - B CUNG seed, ba seed ghep cap s42 / s43 / s44 (luat Claude de xuat 02/11, cho Tuan duyet: Event 2026-11-02
#      "Tuan yeu cau dung B s43 / s44 ..."; SPEC B.2ah) - SAU C6S, TRUOC C9S ====
# So = fus_attn_hr gop trong so ep1-10 (bo A, p_gopA), val gop view (CUNG _gop_v / _so3 cua C6S). Moi seed mot cap B / C: CUNG thu tu lo / khoi
# dong / lambda, CUNG doc 1 view, CUNG epoch backbone; C = B + dau J 6 view luc train. s42: B = luot skip7, C = luot goi o cat (stage4_nb78z, CHI
# doc); s43 / s44: stage4_nb84 (C CHI doc; B = luot nay, hoac luot B seed kia neu da xong). Nguon co moc: tinh lai = moc +- tol (B cua luot nay =
# bang C6S). LUAT (CHI khi DU moi seed): D = TB cac seed (C - B) toan val; D <= +bien -> 'B KHONG KEM C' (bo dau J 6 view duoc); D >= +nguong_c VA
# TB (C - B) > 0 o CA HAI nua -> 'C HON B' (giu dau 6 view); con lai -> 'CHUA RO' (Tuan quyet). Bao kem (KHONG vao luat): tung seed, tier,
# mldec_hr / fus_tb_hr, CI95 bootstrap theo benh nhan cua D va cua seed nay (tap lop co dinh; lan rut thieu lop -> LOAI, in so lan + ten lop) - CI
# CHI do nhieu chon benh nhan, KHONG do nhieu seed. Chua du seed -> in phan da co, KHONG phan xu. KHONG doc nhan test.
_Q_B = CAU_HINH_C['quyet']
_SS_B = [int(s) for s in _Q_B['seeds']]
assert CAU_HINH_C.get('bien_the') == 'B' and DAU_QD_S == _Q_B['dau'] and all(d in _pvA for d in DAU_S) and int(SEED) in _SS_B, (
    'QUYET DINH B: chay sau C6S cua bien the B (seed thuoc cac seed ghep cap) -> DUNG')


def _nap_gop_b(du, s):
    '''{dau: p_gopA} cua luot o thu muc du (seed s); THIEU tep -> None (luot chua xong Phan 3).'''
    kk = kiem_ke(tuple(du), mau=r'p_gopA_[a-z_]+_s\d+\.npy')[1]
    out = {}
    for d in DAU_S:
        ds = kk.get(f'p_gopA_{d}_s{s}.npy') or []
        if not ds: return None
        if xung_dot(ds): print(f"  ! {'/'.join(du)}/p_gopA_{d}_s{s}.npy: {len(ds)} ban NOI DUNG KHAC -> dung ban MOI NHAT ({ds[0]['mt']})")
        out[d] = np.load(tai(ds[0])).astype('float32')
        assert out[d].shape == np.asarray(_pvA[d]).shape and np.isfinite(out[d]).all(), f"{'/'.join(du)} {d} s{s}: sai hinh / khong huu han -> DUNG"
    return out


_PB_B, _CHUA_B = {}, []
for _s_b in _SS_B:
    _ng_b = _Q_B['nguon'][str(_s_b)]
    _pc_b = _nap_gop_b(_ng_b['C'], _s_b)
    assert _pc_b is not None, f"thieu p_gopA cua C s{_s_b} ({'/'.join(_ng_b['C'])}) -> DUNG"
    _pb_b = {d: np.asarray(_pvA[d], 'float32') for d in DAU_S} if _s_b == int(SEED) else _nap_gop_b(_ng_b['B'], _s_b)
    if _pb_b is None:                                                # luot B seed kia chua xong Phan 3 -> bo qua, in ten
        _CHUA_B.append(_s_b); continue
    _PB_B[_s_b] = dict(B=_pb_b, C=_pc_b)
SD_B = sorted(_PB_B)
KQ_B = {(s, b, d): _so3(_gop_v(_PB_B[s][b][d])) for s in SD_B for b in ('B', 'C') for d in DAU_S}
for _s_b in SD_B:
    for _b_b in ('B', 'C'):
        _m_b = _Q_B['nguon'][str(_s_b)].get(f'moc_{_b_b}')
        if _m_b is None or (_s_b == int(SEED) and _b_b == 'B'): continue
        assert abs(KQ_B[(_s_b, _b_b, _Q_B['dau'])]['toan'] - float(_m_b)) < float(_Q_B['tol_moc']), (
            f"cong: {_b_b} s{_s_b} {_Q_B['dau']} gop tinh lai {KQ_B[(_s_b, _b_b, _Q_B['dau'])]['toan']:.4f} KHAC moc {_m_b} -> DUNG")
assert abs(KQ_B[(int(SEED), 'B', _Q_B['dau'])]['toan'] - KQ_S[(_Q_B['dau'], _KA)]['toan']) < 1e-9, 'QUYET DINH B: so cua luot nay KHAC bang C6S -> DUNG'
CHENH_B = {(s, d): {f: KQ_B[(s, 'C', d)][f] - KQ_B[(s, 'B', d)][f] for f in ('toan', 'nua_A', 'nua_B')} for s in SD_B for d in DAU_S}
D_B = {f: float(np.mean([CHENH_B[(s, _Q_B['dau'])][f] for s in SD_B])) for f in ('toan', 'nua_A', 'nua_B')}
D_TIER_B = {t: float(np.mean([KQ_B[(s, 'C', _Q_B['dau'])]['tier'][t] - KQ_B[(s, 'B', _Q_B['dau'])]['tier'][t] for s in SD_B]))
            for t in ('head', 'medium', 'tail')}
if _CHUA_B:
    PHAN_XU_B = None
elif D_B['toan'] <= float(_Q_B['bien']):
    PHAN_XU_B = 'B KHONG KEM C'
elif D_B['toan'] >= float(_Q_B['nguong_c']) and D_B['nua_A'] > 0 and D_B['nua_B'] > 0:
    PHAN_XU_B = 'C HON B'
else:
    PHAN_XU_B = 'CHUA RO'


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


_G_B = {(s, b): _gop_v(_PB_B[s][b][_Q_B['dau']]) for s in SD_B for b in ('B', 'C')}
_TT_B = {k: _thu_tu_10(YV_S, v) for k, v in _G_B.items()}
_a1_b = {k: _ap_w10(_TT_B[k], np.ones(len(YV_S))) for k in _G_B}
assert all(np.allclose(_a1_b[k], ap_cols(YV_S, _G_B[k]), atol=1e-12, equal_nan=True) for k in _G_B), 'AP co trong so (w = 1) KHAC ap_cols -> DUNG'
LOP_CD_B = np.isfinite(_a1_b[(int(SEED), 'B')])
assert all(np.array_equal(np.isfinite(v), LOP_CD_B) for v in _a1_b.values()), 'tap lop hop le KHAC giua cac luot (cung nhan) -> DUNG'
_bn_b, _inv_b = np.unique(SUB[_IVA], return_inverse=True)
_rng_b, _t0_b, _n_bs_b = np.random.default_rng(int(_Q_B['hat'])), time.time(), int(_Q_B['n_boot'])
BS_B, _thieu_b, _ds_b = {'D': np.full(_n_bs_b, np.nan), 'seed': np.full(_n_bs_b, np.nan)}, {}, {}
for _i_b in range(_n_bs_b):
    _w_b = np.bincount(_rng_b.integers(0, len(_bn_b), len(_bn_b)), minlength=len(_bn_b))[_inv_b].astype('float64')
    _a_b = {k: _ap_w10(_TT_B[k], _w_b)[LOP_CD_B] for k in _G_B}
    _hl_b = np.isfinite(_a_b[(int(SEED), 'B')])
    assert all(np.array_equal(np.isfinite(v), _hl_b) for v in _a_b.values()), 'bootstrap: lop hop le KHAC giua cac luot -> DUNG'
    if not _hl_b.all():                                    # lan rut THIEU lop -> LOAI khoi CI (dem + ten lop; KHONG nanmean am tham)
        for _j_b in np.flatnonzero(LOP_CD_B)[~_hl_b].tolist(): _thieu_b[_j_b] = _thieu_b.get(_j_b, 0) + 1
        continue
    _ds_b = {s: _a_b[(s, 'C')].mean() - _a_b[(s, 'B')].mean() for s in SD_B}
    BS_B['D'][_i_b], BS_B['seed'][_i_b] = float(np.mean(list(_ds_b.values()))), _ds_b[int(SEED)]
N_DUNG_B = int(np.isfinite(BS_B['D']).sum())
CI_B = {k: ((float(np.quantile(v[np.isfinite(v)], 0.025)), float(np.quantile(v[np.isfinite(v)], 0.975))) if N_DUNG_B else (float('nan'),) * 2)
        for k, v in BS_B.items()}
print(f"\n=== QUYET DINH BIEN THE B s{SEED} - C - B cung seed ({_Q_B['dau']} gop trong so ep1-10; val {len(_IVA):,} anh, gop view; hai nua theo benh "
      f"nhan, hat {HAT_NUA_S}) | seed co: {', '.join(f's{s}' for s in SD_B)} | chua xong: {', '.join(f's{s}' for s in _CHUA_B) or '-'} ===")
print(f"{'seed':<6}{'ban':<5}{'toan':>8}{'nua A':>8}{'nua B':>8}{'head':>8}{'medium':>8}{'tail':>8}")
for _s_b in SD_B:
    for _b_b in ('B', 'C'):
        _r_b = KQ_B[(_s_b, _b_b, _Q_B['dau'])]
        print(f"{'s' + str(_s_b):<6}{_b_b:<5}{_r_b['toan']:>8.4f}{_r_b['nua_A']:>8.4f}{_r_b['nua_B']:>8.4f}"
              + ''.join(f"{_r_b['tier'][t]:>8.4f}" for t in ('head', 'medium', 'tail')))
    _c_b = CHENH_B[(_s_b, _Q_B['dau'])]
    print(f"{'':<6}{'C-B':<5}{_c_b['toan']:>+8.4f}{_c_b['nua_A']:>+8.4f}{_c_b['nua_B']:>+8.4f}"
          + ''.join(f"{KQ_B[(_s_b, 'C', _Q_B['dau'])]['tier'][t] - KQ_B[(_s_b, 'B', _Q_B['dau'])]['tier'][t]:>+8.4f}" for t in ('head', 'medium', 'tail'))
          + (f"   CI95 seed nay [{CI_B['seed'][0]:+.4f}; {CI_B['seed'][1]:+.4f}]" if _s_b == int(SEED) else ''))
print(f"D = TB {len(SD_B)} seed (C - B): toan {D_B['toan']:+.4f} | nua A {D_B['nua_A']:+.4f} | nua B {D_B['nua_B']:+.4f} | tier "
      + ' / '.join(f'{D_TIER_B[t]:+.4f}' for t in ('head', 'medium', 'tail')) + f" | CI95 [{CI_B['D'][0]:+.4f}; {CI_B['D'][1]:+.4f}] (bootstrap {_n_bs_b} "
      f"lan, hat {_Q_B['hat']}, {(time.time() - _t0_b) / 60:.1f} phut; {N_DUNG_B} lan du {int(LOP_CD_B.sum())} lop co dinh, loai {_n_bs_b - N_DUNG_B}"
      + (' - lop: ' + ', '.join(f'{LAB40[j]} x{n}' for j, n in sorted(_thieu_b.items())) if _thieu_b else '') + ')'
      + ('' if N_DUNG_B >= 0.95 * _n_bs_b else ' !! < 95% lan rut du lop -> CI KHONG dang tin (bao Claude); luat KHONG doi'))
if PHAN_XU_B is None:
    print(f"CHUA DU seed (thieu B s{', s'.join(map(str, _CHUA_B))}): KHONG phan xu - doc QUYET DINH cua notebook B xong SAU CUNG (hoac chay lai o nay "
          "khi luot kia xong Phan 3)")
else:
    print(f"LUAT (Claude de xuat 02/11, cho Tuan duyet): D <= +{_Q_B['bien']} -> B KHONG KEM C; D >= +{_Q_B['nguong_c']} VA > 0 o CA HAI nua -> C HON B; "
          f"con lai -> CHUA RO | D = {D_B['toan']:+.4f} -> {PHAN_XU_B}")
print('bao kem (KHONG vao luat), C - B tung seed: ' + ' | '.join(
    f'{d}: ' + ' '.join(f"s{s} {CHENH_B[(s, d)]['toan']:+.4f}" for s in SD_B) for d in DAU_S if d != _Q_B['dau'])
      + ' | CI chi do nhieu chon benh nhan (KHONG do nhieu seed ~0.001-0.002)')
_ts_b = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
ghi(f'quyet_dinh_b_s{SEED}_{_ts_b}.json', lambda q: ghi_json(q, dict(
    t=_ts_b, seed=SEED, dau=_Q_B['dau'], seeds=_SS_B, seed_co=SD_B, seed_chua=_CHUA_B, bien=_Q_B['bien'], nguong_c=_Q_B['nguong_c'],
    ket_qua={f's{s} | {b} | {d}': v for (s, b, d), v in KQ_B.items()}, chenh={f's{s} | {d}': v for (s, d), v in CHENH_B.items()}, D=D_B, D_tier=D_TIER_B,
    ci95={k: list(v) for k, v in CI_B.items()}, n_boot=_n_bs_b, hat=_Q_B['hat'], n_rut_du_lop=N_DUNG_B, lop_thieu={LAB40[j]: n for j, n in _thieu_b.items()},
    phan_xu=PHAN_XU_B, nguon=_Q_B['nguon'],
    luat=(f"{_Q_B['dau']} gop trong so ep1-10, D = TB (C - B) cac seed {_SS_B}: D <= {_Q_B['bien']} -> B KHONG KEM C; D >= {_Q_B['nguong_c']} va > 0 o "
          "ca hai nua -> C HON B; con lai CHUA RO (Claude de xuat 02/11, cho Tuan duyet; Event 2026-11-02)"))))
print(f"da ghi quyet_dinh_b_s{SEED}_{_ts_b}.json (len Drive o C9S). KHONG doc nhan test.")
del _SS_B, _PB_B, _CHUA_B, _s_b, _ng_b, _pc_b, _pb_b, _b_b, _m_b, _G_B, _TT_B, _a1_b, _bn_b, _inv_b, _rng_b, _t0_b, _n_bs_b, _thieu_b, _ds_b, _r_b, _c_b, _ts_b
