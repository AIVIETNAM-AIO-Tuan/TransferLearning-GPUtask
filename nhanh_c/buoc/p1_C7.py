# ==== nhanh_c buoc p1_C7 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C7 · BANG 78x J v3b dot 4 — VAL tung epoch (dau doc trong luot) CANH J v3b s42 (GHEP CAP: cung seed, thu tu lo, khoi dong, lambda, lich
#      6 epoch) + dau van lo + lambda + write-back / tran tung tang — BAO KEM (so quyet dinh = 78z tren rs_ ep4, gop trong so). B.4: VAL = trong so
#      EMA (+ dong trong so THO). Chay duoc ngay sau C0 (luot nay: tep local truoc roi Drive; s42: Drive, CHI doc); luot chua xong -> in phan da co ====
assert 'kiem_ke' in globals() and 'ap_cols' in globals() and 'CFG_D4' in globals() and 'tim_moi_noi' in globals(), 'chay C0 truoc'
_SFX7 = '_thu' if THU_NHANH else ''
_PJ7, _LJ7 = 'p_J_bert_dong_s{s}_ep{e}.npy', 'epoch_log_J_bert_dong_ep{e}.json'
_DOI7 = tuple(CFG_D4['ghep_cap']) if SEED == 42 else tuple(CAU_HINH_C['moc_x'])   # nhanh C: seed khac 42 -> canh C s42 (KHONG ghep cap)
_GHEP7 = SEED == 42                                    # nhanh C: chi s42 ghep cap (cung seed / lo / khoi dong)
_L7 = [f'{BIEN_THE_D4}_s{SEED}', _DOI7[1]]
_S7 = {_L7[0]: SEED, _L7[1]: 42}
_KK7 = {_L7[0]: None, _L7[1]: kiem_ke((_DOI7[0], _DOI7[1] + _SFX7))[1]}   # None = thu muc CUA notebook nay


def _lay7(l, ten):
    if _KK7[l] is None:
        return tim_moi_noi(ten)
    ds = _KK7[l].get(ten) or []
    return tai(ds[0]) if ds else None


HAT_NUA = 7801
_su_va = np.unique(SUB[I_VA])
_A_va = np.random.default_rng(HAT_NUA).choice(_su_va, size=len(_su_va) // 2, replace=False)
NUA_V = np.where(np.isin(SUB[I_VA], _A_va), 0, 1)
YV7 = YM[I_VA]
_NUA7 = {'toan': np.ones(len(I_VA), bool), 'A': NUA_V == 0, 'B': NUA_V == 1}
_mh7 = lambda P_, m: float(np.nanmean(ap_cols(YV7[m], P_[m])))
KQ7, LAM7, LOG7 = {}, {}, {}
for _l7 in _L7:
    KQ7[_l7], LAM7[_l7], LOG7[_l7] = {}, None, None
    for _e7 in range(1, EPOCHS + 1):
        _p7 = _lay7(_l7, _PJ7.format(s=_S7[_l7], e=_e7))
        if not _p7: continue
        _P7 = np.load(_p7).astype('float32')
        assert _P7.shape == (len(I_VA), C), f'{_l7} ep{_e7}: p_ {_P7.shape} KHAC ({len(I_VA)}, {C}) -> THU_NHANH khac nhau?'
        KQ7[_l7][_e7] = {h: _mh7(_P7, m) for h, m in _NUA7.items()}
    _h7 = _lay7(_l7, 'he_so_lambda.json')
    if _h7:
        with open(_h7, encoding='utf-8') as _f7: LAM7[_l7] = json.load(_f7)
    for _e7 in range(EPOCHS, 0, -1):
        _g7 = _lay7(_l7, _LJ7.format(e=_e7))
        if _g7:
            with open(_g7, encoding='utf-8') as _f7: LOG7[_l7] = json.load(_f7)
            break
_fm7 = lambda v, w=4: '   -  ' if v is None else f'{v:.{w}f}'
print(f"VAL muc anh (dau doc trong luot), toan | nua A | nua B (benh nhan, hat {HAT_NUA}). Chenh = {_L7[0]} - {_L7[1]} "
      + ('(GHEP CAP: cung seed, thu tu lo, khoi dong, lambda, lich 6 epoch)' if _GHEP7 else '(KHONG ghep cap: khac seed; cung khoi dong luong + lambda J v3b s42, cung lich 6 epoch)'))
print(f"{'ep':<4}" + ''.join(f'{l:>28}' for l in _L7) + f"{'chenh toan / A / B':>28}")
for e in range(1, EPOCHS + 1):
    _r7 = [KQ7[l].get(e) for l in _L7]
    print(f'{e:<4}' + ''.join((f"{_fm7(x['toan'])} | {_fm7(x['A'])} | {_fm7(x['B'])}" if x else 'chua co').rjust(28) for x in _r7)
          + (('    ' + ' / '.join(f"{_r7[0][h] - _r7[1][h]:+.4f}" for h in ('toan', 'A', 'B'))) if (_r7[0] and _r7[1]) else f"{'-':>28}"))
DU7 = _GHEP7 and len(KQ7[_L7[0]]) == EPOCHS and len(KQ7[_L7[1]]) == EPOCHS
CHENH_TB7 = ({h: float(np.mean([KQ7[_L7[0]][e][h] - KQ7[_L7[1]][e][h] for e in range(1, EPOCHS + 1)])) for h in ('toan', 'A', 'B')}
             if DU7 else None)
if CHENH_TB7:
    print(f"\nchenh TB ep1-{EPOCHS} ({_L7[0]} - {_L7[1]}; BAO KEM): toan {CHENH_TB7['toan']:+.4f} | nua A {CHENH_TB7['A']:+.4f} | nua B "
          f"{CHENH_TB7['B']:+.4f}")
_lam7 = {l: (LAM7[l] or {}).get('lam') for l in _L7}
print('\nlambda (he_so_lambda.json): ' + ' || '.join(
    f"{l}: " + (' '.join(f'{k} {float(v):.4g}' for k, v in _lam7[l].items()) if _lam7[l] else 'chua co') for l in _L7))
LAM_TRUNG7 = None
if all(_lam7.values()):
    _ch7 = set(_lam7[_L7[0]]) & set(_lam7[_L7[1]])
    LAM_TRUNG7 = all(_lam7[_L7[0]][k] == _lam7[_L7[1]][k] for k in _ch7)
    print('  lambda cac phan chung ' + ('TRUNG tung so' if LAM_TRUNG7 else '⚠ KHAC: bao Claude (ghep cap hong)')
          + f" | phan rieng luot nay: {sorted(set(_lam7[_L7[0]]) - _ch7) or '-'}")
LO_TRUNG7 = None
if LOG7[_L7[0]] and LOG7[_L7[1]]:
    _g7 = [(e, (LOG7[_L7[0]].get(str(e - 1)) or {}).get('lo_dau'), (LOG7[_L7[1]].get(str(e - 1)) or {}).get('lo_dau')) for e in range(1, EPOCHS + 1)]
    _g7 = [(e, a, b) for e, a, b in _g7 if a and b]
    LO_TRUNG7 = bool(_g7) and all(a == b for _, a, b in _g7)
    print('dau van lo luot nay / luot kia: ' + ', '.join(f"ep{e} {'TRUNG' if a == b else 'LECH'}" for e, a, b in _g7)
          + (' -> TRUNG het' if LO_TRUNG7 else (' -> ⚠ LECH: bao Claude (thu tu lo khac)' if _GHEP7 else ' -> lech (khac seed, dung nhu mong doi)')))
print(f"\nwrite-back ||dh|| / ||h|| tang 8-11 (rho {CFG_JV3B['tran_ghi_rho']}): ghi 3 lo dau | ti lo max | tho anh max | anh cham tran tong")
for _l7 in _L7:
    for e in range(1, EPOCHS + 1):
        _x7 = ((LOG7[_l7] or {}).get(str(e - 1)) or {}).get('luong') or {}
        if not _x7: continue
        _t7 = _x7.get('tran') or {}
        print(f"  {_l7:<26} ep{e}: {' '.join(_fm7(v) for v in (_x7.get('ghi_tang') or [])) or '-'} | "
              + (f"{' '.join(_fm7(v) for v in _t7['ti_max'])} | {' '.join(_fm7(v, 3) for v in _t7['tho_max'])} | "
                 f"{' '.join(f'{v:.1%}' for v in _t7['ti_le_cham'])}" if _t7 else '-'))
if BIEN_THE_D4 == 'dp_ema':                            # 78x dot 4 B.4: VAL trong so THO (bao kem, p_tho_)
    print('\nB.4 VAL muc anh trong so THO (luot nay): toan | nua A | nua B')
    for e in range(1, EPOCHS + 1):
        _pt7 = _lay7(_L7[0], _PJ7.format(s=SEED, e=e).replace('p_', 'p_tho_', 1))
        if _pt7:
            _Pt7 = np.load(_pt7).astype('float32')
            print(f'  ep{e}: ' + ' | '.join(f'{_mh7(_Pt7, m):.4f}' for m in _NUA7.values()))
_ts7 = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
ghi(f'bang_78x_d4_{BIEN_THE_D4}_{_ts7}.json', lambda q: ghi_json(q, dict(
    val=KQ7, chenh_tb=CHENH_TB7, ghep_cap=_GHEP7, lam_trung=LAM_TRUNG7, lo_trung=LO_TRUNG7, lam=_lam7, hat_nua=HAT_NUA, epochs=EPOCHS, seed=SEED,
    luot=CFG_D4_CHON, doi=list(_DOI7),
    luat='Tuan duyet 21/10 (Event "Tuan duyet dot 4 ..."): VAL trong luot CHI bao kem; so quyet dinh = 78z (fus_attn_hr gop trong so) tren rs_ ep4')))
