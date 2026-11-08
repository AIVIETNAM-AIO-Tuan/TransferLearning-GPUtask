# ==== nhanh_c buoc bp3_SO_SANH — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== SO SANH - DO LAI PHAN 2 + 3 CUA B s44 TREN CUNG BACKBONE (Tuan 08/11; Event 2026-11-08) - chay SAU C6S, TRUOC C9S ====
# Luot nay (_p2b) so voi Phan 2 dau tien tren cung backbone (r84z_B_s44) va so da in cua luot goc 03-04/11 (tep mat). Ba dau co gop trong so: gop trong so ep1-10 + TB du doan ep1-10,
# toan val / hai nua theo benh nhan. KHONG luat, KHONG chon: so chinh thuc cua s44 van la Phan 2 dau tien (dat truoc). Thieu tep ben kia -> in, bo qua, KHONG dung. KHONG doc nhan test.
assert isinstance(globals().get('CAU_HINH_C'), dict) and CAU_HINH_C.get('bien_the') == 'Bp', 'buoc bien the Bp: chay qua nhanh_c.buoc(..., bien_the="Bp") -> DUNG'
_SS = CAU_HINH_C['so_sanh']
_DS_SS = [d for d in DAU_S if d in _pvA and (d, _KA) in KQ_S and (d, _KTB) in KQ_S]
assert _DS_SS and DAU_QD_S in _DS_SS, 'so sanh: thieu dau quyet dinh trong C6S -> DUNG'
_kg_ss = kiem_ke(tuple(_SS['gts']), mau=r'p_gopA_[a-z_]+_s\d+\.npy')[1]
_kz_ss = kiem_ke(tuple(_SS['z']), mau=rf'p_[a-z0-9_]+_s{SEED}_ep\d+\.npy')[1]
KQ_SS, _thieu_ss = {}, []
print(f"\n=== SO SANH: luot nay ({'/'.join(DRV_DUONG)}) - {_SS['nhan']} ({'/'.join(_SS['gts'])}, {'/'.join(_SS['z'])}) | val {len(_IVA):,} anh, gop view; hai nua theo benh nhan, "
      f"hat {HAT_NUA_S} | KHONG luat, KHONG chon ===")
print(f"{'dau':<13}{'phep':<22}{'nay':>8}{'dau tien':>10}{'chenh':>8} |{'nua A nay':>10}{'dau tien':>9}{'chenh':>8} |{'nua B nay':>10}{'dau tien':>9}{'chenh':>8}")
for _d_ss in _DS_SS:
    _dg_ss = _kg_ss.get(f'p_gopA_{_d_ss}_s{SEED}.npy') or []
    _dz_ss = [_kz_ss.get(f'p_{_d_ss}_s{SEED}_ep{e}.npy') or [] for e in EP_S]
    for _ph_ss, _nay_ss, _ben_ss in ((_KA, _pvA[_d_ss], (np.load(tai(_dg_ss[0])).astype('float32') if _dg_ss else None)),
                                     (_KTB, np.mean(P_EP[_d_ss], 0), (np.mean([np.load(tai(v[0])).astype('float32') for v in _dz_ss], 0) if all(_dz_ss) else None))):
        _a_ss = KQ_S[(_d_ss, _ph_ss)]
        if _ben_ss is None:
            _thieu_ss.append(f'{_d_ss}|{_ph_ss}')
            print(f"{_d_ss:<13}{_ph_ss:<22}{_a_ss['toan']:>8.4f}{'-':>10}{'':>8} | (thieu tep ben kia trong {'/'.join(_SS['gts'] if _ph_ss == _KA else _SS['z'])} -> bo qua)")
            continue
        assert _ben_ss.shape == np.asarray(_nay_ss).shape and np.isfinite(_ben_ss).all(), f'so sanh {_d_ss} {_ph_ss}: tep ben kia sai hinh / khong huu han -> DUNG'
        _b_ss = _so3(_gop_v(_ben_ss))
        KQ_SS[(_d_ss, _ph_ss)] = dict(nay=_a_ss, dau_tien=_b_ss, chenh={k: _a_ss[k] - _b_ss[k] for k in ('toan', 'nua_A', 'nua_B')})
        print(f"{_d_ss:<13}{_ph_ss:<22}{_a_ss['toan']:>8.4f}{_b_ss['toan']:>10.4f}{_a_ss['toan'] - _b_ss['toan']:>+8.4f} |{_a_ss['nua_A']:>10.4f}{_b_ss['nua_A']:>9.4f}"
              f"{_a_ss['nua_A'] - _b_ss['nua_A']:>+8.4f} |{_a_ss['nua_B']:>10.4f}{_b_ss['nua_B']:>9.4f}{_a_ss['nua_B'] - _b_ss['nua_B']:>+8.4f}")
print(f"so da in cua luot goc ({_SS['goc_in']['nhan']}): {DAU_QD_S} {_KA} {_SS['goc_in']['gop']:.4f} | luot nay {KQ_S[(DAU_QD_S, _KA)]['toan']:.4f} -> chenh "
      f"{KQ_S[(DAU_QD_S, _KA)]['toan'] - _SS['goc_in']['gop']:+.4f}" + (f" | Phan 2 dau tien {KQ_SS[(DAU_QD_S, _KA)]['dau_tien']['toan']:.4f} -> chenh "
      f"{KQ_SS[(DAU_QD_S, _KA)]['dau_tien']['toan'] - _SS['goc_in']['gop']:+.4f}" if (DAU_QD_S, _KA) in KQ_SS else ''))
print('doc: |nay - dau tien| cho nhieu chay lai cua RIENG Phan 2 tren cung backbone; so chinh thuc cua s44 KHONG doi theo bang nay (Event 2026-11-08)')
_ts_ss = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
ghi(f'so_sanh_p2b_{_ts_ss}.json', lambda q: ghi_json(q, dict(
    t=_ts_ss, seed=SEED, luot_nay=list(DRV_DUONG), dau_tien=dict(gts=list(_SS['gts']), z=list(_SS['z'])), goc_in=dict(_SS['goc_in']),
    ket_qua={f'{d}|{p}': v for (d, p), v in KQ_SS.items()}, thieu=_thieu_ss,
    ghi_chu='Tuan 08/11: do nhieu chay lai Phan 2 tren cung backbone; so chinh thuc cua s44 = Phan 2 dau tien (r84z_B_s44); KHONG luat; KHONG doc nhan test.')))
print(f"da ghi so_sanh_p2b_{_ts_ss}.json (len Drive o C9)" + (f" | thieu: {', '.join(_thieu_ss)}" if _thieu_ss else ''))
print('NHAC: xong C9S -> chay 84_chep_tai_khoan_ben.ipynb bang tai khoan ben de chep luot nay')
del _SS, _DS_SS, _kg_ss, _kz_ss, _d_ss, _dg_ss, _dz_ss, _ph_ss, _nay_ss, _ben_ss, _a_ss, _ts_ss, _thieu_ss
