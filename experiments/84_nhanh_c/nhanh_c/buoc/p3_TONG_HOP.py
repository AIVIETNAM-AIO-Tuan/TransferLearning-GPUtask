# ==== nhanh_c buoc p3_TONG_HOP — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== TONG HOP · NHANH C: CAC SEED DA XONG (Tuan duyet 29/10; docs/SPEC_SAU_78Y.md B.2ab) - chay SAU C6S, TRUOC C9 ====
# fus_attn_hr (dau quyet dinh nhanh C) cua moi seed da co tren Drive: gop trong so ep1-10 (bo A, p_gopA) + TB du doan ep1-10 (p_ tung epoch cua
# Phan 2); val gop view, toan / hai nua theo benh nhan / tier - CUNG _gop_v / _so3 cua C6S. s42 = luot goi o cat (stage4_nb78z, CHI doc; cong:
# gop tinh lai = moc +- 5e-4); seed cua notebook nay = so C6S (1e-9). In TB +- do lech chuan qua seed, bo gop cac seed (TB du doan gop A cua cac
# seed), C - A o s42 (A = B.4 v2; seed khac chua co A). KHONG luat (Tuan da chon C). Ghi tong_hop_nhanh_c_<ts>.json + pte_gopA cua bo gop (test:
# CHI du doan, KHONG doc nhan test).
_TH = CAU_HINH_C['tong_hop']
_DAU_TH = 'fus_attn_hr'
assert DAU_QD_S == _DAU_TH and SEED in _TH['seeds'] and _DAU_TH in P_EP, 'tong hop: dau quyet dinh / seed lech -> DUNG'
_g_th, _tb_th, _te_th, _nguon_th, _chua_th = {}, {}, {}, {}, []
for _s in _TH['seeds']:
    _ng = _TH['nguon'][str(_s)]
    if _s == SEED:                                                   # seed cua notebook nay: lay tu C6S (bo nho)
        _g_th[_s], _te_th[_s] = np.asarray(_pvA[_DAU_TH], 'float32'), np.asarray(_pteA[_DAU_TH], 'float32')
        _tb_th[_s] = np.mean([np.asarray(p, 'float32') for p in P_EP[_DAU_TH]], 0)
        _nguon_th[_s] = dict(gts=list(DRV_DUONG), z=list(_DU_S), tu='C6S')
        continue
    _kg = kiem_ke(tuple(_ng['gts']), mau=r'p_gopA_[a-z_]+_s\d+\.npy|pte_gopA_[a-z_]+_s\d+\.npy')[1]
    _dg, _dt = _kg.get(f'p_gopA_{_DAU_TH}_s{_s}.npy') or [], _kg.get(f'pte_gopA_{_DAU_TH}_s{_s}.npy') or []
    if not _dg:                                                      # seed chua xong Phan 3 -> bo qua, in ten
        _chua_th.append(_s); continue
    _kz = kiem_ke(tuple(_ng['z']), mau=rf'p_{_DAU_TH}_s{_s}_ep\d+\.npy')[1]
    _dz = [_kz.get(f'p_{_DAU_TH}_s{_s}_ep{e}.npy') or [] for e in EP_S]
    assert all(_dz), f"tong hop s{_s}: thieu p_{_DAU_TH}_s{_s}_ep* trong {'/'.join(_ng['z'])} -> DUNG"
    _g_th[_s] = np.load(tai(_dg[0])).astype('float32')
    _te_th[_s] = np.load(tai(_dt[0])).astype('float32') if _dt else None
    _tb_th[_s] = np.mean([np.load(tai(v[0])).astype('float32') for v in _dz], 0)
    _nguon_th[_s] = dict(gts=list(_ng['gts']), z=list(_ng['z']), md5_gop=_dg[0].get('md5'))
    assert _g_th[_s].shape == _pvA[_DAU_TH].shape == _tb_th[_s].shape and np.isfinite(_g_th[_s]).all() and np.isfinite(_tb_th[_s]).all(), (
        f'tong hop s{_s}: du doan sai hinh / khong huu han -> DUNG')
SD_TH = sorted(_g_th)
KQ_TH = {s: dict(gop=_so3(_gop_v(_g_th[s])), tb=_so3(_gop_v(_tb_th[s]))) for s in SD_TH}
assert abs(KQ_TH[SEED]['gop']['toan'] - KQ_S[(_DAU_TH, _KA)]['toan']) < 1e-9 and abs(KQ_TH[SEED]['tb']['toan'] - KQ_S[(_DAU_TH, _KTB)]['toan']) < 1e-9, (
    'tong hop: so cua seed nay KHAC bang C6S -> DUNG')
if 42 in KQ_TH and SEED != 42:
    assert abs(KQ_TH[42]['gop']['toan'] - float(_TH['moc_s42_gop'])) < 5e-4, (
        f"cong: C s42 gop tinh lai {KQ_TH[42]['gop']['toan']:.4f} KHAC moc {_TH['moc_s42_gop']} -> DUNG")
KQ_BO_TH = dict(gop=_so3(_gop_v(np.mean([_g_th[s] for s in SD_TH], 0))), tb=_so3(_gop_v(np.mean([_tb_th[s] for s in SD_TH], 0))))
_tk_th = lambda k, f: (float(np.mean([KQ_TH[s][k][f] for s in SD_TH])), (float(np.std([KQ_TH[s][k][f] for s in SD_TH], ddof=1)) if len(SD_TH) > 1
                                                                         else float('nan')))
TK_TH = {k: {f: _tk_th(k, f) for f in ('toan', 'nua_A', 'nua_B')} for k in ('gop', 'tb')}
A_S42_TH = None
if 42 in KQ_TH:                                                      # A = B.4 v2 s42 (chi doc): C - A chi co o s42
    _ka = kiem_ke(tuple(_TH['a_s42']['gts']), mau=r'p_gopA_[a-z_]+_s42\.npy')[1].get(f'p_gopA_{_DAU_TH}_s42.npy') or []
    assert _ka, f"thieu p_gopA_{_DAU_TH}_s42.npy trong {'/'.join(_TH['a_s42']['gts'])} -> DUNG"
    A_S42_TH = _so3(_gop_v(np.load(tai(_ka[0])).astype('float32')))
    assert abs(A_S42_TH['toan'] - float(_TH['a_s42']['moc_gop'])) < 5e-4, f"cong: A s42 tinh lai {A_S42_TH['toan']:.4f} KHAC moc -> DUNG"
_f_th = lambda r: f"{r['toan']:>8.4f}{r['nua_A']:>8.4f}{r['nua_B']:>8.4f}"
print(f"\n=== TONG HOP NHANH C - {_DAU_TH} (val {len(_IVA):,} anh, gop view; hai nua theo benh nhan, hat {HAT_NUA_S}) | seed co: "
      f"{', '.join(f's{s}' for s in SD_TH)} | chua xong: {', '.join(f's{s}' for s in _chua_th) or '-'} ===")
print(f"{'seed':<16}{'gop toan':>9}{'nua A':>8}{'nua B':>8}{'head':>8}{'medium':>8}{'tail':>8}  |{'TB toan':>9}{'nua A':>8}{'nua B':>8}")
for s in SD_TH:
    g, t = KQ_TH[s]['gop'], KQ_TH[s]['tb']
    print(f"{'s' + str(s) + (' (goi o cat)' if s == 42 else ''):<16}{g['toan']:>9.4f}{g['nua_A']:>8.4f}{g['nua_B']:>8.4f}"
          + ''.join(f"{g['tier'][x]:>8.4f}" for x in ('head', 'medium', 'tail')) + f"  |{t['toan']:>9.4f}{t['nua_A']:>8.4f}{t['nua_B']:>8.4f}")
print(f"{'TB +- SD':<16}" + ''.join(f"{TK_TH['gop'][f][0]:>8.4f}+-{TK_TH['gop'][f][1]:.4f}" for f in ('toan', 'nua_A', 'nua_B'))
      + f"  (n = {len(SD_TH)} seed; TB du doan: " + ' / '.join(f"{TK_TH['tb'][f][0]:.4f}+-{TK_TH['tb'][f][1]:.4f}" for f in ('toan', 'nua_A', 'nua_B')) + ')')
print(f"bo gop {len(SD_TH)} seed (TB du doan cac seed): gop A{_f_th(KQ_BO_TH['gop'])} | TB du doan{_f_th(KQ_BO_TH['tb'])}")
if A_S42_TH is not None:
    print(f"C - A o s42 (A = B.4 v2 {A_S42_TH['toan']:.4f}; seed khac CHUA co A): toan {KQ_TH[42]['gop']['toan'] - A_S42_TH['toan']:+.4f} | nua A "
          f"{KQ_TH[42]['gop']['nua_A'] - A_S42_TH['nua_A']:+.4f} | nua B {KQ_TH[42]['gop']['nua_B'] - A_S42_TH['nua_B']:+.4f}")
_ts_th = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
_te_co = [s for s in SD_TH if _te_th.get(s) is not None]
if _te_co == SD_TH:
    ghi(f"pte_gopA_{_DAU_TH}_bo_gop_s{'_'.join(map(str, SD_TH))}.npy", lambda q: np.save(q, np.mean([_te_th[s] for s in SD_TH], 0).astype('float32')))
ghi(f'tong_hop_nhanh_c_{_ts_th}.json', lambda q: ghi_json(q, dict(
    t=_ts_th, dau=_DAU_TH, seed_notebook=SEED, seed_co=SD_TH, seed_chua=_chua_th, ket_qua={str(s): v for s, v in KQ_TH.items()}, tk=TK_TH,
    bo_gop=KQ_BO_TH, a_s42=A_S42_TH, nguon={str(s): v for s, v in _nguon_th.items()}, pte_bo_gop=(_te_co == SD_TH),
    ghi_chu='Tuan 29/10: giu nhanh C, 5 seed; KHONG luat; KHONG doc nhan test (pte chi ghi).')))
print(f"da ghi tong_hop_nhanh_c_{_ts_th}.json" + (f" + pte_gopA bo gop {len(SD_TH)} seed (KHONG cham)" if _te_co == SD_TH else '') + ' (len Drive o C9)')
del _TH, _g_th, _tb_th, _te_th, _te_co, _f_th, _tk_th
