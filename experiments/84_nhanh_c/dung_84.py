# -*- coding: utf-8 -*-
"""Bo dung module 84 - nhanh C chay theo seed (Tuan duyet 29/10; docs/PLAN_SAU_78Y.md §A.4v, docs/SPEC_SAU_78Y.md §B.2ab; Event 2026-10-29
   "Tuan giu nhanh C ...").

   Sinh nhanh_c/buoc/<ten>.py (mot o notebook DA KIEM = mot tep) + nhanh_c/thu_tu.py (thu tu + sha256), va notebook moi seed (chi goi
   nhanh_c.buoc). Nguon:
     Phan 1 = dung_78x_cat.phan1_o() (Phan 1 luot goi o cat s42 da chay) + va 'nhanh C' theo seed;
     Phan 2 / 3 = dung_78x_d4.phan_78z_d4 / phan_78z_gts_d4 (ten luot giu cho -> CAU_HINH_C) + doan skip7 (shared_lo qua token_o7, 4 dau chuan)
       + va nhanh C (cong cau truc checkpoint C, EP_BB dinh VAL bang nhau -> muon hon, ten tep theo SEED, thu muc stage4_nb84) + TONG_HOP;
     o chuyen phan = dung_78x_d4.O_CHUYEN (+ cong suc khoe nhanh C truoc phan 2).
   KHONG sua bo dung / khoi cua 78. Chay: python dung_84.py [--notebook]. Kiem: python test_84.py. Day GitHub: python day_nhanh_c.py."""
import hashlib, importlib.util, io, json, os, re, sys, contextlib

D = os.path.dirname(os.path.abspath(__file__))
D78 = os.path.join(os.path.dirname(D), '78_chi_anh')
for _p in (D78, os.path.join(D78, 'khoi')):
    if _p not in sys.path: sys.path.insert(0, _p)
with contextlib.redirect_stdout(io.StringIO()):
    import dung_78 as B
    import dung_78x as X
    import dung_78z as Z
    import dung_78x_d4 as D4
    import dung_78x_cat as K
_spec = importlib.util.spec_from_file_location('nhanh_c_cau_hinh_dung', os.path.join(D, 'nhanh_c', 'cau_hinh.py'))
CH = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(CH)          # KHONG nap __init__ cua goi (thu_tu.py do bo dung sinh)

GOI = os.path.join(D, 'nhanh_c')
THU_MUC_BUOC = os.path.join(GOI, 'buoc')
TEP_DA_DAY = os.path.join(D, 'da_day.json')          # ghi boi day_nhanh_c.py: commit da day + blob tung tep
REPO_URL = 'https://github.com/AIVIETNAM-AIO-Tuan/TransferLearning-GPUtask.git'
NHANH_GIT = 'nhanh-c'
SEEDS_NB = (43, 44, 45, 46)                          # notebook chay moi (s42 = luot goi o cat da chay)
SEED_THU = 43
RX, RZ, RG = 'RUN_X_GIU_CHO', 'RUN_Z_GIU_CHO', 'RUN_GTS_GIU_CHO'   # ten luot giu cho -> CAU_HINH_C
TEN_Z_GIU = '84_nhanh_c_s<seed>.ipynb'


_DONG_GOC_ID = ("GOC_ID_TAY = os.environ.get('CXRLT_GOC_ID', '')   # nhanh C: ID cxrlt_2024 do o tai goi cua notebook dat (KHONG de trong ma cong "
                "khai); rong -> khoi tim theo duong MIMIC_CXR/planB_out/cxrlt_2024\n")


def goc_id():
    '''ID thu muc cxrlt_2024 (lay tu C0 goc cua bo dung 78) - CHI dua vao notebook (trong may / Colab), KHONG vao goi day GitHub.'''
    with contextlib.redirect_stdout(io.StringIO()):
        c0 = K.phan1_o(False)['C0']
    return re.search(r"^GOC_ID_TAY = '([^']+)'", c0, re.M).group(1)


def chen_dau(code, khoi):
    '''Chen `khoi` ngay TRUOC dong ma dau tien cua o (sau khoi chu thich tieu de).'''
    L = code.split('\n')
    i = next(k for k, l in enumerate(L) if l.strip() and not l.lstrip().startswith('#'))
    return '\n'.join(L[:i] + [khoi.rstrip('\n')] + L[i:])


# ================================================================================================ Phan 1 (78x goi o cat, theo seed)
NHANH_C_C0 = r"""# ---- nhanh C (Tuan duyet 29/10; docs/SPEC_SAU_78Y.md B.2ab; PLAN A.4v): cau hinh theo seed do nhanh_c.buoc(ten, globals(), seed) dat vao
#      CAU_HINH_C (seed, RUN_TAG, thu muc Drive stage4_nb84, moc C s42). Cong thuc = Phan 1 luot goi o cat s42 NGUYEN VAN tru dong 'nhanh C' ----
assert isinstance(globals().get('CAU_HINH_C'), dict) and CAU_HINH_C.get('nhanh') == 'C', 'chay qua nhanh_c.buoc(...) (dat CAU_HINH_C) -> DUNG'
LUAT_THU_S44 = False   # nhanh C: luat thu seed 44 cua J v3b (Event 2026-10-16) KHONG ap
"""

C9_C = r"""# ---- nhanh C (Tuan duyet 29/10; SPEC B.2ab): CONG SUC KHOE truoc phan 2 (thay sang loc so B.4 v2) - VAL trong luot (EMA, muc anh, nhat ky
#      C6) TB ep1-6 cua luot nay so C s42 (nhat ky stage4_nb78x/r78x_J_v3b_dp_ema_v2_o_s42, CHI doc qua API; TB tinh lai = moc +- 5e-4): thap
#      hon moc QUA nguong -> YEU (o chuyen phan dung; Tuan dat TIEP_TUC_C = True trong notebook roi chay lai o chuyen neu quyet chay tiep) ----
SUC_KHOE_C = None
if not THU_NHANH:
    _kk_sk = kiem_ke(tuple(CAU_HINH_C['moc_x']), mau=r'epoch_log_J_bert_dong_ep\d+\.json')[1].get(f'epoch_log_J_bert_dong_ep{EPOCHS}.json') or []
    assert _kk_sk, f"{'/'.join(CAU_HINH_C['moc_x'])}: thieu epoch_log_J_bert_dong_ep{EPOCHS}.json -> KHONG kiem suc khoe duoc, DUNG"
    with open(tai(_kk_sk[0]), encoding='utf-8') as _f_sk: _lg_sk = json.load(_f_sk)
    assert all(str(e) in _lg_sk and str(e) in LOG for e in range(EPOCHS)), 'cong suc khoe: nhat ky thieu epoch -> DUNG'
    _moc_sk = [float(_lg_sk[str(e)]['mAP_anh']) for e in range(EPOCHS)]
    _nay_sk = [float(LOG[str(e)]['mAP_anh']) for e in range(EPOCHS)]
    _mt_sk, _ng_sk = float(CAU_HINH_C['suc_khoe']['moc_tb']), float(CAU_HINH_C['suc_khoe']['nguong'])
    assert abs(float(np.mean(_moc_sk)) - _mt_sk) < 5e-4, f'cong: TB VAL C s42 tinh lai {np.mean(_moc_sk):.4f} KHAC moc {_mt_sk} -> DUNG'
    _ch_sk = float(np.mean(_nay_sk) - np.mean(_moc_sk))
    SUC_KHOE_C = dict(nay=_nay_sk, moc=_moc_sk, tb_nay=float(np.mean(_nay_sk)), tb_moc=float(np.mean(_moc_sk)), chenh=_ch_sk, nguong=_ng_sk,
                      ket_luan=('DAT' if _ch_sk >= -_ng_sk else 'YEU'), md5_moc=_kk_sk[0]['md5'], seed=SEED)
    ghi('suc_khoe_c.json', lambda q: ghi_json(q, SUC_KHOE_C))
    _ok_sk = dong_bo()
    print(f"nhanh C s{SEED} CONG SUC KHOE: VAL trong luot (EMA) TB ep1-{EPOCHS} {SUC_KHOE_C['tb_nay']:.4f} - C s42 {SUC_KHOE_C['tb_moc']:.4f} = "
          f"{_ch_sk:+.4f} -> {SUC_KHOE_C['ket_luan']} (>= -{_ng_sk} DAT; thap hon YEU) | tung epoch nay "
          + ' '.join(f'{v:.4f}' for v in _nay_sk) + ' / C s42 ' + ' '.join(f'{v:.4f}' for v in _moc_sk)
          + ('' if _ok_sk else ' | !! suc_khoe_c.json chua len Drive (chay lai C9)'))
    del _kk_sk, _f_sk, _lg_sk, _moc_sk, _nay_sk, _mt_sk, _ng_sk, _ch_sk, _ok_sk
"""

O_CHUYEN_1_C = r"""# ==== nhanh C: CONG SUC KHOE truoc khi sang phan 2 - DAT -> sang phan 2; YEU -> DUNG (bao Claude); Tuan quyet chay tiep: chay mot o
#      `TIEP_TUC_C = True` trong notebook roi chay lai o nay ====
_sk_chuyen = globals().get('SUC_KHOE_C') or {}
assert _sk_chuyen.get('ket_luan') == 'DAT' or (globals().get('TIEP_TUC_C') is True and _sk_chuyen.get('ket_luan') == 'YEU'), (
    f"CONG SUC KHOE {_sk_chuyen.get('ket_luan')} (chenh {_sk_chuyen.get('chenh')}) -> KHONG sang phan 2. YEU: bao Claude; Tuan quyet chay tiep: "
    'dat TIEP_TUC_C = True roi chay lai o nay')
del _sk_chuyen
"""


def va_p1(P1):
    '''Va nhanh C len Phan 1 cua luot goi o cat (moi doan cu DUNG 1 lan).'''
    P1 = dict(P1)
    c0 = P1['C0']
    P1['C0'] = chen_dau(B.va(c0, [
        (Z._mot_dong(c0, 'THU_NHANH = False'),
         "THU_NHANH = bool(CAU_HINH_C['thu'])   # nhanh C: True = ban thu nhanh (2,048 anh train, 1 epoch, 1,024 anh val) -> RUN_TAG + '_thu'\n"),
        (Z._mot_dong(c0, f"RUN_TAG = '{K.RT_O}'"),
         "RUN_TAG = CAU_HINH_C['run_x'] + ('_thu' if THU_NHANH else '')   # nhanh C: r84x_C_s<seed> (Phan 1 = luot goi o cat s42, theo seed)\n"),
        (Z._mot_dong(c0, 'SEED = 42 '), "SEED = int(CAU_HINH_C['seed'])                         # nhanh C: seed cua luot (THAY SEED 42 cua 38a o tren)\n"),
        (Z._mot_dong(c0, "DRV_DUONG = ('stage4_nb78x', RUN_TAG)"),
         "DRV_DUONG = (CAU_HINH_C['tm'], RUN_TAG)     # nhanh C: thu muc CUA module 84 (stage4_nb84); 78x / J v3b s42 / C s42 CHI DOC qua API\n"),
        ("SEED == int(CFG_D4['seed'])", "SEED in tuple(CAU_HINH_C['seeds'])"),
        (Z._mot_dong(c0, "GOC_ID_TAY = '"), _DONG_GOC_ID),
    ], 'nhanh C p1 C0'), NHANH_C_C0)
    P1['C6'] = B.va(P1['C6'], [
        ('if _kk_lo and not THU_NHANH:\n',
         'if _kk_lo and not THU_NHANH and SEED == 42:                       # nhanh C: ghep cap dau van lo CHI seed 42 (cap J v3b s42)\n'),
        ('del _kk_lo\nfor ep in range(ep0, _EP_CHAY):\n',
         "del _kk_lo\nif LO_S42 is None and not THU_NHANH: print(f'nhanh C s{SEED}: cong ghep cap TAT (khong co cap J v3b s{SEED}; khoi dong luong + "
         "lambda van chep J v3b s42)')\nfor ep in range(ep0, _EP_CHAY):\n"),
        ('    if SEED == 44 and ep + 1 in (1, 2):', '    if LUAT_THU_S44 and SEED == 44 and ep + 1 in (1, 2):'),
    ], 'nhanh C p1 C6')
    c7 = P1['C7']
    P1['C7'] = B.va(c7, [
        (Z._mot_dong(c7, "_DOI7 = tuple(CFG_D4['ghep_cap'])"),
         "_DOI7 = tuple(CFG_D4['ghep_cap']) if SEED == 42 else tuple(CAU_HINH_C['moc_x'])   # nhanh C: seed khac 42 -> canh C s42 (KHONG ghep cap)\n"),
        (Z._mot_dong(c7, '_GHEP7 = True'), '_GHEP7 = SEED == 42                                    # nhanh C: chi s42 ghep cap (cung seed / lo / khoi dong)\n'),
        ("      + '(GHEP CAP: cung seed, thu tu lo, khoi dong, lambda, lich 6 epoch)')\n",
         "      + ('(GHEP CAP: cung seed, thu tu lo, khoi dong, lambda, lich 6 epoch)' if _GHEP7 else '(KHONG ghep cap: khac seed; cung khoi dong luong + "
         "lambda J v3b s42, cung lich 6 epoch)'))\n"),
        ("          + (' -> TRUNG het' if LO_TRUNG7 else ' -> ⚠ LECH: bao Claude (thu tu lo khac)'))\n",
         "          + (' -> TRUNG het' if LO_TRUNG7 else (' -> ⚠ LECH: bao Claude (thu tu lo khac)' if _GHEP7 else ' -> lech (khac seed, dung nhu "
         "mong doi)')))\n"),
    ], 'nhanh C p1 C7')
    assert P1['C9'].count(K.C9_O) == 1
    P1['C9'] = P1['C9'].replace(K.C9_O, C9_C)
    return P1


# ================================================================================================ Phan 2 / 3 (78z, theo seed)
NHANH_C_C0Z = r"""# ---- nhanh C (Tuan duyet 29/10; SPEC B.2ab): cau hinh theo seed (CAU_HINH_C do nhanh_c.buoc dat); backbone = Phan 1 nhanh C cua CHINH seed
#      nay o stage4_nb84 (TM_BB) ----
assert isinstance(globals().get('CAU_HINH_C'), dict) and CAU_HINH_C.get('nhanh') == 'C', 'chay qua nhanh_c.buoc(...) (dat CAU_HINH_C) -> DUNG'
TM_BB = CAU_HINH_C['tm_x']
"""
_C0Z_EP_C = ("EP_BB, EP_BB_TAY = None, None   # nhanh C (Tuan 29/10 \"epoch cuoi truoc khi bi overfitting\"): EP_BB = dinh VAL EMA trong luot, bang nhau "
             "-> epoch MUON hon; dat o cuoi C0\n")
_C0Z_IN_C = "print(f'78y nhanh C s{SEEDS[0]}: backbone {TM_BB}/{BB_RUN} (EP_BB = dinh VAL EMA trong luot, cuoi C0) | SEEDS {SEEDS}')\n"
C0Z_EP_C = r"""# ---- nhanh C (Tuan 29/10; luat 27/10 "doc dong cung o DINH VAL trong luot cua CHINH luot"; SPEC B.2ab; kiem dau vao nhu Codex muc 54): EP_BB =
#      epoch co VAL EMA trong luot (dau J, muc anh, toan val; nhat ky Phan 1) CAO NHAT SAU KHI LAM TRON 4 chu so; bang nhau -> epoch MUON hon
#      (epoch cuoi truoc khi sut); moi gia tri huu han trong [0, 1]; EP_BB_TAY: None hoac so nguyen 1-6 -> ghi de. Nguon chon -> CHON_EP_C ->
#      nguon_backbone.json (Phan 3 doi chieu) ----
_kk_ep = kiem_ke((TM_BB, BB_RUN), mau=r'epoch_log_J_bert_dong_ep\d+\.json|rs_J_bert_dong_ep\d+\.pt')[1]
_lg_ep_ds = _kk_ep.get('epoch_log_J_bert_dong_ep6.json') or []
assert _lg_ep_ds, f'thieu epoch_log_J_bert_dong_ep6.json trong cxrlt_2024/{TM_BB}/{BB_RUN} -> Phan 1 chua xong / chua dong_bo -> DUNG'
with open(tai(_lg_ep_ds[0]), encoding='utf-8') as _f_ep: _lg_ep = json.load(_f_ep)
_tho_ep = [(_lg_ep.get(str(e)) or {}).get('mAP_anh') for e in range(6)]
assert all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0.0 <= v <= 1.0 for v in _tho_ep), (
    f'nhat ky Phan 1: VAL EMA ep1-6 thieu epoch / khong huu han / ngoai [0, 1] ({_tho_ep}) -> DUNG')
assert EP_BB_TAY is None or (type(EP_BB_TAY) is int and 1 <= EP_BB_TAY <= 6), f'EP_BB_TAY = {EP_BB_TAY!r}: phai la None hoac so nguyen 1-6 -> DUNG'
VAL_EMA_P1 = [round(float(v), 4) for v in _tho_ep]
EP_BB_DINH = len(VAL_EMA_P1) - VAL_EMA_P1[::-1].index(max(VAL_EMA_P1))   # lan CUOI cua max: bang nhau -> epoch MUON hon (Tuan 29/10)
EP_BB = EP_BB_DINH if EP_BB_TAY is None else EP_BB_TAY
BB_TEP = f'rs_J_bert_dong_ep{EP_BB}.pt'
assert _kk_ep.get(BB_TEP), f'thieu {BB_TEP} trong cxrlt_2024/{TM_BB}/{BB_RUN} (rs_ nhe cua ep{EP_BB}) -> DUNG'
CHON_EP_C = dict(nhat_ky=dict(ten='epoch_log_J_bert_dong_ep6.json', id=_lg_ep_ds[0].get('id'), md5=_lg_ep_ds[0].get('md5')),
                 val_ema_tho=[float(v) for v in _tho_ep], val_ema_lam_tron=VAL_EMA_P1, dinh=EP_BB_DINH, tay=EP_BB_TAY, ep_bb=EP_BB,
                 quy_tac=('TAY qua EP_BB_TAY' if EP_BB_TAY is not None else
                          'dinh VAL EMA trong luot (dau J, muc anh, toan val), lam tron 4 chu so, bang nhau -> epoch MUON hon (Tuan 29/10)'))
print(f"78z nhanh C s{SEEDS[0]}: EP_BB = {EP_BB} ({'TAY (EP_BB_TAY)' if EP_BB_TAY is not None else 'dinh VAL EMA trong luot, bang nhau -> muon hon'}; dinh "
      f"= ep{EP_BB_DINH}) | VAL EMA Phan 1 ep1-6 " + ' '.join(f'{v:.4f}' for v in VAL_EMA_P1) + ' | tho ' + ' '.join(f'{float(v):.6f}' for v in _tho_ep)
      + f" | nhat ky md5 {CHON_EP_C['nhat_ky']['md5']}")
del _kk_ep, _lg_ep_ds, _lg_ep, _tho_ep
"""
_C0Z_NBB_C = "NGUON_BB['chon_ep'] = CHON_EP_C   # nhanh C: cach chon epoch vao nguon_backbone.json (Phan 3 doi chieu)\n"
C0Z_N2_C = r"""# ---- nhanh C Phan 3: backbone + cach chon epoch PHAI TRUNG nguon Phan 2 (nguon_backbone.json cua stage4_nb84/r84z_C_s<seed>) - KHONG gop
#      trong so dau doc da train tren backbone / epoch khac ----
_du_n2 = (CAU_HINH_C['tm'], CAU_HINH_C['run_z'])
_kk_n2 = kiem_ke(_du_n2, mau=r'nguon_backbone\.json')[1].get('nguon_backbone.json') or []
assert _kk_n2, f"thieu nguon_backbone.json cua Phan 2 ({'/'.join(_du_n2)}) -> Phan 2 chua chay / chua dong_bo -> DUNG"
with open(tai(_kk_n2[0]), encoding='utf-8') as _f_n2: _n2 = json.load(_f_n2)
_lech_n2 = {k: (_n2.get(k), NGUON_BB.get(k)) for k in ('duong', 'tep', 'ep', 'md5', 'cau_hinh', 'chon_ep') if _n2.get(k) != NGUON_BB.get(k)}
assert not _lech_n2, f'Phan 3: backbone / cach chon epoch KHAC Phan 2 {_lech_n2} -> DUNG (KHONG gop trong so dau doc train tren backbone khac)'
print(f"78z nhanh C s{SEEDS[0]} Phan 3: backbone = nguon Phan 2 (ep{EP_BB}, md5 {NGUON_BB['md5']}, cach chon epoch trung)")
del _du_n2, _kk_n2, _f_n2, _n2, _lech_n2
"""
C0Z_C = r"""
# ---- nhanh C (Tuan duyet 29/10; SPEC B.2ab): cau hinh o MONG DOI = O_CFG Phan 1 nhanh C (= luot goi o cat); 4 dau chuan (doc 1 view, duong skip) ----
O_CFG_MONG = __O_CFG__
assert ARMS_GIU == ['mldec', 'mldec_hr', 'fus_tb_hr', 'fus_attn_hr'] and O_CFG_MONG['ten'] == 'o_luoi_phong518_skip7_gop6' \
    and int(O_CFG_MONG['n_o']) == 5 and int(O_CFG_MONG['k_skip']) == 8, 'nhanh C 78z: cau hinh lech -> DUNG'
print(f"78z nhanh C s{SEEDS[0]}: backbone {TM_BB}/{BB_RUN} co skip block {int(O_CFG_MONG['k_skip']) - 1} -> {O_CFG_MONG['k_skip']} cua "
      f"{O_CFG_MONG['n_o']} o (dong cung) | 4 dau chuan | backbone ep{EP_BB} ({'TAY' if EP_BB_TAY is not None else 'dinh VAL trong luot'})")
"""
C2BZ_C = r"""    # ---- nhanh C (Tuan duyet 29/10; SPEC B.2ab): module o DONG CUNG tu rs_ ep EP_BB (EMA) + cong cau truc nhanh C (cfg_o goi o cat; DU dua_o +
    #      bo_o7 + bo_o11 + gop_o; CHI nap dua_o + bo_o7 cho duong skip, 4 dau doc 1 view) + cong 3' (duong chia doi CO skip = forward day du
    #      CO skip, 8 anh val) ----
    _cfg_o = _cfj.get('cfg_o')
    assert isinstance(_cfg_o, dict) and _cfg_o == O_CFG_MONG, f"backbone_ft.pt cua {BB_RUN}: cfg_o {_cfg_o} KHAC cau hinh nhanh C mong doi -> DUNG (luot khac?)"
    assert all(k in _ck for k in ('dua_o', 'bo_o7', 'bo_o11', 'gop_o')), (
        f'{BB_TEP}: module o KHONG dung cau truc nhanh C (can dua_o + bo_o7 + bo_o11 + gop_o) -> DUNG')
    with _FORK_78Y():   # khoi tao (GHI DE ngay khi nap) KHONG tieu dong chung
        BO_O7, DUA_O = BoChinhO(D_CXR, int(_cfg_o['r_adapter'])), DuaVao(D_CXR)
    for _m_o, _k_o in ((BO_O7, 'bo_o7'), (DUA_O, 'dua_o')): _m_o.load_state_dict(_ck[_k_o], strict=True)
    BO_O7, DUA_O = (_m_o.to(dev).eval().requires_grad_(False) for _m_o in (BO_O7, DUA_O))
    SO_O, K_SKIP_O, KHUC_O = int(_cfg_o['n_o']), int(_cfg_o['k_skip']), int(_cfg_o['khuc'])
    assert K_SKIP_O == GRAD_FROM, f'nhanh C: skip vao block {K_SKIP_O} KHAC block dong cung dau {GRAD_FROM} -> DUNG'
    LUOI_O = luoi_518(dev)
    _G_O, _W_O = luoi_con(hinh_luoi(1, dev))
    with torch.no_grad(), torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        _t7z = token_o7(backbone, _xz, LUOI_O, MEAN, STD, KHUC_O, K_SKIP_O - 1)
        _ijz = skip_o(_t7z, BO_O7, DUA_O, _G_O.expand(len(_xz), -1, -1, -1).contiguous(), _W_O.expand(len(_xz), -1, -1).contiguous())
        LUONG.dat_neo(NEO_T[_rz].float())
        _pz3 = upper_luong_hr(backbone, LUONG, cong_skip(run_blocks(backbone.embeddings(_xz), 0, GRAD_FROM), _ijz), mo_ghi=True)[0]
        LUONG.dat_neo(NEO_T[_rz].float())
        with skip_vao_luong(LUONG, _ijz, K_SKIP_O):
            _pf3 = chay_bb_luong(backbone, LUONG, _xz, mo_ghi=True, grad_bb=False)[0]
    _e3o = float((_pz3 - _pf3).abs().max())
    _ti_o = float((_ijz.float().norm(dim=-1).mean() / _pz.float().norm(dim=-1).mean().clamp_min(1e-12)))
    assert _e3o < 1e-3, f'nhanh C: duong chia doi CO skip lech forward day du CO skip {_e3o:.2e} (cong 3 prime) -> DUNG'
    print(f"78z nhanh C C2b: module o (rs_ ep{EP_BB}, {_cfg_o['ten']}; nap dua_o + bo_o7, bo qua bo_o11 / gop_o) | cong 3': chia doi co skip lech "
          f"{_e3o:.1e} | ||inj|| / ||patch cuoi|| {_ti_o:.4f} | {SO_O} o, khuc {KHUC_O}")
    del _cfg_o, _t7z, _ijz, _pz3, _pf3
"""


def _va_bb_c(c, ten):
    '''Thu muc backbone Phan 1: ('stage4_nb78x', BB_RUN) -> (TM_BB, BB_RUN) - CHI cac dang da biet (code + f-string thong bao).'''
    c = c.replace("('stage4_nb78x', BB_RUN)", '(TM_BB, BB_RUN)').replace("['stage4_nb78x', BB_RUN]", '[TM_BB, BB_RUN]')
    c = c.replace('stage4_nb78x/{BB_RUN}', '{TM_BB}/{BB_RUN}')
    con = [l for l in B.dong_ma(c) if 'stage4_nb78x' in l]
    assert not con, f'{ten}: con stage4_nb78x o dong ma: {con[:2]}'
    return c


def _va_78z_c(P, ten, c6_ten, phan3=False):
    '''Nhanh C chung cho Phan 2 (C6) va Phan 3 (C6S): C0 (ten luot + seed + thu muc + EP_BB dinh VAL muon hon + nguon chon; phan 3 + cong
       trung nguon Phan 2), C2b (module o + cong cau truc C + cong 3'), C6 / C6S (shared_lo qua token_o7). Moi doan cu DUNG 1 lan.'''
    P = dict(P)
    c0 = P['C0']
    L = c0.split('\n')
    i_ep = [k for k, l in enumerate(L) if l.startswith('EP_BB = 4 ')]
    assert len(i_ep) == 1 and L[i_ep[0] + 1].lstrip().startswith('#'), f'{ten} C0: dong EP_BB = 4 / chu thich noi tiep lech'
    nap_bb = Z._mot_dong(c0, "_ik_bb, _kk_bb = kiem_ke(('stage4_nb78x', BB_RUN)")
    nbb = Z._mot_dong(c0, '                so_ban=len(_ds_bb), cau_hinh=')
    run_dong = Z._mot_dong(c0, f"RUN_TAG = '{RG if phan3 else RZ}'")
    va0 = [(Z._mot_dong(c0, "BB_TEP = {'J_v3': 'rs_J_bert_dong_ep4.pt'"), K._C0Z_TEP_S7),
           (L[i_ep[0]] + '\n' + L[i_ep[0] + 1] + '\n', _C0Z_EP_C),
           (Z._mot_dong(c0, "print(f'78y: backbone 78x {BB_RUN}/{BB_TEP} (ep{EP_BB})"), _C0Z_IN_C),
           (K._C0Z_TEST_CU, K._C0Z_TEST_MOI), (nap_bb, C0Z_EP_C + nap_bb), (nbb, nbb + _C0Z_NBB_C),
           (Z._mot_dong(c0, 'THU_NHANH = False'), "THU_NHANH = bool(CAU_HINH_C['thu'])   # nhanh C: luot thu (Phan 2 / 3 chi khi Tuan bat)\n"),
           (run_dong, run_dong.replace(f"'{RG if phan3 else RZ}'", "CAU_HINH_C['run_gts']" if phan3 else "CAU_HINH_C['run_z']", 1)),
           (Z._mot_dong(c0, f"BB_RUN = '{RX}'"), "BB_RUN = CAU_HINH_C['run_x']   # nhanh C: Phan 1 cua CHINH seed nay (TM_BB = stage4_nb84)\n"),
           (Z._mot_dong(c0, 'SEEDS = (42,) '), "SEEDS = (int(CAU_HINH_C['seed']),)   # nhanh C: dau doc CUNG seed voi backbone\n"),
           (Z._mot_dong(c0, 'SEED, SIZE, EPOCHS = 42, 518, 10'), "SEED, SIZE, EPOCHS = int(CAU_HINH_C['seed']), 518, 10\n"),
           ("RES57 = os.path.join(OUT_DIR, 'stage4_nb78z', RUN_TAG)", "RES57 = os.path.join(OUT_DIR, CAU_HINH_C['tm'], RUN_TAG)"),
           (Z._mot_dong(c0, "DRV_DUONG = ('stage4_nb78z', RUN_TAG)"), "DRV_DUONG = (CAU_HINH_C['tm'], RUN_TAG)   # nhanh C: stage4_nb84\n"),
           (Z._mot_dong(c0, "GOC_ID_TAY = '"), _DONG_GOC_ID)]
    if phan3:
        xoa_bb = Z._mot_dong(c0, 'del _ik_bb, _kk_bb, _ds_bb, _ds_cf, _p_nbb')
        va0.append((xoa_bb, xoa_bb + C0Z_N2_C))
    c0 = _va_bb_c(B.va(c0, va0, f'{ten} C0'), f'{ten} C0')
    P['C0'] = chen_dau(K.chen_chan_rac(c0).rstrip('\n') + '\n' + C0Z_C.replace('__O_CFG__', repr(K.O_CFG)), NHANH_C_C0Z)
    P['C2'] = _va_bb_c(P['C2'], f'{ten} C2')
    P['C2B'] = B.va(P['C2B'], [(K._C2BZ_KHOI_CU, K.khoi_o(co_gop=False, s7=True) + '\n\n' + K._C2BZ_KHOI_CU),
                               (K._C2BZ_DEL_CU, C2BZ_C + K._C2BZ_DEL_CU), (K._C2BZ_EP_CU, K._C2BZ_EP_MOI)], f'{ten} C2b')
    P[c6_ten] = B.va(P[c6_ten], [(K._C6Z_LO_CU, K._C6Z_LO_S7)], f'{ten} {c6_ten}')
    return P


def phan2_c():
    with contextlib.redirect_stdout(io.StringIO()):
        P2 = dict(D4.phan_78z_d4('dp_ema', run_bb=RX, run_z=RZ))
    P2['C9'] = P2['C9'] + D4.CO_PHAN_2
    return _va_78z_c(P2, 'nhanh C phan 2', 'C6')


# ---- C6S: ten tep theo SEED (luot goc 78z chi s42), thu muc Phan 2 cua seed nay; dau gop + dau quyet dinh nhu skip7 ----
VA_C6S_SEED = [
    (f"# luot goc (stage4_nb78z/{RZ}),", "# luot goc (stage4_nb84/r84z_C_s<seed>),"),
    ("assert tuple(SEEDS) == (42,) and 'SUB' in globals() and 'YM' in globals() and 'ST_ID' in globals(), 'C6S: can seed 42, SUB, YM, ST_ID -> DUNG'",
     "assert tuple(SEEDS) == (SEED,) and 'SUB' in globals() and 'YM' in globals() and 'ST_ID' in globals(), 'C6S: can seed cua luot, SUB, YM, ST_ID -> DUNG'"),
    (f"_DU_S = ('stage4_nb78z', '{RZ}')   # 78z dot 4 dp_ema", "_DU_S = (CAU_HINH_C['tm'], CAU_HINH_C['run_z'])   # nhanh C: Phan 2 cua CHINH seed nay"),
    (r"mau=r'(w_s42_ep\d+\.pt|p_[a-z0-9_]+_s42_ep\d+\.npy)'", r"mau=rf'(w_s{SEED}_ep\d+\.pt|p_[a-z0-9_]+_s{SEED}_ep\d+\.npy)'"),
    ("[f'w_s42_ep{e}.pt' for e in EP_S] + [f'p_{d}_s42_ep{e}.npy' for d in DAU_S for e in EP_S]",
     "[f'w_s{SEED}_ep{e}.pt' for e in EP_S] + [f'p_{d}_s{SEED}_ep{e}.npy' for d in DAU_S for e in EP_S]"),
    ("np.load(_tep_s(f'p_{d}_s42_ep{e}.npy'))", "np.load(_tep_s(f'p_{d}_s{SEED}_ep{e}.npy'))"),
    ("torch.load(_tep_s(f'w_s42_ep{e}.pt')", "torch.load(_tep_s(f'w_s{SEED}_ep{e}.pt')"),
    (f"assert w.get('run') == '{RZ}' and", "assert w.get('run') == CAU_HINH_C['run_z'] and"),
    ("f'w_s42_ep{e}.pt: run / arm / epoch KHAC luot goc -> DUNG'", "f'w_s{SEED}_ep{e}.pt: run / arm / epoch KHAC luot goc -> DUNG'"),
    ("    ghi(f'p_gopA_{d}_s42.npy', lambda q, v=_pvA[d]: np.save(q, v))", "    ghi(f'p_gopA_{d}_s{SEED}.npy', lambda q, v=_pvA[d]: np.save(q, v))"),
    ("    ghi(f'p_gopB_{d}_s42.npy', lambda q, v=_pvB[d]: np.save(q, v))", "    ghi(f'p_gopB_{d}_s{SEED}.npy', lambda q, v=_pvB[d]: np.save(q, v))"),
    ("ghi('w_gopA_s42.pt', lambda q:", "ghi(f'w_gopA_s{SEED}.pt', lambda q:"),
    ("for d in DAU_S: ghi(f'pte_gopA_{d}_s42.npy', lambda q, v=_pteA[d]: np.save(q, v))",
     "for d in DAU_S: ghi(f'pte_gopA_{d}_s{SEED}.npy', lambda q, v=_pteA[d]: np.save(q, v))"),
    ("tep_ghi=[f'p_gopA_{d}_s42.npy' for d in DAU_S] + [f'p_gopB_{d}_s42.npy' for d in DAU_S]",
     "tep_ghi=[f'p_gopA_{d}_s{SEED}.npy' for d in DAU_S] + [f'p_gopB_{d}_s{SEED}.npy' for d in DAU_S]"),
    ("    + ['w_gopA_s42.pt'] + [f'pte_gopA_{d}_s42.npy' for d in DAU_S],", "    + [f'w_gopA_s{SEED}.pt'] + [f'pte_gopA_{d}_s{SEED}.npy' for d in DAU_S],"),
    ("tep p_gop* + w_gopA_s42.pt + {len(DAU_S)} tep pte_gopA (KHONG cham) -> stage4_nb78z/{RUN_TAG} '",
     "tep p_gop* + w_gopA_s{SEED}.pt + {len(DAU_S)} tep pte_gopA (KHONG cham) -> {DRV_DUONG[0]}/{RUN_TAG} '"),
]

TONG_HOP_C = r"""# ==== TONG HOP · NHANH C: CAC SEED DA XONG (Tuan duyet 29/10; docs/SPEC_SAU_78Y.md B.2ab) - chay SAU C6S, TRUOC C9 ====
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
"""


def phan3_c():
    with contextlib.redirect_stdout(io.StringIO()):
        P3 = dict(D4.phan_78z_gts_d4('dp_ema', run_bb=RX, run_z=RZ, run_gts=RG, ten_z=TEN_Z_GIU))
    P3 = _va_78z_c(P3, 'nhanh C phan 3', 'C6S', phan3=True)
    P3['C6S'] = B.va(P3['C6S'], [(K._C6S_DAU_CU, K._C6S_DAU_S7), (Z._mot_dong(P3['C6S'], 'DAU_QD_S = '), K._C6S_QD_S7)] + VA_C6S_SEED,
                     'nhanh C phan 3 C6S')
    P3['TONG_HOP'] = TONG_HOP_C
    return P3


def phan_c():
    '''Ba dict o code (phan 1 / 2 / 3) cua nhanh C (chung cho moi seed: seed doc tu CAU_HINH_C).'''
    with contextlib.redirect_stdout(io.StringIO()):
        P1 = va_p1(K.phan1_o(False))
    P2, P3 = phan2_c(), phan3_c()
    for ph, P in (('p1', P1), ('p2', P2), ('p3', P3)):
        for k, v in P.items():
            if not isinstance(v, str): continue
            for t in (RX, RZ, RG):
                assert t not in v, f'{ph}.{k}: con ten giu cho {t}'
            ma = '\n'.join(B.dong_ma(v))
            assert not re.search(r"'r78[xz]_[A-Za-z0-9_]*s4[3-6]", ma), f'{ph}.{k}: ten luot 78 theo seed moi?'
    for s in (NHANH_C_C0, C9_C, O_CHUYEN_1_C, NHANH_C_C0Z, _C0Z_EP_C, _C0Z_IN_C, C0Z_EP_C, _C0Z_NBB_C, C0Z_N2_C, C0Z_C, C2BZ_C, TONG_HOP_C):
        assert s.split('\n', 1)[-1].isascii(), s[:80]
    return P1, P2, P3


# ================================================================================================ markdown + thu tu o (nhu notebook 78 cu)
MD_C1 = dict(K.MD_O1)
MD_C1['C0'] = K.MD_O1['C0'].rstrip('\n') + ('\n\nNhánh C: cấu hình theo seed (`CAU_HINH_C` do `nhanh_c.buoc` đặt): `SEED`, `RUN_TAG` `r84x_C_s<seed>`, thư mục '
                                            '`stage4_nb84`; khởi động luồng + λ vẫn chép J v3b s42 (như lượt gói s42).\n')
MD_C1['C6'] = K.MD_O1['C6'].rstrip('\n') + ('\n\nNhánh C: cổng ghép cặp dấu vân lô chỉ ở s42; luật thử seed 44 của J v3b tắt.\n')
MD_C1['C9'] = D4.MD_D4['C9'].rstrip('\n') + ('\n\nNhánh C: **cổng sức khoẻ** — VAL trong lượt TB ep1–6 thấp hơn C s42 (0,2619) quá 0,01 → `YEU`, ô chuyển '
                                             'phần dừng (báo Claude); Tuấn quyết chạy tiếp: chạy ô `TIEP_TUC_C = True` rồi chạy lại ô chuyển.\n')


def _ten_buoc(phan, khoa):
    return f'p{phan}_{khoa}'


def danh_sach_o(thu=False):
    '''Danh sach o kieu notebook cu (md + ('code', ten buoc)) va dict ten buoc -> nguon. Phan 2 / 3: o trung nguon dung chung tep z_<khoa>.'''
    P1, P2, P3 = phan_c()
    nguon, o = {}, []
    ph_dang = [1]

    def them_code(ten, code):
        if ten in nguon: assert nguon[ten] == code, ten
        nguon[ten] = code
        o.append(('code', ten, ph_dang[0]))

    c1 = X._cells(P1, X.THU_TU_JV2, MD_C1, D4.md_tieu_de_d4('dp_ema', thu), 'nhanh C phan 1')
    c1 = D4._tieu_de_phan(c1, 1, f"# Phần 1 · `78x` nhánh C — `r84x_C_s<seed>`{' (bản thử nhanh)' if thu else ''} (6 epoch)")
    ma1 = {v: k for k, v in P1.items() if isinstance(v, str)}
    for c in c1:
        if c['cell_type'] == 'markdown': o.append(('md', B.src(c)))
        else: them_code(_ten_buoc(1, ma1[B.src(c)]), B.src(c))
    if thu:
        return o, nguon
    o += [('md', '## Chuyển phần 1 → 2 (cổng sức khoẻ nhánh C)\n')]
    them_code('chuyen_1_2', O_CHUYEN_1_C + D4.o_chuyen_b4v2(1, 2))
    with contextlib.redirect_stdout(io.StringIO()):
        c2 = Z._cells_78zj_bb(P2, D4.md_tieu_de_78z_d4('dp_ema'), '\n\nNhánh C: `C0` chọn `EP_BB` = đỉnh VAL EMA trong lượt (bằng nhau → epoch muộn hơn); `C2b` '
                              'nạp `dua_o` + `bo_o7` đông cứng + cổng cấu trúc nhánh C + cổng 3′; 4 đầu chuẩn (đọc 1 view, đường skip).\n', 'nhanh C phan 2')
        c3 = Z._cells_gts(P3, D4.md_tieu_de_78z_gts_d4('dp_ema'), 'nhanh C phan 3')
    c2 = D4._tieu_de_phan(c2, 2, '# Phần 2 · `78z` chuẩn trên backbone nhánh C — `r84z_C_s<seed>`')
    i9 = max(i for i, c in enumerate(c3) if c['cell_type'] == 'code')
    assert B.src(c3[i9]) == P3['C9S']
    c3 = (c3[:i9 - 1] + [B.cell_md('## TỔNG HỢP · Các seed nhánh C đã xong — `fus_attn_hr` gộp trọng số + TB dự đoán, TB ± độ lệch, bộ gộp các seed '
                                   '(chạy trước `C9`)\n'), B.cell_code(P3['TONG_HOP'])] + c3[i9 - 1:])
    c3 = D4._tieu_de_phan(c3, 3, '# Phần 3 · Gộp trọng số đầu đọc ep1–10 + tổng hợp các seed — `r84z_C_s<seed>_gts`')
    for ph, cc, P in ((2, c2, P2), (3, c3, P3)):
        ph_dang[0] = ph
        ma = {v: k for k, v in P.items() if isinstance(v, str)}
        for c in cc:
            if c['cell_type'] == 'markdown':
                o.append(('md', B.src(c).replace(K.MD_RS_EP4_S7[0], '(rs_ đỉnh VAL trong lượt)')))
            else:
                k = ma[B.src(c)]
                chung = ph == 2 and P3.get(k) == P2[k] or ph == 3 and P2.get(k) == P3[k]
                them_code(f'z_{k}' if chung else _ten_buoc(ph, k), B.src(c))
        if ph == 2:
            o += [('md', '## Chuyển phần 2 → 3\n')]
            them_code('chuyen_2_3', D4.o_chuyen_b4v2(2, 3))
    return o, nguon


def _phan_cua(o):
    '''Thu tu buoc theo phan (1 / 2 / 3); o chuyen 1 -> 2 thuoc phan 1, 2 -> 3 thuoc phan 2.'''
    out = {1: [], 2: [], 3: []}
    for x in o:
        if x[0] == 'code': out[x[2]].append(x[1])
    return out


TIEU_DE_BUOC = '# ==== nhanh_c buoc {ten} — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====\n'


def noi_dung_buoc(ten, code):
    return TIEU_DE_BUOC.format(ten=ten) + code.rstrip('\n') + '\n'


def sinh_goi(ghi=True):
    '''Sinh buoc/<ten>.py + thu_tu.py. Tra dict duong dan tuong doi (trong goi) -> noi dung.'''
    o, nguon = danh_sach_o(False)
    o_thu, nguon_thu = danh_sach_o(True)
    assert all(nguon[k] == v for k, v in nguon_thu.items()), 'ban thu: o Phan 1 KHAC ban chinh (THU_NHANH doc tu CAU_HINH_C) -> phai trung'
    tep = {f'buoc/{ten}.py': noi_dung_buoc(ten, code) for ten, code in nguon.items()}
    for ten, code in nguon.items():
        compile(code, ten, 'exec')
    phan = _phan_cua(o)
    assert phan[1][-1] == 'chuyen_1_2' and phan[2][-1] == 'chuyen_2_3' and phan[3][-1] == 'p3_C9S' and 'p3_TONG_HOP' in phan[3], phan
    assert set(sum(phan.values(), [])) == set(nguon), 'thu tu thieu / thua buoc'
    bam = {ten: hashlib.sha256(tep[f'buoc/{ten}.py'].encode('utf-8')).hexdigest() for ten in sorted(nguon)}
    tt = ('# -*- coding: utf-8 -*-\n"""Thu tu buoc theo phan + sha256 tung tep buoc/<ten>.py - SINH TU DONG boi experiments/84_nhanh_c/dung_84.py (KHONG sua tay)."""\n'
          f'PHAN = {json.dumps({str(k): v for k, v in phan.items()}, indent=1)}\n'
          'PHAN = {int(k): tuple(v) for k, v in PHAN.items()}\n'
          f'BAM = {json.dumps(bam, indent=1, sort_keys=True)}\n')
    tep['thu_tu.py'] = tt
    if ghi:
        os.makedirs(THU_MUC_BUOC, exist_ok=True)
        cu = {f for f in os.listdir(THU_MUC_BUOC) if f.endswith('.py')}
        for f in cu - {os.path.basename(p) for p in tep if p.startswith('buoc/')}:
            os.remove(os.path.join(THU_MUC_BUOC, f))                  # buoc cu khong con trong bo dung
        for p, s in tep.items():
            dp = os.path.join(GOI, *p.split('/'))
            with open(dp, 'w', encoding='utf-8', newline='\n') as f: f.write(s)
    return tep


# ================================================================================================ notebook moi seed (chi goi nhanh_c.buoc)
def blob_git(b):
    return hashlib.sha1(b'blob %d\0' % len(b) + b).hexdigest()


def tep_goi_tren_dia():
    '''Moi tep .py cua goi (duong dan trong nhanh_c/) -> bytes.'''
    out = {}
    for goc, _, fs in os.walk(GOI):
        if '__pycache__' in goc: continue
        for f in fs:
            if f.endswith('.py'):
                p = os.path.join(goc, f)
                out[os.path.relpath(p, GOI).replace(os.sep, '/')] = open(p, 'rb').read()
    return out


def sha_da_day():
    '''Commit da day (da_day.json) - CHI khi goi tren dia TRUNG tung blob voi commit do; khong -> None.'''
    if not os.path.isfile(TEP_DA_DAY): return None
    dd = json.load(open(TEP_DA_DAY, encoding='utf-8'))
    tren = {p: blob_git(b) for p, b in tep_goi_tren_dia().items()}
    return dd['sha'] if dd.get('blob') == tren else None


O_TAI_GOI = r"""# ==== TAI GOI nhanh_c tu GitHub (nhanh mo coi __NHANH__, GHIM commit) - chay MOI phien, TRUOC moi o khac ====
import os, subprocess, sys
NHANH_C_SHA = '__SHA__'
os.environ['CXRLT_GOC_ID'] = '__GOC_ID__'   # ID thu muc cxrlt_2024 - CHI trong notebook (goi cong khai tren GitHub KHONG chua)
_d_goi = '/content/nhanh_c_goi'
_tep_goi = os.path.normcase(os.path.realpath(os.path.join(_d_goi, 'nhanh_c', '__init__.py')))
_goi_cu = sys.modules.get('nhanh_c')
if _goi_cu is not None:
    if os.path.normcase(os.path.realpath(getattr(_goi_cu, '__file__', ''))) != _tep_goi:
        raise RuntimeError('nhanh_c da nap tu duong dan khac -> Restart session truoc khi tai goi')
    if getattr(_goi_cu, '_NHANH_C_SHA_DA_NAP', None) != NHANH_C_SHA:
        raise RuntimeError('nhanh_c da nap tu commit khac hoac chua duoc loader ghim -> Restart session truoc khi tai goi')
if not os.path.isdir(os.path.join(_d_goi, '.git')):
    subprocess.run(['git', 'clone', '--quiet', '--branch', '__NHANH__', '__URL__', _d_goi], check=True)
subprocess.run(['git', '-C', _d_goi, 'fetch', '--quiet', 'origin', '__NHANH__'], check=True)
subprocess.run(['git', '-C', _d_goi, 'checkout', '--quiet', '--detach', NHANH_C_SHA], check=True)
_head_goi = subprocess.run(['git', '-C', _d_goi, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
assert _head_goi == NHANH_C_SHA, f'goi nhanh_c: HEAD {_head_goi} KHAC commit ghim {NHANH_C_SHA} -> DUNG'
if _d_goi not in sys.path: sys.path.insert(0, _d_goi)
import nhanh_c as NC
if os.path.normcase(os.path.realpath(getattr(NC, '__file__', ''))) != _tep_goi:
    raise RuntimeError('nhanh_c import tu duong dan khac goi vua tai -> Restart session')
assert NC.PHIEN_BAN == '__PHIEN_BAN__', NC.PHIEN_BAN
NC._NHANH_C_SHA_DA_NAP = NHANH_C_SHA
print(f'nhanh_c {NC.PHIEN_BAN} @ {NHANH_C_SHA[:12]} | {len(NC.ds_buoc())} buoc | seed __SEED__' + (' (BAN THU)' if __THU__ else ''))
del _d_goi, _head_goi, _tep_goi, _goi_cu
"""


def o_tai_goi(sha, seed, thu):
    return (O_TAI_GOI.replace('__SHA__', sha).replace('__GOC_ID__', goc_id()).replace('__NHANH__', NHANH_GIT).replace('__URL__', REPO_URL)
            .replace('__PHIEN_BAN__', CH.PHIEN_BAN)
            .replace('__SEED__', str(seed)).replace('__THU__', repr(bool(thu))))


def o_goi(ten, seed, thu):
    return f"import nhanh_c as NC; NC.buoc('{ten}', globals(), seed={seed}{', thu=True' if thu else ''})\n"


def md_dau(seed, thu, sha):
    t = CH.ten_luot(seed)
    if thu:
        return (f"# 84 · Nhánh C seed {seed} — **BẢN THỬ NHANH Phần 1** (`{t['run_x']}_thu`)\n\n"
                "Mã: gói `nhanh_c` trên GitHub (nhánh `nhanh-c`, commit ghim `" + sha[:12] + "`); mỗi ô gọi `nhanh_c.buoc(...)` — mã bước = ô đã kiểm "
                "của lượt gói ô cắt s42, khác đúng các dòng theo seed (`docs/SPEC_SAU_78Y.md` §B.2ab).\n\n"
                "Khác bản chính: `thu=True` (2.048 ảnh train, 1 epoch, 1.024 ảnh val; không ghép cặp, không cổng sức khoẻ). Mục đích: kiểm gói chạy "
                "đúng trên Colab, tốc độ / VRAM.\n\n**Chạy:** GPU **G4**, Restart session → **Run all** (~40 phút, ước). **Gửi Claude:** ô tải gói, dòng "
                "`cong o cat` (`C4`), các dòng `buoc … VRAM dinh … img/s` + dòng epoch + `O CAT` (`C6`).\n")
    return f"""# 84 · Nhánh C seed {seed} — một notebook ba phần (Tuấn duyệt 29/10)

**Kế hoạch:** `docs/PLAN_SAU_78Y.md` §A.4v, `docs/SPEC_SAU_78Y.md` §B.2ab, `Event.md` 2026-10-29 "Tuấn giữ nhánh C …". Mã: gói `nhanh_c` trên GitHub
(nhánh `nhanh-c`, commit ghim `{sha[:12]}`); mỗi ô gọi `nhanh_c.buoc(...)` — mã bước = ô đã kiểm của lượt gói ô cắt s42 (Phần 1) / lượt skip7 (Phần 2–3,
đường skip + 4 đầu chuẩn), khác đúng các dòng "nhanh C" theo seed.

1. **Phần 1 — `78x`** (`{t['run_x']}`): công thức lượt gói s42 (B.4 v2 + 5 ô lưới phóng 518 không gradient + adapter + skip block 7 → 8 + đầu J đọc 6 view);
   seed {seed}; khởi động luồng + λ chép J v3b s42. Cuối phần: **cổng sức khoẻ** (VAL trong lượt TB ep1–6 không thấp hơn C s42 0,2619 quá 0,01).
2. **Phần 2 — `78z` chuẩn** (`{t['run_z']}`): 4 đầu chuẩn trên backbone đọc ở **đỉnh VAL trong lượt** (bằng nhau → epoch muộn hơn); không dự đoán test mỗi epoch.
3. **Phần 3 — gộp trọng số** ep1–10 (`{t['run_gts']}`; đầu quyết định `fus_attn_hr`, luật mức 2 `C6S`) + **tổng hợp các seed đã xong** (TB ± độ lệch, bộ gộp
   các seed, C − A ở s42). Không đọc nhãn test.

**Chạy:** GPU **G4**, Restart session → **Run all**, ~21 giờ (ước: phần 1 ~12, phần 2 ~8, phần 3 ~1). Ngắt giữa chừng: Restart → Run all (phần 1 / 2 chạy
tiếp từ tệp resume). **Đừng tắt runtime trước `Phần 3 · C9`.** Ghi `stage4_nb84/…`.

**Gửi Claude:** ô tải gói; các dòng epoch + `O CAT` của `Phần 1 · C6`; dòng `CONG SUC KHOE` (`Phần 1 · C9`); dòng `EP_BB` (`Phần 2 · C0`); `Phần 2 · C2b`
+ `C7`; `Phần 3 · C6S` + toàn bộ `TỔNG HỢP`."""


def cells_nb(seed, thu, sha):
    o, nguon = danh_sach_o(thu)
    cells = [B.cell_md(md_dau(seed, thu, sha)), B.cell_md('## Tải gói `nhanh_c` (GitHub, commit ghim) — chạy trước mọi ô\n'),
             B.cell_code(o_tai_goi(sha, seed, thu))]
    if thu:
        o = o[1:]                                                    # bo md tieu de phan 1 cua notebook cu (md_dau thay)
    for x in o:
        cells.append(B.cell_md(x[1]) if x[0] == 'md' else B.cell_code(o_goi(x[1], seed, thu)))
    return cells


def ten_nb(seed, thu):
    return f"84_nhanh_c_s{seed}{'_thu' if thu else ''}.ipynb"


def dung_nb(ghi=True):
    sha = sha_da_day()
    assert sha, 'goi tren dia KHAC commit da day (hoac chua day) -> chay day_nhanh_c.py truoc khi dung notebook'
    out = {}
    for seed, thu in [(s, False) for s in SEEDS_NB] + [(SEED_THU, True)]:
        cells = cells_nb(seed, thu, sha)
        out[ten_nb(seed, thu)] = cells
        if ghi:
            nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                               'language_info': {'name': 'python'}, 'accelerator': 'GPU'}, 'nbformat': 4, 'nbformat_minor': 0}
            p = os.path.join(D, ten_nb(seed, thu))
            with open(p, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(nb, f, ensure_ascii=False, indent=1)
                f.write('\n')
            print(f'{ten_nb(seed, thu)}: {len(cells)} o -> {p}')
    return out


if __name__ == '__main__':
    t = sinh_goi(True)
    print(f'nhanh_c: {len(t) - 1} buoc + thu_tu.py -> {GOI}')
    if '--notebook' in sys.argv:
        dung_nb(True)
