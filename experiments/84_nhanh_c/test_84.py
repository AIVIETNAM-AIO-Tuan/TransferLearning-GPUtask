# -*- coding: utf-8 -*-
"""Kiem offline module 84 (goi nhanh_c; docs/SPEC_SAU_78Y.md B.2ab) - CPU, KHONG vong train. Chay: python test_84.py"""
import ast, contextlib, difflib, hashlib, importlib.util, io, json, math, os, re, shutil, subprocess, sys, tempfile, types

import numpy as np

sys.dont_write_bytecode = True
D = os.path.dirname(os.path.abspath(__file__))
D78 = os.path.join(os.path.dirname(D), '78_chi_anh')
_spec = importlib.util.spec_from_file_location('dung_84_kiem', os.path.join(D, 'dung_84.py'))
M = importlib.util.module_from_spec(_spec)
with contextlib.redirect_stdout(io.StringIO()):
    _spec.loader.exec_module(M)
B, K, Z, D4, CH = M.B, M.K, M.Z, M.D4, M.CH

KQ = []


def ck(ten, dk, chi_tiet=''):
    KQ.append((ten, bool(dk)))
    print(('  DAT   ' if dk else '  TRUOT ') + ten + (f' | {chi_tiet}' if chi_tiet and not dk else ''))


def tieu_de(s):
    print(f'\n==== {s} ====')


def loi(f, *a, **k):
    try:
        f(*a, **k)
    except Exception as e:                                           # noqa: BLE001 - kiem loai loi
        return e
    return None


with contextlib.redirect_stdout(io.StringIO()):
    P1, P2, P3 = M.phan_c()
    P1_GOC = K.phan1_o(False)
    _, P2_S7, P3_S7 = K.phan_s7()
    P2_O = K.phan2_o()

# =========================================================================================================== 1 · goi tren dia
tieu_de('1 · goi tren dia = bo dung; chi .py; khong bi mat')
with contextlib.redirect_stdout(io.StringIO()):
    TEP = M.sinh_goi(False)
tren = M.tep_goi_tren_dia()
ck('moi tep buoc/ + thu_tu.py tren dia = bo dung sinh lai (tung byte)', all(tren.get(p) == s.encode('utf-8') for p, s in TEP.items()))
ck('goi tren dia: dung tep bo dung sinh + __init__.py + cau_hinh.py, khong thua', set(tren) == set(TEP) | {'__init__.py', 'cau_hinh.py'},
   str(set(tren) ^ (set(TEP) | {'__init__.py', 'cau_hinh.py'})))
_khac = [os.path.join(g, f) for g, _, fs in os.walk(M.GOI) if '__pycache__' not in g for f in fs if not f.endswith('.py')]
ck('goi chi co tep .py (ngoai __pycache__)', not _khac, str(_khac[:3]))
_tt = {}
exec(TEP['thu_tu.py'], _tt)
ck('thu_tu.BAM = sha256 tung tep buoc', all(_tt['BAM'][t] == hashlib.sha256(TEP[f'buoc/{t}.py'].encode('utf-8')).hexdigest() for t in _tt['BAM']))
ck('thu_tu.PHAN: phan 1 ket thuc chuyen_1_2, phan 2 ket thuc chuyen_2_3, phan 3 co TONG_HOP truoc C9S',
   _tt['PHAN'][1][-1] == 'chuyen_1_2' and _tt['PHAN'][2][-1] == 'chuyen_2_3' and _tt['PHAN'][3][-2:] == ('p3_TONG_HOP', 'p3_C9S'))
_MAU_BM = {'email': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z.]{2,}', 'token': r'ghp_|github_pat_|AIza[0-9A-Za-z_-]{20}|ya29\.|\bhf_[A-Za-z]{20}|BEGIN [A-Z ]*PRIVATE KEY',
           'id_drive': r"['\"/=]1[A-Za-z0-9_-]{27,43}['\"\s)/&]"}
_bm = {k: sorted({m for s in tren.values() for m in re.findall(v, s.decode('utf-8'))}) for k, v in _MAU_BM.items()}
ck('khong email / token / ID Drive trong goi (repo cong khai)', not any(_bm.values()), str(_bm))
_so8 = sorted({m for s in tren.values() for m in re.findall(r'(?<![\w.])s?\d{8}(?![\w.])', s.decode('utf-8')) if not m.lstrip('s').startswith(('2025', '2026'))})
ck('khong chuoi 8 chu so kieu ma benh nhan / study (tru ngay 2025-26)', not _so8, str(_so8[:10]))

# =========================================================================================================== 2 · Phan 1 = goi o cat da chay + va
tieu_de('2 · Phan 1 = Phan 1 luot goi o cat da chay, khac DUNG cac dong nhanh C')
_NB_O = os.path.join(D78, K.TEN_O[False])
_nb_o = json.load(open(_NB_O, encoding='utf-8'))['cells']
_ma_o = [B.src(c) for c in _nb_o if c['cell_type'] == 'code']
_vt = [i for i, c in enumerate(_nb_o) if c['cell_type'] == 'markdown' and B.src(c).startswith('# Phần 2')][0]
_ma_o1 = [B.src(c) for c in _nb_o[:_vt] if c['cell_type'] == 'code']
_goc1 = [P1_GOC[k] for k in X_THU] if (X_THU := list(M.X.THU_TU_JV2)) else []
ck('notebook goi o cat DA CHAY: o code Phan 1 = phan1_o() cua bo dung 78 (nguon cua nhanh C)', [s.strip() for s in _ma_o1[:len(_goc1)]] == [s.strip() for s in _goc1],
   f'{len(_ma_o1)} vs {len(_goc1)}')
_CU_DUOC = ('THU_NHANH = False', "RUN_TAG = 'r78x_J_v3b_dp_ema_v2_o_s42'", 'SEED = 42 ', "DRV_DUONG = ('stage4_nb78x', RUN_TAG)", "SEED == int(CFG_D4['seed'])",
            'if _kk_lo and not THU_NHANH:', 'if SEED == 44 and ep + 1 in (1, 2):', "_DOI7 = tuple(CFG_D4['ghep_cap'])", '_GHEP7 = True',
            "'(GHEP CAP: cung seed", "' -> ⚠ LECH: bao Claude (thu tu lo khac)'", "GOC_ID_TAY = '")
_MOI_DUOC = re.compile(r"nhanh C|CAU_HINH_C|LUAT_THU_S44|SUC_KHOE_C|TIEP_TUC_C|_sk|_GHEP7|cung lich 6 epoch\)'\)\)|mong doi\)'\)\)\)|LO_S42 is None")
_c9_moi = set(M.C9_C.split('\n')) | set(M.NHANH_C_C0.split('\n'))
_c9_cu = set(K.C9_O.split('\n'))
_xau = []
for k in P1_GOC:
    if not isinstance(P1_GOC[k], str): continue
    a, b = P1_GOC[k].split('\n'), P1[k].split('\n')
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == 'equal': continue
        for l in a[i1:i2]:
            if l.strip() and not any(x in l for x in _CU_DUOC) and l not in _c9_cu: _xau.append((k, '-', l[:100]))
        for l in b[j1:j2]:
            if l.strip() and not _MOI_DUOC.search(l) and l not in _c9_moi: _xau.append((k, '+', l[:100]))
ck('moi dong khac cua Phan 1 la dong nhanh C da liet ke (khong doi gi khac)', not _xau, str(_xau[:4]))
ck('Phan 1: cac o KHONG va trung tung byte (C1, C2, C3, C4, C4A, C4B, C5)', all(P1[k] == P1_GOC[k] for k in ('C1', 'C2', 'C3', 'C4', 'C4A', 'C4B', 'C5')))

# =========================================================================================================== 3 · Phan 2 / 3 = skip7 da chay + va
tieu_de('3 · Phan 2 / 3 = Phan 2 / 3 luot skip7 da chay (4 dau chuan, duong skip), khac DUNG cac o nhanh C')
_NB_S7 = os.path.join(D78, K.TEN_S7[False])
_nb_s7 = json.load(open(_NB_S7, encoding='utf-8'))['cells']
_ma_s7 = {B.src(c).strip() for c in _nb_s7 if c['cell_type'] == 'code'}
_GIONG2 = ('C1', 'C2C', 'c3', 'C3B', 'C3C', 'C4', 'c5', 'C5B', 'C6', 'C7', 'C9')
ck('Phan 2: o khong phu thuoc seed / thu muc trung skip7 tung byte (C1 C2C c3 C3B C3C C4 c5 C5B C6 C7 C9)', all(P2[k] == P2_S7[k] for k in _GIONG2),
   str([k for k in _GIONG2 if P2[k] != P2_S7[k]]))
ck('... va cung la o DA CHAY trong notebook skip7 (tren dia)', all(P2[k].strip() in _ma_s7 for k in _GIONG2))
_GIONG3 = ('C1', 'C2C', 'c3', 'C3B', 'C3C', 'C4', 'c5', 'C5B', 'C9S')
ck('Phan 3: o khong phu thuoc seed trung skip7 tung byte', all(P3[k] == P3_S7[k] for k in _GIONG3), str([k for k in _GIONG3 if P3[k] != P3_S7[k]]))
_d2 = [l for op in difflib.ndiff(P2_S7['C2'].split('\n'), P2['C2'].split('\n')) for l in [op] if l[:1] in '+-']
ck('Phan 2 C2: chi doi thong bao thu muc backbone (stage4_nb78x -> TM_BB)', len(_d2) == 2 and 'TM_BB' in _d2[1] and 'stage4_nb78x' in _d2[0], str(_d2))
_d2b = [(l[:1], l[2:]) for l in difflib.ndiff(P2_S7['C2B'].split('\n'), P2['C2B'].split('\n')) if l[:1] in '+-']
_s7l, _cl = set(K.C2BZ_S7.split('\n')), set(M.C2BZ_C.split('\n'))
ck('Phan 2 C2b: chi khac khoi cong cau truc (bo dong cua C2BZ skip7, them dong cua C2BZ nhanh C)',
   _d2b and all((d == '-' and l in _s7l) or (d == '+' and l in _cl) for d, l in _d2b), str([x for x in _d2b if not ((x[0] == '-' and x[1] in _s7l) or
                                                                                                                (x[0] == '+' and x[1] in _cl))][:3]))
ck('Phan 2 / 3 C2b giong nhau (mot tep z_C2B)', P2['C2B'] == P3['C2B'])
ck('Phan 3 C6S: dau gop + dau quyet dinh nhu skip7', "DAU_S, TOL_S, NG_TOAN_S, NG_NUA_S, HAT_NUA_S = ('mldec_hr', 'fus_attn_hr', 'fus_tb_hr')" in P3['C6S']
   and "DAU_QD_S = 'fus_attn_hr'" in P3['C6S'])
_78 = [(k, l.strip()[:80]) for P in (P2, P3) for k, v in P.items() if isinstance(v, str) for l in B.dong_ma(v)
       if ('stage4_nb78x' in l or 'stage4_nb78z' in l) and not l.startswith('O_CFG_MONG = ')]
ck('Phan 2 / 3: khong con stage4_nb78x / stage4_nb78z o dong ma (tru O_CFG_MONG = cfg_o cua cong thuc, phai trung checkpoint)', not _78, str(_78[:3]))
ck('O_CFG_MONG Phan 2 / 3 = O_CFG Phan 1 nhanh C (cfg_o luu trong checkpoint)', f'O_CFG_MONG = {K.O_CFG!r}' in P2['C0'] and f'O_CFG = {K.O_CFG!r}' in P1['C0'])
_s42 = [l.strip()[:90] for l in B.dong_ma(P3['C6S']) if re.search(r'_s42|\(42,\)|== 42\b', l)]
ck('Phan 3 C6S: khong con ten tep / seed 42 cung o dong ma', not _s42, str(_s42[:3]))
ck('ca ba C0: GOC_ID_TAY doc bien moi truong CXRLT_GOC_ID (ID Drive KHONG trong ma)', all(M._DONG_GOC_ID in P['C0'] and re.search(r"GOC_ID_TAY = '1", P['C0']) is None
                                                                                           for P in (P1, P2, P3)))

# ---- 4 dau so 5 dau (Phan 2 goi o cat s42): cac buoc ngau nhien / cat gradient cua 4 dau chung KHONG phu thuoc dau thu 5 ----
_c6o = P2_O['C6'].split('\n')
_i_giu = [i for i, l in enumerate(_c6o) if 'pf._giu = rut_giu_view(' in l and not l.lstrip().startswith('#')]
_i_arm = [i for i, l in enumerate(_c6o) if l.strip() == 'for t in ARMS:' and _i_giu and i > _i_giu[0]]
_tht = lambda l: len(l) - len(l.lstrip())
ck('goi s42 Phan 2: mat na bo view rut MOT lan moi lo (GEN_FUS), NGOAI vong arm', len(_i_giu) == 1 and _i_arm and _tht(_c6o[_i_giu[0]]) <= _tht(_c6o[_i_arm[0]]))
ck('goi s42 Phan 2: dau fus_attn_hr_o la arm CUOI (4 dau chuan tao + chay truoc)', "'fus_attn_hr', 'fus_attn_hr_o')   # 78t: 17 arm" in P2_O['C0']
   and "ARMS_GIU = ['mldec', 'mldec_hr', 'fus_tb_hr', 'fus_attn_hr', 'fus_attn_hr_o']" in P2_O['C0'])
ck('goi s42 Phan 2: nhanh 6 view chi chay khi arm co gop_o, trong _FORK_78Y (dropout khong tieu dong chung)',
   "if hasattr(m, 'gop_o') else None)" in P2_O['C6'] and K._C6Z_FUS_MOI in P2_O['C6'])
_clip = [l for l in P2_O['C6'].split('\n') if 'clip_grad_norm_(ps, CLIP' in l]
ck('goi s42 Phan 2: cat gradient RIENG tung arm (clip_grad_norm_(ps, ...) trong vong arm)', len(_clip) == 1 and _tht(_clip[0]) > _tht(_c6o[_i_arm[0]]))
_kh = [(k, l) for k in P2_O if isinstance(P2_O[k], str) and k not in ('C0', 'C2B', 'C4', 'C6', 'C9')
       for l in difflib.unified_diff(P2_S7.get(k, '').split('\n'), P2_O[k].split('\n'), lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')]
ck('goi s42 Phan 2 so skip7: ngoai C0 / C2b / C4 / C6 / C9 khong khac (cung ma 78z)', not _kh, str(_kh[:3]))

# =========================================================================================================== 4 · dong cau hinh theo seed
tieu_de('4 · dong cau hinh theo seed (chay that tung dong) + cong cau hinh')


def dong_co(code, dau):
    return Z._mot_dong(code, dau)


def chay_dong(code, dau_ds, ns):
    for d in dau_ds:
        exec(compile(dong_co(code, d), d, 'exec'), ns)
    return ns


_ok = True
for s in CH.SEEDS:
    for thu in (False, True):
        cf = CH.cau_hinh(s, thu)
        ns1 = chay_dong(P1['C0'], ('THU_NHANH = ', 'RUN_TAG = ', 'SEED = int(', 'DRV_DUONG = (CAU'), dict(CAU_HINH_C=cf))
        _ok &= (ns1['THU_NHANH'] is thu and ns1['SEED'] == s and ns1['RUN_TAG'] == f'r84x_C_s{s}' + ('_thu' if thu else '')
                and ns1['DRV_DUONG'] == ('stage4_nb84', ns1['RUN_TAG']))
        for P, rt in ((P2, f'r84z_C_s{s}'), (P3, f'r84z_C_s{s}_gts')):
            ns2 = chay_dong(P['C0'], ('THU_NHANH = ', 'RUN_TAG = ', 'BB_RUN = ', 'SEEDS = ', 'SEED, SIZE, EPOCHS = ', 'DRV_DUONG = (CAU', 'TM_BB = '),
                            dict(CAU_HINH_C=cf))
            _ok &= (ns2['RUN_TAG'] == rt + ('_thu' if (thu and P is P2) else '') and ns2['BB_RUN'] == f'r84x_C_s{s}' and ns2['SEEDS'] == (s,)
                    and ns2['SEED'] == s and ns2['EPOCHS'] == 10 and ns2['DRV_DUONG'] == ('stage4_nb84', ns2['RUN_TAG']) and ns2['TM_BB'] == 'stage4_nb84')
ck('Phan 1 / 2 / 3: SEED, RUN_TAG, BB_RUN, SEEDS, DRV_DUONG, TM_BB dung cho s42-s46 (chinh + thu)', _ok)
ck('cau_hinh: tu choi seed ngoai danh sach / bool / chuoi / thu khong bool',
   all(isinstance(loi(CH.cau_hinh, x), ValueError) for x in (41, 47, True, '43', 43.0)) and isinstance(loi(CH.cau_hinh, 43, 1), ValueError))
_a1 = dong_co(P1['C0'], 'assert CFG_D4[')
_ns = dict(CFG_D4=dict(nen='v', seed=42, epochs=6, dp_ema={}), CFG_JV3B=dict(phien_ban='v'), EPOCHS=6, BIEN_THE_D4='dp_ema')
_t4 = []
for s in (42, 43, 46):
    _ns.update(SEED=s, CAU_HINH_C=CH.cau_hinh(s))
    _t4.append(loi(exec, _a1 + Z._mot_dong(P1['C0'], "    BIEN_THE_D4 in CFG_D4), f\"jca_v3b_d4:"), dict(_ns)) is None)
_ns.update(SEED=41, CAU_HINH_C=CH.cau_hinh(43))
_t4.append(loi(exec, _a1 + Z._mot_dong(P1['C0'], "    BIEN_THE_D4 in CFG_D4), f\"jca_v3b_d4:"), dict(_ns)) is not None)
ck('Phan 1 C0: cong cau hinh dot 4 nhan s42 / s43 / s46, tu choi seed la', all(_t4))
_khoi = M.NHANH_C_C0
ck('Phan 1 C0: khoi nhanh C dung khi thieu CAU_HINH_C (chay ngoai nhanh_c.buoc)', isinstance(loi(exec, _khoi, {}), AssertionError)
   and loi(exec, _khoi, dict(CAU_HINH_C=CH.cau_hinh(43))) is None)
_i_luat = P1['C0'].index('LUAT_THU_S44 = False')
ck('Phan 1: LUAT_THU_S44 = False dat o C0 (truoc C6); C6 chi chay luat seed 44 khi co bat', _i_luat > 0
   and '    if LUAT_THU_S44 and SEED == 44 and ep + 1 in (1, 2):' in P1['C6'] and P1['C6'].count('SEED == 44') == 1)
ck('Phan 1 C6: cong ghep cap dau van lo CHI seed 42', 'if _kk_lo and not THU_NHANH and SEED == 42:' in P1['C6'] and 'LO_S42 is None and not THU_NHANH' in P1['C6'])
_ok7 = True
for s, ghep in ((42, True), (43, False), (46, False)):
    ns7 = chay_dong(P1['C7'], ('_DOI7 = ', '_GHEP7 = '), dict(CFG_D4=dict(ghep_cap=['stage4_nb78x', 'r78x_J_v3b_s42']), SEED=s, CAU_HINH_C=CH.cau_hinh(s)))
    _ok7 &= ns7['_GHEP7'] is ghep and ns7['_DOI7'] == (('stage4_nb78x', 'r78x_J_v3b_s42') if ghep else ('stage4_nb78x', 'r78x_J_v3b_dp_ema_v2_o_s42'))
ck('Phan 1 C7: s42 canh J v3b s42 (ghep cap); seed khac canh C s42 (khong ghep cap)', _ok7)

# =========================================================================================================== 5 · chon epoch backbone (Phan 2 / 3 C0)
tieu_de('5 · chon epoch backbone: dinh VAL EMA trong luot, lam tron 4 chu so, bang nhau -> epoch MUON hon')


def chay_ep(vals, tay=None, co_rs=None, co_lg=True):
    log = {str(e): {'mAP_anh': v} for e, v in enumerate(vals) if v is not None}
    td = tempfile.mkdtemp()
    p = os.path.join(td, 'lg.json'); json.dump(log, open(p, 'w'))
    kho = {}
    if co_lg: kho['epoch_log_J_bert_dong_ep6.json'] = [dict(id='i1', md5='m1', p=p)]
    for e in (co_rs if co_rs is not None else range(1, 7)): kho[f'rs_J_bert_dong_ep{e}.pt'] = [dict(id=f'r{e}', md5='x')]
    ns = dict(kiem_ke=lambda du, mau=None: (None, {k: v for k, v in kho.items() if mau is None or re.fullmatch(mau, k)}), tai=lambda d: d['p'], json=json,
              math=math, TM_BB='stage4_nb84', BB_RUN='r84x_C_s43', EP_BB_TAY=tay, SEEDS=(43,))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(M.C0Z_EP_C, ns)
    return ns


_ca = [([.25, .26, .27, .265, .26, .255], 3), ([.25, .26, .2650, .2668, .2656, .2640], 4), ([.25, .26, .2701, .27004, .2699, .2701], 6),
       ([.2501, .2580, .2633, .2656, .2676, .2668], 5), ([.27, .26, .25, .24, .23, .22], 1)]
ck('dinh duy nhat / bang nhau sau lam tron -> epoch MUON hon / C s42 (ep5) / dinh ep1', all(chay_ep(v)['EP_BB'] == e for v, e in _ca),
   str([chay_ep(v)['EP_BB'] for v, _ in _ca]))
ck('ghi de EP_BB_TAY = 3 -> ep3; CHON_EP_C ghi quy tac TAY', chay_ep(_ca[0][0], tay=3)['EP_BB'] == 3 and chay_ep(_ca[0][0], tay=3)['CHON_EP_C']['quy_tac'].startswith('TAY'))
ck('CHON_EP_C: quy tac "bang nhau -> epoch MUON hon", so tho + lam tron, md5 nhat ky', (lambda n: 'MUON hon' in n['CHON_EP_C']['quy_tac'] and
                                                                                         n['CHON_EP_C']['nhat_ky']['md5'] == 'm1' and n['CHON_EP_C']['val_ema_lam_tron'][2] == 0.2701)(chay_ep(_ca[2][0])))
ck('dung: thieu epoch / NaN / Inf / ngoai [0,1] / thieu nhat ky / thieu rs_ dinh / TAY 7, 2.9, True',
   all(isinstance(loi(chay_ep, *a, **k), AssertionError) for a, k in (((_ca[0][0][:5] + [None],), {}), (([.2, float('nan'), .2, .2, .2, .2],), {}),
                                                                       (([.2, float('inf'), .2, .2, .2, .2],), {}), (([.2, 1.5, .2, .2, .2, .2],), {}),
                                                                       ((_ca[0][0],), dict(co_lg=False)), ((_ca[0][0],), dict(co_rs=[1, 2, 4, 5, 6])),
                                                                       ((_ca[0][0],), dict(tay=7)), ((_ca[0][0],), dict(tay=2.9)), ((_ca[0][0],), dict(tay=True)))))

# =========================================================================================================== 6 · cong suc khoe + o chuyen phan 1 -> 2
tieu_de('6 · cong suc khoe (C9) + o chuyen phan 1 -> 2')
_MOC = [0.2501, 0.2580, 0.2633, 0.2656, 0.2676, 0.2668]


def chay_sk(nay, moc=_MOC, thu=False, tb_moc=0.2619):
    td = tempfile.mkdtemp()
    p = os.path.join(td, 'lg.json'); json.dump({str(e): {'mAP_anh': v} for e, v in enumerate(moc)}, open(p, 'w'))
    da_ghi = {}
    cf = CH.cau_hinh(43); cf['suc_khoe']['moc_tb'] = tb_moc
    ns = dict(THU_NHANH=thu, EPOCHS=6, CAU_HINH_C=cf, SEED=43, LOG={str(e): {'mAP_anh': v} for e, v in enumerate(nay)}, json=json, np=np,
              kiem_ke=lambda du, mau=None: (None, {'epoch_log_J_bert_dong_ep6.json': [dict(id='a', md5='m', p=p)]}), tai=lambda d: d['p'],
              ghi=lambda ten, f: da_ghi.__setitem__(ten, f), ghi_json=lambda q, x: None, dong_bo=lambda: True)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(M.C9_C, ns)
    return ns, da_ghi


_n1, _g1 = chay_sk([v - 0.005 for v in _MOC])
_n2, _ = chay_sk([v - 0.012 for v in _MOC])
_n3, _g3 = chay_sk([v + 0.01 for v in _MOC], thu=True)
ck('cong suc khoe: thap hon 0,005 -> DAT (ghi suc_khoe_c.json); thap hon 0,012 -> YEU; THU_NHANH -> khong chay',
   _n1['SUC_KHOE_C']['ket_luan'] == 'DAT' and 'suc_khoe_c.json' in _g1 and _n2['SUC_KHOE_C']['ket_luan'] == 'YEU' and _n3['SUC_KHOE_C'] is None and not _g3)
ck('cong suc khoe: dung khi TB C s42 tinh lai lech moc (> 5e-4)', isinstance(loi(chay_sk, _MOC, tb_moc=0.2700), AssertionError))
_chuyen = M.O_CHUYEN_1_C
_t6 = [loi(exec, _chuyen, dict(SUC_KHOE_C=dict(ket_luan='DAT'))) is None,
       isinstance(loi(exec, _chuyen, dict(SUC_KHOE_C=dict(ket_luan='YEU', chenh=-0.02))), AssertionError),
       loi(exec, _chuyen, dict(SUC_KHOE_C=dict(ket_luan='YEU'), TIEP_TUC_C=True)) is None,
       isinstance(loi(exec, _chuyen, dict(SUC_KHOE_C=dict(ket_luan='YEU'), TIEP_TUC_C=1)), AssertionError),
       isinstance(loi(exec, _chuyen, dict()), AssertionError)]
ck('o chuyen 1 -> 2: DAT qua; YEU dung; YEU + TIEP_TUC_C = True qua; TIEP_TUC_C = 1 dung; thieu SUC_KHOE_C dung', all(_t6), str(_t6))

# =========================================================================================================== 7 · C2b cong cau truc nhanh C
tieu_de('7 · C2b: cong cau truc checkpoint nhanh C')
_C2B_DAU = M.C2BZ_C.split('    with _FORK_78Y():')[0]


def chay_c2b(cfg_o, khoa):
    return loi(exec, 'if True:\n' + _C2B_DAU, dict(_cfj=dict(cfg_o=cfg_o), _ck={k: {} for k in khoa}, O_CFG_MONG=dict(K.O_CFG), BB_RUN='r', BB_TEP='t'))


ck('nhan checkpoint nhanh C (cfg_o goi o cat, du dua_o / bo_o7 / bo_o11 / gop_o)', chay_c2b(dict(K.O_CFG), ('dua_o', 'bo_o7', 'bo_o11', 'gop_o')) is None)
ck('tu choi checkpoint skip7 (cfg_o skip7 / thieu bo_o11, gop_o) va B.4 v2 (khong cfg_o)',
   all(isinstance(chay_c2b(*a), AssertionError) for a in ((dict(K.S7_CFG), ('dua_o', 'bo_o7')), (dict(K.O_CFG), ('dua_o', 'bo_o7')), (None, ()))))
ck("C2b: chi nap dua_o + bo_o7 (duong skip); khoi nhung co token_o7", "for _m_o, _k_o in ((BO_O7, 'bo_o7'), (DUA_O, 'dua_o'))" in P2['C2B']
   and 'def token_o7(' in P2['C2B'] and 'BO_O11' not in '\n'.join(B.dong_ma(M.C2BZ_C)))

# =========================================================================================================== 8 · C6S ten tep theo SEED
tieu_de('8 · Phan 3 C6S: ten tep theo SEED (s42 -> giong het luot cu)')
_c6s = P3['C6S']
_cay = ast.parse(_c6s)
_hang = [n.value for n in ast.walk(_cay) if isinstance(n, ast.Constant) and isinstance(n.value, str) and 'SEED' in n.value and '{SEED}' in n.value]
ck('khong chuoi THUONG nao chua {SEED} (moi cho la f-string)', not _hang, str(_hang[:3]))
_fs = [n for n in ast.walk(_cay) if isinstance(n, ast.JoinedStr) and any(isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name) and v.value.id == 'SEED'
                                                                        for v in n.values)]
_mau42 = []
for n in _fs:
    env = dict(SEED=42, e=7, d='fus_attn_hr', RUN_TAG='R', DRV_DUONG=('stage4_nb78z', 'R'), TEN_JSON_S='J', DAU_S=('a',))
    try:
        _mau42.append(eval(compile(ast.Expression(n), 'f', 'eval'), env))
    except NameError:
        pass
_cu6 = [m for m in re.findall(r"'([^']*s42[^']*)'", P3_S7['C6S'])]
ck('moi f-string co {SEED} (SEED = 42) cho dung ten luot cu (w_s42_ep7.pt, p_fus_attn_hr_s42_ep7.npy, p_gopA_..._s42.npy, w_gopA_s42.pt)',
   {'w_s42_ep7.pt', 'p_fus_attn_hr_s42_ep7.npy', 'p_gopA_fus_attn_hr_s42.npy', 'p_gopB_fus_attn_hr_s42.npy', 'pte_gopA_fus_attn_hr_s42.npy', 'w_gopA_s42.pt'}
   <= set(_mau42), str(sorted(set(_mau42))[:8]))
_re_moi = eval(compile(ast.Expression(next(n for n in _fs if '_ep' in ast.unparse(n) and 'w_s' in ast.unparse(n) and '\\d' in ast.unparse(n))), 'r', 'eval'), dict(SEED=42))
ck('mau kiem ke rf-string (SEED = 42) = mau r-string cu', _re_moi == r'(w_s42_ep\d+\.pt|p_[a-z0-9_]+_s42_ep\d+\.npy)', _re_moi)

# =========================================================================================================== 9 · TONG_HOP tren du doan gia
tieu_de('9 · TONG_HOP tren du doan gia (seed nay s43; co s42, s44; chua s45, s46)')
N_V, N_T, C_ = 60, 30, 4
_rg = np.random.default_rng(0)


def _so3_gia(p):
    return dict(toan=float(p[:, 0].mean()), nua_A=float(p[:30, 1].mean()), nua_B=float(p[30:, 2].mean()), tier=dict(head=0.5, medium=0.2, tail=0.1))


def chay_th(co=(42, 44), seed=43, moc42=None, a_lech=False, thieu_ep=False, pte_thieu=()):
    td = tempfile.mkdtemp()
    kho = {}

    def dat(du, ten, arr):
        p = os.path.join(td, f"{'_'.join(du)}_{ten}"); np.save(p, arr)
        kho.setdefault(tuple(du), {})[ten] = [dict(id=ten, md5='m' + ten, p=p)]

    cf = CH.cau_hinh(seed)
    gop = {s: _rg.random((N_V, C_)).astype('float32') for s in CH.SEEDS}
    if moc42 is not None: cf['tong_hop']['moc_s42_gop'] = moc42
    else: cf['tong_hop']['moc_s42_gop'] = float(gop[42][:, 0].mean())
    for s in co:
        ng = cf['tong_hop']['nguon'][str(s)]
        dat(ng['gts'], f'p_gopA_fus_attn_hr_s{s}.npy', gop[s])
        if s not in pte_thieu: dat(ng['gts'], f'pte_gopA_fus_attn_hr_s{s}.npy', _rg.random((N_T, C_)).astype('float32'))
        for e in range(1, 11):
            if thieu_ep and e == 10: continue
            dat(ng['z'], f'p_fus_attn_hr_s{s}_ep{e}.npy', gop[s] + e * 0.001)
    a = _rg.random((N_V, C_)).astype('float32')
    cf['tong_hop']['a_s42']['moc_gop'] = float(a[:, 0].mean()) + (0.01 if a_lech else 0.0)
    dat(cf['tong_hop']['a_s42']['gts'], 'p_gopA_fus_attn_hr_s42.npy', a)
    pv = gop[seed]; pep = [pv + e * 0.001 for e in range(1, 11)]
    ghi_ra = {}
    ns = dict(CAU_HINH_C=cf, SEED=seed, DAU_QD_S='fus_attn_hr', P_EP={'fus_attn_hr': pep}, _pvA={'fus_attn_hr': pv}, _pteA={'fus_attn_hr': _rg.random((N_T, C_))},
              DRV_DUONG=('stage4_nb84', f'r84z_C_s{seed}_gts'), _DU_S=('stage4_nb84', f'r84z_C_s{seed}'), EP_S=list(range(1, 11)), _IVA=np.arange(N_V),
              HAT_NUA_S=7801, _gop_v=lambda p: p, _so3=_so3_gia, _KA='gop A', _KTB='TB', np=np, datetime=__import__('datetime'),
              kiem_ke=lambda du, mau=None: (None, {k: v for k, v in kho.get(tuple(du), {}).items() if mau is None or re.fullmatch(mau, k)}),
              tai=lambda d: d['p'], ghi=lambda ten, f: ghi_ra.__setitem__(ten, f), ghi_json=lambda q, x: None)
    ns['KQ_S'] = {('fus_attn_hr', 'gop A'): _so3_gia(pv), ('fus_attn_hr', 'TB'): _so3_gia(np.mean(pep, 0))}
    with contextlib.redirect_stdout(io.StringIO()) as out:
        exec(M.TONG_HOP_C, ns)
    return ns, ghi_ra, out.getvalue(), gop


_t9, _g9, _o9, _gop9 = chay_th()
ck('seed co = s42, s43 (notebook nay), s44; chua = s45, s46', _t9['SD_TH'] == [42, 43, 44] and 's45, s46' in _o9)
ck('so tung seed = _so3(_gop_v(p_gopA)) / TB du doan ep1-10; bo gop = TB du doan cac seed',
   abs(_t9['KQ_TH'][44]['gop']['toan'] - float(_gop9[44][:, 0].mean())) < 1e-6
   and abs(_t9['KQ_TH'][42]['tb']['toan'] - float(np.mean([_gop9[42] + e * 0.001 for e in range(1, 11)], 0)[:, 0].mean())) < 1e-6
   and abs(_t9['KQ_BO_TH']['gop']['toan'] - float(np.mean([_gop9[s] for s in (42, 43, 44)], 0)[:, 0].mean())) < 1e-6)
ck('TB +- SD (ddof 1) qua 3 seed', abs(_t9['TK_TH']['gop']['toan'][1] - float(np.std([float(_gop9[s][:, 0].mean()) for s in (42, 43, 44)], ddof=1))) < 1e-9)
ck('ghi json + pte bo gop 3 seed (du pte) ; C - A o s42 in ra', any(k.startswith('tong_hop_nhanh_c_') for k in _g9) and 'pte_gopA_fus_attn_hr_bo_gop_s42_43_44.npy' in _g9
   and 'C - A o s42' in _o9)
_, _g9b, _, _ = chay_th(pte_thieu=(44,))
ck('thieu pte mot seed -> KHONG ghi pte bo gop', not any(k.startswith('pte_') for k in _g9b))
ck('dung: moc s42 lech / A s42 lech / thieu p_ ep10 cua seed co p_gopA', all(isinstance(loi(chay_th, **k), AssertionError)
                                                                            for k in (dict(moc42=0.9), dict(a_lech=True), dict(thieu_ep=True))))
_t9c, _, _o9c, _ = chay_th(co=(), seed=44)
ck('chi co seed notebook nay -> van in (n = 1, SD = nan), khong C - A', _t9c['SD_TH'] == [44] and 'C - A' not in _o9c)

# =========================================================================================================== 10 · nhanh_c.buoc (nap goi that)
tieu_de('10 · nhanh_c.buoc: loi dau vao, cong tron seed, sha256, ngu nghia o chuyen phan (%reset)')
sys.path.insert(0, D)
for _m in [m for m in sys.modules if m == 'nhanh_c' or m.startswith('nhanh_c.')]: del sys.modules[_m]
import nhanh_c as NC                                                 # noqa: E402
ck('nap goi: PHIEN_BAN, 3 phan, ds_buoc() = thu tu notebook', NC.PHIEN_BAN == CH.PHIEN_BAN and NC.ds_buoc(1)[0] == 'p1_C0'
   and NC.ds_buoc() == list(_tt['PHAN'][1]) + list(_tt['PHAN'][2]) + list(_tt['PHAN'][3]))
ck('buoc: ten la -> KeyError; ns khong dict -> TypeError; seed la -> ValueError',
   isinstance(loi(NC.buoc, 'p9_C0', {}, 43), KeyError) and isinstance(loi(NC.buoc, 'p1_C0', [], 43), TypeError)
   and isinstance(loi(NC.buoc, 'p1_C0', {}, 47), ValueError))
ck('buoc: namespace dang mang seed khac / thu khac -> ValueError (khong chay gi)',
   isinstance(loi(NC.buoc, 'chuyen_2_3', dict(CAU_HINH_C=CH.cau_hinh(44)), 43), ValueError)
   and isinstance(loi(NC.buoc, 'chuyen_2_3', dict(CAU_HINH_C=CH.cau_hinh(43, True)), 43), ValueError))


class _IP:
    def __init__(s, ns): s.ns = ns

    def run_line_magic(s, ten, arg):
        assert (ten, arg) == ('reset', '-f')
        for k in list(s.ns):
            if k not in ('__builtins__', 'get_ipython'): del s.ns[k]


_ns10 = dict(_OK_PHAN_2=True, RUN_TAG='x', ghi=1, dong_bo=1, DRV_DUONG=1, TEN_RESUME=1)
_ns10['get_ipython'] = lambda: _IP(_ns10)
with contextlib.redirect_stdout(io.StringIO()) as _o10:
    _e10 = loi(NC.buoc, 'chuyen_2_3', _ns10, 43)
ck('o chuyen 2 -> 3 qua buoc: %reset xoa namespace notebook (ke ca CAU_HINH_C), in "phan 3: bat dau"', _e10 is None and 'RUN_TAG' not in _ns10
   and 'CAU_HINH_C' not in _ns10 and 'phan 3: bat dau tu namespace sach' in _o10.getvalue(), repr(_e10))
_ns10b = dict(get_ipython=lambda: _IP(_ns10b))
ck('o chuyen 2 -> 3 khi phan 2 chua xong (_OK_PHAN_2 thieu) -> dung', isinstance(loi(NC.buoc, 'chuyen_2_3', _ns10b, 43), AssertionError))
_tmp = tempfile.mkdtemp()
shutil.copytree(M.GOI, os.path.join(_tmp, 'nhanh_c'), ignore=shutil.ignore_patterns('__pycache__'))
with open(os.path.join(_tmp, 'nhanh_c', 'buoc', 'chuyen_2_3.py'), 'a', encoding='utf-8') as f: f.write('# sua tay\n')
_r = subprocess.run([sys.executable, '-I', '-X', 'utf8', '-c', "import sys; sys.path.insert(0, sys.argv[1]); import nhanh_c as N\n"
                     "try:\n    N.buoc('chuyen_2_3', {}, 43)\nexcept RuntimeError as e:\n    print('RT', e)\n", _tmp], capture_output=True, text=True)
ck('buoc: tep buoc bi sua tay (sha256 khac) -> RuntimeError, khong chay', _r.stdout.startswith('RT'), _r.stdout + _r.stderr)
_tmp2 = tempfile.mkdtemp()
shutil.copytree(M.GOI, os.path.join(_tmp2, 'nhanh_c'), ignore=shutil.ignore_patterns('__pycache__'))
for _f in os.listdir(os.path.join(_tmp2, 'nhanh_c', 'buoc')):
    _p = os.path.join(_tmp2, 'nhanh_c', 'buoc', _f)
    _b = open(_p, 'rb').read()                                       # doc TRUOC khi mo ghi (mo 'wb' xoa trang tep)
    open(_p, 'wb').write(_b.replace(b'\n', b'\r\n'))                 # nhu git autocrlf tren Windows
_r2 = subprocess.run([sys.executable, '-I', '-X', 'utf8', '-c', "import sys; sys.path.insert(0, sys.argv[1]); import nhanh_c as N\n"
                      "class I:\n    def __init__(s, ns): s.ns = ns\n    def run_line_magic(s, a, b):\n        [s.ns.pop(k) for k in list(s.ns) if k not in ('__builtins__', 'get_ipython')]\n"
                      "ns = dict(_OK_PHAN_2=True)\nns['get_ipython'] = lambda: I(ns)\nN.buoc('chuyen_2_3', ns, 43)\nprint('OK', 'RUN_TAG' not in ns)\n", _tmp2],
                     capture_output=True, text=True)
ck('buoc: tep buoc bi doi LF -> CRLF (git autocrlf khi checkout) van qua sha256 va chay dung', 'OK True' in _r2.stdout, _r2.stdout[-300:] + _r2.stderr[-300:])

# =========================================================================================================== 11 · notebook + commit ghim
tieu_de('11 · notebook moi seed (chi khi goi tren dia = commit da day)')
_sha = M.sha_da_day()
if _sha is None:
    print('  (bo qua: chua day hoac goi KHAC commit da day -> chay day_nhanh_c.py roi dung_84.py --notebook)')
else:
    with contextlib.redirect_stdout(io.StringIO()):
        _nbs = M.dung_nb(False)
    _ok11, _ok12 = True, True
    for ten, cells in _nbs.items():
        p = os.path.join(D, ten)
        _ok11 &= os.path.isfile(p) and [B.src(c) for c in json.load(open(p, encoding='utf-8'))['cells']] == [B.src(c) for c in cells]
        s = int(re.search(r'_s(\d+)', ten).group(1)); thu = ten.endswith('_thu.ipynb')
        code = [B.src(c) for c in cells if c['cell_type'] == 'code']
        _ok12 &= f"NHANH_C_SHA = '{_sha}'" in code[0] and f"os.environ['CXRLT_GOC_ID'] = '{M.goc_id()}'" in code[0] and all(re.fullmatch(rf"import nhanh_c as NC; NC\.buoc\('[A-Za-z0-9_]+', globals\(\), seed={s}"
                                                                          + (r", thu=True" if thu else '') + r"\)\n", c) for c in code[1:])
        buoc_nb = [re.search(r"buoc\('([^']+)'", c).group(1) for c in code[1:]]
        _ok12 &= buoc_nb == (list(_tt['PHAN'][1][:-1]) if thu else NC.ds_buoc())
    ck('notebook tren dia = bo dung (4 seed + ban thu s43)', _ok11 and len(_nbs) == 5)
    ck('o dau ghim dung commit da day; moi o sau = MOT dong buoc(..., seed) dung thu tu; ban thu chi Phan 1 (thu=True)', _ok12)

# =========================================================================================================== 12 · loader: module da nap phai dung commit + duong dan
tieu_de('12 · loader GitHub: khoa module da nap theo commit + duong dan canonical (KHONG goi mang)')


def chay_loader_cache(kieu):
    from unittest import mock
    sha = 'b' * 40
    cu_mod = {k: v for k, v in sys.modules.items() if k == 'nhanh_c' or k.startswith('nhanh_c.')}
    cu_path = list(sys.path)
    with tempfile.TemporaryDirectory(prefix='kiem84_loader_') as td:
        td = os.path.realpath(td)
        assert os.path.dirname(td) == os.path.realpath(tempfile.gettempdir()), 'thu muc fixture phai trong temp'
        shutil.copytree(M.GOI, os.path.join(td, 'nhanh_c'), ignore=shutil.ignore_patterns('__pycache__'))
        os.mkdir(os.path.join(td, '.git'))                     # chi marker; moi lenh git duoi day duoc gia lap
        lenh = []

        def git_gia(args, **kw):
            lenh.append(args)
            return types.SimpleNamespace(stdout=sha + '\n', stderr='', returncode=0)

        try:
            for k in cu_mod: sys.modules.pop(k, None)
            if kieu != 'moi':
                sys.path.insert(0, td)
                import nhanh_c as nap_cu
                if kieu != 'khong_ghim': nap_cu._NHANH_C_SHA_DA_NAP = sha
                if kieu == 'commit_khac':
                    nap_cu._NHANH_C_SHA_DA_NAP = 'a' * 40
                    p = os.path.join(td, 'nhanh_c', 'cau_hinh.py')
                    with open(p, encoding='utf-8') as f: s = f.read()
                    assert 'NGUONG_SUC_KHOE = 0.01' in s
                    with open(p, 'w', encoding='utf-8') as f:
                        f.write(s.replace('NGUONG_SUC_KHOE = 0.01', 'NGUONG_SUC_KHOE = 0.02'))
                if kieu == 'tien_to_giong':
                    nap_cu.__file__ = os.path.join(td + '_cu', 'nhanh_c', '__init__.py')
            ma = M.o_tai_goi(sha, 43, False).replace("_d_goi = '/content/nhanh_c_goi'", f'_d_goi = {td!r}')
            assert f'_d_goi = {td!r}' in ma
            ns = {}
            with mock.patch.object(subprocess, 'run', git_gia), mock.patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                e = loi(exec, ma, ns)
            nap = sys.modules.get('nhanh_c')
            return dict(loi=e, lenh=list(lenh), sha=getattr(nap, '_NHANH_C_SHA_DA_NAP', None),
                        nguong=(nap.cau_hinh(43)['suc_khoe']['nguong'] if nap else None),
                        duong_dung=bool(nap and os.path.realpath(nap.__file__) == os.path.join(td, 'nhanh_c', '__init__.py')))
        finally:
            for k in list(sys.modules):
                if k == 'nhanh_c' or k.startswith('nhanh_c.'): sys.modules.pop(k, None)
            sys.modules.update(cu_mod)
            sys.path[:] = cu_path


_l12 = chay_loader_cache('moi')
ck('loader: nap moi tu dung duong dan -> ghi commit da nap len module', _l12['loi'] is None and _l12['duong_dung'] and _l12['sha'] == 'b' * 40,
   repr(_l12['loi']))
_l12 = chay_loader_cache('cung_commit')
ck('loader: module cung commit + duong dan duoc chay lai', _l12['loi'] is None and _l12['sha'] == 'b' * 40 and bool(_l12['lenh']), repr(_l12['loi']))
_l12 = chay_loader_cache('khong_ghim')
ck('loader: module da import ngoai loader (chua ghim) -> dung TRUOC git, yeu cau Restart session',
   isinstance(_l12['loi'], RuntimeError) and 'Restart session' in str(_l12['loi']) and not _l12['lenh'], repr(_l12['loi']))
_l12 = chay_loader_cache('commit_khac')
ck('loader: doi commit chi sua cau_hinh trong cung kernel -> dung TRUOC git, khong am tham dung cau hinh cu',
   isinstance(_l12['loi'], RuntimeError) and 'Restart session' in str(_l12['loi']) and not _l12['lenh'] and _l12['nguong'] == 0.01, repr(_l12['loi']))
_l12 = chay_loader_cache('tien_to_giong')
ck('loader: duong dan chi giong tien to (..._cu) bi tu choi TRUOC git',
   isinstance(_l12['loi'], RuntimeError) and 'Restart session' in str(_l12['loi']) and not _l12['lenh'], repr(_l12['loi']))

print(f"\n{sum(d for _, d in KQ)}/{len(KQ)} DAT" + ('' if all(d for _, d in KQ) else ' | TRUOT: ' + '; '.join(t for t, d in KQ if not d)))
sys.exit(0 if all(d for _, d in KQ) else 1)
