# ==== nhanh_c buoc p1_C3 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C3 · BANG LUC TRAIN (CHI benh nhan train, TRU 2% giu rieng) — CHI DOC qua API, KHONG ghi Drive ====
# Chung moi luot: hop 36 vung 78f (TF_HOP o duong train: hop di cung anh) + vung goi y 78q (loss B3; khoi attention trong vung =
# chan doan MOI luot). Rieng: B1 / B2 cau report 78s (CSR theo study POOL); B4 dich day 78s. Luc cham KHONG doc gi o day (cong C6).
assert 'HEAD' in globals() and 'DS' in globals(), 'chay C2 truoc'
NGUON_78 = {'f': ('stage4_nb78f', 'r78f_v1'), 'q': ('stage4_nb78q', 'r78q_v1'), 's': ('stage4_nb78s', 'r78s_v1')}   # CHI DOC
TI_LE_GIU, HAT_GIU = 0.02, 7802                       # 2% benh nhan train GIU RIENG (CUNG hat 78g / 78m)
O_LUOI = 14                                           # patch RAD-DINO 14 px -> luoi G_LUOI x G_LUOI token
G_LUOI = SIZE // O_LUOI
assert G_LUOI * G_LUOI == N_TOK, (G_LUOI, N_TOK)
_R_DAY = DICH_78S['vung_day']['r']
# TAT, KHONG xoa (Tuan 07/10; 78s that: chi 21 o ket cau duong tren ca POOL): phan PHAN_TAT bo khoi loss + lambda (B4 = thuoc
# vung + bat thuong; dau DauDay van du kenh, kenh tat khong co gradient). Bat lai: PHAN_TAT = () (doi -> RUN_TAG moi).
PHAN_TAT = ('ket_cau',)
assert set(PHAN_TAT) <= {'thuoc_vung', 'bat_thuong', 'ket_cau'}, f'PHAN_TAT {PHAN_TAT}: chi tat duoc phan B4'
PHAN_CFG = {'B0': {}, 'B1': {'cau': 0.3}, 'B2': {'cau': 0.3}, 'B3': {'vung': 0.3},
            'B4': {k: float(_R_DAY[k]) for k in ('thuoc_vung', 'bat_thuong', 'ket_cau') if k not in PHAN_TAT}}[LUOT_78U]   # phan -> r
assert all(r > 0 for r in PHAN_CFG.values()) and (PHAN_CFG or LUOT_78U == 'B0'), f'{LUOT_78U}: PHAN_CFG {PHAN_CFG} (PHAN_TAT {PHAN_TAT})'
K_KC = len(DICH_78S['vung_day']['ket_cau'])           # 5 kenh ket cau (B4)
_KK78 = {}


def _lay78(nb, ten):
    if nb not in _KK78: _KK78[nb] = kiem_ke(NGUON_78[nb])[1]          # MOT lan kiem ke moi notebook nguon
    ds = _KK78[nb].get(ten) or []
    assert ds, f'thieu {ten} trong cxrlt_2024/{"/".join(NGUON_78[nb])} -> chay 78{nb} truoc'
    if xung_dot(ds): print(f'  ⚠ {ten}: {len(ds)} ban KHAC noi dung -> dung ban moi nhat ({ds[0]["mt"]})')
    return tai(ds[0])


def _nguon(nb, ten):
    '''Ban DA dung (moi nhat) cua `ten` -> dict(tep, id, md5): ghi vao meta backbone_ft.pt.'''
    b = _KK78[nb][ten][0]
    return dict(tep=f'{"/".join(NGUON_78[nb])}/{ten}', id=b['id'], md5=b['md5'])


def _chuoi(o):
    '''Moi gia tri chuoi (de quy) cua json -> tim phien ban cau hinh / mo hinh ghim ma KHONG phu thuoc ten khoa cua meta 78s.'''
    if isinstance(o, dict): return [x for v in o.values() for x in _chuoi(v)]
    if isinstance(o, list): return [x for v in o for x in _chuoi(v)]
    return [o] if isinstance(o, str) else []


_t0 = time.time()
META_78Q = json.load(open(_lay78('q', 'meta_78q.json'), encoding='utf-8'))
assert META_78Q['muc_tieu']['phien_ban'] == MUC_TIEU['phien_ban'], 'muc tieu luc chay 78q KHAC cau hinh kho -> DUNG'
VUNG = list(META_78Q['vung']); R_VUNG = len(VUNG); VUNG_IDX = {v: i for i, v in enumerate(VUNG)}
assert R_VUNG == 36, R_VUNG
VG = pd.read_parquet(_lay78('q', 'vung_goi_y_78q.parquet'))                    # anh (78f), dicom_id, nhan, mat_na, nguon
ANH_F = pd.read_parquet(_lay78('f', 'anh_train.parquet'), columns=['anh', 'dicom_id', 'W', 'H'])
VT = pd.read_parquet(_lay78('f', 'vung_train.parquet'), columns=['anh', 'vung', 'x1', 'y1', 'x2', 'y2'])
assert ANH_F.anh.tolist() == list(range(len(ANH_F))), 'anh_train 78f: cot anh phai la 0..n-1'
assert VG.anh.between(0, len(ANH_F) - 1).all() and (ANH_F.dicom_id.to_numpy()[VG.anh.to_numpy()] == VG.dicom_id.to_numpy()).all(), \
    'vung_goi_y (78q) va anh_train (78f) KHAC phien ban (anh <-> dicom lech) -> DUNG'
assert VT.anh.between(0, len(ANH_F) - 1).all() and VT.vung.between(0, R_VUNG - 1).all(), 'vung_train 78f: chi so anh / vung ngoai bang'
N_O = len(ORDER)
_ri = pd.Series(np.arange(N_O), index=ORDER.dicom_id.to_numpy())
for _ten, _d in (('vung_goi_y_78q', VG.dicom_id), ('anh_train (78f)', ANH_F.dicom_id)):
    assert _d.isin(_ri.index).all() and TF_[_ri.loc[_d].to_numpy()].all(), f'{_ten}: co anh NGOAI train -> DUNG'

# ---- 2% benh nhan GIU RIENG (chan doan; KHONG vao loss phu) ----
GIU_RIENG = chon_giu_rieng(SUB[TF_], TI_LE_GIU, HAT_GIU)
_vao = TF_ & ~np.isin(SUB, GIU_RIENG)
I_GIU = np.where(TF_ & np.isin(SUB, GIU_RIENG))[0]
POOL = np.where(_vao)[0]                              # anh train KHONG giu rieng -> vao loss phu
POOL_HANG = np.zeros(N_O, bool); POOL_HANG[POOL] = True
I_GIU_C = I_GIU[:512] if THU_NHANH else I_GIU         # chan doan giu rieng (THU_NHANH: 512 anh)

# ---- vung: hop 36 vung -> toa do anh 518; mat na vung goi y (du phong / khong) theo anh 78f ----
SG_HANG = np.full(N_O, -1, 'int64'); SG_HANG[_ri.loc[ANH_F.dicom_id].to_numpy()] = ANH_F.anh.to_numpy()
HOP518 = bang_hop_518(VT.anh, VT.vung, VT.x1, VT.y1, VT.x2, VT.y2, ANH_F.W.to_numpy(), ANH_F.H.to_numpy(), len(ANH_F),
                      R_VUNG, 1022, SIZE)
M_DP = np.zeros((len(ANH_F), C), 'int64'); M_KDP = np.zeros((len(ANH_F), C), 'int64')
M_DP[VG.anh.to_numpy(), VG.nhan.to_numpy()] = VG.mat_na.to_numpy()
_kdp = VG.nguon.astype(str).to_numpy() != 'du_phong'
M_KDP[VG.anh.to_numpy()[_kdp], VG.nhan.to_numpy()[_kdp]] = VG.mat_na.to_numpy()[_kdp]
# cong HINH HOC: kho anh 518 = letterbox 1022 -> 518 (C2). Vung DEM ngoai khung anh goc phai TOI, trong khung SANG.
_lech_vuong = ANH_F[(ANH_F.W - ANH_F.H).abs() > 0.05 * ANH_F[['W', 'H']].max(1)]    # anh KHONG vuong -> co vung dem
assert len(_lech_vuong), 'khong anh thang nao khong vuong -> khong kiem duoc hinh hoc kho anh'
_mau = _lech_vuong.sample(n=min(200, len(_lech_vuong)), random_state=0)
_toi, _sang = [], []
for _r in _mau.itertuples(index=False):
    _x0, _y0, _x1, _y1 = (float(v) * SIZE / 1022 for v in hop_sang_letterbox(0.0, 0.0, _r.W, _r.H, _r.W, _r.H, 1022))
    _im = np.asarray(Image.open(os.path.join(LOCAL, f'{_r.dicom_id}.jpg')).convert('L'), 'float32')
    _ngoai = np.ones(_im.shape, bool)
    _ngoai[max(int(_y0) - 3, 0):int(np.ceil(_y1)) + 3, max(int(_x0) - 3, 0):int(np.ceil(_x1)) + 3] = False
    _trong = np.zeros(_im.shape, bool)
    _trong[int(np.ceil(_y0)) + 3:int(_y1) - 3, int(np.ceil(_x0)) + 3:int(_x1) - 3] = True
    if _ngoai.sum() >= 100 and _trong.sum() >= 100: _toi.append(_im[_ngoai].mean()); _sang.append(_im[_trong].mean())
assert len(_toi) >= min(50, len(_mau)) and np.median(_toi) < 8 and np.median(_sang) > 40, (
    f'hinh hoc kho 518 KHAC letterbox 1022 -> 518: vung dem trung vi {np.median(_toi):.1f}, khung anh {np.median(_sang):.1f} '
    f'({len(_toi)} anh) -> hop vung LECH anh, DUNG')
print(f'cong hinh hoc: {len(_toi)} anh — vung dem ngoai letterbox trung vi {np.median(_toi):.1f} / 255, trong khung '
      f'{np.median(_sang):.1f} / 255 -> hop vung khop anh 518')
NHAN_PHOI = [LAB40.index(l) for l in MUC_TIEU['vung_du_phong']['nhan']]         # 5 nhan trong phoi (chan doan tach rieng)
PHOI_T = torch.zeros(C, dtype=torch.bool, device=dev); PHOI_T[NHAN_PHOI] = True
SG_HANG_T, POOL_T = torch.from_numpy(SG_HANG).to(dev), torch.from_numpy(POOL_HANG).to(dev)
M_DP_T, M_KDP_T = torch.from_numpy(M_DP).to(dev), torch.from_numpy(M_KDP).to(dev)
NGUON_TEP = [_nguon('q', 'meta_78q.json'), _nguon('q', 'vung_goi_y_78q.parquet'), _nguon('f', 'anh_train.parquet'),
             _nguon('f', 'vung_train.parquet')]
print(f'C3 78u ({LUOT_78U}): POOL {len(POOL):,} anh train ({len(np.unique(SUB[POOL])):,} benh nhan) | GIU RIENG {len(GIU_RIENG):,} '
      f'benh nhan, {len(I_GIU):,} anh (chan doan {len(I_GIU_C):,}) | hop: {int((SG_HANG >= 0).sum()):,} anh thang | (anh, nhan) vung '
      f'goi y: du phong {int((M_DP > 0).sum()):,} / khong {int((M_KDP > 0).sum()):,} | {time.time() - _t0:.0f} s')

# ---- 78s (B1 / B2 / B4): cung phien ban cau hinh dich (tim trong MOI chuoi cua meta_78s) ----
META_78S, _GT78S = None, []
if LUOT_78U in ('B1', 'B2', 'B4'):
    META_78S = json.load(open(_lay78('s', 'meta_78s.json'), encoding='utf-8'))
    _GT78S = _chuoi(META_78S)
    assert DICH_78S['phien_ban'] in _GT78S, (f"meta_78s KHONG ghi phien ban cau hinh {DICH_78S['phien_ban']} -> 78s chay voi "
                                              'dich_78s.json KHAC kho, DUNG')
    NGUON_TEP.append(_nguon('s', 'meta_78s.json'))

# ---- B1 / B2: cau report TRAIN (78s) — CSR theo chi so study POOL; hang ORDER -> study POOL (ST_T) hoac -1 (giu rieng / ngoai train
#      / study KHONG co cau, quy tac khoi_cau); bang RIENG cho study giu rieng (CHI chan doan truy hoi) ----
ST_T = PTR_T = IDX_T = ST_G_T = PTR_G_T = IDX_G_T = MA_G_T = W_CAU_T = SUB_T = E_CAU_T = TK_CAU_T = TOK_T = None
ST_HANG = ST_G_HANG = None; D_TXT = None
if LUOT_78U in ('B1', 'B2'):
    CS = pd.read_parquet(_lay78('s', 'cau_study_train.parquet'), columns=['study_id', 'ma_cau'])
    CT = pd.read_parquet(_lay78('s', 'cau_train.parquet'), columns=['ma_cau', 'tan_so'])     # KHONG doc cot text (luat PhysioNet)
    N_CAU = len(CT)
    assert np.array_equal(CT.ma_cau.to_numpy('int64'), np.arange(N_CAU)), 'cau_train 78s: ma_cau phai la 0..n-1 (= hang vector)'
    _sid = ORDER.study_id.to_numpy().astype('int64')
    _cs_st, _cs_ma = CS.study_id.to_numpy('int64'), CS.ma_cau.to_numpy('int64')
    assert np.isin(_cs_st, _sid[TR]).all(), 'cau_study_train 78s co study NGOAI train -> DUNG'
    assert len(_cs_ma) and _cs_ma.min() >= 0 and _cs_ma.max() < N_CAU, 'cau_study_train 78s: ma_cau ngoai cau_train -> DUNG'
    assert np.array_equal(np.bincount(_cs_ma, minlength=N_CAU), CT.tan_so.to_numpy('int64')), \
        'cau_train 78s: tan_so != so study train chua cau -> hai tep 78s lech phien ban, DUNG'

    def _csr(hangs):
        '''Hang ORDER `hangs` -> (st_hang [N_O]: chi so study (0..n-1, xep theo study_id) hoac -1, ptr, idx). CHI study co >= 1 cau.'''
        st_u = np.unique(_sid[hangs]); st_u = st_u[np.isin(st_u, _cs_st)]
        m = np.isin(_cs_st, st_u)
        ptr, idx = bang_cau(np.searchsorted(st_u, _cs_st[m]), _cs_ma[m], len(st_u))
        st_h = np.full(N_O, -1, 'int64'); h = hangs[np.isin(_sid[hangs], st_u)]
        st_h[h] = np.searchsorted(st_u, _sid[h])
        return st_h, ptr, idx

    ST_HANG, _p, _i = _csr(POOL)
    ST_G_HANG, _pg, _ig = _csr(I_GIU)
    assert (ST_HANG[~POOL_HANG] == -1).all() and (ST_G_HANG[POOL_HANG] == -1).all() and (ST_G_HANG[~TF_] == -1).all(), \
        'bang cau: anh giu rieng / ngoai train co st >= 0 -> DUNG'
    ST_T, PTR_T, IDX_T = torch.from_numpy(ST_HANG).to(dev), _p.to(dev), _i.to(dev)
    ST_G_T, PTR_G_T, IDX_G_T = torch.from_numpy(ST_G_HANG).to(dev), _pg.to(dev), _ig.to(dev)
    MA_G_T = torch.unique(IDX_G_T)                                    # kho cau cua cac study giu rieng (truy hoi)
    W_CAU_T = trong_so_cau(CT.tan_so.to_numpy('int64')).to(dev)
    SUB_T = torch.from_numpy(SUB.astype('int64')).to(dev)
    NGUON_TEP += [_nguon('s', 'cau_study_train.parquet'), _nguon('s', 'cau_train.parquet')]
    if LUOT_78U == 'B1':
        _E = np.load(_lay78('s', 'cau_emb.npy')); _TK = np.load(_lay78('s', 'cau_emb_thong_ke.npy'))
        assert _E.dtype == np.float16 and _E.ndim == 2 and _E.shape[0] == N_CAU and np.isfinite(_E).all(), 'cau_emb 78s sai hinh / kieu'
        assert _TK.dtype == np.float32 and _TK.shape == (2, _E.shape[1]) and (_TK[1] > 0).all(), 'cau_emb_thong_ke 78s sai'
        D_TXT = int(_E.shape[1])
        E_CAU_T, TK_CAU_T = torch.from_numpy(_E).to(dev), torch.from_numpy(_TK).to(dev)
        NGUON_TEP += [_nguon('s', 'cau_emb.npy'), _nguon('s', 'cau_emb_thong_ke.npy')]
        del _E, _TK
    else:
        _TOK = np.load(_lay78('s', 'cau_token.npy')); _nz = _TOK != 0
        assert _TOK.dtype == np.int32 and _TOK.shape == (N_CAU, 64) and _nz[:, 0].all() and not (_nz[:, 1:] & ~_nz[:, :-1]).any(), \
            'cau_token 78s sai: int32 [n_cau, 64], token dau khac 0, pad 0 BEN PHAI'
        assert TXT_MODEL in _GT78S and TXT_REV in _GT78S, ('meta_78s KHONG ghi BiomedBERT ghim (TXT_MODEL @ TXT_REV) -> token 78s '
                                                         'khong chac khop thap B2, DUNG')
        TOK_T = torch.from_numpy(_TOK).to(dev)
        NGUON_TEP.append(_nguon('s', 'cau_token.npy'))
        # D-T2 (quyet dinh 07/10): thap hoc cung z-score dau ra TB bang CUNG thong ke voi vector dong cung cua B1 -> B2 - B1 CHI
        # khac o tinh hoc cung (vector BiomedBERT tho gan nhu cung huong, cos ~0.98)
        _TK = np.load(_lay78('s', 'cau_emb_thong_ke.npy'))
        assert _TK.dtype == np.float32 and _TK.ndim == 2 and _TK.shape[0] == 2 and np.isfinite(_TK).all() and (_TK[1] > 0).all(), \
            'cau_emb_thong_ke 78s sai'
        TK_CAU_T = torch.from_numpy(_TK).to(dev)
        NGUON_TEP.append(_nguon('s', 'cau_emb_thong_ke.npy'))
        del _TOK, _nz, _TK
    print(f'cau 78s: {N_CAU:,} cau khac nhau | POOL: {int((ST_HANG >= 0).sum()):,} anh / {len(_p) - 1:,} study co cau ({int(_i.numel()):,} '
          f'cap) | GIU RIENG: {int((ST_G_HANG >= 0).sum()):,} anh / {len(_pg) - 1:,} study, kho truy hoi {int(MA_G_T.numel()):,} cau')
    del CS, CT, _cs_st, _cs_ma

# ---- B4: dich day 78s theo (anh 78f, vung); loss CHI anh POOL co scene graph, chan doan CHI anh giu rieng ----
DV_HANG = DV_HANG_G = DV_HANG_T = DV_HANG_G_T = BT_T = KC_T = BT_NP = KC_NP = None
if LUOT_78U == 'B4':
    _DVI = pd.read_parquet(_lay78('s', 'dich_vung_index.parquet'), columns=['anh', 'dicom_id'])
    assert np.array_equal(_DVI.anh.to_numpy('int64'), np.arange(len(ANH_F))) and \
        (_DVI.dicom_id.to_numpy() == ANH_F.dicom_id.to_numpy()).all(), 'dich_vung_index 78s KHAC thu tu anh_train 78f -> DUNG'
    with np.load(_lay78('s', 'dich_vung_train.npz')) as _z: BT_NP, KC_NP = _z['bat_thuong'], _z['ket_cau']
    assert BT_NP.dtype == np.int8 and BT_NP.shape == (len(ANH_F), R_VUNG) and KC_NP.dtype == np.int8 \
        and KC_NP.shape == (len(ANH_F), R_VUNG, K_KC), f'dich_vung_train 78s sai hinh / kieu {BT_NP.shape} {KC_NP.shape}'
    assert set(np.unique(BT_NP).tolist()) <= {-1, 0, 1} and set(np.unique(KC_NP).tolist()) <= {-1, 0, 1}, 'dich day ngoai {-1, 0, 1}'
    DV_HANG = np.where(POOL_HANG & (SG_HANG >= 0), SG_HANG, -1)               # loss: anh POOL co scene graph -> hang 78f
    DV_HANG_G = np.full(N_O, -1, 'int64'); DV_HANG_G[I_GIU] = SG_HANG[I_GIU]  # chan doan: anh giu rieng co scene graph
    DV_HANG_T, DV_HANG_G_T = torch.from_numpy(DV_HANG).to(dev), torch.from_numpy(DV_HANG_G).to(dev)
    BT_T, KC_T = torch.from_numpy(BT_NP).to(dev), torch.from_numpy(KC_NP).to(dev)
    NGUON_TEP += [_nguon('s', 'dich_vung_index.parquet'), _nguon('s', 'dich_vung_train.npz')]
    print(f'dich day 78s: {len(ANH_F):,} anh x {R_VUNG} vung | bat thuong (vung) 1 / 0 / -1: '
          + ' / '.join(f'{int((BT_NP == v).sum()):,}' for v in (1, 0, -1)) + f' | ket cau {K_KC} kenh 1 / 0: '
          + ' / '.join(f'{int((KC_NP == v).sum()):,}' for v in (1, 0)) + f' | loss: {int((DV_HANG >= 0).sum()):,} anh POOL')
    del _DVI
# ==== 79a · DU LIEU LUONG VUNG (CHI anh train 78f / 78s; doc QUA API, KHONG ghi Drive) ====
# PHAN_CFG: phan 'cau' cua nen B2 / B1 (NGUYEN VAN) + phan luong cua luot (r gia dinh, KHONG chinh o vong sang loc).
assert PHAN_CFG == {'cau': 0.3} and float(LUONG_CFG['phan']['cau']) == PHAN_CFG['cau'], (PHAN_CFG, LUONG_CFG['phan'])
PHAN_CFG = {k: float(v) for k, v in LUONG_CFG['phan'].items()}
assert set(PHAN_CFG) <= {'cau', 'vung_cau', 'bat_thuong', 'hop'} and all(r > 0 for r in PHAN_CFG.values()), PHAN_CFG
assert ('hop' not in PHAN_CFG) or bool(LUONG_CFG['neo']), 'loss hop can neo (luot khong neo: hop giu nguyen hop_tb)'
TOI_DA_VUNG = int(CFG_79['toi_da_vung'])
_t0v = time.time()
# (a) cau vung 78f: cap (anh 78f, vung) -> hang_txt (cau TRUNG CHU = MOT hang) -> CSR theo cap = anh * R_VUNG + vung
VC = pd.read_parquet(_lay78('f', 'vung_cau_train.parquet'), columns=['anh', 'vung', 'hang_cau'])
_CI = pd.read_parquet(_lay78('f', 'cau_vung_index.parquet'), columns=['hang', 'hang_txt'])
assert np.array_equal(_CI.hang.to_numpy('int64'), np.arange(len(_CI))), 'cau_vung_index 78f: cot hang phai la 0..n-1'
assert VC.anh.between(0, len(ANH_F) - 1).all() and VC.vung.between(0, R_VUNG - 1).all() and VC.hang_cau.between(0, len(_CI) - 1).all(), \
    'vung_cau_train 78f: chi so anh / vung / cau ngoai bang -> 78f lech phien ban'
_ht = _CI.hang_txt.to_numpy('int64')[VC.hang_cau.to_numpy('int64')]
_cap = VC.anh.to_numpy('int64') * R_VUNG + VC.vung.to_numpy('int64')
_kv0 = np.unique(_cap * (1 << 31) + _ht); _cap, _ht = _kv0 >> 31, _kv0 & ((1 << 31) - 1)        # bo cap (cap, cau) trung
N_TXT_V = int(_CI.hang_txt.max()) + 1
_pv, _iv = bang_cau(_cap, _ht, len(ANH_F) * R_VUNG)
PTR_V_T, IDX_V_T = _pv.to(dev), _iv.to(dev)
W_V_T = trong_so_cau(np.maximum(np.bincount(_ht, minlength=N_TXT_V), 1)).to(dev)     # w = 1 / log(1 + so cap co cau)
CO_CAU_V = (np.diff(_pv.numpy()).reshape(len(ANH_F), R_VUNG) > 0).any(1)             # anh 78f co >= 1 cau vung
NGUON_TEP += [_nguon('f', 'vung_cau_train.parquet'), _nguon('f', 'cau_vung_index.parquet')]
# (b) phia van ban: thap HOC (nen B2) -> token cau vung; thap DONG (nen B1) -> vector BiomedBERT 78f (CPU, gather theo lo)
TOK_VUNG_T = E_VUNG_T = None; N_CAT_V = 0
if THAP_HOC:
    # VAN BAN cau vung: CHI tren Colab, token hoa bang CUNG tokenizer / so token voi cau_token 78s (thap B2); KHONG in, KHONG ghi,
    # xoa ngay sau khi token hoa (luat PhysioNet).
    from transformers import AutoTokenizer
    _can = np.unique(_ht)
    _vb = pd.read_parquet(_lay78('f', 'cau_van_ban.parquet'), columns=['hang_txt', 'text'])
    assert np.array_equal(_vb.hang_txt.to_numpy('int64'), np.arange(len(_vb))) and len(_vb) == N_TXT_V, \
        'cau_van_ban 78f lech cau_vung_index -> DUNG'
    _tkv, N_CAT_V = token_hoa_cau(AutoTokenizer.from_pretrained(TXT_MODEL, revision=TXT_REV), _vb.text.to_numpy()[_can].tolist(),
                                  int(DICH_78S['cau']['max_tok']))
    del _vb
    _TV = np.zeros((N_TXT_V, _tkv.shape[1]), 'int32'); _TV[_can] = _tkv
    assert (_TV[_can, 0] != 0).all(), 'cau vung token rong -> DUNG'
    TOK_VUNG_T = torch.from_numpy(_TV).to(dev)
    NGUON_TEP.append(_nguon('f', 'cau_van_ban.parquet'))
    del _tkv, _TV, _can
else:
    _E = np.load(_lay78('f', 'cau_vung_emb.npy'))
    assert _E.dtype == np.float16 and _E.shape == (N_TXT_V, TK_CAU_T.shape[1]) and np.isfinite(_E).all(), 'cau_vung_emb 78f sai hinh / kieu'
    E_VUNG_T = torch.from_numpy(_E)                                     # CPU (lon); x_vung gather theo lo
    NGUON_TEP.append(_nguon('f', 'cau_vung_emb.npy'))
    del _E
# (c) co bat thuong tung (anh 78f, vung) — dich_vung_train 78s (CUNG bang ma B4 doc)
_DVI = pd.read_parquet(_lay78('s', 'dich_vung_index.parquet'), columns=['anh', 'dicom_id'])
assert np.array_equal(_DVI.anh.to_numpy('int64'), np.arange(len(ANH_F))) and (_DVI.dicom_id.to_numpy() == ANH_F.dicom_id.to_numpy()).all(), \
    'dich_vung_index 78s KHAC thu tu anh_train 78f -> DUNG'
with np.load(_lay78('s', 'dich_vung_train.npz')) as _z: _BT = _z['bat_thuong']
assert _BT.dtype == np.int8 and _BT.shape == (len(ANH_F), R_VUNG) and set(np.unique(_BT).tolist()) <= {-1, 0, 1}, 'bat_thuong 78s sai'
BT_V_T = torch.from_numpy(_BT.astype('int64')).to(dev)
NGUON_TEP += [_nguon('s', 'dich_vung_index.parquet'), _nguon('s', 'dich_vung_train.npz')]
# (d) hop khoi dau cua luong: TB hop HOP518 tren anh POOL co scene graph (CHI train)
HOP_TB = hop_tb(HOP518, SG_HANG, POOL, R_VUNG, SIZE)
_np_v = int(CO_CAU_V[SG_HANG[POOL][SG_HANG[POOL] >= 0]].sum())
print(f'C3 79a: cau vung {len(_cap):,} cap (anh, vung, cau) | {int((np.diff(_pv.numpy()) > 0).sum()):,} cap (anh, vung) co cau | '
      f'{len(np.unique(_ht)):,} cau khac nhau / {N_TXT_V:,} | anh POOL co cau vung {_np_v:,} | thap {"HOC: token " + str(int(DICH_78S["cau"]["max_tok"])) + f", {N_CAT_V:,} cau cham tran" if THAP_HOC else "DONG: vector 78f"} | '
      f'bat thuong 1 / 0 / -1: ' + ' / '.join(f'{int((_BT == v).sum()):,}' for v in (1, 0, -1))
      + f' | hop TB: w {HOP_TB[:, 2].min():.2f}-{HOP_TB[:, 2].max():.2f}, h {HOP_TB[:, 3].min():.2f}-{HOP_TB[:, 3].max():.2f} | {time.time() - _t0v:.0f} s')
del VC, _CI, _ht, _cap, _kv0, _pv, _iv, _DVI, _BT
VC = None
# ==== 78x · DU LIEU J v2 (docs/PLAN_JCA_V2.md §4.1; CHI anh train, doc QUA API, KHONG ghi Drive) ====
assert LUOT_79A == 'J_bert_dong' and LUOT_78U == 'B1' and not THAP_HOC, '78x: nen CO DINH J_bert_dong (B1)'
PHAN_CFG.update({k: float(v) for k, v in CFG_JV2['phan_them'].items()})
assert set(PHAN_CFG) == {'cau', 'vung_cau', 'bat_thuong', 'hop', 'nhan_vung', 'nhat_quan'} and all(r > 0 for r in PHAN_CFG.values()), PHAN_CFG
_t0x = time.time()
# (a) neo theo view (frontal) — view_group CO SAN luc cham (KHONG phai thong tin chi-train); study theo hang (y B)
NEO = V.view_group.to_numpy() == CFG_JV2['view_neo']
NEO_T = torch.from_numpy(NEO).to(dev)
ST_ID_T = torch.from_numpy(ST_ID).to(dev)
# (b) y A: hang ORDER -> anh 78f mang NGU NGHIA VUNG cua STUDY: co scene graph rieng -> chinh no; khong -> anh 78f cung study (chi so
#     nho nhat; thuoc tinh ImaGenome TRUNG giua anh cung study: 16.185 / 16.185 study, kiem trong may 12/10). CHI train.
_st78f = ST_ID[_ri.loc[ANH_F.dicom_id].to_numpy()]
_st_sg = pd.Series(ANH_F.anh.to_numpy('int64'), index=_st78f).groupby(level=0).min()
SG_ST = np.where(SG_HANG >= 0, SG_HANG, pd.Series(ST_ID).map(_st_sg).fillna(-1).to_numpy().astype('int64'))
SG_ST = np.where(TF_, SG_ST, -1).astype('int64')
SG_ST_T = torch.from_numpy(SG_ST).to(dev)
# (c) y 2: tap vung S cua (anh 78f, nhan) = vung CO HOP THAT (moi nhan) + vung THAY THE (CHI 3 nhan, Tuan 12/10) -> M1 / MC (cay cha-con)
_kj = sorted(n for n in _KK78['f'] if re.fullmatch(r'kiem_78f_\d{8}_\d{6}\.json', n))
assert _kj, 'thieu kiem_78f_*.json (78f C5) -> chay 78f truoc'
K78F = json.load(open(tai(_KK78['f'][_kj[-1]][0]), encoding='utf-8'))
assert list(K78F['vung']) == list(VUNG) and K78F['anh_xa'] == ANH_XA_78X['phien_ban'], 'vung / phien ban anh xa KHAC luc chay 78f -> DUNG'
assert all(v in VUNG for c, vs in CFG_JV2['cay'].items() for v in [c] + list(vs)), 'cay cha-con: ten vung ngoai 36 vung -> DUNG'
_TT = pd.read_parquet(_lay78('f', 'thuoc_tinh_train.parquet'), columns=['anh', 'vung', 'khoa', 'thay'])
assert _TT.anh.between(0, len(ANH_F) - 1).all() and _TT.vung.between(0, R_VUNG - 1).all() and _TT.thay.isin([0, 1]).all(), \
    'thuoc_tinh_train 78f: chi so anh / vung / thay ngoai bang -> DUNG'
_KN = {k: [LAB40.index(l) for l in ls] for k, ls in ANH_XA_78X['thuoc_tinh'].items()}
M_S = mat_na_s(_TT.anh, _TT.vung, _TT.khoa.astype(str), _TT.thay, _KN, [LAB40.index(l) for l in CFG_JV2['thay_the_nhan']], len(ANH_F), C)
CO_DINH_V3 = bit_co_dinh(ANH_XA_78X['co_dinh'], VUNG, LAB40)          # 78x J v3 (y D): 14 nhan -> vung co dinh
_bs3 = int((M_S != 0).sum())
M_S = bo_sung_co_dinh(M_S, CO_DINH_V3)                                  # vung co dinh = target du phong khi ImaGenome khong co vi tri
_bs3 = int((M_S != 0).sum()) - _bs3
CHA_V, CON_V = cay_vung(VUNG, CFG_JV2['cay'])
M1_NP, MC_NP = mo_rong_cha_con(M_S, CHA_V, CON_V)
M1_T, MC_T = torch.from_numpy(M1_NP).to(dev), torch.from_numpy(MC_NP).to(dev)
MIL_T = torch.zeros(C, dtype=torch.bool, device=dev); MIL_T[[LAB40.index(l) for l in CFG_JV2['mil']]] = True
NV_T = MIL_T.clone(); NV_T[sorted(CO_DINH_V3)] = True                  # 78x J v3: loss nhan-vung = 31 MIL + nhan co dinh (33)
NGUON_TEP += [_nguon('f', 'thuoc_tinh_train.parquet'), _nguon('f', _kj[-1])]
# (d) bao cao (CHI so dem): anh POOL theo nguon ngu nghia vung; do phu vi tri ca duong (anh POOL co ngu nghia, nhan MIL) theo tier
_pl = POOL_HANG
print(f"C3 78x: POOL {int(_pl.sum()):,} anh | frontal (neo) {int((_pl & NEO).sum()):,} / khong neo {int((_pl & ~NEO).sum()):,} | ngu nghia vung: "
      f"scene graph rieng {int((_pl & (SG_HANG >= 0)).sum()):,}, qua study (y A) {int((_pl & (SG_HANG < 0) & (SG_ST >= 0)).sum()):,}, "
      f"khong {int((_pl & (SG_ST < 0)).sum()):,} | {len(np.unique(ST_ID[I_TF])):,} study train (TB {len(I_TF) / len(np.unique(ST_ID[I_TF])):.2f} anh)")
_m = _pl & (SG_ST >= 0)
_Yp = YM[_m] > 0.5; _biet = (M1_NP[SG_ST[_m]] != 0) & _Yp
_tl = _biet.sum(0) / np.maximum(_Yp.sum(0), 1)
_mil = MIL_T.cpu().numpy()
print('do phu vi tri (ca duong POOL co ngu nghia vung, nhan MIL) trung vi theo tier: ' + ' | '.join(
    f"{t} {np.median(_tl[np.intersect1d(TIER[t], np.flatnonzero(_mil))]):.1%}" for t in ('head', 'medium', 'tail'))
      + f" | MIL {int(_mil.sum())} nhan | (anh, nhan) co mat na con: {int((MC_NP != 0).sum()):,} | {time.time() - _t0x:.0f} s")
del _TT, _st78f, _st_sg, _m, _Yp, _biet, _tl, _mil
# ==== 78x J v3 (docs/PLAN_JCA_V2.md §5; Tuan 13/10): lambda x3,23 + nguon nhan (y C; ngu_canh_train 78s, CHI train) ====
HE_SO_LAM = float(CFG_JV3['he_so_lam'])
PHAN_CFG = {k: v * HE_SO_LAM for k, v in PHAN_CFG.items()}            # lambda x3,23 = cuong do lan 1 (chuan hoa duong B1)
W_NGUON = float(CFG_JV3['nguon']['trong_so_ce'])
PHAN_CO_DINH = {'nguon': W_NGUON}                                      # CE nguon: lambda CO DINH (cat gradient ve backbone -> khong hieu chinh)
PHAN_CFG['nguon'] = W_NGUON
assert set(PHAN_CFG) == {'cau', 'vung_cau', 'bat_thuong', 'hop', 'nhan_vung', 'nhat_quan', 'nguon'}, PHAN_CFG
_NC3 = pd.read_parquet(_lay78('s', 'ngu_canh_train.parquet'), columns=['study_id', 'nhan', 'ngu_canh'])
assert np.isin(_NC3.study_id.to_numpy('int64'), ST_ID[TF_]).all(), 'ngu_canh_train 78s co study NGOAI train -> DUNG'
assert _NC3.nhan.between(0, C - 1).all(), 'ngu_canh_train 78s: chi so nhan ngoai 0..C-1 -> DUNG'
_ma3 = ma_nguon(_NC3.ngu_canh.astype(str).to_numpy(), CFG_JV3['nguon']['bang'])
NGUON_NHAN_NP = nguon_hang(ST_ID, POOL_HANG, YM, _NC3.study_id, _NC3.nhan, _ma3, C)
NGUON_NHAN_T = torch.from_numpy(NGUON_NHAN_NP).to(dev)
NGUON_TEP.append(_nguon('s', 'ngu_canh_train.parquet'))
_dp3 = int(((YM == 1) & POOL_HANG[:, None]).sum())
print(f"C3 78x J v3: lambda x{HE_SO_LAM} -> r {({k: round(v, 3) for k, v in PHAN_CFG.items()})} | nguon nhan: {int((NGUON_NHAN_NP >= 0).sum()):,} / "
      f"{_dp3:,} cap duong POOL biet nguon | theo nguon " + ' / '.join(
          f"{t} {int((NGUON_NHAN_NP == i).sum()):,}" for i, t in enumerate(CFG_JV3['nguon']['thu_tu']))
      + f" | kenh co dinh {len(CO_DINH_V3)} nhan, loss nhan-vung {int(NV_T.sum())} nhan | M_S them tu vung co dinh {_bs3:,} o")
del _NC3, _ma3
_KHOA_C3 = (THU_NHANH, RUN_TAG, LUOT_78U)             # C4 / C5 / C6 kiem: doi C0 thi PHAI chay lai C3
