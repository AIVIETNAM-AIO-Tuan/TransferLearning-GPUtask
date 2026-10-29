# ==== nhanh_c buoc p1_C4 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C4 · MODULE LOSS PHU cua luot (tao trong fork_rng, hat rieng) + ham loss phu + CONG dong ngau nhien ====
# HAI generator RIENG (khoi_cau, quy tac (iii)): GEN_PHU rut cau ung vien (B1 / B2), GEN_THAP CHI cho dropout cua thap BiomedBERT
# hoc cung (B2) -> B1 va B2 thay CUNG dong ung vien. Dong ngau nhien CHUNG (thu tu lo, hat worker, tang cuong, dropout dau doc)
# KHONG doi o moi luot B0..B4.
assert _KHOA_C3 == (THU_NHANH, RUN_TAG, LUOT_78U), 'C3 chay voi cau hinh C0 KHAC -> chay lai C0 -> C1 -> C2 -> C3 -> C4'
SEED_THAP, SEED_PHU, SEED_DAY = SEED + 2, SEED + 5, SEED + 6   # hat khoi tao: thap cau (78t: seed + 2), PhuCau (78t: seed + 5), DauDay
D_AN, D_CHIEU, TAU0 = 512, 256, 0.07                  # PhuCau B1 / B2 (khoi_cau; nhu 78t cau_bert / cau_clip)
THAP_MO, LR_THAP = 6, 1e-5                            # B2 (nhu 78t cau_clip): mo 6 tang tren cung, LR 1e-5; dau chieu: LR dau doc
TOI_DA_CAU = int(DICH_78S['cau']['toi_da_moi_study']) # so cau rut / study moi buoc (ung vien CHI trong lo)


def x_cau(ma):
    '''B1: vector BiomedBERT cua cau `ma` (78s, float16) z-score theo cau_emb_thong_ke -> float32 (vao ThapVector).'''
    return (E_CAU_T[ma].float() - TK_CAU_T[0]) / TK_CAU_T[1]


def thap_bert(tok, gen):
    '''B2: thap BiomedBERT hoc cung (forward bf16 autocast); dropout (train) rut tu `gen` = GEN_THAP (ThapBert fork dong chung);
       eval: gen = None.'''
    with torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        return PHU.text(tok.long(), gen=gen)


def tao_phu():
    '''Module loss phu cua LUOT_78U (None voi B0 / B3), tao trong fork_rng voi hat rieng -> KHONG tieu dong ngau nhien chung.
       B2: BiomedBERT ghim nap TRUOC (fork rieng), mo THAP_MO tang tren (mo_bert_tren). Thap gieo SEED_THAP NGAY truoc khi tao ->
       ThapVector.mlp (B1) TRUNG ThapBert.dau (B2); PhuCau gieo SEED_PHU -> chieu anh + nhiet do TRUNG giua B1 va B2 (nhu build_phu
       78t). B4: DauDay gieo SEED_DAY.'''
    if LUOT_78U not in ('B1', 'B2', 'B4'): return None
    if LUOT_78U == 'B2':
        with _fork():
            bert = mo_bert_tren(AutoModel.from_pretrained(TXT_MODEL, revision=TXT_REV), THAP_MO)
        assert getattr(bert.config, 'pad_token_id', 0) == 0 and len(bert.encoder.layer) > THAP_MO, (
            'BiomedBERT: pad id phai 0 (cau_token 78s) va so tang > THAP_MO')
        if CKPT_THAP: bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    with _fork():
        if LUOT_78U == 'B4':
            torch.manual_seed(SEED_DAY); m = DauDay(D_CXR, R_VUNG, K_KC)
        else:
            torch.manual_seed(SEED_THAP)
            thap = (ThapVector(D_TXT, D_AN, D_CHIEU) if LUOT_78U == 'B1'
                    else ThapBert(bert, D_AN, D_CHIEU, chuan=TK_CAU_T.cpu()))   # D-T2: z-score co dinh truoc dau chieu
            torch.manual_seed(SEED_PHU)
            m = PhuCau(D_CXR, thap, d_an=D_AN, d_chieu=D_CHIEU, tau0=TAU0)
    return m.to(dev)


def phu_78u(patch, cls, kv, b_hop, b_co, rows, y, gen_phu, gen_thap):
    '''Loss phu cua LUOT_78U tren MOT micro-batch (CHI luc train) -> dict {phan: loss} (+ 'dung1' tensor o B1 / B2, 'khoi' o B3) hoac
       None neu khong phan nao co cap hop le. rows = hang ORDER (dev); b_hop / b_co = hop DA bien doi (TF_HOP). Anh giu rieng /
       ngoai train / khong co bang -> KHONG vao (st = -1, POOL False, g = -1). Ngau nhien CHI tu hai generator RIENG: gen_phu (rut cau
       ung vien, MOT lan / buoc) va gen_thap (CHI dropout thap BiomedBERT hoc cung, B2).'''
    o = {}
    if LUOT_78U in ('B1', 'B2'):
        st = ST_T[rows]
        if bool((st >= 0).any()):                     # khong neo -> KHONG goi (quy tac (i) cua khoi_cau)
            ma_u = lay_cau_lo(st, PTR_T, IDX_T, TOI_DA_CAU, gen_phu)
            P = ma_tran_duong(st, PTR_T, IDX_T, ma_u)
            loai = mat_na_loai(P, SUB_T[rows], st)
            z_cau = PHU.text(x_cau(ma_u)) if LUOT_78U == 'B1' else thap_bert(TOK_T[ma_u], gen_thap)
            o['cau'], o['dung1'] = infonce_cau(PHU.z_anh(cls), z_cau, P, loai, W_CAU_T[ma_u], PHU.tau(), st >= 0)
    elif LUOT_78U == 'B3':
        g = SG_HANG_T[rows]; hl = (g >= 0) & POOL_T[rows]
        if bool(hl.any()):
            Mn = mat_na_nhan(M_KDP_T[g[hl]], o_luoi_tu_hop(b_hop[hl], b_co[hl], G_LUOI, O_LUOI), R_VUNG)
            q = HEAD.q.weight.unsqueeze(0).expand(int(hl.sum()), -1, -1)
            lv, _, kh = loss_vung(attn_truoc_dropout(HEAD.img.at, q, kv[hl]), Mn, y[hl])
            if lv is not None: o['vung'], o['khoi'] = lv, kh
    elif LUOT_78U == 'B4':
        g = DV_HANG_T[rows]; hl = g >= 0
        if bool(hl.any()):
            o.update({q: v for q, v in loss_day(PHU(patch[hl]), dich_o_day(b_hop[hl], b_co[hl], g[hl], BT_T, KC_T, G_LUOI, O_LUOI),
                                                R_VUNG).items() if q not in PHAN_TAT})   # phan TAT (C3): khong vao loss
    return o or None


def x_vung(ma):
    '''J_bert_dong: vector BiomedBERT DONG CUNG cua cau vung `ma` (78f, float16, CPU) z-score bang thong ke 78s (nhu x_cau) -> float32.'''
    return (E_VUNG_T[ma.cpu()].to(dev, non_blocking=True).float() - TK_CAU_T[0]) / TK_CAU_T[1]


def phu_luong(ra, rows, b_hop, b_co, gen_vung, gen_thap_vung, dong_thap=False, y=None):
    """78x (J v2): phan LUONG VUNG cua loss phu (CHI luc train) -> dict {phan: loss} (+ 'dung1_vung'). Y A: cau vung / bat thuong /
       nhan_vung lay NGU NGHIA VUNG cua STUDY (SG_ST_T: scene graph rieng -> chinh no; lateral / khong scene graph -> anh 78f cung
       study); hop CHI anh co scene graph RIENG va NEO (frontal). Anh giu rieng / ngoai train KHONG vao (POOL_T). Ngau nhien CHI tu
       gen_vung / gen_thap_vung (nhu 79a). nhan_vung can y (nhan CXR-LT cua lo)."""
    o = {}; r = ra['r']
    gs = SG_ST_T[rows]; hs = (gs >= 0) & POOL_T[rows]                       # ngu nghia vung cua study (y A)
    g = SG_HANG_T[rows]; hh = (g >= 0) & POOL_T[rows] & NEO_T[rows]          # hop: scene graph RIENG + frontal (neo)
    if 'vung_cau' in PHAN_CFG:
        bi, vi, cap = cap_vung_lo(rows, SG_ST_T, POOL_T, PTR_V_T, R_VUNG)
        if bi.numel():
            ma_u, P, loai = dich_vung_cau(cap, SUB_T[rows][bi], gs[bi], PTR_V_T, IDX_V_T, TOI_DA_VUNG, gen_vung)
            with torch.set_grad_enabled(torch.is_grad_enabled() and not dong_thap):
                z_v = thap_bert(TOK_VUNG_T[ma_u], gen_thap_vung) if THAP_HOC else PHU.text(x_vung(ma_u))
            o['vung_cau'], o['dung1_vung'] = infonce_cau(PHU_V.z_anh(r[bi, vi]), z_v, P, loai, W_V_T[ma_u], PHU_V.tau())
    if 'bat_thuong' in PHAN_CFG and bool(hs.any()):
        lb = loss_bt(PHU_V.bt_logit(r[hs]), BT_V_T[gs[hs]])
        if lb is not None: o['bat_thuong'] = lb
    if 'hop' in PHAN_CFG and bool(hh.any()):
        lh = loss_hop(ra['hop_cac_tang'][:, hh], b_hop[hh], b_co[hh], SIZE)
        if lh is not None: o['hop'] = lh
    if 'nhan_vung' in PHAN_CFG and bool(hs.any()):
        assert y is not None, 'phu_luong: nhan_vung can y (nhan cua lo)'
        zv = ra.get('zv')
        if zv is None: zv = DAUV.z(r)                                       # khoi dong C4b: dau doc khong chay
        T_, V_ = dich_nhan_vung(y, torch.where(hs, gs, torch.full_like(gs, -1)), M1_T, MC_T, NV_T, R_VUNG)
        ln = asl_mat_na(zv, T_, V_)
        if ln is not None: o['nhan_vung'] = ln
    return o


def phu_79(patch, cls, kv, b_hop, b_co, rows, y, gen_phu, gen_thap, ra, gen_vung, gen_thap_vung):
    """78x: loss phu = phu_78u (phan 'cau' cua nen B1, NGUYEN VAN, chay TRUOC) + phu_luong (J v2) + y B (nhat_quan: CHI anh POOL;
       can ra['lg'] tu dau_v2)."""
    o = phu_78u(patch, cls, kv, b_hop, b_co, rows, y, gen_phu, gen_thap) or {}
    o.update(phu_luong(ra, rows, b_hop, b_co, gen_vung, gen_thap_vung, y=y))
    if 'nhat_quan' in PHAN_CFG and ra.get('lg') is not None:
        m = POOL_T[rows]
        if bool(m.any()):
            lq = loss_nhat_quan(ra['lg'][m], ST_ID_T[rows][m], NEO_T[rows][m])
            if lq is not None: o['nhat_quan'] = lq
    if 'nguon' in PHAN_CFG and ra.get('h_dd') is not None:              # 78x J v3 (y C): CE nguon CHI vao DAU_NGUON (h da cat gradient)
        m = POOL_T[rows]
        if bool(m.any()):
            ln = loss_nguon(DAU_NGUON.delta(ra['h_dd'][m]), NGUON_NHAN_T[rows][m])
            if ln is not None: o['nguon'] = ln
    return o or None


def phu_hoc():
    '''Tham so HOC cua module loss phu (B2: CHI phan mo cua thap + dau chieu, chieu anh, nhiet do) -> luu trong rs_.'''
    return {} if PHU is None else {n: p.detach() for n, p in PHU.named_parameters() if p.requires_grad}


def nap_phu_hoc(sd):
    '''Nap lai tham so HOC tu rs_: dung DU ten tham so hoc, khong thua, khong thieu.'''
    if PHU is None:
        assert not sd, 'rs_ co module loss phu ma luot nay khong co -> DUNG'
        return
    hoc = {n for n, p in PHU.named_parameters() if p.requires_grad}
    assert set(sd) == hoc, f'rs_: tham so hoc cua module loss phu lech ({sorted(set(sd) ^ hoc)[:3]}) -> DUNG'
    with torch.no_grad():
        for n, p in PHU.named_parameters():
            if n in sd: p.copy_(sd[n].to(p.device, p.dtype))


_s0 = _sts(); PHU = tao_phu()
assert all(torch.equal(a, b) for a, b in zip(_sts(), _s0)), 'tao module loss phu TIEU dong ngau nhien chung -> DUNG'
PARAM_PHU = [p for p in PHU.parameters() if p.requires_grad] if PHU is not None else []
# ---- 79a: luong vung + phia anh cua dich vung — tao trong fork_rng voi hat RIENG (KHONG tieu dong chung; PHU giu NGUYEN B2 / B1) ----
_s0 = _sts()
with _fork():
    torch.manual_seed(SEED + int(CFG_79['hat']['luong']))
    LUONG = LuongVungNeo(HOP_TB, d=D_CXR, R=R_VUNG, tang=tuple(CFG_79['tang']), n_head=int(CFG_79['n_head']), ff=int(CFG_79['ff']),
                      beta=float(CFG_79['beta']), t_mem=float(CFG_79['t_mem']), hai_chieu=bool(LUONG_CFG['hai_chieu']),
                      neo=bool(LUONG_CFG['neo']), luoi=G_LUOI, ckpt=bool(CFG_79['ckpt_jca'])).to(dev)
with _fork():
    torch.manual_seed(SEED + int(CFG_79['hat']['phu_vung']))
    PHU_V = PhuVung(D_CXR, d_an=D_AN, d_chieu=D_CHIEU, tau0=TAU0).to(dev)
assert all(torch.equal(a, b) for a, b in zip(_sts(), _s0)), 'tao LUONG / PHU_V TIEU dong ngau nhien chung -> DUNG'
LUONG.eval(); PHU_V.eval()
PARAM_LUONG = list(LUONG.parameters()) + list(PHU_V.parameters())
# ---- 78x: y 1 (DOCV) + y 2 / kenh MIL (DAUV) — tao trong fork_rng voi hat RIENG (KHONG tieu dong chung) ----
_s0 = _sts()
with _fork():
    torch.manual_seed(SEED + int(CFG_JV2['hat']['doc_vung']))
    DOCV = DocVung(R_VUNG, D_CXR, D_MLD).to(dev)
with _fork():
    torch.manual_seed(SEED + int(CFG_JV2['hat']['dau_vung']))
    DAUV = DauVung(D_CXR, C, [LAB40.index(l) for l in CFG_JV2['mil']]).to(dev)
assert all(torch.equal(a, b) for a, b in zip(_sts(), _s0)), 'tao DOCV / DAUV TIEU dong ngau nhien chung -> DUNG'
DOCV.eval(); DAUV.eval()
PARAM_LUONG += list(DAUV.ln.parameters()) + list(DAUV.lin.parameters())   # head nhan-vung: phia LUONG (khoi dong C4b + LR dau doc)
PARAM_V2 = list(DOCV.parameters()) + [DAUV.w]                              # phia DAU DOC: nhom dau doc + cat chung PARAM_DB (C6)
assert len({id(p) for p in PARAM_LUONG + PARAM_V2}) == len(PARAM_LUONG) + len(PARAM_V2) == (
    sum(1 for _ in LUONG.parameters()) + sum(1 for _ in PHU_V.parameters()) + sum(1 for _ in DOCV.parameters())
    + sum(1 for _ in DAUV.parameters())), 'nhom tham so J v2 trung / thieu -> DUNG'
# ---- 78x J v3: y C (DAU_NGUON) + y D (KENH_CD) — tham so khoi tao 0 (khong rut so ngau nhien) -> phia DAU DOC; y 5: write-back x g ----
_s0 = _sts()
DAU_NGUON = NguonNhan(C, D_MLD, len(CFG_JV3['nguon']['thu_tu'])).to(dev)
KENH_CD = KenhCoDinh(CO_DINH_V3, R_VUNG).to(dev)
assert all(torch.equal(a, b) for a, b in zip(_sts(), _s0)), 'tao DAU_NGUON / KENH_CD TIEU dong ngau nhien chung -> DUNG'
DAU_NGUON.eval(); KENH_CD.eval()
PARAM_V2 += list(DAU_NGUON.parameters()) + list(KENH_CD.parameters())
N_GHI_NGUOC = tang_ghi_nguoc(LUONG, float(CFG_JV3['ghi_nguoc_g']))
assert N_GHI_NGUOC == len(LUONG.tang) and all(float(l.ca_h.o.weight.detach().abs().max()) == 0 for l in LUONG.lop), \
    'y 5: W_o phai van = 0 (khoi dau trung tinh)'
N_TRAN = ap_tran_ghi(LUONG, float(CFG_JV3B['tran_ghi_rho']))          # 78x J v3b: tran write-back theo anh, SAU x5 (docs SPEC B.1d)
assert N_TRAN == len(LUONG.tang) and all(type(l) is LopJCANeoTran and l.rho == float(CFG_JV3B['tran_ghi_rho']) for l in LUONG.lop) \
    and all(type(l.ca_h.o) is LinearNhan for l in LUONG.lop), 'J v3b: tran / x5 chua dung o moi lop JCA -> DUNG'
print(f"78x J v3b C4: tran write-back rho {CFG_JV3B['tran_ghi_rho']} o {N_TRAN} lop JCA (tang {LUONG.tang}), x{CFG_JV3['ghi_nguoc_g']} giu; "
      'W_o van = 0')
if LUOT_78U == 'B2':
    _nl = len(PHU.text.mdl.encoder.layer)
    assert all(p.requires_grad == (n.startswith('encoder.layer.') and int(n.split('.')[2]) >= _nl - THAP_MO)
               for n, p in PHU.text.mdl.named_parameters()) and all(p.requires_grad for p in PHU.text.dau.parameters()), \
        'thap B2: CHI THAP_MO tang tren + dau chieu duoc hoc (embeddings + tang duoi + pooler dong cung)'
    assert int(TOK_T.max()) < PHU.text.mdl.config.vocab_size, 'token 78s ngoai tu vung cua thap B2 -> DUNG'
print(f'78u {LUOT_78U}: loss phu {PHAN_CFG or "KHONG (doi chung)"} | module '
      + (f'{type(PHU).__name__}: {sum(p.numel() for p in PARAM_PHU):,} tham so hoc' if PHU is not None else 'khong co')
      + (f' (thap: {THAP_MO}/{len(PHU.text.mdl.encoder.layer)} tang mo, LR {LR_THAP:g}, checkpointing {CKPT_THAP})'
         if LUOT_78U == 'B2' else ''))

# ---- cong thap (B2; quy tac (ii)): cung GEN_THAP + dong chung khac -> TRUNG tung bit; GEN_THAP khac -> khac; dong chung khong doi ----
if LUOT_78U == 'B2':
    _tk = TOK_T[:8]
    _g1 = torch.Generator(device=dev); _g1.manual_seed(1); _gs1 = _g1.get_state()
    with _fork(), torch.no_grad():
        PHU.train()
        torch.manual_seed(11); _a0 = _sts(); _za = thap_bert(_tk, _g1); _a1 = _sts()
        _g1.set_state(_gs1); torch.manual_seed(12); _zb = thap_bert(_tk, _g1)
        _g1.manual_seed(2); _zc = thap_bert(_tk, _g1)
    PHU.eval()
    assert all(torch.equal(a, b) for a, b in zip(_a0, _a1)), 'thap B2: forward DOI dong ngau nhien CHUNG -> DUNG'
    assert torch.equal(_za, _zb), 'thap B2: cung GEN_THAP, dong chung khac -> ket qua KHAC (dropout rut tu dong chung) -> DUNG'
    assert not torch.equal(_za, _zc), 'thap B2: GEN_THAP khac ma ket qua giong -> dropout KHONG rut tu generator rieng -> DUNG'
    del _tk, _g1, _gs1, _a0, _a1, _za, _zb, _zc
    print('cong thap B2: cung GEN_THAP + hai dong chung khac -> TRUNG tung bit | GEN_THAP khac -> khac | dong chung khong doi')

# ---- cong DONG NGAU NHIEN tren MOT lo train THAT (duong train: anh + hop): moi phan loss phu co mat + huu han; rut cau CHI tu
#      GEN_PHU, dropout thap CHI tu GEN_THAP (B2; B1 khong cham GEN_THAP); dong chung KHONG doi ----
if PHAN_CFG:
    _c = I_TF[_vao[I_TF] & (SG_HANG[I_TF] >= 0)]
    if LUOT_78U in ('B1', 'B2'): _c = _c[ST_HANG[_c] >= 0]
    _c = _c[CO_CAU_V[SG_HANG[_c]]]                    # 79a: anh co cau vung -> moi phan luong co mat trong lo kiem
    if LUOT_78U == 'B3': _c = _c[((M_KDP[SG_HANG[_c]] != 0) & (YM[_c] > 0.5)).any(1)]
    if LUOT_78U == 'B4': _c = _c[(BT_NP[SG_HANG[_c]] >= 0).any(1) & (KC_NP[SG_HANG[_c]] >= 0).any((1, 2))]
    # 78x: lo kiem CO cap frontal + khong frontal CUNG study (y A co anh khong scene graph; y B co cap) -> moi phan co mat
    _stc = ST_ID[_c]
    _kf = I_TF[_vao[I_TF] & ~NEO[I_TF] & (SG_HANG[I_TF] < 0) & (SG_ST[I_TF] >= 0) & np.isin(ST_ID[I_TF], _stc)]
    _c8 = _c[np.isin(_stc, ST_ID[_kf])][:8]
    _i1 = np.concatenate([_c8, _kf[np.isin(ST_ID[_kf], ST_ID[_c8])][:8]])
    assert len(_i1), f'{LUOT_78U}: khong anh train POOL nao du dieu kien cho cong loss phu -> bang C3 rong, DUNG'
    _b1 = next(iter(loader(_i1, train_tf, False, len(_i1), nw=0)))
    _gq = torch.Generator(device=dev); _gq.manual_seed(1); _gs = _gq.get_state()
    _gt = torch.Generator(device=dev); _gt.manual_seed(2); _gts = _gt.get_state()
    _gv = torch.Generator(device=dev); _gv.manual_seed(3); _gtv = torch.Generator(device=dev); _gtv.manual_seed(4)
    with torch.no_grad():
        bb_ft.eval(); HEAD.eval()
        if PHU is not None: PHU.train()
        _pt, _cq, _ra = fwd_luong_v2(_b1['image'].to(dev), False, _b1['row'].to(dev)); _kv = dau_v2(_pt, _ra)[2]
        _s0 = _sts()
        _oq = phu_79(_pt, _cq, _kv, _b1['hop'].to(dev), _b1['co_hop'].to(dev), _b1['row'].to(dev), _b1['y'].to(dev), _gq, _gt,
                     _ra, _gv, _gtv)
        _s1 = _sts()
    if PHU is not None: PHU.eval()
    assert all(torch.equal(a, b) for a, b in zip(_s1, _s0)), 'loss phu TIEU dong ngau nhien CHUNG -> DUNG'
    assert (LUOT_78U in ('B1', 'B2')) == (not torch.equal(_gq.get_state(), _gs)), 'rut cau KHONG (chi) tu GEN_PHU -> DUNG'
    assert (LUOT_78U == 'B2') == (not torch.equal(_gt.get_state(), _gts)), 'GEN_THAP: CHI dropout thap B2 duoc rut tu no -> DUNG'
    _thieu = [k for k in PHAN_CFG if _oq is None or k not in _oq or not bool(torch.isfinite(_oq[k]))]
    assert not _thieu, f'lo kiem {len(_i1)} anh POOL: THIEU / khong huu han phan loss phu {_thieu} -> DUNG'
    print(f'cong loss phu ({len(_i1)} anh POOL, duong train): ' + ', '.join(f'{k} {float(_oq[k]):.4f}' for k in PHAN_CFG)
          + ' | dong ngau nhien chung khong doi | rut cau: GEN_PHU | dropout thap: GEN_THAP')
    del _c, _i1, _b1, _gq, _gs, _gt, _gts, _pt, _cq, _kv, _s0, _s1, _oq, _thieu, _ra, _gv, _gtv
# ---- cong 79a: luong moi tao (ghi nguoc khoi tao 0) -> fwd_luong TRUNG fwd_ft (lo val THAT); hop moi tang = hop_tb ----
_bz = next(iter(loader(I_VA[:4], eval_tf, False, 4, nw=0)))
_xz = _bz['image'].to(dev)
_p0, _c0 = fwd_ft(_xz, False); _p1, _c1, _r1 = fwd_luong_v2(_xz, False, _bz['row'].to(dev))
_dz = max(float((_p0 - _p1).abs().max()), float((_c0 - _c1).abs().max()))
assert _dz <= 1e-5, f'luong moi tao DOI duong anh ({_dz:.2e}; ghi nguoc phai khoi tao 0) -> DUNG'
assert torch.allclose(_r1['hop_cac_tang'], LUONG.hop_tb[None, None].expand_as(_r1['hop_cac_tang']), atol=1e-5), 'hop khoi dau != hop_tb'
print(f"79a luong {LUOT_79A}: {sum(p.numel() for p in LUONG.parameters()):,} tham so luong + {sum(p.numel() for p in PHU_V.parameters()):,} "
      f"PHU_V | tang {LUONG.tang} | hai chieu {LUONG.hai_chieu} | neo {bool(LUONG_CFG['neo'])} | thap {'HOC' if THAP_HOC else 'DONG'} "
      f"(doi chung ghep cap {LUONG_CFG['doi_chung']}) | cong: fwd_luong = fwd_ft ({'TUNG BIT' if _dz == 0 else f'lech {_dz:.1e}'})")
del _bz, _xz, _p0, _c0, _p1, _c1, _r1, _dz
# ---- cong 78x: (1) dau_h_v2 khong token vung = dau_h 78u TUNG BIT; (2) MIL w = 0 -> logit J v2 = dau_h_v2 TUNG BIT; (3) token vung
#      DOI logit (y 1 song); (4) neo theo view: mau khong frontal giu hop_tb; (5) duong cham chi doc anh + view ----
_bz = next(iter(loader(I_VA[:8], eval_tf, False, 8, nw=0)))
_xz, _rz = _bz['image'].to(dev), _bz['row'].to(dev)
with torch.no_grad():
    _pz, _cz, _raz = fwd_luong_v2(_xz, False, _rz)
    _l0 = dau_h(HEAD, _pz)[0]; _l1 = dau_h_v2(HEAD, _pz)[0]; _l2 = dau_h_v2(HEAD, _pz, DOCV(HEAD.pin, _raz['r']))[0]
    _l3 = dau_v2(_pz, _raz)[0]
assert torch.equal(_l0, _l1), 'dau_h_v2 khong token vung KHAC dau_h -> DUNG'
assert float(DAUV.w.detach().abs().max()) == 0 and torch.equal(_l2, _l3), 'MIL w khoi tao KHAC 0 / dau_v2 lech -> DUNG'
_dl = float((_l2 - _l1).abs().max())
assert _dl > 0, 'token vung KHONG doi logit (y 1 chet) -> DUNG'
_kn = ~NEO_T[_rz]
print(f'cong 78x: dau_h_v2 = dau_h TUNG BIT | MIL w = 0 -> logit = dau doc y 1 TUNG BIT | token vung doi logit (max {_dl:.2e}) | lo val 8 '
      f'anh: {int(NEO_T[_rz].sum())} frontal (neo) / {int(_kn.sum())} khong neo')
del _bz, _xz, _rz, _pz, _cz, _raz, _l0, _l1, _l2, _l3, _dl, _kn
# ---- 78x o cat (Tuan duyet 26/10; SPEC B.2z): module o — adapter x 2 trong fork_rng (hat rieng), phep chieu skip (trong fork: nn.Linear rut so truoc khi dat 0) +
#      bo gop 6 view (tham so 0, khong rut so); vao nhom dau doc (PARAM_V2: LR dau doc, clip chung PARAM_DB) ----
_s0_o = _sts()
with _fork():
    torch.manual_seed(SEED + int(O_CFG['hat']['bo_o7']))
    BO_O7 = BoChinhO(D_CXR, int(O_CFG['r_adapter'])).to(dev)
with _fork():
    torch.manual_seed(SEED + int(O_CFG['hat']['bo_o11']))
    BO_O11 = BoChinhO(D_CXR, int(O_CFG['r_adapter'])).to(dev)
with _fork():                                                       # nn.Linear cua DuaVao rut so TRUOC khi dat 0
    DUA_O = DuaVao(D_CXR).to(dev)
GOP_O = GopStudy('attn', C, D_MLD, n_view=1 + SO_O).to(dev)
assert all(torch.equal(a, b) for a, b in zip(_sts(), _s0_o)), 'tao module o TIEU dong ngau nhien chung -> DUNG'
BO_O7.eval(); BO_O11.eval(); DUA_O.eval(); GOP_O.eval()
PARAM_V2 += list(DUA_O.parameters()) + list(BO_O7.parameters()) + list(BO_O11.parameters()) + list(GOP_O.parameters())
LUOI_O = luoi_518(dev)
_G_O, _W_O = luoi_con(hinh_luoi(1, dev))


def luoi_o(B):
    '''(g, w) cua luoi_con cho B anh (5 o luoi CUNG hinh hoc o moi anh), lien tuc.'''
    return _G_O.expand(int(B), -1, -1, -1).contiguous(), _W_O.expand(int(B), -1, -1).contiguous()


def fwd_o(img, grad, rows, mo_ghi=True, grad_bb=True, do=None):
    '''78x o cat: 5 o phong tu img (KHONG gradient ve backbone, chia khuc KHUC_O) -> token sau block 7 + token cuoi; skip (BO_O7 -> 2 x 2 diem
       con -> DUA_O; tinh lai luc backward) cong vao dau vao block 8 cua anh toan canh trong fwd_luong_v2 (boc luong.buoc); token o cuoi ->
       ra['o11'] (dau_v2_o). -> (patch, cls, ra) nhu fwd_luong_v2. Duong o khong rut so ngau nhien (drop-path CHI o anh toan canh, GEN_DP).'''
    with torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        t7, t11 = token_o(bb_ft, img, LUOI_O, MEAN, STD, KHUC_O, K_SKIP_O - 1)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        inj = skip_o(t7, BO_O7, DUA_O, *luoi_o(len(img)), ckpt=bool(grad) and torch.is_grad_enabled())
    del t7
    with skip_vao_luong(LUONG, inj, K_SKIP_O, do):
        p, c, ra = fwd_luong_v2(img, grad, rows, mo_ghi=mo_ghi, grad_bb=grad_bb)
    ra['o11'] = t11
    return p, c, ra


def dau_v2_o(patch, ra):
    '''78x o cat: dau_v2 + doc 6 view - dau J (HEAD + DOCV) doc RIENG anh toan canh va tung o (token cuoi qua BO_O11, float32 nhu anh toan
       canh; token vung cua anh toan canh dung chung) -> GOP_O (GopStudy attn, 6 view, tham so 0 = TB) -> h_bag (dong view 0) -> HEAD.out
       -> nguon / MIL / kenh co dinh nhu dau_v2. -> (logit [B, C], h_bag, kv = pin(patch) anh toan canh -> loss phu giu nguyen tren anh
       toan canh). Khi co grad: MOI o tinh lai luc backward (checkpoint) -> dinh VRAM ~ 1 o.'''
    zv = DAUV.z(ra['r'])
    kvv = DOCV(HEAD.pin, ra['r'])
    _, h0, kv = dau_h_v2(HEAD, patch, kvv)
    o11 = ra['o11']

    def _h_o(t, kv_):
        return dau_h_v2(HEAD, BO_O11(t.float()), kv_)[1]
    h_o = torch.stack([(_tuc_luong.checkpoint(_h_o, o11[:, k], kvv, use_reentrant=False) if torch.is_grad_enabled() else _h_o(o11[:, k], kvv))
                       for k in range(int(o11.shape[1]))], 1)
    h = gop_view_o(GOP_O, h0, h_o)
    lg = HEAD.out(h)
    dn = DAU_NGUON.delta(h)
    lg = KENH_CD(DAUV.mil(DAU_NGUON.gop(lg, dn), zv), zv)
    ra['zv'], ra['lg'], ra['h_dd'] = zv, lg, h.detach()
    return lg, h, kv


# ---- cong o (lo val THAT, eval): (1) phep chieu skip khoi tao 0 -> fwd_o = fwd_luong_v2 (<= 1e-5), duong o KHONG rut so ngau nhien; (2) adapter
#      = dong nhat TUNG BIT; (3) 6 view GIONG nhau -> dau_v2_o = dau_v2 (<= 1e-4), o that DOI logit; (4) truong nhin o; (5) tham so 0;
#      (6) du doan co o TAT DINH (hai lan = tung bit; GopStudy dung index_add_ atomic -> gop 6 view tinh dang day); (7) dang day = GopStudy ----
_bo_o = next(iter(loader(I_VA[:4], eval_tf, False, 4, nw=0)))
_xo_o, _ro_o = _bo_o['image'].to(dev), _bo_o['row'].to(dev)
with torch.no_grad():
    bb_ft.eval(); HEAD.eval(); LUONG.eval(); DOCV.eval(); DAUV.eval()
    _pa_o, _ca_o, _raa_o = fwd_luong_v2(_xo_o, False, _ro_o)
    _s1_o = _sts(); _pb_o, _cb_o, _rab_o = fwd_o(_xo_o, False, _ro_o); _s2_o = _sts()
    _la_o = dau_v2(_pa_o, _raa_o)[0]
    _lo_o = dau_v2_o(_pb_o, dict(_rab_o))[0]
    _l6_o = dau_v2_o(_pb_o, dict(_rab_o, o11=_pb_o[:, None].expand(-1, SO_O, -1, -1)))[0]
    _pb2_o, _, _rab2_o = fwd_o(_xo_o, False, _ro_o)                 # (6) TAT DINH: lan 2 = lan 1 tung bit (cong duong cham C6 so tung bit)
    _lo2_o = dau_v2_o(_pb2_o, dict(_rab2_o))[0]
    _kv_o = DOCV(HEAD.pin, _rab_o['r'])                             # (7) gop 6 view dang day = GopStudy.forward, tham so ngau nhien
    _h0_o = dau_h_v2(HEAD, _pb_o, _kv_o)[1]                         #     (generator rieng, khong cham dong chung)
    _ho_o = torch.stack([dau_h_v2(HEAD, BO_O11(_rab_o['o11'][:, _k_o].float()), _kv_o)[1] for _k_o in range(SO_O)], 1)
    _gr_o = GopStudy('attn', C, D_MLD, n_view=1 + SO_O).to(dev)
    _gg_o = torch.Generator(device=dev).manual_seed(7)
    for _p_o in (_gr_o.e_view, _gr_o.w, _gr_o.b, _gr_o.ln.weight, _gr_o.ln.bias):
        _p_o.copy_(torch.randn(_p_o.shape, device=dev, generator=_gg_o) * 0.5 + (1.0 if _p_o is _gr_o.ln.weight else 0.0))
    _hd_o = gop_view_o(_gr_o, _h0_o, _ho_o)
    _hs_o = _gr_o(torch.cat([_h0_o[:, None], _ho_o], 1).flatten(0, 1), torch.arange(len(_h0_o), device=dev).repeat_interleave(1 + SO_O),
                  torch.arange(1 + SO_O, device=dev).repeat(len(_h0_o)))[::1 + SO_O]
    _dg_o = float((_hd_o - _hs_o).abs().max() / _hs_o.abs().max().clamp_min(1e-12))
    _hd2_o = gop_view_o(_gr_o, _h0_o, _ho_o)
    _to_o = torch.randn(2, 3, D_CXR, device=dev, generator=torch.Generator(device=dev).manual_seed(5))
    _oo_o = o_phong_518(_xo_o, LUOI_O, MEAN, STD)
    _tq_o = []
    for _k_o, (_x0_o, _y0_o) in enumerate(O_1022):
        _x5_o, _y5_o, _c5_o = _x0_o * S_518 / S_1022, _y0_o * S_518 / S_1022, CANH_O * S_518 / S_1022
        _va_o = F.interpolate(_oo_o[:, _k_o:_k_o + 1].float(), size=(64, 64), mode='area').flatten(1)
        _vb_o = F.interpolate(_xo_o[:, :1, int(round(_y5_o)):int(round(_y5_o + _c5_o)), int(round(_x5_o)):int(round(_x5_o + _c5_o))].float(), size=(64, 64),
                            mode='area').flatten(1)
        _va_o, _vb_o = _va_o - _va_o.mean(1, keepdim=True), _vb_o - _vb_o.mean(1, keepdim=True)
        _tq_o.append((_va_o * _vb_o).sum(1) / (_va_o.norm(dim=1) * _vb_o.norm(dim=1)).clamp_min(1e-6))
    _tq_o = torch.cat(_tq_o)
_dz_o = max(float((_pa_o - _pb_o).abs().max()), float((_ca_o - _cb_o).abs().max()), float((_raa_o['r'] - _rab_o['r']).abs().max()))
assert _dz_o <= 1e-5, f'o cat: phep chieu skip khoi tao 0 VAN doi duong anh toan canh ({_dz_o:.2e}) -> DUNG'
assert all(torch.equal(a, b) for a, b in zip(_s1_o, _s2_o)), 'o cat: duong o TIEU dong ngau nhien chung -> DUNG'
assert torch.equal(BO_O7(_to_o), _to_o) and torch.equal(BO_O11(_to_o), _to_o), 'o cat: adapter khoi tao KHAC dong nhat -> DUNG'
_d6_o = float((_l6_o - _la_o).abs().max())
assert _d6_o <= 1e-4 and float((_lo_o - _la_o).abs().max()) > 1e-4, (f'o cat: 6 view giong nhau KHAC dau_v2 ({_d6_o:.2e}) hoac o that KHONG doi logit '
                                                                 '-> DUNG')
assert float(_tq_o.median()) >= 0.98 and float(_tq_o.min()) >= 0.9, (f'o cat: truong nhin o LECH vung anh (tuong quan trung vi {float(_tq_o.median()):.4f}, '
                                                                  f'min {float(_tq_o.min()):.4f}) -> DUNG')
assert torch.equal(_lo_o, _lo2_o) and torch.equal(_hd_o, _hd2_o), (f'o cat: duong du doan co o KHONG tat dinh (hai lan lech '
                                                                 f'{float((_lo_o - _lo2_o).abs().max()):.2e}) -> cong duong cham C6 se dung -> DUNG')
assert _dg_o <= 1e-5, f'o cat: gop 6 view dang day KHAC GopStudy.forward (lech tuong doi {_dg_o:.2e}) -> DUNG'
assert all(float(p.detach().abs().max()) == 0 for p in list(DUA_O.chieu.parameters()) + [GOP_O.e_view, GOP_O.w, GOP_O.b]) \
    and all(float(m.ra.weight.detach().abs().max()) == 0 for m in (BO_O7, BO_O11)), 'o cat: phep chieu / bo gop / adapter KHONG khoi tao 0 -> DUNG'
print(f"cong o cat: skip luc dau -> duong anh toan canh lech {_dz_o:.1e} | dong ngau nhien khong doi | adapter = dong nhat | 6 view giong nhau = "
      f"dau_v2 (lech {_d6_o:.1e}), o that doi logit (max {float((_lo_o - _la_o).abs().max()):.2e}) | truong nhin o: tuong quan trung vi "
      f"{float(_tq_o.median()):.4f} (min {float(_tq_o.min()):.4f}) | du doan tat dinh (hai lan = tung bit); gop 6 view dang day = GopStudy "
      f"(lech tuong doi {_dg_o:.1e}) | {sum(p.numel() for m in (DUA_O, BO_O7, BO_O11, GOP_O) for p in m.parameters()):,} tham so moi")
del _bo_o, _xo_o, _ro_o, _pa_o, _ca_o, _raa_o, _pb_o, _cb_o, _rab_o, _la_o, _lo_o, _l6_o, _to_o, _oo_o, _tq_o, _va_o, _vb_o, _s1_o, _s2_o
del _pb2_o, _rab2_o, _lo2_o, _kv_o, _h0_o, _ho_o, _gr_o, _gg_o, _p_o, _hd_o, _hs_o, _hd2_o
_KHOA_C4 = (THU_NHANH, RUN_TAG, LUOT_78U, CKPT_THAP)  # C5 / C6 kiem
