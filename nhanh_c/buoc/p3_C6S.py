# ==== nhanh_c buoc p3_C6S — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C6S · GOP TRONG SO dau doc ep1-10 -> MOT bo (mo hinh don muc 2) — du doan lai val, so voi TB du doan — KHONG train ====
# Tuan 18/10 (Event 2026-10-18 "Tuan quyet: mo hinh don ..."; docs/SPEC_SAU_78Y.md B.2h). Phan dinh nghia + cong cua C6 cua
# 84_nhanh_c_s<seed> NGUYEN VAN (toi truy_hoi_cau, nhu Task 2), roi phan moi (cuoi cell): nap w_ ep1-10 + p_ (val) ep1-10 cua
# luot goc (stage4_nb84/r84z_C_s<seed>), cong tai lap (ep10 du doan lai = p_ da luu), gop trong so bo A (ep1-10) / bo B (nua sau), du doan
# lai val, luat dat truoc tren fus_attn_hr; ghi w_gopA / p_gop* / pte_gopA (KHONG cham). KHONG doc nhan test.
crit = LOSS_MOI().to(dev)      # <- DIEM DUY NHAT khac 64 ve khoa hoc
MIDS_T, MCNT_T = torch.from_numpy(MIDS), torch.from_numpy(MCNT)
FEAT_T = torch.from_numpy(FEAT)
# Lay tham so 2-D dau tien, KHONG phai `next(iter(...))`: tham so dau cua mot `Dinov2Layer` la
# `norm1.weight` (1-D), ma nhom ndim<=1 von da co weight_decay=0 -> do o do thi MU voi chinh thu
# ta muon theo doi. Weight decay chi tac dong len ma tran 2-D.
# Van KHONG bam vao ten thuoc tinh cua HF (`.attention.attention.query`) — gia dinh do da lam hong `39`.
_first2d = lambda m: next(p for p in m.parameters() if p.ndim > 1)
_REF = {i: _first2d(LAYERS[i]).detach().clone() for i in FT_BLOCKS}

@torch.no_grad()
def shared_lo(img):
    '''78z skip7: block 0-7 cua anh toan canh + skip block 7 cua 5 o (token_o7; module o DONG CUNG, C2b) -> h8 (co CLS).'''
    with torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        t7 = token_o7(backbone, img, LUOI_O, MEAN, STD, KHUC_O, K_SKIP_O - 1)
        inj = skip_o(t7, BO_O7, DUA_O, _G_O.expand(len(img), -1, -1, -1).contiguous(), _W_O.expand(len(img), -1, -1).contiguous())
        del t7
        return cong_skip(run_blocks(backbone.embeddings(img), 0, GRAD_FROM), inj)

@torch.no_grad()
def upper_frozen(h, rows=None):
    '''Block 8-11 GOC. Tinh MOT lan cho ca batch — moi arm dong cung dung chung. 78y J (LUONG): block 8-11 + luong vung dong cung
       (neo theo view cua `rows`, write-back) -> patch; token vung r gan vao tensor ra (`._r_vung`) cho dau `mldec_hr`.'''
    with torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        if LUONG is None:
            return backbone.layernorm(run_blocks(h, GRAD_FROM, NL))[:, 1:].float()
        assert rows is not None and rows.shape[0] == h.shape[0], '78y J: upper_frozen can rows cua lo (neo theo view) -> DUNG'
        LUONG.dat_neo(NEO_T[rows].float())
        p, r, _ = upper_luong_hr(backbone, LUONG, h, mo_ghi=True)
    p._r_vung = r
    p._rows = rows   # 78z J: hang ORDER cua lo -> dau fusion biet study / view (ST_FUS_T / VIEW_FUS_T)
    return p

def upper_ft(h, t):
    '''Block 8-11 BAN SAO cua arm `t`, CO graph.'''
    with torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        for k in range(len(FT_BLOCKS)): h = _out(UPF[t][k](h))
        return UPF[t][-1](h)[:, 1:].float()

def arm_forward(t, pf, ppf, hl, hlp, erow, has, day_du=False):
    '''`pf`/`ppf` = dac trung DONG CUNG dung chung. Arm `ft` bo qua chung, tu tinh tu `hl`.
       78d: arm `hai` (hoc_chung) — MAC DINH tra duong CHI ANH (duong CHAM: khong doc EHR, khong phim truoc);
       `day_du=True`: duong day du (chi dung luc train).'''
    cfg = ARMS[t]
    if cfg.get('hai') and not day_du: return MOD[t](pf, None)[0]
    et = ETK[t](FEAT_T[erow].to(dev), MIDS_T[erow].to(dev), MCNT_T[erow].to(dev)) \
         if cfg['ehr'] else None
    if cfg['ft']: p, pp = upper_ft(hl, t), (upper_ft(hlp, t) if cfg['img'] else None)   # 78m: khong phim truoc
    else:         p, pp = pf, ppf
    if cfg['img']: p = IMG[t](p, pp, has)
    if cfg.get('fus'): return logit_fus(MOD[t], p)[0]   # 78z J: logit STUDY (fusion tren mldec_hr) phat lai cho tung anh
    if cfg.get('hr'):   # 78y: dau J doc [patch ; token vung]; dropout trong fork_rng (C2b) -> dong chung GIU nhu luot B1_e6
        with _FORK_78Y(): return mldec_h_hr(MOD[t], q_cua(MOD[t]), p, p._r_vung)[0]
    return MOD[t](p, et)[0]

def arm_forward_phu(t, pf, hl):
    '''78g: MOT forward decoder -> (logit, h, kv) cho MOI arm (khong arm mo khoa). Logit TRUNG arm_forward tung bit (cong C6).'''
    if ARMS[t].get('fus'): return logit_fus(MOD[t], pf)   # 78z J: (logit study, h, kv) nhu mldec_h_hr
    if ARMS[t].get('hr'):   # 78y: mldec_hr -> (logit, h, kv) nhu mldec_h; dropout trong fork_rng (C2b)
        with _FORK_78Y(): return mldec_h_hr(MOD[t], q_cua(MOD[t]), pf, pf._r_vung)
    return mldec_h(MOD[t], pf)


def logit_fus(m, pf):
    '''78z J: dau fusion tren mldec_hr -> (logit STUDY phat lai cho tung anh, h, kv). h = 40 dau ra query cua mldec_h_hr (doc
       [patch ; token vung]) cua RIENG dau; GopStudy gom theo study (ST_FUS_T) + view luc cham (VIEW_FUS_T) cua hang lo (pf._rows);
       bo view (pf._giu) CHI khi dau o che do train. Dropout cua decoder trong fork_rng nhu mldec_hr (78y). Gop CHI trong lo, CHI
       theo study_id.'''
    with _FORK_78Y(): _, h, kv = mldec_h_hr(m, q_cua(m), pf, pf._r_vung)
    r = pf._rows
    hS = m.fus(h, ST_FUS_T[r], VIEW_FUS_T[r], getattr(pf, '_giu', None) if m.training else None)
    return m.out(hS), h, kv


def _loader_eval(idx, tf, bs, perm=None):
    '''(nhu 78m) DataLoader cua predict / chan doan giu rieng — NW_EVAL worker, KHONG persistent; eval tat dinh.'''
    return DataLoader(DS(idx, tf, perm), batch_sampler=LoGomStudy(ST_ID[np.asarray(idx)], bs), num_workers=NW_EVAL, pin_memory=True,
                      persistent_workers=False, prefetch_factor=2 if NW_EVAL > 0 else None)   # 78z: lo gom TRON study (fusion)


@torch.no_grad()
def predict(idx, perm=None, bs=32, arms=None):
    '''78d: `arms` = chi tinh cac arm nay (mac dinh: tat ca). Khong arm nao can phim truoc (vd CHI arm cham)
       thi KHONG chay backbone tren phim truoc.'''
    arms = list(ARMS) if arms is None else list(arms)
    can_phim = any(ARMS[t]['img'] and not ARMS[t].get('hai') for t in arms)
    MOD.eval(); ETK.eval(); IMG.eval(); UPF.eval()
    P = {t: np.zeros((len(idx), C), 'float32') for t in arms}
    for b in tqdm(_loader_eval(idx, eval_tf, bs, perm),   # 78m: worker KHONG persistent (ra soat 02/10)
                  desc='eval', leave=False):
        hl = shared_lo(b['image'].to(dev, non_blocking=True)); hs = b['has'].to(dev)
        hlp = shared_lo(b['image_prev'].to(dev, non_blocking=True)) if can_phim else hl
        pf = upper_frozen(hl, b['row'].to(dev)) if globals().get('CAN_PF', True) else None   # 78m: luot B (moi arm mo khoa) khong can
        ppf = upper_frozen(hlp, b['row'].to(dev)) if can_phim else pf
        for t in arms:
            P[t][b['pos'].numpy()] = torch.sigmoid(
                arm_forward(t, pf, ppf, hl, hlp, b['erow'], hs).float()).cpu().numpy()
    assert not any(np.isnan(v).any() for v in P.values()), 'predict cho ra NaN'
    return P

def loader_train(idx, tf, bs):
    '''78t (D-T3, quyet dinh 07/10): DataLoader train = loader(idx, tf, True, bs) cua C2 TRU worker KHONG persistent (nhu 78u). Moi
       epoch iterator MOI rut hat worker + thu tu lo tu dong ngau nhien chung; dong chung luu trong rs_ (lay_rng, SAU du doan / chan
       doan cuoi epoch) va khoi phuc TRUOC vong epoch -> resume sau khi kernel chet CUNG thu tu lo va tang cuong voi chay lien.
       Worker persistent: phien moi rut them hat worker -> lech (kiem 07/10 tren DataLoader that, 2 worker).'''
    return DataLoader(DS(idx, tf, None), batch_size=bs, sampler=LoTheoStudy(ST_ID[np.asarray(idx)]), num_workers=NW, pin_memory=True,
                      drop_last=True, persistent_workers=False, prefetch_factor=PF_78M if NW > 0 else None)   # 78z: study lien nhau


LOG = {}                                   # nap lai tu Drive (API) khi resume — xem duoi
DL = loader_train(I_TF, train_tf, BATCH)   # 78t (D-T3): worker KHONG persistent -> resume CUNG thu tu lo + tang cuong
_IVA = I_VA[:256] if THU_NHANH else I_VA   # THU_NHANH: vai phut, ghi vao RUN_TAG ..._thu
_ITE = I_TE[:256] if THU_NHANH else I_TE
_EP_CHAY = 1 if THU_NHANH else EPOCHS

# ---- 78t: kiem truoc vong seed + chan doan + LAMBDA (chay that, truoc moi buoc train) ----
assert 'ST_T' in globals() and 'mldec_h' in globals() and 'infonce_cau' in globals() and 'DauDich' in globals(), '78t: chay C3c truoc'
assert 'build_phu' in globals() and set(PHU) == {t for t, c in PHU_CFG.items() if c['dich'] or c['cau']}, '78t: chay C4 cua 78t truoc C6'
assert _KHOA_C3C == (THU_NHANH, RUN_TAG) and _KHOA_C4 == (THU_NHANH, RUN_TAG, BATCH, ACCUM, ARMS_GIU), (
    f'C3c / C4 chay voi cau hinh C0 KHAC ({_KHOA_C3C}, {_KHOA_C4}) -> chay lai C0 -> C1 -> C2 -> C3 -> C3b -> C3c -> C4 -> C5 -> C6')
assert not CAN_PHIM and not any(c['ehr'] or c['img'] or c['ft'] for c in ARMS.values()) and ACCUM == 1, '78t: arm chi anh, dong cung'
CAN_PF = True
_fork = lambda: torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev == 'cuda' else [])
_sts = lambda: [torch.get_rng_state()] + ([torch.cuda.get_rng_state()] if dev == 'cuda' else [])


def tham_so_chung(t):
    '''Tham so ma CA loss nhan lan loss phu cua arm `t` cung cham (do ti le gradient): decoder TRU GroupFC.'''
    return [p for n, p in MOD[t].named_parameters() if p.requires_grad and not n.startswith('out.')]


def buoc_cau(rows, gen):
    '''Ung vien cau CHUNG moi arm cau trong MOT buoc (khoi_cau): st = ST_T[rows] (-1 = khong vao loss cau); KHONG hang nao st >= 0 ->
       None (KHONG goi khoi: luat (i) spec 78t). ma_u = lay_cau_lo(st, ..., TOI_DA_CAU, gen) - gen = GEN_PHU, CHI rut ung vien (luat
       (iii): dropout thap hoc cung rut tu GEN_THAP rieng -> dong ung vien TRUNG giua luot co / khong thap); P / loai / w.'''
    st = ST_T[rows]
    if not bool((st >= 0).any()): return None
    ma_u = lay_cau_lo(st, PTR_T, IDX_T, TOI_DA_CAU, gen)
    P = ma_tran_duong(st, PTR_T, IDX_T, ma_u)
    return dict(ma_u=ma_u, P=P, loai=mat_na_loai(P, SUB_T[rows], st), w=W_CAU_T[ma_u], hl=st >= 0)


def z_cau_cua(t, cau, gen_thap):
    '''Vector cau (thap cua arm cau t) cua ung vien cau['ma_u']: bert -> ThapVector(Z_CAU_T, z-score); clip -> ThapBert(TOK_T)
       duoi bf16 autocast, dropout (train) CHI tu gen_thap = GEN_THAP (luat (iii) spec 78t; eval: khong can, None duoc).'''
    m = PHU[t]['cau']
    if PHU_CFG[t]['cau'] == 'bert': return m.text(Z_CAU_T[cau['ma_u']].float())
    with torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        return m.text(TOK_T[cau['ma_u']], gen=gen_thap)


def phu_78t(t, h, kv, b_hop, b_co, rows, y, cau, gen_thap):
    '''Loss phu cua arm `t` tren MOT micro-batch (CHI luc train) -> dict: vung + khoi (khoi attention trong vung) | dich | cau +
       dung1 (tensor khong grad); None neu khong phan nao co cap hop le. rows = hang ORDER (dev); cau = buoc_cau (CHUNG moi arm cau
       trong buoc, rut tu GEN_PHU; None = khong neo -> KHONG goi infonce_cau); gen_thap = GEN_THAP (CHI dropout thap hoc cung).
       Anh giu rieng / ngoai POOL KHONG vao (HANG2POOL = -1, HANG2DICH -> hang -1, ST = -1).'''
    c = PHU_CFG[t]; o = {}
    if c['vung']:
        p = HANG2POOL_T[rows]; g = SG_HANG_T[rows]; hl = (g >= 0) & (p >= 0)
        if bool(hl.any()):
            Mn = mat_na_nhan(M_KDP_T[g[hl]], o_luoi_tu_hop(b_hop[hl], b_co[hl], G_LUOI, O_LUOI), R_VUNG)
            q = q_cua(MOD[t]).unsqueeze(0).expand(int(hl.sum()), -1, -1)
            lv, _, kh = loss_vung(attn_truoc_dropout(MOD[t].img.at, q, kv[hl]), Mn, y[hl])
            if lv is not None: o['vung'], o['khoi'] = lv, kh
    if c['dich']:
        dau = PHU[t]['dich']; hd = HANG2DICH_T[rows]
        ld = loss_dich(dau(h), dau.chon_dich({k: DICH_T[k][hd] for k in dau.nhom}))
        if ld is not None: o['dich'] = ld
    if c['cau'] and cau is not None:
        m = PHU[t]['cau']
        o['cau'], o['dung1'] = infonce_cau(m.z_anh(h.mean(1)), z_cau_cua(t, cau, gen_thap), cau['P'], cau['loai'], cau['w'],
                                           m.tau(), cau['hl'])
    return o or None


def _khoa_dong_cung():
    '''Khoa state_dict cua PHU thuoc thap BiomedBERT KHONG hoc (embeddings + tang duoi) = ban goc: KHONG luu vao w_ (Drive).'''
    return {f'{t}.cau.text.mdl.{n}' for t in PHU if PHU_CFG[t]['cau'] == 'clip'
            for n, p in PHU[t]['cau'].text.mdl.named_parameters() if not p.requires_grad}


def sd_phu():
    '''state_dict cua PHU de luu (w_): moi module phu, thap hoc cung CHI phan MO (spec 78t: Drive co gioi han).'''
    bo = _khoa_dong_cung()
    return {k: v for k, v in PHU.state_dict().items() if k not in bo}


def nap_phu(sd):
    '''Nap sd_phu(): phan dong cung cua thap giu ban goc (_BERT_GOC); thieu / thua khoa khac -> DUNG.'''
    kq = PHU.load_state_dict(sd, strict=False)
    assert not kq.unexpected_keys and set(kq.missing_keys) == _khoa_dong_cung(), ('w_: module phu lech cau hinh', kq)


def lay_rng():
    '''Trang thai MOI dong ngau nhien cua vong train (luu trong rs_): dong chung CPU + CUDA (thu tu lo, dropout dau doc) + GEN_PHU
       (ung vien cau) + GEN_THAP (dropout thap hoc cung).'''
    return dict(cpu=torch.get_rng_state(), cuda=(torch.cuda.get_rng_state_all() if dev == 'cuda' else None),
                gen_phu=GEN_PHU.get_state(), gen_thap=GEN_THAP.get_state(), gen_fus=GEN_FUS.get_state())   # 78z: + mat na bo view


def dat_rng(r):
    '''Khoi phuc lay_rng() (resume: dong chung + GEN_PHU + GEN_THAP nhu luc ghi rs_). torch.load map_location dua tensor sang dev ->
       ve CPU truoc khi dat. (DataLoader that, persistent worker: hat worker rut lai luc tao iterator moi -> thu tu lo sau resume
       KHONG chac trung tung bit tren GPU; mo phong (test_78t) trung tung bit.)'''
    torch.set_rng_state(r['cpu'].cpu())
    if dev == 'cuda': torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])
    GEN_PHU.set_state(r['gen_phu'].cpu()); GEN_THAP.set_state(r['gen_thap'].cpu())
    GEN_FUS.set_state(r['gen_fus'].cpu())                                    # 78z: mat na bo view (resume trung chay lien)


# Cong DUONG CHAM (batch val THAT): GO HET (None) moi bang loss phu (vung, dich, cau, vector / token cau, E van ban, dau ra 78e / 78f /
# 78q / 78s con giu, nhom POOL / giu rieng) + module PHU + BiomedBERT goc -> DataLoader cham tra anh TRUNG va du doan MOI arm TRUNG
# TUNG BIT (doc bat cu bang nao -> loi -> DUNG).
_TEN_PHU = ('HOP518', 'SG_HANG', 'SG_HANG_T', 'M_KDP', 'M_KDP_T', 'HANG2POOL', 'HANG2POOL_T', 'HANG2DICH', 'HANG2DICH_T', 'DICH_T',
            'DICH_G', 'PTR', 'IDX', 'ST', 'PTR_T', 'IDX_T', 'ST_T', 'PTR_G', 'IDX_G', 'ST_G', 'PTR_G_T', 'IDX_G_T', 'ST_G_T', 'MA_G_T',
            'SUB_T', 'Z_CAU_T', 'TOK_T', 'W_CAU_T', 'E_ARM', 'E_MOTA_T', 'PHU', '_BERT_GOC',
            'CAU_T', 'TK_CAU', 'DICH_IDX', 'DICH_TEN', 'META_78S', 'E_MOTA', 'E_5MT', 'E_CAU_LOP', 'E_NM', 'NM_META', 'VG', 'ANH_F', 'VT',
            'IDX3', 'IDX4', 'PHOI_T', 'POOL', 'I_GIU', 'GIU_RIENG',
            'FEAT', 'FEAT_T', 'MIDS', 'MIDS_T', 'MCNT', 'MCNT_T')   # 78z (SPEC_SAU_78Y B.0): + bien giu cho EHR / tui vi khuan
_b0 = next(iter(loader(I_VA[:8], eval_tf, False, 8, nw=0)))
with torch.no_grad():
    MOD.eval(); PHU.eval()
    _hl0 = shared_lo(_b0['image'].to(dev)); _pf0 = upper_frozen(_hl0, _b0['row'].to(dev)); _hs0 = _b0['has'].to(dev)
    _r1 = {t: arm_forward(t, _pf0, _pf0, _hl0, _hl0, _b0['erow'], _hs0) for t in ARMS}
    assert all(torch.equal(_r1[t], arm_forward_phu(t, _pf0, _hl0)[0]) for t in ARMS), 'arm_forward_phu lech arm_forward'
    _G0 = {k: globals()[k] for k in _TEN_PHU}
    try:
        globals().update(dict.fromkeys(_TEN_PHU))
        _b2 = next(iter(loader(I_VA[:8], eval_tf, False, 8, nw=0)))
        _pf2 = upper_frozen(_hl0, _b2['row'].to(dev))   # 78y: block 8-11 (+ J: luong vung + neo) TINH LAI khi GO bang
        _r2 = {t: arm_forward(t, _pf2, _pf2, _hl0, _hl0, _b2['erow'], _hs0) for t in ARMS}
    except Exception as _e:
        raise AssertionError(f'duong CHAM doc bang loss phu / PHU ({type(_e).__name__}: {_e}) -> DUNG')
    finally:
        globals().update(_G0)
assert torch.equal(_b0['image'], _b2['image']) and torch.equal(_pf2, _pf0) \
    and (LUONG is None or torch.equal(_pf2._r_vung, _pf0._r_vung)) and torch.equal(_pf2._rows, _pf0._rows) \
    and all(torch.equal(_r1[t], _r2[t]) for t in ARMS), \
    'duong CHAM doc bang loss phu (vung / dich / cau) -> DUNG'
# Cong DONG NGAU NHIEN: ung vien cau + thap hoc cung (dropout) + loss phu (train) KHONG cham dong chung; HAI generator rieng (luat
# (iii)): ung vien CHI tu GEN_PHU, dropout thap CHI tu GEN_THAP (thap KHONG tieu GEN_PHU -> dong ung vien trung giua cac luot).
_i1 = I_TF[(SG_HANG[I_TF] >= 0) & (HANG2POOL[I_TF] >= 0) & (ST[I_TF] >= 0)][:32]   # anh thang POOL co cau: du moi phan loss phu
assert len(_i1), 'khong anh thang POOL nao co cau 78s -> khong kiem duoc loss phu (78s / 78f lech phien ban?) -> DUNG'
_b1 = next(iter(loader(_i1, train_tf, False, len(_i1), nw=0)))                 # duong train: anh + hop
_gq = torch.Generator(device=dev); _gq.manual_seed(1); _gs = _gq.get_state()     # vai tro GEN_PHU
_gt = torch.Generator(device=dev); _gt.manual_seed(2); _gts = _gt.get_state()    # vai tro GEN_THAP
with torch.no_grad():
    PHU.train()
    _rw1 = _b1['row'].to(dev); _hl1 = shared_lo(_b1['image'].to(dev)); _pf1 = upper_frozen(_hl1, _rw1)
    _F1 = {t: arm_forward_phu(t, _pf1, _hl1) for t in PHU_CFG}
    _s0 = _sts()
    _c1 = buoc_cau(_rw1, _gq) if CO_CAU else None
    _gs1 = _gq.get_state()
    _oq = {t: phu_78t(t, _F1[t][1], _F1[t][2], _b1['hop'].to(dev), _b1['co_hop'].to(dev), _rw1, _b1['y'].to(dev), _c1, _gt)
           for t in PHU_CFG}
assert all(torch.equal(a, b) for a, b in zip(_sts(), _s0)), 'loss phu / ung vien cau / thap hoc cung TIEU dong ngau nhien CHUNG'
_thieu_p = [t for t, o in _oq.items() if o is None or any((k in o) != bool(PHU_CFG[t][k]) for k in PHAN)]
_khong_hh = [t for t, o in _oq.items() if o is not None
             and not all(bool(torch.isfinite(torch.as_tensor(o[k]))) for k in PHAN if k in o)]
assert not _thieu_p and not _khong_hh, (f'lo kiem {len(_i1)} anh thang train: arm THIEU phan loss phu {_thieu_p} / loss phu '
                                        f'khong huu han {_khong_hh} -> DUNG')
assert not CO_CAU or not torch.equal(_gs1, _gs), 'ung vien cau KHONG rut tu generator rieng (GEN_PHU)'
assert torch.equal(_gq.get_state(), _gs1), 'loss phu / thap hoc cung rut tu GEN_PHU -> dong ung vien LECH giua luot co / khong thap'
assert torch.equal(_gt.get_state(), _gts) != CO_CLIP, 'dropout thap hoc cung KHONG CHI rut tu GEN_THAP -> DUNG'
# Cong RNG cua THAP hoc cung (luat (ii) spec 78t, chay TRONG fork): CUNG trang thai generator rieng (vai tro GEN_THAP) + HAI trang
# thai dong chung KHAC -> ra TRUNG TUNG BIT; generator khac -> ra khac; dong chung (CPU + CUDA) KHONG doi sau moi lan goi.
_TC = [t for t in PHU_CFG if PHU_CFG[t]['cau'] == 'clip']
if _TC:
    with _fork(), torch.no_grad():
        PHU.train(); _cu = dict(ma_u=torch.arange(min(64, N_CAU), device=dev)); _gth = torch.Generator(device=dev)
        _gth.manual_seed(11); _gsh = _gth.get_state()
        torch.manual_seed(101); _ra = _sts(); _z1 = z_cau_cua(_TC[0], _cu, _gth); _okr = all(torch.equal(a, b) for a, b in zip(_sts(), _ra))
        _gth.set_state(_gsh); torch.manual_seed(202); _ra = _sts(); _z2 = z_cau_cua(_TC[0], _cu, _gth)
        _okr &= all(torch.equal(a, b) for a, b in zip(_sts(), _ra))
        _gth.manual_seed(12); _z3 = z_cau_cua(_TC[0], _cu, _gth)
    assert _okr and torch.equal(_z1, _z2) and not torch.equal(_z1, _z3), (
        'thap BiomedBERT hoc cung: dropout KHONG CHI tu generator rieng (dong chung khac -> ra khac / dong chung bi tieu) -> DUNG')
    del _z1, _z2, _z3, _cu
del _b0, _b2, _G0, _hl0, _pf0, _r1, _r2, _b1, _rw1, _hl1, _pf1, _F1, _c1, _oq, _gq, _gt, _thieu_p, _khong_hh
print('cong 78t: duong cham KHONG doc bang loss phu (go het bang + PHU -> trung tung bit) | arm_forward_phu = arm_forward | '
      'ung vien cau (GEN_PHU) + dropout thap (GEN_THAP): hai dong RIENG, dong chung khong doi | moi arm co du phan loss phu'
      + (' | thap hoc cung: cung generator -> trung tung bit, dong chung khong doi' if _TC else ''))
# ---- 78z J cong fusion o duong cham (lo val THAT): (1) LoGomStudy tren _IVA / _ITE: moi vi tri DUNG mot lan, KHONG study nao bi cat qua hai
#      lo; (2) DOC LAP giua study: logit fusion cua mot study KHONG doi khi doi cac study khac trong lo (gop CHI trong study); (3) eval BO QUA
#      mat na bo view (pf._giu); (4) duong fusion KHONG doc bang loss phu (cong tren) va CHI doc ST_FUS_T / VIEW_FUS_T.
#      17/10: (2) va (3) so theo NGUONG TOL_FUS (lech logit tuyet doi), KHONG tung bit: GopStudy gop bang index_add_ -> tren GPU cong atomic,
#      thu tu cong doi giua hai lan goi -> lech bit cuoi (~1e-6). Loi that (tron study / eval dung mat na "bo het anh") lech >> 1e-2.
TOL_FUS = 1e-4
for _ten_g, _ix_g in (('val', _IVA), ('test', _ITE)):
    _lo_g = list(iter(LoGomStudy(ST_ID[_ix_g], LO_DU_DOAN)))
    _so_lo = {}
    for _k_g, _l_g in enumerate(_lo_g):
        for _s_g in set(ST_ID[_ix_g][_l_g].tolist()): _so_lo.setdefault(_s_g, set()).add(_k_g)
    assert sorted(v for l in _lo_g for v in l) == list(range(len(_ix_g))) and all(len(v) == 1 for v in _so_lo.values()), (
        f'cong fusion (1): lo du doan {_ten_g} cat study / thieu vi tri -> DUNG')
_b5 = next(iter(_loader_eval(_IVA[:96], eval_tf, 96)))
with torch.no_grad():
    MOD.eval()
    _hl5 = shared_lo(_b5['image'].to(dev)); _rw5 = _b5['row'].to(dev); _pf5 = upper_frozen(_hl5, _rw5)
    _st5 = ST_FUS_T[_rw5]; _u5 = torch.unique(_st5)
    assert len(_u5) >= 3, 'lo kiem fusion can >= 3 study -> DUNG'


    def _con5(ix):
        p = _pf5[ix]; p._rows = _rw5[ix]; p._r_vung = _pf5._r_vung[ix]
        return p


    _ia5 = torch.where((_st5 == _u5[0]) | (_st5 == _u5[1]))[0]; _ib5 = torch.where((_st5 == _u5[0]) | (_st5 == _u5[2]))[0]
    _na5, _nb5 = int((_st5[_ia5] == _u5[0]).sum()), int((_st5[_ib5] == _u5[0]).sum())
    _lech5, _lech3 = {}, {}
    for t in _FUS:
        with _FORK_78Y(): _h5 = mldec_h_hr(MOD[t], q_cua(MOD[t]), _pf5, _pf5._r_vung)[1]   # 78z J: query cua mldec_hr
        _a5 = MOD[t].out(MOD[t].fus(_h5[_ia5], _st5[_ia5], VIEW_FUS_T[_rw5[_ia5]]))[_st5[_ia5] == _u5[0]]
        _c5 = MOD[t].out(MOD[t].fus(_h5[_ib5], _st5[_ib5], VIEW_FUS_T[_rw5[_ib5]]))[_st5[_ib5] == _u5[0]]
        _lech5[t] = float((_a5 - _c5).abs().max())
        _pg5 = _con5(torch.arange(len(_rw5), device=dev)); _pg5._giu = torch.zeros(len(_rw5), dtype=torch.bool, device=dev)
        _lech3[t] = float((logit_fus(MOD[t], _pg5)[0] - logit_fus(MOD[t], _con5(torch.arange(len(_rw5), device=dev)))[0]).abs().nan_to_num(
            float('inf')).max())
    MOD.train()
assert _na5 == _nb5 and all(v <= TOL_FUS for v in _lech5.values()), (
    f'cong fusion (2): logit study PHU THUOC study khac trong lo {_lech5} (nguong {TOL_FUS:g}) -> DUNG')
assert all(v <= TOL_FUS for v in _lech3.values()), (
    f'cong fusion (3): o eval VAN dung mat na bo view (lech logit {_lech3}, nguong {TOL_FUS:g}) -> DUNG')
print(f"cong fusion: lo du doan gom TRON study (val {len(_IVA):,} / test {len(_ITE):,} anh) | doc lap giua study (lech max "
      f"{max(_lech5.values()):.1e}) | eval bo qua mat na bo view (lech max {max(_lech3.values()):.1e}) | nguong {TOL_FUS:g} (atomic GPU)")
del _b5, _hl5, _rw5, _pf5, _st5, _u5, _ia5, _ib5, _h5, _a5, _c5, _pg5, _lo_g, _so_lo, _lech3
# ======================================================================== C6S phan moi (gop trong so, Tuan 18/10) ===================
assert not THU_NHANH and not CAN_PHIM and all(not (c['ehr'] or c['img'] or c['ft']) for c in ARMS.values()), 'C6S: chi dau dong cung chi anh, khong THU_NHANH -> DUNG'
assert tuple(SEEDS) == (SEED,) and 'SUB' in globals() and 'YM' in globals() and 'ST_ID' in globals(), 'C6S: can seed cua luot, SUB, YM, ST_ID -> DUNG'
DAU_S, TOL_S, NG_TOAN_S, NG_NUA_S, HAT_NUA_S = ('mldec_hr', 'fus_attn_hr', 'fus_tb_hr'), 0.002, 0.002, 0.003, 7801   # 78z skip7: + fus_tb_hr (bao kem)
DAU_QD_S = 'fus_attn_hr'                                     # 78z skip7: dau QUYET DINH (nhu B.4 v2); mldec_hr / fus_tb_hr in kem
assert set(DAU_S) <= set(ARMS) and DAU_QD_S in DAU_S, f'dau {DAU_S} khong co trong ARMS -> DUNG'
_DU_S = (CAU_HINH_C['tm'], CAU_HINH_C['run_z'])   # nhanh C: Phan 2 cua CHINH seed nay
# ---- (1) nguon: w_ ep1-10 + p_ (val) ep1-10 cua luot goc (CHI doc) ----
EP_S = list(range(1, EPOCHS + 1))
assert len(EP_S) >= 3, EP_S
EP_S_A, EP_S_B = EP_S, EP_S[len(EP_S) // 2:]                 # bo A = moi epoch (chinh); bo B = nua sau (so phu)
_ik_s, _kk_s = kiem_ke(_DU_S, mau=rf'(w_s{SEED}_ep\d+\.pt|p_[a-z0-9_]+_s{SEED}_ep\d+\.npy)')
_thieu_s = [t for t in [f'w_s{SEED}_ep{e}.pt' for e in EP_S] + [f'p_{d}_s{SEED}_ep{e}.npy' for d in DAU_S for e in EP_S] if not _kk_s.get(t)]
assert not _thieu_s, f"thieu {len(_thieu_s)} tep trong {'/'.join(_DU_S)} (vd {_thieu_s[:3]}) -> DUNG"
TEP_S = {}


def _tep_s(ten):
    v = _kk_s[ten]
    if xung_dot(v): print(f"  ! {ten}: {len(v)} ban NOI DUNG KHAC -> dung ban MOI NHAT ({v[0]['mt']})")
    TEP_S[ten] = dict(id=v[0]['id'], md5=v[0]['md5'], so_ban=len(v))
    return tai(v[0])


# ---- (2) val: nhan, gop view (TB du doan cac anh CUNG study), hai nua theo BENH NHAN (= C1 cua bang 78a, hat 7801) ----
YV_S = YM[_IVA]
_us_v, _inv_v = np.unique(ST_ID[_IVA], return_inverse=True)


def _gop_v(Q):
    S = np.zeros((len(_us_v), Q.shape[1]), 'float64'); np.add.at(S, _inv_v, Q)
    return (S / np.bincount(_inv_v, minlength=len(_us_v))[:, None])[_inv_v].astype('float32')


_su_s = np.unique(SUB[_IVA])
_A_s = np.random.default_rng(HAT_NUA_S).choice(_su_s, size=len(_su_s) // 2, replace=False)
NUA_S = np.where(np.isin(SUB[_IVA], _A_s), 0, 1)
assert not (set(SUB[_IVA][NUA_S == 0].tolist()) & set(SUB[_IVA][NUA_S == 1].tolist())), 'benh nhan o CA HAI nua -> DUNG'


def _so3(Q):
    '''mAP macro (gop view da ap): toan val, nua A, nua B, tier.'''
    a = ap_cols(YV_S, Q)
    return dict(toan=float(np.nanmean(a)), nua_A=float(np.nanmean(ap_cols(YV_S[NUA_S == 0], Q[NUA_S == 0]))),
                nua_B=float(np.nanmean(ap_cols(YV_S[NUA_S == 1], Q[NUA_S == 1]))), tier={t: float(np.nanmean(a[i])) for t, i in TIER.items()})


# ---- (3) p_ da luu: tung epoch + TB DU DOAN ep1-10 (= thanh vien cua bang 78a) ----
P_EP = {d: [np.load(_tep_s(f'p_{d}_s{SEED}_ep{e}.npy')).astype('float32') for e in EP_S] for d in DAU_S}
for d in DAU_S:
    for e, p in zip(EP_S, P_EP[d]):
        assert p.shape == (len(_IVA), C) and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1, (d, e, p.shape)
_KTB, _KEP = f'TB du doan ep{EP_S[0]}-{EP_S[-1]}', f'ep{EP_S[-1]}'
_KA, _KB = f'gop trong so ep{EP_S_A[0]}-{EP_S_A[-1]}', f'gop trong so ep{EP_S_B[0]}-{EP_S_B[-1]}'
KQ_S = {}
for d in DAU_S:
    for e, p in zip(EP_S, P_EP[d]): KQ_S[(d, f'ep{e}')] = _so3(_gop_v(p))
    KQ_S[(d, _KTB)] = _so3(_gop_v(np.mean(P_EP[d], 0)))
# ---- (4) cong tai lap: nap w_ ep cuoi -> du doan lai val = p_ da luu (lech <= TOL_S) ----
def _nap_w(e):
    w = torch.load(_tep_s(f'w_s{SEED}_ep{e}.pt'), map_location='cpu', weights_only=False)
    assert w.get('run') == CAU_HINH_C['run_z'] and list(w.get('arms') or []) == list(ARMS) and int(w['ep']) == e - 1, (
        f'w_s{SEED}_ep{e}.pt: run / arm / epoch KHAC luot goc -> DUNG')
    assert not w['etk'] and not w['img'] and not w['upf'], 'dau chi anh: etk / img / upf trong w_ phai rong -> DUNG'
    return w


def _du_doan(sd_mod, idx):
    MOD.load_state_dict(sd_mod)
    return predict(idx, arms=list(DAU_S))


_t0_s = time.time()
_pv_cuoi = _du_doan(_nap_w(EP_S[-1])['mod'], _IVA)
LECH_S = {d: float(np.abs(_pv_cuoi[d] - P_EP[d][-1]).max()) for d in DAU_S}
print(f"cong tai lap ({_KEP}, val {len(_IVA):,} anh): du doan lai - p_ da luu: lech max " + ', '.join(f'{d} {v:.1e}' for d, v in LECH_S.items())
      + f' (<= {TOL_S}) | {(time.time() - _t0_s) / 60:.1f} phut', flush=True)
assert max(LECH_S.values()) <= TOL_S, 'cong tai lap: du doan lai KHAC p_ da luu -> nap dau doc / duong du doan KHAC luot goc. DUNG, KHONG gop.'
del _pv_cuoi
# ---- (5) gop trong so: TB tung tham so FLOAT cua MOD qua cac epoch (float64 -> dtype goc); buffer khong float phai GIONG NHAU ----
def _gop_w(eps):
    ws = [_nap_w(e)['mod'] for e in eps]
    ks = list(ws[0])
    assert all(list(w) == ks for w in ws), 'state_dict cua MOD KHAC khoa giua cac epoch -> DUNG'
    out, n_float = {}, 0
    for k in ks:
        ts = [w[k] for w in ws]
        if torch.is_floating_point(ts[0]):
            out[k] = torch.stack([t.to(torch.float64) for t in ts], 0).mean(0).to(ts[0].dtype); n_float += out[k].numel()
        else:
            assert all(torch.equal(t, ts[0]) for t in ts), f'{k}: buffer khong float KHAC nhau giua cac epoch -> DUNG'
            out[k] = ts[0].clone()
    return out, int(n_float)


SOUP_A, N_FLOAT_S = _gop_w(EP_S_A)
SOUP_B, _ = _gop_w(EP_S_B)
_pvA = _du_doan(SOUP_A, _IVA)
_pvB = _du_doan(SOUP_B, _IVA)
for d in DAU_S:
    KQ_S[(d, _KA)] = _so3(_gop_v(_pvA[d])); KQ_S[(d, _KB)] = _so3(_gop_v(_pvB[d]))
print(f'gop trong so: {N_FLOAT_S:,} tham so float cua MOD (TB tung tham so) | bo A {EP_S_A[0]}-{EP_S_A[-1]} | bo B {EP_S_B[0]}-{EP_S_B[-1]} | '
      f'{(time.time() - _t0_s) / 60:.1f} phut', flush=True)
# ---- (6) LUAT (dat TRUOC, Event 2026-10-18) tren DAU_QD_S, gop view: A - TB >= -NG_TOAN toan val VA >= -NG_NUA moi nua VA A > ep cuoi -> MUC 2 ----
def _xet_s(ka):
    a, tb, ep = KQ_S[(DAU_QD_S, ka)], KQ_S[(DAU_QD_S, _KTB)], KQ_S[(DAU_QD_S, _KEP)]
    d = dict(toan=a['toan'] - tb['toan'], nua_A=a['nua_A'] - tb['nua_A'], nua_B=a['nua_B'] - tb['nua_B'], so_ep_cuoi=a['toan'] - ep['toan'])
    d['dat'] = bool(d['toan'] >= -NG_TOAN_S and d['nua_A'] >= -NG_NUA_S and d['nua_B'] >= -NG_NUA_S and d['so_ep_cuoi'] > 0)
    return d


XET_S = {_KA: _xet_s(_KA), _KB: _xet_s(_KB)}
PHAN_XU_S = (f'MUC 2: mo hinh don = {_KA} cua {DAU_QD_S}' if XET_S[_KA]['dat'] else
             (f'MUC 1: giu TB du doan ep{EP_S[0]}-{EP_S[-1]}; so phu {_KB} DAT -> Tuan quyet' if XET_S[_KB]['dat'] else
              f'MUC 1: giu TB du doan ep{EP_S[0]}-{EP_S[-1]}'))
print(f"\n=== GOP TRONG SO (val {len(_IVA):,} anh, gop view; hai nua theo benh nhan, hat {HAT_NUA_S}) ===")
print(f"{'dau':<13}{'ban':<27}{'toan':>8}{'nua A':>8}{'nua B':>8}{'head':>8}{'medium':>8}{'tail':>8}")
for d in DAU_S:
    for k in [f'ep{e}' for e in EP_S] + [_KTB, _KA, _KB]:
        r = KQ_S[(d, k)]
        print(f"{d:<13}{k:<27}{r['toan']:>8.4f}{r['nua_A']:>8.4f}{r['nua_B']:>8.4f}" + ''.join(f"{r['tier'][t]:>8.4f}" for t in ('head', 'medium', 'tail')))
print(f"\nLUAT (dat truoc) tren {DAU_QD_S}: gop - TB du doan >= -{NG_TOAN_S} toan val VA >= -{NG_NUA_S} moi nua VA gop > {_KEP} -> MUC 2")
for k, x in XET_S.items():
    print(f"  {k:<27} - TB: toan {x['toan']:+.4f} | nua A {x['nua_A']:+.4f} | nua B {x['nua_B']:+.4f} | - {_KEP} {x['so_ep_cuoi']:+.4f} -> "
          + ('DAT' if x['dat'] else 'KHONG') + ('   <- luat' if k == _KA else '   (so phu)'))
print(f'=> {PHAN_XU_S}')
# ---- (7) ghi (local; len Drive o C9): p_ (val) hai bo, w_ bo A, pte_ bo A (test: CHI ghi du doan, KHONG doc nhan test), json ----
for d in DAU_S:
    ghi(f'p_gopA_{d}_s{SEED}.npy', lambda q, v=_pvA[d]: np.save(q, v))
    ghi(f'p_gopB_{d}_s{SEED}.npy', lambda q, v=_pvB[d]: np.save(q, v))
ghi(f'w_gopA_s{SEED}.pt', lambda q: torch.save(dict(run=RUN_TAG, nguon='/'.join(_DU_S), eps=EP_S_A, arms=list(ARMS), mod=SOUP_A, etk={}, img={},
                                                upf={}, cach='TB tung tham so float cua MOD qua cac epoch (float64 -> dtype goc)'), q))
_pteA = _du_doan(SOUP_A, _ITE)
for d in DAU_S: ghi(f'pte_gopA_{d}_s{SEED}.npy', lambda q, v=_pteA[d]: np.save(q, v))
_ts_s = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
TEN_JSON_S = f'gop_trong_so_{_ts_s}.json'
ghi(TEN_JSON_S, lambda q: ghi_json(q, dict(
    t=_ts_s, dau=list(DAU_S), dau_quyet_dinh=DAU_QD_S, epoch=EP_S, bo_A=EP_S_A, bo_B=EP_S_B, n_float=N_FLOAT_S, so_anh_val=int(len(_IVA)),
    hat_nua=HAT_NUA_S, ket_qua={f'{d} | {k}': v for (d, k), v in KQ_S.items()}, xet=XET_S, phan_xu=PHAN_XU_S,
    luat=(f'tren {DAU_QD_S}, gop view: gop trong so ep{EP_S_A[0]}-{EP_S_A[-1]} - TB du doan >= -{NG_TOAN_S} toan val VA >= -{NG_NUA_S} moi nua '
          f'VA > {_KEP} -> MUC 2; khong -> MUC 1; chi bo B dat -> Tuan quyet. Dat TRUOC khi co so (Event.md 2026-10-18).'),
    cong_tai_lap=dict(lech=LECH_S, tol=TOL_S), nguon=TEP_S, tep_ghi=[f'p_gopA_{d}_s{SEED}.npy' for d in DAU_S] + [f'p_gopB_{d}_s{SEED}.npy' for d in DAU_S]
    + [f'w_gopA_s{SEED}.pt'] + [f'pte_gopA_{d}_s{SEED}.npy' for d in DAU_S],
    ghi_chu='Tuan 18/10: mo hinh don muc 2 = gop trong so dau doc; KHONG train; KHONG doc nhan test (pte_gopA chi ghi, chua cham).')))
print(f'\nda ghi {TEN_JSON_S} + {2 * len(DAU_S)} tep p_gop* + w_gopA_s{SEED}.pt + {len(DAU_S)} tep pte_gopA (KHONG cham) -> {DRV_DUONG[0]}/{RUN_TAG} '
      f'(len Drive o C9). KHONG doc nhan test. Ghi Event.md: {PHAN_XU_S}')
