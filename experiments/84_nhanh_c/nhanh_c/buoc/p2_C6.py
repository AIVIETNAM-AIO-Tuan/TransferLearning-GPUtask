# ==== nhanh_c buoc p2_C6 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
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


@torch.no_grad()
def truy_hoi_cau(t, f):
    '''Truy hoi anh -> cau tren nhom GIU RIENG (arm cau t, eval): ung vien = moi cau KHAC NHAU cua cac study giu rieng (MA_G_T); duong
       = cau cua CHINH study (CSR giu rieng; cau cung benh nhan khac study KHONG loai - chi chan doan). f [n_giu, d] = TB h tung anh
       (thu tu I_GIU). -> cau_top1 (ti le anh co cau xep dau la duong), cau_hit10 (>= 1 duong trong 10 dau) + muc ngau nhien.'''
    m = PHU[t]['cau']; co = ST_G_T >= 0; n, U = int(co.sum()), int(MA_G_T.numel())
    if n == 0 or U < 10: return {}
    za = F.normalize(m.z_anh(f[co]), dim=-1)
    zc = torch.cat([F.normalize(z_cau_cua(t, dict(ma_u=MA_G_T[i:i + 2048]), None).float(), dim=-1) for i in range(0, U, 2048)])
    st = ST_G_T[co]; t1, t10, c1, c10 = [], [], [], []
    for i in range(0, n, 512):
        P = ma_tran_duong(st[i:i + 512], PTR_G_T, IDX_G_T, MA_G_T)
        top = (za[i:i + 512] @ zc.T).topk(10, dim=1).indices
        t1.append(P.gather(1, top[:, :1])[:, 0].float()); t10.append(P.gather(1, top).any(1).float())
        npos = P.sum(1).double(); kq = torch.ones_like(npos)
        for k in range(10): kq = kq * ((U - npos - k).clamp_min(0) / (U - k))
        c1.append(npos / U); c10.append(1 - kq)
    return dict(cau_top1=float(torch.cat(t1).mean()), cau_hit10=float(torch.cat(t10).mean()),
                cau_top1_nn=float(torch.cat(c1).mean()), cau_hit10_nn=float(torch.cat(c10).mean()), cau_n_anh=n, cau_n_ung=U)


@torch.no_grad()
def chan_doan_giu():
    '''78t: chan doan tren nhom GIU RIENG (KHONG vao loss phu), mo hinh o eval:
       - MOI arm: khoi attention lop -> patch trong vung goi y KHONG du phong (TB tren cap (anh thang, nhan duong co vung); txtq_k4:
         TB 4 nguyen mau cua lop); tach rieng 5 nhan trong phoi (_phoi).
       - arm dich: AUC tung bit (T1 / T2 TB tren lop co du hai lop; T5 / T6 / T7 tung bit) + do chinh xac T3 / T4 (chan_doan_dich).
       - arm cau: truy hoi anh -> cau trong kho cau cua cac study giu rieng (truy_hoi_cau).
       - query khong co dinh: cos(q hien tai, q luc dau) TB / nho nhat theo hang; ||Delta|| / ||co so|| (txtq_du); dem nguyen mau
         thang tren cap (anh, lop) duong [40, 4] (txtq_k4).'''
    MOD.eval(); PHU.eval()
    KH = {t: {'kdp': [], 'kdp_phoi': []} for t in ARMS}; H = {t: [] for t in PHU_CFG if PHU_CFG[t]['cau']}
    OD = {t: {k: [] for k in PHU[t]['dich'].nhom} for t in PHU_CFG if PHU_CFG[t]['dich']}
    TH = {t: torch.zeros(C, K_NM, dtype=torch.long, device=dev) for t in ARMS if ARMS[t].get('q') == 'k4'}; pos = []
    for b in _loader_eval(I_GIU, eval_tf, 64):
        hl = shared_lo(b['image'].to(dev, non_blocking=True)); pf = upper_frozen(hl, b['row'].to(dev))
        g = SG_HANG_T[b['row'].to(dev)]; co = g >= 0; y = b['y'].to(dev)
        if bool(co.any()):
            hp = torch.from_numpy(HOP518[g[co].cpu().numpy()]).to(dev)
            Mk = mat_na_nhan(M_KDP_T[g[co]], o_luoi_tu_hop(hp, torch.isfinite(hp).all(-1), G_LUOI, O_LUOI), R_VUNG)
        for t in ARMS:
            _, h, kv = mldec_h(MOD[t], pf)
            if bool(co.any()) and not ARMS[t].get('hr'):   # 78y: mldec_hr co khoa token vung -> attention patch-only SAI nghia, KHONG do
                q = q_cua(MOD[t])
                A = gop_query_k(attn_truoc_dropout(MOD[t].img.at, q.unsqueeze(0).expand(int(co.sum()), -1, -1), kv[co]), q.shape[0] // C)
                kh = (A * Mk.float()).sum(-1); hl_ = (y[co] > 0.5) & Mk.any(-1)
                KH[t]['kdp'].append(kh[hl_]); KH[t]['kdp_phoi'].append(kh[hl_ & PHOI_T])
            if t in OD:
                for k, v in PHU[t]['dich'](h).items(): OD[t][k].append(v.float().cpu())
            if t in H: H[t].append(h.mean(1).float())
            if t in TH: TH[t] += dem_thang(MOD[t].out.logit_k(h), y)
        pos.append(b['pos'])
    assert torch.equal(torch.cat(pos), torch.arange(len(I_GIU))), 'thu tu anh giu rieng lech'
    _tb = lambda v: float(torch.cat(v).mean()) if v and sum(x.numel() for x in v) else None
    vung = {t: {f'khoi_{k_}': _tb(v) for k_, v in KH[t].items()} for t in ARMS}
    dich = {}
    for t, od in OD.items():
        dau = PHU[t]['dich']
        tg = {k: v.cpu().numpy() for k, v in dau.chon_dich({k: DICH_G[k] for k in dau.nhom}).items()}
        dich[t] = chan_doan_dich({k: torch.cat(v).numpy() for k, v in od.items()}, tg)
    cau = {t: truy_hoi_cau(t, torch.cat(v)) for t, v in H.items()}
    txt = {}
    for t in ARMS:
        if not ARMS[t].get('q'): continue
        cq = cos_query(q_cua(MOD[t]), Q_INIT[t])
        r = dict(cos_q=float(cq.mean()), cos_q_min=float(cq.min()))
        if ARMS[t]['q'] == 'du': r['ti_le_delta'] = MOD[t].qtxt.ti_le_delta()
        if t in TH: r['thang'] = TH[t].tolist()
        txt[t] = r
    return dict(vung=vung, dich=dich, cau=cau, txt=txt)


TEP_LAM = 'he_so_lambda.json'          # MOT tep cho luot (moi seed CUNG thu muc dung CUNG lambda da chot)


def hieu_chinh_lambda():
    '''(nhu 78g, them dich / cau) lambda[arm][phan] = r x TB ||grad L_nhan|| / TB ||grad L_phan|| tren tham_so_chung, do tren BAN
       SAO mo hinh (dong ngau nhien RIENG, KHONG luu): HC_BUOC buoc chi loss nhan tren arm doi chung, chep phan chung decoder sang moi
       arm co loss phu (ho txtq: qtxt giu khoi tao - load_state_dict(strict=False) CHI thieu qtxt.*), roi do tren HC_DO micro-batch
       (ung vien cau + dropout thap: generator rieng). Buoc / micro-batch co gradient khong huu han bi BO. Dong chung / MOD / PHU
       KHONG doi.'''
    global MOD, ETK, IMG, UPF, PHU
    _goc = (MOD, ETK, IMG, UPF, PHU); t0 = time.time()
    try:
        with _fork():
            torch.manual_seed(SEEDS[0] + 500)
            MOD, ETK, IMG, UPF = build_arms(SEEDS[0]); PHU = build_phu(SEEDS[0])
            a0 = DOI_CHUNG
            pm = [p for p in MOD[a0].parameters() if p.requires_grad]
            op = torch.optim.AdamW([{'params': [p for p in pm if p.ndim > 1], 'weight_decay': WD, 'lr': LR},
                                    {'params': [p for p in pm if p.ndim <= 1], 'weight_decay': 0.0, 'lr': LR}])
            st_ = math.ceil(len(I_TF) / (BATCH * ACCUM)) * EPOCHS; wm_ = int(WARMUP * st_)
            sc = torch.optim.lr_scheduler.LambdaLR(op, lambda s: s / max(wm_, 1) if s < wm_ else
                                                   0.5 * (1 + math.cos(math.pi * (s - wm_) / max(st_ - wm_, 1))))
            dl_ = loader(I_TF, train_tf, True, BATCH); it = iter(dl_)
            MOD.train(); PHU.train()
            for _s in range(HC_BUOC):
                b = next(it); hl = shared_lo(b['image'].to(dev, non_blocking=True))
                crit(arm_forward_phu(a0, upper_frozen(hl, b['row'].to(dev)), hl)[0].float(), b['y'].to(dev)).backward()
                if not torch.isfinite(torch.nn.utils.clip_grad_norm_(pm, CLIP, error_if_nonfinite=False)):
                    op.zero_grad(set_to_none=True); continue          # nhu C6: gradient khong huu han -> bo buoc
                op.step(); sc.step(); op.zero_grad(set_to_none=True)
            _sd = MOD[a0].state_dict()
            for t in PHU_CFG:                                         # moi arm co loss phu: CUNG phan chung (qtxt giu khoi tao)
                _kq = MOD[t].load_state_dict(_sd, strict=False)
                assert not _kq.unexpected_keys and all(k.startswith('qtxt.') for k in _kq.missing_keys), (t, _kq)
            gen = torch.Generator(device=dev); gen.manual_seed(SEEDS[0] + 501)       # ung vien cau (vai tro GEN_PHU)
            gen_t = torch.Generator(device=dev); gen_t.manual_seed(SEEDS[0] + 502)   # dropout thap hoc cung (vai tro GEN_THAP)
            G = {t: dict(nhan=[], **{k: [] for k in PHAN}) for t in PHU_CFG}
            for _s in range(HC_DO if PHU_CFG else 0):
                b = next(it); hl = shared_lo(b['image'].to(dev, non_blocking=True)); pf = upper_frozen(hl, b['row'].to(dev))
                y = b['y'].to(dev); rw = b['row'].to(dev); bh = b['hop'].to(dev); bc = b['co_hop'].to(dev)
                cau = buoc_cau(rw, gen) if CO_CAU else None
                for t in PHU_CFG:
                    lg, h, kv = arm_forward_phu(t, pf, hl)
                    o = phu_78t(t, h, kv, bh, bc, rw, y, cau, gen_t)
                    if o is None: continue
                    sh = tham_so_chung(t)
                    _g = dict(nhan=chuan_grad(crit(lg.float(), y), sh))
                    for k in PHAN:
                        if k in o: _g[k] = chuan_grad(o[k], sh)
                    if all(np.isfinite(v) for v in _g.values()):
                        for k, v in _g.items(): G[t][k].append(v)
                    del lg, h, kv, o
            del it, dl_, op, sc
    finally:
        MOD, ETK, IMG, UPF, PHU = _goc
    if dev == 'cuda': torch.cuda.empty_cache()
    _thieu = [(t, k) for t, c in PHU_CFG.items() for k in PHAN if c[k] and not G[t][k]]
    assert not _thieu, f'hieu chinh: khong micro-batch nao co loss phu {_thieu}'
    tk = lambda v: dict(tb=float(np.mean(v)), trung_vi=float(np.median(v)), p10=float(np.percentile(v, 10)),
                        p90=float(np.percentile(v, 90)), n=len(v))
    lam = {t: {k: float(c['r_' + k] * np.mean(G[t]['nhan']) / np.mean(G[t][k])) for k in PHAN if c[k]}
           for t, c in PHU_CFG.items()}
    return dict(CAU_HINH_LAM, seed_hieu_chinh=SEEDS[0], hc_buoc=HC_BUOC, hc_do=HC_DO,
                g={t: {k: tk(v) for k, v in G[t].items() if v} for t in G}, lam=lam,
                giay=round(time.time() - t0, 1), gpu=(torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'),
                t=datetime.datetime.now().isoformat(timespec='seconds'))


# Cau hinh ma lambda thuoc ve: sai bat cu muc nao -> DUNG (tep lambda cua cau hinh khac). Luot rut gon / tach (LAM_TU): lambda LAY TU
# luot s42 du 17 arm - phai PHU moi arm co loss phu cua luot nay (cung cau hinh arm) va cung cong thuc (_KHOA_LAM_TU).
CAU_HINH_LAM = json.loads(json.dumps(dict(run=RUN_TAG, arms=list(ARMS), phu_cfg=PHU_CFG, ti_le_tang=TI_LE_TANG, batch=BATCH,
                                          accum=ACCUM, toi_da_cau=TOI_DA_CAU, thap_mo=THAP_MO, lr_thap=LR_THAP, tau0_cau=TAU0_CAU,
                                          dich=DICH_TEN['phien_ban'], muc_tieu=MUC_TIEU['phien_ban'], dich_tat=DICH_TAT,
                                          dich_bit_tat=DICH_BIT_TAT)))
_KHOA_LAM_TU = ('batch', 'accum', 'toi_da_cau', 'thap_mo', 'lr_thap', 'tau0_cau', 'dich', 'muc_tieu', 'dich_tat', 'dich_bit_tat')
assert LAM_TU is None or tuple(LAM_TU) != tuple(DRV_DUONG), 'LAM_TU tro vao CHINH thu muc luot nay -> de LAM_TU = None'
# ---- 78z J (ghi mot lan, Tuan 17/10; khoi_ghi_cuoi): MOT tep resume. Co -> nap TRUOC lambda / vong seed: tra tep ket qua dang cho
#      (he_so_lambda.json, p_ / pte_ / nhat ky cua cac epoch da xong) ve hang cho + trang thai train (vong seed nap). Khac luot / arm /
#      backbone -> DUNG (doi RUN_TAG, KHONG xoa gi). KHONG con cap w_ / rs_ de nap: w_ chi de luu, rs_ khong ghi nua. ----
assert globals().get('GHI_CUOI') is True and 'luu_resume' in globals() and 'khoi_phuc_cho' in globals(), '78z J: can khoi_ghi_cuoi o C0 -> DUNG'
_RES, _lp_res = None, tim_moi_noi(TEN_RESUME)
if _lp_res:
    _RES = torch.load(_lp_res, map_location='cpu', weights_only=False)
    assert _RES.get('run') == RUN_TAG and list(_RES.get('arms') or []) == list(ARMS) and (
        (_RES.get('nguon_bb') or {}).get('md5') == NGUON_BB['md5']), (
        f"{TEN_RESUME}: luot / arm / backbone KHAC ({_RES.get('run')}, {_RES.get('arms')}, md5 {(_RES.get('nguon_bb') or {}).get('md5')} != "
        f"{NGUON_BB['md5']}) -> DUNG (doi RUN_TAG, KHONG xoa gi)")
    khoi_phuc_cho(_RES['cho'])
    print(f"78z J: RESUME tu {TEN_RESUME}: seed {_RES['seed']}, xong epoch {int(_RES['ep']) + 1}")
_lp_lam = tim_moi_noi(TEP_LAM)
if _lp_lam:
    with open(_lp_lam) as _fl: LAM_J = json.load(_fl)
    tai_len_lai(TEP_LAM)                         # tep CHI con o local (hang cho phien truoc mat) -> tai len lai
    print(f"lambda: DOC LAI {TEP_LAM} ({LAM_J['t']}) - KHONG tinh lai")
else:
    if LAM_TU is not None:
        _ik, _kk = kiem_ke(tuple(LAM_TU), mau=re.escape(TEP_LAM))
        _dl = _kk.get(TEP_LAM) or []
        assert _dl, f'thieu {TEP_LAM} trong cxrlt_2024/{"/".join(LAM_TU)} -> chay C6 cua luot s42 (hieu chinh lambda) truoc'
        if xung_dot(_dl): print(f'  !! {TEP_LAM} ({"/".join(LAM_TU)}): {len(_dl)} ban KHAC noi dung -> dung ban moi nhat ({_dl[0]["mt"]})')
        with open(tai(_dl[0])) as _fl: _LJ = json.load(_fl)
        _hc_lj = (_LJ.get('hc_buoc'), _LJ.get('hc_do'))
        assert THU_NHANH or not (str(_LJ.get('run', '')).endswith('_thu') or (None not in _hc_lj and _hc_lj != (HC_BUOC, HC_DO))), (
            f"{TEP_LAM} cua {'/'.join(LAM_TU)} la lambda cua luot THU NHANH (run {_LJ.get('run')}, hieu chinh {_hc_lj[0]} + "
            f"{_hc_lj[1]} buoc) -> DUNG: chay C6 luot s42 du (THU_NHANH = False) truoc")
        _ko = [t for t in PHU_CFG if t not in (_LJ.get('lam') or {}) or (_LJ.get('phu_cfg') or {}).get(t) != CAU_HINH_LAM['phu_cfg'][t]]
        assert not _ko, f'{TEP_LAM} cua {"/".join(LAM_TU)} KHONG phu arm {_ko} (thieu / cau hinh arm khac) -> DUNG'
        _lech = {k: (_LJ.get(k), CAU_HINH_LAM[k]) for k in _KHOA_LAM_TU if _LJ.get(k) != CAU_HINH_LAM[k]}
        assert not _lech, f'{TEP_LAM} cua {"/".join(LAM_TU)} thuoc cong thuc KHAC: {_lech} -> DUNG'
        LAM_J = dict(CAU_HINH_LAM, g=({t: _LJ['g'][t] for t in PHU_CFG} if _LJ.get('g') else None),
                     lam={t: _LJ['lam'][t] for t in PHU_CFG}, hc_buoc=_LJ.get('hc_buoc'), hc_do=_LJ.get('hc_do'),
                     nguon=dict(duong=list(LAM_TU), id=_dl[0]['id'], md5=_dl[0]['md5'], run=_LJ.get('run'), t=_LJ.get('t')),
                     t=datetime.datetime.now().isoformat(timespec='seconds'))
        print(f"lambda: LAY TU {'/'.join(LAM_TU)}/{TEP_LAM} (luot {_LJ.get('run')}, {_LJ.get('t')}) - KHONG hieu chinh lai")
        del _ik, _kk, _LJ
    else:
        _da = sorted(x for x in ten_tren_drive() if re.fullmatch(r'(w|rs)_s\d+_ep\d+\.pt', x))
        if _da:
            # tep lambda MAT nhung DA co checkpoint -> KHOI PHUC lambda tu w_ moi nhat (w_ luu `lam`), KHONG tinh lai giua luot.
            _wl = [x for x in _da if x.startswith('w_')]
            assert _wl, f'Drive co {_da[:3]} nhung KHONG co w_ va THIEU {TEP_LAM} -> KHONG khoi phuc duoc lambda. DUNG.'
            _wn = max(_wl, key=lambda x: int(re.search(r'_ep(\d+)\.pt$', x).group(1)))
            try: _wd = torch.load(tim_moi_noi(_wn), map_location='cpu', weights_only=False)
            except Exception as _e:
                raise AssertionError(f'THIEU {TEP_LAM} va KHONG doc duoc {_wn} ({type(_e).__name__}) -> KHONG khoi phuc '
                                     'duoc lambda. DUNG.')
            assert isinstance(_wd, dict) and _wd.get('run') == RUN_TAG and list(_wd.get('arms', [])) == list(ARMS) \
                and isinstance(_wd.get('lam'), dict), f'{_wn}: khong co lambda cua luot {RUN_TAG} -> KHONG khoi phuc duoc. DUNG.'
            LAM_J = dict(CAU_HINH_LAM, g=None, khoi_phuc_tu=_wn, t=datetime.datetime.now().isoformat(timespec='seconds'),
                         lam={t: {k: float(v) for k, v in d.items() if PHU_CFG[t][k]} for t, d in _wd['lam'].items()})
            del _wd
            print(f'!! THIEU {TEP_LAM} tren Drive nhung co {_wn} -> KHOI PHUC lambda tu w_ (khong tinh lai) va ghi lai tep')
        else:
            print(f'hieu chinh lambda: {HC_BUOC} buoc chi loss nhan + {HC_DO} micro-batch do (ban sao mo hinh, khong luu) ...',
                  flush=True)
            LAM_J = hieu_chinh_lambda()
    _lv = [float(LAM_J['lam'][t].get(k, 0.0)) for t in PHU_CFG for k in PHAN]
    assert set(LAM_J['lam']) == set(PHU_CFG) and all(np.isfinite(v) and v >= 0 for v in _lv) and all(
        set(LAM_J['lam'][t]) == {k for k in PHAN if PHU_CFG[t][k]} and sum(LAM_J['lam'][t].values()) > 0 for t in PHU_CFG), (
        f"lambda HONG {LAM_J['lam']} -> KHONG ghi tep, DUNG (chay lai C6 de hieu chinh lai)")
    ghi(TEP_LAM, lambda q: ghi_json(q, LAM_J))
_lech = {k: (LAM_J.get(k), v) for k, v in CAU_HINH_LAM.items() if LAM_J.get(k) != v}
assert not _lech, f'{TEP_LAM} thuoc cau hinh KHAC: {_lech} -> DUNG (RUN_TAG / arm / h(t) / batch / cau / dich)'
assert (LAM_J.get('nguon') or {}).get('duong') == (list(LAM_TU) if LAM_TU is not None else None), (
    f"{TEP_LAM}: nguon lambda {(LAM_J.get('nguon') or {}).get('duong')} KHAC LAM_TU {LAM_TU} (C0) -> DUNG")
LAM = {t: {k: float(LAM_J['lam'][t].get(k, 0.0)) for k in PHAN} for t in PHU_CFG}
assert all(np.isfinite(v) and v >= 0 for d in LAM.values() for v in d.values()) and all(sum(d.values()) > 0
                                                                                      for d in LAM.values()), LAM
print(f"lambda = r x TB ||grad nhan|| / TB ||grad phu|| ({LAM_J.get('hc_buoc')} buoc chi nhan, {LAM_J.get('hc_do')} micro-batch "
      'do):' + ('' if LAM_J.get('g') else ' (khong arm nao co loss phu -> lambda rong)' if not PHU_CFG else f" KHOI PHUC tu {LAM_J.get('khoi_phuc_tu')}")
      + (f" | LAY TU {'/'.join(LAM_J['nguon']['duong'])}" if LAM_J.get('nguon') else ''))
for t, c in PHU_CFG.items():
    print(f'  {t:<12} ' + ' | '.join(
        (f"{k}: r {c['r_' + k]:g} x {LAM_J['g'][t]['nhan']['tb']:.3g} / {LAM_J['g'][t][k]['tb']:.3g} = {LAM[t][k]:.4g}"
         if LAM_J.get('g') else f'{k}: {LAM[t][k]:.4g}') for k in PHAN if c[k]))
if CHI_LAM: print(f'CHI_LAM: {TEP_LAM} cua luot du ({len(PHU_CFG)} arm co loss phu) da ghi -> KHONG train; notebook tach: ARMS_GIU + '
                  f'LUOT_78T + LAM_TU = {DRV_DUONG}')
for seed in (() if CHI_LAM else SEEDS):
    sk = f's{seed}'
    _co = set(ten_tren_drive()) | set(DA_GHI)   # hoi API + tep phien nay DA ghi (ke ca dang cho tai len; sua cua 78l)
    _thieu = [f'p_{t}_{sk}_ep{e}.npy' for t in ARMS for e in range(1, _EP_CHAY + 1)
              if f'p_{t}_{sk}_ep{e}.npy' not in _co]
    # 78k (ra soat 01/10): epoch CUOI chi tinh la xong khi w_ + epoch_log + pte_ (arm cham) cua no da len Drive. Mat phien
    # sau khi ghi p_ nhung truoc w_ -> KHONG bo qua: chay tiep tu cap rs_/w_ cuoi va LAM LAI epoch cuoi (ban trung ten
    # moi hon thang: tim_moi_noi / kiem_ke lay ban MOI NHAT). Khong doi rs_: sau epoch cuoi khong ai doc no.
    _thieu += [x for x in [f'w_{sk}_ep{_EP_CHAY}.pt', f'epoch_log_{sk}_ep{_EP_CHAY}.json']
               + ([f'pte_{t}_{sk}_ep{_EP_CHAY}.npy' for t in ARMS_CHAM] if DO_TEST else []) if x not in _co]
    if not _thieu: print(f'{sk}: DU {len(ARMS) * _EP_CHAY} file p_ tren Drive, bo qua'); continue
    MOD, ETK, IMG, UPF = build_arms(seed)
    PHU = build_phu(seed)                               # 78m: module loss phu (khoi tao KHONG tieu dong chung)
    SHARED = {t: tham_so_chung(t) for t in PHU_CFG}       # 78g: MOI arm co loss phu (ke ca arm chi vung)
    Q_INIT = {t: q_cua(MOD[t]).detach().clone() for t in ARMS if ARMS[t].get('q')}   # 78t: query LUC DAU (chan doan troi)
    _P_BERT = [p for t in PHU if PHU_CFG[t]['cau'] == 'clip' for p in PHU[t]['cau'].text.mdl.parameters() if p.requires_grad]
    _ID_BERT = {id(p) for t in PHU if PHU_CFG[t]['cau'] == 'clip' for p in PHU[t]['cau'].text.mdl.parameters()}
    PARAM_OF = {t: [p for m in ([MOD[t]] + [x[t] for x in (ETK, IMG, UPF) if t in x])
                    for p in m.parameters() if p.requires_grad] for t in ARMS}
    N_CLIP = {t: 0 for t in ARMS}; N_SKIP = {t: 0 for t in ARMS}; N_STEP = 0
    # Block mo bang dung LR/10 (chuan LP-FT): LR day du se xoa kien thuc pretrain trong vai buoc.
    _ftp = {id(p) for t in UPF for p in UPF[t].parameters()}
    rest = [p for m in (MOD, ETK, IMG) for p in m.parameters() if p.requires_grad]
    ftpp = [p for t in UPF for p in UPF[t].parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{'params': [p for p in rest if p.ndim > 1], 'weight_decay': WD, 'lr': LR},
         {'params': [p for p in rest if p.ndim <= 1], 'weight_decay': 0.0, 'lr': LR},
         # KHONG weight decay tren block pretrain. AdamW tach roi co tham so `w(1-lr*wd)` moi buoc;
         # voi LR/10 thi luc co (~0.4%/4k buoc) manh ngang gradient nhiem vu, nen `drift` do duoc
         # phan lon la BI BAO MON chu khong phai thich nghi — va do dung nhip `G_ft` di xuong.
         {'params': [p for p in ftpp if p.ndim > 1], 'weight_decay': 0.0, 'lr': LR/LR_FT_DIV},
         {'params': [p for p in ftpp if p.ndim <= 1], 'weight_decay': 0.0, 'lr': LR/LR_FT_DIV},
         # 78t: module loss phu (dau dich, chieu anh, thap cau, nhiet do) - NHOM o CUOI, LR nhu dau doc; tang MO cua thap
         # BiomedBERT (cau_clip): nhom RIENG LR_THAP 1e-5, WD tham so 2-D, CUNG warmup + cosine (LambdaLR chung)
         {'params': [p for p in PHU.parameters() if p.requires_grad and id(p) not in _ID_BERT and p.ndim > 1],
          'weight_decay': WD, 'lr': LR},
         {'params': [p for p in PHU.parameters() if p.requires_grad and id(p) not in _ID_BERT and p.ndim <= 1],
          'weight_decay': 0.0, 'lr': LR},
         {'params': [p for p in _P_BERT if p.ndim > 1], 'weight_decay': WD, 'lr': LR_THAP},
         {'params': [p for p in _P_BERT if p.ndim <= 1], 'weight_decay': 0.0, 'lr': LR_THAP}])
    steps = math.ceil(len(I_TF) / (BATCH*ACCUM)) * EPOCHS; warm = int(WARMUP*steps)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s/max(warm,1) if s < warm else
                                            0.5*(1+math.cos(math.pi*(s-warm)/max(steps-warm,1))))
    LOG.setdefault(sk, {}); ep0 = 0; _RNG_RS = None
    # RESUME CO optimizer: trong so (w_) + optimizer/lich (rs_) cua CUNG epoch, tim QUA API (moi ban trung
    # ten). 22/09: khoi dong lai AdamW voi moment RONG lam CNN mat 0.0318 macro va khong hoi phuc.
    for _e in range(_EP_CHAY, 0, -1):
        if not (_RES is not None and int(_RES['seed']) == seed and int(_RES['ep']) == _e - 1): continue   # 78z J: CHI tep resume (ghi mot lan)
        _w = _r = _RES                                       # 78z J: tep resume = w_ + rs_ cu (CUNG khoa)
        assert list(_w['arms']) == list(ARMS) and list(_r['arms']) == list(ARMS), (
            f"checkpoint co arm {list(_w['arms'])}, notebook co {list(ARMS)} -> DOI RUN_TAG o C0, KHONG xoa gi.")
        assert int(_w['ep']) == int(_r['ep']) == _e - 1 and int(_w['seed']) == int(_r['seed']) == seed, (
            f'w_/rs_ ep{_e} ghi ep/seed khac ten -> DUNG')
        MOD.load_state_dict(_w['mod']); ETK.load_state_dict(_w['etk'])
        IMG.load_state_dict(_w['img']); UPF.load_state_dict(_w['upf'])
        nap_phu(_w['phu'])                                  # 78t: module loss phu (thap: CHI phan mo; phan dong cung = goc)
        assert json.loads(json.dumps(_w.get('lam'))) == json.loads(json.dumps(LAM)), (
            f'w_ ep{_e} luu lambda KHAC tep lambda hien tai -> DUNG (tep lambda bi thay giua luot?)')
        opt.load_state_dict(_r['opt']); sch.load_state_dict(_r['sch'])
        N_CLIP, N_SKIP, N_STEP, ep0 = _r['n_clip'], _r['n_skip'], _r['n_step'], _e
        _RNG_RS = _r['rng']                                 # 78t: dong ngau nhien luc ghi rs_ (khoi phuc TRUOC vong epoch)
        for _kl in range(_e, 0, -1):    # 78m (sua cua 78l, ra soat 01/10): log epoch _e mat -> ban GAN NHAT con lai
            _lp = tim_moi_noi(f'epoch_log_{sk}_ep{_kl}.json')
            if _lp: break
        if _lp:
            with open(_lp) as _f: LOG[sk] = json.load(_f).get(sk, {})
            if _kl != _e: print(f'   !! thieu epoch_log_{sk}_ep{_e}.json -> nap ban ep{_kl}: nhat ky THIEU epoch '
                                f'{list(range(_kl + 1, _e + 1))} (p_ van du; C7 bao)')
        LOG[sk].setdefault('resume', []).append(ep0)
        print(f'{sk}: resume tu epoch {ep0+1}/{_EP_CHAY} — trong so + moment AdamW + lich ({N_STEP} buoc)')
        del _w, _r; break
    if ep0 == 0: print(f'{sk}: bat dau tu epoch 1/{_EP_CHAY} (khong co tep resume {TEN_RESUME})')
    # 78m (sua cua 78l, ra soat 01/10): tep cua epoch <= ep0 CHI con o local (hang cho phien truoc mat khi chay lai C0)
    # -> tai len lai. Tep epoch > ep0 se GHI LAI khi train nen KHONG dang ky (ghi() cam trung ten).
    for _e in range(1, ep0 + 1):
        for _t in ([f'p_{t}_{sk}_ep{_e}.npy' for t in ARMS]
                   + ([f'pte_{t}_{sk}_ep{_e}.npy' for t in ARMS_CHAM] if DO_TEST else [])
                   + [f'rs_{sk}_ep{_e}.pt', f'w_{sk}_ep{_e}.pt', f'epoch_log_{sk}_ep{_e}.json']):
            tai_len_lai(_t)
    GEN_PHU = torch.Generator(device=dev); GEN_PHU.manual_seed(seed * 1000 + 17)    # 78t: dong RIENG CHI ung vien cau
    GEN_THAP = torch.Generator(device=dev); GEN_THAP.manual_seed(seed * 1000 + 19)  # 78t: dong RIENG CHI dropout thap hoc cung
    GEN_FUS = torch.Generator(device=dev); GEN_FUS.manual_seed(seed * 1000 + int(CFG_FS['hat_gen_fus']))   # 78z: CHI mat na bo view
    if _RNG_RS is not None: dat_rng(_RNG_RS)            # 78t: resume - dong chung + GEN_PHU + GEN_THAP nhu luc ghi rs_ ep0

    for ep in range(ep0, _EP_CHAY):
        # `UPF` chay EVAL y het ban goc (`backbone` luon o .eval()). No VAN nhan gradient va VAN
        # cap nhat — `eval()` chi tat dropout/drop-path. Neu de `.train()` thi `tpb_ft` co them
        # regularization NGAU NHIEN ma `tpb` khong co, va `G_ft` khong con do rieng "trong so thich
        # nghi" nua. C2 in ra cau hinh dropout de biet rui ro nay co that hay bang 0.
        MOD.train(); ETK.train(); IMG.train(); UPF.eval()
        PHU.train()                                         # 78t: dau dich / cau; thap BiomedBERT: dropout tu GEN_THAP
        if dev == 'cuda': torch.cuda.reset_peak_memory_stats()   # 78m: VRAM dinh cua RIENG vong train
        t0, tot, nb = time.time(), {t: 0.0 for t in ARMS}, 0
        TOT_PHU = {t: dict(vung=0.0, dich=0.0, cau=0.0, khoi=0.0, n_vung=0, n_dich=0, n_cau=0) for t in PHU_CFG}
        D1 = {t: [torch.zeros((), device=dev), torch.zeros((), device=dev)] for t in PHU_CFG if PHU_CFG[t]['cau']}   # dung1
        SO_U = []                                           # 78t: so ung vien cau moi buoc (do U cua uoc VRAM thap)
        N_PHU = {t: 0 for t in PHU_CFG}
        PHU_BO, CAT_PHU, BO_CAT_PHU = ({t: 0 for t in PHU_CFG} for _ in range(3)); TL = {t: [] for t in PHU_CFG}
        opt.zero_grad(set_to_none=True)
        for j, b in enumerate(tqdm(DL, desc=f'{sk} ep{ep+1}/{EPOCHS}', leave=False)):
            if THU_NHANH and j >= 20 * ACCUM: break   # 78m (ra soat 02/10): 20 buoc de do toc do / VRAM (75a: 2)
            hl = shared_lo(b['image'].to(dev, non_blocking=True))
            hlp = hl                                            # 78m: KHONG phim truoc (CAN_PHIM = False)
            y = b['y'].to(dev); hs = b['has'].to(dev)
            pf = upper_frozen(hl, b['row'].to(dev)) if CAN_PF else None; ppf = pf  # MOT lan cho ca batch (luot B: khong can)
            if pf is not None: pf._giu = rut_giu_view(ST_FUS_T[pf._rows], GEN_FUS, P_BO_VIEW)   # 78z: bo view LUC TRAIN (GEN_FUS rieng)
            n_lui, bad = 0, []                                  # 78t (D-T1): so arm DA backward trong buoc nay
            _rw = b['row'].to(dev)                              # 78g: hang ORDER cua anh trong lo
            _bh, _bc = b['hop'].to(dev, non_blocking=True), b['co_hop'].to(dev, non_blocking=True)   # hop vung DA bien doi
            _hs = he_so_tang(N_STEP, steps, TI_LE_TANG)         # h(t) theo SO BUOC toi uu (dung ca sau resume)
            _cau = buoc_cau(_rw, GEN_PHU) if (_hs > 0 and CO_CAU) else None   # 78t: ung vien cau CHUNG moi arm cau (None = khong neo)
            if _cau is not None: SO_U.append(int(_cau['ma_u'].numel()))   # so ung vien cau / buoc (khong dong bo thiet bi)
            for t in ARMS:
                lg, _h, _kv = arm_forward_phu(t, pf, hl)          # MOT forward decoder: logit + h + kv cho loss phu
                if not torch.isfinite(lg).all():
                    bad.append(t); N_SKIP[t] += 1; del lg, _h, _kv; continue
                l_ = crit(lg.float(), y); tot[t] += float(l_.detach())
                if t in PHU_CFG and _hs > 0:                    # 78t: LOSS PHU (CHI luc train), lambda chot truoc vong
                    o = phu_78t(t, _h, _kv, _bh, _bc, _rw, y, _cau, GEN_THAP)
                    if o is not None:
                        lp = sum(LAM[t][k] * o[k] for k in PHAN if k in o)
                        if torch.isfinite(lp):
                            if j < N_DO_TL:     # chan doan: ||grad lambda*L_phu|| / ||grad L_nhan|| (KHONG nhan h(t))
                                TL[t].append(chuan_grad(lp, SHARED[t]) / max(chuan_grad(l_, SHARED[t]), 1e-12))
                            l_ = l_ + _hs * lp; N_PHU[t] += 1
                            for k in PHAN + ('khoi',):          # khoi: so Python; loss: tensor -> detach (khong canh bao)
                                if k in o: TOT_PHU[t][k] += float(o[k].detach() if torch.is_tensor(o[k]) else o[k])
                            for k in PHAN: TOT_PHU[t]['n_' + k] += int(k in o)
                            if 'dung1' in o:                     # dung1 nan khi khong neo -> CHI cong khi huu han, dem RIENG
                                _f1 = torch.isfinite(o['dung1'])
                                D1[t][0] += torch.where(_f1, o['dung1'], torch.zeros_like(o['dung1'])); D1[t][1] += _f1.float()
                        else:
                            PHU_BO[t] += 1                       # loss phu khong huu han -> BO phan phu, DEM
                # 78t (D-T1, quyet dinh 07/10): backward RIENG arm nay NGAY sau forward cua no (78g: MOT backward tren TONG moi arm)
                # -> do thi cua arm (kv, K / V, attention + dropout, thap BiomedBERT) giai phong TRUOC forward arm sau: VRAM dinh ~
                # MOT arm thay vi 17. Arm KHONG chung tham so hoc (backbone dong cung, pf khong grad) va thu tu forward (dong dropout)
                # giu nguyen -> gradient, tham so sau buoc, dong ngau nhien TRUNG TUNG BIT ban tong (test_78t muc 4).
                (l_ / ACCUM).backward(); n_lui += 1
                del lg, _h, _kv, l_; o = lp = None              # bo moi tham chieu toi do thi arm (ca nhanh phu bi BO) -> giai phong
            if bad and (sum(N_SKIP.values()) <= 3 or j % 400 == 0):
                print(f'   [{sk} ep{ep+1} buoc {j}] logit khong huu han: {bad}')
            if not n_lui: opt.zero_grad(set_to_none=True); continue   # 78t (D-T1): KHONG arm nao huu han -> bo buoc
            nb += 1                                                    # 78t (D-T1): backward DA chay trong vong arm
            if (j + 1) % ACCUM == 0:
                # Kiem gradient TRUOC khi buoc: total_norm = nan cho clip_coef = nan va BOI BAN
                # toan bo tham so -> mot buoc hong lam chet vinh vien.
                for t, ps in PARAM_OF.items():
                    gn = torch.nn.utils.clip_grad_norm_(ps, CLIP, error_if_nonfinite=False)
                    if not torch.isfinite(gn):
                        for p in ps:
                            if p.grad is not None: p.grad.zero_()
                        N_SKIP[t] += 1
                    elif float(gn) > CLIP: N_CLIP[t] += 1
                for t in PHU:                                    # 78m: module loss phu cat RIENG tung arm
                    gp = torch.nn.utils.clip_grad_norm_(list(PHU[t].parameters()), CLIP, error_if_nonfinite=False)
                    if not torch.isfinite(gp):
                        for p in PHU[t].parameters(): p.grad = None
                        BO_CAT_PHU[t] += 1
                    elif float(gp) > CLIP: CAT_PHU[t] += 1
                N_STEP += 1; opt.step()
                sch.step(); opt.zero_grad(set_to_none=True)

        _t_train = time.time() - t0                            # 78m: toc do / VRAM cua RIENG vong train
        _vr = (torch.cuda.max_memory_allocated() / 1e9) if dev == 'cuda' else 0.0
        PV = predict(_IVA)
        for t, p in PV.items():
            ghi(f'p_{t}_{sk}_ep{ep+1}.npy', lambda q, v=p: np.save(q, v))
        # KHONG luu `psl_`: lat chon da NAM TRONG train o luot nay, nen du doan tren no la
        # TRONG MAU. Luu ra la moi nguoi (ke ca ta) dung no de chon.
        if DO_TEST:          # MOI epoch; dac trung XGB MOI (ghim). CHI ghi du doan, KHONG doc nhan test
            PT = predict(_ITE, arms=ARMS_CHAM)   # 78d: CHI arm cham (chi anh); thay KHONG co pte_
            for t, p in PT.items():
                ghi(f'pte_{t}_{sk}_ep{ep+1}.npy', lambda q, v=p: np.save(q, v))
        with torch.no_grad():
            # TI LE BIEN DO THUC cua nhanh phim truoc: ‖phan du‖ / ‖patch‖. `pr_norm` chi la so
            # GIAN TIEP; day moi la dai luong vuot 1 thi sup. Do tren mot lo val co dinh.
            _bd = next(iter(loader(I_VA[:32], eval_tf, False, 32, nw=0)))
            _hb = shared_lo(_bd['image'].to(dev))
            _hbp = shared_lo(_bd['image_prev'].to(dev)) if CAN_PHIM else _hb   # 78m: image_prev la tensor 1 phan tu
            _hs2 = _bd['has'].to(dev)
            _pb, _ppb = upper_frozen(_hb, _bd['row'].to(dev)), upper_frozen(_hbp, _bd['row'].to(dev))
            _ri = {}
            for t in IMG:
                _src = upper_ft(_hb, t) if ARMS[t]['ft'] else _pb
                _srp = upper_ft(_hbp, t) if ARMS[t]['ft'] else _ppb
                _ri[t] = float(((IMG[t](_src, _srp, _hs2) - _src).norm(dim=-1)
                                / (_src.norm(dim=-1) + 1e-9)).mean())
            _dr = {t: float(max((_first2d(UPF[t][k]) - _REF[i]).norm()
                                / (_REF[i].norm() + 1e-9) for k, i in enumerate(FT_BLOCKS)))
                   for t in UPF}
            # `drift` mot minh khong phan biet duoc BI CO LAI voi XOAY HUONG. `wnorm` = ‖w‖/‖w_goc‖:
            # < 1 la bi bao mon, ~1 ma drift van tang la thich nghi that.
            _wn = {t: float(np.mean([float(_first2d(UPF[t][k]).norm() / (_REF[i].norm() + 1e-9))
                                     for k, i in enumerate(FT_BLOCKS)])) for t in UPF}
        _e2 = {}
        for t, p in PV.items():
            ap = ap_cols(YM[_IVA], p)
            # KHONG co `sel`: lat chon nam trong train o luot nay (xem C0)
            _e2[t] = dict(mAP_anh=float(np.nanmean(ap)), **tiers(ap),
                          pr_norm=float(IMG[t].pr.weight.norm()) if t in IMG else 0.0,
                          clip_pct=N_CLIP[t]/max(N_STEP, 1), drift=_dr.get(t, 0.0),
                          ratio_img=_ri.get(t, 0.0), wnorm=_wn.get(t, 0.0),
                          loss_nhan=tot[t] / max(nb, 1))
        with _fork(): CD = chan_doan_giu()     # 78t: khoi attention (moi arm), AUC dich, truy hoi cau, troi query tren nhom GIU RIENG;
                                               # fork: DataLoader moi KHONG tieu dong ngau nhien chung
        for t in ARMS:
            _e2[t].update(CD['vung'][t], **CD['txt'].get(t, {}))
            if t not in PHU_CFG: continue
            _o = TOT_PHU[t]
            _e2[t].update(**{f'loss_{k}': (_o[k] / max(_o['n_' + k], 1) if PHU_CFG[t][k] else None) for k in PHAN},
                          khoi=(_o['khoi'] / max(_o['n_vung'], 1) if PHU_CFG[t]['vung'] else None),
                          dung1=(float(D1[t][0] / D1[t][1]) if t in D1 and float(D1[t][1]) > 0 else None),
                          tau=(float(PHU[t]['cau'].tau().detach()) if PHU_CFG[t]['cau'] else None), lam=LAM[t],
                          he_so=he_so_tang(N_STEP, steps, TI_LE_TANG),
                          ti_le_grad=(float(np.mean(TL[t])) if TL[t] else None), n_phu=N_PHU[t], phu_bo=PHU_BO[t],
                          cat_phu=CAT_PHU[t], bo_cat_phu=BO_CAT_PHU[t],
                          so_ung_cau=(dict(tb=float(np.mean(SO_U)), max=int(max(SO_U))) if PHU_CFG[t]['cau'] and SO_U else None),
                          **CD['dich'].get(t, {}), **CD['cau'].get(t, {}))
        LOG[sk][str(ep)] = _e2
        ghi(f'epoch_log_{sk}_ep{ep+1}.json', lambda q: ghi_json(q, LOG))
        # Trong so (~40 MB) va optimizer/lich TACH hai file, TEN RIENG moi epoch -> khong bao gio ghi de
        # (53/r3 mat trong so ep2-10 vi ghi de MOT file qua FUSE). Resume can CA HAI cua cung epoch.
        ghi(f'w_{sk}_ep{ep+1}.pt', lambda q: torch.save(dict(
            mod=MOD.state_dict(), etk=ETK.state_dict(), img=IMG.state_dict(), upf=UPF.state_dict(),
            phu=sd_phu(), lam=LAM,                            # 78t: module loss phu (thap: CHI phan mo) + lambda da chot
            ep=ep, seed=seed, arms=list(ARMS), run=RUN_TAG), q))
        luu_resume(lambda q: torch.save(dict(   # 78z J (ghi mot lan, Tuan 17/10): MOT tep resume GHI DE = w_ + rs_ + tep dang cho
            mod=MOD.state_dict(), etk=ETK.state_dict(), img=IMG.state_dict(), upf=UPF.state_dict(), phu=sd_phu(), lam=LAM,
            opt=opt.state_dict(), sch=sch.state_dict(), n_clip=N_CLIP, n_skip=N_SKIP, n_step=N_STEP, rng=lay_rng(),
            ep=ep, seed=seed, arms=list(ARMS), run=RUN_TAG, nguon_bb=NGUON_BB, cho=cho_nho()), q))
        drive_gate(f'{sk} ep{ep+1}')          # KHONG nem loi (luat 4): hang cho, thu lai lan ghi sau
        g = lambda a, b: _e2[a]['mAP_anh'] - _e2[b]['mAP_anh']
        print(f'{sk} ep{ep+1:>2} | {nb*BATCH/max(time.time()-t0,1e-9):.0f} img/s | '
              f'{(time.time()-t0)/60:.1f} ph | loss ' +
              ' '.join(f'{t}:{tot[t]/max(nb,1):.3f}' for t in ARMS))
        print('   VAL      ' + ' '.join(f'{t:>9}:{_e2[t]["mAP_anh"]:.4f}' for t in ARMS))
        print('   TAIL     ' + ' '.join(f'{t:>9}:{_e2[t]["tail"]:.4f}' for t in ARMS))
        print('   ^ VAL chi de THEO DOI suc khoe — E=10 da KHOA, khong duoc doi gi theo dong nay')
        print(f'   TRAIN    {nb * BATCH / max(_t_train, 1e-9):.0f} img/s (RIENG vong train, {_t_train / 60:.1f} ph) | VRAM '
              f'dinh {_vr:.1f} / {(torch.cuda.get_device_properties(0).total_memory / 1e9) if dev == "cuda" else 0:.0f} GB (uoc C4 '
              f'~{VRAM_UOC:.0f}; > 90 GB -> tach hai notebook: ARMS_GIU + LUOT_78T + LAM_TU)'
              + (f' | ung vien cau / buoc TB {np.mean(SO_U):.0f}, max {max(SO_U)}' if SO_U else ''))
        if THU_NHANH:                                       # 78t (D-T1): VRAM DO DUOC -> MOT notebook hay TACH (Tuan quyet)
            print(f'   THU_NHANH: VRAM DINH DO DUOC {_vr:.1f} GB ({len(ARMS)} arm, backward RIENG tung arm; uoc C4 ~{VRAM_UOC:.0f} '
                  'GB) -> ' + ('> 90 GB: TACH notebook - CHI_LAM = True mot lan (lambda luot du), roi moi notebook ARMS_GIU (co '
                               'mldec) + LUOT_78T + LAM_TU' if _vr > 90 else 'vua MOT notebook (khong can tach)'))
        print(f'   CHENH - {DOI_CHUNG}: ' + ' '.join(f'{t}:{g(t, DOI_CHUNG):+.4f}' for t in ARMS if t != DOI_CHUNG)
              + '   (1 seed: < 0.003 chua phan biet voi nhieu seed)')
        _fm = lambda v, w=3: '-' if v is None else f'{v:.{w}f}'
        _tbl = lambda v: None if not v or all(x is None for x in v) else float(np.mean([x for x in v if x is not None]))
        print(f'   PHU (h {he_so_tang(N_STEP, steps, TI_LE_TANG):.2f}) ' + ' | '.join(
            f"{t} " + ' '.join(f"{k} {_fm(_e2[t]['loss_' + k])}" for k in PHAN if PHU_CFG[t][k])
            + (f" tau {_fm(_e2[t]['tau'])} dung1 {_fm(_e2[t]['dung1'])}" if PHU_CFG[t]['cau'] else '')
            + f" g {_fm(_e2[t]['ti_le_grad'], 2)}" for t in PHU_CFG))
        print('   GIU RIENG dich AUC T1/T2/T5/T6/T7 | acc T3/T4 ' + ' '.join(
            f"{t}:" + '/'.join(_fm(_e2[t].get(k + '_auc') if k in ('T1', 'T2') else _tbl(_e2[t].get(k + '_auc')))
                               for k in ('T1', 'T2', 'T5', 'T6', 'T7')) + f"|{_fm(_e2[t].get('T3_acc'))}/{_fm(_e2[t].get('T4_acc'))}"
            for t in PHU_CFG if PHU_CFG[t]['dich']))
        print('   GIU RIENG cau top1 / hit10 (ngau nhien) ' + ' '.join(
            f"{t}:{_fm(_e2[t].get('cau_top1'))}/{_fm(_e2[t].get('cau_hit10'))} ({_fm(_e2[t].get('cau_top1_nn'))}/"
            f"{_fm(_e2[t].get('cau_hit10_nn'))})" for t in PHU_CFG if PHU_CFG[t]['cau']))
        print('   KHOI ATTENTION trong vung kdp (5 nhan trong phoi) ' + ' '.join(
            f"{t}:{_fm(_e2[t].get('khoi_kdp'))} ({_fm(_e2[t].get('khoi_kdp_phoi'))})" for t in ARMS))
        print('   QUERY cos(nay, dau) TB ' + ' '.join(f"{t}:{_fm(_e2[t].get('cos_q'))}" for t in ARMS if 'cos_q' in _e2[t])
              + ''.join(f" | {t} ||Delta|| / ||co so|| {_fm(_e2[t]['ti_le_delta'])}" for t in ARMS if 'ti_le_delta' in _e2[t]))
        print(f"   bo phu {sum(_e2[t]['phu_bo'] for t in PHU_CFG)} | cat phu {sum(_e2[t]['cat_phu'] for t in PHU_CFG)} "
              f"| bo cat phu {sum(_e2[t]['bo_cat_phu'] for t in PHU_CFG)} | clip dau doc "
              f"{max(v['clip_pct'] for v in _e2.values()):.1%}")
        if max(N_SKIP.values()) > 0:
            print('   BUOC BI BO: ' + ' '.join(f'{t}:{v}' for t, v in N_SKIP.items() if v > 0))
    LOG[sk]['done'] = True
    ghi(f'epoch_log_{sk}_cuoi.json', lambda q: ghi_json(q, LOG))   # trong so cuoi = w_ epoch cuoi
    print(f'{sk}: xong')
dong_bo()                                    # 78m (sua cua 78l, ra soat 01/10): bao NGAY tep nao chua len Drive
print('tat ca seed xong')
