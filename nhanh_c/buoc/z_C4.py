# ==== nhanh_c buoc z_C4 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# 78t (Tuan duyet 06/10; docs/PLAN_SAU_78G.md 4-6, spec 78t) - MOI arm CHI ANH luc cham, dung chung MOT forward backbone.
# Loss phu CHI luc train (PHU_CFG, MOI phan mot r; lambda = r x ||grad L_nhan|| / ||grad L_phan||, hieu chinh dau C6, CHOT truoc
# train): vung = phat attention lop -> patch ngoai vung goi y KHONG du phong (vung_kdp 78g); dich = 7 nhom dich moi doc h (DauDich,
# bo luc cham); cau = InfoNCE da duong muc cau anh <-> cau report train (PhuCau: vector BiomedBERT dong cung | thap hoc cung).
# Query lop (q): co dinh | hoc ('hoc') | van ban ('txt', 'du' = + Delta, 'k4' = 4 nguyen mau / lop); E = nguon vector lop.
#   mldec      doi chung (78g)                      q_hoc      query hoc duoc (78g)
#   txtq       40 mo ta lop 78e (78g)                txtq_nn    E ngau nhien N(0, 1) (doi chung NGHIA van ban)
#   txtq_5mt   TB 5 cach dien dat (78s)              txtq_cau   TB cau report khang dinh (78s)
#   txtq_k4    4 nguyen mau / lop -> 160 query, LSE - log 4      txtq_du    txtq + Delta hoc duoc (khoi tao 0)
#   txtq_vung  txtq + vung kdp (0.3)                 vung_kdp   vung kdp (0.3) (78g)
#   dich_thay  T1 + T2 | dich_tt  T3 + T4 + T7 | dich_tb  T5 + T6 | dich_tat_ca / txtq_dich  T1..T7 (r 0.3; txtq_dich: query txtq)
#   cau_bert   loss cau, vector cau BiomedBERT dong cung (0.3)   cau_clip   loss cau, thap BiomedBERT hoc cung (0.3)
# Nhom DICH_TAT (C3c; mac dinh T3 + T7, Tuan 07/10) bo khoi moi arm, KHONG xoa: dich_tt = T4, dich_tat_ca / txtq_dich = T1 T2 T4-T6.
_dich_bat = lambda *g: tuple(x for x in g if x not in DICH_TAT)
_TAT_CA_DICH = _dich_bat('T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7')
_ARMS_78T = {
  'mldec':       dict(),
  'q_hoc':       dict(q='hoc'),
  'txtq':        dict(q='txt', E='mota'),
  'txtq_nn':     dict(q='txt', E='nn'),
  'txtq_5mt':    dict(q='txt', E='5mt'),
  'txtq_cau':    dict(q='txt', E='cau'),
  'txtq_k4':     dict(q='k4', E='k4'),
  'txtq_du':     dict(q='du', E='mota'),
  'txtq_vung':   dict(q='txt', E='mota', phu=dict(vung='kdp', r_vung=R_CAO)),
  'vung_kdp':    dict(phu=dict(vung='kdp', r_vung=R_CAO)),
  'dich_thay':   dict(phu=dict(dich=_dich_bat('T1', 'T2'), r_dich=R_CAO)),
  'dich_tt':     dict(phu=dict(dich=_dich_bat('T3', 'T4', 'T7'), r_dich=R_CAO)),
  'dich_tb':     dict(phu=dict(dich=_dich_bat('T5', 'T6'), r_dich=R_CAO)),
  'dich_tat_ca': dict(phu=dict(dich=_TAT_CA_DICH, r_dich=R_CAO)),
  'txtq_dich':   dict(q='txt', E='mota', phu=dict(dich=_TAT_CA_DICH, r_dich=R_CAO)),
  'cau_bert':    dict(phu=dict(cau='bert', r_cau=R_CAO)),
  'cau_clip':    dict(phu=dict(cau='clip', r_cau=R_CAO)),
  'mldec_hr':    dict(hr=True),          # 78y: dau J doc [patch ; token vung] (DocVung gan vao MLDecoder; CHI khi co LUONG)
  'fus_tb_hr':   dict(hr=True, fus='tb'),    # 78z J: fusion tren mldec_hr, TB dac trung cac anh cung study + nhung view
  'fus_attn_hr': dict(hr=True, fus='attn'),  # 78z J: fusion tren mldec_hr, attention giua cac anh cua study, rieng tung lop
}
assert tuple(_ARMS_78T) == tuple(TEN_ARM_78T), 'bang arm C4 lech danh sach ten C0 (TEN_ARM_78T)'
_rong = [t for t, c in _ARMS_78T.items() if 'dich' in c.get('phu', {}) and not c['phu']['dich']]
assert not _rong, f'DICH_TAT {DICH_TAT} (C3c) tat HET nhom dich cua arm {_rong} -> bat lai mot nhom, hoac bo arm (ARMS_GIU)'
_GIU = list(_ARMS_78T) if ARMS_GIU is None else list(ARMS_GIU)
ARMS = {t: dict(ehr=False, img=False, ft=False, **{k: c[k] for k in ('q', 'E', 'hr', 'fus') if k in c}) for t, c in _ARMS_78T.items() if t in _GIU}
PHAN = ('vung', 'dich', 'cau')
PHU_CFG = {t: dict(dict(vung=None, r_vung=0.0, dich=None, r_dich=0.0, cau=None, r_cau=0.0), **c['phu'])
           for t, c in _ARMS_78T.items() if 'phu' in c and t in ARMS}
ARMS_CHAM = list(ARMS)                                  # moi arm chi anh -> deu cham (pte_)
DOI_CHUNG = 'mldec'
assert list(ARMS)[0] == DOI_CHUNG and DOI_CHUNG not in PHU_CFG and (ARMS_GIU is not None or len(ARMS) == 17), list(ARMS)
assert all(all((c[k] is not None) == (c['r_' + k] > 0) for k in PHAN) and c['vung'] in (None, 'kdp') and c['cau'] in (None, 'bert', 'clip')
           and (c['dich'] is None or (len(c['dich']) > 0 and set(c['dich']) <= set(NHOM_DICH))) for c in PHU_CFG.values()), PHU_CFG
CO_CAU = any(c['cau'] for c in PHU_CFG.values())
CO_CLIP = any(c['cau'] == 'clip' for c in PHU_CFG.values())
# cau_clip: BiomedBERT GHIM (TXT_MODEL@TXT_REV, = 78e / 78s) nap MOT lan (CPU, eval, DONG CUNG - ban mau, khong bao gio hoc;
# trong fork_rng -> khong tieu dong chung); build_phu chep ban MOI moi lan (khong tai lai). Pad id 0 (cau_token 78s dem 0).
if CO_CLIP:
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev == 'cuda' else []):
        _BERT_GOC = AutoModel.from_pretrained(TXT_MODEL, revision=TXT_REV).eval().requires_grad_(False)
    assert getattr(_BERT_GOC.config, 'pad_token_id', 0) == 0 and len(_BERT_GOC.encoder.layer) > THAP_MO, (
        'BiomedBERT: pad id phai 0 (cau_token 78s) va so tang > THAP_MO')
else:
    _BERT_GOC = None
print(f'78t ({RUN_TAG}): {len(ARMS)} arm | doi chung {DOI_CHUNG} | query: '
      + ' '.join(f"{t}({c['q']}{'/' + c['E'] if 'E' in c else ''})" for t, c in ARMS.items() if c.get('q')) + ' | loss phu: '
      + ' '.join(f"{t}(" + ', '.join(f"{k} {c[k] if isinstance(c[k], str) else '+'.join(c[k])} r {c['r_' + k]:g}" for k in PHAN if c[k])
                 + ')' for t, c in PHU_CFG.items()))
assert '_KHOA_C2B' in globals() and _KHOA_C2B == (THU_NHANH, RUN_TAG, LUOT_78Y), 'chay C2b (78y) truoc C4 -> DUNG'
assert '_KHOA_C2C' in globals() and _KHOA_C2C == (THU_NHANH, RUN_TAG), 'chay C2c (78z) truoc C4 -> DUNG'
_KHOA_C4 = (THU_NHANH, RUN_TAG, BATCH, ACCUM, ARMS_GIU)   # C6 kiem: doi C0 thi PHAI chay lai C4
FT_ARMS = [t for t, c in ARMS.items() if c['ft']]

def build_arms(seed=SEED):
    M, E, I, U = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
    for tag, cfg in ARMS.items():
        torch.manual_seed(seed)     # gieo lai TRUOC TUNG arm -> khoi tao giong het nhau
        if cfg['ehr']: E[tag] = ColTokBag(FEAT.shape[1], D_ETOK, False, V=V_MIC)
        M[tag] = (MLDecoderHaiDuong if cfg.get('hai') else MLDecoderTxtQK if cfg.get('q') == 'k4'
                  else MLDecoderTxtQ if cfg.get('q') in ('txt', 'du') else MLDecoder)(
            C, D_CXR, D_MLD, d_ehr_in=D_ETOK, t1=cfg['ehr'],
            **({'E_mo_ta': E_ARM[cfg['E']], 'hat': seed + 3} if cfg.get('q') in ('txt', 'du', 'k4') else {}),
            **({'K': K_NM} if cfg.get('q') == 'k4' else {}))   # 78t: ho txtq - CUNG hat qtxt (78g: seed + 3), chi E khac
        if cfg.get('q') == 'du': M[tag].qtxt = QueryTuTextDu(M[tag].qtxt)   # 78t txtq_du: + Delta (khoi tao 0)
        if cfg.get('q') == 'hoc': M[tag].q.weight.requires_grad_(True)   # 78g q_hoc: query HOC DUOC, cung khoi tao mldec
        if cfg.get('hr'):                                                 # 78y mldec_hr: + DocVung (LN + e_* = 0, khong rut ngau nhien)
            assert LUONG is not None, 'mldec_hr can luong vung (C2b, luot J_v3) -> DUNG'
            M[tag].docv = DocVung(LUONG.R, D_CXR, D_MLD)
        if cfg.get('fus'):                                                # 78z J fus_*_hr: + GopStudy (tham so 0, khong rut ngau nhien)
            M[tag].fus = GopStudy(cfg['fus'], C, D_MLD, n_view=int(CFG_FS['n_view']))
        if cfg['img']: I[tag] = Tier1Img(D_CXR)
        if cfg['ft']:
            # BAN SAO RIENG cua block 8-11 + layernorm. Dung chung thi viec cap nhat chung se
            # doi dac trung cua `mldec` va `tpb` -> pha luon hai doi chung.
            U[tag] = nn.ModuleList([copy.deepcopy(LAYERS[i]) for i in FT_BLOCKS]
                                   + [copy.deepcopy(backbone.layernorm)])
            for p in U[tag].parameters(): p.requires_grad_(True)
    return M.to(dev), E.to(dev), I.to(dev), U.to(dev)


def _bert_moi():
    '''Ban sao MOI cua BiomedBERT goc (_BERT_GOC, C4 nap MOT lan; CPU, eval) - khong tai lai, khong rut so ngau nhien.'''
    return copy.deepcopy(_BERT_GOC)


def build_phu(seed=SEED):
    '''78t: module loss phu cua tung arm co `dich` / `cau` (arm CHI vung KHONG co tham so phu) -> PHU[t] = ModuleDict:
       dich = DauDich (nhom cua arm; khoi tao THEO NHOM tu hat rieng -> cung nhom cung khoi tao giua cac arm); cau = PhuCau (anh:
       ChieuAnh D_MLD -> 512 -> 256 tren TB h; text: ThapVector tren vector cau z-score (bert) | ThapBert tren token, BiomedBERT MO
       THAP_MO tang tren (clip); nhiet do hoc duoc). Gieo lai TRUOC TUNG module trong fork_rng -> KHONG tieu dong ngau nhien chung;
       cung hat -> ChieuAnh / NhietDo cua hai arm cau TRUNG nhau, dau chieu ThapVector = ThapBert.dau khi cung hinh.'''
    P = nn.ModuleDict()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev == 'cuda' else []):
        for t, c in PHU_CFG.items():
            m = nn.ModuleDict()
            if c['dich']:
                m['dich'] = DauDich(D_MLD, C, IDX3, IDX4, N5, N6, N7, c['dich'], hat=seed + 4)
            if c['cau']:
                torch.manual_seed(seed + 2)
                thap = (ThapVector(D_TEXT, 512, 256) if c['cau'] == 'bert'
                        else ThapBert(mo_bert_tren(_bert_moi(), THAP_MO), 512, 256, chuan=TK_CAU))   # D-T2: z-score co dinh
                torch.manual_seed(seed + 5)
                m['cau'] = PhuCau(D_MLD, thap, d_an=512, d_chieu=256, tau0=TAU0_CAU)
            if len(m): P[t] = m
    return P.to(dev)

MOD, ETK, IMG, UPF = build_arms()
NP_ = {t: sum(p.numel() for m in ([MOD[t]] + [x[t] for x in (ETK, IMG, UPF) if t in x])
              for p in m.parameters() if p.requires_grad) for t in ARMS}
print(f'{"arm":<12} {"tham so":>12} {"trong do block 8-11":>21}')
for t in ARMS:
    nu = sum(p.numel() for p in UPF[t].parameters()) if t in UPF else 0
    print(f'  {t:<10} {NP_[t]:>12,} {nu:>21,}')

_p = torch.randn(4, N_TOK, D_CXR, device=dev)
assert not any(p.requires_grad for p in backbone.parameters()), 'backbone GOC phai dong cung'
# cong 2 (78t): MOI arm CHI ANH; CHI q_hoc hoc query; ho txtq (txt / du / k4) co query tu text; phan chung decoder CUNG khoi tao
# (moi tham so TRU qtxt.* va out.* cua txtq_k4)
assert len(ETK) == 0 and len(IMG) == 0 and len(UPF) == 0 and all(MOD[t].t1 is None for t in ARMS), '78t: arm co EHR / phim truoc'
assert all(MOD[t].q.weight.requires_grad == (ARMS[t].get('q') == 'hoc') for t in ARMS), 'chi q_hoc duoc hoc query'
assert all(hasattr(MOD[t], 'qtxt') == (ARMS[t].get('q') in ('txt', 'du', 'k4')) for t in ARMS), 'chi ho txtq co query tu text'
_sd0 = MOD[DOI_CHUNG].state_dict()
_rieng = lambda t, k: k.startswith('qtxt.') or (ARMS[t].get('q') == 'k4' and k.startswith('out.'))
assert all(set(MOD[t].state_dict()) == set(_sd0) | {k for k in MOD[t].state_dict() if k.startswith('qtxt.')
                                                     or (ARMS[t].get('hr') and k.startswith('docv.'))
                                                     or (ARMS[t].get('fus') and k.startswith('fus.'))}   # 78y: + DocVung; 78z J: + GopStudy
           and all(torch.equal(_sd0[k], MOD[t].state_dict()[k]) for k in _sd0 if not _rieng(t, k)) for t in ARMS), \
    'decoder cac arm KHAC khoi tao (phan chung)'
# ho txtq: CUNG Linear / LayerNorm cua qtxt (CUNG hat), E = E_ARM cua arm; txtq_du: Delta = 0; txtq_k4: out = GroupFC mldec lap K_NM
_TXT = [t for t in ARMS if hasattr(MOD[t], 'qtxt')]
_qs = lambda t: {k: v for k, v in MOD[t].qtxt.state_dict().items() if k.split('.')[0] in ('lin', 'ln')}
assert all(_qs(t).keys() == _qs(_TXT[0]).keys() and all(torch.equal(v, _qs(_TXT[0])[k]) for k, v in _qs(t).items())
           and torch.equal(MOD[t].qtxt.E, E_ARM[ARMS[t]['E']].to(dev)) for t in _TXT), 'ho txtq: Linear / LayerNorm KHAC nhau hoac E lech'
assert all(not MOD[t].qtxt.delta.any() for t in _TXT if ARMS[t]['q'] == 'du'), 'txtq_du: Delta phai = 0 luc dau'
assert all(torch.equal(MOD[t].out.w, _sd0['out.w'].repeat_interleave(K_NM, 0))
           and torch.equal(MOD[t].out.b, _sd0['out.b'].repeat_interleave(K_NM, 0)) for t in _TXT if ARMS[t]['q'] == 'k4'), \
    'txtq_k4: dau ra phai = GroupFC cua mldec lap K_NM lan'
_frk = lambda: torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev == 'cuda' else [])
with torch.no_grad():
    _L4 = {t: MOD[t].eval()(_p, None)[0] for t in ARMS}
    if 'txtq' in ARMS:                   # txtq = arm dung bang LOP cua 78g (MLDecoderTxtQ, CUNG khoi tao) TUNG BIT
        with _frk():
            torch.manual_seed(SEED)
            _m78 = MLDecoderTxtQ(C, D_CXR, D_MLD, d_ehr_in=D_ETOK, t1=False, E_mo_ta=E_MOTA_T, hat=SEED + 3).to(dev).eval()
        assert torch.equal(_m78(_p, None)[0], _L4['txtq']), 'txtq KHAC arm dung bang lop 78g (cung khoi tao) -> DUNG'
    if {'txtq', 'txtq_nn'} <= set(ARMS):  # txtq_nn chi khac txtq o E: doi E -> logit TRUNG txtq
        _sa, _sb = MOD['txtq'].state_dict(), MOD['txtq_nn'].state_dict()
        assert {k for k in _sa if not torch.equal(_sa[k], _sb[k])} == {'qtxt.E'}, 'txtq_nn khac txtq NGOAI E -> DUNG'
        _mx = copy.deepcopy(MOD['txtq_nn']); _mx.qtxt.E.copy_(MOD['txtq'].qtxt.E)
        assert torch.equal(_mx(_p, None)[0], _L4['txtq']), 'txtq_nn voi E cua txtq KHAC txtq -> DUNG'
    if {'txtq', 'txtq_du'} <= set(ARMS):
        assert torch.equal(_L4['txtq_du'], _L4['txtq']), 'txtq_du luc dau (Delta = 0) KHAC txtq -> DUNG'
    if {'txtq', 'txtq_k4'} <= set(ARMS):  # K nguyen mau GIONG nhau (E txtq lap K) -> = txtq (LSE - log K)
        with _frk():
            torch.manual_seed(SEED)
            _m4 = MLDecoderTxtQK(C, D_CXR, D_MLD, d_ehr_in=D_ETOK, t1=False, E_mo_ta=E_MOTA_T.repeat_interleave(K_NM, 0),
                                 hat=SEED + 3, K=K_NM).to(dev).eval()
        _d4 = float((_m4(_p, None)[0] - _L4['txtq']).abs().max())
        assert _d4 <= 1e-4, f'txtq_k4 voi {K_NM} nguyen mau GIONG nhau lech txtq {_d4:.1e} (LSE - log K sai?) -> DUNG'
print('cong txtq: txtq = lop 78g TUNG BIT | txtq_nn chi khac E | txtq_du luc dau = txtq | txtq_k4 nguyen mau giong nhau = txtq | '
      'ho txtq cung Linear / LayerNorm, phan chung decoder cung khoi tao')
# cong mldec_h (C3c): logit TRUNG forward tung bit (eval; train cung dong ngau nhien); attention tinh lai = MHA o eval
with torch.no_grad():
    for t in ARMS:
        _l1 = MOD[t].eval()(_p, None)[0]; _l2, _h2, _k2 = mldec_h(MOD[t], _p)
        assert torch.equal(_l1, _l2) and tuple(_h2.shape) == (4, q_cua(MOD[t]).shape[0], D_MLD), t
        _qr = q_cua(MOD[t]).unsqueeze(0).expand(4, -1, -1)      # KHONG ten `_q`: ham cua khoi Drive (C0)
        _wm = MOD[t].img.at(_qr, _k2, _k2, need_weights=True, average_attn_weights=True)[1]
        assert torch.allclose(attn_truoc_dropout(MOD[t].img.at, _qr, _k2), _wm, atol=1e-5), f'{t}: attention tinh lai lech MHA'
        torch.manual_seed(5); _l1 = MOD[t].train()(_p, None)[0]
        torch.manual_seed(5); _l2 = mldec_h(MOD[t], _p)[0]
        assert torch.equal(_l1, _l2), f'{t}: mldec_h lech forward o che do train'
print('cong 2: moi arm chi anh, phan chung decoder cung khoi tao | mldec_h = forward tung bit (eval + train) | attention tinh '
      'lai = MHA (eval)')
# 78y cong mldec_hr: CHI dau `hr` co DocVung; luc dau e_loai = e_vung = 0 (LN mac dinh); mldec_h_hr = dau_h_v2 cua J v2 (khoi_jca_v2)
# TUNG BIT (eval, CUNG query co dinh); token vung DOI logit (dau that su doc r). Rut so trong fork_rng -> KHONG tieu dong ngau nhien chung.
_HR = [t for t in ARMS if ARMS[t].get('hr')]
assert all(hasattr(MOD[t], 'docv') == bool(ARMS[t].get('hr')) for t in ARMS), 'chi dau hr (mldec_hr) co DocVung -> DUNG'
if _HR:
    with torch.no_grad(), _frk():
        torch.manual_seed(SEED + 78)
        _rr = torch.randn(4, LUONG.R, D_CXR, device=dev)
        for t in _HR:
            _dv = MOD[t].eval().docv
            assert not MOD[t].docv.e_loai.any() and not MOD[t].docv.e_vung.any(), f'{t}: DocVung e_* KHAC 0 luc dau -> DUNG'
            _a = mldec_h_hr(MOD[t], q_cua(MOD[t]), _p, _rr); _b = dau_h_v2(MOD[t], _p, _dv(MOD[t].pin, _rr))
            assert all(torch.equal(x, y) for x, y in zip(_a, _b)), f'{t}: mldec_h_hr lech dau_h_v2 (J v2) -> DUNG'
            assert not torch.equal(_a[0], mldec_h_hr(MOD[t], q_cua(MOD[t]), _p, 2 * _rr + 1)[0]), f'{t}: token vung KHONG doi logit -> DUNG'
            MOD[t].train()
    print(f"cong mldec_hr ({', '.join(_HR)}): DocVung luc dau e_* = 0 | = dau_h_v2 (J v2) tung bit | token vung doi logit")
# 78z J cong fusion (khoi tao): fus_*_hr = MLDecoder + DocVung CUNG khoi tao mldec_hr + GopStudy tham so 0 -> study MOT anh: logit
# fusion = logit mldec_hr TUNG BIT; luc dau fus_tb_hr = fus_attn_hr TUNG BIT; study hai anh = TB logit; KHONG rut so ngau nhien chung.
_FUS = [t for t in ARMS if ARMS[t].get('fus')]
assert _FUS == ['fus_tb_hr', 'fus_attn_hr'] and all(ARMS[t].get('hr') for t in _FUS) and 'mldec_hr' in ARMS and all(
    hasattr(MOD[t], 'fus') == bool(ARMS[t].get('fus')) for t in ARMS), 'chi dau fus_*_hr co GopStudy (tren mldec_hr) -> DUNG'
assert all(MOD[t].fus.kieu == ARMS[t]['fus'] and not MOD[t].fus.e_view.any() and (ARMS[t]['fus'] == 'tb' or (
    not MOD[t].fus.w.any() and not MOD[t].fus.b.any())) for t in _FUS), 'GopStudy khoi tao KHAC 0 / sai kieu -> DUNG'
_s0f = torch.get_rng_state()
with torch.no_grad(), _frk():
    for t in ARMS: MOD[t].eval()
    torch.manual_seed(SEED + 79)
    _rrf = torch.randn(4, LUONG.R, D_CXR, device=dev)
    _stf = torch.tensor([5, 5, 6, 7], device=dev); _vwf = torch.tensor([0, 1, 0, 2], device=dev)
    _l0f = mldec_h_hr(MOD['mldec_hr'], q_cua(MOD['mldec_hr']), _p, _rrf)[0]
    _lf = {t: MOD[t].out(MOD[t].fus(mldec_h_hr(MOD[t], q_cua(MOD[t]), _p, _rrf)[1], _stf, _vwf)) for t in _FUS}
    for t in ARMS: MOD[t].train()
assert all(torch.equal(_lf[t][2:], _l0f[2:]) for t in _FUS), 'study MOT anh: logit fusion KHAC mldec_hr (khoi tao) -> DUNG'
assert torch.equal(_lf['fus_tb_hr'], _lf['fus_attn_hr']), 'luc dau fus_tb_hr KHAC fus_attn_hr -> DUNG'
assert torch.allclose(_lf['fus_tb_hr'][0], 0.5 * (_l0f[0] + _l0f[1]), atol=1e-5) and torch.equal(_lf['fus_tb_hr'][0], _lf['fus_tb_hr'][1]), (
    'study hai anh: logit fusion luc dau KHAC TB logit hai anh (GroupFC tuyen tinh) -> DUNG')
assert torch.equal(_s0f, torch.get_rng_state()), 'cong fusion TIEU dong ngau nhien chung -> DUNG'
print(f"cong fusion J ({', '.join(_FUS)}): GopStudy tham so 0 | study mot anh = mldec_hr tung bit | luc dau tb = attn tung bit | study hai "
      'anh = TB logit | dong ngau nhien chung khong doi')
del _stf, _vwf, _l0f, _lf, _s0f, _rrf
# cong PHU: build_phu KHONG tieu dong ngau nhien chung; dung module moi arm; thap cau_clip CHI hoc THAP_MO tang tren + dau chieu
_sts = lambda: [torch.get_rng_state()] + ([torch.cuda.get_rng_state()] if dev == 'cuda' else [])
_s0 = _sts(); PHU = build_phu()
assert all(torch.equal(a, b) for a, b in zip(_sts(), _s0)), 'build_phu TIEU dong ngau nhien chung'
assert set(PHU) == {t for t, c in PHU_CFG.items() if c['dich'] or c['cau']} and all(
    ('dich' in PHU[t]) == bool(PHU_CFG[t]['dich']) and ('cau' in PHU[t]) == bool(PHU_CFG[t]['cau']) for t in PHU), 'PHU sai cau hinh'
assert all(PHU[t]['dich'].nhom == tuple(PHU_CFG[t]['dich']) for t in PHU if 'dich' in PHU[t]), 'dau dich sai nhom'
_TB4 = [PHU[t]['cau'].text for t in PHU if PHU_CFG[t]['cau'] == 'clip']
assert all(p.requires_grad == (n.startswith('encoder.layer.') and int(n.split('.')[2]) >= len(m.mdl.encoder.layer) - THAP_MO)
           for m in _TB4 for n, p in m.mdl.named_parameters()) and all(p.requires_grad for m in _TB4 for p in m.dau.parameters()), \
    'thap cau_clip: CHI THAP_MO tang tren + dau chieu duoc hoc (embeddings + tang duoi dong cung)'
if {'cau_bert', 'cau_clip'} <= set(PHU):     # cung khoi tao phan chieu giua hai arm cau (chi khac thap)
    _ab = [PHU[t]['cau'] for t in ('cau_bert', 'cau_clip')]
    assert all(torch.equal(a, b) for a, b in zip(_ab[0].anh.state_dict().values(), _ab[1].anh.state_dict().values())) \
        and torch.equal(_ab[0].nhiet.tho, _ab[1].nhiet.tho), 'cau_bert / cau_clip KHAC khoi tao chieu anh / nhiet do'
    _ma, _mb = _ab[0].text.mlp.state_dict(), _ab[1].text.dau.state_dict()
    assert {k: v.shape for k, v in _ma.items()} != {k: v.shape for k, v in _mb.items()} or all(
        torch.equal(_ma[k], _mb[k]) for k in _ma), 'cau_bert / cau_clip: dau chieu CUNG hinh nhung KHAC khoi tao'
_np4 = lambda m, hoc: sum(p.numel() for p in m.parameters() if p.requires_grad == hoc)
print('cong PHU: ' + ' '.join(f'{t}:{_np4(PHU[t], True):,}' + (f' (+{_np4(PHU[t], False):,} dong cung)' if _np4(PHU[t], False) else '')
                              for t in PHU) + ' tham so hoc')
# ---- uoc luong theo cau hinh arm THAT: VRAM (uoc_vram_78t, C0) + Drive (tu so tham so; thap: CHI phan mo) ----
VRAM_UOC, _VG4, _VT4 = uoc_vram_78t(len(ARMS), sum(bool(c['vung']) for c in PHU_CFG.values()),
                                    sum(c.get('q') == 'k4' for c in ARMS.values()), CO_CLIP)
_n_hoc = sum(p.numel() for t in ARMS for p in MOD[t].parameters() if p.requires_grad) + _np4(PHU, True)
_GB_EP = (12 * _n_hoc + 4 * C * (len(I_VA) + len(I_TE)) * len(ARMS)) / 1e9
print(f'78t C4: VRAM UOC ~{VRAM_UOC:.0f} GB / buoc, backward RIENG tung arm (' + ', '.join(f'{k} {v:.1f}' for k, v in _VG4.items())
      + f'; MOT backward tren tong ~{_VT4:.0f} GB)'
      + (' -> > 90 GB: neu THU_NHANH do that cung > 90 GB, TACH hai notebook (moi ben co mldec): ARMS_GIU A = ho txtq + vung, '
         'B = dich + cau; LUOT_78T rieng; LAM_TU = lambda cua luot du' if VRAM_UOC > 90 else ''))
print(f'78t C4: Drive ~{_GB_EP:.2f} GB / epoch (w_ + rs_ = 3 x 4 byte x {_n_hoc:,} tham so hoc; p_ / pte_ {len(ARMS)} arm) -> '
      f'~{EPOCHS * _GB_EP:.1f} GB / luot {EPOCHS} epoch')
del _L4, _sd0
MOD, ETK, IMG, UPF = build_arms()
