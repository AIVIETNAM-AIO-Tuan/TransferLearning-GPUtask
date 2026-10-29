# ==== nhanh_c buoc p1_C6 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C6 · HUAN LUYEN 78x — cong duong cham, roi vong ft 38a (EPOCHS 6, LO THEO STUDY, BATCH 144 x 2, clip 5, ASL(4,0)) + loss
#      phu cua luot. Moi epoch: du doan val (p_), nhat ky, chan doan giu rieng; backbone_ft.pt sau epoch cuoi. 78x dot 4 (ghi mot lan):
#      MOT tep resume ghi de moi epoch + rs_ NHE (khong AdamW) MOI epoch ====
assert _KHOA_C5 == (THU_NHANH, RUN_TAG, LUOT_78U, CKPT_THAP) and set(LAM) == set(PHAN_CFG), (
    'C3 / C4 / C5 chay voi cau hinh C0 KHAC -> chay lai C0 -> C1 -> C2 -> C3 -> C4 -> C5 -> C6')
SK = f's{SEED}'
TEN_P = lambda e: f'p_{LUOT_79A}_{SK}_ep{e}.npy'      # du doan VAL float16 [n_val, 40] (78a doc), dau doc 38a
TEN_LOG = lambda e: f'epoch_log_{LUOT_79A}_ep{e}.json'
TEN_RS = lambda e: f'rs_{LUOT_79A}_ep{e}.pt'
TEN_BB = 'backbone_ft.pt'
TEN_P_THO = lambda e: TEN_P(e).replace('p_', 'p_tho_', 1)        # 78x dot 4 B.4: du doan val trong so THO
TEN_RS_THO = lambda e: TEN_RS(e).replace('rs_', 'rs_tho_', 1)    # 78x dot 4 B.4: trong so THO ep_bb
_so = lambda v: None if v is None or not np.isfinite(v) else float(v)


@torch.no_grad()
def predict(idx, bs=64):
    '''predict cua 38a, CHI dau doc ft: sigmoid(dau_v2(fwd_luong_v2(anh, view))) (78x: backbone + luong neo theo view + y 1 + kenh MIL) — duong cham CHI doc anh (worker KHONG persistent).'''
    bb_ft.eval(); HEAD.eval(); LUONG.eval(); DOCV.eval(); DAUV.eval()
    P = np.zeros((len(idx), C), 'float32')
    for b in tqdm(loader(idx, eval_tf, False, bs, nw=NW_EVAL), desc='val', leave=False):
        _p, _c, _ra = fwd_o(b['image'].to(dev, non_blocking=True), False, b['row'].to(dev))   # 78x o cat: + 5 o (skip block 7, token o cuoi)
        P[b['pos'].numpy()] = torch.sigmoid(dau_v2_o(_p, _ra)[0].float()).cpu().numpy()          # 78x o cat: dau doc 6 view
    assert not np.isnan(P).any(), 'predict cho ra NaN'
    return P


@torch.no_grad()
def chan_doan_giu():
    '''Chan doan tren nhom GIU RIENG (2% benh nhan train: vao loss nhan nhu thuong, KHONG vao loss phu), mo hinh o eval:
       - MOI luot: khoi attention cua dau doc 38a trong vung goi y 78q (co / khong du phong; tach 5 nhan trong phoi) — nhu 78g;
       - B1 / B2: truy hoi anh -> cau trong kho cau cua cac study giu rieng: top-1 da duong, top-10, va muc ngau nhien tuong ung;
       - B4: AUC tung kenh dich day (36 thuoc vung, bat thuong, 5 ket cau) tren hop KHONG bien doi.'''
    bb_ft.eval(); HEAD.eval(); LUONG.eval(); PHU_V.eval(); DOCV.eval(); DAUV.eval()
    if PHU is not None: PHU.eval()
    KH = {k: [] for k in ('dp', 'kdp', 'dp_phoi', 'kdp_phoi')}; ZA, STA, pos = [], [], []
    AUC = TichLuyAUC(R_VUNG + 1 + K_KC) if LUOT_78U == 'B4' else None
    AUC_V = TichLuyAUC(R_VUNG); IOU_V, IOU_0, GHI_G = [], [], []                 # 79a: luong vung (anh giu rieng)
    for b in loader(I_GIU_C, eval_tf, False, 64, nw=NW_EVAL):
        rows, y = b['row'].to(dev), b['y'].to(dev)
        patch, cls, ra = fwd_o(b['image'].to(dev, non_blocking=True), False, rows)   # 78x o cat
        kv = dau_h(HEAD, patch)[2]
        if ra['ghi']: GHI_G.append([float(t) for t in ra['ghi']])
        g = SG_HANG_T[rows]; co = g >= 0
        if bool(co.any()):
            hp = torch.from_numpy(HOP518[g[co].cpu().numpy()]).to(dev); cv = torch.isfinite(hp).all(-1)
            tr = o_luoi_tu_hop(hp, cv, G_LUOI, O_LUOI)
            MM = dict(dp=mat_na_nhan(M_DP_T[g[co]], tr, R_VUNG), kdp=mat_na_nhan(M_KDP_T[g[co]], tr, R_VUNG))
            A = attn_truoc_dropout(HEAD.img.at, HEAD.q.weight.unsqueeze(0).expand(int(co.sum()), -1, -1), kv[co])
            for k_, M_ in MM.items():
                kh = (A * M_.float()).sum(-1); hl_ = (y[co] > 0.5) & M_.any(-1)
                KH[k_].append(kh[hl_]); KH[k_ + '_phoi'].append(kh[hl_ & PHOI_T])
            if AUC is not None:
                d = dich_o_day(hp, cv, DV_HANG_G_T[rows][co], BT_T, KC_T, G_LUOI, O_LUOI)
                AUC.them(PHU(patch[co]), torch.cat([d[0], d[1][..., None], d[2]], -1))
            hpx = cxcywh_sang_xyxy(ra['hop_cac_tang'][-1][co]) * SIZE        # 79a: hop tang cuoi (anh giu rieng, hop GOC 518)
            h0x = cxcywh_sang_xyxy(LUONG.hop_tb)[None].expand_as(hpx) * SIZE
            IOU_V.append(iou_cap(hpx[cv], hp[cv])); IOU_0.append(iou_cap(h0x[cv], hp[cv]))
            AUC_V.them(PHU_V.bt_logit(ra['r'][co]), BT_V_T[g[co]])
        if LUOT_78U in ('B1', 'B2'):
            sg = ST_G_T[rows]; m = sg >= 0
            if bool(m.any()): ZA.append(F.normalize(PHU.z_anh(cls[m]), dim=-1)); STA.append(sg[m])
        pos.append(b['pos'])
    assert torch.equal(torch.cat(pos), torch.arange(len(I_GIU_C))), 'thu tu anh giu rieng lech'
    _tb = lambda v: float(torch.cat(v).mean()) if v and sum(x.numel() for x in v) else None
    out = {f'khoi_{k}': _tb(v) for k, v in KH.items()}
    if ZA and int(MA_G_T.numel()):
        za, sta, U = torch.cat(ZA), torch.cat(STA), int(MA_G_T.numel())
        zc = torch.cat([F.normalize((PHU.text(x_cau(MA_G_T[i:i + 1024])) if LUOT_78U == 'B1'
                                     else thap_bert(TOK_T[MA_G_T[i:i + 1024]], None)).float(), dim=-1) for i in range(0, U, 1024)])
        k10 = torch.arange(min(10, U), device=dev, dtype=torch.float64); t1 = t10 = c1 = c10 = 0.0
        for i in range(0, len(za), 256):
            S = za[i:i + 256] @ zc.T
            P = ma_tran_duong(sta[i:i + 256], PTR_G_T, IDX_G_T, MA_G_T); n = P.sum(1).double()
            t1 += float(P.gather(1, S.argmax(1, keepdim=True)).sum())
            t10 += float(P.gather(1, S.topk(min(10, U), dim=1).indices).any(1).sum())
            c1 += float((n / U).sum())
            c10 += float((1 - ((U - n[:, None] - k10) / (U - k10)).clamp_min(0).prod(1)).sum())
        out.update(cau_top1=t1 / len(za), cau_top10=t10 / len(za), cau_nn1=c1 / len(za), cau_nn10=c10 / len(za),
                   cau_n_neo=int(len(za)), cau_n_kho=U)
    _au = np.array(AUC_V.auc(), 'float64')
    out.update(luong_iou=_tb(IOU_V), luong_iou_tb=_tb(IOU_0),
               luong_auc_bt=(float(np.nanmean(_au)) if np.isfinite(_au).any() else None),
               luong_ghi=([float(v) for v in np.mean(GHI_G, 0)] if GHI_G and GHI_G[0] else None))
    if AUC is not None:
        a = np.array(AUC.auc(), 'float64'); _nm = lambda v: float(np.nanmean(v)) if np.isfinite(v).any() else None
        out.update(day_thuoc_vung=_nm(a[:R_VUNG]), day_bat_thuong=_so(a[R_VUNG]), day_ket_cau=[_so(v) for v in a[R_VUNG + 1:]],
                   day_thuoc_vung_kenh=[_so(v) for v in a[:R_VUNG]])
    return out


# Cong DUONG CHAM (batch val THAT): GO HET (None) moi bang / module CHI dung luc train -> anh cua duong cham va du doan val TRUNG TUNG
# BIT (doc bat cu bang nao -> loi -> DUNG). Luat 2 (spec 07/10), nhu 78g C6.
_TEN_PHU = ('HOP518', 'SG_HANG', 'SG_HANG_T', 'POOL', 'POOL_HANG', 'POOL_T', 'M_DP', 'M_KDP', 'M_DP_T', 'M_KDP_T', 'VG', 'VT', 'ANH_F',
            'ST_HANG', 'ST_G_HANG', 'ST_T', 'PTR_T', 'IDX_T', 'ST_G_T', 'PTR_G_T', 'IDX_G_T', 'MA_G_T', 'W_CAU_T', 'SUB_T', 'E_CAU_T',
            'TK_CAU_T', 'TOK_T', 'DV_HANG', 'DV_HANG_G', 'DV_HANG_T', 'DV_HANG_G_T', 'BT_T', 'KC_T', 'BT_NP', 'KC_NP', 'PHU', 'PARAM_PHU',
            'PTR_V_T', 'IDX_V_T', 'W_V_T', 'TOK_VUNG_T', 'E_VUNG_T', 'BT_V_T', 'PHU_V', 'PARAM_LUONG', 'CO_CAU_V', 'HOP_TB',
            'SG_ST', 'SG_ST_T', 'M_S', 'M1_NP', 'MC_NP', 'M1_T', 'MC_T', 'MIL_T', 'ST_ID', 'ST_ID_T', 'PARAM_V2', 'K78F', 'NGUON_NHAN_T', 'NGUON_NHAN_NP', 'NV_T', 'CO_DINH_V3')
_b0 = next(iter(loader(I_VA[:8], eval_tf, False, 8, nw=0)))
_P1 = predict(I_VA[:8], bs=8)
_G0 = {k: globals()[k] for k in _TEN_PHU}
try:
    globals().update(dict.fromkeys(_TEN_PHU))
    _b2 = next(iter(loader(I_VA[:8], eval_tf, False, 8, nw=0)))
    _P2 = predict(I_VA[:8], bs=8)
except Exception as _e:
    raise AssertionError(f'duong CHAM doc bang / module loss phu ({type(_e).__name__}: {_e}) -> DUNG')
finally:
    globals().update(_G0)
assert torch.equal(_b0['image'], _b2['image']) and np.array_equal(_P1, _P2), 'duong CHAM doi khi go bang loss phu -> DUNG'
del _b0, _b2, _P1, _P2, _G0
print('cong duong cham: go het bang / module loss phu -> anh + du doan val TRUNG tung bit')

PARAM_DB = [p for p in HEAD.parameters() if p.requires_grad] + [p for p in bb_ft.parameters() if p.requires_grad] + PARAM_V2   # = P_ARM['ft'] 38a
PARAM_BB = tham_so_bb()


def nhom_phu(sp):
    '''78u: nhom tham so module loss phu, o CUOI danh sach nhom (nhom dau doc / backbone y het B0). B2 (nhu 78t cau_clip): CHI cac
       tang MO cua BiomedBERT (PHU.text.mdl) LR_THAP rieng; dau chieu thap (text.dau), chieu anh, nhiet do (+ B1, B4): LR dau doc
       -> B1 va B2 khac nhau CHI o thap. WD tren tham so ndim > 1 (sp cua 38a); LambdaLR chung (warmup + cosine) cho moi nhom.'''
    if PHU is None: return []
    if LUOT_78U == 'B2':
        return (sp([p for n, p in PHU.named_parameters() if p.requires_grad and not n.startswith('text.mdl.')], LR_HEAD)
                + sp([p for p in PHU.text.mdl.parameters() if p.requires_grad], LR_THAP))
    return sp(PARAM_PHU, LR_HEAD)


def make_opt():
    '''Backbone LR = LR dau doc / 10 (LP-FT). LambdaLR nhan CHUNG mot he so len moi group
       nen ti le 1/10 giu nguyen suot lich.'''
    hp = [p for p in HEAD.parameters() if p.requires_grad] + PARAM_V2   # 78u: MOT dau doc; 78x: + DOCV / MIL (y 1, kenh MIL)
    bp = [p for p in bb_ft.parameters() if p.requires_grad]
    sp = lambda ps, lr: [{'params': [p for p in ps if p.ndim > 1],  'lr': lr, 'weight_decay': WD},
                         {'params': [p for p in ps if p.ndim <= 1], 'lr': lr, 'weight_decay': 0.0}]
    return torch.optim.AdamW(sp(hp, LR_HEAD) + sp(bp, LR_HEAD * LR_MULT_BB) + nhom_phu(sp)   # 78u: nhom loss phu o CUOI
                             + sp(PARAM_LUONG, LR_HEAD))                 # 79a: luong + PHU_V SAU nhom cua nen (thu tu nen giu nguyen)


def bb_38a():
    '''state_dict backbone theo ten CU cua 38a / tep hub (ve_ten_38a; khu hoi da kiem o C2) -> rs_ / backbone_ft.pt nap duoc o CA
       HAI bo cuc ten Dinov2 bang cach nap cua 78g (doi_ten_dinov2) — resume C6 va 78v.'''
    sd = ve_ten_38a(bb_ft.state_dict())
    assert set(sd) == set(KHOA_38A), 'backbone: khoa sau ve_ten_38a KHAC khoa tep hub (38a) -> DUNG'
    return sd


def ten_tham_so_opt():
    '''Ten tham so cua opt theo DUNG thu tu nhom (backbone theo ten CU 38a: ve_ten_38a) -> rs_ ghi, resume so khop. Moment AdamW nap
       theo VI TRI: transformers doi bo cuc / thu tu tham so Dinov2 giua hai phien (nhu 06/10) -> moment gan NHAM tensor cung hinh
       ma khong bao loi -> DUNG thay vi sai lang le.'''
    ten = {id(p): 'bb.' + n for n, p in ve_ten_38a(dict(bb_ft.named_parameters())).items()}
    ten.update({id(p): 'head.' + n for n, p in HEAD.named_parameters()})
    if PHU is not None: ten.update({id(p): 'phu.' + n for n, p in PHU.named_parameters()})
    ten.update({id(p): 'luong.' + n for n, p in LUONG.named_parameters()})
    ten.update({id(p): 'phu_v.' + n for n, p in PHU_V.named_parameters()})
    ten.update({id(p): 'docv.' + n for n, p in DOCV.named_parameters()})
    ten.update({id(p): 'dauv.' + n for n, p in DAUV.named_parameters()})
    ten.update({id(p): 'dnguon.' + n for n, p in DAU_NGUON.named_parameters()})
    ten.update({id(p): 'kcd.' + n for n, p in KENH_CD.named_parameters()})
    for _t_o, _m_o in (('dua_o', DUA_O), ('bo_o7', BO_O7), ('bo_o11', BO_O11), ('gop_o', GOP_O)):   # 78x o cat
        ten.update({id(p): f'{_t_o}.' + n for n, p in _m_o.named_parameters()})
    return [ten[id(p)] for gr in opt.param_groups for p in gr['params']]


def meta_bb(e2, ep):
    '''meta cua backbone_ft.pt (78v / 78a doc): cong thuc, luot, lambda, nguon (id / md5), bo cuc ten khoa bb, val cua epoch cuoi.'''
    return dict(luot=LUOT_78U, luot_79a=LUOT_79A, cfg_79=CFG_79, cfg_jv2=CFG_JV2, cfg_jv3=CFG_JV3, cfg_jv3b=CFG_JV3B, cfg_d4=CFG_D4_CHON, run=RUN_TAG, seed=SEED, thu_nhanh=THU_NHANH, epochs=EPOCHS, ep=ep,
                cong_thuc=dict(nguon='38a a2_E3', batch=BATCH, accum=ACCUM, lr_head=LR_HEAD, lr_mult_bb=LR_MULT_BB, wd=WD,
                               warmup=WARMUP, clip=CLIP, asl=[4, 0], size=SIZE, d_mld=D_MLD, steps=steps, warm=warm,
                               du_lieu='toan bo anh train', n_train=int(len(I_TF)),
                               tang_cuong='TF_HOP (v2: RandomResizedCrop 0.85-1 / 0.95-1.05 bicubic, RandomRotation 7, '
                                          'ClampBoundingBoxes) + eval_tf'),
                phan=PHAN_CFG, lam=LAM, ti_le_tang=TI_LE_TANG, hc_do=HC_DO,
                thap=(dict(model=TXT_MODEL, revision=TXT_REV, thap_mo=THAP_MO, lr=LR_THAP, checkpointing=CKPT_THAP)
                      if LUOT_78U == 'B2' else None),
                dau_doc_37=dict(CK37, duong='/'.join(NGUON_37) + '/resume.pt'), nguon=NGUON_TEP,
                giu_rieng=dict(ti_le=TI_LE_GIU, hat=HAT_GIU, n_benh_nhan=int(len(GIU_RIENG))),
                tf_ver=_TF_VER, ten_khoa_bb='cu (38a / tep hub; ve_ten_38a)', bo_cuc_chay=('moi' if TEN_MOI_BB else 'cu'),
                torch=torch.__version__, hat=dict(chung=SEED, gen_phu=SEED * 1000 + 17, gen_thap=SEED * 1000 + 19, thap=SEED_THAP,
                                                   phu=SEED_PHU, day=SEED_DAY),
                n_buoc=N_STEP, n_clip=N_CLIP, n_bo_buoc=N_SKIP,
                val={k: e2[k] for k in ('mAP_anh', 'mAP_study', 'head', 'medium', 'tail')},
                t=datetime.datetime.now().isoformat(timespec='seconds'))


DL = loader_study(I_TF, train_tf, BATCH)                          # 78x: lo theo study (y B), CUNG hoan vi o hai luot
opt = make_opt()
steps = math.ceil(len(I_TF) / (BATCH * ACCUM)) * EPOCHS; warm = int(WARMUP * steps)
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s/max(warm,1) if s < warm else
                                        0.5*(1+math.cos(math.pi*(s-warm)/max(steps-warm,1))))
N_CLIP = N_STEP = N_SKIP = 0; LOG = {}; ep0 = 0
GEN_PHU = torch.Generator(device=dev); GEN_PHU.manual_seed(SEED * 1000 + 17)    # dong RIENG: rut cau ung vien (B1 / B2)
GEN_THAP = torch.Generator(device=dev); GEN_THAP.manual_seed(SEED * 1000 + 19)  # dong RIENG: CHI dropout thap hoc cung (B2)
GEN_VUNG = torch.Generator(device=dev); GEN_VUNG.manual_seed(SEED * 1000 + 23)  # 79a: dong RIENG rut cau vung
GEN_THAP_VUNG = torch.Generator(device=dev); GEN_THAP_VUNG.manual_seed(SEED * 1000 + 29)  # 79a: dropout thap khi ma hoa cau vung
_EMA_SD = _GDP = None                                             # 78x dot 4 B.4: EMA + generator drop-path cua tep resume
_co = set(ten_tren_drive()) | set(DA_GHI)
if TEN_BB in _co and TEN_LOG(_EP_CHAY) in _co:
    ep0 = _EP_CHAY
    with open(tim_moi_noi(TEN_LOG(_EP_CHAY))) as _f: LOG = json.load(_f)
    print(f'{RUN_TAG}: DA CO {TEN_BB} + nhat ky epoch {_EP_CHAY} tren Drive -> bo qua train (C7 doc nhat ky)')
else:
    # RESUME: rs_ moi nhat (QUA API, moi ban trung ten) -> trong so + AdamW + lich + module loss phu + DONG NGAU NHIEN (chung +
    # GEN_PHU + GEN_THAP)
    for _e in range(_EP_CHAY - 1, 0, -1):
        if not (_RES is not None and int(_RES['ep']) == _e - 1): continue   # 78x dot 4 (ghi mot lan): CHI tep resume (nap o C5)
        _r = _RES                                            # tep resume = rs_ cu (CUNG khoa) + cho
        assert _r.get('run') == RUN_TAG and _r.get('luot') == LUOT_78U and int(_r['ep']) == _e - 1, (
            f'{TEN_RS(_e)} cua luot / epoch KHAC -> DOI RUN_TAG o C0, KHONG xoa gi')
        assert json.loads(json.dumps(_r.get('lam'))) == json.loads(json.dumps(LAM)), (
            f'{TEN_RS(_e)} luu lambda KHAC tep lambda hien tai -> DUNG (tep lambda bi thay giua luot?)')
        assert _r.get('ten_opt') == ten_tham_so_opt(), (f'{TEN_RS(_e)}: ten / thu tu tham so cua optimizer KHAC luc ghi '
                                                       '(transformers doi bo cuc Dinov2?) -> moment AdamW se gan NHAM, DUNG')
        bb_ft.load_state_dict(doi_ten_dinov2(_r['bb'], bb_ft.state_dict()), strict=True)   # NAP_BB_78G: ten CU -> MOI neu can
        HEAD.load_state_dict(_r['head']); nap_phu_hoc(_r['phu'])
        LUONG.load_state_dict(_r['luong'], strict=True); PHU_V.load_state_dict(_r['phu_v'], strict=True)
        DOCV.load_state_dict(_r['docv'], strict=True); DAUV.load_state_dict(_r['dauv'], strict=True)
        DAU_NGUON.load_state_dict(_r['dnguon'], strict=True); KENH_CD.load_state_dict(_r['kcd'], strict=True)
        assert _r.get('cfg_o') == O_CFG_CHON, f"{TEN_RESUME}: cfg_o {_r.get('cfg_o')} KHAC O_CFG -> DUNG"   # 78x o cat
        DUA_O.load_state_dict(_r['dua_o'], strict=True); BO_O7.load_state_dict(_r['bo_o7'], strict=True)
        BO_O11.load_state_dict(_r['bo_o11'], strict=True); GOP_O.load_state_dict(_r['gop_o'], strict=True)
        opt.load_state_dict(_r['opt']); sch.load_state_dict(_r['sch'])
        N_STEP, N_CLIP, N_SKIP, LOG = int(_r['n_step']), int(_r['n_clip']), int(_r['n_skip']), _r['log']
        torch.set_rng_state(_r['rng_cpu'].cpu())
        if dev == 'cuda' and _r.get('rng_cuda') is not None: torch.cuda.set_rng_state(_r['rng_cuda'].cpu())
        GEN_PHU.set_state(_r['gen_phu'].cpu()); GEN_THAP.set_state(_r['gen_thap'].cpu())
        GEN_VUNG.set_state(_r['gen_vung'].cpu()); GEN_THAP_VUNG.set_state(_r['gen_thap_vung'].cpu())
        ep0 = _e
        LOG.setdefault('resume', []).append(ep0)
        print(f'{RUN_TAG}: RESUME tu epoch {ep0 + 1}/{_EP_CHAY} — trong so + moment AdamW + lich ({N_STEP} buoc) + dong ngau nhien')
        _EMA_SD, _GDP = _r.get('ema_sd'), _r.get('gen_dp')   # 78x dot 4 B.4
        del _r; _RES = None; break                          # 78x dot 4: giai phong tep resume (RAM)
assert _RES is None or ep0 == _EP_CHAY, (f'{TEN_RESUME} (xong ep{int(_RES["ep"]) + 1}) KHONG nap duoc vao vong train (thieu '
                                         'backbone_ft.pt / nhat ky?) -> DUNG, bao Claude')   # 78x dot 4: KHONG train lai tu dau
EMA = EmaTrongSo({'bb_ft': bb_ft, 'LUONG': LUONG, 'HEAD': HEAD, 'PHU_V': PHU_V, 'DOCV': DOCV, 'DAUV': DAUV, 'DAU_NGUON': DAU_NGUON,
                  'KENH_CD': KENH_CD, 'DUA_O': DUA_O, 'BO_O7': BO_O7, 'BO_O11': BO_O11, 'GOP_O': GOP_O},
                 float(CFG_D4['dp_ema']['ema']))   # 78x o cat: + 4 module o   # 78x dot 4 B.4: EMA trong so (SAU nap resume)
assert _EMA_SD is not None or ep0 in (0, _EP_CHAY), (f'78x dot 4 B.4: resume tu epoch {ep0} ma tep resume KHONG co EMA / generator drop-path '
                                                     '-> DUNG (EMA khoi dong lai giua chung = luot khac)')
if _EMA_SD is not None:
    EMA.load_state_dict(_EMA_SD); GEN_DP.set_state(_GDP.cpu())
    print(f'78x dot 4 B.4: EMA + generator drop-path NAP tu tep resume ({EMA.n} lan cap nhat)')
del _EMA_SD, _GDP
if ep0 == 0:
    GEN_DP.manual_seed(SEED * 1000 + int(CFG_D4['dp_ema']['hat_dp'])); torch.manual_seed(SEED)       # 78u: dong ngau nhien CHUNG bat dau tu SEED -> B0..B4 CUNG thu tu lo, tang cuong, dropout dau doc
    print(f'{RUN_TAG}: bat dau tu epoch 1/{_EP_CHAY} (dong chung gieo SEED {SEED})')
for _e in range(1, ep0 + 1):      # tep cua epoch da xong CHI con o local (hang cho phien truoc mat) -> tai len lai qua khoi ghi chuan
    for _t in (TEN_P(_e), TEN_P_THO(_e), TEN_LOG(_e), TEN_RS(_e)) + ((TEN_RS_THO(_e),) if _e == EP_BB_D4 else ()) + ((TEN_BB,) if _e == _EP_CHAY else ()):   # 78x dot 4 B.4: + p_tho_ / rs_tho_
        tai_len_lai(_t)

# ---- B.4 v2 (Tuan 22/10): CONG GHEP CAP - dau van 3 lo dau MOI epoch phai TRUNG J v3b s42 (nhat ky cua s42, CHI doc qua API); lech -> DUNG ----
_kk_lo = kiem_ke(tuple(CFG_D4['ghep_cap']), mau=r'epoch_log_J_bert_dong_ep\d+\.json')[1].get(f'epoch_log_J_bert_dong_ep{EPOCHS}.json') or []
assert THU_NHANH or _kk_lo, f"{'/'.join(CFG_D4['ghep_cap'])}: thieu epoch_log_J_bert_dong_ep{EPOCHS}.json -> KHONG kiem duoc ghep cap, DUNG"
LO_S42 = None
if _kk_lo and not THU_NHANH and SEED == 42:                       # nhanh C: ghep cap dau van lo CHI seed 42 (cap J v3b s42)
    with open(tai(_kk_lo[0]), encoding='utf-8') as _f_lo: LO_S42 = {k: (v or {}).get('lo_dau') for k, v in json.load(_f_lo).items()}
    assert all(LO_S42.get(str(e)) for e in range(EPOCHS)), (
        f'nhat ky s42 thieu dau van lo o epoch {[e + 1 for e in range(EPOCHS) if not LO_S42.get(str(e))]} -> DUNG')
    print(f"B.4 v2: cong ghep cap BAT - dau van 3 lo dau ep1-{EPOCHS} cua J v3b s42 ({'/'.join(CFG_D4['ghep_cap'])}, md5 "
          f"{str(_kk_lo[0]['md5'])[:8]})")
del _kk_lo
if LO_S42 is None and not THU_NHANH: print(f'nhanh C s{SEED}: cong ghep cap TAT (khong co cap J v3b s{SEED}; khoi dong luong + lambda van chep J v3b s42)')
for ep in range(ep0, _EP_CHAY):
    bb_ft.train(); HEAD.train(); LUONG.train(); PHU_V.train(); DOCV.train(); DAUV.train()
    if PHU is not None: PHU.train()
    if dev == 'cuda': torch.cuda.reset_peak_memory_stats()
    t0, tot, nb = time.time(), 0.0, 0
    TOT = {q: 0.0 for q in PHAN_CFG}; NP_ = {q: 0 for q in PHAN_CFG}; TL = []
    DUNG, N_DUNG = torch.zeros((), device=dev), torch.zeros((), device=dev); KHOI_T, N_KHOI = 0.0, 0
    N_PHU = PHU_BO = CAT_PHU = BO_CAT_PHU = 0
    LO_DAU = []                                                     # 78x: dau van thu tu lo (ghep cap hai luot, C7)
    DO_O = []                                                       # 78x o cat: ||inj|| / ||h|| (N_DO_TL micro-batch dau)
    CAT_LUONG = BO_CAT_LUONG = 0; GHI_T = []; DV_T, N_DV = torch.zeros((), device=dev), torch.zeros((), device=dev)
    TRAN_TI, TRAN_THO, TRAN_CHAM = (torch.zeros(len(LUONG.tang), device=dev) for _ in range(3))   # 78x J v3b: tran MOI buoc (thiet bi)
    TRAN_N = 0; GL_T = []
    opt.zero_grad(set_to_none=True)
    for k, b in enumerate(tqdm(DL, desc=f'{LUOT_78U} ep{ep + 1}/{EPOCHS}', leave=False)):
        img = b['image'].to(dev, non_blocking=True); y = b['y'].to(dev)
        if k < 3: LO_DAU.append(b['row'].numpy())
        if k == 2 and LO_S42 is not None:                              # B.4 v2: CONG GHEP CAP ngay sau 3 lo dau cua epoch
            assert dau_van_lo(LO_DAU) == LO_S42[str(ep)], (f'B.4 v2 ep{ep + 1}: dau van 3 lo dau {dau_van_lo(LO_DAU)} KHAC J v3b s42 '
                                                            f'{LO_S42[str(ep)]} -> thu tu lo LECH, DUNG (bao Claude)')
        patch, cls, ra = fwd_o(img, True, b['row'].to(dev), do=(DO_O if k < N_DO_TL else None))    # 78x o cat: + 5 o (skip block 7, token o cuoi)
        if k < N_DO_TL and ra['ghi']: GHI_T.append([float(t) for t in ra['ghi']])
        if ra['ghi']:                                                  # 78x J v3b: gom tran MOI buoc tren thiet bi (KHONG dong bo GPU)
            TRAN_TI = torch.maximum(TRAN_TI, torch.stack([t.float() for t in ra['ghi']]))
            TRAN_THO = torch.maximum(TRAN_THO, torch.stack([l._tk[0].float() for l in LUONG.lop]))
            TRAN_CHAM = TRAN_CHAM + torch.stack([l._tk[1].float() for l in LUONG.lop]); TRAN_N += int(img.shape[0])
        lg, _h, _kv = dau_v2_o(patch, ra)                              # 78x o cat: dau doc 6 view (gop o) + kenh MIL; ra['lg'] cho y B
        if not torch.isfinite(lg).all():
            raise RuntimeError(f'ep{ep + 1} buoc {k}: dau doc cho ra logit NaN/inf')
        loss = crit(lg.float(), y); tot += float(loss.detach()); nb += 1
        _hs = he_so_tang(N_STEP, steps, TI_LE_TANG)                    # h(t) theo SO BUOC toi uu (dung ca sau resume)
        if PHAN_CFG and _hs > 0:                                       # LOSS PHU (CHI luc train), lambda chot o C5
            o = phu_79(patch, cls, _kv, b['hop'].to(dev, non_blocking=True), b['co_hop'].to(dev, non_blocking=True),
                       b['row'].to(dev), y, GEN_PHU, GEN_THAP, ra, GEN_VUNG, GEN_THAP_VUNG)
            if o is not None:
                lp = sum(LAM[q] * o[q] for q in PHAN_CFG if q in o)
                if torch.isfinite(lp):
                    if k < N_DO_TL:    # chan doan: ||grad_bb lambda L_phu|| / ||grad_bb L_nhan|| (KHONG nhan h(t))
                        TL.append(chuan_grad(lp, PARAM_BB) / max(chuan_grad(loss, PARAM_BB), 1e-12))
                    loss = loss + _hs * lp; N_PHU += 1
                    for q in PHAN_CFG:
                        if q in o: TOT[q] += float(o[q].detach()); NP_[q] += 1
                    if 'dung1' in o:                                   # khong neo -> nan: CHI cong khi huu han, dem RIENG
                        _fd = torch.isfinite(o['dung1']); DUNG += torch.where(_fd, o['dung1'], 0.0); N_DUNG += _fd
                    if 'khoi' in o: KHOI_T += o['khoi']; N_KHOI += 1
                    if 'dung1_vung' in o:                              # 79a: dem nhu dung1 (chi khi huu han)
                        _fv = torch.isfinite(o['dung1_vung']); DV_T += torch.where(_fv, o['dung1_vung'], 0.0); N_DV += _fv
                else:
                    PHU_BO += 1                                        # loss phu khong huu han -> BO phan phu, DEM
        (loss / ACCUM).backward(); ra.pop('o11', None)                # 78x o cat: tra token o cuoi (bo nho) truoc lo sau
        if (k + 1) % ACCUM == 0:
            gn = torch.nn.utils.clip_grad_norm_(PARAM_DB, CLIP)       # 38a: clip dau doc + backbone CHUNG
            if not torch.isfinite(gn):                                 # 78u: gradient khong huu han -> BO grad buoc nay
                for p in PARAM_DB: p.grad = None
                N_SKIP += 1
            elif float(gn) > CLIP: N_CLIP += 1
            if PARAM_PHU:                                              # module loss phu: cat RIENG (nhu 78g)
                gp = torch.nn.utils.clip_grad_norm_(PARAM_PHU, CLIP)
                if not torch.isfinite(gp):
                    for p in PARAM_PHU: p.grad = None
                    BO_CAT_PHU += 1
                elif float(gp) > CLIP: CAT_PHU += 1
            if PARAM_LUONG:                                            # 79a: luong + PHU_V cat RIENG (nhu PARAM_PHU)
                gl = torch.nn.utils.clip_grad_norm_(PARAM_LUONG, CLIP)
                if not torch.isfinite(gl):
                    for p in PARAM_LUONG: p.grad = None
                    BO_CAT_LUONG += 1
                elif float(gl) > CLIP: CAT_LUONG += 1
                if torch.isfinite(gl): GL_T.append(float(gl))           # 78x J v3b: chuan gradient luong MOI buoc (truoc cat)
            N_STEP += 1
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            EMA.cap_nhat()                                     # 78x dot 4 B.4: EMA sau MOI buoc toi uu
            if THU_NHANH and dev == 'cuda':
                print(f'   buoc {N_STEP}: VRAM dinh {torch.cuda.max_memory_allocated() / 1e9:.1f} / '
                      f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB | {nb * BATCH / max(time.time() - t0, 1e-9):.0f} '
                      'img/s', flush=True)
    _t_train = time.time() - t0
    _vr = (torch.cuda.max_memory_allocated() / 1e9) if dev == 'cuda' else 0.0
    with _fork(): PV_THO = predict(I_VA)                              # B.4 v2 (22/10): du doan THO TRONG fork -> dong ngau nhien chung NHU s42 (lan 1: ngoai fork -> lech thu tu lo tu ep2)
    ghi(TEN_P_THO(ep + 1), lambda q, v=PV_THO: np.save(q, v.astype('float16')))
    _ai_tho, _as_tho = score(PV_THO)
    with EMA.ap():                                                     # 78x dot 4 B.4: du doan CHINH + chan doan giu rieng = trong so EMA
        PV = predict(I_VA)
        ghi(TEN_P(ep + 1), lambda q, v=PV: np.save(q, v.astype('float16')))
        _ai, _as = score(PV)
        with _fork(): CD = chan_doan_giu()                             # fork: DataLoader moi KHONG tieu dong chung
    _e2 = dict(mAP_anh=float(np.nanmean(_ai)), mAP_study=float(np.nanmean(_as)), **tiers(_ai), ap_anh=[_so(v) for v in _ai],
               loss_nhan=tot / max(nb, 1), n_buoc=N_STEP, clip_pct=N_CLIP / max(N_STEP, 1), n_bo_buoc=N_SKIP,
               img_s=nb * BATCH / max(_t_train, 1e-9), phut=_t_train / 60, vram_gb=_vr, he_so=he_so_tang(N_STEP, steps, TI_LE_TANG),
               lam=LAM, n_phu=N_PHU, phu_bo=PHU_BO, cat_phu=CAT_PHU, bo_cat_phu=BO_CAT_PHU,
               ti_le_grad=(float(np.mean(TL)) if TL else None), **{f'loss_{q}': (TOT[q] / NP_[q] if NP_[q] else None) for q in PHAN_CFG},
               dung1=(float(DUNG / N_DUNG) if float(N_DUNG) > 0 else None), khoi=(KHOI_T / N_KHOI if N_KHOI else None),
               tau=(float(PHU.tau().detach()) if LUOT_78U in ('B1', 'B2') else None), giu=CD)
    _e2['luong'] = dict(ghi_tang=([float(v) for v in np.mean(GHI_T, 0)] if GHI_T else None),
                        dung1_vung=(float(DV_T / N_DV) if float(N_DV) > 0 else None), cat=CAT_LUONG, bo_cat=BO_CAT_LUONG)
    _gl = np.asarray(GL_T, 'float64')                                  # 78x J v3b: tran write-back + gradient luong cua CA epoch
    _e2['luong']['tran'] = dict(rho=float(CFG_JV3B['tran_ghi_rho']), ti_max=[float(v) for v in TRAN_TI.cpu()],
                                tho_max=[float(v) for v in TRAN_THO.cpu()], n_anh=TRAN_N,
                                ti_le_cham=[float(v) / max(TRAN_N, 1) for v in TRAN_CHAM.cpu()],
                                gl_p50=(float(np.percentile(_gl, 50)) if len(_gl) else None),
                                gl_p99=(float(np.percentile(_gl, 99)) if len(_gl) else None),
                                gl_max=(float(_gl.max()) if len(_gl) else None), n_gl=int(len(_gl)))
    _e2['lo_dau'] = dau_van_lo(LO_DAU)
    _e2['ema'] = dict(n=EMA.n, he_so=EMA.he_so(), decay=EMA.decay, mAP_anh_tho=float(np.nanmean(_ai_tho)),   # 78x dot 4 B.4
                      mAP_study_tho=float(np.nanmean(_as_tho)), **{'tho_' + k: v for k, v in tiers(_ai_tho).items()})
    _e2['dp'] = dict(ti_le=TI_LE_DP)
    _e2['o'] = thong_ke_o(GOP_O, BO_O7, BO_O11, DUA_O, DO_O)            # 78x o cat: chan doan duong o (so tong hop)
    _e2['jv2'] = dict(mil_w_tb=float(DAUV.w.detach().abs().mean()), mil_w_max=float(DAUV.w.detach().abs().max()),
                      kcd_w_tb=float(KENH_CD.w.detach().abs().mean()), kcd_w_max=float(KENH_CD.w.detach().abs().max()),
                      nguon_w_tb=float(DAU_NGUON.w.detach().abs().mean()), ghi_nguoc_g=float(CFG_JV3['ghi_nguoc_g']))
    LOG[str(ep)] = _e2
    ghi(TEN_LOG(ep + 1), lambda q: ghi_json(q, LOG))
    _rs = dict(                  # 78x dot 4 (ghi mot lan): = rs_ cu (CUNG khoa) MOI epoch + cau hinh dot 4 -> tep resume + rs_ NHE
        bb=bb_38a(), head=HEAD.state_dict(), phu=phu_hoc(), opt=opt.state_dict(), sch=sch.state_dict(),
        luong=LUONG.state_dict(), phu_v=PHU_V.state_dict(), docv=DOCV.state_dict(), dauv=DAUV.state_dict(),
        dnguon=DAU_NGUON.state_dict(), kcd=KENH_CD.state_dict(),
        gen_vung=GEN_VUNG.get_state(),
        gen_thap_vung=GEN_THAP_VUNG.get_state(),
        n_step=N_STEP, n_clip=N_CLIP, n_skip=N_SKIP, log=LOG, lam=LAM, ep=ep, run=RUN_TAG, luot=LUOT_78U,
        rng_cpu=torch.get_rng_state(), rng_cuda=(torch.cuda.get_rng_state() if dev == 'cuda' else None),
        gen_phu=GEN_PHU.get_state(), gen_thap=GEN_THAP.get_state(), tf_ver=_TF_VER, ten_moi_bb=TEN_MOI_BB,
        ten_opt=ten_tham_so_opt(), cfg_d4=CFG_D4_CHON, cfg_o=O_CFG_CHON, dua_o=DUA_O.state_dict(), bo_o7=BO_O7.state_dict(),
        bo_o11=BO_O11.state_dict(), gop_o=GOP_O.state_dict())   # 78x o cat: + module o
    _rs.update(ema_sd=EMA.state_dict(), gen_dp=GEN_DP.get_state())    # 78x dot 4 B.4: tep resume mang EMA + generator drop-path
    _rs_nhe = {k: v for k, v in _rs.items() if k not in ('opt', 'sch', 'ema_sd', 'gen_dp')}
    _rs_nhe.update(bb=ve_ten_38a(EMA.sd_module('bb_ft')), head=EMA.sd_module('HEAD'), luong=EMA.sd_module('LUONG'), phu_v=EMA.sd_module('PHU_V'),
                   docv=EMA.sd_module('DOCV'), dauv=EMA.sd_module('DAUV'), dnguon=EMA.sd_module('DAU_NGUON'), kcd=EMA.sd_module('KENH_CD'),
                   dua_o=EMA.sd_module('DUA_O'), bo_o7=EMA.sd_module('BO_O7'), bo_o11=EMA.sd_module('BO_O11'),   # 78x o cat
                   gop_o=EMA.sd_module('GOP_O'), ema=dict(decay=EMA.decay, n=EMA.n, he_so=EMA.he_so(), trong_so='EMA'))
    assert set(_rs_nhe['bb']) == set(KHOA_38A), 'rs_ EMA: khoa backbone KHAC khoa 38a -> DUNG'
    ghi(TEN_RS(ep + 1), lambda q: torch.save(_rs_nhe, q))             # rs_ NHE MOI epoch = trong so EMA (78z: ep4; tai len o luu_resume)
    if ep + 1 == EP_BB_D4:                                             # trong so THO ep4: quy nguyen nhan drop-path / EMA neu can
        ghi(TEN_RS_THO(ep + 1), lambda q: torch.save({k: v for k, v in _rs.items() if k not in ('opt', 'sch', 'ema_sd', 'gen_dp')}, q))
    del _rs_nhe
    if ep + 1 == _EP_CHAY:       # backbone cho 78v: khoa nhu 38a (bb, head, run, E, ep) + luot / tf_ver / meta
        with EMA.ap(): ghi(TEN_BB, lambda q: torch.save(dict(bb=bb_38a(), head=HEAD.state_dict(), run=RUN_TAG, luot=LUOT_78U, E=EPOCHS,
                                              luot_79a=LUOT_79A, luong=LUONG.state_dict(), phu_v=PHU_V.state_dict(),
                                              docv=DOCV.state_dict(), dauv=DAUV.state_dict(), cfg_jv2=CFG_JV2,
                                              dnguon=DAU_NGUON.state_dict(), kcd=KENH_CD.state_dict(), cfg_jv3=CFG_JV3,
                                              cfg_jv3b=CFG_JV3B, cfg_d4=CFG_D4_CHON, cfg_o=O_CFG_CHON, dua_o=DUA_O.state_dict(),
                                              bo_o7=BO_O7.state_dict(), bo_o11=BO_O11.state_dict(), gop_o=GOP_O.state_dict(),   # 78x o cat
                                              phu=phu_hoc(), cfg_luong=dict(CFG_79, luot_chon=LUOT_79A),
                                              hop_tb=LUONG.hop_tb.detach().cpu(),
                                              ep=ep + 1, tf_ver=_TF_VER, meta=meta_bb(_e2, ep + 1)), q))
    luu_resume(lambda q: torch.save(dict(_rs, cho=cho_nho()), q))   # 78x dot 4 (ghi mot lan): MOT tep resume GHI DE (+ tep lon dang cho len Drive)
    del _rs
    _fm = lambda v, w=4: '-' if v is None else f'{v:.{w}f}'
    print(f"{LUOT_78U} ep{ep + 1}/{_EP_CHAY} | loss nhan {_e2['loss_nhan']:.4f} | VAL mAP anh {_e2['mAP_anh']:.4f} (study "
          f"{_e2['mAP_study']:.4f}) head {_e2['head']:.4f} medium {_e2['medium']:.4f} tail {_e2['tail']:.4f} | {_e2['img_s']:.0f} img/s"
          f", {_e2['phut']:.1f} ph | VRAM dinh {_vr:.1f} GB | clip {_e2['clip_pct']:.0%} | bo buoc {N_SKIP}")
    if PHAN_CFG:
        print(f"   PHU (h {_e2['he_so']:.2f}) " + ' | '.join(f"{q} {_fm(_e2['loss_' + q])} (lambda {LAM[q]:.3g})" for q in PHAN_CFG)
              + f" | g {_fm(_e2['ti_le_grad'], 2)} | dung1 {_fm(_e2['dung1'], 3)} | khoi {_fm(_e2['khoi'], 3)} | tau "
              f"{_fm(_e2['tau'], 3)} | bo phu {PHU_BO} | cat phu {CAT_PHU} / bo {BO_CAT_PHU}")
    print('   GIU RIENG: khoi attention trong vung du phong / khong (5 nhan phoi) '
          f"{_fm(CD['khoi_dp'], 3)} / {_fm(CD['khoi_kdp'], 3)} ({_fm(CD['khoi_dp_phoi'], 3)} / {_fm(CD['khoi_kdp_phoi'], 3)})"
          + (f" | cau top1 {_fm(CD.get('cau_top1'), 3)} (nn {_fm(CD.get('cau_nn1'), 3)}) top10 {_fm(CD.get('cau_top10'), 3)} (nn "
             f"{_fm(CD.get('cau_nn10'), 3)})" if 'cau_top1' in CD else '')
          + (f" | AUC day thuoc vung {_fm(CD.get('day_thuoc_vung'), 3)} bat thuong {_fm(CD.get('day_bat_thuong'), 3)} ket cau "
             + ('(TAT, kenh khong hoc) ' if 'ket_cau' in PHAN_TAT else '')
             + ' '.join(_fm(v, 3) for v in CD.get('day_ket_cau', [])) if 'day_thuoc_vung' in CD else ''))
    _lv = _e2['luong']
    print(f"   LUONG: ghi nguoc ||dh||/||h|| tang {LUONG.tang}: {' '.join(_fm(v, 4) for v in (_lv['ghi_tang'] or []))} | dung1 vung "
          f"{_fm(_lv['dung1_vung'], 3)} | cat {_lv['cat']} / bo {_lv['bo_cat']} | GIU RIENG: IoU hop {_fm(CD.get('luong_iou'), 3)} "
          f"(hop TB {_fm(CD.get('luong_iou_tb'), 3)}) | AUC bat thuong {_fm(CD.get('luong_auc_bt'), 3)}")
    _tr = _lv['tran']
    print(f"   TRAN write-back (rho {_tr['rho']:g}, theo anh, MOI buoc): ti lo max {' '.join(_fm(v, 4) for v in _tr['ti_max'])} | tho anh "
          f"max {' '.join(_fm(v, 3) for v in _tr['tho_max'])} | anh cham tran {' '.join(f'{v:.1%}' for v in _tr['ti_le_cham'])} | grad "
          f"luong p50 / p99 / max {_fm(_tr['gl_p50'], 3)} / {_fm(_tr['gl_p99'], 3)} / {_fm(_tr['gl_max'], 3)} (cat > {CLIP:g}: {_lv['cat']})")
    print(f"   B.4: VAL mAP anh EMA {_e2['mAP_anh']:.4f} | tho {_e2['ema']['mAP_anh_tho']:.4f} (study EMA {_e2['mAP_study']:.4f} / tho "
          f"{_e2['ema']['mAP_study_tho']:.4f}) | EMA {EMA.n} lan cap nhat (he so {EMA.he_so():.4f}) | drop-path max {max(TI_LE_DP):.2f}")
    _o6 = _e2['o']                                                     # 78x o cat
    print(f"   O CAT: ||inj|| / ||h|| (block 7, {len(DO_O)} lo dau) {_fm(_o6['ti_le_inj'], 4)} | ||ra|| adapter block 7 / cuoi {_o6['bo7']:.3e} / "
          f"{_o6['bo11']:.3e} | ||chieu skip|| {_o6['dua']:.3e} | gop 6 view: ||w|| {_o6['gop_w']:.3e}, |b| theo view "
          + ' '.join(f'{v:.3f}' for v in _o6['gop_b_view']))
    if LUAT_THU_S44 and SEED == 44 and ep + 1 in (1, 2):                                # 78x J v3b: LUAT THU seed 44 (dat truoc, Event 2026-10-16)
        _ng = float(CFG_JV3B['luat_thu_s44'][f'ep{ep + 1}'])
        print(f"   >>> LUAT THU seed 44 ep{ep + 1}: VAL {_e2['mAP_anh']:.4f} {'>=' if _e2['mAP_anh'] >= _ng else '<'} {_ng} -> "
              + ('DAT' if _e2['mAP_anh'] >= _ng else 'KHONG DAT -> J v3b TRUOT luat thu: NGAT luot, bao Claude (J dung han)')
              + (' (can CA ep1 va ep2 dat)' if ep == 0 else ''))
    print(f"   J v3: MIL |w| TB {_e2['jv2']['mil_w_tb']:.4f} / max {_e2['jv2']['mil_w_max']:.4f} | kenh co dinh |w| TB "
          f"{_e2['jv2']['kcd_w_tb']:.4f} / max {_e2['jv2']['kcd_w_max']:.4f} | dau nguon |w| TB {_e2['jv2']['nguon_w_tb']:.2e} | "
          f"write-back x{_e2['jv2']['ghi_nguoc_g']:g} | dau van lo {_e2['lo_dau']}")
    print('   ^ VAL: chon epoch theo luat hai nua o C7 (early stopping kieu a); do QUYET DINH = frozen reading epoch da chon')
dong_bo()                                    # bao NGAY tep nao chua len Drive (nhu 78g)
print(f'{RUN_TAG}: xong')
