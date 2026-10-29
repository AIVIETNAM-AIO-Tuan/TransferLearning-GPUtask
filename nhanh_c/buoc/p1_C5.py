# ==== nhanh_c buoc p1_C5 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C5 · HIEU CHINH LAMBDA tren tham so BACKBONE (lan dau; phien sau / resume DOC LAI he_so_lambda.json) ====
# lambda[phan] = r x TB ||grad_bb L_nhan|| / TB ||grad_bb L_phan||, do tren HC_DO micro-batch voi mo hinh KHOI DAU (dau doc 37 DA hoc
# -> HC_BUOC 0: KHONG buoc toi uu nao), trong fork_rng + DataLoader RIENG + generator RIENG -> dong chung / trong so KHONG doi. Chot
# TRUOC train; h(t) tang 0 -> 1 trong 10% so buoc dau (nhu 78g).
assert _KHOA_C4 == (THU_NHANH, RUN_TAG, LUOT_78U, CKPT_THAP), 'C4 chay voi cau hinh C0 KHAC -> chay lai C0 -> ... -> C4'
assert globals().get('_KHOA_C4B') == _KHOA_C4, 'chay C4a -> C4b (khoi dong luong) truoc C5'
HC_DO = 3 if THU_NHANH else 30                        # micro-batch do (HC_BUOC = 0)
TI_LE_TANG = 0.0 if THU_NHANH else 0.1                # h(t): 0 -> 1 trong 10% so buoc toi uu dau
N_DO_TL = 3                                           # moi epoch: ||grad_bb lambda L_phu|| / ||grad_bb L_nhan|| tren 3 micro-batch dau
TEP_LAM = 'he_so_lambda.json'
CAU_HINH_LAM = json.loads(json.dumps(dict(run=RUN_TAG, luot=LUOT_78U, phan=PHAN_CFG, batch=BATCH, accum=ACCUM, hc_do=HC_DO,
                                          ti_le_tang=TI_LE_TANG, toi_da_cau=TOI_DA_CAU, thap_mo=THAP_MO,
                                          dich=DICH_78S['phien_ban'], tham_so='backbone', luot_79a=LUOT_79A, luong=CFG_79['phien_ban'], jca_v2=CFG_JV2['phien_ban'], lo='study', nhan_chuan='duong_B1', jca_v3=CFG_JV3['phien_ban'], he_so_lam=HE_SO_LAM)))


def tham_so_bb():
    return [p for p in bb_ft.parameters() if p.requires_grad]


assert not ({id(p) for p in tham_so_bb()} & ({id(p) for p in PARAM_LUONG + PARAM_V2} | {id(p) for p in PHU.parameters()})), \
    'tap tham so do lambda (backbone) chua tham so luong / thap -> DUNG (bai hoc lambda 78t)'


def hieu_chinh_lambda():
    '''-> dict(CAU_HINH_LAM, g = thong ke ||grad_bb|| moi phan, lam). Micro-batch co gradient khong huu han bi BO.'''
    t0 = time.time(); G = {k: [] for k in ['nhan', 'nhan_v2'] + [q for q in PHAN_CFG if q not in PHAN_CO_DINH]}; sh = tham_so_bb()   # 78x: nhan = DUONG B1
    with _fork():
        torch.manual_seed(SEED + 500)
        gp = torch.Generator(device=dev); gp.manual_seed(SEED + 501)      # rut cau (nhu GEN_PHU)
        gt = torch.Generator(device=dev); gt.manual_seed(SEED + 502)      # dropout thap B2 (nhu GEN_THAP)
        gv = torch.Generator(device=dev); gv.manual_seed(SEED + 503)      # 79a: rut cau vung (nhu GEN_VUNG)
        gtv = torch.Generator(device=dev); gtv.manual_seed(SEED + 504)    # 79a: dropout thap khi ma hoa cau vung
        dl_ = loader_study(I_TF, train_tf, BATCH); it = iter(dl_)          # 78x: lo theo study (CUNG o hai luot)
        bb_ft.train(); HEAD.train(); LUONG.train(); PHU_V.train(); DOCV.train(); DAUV.train()
        if PHU is not None: PHU.train()
        for _s in range(HC_DO):
            try: b = next(it)
            except StopIteration: it = iter(dl_); b = next(it)
            y = b['y'].to(dev)
            patch, cls, ra = fwd_luong_v2(b['image'].to(dev, non_blocking=True), True, b['row'].to(dev))
            lg0 = dau_h(HEAD, patch)[0]                    # 78x: DUONG B1 (KHONG token vung / MIL) -> chuan hoa lambda (Tuan 12/10, A)
            with _fork(): lg, _h, kv = dau_v2(patch, ra)   # 78x: dropout dau doc J v2 trong fork -> dong chung TRUNG doi chung
            o = phu_79(patch, cls, kv, b['hop'].to(dev), b['co_hop'].to(dev), b['row'].to(dev), y, gp, gt, ra, gv, gtv)
            if o is not None:
                _g = dict(nhan=chuan_grad(crit(lg0.float(), y), sh), nhan_v2=chuan_grad(crit(lg.float(), y), sh))
                for k in PHAN_CFG:
                    if k in o and k not in PHAN_CO_DINH: _g[k] = chuan_grad(o[k], sh)   # J v3: CE nguon khong toi backbone
                if all(np.isfinite(v) for v in _g.values()):
                    for k, v in _g.items(): G[k].append(v)
            del patch, cls, lg, lg0, _h, kv, o, ra
        del it, dl_
    bb_ft.eval(); HEAD.eval(); LUONG.eval(); PHU_V.eval(); DOCV.eval(); DAUV.eval()
    if PHU is not None: PHU.eval()
    if dev == 'cuda': torch.cuda.empty_cache()
    _thieu = [k for k in PHAN_CFG if k not in PHAN_CO_DINH and not G[k]]
    assert not _thieu, f'hieu chinh: khong micro-batch nao co phan loss phu {_thieu} -> DUNG'
    tk = lambda v: dict(tb=float(np.mean(v)), trung_vi=float(np.median(v)), p10=float(np.percentile(v, 10)),
                        p90=float(np.percentile(v, 90)), n=len(v))
    lam = {k: float(r * np.mean(G['nhan']) / np.mean(G[k])) for k, r in PHAN_CFG.items() if k not in PHAN_CO_DINH}
    lam.update({k: float(v) for k, v in PHAN_CO_DINH.items()})   # J v3: lambda CO DINH (CE nguon, cat gradient ve backbone)
    return dict(CAU_HINH_LAM, seed_hieu_chinh=SEED, g={k: tk(v) for k, v in G.items() if v}, lam=lam,
                giay=round(time.time() - t0, 1), gpu=(torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'),
                t=datetime.datetime.now().isoformat(timespec='seconds'))


# ---- 78x J v3b dot 4 (Tuan duyet 21/10; ghi mot lan): (a) MOT tep resume -> nap TRUOC khi tim tep lambda; (b) lambda CHEP tu J v3b s42
#      (ghep cap; KHONG hieu chinh lai) ----
assert globals().get('GHI_CUOI') is True and 'luu_resume' in globals() and 'khoi_phuc_cho' in globals(), (
    '78x dot 4: can khoi_ghi_cuoi o C0 -> DUNG')
_RES, _lp_res = None, tim_moi_noi(TEN_RESUME)
if _lp_res:
    _RES = torch.load(_lp_res, map_location='cpu', weights_only=False)
    assert _RES.get('run') == RUN_TAG and _RES.get('luot') == LUOT_78U and (_RES.get('cfg_d4') or {}).get('ten') == BIEN_THE_D4, (
        f"{TEN_RESUME}: luot / bien the KHAC ({_RES.get('run')}, {(_RES.get('cfg_d4') or {}).get('ten')}) -> DUNG (doi RUN_TAG, KHONG xoa gi)")
    khoi_phuc_cho(_RES['cho'])
    print(f"78x dot 4: RESUME tu {TEN_RESUME}: xong epoch {int(_RES['ep']) + 1}")
if not THU_NHANH and PHAN_CFG and not tim_moi_noi(TEP_LAM):
    _kk_l = kiem_ke(tuple(CFG_D4['ghep_cap']), mau=r'he_so_lambda\.json')[1].get(TEP_LAM) or []
    assert _kk_l, f"{'/'.join(CFG_D4['ghep_cap'])}: KHONG co {TEP_LAM} -> DUNG (ghep cap can lambda cua s42; bao Claude)"
    with open(tai(_kk_l[0]), encoding='utf-8') as _fl: _lj0 = json.load(_fl)
    assert _lj0.get('run') == CFG_D4['ghep_cap'][1] and isinstance(_lj0.get('lam'), dict), (
        f"{TEP_LAM} cua {'/'.join(CFG_D4['ghep_cap'])}: run {_lj0.get('run')} -> DUNG")
    _lam_d4, _lk_d4 = dict(_lj0['lam']), None
    _lj_d4 = dict(CAU_HINH_LAM, g=_lj0.get('g'), lam=_lam_d4, nguon_lam=CFG_D4['ghep_cap'][1], md5_nguon=_kk_l[0]['md5'], lam_khe=_lk_d4,
                  t=datetime.datetime.now().isoformat(timespec='seconds'))
    ghi(TEP_LAM, lambda q: ghi_json(q, _lj_d4))
    print(f"78x dot 4: lambda CHEP tu {'/'.join(CFG_D4['ghep_cap'])} (md5 {str(_kk_l[0]['md5'])[:8]})"
          + ' — KHONG hieu chinh lai')
    del _lj0, _kk_l, _lam_d4, _lj_d4, _lk_d4
LAM_J = None
if not PHAN_CFG:
    print('B0: KHONG loss phu -> khong hieu chinh lambda')
else:
    _lp_lam = tim_moi_noi(TEP_LAM)
    if _lp_lam:
        with open(_lp_lam) as _fl: LAM_J = json.load(_fl)
        tai_len_lai(TEP_LAM)                         # tep CHI con o local (hang cho phien truoc mat) -> tai len lai
        print(f"lambda: DOC LAI {TEP_LAM} ({LAM_J['t']}) — KHONG tinh lai")
    else:
        _da = sorted(x for x in ten_tren_drive() if re.fullmatch(rf'rs_{LUOT_79A}_ep\d+\.pt|backbone_ft\.pt', x))
        if _da:
            # tep lambda MAT nhung DA co checkpoint -> KHOI PHUC lambda tu checkpoint moi nhat (luu `lam`), KHONG tinh lai giua luot
            _ckl = 'backbone_ft.pt' if 'backbone_ft.pt' in _da else max(_da, key=lambda x: int(re.search(r'_ep(\d+)\.pt$', x).group(1)))
            try: _cd = torch.load(tim_moi_noi(_ckl), map_location='cpu', weights_only=False)
            except Exception as _e:
                raise AssertionError(f'THIEU {TEP_LAM} va KHONG doc duoc {_ckl} ({type(_e).__name__}) -> KHONG khoi phuc duoc '
                                     'lambda. DUNG.')
            _lam_ck = (_cd.get('lam') if 'lam' in _cd else (_cd.get('meta') or {}).get('lam')) if isinstance(_cd, dict) else None
            assert isinstance(_cd, dict) and _cd.get('run') == RUN_TAG and isinstance(_lam_ck, dict), (
                f'{_ckl}: khong co lambda cua luot {RUN_TAG} -> KHONG khoi phuc duoc. DUNG.')
            LAM_J = dict(CAU_HINH_LAM, g=None, khoi_phuc_tu=_ckl, lam={k: float(v) for k, v in _lam_ck.items()},
                         t=datetime.datetime.now().isoformat(timespec='seconds'))
            del _cd
            print(f'!! THIEU {TEP_LAM} tren Drive nhung co {_ckl} -> KHOI PHUC lambda (khong tinh lai) va ghi lai tep')
        else:
            print(f'hieu chinh lambda: {HC_DO} micro-batch, tham so backbone (mo hinh khoi dau, khong buoc toi uu) ...', flush=True)
            LAM_J = hieu_chinh_lambda()
        assert set(LAM_J['lam']) == set(PHAN_CFG) and all(np.isfinite(v) and v > 0 for v in LAM_J['lam'].values()), (
            f"lambda HONG {LAM_J['lam']} -> KHONG ghi tep, DUNG (chay lai C5)")
        ghi(TEP_LAM, lambda q: ghi_json(q, LAM_J))
    _lech = {k: (LAM_J.get(k), v) for k, v in CAU_HINH_LAM.items() if LAM_J.get(k) != v}
    assert not _lech, f'{TEP_LAM} thuoc cau hinh KHAC: {_lech} -> DUNG (RUN_TAG / luot / phan / batch / h(t) / so micro-batch do)'
LAM = {k: float(LAM_J['lam'][k]) for k in PHAN_CFG} if PHAN_CFG else {}
assert all(np.isfinite(v) and v > 0 for v in LAM.values()), LAM
if LAM:
    print('lambda = r x TB ||grad_bb nhan|| / TB ||grad_bb phan||' + ('' if LAM_J.get('g') else f" (KHOI PHUC tu {LAM_J.get('khoi_phuc_tu')})")
          + ': ' + ' | '.join((f"{k}: r {r:g} x {LAM_J['g']['nhan']['tb']:.3g} / {LAM_J['g'][k]['tb']:.3g} = {LAM[k]:.4g}"
                               if LAM_J.get('g') and k in LAM_J['g'] else f'{k}: {LAM[k]:.4g} (co dinh)') for k, r in PHAN_CFG.items()))
# ---- 78x J v3b dot 4: lambda DANG DUNG = lambda cua J v3b s42 TUNG SO; THU_NHANH: hieu chinh
#      nhu thuong ----
if not THU_NHANH and LAM:
    _kk_l2 = kiem_ke(tuple(CFG_D4['ghep_cap']), mau=r'he_so_lambda\.json')[1].get(TEP_LAM) or []
    assert _kk_l2, f"{'/'.join(CFG_D4['ghep_cap'])}: KHONG co {TEP_LAM} -> DUNG"
    with open(tai(_kk_l2[0]), encoding='utf-8') as _fl: _lam_s42 = json.load(_fl)['lam']
    _lech_d4 = {k: (LAM.get(k), _lam_s42.get(k)) for k in set(LAM) | set(_lam_s42) if LAM.get(k) != _lam_s42.get(k)}
    assert not _lech_d4, f'lambda dang dung KHAC lambda cua J v3b s42 ({_lech_d4}) -> DUNG (ghep cap hong)'
    print(f"lambda = lambda cua J v3b s42 TUNG SO ({len(LAM)} phan) -> ghep cap")
    del _kk_l2, _lam_s42, _lech_d4
_KHOA_C5 = (THU_NHANH, RUN_TAG, LUOT_78U, CKPT_THAP)  # C6 kiem
