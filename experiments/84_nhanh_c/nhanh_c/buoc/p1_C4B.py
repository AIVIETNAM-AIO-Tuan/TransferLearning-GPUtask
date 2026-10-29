# ==== nhanh_c buoc p1_C4B — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C4b · KHOI DONG LUONG VUNG (79a) — backbone + thap van ban DONG CUNG (dung trang thai khoi tao cua nen B2 / B1), duong ghi nguoc
#      DONG; CHI LUONG + PHU_V hoc. Chay trong fork_rng + DataLoader RIENG -> dong ngau nhien CHUNG (thu tu lo / tang cuong C6) KHONG doi ====
assert _KHOA_C4 == (THU_NHANH, RUN_TAG, LUOT_78U, CKPT_THAP), 'C4 chay voi cau hinh C0 KHAC -> chay lai C0 -> ... -> C4'
assert '_NHIEU_79A' in globals(), 'chay C4a (loc nhieu worker) truoc C4b'
TEN_KD = 'luong_khoi_dong.pt'
_KD = CFG_79['khoi_dong']
PHAN_KD = [q for q in PHAN_CFG if q not in ('cau', 'nhat_quan', 'nguon')]   # 78x: y B / y C can dau doc -> KHONG o khoi dong
N_MOC = 5
assert PHAN_KD, f'{LUOT_79A}: khong co phan luong nao de khoi dong'


def khoi_dong():
    '''-> (lich su: TB loss tung phan o N_MOC doan deu, so buoc, so buoc bo, giay). AdamW (LR _KD['lr'], warmup 5% roi cosine) tren
       PARAM_LUONG; tong loss = tong cac phan luong (trong so 1); cat grad CLIP. Thap van ban: eval + no_grad (dong_thap).'''
    t0 = time.time()
    with _fork():
        torch.manual_seed(SEED + 600)
        gv = torch.Generator(device=dev); gv.manual_seed(SEED + 601)
        gtv = torch.Generator(device=dev); gtv.manual_seed(SEED + 602)
        dl = loader(I_TF, train_tf, True, BATCH)
        n = int(_KD['buoc_thu_nhanh']) if THU_NHANH else int(round(float(_KD['epoch']) * len(dl)))
        o_kd = torch.optim.AdamW([{'params': [p for p in PARAM_LUONG if p.ndim > 1], 'weight_decay': WD},
                                  {'params': [p for p in PARAM_LUONG if p.ndim <= 1], 'weight_decay': 0.0}], lr=float(_KD['lr']))
        w_kd = max(1, int(0.05 * n))
        s_kd = torch.optim.lr_scheduler.LambdaLR(o_kd, lambda s: (s + 1) / w_kd if s < w_kd else
                                                 0.5 * (1 + math.cos(math.pi * (s - w_kd) / max(n - w_kd, 1))))
        bb_ft.eval(); HEAD.eval(); PHU.eval(); LUONG.train(); PHU_V.train(); DAUV.train()
        TONG = {q: [0.0] * N_MOC for q in PHAN_KD}; DEM = {q: [0] * N_MOC for q in PHAN_KD}
        it = iter(dl); n_bo = 0
        for k in tqdm(range(n), desc='khoi dong luong', leave=False):
            try: b = next(it)
            except StopIteration: it = iter(dl); b = next(it)
            rows = b['row'].to(dev)
            _, _, ra = fwd_luong_v2(b['image'].to(dev, non_blocking=True), True, rows, mo_ghi=False, grad_bb=False)
            o = phu_luong(ra, rows, b['hop'].to(dev, non_blocking=True), b['co_hop'].to(dev, non_blocking=True), gv, gtv, dong_thap=True,
                          y=b['y'].to(dev, non_blocking=True))
            ls = [o[q] for q in PHAN_KD if q in o]
            if not ls: continue
            tong = sum(ls)
            if not torch.isfinite(tong):
                n_bo += 1; continue
            o_kd.zero_grad(set_to_none=True); tong.backward()
            torch.nn.utils.clip_grad_norm_(PARAM_LUONG, CLIP); o_kd.step(); s_kd.step()
            m = min(N_MOC - 1, k * N_MOC // n)
            for q in PHAN_KD:
                if q in o: TONG[q][m] += float(o[q].detach()); DEM[q][m] += 1
        o_kd.zero_grad(set_to_none=True)
        del it, dl, o_kd, s_kd
    LUONG.eval(); PHU_V.eval(); DAUV.eval()
    if dev == 'cuda': torch.cuda.empty_cache()
    ls = {q: [TONG[q][i] / DEM[q][i] if DEM[q][i] else None for i in range(N_MOC)] for q in PHAN_KD}
    return ls, n, n_bo, time.time() - t0


# ---- 78x J v3b dot 4 (Tuan duyet 21/10; SPEC B.2n): KHOI DONG LUONG CHUNG voi J v3b s42 — khoi dong DONG write-back (mo_ghi=False) nen
#      CUNG phep tinh voi s42 -> CHEP tep cua s42 (CHI doc, qua API) thay vi tinh lai: diem bat dau ghep cap ----
if not THU_NHANH and not tim_moi_noi(TEN_KD):
    _kk_kd = kiem_ke(tuple(CFG_D4['ghep_cap']), mau=r'luong_khoi_dong\.pt')[1].get(TEN_KD) or []
    assert _kk_kd, (f"{'/'.join(CFG_D4['ghep_cap'])}: KHONG co {TEN_KD} -> DUNG (ghep cap can khoi dong cua s42; bao Claude, KHONG tu tinh "
                    'lai)')
    _kd0 = torch.load(tai(_kk_kd[0]), map_location='cpu', weights_only=False)
    assert _kd0.get('run') == CFG_D4['ghep_cap'][1] and _kd0.get('luot_79a') == LUOT_79A, (
        f"{TEN_KD} cua {'/'.join(CFG_D4['ghep_cap'])}: run / luot KHAC ({_kd0.get('run')}, {_kd0.get('luot_79a')}) -> DUNG")
    ghi(TEN_KD, lambda q: torch.save(dict(_kd0, run=RUN_TAG, nguon_khoi_dong=CFG_D4['ghep_cap'][1], md5_nguon=_kk_kd[0]['md5']), q))
    print(f"78x dot 4: khoi dong luong CHEP tu {'/'.join(CFG_D4['ghep_cap'])} (md5 {str(_kk_kd[0]['md5'])[:8]}, {_kd0.get('n_buoc')} "
          'buoc) — KHONG tinh lai')
    del _kd0, _kk_kd
_s0 = _sts()
_lp_kd = tim_moi_noi(TEN_KD)
if _lp_kd:
    _kd = torch.load(_lp_kd, map_location=dev, weights_only=False)
    assert _kd.get('run') == RUN_TAG and _kd.get('luot_79a') == LUOT_79A, f'{TEN_KD} cua luot / run KHAC -> DOI RUN_TAG o C0, KHONG xoa gi'
    LUONG.load_state_dict(_kd['luong'], strict=True); PHU_V.load_state_dict(_kd['phu_v'], strict=True)
    DAUV.load_state_dict(_kd['dauv'], strict=True)
    LS_KD, N_KD = _kd['lich_su'], int(_kd['n_buoc'])
    tai_len_lai(TEN_KD)
    print(f"khoi dong: DOC LAI {TEN_KD} ({_kd['t']}, {N_KD} buoc) — KHONG chay lai")
    del _kd
else:
    print(f"khoi dong luong: {'%d buoc (THU_NHANH)' % int(_KD['buoc_thu_nhanh']) if THU_NHANH else '%g epoch' % float(_KD['epoch'])}, "
          f"LR {_KD['lr']}, phan {PHAN_KD} ...", flush=True)
    LS_KD, N_KD, _bo_kd, _giay_kd = khoi_dong()
    ghi(TEN_KD, lambda q: torch.save(dict(luong=LUONG.state_dict(), phu_v=PHU_V.state_dict(), dauv=DAUV.state_dict(), run=RUN_TAG, luot_79a=LUOT_79A,
                                          lich_su=LS_KD, n_buoc=N_KD, n_bo=_bo_kd, giay=round(_giay_kd, 1), cfg=_KD,
                                          t=datetime.datetime.now().isoformat(timespec='seconds')), q))
    print(f'khoi dong: {N_KD} buoc, {_giay_kd / 60:.1f} phut, bo {_bo_kd} buoc loss khong huu han')
assert all(torch.equal(a, b) for a, b in zip(_sts(), _s0)), 'khoi dong TIEU dong ngau nhien chung -> DUNG'
assert all(float(l.ca_h.o.weight.abs().max()) == 0 and float(l.ca_h.o.bias.abs().max()) == 0 for l in LUONG.lop), \
    'duong ghi nguoc DOI trong khoi dong (phai giu 0 toi C6) -> DUNG'
_fk = lambda v: '   -  ' if v is None else f'{v:6.3f}'
print(f'loss khoi dong theo {N_MOC} doan deu (TB moi doan):')
for _qk in PHAN_KD: print(f'   {_qk:<11}' + ' '.join(_fk(v) for v in LS_KD[_qk]))   # KHONG ten `_q`: ham cua khoi Drive (C0)
_tg = [sum(LS_KD[q][i] for q in PHAN_KD if LS_KD[q][i] is not None) for i in range(N_MOC)]
_giam = (_tg[-2] - _tg[-1]) / max(abs(_tg[-2]), 1e-12)
print(f'   tong       ' + ' '.join(f'{v:6.3f}' for v in _tg) + f' | doan cuoi giam {_giam:.1%} so voi doan truoc -> '
      + ('CON GIAM MANH: Tuan can nhac tang khoi_dong.epoch len 2-3 (CFG_79 o C0, RUN_TAG moi) — KHONG tu doi'
         if _giam > float(_KD['keo_dai_neu_giam']) else 'da phang: giu 1 epoch'))
_KHOA_C4B = _KHOA_C4
