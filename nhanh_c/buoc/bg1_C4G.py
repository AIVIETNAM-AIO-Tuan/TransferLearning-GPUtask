# ==== nhanh_c buoc bg1_C4G — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C4g · BIEN THE Bg (Tuan duyet 04/11; docs/SPEC_SAU_78Y.md B.2ai; PLAN A.4y): cong thuc B + gradient qua embeddings + block 0-7 cua 5 o CHI qua skip ====
# Co che hai luot cua Cg (g1_C4G) tren duong o cua B: luot 1 = fwd_o cua B, t7 (token_o7, khong grad) lam LA requires_grad -> skip -> loss ->
# backward (gradient duong toan canh + dL/dt7); luot 2 (day_grad_o7, bg1_C6 goi NGAY sau backward cua micro-batch): tinh lai t7 theo khuc KHUC_G
# view CO gradient (CUNG o_phong_518 / autocast / vong token_o7, dung o block 7) -> torch.autograd.backward(t7, dL/dt7) -> CONG vao .grad cua
# embeddings + block 0-7 (tham so CHUNG voi anh toan canh). O KHONG di qua block 8-11 (nhu B) -> block 8-11 + LN cuoi khong nhan gi tu duong o.
# Duong o KHONG rut so ngau nhien -> thu tu lo / tang cuong / mat na drop-path TRUNG B cung seed (cong ghep cap o bg1_C6). Ham t7_o_khuc /
# day_grad_o7 / thong_ke_go / in_go = ham cua g1_C4G (Cg) NGUYEN VAN.
assert CAU_HINH_C.get('bien_the') == 'Bg' and '_KHOA_C4' in globals() and 'fwd_o' in globals() and 'DUA_O' in globals() \
    and 'BO_O11' not in globals(), 'bg1_C4G: chay qua nhanh_c.buoc(..., bien_the="Bg") NGAY sau bg1_C4 (duong o cua B) -> DUNG'
CFG_G = dict(CAU_HINH_C['grad_o7'])
KHUC_G, TOL_G, TOL_T7 = int(CFG_G['khuc']), 1e-2, 1e-2
assert CFG_G['duong'] == 'skip' and int(CFG_G['tang_cuoi']) == K_SKIP_O - 1 == 7 and CFG_G['embeddings'] is True and 1 <= KHUC_G <= KHUC_O, (
    f'bien the Bg: cau hinh grad_o7 {CFG_G} lech -> DUNG')
if not getattr(fwd_o, '_bien_the_bg', False):
    _FWD_O_B = fwd_o                                                  # fwd_o cua B (bg1_C4 = C4 skip7); chay lai o nay KHONG boc chong
THAM_SO_G = [p for p in list(bb_ft.embeddings.parameters()) + [p for l in range(int(CFG_G['tang_cuoi']) + 1)
                                                                for p in bb_ft.encoder.layer[l].parameters()] if p.requires_grad]
_ID_G = {id(p) for p in THAM_SO_G}
THAM_SO_TREN = [p for p in bb_ft.parameters() if p.requires_grad and id(p) not in _ID_G]   # block 8-11 + LN cuoi: duong o KHONG cham
GO_T = []


def t7_o_khuc(bb, o, k7, dtype):
    '''o [M, 518, 518] (o phong o_phong_518, da chuan hoa; M view) -> t7 [M, N, d] = CUNG phep tinh voi vong token_o (embeddings -> block
       0..k7 qua _out_luong), dung o block k7 (KHONG block 8-11 / layernorm). Gradient + autocast theo ngu canh noi goi.'''
    h = bb.embeddings(o.unsqueeze(1).expand(-1, 3, -1, -1).contiguous().to(dtype))
    for l in range(int(k7) + 1):
        h = _out_luong(bb.encoder.layer[l](h))
    return h[:, 1:]


def fwd_o(img, grad, rows, mo_ghi=True, grad_bb=True, do=None, dua=None):
    '''Bien the Bg: = fwd_o cua B (_FWD_O_B); khi grad (va grad dang bat, dua None): t7 (token_o7, khong grad - nhu B) -> LA requires_grad -> skip
       (BO_O7 -> 2 x 2 diem con -> DUA_O, tinh lai luc backward) -> ra['t7_la'] (day_grad_o7 doc t7_la.grad sau backward). grad False (du doan /
       cham / chan doan) hoac `dua` (doi chung duong o cua cong C4) -> GOI DUNG fwd_o cua B.'''
    if not (grad and torch.is_grad_enabled()) or dua is not None:
        return _FWD_O_B(img, grad, rows, mo_ghi=mo_ghi, grad_bb=grad_bb, do=do, dua=dua)
    with torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        t7 = token_o7(bb_ft, img, LUOI_O, MEAN, STD, KHUC_O, K_SKIP_O - 1)
    t7 = t7.detach().requires_grad_(True)
    with torch.enable_grad(), torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        inj = skip_o(t7, BO_O7, DUA_O, *luoi_o(len(img)), ckpt=True)
    with skip_vao_luong(LUONG, inj, K_SKIP_O, do):
        p, c, ra = fwd_luong_v2(img, grad, rows, mo_ghi=mo_ghi, grad_bb=grad_bb)
    ra['t7_la'] = t7
    return p, c, ra


fwd_o._bien_the_bg = True


def day_grad_o7(img, t7_la, k, chan_doan=None):
    '''LUOT 2 cua micro-batch k (goi NGAY sau (loss / ACCUM).backward()): g7 = t7_la.grad -> tinh lai t7 theo khuc KHUC_G view CO gradient
       (CUNG o_phong_518 + autocast nhu luot 1) -> torch.autograd.backward(t7, g7) -> CONG vao .grad cua THAM_SO_G (embeddings + block
       0-7). Chan doan (mac dinh: k % ACCUM == 0 va k // ACCUM < N_DO_TL = micro-batch DAU cua N_DO_TL buoc dau): ||g truoc luot 2|| (duong
       toan canh + loss phu cua micro-batch nay), ||g o|| (phan luot 2 cong them), cos giua hai phan, ||dL/dt7|| -> GO_T.'''
    if t7_la is None or t7_la.grad is None: return
    g7 = t7_la.grad
    do = ((k % ACCUM == 0) and (k // ACCUM < N_DO_TL)) if chan_doan is None else bool(chan_doan)
    g0 = [(torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()) for p in THAM_SO_G] if do else None
    with torch.no_grad(), torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        o = o_phong_518(img, LUOI_O, MEAN, STD)
    B, K = int(o.shape[0]), int(o.shape[1])
    o = o.reshape(B * K, o.shape[2], o.shape[3])
    g = g7.reshape(B * K, g7.shape[2], g7.shape[3])
    for i in range(0, B * K, KHUC_G):
        with torch.enable_grad(), torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
            t = t7_o_khuc(bb_ft, o[i:i + KHUC_G], K_SKIP_O - 1, img.dtype)
        torch.autograd.backward(t, g[i:i + KHUC_G].to(t.dtype))
        del t
    if do:
        with torch.no_grad():
            d = [(torch.zeros_like(p) if p.grad is None else p.grad.detach()) - a for p, a in zip(THAM_SO_G, g0)]
            n0 = torch.sqrt(sum((a.float() ** 2).sum() for a in g0))
            nd = torch.sqrt(sum((x.float() ** 2).sum() for x in d))
            dot = sum((a.float() * x.float()).sum() for a, x in zip(g0, d))
            GO_T.append(dict(g_toan_canh=float(n0), g_o=float(nd), cos=float(dot / (n0 * nd).clamp_min(1e-30)), g7=float(g7.float().norm())))
        del g0, d


def thong_ke_go(go):
    '''So tong hop cua GO_T (micro-batch chan doan) -> epoch_log['grad_o7'].'''
    if not go: return dict(n=0, cfg=dict(CFG_G))
    a = {q: [float(x[q]) for x in go] for q in ('g_toan_canh', 'g_o', 'cos', 'g7')}
    return dict(n=len(go), **{f'{q}_tb': float(np.mean(v)) for q, v in a.items()},
                ti_le_tb=float(np.mean([x['g_o'] / max(x['g_toan_canh'], 1e-30) for x in go])), cfg=dict(CFG_G))


def in_go(t):
    '''Mot dong in cua thong_ke_go (C6, sau dong O CAT).'''
    if not t.get('n'): return '   GRAD O THAP: chua do (khong co micro-batch chan doan)'
    return (f"   GRAD O THAP (embeddings + block 0-7, {t['n']} buoc dau): ||g toan canh|| TB {t['g_toan_canh_tb']:.3e} | ||g o|| TB "
            f"{t['g_o_tb']:.3e} (ti le TB {t['ti_le_tb']:.3f}) | cos TB {t['cos_tb']:+.3f} | ||dL/dt7|| TB {t['g7_tb']:.3e} | khuc {KHUC_G} view")


# ---- cong bien the Bg (lo val THAT, eval; trong fork_rng; module o NHIEU tam thoi bang generator rieng roi KHOI PHUC tung bit):
#      (1) fwd_o Bg: grad False = fwd_o B; grad True: patch / cls / token vung lech <= 1e-5; (2) hai luot = MOT do thi (t7 tinh CO gradient trong
#      cung do thi) tren MOI tham so hoc: lech tuong doi <= TOL_G, cos >= 0.999, phan duong o khac 0; (3) luot 2 KHONG cong gi vao block 8-11 /
#      LN cuoi (tung bit); (4) t7 tinh lai theo khuc KHUC_G = t7 cua token_o7 (64 anh, khuc KHUC_O): lech tuong doi <= TOL_T7; (5) dong ngau
#      nhien chung khong doi, module o khoi phuc tung bit, .grad xoa ----
_s0_g = _sts()
_sd_g = {n: {q: v.detach().clone() for q, v in m.state_dict().items()} for n, m in (('DUA_O', DUA_O), ('BO_O7', BO_O7))}
_mods_g = [bb_ft, HEAD, LUONG, DOCV, DAUV, DAU_NGUON, KENH_CD, DUA_O, BO_O7]
_che_do_g = [(sm, sm.training) for _m_g in _mods_g for sm in _m_g.modules()]   # che do train / eval TUNG module con -> tra nguyen sau cong
_P_g, _thay_g = [], set()
for _m_g in _mods_g:
    for _p_g in _m_g.parameters():
        if _p_g.requires_grad and id(_p_g) not in _thay_g: _thay_g.add(id(_p_g)); _P_g.append(_p_g)
_xoa_g = lambda: [setattr(p, 'grad', None) for p in _P_g]
_chep_g = lambda ps: [None if p.grad is None else p.grad.detach().clone() for p in ps]
with _fork():
    _gen_g = torch.Generator(device=dev).manual_seed(911)
    with torch.no_grad():
        for _m_g in (DUA_O, BO_O7):
            for _p_g in _m_g.parameters(): _p_g.add_(torch.randn(_p_g.shape, generator=_gen_g, device=dev).to(_p_g.dtype) * 0.05)
    for _m_g in _mods_g: _m_g.eval()
    _bg = next(iter(loader(I_VA[:4], eval_tf, False, 4, nw=0)))
    _xg, _rg = _bg['image'].to(dev), _bg['row'].to(dev)
    _Wg = torch.randn(len(_xg), C, generator=_gen_g, device=dev)
    _lg_g = lambda p, ra: (dau_v2(p, dict(ra))[0].float() * _Wg).sum()
    _so_g = lambda a, b: max(float((a[0] - b[0]).abs().max()), float((a[1] - b[1]).abs().max()), float((a[2]['r'] - b[2]['r']).abs().max()))
    with torch.no_grad():                                                                  # (1) grad False
        _d1_g = _so_g(_FWD_O_B(_xg, False, _rg), fwd_o(_xg, False, _rg))
    _xoa_g()
    with torch.enable_grad():                                                              # (1) grad True
        _a_g, _b_g = _FWD_O_B(_xg, True, _rg), fwd_o(_xg, True, _rg)
        _d2_g = _so_g([x.detach() for x in _a_g[:2]] + [{q: (v.detach() if torch.is_tensor(v) else v) for q, v in _a_g[2].items()}],
                      [x.detach() for x in _b_g[:2]] + [{q: (v.detach() if torch.is_tensor(v) else v) for q, v in _b_g[2].items()}])
    del _a_g, _b_g
    _xoa_g()
    with torch.enable_grad():                                                              # (2A) MOT do thi: t7 CO gradient trong do thi
        with torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
            _oA = o_phong_518(_xg, LUOI_O, MEAN, STD)
            _t7A = t7_o_khuc(bb_ft, _oA.reshape(-1, _oA.shape[2], _oA.shape[3]), K_SKIP_O - 1, _xg.dtype)
            _t7A = _t7A.reshape(len(_xg), SO_O, _t7A.shape[1], _t7A.shape[2])
            _injA = skip_o(_t7A, BO_O7, DUA_O, *luoi_o(len(_xg)), ckpt=True)
        with skip_vao_luong(LUONG, _injA, K_SKIP_O):
            _pA, _cA, _raA = fwd_luong_v2(_xg, True, _rg)
        _lA = _lg_g(_pA, _raA)
    _lA.backward()
    _gA = _chep_g(_P_g)
    del _oA, _t7A, _injA, _pA, _cA, _raA, _lA
    _xoa_g()
    with torch.enable_grad():                                                              # (2B) HAI luot
        _pB, _cB, _raB = fwd_o(_xg, True, _rg)
        _lB = _lg_g(_pB, _raB)
    _lB.backward()
    _gB1, _tren0 = _chep_g(_P_g), _chep_g(THAM_SO_TREN)
    day_grad_o7(_xg, _raB.pop('t7_la'), -1, chan_doan=False)
    _gB, _tren1 = _chep_g(_P_g), _chep_g(THAM_SO_TREN)
    del _pB, _cB, _raB, _lB
    _xoa_g()
    _b64 = next(iter(loader(I_VA[:64], eval_tf, False, 64, nw=0)))                           # (4) t7 tinh lai theo khuc = token_o7
    _x64 = _b64['image'].to(dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        _t7r = token_o7(bb_ft, _x64, LUOI_O, MEAN, STD, KHUC_O, K_SKIP_O - 1)
        _o64 = o_phong_518(_x64, LUOI_O, MEAN, STD)
        _o64 = _o64.reshape(-1, _o64.shape[2], _o64.shape[3])
        _t7k = torch.cat([t7_o_khuc(bb_ft, _o64[i:i + KHUC_G], K_SKIP_O - 1, _x64.dtype) for i in range(0, len(_o64), KHUC_G)])
    _rel_t7 = float((_t7k.float() - _t7r.reshape(_t7k.shape).float()).norm() / _t7r.float().norm().clamp_min(1e-30))
    del _b64, _x64, _t7r, _o64, _t7k
    with torch.no_grad():                                                                  # (5) khoi phuc module o
        for _n_g, _m_g in (('DUA_O', DUA_O), ('BO_O7', BO_O7)): _m_g.load_state_dict(_sd_g[_n_g], strict=True)
for _sm_g, _tr_g in _che_do_g: _sm_g.training = _tr_g                                    # che do train / eval nhu SAU C4 (tung module con)
_none_g = all((a is None) == (b is None) for a, b in zip(_gA, _gB))
_fa_g = torch.cat([a.float().flatten() for a in _gA if a is not None])
_fb_g = torch.cat([b.float().flatten() for b in _gB if b is not None])
_rel_g = float((_fa_g - _fb_g).norm() / _fa_g.norm().clamp_min(1e-30))
_cos_g = float((_fa_g * _fb_g).sum() / (_fa_g.norm() * _fb_g.norm()).clamp_min(1e-30))
_ix_g = [i for i, p in enumerate(_P_g) if id(p) in _ID_G]
_go_g = float(torch.sqrt(sum(((_gB[i] - (0 if _gB1[i] is None else _gB1[i])).float() ** 2).sum() for i in _ix_g if _gB[i] is not None)))
_tren_g = all((a is None and b is None) or (a is not None and b is not None and torch.equal(a, b)) for a, b in zip(_tren0, _tren1))
_kp_g = all(torch.equal(v, m.state_dict()[q]) for n, m in (('DUA_O', DUA_O), ('BO_O7', BO_O7)) for q, v in _sd_g[n].items())
assert _d1_g <= 1e-6 and _d2_g <= 1e-5, f'bien the Bg: fwd_o KHAC fwd_o cua B (grad False {_d1_g:.2e} / grad True {_d2_g:.2e}) -> DUNG'
assert _none_g and _rel_g <= TOL_G and _cos_g >= 0.999 and _go_g > 0, (
    f'bien the Bg: hai luot KHAC mot do thi (lech tuong doi {_rel_g:.2e}, cos {_cos_g:.6f}, phan duong o {_go_g:.2e}) -> DUNG')
assert _tren_g, 'bien the Bg: luot 2 CONG gradient vao block 8-11 / LN cuoi -> DUNG'
assert _rel_t7 <= TOL_T7, f'bien the Bg: t7 tinh lai theo khuc {KHUC_G} LECH t7 cua token_o7 ({_rel_t7:.2e}) -> DUNG'
assert _kp_g and all(torch.equal(a, b) for a, b in zip(_sts(), _s0_g)) and all(p.grad is None for p in _P_g), (
    'bien the Bg: module o chua khoi phuc / dong ngau nhien chung doi / .grad con -> DUNG')
print(f"cong bien the Bg: fwd_o = B (grad False lech {_d1_g:.1e}, grad True lech {_d2_g:.1e}) | hai luot = mot do thi: lech tuong doi {_rel_g:.1e}, "
      f"cos {_cos_g:.6f} (phan duong o ||g|| {_go_g:.3e} > 0) | luot 2 khong cham block 8-11 / LN cuoi (tung bit) | t7 tinh lai khuc {KHUC_G} = "
      f"token_o7 khuc {KHUC_O} (lech tuong doi {_rel_t7:.1e}, 64 anh) | dong ngau nhien khong doi, module o khoi phuc tung bit | "
      f"{len(THAM_SO_G)} tensor embeddings + block 0-7 ({sum(p.numel() for p in THAM_SO_G):,} tham so) nhan gradient tu o")
del _s0_g, _sd_g, _mods_g, _P_g, _thay_g, _m_g, _p_g, _gen_g, _bg, _xg, _rg, _Wg, _lg_g, _so_g, _d1_g, _d2_g, _gA, _gB1, _tren0, _gB, _tren1
del _rel_t7, _none_g, _fa_g, _fb_g, _rel_g, _cos_g, _ix_g, _go_g, _tren_g, _kp_g, _n_g, _xoa_g, _chep_g, _che_do_g, _sm_g, _tr_g
