# -*- coding: utf-8 -*-
"""Cau hinh nhanh C theo seed (Tuan duyet 29/10; docs/SPEC_SAU_78Y.md B.2ab; PLAN A.4v). nhanh_c.buoc(ten, ns, seed) dat ket qua cau_hinh(seed, thu)
   vao ns['CAU_HINH_C']; moi buoc doc tu do (seed, RUN_TAG, thu muc Drive, moc C s42) - KHONG hang so an trong buoc."""

PHIEN_BAN = 'nhanh_c_v1'
SEEDS = (42, 43, 44, 45, 46)                     # s42 = luot goi o cat da chay (dung lai); s43-s46 chay bang goi nay
TM = 'stage4_nb84'                               # thu muc Drive cua module 84 (moi luot nhanh C ghi vao day)

# luot goi o cat s42 = seed 1 cua nhanh C (CHI doc qua API)
MOC_C_S42 = dict(x=('stage4_nb78x', 'r78x_J_v3b_dp_ema_v2_o_s42'), z=('stage4_nb78z', 'r78z_Jv3b_dp_ema_v2_o_s42'),
                 gts=('stage4_nb78z', 'r78z_Jv3b_dp_ema_v2_o_s42_gts'), gop=0.2915, tb_p1=0.2619)
# A = B.4 v2 s42 (khong skip, dau J 1 view) - chi co o s42
MOC_A_S42 = dict(gts=('stage4_nb78z', 'r78z_Jv3b_dp_ema_v2_s42_gts'), gop=0.2881)
NGUONG_SUC_KHOE = 0.01                           # Phan 1: TB VAL trong luot ep1-6 thap hon C s42 qua nguong nay -> YEU (Tuan quyet)

# ---- bien the Cg (Tuan 02/11; docs/SPEC_SAU_78Y.md B.2ae; PLAN A.4w): gradient qua embeddings + block 0-7 cua 5 o CHI qua skip; ghep cap
#      voi C CUNG seed (cung thu tu lo / khoi dong / lambda). Moc C cung seed = so da in o C9 (TB VAL EMA ep1-6) / C6S (gop trong so) ----
BIEN_THE = ('C', 'Cg')
SEEDS_CG = (43, 44)
MOC_C_SEED = {43: dict(tb_p1=0.2616, gop=0.2870), 44: dict(tb_p1=0.2602, gop=0.2867)}
CFG_GRAD_O7 = dict(phien_ban='g_o7_v1', duong='skip', tang_cuoi=7, embeddings=True, khuc=48)
QUYET_CG = dict(dau='fus_attn_hr', nguong=0.003, n_boot=2000, hat=22101, tol_moc=5e-4)


def ten_luot(seed, bien_the='C'):
    '''RUN_TAG ba phan cua mot seed (thu muc TM); bien the Cg -> r84x_Cg_s<seed> ...'''
    p = {'C': 'C', 'Cg': 'Cg'}[bien_the]
    return dict(run_x=f'r84x_{p}_s{seed}', run_z=f'r84z_{p}_s{seed}', run_gts=f'r84z_{p}_s{seed}_gts')


def _nguon_seed(s):
    if s == 42:
        return dict(z=list(MOC_C_S42['z']), gts=list(MOC_C_S42['gts']))
    t = ten_luot(s)
    return dict(z=[TM, t['run_z']], gts=[TM, t['run_gts']])


def cau_hinh(seed, thu=False, bien_the='C'):
    '''Cau hinh mot luot nhanh C (bien_the 'C', mac dinh - KHONG doi so voi commit truoc) hoac bien the 'Cg' (seed thuoc SEEDS_CG). seed phai
       thuoc SEEDS (so nguyen, khong bool).'''
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise ValueError(f'nhanh C: seed {seed!r} khong hop le (can mot trong {SEEDS})')
    if not isinstance(thu, bool):
        raise ValueError(f'nhanh C: thu phai la bool, nhan {thu!r}')
    if bien_the not in BIEN_THE:
        raise ValueError(f'nhanh C: bien the {bien_the!r} khong hop le (can mot trong {BIEN_THE})')
    if bien_the == 'Cg' and seed not in SEEDS_CG:
        raise ValueError(f'nhanh C bien the Cg: seed {seed!r} khong hop le (can mot trong {SEEDS_CG}: C cung seed da xong)')
    cfg = dict(nhanh='C', phien_ban=PHIEN_BAN, seed=seed, thu=thu, seeds=list(SEEDS), tm=TM, tm_x=TM, **ten_luot(seed, bien_the),
               moc_x=list(MOC_C_S42['x']), suc_khoe=dict(moc_tb=MOC_C_S42['tb_p1'], nguong=NGUONG_SUC_KHOE),
               tong_hop=dict(seeds=list(SEEDS), nguon={str(s): _nguon_seed(s) for s in SEEDS}, moc_s42_gop=MOC_C_S42['gop'],
                             a_s42=dict(gts=list(MOC_A_S42['gts']), moc_gop=MOC_A_S42['gop'])))
    if bien_the == 'C':
        return cfg
    c_seed = ten_luot(seed)                                       # C CUNG seed (doi chung ghep cap, CHI doc)
    cfg.update(bien_the='Cg', nhan_moc=f'C s{seed}', seed_moc=seed, moc_x=[TM, c_seed['run_x']], ghep_cap_x=[TM, c_seed['run_x']],
               suc_khoe=dict(moc_tb=MOC_C_SEED[seed]['tb_p1'], nguong=NGUONG_SUC_KHOE), grad_o7=dict(CFG_GRAD_O7),
               quyet=dict(QUYET_CG, gts=[TM, c_seed['run_gts']], moc_gop=MOC_C_SEED[seed]['gop']))
    return cfg
