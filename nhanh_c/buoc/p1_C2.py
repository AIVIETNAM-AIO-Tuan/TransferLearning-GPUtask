# ==== nhanh_c buoc p1_C2 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C2 · RAD-DINO GOC (hub) + bien doi (38a + TF_HOP 78g) + DS (anh + hop vung) + dau doc 38a khoi tao tu 37 (QUA API) + cong ====
assert 'kiem_ke' in globals() and 'bien_doi_kem_hop' in globals() and 'ap_cols' in globals(), 'chay C0 truoc'
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from torchvision import transforms as T

proc = AutoImageProcessor.from_pretrained('microsoft/rad-dino')
MEAN, STD = proc.image_mean[0], proc.image_std[0]
train_tf = T.Compose([T.RandomResizedCrop(SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05),
                                          interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomRotation(7), T.ToTensor(), T.Normalize([MEAN]*3, [STD]*3)])
eval_tf  = T.Compose([T.ToTensor(), T.Normalize([MEAN]*3, [STD]*3)])
# 78g: CUNG phep bien doi hinh hoc voi train_tf (RandomResizedCrop + RandomRotation, cung tham so) nhung qua
# torchvision.transforms.v2: MOT lan rut tham so cho CA anh lan hop vung ImaGenome (arm vung, khoi_hop_anh). Anh roi qua
# eval_tf (ToTensor + Normalize) -> CUNG dang tensor voi train_tf. `train_tf` giu lam CO: DS thay s.tf is train_tf -> di
# duong anh + hop. Hop ngoai anh sau xoay -> ClampBoundingBoxes; hop suy bien (< 1 px) -> coi nhu khong co.
from torchvision.transforms import v2 as T2
TF_HOP = T2.Compose([T2.RandomResizedCrop(SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05),
                                          interpolation=T2.InterpolationMode.BICUBIC),
                     T2.RandomRotation(7), T2.ClampBoundingBoxes()])

# ---- backbone: trong so GOC microsoft/rad-dino tren hub (KHONG checkpoint 38a), fine-tune TOAN BO (nhu bb_ft cua 38a) ----
_sd0 = {k: v.clone() for k, v in AutoModel.from_pretrained('microsoft/rad-dino').state_dict().items()}
bb_ft = AutoModel.from_pretrained('microsoft/rad-dino')
# 78g (Colab 06/10, Python 3.13): transformers >= 5.18 DOI TEN lop attention cua Dinov2 — attention.attention.query / key /
# value -> attention.q_proj / k_proj / v_proj, attention.output.dense -> attention.o_proj (bang doi ten `Dinov2Model` trong
# conversion_mapping.py cua chinh transformers). from_pretrained tu doi; torch.load + load_state_dict thi KHONG. Checkpoint 38a
# luu theo ten CU -> doi ten CHI khi mo hinh dang chay dung ten MOI; van strict=True. Cong: tep safetensors GOC cua RAD-DINO
# tren hub (ten cu) doi ten bang CUNG ham phai TRUNG TUNG BIT ban from_pretrained tu doi (_sd0) -> bang doi ten khop transformers.
from transformers import __version__ as _TF_VER
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as _doc_st
_DOI_TEN_DINOV2 = (('.attention.attention.query.', '.attention.q_proj.'), ('.attention.attention.key.', '.attention.k_proj.'),
                   ('.attention.attention.value.', '.attention.v_proj.'), ('.attention.output.dense.', '.attention.o_proj.'))


def doi_ten_dinov2(sd, khoa_mo_hinh):
    '''state_dict Dinov2 -> ten khoa cua mo hinh dang chay: da khop -> giu nguyen; khong -> doi ten CU sang MOI (chi doi ten).'''
    if set(sd) <= set(khoa_mo_hinh): return dict(sd)
    out = {}
    for k, v in sd.items():
        for a, b in _DOI_TEN_DINOV2: k = k.replace(a, b)
        out[k] = v
    return out


_st_hub = _doc_st(hf_hub_download('microsoft/rad-dino', 'model.safetensors'))   # 78u: giu ten khoa CU (38a / tep hub)
KHOA_38A = sorted(_st_hub)
_hub = doi_ten_dinov2(_st_hub, _sd0)
assert set(_hub) == set(_sd0) and all(torch.equal(_hub[k].to(_sd0[k].dtype), _sd0[k]) for k in _sd0), (
    f'transformers {_TF_VER}: doi ten Dinov2 KHAC cach from_pretrained tu doi (tep hub RAD-DINO) -> DUNG')
del _hub, _st_hub
# 78u: KHONG nap checkpoint 38a — backbone = trong so GOC tren hub. rs_ / backbone_ft.pt LUU theo ten CU cua 38a (ve_ten_38a,
# nghich dao doi_ten_dinov2) -> resume (C6) va 78v nap bang CHINH cach nap cua 78g (doi_ten_dinov2) o CA HAI bo cuc ten.
TEN_MOI_BB = any('.attention.q_proj.' in k for k in _sd0)


def ve_ten_38a(sd):
    '''state_dict Dinov2 theo ten cua mo hinh DANG CHAY -> ten CU (38a / tep hub): nghich dao doi_ten_dinov2 (chi doi ten).'''
    out = {}
    for k, v in sd.items():
        for a, b in _DOI_TEN_DINOV2: k = k.replace(b, a)
        out[k] = v
    return out


_ve = ve_ten_38a(_sd0); _lai = doi_ten_dinov2(_ve, _sd0)
assert set(_ve) == set(KHOA_38A) and set(_lai) == set(_sd0) and all(_lai[k] is _sd0[k] for k in _sd0), (
    f'transformers {_TF_VER}: ve_ten_38a KHONG la nghich dao doi_ten_dinov2 tren khoa hub -> rs_ / backbone_ft.pt se sai ten, DUNG')
del _ve, _lai
print(f'transformers {_TF_VER}: bo cuc ten Dinov2 ' + ('MOI (q/k/v/o_proj)' if TEN_MOI_BB else 'CU (attention.attention.*)')
      + ' | tep hub doi ten = from_pretrained tung bit (cong) | rs_ / backbone_ft.pt luu ten CU 38a (khu hoi = cong)')
assert set(bb_ft.state_dict()) == set(_sd0) and all(torch.equal(v, _sd0[k]) for k, v in bb_ft.state_dict().items()), \
    'backbone KHAC trong so hub (from_pretrained) -> DUNG'
del _sd0
bb_ft = bb_ft.to(dev)
for p in bb_ft.parameters(): p.requires_grad_(True)
D_CXR = bb_ft.config.hidden_size
_n_bb = sum(p.numel() for p in bb_ft.parameters())
_cfg = bb_ft.config
_dp = (getattr(_cfg, 'attention_probs_dropout_prob', 0), getattr(_cfg, 'hidden_dropout_prob', 0), getattr(_cfg, 'drop_path_rate', 0))
print(f'backbone: hidden {D_CXR} | patch {_cfg.patch_size} | {_n_bb:,} tham so (MOI tham so hoc) | dropout (attn, hidden, '
      f'drop_path) = {_dp} (khac 0 -> rut tu dong CHUNG, nhu nhau o moi luot)')


def fwd(bb, img, grad):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        h = bb(pixel_values=img).last_hidden_state
    return h[:, 1:].float()


def fwd_ft(img, grad):
    '''78u: MOT forward bb_ft (autocast nhu fwd 38a) -> (patch [B, N_TOK, d] float32 = fwd(bb_ft, img, grad) TUNG BIT (cong duoi),
       cls [B, d] float32 = token CLS sau LayerNorm cuoi: vector anh cua B1 / B2).'''
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        h = bb_ft(pixel_values=img).last_hidden_state
    return h[:, 1:].float(), h[:, 0].float()



def fwd_luong(img, grad, mo_ghi=True, grad_bb=True):
    """79a: MOT forward bb_ft + LUONG (khoi_luong_vung.chay_bb_luong) trong CUNG autocast / grad context nhu fwd_ft -> (patch [B, N_TOK,
       d] float32, cls [B, d] float32, ra = dau ra luong). LUONG tao o C4 (TRUOC C4 KHONG goi). Ghi nguoc = 0 -> TRUNG fwd_ft (cong C4).
       grad_bb=False (CHI khoi dong C4b, mo_ghi=False): block chay no_grad, luong van hoc."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        return chay_bb_luong_dp(bb_ft, LUONG, img, mo_ghi=mo_ghi, grad_bb=grad_bb, ckpt_bb=CKPT_BB, ti_le=TI_LE_DP, gen=GEN_DP)   # 78x dot 4 B.4: stochastic depth theo block



def fwd_luong_v2(img, grad, rows, mo_ghi=True, grad_bb=True):
    """78x: fwd_luong voi neo THEO MAU tu view_group (NEO_T, C3 — co san luc cham): frontal neo nhu 79a, lateral / unknown attention
       tu do. rows = hang ORDER (dev) cua lo."""
    LUONG.dat_neo(NEO_T[rows].float())
    return fwd_luong(img, grad, mo_ghi=mo_ghi, grad_bb=grad_bb)


def dau_v2(patch, ra):
    """78x: dau doc J v2 -> (logit sau MIL [B, C], h, kv patch). Y 1: K/V = [patch ; token vung] (DOCV); y 2: z vung (DAUV) -> ra['zv'];
       kenh MIL (w khoi tao 0) -> ra['lg'] (y B). Chi doc anh (+ view qua luong) — duong cham."""
    zv = DAUV.z(ra['r'])
    lg, h, kv = dau_h_v2(HEAD, patch, DOCV(HEAD.pin, ra['r']))
    dn = DAU_NGUON.delta(h)                                   # 78x J v3 (y C): offset 5 nguon / lop (khoi tao 0)
    lg = KENH_CD(DAUV.mil(DAU_NGUON.gop(lg, dn), zv), zv)    # nguon -> MIL toan cuc (31 nhan) -> kenh vung co dinh (14 nhan, y D)
    ra['zv'], ra['lg'], ra['h_dd'] = zv, lg, h.detach()        # h_dd: CE nguon CHI vao DAU_NGUON (cat gradient ve backbone)
    return lg, h, kv


class DS(Dataset):
    '''78u: anh + nhan. Duong train (tf is train_tf): anh + hop 36 vung (78f, C3) qua CUNG phep bien doi TF_HOP — MOI luot, ke ca
       B0 (MOT dong tang cuong chung; B3 / B4 can hop di cung anh). Duong cham (eval_tf): CHI doc anh.'''
    def __init__(s, idx, tf): s.idx, s.tf = np.asarray(idx), tf
    def __len__(s): return len(s.idx)
    def _img(s, i):
        im = Image.open(os.path.join(LOCAL, f'{ORDER.dicom_id.iloc[i]}.jpg')).convert('L')
        return s.tf(Image.merge('RGB', (im, im, im)))
    def _img_hop(s, i):
        '''78g: anh train + hop 36 vung (C3c: HOP518, toa do anh 518) qua CUNG phep bien doi TF_HOP. Anh khong co scene
           graph (anh nghieng / ngoai ImaGenome) -> khong hop (co = False).'''
        im = Image.open(os.path.join(LOCAL, f'{ORDER.dicom_id.iloc[i]}.jpg')).convert('L')
        g = int(SG_HANG[i])
        hop = HOP518[g] if g >= 0 else np.full((R_VUNG, 4), np.nan, 'float32')
        im2, hop2, co2 = bien_doi_kem_hop(Image.merge('RGB', (im, im, im)), hop, TF_HOP)
        return eval_tf(im2), torch.from_numpy(np.nan_to_num(hop2)), torch.from_numpy(co2)
    def __getitem__(s, k):
        i = int(s.idx[k])
        if s.tf is train_tf: x, hop, co_hop = s._img_hop(i)          # 78g: anh + hop vung CUNG phep bien doi
        else: x = s._img(i); hop, co_hop = torch.zeros(1), torch.zeros(1, dtype=torch.bool)
        return dict(image=x, hop=hop, co_hop=co_hop, y=torch.from_numpy(YM[i]), row=i, pos=k)


def loader(idx, tf, shuffle, bs, nw=NW):
    '''loader cua 38a; 78u: worker KHONG persistent -> moi lan lap (moi epoch) worker MOI, hat lay tu dong CHUNG luc bat dau -> luu /
       nap lai dong chung trong rs_ (C6) => resume sau khi kernel chet TRUNG luot lien mach (38a: persistent_workers=nw > 0).'''
    return DataLoader(DS(idx, tf), batch_size=bs, shuffle=shuffle, num_workers=nw,
                      pin_memory=True, drop_last=shuffle,
                      persistent_workers=False, prefetch_factor=4 if nw > 0 else None)



ST_ID = ORDER.study_id.to_numpy().astype('int64')    # 78x: study theo hang ORDER (lo theo study; y B)


def loader_study(idx, tf, bs, nw=NW):
    """78x (Tuan 12/10, y B): nhu loader(idx, tf, True, bs) nhung thu tu = LoTheoStudy (anh CUNG study lien nhau -> cung lo) — CUNG
       o J v2 va doi chung B1 6 epoch (ghep cap). drop_last nhu loader shuffle -> CUNG so lo / epoch, CUNG lich LR."""
    return DataLoader(DS(idx, tf), batch_size=bs, sampler=LoTheoStudy(ST_ID[idx]), num_workers=nw,
                      pin_memory=True, drop_last=True, persistent_workers=False, prefetch_factor=4 if nw > 0 else None)


# ---- lop 38a (NGUYEN VAN ca cell C3 cua 38a) ----
class GroupFC(nn.Module):
    def __init__(s, C, d):
        super().__init__(); s.w = nn.Parameter(torch.randn(C, d) * d ** -0.5); s.b = nn.Parameter(torch.zeros(C))
    def forward(s, h): return (h * s.w).sum(-1) + s.b

class XAttn(nn.Module):
    def __init__(s, d, nhead=8, ff=1024, drop=0.1):
        super().__init__()
        s.at = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
    def forward(s, q, kv, need_w=False):
        a, w = s.at(q, kv, kv, need_weights=need_w, average_attn_weights=True)
        h = s.n1(q + a)
        return s.n2(h + s.ff(h)), w

class MLDecoder(nn.Module):
    '''Q = C query lop (KHONG hoc), K/V = 1369 patch token. Bo self-attention giua cac query
       nen chi phi TUYEN TINH theo so lop: 40 x 1369, khong phai 1369^2 = 1.87M.'''
    def __init__(s, C, d_in, d=512, nhead=8, ff=1024, drop=0.1):
        super().__init__()
        s.q = nn.Embedding(C, d); s.q.weight.requires_grad_(False)
        s.pin = nn.Linear(d_in, d)
        s.img = XAttn(d, nhead, ff, drop)
        s.out = GroupFC(C, d)
    def forward(s, patch, need_w=False):
        q = s.q.weight.unsqueeze(0).expand(patch.shape[0], -1, -1)
        h, w = s.img(q, s.pin(patch), need_w)
        return s.out(h), w

class ASL(nn.Module):
    '''Verbatim nb11/12/13/15. gn=4, gp=0 la cau hinh 35 da chot.'''
    def __init__(s, gn=4, gp=0, clip=0.05, eps=1e-8):
        super().__init__(); s.gn, s.gp, s.clip, s.eps = gn, gp, clip, eps
    def forward(s, x, y):
        y = y.float(); xp = torch.sigmoid(x); xn = 1 - xp
        if s.clip: xn = (xn + s.clip).clamp(max=1)
        lo = y*torch.log(xp.clamp(min=s.eps)) + (1-y)*torch.log(xn.clamp(min=s.eps))
        pt = xp*y + xn*(1-y); g = s.gp*y + s.gn*(1-y)
        return -(lo * torch.pow(1-pt, g)).mean()
print('kien truc san sang')

# ---- dau doc: MLDecoder 38a (query CO DINH), khoi tao tu `mldec` cua 37 (v3_E12_binfix) — tep doc QUA API (38a: glob FUSE) ----
NGUON_37 = ('stage4_nb37', 'v3_E12_binfix', 'ckpt')
torch.manual_seed(SEED)
HEAD = MLDecoder(C, D_CXR, D_MLD).to(dev)
_k37 = kiem_ke(NGUON_37, mau=r'resume\.pt')[1].get('resume.pt') or []
assert _k37, f'thieu resume.pt trong cxrlt_2024/{"/".join(NGUON_37)} -> KHONG khoi tao duoc dau doc tu 37, DUNG'
if xung_dot(_k37): print(f'  ⚠ resume.pt (37): {len(_k37)} ban KHAC noi dung -> dung ban moi nhat ({_k37[0]["mt"]})')
CK37 = dict(id=_k37[0]['id'], md5=_k37[0]['md5'], mt=_k37[0]['mt'])
_sd = torch.load(tai(_k37[0]), map_location='cpu', weights_only=False)['mod']
_h = {k[len('mldec.'):]: v for k, v in _sd.items() if k.startswith('mldec.')}
assert set(_h) == set(HEAD.state_dict()), (
    f'ckpt 37 thieu / thua khoa dau doc ({sorted(set(HEAD.state_dict()) ^ set(_h))[:3]}) -> DUNG (KHONG khoi tao ngau nhien nhu 38a)')
HEAD.load_state_dict(_h); del _sd, _h
crit = ASL(gn=4, gp=0).to(dev)                        # 38a C7
# ---- 78x dot 4 B.4 (Tuan duyet 21/10; SPEC B.2n): stochastic depth theo block backbone, p_i = dp x i / (L - 1); mat na tu generator RIENG
#      (ngoai checkpoint) -> dong ngau nhien chung (thu tu lo, dropout dau doc) KHONG doi; C6 gieo lai GEN_DP luc bat dau train ----
TI_LE_DP = ti_le_dp(len(bb_ft.encoder.layer), float(CFG_D4['dp_ema']['dp']))
GEN_DP = torch.Generator(device=dev); GEN_DP.manual_seed(SEED * 1000 + int(CFG_D4['dp_ema']['hat_dp']))
print(f"78x dot 4 B.4: stochastic depth {len(TI_LE_DP)} block, p tu {TI_LE_DP[0]:.3f} toi {TI_LE_DP[-1]:.3f} (TB "
      f"{sum(TI_LE_DP) / len(TI_LE_DP):.3f}) | EMA {CFG_D4['dp_ema']['ema']}")
print(f"dau doc mldec 37: {'/'.join(NGUON_37)}/resume.pt (md5 {CK37['md5']}) | {sum(p.numel() for p in HEAD.parameters()):,} tham so")


def dau_h(m, patch):
    '''78u: MOT forward dau doc 38a -> (logit, h = 40 dau ra query TRUOC GroupFC, kv = pin(patch)). CUNG phep tinh voi
       MLDecoder.forward -> logit TRUNG tung bit (cong duoi: eval + train cung dong ngau nhien). kv: attention lop -> patch (B3).'''
    q = m.q.weight.unsqueeze(0).expand(patch.shape[0], -1, -1)
    kv = m.pin(patch)
    h, _ = m.img(q, kv, False)
    return m.out(h), h, kv


# ---- cong (38a C4 + 78u) ----
assert all(p.requires_grad for p in bb_ft.parameters()), 'bb_ft phai hoc'
assert not HEAD.q.weight.requires_grad, 'query_embed dang hoc (thiet ke bai goc)'
_b = next(iter(loader(I_VA[:4], eval_tf, False, 4, nw=0)))
_x = _b['image'].to(dev)
_p, _cl = fwd_ft(_x, False)
N_TOK = _p.shape[1]
assert N_TOK == (SIZE // bb_ft.config.patch_size) ** 2 and tuple(_cl.shape) == (len(_x), D_CXR), (N_TOK, tuple(_cl.shape))
assert torch.equal(_p, fwd(bb_ft, _x, False)), 'fwd_ft lech fwd cua 38a -> DUNG'
with torch.no_grad():
    HEAD.eval()
    _l1 = HEAD(_p)[0]; _l2, _h2, _k2 = dau_h(HEAD, _p)
    assert torch.equal(_l1, _l2) and tuple(_h2.shape) == (len(_x), C, D_MLD), 'dau_h lech MLDecoder.forward (eval) -> DUNG'
    _qr = HEAD.q.weight.unsqueeze(0).expand(len(_x), -1, -1)    # KHONG ten `_q`: ham cua khoi Drive (C0)
    _wm = HEAD.img.at(_qr, _k2, _k2, need_weights=True, average_attn_weights=True)[1]
    assert torch.allclose(attn_truoc_dropout(HEAD.img.at, _qr, _k2), _wm, atol=1e-5), (
        'attention tinh lai (khoi_loss) lech MHA cua dau doc 38a -> viet lai theo lop cua no (plan 6.7), DUNG')
    with _fork():
        HEAD.train()
        torch.manual_seed(5); _l1 = HEAD(_p)[0]
        torch.manual_seed(5); _l2 = dau_h(HEAD, _p)[0]
    assert torch.equal(_l1, _l2), 'dau_h lech MLDecoder.forward o che do train -> DUNG'
HEAD.eval()
del _b, _x, _p, _cl, _l1, _l2, _h2, _k2, _qr, _wm
print(f'cong: {N_TOK} token patch | fwd_ft = fwd 38a tung bit | dau_h = MLDecoder.forward tung bit (eval + train) | attention '
      'tinh lai truoc dropout = MHA (eval)')
