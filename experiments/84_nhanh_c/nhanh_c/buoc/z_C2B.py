# ==== nhanh_c buoc z_C2B — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C2b · 78y: LUONG VUNG J v3 DONG CUNG (CHI luot J_v3; luot B1_e6: LUONG = None) — kien truc + g lay TRONG checkpoint; 4 cong ====
# Khoi nhung NGUYEN VAN (cuoi cell): khoi_luong_vung (79, CHI DOC), khoi_jca_v2, khoi_jca_v3, khoi_doc_dong_cung (78y).
# CONG (truot -> DUNG, KHONG train dau doc tren dac trung sai):
#   (1) backbone_ft.pt CUNG luot (run, luot_79a J_bert_dong) co cau hinh luong (cfg_luong / cfg_jv2 / cfg_jv3); hop_tb trung rs_ ep4;
#   (2) dung LuongVungNeo DUNG cau hinh do + write-back x g (cfg_jv3['ghi_nguoc_g']), nap state_dict luong cua rs_ ep4 STRICT;
#   (3) duong CHIA DOI (block 0-7 roi block 8-11 + luong: upper_luong_hr) = forward DAY DU chay_bb_luong tren 8 anh val (patch + token
#       vung; < 1e-3 nhu cong 1 cua C2);
#   (4) ti le ghi nguoc tung lop JCA cung co voi nhat ky ep4 cua 78x (0.33x - 3x) -> bat g / cau hinh sai (g sai 5 lan -> truot).
# Neo theo view = view_group (CO SAN luc cham; doan P_VIEW / V NGUYEN VAN 38a): frontal = neo (nhu 78x).

# ---- khoi 79 (NGUYEN VAN experiments/79_phat_minh/khoi/khoi_luong_vung.py, CHI DOC) ----
import torch.utils.checkpoint as _tuc_luong
from torchvision.ops import generalized_box_iou_loss as _giou_luong


def _out_luong(o):
    '''Dau ra mot block Dinov2: tuple / list (transformers cu) -> phan tu dau; tensor (transformers >= 5) -> giu nguyen.'''
    return o[0] if isinstance(o, (tuple, list)) else o


def _ma_hoa_hop(hop, n=32):
    '''Hop [..., 4] (cx, cy, w, h) chuan hoa -> sin / cos [..., 8n] (nhu DAB-DETR, nhiet do 10000). Khong tham so, khong ngau
       nhien; float32.'''
    t = 10000.0 ** (torch.arange(n, device=hop.device, dtype=torch.float32) / n)
    x = hop.float()[..., None] * (2.0 * math.pi) / t
    return torch.cat([x.sin(), x.cos()], -1).flatten(-2)


def mat_na_hop(hop, luoi=37, t=0.5):
    '''Hop (cx, cy, w, h) chuan hoa [..., R, 4] -> mat na mem [..., R, luoi * luoi] trong (0, 1):
       m = s((x - x1) / t) s((x2 - x) / t) s((y - y1) / t) s((y2 - y) / t), s = sigmoid, (x, y) = tam o luoi, moi do dai theo DON VI
       O LUOI (t = 0.5 o). Thu tu o theo HANG (chi so y * luoi + x), dung thu tu token patch cua Dinov2. float32.'''
    c = torch.arange(luoi, device=hop.device, dtype=torch.float32) + 0.5
    hp = hop.float() * luoi
    x1, x2 = hp[..., 0] - hp[..., 2] / 2, hp[..., 0] + hp[..., 2] / 2
    y1, y2 = hp[..., 1] - hp[..., 3] / 2, hp[..., 1] + hp[..., 3] / 2
    mx = torch.sigmoid((c - x1[..., None]) / t) * torch.sigmoid((x2[..., None] - c) / t)
    my = torch.sigmoid((c - y1[..., None]) / t) * torch.sigmoid((y2[..., None] - c) / t)
    return (my[..., :, None] * mx[..., None, :]).flatten(-2)


class CrossAttnBias(nn.Module):
    '''Cross-attention nhieu dau co thien lech cong: q [B, Lq, d], kv [B, Lk, d], bias [B, 1, Lq, Lk] | None -> [B, Lq, d].
       q / k / v / o = Linear(d, d); F.scaled_dot_product_attention; KHONG dropout. zero_out=True -> o.weight = o.bias = 0 (dau
       ra DUNG 0 luc khoi tao). trong_so(...) = attention TB cac dau [B, Lq, Lk] (chan doan / kiem).'''
    def __init__(s, d, n_head, zero_out=False):
        super().__init__()
        assert d % n_head == 0, (d, n_head)
        s.nh, s.dh = int(n_head), d // n_head
        s.q, s.k, s.v, s.o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        if zero_out:
            nn.init.zeros_(s.o.weight); nn.init.zeros_(s.o.bias)

    def _qkv(s, q, kv):
        B, Lq, d = q.shape; Lk = kv.shape[1]
        Q = s.q(q).view(B, Lq, s.nh, s.dh).transpose(1, 2)
        K = s.k(kv).view(B, Lk, s.nh, s.dh).transpose(1, 2)
        V = s.v(kv).view(B, Lk, s.nh, s.dh).transpose(1, 2)
        return Q, K, V

    def forward(s, q, kv, bias=None):
        Q, K, V = s._qkv(q, kv)
        a = F.scaled_dot_product_attention(Q, K, V, attn_mask=None if bias is None else bias.to(Q.dtype), dropout_p=0.0)
        return s.o(a.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1))

    @torch.no_grad()
    def trong_so(s, q, kv, bias=None):
        Q, K, _ = s._qkv(q, kv)
        S = (Q.float() @ K.float().transpose(-1, -2)) / math.sqrt(s.dh)
        if bias is not None: S = S + bias.float()
        return S.softmax(-1).mean(1)


class LopJCA(nn.Module):
    '''MOT lop JCA sau block l. forward(h [B, 1 + N, d], r [B, R, d], hop [B, R, 4], p [B, R, d] ma hoa vi tri cua hop, mo_ghi)
       -> (h', r', hop', ti): ti = ||dh|| / ||h|| (0-d, khong grad) khi mo_ghi, None khi khong.
         r1 = r + SA(LN(r + p));  r2 = r1 + CA_r(LN(r1 + p), LN(h[:, 1:]), bias_r);  r3 = r2 + FFN(LN(r2))
         mo_ghi: h' = h + CA_h(LN(h), LN(r + p), bias_h)   (r CU: cap nhat dong thoi; CA_h.o khoi tao 0)
         neo: hop' = sigmoid(logit(hop) + MLP_hop(r3))      (lop cuoi MLP_hop khoi tao 0)
       bias_r = -beta (1 - mat_na_hop(hop.detach())) [B, 1, R, N] (None neu khong neo); bias_h = bias_r chuyen vi, them hang 0 cho CLS.'''
    def __init__(s, d, n_head, ff, beta, t_mem, luoi, neo):
        super().__init__()
        s.beta, s.t_mem, s.luoi, s.neo = float(beta), float(t_mem), int(luoi), bool(neo)
        s.ln_sa, s.sa = nn.LayerNorm(d), CrossAttnBias(d, n_head)
        s.ln_r, s.ln_h, s.ca_r = nn.LayerNorm(d), nn.LayerNorm(d), CrossAttnBias(d, n_head)
        s.ln_ff, s.ff = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        s.ln_h2, s.ln_r2, s.ca_h = nn.LayerNorm(d), nn.LayerNorm(d), CrossAttnBias(d, n_head, zero_out=True)
        if s.neo:
            s.mlp_hop = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
            nn.init.zeros_(s.mlp_hop[-1].weight); nn.init.zeros_(s.mlp_hop[-1].bias)

    def bias_vung(s, hop):
        if not s.neo: return None
        return (-s.beta * (1.0 - mat_na_hop(hop.detach(), s.luoi, s.t_mem)))[:, None]

    def forward(s, h, r, hop, p, mo_ghi):
        b_r = s.bias_vung(hop)
        x = s.ln_sa(r + p)
        r1 = r + s.sa(x, x)
        r2 = r1 + s.ca_r(s.ln_r(r1 + p), s.ln_h(h[:, 1:]), b_r)
        r3 = r2 + s.ff(s.ln_ff(r2))
        ti = None
        if mo_ghi:
            bh = None if b_r is None else torch.cat(
                [b_r.new_zeros(b_r.shape[0], 1, 1, b_r.shape[2]), b_r.transpose(-1, -2)], 2)
            dh = s.ca_h(s.ln_h2(h), s.ln_r2(r + p), bh)
            with torch.no_grad():
                ti = dh.float().norm() / h.float().norm().clamp_min(1e-12)
            h = h + dh
        if s.neo:
            hop = torch.sigmoid(torch.logit(hop.float().clamp(1e-4, 1.0 - 1e-4)) + s.mlp_hop(r3).float())
        return h, r3, hop, ti


class LuongVung(nn.Module):
    '''Luong vung: R token + MOT LopJCA cho moi block trong `tang`. hop_tb [R, 4] (cx, cy, w, h) chuan hoa = buffer (luu trong
       state_dict). e [R, d] ~ N(0, 0.02); pos = MLP(_ma_hoa_hop) DUNG CHUNG moi lop. ckpt=True: tinh lai kich hoat cua tung lop
       (torch.utils.checkpoint, use_reentrant=False) khi train co grad — khong doi ket qua (khong dropout, khong ngau nhien).
       mo_ghi thuc = mo_ghi and hai_chieu.
         st = bat_dau(B); h = buoc(l, h, st, mo_ghi) sau moi block l (l ngoai tang -> tra h nguyen); dau_ra(st) ->
         dict(r [B, R, d], hop_cac_tang [L, B, R, 4] | None, ghi = list ti tung lop)'''
    def __init__(s, hop_tb, d=768, R=36, tang=(8, 9, 10, 11), n_head=12, ff=1536, beta=4.0, t_mem=0.5, hai_chieu=True,
                 neo=True, luoi=37, ckpt=True, n_pos=32):
        super().__init__()
        hop_tb = torch.as_tensor(np.asarray(hop_tb, dtype='float32'))
        assert tuple(hop_tb.shape) == (int(R), 4) and bool(((hop_tb > 0) & (hop_tb <= 1)).all()), \
            f'hop_tb phai [R, 4] trong (0, 1]: {tuple(hop_tb.shape)}'
        s.R, s.tang, s.hai_chieu, s.ckpt, s.n_pos = int(R), tuple(int(t) for t in tang), bool(hai_chieu), bool(ckpt), int(n_pos)
        assert len(s.tang) >= 1 and list(s.tang) == sorted(set(s.tang)), f'tang phai tang ngat: {s.tang}'
        s._vt = {t: i for i, t in enumerate(s.tang)}
        s.register_buffer('hop_tb', hop_tb.clone())
        s.e = nn.Parameter(torch.randn(s.R, d) * 0.02)
        s.pos = nn.Sequential(nn.Linear(8 * s.n_pos, d), nn.GELU(), nn.Linear(d, d))
        s.lop = nn.ModuleList([LopJCA(d, n_head, ff, beta, t_mem, luoi, neo) for _ in s.tang])

    def bat_dau(s, B):
        return dict(r=s.e[None].expand(int(B), -1, -1), hop=s.hop_tb[None].expand(int(B), -1, -1),
                    hop_cac_tang=[], ghi=[])

    def buoc(s, l, h, st, mo_ghi):
        i = s._vt.get(int(l))
        if i is None: return h
        lop, mg = s.lop[i], bool(mo_ghi and s.hai_chieu)
        p = s.pos(_ma_hoa_hop(st['hop'].detach(), s.n_pos))
        if s.ckpt and s.training and torch.is_grad_enabled():
            h, r, hop, ti = _tuc_luong.checkpoint(lop, h, st['r'], st['hop'], p, mg, use_reentrant=False)
        else:
            h, r, hop, ti = lop(h, st['r'], st['hop'], p, mg)
        st['r'], st['hop'] = r, hop
        st['hop_cac_tang'].append(hop)
        if ti is not None: st['ghi'].append(ti)
        return h

    def dau_ra(s, st):
        return dict(r=st['r'], hop_cac_tang=(torch.stack(st['hop_cac_tang']) if st['hop_cac_tang'] else None), ghi=st['ghi'])


class PhuVung(nn.Module):
    '''Phia ANH cua dich vung (KHONG chua thap van ban — thap la PHU.text cua B1 / B2, dung chung, tranh tham so trung nhom
       optimizer): anh = ChieuAnh(d, d_an, d_chieu) (nhu PhuCau.anh), nhiet = NhietDo(tau0) (RIENG voi cau study), bt = Linear(d, 1).
       z_anh(r) [..., d_chieu]; tau(); bt_logit(r) [...]. Khoi tao TIEU dong chung -> tao trong fork_rng voi hat rieng.'''
    def __init__(s, d=768, d_an=512, d_chieu=256, tau0=0.07):
        super().__init__()
        s.anh = ChieuAnh(d, d_an=d_an, d_chieu=d_chieu); s.nhiet = NhietDo(tau0); s.bt = nn.Linear(d, 1)

    def z_anh(s, r):
        return s.anh(r.float())[0]

    def tau(s):
        return s.nhiet.tau()

    def bt_logit(s, r):
        return s.bt(r.float())[..., 0]


def chay_bb_luong(bb, luong, img, mo_ghi=True, grad_bb=True, ckpt_bb=()):
    '''Forward Dinov2 `bb` (embeddings -> encoder.layer 0..L-1 -> layernorm, = bb(pixel_values).last_hidden_state) voi luong chen
       sau cac block trong luong.tang -> (patch [B, N, d] float32, cls [B, d] float32, luong.dau_ra). Autocast / grad do noi goi
       dat. grad_bb=False: block chay no_grad (luong van hoc) — CHI cho phep khi duong ghi nguoc dong (neu khong, grad cua luong
       qua cac block sau bi cat am tham). Chi goi layer(h): khong phu thuoc ten module con (transformers >= 5.18 doi ten).
       ckpt_bb: chi so block backbone TINH LAI kich hoat luc backward (torch.utils.checkpoint, use_reentrant=False) khi co grad —
       giam VRAM, KHONG doi ket qua (block Dinov2 cua RAD-DINO khong dropout; checkpoint giu nguyen trang thai RNG + autocast).'''
    g = torch.is_grad_enabled(); gb = g and bool(grad_bb); ck = {int(i) for i in ckpt_bb}
    assert grad_bb or not g or not (mo_ghi and luong.hai_chieu), \
        'chay_bb_luong: backbone dong (grad_bb=False) ma duong ghi nguoc mo -> grad cua luong bi cat, DUNG'
    with torch.set_grad_enabled(gb):
        h = bb.embeddings(img)
    st = luong.bat_dau(img.shape[0])
    for l, lop in enumerate(bb.encoder.layer):
        with torch.set_grad_enabled(gb):
            h = _out_luong(_tuc_luong.checkpoint(lop, h, use_reentrant=False) if (gb and l in ck) else lop(h))
        h = luong.buoc(l, h, st, mo_ghi)
    with torch.set_grad_enabled(gb):
        h = bb.layernorm(h)
    return h[:, 1:].float(), h[:, 0].float(), luong.dau_ra(st)


@torch.no_grad()
def upper_luong(bb, luong, h8, mo_ghi=True):
    '''Doc dong cung (79b): h8 = dau ra block luong.tang[0] - 1 (shared_lo cua 78v) -> block tang[0]..L-1 + luong + layernorm
       -> patch [B, N, d] float32. mo_ghi=False -> TRUNG upper_frozen goc cua 78v (cung block, cung layernorm).'''
    st = luong.bat_dau(h8.shape[0]); h = h8
    L = bb.encoder.layer
    for l in range(luong.tang[0], len(L)):
        h = _out_luong(L[l](h))
        h = luong.buoc(l, h, st, mo_ghi)
    return bb.layernorm(h)[:, 1:].float()


def cxcywh_sang_xyxy(b):
    c = b.float()
    return torch.stack([c[..., 0] - c[..., 2] / 2, c[..., 1] - c[..., 3] / 2, c[..., 0] + c[..., 2] / 2, c[..., 1] + c[..., 3] / 2], -1)


def xyxy_sang_cxcywh(b):
    c = b.float()
    return torch.stack([(c[..., 0] + c[..., 2]) / 2, (c[..., 1] + c[..., 3]) / 2, c[..., 2] - c[..., 0], c[..., 3] - c[..., 1]], -1)


def hop_tb(HOP518, SG_HANG, POOL, R, SIZE):
    '''Hop khoi dau cua luong: TB hop HOP518 [n_anh_78f, R, 4] (x1, y1, x2, y2 toa do anh SIZE; NaN = khong co) tren anh POOL
       (CHI train, khong giu rieng) co scene graph (SG_HANG >= 0) -> [R, 4] float32 (cx, cy, w, h) chuan hoa, kep [1e-3, 1].
       Vung khong co mau nao -> hop ca anh (0.5, 0.5, 1, 1). numpy, CPU.'''
    g = np.asarray(SG_HANG)[np.asarray(POOL)]; g = g[g >= 0]
    H = np.asarray(HOP518); out = np.tile(np.array([0.5, 0.5, 1.0, 1.0], 'float64'), (int(R), 1))
    for k in range(int(R)):
        v = H[g, k]; v = v[np.isfinite(v).all(1)]
        if len(v):
            x1, y1, x2, y2 = v.mean(0) / float(SIZE)
            out[k] = ((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1e-3), max(y2 - y1, 1e-3))
    return np.clip(out, 1e-3, 1.0).astype('float32')


def loss_hop(hop_cac_tang, gt_xyxy, co, SIZE):
    '''Hop moi tang: L1 (cx, cy, w, h chuan hoa) + (1 - GIoU), TB tren tang x cap hop le. hop_cac_tang [L, B, R, 4]; gt_xyxy
       [B, R, 4] toa do anh SIZE (DA bien doi cung anh, TF_HOP); co [B, R] bool. Khong cap hop le -> None.'''
    co = co.bool()
    if hop_cac_tang is None or not bool(co.any()): return None
    g = gt_xyxy.float() / float(SIZE)
    p = hop_cac_tang.float()[:, co]
    gx = g[co]
    l1 = (p - xyxy_sang_cxcywh(gx)[None]).abs().sum(-1).mean()
    gi = _giou_luong(cxcywh_sang_xyxy(p).reshape(-1, 4), gx[None].expand(p.shape[0], -1, -1).reshape(-1, 4), reduction='mean')
    return l1 + gi


def loss_bt(logit, y):
    '''Co bat thuong tung vung: BCE tren y in {0, 1}; y = -1 (khong biet) bi bo. Khong phan tu -> None.'''
    m = y >= 0
    if not bool(m.any()): return None
    return F.binary_cross_entropy_with_logits(logit.float()[m], y[m].float())


def cap_vung_lo(rows, SG_HANG_T, POOL_T, PTR_V, R):
    '''Cap (anh trong lo, vung) vao loss cau vung: anh POOL co scene graph, vung co >= 1 cau (CSR PTR_V theo cap = g * R + vung).
       -> (b_idx, vung, cap) long [Np]; khong cap -> tensor rong.'''
    R = int(R)
    g = SG_HANG_T[rows]; hl = (g >= 0) & POOL_T[rows]
    b = torch.nonzero(hl, as_tuple=True)[0]
    vr = torch.arange(R, device=rows.device)
    cap = g[b][:, None] * R + vr[None, :]
    co = PTR_V[cap + 1] > PTR_V[cap]
    return b[:, None].expand(-1, R)[co], vr[None, :].expand(b.numel(), -1)[co], cap[co]


def mat_na_loai_vung(P, sub, anh):
    '''loai bool [Np, U] = ~P VA co cap b' CUNG benh nhan, KHAC anh, P[b', u]: cau vung cua anh khac cung benh nhan (benh man
       tinh cung vung) KHONG lam am. Cau cua VUNG KHAC trong CUNG anh van la am (day token vung phan biet vung).'''
    P = P.bool()
    A = (sub[:, None] == sub[None, :]) & (anh[:, None] != anh[None, :])
    with torch.autocast(device_type=P.device.type, enabled=False):
        E = (A.float() @ P.float()) > 0
    return E & ~P


def dich_vung_cau(cap, sub, anh, PTR_V, IDX_V, toi_da, gen):
    '''Dich InfoNCE cau vung cho cac cap trong lo: ung vien = lay_cau_lo (khoi_cau; "study" = cap, ngau nhien CHI tu `gen`),
       duong = ma_tran_duong (cau trung chu = MOT ung vien, duong voi moi cap co no), loai = mat_na_loai_vung.
       -> (ma_u [U], P [Np, U], loai [Np, U]).'''
    ma_u = lay_cau_lo(cap, PTR_V, IDX_V, toi_da, gen)
    P = ma_tran_duong(cap, PTR_V, IDX_V, ma_u)
    return ma_u, P, mat_na_loai_vung(P, sub, anh)


def token_hoa_cau(TOK, texts, max_tok=64, lo=4096):
    '''DUNG cach 78s lam cau_token: TOK(lo, truncation=True, max_length=max_tok)['input_ids'] -> int32 [n, max_tok], [CLS] ... [SEP],
       dem 0 ben PHAI. Tra token (khong giu van ban) + so cau bi cat.'''
    assert TOK.pad_token_id == 0, f'pad id cua tokenizer = {TOK.pad_token_id} != 0 (ThapBert dem 0 ben phai) -> DUNG'
    out = np.zeros((len(texts), int(max_tok)), 'int32'); n_cat = 0
    for i in range(0, len(texts), int(lo)):
        for r, x in enumerate(TOK(list(texts[i:i + int(lo)]), truncation=True, max_length=int(max_tok))['input_ids']):
            out[i + r, :len(x)] = x; n_cat += int(len(x) >= int(max_tok))
    return out, n_cat


@torch.no_grad()
def iou_cap(a_xyxy, b_xyxy):
    '''IoU tung cap hop [..., 4] (x1, y1, x2, y2) — chan doan.'''
    a, b = a_xyxy.float(), b_xyxy.float()
    iw = (torch.minimum(a[..., 2], b[..., 2]) - torch.maximum(a[..., 0], b[..., 0])).clamp_min(0)
    ih = (torch.minimum(a[..., 3], b[..., 3]) - torch.maximum(a[..., 1], b[..., 1])).clamp_min(0)
    gi = iw * ih
    dt = lambda x: (x[..., 2] - x[..., 0]).clamp_min(0) * (x[..., 3] - x[..., 1]).clamp_min(0)
    return gi / (dt(a) + dt(b) - gi).clamp_min(1e-9)

# ---- khoi 78x (NGUYEN VAN experiments/78_chi_anh/khoi/khoi_jca_v2.py) ----


class LopJCANeo(LopJCA):
    '''LopJCA (79, CHI DOC) + neo THEO MAU. forward(h, r, hop, p, mo_ghi, neo [B] float 0 / 1 | None): neo = 1 -> Y HET LopJCA;
       neo = 0 -> thien lech vung (ca bias_r lan bias_h) = 0 va hop GIU nguyen. neo la DOI SO -> torch.utils.checkpoint tinh lai dung
       mau. KHONG them tham so (LuongVungNeo doi lop cac LopJCA da tao).'''
    _neo = None

    def bias_vung(s, hop):
        b = super().bias_vung(hop)
        if b is None or s._neo is None: return b
        return b * s._neo.to(b.dtype)[:, None, None, None]

    def forward(s, h, r, hop, p, mo_ghi, neo=None):
        s._neo = None if neo is None else neo.float()
        try:
            h2, r2, hop2, ti = super().forward(h, r, hop, p, mo_ghi)
        finally:
            s._neo = None
        if neo is not None and s.neo:
            hop2 = torch.where(neo.bool()[:, None, None], hop2, hop.to(hop2.dtype))
        return h2, r2, hop2, ti


class LuongVungNeo(LuongVung):
    '''LuongVung (79, CHI DOC) voi lop LopJCANeo — DOI LOP cac LopJCA vua tao (khoi tao TRUNG LuongVung cung hat, khong rut them so
       ngau nhien, state_dict cung khoa). dat_neo(neo [B]) NGAY TRUOC moi forward (chay_bb_luong / upper_luong goi bat_dau): bat_dau LAY
       va XOA -> quen dat thi DUNG, khong bao gio dung neo cua lo truoc. Moi mau neo = 1 -> TRUNG LuongVung tung bit.'''
    def __init__(s, hop_tb, **kw):
        super().__init__(hop_tb, **kw)
        for l in s.lop: l.__class__ = LopJCANeo
        s._neo_lo = None

    def dat_neo(s, neo):
        s._neo_lo = neo

    def bat_dau(s, B):
        neo, s._neo_lo = s._neo_lo, None
        assert neo is not None, 'LuongVungNeo: chua dat_neo(neo) truoc forward -> DUNG (khong dung neo cua lo truoc)'
        assert tuple(neo.shape) == (int(B),), f'neo {tuple(neo.shape)} != ({int(B)},)'
        st = super().bat_dau(B)
        st['neo'] = neo.to(st['hop'].device).float()
        return st

    def buoc(s, l, h, st, mo_ghi):
        '''= LuongVung.buoc + truyen st['neo'] vao lop (ca nhanh checkpoint).'''
        i = s._vt.get(int(l))
        if i is None: return h
        lop, mg = s.lop[i], bool(mo_ghi and s.hai_chieu)
        p = s.pos(_ma_hoa_hop(st['hop'].detach(), s.n_pos))
        if s.ckpt and s.training and torch.is_grad_enabled():
            h, r, hop, ti = _tuc_luong.checkpoint(lop, h, st['r'], st['hop'], p, mg, st['neo'], use_reentrant=False)
        else:
            h, r, hop, ti = lop(h, st['r'], st['hop'], p, mg, st['neo'])
        st['r'], st['hop'] = r, hop
        st['hop_cac_tang'].append(hop)
        if ti is not None: st['ghi'].append(ti)
        return h


class DocVung(nn.Module):
    '''Y 1: token vung -> khoa / gia tri THEM cho ML-Decoder: pin(LN(r)) + e_loai + e_vung[k] (pin = phep chieu patch CUA dau doc;
       e_* khoi tao 0). forward(pin, r [B, R, d_in]) -> [B, R, d]. Khong ngau nhien.'''
    def __init__(s, R, d_in, d):
        super().__init__()
        s.ln = nn.LayerNorm(d_in)
        s.e_loai = nn.Parameter(torch.zeros(d))
        s.e_vung = nn.Parameter(torch.zeros(int(R), d))

    def forward(s, pin, r):
        return pin(s.ln(r.float())) + s.e_loai + s.e_vung


def dau_h_v2(m, patch, kv_vung=None):
    '''dau_h cua 78u (MLDecoder 38a) + y 1: K/V = [pin(patch) ; kv_vung] -> (logit, h = 40 dau ra query TRUOC GroupFC, kv = pin(patch)
       CHI phan patch). kv_vung None -> TRUNG dau_h tung bit.'''
    q = m.q.weight.unsqueeze(0).expand(patch.shape[0], -1, -1)
    kv = m.pin(patch)
    h, _ = m.img(q, kv if kv_vung is None else torch.cat([kv, kv_vung.to(kv.dtype)], 1), False)
    return m.out(h), h, kv


class DauVung(nn.Module):
    '''Y 2 + kenh MIL. z(r) [B, R, C] = Linear(LN(r)) = logit "nhan j o vung k". mil(logit, z): logit_j + w_j * max_k z[:, k, j] CHI
       voi j trong mil_idx (w khoi tao 0 -> luc dau TRUNG logit classifier); nhan ngoai mil_idx KHONG doi.'''
    def __init__(s, d_in, C, mil_idx):
        super().__init__()
        s.ln, s.lin = nn.LayerNorm(d_in), nn.Linear(d_in, int(C))
        mi = sorted({int(i) for i in mil_idx})
        assert all(0 <= i < int(C) for i in mi), mi
        s.register_buffer('mil_idx', torch.as_tensor(mi, dtype=torch.long))
        s.w = nn.Parameter(torch.zeros(len(mi)))

    def z(s, r):
        return s.lin(s.ln(r.float()))

    def mil(s, logit, z):
        if not s.mil_idx.numel(): return logit
        return logit.index_add(1, s.mil_idx, (s.w * z[:, :, s.mil_idx].amax(1)).to(logit.dtype))


def mat_na_s(anh, vung, khoa, thay, khoa_nhan, nhan_thay, n_anh, C):
    '''Thuoc tinh 'yes' cua 78f (thuoc_tinh_train.parquet: anh, vung, khoa "loai|gia tri", thay) -> M [n_anh, C] int64 mat na bit tap
       vung S (§4.1): vung CO HOP THAT (thay == 0) cho MOI nhan; them vung THAY THE (thay == 1: chest wall -> phoi cung ben, neck ->
       trachea + upper mediastinum, shoulder -> clavicle) CHI cho nhan trong `nhan_thay` (chi so; Tuan 12/10). khoa_nhan: khoa ->
       [chi so nhan]; khoa khong anh xa bi bo. nhan_thay rong -> = mat_na_vi_tri_that cua khoi_muc_tieu (78q).'''
    anh = np.asarray(anh, 'int64'); vung = np.asarray(vung, 'int64'); khoa = np.asarray(khoa, dtype=object); thay = np.asarray(thay)
    M = np.zeros((int(n_anh), int(C)), 'int64'); nt = {int(j) for j in nhan_thay}
    for k, js in khoa_nhan.items():
        mk = khoa == k
        if not mk.any(): continue
        for j in js:
            g = mk & ((thay == 0) | (int(j) in nt))
            np.bitwise_or.at(M, (anh[g], np.full(int(g.sum()), int(j), 'int64')), np.left_shift(np.int64(1), vung[g]))
    return M


def cay_vung(VUNG, cay):
    '''cay {cha: [con, ...]} (ten vung) -> (CHA [R] int64: chi so cha hoac -1, CON [R] int64: mat na bit cac con). Moi con MOT cha;
       ten ngoai VUNG -> loi.'''
    idx = {v: i for i, v in enumerate(VUNG)}
    R = len(VUNG); CHA = np.full(R, -1, 'int64'); CON = np.zeros(R, 'int64')
    for c, ds in cay.items():
        for x in ds:
            assert CHA[idx[x]] == -1, f'vung {x} co hai cha'
            CHA[idx[x]] = idx[c]; CON[idx[c]] |= np.int64(1) << np.int64(idx[x])
    return CHA, CON


def mo_rong_cha_con(M, CHA, CON):
    '''M [...] int64 mat na bit tap vung S cua (anh, nhan) -> (M1, MC):
       M1 = S | cha(S) (vung CHUA ton thuong -> 1; ImaGenome tu lan con -> phoi ~98%, day la phan con lai);
       MC = cac con cua vung c thuoc M1 KHI S KHONG co con nao cua c (report CHI nhac muc phoi -> chua biet zone nao -> mask).
       S co >= 1 con cua c -> cac con khac cua c giu 0 (report da khu tru).'''
    M = np.asarray(M, 'int64'); M1 = M.copy(); MC = np.zeros_like(M)
    for k in range(len(CHA)):
        if CHA[k] >= 0:
            M1 = np.where(((M >> np.int64(k)) & 1) == 1, M1 | (np.int64(1) << np.int64(CHA[k])), M1)
    for c in range(len(CON)):
        if CON[c]:
            MC = np.where((((M1 >> np.int64(c)) & 1) == 1) & ((M & CON[c]) == 0), MC | CON[c], MC)
    return M1, MC & ~M1


def dich_nhan_vung(y, g, M1_T, MC_T, mil_T, R):
    '''Target y 2 cho MOT lo. y [B, C] {0, 1}; g [B] long: anh 78f mang ngu nghia vung cua STUDY (y A) hoac -1; M1_T / MC_T [n_anh, C]
       int64; mil_T [C] bool -> (T [B, R, C] float, V [B, R, C] bool):
       nhan am -> T 0, V moi vung; duong + M1 = 0 (chua biet vi tri) -> V False; duong + M1 != 0 -> T = 1 tren M1, V False tren MC;
       g < 0 hoac nhan ngoai MIL -> V False.'''
    co = g >= 0
    gg = g.clamp_min(0)
    m1, mc = M1_T[gg], MC_T[gg]
    bit = torch.bitwise_left_shift(torch.ones(int(R), dtype=torch.long, device=y.device), torch.arange(int(R), device=y.device))
    in1 = (m1[:, None, :] & bit[None, :, None]) != 0
    inc = (mc[:, None, :] & bit[None, :, None]) != 0
    duong = (y > 0.5)[:, None, :]
    T = (duong & in1).float()
    Vm = (~duong | ((m1 != 0)[:, None, :] & ~inc)) & co[:, None, None] & mil_T.bool()[None, None, :]
    return T, Vm


def asl_mat_na(x, y, v, gn=4.0, gp=0.0, clip=0.05, eps=1e-8):
    '''ASL (cong thuc NGUYEN VAN lop ASL cua 38a) tung phan tu, TB tren phan tu v = True. Khong phan tu nao -> None.'''
    if not bool(v.any()): return None
    x, y = x.float()[v], y.float()[v]
    xp = torch.sigmoid(x); xn = 1 - xp
    if clip: xn = (xn + clip).clamp(max=1)
    lo = y * torch.log(xp.clamp(min=eps)) + (1 - y) * torch.log(xn.clamp(min=eps))
    pt = xp * y + xn * (1 - y); g = gp * y + gn * (1 - y)
    return -(lo * torch.pow(1 - pt, g)).mean()


def loss_nhat_quan(logit, st, frontal):
    '''Y B: logit [B, C] (sau MIL); st [B] long (study); frontal [B] bool (view_group frontal). Thay = TB sigmoid logit cac anh frontal
       CUNG study trong lo (detach); tro = anh KHONG frontal ma study co frontal trong lo. -> BCE TB (tro x C); khong cap -> None.'''
    fr = frontal.bool()
    assert logit.shape[0] == st.shape[0] == fr.shape[0], f'loss_nhat_quan: lo lech {logit.shape[0]} / {st.shape[0]} / {fr.shape[0]}'
    u, inv = torch.unique(st, return_inverse=True)
    pf = torch.sigmoid(logit.detach().float())
    s = torch.zeros(len(u), logit.shape[1], device=logit.device).index_add_(0, inv[fr], pf[fr])
    n = torch.zeros(len(u), device=logit.device).index_add_(0, inv[fr], torch.ones(int(fr.sum()), device=logit.device))
    tro = ~fr & (n[inv] > 0)
    if not bool(tro.any()): return None
    return F.binary_cross_entropy_with_logits(logit[tro].float(), s[inv[tro]] / n[inv[tro]][:, None])

# ---- khoi 78x (NGUYEN VAN experiments/78_chi_anh/khoi/khoi_jca_v3.py) ----


class NguonNhan(nn.Module):
    '''Y C: offset cho K nguon / lop tren h [B, C, d] (dau ra query cua ML-Decoder, TRUOC GroupFC). delta(h) [B, C, K] = <h_j, w_jk>
       + b_jk (w, b khoi tao 0 -> khong rut so ngau nhien). gop(base [B, C], delta) = base + logsumexp_k(delta) - logsumexp_k(0)
       (CUNG thiet bi / kieu -> TRUNG base tung bit khi delta = 0) = logsumexp_k(base + delta_k) - log K.'''
    def __init__(s, C, d, K=5):
        super().__init__()
        s.K = int(K)
        s.w = nn.Parameter(torch.zeros(int(C), s.K, int(d)))
        s.b = nn.Parameter(torch.zeros(int(C), s.K))

    def delta(s, h):
        return torch.einsum('bcd,ckd->bck', h.float(), s.w) + s.b

    @staticmethod
    def gop(base, delta):
        return base + (torch.logsumexp(delta, -1) - torch.logsumexp(torch.zeros_like(delta), -1)).to(base.dtype)


def loss_nguon(delta, s):
    '''CE "nguon nao": delta [B, C, K], s [B, C] int (nguon 0..K-1 hoac -1). TB tren cap s >= 0; khong cap nao -> None.'''
    m = s >= 0
    if not bool(m.any()): return None
    return F.cross_entropy(delta[m].float(), s[m].long())


def ma_nguon(ngu_canh, bang):
    '''Ngu canh 78s (chuoi) -> chi so nguon int8 theo `bang` {ngu canh: nguon}; ngoai bang (khong_tu_khoa, khong_report, la) -> -1.'''
    return np.array([int(bang.get(str(g), -1)) for g in ngu_canh], 'int8')


def nguon_hang(st_hang, pool, Y, st_nc, nhan_nc, ma_nc, C):
    '''Bang nguon theo hang ORDER [N_O, C] int8: hang POOL, Y = 1, (study, nhan) co trong bang 78s -> ma nguon; con lai -1.
       st_hang [N_O] study cua hang; pool [N_O] bool; Y [N_O, C] (NaN -> khong duong); st_nc / nhan_nc / ma_nc: cot bang 78s.'''
    C = int(C); st_hang = np.asarray(st_hang, 'int64'); Y = np.asarray(Y)
    k = pd.Index(np.asarray(st_nc, 'int64') * C + np.asarray(nhan_nc, 'int64'))
    assert k.is_unique, 'bang ngu canh: (study, nhan) TRUNG'
    out = np.full(Y.shape, -1, 'int8')
    i, j = np.nonzero((Y == 1) & np.asarray(pool, bool)[:, None])
    p = k.get_indexer(st_hang[i] * C + j)
    co = p >= 0
    out[i[co], j[co]] = np.asarray(ma_nc, 'int8')[p[co]]
    return out


class KenhCoDinh(nn.Module):
    '''Y D: logit_j += w_j * max_{k thuoc vung co dinh cua j} z[:, k, j] cho nhan j co vung co dinh (w khoi tao 0 -> luc dau TRUNG
       logit). co_dinh {chi so nhan: [chi so vung, ...]} (moi nhan >= 1 vung); R so vung. z [B, R, C] (DauVung.z).'''
    def __init__(s, co_dinh, R):
        super().__init__()
        idx = sorted(int(j) for j in co_dinh)
        m = torch.zeros(len(idx), int(R), dtype=torch.bool)
        for a, j in enumerate(idx):
            ks = sorted({int(k) for k in co_dinh[j]})
            assert ks and all(0 <= k < int(R) for k in ks), (j, ks)
            m[a, ks] = True
        s.register_buffer('idx', torch.as_tensor(idx, dtype=torch.long))
        s.register_buffer('mat_na', m)
        s.w = nn.Parameter(torch.zeros(len(idx)))

    def forward(s, logit, z):
        if not s.idx.numel(): return logit
        zz = z[:, :, s.idx].float().masked_fill(~s.mat_na.T[None], float('-inf')).amax(1)
        return logit.index_add(1, s.idx, (s.w * zz).to(logit.dtype))


def bit_co_dinh(co_dinh_ten, VUNG, LAB):
    '''anh_xa['co_dinh'] {ten nhan: [ten vung]} -> {chi so nhan: [chi so vung]} (ten la -> KeyError).'''
    vi = {v: i for i, v in enumerate(VUNG)}; li = {l: j for j, l in enumerate(LAB)}
    return {li[l]: [vi[v] for v in vs] for l, vs in co_dinh_ten.items()}


def bo_sung_co_dinh(M, co_dinh):
    '''M [n, C] int64 mat na bit tap vung S cua (anh, nhan); nhan co vung co dinh: o M = 0 (ImaGenome khong co vi tri) -> bit cac vung
       co dinh (vung du phong). Tra BAN SAO; o M != 0 GIU NGUYEN (vi tri ImaGenome uu tien).'''
    M = np.array(M, 'int64', copy=True)
    for j, ks in co_dinh.items():
        b = np.int64(0)
        for k in ks: b |= np.int64(1) << np.int64(int(k))
        M[:, int(j)] = np.where(M[:, int(j)] == 0, b, M[:, int(j)])
    return M


class LinearNhan(nn.Linear):
    '''Y 5: nn.Linear voi dau ra x g (g co dinh, KHONG phai tham so). Dung bang cach DOI LOP ca_h.o cua LopJCA (cung tham so, cung khoa
       state_dict). W_o = 0 -> dau ra 0 (trung tinh); voi Adam, buoc cap nhat W_o ~ LR -> write-back doi ~g lan nhanh hon.'''
    g = 1.0

    def forward(s, x):
        return F.linear(x, s.weight, s.bias) * s.g


def tang_ghi_nguoc(luong, g):
    '''Moi lop JCA cua `luong`: ca_h.o (W_o) -> LinearNhan voi he so g. Tra so lop da doi.'''
    n = 0
    for l in luong.lop:
        l.ca_h.o.__class__ = LinearNhan; l.ca_h.o.g = float(g); n += 1
    return n

# ---- khoi 78x J v3b (NGUYEN VAN experiments/78_chi_anh/khoi/khoi_jca_v3b.py) ----


class LopJCANeoTran(LopJCANeo):
    '''J v3b: LopJCANeo + tran write-back theo anh (xem dau tep). Tra (h', r', hop', ti) nhu lop cha; ti = ||dh sau tran|| / ||h|| tren CA lo
       khi co anh cham tran, con khong = ti cua lop cha (torch.where, KHONG dong bo GPU). Thong ke KHONG grad (tensor tren thiet bi, doc
       NGAY sau forward; checkpoint tinh lai ghi de bang CUNG gia tri): _tk = (ti le tho lon nhat theo anh, so anh cham tran).'''
    rho = None
    _tk = None

    def forward(s, h, r, hop, p, mo_ghi, neo=None):
        h2, r2, hop2, ti = super().forward(h, r, hop, p, mo_ghi, neo)
        if not mo_ghi or s.rho is None:
            return h2, r2, hop2, ti
        B = h.shape[0]
        dh = h2 - h                                                   # phan DA cong vao h (sau lam tron cua phep cong)
        nh = h.float().reshape(B, -1).norm(dim=1)
        nd = dh.float().reshape(B, -1).norm(dim=1)
        tho = nd / nh.clamp_min(1e-12)
        cham = tho > float(s.rho)
        he = torch.where(cham, float(s.rho) * nh / nd.clamp_min(1e-12), torch.ones_like(nd))
        hinh = (B,) + (1,) * (h.dim() - 1)
        m = cham.view(hinh)
        dh_t = he.view(hinh).to(dh.dtype) * dh
        h3 = torch.where(m, h + dh_t, h2)                             # anh duoi tran: h' cua lop cha TUNG BIT
        with torch.no_grad():                                         # ||dh sau tran cua anh b|| = he_b ||dh_b|| -> khong dung tensor lon
            ti3 = (he * nd).square().sum().sqrt() / nh.square().sum().sqrt().clamp_min(1e-12)
            ti = ti3 if ti is None else torch.where(cham.any(), ti3, ti)
            s._tk = (tho.detach().max(), cham.sum())
        return h3, r2, hop2, ti


def ap_tran_ghi(luong, rho):
    '''Moi lop JCA cua `luong` (PHAI la LopJCANeo) -> LopJCANeoTran voi tran rho (> 0). Tra so lop da doi. Khong them tham so / buffer.'''
    rho = float(rho)
    assert rho > 0, f'ap_tran_ghi: rho {rho} phai > 0'
    n = 0
    for l in luong.lop:
        assert type(l) is LopJCANeo, f'ap_tran_ghi: lop {type(l).__name__} khong phai LopJCANeo -> DUNG'
        l.__class__ = LopJCANeoTran; l.rho = rho; n += 1
    return n

# ---- khoi 78y (NGUYEN VAN experiments/78_chi_anh/khoi/khoi_doc_dong_cung.py) ----


@torch.no_grad()
def upper_luong_hr(bb, luong, h8, mo_ghi=True):
    '''Doc dong cung J (78y): = upper_luong (79) + tra them token vung r sau lop JCA cuoi va ti le ghi nguoc tung lop.
       h8 = dau ra block luong.tang[0] - 1 (shared_lo cua 78t / 78v). -> (patch [B, N, d] float32, r [B, R, d] float32, ghi list).'''
    st = luong.bat_dau(h8.shape[0]); h = h8
    L = bb.encoder.layer
    for l in range(luong.tang[0], len(L)):
        h = _out_luong(L[l](h))
        h = luong.buoc(l, h, st, mo_ghi)
    return bb.layernorm(h)[:, 1:].float(), luong.dau_ra(st)['r'].float(), list(st['ghi'])


def mldec_h_hr(m, q, patch, r):
    '''Dau J doc [patch ; token vung] (78y `mldec_hr`): K / V = [pin(patch) ; m.docv(pin, r)], query q [C, d] (q_cua cua 78t)
       -> (logit, h = dau ra query TRUOC GroupFC, kv = pin(patch) CHI phan patch). Nhu dau_h_v2 cua J v2 (khoi_jca_v2).'''
    kv = m.pin(patch)
    h, _ = m.img(q.unsqueeze(0).expand(patch.shape[0], -1, -1), torch.cat([kv, m.docv(m.pin, r).to(kv.dtype)], 1), False)
    return m.out(h), h, kv


# ---- doan NGUYEN VAN experiments/78_chi_anh/khoi/khoi_cat_vung.py ----


S_1022, S_518, CANH_O = 1022, 518, 518


O_1022 = ((0, 0), (504, 0), (0, 504), (504, 504), (252, 252))     # (x0, y0) goc tren-trai tung o tren anh 1022: 2 x 2 + o giua


assert all(0 <= v and v + CANH_O <= S_1022 for o in O_1022 for v in o)


def luoi_518(dev='cpu'):
    '''Luoi grid_sample [5, 518, 518, 2] (thu tu x, y) cho cac o tren anh 518. Tam pixel dau ra j cua o (x0, y0) -> toa do chuan hoa
       g = 2 (x0 + j + 0,5) / 1022 - 1 (align_corners=False) <=> toa do pixel nguon (x0 + j + 0,5) * 518 / 1022 - 0,5 (Codex muc 44.3).'''
    j = torch.arange(CANH_O, dtype=torch.float64)
    out = []
    for x0, y0 in O_1022:
        gx, gy = 2 * (x0 + j + 0.5) / S_1022 - 1, 2 * (y0 + j + 0.5) / S_1022 - 1
        GY, GX = torch.meshgrid(gy, gx, indexing='ij')
        out.append(torch.stack([GX, GY], -1))
    return torch.stack(out).float().to(dev)


def lay_mau_518(x, luoi):
    '''x [B, 518, 518] (uint8 / float) -> [B, 5, 518, 518] float: bicubic, align_corners=False, padding 'border' (CHUA lam tron).'''
    xf = x.float().unsqueeze(1)
    B = xf.shape[0]
    return torch.stack([F.grid_sample(xf, luoi[k].unsqueeze(0).expand(B, -1, -1, -1), mode='bicubic', padding_mode='border',
                                      align_corners=False)[:, 0] for k in range(luoi.shape[0])], 1)


# ---- doan NGUYEN VAN experiments/78_chi_anh/khoi/khoi_cat_sau.py ----


N_LUOI = S_518 // 14                 # 37 token moi canh (patch 14)


TI_1022 = S_1022 / S_518             # toa do anh 518 -> anh 1022


def hinh_luoi(B, dev='cpu'):
    '''-> [B, 5, 3] long (x0, y0, L) tren anh 1022 cua 5 o luoi (O_1022, canh 518).'''
    g = torch.tensor([[x0, y0, CANH_O] for x0, y0 in O_1022], dtype=torch.long, device=dev)
    return g[None].expand(int(B), -1, -1).clone()


def luoi_con(geo):
    '''geo [B, K, 3] (x0, y0, L tren anh 1022) -> (g [B, K, P, 2] float32, w [B, K, P] float32), P = 37 * 37 * 4, thu tu (i, j, qy, qx):
       diem con q cua token toan anh (i, j) o toa do 1022: p_x = (14 j + 3,5 + 7 qx) 1022 / 518 (y tuong tu); g = 2 (p - x0) / L - 1
       (grid_sample align_corners=False tren luoi token 37 x 37 cua o); w = 1 neu diem con nam trong o.'''
    dev = geo.device
    t = torch.arange(N_LUOI, dtype=torch.float64, device=dev)
    q = torch.tensor([3.5, 10.5], dtype=torch.float64, device=dev)
    pc = (14 * t[:, None] + q[None]) * TI_1022                               # [37, 2]: toa do con theo mot truc
    py = pc[:, None, :, None].expand(N_LUOI, N_LUOI, 2, 2).reshape(-1)       # theo (i, qy)
    px = pc[None, :, None, :].expand(N_LUOI, N_LUOI, 2, 2).reshape(-1)       # theo (j, qx)
    gf = geo.to(torch.float64)
    x0, y0, L = gf[..., 0:1], gf[..., 1:2], gf[..., 2:3]
    dx, dy = px[None, None] - x0, py[None, None] - y0
    w = ((dx >= 0) & (dx < L) & (dy >= 0) & (dy < L)).float()
    g = torch.stack([2 * dx / L - 1, 2 * dy / L - 1], -1).float()
    return g, w


@torch.no_grad()
def lay_mau_con(c, g, w):
    '''Token o c [B, K, N, d] (KHONG CLS; N = 37 * 37) + (g, w) cua luoi_con -> (v [B, N, 4, d] float32 = TB co trong so w qua K o cua gia
       tri bilinear tai 4 diem con cua moi token toan anh, phu [B, N] bool = ca 4 diem con co >= 1 o phu). Khong tham so, khong grad.'''
    B, K, N, d = c.shape
    n = int(round(N ** 0.5))
    assert n * n == N and tuple(g.shape[:2]) == (B, K) and g.shape[2] == 4 * N, (tuple(c.shape), tuple(g.shape))
    tong = torch.zeros(B, 4 * N, d, dtype=torch.float32, device=c.device)
    for k in range(K):
        m = c[:, k].reshape(B, n, n, d).permute(0, 3, 1, 2).float()
        s = F.grid_sample(m, g[:, k][:, None].float(), mode='bilinear', padding_mode='border', align_corners=False)[:, :, 0]
        tong += s.transpose(1, 2) * w[:, k, :, None]
    ws = w.sum(1)
    v = tong / ws.clamp(min=1.0)[..., None]
    return v.reshape(B, N, 4, d), (ws.reshape(B, N, 4) > 0).all(-1)


class DuaVao(nn.Module):
    '''Skip bac thap (cach a): v [B, N, 4, d] (lay_mau_con) -> LayerNorm(d) -> ghep 4 d -> Linear(4 d, d) KHOI TAO 0 -> x phu -> [B, N, d].
       Luc khoi tao dau ra = 0 tung bit -> nhanh skip = nhanh 0.'''
    def __init__(s, d=768):
        super().__init__()
        s.ln = nn.LayerNorm(d)
        s.chieu = nn.Linear(4 * d, d)
        nn.init.zeros_(s.chieu.weight)
        nn.init.zeros_(s.chieu.bias)

    def forward(s, v, phu):
        B, N, q, d = v.shape
        return s.chieu(s.ln(v).reshape(B, N, q * d)) * phu[..., None].to(v.dtype)


# ---- khoi NGUYEN VAN experiments/78_chi_anh/khoi/khoi_cat_j.py ----


import contextlib
lay_mau_con_grad = lay_mau_con.__wrapped__        # CUNG phep tinh voi lay_mau_con (khoi_cat_sau, boc @torch.no_grad) nhung CO gradient


def o_phong_518(img, luoi, mean, std):
    '''img [B, 3, 518, 518] float = anh 518 DA chuan hoa ((x / 255 - mean) / std, 3 kenh giong nhau) -> 5 o [B, K, 518, 518] float32 CUNG
       chuan hoa: bicubic tren kenh 0 (lay_mau_518, align_corners=False, padding border), kep ve khoang chuan hoa cua [0, 255] (nhu kep
       uint8 cua cat_518_phong; khong lam tron muc xam).'''
    o = lay_mau_518(img[:, 0], luoi)
    return o.clamp((0.0 - float(mean)) / float(std), (1.0 - float(mean)) / float(std))


@torch.no_grad()
def token_o(bb, img, luoi, mean, std, khuc, k7):
    '''5 o phong (o_phong_518) qua bb (embeddings -> block 0..L-1 -> layernorm) KHONG gradient, chia khuc `khuc` view (autocast do noi goi
       dat) -> (t7 [B, K, N, d] = token patch sau block k7 (0-based), t11 [B, K, N, d] = token patch cuoi sau layernorm); bo CLS.
       Khong qua luong vung, khong drop-path (drop-path B.4 CHI o duong anh toan canh); RAD-DINO dropout = 0 -> khong rut so.'''
    B = int(img.shape[0])
    o = o_phong_518(img, luoi, mean, std)
    K = int(o.shape[1])
    o = o.reshape(B * K, o.shape[2], o.shape[3])
    t7, t11 = [], []
    for i in range(0, B * K, int(khuc)):
        h = bb.embeddings(o[i:i + int(khuc)].unsqueeze(1).expand(-1, 3, -1, -1).contiguous().to(img.dtype))
        for l, lop in enumerate(bb.encoder.layer):
            h = _out_luong(lop(h))
            if l == int(k7): t7.append(h[:, 1:])
        t11.append(bb.layernorm(h)[:, 1:])
    t7, t11 = torch.cat(t7), torch.cat(t11)
    return t7.reshape(B, K, t7.shape[1], t7.shape[2]), t11.reshape(B, K, t11.shape[1], t11.shape[2])


class BoChinhO(nn.Module):
    '''Adapter duong o (Tuan 26/10: "mo nho truoc de tranh sup va danh gia tiem nang"): x + ra(GELU(vao(LN(x)))), vao = Linear(d, r),
       ra = Linear(r, d) KHOI TAO 0 -> luc dau = x TUNG BIT. `vao` rut so -> ben goi tao trong fork_rng voi hat rieng.'''
    def __init__(s, d=768, r=64):
        super().__init__()
        s.ln, s.vao, s.ra = nn.LayerNorm(d), nn.Linear(d, int(r)), nn.Linear(int(r), d)
        nn.init.zeros_(s.ra.weight)
        nn.init.zeros_(s.ra.bias)

    def forward(s, x):
        return x + s.ra(F.gelu(s.vao(s.ln(x.to(s.ln.weight.dtype))))).to(x.dtype)      # nhanh tinh theo kieu tham so (autocast van ap)


def skip_o(t7, bo, dua, g, w, ckpt=False):
    '''Skip block 7 (cach a cua pha 1b) CO gradient vao adapter + phep chieu: t7 [B, K, N, d] (token o sau block 7, khong grad) -> adapter
       `bo` -> lay mau 2 x 2 diem con / token toan anh (g [B, K, P, 2], w [B, K, P] cua luoi_con) -> DuaVao `dua` -> inj [B, N, d].
       ckpt=True: tinh lai luc backward (torch.utils.checkpoint, use_reentrant=False) -> khong giu kich hoat [B, K, N, d] float32.'''
    def f(t):
        v, phu = lay_mau_con_grad(bo(t), g, w)
        return dua(v, phu)
    return _tuc_luong.checkpoint(f, t7, use_reentrant=False) if ckpt else f(t7)


def cong_skip(h, inj):
    '''h [B, 1 + N, d] (co CLS) + inj [B, N, d] vao token patch; CLS giu nguyen. inj None -> h (cung object).'''
    if inj is None: return h
    return torch.cat([h[:, :1], h[:, 1:] + inj.to(h.dtype)], 1)


@contextlib.contextmanager
def skip_vao_luong(luong, inj, k_skip=8, do=None):
    '''Trong with: luong.buoc(k_skip - 1, h, ...) tra cong_skip(h, inj) -> dau vao block k_skip co skip; dung voi chay_bb_luong (79) /
       chay_bb_luong_dp (B.4) KHONG sua (ca hai goi luong.buoc sau MOI block). do = list | None: them ||inj|| / ||h patch|| TB anh (tensor
       khong grad, khong dong bo GPU). inj None -> khong doi gi. Ra khoi with (ke ca loi): tra luong.buoc goc.'''
    if inj is None:
        yield luong
        return
    assert 'buoc' not in vars(luong), 'skip_vao_luong: luong.buoc da bi boc (long nhau?) -> DUNG'
    goc = luong.buoc

    def buoc(l, h, st, mo_ghi):
        h = goc(l, h, st, mo_ghi)
        if int(l) != int(k_skip) - 1: return h
        if do is not None:
            with torch.no_grad():
                do.append((inj.float().norm(dim=-1).mean(1) / h[:, 1:].float().norm(dim=-1).mean(1).clamp_min(1e-12)).mean())
        return cong_skip(h, inj)
    luong.buoc = buoc
    try:
        yield luong
    finally:
        del luong.buoc


def gop_view_o(gop, h0, h_o):
    '''h0 [B, Q, d] (view 0 = anh toan canh) + h_o [B, K, Q, d] (K o) -> h_bag [B, Q, d] = CUNG phep tinh GopStudy `gop` (kieu 'attn',
       giu None, n_view = 1 + K) tren 1 + K view cua TUNG anh (view = 0..K), viet DANG DAY: amax / tong theo truc view thay scatter_reduce /
       index_add_ (atomic -> GPU lech bit giua hai lan chay, Colab 26/10: cong duong cham C6 dung) -> TAT DINH. Dung tham so cua `gop`
       (e_view, ln, w, b) -> state_dict khong doi. Tham so gop = 0 -> TB 1 + K view.'''
    V = 1 + int(h_o.shape[1])
    assert gop.kieu == 'attn' and gop.n_view == V and tuple(h_o.shape[2:]) == tuple(h0.shape[1:]), (gop.kieu, gop.n_view, tuple(h_o.shape))
    h = torch.cat([h0.unsqueeze(1), h_o.to(h0.dtype)], 1)                                       # [B, V, Q, d]
    hv = h + gop.e_view.unsqueeze(1).to(h.dtype)                                                 # + e_view[v]
    sc = (gop.ln(h) * gop.w).sum(-1) + gop.b                                                     # [B, V, Q]
    w = torch.exp(sc - sc.amax(1, keepdim=True))
    return (w.unsqueeze(-1).to(hv.dtype) * hv).sum(1) / w.sum(1).unsqueeze(-1).to(hv.dtype)


@torch.no_grad()
def thong_ke_o(gop, bo7, bo11, dua, do=None):
    '''Chan doan duong o (so tong hop, khong grad): ||inj|| / ||h patch block 7|| TB (tu `do` cua skip_vao_luong), chuan Linear ra cua hai
       adapter, chuan phep chieu skip, |b| TB va ||e_view|| theo view + ||w|| cua bo gop view.'''
    return dict(ti_le_inj=(float(torch.stack(do).mean()) if do else None), bo7=float(bo7.ra.weight.norm()), bo11=float(bo11.ra.weight.norm()),
                dua=float(dua.chieu.weight.norm()), gop_w=float(gop.w.norm()), gop_b_view=[float(v) for v in gop.b.abs().mean(1)],
                gop_e_view=[float(v) for v in gop.e_view.norm(dim=1)])


# ==== KHOI SKIP7 (Tuan duyet 28/10; SPEC B.2aa): CHI THEM duoi day, KHONG sua phan tren ====
@torch.no_grad()
def token_o7(bb, img, luoi, mean, std, khuc, k7):
    '''Nhu token_o nhung DUNG sau block k7 (khong tinh block > k7, khong layernorm cuoi): 5 o phong (o_phong_518) qua bb KHONG gradient, chia
       khuc `khuc` view -> t7 [B, K, N, d] = token patch sau block k7 (0-based), bo CLS. CUNG phep tinh / thu tu voi token_o -> = t7 cua no
       tung bit. Khong qua luong vung, khong drop-path; RAD-DINO dropout = 0 -> khong rut so.'''
    B = int(img.shape[0])
    o = o_phong_518(img, luoi, mean, std)
    K = int(o.shape[1])
    o = o.reshape(B * K, o.shape[2], o.shape[3])
    t7 = []
    for i in range(0, B * K, int(khuc)):
        h = bb.embeddings(o[i:i + int(khuc)].unsqueeze(1).expand(-1, 3, -1, -1).contiguous().to(img.dtype))
        for lop in bb.encoder.layer[:int(k7) + 1]:
            h = _out_luong(lop(h))
        t7.append(h[:, 1:])
    t7 = torch.cat(t7)
    return t7.reshape(B, K, t7.shape[1], t7.shape[2])


@torch.no_grad()
def thong_ke_o7(bo7, dua, do=None):
    '''Chan doan duong o cua bien the skip7 (so tong hop, khong grad): ||inj|| / ||h patch block 7|| TB (tu `do` cua skip_vao_luong), chuan
       Linear ra cua adapter BO_O7, chuan phep chieu skip.'''
    return dict(ti_le_inj=(float(torch.stack(do).mean()) if do else None), bo7=float(bo7.ra.weight.norm()), dua=float(dua.chieu.weight.norm()))


# ---- 78y: luong J v3 dong cung ----
# MOI cho RUT SO rieng cua J (khoi tao luong truoc khi nap, lo cua cong 3, dropout cua dau mldec_hr o C6) chay trong fork_rng -> dong
# ngau nhien chung (thu tu lo, mat na dropout cua 3 dau 78v) TRUNG luot B1_e6 -> so J - B1_e6 GHEP CAP ca dong ngau nhien.
_FORK_78Y = lambda: torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev == 'cuda' else [])
LUONG = None
if LUOT_78Y == 'J_v3':
    _cfj = torch.load(BB_CFG_SRC, map_location='cpu', weights_only=False)
    assert _cfj.get('run') == BB_RUN and _cfj.get('luot_79a') == 'J_bert_dong' and all(
        isinstance(_cfj.get(k), dict) for k in ('cfg_luong', 'cfg_jv2', 'cfg_jv3')), (
        f'backbone_ft.pt cua {BB_RUN}: run / luot_79a / cau hinh luong KHAC (cong 1) -> DUNG')
    CFG_79J, CFG_JV2J, CFG_JV3J = _cfj['cfg_luong'], _cfj['cfg_jv2'], _cfj['cfg_jv3']
    LUONG_CFG_J = CFG_79J['luot'][CFG_79J['luot_chon']]
    _hop_j = _cfj['hop_tb'].float()
    assert CFG_79J['luot_chon'] == 'J_bert_dong' and torch.equal(_hop_j, _ck['luong']['hop_tb'].float().cpu()), (
        'hop_tb cua backbone_ft.pt KHAC rs_ ep4 (luot khac?) (cong 1) -> DUNG')
    G_LUOI_J = SIZE // backbone.config.patch_size
    assert G_LUOI_J ** 2 == N_TOK, (G_LUOI_J, N_TOK)
    with _FORK_78Y():   # khoi tao ngau nhien (GHI DE ngay khi nap) KHONG tieu dong chung
        LUONG = LuongVungNeo(_hop_j.numpy(), d=D_CXR, R=int(_hop_j.shape[0]), tang=tuple(CFG_79J['tang']),
                             n_head=int(CFG_79J['n_head']), ff=int(CFG_79J['ff']), beta=float(CFG_79J['beta']),
                             t_mem=float(CFG_79J['t_mem']), hai_chieu=bool(LUONG_CFG_J['hai_chieu']), neo=bool(LUONG_CFG_J['neo']),
                             luoi=G_LUOI_J, ckpt=False)
    N_GHI_NGUOC_J = tang_ghi_nguoc(LUONG, float(CFG_JV3J['ghi_nguoc_g']))
    assert isinstance(_cfj.get('cfg_jv3b'), dict), f'backbone_ft.pt cua {BB_RUN} KHONG co cfg_jv3b (khong phai luot J v3b) -> DUNG'
    CFG_JV3BJ = _cfj['cfg_jv3b']
    N_TRAN_J = ap_tran_ghi(LUONG, float(CFG_JV3BJ['tran_ghi_rho']))   # 78y J v3b: frozen reading doc DUNG mo hinh co tran (SAU x5)
    assert N_TRAN_J == N_GHI_NGUOC_J and all(type(l) is LopJCANeoTran and l.rho == float(CFG_JV3BJ['tran_ghi_rho']) for l in LUONG.lop), \
        'J v3b: tran chua gan du lop JCA -> DUNG'
    # ---- 78z dot 4 (Tuan duyet 21/10; SPEC B.2n): luot = cfg_d4 cua backbone_ft.pt (78x dot 4): ten / ep_bb / 6 epoch / KHONG khe; B.4: rs_
    #      ep4 = trong so EMA (khoa 'ema' trong rs_) <-> cfg_d4 trong_so 'EMA' ----
    _cd4 = _cfj.get('cfg_d4')
    assert isinstance(_cd4, dict) and _cd4.get('ten') == BIEN_THE_BB and 1 <= EP_BB <= int(_cd4.get('epochs', -1)) and int(_cd4.get('epochs', -1)) == 6 \
        and _cd4.get('khe') is None and not any(k.startswith('khe.') for k in _ck['luong']), (
        f"backbone_ft.pt cua {BB_RUN}: cfg_d4 {_cd4} KHAC luot {BIEN_THE_BB} / ep_bb {EP_BB} / 6 epoch / khong khe -> DUNG")
    _ema4 = _ck.get('ema')
    assert (_cd4.get('trong_so') == 'EMA') == isinstance(_ema4, dict) and (_ema4 is None or _ema4.get('trong_so') == 'EMA'), (
        f"{BB_TEP}: trong so {'EMA' if _ema4 else 'tho'} KHAC cfg_d4 trong_so {_cd4.get('trong_so')} -> DUNG")
    print(f"78z dot 4 C2b: backbone {BB_RUN} (rs_ ep{EP_BB}) | luot {BIEN_THE_BB} | trong so {_cd4.get('trong_so')}"
          + (f" (EMA {_ema4['n']} lan cap nhat, he so {_ema4['he_so']:.4f})" if _ema4 else ''))
    LUONG.load_state_dict(_ck['luong'], strict=True)
    LUONG = LUONG.to(dev).eval().requires_grad_(False)
    assert N_GHI_NGUOC_J == len(LUONG.tang) and GRAD_FROM == LUONG.tang[0] and list(LUONG.tang) == list(range(GRAD_FROM, NL)), (
        f'luong tang {LUONG.tang} KHONG khop block dong cung {GRAD_FROM}..{NL - 1} cua 78v (cong 2) -> DUNG')
    # neo theo view: view_group CO SAN luc cham (doan NGUYEN VAN 38a, thut vao; CUNG ORDER emb_index + CUNG tep view nhu 78x) — frontal
    # = neo (cfg_jv2 view_neo, nhu 78x)
    P_VIEW = os.path.join(OUT_DIR, 'cxrlt_view_resolved.parquet')
    V = ((pd.read_parquet(P_VIEW)[['dicom_id', 'view_group', 'use']]
            .set_index('dicom_id').loc[ORDER.dicom_id].reset_index()) if os.path.exists(P_VIEW)
         else ORDER[['dicom_id', 'view_group']].assign(use=lambda d: d.view_group != 'unknown'))
    NEO = V.view_group.to_numpy() == CFG_JV2J['view_neo']
    # 78y view LUC CHAM (Tuan 15/10 + 16/10; docs/SPEC_SAU_78Y.md B.1c): anh VAL / TEST lay co neo tu tep view luc cham (C0: VIEW_CHAM_SRC,
    # md5 ghim); anh TRAIN giu tep cu (cung neo voi luc train backbone 78x). Cong: phu moi anh val / test; so anh doi co = buoc 1.0.
    _vc = pd.read_parquet(VIEW_CHAM_SRC, columns=['dicom_id', 'view_group']).set_index('dicom_id').view_group
    _dic = ORDER.dicom_id.to_numpy()
    _cham = np.zeros(len(_dic), bool); _cham[I_VA] = True; _cham[I_TE] = True
    assert pd.Index(_dic[_cham]).isin(_vc.index).all(), 'tep view luc cham thieu anh val / test cua ORDER -> DUNG'
    NEO_CU = np.asarray(NEO, bool).copy()
    NEO = NEO_CU.copy()
    NEO[_cham] = _vc.reindex(_dic[_cham]).astype(str).to_numpy() == CFG_JV2J['view_neo']
    NEO_DOI = {t: int((NEO != NEO_CU)[ix].sum()) for t, ix in (('val', I_VA), ('test', I_TE))}
    assert NEO_DOI == NEO_DOI_DAT_TRUOC and not (NEO != NEO_CU)[~_cham].any(), (
        f'so anh doi co neo {NEO_DOI} KHAC buoc 1.0 {NEO_DOI_DAT_TRUOC} hoac doi o anh train -> tep view / ORDER khac. DUNG, bao Claude')
    print(f"78y C2b: co neo anh val / test theo view LUC CHAM (md5 {MD5_VIEW_CHAM[:8]}): doi val {NEO_DOI['val']} / test {NEO_DOI['test']} "
          'anh (= buoc 1.0); anh train giu tep cu')
    del _vc, _dic, _cham
    NEO_T = torch.from_numpy(NEO).to(dev)
    # cong (3): duong chia doi = forward day du (8 anh val, neo theo view that)
    with _FORK_78Y():   # tao iter DataLoader rut 1 so -> trong fork
        _bz = next(iter(loader(I_VA[:8], eval_tf, False, 8, nw=0)))
    _xz, _rz = _bz['image'].to(dev), _bz['row'].to(dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        LUONG.dat_neo(NEO_T[_rz].float())
        _pz, _rv, _gz = upper_luong_hr(backbone, LUONG, run_blocks(backbone.embeddings(_xz), 0, GRAD_FROM), mo_ghi=True)
        LUONG.dat_neo(NEO_T[_rz].float())
        _pf, _cf, _raf = chay_bb_luong(backbone, LUONG, _xz, mo_ghi=True, grad_bb=False)
    _e3 = max(float((_pz - _pf).abs().max()), float((_rv - _raf['r'].float()).abs().max()))
    assert _e3 < 1e-3, f'duong chia doi J lech forward day du {_e3:.2e} (cong 3) -> DUNG'
    # cong (4): ti le ghi nguoc tung lop ~ nhat ky ep4 cua 78x (bat g / cau hinh sai)
    _gt = (((_ck.get('log') or {}).get(str(EP_BB - 1)) or {}).get('luong') or {}).get('ghi_tang')
    _gn = [float(t) for t in _gz]
    assert _gt and len(_gt) == len(_gn) == len(LUONG.tang), f'nhat ky ep{EP_BB} khong co ghi_tang / so lop lech ({_gt}, {_gn}) -> DUNG'
    _ti4 = [a / b for a, b in zip(_gn, _gt)]
    assert all(1 / 3 <= t <= 3 for t in _ti4), (
        f'ti le ghi nguoc 8 anh val {[round(x, 4) for x in _gn]} lech nhat ky ep{EP_BB} {[round(x, 4) for x in _gt]} qua 3 lan (cong 4: g / '
        'cau hinh sai?) -> DUNG')
    print(f"78y C2b: luong J v3 dong cung | R {LUONG.R} | tang {LUONG.tang} | g {float(CFG_JV3J['ghi_nguoc_g']):g} | neo theo view "
          f"(frontal {NEO.mean():.1%} anh) | cong 3: chia doi lech {_e3:.1e} | cong 4: ghi nguoc {[round(x, 4) for x in _gn]} / nhat ky "
          f"ep{EP_BB} {[round(x, 4) for x in _gt]}")
    print(f"78y C2b J v3b: tran write-back rho {CFG_JV3BJ['tran_ghi_rho']} o {N_TRAN_J} lop JCA (cfg_jv3b cua backbone_ft.pt, "
          f"{CFG_JV3BJ['phien_ban']})")
    # ---- nhanh C (Tuan duyet 29/10; SPEC B.2ab): module o DONG CUNG tu rs_ ep EP_BB (EMA) + cong cau truc nhanh C (cfg_o goi o cat; DU dua_o +
    #      bo_o7 + bo_o11 + gop_o; CHI nap dua_o + bo_o7 cho duong skip, 4 dau doc 1 view) + cong 3' (duong chia doi CO skip = forward day du
    #      CO skip, 8 anh val) ----
    _cfg_o = _cfj.get('cfg_o')
    assert isinstance(_cfg_o, dict) and _cfg_o == O_CFG_MONG, f"backbone_ft.pt cua {BB_RUN}: cfg_o {_cfg_o} KHAC cau hinh nhanh C mong doi -> DUNG (luot khac?)"
    assert all(k in _ck for k in ('dua_o', 'bo_o7', 'bo_o11', 'gop_o')), (
        f'{BB_TEP}: module o KHONG dung cau truc nhanh C (can dua_o + bo_o7 + bo_o11 + gop_o) -> DUNG')
    with _FORK_78Y():   # khoi tao (GHI DE ngay khi nap) KHONG tieu dong chung
        BO_O7, DUA_O = BoChinhO(D_CXR, int(_cfg_o['r_adapter'])), DuaVao(D_CXR)
    for _m_o, _k_o in ((BO_O7, 'bo_o7'), (DUA_O, 'dua_o')): _m_o.load_state_dict(_ck[_k_o], strict=True)
    BO_O7, DUA_O = (_m_o.to(dev).eval().requires_grad_(False) for _m_o in (BO_O7, DUA_O))
    SO_O, K_SKIP_O, KHUC_O = int(_cfg_o['n_o']), int(_cfg_o['k_skip']), int(_cfg_o['khuc'])
    assert K_SKIP_O == GRAD_FROM, f'nhanh C: skip vao block {K_SKIP_O} KHAC block dong cung dau {GRAD_FROM} -> DUNG'
    LUOI_O = luoi_518(dev)
    _G_O, _W_O = luoi_con(hinh_luoi(1, dev))
    with torch.no_grad(), torch.autocast('cuda', dtype=AMP, enabled=dev == 'cuda'):
        _t7z = token_o7(backbone, _xz, LUOI_O, MEAN, STD, KHUC_O, K_SKIP_O - 1)
        _ijz = skip_o(_t7z, BO_O7, DUA_O, _G_O.expand(len(_xz), -1, -1, -1).contiguous(), _W_O.expand(len(_xz), -1, -1).contiguous())
        LUONG.dat_neo(NEO_T[_rz].float())
        _pz3 = upper_luong_hr(backbone, LUONG, cong_skip(run_blocks(backbone.embeddings(_xz), 0, GRAD_FROM), _ijz), mo_ghi=True)[0]
        LUONG.dat_neo(NEO_T[_rz].float())
        with skip_vao_luong(LUONG, _ijz, K_SKIP_O):
            _pf3 = chay_bb_luong(backbone, LUONG, _xz, mo_ghi=True, grad_bb=False)[0]
    _e3o = float((_pz3 - _pf3).abs().max())
    _ti_o = float((_ijz.float().norm(dim=-1).mean() / _pz.float().norm(dim=-1).mean().clamp_min(1e-12)))
    assert _e3o < 1e-3, f'nhanh C: duong chia doi CO skip lech forward day du CO skip {_e3o:.2e} (cong 3 prime) -> DUNG'
    print(f"78z nhanh C C2b: module o (rs_ ep{EP_BB}, {_cfg_o['ten']}; nap dua_o + bo_o7, bo qua bo_o11 / gop_o) | cong 3': chia doi co skip lech "
          f"{_e3o:.1e} | ||inj|| / ||patch cuoi|| {_ti_o:.4f} | {SO_O} o, khuc {KHUC_O}")
    del _cfg_o, _t7z, _ijz, _pz3, _pf3
    del _cfj, _bz, _xz, _rz, _pz, _rv, _gz, _pf, _cf, _raf
else:
    print(f'78y C2b: luot {LUOT_78Y} -> KHONG co luong vung (LUONG = None); upper_frozen = block 8-11 goc nhu 78v')
_KHOA_C2B = (THU_NHANH, RUN_TAG, LUOT_78Y)   # C4 / C6 kiem: doi C0 thi PHAI chay lai C2b
