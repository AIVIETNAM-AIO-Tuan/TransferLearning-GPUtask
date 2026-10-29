# ==== nhanh_c buoc z_c3 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
class GroupFC(nn.Module):
    def __init__(s, C, d):
        super().__init__()
        s.w = nn.Parameter(torch.randn(C, d) * d ** -0.5); s.b = nn.Parameter(torch.zeros(C))
    def forward(s, h): return (h * s.w).sum(-1) + s.b

class XAttn(nn.Module):
    def __init__(s, d, nhead=8, ff=2048, drop=0.1):
        super().__init__()
        s.at = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
    def forward(s, q, kv, need_w=False):
        a, w = s.at(q, kv, kv, need_weights=need_w, average_attn_weights=True)
        h = s.n1(q + a); return s.n2(h + s.ff(h)), w

class MicroBag(nn.Module):
    '''`const=True`: MOT chi so, khong phai MIC_L ban sao (tui that TB 1.7 vi khuan).'''
    def __init__(s, V, d, const=False):
        super().__init__()
        s.const = const; s.w = nn.Parameter(torch.randn(V, d) * d ** -0.5)
        s.mlp = nn.Sequential(nn.GELU(), nn.Linear(d, d))
    def forward(s, ids, w):
        if s.const:
            ids = torch.zeros(ids.shape[0], 1, dtype=torch.long, device=ids.device)
            w = torch.ones(ids.shape[0], 1, device=ids.device)
        return s.mlp(F.embedding_bag(ids, s.w, per_sample_weights=w, mode='sum'))

class ColTokBag(nn.Module):
    '''d_in token COT + 1 TONG HOP + 1 BAG. Chuoi nay CHUA k1 nhu truong hop rieng.'''
    def __init__(s, d_in, d=256, const=False, V=None):
        super().__init__()
        s.W = nn.Parameter(torch.randn(d_in, d) * d ** -0.5)
        s.b = nn.Parameter(torch.zeros(d_in, d))
        s.joint = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        s.ln = nn.LayerNorm(d); s.const = const
        if const: s.c = nn.Parameter(torch.randn(d_in + 1, d) * 0.02)
        s.mb = MicroBag(V, d, const) if V else None
        if s.mb is not None:
            s.bln = nn.LayerNorm(d); s.cln = nn.LayerNorm(d) if const else nn.Identity()
    def forward(s, e, mids=None, mcnt=None):
        if s.const: base = s.c.unsqueeze(0).expand(e.shape[0], -1, -1)
        else:
            col = s.b.unsqueeze(0) + e.unsqueeze(-1) * s.W.unsqueeze(0)
            base = s.ln(torch.cat([col, s.joint(e).unsqueeze(1)], 1))
        if s.mb is None: return base
        return torch.cat([s.cln(base), s.bln(s.mb(mids, mcnt)).unsqueeze(1)], 1)

class Tier1(nn.Module):
    '''Q = 1369 PATCH, K/V = token EHR. `pr` zero-init; `ln_a` chuan hoa DAU VAO cua phep chieu
       (38d: thieu no thi arm doi chung sup 0.2416 -> 0.0681).'''
    def __init__(s, d, nhead=8, drop=0.1):
        super().__init__()
        s.at = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        s.n = nn.LayerNorm(d); s.ln_a = nn.LayerNorm(d); s.pr = nn.Linear(d, d)
        nn.init.zeros_(s.pr.weight); nn.init.zeros_(s.pr.bias)
    def forward(s, patch, etok):
        a, _ = s.at(s.n(patch), etok, etok, need_weights=False)
        return patch + s.pr(s.ln_a(a))

class Tier1Img(nn.Module):
    '''Phim truoc o muc PATCH, chay o khong gian D_CXR (TRUOC `pin`).
       KHONG tru thang `patch - prior`: hai phim khong duoc dang ky (tu the, do hit vao). '''
    def __init__(s, d, nhead=8, drop=0.1, diff=True):
        super().__init__()
        s.at = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        s.nq = nn.LayerNorm(d); s.nk = nn.LayerNorm(d); s.ln_a = nn.LayerNorm(d)
        # LAYERSCALE. `pr` zero-init cho dong nhat thuc o buoc 0 nhung KHONG chan duoc toc do
        # lon len: AdamW cap nhat ~lr/buoc/tham so BAT KE gradient nho the nao, nen mot nhanh xuat
        # phat tu 0 lon TUYEN TINH theo buoc x LR. Do `ln_a` chuan hoa `patch - a` (thuong gan 0,
        # phan lon la nhieu) ve chuan don vi, bien do dau ra do TRON VEN `‖pr‖` quyet dinh. Khi no
        # at `patch` thi ML-Decoder doi che do dot ngot -> sup. Da xay ra 3 lan, luon o ep2-4 tuc
        # ngay sau dinh warmup (0.05 x E = nua epoch 1), va `clip` do 0.0% vi gradient VAN binh
        # thuong — van de o BUOC DI, khong o gradient.
        # Nay `pr` khoi tao CHUAN va mot he so LayerScale `1e-4` giu phan du gan 0 luc dau; ca hai
        # deu co gradient that, va he so phai HOC de lon len thay vi lon lên vi quan tinh AdamW.
        s.pr = nn.Linear(d, d)
        s.gamma = nn.Parameter(torch.full((d,), 1e-4))
        s.diff = diff
    def forward(s, patch, prior, has):
        a, _ = s.at(s.nq(patch), s.nk(prior), s.nk(prior), need_weights=False)
        z = s.gamma * s.pr(s.ln_a(patch - a if s.diff else a))
        return patch + z * has.view(-1, 1, 1)      # khong co phim truoc -> mask nhanh di han

class MLDecoder(nn.Module):
    '''Q = C query lop (KHONG hoc, dung theo bai goc). K/V = patch, co the da bi Tier1 dieu bien.'''
    def __init__(s, C, d_in, d=512, nhead=8, ff=1024, drop=0.1, d_ehr_in=0, t1=False):
        super().__init__()
        s.q = nn.Embedding(C, d); s.q.weight.requires_grad_(False)
        s.pin = nn.Linear(d_in, d); s.img = XAttn(d, nhead, ff, drop)
        s.t1 = Tier1(d, nhead, drop) if t1 else None
        if t1: s.ein = nn.Linear(d_ehr_in, d)
        s.out = GroupFC(C, d)
    def forward(s, patch, etok=None, need_w=False):
        q = s.q.weight.unsqueeze(0).expand(patch.shape[0], -1, -1)
        kv = s.pin(patch)
        if s.t1 is not None: kv = s.t1(kv, s.ein(etok))
        h, w = s.img(q, kv, need_w); return s.out(h), w

class ASL(nn.Module):
    '''Verbatim nb11/12/13/15/35/37/38*/44*/45/52*/53/56. gn=4, gp=0 la cau hinh 35 da chot.'''
    def __init__(s, gn=4, gp=0, clip=0.05, eps=1e-8):
        super().__init__(); s.gn, s.gp, s.clip, s.eps = gn, gp, clip, eps
    def forward(s, x, y):
        y = y.float(); xp = torch.sigmoid(x); xn = 1 - xp
        if s.clip: xn = (xn + s.clip).clamp(max=1)
        lo = y*torch.log(xp.clamp(min=s.eps)) + (1-y)*torch.log(xn.clamp(min=s.eps))
        pt = xp*y + xn*(1-y); g = s.gp*y + s.gn*(1-y)
        return -(lo * torch.pow(1-pt, g)).mean()
print('kien truc san sang')
