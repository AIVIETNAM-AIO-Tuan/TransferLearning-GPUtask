# ==== nhanh_c buoc z_C3B — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C3b · HAM MAT MAT — diem DUY NHAT khac 64 (chep nguyen van C3b cua 74a) ====
# Lop loss lay NGUYEN VAN tu nb72 C6 (da qua cong bit-for-bit: ASLel==ASL cua 68b,
# WASL(w=1)==ASL(4,0), TronCot(A,A)==A). Cong do duoc lap lai ngay duoi.
import torch.nn.functional as F

CHUAN_HOA_DBL = True        # xem chu thich o DBLChuan

class WASL(nn.Module):
    """ASL + trong so lien lop CHI tren so hang duong. w=1 -> identical ASL. Verbatim nb12."""
    def __init__(s, w, gn=4, gp=0, clip=0.05, eps=1e-8):
        super().__init__(); s.gn, s.gp, s.clip, s.eps = gn, gp, clip, eps
        s.register_buffer('w', torch.as_tensor(np.asarray(w, 'float32')))
    def el(s, x, y):
        y = y.float(); xp = torch.sigmoid(x); xn = 1 - xp
        if s.clip: xn = (xn + s.clip).clamp(max=1)
        lp = y*torch.log(xp.clamp(min=s.eps)); ln = (1-y)*torch.log(xn.clamp(min=s.eps))
        pt = xp*y + xn*(1-y); g = s.gp*y + s.gn*(1-y); mod = torch.pow(1-pt, g)
        return -(s.w*lp)*mod - ln*mod
    def forward(s, x, y): return s.el(x, y).mean()


class DBLoss(nn.Module):
    """Distribution-Balanced Loss (Wu ECCV'20) — verbatim PlanB_LPFT_MV73_DBML.ipynb."""
    def __init__(s, class_freq, train_num, map_a=0.1, map_b=10.0, map_g=0.3,
                 neg_scale=2.0, init_bias=0.05, eps=1e-6):
        super().__init__()
        cf = torch.tensor(class_freq, dtype=torch.float32)
        s.register_buffer('freq_inv', 1.0/(cf+eps))
        s.register_buffer('bias', -torch.log(train_num/(cf+eps)-1)*init_bias)
        s.a, s.b, s.g, s.neg, s.eps = map_a, map_b, map_g, neg_scale, eps
    def rw(s, y):
        repeat = (y*s.freq_inv).sum(1, keepdim=True)
        return s.a + torch.sigmoid(s.b*(s.freq_inv/(repeat+s.eps) - s.g))
    def el(s, z, y):
        y = y.float(); w = s.rw(y); zb = z - s.bias
        return -(w*(y*F.logsigmoid(zb) + (1-y)*(1.0/s.neg)*F.logsigmoid(-s.neg*zb)))
    def forward(s, z, y): return s.el(z, y).mean()


class TronCot(nn.Module):
    """Loss `a` cho mot so cot, `b` cho phan con lai. Ghep o muc PHAN TU."""
    def __init__(s, a, b, cot_a, C_):
        super().__init__(); s.a, s.b = a, b
        m = torch.zeros(C_); m[np.asarray(cot_a)] = 1.0
        s.register_buffer('m', m)
    def forward(s, x, y): return (s.m*s.a.el(x, y) + (1-s.m)*s.b.el(x, y)).mean()


class DBLChuan(nn.Module):
    """DBL verbatim, chia them cho mean(w) do tren mot mau train.

       Ly do: trong so rebalancing cua DBL nam trong [0.1, 1.1] va mean THUC TE
       khac 1. Mean != 1 la mot phep doi LEARNING RATE, khong phai doi HINH DANG
       tai trong so. Khong chuan hoa thi phep so tron lan hai thu, va ta khong
       biet DBL thua vi tai trong so sai hay chi vi no hoc cham hon.
       Dat CHUAN_HOA_DBL=False de co ban bit-exact Wu et al."""
    def __init__(s, db, k):
        super().__init__(); s.db = db; s.k = float(k)
    def el(s, z, y): return s.db.el(z, y) / s.k
    def forward(s, z, y): return s.el(z, y).mean()


class ASLel(ASL):
    """`ASL` cua 68b/70a chi co `forward`, khong co `el()` — ma `TronCot` can `el()`
       de ghep hai loss o muc PHAN TU. Lop con nay them `el()` va dinh nghia lai
       `forward = el().mean()`. Cong ngay duoi kiem no KHONG lam doi `forward`.

       (nb73 da co lop nay tu dau; nb72 bi quen — loi lo ra khi chay that 17/09.)"""
    def el(s, x, y):
        y = y.float(); xp = torch.sigmoid(x); xn = 1 - xp
        if s.clip: xn = (xn + s.clip).clamp(max=1)
        lo = y*torch.log(xp.clamp(min=s.eps)) + (1-y)*torch.log(xn.clamp(min=s.eps))
        pt = xp*y + xn*(1-y); g = s.gp*y + s.gn*(1-y)
        return -(lo * torch.pow(1-pt, g))
    def forward(s, x, y): return s.el(x, y).mean()


def class_weights(n_pos, alpha, lo=0.5, hi=3.0, tol=1e-12, iters=200):
    """w = clip(s*n^-alpha, lo, hi), s chia doi de mean(w)=1. Verbatim nb12/15.
       BO KEP = lo=0.0, hi=np.inf."""
    n = np.asarray(n_pos, 'float64'); assert np.all(n > 0)
    if alpha == 0: return np.ones(len(n))
    x = n ** (-alpha)
    f = lambda s: np.clip(s*x, lo, hi).mean() - 1.0
    s_lo, s_hi = 1e-12, 1.0
    while f(s_hi) < 0: s_hi *= 2.0
    for _ in range(iters):
        s = 0.5*(s_lo + s_hi)
        if f(s) < 0: s_lo = s
        else:        s_hi = s
        if s_hi - s_lo < tol*max(1.0, s_hi): break
    return np.clip(0.5*(s_lo+s_hi)*x, lo, hi)

N_POS_TF = YM[I_TF].sum(0)
assert N_POS_TF.min() > 0 and N_POS_TF.max() < len(I_TF), 'DBL se chia 0 / log(<=0)'


def _dbl():
    d = DBLoss(N_POS_TF, float(len(I_TF)))
    if not CHUAN_HOA_DBL: return d
    _k = float(d.to(dev).rw(torch.from_numpy(YM[I_TF[:8192]]).to(dev)).mean())
    return DBLChuan(d, _k)


_W = lambda a: WASL(class_weights(N_POS_TF, a, 0.0, np.inf), gn=4, gp=0)

# ---- cong bit-for-bit: bien the phai TRUNG KHIT ASL(4,0) o truong hop suy bien ----
torch.manual_seed(0)
_x = torch.randn(32, C, device=dev); _y = (torch.rand(32, C, device=dev) < .15).float()
assert torch.equal(ASLel(gn=4, gp=0).to(dev)(_x, _y), ASL(gn=4, gp=0).to(dev)(_x, _y)), \
    'ASLel != ASL cua 68b — them el() da lam doi forward'
assert torch.equal(WASL(np.ones(C), gn=4, gp=0).to(dev)(_x, _y),
                   ASLel(gn=4, gp=0).to(dev)(_x, _y)), 'WASL(w=1) != ASL(4,0)'
assert torch.equal(TronCot(ASLel(gn=4, gp=0), ASLel(gn=4, gp=0), TIER['tail'], C).to(dev)(_x, _y),
                   ASLel(gn=4, gp=0).to(dev)(_x, _y)), 'TronCot(A,A) != A'
print('cong: ASLel==ASL cua 68b | WASL(w=1)==ASL(4,0) | TronCot(A,A)==A')

LOSS_78D = 'asl40'     # 'asl40' = ASL(4,0) nhu 64/r2a (MAC DINH, docs/PLAN_CHI_ANH.md); 'wasl25' = nhu 75a / 74a
LOSS_MOI = (lambda: ASLel(gn=4, gp=0)) if LOSS_78D == 'asl40' else (lambda: _W(0.25))
TEN_LOSS = {'asl40': 'asl40', 'wasl25': 'wasl25_nokep'}[LOSS_78D]
print(f'\nLOSS CUA LUOT NAY: {TEN_LOSS}')
print(f'  moc so sanh = 64/r2a s42 (CUNG seed, CUNG moi thu, loss ASL(4,0)), arm tpb:')
print(f'    ep4 0.2550 | ep6 0.2552 | ep10 0.2524   <- nhat ky 64 (61 C5)')
print(f'  san nhieu seed cua 64: |s42 0.2524 - s43 0.2556| = 0.0032 (ep10)')
print(f'  -> quyet dinh o muc ENSEMBLE (nhu 74d), khong o muc thanh vien.')
_w25 = class_weights(N_POS_TF, 0.25, 0.0, np.inf)
print(f'  n+ tren I_TF: {N_POS_TF.min():.0f}..{N_POS_TF.max():.0f} ({N_POS_TF.max()/N_POS_TF.min():.0f}x)')
print(f'  alpha=0.25 KHONG kep: w {_w25.min():.3f}..{_w25.max():.3f} | mean {_w25.mean():.6f}')
