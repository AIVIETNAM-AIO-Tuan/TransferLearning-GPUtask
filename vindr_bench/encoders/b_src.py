# -*- coding: utf-8 -*-
"""Mã forward của B (78z skip7: RAD-DINO tinh chỉnh + luồng vùng J v3b có trần ghi ngược + skip block 7 từ 5 ô) — CHỈ phần cần để chạy B ĐÔNG CỨNG.

Mọi định nghĩa dưới đây CHÉP NGUYÊN VĂN (không sửa chữ nào) từ các tệp khối đã chạy ở 78x / 78y / 78z / 85f / 86b_B:
experiments/79_phat_minh/khoi/khoi_luong_vung.py, experiments/78_chi_anh/khoi/{khoi_jca_v2, khoi_jca_v3, khoi_jca_v3b, khoi_doc_dong_cung, khoi_cat_vung,
khoi_cat_sau, khoi_cat_j}.py và ô C2 của 86b_B (đổi tên khoá Dinov2). Notebook 85f / 86b_B nhúng đúng các đoạn này vào ô; tests/test_b.py so cây cú pháp
từng định nghĩa với tệp nguồn VÀ với ô của 86b_B. Phần chỉ dùng lúc học (loss, câu chữ, nhãn vùng, ...) KHÔNG chép.

Lớp bọc dùng các định nghĩa này: encoders/b.py. Chỉ dùng cú pháp / API có ở Python 3.9 + torch 2.0."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _tuc_luong          # = tên trong khoi_luong_vung (nhánh tính lại kích hoạt chỉ dùng khi học)


# ---- experiments/79_phat_minh/khoi/khoi_luong_vung.py (NGUYÊN VĂN): _out_luong, _ma_hoa_hop, mat_na_hop, CrossAttnBias, LopJCA, LuongVung ----
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


# ---- experiments/78_chi_anh/khoi/khoi_jca_v2.py (NGUYÊN VĂN): LopJCANeo, LuongVungNeo ----
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


# ---- experiments/78_chi_anh/khoi/khoi_jca_v3.py (NGUYÊN VĂN): LinearNhan, tang_ghi_nguoc ----
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


# ---- experiments/78_chi_anh/khoi/khoi_jca_v3b.py (NGUYÊN VĂN): LopJCANeoTran, ap_tran_ghi ----
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


# ---- experiments/78_chi_anh/khoi/khoi_doc_dong_cung.py (NGUYÊN VĂN): upper_luong_hr ----
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


# ---- experiments/78_chi_anh/khoi/khoi_cat_vung.py (NGUYÊN VĂN): S_1022, O_1022, luoi_518, lay_mau_518 ----
S_1022, S_518, CANH_O = 1022, 518, 518


O_1022 = ((0, 0), (504, 0), (0, 504), (504, 504), (252, 252))     # (x0, y0) goc tren-trai tung o tren anh 1022: 2 x 2 + o giua


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


# ---- experiments/78_chi_anh/khoi/khoi_cat_sau.py (NGUYÊN VĂN): N_LUOI, TI_1022, hinh_luoi, luoi_con, lay_mau_con, DuaVao ----
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


# ---- experiments/78_chi_anh/khoi/khoi_cat_j.py (NGUYÊN VĂN): lay_mau_con_grad, o_phong_518, BoChinhO, skip_o, cong_skip, token_o7 ----
lay_mau_con_grad = lay_mau_con.__wrapped__        # CUNG phep tinh voi lay_mau_con (khoi_cat_sau, boc @torch.no_grad) nhung CO gradient


def o_phong_518(img, luoi, mean, std):
    '''img [B, 3, 518, 518] float = anh 518 DA chuan hoa ((x / 255 - mean) / std, 3 kenh giong nhau) -> 5 o [B, K, 518, 518] float32 CUNG
       chuan hoa: bicubic tren kenh 0 (lay_mau_518, align_corners=False, padding border), kep ve khoang chuan hoa cua [0, 255] (nhu kep
       uint8 cua cat_518_phong; khong lam tron muc xam).'''
    o = lay_mau_518(img[:, 0], luoi)
    return o.clamp((0.0 - float(mean)) / float(std), (1.0 - float(mean)) / float(std))


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


# ---- 86_vindr_giao_thuc/86b_B.ipynb ô C2 (= C2 của 78z / 85f, NGUYÊN VĂN): _DOI_TEN_DINOV2, doi_ten_dinov2 ----
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
