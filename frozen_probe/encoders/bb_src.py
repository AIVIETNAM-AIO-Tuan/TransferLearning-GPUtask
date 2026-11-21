# -*- coding: utf-8 -*-
"""Mặt nạ encoder khác (DINOv3 / CheXFound / BiomedCLIP) + lớp luồng vùng có P token đầu — CHÉP NGUYÊN VĂN phần A và phần B của
experiments/78_chi_anh/khoi/khoi_bb_khac.py (nguồn duy nhất mà dung_78x_bb.py chép vào mọi notebook 78x / 78r của encoder khác; PLAN_92 §13).
Từ dòng `# ==== KHOI A ====` tới hết tệp: trùng tệp nguồn TỪNG KÝ TỰ (tests/test_bb.py so). Chỉ đầu tệp khác nguồn: phần B cần LopJCA / LopJCANeoTran —
lấy từ vindr_bench.encoders.b_src (bản chép nguyên văn đã kiểm của khối 79 khoi_luong_vung / 78 khoi_jca_v3b, tests/test_b.py của 89), không từ đường
dẫn module 78 / 79 (không có trên Colab)."""
import math, re, types
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from vindr_bench.encoders.b_src import LopJCA, LopJCANeoTran
# ==== KHOI A ====
import types as _ty_bb


def _ra_bb(o):
    '''Dau ra mot block: tuple / list (transformers cu) -> phan tu dau; tensor -> giu nguyen.'''
    return o[0] if isinstance(o, (tuple, list)) else o


class _ChuanVao(nn.Module):
    '''Anh duong 518 (chuan hoa RAD-DINO: (x - m_in) / s_in, 3 kenh) -> chuan hoa cua encoder ((x - m_ra) / s_ra) -> thu nho S x S
       (bilinear, antialias) khi kich thuoc khac S. float32, TAT autocast; khong tham so (buffer KHONG luu), khong ngau nhien.'''
    def __init__(s, m_in, s_in, m_ra, s_ra, S):
        super().__init__()
        f = lambda v: torch.as_tensor(np.asarray(v, dtype='float64').reshape(-1), dtype=torch.float64).expand(3).reshape(1, 3, 1, 1)
        a, b = f(s_in) / f(s_ra), (f(m_in) - f(m_ra)) / f(s_ra)          # x_ra = x * a + b
        s.register_buffer('a', a.float(), persistent=False)
        s.register_buffer('b', b.float(), persistent=False)
        s.S = int(S)

    def forward(s, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float() * s.a + s.b
            if x.shape[-1] != s.S or x.shape[-2] != s.S:
                x = F.interpolate(x, size=(s.S, s.S), mode='bilinear', align_corners=False, antialias=True)
        return x


class _ChuanCuoi(nn.Module):
    '''norm cuoi cua encoder + BO P - 1 token register (giu CLS o 0 va token patch) -> [B, 1 + N, d].'''
    def __init__(s, norm, P):
        super().__init__()
        s.norm, s.P = norm, int(P)

    def forward(s, h):
        n = s.norm(h)
        return n if s.P == 1 else torch.cat([n[:, :1], n[:, s.P:]], 1)


class _HangRope(nn.Module):
    '''(cos, sin) RoPE cua DINOv3 cho MOT luoi, tinh MOT lan o eval (khong dich / gian / jitter ngau nhien), buffer KHONG luu.'''
    def __init__(s, rope, S):
        super().__init__()
        tr = rope.training
        rope.eval()
        with torch.no_grad():
            c, n = rope(torch.zeros(1, 3, int(S), int(S), dtype=torch.float32, device=rope.inv_freq.device))
        rope.train(tr)
        s.register_buffer('cos', c.float(), persistent=False)
        s.register_buffer('sin', n.float(), persistent=False)


class _LopRope(nn.Module):
    '''Block DINOv3 nhan RoPE hang: forward(h) = lop(h, position_embeddings=(cos, sin)).'''
    def __init__(s, lop, hang):
        super().__init__()
        s.lop, s.hang = lop, hang

    def forward(s, h):
        return _ra_bb(s.lop(h, position_embeddings=(s.hang.cos, s.hang.sin)))


class _NhungHF(nn.Module):
    '''embeddings HF (DINOv3ViTEmbeddings / Dinov2Embeddings): chuan_vao -> emb(pixel_values) -> [CLS, register, patch].'''
    def __init__(s, chuan_vao, emb):
        super().__init__()
        s.chuan_vao, s.emb = chuan_vao, emb

    def forward(s, img):
        return s.emb(s.chuan_vao(img))


class _NhungTimm(nn.Module):
    '''embeddings cua VisionTransformer timm (class token, pos_embed tuyet doi gom ca vi tri CLS, KHONG register, KHONG kich thuoc dong):
       = patch_embed -> _pos_embed -> patch_drop -> norm_pre cua timm.forward_features (chi phan truoc blocks).'''
    def __init__(s, chuan_vao, trunk):
        super().__init__()
        assert getattr(trunk, 'num_prefix_tokens', 1) == 1 and getattr(trunk, 'cls_token', None) is not None, 'timm ViT phai co CLS, khong register'
        assert not getattr(trunk, 'no_embed_class', False) and not getattr(trunk, 'dynamic_img_size', False), 'timm ViT: cau hinh pos_embed la'
        s.chuan_vao = chuan_vao
        s.patch_embed, s.cls_token, s.pos_embed = trunk.patch_embed, trunk.cls_token, trunk.pos_embed
        s.pos_drop, s.patch_drop, s.norm_pre = trunk.pos_drop, trunk.patch_drop, trunk.norm_pre

    def forward(s, img):
        x = s.patch_embed(s.chuan_vao(img))
        x = torch.cat([s.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + s.pos_embed
        return s.norm_pre(s.patch_drop(s.pos_drop(x)))


class MatNaBB(nn.Module):
    '''Giao dien Dinov2 cho encoder khac (xem dau tep). ten / n_dau (P) / luoi / kich_thuoc; config: hidden_size, num_hidden_layers,
       patch_size, num_attention_heads, image_size, cac ti le dropout = 0.'''
    def __init__(s, ten, nhung, lops, norm, P, d, patch, S, n_head):
        super().__init__()
        assert int(S) % int(patch) == 0, (S, patch)
        s.ten, s.n_dau, s.kich_thuoc, s.luoi = str(ten), int(P), int(S), int(S) // int(patch)
        s.embeddings = nhung
        s.encoder = nn.Module()
        s.encoder.layer = nn.ModuleList(lops)
        s.layernorm = _ChuanCuoi(norm, P)
        s.config = _ty_bb.SimpleNamespace(hidden_size=int(d), num_hidden_layers=len(lops), patch_size=int(patch), num_attention_heads=int(n_head),
                                          image_size=int(S), attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0, drop_path_rate=0.0,
                                          ten=str(ten), n_dau=int(P))

    def forward(s, pixel_values, **kw):
        h = s.embeddings(pixel_values)
        for lop in s.encoder.layer: h = _ra_bb(lop(h))
        return _ty_bb.SimpleNamespace(last_hidden_state=s.layernorm(h))


def gan_goc(mb, kieu, goc):
    '''Giu tham chieu mo hinh GOC (KHONG dang ky module con, KHONG vao state_dict / parameters) cho cong so_goc. kieu 'hf' | 'timm'.'''
    object.__setattr__(mb, '_goc_bb', (str(kieu), goc))
    return mb


@torch.no_grad()
def so_goc(mb, x):
    '''Forward GOC cua encoder (HF: model(pixel_values).last_hidden_state; timm: forward_features) tren anh DA qua _ChuanVao cua mat na,
       bo register -> [B, 1 + N, d]. Cong trong notebook: mat na (eval) phai TRUNG tung bit.'''
    kieu, goc = mb._goc_bb
    xx = mb.embeddings.chuan_vao(x)
    if kieu == 'timm': return goc.forward_features(xx)
    o = goc(pixel_values=xx).last_hidden_state
    return o if mb.n_dau == 1 else torch.cat([o[:, :1], o[:, mb.n_dau:]], 1)


def bo_goc(mb):
    '''Bo tham chieu mo hinh goc sau cong (giai phong phan rieng cua mo hinh goc, vd RoPE / pooler).'''
    object.__setattr__(mb, '_goc_bb', None)


def tao_dinov3(goc, m_in, s_in, m_ra, s_ra, S=512):
    '''goc = DINOv3ViTModel (HF, DA nap trong so; drop_path_rate 0, attention_dropout 0) -> MatNaBB (P = 1 + so register).'''
    cfg = goc.config
    assert float(getattr(cfg, 'drop_path_rate', 0.0)) == 0.0 and float(getattr(cfg, 'attention_dropout', 0.0)) == 0.0, \
        'DINOv3: drop_path_rate / attention_dropout phai 0 (stochastic depth cua B.4 o muc block) -> DUNG'
    lops = goc.model.layer if hasattr(goc, 'model') and hasattr(goc.model, 'layer') else (goc.layer if hasattr(goc, 'layer') else goc.encoder.layer)
    hang = _HangRope(goc.rope_embeddings, S)
    P = 1 + int(cfg.num_register_tokens)
    return gan_goc(MatNaBB('dinov3', _NhungHF(_ChuanVao(m_in, s_in, m_ra, s_ra, S), goc.embeddings), [_LopRope(l, hang) for l in lops], goc.norm,
                           P, cfg.hidden_size, cfg.patch_size, S, cfg.num_attention_heads), 'hf', goc)


def tao_dinov2_hf(goc, m_in, s_in, m_ra, s_ra, S=512, ten='dinov2'):
    '''goc = Dinov2Model / Dinov2WithRegistersModel (HF, DA nap trong so, drop_path_rate 0) -> MatNaBB (P = 1 + so register).'''
    cfg = goc.config
    assert float(getattr(cfg, 'drop_path_rate', 0.0)) == 0.0 and float(getattr(cfg, 'attention_probs_dropout_prob', 0.0)) == 0.0 \
        and float(getattr(cfg, 'hidden_dropout_prob', 0.0)) == 0.0, 'Dinov2: dropout / drop_path phai 0 -> DUNG'
    P = 1 + int(getattr(cfg, 'num_register_tokens', 0))
    return gan_goc(MatNaBB(ten, _NhungHF(_ChuanVao(m_in, s_in, m_ra, s_ra, S), goc.embeddings), list(goc.encoder.layer), goc.layernorm, P,
                           cfg.hidden_size, cfg.patch_size, S, cfg.num_attention_heads), 'hf', goc)


def tao_timm_vit(trunk, m_in, s_in, m_ra, s_ra, S=512, ten='timm_vit'):
    '''trunk = VisionTransformer timm (CLS, KHONG register, pos_embed DA noi suy toi luoi S / patch) -> MatNaBB (P = 1).'''
    patch = int(trunk.patch_embed.patch_size[0])
    g = int(S) // patch
    assert tuple(trunk.pos_embed.shape[1:]) == (1 + g * g, trunk.embed_dim), (
        f'pos_embed {tuple(trunk.pos_embed.shape)} != luoi {g} x {g} + CLS -> noi suy truoc (noi_suy_pos)')
    return gan_goc(MatNaBB(ten, _NhungTimm(_ChuanVao(m_in, s_in, m_ra, s_ra, S), trunk), list(trunk.blocks), trunk.norm, 1,
                           trunk.embed_dim, patch, S, trunk.blocks[0].attn.num_heads), 'timm', trunk)


def noi_suy_pos(pe, g_moi, n_dau=1):
    '''pos_embed [1, n_dau + g * g, d] -> [1, n_dau + g_moi^2, d]: phan luoi noi suy bicubic (float32, align_corners False, antialias=True
       ca khi phong - DUNG nhu timm resample_abs_pos_embed; voi bicubic antialias doi he so ca khi phong), phan dau giu nguyen.'''
    dau, luoi = pe[:, :n_dau], pe[:, n_dau:]
    g = int(round(math.sqrt(luoi.shape[1])))
    assert g * g == luoi.shape[1], luoi.shape
    if g == int(g_moi): return pe.clone()
    x = luoi.reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(int(g_moi), int(g_moi)), mode='bicubic', align_corners=False, antialias=True)
    return torch.cat([dau, x.permute(0, 2, 3, 1).reshape(1, int(g_moi) * int(g_moi), -1).to(pe.dtype)], 1)


def _bo_vo_dinov2(sd):
    '''Boc vo checkpoint DINOv2 goc: {'teacher': ...} / {'model': ...} / {'state_dict': ...} (lop ngoai KHONG co tensor) -> dict khoa -> tensor.'''
    for _ in range(3):
        if any(torch.is_tensor(v) for v in sd.values()): break
        k = next((t for t in ('teacher', 'model', 'state_dict') if isinstance(sd.get(t), dict)), None)
        assert k is not None, f'checkpoint DINOv2: khong thay state_dict (khoa ngoai {sorted(sd)[:6]}) -> DUNG'
        sd = sd[k]
    return sd


def doi_khoa_dinov2_goc(sd, khoa_hf):
    '''state_dict DINOv2 GOC (facebookresearch; co the boc 'teacher' / 'model', tien to 'backbone.', block chunk 'blocks.<c>.<i>.') ->
       state_dict theo khoa `khoa_hf` cua Dinov2Model / Dinov2WithRegistersModel HF dang chay (ten attention CU attention.attention.* hoac
       MOI q_proj / k_proj / v_proj / o_proj - theo khoa_hf). qkv gop [3d, d] tach q, k, v. Thieu layer scale -> 1. Khoa la / thieu -> DUNG.'''
    sd = _bo_vo_dinov2(sd)
    g = {}
    for k, v in sd.items():
        if not torch.is_tensor(v): continue
        if k.startswith('backbone.'): k = k[len('backbone.'):]
        elif any(k.startswith(t) for t in ('dino_head.', 'ibot_head.', 'head.')): continue
        g[re.sub(r'^blocks\.\d+\.(\d+)\.', r'blocks.\1.', k) if re.match(r'^blocks\.\d+\.\d+\.', k) else k] = v
    khoa_hf = set(khoa_hf)
    moi = any('.attention.q_proj.' in k for k in khoa_hf)
    ra, dung = {}, set()

    def dat(kh, v, kg):
        assert kh in khoa_hf, f'khoa HF {kh} (tu {kg}) khong co trong mo hinh -> cau hinh HF lech checkpoint, DUNG'
        ra[kh] = v; dung.add(kg)

    def lay(kg):
        assert kg in g, f'doi khoa DINOv2 goc: thieu khoa goc {kg} -> checkpoint khong dung dang DINOv2, DUNG (bao Claude)'
        return g[kg]
    for a, b in (('cls_token', 'embeddings.cls_token'), ('mask_token', 'embeddings.mask_token'), ('register_tokens', 'embeddings.register_tokens'),
                 ('pos_embed', 'embeddings.position_embeddings'), ('patch_embed.proj.weight', 'embeddings.patch_embeddings.projection.weight'),
                 ('patch_embed.proj.bias', 'embeddings.patch_embeddings.projection.bias'), ('norm.weight', 'layernorm.weight'),
                 ('norm.bias', 'layernorm.bias')):
        if a in g and (a != 'mask_token' or b in khoa_hf): dat(b, g[a], a)     # mask_token: CHI khi HF co (use_mask_token)
    if 'embeddings.mask_token' in khoa_hf and 'embeddings.mask_token' not in ra:
        d = g['cls_token'].shape[-1]; ra['embeddings.mask_token'] = torch.zeros(1, d, dtype=g['cls_token'].dtype)
    nb = sorted({int(m.group(1)) for k in g for m in [re.match(r'^blocks\.(\d+)\.', k)] if m})
    att = (lambda i, t: f'encoder.layer.{i}.attention.{t}_proj') if moi else (lambda i, t: f'encoder.layer.{i}.attention.attention.{t}')
    out = (lambda i: f'encoder.layer.{i}.attention.o_proj') if moi else (lambda i: f'encoder.layer.{i}.attention.output.dense')
    for i in nb:
        p, e = f'blocks.{i}.', f'encoder.layer.{i}.'
        for t in ('norm1', 'norm2'):
            for w in ('weight', 'bias'): dat(f'{e}{t}.{w}', lay(f'{p}{t}.{w}'), f'{p}{t}.{w}')
        W = lay(f'{p}attn.qkv.weight'); d = W.shape[1]
        for j, t in enumerate(('q', 'k', 'v') if moi else ('query', 'key', 'value')):
            dat(f'{att(i, t)}.weight', W[j * d:(j + 1) * d].clone(), f'{p}attn.qkv.weight')
            if f'{p}attn.qkv.bias' in g: dat(f'{att(i, t)}.bias', g[f'{p}attn.qkv.bias'][j * d:(j + 1) * d].clone(), f'{p}attn.qkv.bias')
        for w in ('weight', 'bias'):
            if f'{p}attn.proj.{w}' in g: dat(f'{out(i)}.{w}', g[f'{p}attn.proj.{w}'], f'{p}attn.proj.{w}')
        for a, b in (('ls1.gamma', 'layer_scale1.lambda1'), ('ls2.gamma', 'layer_scale2.lambda1')):
            if f'{p}{a}' in g: dat(f'{e}{b}', g[f'{p}{a}'], f'{p}{a}')
            elif f'{e}{b}' in khoa_hf: ra[f'{e}{b}'] = torch.ones(d, dtype=W.dtype)
        if f'{p}mlp.fc1.weight' in g:                                    # FFN MLP: fc1 / fc2 (moi ban transformers)
            for a in ('mlp.fc1', 'mlp.fc2'):
                for w in ('weight', 'bias'):
                    if f'{p}{a}.{w}' in g: dat(f'{e}{a}.{w}', g[f'{p}{a}.{w}'], f'{p}{a}.{w}')
        elif f'{e}mlp.gate_proj.weight' in khoa_hf:                      # SwiGLU, transformers >= 5.18 (Colab 5.20.1): w12 = [gate ; up]
            h = lay(f'{p}mlp.w12.weight').shape[0] // 2                  # (silu o NUA DAU, nhu DINOv2 / xformers), w3 = down
            for w in ('weight', 'bias'):
                if f'{p}mlp.w12.{w}' in g:
                    x = g[f'{p}mlp.w12.{w}']
                    dat(f'{e}mlp.gate_proj.{w}', x[:h].clone(), f'{p}mlp.w12.{w}'); dat(f'{e}mlp.up_proj.{w}', x[h:].clone(), f'{p}mlp.w12.{w}')
                if f'{p}mlp.w3.{w}' in g: dat(f'{e}mlp.down_proj.{w}', g[f'{p}mlp.w3.{w}'], f'{p}mlp.w3.{w}')
        else:                                                            # SwiGLU, transformers cu: weights_in (= w12) / weights_out (= w3)
            for a, b in (('mlp.w12', 'mlp.weights_in'), ('mlp.w3', 'mlp.weights_out')):
                for w in ('weight', 'bias'):
                    if f'{p}{a}.{w}' in g: dat(f'{e}{b}.{w}', g[f'{p}{a}.{w}'], f'{p}{a}.{w}')
    la = sorted(set(g) - dung - {'mask_token'})
    thieu = sorted(khoa_hf - set(ra))
    assert not la and not thieu, f'doi khoa DINOv2 goc: khoa goc KHONG dung {la[:6]} | khoa HF thieu {thieu[:6]} -> DUNG (bao Claude)'
    return ra


def cau_hinh_dinov2_goc(sd):
    '''Suy cau hinh HF tu HINH tensor cua state_dict DINOv2 goc (sau khi bo 'teacher' / 'backbone.'): dict(d, L, patch, n_reg, luoi,
       mlp_ratio, swiglu, qkv_bias, n_head = d / 64).'''
    sd = _bo_vo_dinov2(sd)
    g = {(k[len('backbone.'):] if k.startswith('backbone.') else k): v for k, v in sd.items() if torch.is_tensor(v)}
    g = {re.sub(r'^blocks\.\d+\.(\d+)\.', r'blocks.\1.', k) if re.match(r'^blocks\.\d+\.\d+\.', k) else k: v for k, v in g.items()}
    d = int(g['cls_token'].shape[-1]); patch = int(g['patch_embed.proj.weight'].shape[-1])
    L = 1 + max(int(re.match(r'^blocks\.(\d+)\.', k).group(1)) for k in g if re.match(r'^blocks\.\d+\.', k))
    n_reg = int(g['register_tokens'].shape[1]) if 'register_tokens' in g else 0
    n_pos = int(g['pos_embed'].shape[1]) - 1; luoi = int(round(math.sqrt(n_pos)))
    assert luoi * luoi == n_pos, n_pos
    swiglu = 'blocks.0.mlp.w12.weight' in g
    an = int(g['blocks.0.mlp.w12.weight'].shape[0]) // 2 if swiglu else int(g['blocks.0.mlp.fc1.weight'].shape[0])
    assert d % 64 == 0, d
    # mlp_ratio cua HF Dinov2Config la SO NGUYEN danh nghia (config strict): MLP an = d r; SwiGLU (HF va DINOv2 SwiGLUFFNFused cung lam tron)
    # an = (int(int(d r) * 2 / 3) + 7) // 8 * 8 -> tim r (CheXFound ViT-L SwiGLU: an 2736 -> r 4; ban 23/10 truyen 2736 / 1024 -> loi Colab)
    rs = [r for r in range(1, 33) if (((int(int(d * r) * 2 / 3) + 7) // 8 * 8) if swiglu else d * r) == an]
    assert len(rs) == 1, f'FFN an {an} (d {d}, swiglu {swiglu}) khong ung voi DUNG mot mlp_ratio nguyen cua HF Dinov2 ({rs}) -> DUNG (bao Claude)'
    return dict(d=d, L=L, patch=patch, n_reg=n_reg, luoi=luoi, mlp_ratio=rs[0], swiglu=swiglu, qkv_bias='blocks.0.attn.qkv.bias' in g,
                n_head=d // 64, ls='blocks.0.ls1.gamma' in g)
# ==== HET A ====
# ==== KHOI B ====


class LopJCAP(LopJCA):
    '''LopJCA (79, CHI DOC) voi P = n_dau token DAU (CLS + register): token patch = h[:, P:]; thien lech ghi nguoc them P hang 0 (token dau
       doc token vung KHONG thien lech, nhu CLS cua LopJCA). P = 1 -> TRUNG LopJCA.forward tung bit. n_dau la thuoc tinh lop / doi tuong.'''
    n_dau = 1

    def forward(s, h, r, hop, p, mo_ghi):
        P = int(s.n_dau)
        b_r = s.bias_vung(hop)
        x = s.ln_sa(r + p)
        r1 = r + s.sa(x, x)
        r2 = r1 + s.ca_r(s.ln_r(r1 + p), s.ln_h(h[:, P:]), b_r)
        r3 = r2 + s.ff(s.ln_ff(r2))
        ti = None
        if mo_ghi:
            bh = None if b_r is None else torch.cat(
                [b_r.new_zeros(b_r.shape[0], 1, P, b_r.shape[2]), b_r.transpose(-1, -2)], 2)
            dh = s.ca_h(s.ln_h2(h), s.ln_r2(r + p), bh)
            with torch.no_grad():
                ti = dh.float().norm() / h.float().norm().clamp_min(1e-12)
            h = h + dh
        if s.neo:
            hop = torch.sigmoid(torch.logit(hop.float().clamp(1e-4, 1.0 - 1e-4)) + s.mlp_hop(r3).float())
        return h, r3, hop, ti


class LopJCANeoTranP(LopJCANeoTran, LopJCAP):
    '''J v3b (neo theo mau + tran write-back) voi P token dau. MRO: LopJCANeoTran -> LopJCANeo -> LopJCAP -> LopJCA.'''
    pass


def ap_n_dau(luong, P):
    '''Moi lop cua `luong` (PHAI la LopJCANeoTran, tuc SAU ap_tran_ghi) -> LopJCANeoTranP voi n_dau = P (>= 1). Khong them tham so / buffer.
       Tra so lop da doi.'''
    P = int(P)
    assert P >= 1, P
    n = 0
    for l in luong.lop:
        assert type(l) is LopJCANeoTran, f'ap_n_dau: lop {type(l).__name__} khong phai LopJCANeoTran (goi SAU ap_tran_ghi) -> DUNG'
        l.__class__ = LopJCANeoTranP; l.n_dau = P; n += 1
    return n
