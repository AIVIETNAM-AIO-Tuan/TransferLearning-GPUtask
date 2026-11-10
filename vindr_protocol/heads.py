"""Đầu đọc: ML-Decoder của ta (lớp NGUYÊN VĂN ô B của 85f = c3 / C2B / C3C của 78z) + tạo / chạy 3 loại đầu của giao thức 86
(Linear, GLoRI: mã CheXFound; ML-Decoder: đọc patch tầng cuối, bỏ đầu "chỉ patch" của B — PLAN_86 §4.3)."""
import torch
import torch.nn as nn

from .chexfound import GLoRI, LinearClassifier


# ==== lớp ML-Decoder — NGUYÊN VĂN ô B của 85f (test_vindr_protocol so cây cú pháp với ô gốc) ====
class XAttn(nn.Module):
    def __init__(s, d, nhead=8, ff=2048, drop=0.1):
        super().__init__()
        s.at = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
    def forward(s, q, kv, need_w=False):
        a, w = s.at(q, kv, kv, need_weights=need_w, average_attn_weights=True)
        h = s.n1(q + a); return s.n2(h + s.ff(h)), w


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


class GroupFC(nn.Module):
    def __init__(s, C, d):
        super().__init__()
        s.w = nn.Parameter(torch.randn(C, d) * d ** -0.5); s.b = nn.Parameter(torch.zeros(C))
    def forward(s, h): return (h * s.w).sum(-1) + s.b


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


def mldec_h_hr(m, q, patch, r):
    '''Dau J doc [patch ; token vung] (78y `mldec_hr`): K / V = [pin(patch) ; m.docv(pin, r)], query q [C, d] (q_cua cua 78t)
       -> (logit, h = dau ra query TRUOC GroupFC, kv = pin(patch) CHI phan patch). Nhu dau_h_v2 cua J v2 (khoi_jca_v2).'''
    kv = m.pin(patch)
    h, _ = m.img(q.unsqueeze(0).expand(patch.shape[0], -1, -1), torch.cat([kv, m.docv(m.pin, r).to(kv.dtype)], 1), False)
    return m.out(h), h, kv


def q_cua(m):
    '''Query lop [40, d] cua decoder `m` (txtq: tu text; q_hoc: hoc duoc; con lai: co dinh).'''
    return m.qtxt() if hasattr(m, 'qtxt') else m.q.weight


def mldec_h(m, patch):
    '''MOT forward decoder -> (logit, h = 40 dau ra query TRUOC GroupFC, kv = pin(patch)). CUNG phep tinh MLDecoder /
       MLDecoderTxtQ.forward (khong Tier1) -> logit TRUNG tung bit (cong C4, eval + train cung dong ngau nhien).'''
    q = q_cua(m).unsqueeze(0).expand(patch.shape[0], -1, -1)
    kv = m.pin(patch)
    h, _ = m.img(q, kv, False)
    return m.out(h), h, kv


# ==== tạo / chạy đầu của giao thức 86 ====
def build_head(head_type, seed, dim, num_classes, device, protocol):
    '''Đầu mới; khởi tạo CHỈ phụ thuộc `seed` (RNG riêng, không động RNG chung). lin = setup_linear_classifiers (4 tầng, không avgpool);
       glori = setup_glori (decoder 768, cat_cls); ml = đầu ML-Decoder của 85f (28 query cố định).'''
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if head_type == 'lin':
            m = LinearClassifier(protocol.n_last_blocks * dim, use_n_blocks=protocol.n_last_blocks, use_avgpool=False, num_classes=num_classes)
        elif head_type == 'glori':
            m = GLoRI(num_classes=num_classes, decoder_embedding=protocol.glori_dim, initial_num_features=protocol.n_last_blocks * dim,
                      use_n_blocks=protocol.n_last_blocks, cat_cls=True)
        elif head_type == 'ml':
            m = MLDecoder(num_classes, dim, d=protocol.mldecoder_dim, nhead=8, ff=1024, drop=0.1)
        else:
            raise ValueError(head_type)
    return m.to(device)


def run_head(head_type, model, blocks, last_patch):
    '''-> logit [B, C]. Linear / GLoRI nhận list các view như mã gốc ([blocks]); ML-Decoder nhận patch tầng cuối.'''
    return mldec_h(model, last_patch.float())[0] if head_type == 'ml' else model([blocks])
