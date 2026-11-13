# -*- coding: utf-8 -*-
"""Đầu ML-Decoder của giao thức (PLAN_89 §3): token patch TẦNG CUỐI -> C query lớp cố định (không học) -> cross-attention 1 tầng -> GroupFC.

Lớp ML-Decoder (XAttn, Tier1, GroupFC, MLDecoder) chép NGUYÊN VĂN từ vindr_protocol/heads.py của module 86 (= ô B của 85f; tests/test_ver1.py so cây
cú pháp với tệp nguồn). Cấu hình = đầu 'ml' của 86 (build_head): d = 512, 8 head, FFN 1024, dropout 0,1, không Tier1.

Ver1 (mã CheXFound): linear_glori.py không có đầu này -> tệp chạy (protocols/chexfound_env/launch.py) thay hàm setup_glori của họ bằng
setup_mldecoder (cùng chữ ký, cùng tên đầu "linear:blocks=1:avgpool=False:lr=…", cùng nhóm tham số / DDP) khi chạy với --glori --n-last-blocks 1:
optimizer, lịch lr, loss, chấm, chọn lr, pha 2 đều là mã của họ. Đầu đọc nhận đặc trưng của họ (tuple n tầng × (patch, vector)), dùng patch tầng cuối
ở float32 như create_mldecoder_input của GLoRI.

Chỉ cú pháp / API có ở Python 3.9 + torch 2.0."""
import torch
import torch.nn as nn

MLDECODER_DIM, MLDECODER_HEADS, MLDECODER_FF, MLDECODER_DROPOUT = 512, 8, 1024, 0.1     # = build_head('ml') của 86 (Protocol.mldecoder_dim = 512)


# ==== lớp ML-Decoder — NGUYÊN VĂN vindr_protocol/heads.py (86) dòng 10–56 = ô B của 85f ====
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


# ==== giao diện đầu đọc của mã CheXFound (Ver1) ====
class MLDecoderOnTheirFeatures(nn.Module):
    '''forward(x): x = list các view như mã họ đưa vào đầu đọc ([đặc trưng]); đặc trưng = tuple n tầng cuối × (patch [B, T, D], vector [B, D]).
       -> logit [B, C] từ patch của tầng CUỐI (view 0), float32.'''
    def __init__(self, num_classes, d_in, d=MLDECODER_DIM, nhead=MLDECODER_HEADS, ff=MLDECODER_FF, drop=MLDECODER_DROPOUT):
        super().__init__()
        self.decoder = MLDecoder(num_classes, d_in, d=d, nhead=nhead, ff=ff, drop=drop)

    def forward(self, x):
        return self.decoder(x[0][-1][0].float())[0]


def head_name(n, avgpool, lr):
    '''Tên đầu theo cách của setup_glori / setup_linear_classifiers của họ (runner đọc lr từ tên).'''
    return f"linear:blocks={n}:avgpool={avgpool}:lr={lr:.10f}".replace(".", "_")


def setup_mldecoder(sample_output, n_last_blocks_list, learning_rates, avgpools=(False,), num_classes=14, multiview=False, decoder_dim=768,
                    cat_cls=False):
    '''Thay setup_glori (chexfound/eval/classification/utils.py @ 7e288d0) — CÙNG chữ ký và cấu trúc vòng lặp (n tầng × avgpool × lr), cùng tên
       đầu, cùng {"params", "lr"} mỗi đầu, AllClassifiers + DDP khi phân tán như hàm gốc; khác đúng lớp đầu đọc. decoder_dim (của GLoRI) bỏ qua.
       Chỉ nhận đúng cấu hình của giao thức: một view, không cat_cls, n_last_blocks = [1] (đầu chỉ đọc tầng cuối).'''
    import chexfound.distributed as distributed
    from chexfound.eval.classification.utils import AllClassifiers
    assert not multiview and not cat_cls, 'ML-Decoder: khong multiview, khong --cat-cls -> DUNG'
    assert list(n_last_blocks_list) == [1], f'ML-Decoder doc tang cuoi: --n-last-blocks phai la 1 (nhan {list(n_last_blocks_list)}) -> DUNG'
    d_in = int(sample_output[-1][0].shape[-1])
    heads, groups = nn.ModuleDict(), []
    for n in n_last_blocks_list:
        for avgpool in avgpools:
            for lr in learning_rates:
                head = MLDecoderOnTheirFeatures(num_classes, d_in)
                if torch.cuda.is_available(): head = head.cuda()        # như .cuda() của hàm gốc (bộ kiểm CPU: giữ CPU)
                heads[head_name(n, avgpool, lr)] = head
                groups.append({"params": head.parameters(), "lr": lr})
    classifiers = AllClassifiers(heads)
    if distributed.is_enabled():
        classifiers = nn.parallel.DistributedDataParallel(classifiers)
    return classifiers, groups
