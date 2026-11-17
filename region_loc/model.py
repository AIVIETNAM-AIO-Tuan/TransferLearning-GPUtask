# -*- coding: utf-8 -*-
"""Mô hình cuối (ArrE s42) cho phép thử định vị — KHÔNG có mã mô hình mới.

  encoder + luồng vùng : lớp bọc `vindr_bench.encoders.b.B` (đã kiểm từng bit với đường 86b_B); forward đi qua đúng `B._blocks`, chỉ BẮT lại trạng thái
                         của luồng (`luong.bat_dau` trả về `st`) để đọc hộp tầng cuối và token vùng cuối — không chép lại vòng lặp
  đầu đọc trong lượt   : `vindr_bench.heads.mldecoder.MLDecoder` nạp chặt khoá 'head' của checkpoint; K / V = [pin(patch) ; DocVung(pin, r)] như
                         `dau_h_v2` (khoi_jca_v2) -> attention của 40 query lớp trên 1369 patch + 36 token vùng
  cờ bất thường        : Linear(768, 1) nạp 'phu_v' (bt.weight, bt.bias) — `PhuVung.bt_logit(r) = bt(r.float())[..., 0]` (khoi_luong_vung, 79)
  nhãn theo vùng       : `DauVung.z(r) = lin(ln(r))` nạp chặt khoá 'dauv' -> logit [B, 36, 40]

`DocVung`, `DauVung` chép NGUYÊN VĂN từ experiments/78_chi_anh/khoi/khoi_jca_v2.py (tests/test_region_loc.py so cây cú pháp; lớp ML-Decoder của gói
vindr_bench được so với lớp trong notebook huấn luyện). Mọi token vùng đưa vào ba đầu là `ra['r']` (token vùng SAU block cuối), như lúc học."""
import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import geometry as G


# ==== NGUYÊN VĂN khoi_jca_v2.py (78) ====
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


# ==== của gói ====
class Loc(nn.Module):
    '''forward(x [B, 3, 518, 518] chuẩn hoá gốc RAD-DINO) -> dict tensor CPU float32:
         box [B, 36, 4] (x1, y1, x2, y2) chuẩn hoá — hộp tầng cuối của luồng vùng
         bt  [B, 36]      logit cờ bất thường từng vùng
         z   [B, 36, 40]  logit nhãn theo vùng
         attn [B, 40, 1369 + 36] attention (trung bình các head) của từng query lớp
         logit0 [B, 40]   đầu ra GroupFC của đầu đọc (CHƯA qua kênh nguồn / MIL / vùng cố định của lượt học — chỉ để soi, không phải dự đoán của mô hình)'''
    def __init__(self, enc, head, docv, dauv, bt, info):
        super().__init__()
        self.enc, self.head, self.docv, self.dauv, self.bt, self.info = enc, head, docv, dauv, bt, info

    @torch.no_grad()
    def forward(self, x):
        luong, cap = self.enc.luong, {}
        goc = luong.bat_dau

        def bat_dau(n):
            st = goc(n)
            cap['st'] = st
            return st
        luong.bat_dau = bat_dau                                      # thuộc tính của ĐỐI TƯỢNG che phương thức trong một lần forward
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.is_cuda):
                h = self.enc._blocks(self.enc.to_native(x), 1)[0]
        finally:
            del luong.bat_dau
        assert 'st' in cap and getattr(luong, 'bat_dau').__func__ is type(luong).bat_dau, 'khong bat duoc trang thai luong vung / chua tra lai phuong thuc -> DUNG'
        ra = luong.dau_ra(cap['st'])
        hop = ra['hop_cac_tang']
        assert hop is not None and hop.shape[0] == len(luong.tang) and tuple(hop.shape[1:]) == (x.shape[0], C.N_REGIONS, 4), (
            f'hop_cac_tang {None if hop is None else tuple(hop.shape)} khac ({len(luong.tang)}, B, {C.N_REGIONS}, 4) -> DUNG')
        patch, r = h[:, 1:].float(), ra['r'].float()
        assert patch.shape[1] == C.GRID * C.GRID, tuple(patch.shape)
        m = self.head
        q = m.q.weight.unsqueeze(0).expand(patch.shape[0], -1, -1)
        kv = m.pin(patch)
        kvc = torch.cat([kv, self.docv(m.pin, r).to(kv.dtype)], 1)
        hq, _ = m.img(q, kvc, False)                                 # dung duong cua dau_h_v2 (need_weights False) -> logit0 trung luc hoc tung bit
        w = m.img.at(q, kvc, kvc, need_weights=True, average_attn_weights=True)[1]   # attention tinh rieng (need_weights doi duong tinh cua MHA)
        b = hop[-1].float()
        box = torch.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2, b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], -1).clamp(0, 1)
        return dict(box=box.cpu(), bt=self.bt(r)[..., 0].float().cpu(), z=self.dauv.z(r).float().cpu(), attn=w.float().cpu(),
                    logit0=m.out(hq).float().cpu())


def build(path=None, cfg_path=None, attn=None, khuc=None):
    '''-> Loc ở eval, requires_grad False. path / cfg_path None: tệp đã lấy về bởi vindr_bench.encoders.files.fetch (VB_WEIGHTS_DIR).'''
    from vindr_bench.checks import safe_load
    from vindr_bench.encoders import b as BW
    from vindr_bench.encoders import files
    from vindr_bench.encoders.spec import SPECS
    from vindr_bench.heads.mldecoder import MLDecoder
    spec = SPECS[C.ENCODER]
    assert (spec.image_size, spec.patch_size, spec.embed_dim) == (C.SIZE, C.PATCH, 768), spec
    enc = BW.B(input_mean=spec.mean, input_std=spec.std, path=path, cfg_path=cfg_path, attn=attn, khuc=khuc).eval()
    assert float((enc.scale - 1).abs().max()) == 0.0 and float(enc.shift.abs().max()) == 0.0, 'lop boc phai nhan anh o chuan hoa goc -> DUNG'
    assert enc.neo == 1.0 and enc.luong.R == C.N_REGIONS, (enc.neo, enc.luong.R)
    path = path or files.weights_path(spec)
    ck, src = safe_load.load_checkpoint(path, expect_md5=spec.file_md5, expect_size=spec.file_size)
    assert ck.get('run') == BW.RUN and ck.get('ep') == BW.EPOCH - 1, (ck.get('run'), ck.get('ep'))
    d = spec.embed_dim
    head = MLDecoder(C.N_LABELS, d)
    head.load_state_dict(ck['head'], strict=True)
    docv = DocVung(C.N_REGIONS, d, head.pin.out_features)
    docv.load_state_dict(ck['docv'], strict=True)
    mil = [int(i) for i in ck['dauv']['mil_idx'].tolist()]
    dauv = DauVung(d, C.N_LABELS, mil)
    dauv.load_state_dict(ck['dauv'], strict=True)
    bt = nn.Linear(d, 1)
    bt.load_state_dict({'weight': ck['phu_v']['bt.weight'], 'bias': ck['phu_v']['bt.bias']}, strict=True)
    thieu = [f for f in C.FINDINGS if C.LAB40.index(f) not in mil]
    assert not thieu, f'dau hieu MS-CXR khong nam trong kenh nhan-vung (mil_idx) cua checkpoint: {thieu} -> DUNG'
    log = ck.get('log') or {}
    ep = log.get(str(ck['ep'])) or log.get(ck['ep']) or {}
    giu = (ep.get('giu') if isinstance(ep, dict) else None) or {}      # nhật ký epoch cuối, mục 'giu' = nhóm bệnh nhân train GIỮ RIÊNG (hộp silver)
    info = dict(run=ck['run'], ep=int(ck['ep']) + 1, nguon_tep=src, mil_idx=mil, n_head=len(ck['head']), n_docv=len(ck['docv']), n_dauv=len(ck['dauv']),
                nhat_ky={k: (float(giu[k]) if giu.get(k) is not None else None) for k in ('luong_iou', 'luong_iou_tb', 'luong_auc_bt')},
                nap_encoder=dict(run=BW.RUN, tep=spec.drive_file))
    del ck
    loc = Loc(enc, head, docv, dauv, bt, info).eval()
    loc.requires_grad_(False)
    return loc


def order_gate(hop_tb, gold, n_regions=C.N_REGIONS):
    '''Cổng thứ tự vùng: hop_tb [R, 4] (cx, cy, w, h) của checkpoint so với hộp gold TRUNG BÌNH theo tên vùng (gold đã qua tables.gold_frame, cột k,
       g1..g4; chỉ hàng train). Mỗi vùng có gold phải khớp nhất với CHÍNH NÓ và IoU >= MIN_ORDER_IOU. -> dict(iou_tb, iou_min, n_vung).'''
    hx = G.cxcywh_to_xyxy(np.asarray(hop_tb, 'float64'))
    assert hx.shape == (n_regions, 4), hx.shape
    tb = gold.groupby('k')[['g1', 'g2', 'g3', 'g4']].mean()
    ks = tb.index.to_numpy().astype(int)
    M = np.stack([G.iou(hx, np.repeat(tb.loc[[k]].to_numpy(), n_regions, 0)) for k in ks], 1)       # [R vùng của luồng, vùng gold]
    dung = np.array([M[k, j] for j, k in enumerate(ks)])
    nham = [int(k) for j, k in enumerate(ks) if int(np.argmax(M[:, j])) != int(k)]
    assert not nham and float(dung.min()) >= C.MIN_ORDER_IOU, (
        f'thu tu vung: {len(nham)} vung khop nhat voi vung KHAC ({nham}); IoU nho nhat {dung.min():.3f} (can >= {C.MIN_ORDER_IOU}) -> danh sach vung / '
        f'hinh hoc sai -> DUNG')
    return dict(iou_tb=float(dung.mean()), iou_min=float(dung.min()), n_vung=int(len(ks)))
