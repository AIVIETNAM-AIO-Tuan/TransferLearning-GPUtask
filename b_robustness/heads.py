"""Học lại đầu P2 trên VinDr train, backbone ĐÔNG CỨNG (= ô H của 85f, mọi hàng 85f dùng nguyên văn; test_87 kiểm trùng từng bit trên CPU).

Đầu: 'ml' (ML-Decoder, 28 query cố định, + DocVung khi có token vùng = mldec_hr), 'ml_patch' (cùng đầu, KHÔNG token vùng; chỉ khi có vùng),
'lin' (linear trên [CLS ; TB patch], chuẩn hoá z theo train). Mỗi seed chọn lr + epoch theo mAP VAL 21 lớp — KHÔNG nhìn test.
Lưới khoá (Tuấn 09/11): ML-Decoder {1e-4 … 3e-2}, 15 epoch, lô 64; linear {1e-4 … 1e-1}, 100 epoch, lô 1024; AdamW wd 1e-4; BCE.
Phân kỳ (dự đoán val NaN / inf): dừng lần học, giữ epoch tốt nhất trước đó; phân kỳ ngay epoch đầu -> không bao giờ được chọn."""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .metrics import ap_cols


def train_p2_heads(tok, vung, glob, labels, split, label_names, main_classes, ops, row, seeds=config.HEAD_SEEDS,
                   ml=(config.ML_GRID, config.ML_EPOCHS, config.ML_BATCH, config.ML_WD), lin=(config.LIN_GRID, config.LIN_EPOCHS, config.LIN_BATCH, config.LIN_WD),
                   ap_fn=ap_cols, log=print):
    '''tok [N, T, d] fp16, vung [N, R, d] fp16 hoặc None, glob [N, dg] float32, labels [N, 28], split [N] (train / val / test).
       ops = dict(MLDecoder, DocVung, mldec_h, mldec_h_hr, q_cua) — lớp / hàm đầu đọc của 78z (Part3View.head_ops()).
       -> dict(heads={đầu: dict(val=[5, n_val, 28], test=[5, n_test, 28], chon=[...])}, config=cấu hình học).'''
    LUOI_ML, E_ML, LO_ML, WD_ML = ml
    LUOI_LIN, E_LIN, LO_LIN, WD_LIN = lin
    MLDecoder, DocVung, mldec_h, mldec_h_hr, q_cua = (ops[k] for k in ('MLDecoder', 'DocVung', 'mldec_h', 'mldec_h_hr', 'q_cua'))
    cfg = dict(seed=list(seeds), loss='BCE', chon='mAP val tren 21 lop so chinh P2; hoa -> phan tu dau cua luoi',
               phan_ky='du doan val NaN / inf -> dung lan hoc, giu epoch tot nhat truoc do; phan ky ngay epoch dau -> khong duoc chon',
               ml=dict(lr=list(LUOI_ML), epoch=E_ML, lo=LO_ML, wd=WD_ML, d=512, nhead=8, ff=1024, drop=0.1, lich='warmup 1 epoch + cosine', autocast='bf16'),
               lin=dict(lr=list(LUOI_LIN), epoch=E_LIN, lo=LO_LIN, wd=WD_LIN, chuan_hoa='z theo train', lich='hang so'))
    assert tok.dtype == torch.float16 and tok.shape[0] == len(labels) == len(split) == glob.shape[0] and (vung is None or vung.shape[0] == len(labels)), (
        'dau vao P2_HOC sai hinh -> DUNG')
    itr, iva, ite = (torch.as_tensor(np.flatnonzero(split == t), device=tok.device) for t in ('train', 'val', 'test'))
    jc = [label_names.index(t) for t in main_classes]
    Yt = torch.as_tensor(labels, dtype=torch.float32, device=tok.device)
    Yv = labels[iva.cpu().numpy()][:, jc]
    n_cls = len(label_names)

    def map_val(P):
        '''mAP val trên 21 lớp số chính (lớp không có dương -> NaN, bỏ; 85e đã kiểm mọi lớp số chính >= 1 ca val).'''
        return float(np.nanmean(ap_fn(Yv, P[:, jc])))

    def schedule(opt, n_steps, n_warm):
        '''warmup tuyến tính n_warm bước rồi cosine về 0.'''
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda b: (b + 1) / n_warm if b < n_warm
                                                 else 0.5 * (1 + math.cos(math.pi * (b - n_warm) / max(1, n_steps - n_warm))))

    @torch.no_grad()
    def predict(f, idx, lo=256):
        return np.concatenate([torch.sigmoid(f(idx[b:b + lo]).float()).cpu().numpy() for b in range(0, len(idx), lo)])

    def fit(make, f, lr, wd, E, lo, seed, sched=True):
        '''Một lần học: khởi tạo theo seed -> E epoch AdamW + BCE trên train (thứ tự lô theo seed) -> giữ trạng thái có mAP val cao nhất.'''
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        m = make()
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        n_lo = math.ceil(len(itr) / lo)
        sch = schedule(opt, E * n_lo, n_lo) if sched else None
        best, diverged = (-1.0, None, -1), False
        for ep in range(E):
            m.train()
            perm = itr[torch.randperm(len(itr), generator=g).to(itr.device)]
            for b in range(0, len(perm), lo):
                ii = perm[b:b + lo]
                loss = F.binary_cross_entropy_with_logits(f(m, ii).float(), Yt[ii])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                if sch is not None: sch.step()
            m.eval()
            pv = predict(lambda ii: f(m, ii), iva)
            if not np.isfinite(pv).all():
                diverged = True; break
            mv = map_val(pv)
            if mv > best[0]: best = (mv, {k: v.detach().clone() for k, v in m.state_dict().items()}, ep + 1)
        if best[1] is None:
            return dict(map_val=float('-inf'), ep=0, pv=None, pt=None, phan_ky=True)
        m.load_state_dict(best[1]); m.eval()
        return dict(map_val=best[0], ep=best[2], pv=predict(lambda ii: f(m, ii), iva), pt=predict(lambda ii: f(m, ii), ite), phan_ky=diverged)

    def f_ml(v):
        def f(m, ii):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=tok.is_cuda):
                if v is None: return mldec_h(m, tok[ii].float())[0]                   # fp16 -> fp32 CHÍNH XÁC; autocast ép bf16
                return mldec_h_hr(m, q_cua(m), tok[ii].float(), v[ii].float())[0]
        return f

    def make_ml(v):
        def make():
            m = MLDecoder(n_cls, tok.shape[2], d=512, nhead=8, ff=1024, drop=0.1)     # = đầu MIMIC (c3 của 78t), 28 query cố định
            if v is not None: m.docv = DocVung(v.shape[1], tok.shape[2], 512)        # = mldec_hr (DocVung không rút số ngẫu nhiên)
            return m.to(tok.device)
        return make

    mu = glob[itr].mean(0, keepdim=True)
    sd = glob[itr].std(0, keepdim=True).clamp_min(1e-6)
    Gz = ((glob - mu) / sd).float()
    out = {}

    def run(name, fit_one, grid, E):
        kq, t0 = [], time.time()
        for s in seeds:
            runs = {h: fit_one(h, s) for h in grid}
            h_best = max(grid, key=lambda h: runs[h]['map_val'])                       # hoà -> phần tử ĐẦU của lưới
            assert np.isfinite(runs[h_best]['map_val']), f'{row} | {name}: MOI lr deu phan ky o seed {s} -> DUNG, bao Claude'
            kq.append(dict(seed=s, sieu=h_best, **runs[h_best], pk_lr=[h for h in grid if runs[h]['phan_ky']],
                           luoi={str(h): (round(runs[h]['map_val'], 6) if np.isfinite(runs[h]['map_val']) else None) for h in grid}))
        out[name] = dict(val=np.stack([k['pv'] for k in kq]).astype('float32'), test=np.stack([k['pt'] for k in kq]).astype('float32'),
                         chon=[dict(seed=k['seed'], sieu=k['sieu'], ep=k['ep'], map_val=round(k['map_val'], 6), luoi=k['luoi'], phan_ky=k['pk_lr']) for k in kq])
        mv = [k['map_val'] for k in kq]
        log(f"{row} | {name}: mAP val {len(main_classes)} lop {np.mean(mv):.4f} +- {np.std(mv):.4f} ({len(seeds)} seed) | chon "
            + ', '.join(f"{k['sieu']:g}/ep{k['ep']}" for k in kq) + f' | {(time.time() - t0) / 60:.1f} phut')
        # chẩn đoán (CHỈ in, KHÔNG đổi lựa chọn): mAP val TB theo lr; lr chọn ở MÉP lưới / epoch chọn gần cuối
        tb = {h: [k['luoi'][str(h)] for k in kq if k['luoi'][str(h)] is not None] for h in grid}
        n_pk = {h: sum(h in k['pk_lr'] for k in kq) for h in grid}
        n_max, n_min, n_ep = (sum(k['sieu'] == max(grid) for k in kq), sum(k['sieu'] == min(grid) for k in kq), sum(k['ep'] >= 0.9 * E for k in kq))
        log(f"   {name}: mAP val TB {len(kq)} seed theo lr: " + ', '.join(f'{h:g} ' + (f'{np.mean(v):.4f}' if v else 'phan ky') for h, v in tb.items())
            + (' | phan ky (so seed): ' + ', '.join(f'{h:g} x{n}' for h, n in n_pk.items() if n) if any(n_pk.values()) else '')
            + (f' | ⚠ {n_max} / {len(kq)} seed chon lr LON NHAT cua luoi' if n_max else '') + (f' | ⚠ {n_min} / {len(kq)} seed chon lr NHO NHAT cua luoi' if n_min else '')
            + (f' | ⚠ {n_ep} / {len(kq)} seed chon epoch >= 90 % cua {E}' if n_ep else ''))

    run('ml', lambda lr, s: fit(make_ml(vung), f_ml(vung), lr, WD_ML, E_ML, LO_ML, s), LUOI_ML, E_ML)
    if vung is not None:
        run('ml_patch', lambda lr, s: fit(make_ml(None), f_ml(None), lr, WD_ML, E_ML, LO_ML, s), LUOI_ML, E_ML)
    run('lin', lambda lr, s: fit(lambda: nn.Linear(Gz.shape[1], n_cls).to(Gz.device), lambda m, ii: m(Gz[ii]), lr, WD_LIN, E_LIN, LO_LIN, s, sched=False),
        LUOI_LIN, E_LIN)
    assert all(np.isfinite(v['val']).all() and np.isfinite(v['test']).all() for v in out.values()), 'du doan P2 co NaN -> DUNG'
    log(f'P2 {row}: xong {list(out)} — CHUA cham test (87d)')
    return dict(heads=out, config=cfg)
