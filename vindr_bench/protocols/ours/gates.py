# -*- coding: utf-8 -*-
"""Cổng của notebook Ver2 (PLAN_89 §8), chạy trong tiến trình của lượt:

    library_gate   bộ thư viện = bộ ĐÃ CHỐT (§8 mục 7): torch / CUDA / transformers / timm khác -> dừng
    gpu_gate       GPU của Ver2 = G4 (RTX PRO 6000) — "cùng GPU" (§1 Runtime)
    data_gate      vân tay dữ liệu = vân tay mọi lượt Ver1 đã chạy (§8 mục 6)
    numeric_check  đặc trưng bf16 (autocast của lượt) so với fp32 trên cùng GPU, cùng ảnh
    memory_ok      đỉnh bộ nhớ GPU của lượt chạy thử <= 70 % (§8 mục 9)

numeric_check KHÔNG dùng ngưỡng của Ver1 (hiệu chỉnh cho fp16): bf16 ít hơn fp16 3 bit phần định trị -> 1 - cos lớn hơn ~64 lần, và nền bf16 trên GPU
chưa đo được trước lượt đầu. Vì vậy chỉ DỪNG khi hỏng nặng (không hữu hạn / trung vị cos < 0,99 — sai kernel, tràn số làm tụt phần lớn token), còn
lại GHI số (trung vị, phân vị 1e-2, nhỏ nhất, lệch lớn nhất) + cờ cảnh báo; số đo ở các lượt đầu là nền cho bảng gộp.

Hiệu chỉnh ngưỡng (15/11, CPU autocast bf16 so với fp32, trọng số thật, 5 ảnh X-quang mẫu công khai của repo CheXFound — KHÔNG phải nền GPU): token patch
4 tầng — trung vị cos 0,99988 (CheXzero) … 0,99998; phân vị 1e-2 0,9962 (CheXFound), 0,9968 (RAD-DINO), 0,9970 (CheXWorld), >= 0,9992 các hàng còn lại;
nhỏ nhất tới 0,94 (RAD-DINO: đuôi nặng -> không dùng min); vector toàn ảnh: trung vị >= 0,99994. Ngưỡng dừng (trung vị 0,99) cách nền ~80 lần theo
1 - cos; ngưỡng cảnh báo (trung vị 0,999; phân vị 1e-2 0,98) cách nền >= 5 lần."""
import importlib
import sys

import numpy as np
import torch

# Bộ thư viện đã chốt = image Colab ngày 14/11/2026 (output C2 / C4 của các notebook Ver1: torch 2.14.1+cu130, transformers 5.22.1, timm 1.0.31;
# `86b` trên G4 ngày 11/11 cũng transformers 5.22.1 -> G4 và A100 cùng image). Khác -> dừng, báo Claude (không tự cài lại).
VER2_LIBS = {'torch': '2.14.1+cu130', 'transformers': '5.22.1', 'timm': '1.0.31'}
RECORDED = ('torchvision', 'numpy', 'PIL', 'huggingface_hub', 'safetensors', 'torchmetrics', 'pandas')
GPU_MARK = 'RTX PRO 6000'                 # torch.cuda.get_device_name trên G4: 'NVIDIA RTX PRO 6000 Blackwell Server Edition' (86b)
MEM_LIMIT = 0.70
COS_STOP = 0.99                           # trung vị cos bf16 vs fp32 dưới mức này = hỏng nặng -> dừng
COS_WARN = dict(trung_vi=0.999, q1e2=0.98)
# md5 (8 ký tự đầu) khoá / nhãn từng tách — in ở C3 của MỌI lượt Ver1 đã chạy (RAD-DINO, BiomedCLIP, CheXFound, CheXWorld, CheXzero, EVA-X; kho kcb)
FINGERPRINT = {'train': ('4df91940', '385112c0'), 'val': ('f66f16fb', 'e752a586'), 'test': ('1e8c5cf6', '9762e804')}


def versions():
    v = dict(python=sys.version.split()[0], cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
    for m in tuple(VER2_LIBS) + RECORDED:
        try:
            v[m] = getattr(importlib.import_module(m), '__version__', '?')
        except Exception as e:                                   # thiếu thư viện: ghi tên lỗi, cổng bên dưới quyết
            v[m] = f'THIEU ({type(e).__name__})'
    return v


def library_gate(v=None, pinned=None):
    '''-> (đạt, {thư viện: [có, ghim]} các chỗ khác).'''
    v, pinned = v or versions(), pinned or VER2_LIBS
    diff = {m: [v.get(m), want] for m, want in pinned.items() if v.get(m) != want}
    return not diff, diff


def gpu_info():
    if not torch.cuda.is_available():
        return dict(ten=None, tong_gib=None)
    return dict(ten=torch.cuda.get_device_name(0), tong_gib=torch.cuda.get_device_properties(0).total_memory / 2 ** 30)


def gpu_gate(info=None, mark=GPU_MARK):
    info = info or gpu_info()
    return bool(info['ten'] and mark in info['ten'])


def data_gate(fingerprint, pinned=None):
    '''Vân tay của kho đang nạp so với vân tay các lượt Ver1. -> (trùng, {tách: [có, ghim]} các chỗ khác).'''
    pinned = pinned or FINGERPRINT
    diff = {s: [[fingerprint[s]['md5_khoa'][:8], fingerprint[s]['md5_nhan'][:8]], list(want)] for s, want in pinned.items()
            if (fingerprint[s]['md5_khoa'][:8], fingerprint[s]['md5_nhan'][:8]) != tuple(want)}
    return not diff, diff


def _cos_stats(a, b):
    '''a, b [..., D] -> thống kê cos theo từng vector + lệch tuyệt đối lớn nhất (tính ở float64 trên CPU).'''
    a, b = a.detach().double().cpu().reshape(-1, a.shape[-1]), b.detach().double().cpu().reshape(-1, b.shape[-1])
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).numpy()
    fin = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return dict(huu_han=fin, so_vector=int(len(cos)), lech_max=float((a - b).abs().max()), cos_min=float(np.min(cos)),
                cos_q1e2=float(np.quantile(cos, 1e-2)), cos_trung_vi=float(np.median(cos)))


@torch.no_grad()
def numeric_check(backbone, x):
    '''Đặc trưng của lượt (autocast bf16) so với fp32 trên CÙNG thiết bị, cùng lô ảnh `x` (đã chuẩn hoá ImageNet). CPU: hai đường là một -> lệch 0.
       -> dict(patch, vector: thống kê gộp 4 tầng; tang: cos trung vị từng tầng; dat; canh_bao).'''
    x = x.to(backbone.device)
    b_amp, _ = backbone.features(x)
    b_ref, _ = backbone.features(x, autocast=False)
    assert len(b_amp) == len(b_ref) == 4, 'dac trung KHONG du 4 tang -> DUNG'
    n, d = backbone.enc.spec.n_patches, backbone.dim
    for p, v in b_amp:
        assert tuple(p.shape) == (x.shape[0], n, d) and tuple(v.shape) == (x.shape[0], d), f'hinh dac trung {tuple(p.shape)} / {tuple(v.shape)} SAI -> DUNG'
    patch = _cos_stats(torch.stack([p for p, _ in b_amp]), torch.stack([p for p, _ in b_ref]))
    vector = _cos_stats(torch.stack([v for _, v in b_amp]), torch.stack([v for _, v in b_ref]))
    layers = [_cos_stats(pa, pr)['cos_trung_vi'] for (pa, _), (pr, _) in zip(b_amp, b_ref)]
    khac_tang = float((b_ref[0][0].float() - b_ref[-1][0].float()).abs().max())
    ok = bool(patch['huu_han'] and vector['huu_han'] and patch['cos_trung_vi'] >= COS_STOP and vector['cos_trung_vi'] >= COS_STOP and khac_tang > 0)
    warn = [f'{name}: {k} {s[key]:.6f} < {lim}' for name, s in (('patch', patch), ('vector', vector))
            for k, key, lim in (('trung vi', 'cos_trung_vi', COS_WARN['trung_vi']), ('phan vi 1e-2', 'cos_q1e2', COS_WARN['q1e2'])) if s[key] < lim]
    return dict(patch=patch, vector=vector, cos_trung_vi_tung_tang=layers, khac_tang_dau_cuoi=khac_tang, dtype_luot=str(b_amp[-1][0].dtype),
                dtype_doi_chieu=str(b_ref[-1][0].dtype), so_anh=int(x.shape[0]), dat=ok, canh_bao=warn, nguong_dung=COS_STOP)


def numeric_text(g):
    f = lambda s: (f"lech max {s['lech_max']:.2e} | cos trung vi {s['cos_trung_vi']:.6f}, phan vi 1e-2 {s['cos_q1e2']:.6f}, nho nhat {s['cos_min']:.6f} "
                   f"({s['so_vector']:,} vector)")
    return (f"so hoc {g['dtype_luot'].replace('torch.', '')} vs {g['dtype_doi_chieu'].replace('torch.', '')} tren {g['so_anh']} anh val, 4 tang: patch "
            f"{f(g['patch'])} | vector toan anh {f(g['vector'])} | chi dung khi trung vi cos < {g['nguong_dung']}: {'DAT' if g['dat'] else 'TRUOT'}"
            + (f" | CANH BAO (ghi lai, khong dung): {'; '.join(g['canh_bao'])}" if g['canh_bao'] else ''))


def memory_ok(peak_bytes, total_bytes, limit=MEM_LIMIT):
    return bool(total_bytes and peak_bytes / total_bytes <= limit)
