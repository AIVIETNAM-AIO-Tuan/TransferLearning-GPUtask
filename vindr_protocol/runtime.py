"""Môi trường chạy: thiết bị, kiểu autocast của backbone, TF32 cho đầu đọc, RAM, số worker."""
import os

import torch


def on_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def device_and_amp(require_gpu_on_colab=True):
    '''-> (thiết bị, dtype autocast của backbone). bf16 khi GPU hỗ trợ, như 85 (mã CheXFound dùng fp16 — PLAN_86 F13, ghi ở bảng sai khác).'''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if require_gpu_on_colab and device != 'cuda' and on_colab():
        raise RuntimeError('cần runtime GPU (G4) -> DỪNG')
    amp = torch.bfloat16 if (device == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16
    return device, amp


def device_name(device):
    if device == 'cuda':
        return f'{torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB'
    return 'cpu'


def allow_tf32_heads():
    '''Đầu đọc học ở fp32 như mã gốc; CHO PHÉP TF32 matmul (PLAN_86 §4.4).'''
    torch.set_float32_matmul_precision('high')


def ram_gb():
    '''(GB còn trống, GB tổng) của máy; không có psutil -> (None, None).'''
    try:
        import psutil
        vm = psutil.virtual_memory()
        return round(vm.available / 1e9, 1), round(vm.total / 1e9, 1)
    except Exception:
        return None, None


def default_num_workers(cap=6):
    '''6 worker x prefetch 2 x lô 256 ảnh 3x518x518 fp32 ~ 10 GB RAM đang chờ; đầu đọc mỗi bước vẫn chậm hơn nạp dữ liệu.'''
    return min(cap, max(1, (os.cpu_count() or 2) - 2))
