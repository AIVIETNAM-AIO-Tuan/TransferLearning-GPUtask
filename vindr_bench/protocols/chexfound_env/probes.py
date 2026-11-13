# -*- coding: utf-8 -*-
"""Cổng số học của Ver1 (PLAN_89 §8 mục 1, 4) — điều phối từ notebook: một đầu vào chung (biến đổi đánh giá của họ, môi trường cũ) rồi SÁU tiến trình
đặc trưng của lớp bọc (= thiết kế C4 của 86j, đã chạy 13/11):

    cũ (Python của họ):  GPU fp16 (như lượt học) · GPU fp32 · CPU fp32
    mới (Python notebook): GPU fp32 · GPU fp16 · CPU fp32          (RAD-DINO: attention eager = thuật toán duy nhất của transformers 4.36.2)

  cổng fp16     = cũ fp16 vs cũ fp32, ngưỡng theo NỀN mới fp16 vs mới fp32       (checks.features.relative_gate: tương đối + chặn bền)
  cổng GPU–CPU  = cũ GPU vs cũ CPU (fp32), nền mới GPU vs mới CPU
  thông tin     = cũ fp32 vs mới fp32 (khác phiên bản thư viện; không phải cổng)
  nạp trọng số  = trong MỖI tiến trình: thông tin nạp của thư viện + từng tensor = tệp (checks.weights)."""
import json
import os
import subprocess
import sys
import time

from ...checks import features as F
from .environment import child_env

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, 'probe_features.py')
N_IMAGES = 4
TIMEOUT_S = 30 * 60
NEW_ENV_ATTN = {'raddino': 'eager'}           # môi trường mới: cùng thuật toán attention với môi trường cũ (transformers 4.36.2 chỉ có eager)
RUNS = (('cu_fp16', 'cu', 'fp16', 'cuda'), ('cu_fp32', 'cu', 'none', 'cuda'), ('cu_cpu', 'cu', 'none', 'cpu'),
        ('moi_fp32', 'moi', 'none', 'cuda'), ('moi_fp16', 'moi', 'fp16', 'cuda'), ('moi_cpu', 'moi', 'none', 'cpu'))
STATS = (('cu fp16 vs fp32', 'cu_fp32', 'cu_fp16'), ('nen: moi fp16 vs fp32', 'moi_fp32', 'moi_fp16'), ('cu GPU vs CPU fp32', 'cu_cpu', 'cu_fp32'),
         ('nen: moi GPU vs CPU fp32', 'moi_cpu', 'moi_fp32'), ('thong tin: cu vs moi fp32 (ref cu)', 'cu_fp32', 'moi_fp32'))


def _new_env_vars(package_dir):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([package_dir] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    return env


def _run(cmd, env_vars, timeout_s, tag):
    t0 = time.time()
    try:
        # utf-8 + replace: thanh tiến độ của transformers mới (ký tự khối) không làm mất đuôi lỗi
        r = subprocess.run(cmd, env=env_vars, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        return None, dict(ma_thoat=None, qua_gio=True, phut=(time.time() - t0) / 60, duoi_loi=str(e)[-500:])
    info = json.loads(r.stdout.split(tag)[-1].strip().splitlines()[0]) if (r.returncode == 0 and tag in r.stdout) else None
    return info, dict(ma_thoat=r.returncode, qua_gio=False, phut=(time.time() - t0) / 60, duoi_loi=(r.stderr or '')[-3000:])


def make_input(env, csv_root, out_npy, size, n=N_IMAGES):
    '''n ảnh train đầu qua biến đổi đánh giá CỦA HỌ (Python của họ) -> .npy dùng chung cho 6 tiến trình.'''
    info, run = _run([env.python, SCRIPT, '--make-input', '--repo', env.repo, '--csv-root', csv_root, '--size', str(size), '--n', str(n),
                      '--out', out_npy], child_env(env), TIMEOUT_S, 'DV=')
    assert info and info['hinh'] == [n, 3, size, size], f"tao dau vao dac trung THAT BAI: {run['duoi_loi'][-1500:]} -> DUNG"
    return info


def run_features(out_npz, input_npy, encoder, amp, device, python, env_vars, attn=None):
    '''-> dict(dat, info, ma_thoat, phut, duoi_loi, tep). dat = chạy xong, hữu hạn, nạp trọng số đạt, không thiếu khoá.'''
    cmd = [python, SCRIPT, '--encoder', encoder, '--input', input_npy, '--out', out_npz, '--amp', amp, '--device', device]
    if attn: cmd += ['--attn', attn]
    info, run = _run(cmd, env_vars, TIMEOUT_S, 'DT=')
    ok = bool(info and info['huu_han'] and info['thieu_khoa'] == 0 and (info.get('nap') or {}).get('dat') is True)
    return dict(dat=ok, info=info, tep=out_npz, **run)


def numeric_gate(env, encoder, csv_root, out_dir, size, log=print):
    '''Đầu vào chung + 6 tiến trình + hai cổng (fp16, GPU–CPU) + thống kê. Trượt bất kỳ cổng nào (kể cả nạp trọng số) -> AssertionError.'''
    from ...checks import weights
    os.makedirs(out_dir, exist_ok=True)
    inp = os.path.join(out_dir, 'dac_trung_dau_vao.npy')
    dv = make_input(env, csv_root, inp, size)
    attn_new = NEW_ENV_ATTN.get(encoder)
    new_vars = _new_env_vars(env.package_dir)
    dt = {}
    for tag, side, amp, dev in RUNS:
        old = side == 'cu'
        dt[tag] = run_features(os.path.join(out_dir, f'dt_{tag}.npz'), inp, encoder, amp, dev, env.python if old else sys.executable,
                               child_env(env) if old else new_vars, attn=None if old else attn_new)
        r = dt[tag]
        nap = (r['info'] or {}).get('nap')
        assert r['dat'], (f"dac trung {tag} ({encoder}) LOI / khong huu han / nap trong so KHONG dat: "
                          + (weights.loading_text(nap) if nap else r['duoi_loi'][-1500:]) + ' -> DUNG')
        log(f"{tag:9s} {weights.loading_text(nap)} | attention {r['info'].get('attention') or r['info'].get('attn') or '-'} | "
            f"{r['info']['thiet_bi']} {r['info']['amp']} {r['phut']:.1f} phut")
    cmp = lambda a, b: F.compare_features(dt[a]['tep'], dt[b]['tep'])
    gate_fp16 = F.relative_gate(cmp('cu_fp32', 'cu_fp16'), cmp('moi_fp32', 'moi_fp16'))
    gate_dev = F.relative_gate(cmp('cu_cpu', 'cu_fp32'), cmp('moi_cpu', 'moi_fp32'))
    cross = cmp('moi_fp32', 'cu_fp32')
    stats = {name: F.feature_stats(dt[a]['tep'], dt[b]['tep']) for name, a, b in STATS}
    for name, st in stats.items():
        log(F.stats_text(name, st))
    log(F.gate_text('CONG fp16 (cu fp16 vs fp32, theo nen moi)', gate_fp16))
    log(F.gate_text('CONG GPU-CPU (cu GPU vs CPU fp32, theo nen moi)', gate_dev))
    log(f"thong tin (khong cong) cu vs moi fp32: lech {cross['lech_tuong_doi']:.2e}, cos_min {cross['cos_min']:.6f}, cos trung vi "
        f"{cross['cos_trung_vi']:.6f} | dac trung {dt['cu_fp16']['info']['hinh']} | anh {size}")
    out = dict(dau_vao=dv, dac_trung=dt, nap={k: v['info']['nap'] for k, v in dt.items()},
               cong_fp16=gate_fp16, cong_gpu_cpu=gate_dev, cheo_moi_truong=cross, thong_ke=stats, attention_moi=attn_new)
    assert gate_fp16['dat'], f'lop boc {encoder} o moi truong cu: fp16 vs fp32 TRUOT -> DUNG, gui Claude output C4'
    assert gate_dev['dat'], f'lop boc {encoder} o moi truong cu: GPU vs CPU TRUOT -> DUNG, gui Claude output C4'
    return out
