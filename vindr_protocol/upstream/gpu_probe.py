# -*- coding: utf-8 -*-
"""86h (PLAN_86 §15): môi trường thư viện của CheXFound (Python 3.9 + torch 2.0.0+cu117 + xFormers 0.0.18) có chạy ĐÚNG trên G4 (Blackwell, sm_120)
không — kết luận bằng phép tính thật, không bằng thông báo hỗ trợ (Event 2026-11-12 "Đính chính: … không chạy được trên G4 …").

- run_ops: probe_ops.py trong tiến trình riêng bằng Python của MỘT môi trường (cũ / mới — đối chứng sạch trên cùng GPU), ghi từng dòng JSON.
- list_ptx: PTX nhúng trong thư viện GPU của môi trường (cuobjdump --list-ptx; không có công cụ -> ghi "không có").
- make_feature_input / run_features / compare_features: đặc trưng 4 tầng cuối CheXFound bằng hàm của họ (probe_features.py), cùng đầu vào.
- verdict: luật kết luận ĐẶT TRƯỚC (§15.2).
- 86j (§16.4, 13/11 lần 2): relative_gate (ngưỡng theo NỀN) -> encoder_gate (fp16 vs fp32) / device_gate (GPU vs CPU); feature_stats / stats_text /
  gate_text (phân vị, theo tầng; chỉ đọc / lưu).
Bộ đệm JIT của driver: CUDA_CACHE_PATH = JIT_CACHE_DIR, CUDA_CACHE_MAXSIZE = 4 GiB (mọi tiến trình con dùng chung trong phiên)."""
import glob, json, os, re, shutil, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPS_SCRIPT = os.path.join(HERE, 'probe_ops.py')
FEAT_SCRIPT = os.path.join(HERE, 'probe_features.py')
JIT_CACHE_DIR = '/content/cuda_jit_cache'
JIT_CACHE_MAX = 4 * 2 ** 30
OPS_TIMEOUT_S = 60 * 60          # quá giờ -> QUA_GIO (không kết luận, giữ phần đã đo)
FEAT_TIMEOUT_S = 30 * 60
FEATURE_IMAGES = 4
FEATURE_RULE = dict(he_so=2.0, cong=1e-3, cos_min=0.999)        # §15.2 (4)
G4_CAPABILITY = [12, 0]
PTX_LIB_KEYS = ('torch_cuda', 'cublas', 'cudnn', 'nccl', 'xformers', '_C')


def child_env(isolated=True, extra=None):
    '''Biến môi trường cho tiến trình con. isolated (môi trường của họ): bỏ PYTHONPATH / PYTHONHOME của notebook, tắt site người dùng, bỏ
       expandable_segments (torch 2.0 chưa có). Mọi tiến trình: bộ đệm JIT chung.'''
    env = dict(os.environ)
    if isolated:
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        env['PYTHONNOUSERSITE'] = '1'
        if 'expandable_segments' in env.get('PYTORCH_CUDA_ALLOC_CONF', ''): env.pop('PYTORCH_CUDA_ALLOC_CONF')
    os.makedirs(JIT_CACHE_DIR, exist_ok=True)
    env.update(CUDA_CACHE_PATH=JIT_CACHE_DIR, CUDA_CACHE_MAXSIZE=str(JIT_CACHE_MAX))
    env.update(extra or {})
    return env


def _read_jsonl(path):
    recs = []
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    break                                   # dòng cuối ghi dở (tiến trình bị giết) -> bỏ
    return recs


def run_ops(out_path, python=None, isolated=True, device='cuda', fail=None, timeout_s=OPS_TIMEOUT_S, env_extra=None):
    '''probe_ops.py bằng `python` (None -> Python của notebook, không cô lập). -> dict(thong_tin, phep_tinh, xong, qua_gio, ma_thoat, phut, duoi_loi).'''
    py = python or sys.executable
    cmd = [py, OPS_SCRIPT, out_path, '--device', device] + (['--fail', fail] if fail else [])
    t0, qua_gio, rc, err = time.time(), False, None, ''
    try:
        r = subprocess.run(cmd, env=child_env(isolated and python is not None, env_extra), capture_output=True, text=True, timeout=timeout_s)
        rc, err = r.returncode, (r.stderr or '')[-3000:]
    except subprocess.TimeoutExpired as e:
        qua_gio, err = True, str(e)[-500:]
    recs = _read_jsonl(out_path)
    info = next((r for r in recs if r.get('loai') == 'thong_tin'), None)
    ops = [r for r in recs if r.get('loai') == 'phep_tinh']
    done = next((r for r in recs if r.get('loai') == 'xong'), None)
    return dict(thong_tin=info, phep_tinh=ops, xong=done is not None, ly_do_xong=(done or {}).get('ly_do'), qua_gio=qua_gio, ma_thoat=rc,
                phut=(time.time() - t0) / 60, duoi_loi=err, python=py)


_LIBS_CODE = '''import glob, importlib.util, json, os, site
import torch
libs = sorted(glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "*.so*")))
for sp in site.getsitepackages():
    libs += sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib", "*.so*")))
s = importlib.util.find_spec("xformers")
if s and s.origin:
    libs += sorted(glob.glob(os.path.join(os.path.dirname(s.origin), "*.so")))
print("LIBS=" + json.dumps(libs))
'''


def find_cuobjdump():
    return shutil.which('cuobjdump') or next((p for p in ('/usr/local/cuda/bin/cuobjdump',) if os.path.exists(p)), None)


def list_ptx(python=None, isolated=True, timeout_s=300):
    '''PTX nhúng trong thư viện GPU của môi trường (libtorch_cuda, cuBLAS, cuDNN, NCCL, xFormers): số tệp PTX + kiến trúc trong tên. Chỉ ghi thông tin
       (không vào luật). -> dict(co_cuobjdump, thu_vien={tên: dict(so_ptx, kien_truc_ptx) | dict(loi) | dict(qua_gio)}).'''
    tool = find_cuobjdump()
    if not tool: return dict(co_cuobjdump=False, thu_vien={})
    r = subprocess.run([python or sys.executable, '-c', _LIBS_CODE], env=child_env(isolated and python is not None), capture_output=True, text=True,
                       timeout=600)
    if r.returncode != 0 or 'LIBS=' not in r.stdout: return dict(co_cuobjdump=True, loi=(r.stderr or '')[-500:], thu_vien={})
    libs = [p for p in json.loads(r.stdout.split('LIBS=')[-1].strip().splitlines()[0]) if any(k in os.path.basename(p) for k in PTX_LIB_KEYS)]
    out = {}
    for lib in libs:
        name = os.path.basename(lib)
        try:
            p = subprocess.run([tool, '--list-ptx', lib], capture_output=True, text=True, timeout=timeout_s)
            names = [l.split(':', 1)[1].strip() for l in p.stdout.splitlines() if l.startswith('PTX file')]
            out[name] = dict(so_ptx=len(names), kien_truc_ptx=sorted({m for n in names for m in re.findall(r'sm_(\d+)', n)}, key=int))
            if p.returncode != 0 and not names: out[name] = dict(loi=(p.stderr or p.stdout).strip()[:200])
        except subprocess.TimeoutExpired:
            out[name] = dict(qua_gio=True)
    return dict(co_cuobjdump=True, cong_cu=tool, thu_vien=out)


def make_feature_input(vindr, out_npy, n=FEATURE_IMAGES, size=512):
    '''n ảnh train đầu (biến đổi ĐÁNH GIÁ của họ: Resize + CenterCrop + RescaleImage + chuẩn hoá ImageNet, ảnh `size`) -> .npy float32 dùng chung cho
       mọi lượt đặc trưng (cùng đầu vào từng bit). Chỉ tensor, không in gì của ảnh.'''
    from ..chexfound.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from ..config import BackboneSpec
    from ..data import VinDrImages, build_transforms
    _, tf_eval = build_transforms(BackboneSpec('chexfound', size, tuple(IMAGENET_DEFAULT_MEAN), tuple(IMAGENET_DEFAULT_STD)))
    ds = VinDrImages(vindr.keys, vindr.image_dir, tf_eval)
    x = np.stack([ds[int(j)][0].numpy() for j in vindr.indices('train')[:n]]).astype(np.float32)
    np.save(out_npy, x)
    return dict(so_anh=int(x.shape[0]), hinh=list(x.shape))


def run_features(out_npz, repo, config, ckpt, input_npy, amp, python=None, isolated=True, device='cuda', opts=('student.block_chunks=4',),
                 timeout_s=FEAT_TIMEOUT_S, encoder=None, pkg=None, attn=None):
    '''probe_features.py bằng `python` -> dict(dat, info, ma_thoat, phut, duoi_loi, tep). encoder (86j): lớp bọc encoders_ho thay CheXFound (pkg =
       thư mục chứa vindr_protocol; config / ckpt bỏ qua); dat còn đòi cổng nạp trọng số THẬT của encoders_ho (info['nap']['dat'] — 13/11).
       attn (86j §16.4 13/11 lần 2; chỉ cùng encoder RAD-DINO, chỉ môi trường mới): ép attention ('eager') cho phép đo nền.'''
    assert attn is None or encoder, 'attn chi dung voi encoder'
    cmd = [python or sys.executable, FEAT_SCRIPT, '--repo', repo, '--input', input_npy, '--out', out_npz, '--amp', amp, '--device', device]
    if encoder:
        cmd += ['--encoder', encoder, '--pkg', pkg or os.path.dirname(os.path.dirname(HERE))] + (['--attn', attn] if attn else [])
    else:
        cmd += ['--config', config, '--ckpt', ckpt, '--opts', *opts]
    t0 = time.time()
    try:
        # utf-8 + replace: thanh tiến độ của transformers 5.x (ký tự khối) giải mã theo locale cp1252 (máy Windows) làm mất đuôi lỗi (13/11)
        r = subprocess.run(cmd, env=child_env(isolated and python is not None), capture_output=True, text=True, encoding='utf-8', errors='replace',
                           timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        return dict(dat=False, qua_gio=True, info=None, ma_thoat=None, phut=(time.time() - t0) / 60, duoi_loi=str(e)[-500:], tep=out_npz)
    info = json.loads(r.stdout.split('DT=')[-1].strip().splitlines()[0]) if (r.returncode == 0 and 'DT=' in r.stdout) else None
    ok = bool(info and info['huu_han'] and info['thieu_khoa'] == 0 and (not encoder or (info.get('nap') or {}).get('dat') is True))
    return dict(dat=ok, qua_gio=False, info=info, ma_thoat=r.returncode, phut=(time.time() - t0) / 60, duoi_loi=(r.stderr or '')[-3000:], tep=out_npz)


def compare_features(ref_npz, test_npz):
    '''Sai số tương đối (max |Δ| / max |tham chiếu|) + cosine theo token (patch và cls, 4 tầng, mọi ảnh; phần chỉ có ở cả hai tệp — lớp bọc không
       token patch như BiomedCLIP: chỉ cls). -> dict.'''
    r, a = np.load(ref_npz), np.load(test_npz)
    parts = [p for p in ('patch', 'cls') if p in r.files and p in a.files]
    assert 'cls' in parts, 'thieu cls -> DUNG'
    out = {}
    for part in parts:
        R, A = r[part].astype(np.float64), a[part].astype(np.float64)
        assert R.shape == A.shape, (part, R.shape, A.shape)
        Rf, Af = R.reshape(-1, R.shape[-1]), A.reshape(-1, A.shape[-1])
        cos = (Rf * Af).sum(1) / np.maximum(np.linalg.norm(Rf, axis=1) * np.linalg.norm(Af, axis=1), 1e-12)
        out[part] = dict(lech_tuong_doi=float(np.abs(A - R).max() / max(np.abs(R).max(), 1e-12)), cos_min=float(cos.min()), cos_tb=float(cos.mean()))
    out['lech_tuong_doi'] = max(out[p]['lech_tuong_doi'] for p in parts)
    out['cos_min'] = min(out[p]['cos_min'] for p in parts)
    out['phan'] = parts
    return out


# 86j §16.4 (sửa 13/11 lần 2, Tuấn duyệt): ngưỡng TƯƠNG ĐỐI theo NỀN = cùng phép so ở môi trường mới (cùng lớp bọc / ảnh / GPU; RAD-DINO eager).
# Bản trước (lệch ≤ 2e-2, cos_min ≥ 0,999 tuyệt đối, chép từ §15.2 chưa hiệu chỉnh) dừng raddino_kieu_ho ở 0,99853 trong khi nền eager = 0,99829.
# 1 − cos ≈ góc² / 2 -> hệ số 4 trên 1 − cos = sai số gấp 2 (khớp hệ số 2 của lệch, như §15.2 (4)); lech_tran / cos_san chặn hỏng nặng.
RELATIVE_RULE = dict(he_so_lech=2.0, cong_lech=1e-3, he_so_thieu_cos=4.0, cong_thieu_cos=1e-4, lech_tran=2e-2, cos_san=0.99)


def relative_gate(cmp_test, cmp_floor, rule=RELATIVE_RULE):
    '''86j §16.4: cmp_* = compare_features(tham chiếu, thử) của phép so cần kiểm và của NỀN. Đạt ⇔ lệch ≤ he_so_lech × lệch_nền + cong_lech VÀ
       1 − cos_min ≤ he_so_thieu_cos × (1 − cos_min_nền) + cong_thieu_cos VÀ lệch ≤ lech_tran VÀ cos_min ≥ cos_san. -> dict(dat, ngưỡng, nền, luật).'''
    lim_l = rule['he_so_lech'] * cmp_floor['lech_tuong_doi'] + rule['cong_lech']
    lim_t = rule['he_so_thieu_cos'] * (1.0 - cmp_floor['cos_min']) + rule['cong_thieu_cos']
    lech, cos = cmp_test['lech_tuong_doi'], cmp_test['cos_min']
    dat = lech <= lim_l and (1.0 - cos) <= lim_t and lech <= rule['lech_tran'] and cos >= rule['cos_san']
    return dict(dat=bool(dat), lech=lech, cos_min=cos, nguong_lech=lim_l, nguong_cos_min=1.0 - lim_t, lech_nen=cmp_floor['lech_tuong_doi'],
                cos_min_nen=cmp_floor['cos_min'], luat=dict(rule), phan=cmp_test['phan'], phan_nen=cmp_floor['phan'])


def encoder_gate(cmp_old, cmp_floor):
    '''86j §16.4 cổng fp16: cmp_old = compare_features(cũ fp32, cũ fp16) — môi trường của họ, như lượt học; cmp_floor = compare_features(mới fp32,
       mới fp16), cùng lớp bọc / ảnh / GPU (RAD-DINO eager). Cổng nạp trọng số nằm ở run_features.'''
    assert cmp_old['phan'] == cmp_floor['phan'], (cmp_old['phan'], cmp_floor['phan'])
    return relative_gate(cmp_old, cmp_floor)


def device_gate(cmp_old, cmp_floor):
    '''86j §16.4 cổng GPU–CPU: cmp_old = compare_features(cũ CPU fp32, cũ GPU fp32) — bắt lỗi GPU chung cho fp16 và fp32 (nội suy, layernorm,
       softmax chạy fp32) mà cổng fp16 không thấy; cmp_floor = compare_features(mới CPU fp32, mới GPU fp32).'''
    assert cmp_old['phan'] == cmp_floor['phan'], (cmp_old['phan'], cmp_floor['phan'])
    return relative_gate(cmp_old, cmp_floor)


def gate_text(name, g):
    '''Một dòng tóm tắt kết quả relative_gate.'''
    return (f"{name}: lech {g['lech']:.2e} <= {g['nguong_lech']:.2e} (nen {g['lech_nen']:.2e}) | cos_min {g['cos_min']:.6f} >= {g['nguong_cos_min']:.6f} "
            f"(nen {g['cos_min_nen']:.6f}) | chan hong nang: lech <= {g['luat']['lech_tran']}, cos_min >= {g['luat']['cos_san']} -> "
            f"{'DAT' if g['dat'] else 'TRUOT'}")


def feature_stats(ref_npz, test_npz, n_worst=5):
    '''Thống kê cos theo token cho đọc / lưu (KHÔNG phải cổng): mỗi phần (patch / cls) -> lệch (như compare_features), cos_min, phân vị cos 1e-4 /
       1e-3 / 1e-2 / 0,5, số vector dưới 0,999, cos_min theo tầng (-n..-1), n_worst vector tệ nhất (tầng, cos, độ lớn / trung vị độ lớn cùng tầng ở
       tham chiếu). Tầng = trục 0 của mảng (4 tầng cuối).'''
    r, a = np.load(ref_npz), np.load(test_npz)
    out = {}
    for part in [p for p in ('patch', 'cls') if p in r.files and p in a.files]:
        R, A = r[part].astype(np.float64), a[part].astype(np.float64)
        assert R.shape == A.shape, (part, R.shape, A.shape)
        R, A = R.reshape(R.shape[0], -1, R.shape[-1]), A.reshape(A.shape[0], -1, A.shape[-1])
        nr = np.linalg.norm(R, axis=-1)
        cos = (R * A).sum(-1) / np.maximum(nr * np.linalg.norm(A, axis=-1), 1e-12)
        ratio = nr / np.maximum(np.median(nr, axis=1, keepdims=True), 1e-12)
        worst = np.argsort(cos, axis=None)[:n_worst]
        out[part] = dict(lech=float(np.abs(A - R).max() / max(np.abs(R).max(), 1e-12)), cos_min=float(cos.min()),
                         phan_vi=[float(v) for v in np.quantile(cos, [1e-4, 1e-3, 1e-2, 0.5])], duoi_0999=int((cos < 0.999).sum()), tong=int(cos.size),
                         theo_tang=[float(v) for v in cos.min(axis=1)],
                         te_nhat=[[int(i // cos.shape[1]) - cos.shape[0], float(cos.flat[i]), float(ratio.flat[i])] for i in worst])
    return out


def stats_text(name, st):
    '''Một dòng mỗi phần cho output notebook (vector tệ nhất chỉ lưu JSON).'''
    return '\n'.join(f"{name} [{p}] lech {v['lech']:.2e} | cos_min {v['cos_min']:.6f} | cos q 1e-4 / 1e-3 / 1e-2 / 0,5: "
                     + ' / '.join(f'{x:.6f}' for x in v['phan_vi']) + f" | duoi 0,999: {v['duoi_0999']}/{v['tong']} | theo tang: "
                     + ' '.join(f'{x:.6f}' for x in v['theo_tang']) for p, v in st.items())


def feature_rule(cmp_old, cmp_new, rule=FEATURE_RULE):
    '''§15.2 (4): lệch(cũ) ≤ he_so × lệch(mới) + cong VÀ cos_min(cũ) ≥ cos_min. -> dict(dat, nguong_lech, ...).'''
    lim = rule['he_so'] * cmp_new['lech_tuong_doi'] + rule['cong']
    return dict(dat=bool(cmp_old['lech_tuong_doi'] <= lim and cmp_old['cos_min'] >= rule['cos_min']), nguong_lech=lim, lech_cu=cmp_old['lech_tuong_doi'],
                lech_moi=cmp_new['lech_tuong_doi'], cos_min_cu=cmp_old['cos_min'], cos_min_moi=cmp_new['cos_min'], luat=dict(rule))


def _gate_ops(run):
    return [o for o in run['phep_tinh'] if o.get('gate')]


def verdict(old, new, features=None, smokes=None, gate_names=None):
    '''Luật §15.2 (đặt trước). old / new = kết quả run_ops (môi trường của họ / đối chứng môi trường mới, cùng GPU); features = dict(cu=run_features fp16
       môi trường cũ, moi=…fp16 môi trường mới, tham_chieu=…fp32 môi trường mới, so=dict(cu=compare, moi=compare) | None); smokes = {kind: smoke_test |
       dict(dat=False, loi)}. Thứ tự: KHONG_PHAI_G4 > PHEP_THU_HONG > KHONG_CHAY > QUA_GIO > CHAY_NHUNG_LECH > CHUA_DU > CHAY_DUOC.'''
    reasons = []
    names = list(gate_names) if gate_names else [o['ten'] for o in _gate_ops(new)] or [o['ten'] for o in _gate_ops(old)]
    cap = ((old.get('thong_tin') or {}).get('capability') or (new.get('thong_tin') or {}).get('capability'))
    if cap != G4_CAPABILITY:
        return dict(ket_luan='KHONG_PHAI_G4', ly_do=[f'capability {cap} != {G4_CAPABILITY} -> phep thu nay danh cho G4'], capability=cap)
    # đối chứng sạch: môi trường mới phải đạt mọi phép gate (xFormers không có -> bỏ mục đó)
    new_by = {o['ten']: o for o in new['phep_tinh']}
    bad_ctrl = [n for n in names if not (new_by.get(n, {}).get('trang_thai') in ('khong_co',) or new_by.get(n, {}).get('dat') is True)]
    if not (new.get('thong_tin') or {}).get('cuda_dung_duoc') or bad_ctrl or not new['xong']:
        return dict(ket_luan='PHEP_THU_HONG', ly_do=[f'doi chung (moi truong moi) truot: {bad_ctrl or "khong chay het"}'], capability=cap)
    for k, ten in (('moi', 'moi truong moi fp16'), ('tham_chieu', 'tham chieu fp32')):
        if features and features.get(k) is not None and not features[k].get('dat'):
            return dict(ket_luan='PHEP_THU_HONG', ly_do=[f'dac trung {ten} (doi chung) loi / khong huu han'], capability=cap)
    info = old.get('thong_tin') or {}
    old_by = {o['ten']: o for o in old['phep_tinh']}
    if not info.get('cuda_dung_duoc') or info.get('loi_khoi_tao'):
        return dict(ket_luan='KHONG_CHAY', ly_do=[f"moi truong cu khong dung duoc CUDA ({info.get('loi_khoi_tao') or 'cuda khong san sang'})"], capability=cap)
    errs = [f"{n}: {old_by[n].get('loi', '')[:200]}" for n in names if old_by.get(n, {}).get('trang_thai') == 'loi']
    errs += [f"chay thu {k}: {str(v.get('loi', ''))[-300:]}" for k, v in (smokes or {}).items() if v.get('loi')]
    numeric = []
    fc = (features or {}).get('cu')
    if fc is not None and not fc.get('dat') and not fc.get('qua_gio'):
        if fc.get('ma_thoat') == 0 and fc.get('info'):          # chạy xong nhưng NaN / inf hoặc thiếu khoá -> lệch số, không phải "không chạy"
            numeric.append(f"dac trung moi truong cu: huu han {fc['info'].get('huu_han')}, thieu khoa {fc['info'].get('thieu_khoa')}")
        else:
            errs.append(f"dac trung moi truong cu: ma thoat {fc.get('ma_thoat')} | {fc.get('duoi_loi', '')[-300:]}")
    if not old['xong'] and not old['qua_gio']:
        errs.append(f"tien trinh phep tinh dung giua chung (ma thoat {old['ma_thoat']}): {old['duoi_loi'][-300:]}")
    if errs:
        return dict(ket_luan='KHONG_CHAY', ly_do=errs, capability=cap)
    if old['qua_gio'] or (features and (features.get('cu') or {}).get('qua_gio')):
        return dict(ket_luan='QUA_GIO', ly_do=['tien trinh con vuot gio cho -> khong ket luan, giu phan da do'], capability=cap)
    missing = [n for n in names if n not in old_by]
    numeric += [f"{n}: lech {old_by[n].get('lech')} > nguong {old_by[n].get('nguong')}" for n in names
                if old_by.get(n, {}).get('trang_thai') == 'chay' and not old_by[n].get('dat')]
    if names and 'xformers_mea' in names and old_by.get('xformers_mea', {}).get('trang_thai') == 'khong_co':
        numeric.append('xformers_mea: moi truong cu KHONG co xFormers (phai co 0.0.18)')
    rule = feature_rule(features['so']['cu'], features['so']['moi']) if features and features.get('so') else None
    if rule is not None and not rule['dat']:
        numeric.append(f"dac trung: lech cu {rule['lech_cu']:.2e} (nguong {rule['nguong_lech']:.2e}), cos_min cu {rule['cos_min_cu']:.6f}")
    smoke_bad = [k for k, v in (smokes or {}).items() if not v.get('dat')]
    if numeric or smoke_bad:
        return dict(ket_luan='CHAY_NHUNG_LECH', ly_do=numeric + [f'chay thu {k} khong dat hai cong' for k in smoke_bad], capability=cap, luat_dac_trung=rule)
    need = []
    if missing: need.append(f'thieu phep tinh: {missing}')
    if rule is None: need.append('chua so dac trung (C4)')
    if not smokes or set(smokes) != {'lin', 'glori'}: need.append('chua du chay thu Linear + GLoRI (C5)')
    if need:
        return dict(ket_luan='CHUA_DU', ly_do=need, capability=cap, luat_dac_trung=rule)
    return dict(ket_luan='CHAY_DUOC', ly_do=['moi phep tinh gate + dac trung + chay thu dat (PLAN_86 15.2)'], capability=cap, luat_dac_trung=rule)


def ops_table(run, title):
    '''Dòng in gọn cho notebook: từng phép tính — trạng thái, lệch / ngưỡng, ms lần đầu / lần hai (lần đầu gồm dịch JIT nếu có).'''
    info = run.get('thong_tin') or {}
    lines = [f"== {title}: Python {info.get('python')} | torch {(info.get('phien_ban') or {}).get('torch')} (CUDA {info.get('cuda')}) | xFormers "
             f"{(info.get('phien_ban') or {}).get('xformers')} | GPU {info.get('gpu')} cap {info.get('capability')} | arch_list {info.get('arch_list')} | NVCC "
             f"SASS {info.get('nvcc_sass')} PTX {info.get('nvcc_ptx')} | cuDNN {info.get('cudnn')} | NCCL {info.get('nccl')} | khoi tao "
             f"{info.get('ms_khoi_tao', float('nan')):.0f} ms | {run['phut']:.1f} phut" + (' | QUA GIO' if run['qua_gio'] else '')
             + ('' if run['xong'] else f" | KHONG CHAY HET (ma thoat {run['ma_thoat']})")]
    if info.get('canh_bao'): lines.append(f"   canh bao luc khoi tao: {info['canh_bao']}")
    if info.get('loi_khoi_tao'): lines.append(f"   LOI khoi tao: {info['loi_khoi_tao']}")
    for o in run['phep_tinh']:
        tag = '' if o.get('gate') else ' (thong tin)'
        if o['trang_thai'] == 'chay':
            lines.append(f"   [{'DAT ' if o['dat'] else 'TRUOT'}] {o['ten']:26s}{tag} lech {o['lech']:.2e} / nguong {o['nguong']:.0e} | "
                         f"{o['ms_lan_dau']:8.1f} / {o['ms_lan_hai']:7.1f} ms")
        elif o['trang_thai'] == 'loi':
            lines.append(f"   [LOI ] {o['ten']:26s}{tag} {o['loi'][:220]}")
        else:
            lines.append(f"   [ -- ] {o['ten']:26s}{tag} {o['trang_thai']}")
    return '\n'.join(lines)
