# -*- coding: utf-8 -*-
"""Môi trường của Ver1 (PLAN_89 §1, §4.1) — chép từ vindr_protocol/upstream/runner.py @ 0.8.2 (đã chạy ở 86f / 86j trên A100), gom về một chỗ:

  - môi trường THƯ VIỆN CỦA HỌ cho mọi tiến trình torchrun (conda-extras.yaml của họ: Python 3.9, torch 2.0.0 + CUDA 11.7, torchvision 0.15.0,
    xFormers 0.0.18, torchmetrics 0.10.3, fvcore, iopath, omegaconf; gói họ import mà không ghim: bản cùng thời 12/2023);
  - repo CheXFound công khai, checkout commit ghim, KHÔNG sửa tệp nào (kiểm HEAD, cây sạch, md5 linear_glori.py);
  - Python của notebook: chỉ điều phối + chấm (torchmetrics 0.10.3) + lượt đo đặc trưng "môi trường mới" (transformers / timm của Colab).

Trạng thái môi trường đi theo đối tượng TheirEnv (không biến toàn cục)."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict, dataclass

CHEXFOUND_COMMIT = '7e288d06532f24abedb7d00781185c36727d9d69'
CHEXFOUND_URL = 'https://github.com/RPIDIAL/CheXFound.git'
LINEAR_GLORI = 'chexfound/eval/classification/linear_glori.py'
LINEAR_GLORI_MD5 = '123bf478c4cdaf548084e618facb4f5f'          # md5 (LF) bản đã đọc (= 86)

THEIR_PYTHON = '3.9'
THEIR_CORE = {'torch': '2.0.0', 'torchvision': '0.15.0', 'xformers': '0.0.18', 'torchmetrics': '0.10.3'}
# Linux: torch / torchvision bản CUDA 11.7 từ kho PyTorch (torchvision 0.15.0 trên PyPI đòi torch==2.0.0+cu117 mà PyPI không có — 86f 12/11).
THEIR_TORCH = {'linux-64': (('torch==2.0.0+cu117', 'torchvision==0.15.0+cu117'), 'https://download.pytorch.org/whl/cu117'),
               'win-64': (('torch==2.0.0', 'torchvision==0.15.0'), None)}
THEIR_PIP = ('xformers==0.0.18', 'torchmetrics==0.10.3', 'numpy==1.26.4', 'setuptools<70', 'fvcore==0.1.5.post20221221', 'iopath==0.1.10',
             'omegaconf==2.3.0', 'pandas==2.1.4', 'pillow==10.2.0', 'scipy==1.11.4', 'scikit-learn==1.3.2', 'scikit-image==0.22.0',
             'scikit-multilearn==0.2.0', 'toolz==0.12.1', 'transformers==4.36.2', 'accelerate==0.25.0', 'peft==0.7.1', 'open_clip_torch==2.23.0',
             'timm==0.9.12')
THEIR_ENV_DIR = '/content/moi_truong_ho'
MICROMAMBA_URL = 'https://micro.mamba.pm/api/micromamba/{platform}/latest'
# Python của notebook: torchmetrics 0.10.3 (chấm như mã họ; cần pkg_resources -> setuptools<80). Thư viện lõi của Colab KHÔNG được đổi.
NOTEBOOK_PIP = ('setuptools<80', 'torchmetrics==0.10.3')
GUARD_PACKAGES = ('torch', 'torchvision', 'numpy', 'pillow', 'transformers')
NEW_ENV_MODULES = ('transformers', 'timm', 'safetensors', 'huggingface_hub')   # lượt đặc trưng "môi trường mới" (nền của cổng số học)
MIN_FREE_GB = 40                 # checkpoint mã họ ghi (GLoRI ~8 GB / lượt) + kho ảnh


@dataclass
class TheirEnv:
    python: str                  # Python 3.9 của môi trường họ (mọi tiến trình torchrun / đo đặc trưng "cũ")
    repo: str                    # repo CheXFound @ commit ghim
    package_dir: str             # thư mục CHỨA vindr_bench (gốc clone của gói)

    def to_dict(self):
        return asdict(self)


def _git(*args):
    r = subprocess.run(['git', *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[-500:]}")
    return r.stdout.strip()


def md5_lf(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read().replace(b'\r\n', b'\n')).hexdigest()


def clone_chexfound(dest, commit=CHEXFOUND_COMMIT, url=CHEXFOUND_URL):
    '''Clone repo CheXFound (nếu chưa có), checkout --detach commit ghim; kiểm HEAD, cây sạch, md5 linear_glori.py. -> dict.'''
    if not os.path.isdir(os.path.join(dest, '.git')):
        _git('clone', '--quiet', url, dest)
    _git('-C', dest, 'checkout', '--quiet', '--detach', commit)
    head = _git('-C', dest, 'rev-parse', 'HEAD')
    dirty = _git('-C', dest, 'status', '--porcelain', '--untracked-files=no')
    md5 = md5_lf(os.path.join(dest, LINEAR_GLORI))
    assert head == commit and not dirty and md5 == LINEAR_GLORI_MD5, (
        f'repo CheXFound: HEAD {head[:12]} / sua {dirty[:80]!r} / md5 linear_glori {md5} KHAC ban da doc -> DUNG')
    return dict(repo=dest, head=head, md5_linear_glori=md5)


def _version(name):
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def install_notebook_packages(disk='/content', min_free_gb=MIN_FREE_GB):
    '''Python notebook: pip NOTEBOOK_PIP (+ timm nếu thiếu); thư viện lõi của Colab KHÔNG đổi; torchmetrics nạp được ngay = 0.10.3; module của lượt
       "môi trường mới" có đủ; đĩa >= min_free_gb. -> dict.'''
    import importlib
    import importlib.util
    import warnings
    pkgs = list(NOTEBOOK_PIP) + ([] if importlib.util.find_spec('timm') else ['timm'])
    before = {m: _version(m) for m in GUARD_PACKAGES}
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-input', *pkgs], capture_output=True, text=True)
    assert r.returncode == 0, f'pip install THAT BAI:\n{(r.stderr or r.stdout)[-3000:]}'
    importlib.invalidate_caches()
    after = {m: _version(m) for m in GUARD_PACKAGES}
    changed = {m: [before[m], after[m]] for m in GUARD_PACKAGES if before[m] != after[m]}
    assert not changed, f'pip DOI thu vien loi cua Colab {changed} -> DUNG: Runtime > Disconnect and delete runtime, roi bao Claude'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import torchmetrics
    assert torchmetrics.__version__ == '0.10.3', (f'torchmetrics trong tien trinh nay = {torchmetrics.__version__} (nap TRUOC khi pip?) -> '
                                                  'Runtime > Restart session, chay lai tu C0')
    missing = [m for m in NEW_ENV_MODULES if importlib.util.find_spec(m) is None]
    assert not missing, f'Python notebook thieu {missing} (luot dac trung moi truong moi) -> DUNG'
    free = shutil.disk_usage(disk).free / 2 ** 30 if os.path.isdir(disk) else None
    assert free is None or free >= min_free_gb, f'o dia {disk} con {free:.0f} GB < {min_free_gb} GB -> DUNG'
    conflicts = [s.strip() for s in (r.stderr + '\n' + r.stdout).splitlines() if ' requires ' in s and 'but you have' in s]
    return dict(goi=pkgs, loi_giu_nguyen=after, moi={m: _version(m) for m in ('torchmetrics', 'setuptools', *NEW_ENV_MODULES)},
                xung_dot_pip=conflicts[:10], dia_trong_gb=free, python=sys.version.split()[0])


def _platform():
    return 'win-64' if os.name == 'nt' else 'linux-64'


def _env_python(dest):
    return os.path.join(dest, 'python.exe') if os.name == 'nt' else os.path.join(dest, 'bin', 'python')


def _get_micromamba(mm_dir, log=print):
    import io
    import urllib.request
    member = 'Library/bin/micromamba.exe' if os.name == 'nt' else 'bin/micromamba'
    mm = os.path.join(mm_dir, *member.split('/'))
    if os.path.exists(mm):
        return mm
    os.makedirs(mm_dir, exist_ok=True)
    data = urllib.request.urlopen(MICROMAMBA_URL.format(platform=_platform()), timeout=600).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:bz2') as tf:
        m = tf.getmember(member)
        assert m.isfile() and not os.path.isabs(m.name) and '..' not in m.name.split('/'), f'goi micromamba la ({m.name}) -> DUNG'
        try:
            tf.extract(m, mm_dir, filter='data')
        except TypeError:
            tf.extract(m, mm_dir)
    os.chmod(mm, 0o755)
    log(f'  micromamba: {len(data) / 2 ** 20:.0f} MB tai ve')
    return mm


_PROBE_ENV = '''RUN_GPU = {run_gpu}
import json, importlib, sys, warnings
warnings.simplefilter("ignore")
v = {{}}
for m in ("torch", "torchvision", "xformers", "torchmetrics", "numpy", "pandas", "PIL", "scipy", "sklearn", "skimage", "skmultilearn", "fvcore",
          "iopath", "omegaconf", "peft", "transformers", "open_clip", "timm", "toolz", "safetensors", "huggingface_hub"):
    try: v[m] = getattr(importlib.import_module(m), "__version__", "co")
    except Exception: v[m] = None
import torch
info = dict(phien_ban=v, python=sys.version.split()[0], cuda=torch.version.cuda, cuda_dung_duoc=torch.cuda.is_available(),
            gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            bo_nho_gb=(torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if torch.cuda.is_available() else None),
            cudnn=(torch.backends.cudnn.version() if torch.cuda.is_available() else None))
if RUN_GPU and torch.cuda.is_available() and v["xformers"]:
    from xformers.ops import memory_efficient_attention
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, w = (torch.randn(2, 257, 16, 64, device="cuda", generator=g).half() for _ in range(3))
    o = memory_efficient_attention(q, k, w).float()
    a = torch.softmax(q.float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-1, -2) / 8.0, -1) @ w.float().transpose(1, 2)
    info["xformers_lech"] = float((o - a.transpose(1, 2)).abs().max())
print("MT=" + json.dumps(info))
'''


def isolated_env(extra_path=()):
    '''Biến môi trường cho Python của họ: PYTHONPATH CHỈ extra_path, không site người dùng, bỏ PYTHONHOME, bỏ expandable_segments (torch 2.0 chưa có).'''
    env = dict(os.environ, PYTHONNOUSERSITE='1')
    env.pop('PYTHONHOME', None)
    if extra_path:
        env['PYTHONPATH'] = os.pathsep.join(extra_path)
    else:
        env.pop('PYTHONPATH', None)
    if 'expandable_segments' in env.get('PYTORCH_CUDA_ALLOC_CONF', ''): env.pop('PYTORCH_CUDA_ALLOC_CONF')
    return env


def _probe_env(python, require_cuda=True):
    r = subprocess.run([python, '-c', _PROBE_ENV.format(run_gpu=bool(require_cuda))], env=isolated_env(), capture_output=True, text=True)
    assert r.returncode == 0 and 'MT=' in r.stdout, f'kiem moi truong THAT BAI:\n{r.stderr[-3000:]}'
    info = json.loads(r.stdout.split('MT=')[-1].strip().splitlines()[0])
    if require_cuda:
        assert info['cuda_dung_duoc'], f"moi truong cua ho: torch {info['phien_ban']['torch']} KHONG dung duoc GPU {info['gpu']} -> DUNG (GPU phai A100)"
        assert info['cuda'] == '11.7', f"CUDA cua torch = {info['cuda']} != 11.7 -> DUNG"
        assert 'A100' in (info['gpu'] or '') and info['bo_nho_gb'] >= 70, (
            f"GPU {info['gpu']} ({info['bo_nho_gb']:.0f} GB): Ver1 can A100 80 GB (Runtime > Change runtime type) -> DUNG")
        assert info.get('xformers_lech') is not None and info['xformers_lech'] < 2e-2, f"xFormers tren GPU lech {info.get('xformers_lech')} -> DUNG"
    return info


def _pip_env():
    dropped = sorted(k for k in os.environ if k.startswith(('PIP_', 'UV_', 'PYTHON')))
    env = {k: v for k, v in os.environ.items() if k not in dropped}
    env.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull)
    return env, dropped


def their_pip_command(python, skip=()):
    torch_pins, index = THEIR_TORCH[_platform()]
    pkgs = [p for p in (*torch_pins, *THEIR_PIP) if p.split('==')[0].split('<')[0] not in skip]
    cmd = [python, '-m', 'pip', 'install', '--no-input', '--disable-pip-version-check', '--progress-bar', 'off']
    if index: cmd += ['--extra-index-url', index]
    return cmd + pkgs, pkgs


def build_their_env(dest=THEIR_ENV_DIR, skip=(), require_cuda=True, log=print):
    '''micromamba tạo Python 3.9 (conda-forge) tại dest + pip THEIR_TORCH / THEIR_PIP (bỏ skip — chỉ bản kiểm trên máy CPU). Cổng: Python 3.9, đúng
       THEIR_CORE, numpy 1.x, (require_cuda) CUDA 11.7 dùng được trên A100 + xFormers đúng trên GPU. Gọi lại khi đã dựng: chỉ cài phần thiếu + kiểm.'''
    t0 = time.time()
    py = _env_python(dest)
    if not os.path.exists(py):
        mm_dir = dest + '_micromamba'
        mm = _get_micromamba(mm_dir, log=log)
        r = subprocess.run([mm, 'create', '-y', '-q', '-p', dest, '-c', 'conda-forge', '--override-channels', f'python={THEIR_PYTHON}', 'pip'],
                           capture_output=True, text=True, env=dict(os.environ, MAMBA_ROOT_PREFIX=mm_dir))
        assert r.returncode == 0 and os.path.exists(py), f'micromamba tao Python {THEIR_PYTHON} THAT BAI:\n{(r.stderr or r.stdout)[-3000:]}'
        log(f'  Python {THEIR_PYTHON} (conda-forge) tai {dest}: {(time.time() - t0) / 60:.1f} phut')
    cmd, pkgs = their_pip_command(py, skip=skip)
    env, dropped = _pip_env()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0, f'pip cai thu vien cua ho THAT BAI (bien bo khoi pip: {dropped}):\n{(r.stdout + chr(10) + r.stderr)[-4000:]}'
    log(f'  pip: {len(pkgs)} goi cai xong ({(time.time() - t0) / 60:.1f} phut tu dau)')
    info = _probe_env(py, require_cuda=require_cuda)
    v = info['phien_ban']
    assert info['python'].startswith(THEIR_PYTHON + '.'), f"Python {info['python']} != {THEIR_PYTHON} -> DUNG"
    bad = {k: v.get(k) for k, want in THEIR_CORE.items() if k not in skip and (v.get(k) or '').split('+')[0] != want}
    assert not bad, f'phien ban KHAC conda-extras.yaml cua ho: {bad} (can {THEIR_CORE}) -> DUNG'
    assert (v.get('numpy') or '').startswith('1.'), f"numpy {v.get('numpy')} (torch 2.0 can numpy 1.x) -> DUNG"
    return dict(python=py, thong_tin=info, goi=pkgs, bien_bo_khoi_pip=dropped, phut=(time.time() - t0) / 60)


CHEXFOUND_DRIVE = ('chexfound',)                          # cxrlt_2024/chexfound — tệp công khai của tác giả (cùng tệp 85 / 86)
CONFIG_NAME, CONFIG_MD5 = 'config.yaml', 'a776cfd0d3e993f293d4434901c84e7e'   # md5 bản công khai đã kiểm 11/11 (= 86)
CONFIG_DIR = '/content/chexfound_cau_hinh'


def fetch_config(kiem_ke, tai, xung_dot, dest_dir=CONFIG_DIR, log=print):
    '''config.yaml của CheXFound qua khối đọc Drive chuẩn (md5 = bản công khai) -> bản chép TÊN CỐ ĐỊNH (tên tải về mang ID Drive; mã họ ghi mọi tham số
       vào log). Lớp bọc chỉ cần tệp này cho setup(args) + dtype autocast fp16 của họ; trọng số CheXFound không dùng.'''
    kk = kiem_ke(CHEXFOUND_DRIVE)[1]
    versions = kk.get(CONFIG_NAME)
    assert versions, f"thieu {CONFIG_NAME} trong cxrlt_2024/{'/'.join(CHEXFOUND_DRIVE)} -> DUNG"
    if xung_dot(versions): log(f"  ! {CONFIG_NAME}: {len(versions)} ban NOI DUNG KHAC -> dung ban MOI NHAT ({versions[0]['mt']})")
    src = tai(versions[0])
    with open(src, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    assert md5 == CONFIG_MD5, f'{CONFIG_NAME}: md5 {md5} KHAC ban cong khai da kiem -> DUNG'
    os.makedirs(dest_dir, exist_ok=True)
    dst = os.path.join(dest_dir, CONFIG_NAME)
    shutil.copyfile(src, dst)
    return dst


def child_env(env):
    '''Môi trường cho tiến trình torchrun / đo đặc trưng của họ: PYTHONPATH = repo CheXFound + thư mục chứa vindr_bench, cô lập.'''
    return isolated_env((env.repo, env.package_dir))


_CHECK_IMPORTS = '''import json, importlib, sys, warnings
warnings.simplefilter("ignore")
from chexfound.eval.classification import linear_glori
from chexfound.layers import attention, block, swiglu_ffn
import vindr_bench.protocols.chexfound_env.launch
import vindr_bench.protocols.chexfound_env.dataset
import vindr_bench.heads.mldecoder
import vindr_bench.encoders
v = {"python": sys.version.split()[0]}
for m in ("torch", "torchvision", "torchmetrics", "fvcore", "omegaconf", "peft", "open_clip", "skmultilearn", "transformers", "timm", "xformers"):
    try: v[m] = getattr(importlib.import_module(m), "__version__", "co")
    except Exception: v[m] = None
v["xformers_cua_ho"] = {n: bool(getattr(x, "XFORMERS_AVAILABLE", False)) for n, x in (("attention", attention), ("block", block),
                       ("swiglu_ffn", swiglu_ffn))}
print("PHIEN_BAN=" + json.dumps(v))
'''


def check_imports(env, expect_xformers=True):
    '''Import linear_glori.py của họ + các module của gói chạy trong tiến trình con, bằng Python của họ (như torchrun sẽ làm). -> phiên bản.'''
    r = subprocess.run([env.python, '-c', _CHECK_IMPORTS], env=child_env(env), capture_output=True, text=True)
    assert r.returncode == 0 and 'PHIEN_BAN=' in r.stdout, f'import ma CheXFound / goi THAT BAI:\n{r.stderr[-3000:]}'
    versions = json.loads(r.stdout.split('PHIEN_BAN=')[-1].strip().splitlines()[0])
    assert versions['torchmetrics'] == '0.10.3', f"torchmetrics {versions['torchmetrics']} != 0.10.3 (ban ho ghim) -> DUNG"
    if expect_xformers is not None:
        flags = versions['xformers_cua_ho']
        assert all(f == expect_xformers for f in flags.values()), f'XFORMERS_AVAILABLE trong ma cua ho {flags} != {expect_xformers} -> DUNG'
    return versions
