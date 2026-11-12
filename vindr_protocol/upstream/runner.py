"""Điều phối 86e (PLAN_86 §12): clone + kiểm mã CheXFound (commit ghim, KHÔNG sửa), cài + kiểm thư viện, tải trọng số (cùng tệp 85 / 86b), kiểm nạp
bằng hàm dựng của họ, chạy `linear_glori.py` của họ qua torchrun (theo dõi tiến độ, bắt tràn bộ nhớ), đọc tệp kết quả của họ, chạy thử nhỏ, chạy đầy
đủ một loại đầu, lưu / khôi phục kết quả từng loại đầu qua tệp resume của khối ghi cuối (mất runtime giữa chừng không mất loại đầu đã xong)."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

from ..config import CHEXFOUND_COMMIT, LR_GRID as LR_GRID_DEFAULT       # = learning_rates mặc định của linear_glori.py (13 lr)

CHEXFOUND_URL = 'https://github.com/RPIDIAL/CheXFound.git'
# thư viện mã của họ cần ngoài Colab (conda-extras.yaml của họ: torchmetrics 0.10.3, fvcore, iopath, omegaconf; linear_glori.py import toolz, peft;
# eval/utils.py import skmultilearn, open_clip, transformers). Không có xFormers (họ: 0.0.18) -> mã của họ tự dùng nhánh attention tường minh.
# setuptools<80: torchmetrics 0.10.3 import pkg_resources lúc import — setuptools mới bỏ module này (thử trên máy 12/11: thiếu -> import lỗi).
PIP_PACKAGES = ('setuptools<80', 'torchmetrics==0.10.3', 'fvcore', 'iopath', 'omegaconf', 'peft', 'toolz', 'scikit-multilearn', 'open_clip_torch')
GUARD_PACKAGES = ('torch', 'torchvision', 'numpy', 'pillow', 'transformers')     # thư viện lõi của Colab: pip KHÔNG được đổi phiên bản
NEW_PACKAGES = ('torchmetrics', 'fvcore', 'iopath', 'omegaconf', 'peft', 'toolz', 'scikit-multilearn', 'open_clip_torch', 'setuptools')
MIN_FREE_GB = 40               # checkpoint mã họ ghi: GLoRI ~8 GB (9 đầu x ~100 MB kèm trạng thái AdamW, ghi mỗi 2 epoch), Linear ~0,1 GB
# Trọng số đưa cho mã họ qua LIÊN KẾT TÊN CỐ ĐỊNH: tên tệp khối đọc Drive tải về mang ID Drive, mà mã họ ghi mọi tham số vào log.
WEIGHTS_DIR = '/content/chexfound_trong_so'
LINEAR_GLORI = 'chexfound/eval/classification/linear_glori.py'
LINEAR_GLORI_MD5 = '123bf478c4cdaf548084e618facb4f5f'          # md5 (LF) bản đọc 09/11 (nguon_chexfound_7e288d0/README) = CheXFound-main 12/11
LAUNCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'launch.py')
OPTS = ('student.block_chunks=4',)                               # G7: config.yaml ghi 0, checkpoint 4 khúc -> không chỉnh thì 24 tầng không nạp
BATCH, IMAGE_SIZE = 256, 512                                     # 1 GPU x 256 = lô hiệu dụng 4 x 64 của README; ảnh 512 như script / README của họ
IMAGE_SIZES = (224, 512, 518)   # 512: script / README của họ (86e / 86f); 224: mặc định --image-size (86g §14.1; BiomedCLIP 86j); 518: RAD-DINO (86j §16)
PAPER_LRS = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3)   # §IV-A.2 của bài — Linear VÀ GLoRI (Tuấn 12/11, PLAN_86 §13.1)
GLORI_HALVES = (PAPER_LRS[:5], PAPER_LRS[5:])                   # khi 9 đầu GLoRI cùng lúc tràn bộ nhớ (PLAN_86 §12.3)
# ---- môi trường THƯ VIỆN CỦA HỌ cho 86f (PLAN_86 §13.2): conda-extras.yaml = Python 3.9, pytorch 2.0.0 + pytorch-cuda 11.7, torchvision 0.15.0,
# xformers 0.0.18, torchmetrics 0.10.3, fvcore, iopath, omegaconf. Cài bằng pip (bản PyPI cùng phiên bản, dựng với CUDA 11.7) trong Python 3.9 của
# micromamba (conda-forge). Gói mã họ import nhưng họ không ghim: bản cùng thời (12/2023), chỉ cần import được (không vào phép tính Linear / GLoRI).
THEIR_PYTHON = '3.9'
THEIR_CORE = {'torch': '2.0.0', 'torchvision': '0.15.0', 'xformers': '0.0.18', 'torchmetrics': '0.10.3'}
# torch / torchvision theo nền tảng. Linux (Colab): bản CUDA 11.7 từ KHO CỦA PYTORCH — torchvision 0.15.0 bản Linux trên PyPI khai phụ thuộc
# `torch==2.0.0+cu117` mà PyPI không có (PyPI cấm phiên bản có nhãn cục bộ) -> pip chỉ PyPI KHÔNG giải được (Colab 12/11: ResolutionImpossible; tái hiện
# trên máy bằng --dry-run --platform manylinux). = nguồn của conda-extras.yaml (kênh pytorch, pytorch-cuda 11.7). Windows: chỉ bản kiểm CPU trên máy.
THEIR_TORCH = {'linux-64': (('torch==2.0.0+cu117', 'torchvision==0.15.0+cu117'), 'https://download.pytorch.org/whl/cu117'),
               'win-64': (('torch==2.0.0', 'torchvision==0.15.0'), None)}
THEIR_PIP = ('xformers==0.0.18', 'torchmetrics==0.10.3', 'numpy==1.26.4', 'setuptools<70', 'fvcore==0.1.5.post20221221', 'iopath==0.1.10',
             'omegaconf==2.3.0', 'pandas==2.1.4', 'pillow==10.2.0', 'scipy==1.11.4', 'scikit-learn==1.3.2', 'scikit-image==0.22.0',
             'scikit-multilearn==0.2.0', 'toolz==0.12.1', 'transformers==4.36.2', 'accelerate==0.25.0', 'peft==0.7.1', 'open_clip_torch==2.23.0',
             'timm==0.9.12')
THEIR_ENV_DIR = '/content/moi_truong_ho'
MICROMAMBA_URL = 'https://micro.mamba.pm/api/micromamba/{platform}/latest'
RESULTS_FILE = 'results_eval_linear.json'
_OOM = ('CUDA out of memory', 'OutOfMemoryError', 'CUBLAS_STATUS_ALLOC_FAILED')


def _git(*args):
    r = subprocess.run(['git', *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[-500:]}")
    return r.stdout.strip()


def md5_lf(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read().replace(b'\r\n', b'\n')).hexdigest()


def clone_chexfound(dest, commit=CHEXFOUND_COMMIT, url=CHEXFOUND_URL):
    '''Clone repo CheXFound công khai (nếu chưa có), checkout --detach commit ghim; kiểm HEAD = commit, không sửa tệp theo dõi, md5 linear_glori.py =
       bản đã đọc. -> dict(repo, head, so_tep_py, md5_linear_glori).'''
    if not os.path.isdir(os.path.join(dest, '.git')):
        _git('clone', '--quiet', url, dest)
    _git('-C', dest, 'checkout', '--quiet', '--detach', commit)
    head = _git('-C', dest, 'rev-parse', 'HEAD')
    dirty = _git('-C', dest, 'status', '--porcelain', '--untracked-files=no')
    md5 = md5_lf(os.path.join(dest, LINEAR_GLORI))
    assert head == commit and not dirty and md5 == LINEAR_GLORI_MD5, (
        f'repo CheXFound: HEAD {head[:12]} / sua {dirty[:80]!r} / md5 linear_glori {md5} KHAC ban da doc -> DUNG')
    n_py = sum(f.endswith('.py') for _, _, fs in os.walk(os.path.join(dest, 'chexfound')) for f in fs)
    return dict(repo=dest, head=head, so_tep_py=n_py, md5_linear_glori=md5)


def _version(name):
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def install_packages(packages=PIP_PACKAGES, guard=GUARD_PACKAGES, disk='/content', min_free_gb=MIN_FREE_GB):
    '''pip install thư viện mã họ cần. Cổng: thư viện lõi của Colab (guard) KHÔNG đổi phiên bản; torchmetrics import được NGAY trong tiến trình này và
       = 0.10.3 (ô chấm dùng); ổ đĩa còn >= min_free_gb. Dòng xung đột của pip (gói khác của Colab đòi phiên bản khác) chỉ ghi lại. -> dict.'''
    import importlib
    import warnings
    before = {m: _version(m) for m in guard}
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-input', *packages], capture_output=True, text=True)
    assert r.returncode == 0, f'pip install THAT BAI:\n{(r.stderr or r.stdout)[-3000:]}'
    importlib.invalidate_caches()
    after = {m: _version(m) for m in guard}
    changed = {m: [before[m], after[m]] for m in guard if before[m] != after[m]}
    assert not changed, f'pip DOI thu vien loi cua Colab {changed} -> DUNG: Runtime > Disconnect and delete runtime, roi bao Claude'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import torchmetrics
    assert torchmetrics.__version__ == '0.10.3', (f'torchmetrics trong tien trinh nay = {torchmetrics.__version__} (nap TRUOC khi pip?) -> '
                                                  'Runtime > Restart session, chay lai tu C0')
    conflicts = [s.strip() for s in (r.stderr + '\n' + r.stdout).splitlines() if ' requires ' in s and 'but you have' in s]
    free = shutil.disk_usage(disk).free / 2 ** 30 if os.path.isdir(disk) else None
    assert free is None or free >= min_free_gb, f'o dia {disk} con {free:.0f} GB < {min_free_gb} GB (checkpoint ma ho ghi) -> DUNG'
    return dict(goi=list(packages), loi_giu_nguyen=after, moi={m: _version(m) for m in NEW_PACKAGES}, xung_dot_pip=conflicts[:10],
                dia_trong_gb=free)


CHILD_PYTHON = None       # None -> Python của notebook (86e); 86f: Python của môi trường riêng (use_python)
CHILD_ISOLATED = False    # True -> tiến trình con KHÔNG thừa hưởng PYTHONPATH / site người dùng của Python notebook


def use_python(path, isolated=True):
    '''Mọi tiến trình con (torchrun, kiểm import) chạy bằng `path` (86f: Python 3.9 của môi trường họ); isolated -> PYTHONPATH chỉ repo + gói.'''
    global CHILD_PYTHON, CHILD_ISOLATED
    CHILD_PYTHON, CHILD_ISOLATED = path, bool(isolated)


def _child_python():
    return CHILD_PYTHON or sys.executable


def child_env(repo, package_dir):
    '''Môi trường cho tiến trình con: PYTHONPATH = repo CheXFound + thư mục gói (+ PYTHONPATH cũ nếu KHÔNG cô lập). Môi trường mới (không cô lập):
       bộ cấp phát CUDA expandable_segments (bớt phân mảnh, không đổi số). Môi trường riêng của họ: KHÔNG đặt — torch 2.0 chưa có tuỳ chọn này
       (có từ 2.1) và báo lỗi ngay lần cấp phát CUDA đầu; site người dùng tắt.'''
    env = dict(os.environ)
    inherit = [] if CHILD_ISOLATED else ([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])
    env['PYTHONPATH'] = os.pathsep.join([repo, package_dir] + inherit)
    if CHILD_ISOLATED:
        env['PYTHONNOUSERSITE'] = '1'
        env.pop('PYTHONHOME', None)
        if 'expandable_segments' in env.get('PYTORCH_CUDA_ALLOC_CONF', ''): env.pop('PYTORCH_CUDA_ALLOC_CONF')
    else:
        env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    return env


def check_imports(repo, package_dir, expect_xformers=None):
    '''Import linear_glori.py của họ trong một tiến trình MỚI bằng Python của tiến trình con (như torchrun sẽ làm) + phiên bản thư viện + cờ
       XFORMERS_AVAILABLE ở ba lớp attention / block / swiglu_ffn của họ. expect_xformers True / False -> cổng (86f: nhánh xFormers thật).'''
    code = ('import json, importlib, sys, warnings\n'
            'warnings.simplefilter("ignore")\n'
            'from chexfound.eval.classification import linear_glori\n'
            'import vindr_protocol.upstream.launch\n'
            'from chexfound.layers import attention, block, swiglu_ffn\n'
            'v = {"python": sys.version.split()[0]}\n'
            'for m in ("torch", "torchvision", "torchmetrics", "fvcore", "omegaconf", "peft", "open_clip", "skmultilearn", "transformers", "xformers"):\n'
            '    try: v[m] = getattr(importlib.import_module(m), "__version__", "co")\n'
            '    except Exception: v[m] = None\n'
            'v["xformers_cua_ho"] = {n: bool(getattr(x, "XFORMERS_AVAILABLE", False)) for n, x in (("attention", attention), ("block", block), '
            '("swiglu_ffn", swiglu_ffn))}\n'
            'print("PHIEN_BAN=" + json.dumps(v))\n')
    r = subprocess.run([_child_python(), '-c', code], env=child_env(repo, package_dir), capture_output=True, text=True)
    assert r.returncode == 0 and 'PHIEN_BAN=' in r.stdout, f'import ma CheXFound THAT BAI:\n{scrub(r.stderr[-3000:])}'
    versions = json.loads(r.stdout.split('PHIEN_BAN=')[-1].strip().splitlines()[0])
    assert versions['torchmetrics'] == '0.10.3', f"torchmetrics {versions['torchmetrics']} != 0.10.3 (ban ho ghim) -> DUNG"
    if expect_xformers is not None:
        flags = versions['xformers_cua_ho']
        assert all(f == expect_xformers for f in flags.values()), f'XFORMERS_AVAILABLE trong ma cua ho {flags} != {expect_xformers} -> DUNG'
    return versions


def _platform():
    return 'win-64' if os.name == 'nt' else 'linux-64'


def _env_python(dest):
    return os.path.join(dest, 'python.exe') if os.name == 'nt' else os.path.join(dest, 'bin', 'python')


def _get_micromamba(mm_dir, log=print):
    '''micromamba (một tệp chạy, tải từ micro.mamba.pm) -> đường dẫn. Chỉ lấy đúng tệp chạy trong gói tải về.'''
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
        except TypeError:                                   # Python < 3.10.12 / 3.11.4: extract chưa có filter
            tf.extract(m, mm_dir)
    os.chmod(mm, 0o755)
    log(f'  micromamba: {len(data) / 2 ** 20:.0f} MB tai ve')
    return mm


def _probe_env(python, require_cuda=True):
    '''Trong Python của môi trường: phiên bản, CUDA, GPU, cuDNN; require_cuda + có GPU + xFormers -> chạy memory_efficient_attention trên GPU so với
       attention tường minh (fp32). require_cuda=False (bản kiểm trên máy; 86h — thử môi trường trên GPU chưa được hỗ trợ chính thức, PLAN_86 §15): KHÔNG
       chạy kernel nào ở đây (chỉ hỏi phiên bản / thiết bị) — phép tính GPU thử riêng ở gpu_probe, lỗi không làm hỏng bước dựng. -> dict.'''
    code = (f'RUN_GPU = {bool(require_cuda)}\n'
            'import json, importlib, sys, warnings\n'
            'warnings.simplefilter("ignore")\n'
            'v = {}\n'
            'for m in ("torch", "torchvision", "xformers", "torchmetrics", "numpy", "pandas", "PIL", "scipy", "sklearn", "skimage", "skmultilearn", '
            '"fvcore", "iopath", "omegaconf", "peft", "transformers", "open_clip", "timm", "toolz"):\n'
            '    try: v[m] = getattr(importlib.import_module(m), "__version__", "co")\n'
            '    except Exception: v[m] = None\n'
            'import torch\n'
            'info = dict(phien_ban=v, python=sys.version.split()[0], cuda=torch.version.cuda, cuda_dung_duoc=torch.cuda.is_available(),\n'
            '            gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),\n'
            '            bo_nho_gb=(torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if torch.cuda.is_available() else None),\n'
            '            cudnn=(torch.backends.cudnn.version() if torch.cuda.is_available() else None))\n'
            'if RUN_GPU and torch.cuda.is_available() and v["xformers"]:\n'
            '    from xformers.ops import memory_efficient_attention\n'
            '    g = torch.Generator(device="cuda").manual_seed(0)\n'
            '    q, k, w = (torch.randn(2, 257, 16, 64, device="cuda", generator=g).half() for _ in range(3))\n'
            '    o = memory_efficient_attention(q, k, w).float()\n'
            '    a = torch.softmax(q.float().transpose(1, 2) @ k.float().transpose(1, 2).transpose(-1, -2) / 8.0, -1) @ w.float().transpose(1, 2)\n'
            '    info["xformers_lech"] = float((o - a.transpose(1, 2)).abs().max())\n'
            'print("MT=" + json.dumps(info))\n')
    env = dict(os.environ, PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH', None)
    if 'expandable_segments' in env.get('PYTORCH_CUDA_ALLOC_CONF', ''): env.pop('PYTORCH_CUDA_ALLOC_CONF')
    r = subprocess.run([python, '-c', code], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and 'MT=' in r.stdout, f'kiem moi truong THAT BAI:\n{r.stderr[-3000:]}'
    info = json.loads(r.stdout.split('MT=')[-1].strip().splitlines()[0])
    if require_cuda:
        assert info['cuda_dung_duoc'], f"moi truong cua ho: torch {info['phien_ban']['torch']} KHONG dung duoc GPU {info['gpu']} -> DUNG (GPU phai A100)"
        assert info['cuda'] == '11.7', f"CUDA cua torch = {info['cuda']} != 11.7 -> DUNG"
        assert 'A100' in (info['gpu'] or '') and info['bo_nho_gb'] >= 70, (
            f"GPU {info['gpu']} ({info['bo_nho_gb']:.0f} GB): 86f can A100 80 GB (Runtime > Change runtime type) -> DUNG")
        assert info.get('xformers_lech') is not None and info['xformers_lech'] < 2e-2, f"xFormers tren GPU lech {info.get('xformers_lech')} -> DUNG"
    return info


def _pip_env():
    '''Môi trường cho pip của môi trường riêng: BỎ mọi biến PIP_* / UV_* / PYTHON* của notebook (Colab có thể đặt chỉ mục / ràng buộc riêng), không
       đọc tệp cấu hình pip nào (PIP_CONFIG_FILE = os.devnull), tắt site người dùng. -> (env, [tên biến đã bỏ]).'''
    dropped = sorted(k for k in os.environ if k.startswith(('PIP_', 'UV_', 'PYTHON')))
    env = {k: v for k, v in os.environ.items() if k not in dropped}
    env.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull)
    return env, dropped


def their_pip_command(python, skip=()):
    '''Lệnh pip cài môi trường của họ (nền tảng hiện tại): torch / torchvision theo THEIR_TORCH (+ kho của PyTorch trên Linux), rồi THEIR_PIP (bỏ skip).'''
    torch_pins, index = THEIR_TORCH[_platform()]
    pkgs = [p for p in (*torch_pins, *THEIR_PIP) if p.split('==')[0].split('<')[0] not in skip]
    cmd = [python, '-m', 'pip', 'install', '--no-input', '--disable-pip-version-check', '--progress-bar', 'off']
    if index: cmd += ['--extra-index-url', index]
    return cmd + pkgs, pkgs


def build_their_env(dest=THEIR_ENV_DIR, skip=(), require_cuda=True, log=print):
    '''Môi trường THƯ VIỆN CỦA HỌ (PLAN_86 §13.2): micromamba tạo Python 3.9 (conda-forge) tại dest, pip cài torch / torchvision (THEIR_TORCH) + THEIR_PIP
       (bỏ các gói trong skip — chỉ bản kiểm trên máy CPU: skip=('xformers',), require_cuda=False). Cổng: Python 3.9, đúng phiên bản THEIR_CORE (trừ skip),
       CUDA 11.7 dùng được, A100, xFormers chạy đúng trên GPU. -> dict(python, thong_tin, phut). Gọi lại khi đã dựng: chỉ cài phần thiếu + kiểm.'''
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
    assert r.returncode == 0, (f'pip cai thu vien cua ho THAT BAI (bien bo khoi pip: {dropped}):\n'
                               f'{(r.stdout + chr(10) + r.stderr)[-4000:]}')
    log(f'  pip: {len(pkgs)} goi cai xong ({(time.time() - t0) / 60:.1f} phut tu dau)')
    info = _probe_env(py, require_cuda=require_cuda)
    v = info['phien_ban']
    assert info['python'].startswith(THEIR_PYTHON + '.'), f"Python {info['python']} != {THEIR_PYTHON} -> DUNG"
    bad = {k: v.get(k) for k, want in THEIR_CORE.items() if k not in skip and (v.get(k) or '').split('+')[0] != want}
    assert not bad, f'phien ban KHAC conda-extras.yaml cua ho: {bad} (can {THEIR_CORE}) -> DUNG'
    assert (v.get('numpy') or '').startswith('1.'), f"numpy {v.get('numpy')} (torch 2.0 can numpy 1.x) -> DUNG"
    return dict(python=py, thong_tin=info, goi=pkgs, bien_bo_khoi_pip=dropped, phut=(time.time() - t0) / 60)


def fetch_chexfound_files(kiem_ke, tai, xung_dot, dest_dir=WEIGHTS_DIR, log=print):
    '''teacher_checkpoint.pth + config.yaml từ cxrlt_2024/chexfound (cùng tệp 85 / 86b) qua khối đọc Drive chuẩn; md5 phải = bản công khai đã kiểm.
       -> đường dẫn TÊN CỐ ĐỊNH dest_dir/<tên> (liên kết tới bản tải về; không tạo được liên kết -> chép), không mang ID Drive.'''
    from ..backbones import CHEXFOUND_DRIVE, CHEXFOUND_FILES, CHEXFOUND_MD5, _md5
    kk = kiem_ke(CHEXFOUND_DRIVE)[1]
    os.makedirs(dest_dir, exist_ok=True)
    paths = {}
    for name in CHEXFOUND_FILES:
        versions = kk.get(name)
        assert versions, f"thieu {name} trong cxrlt_2024/{'/'.join(CHEXFOUND_DRIVE)} -> DUNG"
        if xung_dot(versions): log(f"  ! {name}: {len(versions)} ban NOI DUNG KHAC -> dung ban MOI NHAT ({versions[0]['mt']})")
        src = os.path.abspath(tai(versions[0]))
        assert _md5(src) == CHEXFOUND_MD5[name], f'{name}: md5 KHAC ban cong khai da kiem -> DUNG'
        link = os.path.join(dest_dir, name)
        if os.path.lexists(link): os.remove(link)
        try:
            os.symlink(src, link)
        except OSError:
            shutil.copyfile(src, link)
        assert os.path.getsize(link) == os.path.getsize(src), f'{link}: kich thuoc KHAC ban tai ve -> DUNG'
        paths[name] = link
    return paths['teacher_checkpoint.pth'], paths['config.yaml']


_CHECK_LOAD_CODE = '''import json, sys, warnings
warnings.simplefilter("ignore")
import torch
from omegaconf import OmegaConf
from chexfound.configs import dinov2_default_config
from chexfound.models import build_model_from_cfg
from chexfound.layers import attention, swiglu_ffn
cfg_path, ck_path, opts = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
sd = torch.load(ck_path, map_location="cpu")
sd = sd["teacher"] if "teacher" in sd else sd
sd = {k.replace("module.", ""): v for k, v in sd.items()}
sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
out = dict(xformers=dict(attention=bool(attention.XFORMERS_AVAILABLE), swiglu_ffn=bool(swiglu_ffn.XFORMERS_AVAILABLE)))
for name, extra in (("config_phat_hanh", []), ("voi_opts", opts)):
    cfg = OmegaConf.merge(OmegaConf.create(dinov2_default_config), OmegaConf.load(cfg_path), OmegaConf.from_cli(extra))
    model, _ = build_model_from_cfg(cfg, only_teacher=True)
    msg = model.load_state_dict(sd, strict=False)
    out[name] = dict(block_chunks=int(cfg.student.block_chunks), so_khoa_mo_hinh=len(model.state_dict()), thieu=len(msg.missing_keys),
                     thieu_tang=sum(k.startswith("blocks.") for k in msg.missing_keys),
                     thua_ngoai_dau=sorted({k.split(".")[0] for k in msg.unexpected_keys} - {"dino_head", "ibot_head"}),
                     lop_mlp=type(model.blocks[0][-1].mlp if cfg.student.block_chunks else model.blocks[0].mlp).__name__)
    del model
print("NAP=" + json.dumps(out))
'''


def check_weight_loading(repo, config_path, checkpoint_path, opts=OPTS, package_dir=None):
    '''Dựng encoder bằng HÀM CỦA HỌ (build_model_from_cfg trên config gộp như get_cfg_from_args: mặc định + config.yaml + --opts) và nạp như
       load_pretrained_weights của họ (khoá 'teacher', bỏ 'module.' / 'backbone.', strict=False) — hai lần: config phát hành nguyên trạng và có opts.
       Chạy trong TIẾN TRÌNH CON bằng Python của tiến trình con (= nơi mã họ chạy): 86f kiểm đúng các lớp xFormers (MemEffAttention, SwiGLUFFNFused)
       mà torchrun sẽ dựng. -> {ten: dict(block_chunks, thieu, thieu_tang, thua_ngoai_dau, lop_mlp), xformers: {...}}. Không chạy GPU.'''
    r = subprocess.run([_child_python(), '-c', _CHECK_LOAD_CODE, config_path, checkpoint_path, json.dumps(list(opts))],
                       env=child_env(repo, package_dir or repo), capture_output=True, text=True)
    assert r.returncode == 0 and 'NAP=' in r.stdout, f'kiem nap trong so (tien trinh con) THAT BAI:\n{scrub(r.stderr[-3000:])}'
    return json.loads(r.stdout.split('NAP=')[-1].strip().splitlines()[0])


def run_torchrun(repo, package_dir, args, log_path, poll_s=180, log=print):
    '''torchrun --nproc_per_node=1 launch.py <args> (mã của họ); stdout + stderr vào log_path; mỗi poll_s giây in dòng tiến độ cuối của họ.
       -> dict(ma_thoat, phut, tran_bo_nho, duoi_log).'''
    env = child_env(repo, package_dir)
    env.pop('VP_ENCODER', None)
    if ENCODER: env['VP_ENCODER'] = ENCODER                  # 86j: tệp chạy thay mô hình CheXFound bằng lớp bọc này (launch.inject_encoder)
    if USE_TORCHRUN:
        cmd = [_child_python(), '-m', 'torch.distributed.run', '--nproc_per_node=1', '--standalone', LAUNCH, *[str(a) for a in args]]
    else:                    # bản kiểm trên máy (Windows: torchrun --standalone cần libuv): một tiến trình, biến môi trường phân tán đặt tay
        cmd = [_child_python(), LAUNCH, *[str(a) for a in args]]
        env.update(RANK='0', WORLD_SIZE='1', LOCAL_RANK='0', LOCAL_WORLD_SIZE='1', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(29500 + os.getpid() % 500),
                   USE_LIBUV='0')
    t0 = time.time()
    next_report = t0 + poll_s
    with open(log_path, 'w', encoding='utf-8') as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(5)
            if time.time() >= next_report:
                next_report += poll_s
                line = _last_progress(log_path)
                if line: log(f'    {(time.time() - t0) / 60:5.1f} phut | {line}')
    text = open(log_path, encoding='utf-8', errors='replace').read()
    return dict(ma_thoat=proc.returncode, phut=(time.time() - t0) / 60, tran_bo_nho=any(s in text for s in _OOM), duoi_log=scrub(text[-4000:]))


_HEX32 = re.compile(r'(?<![0-9A-Za-z])[0-9a-f]{32}(?![0-9A-Za-z])')


def scrub(text):
    '''Đuôi log IN RA notebook (thông báo lỗi): che chuỗi 32 hex = dạng mã ảnh VinDr (tên tệp ảnh trong đường dẫn; luật PhysioNet: mã ảnh không vào
       chat). Tệp log đầy đủ (chay.log, nhật ký trên Drive) giữ nguyên.'''
    return _HEX32.sub('<ma_anh>', text)


def _last_progress(log_path):
    try:
        lines = open(log_path, encoding='utf-8', errors='replace').read().splitlines()
    except OSError:
        return ''
    for line in reversed(lines[-400:]):
        if re.search(r'(Training|Test:)\s+\[', line):
            return re.sub(r'\s+', ' ', line.split(' - ')[-1] if ' - ' in line else line).strip()[:200]
    return ''


def read_results(output_dir):
    '''Đọc results_eval_linear.json của họ (khối "ITER: <bước> <tách>" + một dòng JSON). -> [dict(buoc, tach, ten, ket_qua)].'''
    path = os.path.join(output_dir, RESULTS_FILE)
    lines = open(path, encoding='utf-8').read().splitlines()
    out = []
    for i, line in enumerate(lines):
        m = re.match(r'^ITER: (\d+) (\w+)\s*$', line)
        if m and i + 1 < len(lines) and lines[i + 1].startswith('{'):
            best = json.loads(lines[i + 1])['best_classifier']
            out.append(dict(buoc=int(m.group(1)), tach=m.group(2), ten=best['name'], ket_qua=best['results']))
    assert out, f'{path}: khong doc duoc khoi ket qua nao -> DUNG'
    return out


def lr_from_name(name):
    '''"linear:blocks=4:avgpool=False:lr=0_0005000000" -> 0.0005 (cách họ đặt tên đầu: lr:.10f, '.' -> '_').'''
    m = re.search(r'lr=(\d+)_(\d+)', name)
    assert m, name
    return float(f'{m.group(1)}.{m.group(2)}')


CHECK_TOL = 5e-5          # |AUROC tính lại trên CPU − số họ tính trên GPU| từ CÙNG preds


def read_evaluations(run_dir):
    '''Các lần chấm launch.record_evaluations đã ghi trong thư mục lượt chạy, theo thứ tự gọi -> [dict(thu_tu, tach, iteration, prefix, ten (tên đầu),
       so_ho_tinh ({tên đầu: {chỉ số: giá trị họ tính}}), npz)].'''
    from .launch import EVAL_DIR
    d = os.path.join(run_dir, EVAL_DIR)
    recs = []
    for f in (sorted(os.listdir(d)) if os.path.isdir(d) else []):
        if f.endswith('.json'):
            with open(os.path.join(d, f), encoding='utf-8') as fh:
                rec = json.load(fh)
            rec['npz'] = os.path.join(d, f[:-len('.json')] + '.npz')
            recs.append(rec)
    assert recs, f'{d}: khong co lan cham nao duoc ghi -> DUNG'
    return sorted(recs, key=lambda r: r['thu_tu'])


def load_evaluation(rec):
    '''-> (target [N, C] int64, {tên đầu: preds [N, C] float32}) — đúng tensor đã vào metric.update của họ.'''
    import numpy as np
    with np.load(rec['npz']) as z:
        return z['target'].astype(np.int64), {name: z[f'preds_{j}'] for j, name in enumerate(rec['ten'])}


def _csv_labels(csv_root, split):
    import pandas as pd
    df = pd.read_csv(os.path.join(csv_root, f'{split}.csv'))
    return df.drop(columns=['Path']).to_numpy().astype('int64')


def check_run(run, csv_root):
    '''Cổng trên MỘT lượt học (đọc lần chấm đã ghi + results_eval_linear.json của họ):
       (1) lần chấm val CUỐI (pha 1): AUROC tính lại (torchmetrics 0.10.3, CPU) từng lr = số họ tính; lr họ chọn = lr có AUROC (số họ tính) lớn nhất,
           hoà -> lr đứng trước (so sánh '>' của họ); kèm: lr có AUROC tính lại lớn nhất có trùng không; target = val.csv;
       (2) lần chấm test CUỐI (pha 2, một đầu = lr chọn): AUROC tính lại = số họ tính = khối test cuối họ ghi; target = test.csv theo thứ tự.
       -> dict(val, test, du_doan_test (preds, target)).'''
    from .report import tm_scores
    recs, blocks = read_evaluations(run['thu_muc']), run['ket_qua']
    out = {}
    for split in ('val', 'test'):
        rs, bs = [r for r in recs if r['tach'] == split], [b for b in blocks if b['tach'] == split]
        assert rs and bs, f"{run['thu_muc']}: thieu lan cham / khoi {split} -> DUNG"
        rec, blk = rs[-1], bs[-1]
        target, preds = load_evaluation(rec)
        assert np_equal(target, _csv_labels(csv_root, split)), f'{split}: target lan cham KHAC nhan {split}.csv (thu tu) -> DUNG'
        ours = {n: tm_scores(p, target)[1] for n, p in preds.items()}
        theirs = {n: rec['so_ho_tinh'][n]['auroc'] for n in rec['ten']}
        best_theirs = max(rec['ten'], key=lambda n: (theirs[n], -rec['ten'].index(n)))
        best_ours = max(rec['ten'], key=lambda n: (ours[n], -rec['ten'].index(n)))
        lech = max(abs(ours[n] - theirs[n]) for n in rec['ten'])
        out[split] = dict(iteration=rec['iteration'], buoc_khoi=blk['buoc'], so_dau=len(rec['ten']), lech_max=float(lech), ten_ho_chon=blk['ten'],
                          auroc_ho_ghi=float(blk['ket_qua']['auroc']), auroc_tinh_lai=float(ours[blk['ten']]) if blk['ten'] in ours else None,
                          chon_dung_luat=bool(blk['ten'] == best_theirs), chon_trung_tinh_lai=bool(best_ours == best_theirs),
                          dat=bool(lech < CHECK_TOL and blk['ten'] == best_theirs and rec['iteration'] == blk['buoc']
                                   and abs(theirs[blk['ten']] - float(blk['ket_qua']['auroc'])) < 1e-6))
        if split == 'test':
            assert len(rec['ten']) == 1, f"lan cham test cuoi co {len(rec['ten'])} dau (pha 2 phai MOT dau) -> DUNG"
            out['du_doan_test'] = (preds[rec['ten'][0]], target)
    return out


def np_equal(a, b):
    import numpy as np
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _dataset_args(csv_root):
    from .vindr_dataset import dataset_str
    return ['--train-dataset', dataset_str('train', csv_root), '--val-dataset', dataset_str('val', csv_root), '--test-dataset', dataset_str('test', csv_root)]


def set_image_size(n):
    '''--image-size cho MỌI lượt chạy sau đó (chạy thử, Linear, GLoRI). 86g: 224 = mặc định của linear_glori.py (PLAN_86 §14.1) — chỉ đổi một
       điều so với 86f. Chỉ nhận IMAGE_SIZES. -> kích thước đã đặt.'''
    global IMAGE_SIZE
    assert int(n) in IMAGE_SIZES, f'kich thuoc anh {n} khong thuoc {IMAGE_SIZES} -> DUNG'
    IMAGE_SIZE = int(n)
    return IMAGE_SIZE


ENCODER = None           # 86j (PLAN_86 §16): tên lớp bọc encoders_ho cho MỌI lượt (biến VP_ENCODER của tiến trình torchrun); None = CheXFound
GLORI_FORCE_HALVES = False   # 86j: GLoRI chạy THẲNG hai nửa lưới 5 + 4 như 86f (cùng cấu trúc chia cho mọi encoder), không thử 9 lr một lượt


def set_encoder(name):
    '''86j: encoder cho MỌI lượt sau đó (chạy thử, Linear, GLoRI) — lớp bọc encoders_ho, thay mô hình CheXFound trong mã họ. -> tên.'''
    global ENCODER
    from .encoders_ho import NAMES
    assert name in NAMES, f'encoder {name!r} khong thuoc {NAMES} -> DUNG'
    ENCODER = name
    return ENCODER


def set_glori_halves(flag=True):
    '''86j: True -> run_head('glori') chạy thẳng GLORI_HALVES (5 + 4) như 86f, kể cả khi 9 lr vừa bộ nhớ (§16.1). -> cờ.'''
    global GLORI_FORCE_HALVES
    GLORI_FORCE_HALVES = bool(flag)
    return GLORI_FORCE_HALVES


EXTRA_ARGS = ()          # tham số thêm cho MỌI lượt (Colab: rỗng = mặc định của họ; bản kiểm trên máy: --num-workers 0)
USE_TORCHRUN = True      # Colab: torchrun --nproc_per_node=1 như README của họ; bản kiểm trên Windows: False


def _common_args(csv_root, checkpoint, config):
    return (['--batch-size', BATCH, '--image-size', IMAGE_SIZE, '--config-file', config, '--pretrained-weights', checkpoint] + _dataset_args(csv_root)
            + list(EXTRA_ARGS))


def _head_args(kind):
    assert kind in ('lin', 'glori'), kind
    return ['--glori', '--cat-cls'] if kind == 'glori' else []


_RUN_DIR_PREFIXES = ('lin_', 'glori_', 'thu_')


def _fresh_dir(out_dir):
    '''Thư mục lượt chạy TRỐNG: mã họ mặc định resume=True — thư mục còn checkpoint của lần trước thì học TIẾP từ đó (không phải lượt mới). Chỉ xoá
       thư mục lượt chạy của bộ điều phối (tên lin_ / glori_ / thu_).'''
    assert os.path.basename(os.path.normpath(out_dir)).startswith(_RUN_DIR_PREFIXES), f'{out_dir}: khong phai thu muc luot chay -> KHONG xoa'
    if os.path.lexists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def train_once(kind, repo, package_dir, csv_root, checkpoint, config, out_dir, lrs=None, extra=(), log=print):
    '''MỘT lượt học của mã họ (pha 1 dò lr trên val + pha 2 học tiếp trên train + val + chấm test của họ) trong thư mục TRỐNG.
       -> dict(thu_muc, chay, ket_qua).'''
    _fresh_dir(out_dir)
    args = _common_args(csv_root, checkpoint, config) + _head_args(kind) + ['--output-dir', out_dir]
    if lrs: args += ['--learning-rates', *[f'{v:g}' for v in lrs]]
    args += list(extra) + ['--opts', *OPTS]
    run = run_torchrun(repo, package_dir, args, os.path.join(out_dir, 'chay.log'), log=log)
    res = read_results(out_dir) if run['ma_thoat'] == 0 else None
    return dict(thu_muc=out_dir, lrs=list(lrs) if lrs else None, chay=run, ket_qua=res)


def phase1_choice(results):
    '''Lựa chọn lr của pha 1 = khối 'val' CUỐI (evaluate_linear_classifiers cuối eval_linear của họ). -> dict(ten, lr, auroc_val).'''
    val = [r for r in results if r['tach'] == 'val']
    assert val, 'khong co khoi val (pha 1) -> DUNG'
    last = val[-1]
    return dict(ten=last['ten'], lr=lr_from_name(last['ten']), auroc_val=float(last['ket_qua']['auroc']), buoc=last['buoc'])


PRED_FILE = 'du_doan_test.npz'      # dự đoán của lần chấm test CUỐI (pha 2) của lượt thắng: preds (sigmoid của họ) + target


def _write_test_predictions(run_dir, preds_target):
    import numpy as np
    path = os.path.join(run_dir, PRED_FILE)
    np.savez_compressed(path, preds=preds_target[0].astype(np.float32), target=preds_target[1].astype(np.int64))
    return path


def run_head(kind, repo, package_dir, csv_root, checkpoint, config, out_root, log=print):
    '''Đầy đủ một loại đầu bằng mã của họ, lưới 9 lr của bài (PAPER_LRS, Linear và GLoRI; Tuấn 12/11), một lượt; GLoRI TRÀN BỘ NHỚ -> hai nửa lưới
       (5 + 4), lr thắng = AUROC val pha 1 cao nhất giữa hai nửa (hoà -> nửa đầu = lr nhỏ hơn, như so sánh '>' của họ). Dự đoán test = lần chấm test
       CUỐI của pha 2 trong lượt thắng (không chạy --infer riêng); cổng check_run trên MỌI lượt (hai nửa: cả hai).'''
    t0 = time.time()
    lrs = PAPER_LRS
    forced = kind == 'glori' and GLORI_FORCE_HALVES
    log(f"== {kind}: hoc bang ma cua ho (luoi {len(lrs)} lr cua bai, lo {BATCH}, anh {IMAGE_SIZE}, 10 + 10 epoch" + (f', encoder {ENCODER}' if ENCODER else '')
        + (', hai nua luoi 5 + 4 BAT BUOC nhu 86f' if forced else '') + ')')
    halves = lambda: [train_once(kind, repo, package_dir, csv_root, checkpoint, config, os.path.join(out_root, f'{kind}_hoc_nua{i + 1}'), half, log=log)
                      for i, half in enumerate(GLORI_HALVES)]
    if forced:
        runs = halves()
    else:
        runs = [train_once(kind, repo, package_dir, csv_root, checkpoint, config, os.path.join(out_root, f'{kind}_hoc'), lrs, log=log)]
        if runs[0]['chay']['ma_thoat'] != 0 and kind == 'glori' and runs[0]['chay']['tran_bo_nho']:
            log('  ! GLoRI 9 lr cung luc TRAN BO NHO -> chay hai nua luoi (PLAN_86 12.3)')
            runs = halves()
    for r in runs:
        assert r['chay']['ma_thoat'] == 0, f"hoc {kind} ({r['thu_muc']}) THAT BAI:\n{r['chay']['duoi_log']}"
    checks = [check_run(r, csv_root) for r in runs]
    for r, c in zip(runs, checks):
        assert c['val']['dat'] and c['test']['dat'], (f"{r['thu_muc']}: cong lan cham KHONG dat (val {c['val']}, test {c['test']}) -> DUNG, gui Claude")
    choices = [phase1_choice(r['ket_qua']) for r in runs]
    k = max(range(len(runs)), key=lambda i: (choices[i]['auroc_val'], -i))
    winner = runs[k]
    test_blocks = [b for b in winner['ket_qua'] if b['tach'] == 'test']
    pred_path = _write_test_predictions(winner['thu_muc'], checks[k]['du_doan_test'])
    return dict(loai=kind, luot=[dict(thu_muc=r['thu_muc'], lrs=r['lrs'], phut=r['chay']['phut'], chon=c, cong={s: x[s] for s in ('val', 'test')})
                                 for r, c, x in zip(runs, choices, checks)],
                thang=k, chon=choices[k], hai_nua=len(runs) > 1, hai_nua_bat_buoc=forced, encoder=ENCODER, anh=IMAGE_SIZE,
                test_ho_ghi_cuoi_pha2=(test_blocks[-1] if test_blocks else None),
                cong={s: checks[k][s] for s in ('val', 'test')}, du_doan=pred_path, phut_tong=(time.time() - t0) / 60)


SMOKE_SIZES = dict(train=512, val=256, test=256)     # lượt chạy thử: n ảnh đầu mỗi tách


def smoke_test(repo, package_dir, csv_root, checkpoint, config, out_root, log=print, kind='lin'):
    '''Chạy thử nhỏ bằng mã của họ (~5 phút): một loại đầu (mặc định Linear; 86h thêm GLoRI — PLAN_86 §15) 2 lr trên bộ CSV nhỏ (SMOKE_SIZES ảnh đầu mỗi
       tách), 1 + 1 epoch -> bắt lỗi môi trường / tệp chạy / đọc kết quả / ghi lần chấm / hai cổng check_run TRƯỚC lượt đầy đủ. Số đo ở đây vô nghĩa
       (không đọc). Không dùng --num-samples (lỗi của họ với torch mới, xem vindr_dataset.write_subset_csvs).'''
    from .vindr_dataset import write_subset_csvs
    small = os.path.join(out_root, 'csv_nho')
    if not os.path.isabs(csv_root): small = os.path.relpath(small)      # giữ kiểu đường dẫn của csv_root (bản kiểm Windows: tương đối)
    sizes = write_subset_csvs(csv_root, small, SMOKE_SIZES)
    run_dir = os.path.join(out_root, 'thu_hoc' if kind == 'lin' else f'thu_hoc_{kind}')
    r = train_once(kind, repo, package_dir, small, checkpoint, config, run_dir, lrs=(1e-4, 1e-3),
                   extra=('--epochs', 1, '--val-epochs', 1, '--eval-period-epochs', 1, '--save-checkpoint-frequency', 1), log=log)
    assert r['chay']['ma_thoat'] == 0, f"chay thu (hoc) THAT BAI:\n{r['chay']['duoi_log']}"
    c = check_run(r, small)
    preds, target = c['du_doan_test']
    ok = (preds.shape == (sizes['test'], len(sizes['class_names'])) and c['val']['dat'] and c['test']['dat'] and c['val']['so_dau'] == 2
          and int(target.sum()) > 0)
    return dict(dat=bool(ok), hinh_du_doan=list(preds.shape), khoi=[(b['buoc'], b['tach']) for b in r['ket_qua']], phut=r['chay']['phut'],
                cong={s: c[s] for s in ('val', 'test')})


# ---- lưu / khôi phục từng loại đầu qua tệp resume của khối ghi cuối (ghi() chỉ local; dong_bo() tải lên MỘT lượt cuối; tệp resume ghi đè) ----
HEAD_FILES = ('du_doan_{label}_{kind}_{t}.npz', 'luot_{label}_{kind}_{t}.json', 'nhat_ky_{label}_{kind}_{t}.tar.gz')


def run_label(run_tag):
    '''Nhãn notebook trong tên tệp kết quả: RUN_TAG 'r86e_…' -> '86e', 'r86f_…' -> '86f' (C6 / C7 dùng chung giữa 86e và 86f).'''
    m = re.match(r'r(86[a-z])_', run_tag)
    assert m, f'RUN_TAG {run_tag!r} khong co dang r86<chu>_ -> DUNG'
    return m.group(1)


def resume_state(path, run_tag, khoi_phuc_cho, log=print):
    '''tim_moi_noi(TEN_RESUME) -> trạng thái dict(run, nhan, t, xong={loại đầu: [tên tệp]}); có tệp thì trả tệp đang chờ về local (khoi_phuc_cho).
       Không có tệp -> trạng thái mới, t = giờ hiện tại (hậu tố tên tệp kết quả của cả lượt).'''
    if not path:
        return dict(run=run_tag, nhan=run_label(run_tag), t=time.strftime('%Y%m%d_%H%M%S'), xong={})
    import torch
    st = torch.load(path, map_location='cpu', weights_only=False)
    assert st.get('run') == run_tag, f"tep resume cua luot KHAC ({st.get('run')} != {run_tag}) -> DOI RUN_TAG o C0, KHONG xoa gi"
    khoi_phuc_cho(st.get('cho'))
    log(f"RESUME: luot {st['t']} da xong {sorted(st['xong']) or '-'} -> khoi phuc, KHONG chay lai")
    return dict(run=st['run'], nhan=run_label(st['run']), t=st['t'], xong={k: list(v) for k, v in st['xong'].items()})


def _write_bytes(path, b):
    with open(path, 'wb') as f:
        f.write(b)


def _write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def save_head(result, kind, t, out_root, ghi, label='86e'):
    '''Kết quả một loại đầu -> 3 tệp qua ghi() (local + hàng chờ): dự đoán test (lần chấm test cuối của pha 2), kết quả (JSON; du_doan = tên tệp),
       nhật ký của họ (log, results_eval_linear.json, config.yaml, mọi lần chấm đã ghi (cham/) của mọi thư mục lượt chạy loại đầu này; KHÔNG .pth).
       result['du_doan'] trỏ sang bản local vừa ghi. -> [tên tệp] (thứ tự HEAD_FILES).'''
    names = [f.format(label=label, kind=kind, t=t) for f in HEAD_FILES]
    with open(result['du_doan'], 'rb') as f:
        pred_bytes = f.read()
    result['du_doan'] = ghi(names[0], lambda q: _write_bytes(q, pred_bytes))
    js = json.loads(json.dumps(result))
    js['du_doan'] = names[0]
    ghi(names[1], lambda q: _write_json(q, js))
    dirs = sorted(d for d in os.listdir(out_root) if d.startswith(f'{kind}_') and os.path.isdir(os.path.join(out_root, d)))

    def _tar(q):
        with tarfile.open(q, 'w:gz') as tf:
            for d in dirs:
                tf.add(os.path.join(out_root, d), arcname=d, filter=lambda ti: None if ti.name.endswith('.pth') else ti)
    ghi(names[2], _tar)
    return names


def restore_head(names, local_dir):
    '''Ngược của save_head trên các tệp khoi_phuc_cho đã trả về local_dir: đọc JSON, du_doan -> đường dẫn local, kiểm dự đoán đọc được.'''
    import numpy as np
    with open(os.path.join(local_dir, names[1]), encoding='utf-8') as f:
        result = json.load(f)
    result['du_doan'] = os.path.join(local_dir, names[0])
    with np.load(result['du_doan']) as z:
        assert z['preds'].ndim == 2 and z['preds'].shape == z['target'].shape and np.isfinite(z['preds']).all(), f'{names[0]}: du doan hong -> DUNG'
    result['khoi_phuc'] = True
    return result


def head_step(kind, state, repo, package_dir, csv_root, checkpoint, config, out_root, local_dir, ghi, save_state, log=print):
    '''Một loại đầu có resume: đã xong trong tệp resume -> khôi phục (không chạy lại); chưa -> run_head, save_head, save_state(state) (ghi đè tệp
       resume, kèm tệp đang chờ).'''
    if kind in state['xong']:
        return restore_head(state['xong'][kind], local_dir)
    result = run_head(kind, repo, package_dir, csv_root, checkpoint, config, out_root, log=log)
    state['xong'][kind] = save_head(result, kind, state['t'], out_root, ghi, label=state.get('nhan') or run_label(state['run']))
    save_state(state)
    return result


def head_summary(result):
    '''Một dòng mô tả lượt của một loại đầu (KHÔNG có số test — số test ở ô chấm).'''
    c, g = result['chon'], result['cong']
    when = 'KHOI PHUC tu tep resume' if result.get('khoi_phuc') else f"{result['phut_tong']:.0f} phut"
    return (f"{result['loai']}: anh {result.get('anh', 512)} | lr chon {c['lr']:g} (AUROC val pha 1 {c['auroc_val']:.4f}, dropout bat)"
            + (' | 2 nua luoi (9 lr cung luc tran bo nho)' if result['hai_nua'] else '') + ' | luot: '
            + ', '.join(f"{os.path.basename(x['thu_muc'])} {x['phut']:.0f} phut" for x in result['luot'])
            + f" | cong lan cham: val {g['val']['so_dau']} lr, AUROC tinh lai - ho tinh lech max {g['val']['lech_max']:.1e}, lr ho chon "
            + ('dung luat' if g['val']['chon_dung_luat'] else 'SAI luat') + f"; test lech {g['test']['lech_max']:.1e} | {when} | so test: o C8")
