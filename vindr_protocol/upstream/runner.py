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
PAPER_GLORI_LRS = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3)   # §IV-A.2 của bài (GLoRI)
GLORI_HALVES = (PAPER_GLORI_LRS[:5], PAPER_GLORI_LRS[5:])        # khi 9 lr cùng lúc tràn bộ nhớ (PLAN_86 §12.3)
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


def child_env(repo, package_dir):
    '''Môi trường cho tiến trình con: PYTHONPATH = repo CheXFound + thư mục gói; bộ cấp phát CUDA expandable_segments (bớt phân mảnh, không đổi số).'''
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([repo, package_dir] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    return env


def check_imports(repo, package_dir):
    '''Import linear_glori.py của họ trong một tiến trình MỚI (như torchrun sẽ làm) + phiên bản thư viện. Lỗi -> AssertionError kèm stderr.'''
    code = ('import json, importlib, warnings\n'
            'warnings.simplefilter("ignore")\n'
            'from chexfound.eval.classification import linear_glori\n'
            'import vindr_protocol.upstream.launch\n'
            'v = {}\n'
            'for m in ("torch", "torchvision", "torchmetrics", "fvcore", "omegaconf", "peft", "open_clip", "skmultilearn", "transformers", "xformers"):\n'
            '    try: v[m] = getattr(importlib.import_module(m), "__version__", "co")\n'
            '    except Exception: v[m] = None\n'
            'print("PHIEN_BAN=" + json.dumps(v))\n')
    r = subprocess.run([sys.executable, '-c', code], env=child_env(repo, package_dir), capture_output=True, text=True)
    assert r.returncode == 0 and 'PHIEN_BAN=' in r.stdout, f'import ma CheXFound THAT BAI:\n{r.stderr[-3000:]}'
    versions = json.loads(r.stdout.split('PHIEN_BAN=')[-1].strip().splitlines()[0])
    assert versions['torchmetrics'] == '0.10.3', f"torchmetrics {versions['torchmetrics']} != 0.10.3 (ban ho ghim) -> DUNG"
    return versions


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


def check_weight_loading(repo, config_path, checkpoint_path, opts=OPTS):
    '''Dựng encoder bằng HÀM CỦA HỌ (build_model_from_cfg trên config gộp như get_cfg_from_args: mặc định + config.yaml + --opts) và nạp như
       load_pretrained_weights của họ (khoá 'teacher', bỏ 'module.' / 'backbone.', strict=False) — hai lần: config phát hành nguyên trạng và có opts.
       -> {ten: dict(block_chunks, thieu, thieu_tang, thua_ngoai_dau)}. Không chạy GPU.'''
    import warnings

    import torch
    if repo not in sys.path: sys.path.insert(0, repo)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from omegaconf import OmegaConf

        from chexfound.configs import dinov2_default_config
        from chexfound.models import build_model_from_cfg
    sd = torch.load(checkpoint_path, map_location='cpu')
    sd = sd['teacher'] if 'teacher' in sd else sd
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    sd = {k.replace('backbone.', ''): v for k, v in sd.items()}
    out = {}
    for name, extra in (('config_phat_hanh', []), ('voi_opts', list(opts))):
        cfg = OmegaConf.merge(OmegaConf.create(dinov2_default_config), OmegaConf.load(config_path), OmegaConf.from_cli(extra))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model, _ = build_model_from_cfg(cfg, only_teacher=True)
        msg = model.load_state_dict(sd, strict=False)
        out[name] = dict(block_chunks=int(cfg.student.block_chunks), so_khoa_mo_hinh=len(model.state_dict()), thieu=len(msg.missing_keys),
                         thieu_tang=sum(k.startswith('blocks.') for k in msg.missing_keys),
                         thua_ngoai_dau=sorted({k.split('.')[0] for k in msg.unexpected_keys} - {'dino_head', 'ibot_head'}))
        del model
    return out


def run_torchrun(repo, package_dir, args, log_path, poll_s=180, log=print):
    '''torchrun --nproc_per_node=1 launch.py <args> (mã của họ); stdout + stderr vào log_path; mỗi poll_s giây in dòng tiến độ cuối của họ.
       -> dict(ma_thoat, phut, tran_bo_nho, duoi_log).'''
    env = child_env(repo, package_dir)
    if USE_TORCHRUN:
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=1', '--standalone', LAUNCH, *[str(a) for a in args]]
    else:                    # bản kiểm trên máy (Windows: torchrun --standalone cần libuv): một tiến trình, biến môi trường phân tán đặt tay
        cmd = [sys.executable, LAUNCH, *[str(a) for a in args]]
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


def parse_infer_metrics(log_text):
    '''Dòng "-- Classifier: <tên> * {'ap': tensor(x...), 'auroc': tensor(y...)}" mà evaluate_linear_classifiers_preds của họ ghi ở --infer.'''
    m = re.search(r"-- Classifier: (\S+) \* \{'ap': tensor\(([-0-9.eE+]+)[^}]*'auroc': tensor\(([-0-9.eE+]+)", log_text)
    assert m, 'khong thay dong chi so --infer cua ho trong log -> DUNG'
    return dict(ten=m.group(1), ap=float(m.group(2)), auroc=float(m.group(3)))


def _dataset_args(csv_root):
    from .vindr_dataset import dataset_str
    return ['--train-dataset', dataset_str('train', csv_root), '--val-dataset', dataset_str('val', csv_root), '--test-dataset', dataset_str('test', csv_root)]


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


def infer(kind, repo, package_dir, csv_root, checkpoint, config, winner_dir, out_dir, log=print):
    '''--infer của họ trên test với đầu pha 2 của lượt thắng (optimal/model_final.pth); --val-metric-type multilabel_ap -> họ in AP + AUROC;
       tệp chạy lưu preds / target. -> dict(chi_so_ho_in, du_doan (đường dẫn), chay).'''
    import numpy as np

    from .launch import PRED_FILE
    _fresh_dir(out_dir)
    args = (['--infer','--classifier-fpath', os.path.join(winner_dir, 'optimal', 'model_final.pth'), '--val-metric-type', 'multilabel_ap',
             '--output-dir', out_dir] + _common_args(csv_root, checkpoint, config) + _head_args(kind) + ['--opts', *OPTS])
    run = run_torchrun(repo, package_dir, args, os.path.join(out_dir, 'chay.log'), log=log)
    assert run['ma_thoat'] == 0, f"--infer {kind} THAT BAI:\n{run['duoi_log']}"
    text = open(os.path.join(out_dir, 'chay.log'), encoding='utf-8', errors='replace').read()
    pred_path = os.path.join(out_dir, PRED_FILE)
    with np.load(pred_path) as z:
        assert z['preds'].ndim == 2 and np.isfinite(z['preds']).all(), 'du doan --infer sai hinh / khong huu han -> DUNG'
    return dict(chi_so_ho_in=parse_infer_metrics(text), du_doan=pred_path, chay=dict(ma_thoat=run['ma_thoat'], phut=run['phut']))


def run_head(kind, repo, package_dir, csv_root, checkpoint, config, out_root, log=print):
    '''Đầy đủ một loại đầu bằng mã của họ. Linear: lưới mặc định 13 lr, một lượt. GLoRI: lưới 9 lr của bài, một lượt; TRÀN BỘ NHỚ -> hai nửa lưới
       (5 + 4), lr thắng = AUROC val pha 1 cao nhất giữa hai nửa (hoà -> nửa đầu = lr nhỏ hơn, như so sánh '>' của họ). Rồi --infer với lượt thắng.'''
    t0 = time.time()
    lrs = None if kind == 'lin' else PAPER_GLORI_LRS
    log(f"== {kind}: hoc bang ma cua ho ({'luoi mac dinh 13 lr' if lrs is None else 'luoi 9 lr cua bai'}, lo {BATCH}, anh {IMAGE_SIZE}, 10 + 10 epoch)")
    runs = [train_once(kind, repo, package_dir, csv_root, checkpoint, config, os.path.join(out_root, f'{kind}_hoc'), lrs, log=log)]
    if runs[0]['chay']['ma_thoat'] != 0 and kind == 'glori' and runs[0]['chay']['tran_bo_nho']:
        log('  ! GLoRI 9 lr cung luc TRAN BO NHO -> chay hai nua luoi (PLAN_86 12.3)')
        runs = [train_once(kind, repo, package_dir, csv_root, checkpoint, config, os.path.join(out_root, f'{kind}_hoc_nua{i + 1}'), half, log=log)
                for i, half in enumerate(GLORI_HALVES)]
    for r in runs:
        assert r['chay']['ma_thoat'] == 0, f"hoc {kind} ({r['thu_muc']}) THAT BAI:\n{r['chay']['duoi_log']}"
    choices = [phase1_choice(r['ket_qua']) for r in runs]
    k = max(range(len(runs)), key=lambda i: (choices[i]['auroc_val'], -i))
    winner = runs[k]
    test_blocks = [b for b in winner['ket_qua'] if b['tach'] == 'test']
    inf = infer(kind, repo, package_dir, csv_root, checkpoint, config, winner['thu_muc'], os.path.join(out_root, f'{kind}_infer'), log=log)
    return dict(loai=kind, luot=[dict(thu_muc=r['thu_muc'], lrs=r['lrs'], phut=r['chay']['phut'], chon=c) for r, c in zip(runs, choices)],
                thang=k, chon=choices[k], hai_nua=len(runs) > 1, test_ho_ghi_cuoi_pha2=(test_blocks[-1] if test_blocks else None), infer=inf,
                phut_tong=(time.time() - t0) / 60)


SMOKE_SIZES = dict(train=512, val=256, test=256)     # lượt chạy thử: n ảnh đầu mỗi tách


def smoke_test(repo, package_dir, csv_root, checkpoint, config, out_root, log=print):
    '''Chạy thử nhỏ bằng mã của họ (~5 phút): Linear 1 lr trên bộ CSV nhỏ (SMOKE_SIZES ảnh đầu mỗi tách), 1 + 1 epoch, rồi --infer -> bắt lỗi môi
       trường / tệp chạy / đọc kết quả / lưu dự đoán TRƯỚC lượt đầy đủ. Số đo ở đây vô nghĩa (không đọc). Không dùng --num-samples (lỗi của họ với
       torch mới, xem vindr_dataset.write_subset_csvs).'''
    import numpy as np

    from .vindr_dataset import write_subset_csvs
    small = os.path.join(out_root, 'csv_nho')
    if not os.path.isabs(csv_root): small = os.path.relpath(small)      # giữ kiểu đường dẫn của csv_root (bản kiểm Windows: tương đối)
    sizes = write_subset_csvs(csv_root, small, SMOKE_SIZES)
    r = train_once('lin', repo, package_dir, small, checkpoint, config, os.path.join(out_root, 'thu_hoc'), lrs=(1e-3,),
                   extra=('--epochs', 1, '--val-epochs', 1, '--eval-period-epochs', 1, '--save-checkpoint-frequency', 1), log=log)
    assert r['chay']['ma_thoat'] == 0, f"chay thu (hoc) THAT BAI:\n{r['chay']['duoi_log']}"
    inf = infer('lin', repo, package_dir, small, checkpoint, config, r['thu_muc'], os.path.join(out_root, 'thu_infer'), log=log)
    with np.load(inf['du_doan']) as z:
        shape, target = z['preds'].shape, z['target']
    ok = (shape == (sizes['test'], len(sizes['class_names'])) and {b['tach'] for b in r['ket_qua']} >= {'val', 'test'}
          and int(target.sum()) > 0)
    return dict(dat=bool(ok), hinh_du_doan=list(shape), khoi=[(b['buoc'], b['tach']) for b in r['ket_qua']], phut=r['chay']['phut'] + inf['chay']['phut'])


# ---- lưu / khôi phục từng loại đầu qua tệp resume của khối ghi cuối (ghi() chỉ local; dong_bo() tải lên MỘT lượt cuối; tệp resume ghi đè) ----
HEAD_FILES = ('du_doan_86e_{kind}_{t}.npz', 'luot_86e_{kind}_{t}.json', 'nhat_ky_86e_{kind}_{t}.tar.gz')


def resume_state(path, run_tag, khoi_phuc_cho, log=print):
    '''tim_moi_noi(TEN_RESUME) -> trạng thái dict(run, t, xong={loại đầu: [tên tệp]}); có tệp thì trả tệp đang chờ về local (khoi_phuc_cho).
       Không có tệp -> trạng thái mới, t = giờ hiện tại (hậu tố tên tệp kết quả của cả lượt).'''
    if not path:
        return dict(run=run_tag, t=time.strftime('%Y%m%d_%H%M%S'), xong={})
    import torch
    st = torch.load(path, map_location='cpu', weights_only=False)
    assert st.get('run') == run_tag, f"tep resume cua luot KHAC ({st.get('run')} != {run_tag}) -> DOI RUN_TAG o C0, KHONG xoa gi"
    khoi_phuc_cho(st.get('cho'))
    log(f"RESUME: luot {st['t']} da xong {sorted(st['xong']) or '-'} -> khoi phuc, KHONG chay lai")
    return dict(run=st['run'], t=st['t'], xong={k: list(v) for k, v in st['xong'].items()})


def _write_bytes(path, b):
    with open(path, 'wb') as f:
        f.write(b)


def _write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def save_head(result, kind, t, out_root, ghi):
    '''Kết quả một loại đầu -> 3 tệp qua ghi() (local + hàng chờ): dự đoán test (.npz của --infer), kết quả (JSON; du_doan = tên tệp), nhật ký của
       họ (log, results_eval_linear.json, config.yaml của mọi thư mục lượt chạy loại đầu này; KHÔNG .pth). result['infer']['du_doan'] trỏ sang bản
       local vừa ghi. -> [tên tệp] (thứ tự HEAD_FILES).'''
    names = [f.format(kind=kind, t=t) for f in HEAD_FILES]
    with open(result['infer']['du_doan'], 'rb') as f:
        pred_bytes = f.read()
    result['infer']['du_doan'] = ghi(names[0], lambda q: _write_bytes(q, pred_bytes))
    js = json.loads(json.dumps(result))
    js['infer']['du_doan'] = names[0]
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
    result['infer']['du_doan'] = os.path.join(local_dir, names[0])
    with np.load(result['infer']['du_doan']) as z:
        assert z['preds'].ndim == 2 and z['preds'].shape == z['target'].shape and np.isfinite(z['preds']).all(), f'{names[0]}: du doan hong -> DUNG'
    result['khoi_phuc'] = True
    return result


def head_step(kind, state, repo, package_dir, csv_root, checkpoint, config, out_root, local_dir, ghi, save_state, log=print):
    '''Một loại đầu có resume: đã xong trong tệp resume -> khôi phục (không chạy lại); chưa -> run_head, save_head, save_state(state) (ghi đè tệp
       resume, kèm tệp đang chờ).'''
    if kind in state['xong']:
        return restore_head(state['xong'][kind], local_dir)
    result = run_head(kind, repo, package_dir, csv_root, checkpoint, config, out_root, log=log)
    state['xong'][kind] = save_head(result, kind, state['t'], out_root, ghi)
    save_state(state)
    return result


def head_summary(result):
    '''Một dòng mô tả lượt của một loại đầu (KHÔNG có số test — số test ở ô chấm).'''
    c = result['chon']
    when = 'KHOI PHUC tu tep resume' if result.get('khoi_phuc') else f"{result['phut_tong']:.0f} phut"
    return (f"{result['loai']}: lr chon {c['lr']:g} (AUROC val pha 1 {c['auroc_val']:.4f}, dropout bat)"
            + (' | 2 nua luoi (9 lr cung luc tran bo nho)' if result['hai_nua'] else '') + ' | luot: '
            + ', '.join(f"{os.path.basename(x['thu_muc'])} {x['phut']:.0f} phut" for x in result['luot']) + f' | {when} | so test: o C8')
