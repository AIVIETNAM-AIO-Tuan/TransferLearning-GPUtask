# -*- coding: utf-8 -*-
"""Điều phối `88c` Shenzhen (PLAN_88 §4.3): chạy linear_glori.py của CheXFound @ 7e288d0 NGUYÊN VĂN qua torchrun bằng hạ tầng `86e`
(vindr_protocol.upstream.runner: clone / kiểm repo, cài thư viện, kiểm nạp trọng số, môi trường thư viện của họ (86f), torchrun + theo dõi tiến độ /
tràn bộ nhớ, đọc results_eval_linear.json, đọc lần chấm đã chép) với tệp chạy riêng shenzhen_launch.py (tạm đặt vào runner.LAUNCH — biến cấp
module mà bộ kiểm 86e cũng đặt — trả lại khi xong; không sửa gói `86`).

Công thức (config.SHENZHEN_RECIPE): script run_eval_shenzhen(_glori).sh của họ với 1 GPU x lô 256 (loss `sum` giữ nguyên), 9 lr của bài qua
--learning-rates, --opts student.block_chunks=4. Linear và GLoRI là HAI lượt riêng (mỗi lượt mã họ đặt seed 0, khởi tạo tuần tự 9 đầu). Dự đoán test =
lần chấm test CUỐI của pha 2 trong lượt (không --infer). Tràn bộ nhớ -> DỪNG và báo (PLAN_88 §4.3 (8): không tự chia lưới).

check_run — cổng trên một lượt (đọc lại mọi tệp lượt chạy ghi; không chạy forward nào):
  lịch chấm (tách, bước) của bộ ghi VÀ của tệp kết quả họ = lịch suy từ công thức (pha 1 val, pha 2 test); lần chấm val: đủ 9 đầu đúng thứ tự lưới,
  target = val.csv; lần chấm test: một đầu = lr họ chọn, target = test.csv; AUROC torchmetrics 0.10.3 tính lại từ dự đoán đã chép = số họ tính
  (< CHECK_TOL); trạng thái BinaryAUROC sau lần chấm = dự đoán đã chép (từng số); lr họ chọn = AUROC val lần cuối lớn nhất, hoà -> lr đứng trước
  (so '>' của họ); nạp checkpoint: pha 1 không nạp gì, pha 2 nạp model_final.pth của pha 1 đủ khoá, md5 trọng số vừa nạp = đầu đã chọn trong
  model_final.pth pha 1; dtype: autocast của mã = fp16, Linear trong encoder ra fp16 khi có GPU (đặc trưng ra qua LayerNorm -> fp32, chỉ ghi), đầu
  ra đầu đọc fp32; đầu đọc ở train() khi chấm (hành vi của họ); có dấu tệp chạy 88c trong log; có bản ghi bộ nhớ cuối lượt; dùng chung đầu vào GLoRI
  đúng như lượt yêu cầu (tắt: mã nguyên văn; bật: mỗi bước học tính MỘT lần cho mọi đầu — số đếm khớp số bước suy từ công thức). Grad NaN chỉ báo.

Đo cho `88a` (PLAN_88 §14.17 — ĐO, không phải lượt kết quả): memory_probe (bộ nhớ đỉnh / tràn của một cấu hình đầu, 1 + 1 epoch, công thức đủ trừ số
epoch), compare_runs (hai lượt: mọi dự đoán ở mọi lần chấm trùng từng số? lệch lớn nhất), log_timing (thời gian / bước và thời gian chờ dữ liệu
trong log của họ)."""
import contextlib
import json
import math
import os
import re
import time

import numpy as np

from . import config, splits
from .shenzhen_launch import ENV_FILE, EVAL88_DIR, LOAD_FILE, MARKER, MEM_FILE, NAN_FILE, SHARE_ENV, SHARE_FILE, state_md5

LAUNCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shenzhen_launch.py')
RUN_DIRS = {'lin': 'lin_shenzhen', 'glori': 'glori_shenzhen'}        # tiền tố lin_ / glori_ / thu_: runner._fresh_dir chỉ xoá thư mục loại này
SMOKE_DIR = 'thu_shenzhen'
SMOKE_LRS = (1e-4, 1e-3)
SMOKE_RECIPE = {'--val-epochs': 1, '--epochs': 1, '--eval-period-epochs': 1, '--save-checkpoint-frequency': 1}
PRED_FILE = 'du_doan_test_shenzhen.npz'
WORLD = 1                                   # 1 GPU (torchrun --nproc_per_node=1)


def _runner():
    from vindr_protocol.upstream import runner
    return runner


def child_paths():
    '''PYTHONPATH thêm cho tiến trình con (ngoài repo CheXFound): thư mục chứa external_benchmarks + thư mục chứa vindr_protocol.'''
    import vindr_protocol
    p88 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p86 = os.path.dirname(os.path.dirname(os.path.abspath(vindr_protocol.__file__)))
    return os.pathsep.join(dict.fromkeys([p88, p86]))


@contextlib.contextmanager
def launch_file(path=None):
    '''runner.run_torchrun chạy tệp runner.LAUNCH -> tạm đặt tệp chạy của 88c (mặc định LAUNCH của module này, tra lúc gọi: bản kiểm CPU đặt lớp vá),
       trả lại khi xong (kể cả khi lỗi).'''
    r = _runner()
    old = r.LAUNCH
    r.LAUNCH = path or LAUNCH
    try:
        yield
    finally:
        r.LAUNCH = old


def check_source(repo):
    '''md5 (bỏ CR) các tệp source lớp bọc dựa vào = bản đã đọc (config.CHEXFOUND_SOURCE_MD5). -> {tệp: md5}; khác -> DỪNG.'''
    r = _runner()
    got = {f: r.md5_lf(os.path.join(repo, *f.split('/'))) for f in config.CHEXFOUND_SOURCE_MD5}
    bad = {f: m for f, m in got.items() if m != config.CHEXFOUND_SOURCE_MD5[f]}
    assert not bad, f'tep source CheXFound KHAC ban da doc: {bad} -> DUNG'
    return got


def check_child_imports(repo):
    '''Tiến trình con (Python + PYTHONPATH như torchrun sẽ dùng) import được lớp ShenzhenL, tệp chạy 88c, linear_glori của họ, bộ ghi 86e. -> phiên bản.'''
    import subprocess
    r = _runner()
    code = ('import json, sys, warnings\nwarnings.simplefilter("ignore")\n'
            'from external_benchmarks import shenzhen_dataset, shenzhen_launch, __version__\n'
            'from chexfound.eval.classification import linear_glori\nfrom vindr_protocol.upstream import launch\nimport torch, torchmetrics\n'
            'print("NHAP=" + json.dumps(dict(goi88=__version__, lop=shenzhen_dataset.ShenzhenL.__mro__[1].__module__, python=sys.version.split()[0], '
            'torch=torch.__version__, torchmetrics=torchmetrics.__version__)))\n')
    p = subprocess.run([r._child_python(), '-c', code], env=r.child_env(repo, child_paths()), capture_output=True, text=True)
    assert p.returncode == 0 and 'NHAP=' in p.stdout, f'tien trinh con KHONG import duoc ShenzhenL / tep chay 88c:\n{p.stderr[-3000:]}'
    v = json.loads(p.stdout.split('NHAP=')[-1].strip().splitlines()[0])
    assert v['lop'] == 'chexfound.data.datasets.shenzhen' and v['torchmetrics'] == '0.10.3', f'tien trinh con: {v} -> DUNG'
    return v


def recipe(overrides=None):
    rec = dict(config.SHENZHEN_RECIPE)
    for k, v in (overrides or {}).items():
        assert k in rec, f'{k}: khong co trong cong thuc Shenzhen'
        rec[k] = v
    return rec


def run_args(kind, csv_root, checkpoint, config_path, out_dir, lrs=config.PAPER_LRS, overrides=None, extra=()):
    '''Tham số dòng lệnh = script của họ (thứ tự không quan trọng với argparse) + 9 lr + EXTRA_ARGS của runner + --opts (cuối: nargs='+').'''
    assert kind in RUN_DIRS, kind
    r = _runner()
    args = [x for kv in recipe(overrides).items() for x in kv]
    args += ['--config-file', config_path, '--pretrained-weights', checkpoint, '--output-dir', out_dir]
    for s in splits.SHENZHEN_SPLIT_NAMES:
        args += [f'--{s}-dataset', splits.dataset_str(s, csv_root)]
    if kind == 'glori':
        args += ['--glori', '--cat-cls']
    args += ['--learning-rates', *[f'{v:g}' for v in lrs]] + list(r.EXTRA_ARGS) + list(extra) + ['--opts', *r.OPTS]
    return [str(a) for a in args]


def head_name(lr, blocks=4, avgpool=False):
    '''Tên đầu mã họ đặt (setup_linear_classifiers / setup_glori; cả GLoRI cũng tiền tố "linear:").'''
    return f'linear:blocks={blocks}:avgpool={avgpool}:lr={lr:.10f}'.replace('.', '_')


def expected_schedule(n_train, n_val, batch, val_epochs, epochs, eval_period_epochs, world=WORLD):
    '''Lịch chấm của eval_linear (linear_glori.py:470–496) -> [(tách, bước)]. Lô = min(batch, n_train) (dòng 556, dùng cả hai pha); pha 1: epoch_length
       = ceil(n_train / (lô x world)), max_iter = epoch_length x val_epochs, chấm val ở bước i nếu (i + 1) % (eval_period_epochs x epoch_length) == 0 và
       i != max_iter − 1, rồi lần cuối ở bước max_iter; pha 2: như thế trên n_train + n_val, epochs, chấm test.'''
    b = min(batch, n_train)
    out = []
    for split, n, ep in (('val', n_train, val_epochs), ('test', n_train + n_val, epochs)):
        el = math.ceil(n / (b * world))
        period, max_iter = eval_period_epochs * el, el * ep
        out += [(split, i) for i in range(max_iter) if period > 0 and (i + 1) % period == 0 and i != max_iter - 1]
        out.append((split, max_iter))
    return out


def tm_binary_auroc(preds, target):
    '''AUROC = BinaryAUROC của torchmetrics 0.10.3 (build_binary_auroc_metric của họ), CPU.'''
    import warnings
    import torch
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import torchmetrics
        from torchmetrics.classification import BinaryAUROC
    assert torchmetrics.__version__ == '0.10.3', f'torchmetrics {torchmetrics.__version__} != 0.10.3 (ban CheXFound ghim) -> DUNG'
    return float(BinaryAUROC()(torch.as_tensor(np.asarray(preds), dtype=torch.float32), torch.as_tensor(np.asarray(target), dtype=torch.long)))


@contextlib.contextmanager
def _env_flag(name, on):
    '''Đặt / bỏ biến môi trường `name` trong lúc chạy (tiến trình con thừa hưởng qua runner.child_env); trả lại giá trị cũ khi xong (kể cả khi lỗi).'''
    old = os.environ.get(name)
    if on:
        os.environ[name] = '1'
    else:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def train_once(kind, repo, csv_root, checkpoint, config_path, out_dir, lrs=config.PAPER_LRS, overrides=None, shared_input=False, log=print):
    '''MỘT lượt mã họ (pha 1 dò lr trên val + pha 2 học tiếp trên train + val, chấm test định kỳ) trong thư mục TRỐNG. shared_input (chỉ GLoRI):
       bật lớp bọc dùng chung đầu vào (shenzhen_launch.share_glori_input). -> dict.'''
    assert not shared_input or kind == 'glori', 'dung chung dau vao chi danh cho GLoRI -> DUNG'
    r = _runner()
    r._fresh_dir(out_dir)
    args = run_args(kind, csv_root, checkpoint, config_path, out_dir, lrs, overrides)
    log_path = os.path.join(out_dir, 'chay.log')
    with launch_file(), _env_flag(SHARE_ENV, shared_input):
        run = r.run_torchrun(repo, child_paths(), args, log_path, log=log)
    with open(log_path, encoding='utf-8', errors='replace') as f:
        run['dau_88c'] = MARKER in f.read()
    res = r.read_results(out_dir) if run['ma_thoat'] == 0 else None
    return dict(thu_muc=out_dir, loai=kind, lrs=list(lrs), recipe=recipe(overrides), args=args, chay=run, ket_qua=res, dung_chung=bool(shared_input))


def _read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def check_run(run, csv_root):
    '''Cổng trên MỘT lượt (mô tả ở đầu tệp). -> dict(dat, lich, val, test, nap, dtype, moi_truong, nan_grad, du_doan_test, ten_anh_test, loi).'''
    import torch
    r = _runner()
    d, rec, lrs = run['thu_muc'], run['recipe'], run['lrs']
    recs, blocks = r.read_evaluations(d), run['ket_qua']
    n_tr = len(splits.read_shenzhen_csv(csv_root, 'train')[1])
    _, y_v = splits.read_shenzhen_csv(csv_root, 'val')
    names_t, y_t = splits.read_shenzhen_csv(csv_root, 'test')
    loi = []

    def gate(name, ok):
        if not ok:
            loi.append(name)
        return bool(ok)

    want = expected_schedule(n_tr, len(y_v), int(rec['--batch-size']), int(rec['--val-epochs']), int(rec['--epochs']), int(rec['--eval-period-epochs']))
    seen, seen_blk = [(x['tach'], x['iteration']) for x in recs], [(b['tach'], b['buoc']) for b in blocks]
    lich = dict(mong=want, ban_ghi=seen, tep_ket_qua=seen_blk, dung=gate('lich cham', seen == want and seen_blk == want))
    gate('dau tep chay 88c trong log', run['chay'].get('dau_88c'))
    env = _read_json(os.path.join(d, ENV_FILE), {})
    gate('autocast cua ma = fp16', env.get('autocast_dtype') == 'torch.float16')
    mem = _read_json(os.path.join(d, MEM_FILE), None)
    gate('co ban ghi bo nho cuoi luot (GPU: co ten GPU)', mem is not None and (mem.get('gpu') is not None) == bool(env.get('gpu')))
    share = _read_json(os.path.join(d, SHARE_FILE), None)
    n_p1, n_p2 = max(i for s, i in want if s == 'val'), max(i for s, i in want if s == 'test')    # số bước học mỗi pha = lần chấm cuối
    if run.get('dung_chung'):
        gate('dung chung dau vao GLoRI: bat; moi buoc hoc tinh MOT lan cho moi dau (so dem = so buoc suy tu cong thuc)',
             share is not None and share.get('bat') is True and env.get('glori_dung_chung_dau_vao') is True
             and share.get('forward') == share.get('goi_goc') == n_p1 + n_p2 and share.get('dung_lai') == (len(lrs) - 1) * n_p1)
    else:
        gate('khong dung chung dau vao GLoRI (ma nguyen van)',
             share is not None and share.get('bat') is False and env.get('glori_dung_chung_dau_vao') is False)
    want_enc = env.get('autocast_dtype') if env.get('gpu') else 'torch.float32'     # Linear trong encoder; CPU: autocast cuda tự tắt (chỉ bản kiểm)
    names_lr = [head_name(v) for v in lrs]
    val_blk = [b for b in blocks if b['tach'] == 'val']
    test_blk = [b for b in blocks if b['tach'] == 'test']
    chosen = val_blk[-1]['ten'] if val_blk else None
    per, last = [], {}
    for x in recs:
        target, preds = r.load_evaluation(x)
        stem = os.path.basename(x['npz'])[:-len('.npz')]
        y = y_v if x['tach'] == 'val' else y_t
        ours = {n: tm_binary_auroc(p, target) for n, p in preds.items()}
        theirs = {n: float(x['so_ho_tinh'][n]['auroc']) for n in x['ten']}
        mpath = os.path.join(d, EVAL88_DIR, stem + '_metric.npz')
        probe = _read_json(os.path.join(d, EVAL88_DIR, stem + '.json'), {})
        if os.path.exists(mpath):
            with np.load(mpath) as z:
                state_ok = (np.array_equal(z['target'], target.reshape(-1))
                            and all(np.array_equal(z[f'preds_{j}'], preds[n].reshape(-1)) for j, n in enumerate(x['ten'])))
        else:
            state_ok = False
        do = probe.get('do', {})
        item = dict(tach=x['tach'], iteration=x['iteration'], so_dau=len(x['ten']), so_anh=int(target.shape[0]),
                    ten_dung=(x['ten'] == names_lr) if x['tach'] == 'val' else (x['ten'] == [chosen]),
                    target_dung=bool(target.ndim == 1 and np.array_equal(target, y)), lech_max=max(abs(ours[n] - theirs[n]) for n in x['ten']),
                    trang_thai_metric=bool(state_ok), linear_ma_hoa=probe.get('ma_hoa', {}).get('linear_ma_hoa'),
                    dac_trung=sorted({str(v.get('dac_trung')) for v in do.values()}) if do else None,
                    dtype_dung=bool(do) and probe.get('ma_hoa', {}).get('linear_ma_hoa') == want_enc
                    and all(v.get('du_doan') == 'torch.float32' for v in do.values()),
                    dau_train=bool(do) and all(v.get('dau_train') is True for v in do.values()))
        per.append(item)
        last[x['tach']] = (x, ours, theirs, preds, target)
    for k in ('ten_dung', 'target_dung', 'trang_thai_metric', 'dtype_dung', 'dau_train'):
        gate(f'moi lan cham: {k}', all(p[k] for p in per))
    gate(f'moi lan cham: AUROC tinh lai - ho tinh < {r.CHECK_TOL}', all(p['lech_max'] < r.CHECK_TOL for p in per))
    out = dict(lich=lich, lan_cham=per, moi_truong=env, nan_grad=_read_json(os.path.join(d, NAN_FILE), None), bo_nho=mem, dung_chung=share, loi=loi)
    if 'val' in last:
        x, ours, theirs, _, _ = last['val']
        best = max(x['ten'], key=lambda n: (theirs[n], -x['ten'].index(n)))
        best_ours = max(x['ten'], key=lambda n: (ours[n], -x['ten'].index(n)))
        out['val'] = dict(ten_chon=chosen, lr_chon=r.lr_from_name(chosen), auroc_ho=theirs, auroc_tinh_lai=ours,
                          chon_dung_luat=gate('lr ho chon dung luat (AUROC val cuoi lon nhat, hoa -> lr dung truoc)', best == chosen),
                          chon_trung_tinh_lai=best_ours == chosen,
                          khoi_khop=gate('khoi val cuoi = lan cham val cuoi', abs(theirs[chosen] - float(val_blk[-1]['ket_qua']['auroc'])) < 1e-9))
    else:
        gate('co lan cham val', False)
    if 'test' in last:
        x, ours, theirs, preds, target = last['test']
        name = x['ten'][0]
        out['test'] = dict(ten=name, auroc_ho_ghi=float(test_blk[-1]['ket_qua']['auroc']), auroc_tinh_lai=ours[name], so_anh=int(target.shape[0]),
                           khop=gate('lan cham test cuoi: mot dau = lr chon, = khoi test cuoi ho ghi',
                                     len(x['ten']) == 1 and name == chosen == test_blk[-1]['ten']
                                     and abs(theirs[name] - float(test_blk[-1]['ket_qua']['auroc'])) < 1e-9 and target.shape[0] == len(names_t)))
        out['du_doan_test'] = (preds[name].reshape(-1), target)
        out['ten_anh_test'] = names_t
    else:
        gate('co lan cham test', False)
    loads = _read_json(os.path.join(d, EVAL88_DIR, LOAD_FILE), [])
    model_loads = [x for x in loads if x.get('nap_mo_hinh')]
    nap = dict(cac_lan=loads)
    if chosen and len(model_loads) == 1 and loads and loads[-1].get('nap_mo_hinh'):
        ld = loads[-1]
        ck = torch.load(os.path.join(d, 'model_final.pth'), map_location='cpu', weights_only=True)
        md5_p1 = state_md5({k: v for k, v in ck['model'].items() if k.startswith(f'classifiers_dict.{chosen}.')})
        nap.update(md5_pha1_dau_chon=md5_p1, md5_sau_nap_pha2=ld.get('md5_sau_nap'))
        gate('pha 2 nap model_final.pth cua pha 1, du khoa, dung dau chon, trong so = pha 1 (md5)',
             ld.get('tep') == 'model_final.pth' and ld.get('thu_muc_tep') == os.path.basename(os.path.normpath(d)) and ld.get('thu_muc_luu') == 'optimal'
             and ld.get('thieu') == [] and not ld.get('sai_hinh') and ld.get('ten_dau') == [chosen] and ld.get('md5_sau_nap') == md5_p1)
        gate('pha 1 khong nap checkpoint nao (thu muc trong)', all(x.get('tep') == '' and not x.get('nap_mo_hinh') for x in loads[:-1]))
    else:
        gate(f'dung mot lan nap mo hinh (pha 2), la lan cuoi: {len(model_loads)} lan', False)
    out['nap'] = nap
    out['dat'] = not loi
    return out


def write_predictions(run_dir, preds, target, names, head):
    '''Dự đoán test cuối pha 2 -> PRED_FILE (preds = sigmoid của họ, float32; target int64; tên ảnh theo thứ tự test.csv; tên đầu).'''
    path = os.path.join(run_dir, PRED_FILE)
    np.savez_compressed(path, preds=np.asarray(preds, np.float32), target=np.asarray(target, np.int64), ten_anh=np.asarray(names), ten_dau=np.asarray(head))
    return path


def run_head(kind, repo, csv_root, checkpoint, config_path, out_root, lrs=config.PAPER_LRS, overrides=None, run_dir=None, log=print):
    '''Một loại đầu trên Shenzhen bằng mã họ: một lượt (seed 0 của mã), cổng check_run, ghi dự đoán test cuối pha 2. Tràn bộ nhớ / lỗi / cổng trượt
       -> ném lỗi (không chạy lại, không chia lưới). -> dict (KHÔNG có số test; số test ở ô chấm).'''
    t0 = time.time()
    r = _runner()
    rec = recipe(overrides)
    log(f"== 88c {kind}: ma CheXFound nguyen van tren Shenzhen ({len(lrs)} lr, lo {rec['--batch-size']}, anh {rec['--image-size']}, "
        f"{rec['--val-epochs']} + {rec['--epochs']} epoch, seed 0 cua ma)")
    run = train_once(kind, repo, csv_root, checkpoint, config_path, os.path.join(out_root, run_dir or RUN_DIRS[kind]), lrs, overrides, log=log)
    if run['chay']['ma_thoat'] != 0:
        if run['chay']['tran_bo_nho']:
            raise RuntimeError(f'{kind}: {len(lrs)} dau cung luc TRAN BO NHO -> DUNG, bao Tuan truoc khi doi cach chay (PLAN_88 §4.3 (8)); '
                               'KHONG tu chia luoi lr / ha kich thuoc anh')
        raise RuntimeError(f"hoc {kind} THAT BAI (ma thoat {run['chay']['ma_thoat']}):\n{run['chay']['duoi_log']}")
    c = check_run(run, csv_root)
    if not c['dat']:
        raise AssertionError(f"{run['thu_muc']}: cong luot KHONG dat {c['loi']} -> DUNG, gui Claude")
    preds, target = c['du_doan_test']
    path = write_predictions(run['thu_muc'], preds, target, c['ten_anh_test'], c['val']['ten_chon'])
    choice = r.phase1_choice(run['ket_qua'])
    return dict(loai=kind, thu_muc=run['thu_muc'], lrs=list(lrs), recipe=rec, args=run['args'], phut=(time.time() - t0) / 60,
                chon=dict(choice, canh_luoi=bool(choice['lr'] == max(lrs))), auroc_val_cac_lr=c['val']['auroc_ho'],
                cong=dict(lich=c['lich'], lan_cham=c['lan_cham'], nap=c['nap'], val={k: v for k, v in c['val'].items() if k != 'auroc_tinh_lai'},
                          test_khop=c['test']['khop']),
                moi_truong=c['moi_truong'], nan_grad=c['nan_grad'], bo_nho=c['bo_nho'], dung_chung=c['dung_chung'],
                thoi_gian=log_timing(os.path.join(run['thu_muc'], 'chay.log')), du_doan=path)


def _dtype_summary(lan_cham):
    return dict(linear_trong_encoder=sorted({str(x['linear_ma_hoa']) for x in lan_cham}),
                dac_trung_ra=sorted({d for x in lan_cham for d in (x['dac_trung'] or [])}))


def smoke_test(repo, csv_root, checkpoint, config_path, out_root, overrides=None, log=print):
    '''Chạy thử trước lượt đầy đủ: Linear 2 lr, 1 + 1 epoch, ĐỦ ảnh (lớp Shenzhen của họ assert 463 / 65 / 134 -> không cắt tập được) -> bắt lỗi môi
       trường / ảnh / tệp chạy / cổng (run_head ném lỗi khi trượt). overrides: chỉ bản kiểm CPU (ảnh nhỏ). Số AUROC ở đây vô nghĩa (không đọc);
       dtype (bằng chứng fp16 trong encoder trên GPU), bộ nhớ đỉnh, thời gian / bước + chờ dữ liệu: ghi lại cho `88a`.'''
    res = run_head('lin', repo, csv_root, checkpoint, config_path, out_root, lrs=SMOKE_LRS, overrides=dict(SMOKE_RECIPE, **(overrides or {})),
                   run_dir=SMOKE_DIR, log=log)
    return dict(dat=True, lich=res['cong']['lich']['ban_ghi'], phut=res['phut'], moi_truong=res['moi_truong'], dtype=_dtype_summary(res['cong']['lan_cham']),
                bo_nho=res['bo_nho'], thoi_gian=res['thoi_gian'], thu_muc=res['thu_muc'])


# ---------------------------------------------------------------- đo cho 88a (PLAN_88 §14.17) ----------------------------------------------------------------
_LOG_ITER = re.compile(r'(Training|Test:)\s*\[\s*(\d+)/(\d+)\].*?time: ([\d.]+)\s+data: ([\d.]+)(?:\s+max mem: (\d+))?')
_LOG_TOTAL = re.compile(r'(Training|Test:) Total time: (\d+):(\d+):(\d+) \(([\d.]+) s / it\)')


def log_timing(log_path):
    '''Dòng tiến độ MetricLogger của họ (chexfound/logging/helpers.py:80–130): 'time' = TB 20 bước gần nhất của thời gian / bước (gồm chờ dữ liệu),
       'data' = TB thời gian chờ lô kế tiếp, 'max mem' = max_memory_allocated (MB, chỉ khi có GPU); dòng 'Total time' cuối mỗi vòng. -> dict(hoc [mỗi
       dòng Training], cham [mỗi dòng Test:], tong_hoc / tong_cham [giây, giây / bước], ty_le_cho_du_lieu (dòng Training cuối: data / time)).'''
    try:
        lines = open(log_path, encoding='utf-8', errors='replace').read().splitlines()
    except OSError:
        return None
    rows = {'Training': [], 'Test:': []}
    totals = {'Training': [], 'Test:': []}
    for line in lines:
        m = _LOG_ITER.search(line)
        if m:
            h, i, n, t, dt, mm = m.groups()
            rows[h].append(dict(buoc=int(i), tong=int(n), giay_buoc=float(t), giay_cho_du_lieu=float(dt), max_mem_mb=int(mm) if mm else None))
            continue
        m = _LOG_TOTAL.search(line)
        if m:
            h, hh, mi, ss, per = m.groups()
            totals[h].append(dict(giay=int(hh) * 3600 + int(mi) * 60 + int(ss), giay_moi_buoc=float(per)))
    last = rows['Training'][-1] if rows['Training'] else None
    return dict(hoc=rows['Training'], cham=rows['Test:'], tong_hoc=totals['Training'], tong_cham=totals['Test:'],
                ty_le_cho_du_lieu=(last['giay_cho_du_lieu'] / last['giay_buoc']) if last and last['giay_buoc'] > 0 else None)


def memory_probe(kind, repo, csv_root, checkpoint, config_path, out_root, run_dir, lrs=config.PAPER_LRS, shared_input=False, overrides=None, log=print):
    '''ĐO (không phải lượt kết quả): mã họ với công thức đủ TRỪ số epoch (SMOKE_RECIPE: 1 + 1 epoch, chấm / lưu mỗi epoch) -> bộ nhớ GPU đỉnh, thời gian.
       run_dir phải bắt đầu bằng 'thu_' (runner._fresh_dir chỉ xoá thư mục lượt chạy). Tràn bộ nhớ -> TRẢ VỀ số đo (tran_bo_nho=True, dat=None),
       không ném lỗi; lỗi khác -> ném lỗi; chạy xong -> cổng check_run phải đạt (kể cả cổng dùng chung đầu vào). overrides: chỉ bản kiểm CPU. -> dict.'''
    assert run_dir.startswith('thu_'), f'{run_dir}: thu muc do phai bat dau bang thu_ -> DUNG'
    t0 = time.time()
    run = train_once(kind, repo, csv_root, checkpoint, config_path, os.path.join(out_root, run_dir), lrs, dict(SMOKE_RECIPE, **(overrides or {})),
                     shared_input=shared_input, log=log)
    d = run['thu_muc']
    out = dict(loai=kind, so_dau=len(lrs), lrs=list(lrs), dung_chung=bool(shared_input), thu_muc=d, ma_thoat=run['chay']['ma_thoat'],
               tran_bo_nho=bool(run['chay']['tran_bo_nho']), bo_nho=_read_json(os.path.join(d, MEM_FILE)),
               dem_dung_chung=_read_json(os.path.join(d, SHARE_FILE)), thoi_gian=log_timing(os.path.join(d, 'chay.log')))
    if run['chay']['ma_thoat'] != 0:
        if not out['tran_bo_nho']:
            raise RuntimeError(f"do {kind} {len(lrs)} dau{' (dung chung dau vao)' if shared_input else ''} THAT BAI (ma thoat {run['chay']['ma_thoat']}, "
                               f"khong phai tran bo nho):\n{run['chay']['duoi_log']}")
        out.update(dat=None, phut=(time.time() - t0) / 60)
        return out
    c = check_run(run, csv_root)
    if not c['dat']:
        raise AssertionError(f"{d}: cong luot do KHONG dat {c['loi']} -> DUNG, gui Claude")
    out.update(dat=True, lich=c['lich']['ban_ghi'], moi_truong=c['moi_truong'], dtype=_dtype_summary(c['lan_cham']), phut=(time.time() - t0) / 60)
    return out


def compare_runs(dir_a, dir_b):
    '''Hai lượt cùng công thức (vd GLoRI không / có dùng chung đầu vào; hoặc hai lần không bọc = mức dao động chạy lại): mọi lần chấm bộ ghi 86e đã
       chép (cham/: target + dự đoán TỪNG đầu + số họ tính) và khối kết quả họ ghi. -> dict(trung_tung_so, cung_tep, so_lan_cham, so_mang, lech_max (|a − b|
       lớn nhất trên mảng số thực khác nhau), khac [tệp:mảng], khoi_trung).'''
    from vindr_protocol.upstream.launch import EVAL_DIR
    r = _runner()
    da, db = os.path.join(dir_a, EVAL_DIR), os.path.join(dir_b, EVAL_DIR)
    fa = sorted(os.listdir(da)) if os.path.isdir(da) else []
    fb = sorted(os.listdir(db)) if os.path.isdir(db) else []
    out = dict(cung_tep=bool(fa) and fa == fb, so_lan_cham=sum(f.endswith('.json') for f in fa), so_mang=0, lech_max=0.0, khac=[])
    for f in sorted(set(fa) & set(fb)):
        pa, pb = os.path.join(da, f), os.path.join(db, f)
        if f.endswith('.npz'):
            with np.load(pa) as za, np.load(pb) as zb:
                if sorted(za.files) != sorted(zb.files):
                    out['khac'].append(f)
                    continue
                for k in za.files:
                    a, b = za[k], zb[k]
                    out['so_mang'] += 1
                    if a.shape != b.shape or a.dtype != b.dtype:
                        out['khac'].append(f'{f}:{k} (hinh / kieu)')
                    elif not np.array_equal(a, b):
                        out['khac'].append(f'{f}:{k}')
                        if np.issubdtype(a.dtype, np.floating):
                            out['lech_max'] = max(out['lech_max'], float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
        elif f.endswith('.json'):
            ja, jb = _read_json(pa), _read_json(pb)
            if any(ja.get(k) != jb.get(k) for k in ('tach', 'iteration', 'ten', 'so_ho_tinh')):
                out['khac'].append(f)
    blocks = lambda d: [(b['buoc'], b['tach'], b['ten'], b['ket_qua']) for b in r.read_results(d)]
    out['khoi_trung'] = blocks(dir_a) == blocks(dir_b)
    out['trung_tung_so'] = out['cung_tep'] and not out['khac'] and out['khoi_trung']
    return out


def head_summary(result):
    '''Một dòng mô tả lượt của một loại đầu (KHÔNG có số test).'''
    c, g = result['chon'], result['cong']
    nan = sum(sum(ph['dau'].values()) for ph in (result['nan_grad'] or []))
    return (f"{result['loai']}: lr chon {c['lr']:g}{' (MEP LUOI)' if c['canh_luoi'] else ''} (AUROC val pha 1 {c['auroc_val']:.4f}) | "
            f"{len(result['lrs'])} lr | lich cham {len(g['lich']['ban_ghi'])} lan dung | nap pha 2 = pha 1 (md5) | grad NaN bi xoa: {nan} | "
            f"autocast {result['moi_truong'].get('autocast_dtype')} | {result['phut']:.0f} phut | so test: o o cham")


# ---------------------------------------------------------------- tóm tắt một dòng cho notebook 88a (không có số AUROC / test) ----------------------------------------------------------------
def _fmt(x, nd=2):
    return '-' if x is None else f'{x:.{nd}f}'


def _mem_text(bn):
    if not bn or bn.get('gpu') is None:
        return 'GPU: khong (CPU)'
    return f"{bn['gpu']}: dinh cap phat {bn['dinh_cap_phat_gb']:.1f} / du tru {bn['dinh_du_tru_gb']:.1f} / tong {bn['tong_gb']:.1f} GiB"


def _step_text(tg):
    th = (tg or {}).get('tong_hoc') or []
    return (f"giay / buoc hoc (vong cuoi) {_fmt(th[-1]['giay_moi_buoc'])}" if th else 'giay / buoc hoc -') + \
        f" | ty le cho du lieu (dong tien do cuoi) {_fmt((tg or {}).get('ty_le_cho_du_lieu'))}"


def run_summary(label, res):
    '''Một dòng cho kết quả smoke_test: lịch chấm, dtype (bằng chứng fp16 trong encoder), bộ nhớ đỉnh, thời gian / bước + chờ dữ liệu.'''
    dt = res.get('dtype') or {}
    return (f"{label}: DAT ({res['phut']:.1f} phut) | lich cham {res['lich']} | autocast {(res.get('moi_truong') or {}).get('autocast_dtype')}, Linear trong "
            f"encoder {dt.get('linear_trong_encoder')}, dac trung ra {dt.get('dac_trung_ra')} | {_mem_text(res.get('bo_nho'))} | {_step_text(res.get('thoi_gian'))}")


def probe_summary(name, r):
    '''Một dòng cho một phép đo memory_probe: số đầu, có dùng chung đầu vào, tràn / xong, bộ nhớ đỉnh, lỗi (nếu tràn), số đếm dùng chung, thời gian.'''
    bn = r.get('bo_nho') or {}
    return (f"{name}: {r['so_dau']} dau{' + DUNG CHUNG dau vao' if r['dung_chung'] else ' (ma nguyen van)'} -> "
            f"{'TRAN BO NHO' if r['tran_bo_nho'] else 'chay xong, cong dat'} | {_mem_text(bn)}"
            + (f" | loi: {bn['loi'][:160]}" if bn.get('loi') else '') + (f" | dem {r['dem_dung_chung']}" if r['dung_chung'] else '')
            + f" | {r['phut']:.1f} phut | {_step_text(r.get('thoi_gian'))}")


def compare_summary(name, c):
    '''Một dòng cho compare_runs.'''
    return (f"{name}: {c['so_lan_cham']} lan cham, {c['so_mang']} mang -> "
            + ('TRUNG tung so' if c['trung_tung_so'] else f"KHAC: lech max {c['lech_max']:.2e}, {len(c['khac'])} mang khac, cung tep {c['cung_tep']}, "
                                                          f"khoi ket qua trung {c['khoi_trung']}"))


def pack_logs(path, out_root, prefixes=('thu_',)):
    '''tar.gz các thư mục lượt (tên bắt đầu bằng prefixes) dưới out_root — log, results_eval_linear.json, config.yaml, mọi lần chấm đã chép, bộ nhớ, đếm —
       BỎ *.pth (checkpoint). -> path.'''
    import tarfile
    dirs = sorted(d for d in (os.listdir(out_root) if os.path.isdir(out_root) else []) if d.startswith(prefixes) and os.path.isdir(os.path.join(out_root, d)))
    with tarfile.open(path, 'w:gz') as tf:
        for d in dirs:
            tf.add(os.path.join(out_root, d), arcname=d, filter=lambda ti: None if ti.name.endswith('.pth') else ti)
    return path
