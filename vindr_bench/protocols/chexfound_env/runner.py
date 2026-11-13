# -*- coding: utf-8 -*-
"""Điều phối Ver1 (PLAN_89 §3, §4): mỗi loại đầu = lượt `linear_glori.py` của họ qua torchrun (launch.py), lưới 9 lr của bài, 10 + 10 epoch, lô 256,
chọn lr theo AUROC val (mã họ), pha 2 học tiếp trên train + val rồi chấm test (mã họ). Chép từ vindr_protocol/upstream/runner.py @ 0.8.2 (86f / 86j),
thêm Linear-1 và ML-Decoder:

    lin1   Linear-1   --n-last-blocks 1                       vector toàn ảnh tầng cuối              9 lr một lượt
    lin4   Linear-4   --n-last-blocks 4                       nối vector 4 tầng cuối (mặc định họ)   9 lr một lượt
    mldec  ML-Decoder --glori --n-last-blocks 1 + VB_HEAD     patch tầng cuối (heads/mldecoder.py)   hai nửa 5 + 4
    glori  GLoRI      --glori --cat-cls --n-last-blocks 4     mã GLoRI công khai của họ               hai nửa 5 + 4

Hai nửa lưới: lr thắng = AUROC val pha 1 cao nhất giữa hai nửa (hoà -> nửa đầu = lr nhỏ hơn, như so sánh '>' của họ); dự đoán test = lần chấm test
CUỐI của pha 2 trong lượt thắng. Mọi lượt qua cổng: (1) tiến trình học tự ghi kết quả nạp trọng số (nap_encoder.json) đạt; (2) lần chấm val cuối:
AUROC tính lại (torchmetrics 0.10.3, CPU) = số họ tính, lr họ chọn đúng luật, tên đầu = lưới lr + số tầng của loại đầu; (3) lần chấm test cuối: một đầu,
target = test.csv; (4) ML-Decoder: tiến trình học tự ghi đã dựng đúng lớp đầu (pha 1: đủ lưới; pha 2: 1 đầu); loại khác: không có bản ghi đó."""
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass

from . import dataset as D
from .environment import TheirEnv, child_env

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCH = os.path.join(HERE, 'launch.py')
EVAL_DIR, NAP_FILE, MLDEC_FILE = 'cham', 'nap_encoder.json', 'dau_mldecoder.json'     # = launch.py (không import launch ở đây: cần chexfound)
MEM_FILE = 'bo_nho_gpu.json'                                                             # = launch.MEM_FILE
# Cổng bộ nhớ của lượt chạy thử (0.2.1): đỉnh cấp phát <= 70 % bộ nhớ GPU. Hiệu chỉnh bằng số đo 13/11 (RAD-DINO 518, lô 256, không chia khúc):
# Linear-1 đỉnh 61,6 / 79,25 GiB (78 %) — qua sát trần; Linear-4 64,6 GiB (82 %) — OOM sau lần chấm val đầu (phân mảnh). Ước sau chia khúc 64:
# Linear ~18–24 GiB, ML-Decoder 5 đầu ~19 GiB, GLoRI 5 đầu ~47 GiB (tools/memory_estimate.py).
MEM_LIMIT = 0.70
RESULTS_FILE = 'results_eval_linear.json'
OPTS = ('student.block_chunks=4',)            # như 86f / 86j (config.yaml của CheXFound phát hành ghi 0; với lớp bọc không ảnh hưởng mô hình)
BATCH = 256
PAPER_LRS = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3)    # bài CheXFound §IV-A.2 (Linear và GLoRI)
HALVES = (PAPER_LRS[:5], PAPER_LRS[5:])
HEADS = {
    'lin1': dict(title='Linear-1', args=('--n-last-blocks', '1'), blocks=1, halves=False, env={}),
    'lin4': dict(title='Linear-4', args=('--n-last-blocks', '4'), blocks=4, halves=False, env={}),
    'mldec': dict(title='ML-Decoder', args=('--glori', '--n-last-blocks', '1'), blocks=1, halves=True, env={'VB_HEAD': 'mldecoder'}),
    'glori': dict(title='GLoRI', args=('--glori', '--cat-cls', '--n-last-blocks', '4'), blocks=4, halves=True, env={}),
}
HEAD_ORDER = ('lin1', 'lin4', 'mldec', 'glori')
CHECK_TOL = 5e-5                              # |AUROC tính lại (CPU) − số họ tính (GPU)| từ CÙNG preds
SMOKE_SIZES = dict(train=512, val=256, test=256)
_OOM = ('CUDA out of memory', 'OutOfMemoryError', 'CUBLAS_STATUS_ALLOC_FAILED')
_HEX32 = re.compile(r'(?<![0-9A-Za-z])[0-9a-f]{32}(?![0-9A-Za-z])')


@dataclass
class Ver1Run:
    '''Cấu hình chung mọi lượt của một notebook Ver1.'''
    env: TheirEnv
    encoder: str                  # tên lớp bọc vindr_bench.encoders (VB_ENCODER)
    image_size: int               # --image-size = độ phân giải gốc của encoder
    csv_root: str
    config_path: str              # config.yaml của CheXFound (setup(args) + autocast fp16 của họ)
    weights_path: str             # --pretrained-weights: lớp bọc thay setup_and_build_model -> không đọc (ghi tên giữ chỗ)
    out_root: str
    extra_args: tuple = ()        # bản kiểm trên máy: ('--num-workers', 0)
    use_torchrun: bool = True     # bản kiểm trên Windows: False (một tiến trình, biến phân tán đặt tay)


def scrub(text):
    '''Đuôi log IN RA notebook: che chuỗi 32 hex (= dạng mã ảnh VinDr trong đường dẫn; luật PhysioNet). Tệp log đầy đủ giữ nguyên.'''
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


def run_torchrun(run, args, log_path, env_extra=None, poll_s=180, log=print):
    '''torchrun --nproc_per_node=1 launch.py <args> bằng Python của họ; log vào log_path; mỗi poll_s giây in dòng tiến độ cuối. -> dict.'''
    env = child_env(run.env)
    for k in ('VB_ENCODER', 'VB_HEAD'):
        env.pop(k, None)
    env['VB_ENCODER'] = run.encoder
    env.update(env_extra or {})
    if run.use_torchrun:
        cmd = [run.env.python, '-m', 'torch.distributed.run', '--nproc_per_node=1', '--standalone', LAUNCH, *[str(a) for a in args]]
    else:
        cmd = [run.env.python, LAUNCH, *[str(a) for a in args]]
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


def read_results(output_dir):
    '''results_eval_linear.json của họ ("ITER: <bước> <tách>" + một dòng JSON) -> [dict(buoc, tach, ten, ket_qua)].'''
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
    '''"linear:blocks=4:avgpool=False:lr=0_0005000000" -> 0.0005.'''
    m = re.search(r'lr=(\d+)_(\d+)', name)
    assert m, name
    return float(f'{m.group(1)}.{m.group(2)}')


def blocks_from_name(name):
    m = re.search(r'blocks=(\d+)', name)
    assert m, name
    return int(m.group(1))


def read_evaluations(run_dir):
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


def _np_equal(a, b):
    import numpy as np
    return a.shape == b.shape and bool(np.array_equal(a, b))


def check_run(r, csv_root, kind, lrs):
    '''Cổng lần chấm của MỘT lượt (đọc lần chấm đã ghi + results_eval_linear.json): val cuối (pha 1) và test cuối (pha 2). -> dict(val, test, du_doan_test).'''
    from ...evaluation.metrics import tm_scores
    recs, blocks = read_evaluations(r['thu_muc']), r['ket_qua']
    want = sorted(float(v) for v in lrs)
    out = {}
    for split in ('val', 'test'):
        rs, bs = [x for x in recs if x['tach'] == split], [b for b in blocks if b['tach'] == split]
        assert rs and bs, f"{r['thu_muc']}: thieu lan cham / khoi {split} -> DUNG"
        rec, blk = rs[-1], bs[-1]
        target, preds = load_evaluation(rec)
        assert _np_equal(target, _csv_labels(csv_root, split)), f'{split}: target lan cham KHAC nhan {split}.csv (thu tu) -> DUNG'
        ours = {n: tm_scores(p, target)[1] for n, p in preds.items()}
        theirs = {n: rec['so_ho_tinh'][n]['auroc'] for n in rec['ten']}
        best_theirs = max(rec['ten'], key=lambda n: (theirs[n], -rec['ten'].index(n)))
        best_ours = max(rec['ten'], key=lambda n: (ours[n], -rec['ten'].index(n)))
        lech = max(abs(ours[n] - theirs[n]) for n in rec['ten'])
        names_ok = all(blocks_from_name(n) == HEADS[kind]['blocks'] for n in rec['ten'])
        if split == 'val':
            names_ok = names_ok and sorted(lr_from_name(n) for n in rec['ten']) == want
        out[split] = dict(iteration=rec['iteration'], buoc_khoi=blk['buoc'], so_dau=len(rec['ten']), lech_max=float(lech), ten_ho_chon=blk['ten'],
                          auroc_ho_ghi=float(blk['ket_qua']['auroc']), auroc_tinh_lai=float(ours[blk['ten']]) if blk['ten'] in ours else None,
                          chon_dung_luat=bool(blk['ten'] == best_theirs), chon_trung_tinh_lai=bool(best_ours == best_theirs), ten_dung_cau_hinh=bool(names_ok),
                          dat=bool(lech < CHECK_TOL and blk['ten'] == best_theirs and rec['iteration'] == blk['buoc'] and names_ok
                                   and abs(theirs[blk['ten']] - float(blk['ket_qua']['auroc'])) < 1e-6))
        if split == 'test':
            assert len(rec['ten']) == 1, f"lan cham test cuoi co {len(rec['ten'])} dau (pha 2 phai MOT dau) -> DUNG"
            out['du_doan_test'] = (preds[rec['ten'][0]], target)
    return out


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def process_checks(r, kind, lrs):
    '''Cổng do CHÍNH tiến trình học ghi: nạp trọng số (nap_encoder.json) + đầu ML-Decoder (dau_mldecoder.json). -> dict(dat, nap, mldecoder, ly_do).'''
    nap = _read_json(os.path.join(r['thu_muc'], NAP_FILE))
    ml = _read_json(os.path.join(r['thu_muc'], MLDEC_FILE))
    why = []
    if not (nap and nap.get('dat') is True):
        why.append('nap trong so: khong co tep / KHONG dat')
    if kind == 'mldec':
        ok = (isinstance(ml, list) and len(ml) == 2 and ml[0]['so_dau'] == len(lrs) and sorted(ml[0]['lr']) == sorted(float(v) for v in lrs)
              and ml[1]['so_dau'] == 1 and all(m['lop'] == ['MLDecoderOnTheirFeatures'] for m in ml))
        if not ok: why.append(f'ML-Decoder: ban ghi dung dau {ml!r:.300} KHAC (pha 1 du luoi, pha 2 mot dau)')
    elif ml is not None:
        why.append(f'{kind}: co ban ghi ML-Decoder (khong duoc tiem) -> sai loai dau')
    return dict(dat=not why, nap=nap, mldecoder=ml, ly_do=why, bo_nho=_read_json(os.path.join(r['thu_muc'], MEM_FILE)))


_RUN_DIR_PREFIXES = tuple(f'{k}_' for k in HEADS) + ('thu_',)


def _fresh_dir(out_dir):
    '''Thư mục lượt TRỐNG (mã họ resume=True: còn checkpoint cũ thì học TIẾP). Chỉ xoá thư mục lượt chạy của bộ điều phối.'''
    assert os.path.basename(os.path.normpath(out_dir)).startswith(_RUN_DIR_PREFIXES), f'{out_dir}: khong phai thu muc luot chay -> KHONG xoa'
    if os.path.lexists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def run_args(run, kind, csv_root, out_dir, lrs, extra=()):
    '''Dòng lệnh đầy đủ của một lượt (ghi vào kết quả: cổng "tham số trùng" khi dùng lại lượt cũ).'''
    a = ['--batch-size', BATCH, '--image-size', run.image_size, '--config-file', run.config_path, '--pretrained-weights', run.weights_path,
         '--train-dataset', D.dataset_str('train', csv_root), '--val-dataset', D.dataset_str('val', csv_root),
         '--test-dataset', D.dataset_str('test', csv_root), *run.extra_args, *HEADS[kind]['args'], '--output-dir', out_dir,
         '--learning-rates', *[f'{v:g}' for v in lrs], *extra, '--opts', *OPTS]
    return [str(x) for x in a]


def train_once(run, kind, out_dir, lrs, csv_root=None, extra=(), log=print):
    '''MỘT lượt mã họ (pha 1 + pha 2) trong thư mục TRỐNG. -> dict(thu_muc, lrs, args, chay, ket_qua, kiem_tien_trinh).'''
    _fresh_dir(out_dir)
    args = run_args(run, kind, csv_root or run.csv_root, out_dir, lrs, extra)
    res = run_torchrun(run, args, os.path.join(out_dir, 'chay.log'), env_extra=HEADS[kind]['env'], log=log)
    r = dict(thu_muc=out_dir, lrs=list(lrs), args=args, chay=res, ket_qua=read_results(out_dir) if res['ma_thoat'] == 0 else None)
    r['kiem_tien_trinh'] = process_checks(r, kind, lrs)
    return r


def phase1_choice(results):
    '''Lựa chọn lr của pha 1 = khối 'val' CUỐI. -> dict(ten, lr, auroc_val, buoc).'''
    val = [x for x in results if x['tach'] == 'val']
    assert val, 'khong co khoi val (pha 1) -> DUNG'
    last = val[-1]
    return dict(ten=last['ten'], lr=lr_from_name(last['ten']), auroc_val=float(last['ket_qua']['auroc']), buoc=last['buoc'])


PRED_FILE = 'du_doan_test.npz'


def _write_test_predictions(run_dir, preds_target):
    import numpy as np
    path = os.path.join(run_dir, PRED_FILE)
    np.savez_compressed(path, preds=preds_target[0].astype(np.float32), target=preds_target[1].astype(np.int64))
    return path


def _assert_run(r, kind):
    assert r['chay']['ma_thoat'] == 0, f"hoc {kind} ({r['thu_muc']}) THAT BAI:\n{r['chay']['duoi_log']}"
    assert r['kiem_tien_trinh']['dat'], f"hoc {kind} ({r['thu_muc']}): cong tien trinh hoc KHONG dat: {r['kiem_tien_trinh']['ly_do']} -> DUNG"


def run_head(run, kind, log=print):
    '''Đầy đủ một loại đầu (lưới 9 lr; GLoRI / ML-Decoder: hai nửa 5 + 4). -> dict kết quả (dự đoán test của lượt thắng ghi ra tệp).'''
    assert kind in HEADS, kind
    t0 = time.time()
    h = HEADS[kind]
    log(f"== {h['title']} ({kind}): ma cua ho, luoi {len(PAPER_LRS)} lr cua bai, lo {BATCH}, anh {run.image_size}, 10 + 10 epoch, encoder {run.encoder}"
        + (', hai nua luoi 5 + 4' if h['halves'] else ''))
    grids = HALVES if h['halves'] else (PAPER_LRS,)
    runs = []
    for i, lrs in enumerate(grids):
        name = f'{kind}_hoc' + (f'_nua{i + 1}' if h['halves'] else '')
        runs.append(train_once(run, kind, os.path.join(run.out_root, name), lrs, log=log))
        _assert_run(runs[-1], kind)
    checks = [check_run(r, run.csv_root, kind, r['lrs']) for r in runs]
    for r, c in zip(runs, checks):
        assert c['val']['dat'] and c['test']['dat'], f"{r['thu_muc']}: cong lan cham KHONG dat (val {c['val']}, test {c['test']}) -> DUNG, gui Claude"
    choices = [phase1_choice(r['ket_qua']) for r in runs]
    k = max(range(len(runs)), key=lambda i: (choices[i]['auroc_val'], -i))
    winner = runs[k]
    test_blocks = [b for b in winner['ket_qua'] if b['tach'] == 'test']
    pred_path = _write_test_predictions(winner['thu_muc'], checks[k]['du_doan_test'])
    return dict(loai=kind, ten_dau=h['title'], encoder=run.encoder, anh=run.image_size, hai_nua=h['halves'],
                luot=[dict(thu_muc=r['thu_muc'], lrs=r['lrs'], args=r['args'], phut=r['chay']['phut'], chon=c, cong={s: x[s] for s in ('val', 'test')},
                           kiem_tien_trinh=r['kiem_tien_trinh']) for r, c, x in zip(runs, choices, checks)],
                thang=k, chon=choices[k], test_ho_ghi_cuoi_pha2=(test_blocks[-1] if test_blocks else None),
                cong={s: checks[k][s] for s in ('val', 'test')}, du_doan=pred_path, phut_tong=(time.time() - t0) / 60)


def smoke_lrs(kind):
    '''Lưới của lượt chạy thử = nhóm đầu LỚN NHẤT của lượt thật (Linear: 9 lr; GLoRI / ML-Decoder: nửa đầu 5 lr) -> bộ nhớ đầu đọc như lượt thật.'''
    return HALVES[0] if HEADS[kind]['halves'] else PAPER_LRS


def memory_ok(mem, limit=MEM_LIMIT):
    '''Cổng bộ nhớ (0.2.1): đỉnh cấp phát của tiến trình học <= limit × bộ nhớ GPU. Không có số đo -> KHÔNG đạt.'''
    return bool(mem) and mem['dinh_cap_phat_gib'] <= limit * mem['tong_gib']


def smoke_test(run, kinds=HEAD_ORDER, log=print):
    '''Chạy thử nhỏ mỗi loại đầu (số đo VÔ NGHĨA): lưới = nhóm đầu lớn nhất của lượt thật (smoke_lrs), lô 256 như lượt thật, SMOKE_SIZES ảnh đầu mỗi
       tách, 1 + 1 epoch -> bắt lỗi môi trường / tiêm / đọc kết quả / ghi lần chấm / mọi cổng + CỔNG BỘ NHỚ (đỉnh <= MEM_LIMIT bộ nhớ GPU) TRƯỚC lượt
       đầy đủ. -> {loại: dict(dat, bo_nho, ...)}.'''
    small = os.path.join(run.out_root, 'csv_nho')
    sizes = D.write_subset_csvs(run.csv_root, small, SMOKE_SIZES)
    out = {}
    for kind in kinds:
        lrs = smoke_lrs(kind)
        r = train_once(run, kind, os.path.join(run.out_root, f'thu_{kind}'), lrs, csv_root=small,
                       extra=('--epochs', 1, '--val-epochs', 1, '--eval-period-epochs', 1, '--save-checkpoint-frequency', 1), log=log)
        _assert_run(r, kind)
        c = check_run(r, small, kind, lrs)
        preds, target = c['du_doan_test']
        mem = r['kiem_tien_trinh']['bo_nho']
        ok = (preds.shape == (sizes['test'], len(sizes['class_names'])) and c['val']['dat'] and c['test']['dat'] and c['val']['so_dau'] == len(lrs)
              and int(target.sum()) > 0 and memory_ok(mem))
        out[kind] = dict(dat=bool(ok), bo_nho=mem, bo_nho_dat=memory_ok(mem), so_dau=len(lrs), hinh_du_doan=list(preds.shape),
                         khoi=[(b['buoc'], b['tach']) for b in r['ket_qua']], phut=r['chay']['phut'], cong={s: c[s] for s in ('val', 'test')},
                         kiem_tien_trinh=r['kiem_tien_trinh'])
    return out


# ---- lưu / khôi phục từng loại đầu qua tệp resume của khối ghi cuối (ghi() chỉ local; dong_bo() tải lên MỘT lượt cuối) ----
HEAD_FILES = ('du_doan_{label}_{kind}_{t}.npz', 'luot_{label}_{kind}_{t}.json', 'nhat_ky_{label}_{kind}_{t}.tar.gz')


def run_label(run_tag):
    '''RUN_TAG 'r89v1_raddino' -> '89v1_raddino'.'''
    m = re.fullmatch(r'r(89[a-z0-9_]+)', run_tag)
    assert m, f'RUN_TAG {run_tag!r} khong co dang r89<...> -> DUNG'
    return m.group(1)


def resume_state(path, run_tag, khoi_phuc_cho, log=print):
    if not path:
        return dict(run=run_tag, nhan=run_label(run_tag), t=time.strftime('%Y%m%d_%H%M%S'), xong={})
    import torch
    st = torch.load(path, map_location='cpu', weights_only=False)
    assert st.get('run') == run_tag, f"tep resume cua luot KHAC ({st.get('run')} != {run_tag}) -> DOI RUN_TAG o C0, KHONG xoa gi"
    khoi_phuc_cho(st.get('cho'))
    log(f"RESUME: luot {st['t']} da xong {sorted(st['xong']) or '-'} -> khoi phuc, KHONG chay lai")
    return dict(run=st['run'], nhan=run_label(st['run']), t=st['t'], xong={k: list(v) for k, v in st['xong'].items()})


def _write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)


def save_head(result, kind, state, out_root, ghi):
    '''-> 3 tệp qua ghi(): dự đoán test, kết quả JSON, nhật ký của họ (log, results_eval_linear.json, lần chấm, bản ghi tiến trình; KHÔNG .pth).'''
    names = [f.format(label=state['nhan'], kind=kind, t=state['t']) for f in HEAD_FILES]
    with open(result['du_doan'], 'rb') as f:
        pred_bytes = f.read()

    def _bytes(q):
        with open(q, 'wb') as fh:
            fh.write(pred_bytes)
    result['du_doan'] = ghi(names[0], _bytes)
    js = json.loads(json.dumps(result, default=str))
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
    import numpy as np
    with open(os.path.join(local_dir, names[1]), encoding='utf-8') as f:
        result = json.load(f)
    result['du_doan'] = os.path.join(local_dir, names[0])
    with np.load(result['du_doan']) as z:
        assert z['preds'].ndim == 2 and z['preds'].shape == z['target'].shape and np.isfinite(z['preds']).all(), f'{names[0]}: du doan hong -> DUNG'
    result['khoi_phuc'] = True
    return result


def head_step(run, kind, state, local_dir, ghi, save_state, log=print):
    '''Một loại đầu có resume: đã xong trong tệp resume -> khôi phục; chưa -> run_head, save_head, save_state(state).'''
    if kind in state['xong']:
        return restore_head(state['xong'][kind], local_dir)
    result = run_head(run, kind, log=log)
    state['xong'][kind] = save_head(result, kind, state, run.out_root, ghi)
    save_state(state)
    return result


def head_summary(result):
    '''Một dòng mô tả lượt của một loại đầu (KHÔNG có số test — số test ở ô chấm).'''
    c, g = result['chon'], result['cong']
    when = 'KHOI PHUC tu tep resume' if result.get('khoi_phuc') else f"{result['phut_tong']:.0f} phut"
    mems = [x['kiem_tien_trinh'].get('bo_nho') for x in result['luot'] if (x.get('kiem_tien_trinh') or {}).get('bo_nho')]
    mem = (f" | GPU dinh {max(m['dinh_cap_phat_gib'] for m in mems):.1f} / {mems[0]['tong_gib']:.1f} GiB" if mems else '')
    edge = ' | lr O BIEN LUOI (PLAN_89 7.5)' if c['lr'] in (min(PAPER_LRS), max(PAPER_LRS)) else ''
    return (f"{result['ten_dau']}: anh {result['anh']} | lr chon {c['lr']:g}{edge} (AUROC val pha 1 {c['auroc_val']:.4f}, dropout bat)"
            + (' | 2 nua luoi 5 + 4' if result['hai_nua'] else '') + mem + ' | luot: '
            + ', '.join(f"{os.path.basename(x['thu_muc'])} {x['phut']:.0f} phut" for x in result['luot'])
            + f" | cong lan cham: val {g['val']['so_dau']} lr, lech tinh lai {g['val']['lech_max']:.1e}, lr ho chon "
            + ('dung luat' if g['val']['chon_dung_luat'] else 'SAI luat') + f"; test lech {g['test']['lech_max']:.1e} | {when} | so test: o C7")
