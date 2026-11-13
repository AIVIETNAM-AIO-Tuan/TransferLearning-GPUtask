# -*- coding: utf-8 -*-
"""Tệp chạy dưới torchrun cho `linear_glori.py` của CheXFound NGUYÊN VĂN (PLAN_89 §4). Cần PYTHONPATH = repo CheXFound (commit ghim) + thư mục chứa
vindr_bench. Chạy: torchrun --nproc_per_node=1 <tệp này> <tham số y như linear_glori.py>.

Không sửa tệp nào của họ: các hàm được tra theo tên trong module lúc gọi nên thay từ ngoài là đủ. Trước khi gọi main() của họ:
  1. đăng ký tên dataset 'VinDr' (bọc chexfound.data.loaders._parse_dataset_str);
  2. GHI MỌI LẦN CHẤM: bọc linear_glori.evaluate / evaluate_linear_classifiers — chép (preds, target) đúng như vừa đưa vào metric.update của họ + số họ
     tính -> <output-dir>/cham/<thứ tự>_<tách>_iter<k>.npz + .json (phép tính, thứ tự gọi, chế độ train / eval của đầu, RNG không đổi);
  3. (biến VB_ENCODER) thay mô hình CheXFound bằng lớp bọc vindr_bench.encoders trong setup_and_build_model — vẫn setup(args) + autocast theo config
     của họ; kết quả kiểm nạp trọng số của CHÍNH tiến trình học -> <output-dir>/nap_encoder.json, trượt -> dừng;
  5. (0.2.1) sau main() của họ: bộ nhớ GPU đỉnh (cấp phát / giữ) của chính tiến trình -> <output-dir>/bo_nho_gpu.json (cổng bộ nhớ ở lượt chạy thử).
  4. (biến VB_HEAD=mldecoder; chỉ với --glori --n-last-blocks 1, không --cat-cls) thay setup_glori bằng heads.mldecoder.setup_mldecoder; mỗi lần dựng
     đầu (pha 1, pha 2) ghi một bản ghi vào <output-dir>/dau_mldecoder.json (số đầu, chiều vào, số tham số) — runner kiểm.
Chép từ vindr_protocol/upstream/launch.py @ 0.8.2 (mục 1–3); mục 4 mới ở 89. Chỉ cú pháp / API có ở Python 3.9 + torch 2.0."""
import json
import os
import sys

EVAL_DIR = 'cham'
NAP_FILE = 'nap_encoder.json'
MLDEC_FILE = 'dau_mldecoder.json'
MEM_FILE = 'bo_nho_gpu.json'      # 0.2.1: bộ nhớ GPU đỉnh của CHÍNH tiến trình học (cả hai pha) — cổng bộ nhớ ở lượt chạy thử
ENCODER_ENV, HEAD_ENV = 'VB_ENCODER', 'VB_HEAD'


def _rank0():
    return int(os.environ.get('RANK', '0')) == 0


def write_memory(out_dir):
    '''Ghi bộ nhớ GPU đỉnh (cấp phát + giữ) của tiến trình này vào out_dir/MEM_FILE (rank 0). Không có GPU -> không ghi, trả None.'''
    import torch
    if not torch.cuda.is_available() or not _rank0():
        return None
    p = torch.cuda.get_device_properties(0)
    rec = dict(gpu=p.name, tong_gib=p.total_memory / 2 ** 30, dinh_cap_phat_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
               dinh_giu_gib=torch.cuda.max_memory_reserved() / 2 ** 30)
    with open(os.path.join(out_dir, MEM_FILE), 'w', encoding='utf-8') as f:
        json.dump(rec, f)
    print('BO_NHO_GPU ' + json.dumps(rec), flush=True)
    return rec


def register_vindr():
    import chexfound.data.loaders as loaders

    from vindr_bench.protocols.chexfound_env import dataset
    original = loaders._parse_dataset_str
    if getattr(original, '_vindr_wrapped', False):
        return original

    def parse(dataset_str):
        if dataset_str.split(':')[0] == dataset.NAME:
            return dataset.parse_dataset_str(dataset_str)
        return original(dataset_str)
    parse._vindr_wrapped = True
    loaders._parse_dataset_str = parse
    return original


def record_evaluations(linear_glori, out_dir):
    '''Bọc linear_glori.evaluate_linear_classifiers (iteration / prefixstring) và linear_glori.evaluate (chép preds / target từng đầu + số họ tính).'''
    import numpy as np
    import torch

    os.makedirs(out_dir, exist_ok=True)
    orig_elc, orig_eval = linear_glori.evaluate_linear_classifiers, linear_glori.evaluate
    ctx = dict(n=0, iteration=None, prefix='')

    class _Recorder(torch.nn.Module):
        def __init__(self, inner, sink):
            super().__init__()
            self.inner, self.sink = inner, sink

        def forward(self, samples, targets):
            out = self.inner(samples, targets)
            # .clone(): trên CPU .cpu() trả CHÍNH tensor đó (cùng bộ nhớ với tensor torchmetrics giữ) -> chép thật
            self.sink.append((out['preds'].detach().float().cpu().clone(), out['target'].detach().cpu().clone()))
            return out

    def evaluate_linear_classifiers(*args, **kwargs):
        ctx.update(iteration=kwargs.get('iteration'), prefix=kwargs.get('prefixstring', ''))
        return orig_elc(*args, **kwargs)

    def evaluate(model, data_loader, postprocessors, metrics, device, *args, **kwargs):
        sinks = {k: [] for k in postprocessors}
        wrapped = {k: _Recorder(p, sinks[k]) for k, p in postprocessors.items()}
        res = orig_eval(model, data_loader, wrapped, metrics, device, *args, **kwargs)
        names = list(postprocessors)
        targets = [torch.cat([t for _, t in sinks[k]]) for k in names]
        assert all(torch.equal(targets[0], t) for t in targets[1:]), 'target khac nhau giua cac dau trong mot lan cham -> DUNG'
        split = getattr(getattr(data_loader.dataset, 'split', None), 'value', 'khong_ro')
        stem = f"{ctx['n']:03d}_{split}_iter{ctx['iteration']}"
        np.savez_compressed(os.path.join(out_dir, stem + '.npz'), target=targets[0].numpy(),
                            **{f'preds_{j}': torch.cat([p for p, _ in sinks[k]]).numpy() for j, k in enumerate(names)})
        stats = {k: {m: float(v) for m, v in res[1][k].items()} for k in names}
        with open(os.path.join(out_dir, stem + '.json'), 'w', encoding='utf-8') as f:
            json.dump(dict(thu_tu=ctx['n'], tach=split, iteration=ctx['iteration'], prefix=ctx['prefix'], ten=names, so_ho_tinh=stats), f)
        ctx['n'] += 1
        return res

    linear_glori.evaluate_linear_classifiers, linear_glori.evaluate = evaluate_linear_classifiers, evaluate
    return orig_elc, orig_eval


def inject_encoder(linear_glori, name):
    '''Thay linear_glori.setup_and_build_model: VẪN setup(args) + get_autocast_dtype(config) của họ, CHỈ thay mô hình bằng lớp bọc `name`.'''
    import torch.backends.cudnn as cudnn
    from chexfound.eval import setup as ev_setup

    from vindr_bench import encoders
    from vindr_bench.checks import weights
    original = linear_glori.setup_and_build_model

    def setup_and_build_model(args):
        cudnn.benchmark = True                                   # như hàm gốc
        config = ev_setup.setup(args)
        model = encoders.build(name, check=False)
        if _rank0():                                             # ghi TRƯỚC khi quyết (trượt vẫn để lại bằng chứng)
            with open(os.path.join(args.output_dir, NAP_FILE), 'w', encoding='utf-8') as f:
                json.dump(model.nap, f, ensure_ascii=False, indent=1, default=str)
        print('NAP_ENCODER ' + weights.loading_text(model.nap), flush=True)
        assert model.nap['dat'], f'{name}: nap trong so KHONG dat -> DUNG (xem {NAP_FILE})'
        model.eval()
        model.cuda()
        return model, ev_setup.get_autocast_dtype(config)
    setup_and_build_model._encoder = name
    linear_glori.setup_and_build_model = setup_and_build_model
    return original


def inject_mldecoder(linear_glori, out_dir):
    '''Thay linear_glori.setup_glori bằng setup_mldecoder; mỗi lần gọi ghi bản ghi (số đầu, chiều vào, số tham số, lr) vào out_dir/MLDEC_FILE.'''
    from vindr_bench.heads import mldecoder
    original = linear_glori.setup_glori
    path = os.path.join(out_dir, MLDEC_FILE)

    def setup_glori(sample_output, n_last_blocks_list, learning_rates, *args, **kwargs):
        classifiers, groups = mldecoder.setup_mldecoder(sample_output, n_last_blocks_list, learning_rates, *args, **kwargs)
        heads = getattr(classifiers, 'module', classifiers).classifiers_dict
        rec = dict(so_dau=len(heads), ten=list(heads), lr=[float(g['lr']) for g in groups], chieu_vao=int(sample_output[-1][0].shape[-1]),
                   so_token=int(sample_output[-1][0].shape[1]), lop=sorted({type(h).__name__ for h in heads.values()}),
                   so_tham_so_moi_dau=sorted({sum(p.numel() for p in h.parameters()) for h in heads.values()}))
        if _rank0():
            old = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else []
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(old + [rec], f, ensure_ascii=False)
        print(f"DAU_MLDECODER {rec['so_dau']} dau, vao {rec['so_token']} x {rec['chieu_vao']}, lop {rec['lop']}", flush=True)
        return classifiers, groups
    linear_glori.setup_glori = setup_glori
    return original


def main(argv=None):
    register_vindr()
    from chexfound.eval.classification import linear_glori
    args = linear_glori.get_args_parser(description='DINOv2 linear evaluation').parse_args(argv)
    assert not args.infer, 'du doan lay tu cac lan cham trong luot hoc -> khong dung --infer'
    head = os.environ.get(HEAD_ENV, '')
    if head:
        assert head == 'mldecoder', f'{HEAD_ENV}={head!r} khong ho tro -> DUNG'
        assert args.glori and not args.cat_cls and list(args.n_last_blocks) == [1], (
            f'ML-Decoder can --glori --n-last-blocks 1, khong --cat-cls (nhan glori={args.glori}, cat_cls={args.cat_cls}, n={args.n_last_blocks}) -> DUNG')
        os.makedirs(args.output_dir, exist_ok=True)
        inject_mldecoder(linear_glori, args.output_dir)
    if os.environ.get(ENCODER_ENV):
        inject_encoder(linear_glori, os.environ[ENCODER_ENV])
    record_evaluations(linear_glori, os.path.join(args.output_dir, EVAL_DIR))
    rc = linear_glori.main(args)
    write_memory(args.output_dir)
    return rc


if __name__ == '__main__':
    # chạy như tệp: bỏ thư mục của tệp khỏi sys.path (dataset.py / runner.py ở đây không được che module cùng tên); gói nạp qua PYTHONPATH
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _here]
    sys.exit(main())
