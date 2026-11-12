"""Tệp chạy dưới torchrun cho `linear_glori.py` của CheXFound NGUYÊN VĂN (PLAN_86 §12.2–12.3). Cần PYTHONPATH chứa repo CheXFound (commit ghim) và
thư mục gói vindr_protocol. Chạy: torchrun --nproc_per_node=1 <đường dẫn tệp này> <tham số y như linear_glori.py>.

Chỉ làm hai việc (thêm việc 3 ở 86j) trước khi gọi main() của họ (không sửa tệp nào của họ; các hàm được tra theo tên trong module lúc gọi nên bọc từ ngoài là đủ):
  1. đăng ký tên dataset 'VinDr': chexfound.data.loaders._parse_dataset_str được bọc — tên 'VinDr' -> vindr_dataset.parse_dataset_str, tên khác ->
     hàm gốc (make_dataset của họ tra hàm này theo tên trong module lúc gọi);
  2. GHI LẠI MỌI LẦN CHẤM của lượt học (Tuấn 12/11: dự đoán lấy thẳng từ lần chấm cuối có sẵn trong lượt học, không chạy --infer riêng):
     linear_glori.evaluate (= chexfound.eval.utils.evaluate, hàm evaluate_linear_classifiers gọi) được bọc — mỗi postprocessor (LinearPostprocessor
     của họ: preds = sigmoid(đầu đọc(đặc trưng))) được bọc để chép (preds, target) ĐÚNG như vừa đưa vào metric.update, rồi gọi evaluate gốc; số họ
     tính (metric.compute) ghi kèm. Phép tính, thứ tự gọi, chế độ train / eval của đầu đọc và RNG không đổi (chỉ chép tensor ra CPU).
     Mỗi lần chấm -> <output-dir>/cham/<thứ tự>_<tách>_iter<k>.npz (target + preds từng đầu) + .json (tên đầu, số họ tính, prefixstring).
  3. (86j, PLAN_86 §16; chỉ khi đặt biến VP_ENCODER) thay mô hình CheXFound bằng lớp bọc encoders_ho trong setup_and_build_model — vẫn setup(args) +
     autocast fp16 theo config của họ (inject_encoder); kết quả kiểm nạp trọng số của CHÍNH tiến trình học -> <output-dir>/nap_encoder.json, trượt -> dừng."""
import json
import os
import sys

EVAL_DIR = 'cham'
NAP_FILE = 'nap_encoder.json'     # 86j: kết quả kiểm nạp trọng số (encoders_ho.build) của tiến trình học — runner đọc lại, cổng + ghi vào kết quả


def register_vindr():
    '''Bọc chexfound.data.loaders._parse_dataset_str: 'VinDr' -> lớp của ta; tên khác -> hàm gốc của họ. Trả hàm gốc (để kiểm).'''
    import chexfound.data.loaders as loaders

    from vindr_protocol.upstream import vindr_dataset
    original = loaders._parse_dataset_str
    if getattr(original, '_vindr_wrapped', False):
        return original
    def parse(dataset_str):
        if dataset_str.split(':')[0] == vindr_dataset.NAME:
            return vindr_dataset.parse_dataset_str(dataset_str)
        return original(dataset_str)
    parse._vindr_wrapped = True
    loaders._parse_dataset_str = parse
    return original


def record_evaluations(linear_glori, out_dir):
    '''Bọc linear_glori.evaluate_linear_classifiers (lấy iteration / prefixstring của lần chấm) và linear_glori.evaluate (chép preds / target từng
       đầu + số họ tính). Trả (hàm evaluate_linear_classifiers gốc, hàm evaluate gốc).'''
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
            # .clone(): trên CPU .cpu() trả CHÍNH tensor đó (cùng bộ nhớ với tensor torchmetrics giữ) -> chép thật, không dùng chung
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


ENCODER_ENV = 'VP_ENCODER'       # 86j (PLAN_86 §16): tên lớp bọc trong encoders_ho; không đặt -> mô hình CheXFound của họ (86e / 86f / 86g)


def inject_encoder(linear_glori, name):
    '''86j: thay linear_glori.setup_and_build_model — VẪN gọi setup(args) (phân tán, log, config) và get_autocast_dtype(config) (fp16 theo config
       của họ) của chexfound.eval.setup, CHỈ thay mô hình bằng lớp bọc `name` (encoders_ho). main() của họ tra tên này trong module lúc gọi. Trả hàm gốc.'''
    import torch.backends.cudnn as cudnn
    from chexfound.eval import setup as ev_setup

    from vindr_protocol.upstream import encoders_ho
    original = linear_glori.setup_and_build_model

    def setup_and_build_model(args):
        cudnn.benchmark = True                                   # như hàm gốc
        config = ev_setup.setup(args)
        model = encoders_ho.build(name, check=False)
        if int(os.environ.get('RANK', '0')) == 0:                # ghi kết quả kiểm nạp TRƯỚC khi quyết (trượt vẫn để lại bằng chứng)
            with open(os.path.join(args.output_dir, NAP_FILE), 'w', encoding='utf-8') as f:
                json.dump(model.nap, f, ensure_ascii=False, indent=1)
        print('NAP_ENCODER ' + encoders_ho.loading_text(model.nap), flush=True)
        assert model.nap['dat'], f'{name}: nap trong so KHONG dat -> DUNG (xem {NAP_FILE})'
        model.cuda()
        return model, ev_setup.get_autocast_dtype(config)
    setup_and_build_model._encoder = name
    linear_glori.setup_and_build_model = setup_and_build_model
    return original


def main(argv=None):
    register_vindr()
    from chexfound.eval.classification import linear_glori
    args = linear_glori.get_args_parser(description='DINOv2 linear evaluation').parse_args(argv)
    assert not args.infer, '86e lay du doan tu cac lan cham trong luot hoc (Tuan 12/11) -> khong dung --infer'
    if os.environ.get(ENCODER_ENV):
        inject_encoder(linear_glori, os.environ[ENCODER_ENV])
    record_evaluations(linear_glori, os.path.join(args.output_dir, EVAL_DIR))
    return linear_glori.main(args)


if __name__ == '__main__':
    sys.exit(main())
