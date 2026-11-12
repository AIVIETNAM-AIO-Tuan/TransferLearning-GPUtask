# -*- coding: utf-8 -*-
"""Tệp chạy dưới torchrun cho `88c` Shenzhen (PLAN_88 §4.3): `linear_glori.py` của CheXFound @ 7e288d0 NGUYÊN VĂN.
PYTHONPATH: repo CheXFound (commit ghim) + thư mục chứa gói external_benchmarks + thư mục chứa gói vindr_protocol (`86`).
Chạy: torchrun --nproc_per_node=1 <tệp này> <tham số y như linear_glori.py>.

Không sửa tệp nào của họ; các tên dưới đây được tra trong module lúc gọi nên bọc từ ngoài là đủ. Trước khi gọi main() của họ:
  1. đăng ký 'ShenzhenL' (shenzhen_dataset): = lớp Shenzhen của họ, chỉ đổi giải mã PNG (convert('L') trước NumPy);
  2. chép mọi lần chấm bằng bộ ghi của `86e` (vindr_protocol.upstream.launch.record_evaluations): preds / target đúng như vào metric.update + số họ
     tính -> <out>/cham/;
  3. bọc NGOÀI bộ ghi đó (record_metric_state): sau mỗi lần chấm đọc TRẠNG THÁI preds / target của BinaryAUROC mà evaluate của họ vừa cập nhật
     (PLAN_88 §4.3 (7)) + dò ở lô đầu: dtype đầu ra Linear đầu tiên TRONG encoder (bằng chứng autocast fp16; đặc trưng ra qua LayerNorm nên là fp32
     dù autocast), dtype đặc trưng / đầu ra đầu đọc, đầu đọc đang train() hay eval() -> <out>/cham88/;
  4. đếm số lần mã họ sẽ xoá grad NaN (PLAN_88 §14.13 (9)): hook CHỈ ĐỌC trên tham số đầu đọc (bọc setup_linear_classifiers / setup_glori để gắn),
     cộng dồn trên thiết bị -> <out>/nan_grad.json cuối lượt;
  5. ghi mọi lần nạp checkpoint của fvcore (đường dẫn, khoá thiếu / thừa, md5 trọng số vừa nạp) -> <out>/cham88/nap_checkpoint.json: pha 1 không
     được nạp gì (thư mục trống), pha 2 nạp model_final.pth của pha 1 (PLAN_88 §4.3 (5), (10));
  6. môi trường: dtype autocast mã họ dùng, dtype tham số, phiên bản, GPU, cờ TF32 / cudnn, xFormers -> <out>/moi_truong_88.json;
  7. cuối lượt (kể cả khi lỗi / tràn bộ nhớ): bộ nhớ GPU đỉnh (max_memory_allocated / reserved) + lỗi nếu có -> <out>/bo_nho_88.json.
Phép tính, thứ tự gọi, chế độ train / eval và RNG không đổi (hook trả None; chỉ đọc .dtype / .training / chép ra CPU) — đối chứng sạch trong
test_external_benchmarks: chạy không có 3–6 -> khối kết quả họ ghi trùng từng số.

TUỲ CHỌN (chỉ khi biến môi trường GLORI_DUNG_CHUNG_DAU_VAO=1; `88a` đo, Tuấn chọn có dùng cho `88c` hay không — PLAN_88 §14.13 (1), §14.17):
  8. share_glori_input: trong MỘT lần AllClassifiers.forward, create_mldecoder_input của glori.py tính MỘT lần cho mọi đầu (mã họ: mỗi đầu một bản
     [B, N, 4D] fp32 giống hệt) -> bớt (số đầu − 1) bản sao; số đếm -> <out>/dung_chung_glori.json. Không đổi giá trị / thứ tự / RNG (kiểm từng số)."""
import os
import sys

if __name__ == '__main__':
    # Chạy như tệp: Python đặt thư mục tệp (external_benchmarks/) ở sys.path[0] -> config.py / datasets.py của gói sẽ CHE module cùng tên (vd thư viện
    # HF 'datasets' mà transformers dò) -> bỏ thư mục này trước mọi import khác; gói được import qua PYTHONPATH (thư mục cha).
    _here = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:] = [p for p in sys.path if os.path.normcase(os.path.abspath(p or '.')) != _here]

import hashlib
import json

EVAL88_DIR = 'cham88'
ENV_FILE = 'moi_truong_88.json'
NAN_FILE = 'nan_grad.json'
LOAD_FILE = 'nap_checkpoint.json'
MEM_FILE = 'bo_nho_88.json'
SHARE_FILE = 'dung_chung_glori.json'
SHARE_ENV = 'GLORI_DUNG_CHUNG_DAU_VAO'      # '1' -> bật share_glori_input (đặt bởi shenzhen_upstream.train_once(shared_input=True))
MARKER = '[88c] shenzhen_launch'


def _is_main():
    return int(os.environ.get('RANK', '0')) == 0


def _dump(path, obj):
    if _is_main():
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)


def state_md5(state_dict, prefix=''):
    '''md5 các tensor (khoá bắt đầu bằng prefix, đã bỏ prefix; thứ tự khoá sắp xếp) — dấu vân tay trọng số đầu đọc, so pha 1 / pha 2.'''
    h = hashlib.md5()
    for k in sorted(k for k in state_dict if k.startswith(prefix)):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(k[len(prefix):].encode())
        h.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def record_environment(linear_glori, path, extra=None):
    '''Bọc setup_and_build_model: gọi gốc, ghi dtype autocast + môi trường (+ extra), trả nguyên kết quả.'''
    orig = linear_glori.setup_and_build_model

    def setup_and_build_model(args):
        model, autocast_dtype = orig(args)
        import torch
        import torch.distributed as dist
        from chexfound.layers import attention, block, swiglu_ffn
        cuda = torch.cuda.is_available()
        _dump(path, dict(extra or {}, autocast_dtype=str(autocast_dtype), dtype_tham_so=str(next(model.parameters()).dtype), python=sys.version.split()[0],
                         torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if cuda else None,
                         cudnn=torch.backends.cudnn.version() if cuda else None, cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
                         tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32), tf32_cudnn=bool(torch.backends.cudnn.allow_tf32),
                         so_tien_trinh=dist.get_world_size() if dist.is_initialized() else 1,
                         xformers_cua_ho={n: bool(getattr(m, 'XFORMERS_AVAILABLE', False)) for n, m in
                                          (('attention', attention), ('block', block), ('swiglu_ffn', swiglu_ffn))}))
        return model, autocast_dtype

    linear_glori.setup_and_build_model = setup_and_build_model
    return orig


def count_nan_grads(linear_glori):
    '''Bọc setup_linear_classifiers / setup_glori: gắn hook CHỈ ĐỌC lên mọi tham số đầu đọc vừa dựng (trả None -> gradient không đổi). Mỗi lần
       dựng (pha 1, pha 2) -> {tên đầu: bộ đếm (bước, tham số) có NaN} trên thiết bị. -> danh sách các lần dựng (đọc cuối lượt).'''
    import torch
    phases = []

    def make_hook(cnt):
        def hook(g):
            cnt.add_(torch.isnan(g).any())
            return None
        return hook

    def wrap(fn):
        def setup(*args, **kwargs):
            classifiers, groups = fn(*args, **kwargs)
            heads = getattr(classifiers, 'module', classifiers).classifiers_dict
            counters = {}
            for name, head in heads.items():
                params = [p for p in head.parameters() if p.requires_grad]
                cnt = torch.zeros((), dtype=torch.int64, device=params[0].device)
                for p in params:
                    p.register_hook(make_hook(cnt))
                counters[name] = cnt
            phases.append(dict(ham=fn.__name__, dau=counters))
            return classifiers, groups
        setup.__name__ = fn.__name__
        return setup

    linear_glori.setup_linear_classifiers = wrap(linear_glori.setup_linear_classifiers)
    linear_glori.setup_glori = wrap(linear_glori.setup_glori)
    return phases


def record_checkpoint_loads(path):
    '''Bọc fvcore Checkpointer.load / _load_model (lớp mã họ dùng): ghi đường dẫn nạp, checkpointables, khoá thiếu / thừa, md5 trọng số mô hình
       ngay sau khi nạp. Gốc được gọi nguyên vẹn.'''
    from fvcore.common import checkpoint as fc
    orig_load, orig_lm = fc.Checkpointer.load, fc.Checkpointer._load_model
    loads = []

    def load(self, path_, checkpointables=None):
        loads.append(dict(tep=os.path.basename(path_) if path_ else '', thu_muc_tep=os.path.basename(os.path.dirname(path_)) if path_ else '',
                          thu_muc_luu=os.path.basename(os.path.normpath(self.save_dir)),
                          checkpointables=None if checkpointables is None else list(checkpointables), nap_mo_hinh=False))
        _dump(path, loads)
        return orig_load(self, path_, checkpointables)

    def _load_model(self, checkpoint):
        inc = orig_lm(self, checkpoint)
        sd = self.model.state_dict()
        loads[-1].update(nap_mo_hinh=True, so_khoa_mo_hinh=len(sd), thieu=list(inc.missing_keys), so_thua=len(inc.unexpected_keys),
                         sai_hinh=[list(map(str, x)) for x in getattr(inc, 'incorrect_shapes', [])], md5_sau_nap=state_md5(sd),
                         ten_dau=sorted({k.split('.')[1] for k in sd if k.startswith('classifiers_dict.')}))
        _dump(path, loads)
        return inc

    fc.Checkpointer.load, fc.Checkpointer._load_model = load, _load_model
    return loads


def record_metric_state(linear_glori, out_dir, eval_dir_86):
    '''Bọc NGOÀI evaluate (= bộ ghi 86e đã bọc): (a) bọc postprocessor bằng bộ dò (lô đầu: dtype đặc trưng tầng cuối / token / dự đoán, hình dự đoán,
       đầu đọc .training); (b) sau khi evaluate của họ xong: đọc trạng thái preds / target của metric (torchmetrics BinaryAUROC, thresholds=None giữ đủ)
       -> <stem>_metric.npz; dò -> <stem>.json. <stem> = tên lần chấm bộ ghi 86e vừa ghi (cùng thứ tự gọi).'''
    import numpy as np
    import torch
    os.makedirs(out_dir, exist_ok=True)
    inner = linear_glori.evaluate
    n = [0]

    class _Probe(torch.nn.Module):
        def __init__(self, post, sink):
            super().__init__()
            self.inner, self.sink = post, sink

        def forward(self, samples, targets):
            out = self.inner(samples, targets)
            if not self.sink:
                last = samples[0][-1]                    # samples = [tuple n_khối x (patch, cls)] (mã họ: một góc nhìn)
                head = getattr(self.inner, 'linear_classifier', None)
                self.sink.update(dac_trung=str(last[0].dtype), token=str(last[1].dtype), du_doan=str(out['preds'].dtype),
                                 hinh_du_doan=list(out['preds'].shape), dau_train=None if head is None else bool(head.training))
            return out

    def evaluate(model, data_loader, postprocessors, metrics, device, *args, **kwargs):
        probes, enc = {k: {} for k in postprocessors}, {}
        # dtype phép tính TRONG encoder: Linear đầu tiên của backbone (đặc trưng ra đã qua LayerNorm -> fp32 dù autocast fp16, không chứng minh được
        # gì); hook chỉ đọc, gỡ ngay sau lần chấm (các bước học không có hook)
        backbone = getattr(model, 'feature_model', None)
        lin = next((m for m in backbone.modules() if isinstance(m, torch.nn.Linear)), None) if backbone is not None else None

        def enc_hook(mod, inp, out):
            if not enc:
                enc.update(linear_ma_hoa=str(out.dtype))

        handle = lin.register_forward_hook(enc_hook) if lin is not None else None
        try:
            res = inner(model, data_loader, {k: _Probe(p, probes[k]) for k, p in postprocessors.items()}, metrics, device, *args, **kwargs)
        finally:
            if handle is not None:
                handle.remove()
        names = list(postprocessors)
        stems = [f[:-len('.json')] for f in sorted(os.listdir(eval_dir_86)) if f.startswith(f'{n[0]:03d}_') and f.endswith('.json')]
        assert len(stems) == 1, f'lan cham {n[0]}: {len(stems)} ban ghi 86e khop -> DUNG'
        state = {}
        for j, k in enumerate(names):
            mets = [m for _, m in metrics[k].items(keep_base=True, copy_state=False)]
            assert len(mets) == 1 and hasattr(mets[0], 'preds'), f'metric cua {k}: {len(mets)} metric / khong co trang thai preds -> DUNG'
            m = mets[0]
            state[f'preds_{j}'] = torch.cat([t.flatten() for t in m.preds]).float().cpu().numpy()
            if j == 0:
                state['target'] = torch.cat([t.flatten() for t in m.target]).cpu().numpy()
        if _is_main():
            np.savez_compressed(os.path.join(out_dir, stems[0] + '_metric.npz'), **state)
        _dump(os.path.join(out_dir, stems[0] + '.json'), dict(thu_tu=n[0], ten=names, metric=type(m).__name__, do=probes, ma_hoa=enc))
        n[0] += 1
        return res

    linear_glori.evaluate = evaluate
    return inner


def share_glori_input():
    '''TUỲ CHỌN (SHARE_ENV=1). GLoRI.forward của họ gọi create_mldecoder_input(x[0]) RIÊNG cho từng đầu (glori.py:164) -> mỗi đầu giữ một bản nối
       [B, N, 4D] fp32 giống hệt tới backward. Trong MỘT lần AllClassifiers.forward (bước học: mọi đầu nhận CÙNG đối tượng danh sách đặc trưng), tính
       một lần rồi trả CÙNG tensor cho mọi đầu; ra khỏi forward thì bỏ tham chiếu (autograd của từng đầu vẫn giữ tensor tới backward như cũ). Ngoài
       AllClassifiers.forward (lúc chấm: postprocessor gọi từng đầu riêng) -> hàm gốc, như mã họ. Không đầu nào sửa tại chỗ x / cls (embed_standart tạo
       tensor mới; relu inplace trên tensor mới đó; cls.unsqueeze().repeat() tạo mới) -> giá trị, dtype, hình, thứ tự phép tính, RNG không đổi.
       Khoá = CHÍNH đối tượng danh sách (so `is`, giữ tham chiếu trong lúc forward -> id không bị dùng lại) + use_n_blocks. -> bộ đếm dict.'''
    from chexfound.eval.classification import glori
    from chexfound.eval.classification import utils as cls_utils
    orig_cmi, orig_fwd = glori.create_mldecoder_input, cls_utils.AllClassifiers.forward
    st = dict(active=False, ref=None, n=None, val=None, goi_goc=0, dung_lai=0, forward=0)

    def create_mldecoder_input(x_tokens_list, use_n_blocks):
        if not st['active']:
            return orig_cmi(x_tokens_list, use_n_blocks)
        if st['ref'] is x_tokens_list and st['n'] == use_n_blocks:
            st['dung_lai'] += 1
            return st['val']
        val = orig_cmi(x_tokens_list, use_n_blocks)
        st.update(ref=x_tokens_list, n=use_n_blocks, val=val)
        st['goi_goc'] += 1
        return val

    def forward(self, inputs, return_attention=False):
        st['active'] = True
        st['forward'] += 1
        try:
            return orig_fwd(self, inputs, return_attention=return_attention)
        finally:
            st.update(active=False, ref=None, n=None, val=None)

    glori.create_mldecoder_input = create_mldecoder_input
    cls_utils.AllClassifiers.forward = forward
    return st


def record_memory(path, error=None):
    '''Bộ nhớ GPU đỉnh của tiến trình (max_memory_allocated / reserved, GiB) + lỗi (kiểu + 500 ký tự đầu) -> path. CPU: gpu None.'''
    import torch
    out = dict(gpu=None)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        out = dict(gpu=props.name, tong_gb=props.total_memory / 2 ** 30, dinh_cap_phat_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                   dinh_du_tru_gb=torch.cuda.max_memory_reserved() / 2 ** 30)
    if error is not None:
        out['loi'] = f'{type(error).__name__}: {str(error)[:500]}'
    _dump(path, out)


def main(argv=None):
    from external_benchmarks import __version__
    from external_benchmarks.config import DATASET_NAME
    from external_benchmarks.shenzhen_dataset import register_shenzhen
    register_shenzhen()
    from chexfound.eval.classification import linear_glori
    from vindr_protocol.upstream.launch import EVAL_DIR, record_evaluations
    args = linear_glori.get_args_parser(description='DINOv2 linear evaluation').parse_args(argv)
    assert not args.infer, '88c lay du doan tu lan cham cuoi trong luot hoc (PLAN_88 §4.3 (7)) -> khong dung --infer'
    for s in (args.train_dataset_str, args.val_dataset_str, args.test_dataset_str):
        assert s and s.split(':')[0] == DATASET_NAME, f'chuoi dataset {s!r}: ca 3 tach phai la {DATASET_NAME} -> DUNG'
    out = args.output_dir
    os.makedirs(os.path.join(out, EVAL88_DIR), exist_ok=True)
    shared = os.environ.get(SHARE_ENV) == '1'
    assert not shared or args.glori, f'{SHARE_ENV}=1 chi danh cho luot GLoRI (--glori) -> DUNG'
    print(f'{MARKER} {__version__}: ShenzhenL (convert L) + ghi lan cham / trang thai metric / grad NaN / nap checkpoint / bo nho'
          + (' + DUNG CHUNG dau vao GLoRI' if shared else ''), flush=True)
    record_environment(linear_glori, os.path.join(out, ENV_FILE), extra=dict(glori_dung_chung_dau_vao=shared))
    phases = count_nan_grads(linear_glori)
    record_checkpoint_loads(os.path.join(out, EVAL88_DIR, LOAD_FILE))
    record_evaluations(linear_glori, os.path.join(out, EVAL_DIR))
    record_metric_state(linear_glori, os.path.join(out, EVAL88_DIR), os.path.join(out, EVAL_DIR))
    share = share_glori_input() if shared else None
    error = None
    try:
        return linear_glori.main(args)
    except BaseException as e:            # tràn bộ nhớ / lỗi: ghi bộ nhớ đỉnh + lỗi rồi NÉM LẠI (không nuốt lỗi)
        error = e
        raise
    finally:
        for fn in (lambda: _dump(os.path.join(out, NAN_FILE), [dict(lan_dung=i + 1, ham=ph['ham'], dau={k: int(c.item()) for k, c in ph['dau'].items()})
                                                                for i, ph in enumerate(phases)]),
                   lambda: record_memory(os.path.join(out, MEM_FILE), error),
                   lambda: _dump(os.path.join(out, SHARE_FILE), dict(bat=shared, **({k: share[k] for k in ('goi_goc', 'dung_lai', 'forward')}
                                                                                     if share else {})))):
            try:                          # một phần ghi lỗi (vd CUDA hỏng sau tràn) không được che lỗi gốc / bỏ các phần ghi khác
                fn()
            except Exception as e2:
                print(f'{MARKER}: ghi cuoi luot loi {type(e2).__name__}: {e2}', flush=True)


if __name__ == '__main__':
    sys.exit(main())
