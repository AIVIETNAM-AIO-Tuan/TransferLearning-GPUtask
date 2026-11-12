# -*- coding: utf-8 -*-
"""86h (PLAN_86 §15): danh sách phép tính THẬT trên GPU, mỗi phép so với CPU fp32 của CÙNG torch — chạy bằng Python của MỘT môi trường (môi trường
của họ: Python 3.9 + torch 2.0.0+cu117 + xFormers 0.0.18; đối chứng sạch: Python của notebook, thư viện mới) trong tiến trình riêng. Ghi từng dòng
JSON (mất tiến trình giữa chừng vẫn còn phần đã đo). Không import vindr_protocol; chỉ cú pháp / API có ở Python 3.9 + torch 2.0.

    python probe_ops.py OUT.jsonl [--device cuda|cpu] [--fail TEN]

--device cpu: CHỈ để kiểm bộ khung trên máy không GPU (phép tính "thiết bị" chạy trên CPU, fp16 -> bf16, NCCL -> gloo). --fail TEN: phép tính TEN ném
lỗi giả (kiểm đường bắt lỗi). Ngưỡng (PLAN_86 §15.2): sai số tương đối = max |Δ| / max |tham chiếu|; fp32 ≤ 1e-4 (TF32 tắt), fp16 / autocast ≤ 2e-2,
dropout tỉ lệ giữ 0,9 ± 0,01, AdamW sau một bước ≤ 1e-4, grad DDP = không DDP ≤ 1e-6."""
import importlib, importlib.util, json, os, re, socket, sys, time, traceback, warnings

TOL_FP32, TOL_FP16 = 1e-4, 2e-2
TOL_DROPOUT, TOL_ADAMW, TOL_DDP = 0.01, 1e-4, 1e-6
P_DROP = 0.1
_T = None          # torch (import trong main để bắt cảnh báo lúc khởi tạo CUDA)


def _g(seed):
    return _T.Generator().manual_seed(seed)


def _rn(*shape, seed):
    return _T.randn(*shape, generator=_g(seed))


def _rel(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    assert a.shape == b.shape, f'hinh khac {tuple(a.shape)} / {tuple(b.shape)}'
    return float((a - b).abs().max() / max(float(b.abs().max()), 1e-12))


def _sync(dev):
    if dev == 'cuda': _T.cuda.synchronize()


# ---------------------------------------------------------------------------------------------------------------- phép tính (dev, H) -> (lệch, ngưỡng)
def op_tao_tensor(dev, H):
    gd = _T.Generator(device=dev).manual_seed(0)
    x = _T.randn(1000, 1000, device=dev, generator=gd)
    z = _T.arange(10, device=dev).float() + _T.zeros(10, device=dev)
    _sync(dev)
    ok = bool(_T.isfinite(x).all()) and abs(float(x.mean())) < 0.01 and abs(float(x.std()) - 1.0) < 0.01 and float(z.sum()) == 45.0
    return (0.0 if ok else 1.0), 0.5


def op_matmul_fp32(dev, H):
    A, B = _rn(512, 1024, seed=1), _rn(1024, 768, seed=2)
    return _rel(A.to(dev) @ B.to(dev), A @ B), TOL_FP32


def op_matmul_fp16(dev, H):
    A, B = _rn(512, 1024, seed=1).to(H), _rn(1024, 768, seed=2).to(H)
    return _rel(A.to(dev) @ B.to(dev), A.float() @ B.float()), TOL_FP16


def op_bmm_fp16(dev, H):
    A, B = _rn(8, 257, 64, seed=3).to(H), _rn(8, 64, 257, seed=4).to(H)
    return _rel(_T.bmm(A.to(dev), B.to(dev)), _T.bmm(A.float(), B.float())), TOL_FP16


def _linear_fb(dev, W, b, x, w_out):
    W, b, x = (t.clone().to(dev).requires_grad_(True) for t in (W, b, x))
    y = _T.nn.functional.linear(x, W, b)
    (y * w_out.to(dev)).sum().backward()
    return [y, x.grad, W.grad, b.grad]


def op_linear_xuoi_nguoc(dev, H):
    W, b, x, w_out = _rn(768, 1024, seed=5) * 0.02, _rn(768, seed=6), _rn(64, 1024, seed=7), _rn(64, 768, seed=8)
    return max(_rel(u, v) for u, v in zip(_linear_fb(dev, W, b, x, w_out), _linear_fb('cpu', W, b, x, w_out))), TOL_FP32


def _patch_inputs():
    return _rn(2, 3, 224, 224, seed=9), _rn(64, 3, 16, 16, seed=10) * 0.02, _rn(64, seed=11)


def op_conv2d_fp32(dev, H):
    x, W, b = _patch_inputs()
    F = _T.nn.functional
    return _rel(F.conv2d(x.to(dev), W.to(dev), b.to(dev), stride=16), F.conv2d(x, W, b, stride=16)), TOL_FP32


def op_conv2d_autocast(dev, H):
    x, W, b = _patch_inputs()
    F = _T.nn.functional
    with _T.autocast(device_type=dev, dtype=H):
        y = F.conv2d(x.to(dev), W.to(dev), b.to(dev), stride=16)
    return _rel(y, F.conv2d(x, W, b, stride=16)), TOL_FP16


def _elementwise(fn, shape, seed):
    x, w = _rn(*shape, seed=seed), _rn(*shape, seed=seed + 1)

    def run(dev):
        t = x.clone().to(dev).requires_grad_(True)
        y = fn(t)
        (y * w.to(dev)).sum().backward()
        return [y, t.grad]
    return run


def _cmp_runs(run, dev):
    return max(_rel(u, v) for u, v in zip(run(dev), run('cpu')))


def op_layernorm(dev, H):
    g, b = _rn(768, seed=21), _rn(768, seed=22)
    run = _elementwise(lambda t: _T.nn.functional.layer_norm(t, (768,), g.to(t.device), b.to(t.device), 1e-6), (16, 257, 768), 23)
    return _cmp_runs(run, dev), TOL_FP32


def op_softmax(dev, H):
    return _cmp_runs(_elementwise(lambda t: _T.softmax(t, -1), (16, 257, 257), 25), dev), TOL_FP32


def op_gelu(dev, H):
    return _cmp_runs(_elementwise(_T.nn.functional.gelu, (16, 257, 768), 27), dev), TOL_FP32


def op_silu(dev, H):
    return _cmp_runs(_elementwise(_T.nn.functional.silu, (16, 257, 768), 29), dev), TOL_FP32


def op_noi_suy_bicubic(dev, H):
    x = _rn(1, 768, 37, 37, seed=31)

    def f(t):
        return _T.nn.functional.interpolate(t, size=(32, 32), mode='bicubic', align_corners=False)
    return _rel(f(x.to(dev)), f(x)), TOL_FP32


def _mha_state():
    with _T.random.fork_rng(devices=[]):
        _T.manual_seed(41)
        return _T.nn.MultiheadAttention(768, 8, dropout=0.0).state_dict()


def op_mha_train(dev, H):
    st, q0, kv0, w = _mha_state(), _rn(28, 4, 768, seed=42), _rn(1024, 4, 768, seed=43), _rn(28, 4, 768, seed=44)

    def run(d):
        m = _T.nn.MultiheadAttention(768, 8, dropout=0.0)
        m.load_state_dict(st)
        m = m.to(d).train()
        q, kv = q0.clone().to(d).requires_grad_(True), kv0.clone().to(d).requires_grad_(True)
        out = m(q, kv, kv)[0]                      # như ML-Decoder / GLoRI: query = lớp, key = value = token ảnh, need_weights mặc định
        (out * w.to(d)).sum().backward()
        return [out, q.grad, kv.grad, m.in_proj_weight.grad, m.out_proj.weight.grad]
    return max(_rel(u, v) for u, v in zip(run(dev), run('cpu'))), TOL_FP32


def op_mha_eval_duong_nhanh(dev, H):
    with _T.random.fork_rng(devices=[]):
        _T.manual_seed(45)
        st = _T.nn.MultiheadAttention(768, 8, batch_first=True).state_dict()
    x = _rn(4, 257, 768, seed=46)

    def run(d):
        m = _T.nn.MultiheadAttention(768, 8, batch_first=True)
        m.load_state_dict(st)
        m = m.to(d).eval()
        with _T.no_grad():
            t = x.to(d)
            return m(t, t, t, need_weights=False)[0]
    return _rel(run(dev), run('cpu')), TOL_FP32


def op_dropout(dev, H):
    _T.manual_seed(51)
    y = _T.nn.functional.dropout(_T.ones(4096, 1024, device=dev), p=P_DROP, training=True)
    kept = y[y != 0]
    keep = float(kept.numel()) / y.numel()
    vals_ok = bool(_T.allclose(kept, _T.full_like(kept, 1.0 / (1.0 - P_DROP))))
    return (abs(keep - (1.0 - P_DROP)) if vals_ok else 1.0), TOL_DROPOUT


def op_bce_nguoc(dev, H):
    W, b, x = _rn(28, 1024, seed=61) * 0.02, _rn(28, seed=62) * 0.1, _rn(256, 1024, seed=63)
    y = (_T.rand(256, 28, generator=_g(64)) < 0.3).float()

    def run(d):
        Wd, bd = W.clone().to(d).requires_grad_(True), b.clone().to(d).requires_grad_(True)
        loss = _T.nn.functional.binary_cross_entropy_with_logits(_T.nn.functional.linear(x.to(d), Wd, bd), y.to(d), reduction='sum')
        loss.backward()
        return [loss.reshape(1), Wd.grad, bd.grad]
    return max(_rel(u, v) for u, v in zip(run(dev), run('cpu'))), TOL_FP32


def op_adamw_buoc(dev, H):
    W, b = _rn(28, 1024, seed=71) * 0.02, _rn(28, seed=72) * 0.1
    gW = _rn(28, 1024, seed=73).sign() * (0.1 + _T.rand(28, 1024, generator=_g(74)))      # grad cố định, |g| >= 0,1 (bước đầu Adam ~ dấu g)
    gb = _rn(28, seed=75).sign() * (0.1 + _T.rand(28, generator=_g(76)))

    def run(d):
        Wd, bd = _T.nn.Parameter(W.clone().to(d)), _T.nn.Parameter(b.clone().to(d))
        opt = _T.optim.AdamW([Wd, bd], lr=1e-3, weight_decay=1e-4)
        for _ in range(3):
            Wd.grad, bd.grad = gW.clone().to(d), gb.clone().to(d)
            opt.step()
        return [Wd.detach(), bd.detach()]
    return max(_rel(u, v) for u, v in zip(run(dev), run('cpu'))), TOL_ADAMW


def _block(x, P):
    F = _T.nn.functional
    Bn, Tn, D = x.shape
    h = F.layer_norm(x, (D,), P['g1'], P['b1'])
    qkv = F.linear(h, P['Wqkv'], P['bqkv']).reshape(Bn, Tn, 3, 8, D // 8).permute(2, 0, 3, 1, 4)
    a = _T.softmax((qkv[0] @ qkv[1].transpose(-2, -1)) / (D // 8) ** 0.5, -1) @ qkv[2]
    x = x + F.linear(a.transpose(1, 2).reshape(Bn, Tn, D), P['Wo'], P['bo'])
    h = F.layer_norm(x, (D,), P['g2'], P['b2'])
    return x + F.linear(F.gelu(F.linear(h, P['W1'], P['c1'])), P['W2'], P['c2'])


def op_khoi_transformer_autocast(dev, H):
    D = 768
    shapes = dict(g1=(D,), b1=(D,), Wqkv=(3 * D, D), bqkv=(3 * D,), Wo=(D, D), bo=(D,), g2=(D,), b2=(D,), W1=(4 * D, D), c1=(4 * D,), W2=(D, 4 * D),
                  c2=(D,))
    P = {k: _rn(*s, seed=80 + i) * (0.02 if len(s) == 2 else 0.1) + (1.0 if k in ('g1', 'g2') else 0.0) for i, (k, s) in enumerate(shapes.items())}
    x = _rn(2, 257, D, seed=99)
    with _T.autocast(device_type=dev, dtype=H):
        y = _block(x.to(dev), {k: v.to(dev) for k, v in P.items()})
    return _rel(y, _block(x, P)), TOL_FP16


def op_sap_xep_tich_luy(dev, H):
    x, t = _rn(100000, seed=101), (_T.rand(100000, generator=_g(102)) < 0.1).long()

    def f(xx, tt):                         # các bước torchmetrics dùng khi chấm AUROC / AP trên GPU (sắp giảm dần, cumsum, chỗ đổi giá trị, unique)
        idx = _T.argsort(xx, descending=True)
        xs, ts = xx[idx], tt[idx]
        d = _T.where(xs[1:] - xs[:-1] != 0)[0]
        return [xs, _T.cumsum(ts, 0).float(), _T.cumsum(1 - ts, 0).float(), xx.sum().reshape(1), xx.max().reshape(1), _T.unique(tt).float(),
                d[:1000].float()]
    return max(_rel(u, v) for u, v in zip(f(x.to(dev), t.to(dev)), f(x, t))), TOL_FP32


def _attn_ref(q, k, v):                    # (B, M, H, K) như xFormers, fp32
    a = _T.softmax(q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / q.shape[-1] ** 0.5, -1) @ v.transpose(1, 2)
    return a.transpose(1, 2)


def op_xformers_mea(dev, H):
    from xformers.ops import memory_efficient_attention
    q, k, v = (_rn(2, 257, 16, 64, seed=s).half() for s in (111, 112, 113))
    o = memory_efficient_attention(q.to(dev), k.to(dev), v.to(dev))
    return _rel(o, _attn_ref(q.float(), k.float(), v.float())), TOL_FP16


def op_sdpa_fp16(dev, H):
    q, k, v = (_rn(2, 16, 257, 64, seed=s).to(H) for s in (121, 122, 123))
    o = _T.nn.functional.scaled_dot_product_attention(q.to(dev), k.to(dev), v.to(dev))
    ref = _T.softmax(q.float() @ k.float().transpose(-1, -2) / 8.0, -1) @ v.float()
    return _rel(o, ref), TOL_FP16


def op_nccl_ddp(dev, H):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    dist.init_process_group('nccl' if dev == 'cuda' else 'gloo', init_method=f'tcp://127.0.0.1:{port}', rank=0, world_size=1)
    try:
        t = _T.ones(4, device=dev)
        dist.all_reduce(t)
        dist.broadcast(t, 0)
        _sync(dev)
        if float(t.sum()) != 4.0: return 1.0, TOL_DDP
        W, x = _rn(8, 64, seed=131), _rn(16, 64, seed=132)

        def grad(wrap):
            m = _T.nn.Linear(64, 8, bias=False)
            with _T.no_grad(): m.weight.copy_(W)
            m = m.to(dev)
            mm = DDP(m, device_ids=[0]) if (wrap and dev == 'cuda') else (DDP(m) if wrap else m)
            mm(x.to(dev)).pow(2).sum().backward()
            return m.weight.grad.detach().cpu()
        return float((grad(True) - grad(False)).abs().max()), TOL_DDP
    finally:
        dist.destroy_process_group()


# gate = đường mã của họ dùng tới (luật §15.2 (2)); sdpa chỉ ghi thông tin
OPS = (('tao_tensor', True, op_tao_tensor), ('matmul_fp32', True, op_matmul_fp32), ('matmul_fp16', True, op_matmul_fp16),
       ('bmm_fp16', True, op_bmm_fp16), ('linear_xuoi_nguoc', True, op_linear_xuoi_nguoc), ('conv2d_fp32', True, op_conv2d_fp32),
       ('conv2d_autocast', True, op_conv2d_autocast), ('layernorm', True, op_layernorm), ('softmax', True, op_softmax), ('gelu', True, op_gelu),
       ('silu', True, op_silu), ('noi_suy_bicubic', True, op_noi_suy_bicubic), ('mha_train', True, op_mha_train),
       ('mha_eval_duong_nhanh', True, op_mha_eval_duong_nhanh), ('dropout', True, op_dropout), ('bce_nguoc', True, op_bce_nguoc),
       ('adamw_buoc', True, op_adamw_buoc), ('khoi_transformer_autocast', True, op_khoi_transformer_autocast),
       ('sap_xep_tich_luy', True, op_sap_xep_tich_luy), ('xformers_mea', True, op_xformers_mea), ('nccl_ddp', True, op_nccl_ddp),
       ('sdpa_fp16', False, op_sdpa_fp16))


def _info(dev):
    v = {}
    for m in ('torch', 'torchvision', 'xformers', 'numpy'):
        try:
            v[m] = getattr(importlib.import_module(m), '__version__', 'co')
        except Exception as e:                      # xFormers cài mà import lỗi: ghi lỗi, không coi là "không có"
            v[m] = None if importlib.util.find_spec(m) is None else f'LOI IMPORT: {type(e).__name__}: {e}'[:300]
    try:
        cfg = _T.__config__.show()
    except Exception:
        cfg = ''
    nvcc = next((l.split(':', 1)[1].strip() for l in cfg.splitlines() if 'NVCC architecture flags' in l), '')
    info = dict(python=sys.version.split()[0], phien_ban=v, cuda=_T.version.cuda, thiet_bi=dev, nvcc_co=nvcc[:600],
                nvcc_sass=sorted(set(re.findall(r'code=sm_(\d+)', nvcc)), key=int), nvcc_ptx=sorted(set(re.findall(r'code=compute_(\d+)', nvcc)), key=int),
                cuda_module_loading=os.environ.get('CUDA_MODULE_LOADING'), cuda_cache_maxsize=os.environ.get('CUDA_CACHE_MAXSIZE'),
                cuda_dung_duoc=bool(_T.cuda.is_available()))
    if info['cuda_dung_duoc']:
        p = _T.cuda.get_device_properties(0)
        info.update(gpu=_T.cuda.get_device_name(0), capability=list(_T.cuda.get_device_capability(0)), bo_nho_gb=p.total_memory / 2 ** 30,
                    arch_list=list(_T.cuda.get_arch_list()), cudnn=_T.backends.cudnn.version())
        try:
            nv = _T.cuda.nccl.version()
            info['nccl'] = '.'.join(map(str, nv)) if isinstance(nv, (tuple, list)) else str(nv)
        except Exception as e:
            info['nccl'] = f'loi: {type(e).__name__}: {e}'[:200]
        t0 = time.time()
        try:
            _T.empty(1, device='cuda')               # khởi tạo ngữ cảnh, không chạy kernel
            _T.cuda.synchronize()
            info['ms_khoi_tao'] = (time.time() - t0) * 1e3
        except Exception as e:
            info['loi_khoi_tao'] = f'{type(e).__name__}: {e}'[:500]
    return info


def main():
    global _T
    a = sys.argv[1:]
    out = a[0]
    dev = a[a.index('--device') + 1] if '--device' in a else 'cuda'
    fail = a[a.index('--fail') + 1] if '--fail' in a else None
    assert dev in ('cuda', 'cpu'), dev
    t_start = time.time()
    f = open(out, 'w', encoding='utf-8')

    def emit(rec):
        f.write(json.dumps(rec) + '\n')
        f.flush()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        import torch
        _T = torch
        info = _info(dev)
    info['canh_bao'] = sorted({str(w.message)[:300] for w in caught if 'capab' in str(w.message).lower() or 'compatib' in str(w.message).lower()})
    emit(dict(loai='thong_tin', **info))
    if dev == 'cuda' and (not info['cuda_dung_duoc'] or info.get('loi_khoi_tao')):
        emit(dict(loai='xong', giay=time.time() - t_start, ly_do='khong dung duoc CUDA'))
        return
    torch.backends.cuda.matmul.allow_tf32 = False          # tham chiếu fp32 đúng nghĩa (TF32 tắt) — chỉ API cũ, có ở torch 2.0 lẫn bản mới
    torch.backends.cudnn.allow_tf32 = False
    H = torch.float16 if dev == 'cuda' else torch.bfloat16
    has_xf = importlib.util.find_spec('xformers') is not None
    for name, gate, fn in OPS:
        rec = dict(loai='phep_tinh', ten=name, gate=gate)
        if name == 'xformers_mea' and not has_xf:
            rec.update(trang_thai='khong_co')
        elif name == 'xformers_mea' and dev != 'cuda':
            rec.update(trang_thai='bo_qua_cpu')
        else:
            try:
                if fail == name: raise RuntimeError(f'loi gia (--fail) o {name}')
                t0 = time.time()
                lech, nguong = fn(dev, H)
                _sync(dev)
                t1 = time.time()
                fn(dev, H)
                _sync(dev)
                t2 = time.time()
                rec.update(trang_thai='chay', lech=lech, nguong=nguong, dat=bool(lech == lech and lech <= nguong), ms_lan_dau=(t1 - t0) * 1e3,
                           ms_lan_hai=(t2 - t1) * 1e3)
            except Exception as e:
                rec.update(trang_thai='loi', dat=False, loi=f'{type(e).__name__}: {e}'[:800], vet=traceback.format_exc()[-1500:])
        emit(rec)
    emit(dict(loai='xong', giay=time.time() - t_start))
    f.close()


if __name__ == '__main__':
    main()
