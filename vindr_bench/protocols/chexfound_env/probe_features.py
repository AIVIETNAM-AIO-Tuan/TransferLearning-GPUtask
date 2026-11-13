# -*- coding: utf-8 -*-
"""Tiến trình đo đặc trưng của lớp bọc (cổng số học PLAN_89 §8 mục 4) — chạy bằng Python của môi trường cũ (của họ) hoặc mới (notebook), cùng đầu vào.
Chép ý của vindr_protocol/upstream/probe_features.py @ 0.8.2 (86j), dùng lớp bọc vindr_bench.encoders.

  1) tạo đầu vào (môi trường cũ, cần --repo):
     python probe_features.py --make-input --repo R --csv-root C --size S --n 4 --out X.npy
     -> n ảnh ĐẦU của tách train qua lớp VinDr (dataset.py) + make_classification_eval_transform(resize_size=S, crop_size=S) CỦA HỌ (Resize + CenterCrop
        + RescaleImage + chuẩn hoá ImageNet) = đúng tensor lượt học đưa vào lớp bọc lúc chấm.
  2) đặc trưng:
     python probe_features.py --encoder TEN --input X.npy --out Y.npz --amp fp16|none [--device cuda|cpu] [--attn eager]
     -> get_intermediate_layers(x, 4, return_class_token=True) dưới inference_mode (+ autocast fp16 nếu --amp fp16, như ModelWithIntermediateLayers của
        họ); --amp none: fp32, TF32 tắt (tham chiếu). Lưu patch [4, B, T, D] + cls [4, B, D] + info (phiên bản, nạp trọng số THẬT của tiến trình này).
        --attn chỉ cho RAD-DINO ở môi trường mới (eager = thuật toán duy nhất của transformers 4.36.2).
Gói nạp qua PYTHONPATH (thư mục chứa vindr_bench). Chỉ cú pháp / API có ở Python 3.9 + torch 2.0."""
import json
import os
import sys
import time
import warnings


def _arg(a, name, default=None):
    return a[a.index(name) + 1] if name in a else default


def make_input(a):
    repo, csv_root, out = _arg(a, '--repo'), _arg(a, '--csv-root'), _arg(a, '--out')
    size, n = int(_arg(a, '--size')), int(_arg(a, '--n', '4'))
    sys.path.insert(0, repo)
    import numpy as np
    from chexfound.data.transforms import make_classification_eval_transform

    from vindr_bench.protocols.chexfound_env.dataset import VinDr, _Split
    ds = VinDr(split=_Split.TRAIN, root=csv_root, transform=make_classification_eval_transform(resize_size=size, crop_size=size))
    x = np.stack([ds[i][0].numpy() for i in range(n)]).astype(np.float32)
    np.save(out, x)
    print('DV=' + json.dumps(dict(so_anh=int(x.shape[0]), hinh=list(x.shape), python=sys.version.split()[0])))


def features(a):
    encoder, inp, out = _arg(a, '--encoder'), _arg(a, '--input'), _arg(a, '--out')
    amp, dev, attn = _arg(a, '--amp', 'fp16'), _arg(a, '--device', 'cuda'), _arg(a, '--attn')
    assert amp in ('fp16', 'none') and dev in ('cuda', 'cpu'), (amp, dev)
    import numpy as np
    import torch

    from vindr_bench import encoders
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kw = dict(attn=attn) if attn else {}
    model = encoders.build(encoder, check=False, **kw)      # nạp THẬT trong tiến trình này; trượt -> ghi lại, bên gọi báo không đạt
    t = model.nap['thong_tin_nap']
    versions = {}
    for m in ('transformers', 'timm'):
        try:
            versions[m] = __import__(m).__version__
        except ImportError:
            versions[m] = None
    info = dict(python=sys.version.split()[0], torch=torch.__version__, amp=amp, thiet_bi=dev, encoder=encoder, nap=model.nap, attn=attn,
                attention=model.nap.get('attention'), phien_ban=versions,
                thieu_khoa=(len(t['thieu']) if t else 0) + model.nap['so_voi_tep']['so_thieu'])
    model = model.eval().to(dev)
    x = torch.from_numpy(np.load(inp)).to(dev)
    H = torch.float16 if dev == 'cuda' else torch.bfloat16
    t0 = time.time()
    with torch.inference_mode(), torch.autocast(device_type=dev, dtype=H, enabled=(amp != 'none')):
        outs = model.get_intermediate_layers(x, 4, return_class_token=True)
        arrays = dict(patch=torch.stack([o[0].float() for o in outs]).cpu().numpy(), cls=torch.stack([o[1].float() for o in outs]).cpu().numpy())
    if dev == 'cuda':
        torch.cuda.synchronize()
    info.update(giay=time.time() - t0, gpu=(torch.cuda.get_device_name(0) if dev == 'cuda' else None),
                huu_han=bool(all(np.isfinite(v).all() for v in arrays.values())), hinh={k: list(v.shape) for k, v in arrays.items()})
    np.savez(out, info=np.array(json.dumps(info, default=str)), **arrays)
    print('DT=' + json.dumps(info, default=str))


if __name__ == '__main__':
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != _here]
    warnings.simplefilter('ignore')
    (make_input if '--make-input' in sys.argv else features)(sys.argv[1:])
