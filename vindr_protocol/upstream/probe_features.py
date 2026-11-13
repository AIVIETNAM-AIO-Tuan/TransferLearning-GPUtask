# -*- coding: utf-8 -*-
"""86h (PLAN_86 §15): đặc trưng 4 tầng cuối CheXFound — get_intermediate_layers(x, 4, return_class_token=True) như ModelWithIntermediateLayers của
linear_glori.py — dựng bằng HÀM CỦA HỌ (build_model_from_cfg trên config gộp: mặc định + config.yaml + opts; nạp như load_pretrained_weights: khoá
'teacher', bỏ 'module.' / 'backbone.', strict=False). CÙNG kịch bản chạy bằng Python của môi trường cũ / mới, cùng đầu vào .npy -> so ở notebook.
Không import vindr_protocol; chỉ cú pháp / API có ở Python 3.9 + torch 2.0.

    python probe_features.py --repo R --config C --ckpt K --input X.npy --out Y.npz --amp fp16|none [--device cuda|cpu] [--opts a=b ...]
    python probe_features.py --repo R --encoder TEN --pkg GOI --input X.npy --out Y.npz --amp fp16|none [--device cuda|cpu] [--attn eager]

--amp none: fp32 với TF32 tắt (tham chiếu). --device cpu: chỉ để kiểm trên máy (autocast cpu = bf16). --encoder (86j, PLAN_86 §16): lớp bọc
vindr_protocol.upstream.encoders_ho thay CheXFound (cần --pkg = thư mục chứa vindr_protocol); lớp bọc không có token patch (BiomedCLIP) -> chỉ lưu cls;
info['nap'] = kết quả kiểm nạp trọng số THẬT của encoders_ho (thông tin nạp của thư viện + so từng tensor với tệp), thieu_khoa = số khoá thiếu đo được.
--attn (§16.4 13/11 lần 2; chỉ RAD-DINO ở môi trường mới): ép attention của transformers (eager = thuật toán của 4.36.2) cho phép đo nền; info['attn']
= attention thực dùng."""
import json, sys, time, warnings


def _arg(a, name, default=None):
    return a[a.index(name) + 1] if name in a else default


def main():
    a = sys.argv[1:]
    repo, config, ckpt, inp, out = (_arg(a, k) for k in ('--repo', '--config', '--ckpt', '--input', '--out'))
    amp, dev, encoder, pkg = _arg(a, '--amp', 'fp16'), _arg(a, '--device', 'cuda'), _arg(a, '--encoder'), _arg(a, '--pkg')
    attn = _arg(a, '--attn')
    opts = a[a.index('--opts') + 1:] if '--opts' in a else []
    assert amp in ('fp16', 'none') and dev in ('cuda', 'cpu'), (amp, dev)
    assert attn is None or encoder, '--attn chi dung voi --encoder'
    sys.path.insert(0, repo)
    warnings.simplefilter('ignore')
    import numpy as np
    import torch
    from chexfound.layers import attention, swiglu_ffn
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    info = dict(python=sys.version.split()[0], torch=torch.__version__, amp=amp, thiet_bi=dev, encoder=encoder or 'chexfound')
    if encoder:
        assert pkg, '--encoder can --pkg (thu muc chua vindr_protocol)'
        sys.path.insert(0, pkg)
        from vindr_protocol.upstream import encoders_ho
        model = encoders_ho.build(encoder, check=False, attn=attn)   # kiểm nạp THẬT trong tiến trình này; trượt -> ghi lại, run_features báo không đạt
        t = model.nap['thong_tin_nap']
        info.update(nap=model.nap, attn=model.nap.get('attention'), thieu_khoa=(len(t['thieu']) if t else 0) + model.nap['so_voi_tep']['so_thieu'])
    else:
        from omegaconf import OmegaConf
        from chexfound.configs import dinov2_default_config
        from chexfound.models import build_model_from_cfg
        cfg = OmegaConf.merge(OmegaConf.create(dinov2_default_config), OmegaConf.load(config), OmegaConf.from_cli(opts))
        model, _ = build_model_from_cfg(cfg, only_teacher=True)
        sd = torch.load(ckpt, map_location='cpu')
        sd = sd['teacher'] if 'teacher' in sd else sd
        sd = {k.replace('module.', '').replace('backbone.', ''): v for k, v in sd.items()}
        msg = model.load_state_dict(sd, strict=False)
        info.update(thieu_khoa=len(msg.missing_keys), thieu_tang=sum(k.startswith('blocks.') for k in msg.missing_keys),
                    block_chunks=int(cfg.student.block_chunks),
                    xformers_cua_ho=dict(attention=bool(attention.XFORMERS_AVAILABLE), swiglu_ffn=bool(swiglu_ffn.XFORMERS_AVAILABLE)))
    model = model.eval().to(dev)
    x = torch.from_numpy(np.load(inp)).to(dev)
    H = torch.float16 if dev == 'cuda' else torch.bfloat16
    t0 = time.time()
    with torch.inference_mode(), torch.autocast(device_type=dev, dtype=H, enabled=(amp != 'none')):
        outs = model.get_intermediate_layers(x, 4, return_class_token=True)
        arrays = dict(cls=torch.stack([o[1].float() for o in outs]).cpu().numpy())
        if all(o[0] is not None for o in outs):
            arrays['patch'] = torch.stack([o[0].float() for o in outs]).cpu().numpy()
    if dev == 'cuda': torch.cuda.synchronize()
    info.update(giay=time.time() - t0, gpu=(torch.cuda.get_device_name(0) if dev == 'cuda' else None),
                huu_han=bool(all(np.isfinite(v).all() for v in arrays.values())), hinh={k: list(v.shape) for k, v in arrays.items()},
                hinh_patch=list(arrays['patch'].shape) if 'patch' in arrays else None, hinh_cls=list(arrays['cls'].shape))
    np.savez(out, info=np.array(json.dumps(info)), **arrays)
    print('DT=' + json.dumps(info))


if __name__ == '__main__':
    main()
