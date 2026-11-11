"""Đặc trưng đông cứng cho đầu đọc: 4 tầng cuối qua layernorm cuối (= get_intermediate_layers(n=4, return_class_token=True, norm=True) của
DINOv2 — PLAN_86 §4.2, F12) + cổng kiểm trên một ảnh thật. Hai đường: HF Dinov2 (RAD-DINO) và mã gốc DINOv2 / CheXFound (gọi thẳng
get_intermediate_layers như ModelWithIntermediateLayers của họ)."""
import torch


@torch.no_grad()
def dinov2_hf(model, x, amp, device, n_blocks=4, with_last=False):
    '''HF Dinov2 (RAD-DINO): hidden_states = đầu ra từng tầng TRƯỚC layernorm cuối. ảnh [B, 3, S, S] ->
       (blocks = n_blocks x (patch [B, N, d], cls [B, d]), patch tầng cuối cho ML-Decoder (= last_hidden_state[:, 1:])) [, last_hidden_state].'''
    with torch.autocast(device, dtype=amp, enabled=(device == 'cuda')):
        out = model(pixel_values=x, output_hidden_states=True)
        normed = [model.layernorm(h) for h in out.hidden_states[-n_blocks:]]
    blocks = tuple((h[:, 1:], h[:, 0]) for h in normed)
    return (blocks, normed[-1][:, 1:], out.last_hidden_state) if with_last else (blocks, normed[-1][:, 1:])


@torch.no_grad()
def dinov2_original(model, x, amp, device, n_blocks=4, with_last=False):
    '''Mã gốc DINOv2 / CheXFound: get_intermediate_layers(x, n_blocks, return_class_token=True) (norm=True mặc định; đã BỎ register) dưới autocast
       — = ModelWithIntermediateLayers của mã CheXFound. -> (blocks, patch tầng cuối) [, forward_features(x) (dict x_norm_*, cho cổng)].'''
    with torch.autocast(device, dtype=amp, enabled=(device == 'cuda')):
        blocks = tuple(model.get_intermediate_layers(x, n_blocks, return_class_token=True))
        ref = model.forward_features(x) if with_last else None
    return (blocks, blocks[-1][0], ref) if with_last else (blocks, blocks[-1][0])


def _last_layer_reference(backbone, ref):
    '''(patch, cls) tầng cuối qua layernorm theo đường forward CHUẨN của mô hình (để so với get_intermediate_layers / hidden_states).'''
    if backbone.kind == 'hf_dinov2':
        return ref[:, 1:], ref[:, 0]
    return ref['x_norm_patchtokens'], ref['x_norm_clstoken']


def check_features(backbone, image, eval_transform):
    '''CỔNG trên MỘT ảnh: tầng cuối qua layernorm = đầu ra chuẩn của mô hình (HF: last_hidden_state; mã gốc: forward_features x_norm_*) từng bit,
       đủ 4 tầng đúng hình (patch = (S / patch)^2, đã bỏ register), hữu hạn, 4 tầng khác nhau; biến đổi đánh giá = min-max từng kênh về [0, 1]
       rồi chuẩn hoá. Sai -> AssertionError. -> {lech, so_patch, dim}.'''
    spec = backbone.spec
    x = eval_transform(image).unsqueeze(0)
    mm = (x * torch.as_tensor(spec.std).view(1, 3, 1, 1) + torch.as_tensor(spec.mean).view(1, 3, 1, 1)).flatten(2)
    blocks, last, ref = backbone.features(x.to(backbone.device), with_last=True)
    ref_patch, ref_cls = _last_layer_reference(backbone, ref)
    n, d = (spec.image_size // backbone.patch_size) ** 2, backbone.dim
    lech = max(float((blocks[-1][0] - ref_patch).abs().max()), float((blocks[-1][1] - ref_cls).abs().max()), float((last - ref_patch).abs().max()))
    assert (len(blocks) == 4 and all(tuple(p.shape) == (1, n, d) and tuple(c.shape) == (1, d) for p, c in blocks) and lech == 0.0
            and all(bool(torch.isfinite(p).all() and torch.isfinite(c).all()) for p, c in blocks)
            and float((blocks[0][0] - blocks[-1][0]).abs().max()) > 0), (
        f'dac trung 4 tang: tang cuoi qua layernorm KHAC dau ra chuan cua mo hinh (lech {lech:.1e}) / sai hinh / khong huu han / 4 tang trung nhau -> DUNG')
    assert float(mm.min(-1).values.abs().max()) < 1e-5 and float((mm.max(-1).values - 1).abs().max()) < 1e-5, 'bien doi khong min-max ve [0, 1] -> DUNG'
    return dict(lech=lech, so_patch=n, dim=d)
