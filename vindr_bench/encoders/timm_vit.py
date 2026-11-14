# -*- coding: utf-8 -*-
"""Phần dùng chung cho các encoder là timm VisionTransformer (BiomedCLIP; bước sau: MRM, CheXWorld): dựng ViT-B/16 224 trần, nạp CHẶT, lấy n tầng
cuối qua norm cuối bằng API công khai của timm (có ở cả timm 0.9.12 của môi trường CheXFound lẫn 1.0.x)."""
import hashlib

import torch


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def vit_b16_224():
    '''timm vit_base_patch16_224 không trọng số, không đầu phân loại (num_classes=0 -> head = Identity; global_pool mặc định 'token', fc_norm = Identity).'''
    import timm
    return timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)


def last_blocks(model, x, n):
    '''-> list n tensor [B, tiền tố + T, D] của n block cuối, đã qua model.norm, token tiền tố đứng trước (khuôn của base.EncoderWrapper._blocks).
       Mô hình không có token tiền tố (CheXWorld): timm 1.0 trả tensor chứ không trả cặp (patch, tiền tố) -> gọi return_prefix_tokens=False.'''
    if int(getattr(model, 'num_prefix_tokens', 0)) == 0:
        outs = [t for t in model.get_intermediate_layers(x, n=n, return_prefix_tokens=False, norm=True)]
    else:
        outs = [torch.cat([prefix, patch], dim=1) for patch, prefix in model.get_intermediate_layers(x, n=n, return_prefix_tokens=True, norm=True)]
    assert len(outs) == n and all(t.dim() == 3 and t.shape[0] == x.shape[0] for t in outs), [tuple(t.shape) for t in outs]
    return outs
