# -*- coding: utf-8 -*-
"""Backbone mới của `88` (PLAN_88 §3.1): MRM và BiomedCLIP — cả hai là timm ViT-B/16 224, ĐÔNG CỨNG, eval().

- MRM (RL4M/MRM-pytorch, tệp `MRM.pth` công khai): checkpoint huấn luyện đầy đủ {'model', 'optimizer', 'epoch', 'scaler', 'args'}; 'model' gồm encoder
  ảnh (150 khoá ViT-B/16) + decoder MAE + nhánh chữ BERT. Chỉ nạp encoder ảnh vào `timm.models.vision_transformer.VisionTransformer` cấu hình như
  `models_vit.vit_base_patch16` của họ (norm LayerNorm eps 1e-6), NẠP CHẶT; khoá còn lại phải thuộc nhóm đã biết. Token toàn ảnh = TB token patch đã qua
  LayerNorm cuối, từng tầng (theo tác giả: nhánh report và RAD-DINO E.2 dùng TB patch; PLAN_88 §14.5). Chuẩn hoá 0,4978 / 0,2449 (mã fine-tune của họ).
- BiomedCLIP (HF microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 @ commit ghim): tháp ảnh = `visual.trunk.*` của open_clip (timm
  'vit_base_patch16_224'; timm_pool rỗng -> pool mặc định 'token' = CLS qua LayerNorm cuối, rồi chiếu 512). Nạp trunk vào timm, KHÔNG cần open_clip lúc
  chạy. Token toàn ảnh = CLS. Chuẩn hoá CLIP theo preprocess_cfg.
Đặc trưng = 4 tầng cuối qua LayerNorm cuối (quy ước `86` = DINOv2 norm=True) -> giao diện `vindr_protocol.backbones.Backbone` (kind 'timm_vit') để
`run_main` / `features.check_features` của `86` dùng thẳng (RSNA); lớp bọc nguồn Shenzhen dùng `intermediate_layers` (dạng get_intermediate_layers).
Nạp checkpoint: liệt kê GLOBAL trong pickle (không chạy pickle), rồi `torch.load(weights_only=True)`; chỉ cho phép thêm đúng `argparse.Namespace`
(phần 'args' của checkpoint MRM) — không bao giờ tắt weights_only."""
import argparse
import collections
import hashlib
import json
import os
import pickletools
import zipfile
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn

from vindr_protocol.backbones import Backbone
from vindr_protocol.config import BackboneSpec

from . import config

MRM_SPEC = BackboneSpec('mrm', 224, config.MRM_MEAN, config.MRM_STD)
BIOMEDCLIP_SPEC = BackboneSpec('biomedclip', 224, config.BIOMEDCLIP_MEAN, config.BIOMEDCLIP_STD)

# GLOBAL được phép trong pickle của checkpoint (torch.save chuẩn + 'args' kiểu Namespace của MAE / MRM). Khác -> dừng, xem tệp trước.
ALLOWED_GLOBALS = {'collections.OrderedDict', 'torch._utils._rebuild_tensor_v2', 'torch._utils._rebuild_parameter', 'argparse.Namespace'} | {
    f'torch.{t}Storage' for t in ('Float', 'Half', 'BFloat16', 'Double', 'Long', 'Int', 'Short', 'Char', 'Byte', 'Bool')}
# Nhóm khoá KHÔNG thuộc encoder ảnh trong 'model' của MRM (decoder MAE + nhánh chữ), đã liệt kê trên tệp thật 12/11
MRM_NON_ENCODER = {'mask_token', 'decoder_pos_embed', 'decoder_embed', 'decoder_blocks', 'decoder_norm', 'decoder_pred', 'bert_encoder', 'bert_mlp'}
N_BLOCKS = 4


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def pickle_globals(path):
    '''Các tham chiếu GLOBAL trong data.pkl của tệp torch.save dạng zip — đọc opcode bằng pickletools, KHÔNG chạy pickle. Tệp không phải zip -> None.
       -> {tên đầy đủ: số lần}.'''
    if not zipfile.is_zipfile(path):
        return None
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith('data.pkl')]
        assert len(names) == 1, f'{path}: {len(names)} data.pkl trong zip -> DUNG'
        data = zf.read(names[0])
    out, strings = collections.Counter(), []
    for op, arg, _ in pickletools.genops(data):
        if op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE', 'BINUNICODE8'):
            strings.append(arg)
        if op.name == 'GLOBAL':
            out[arg.replace(' ', '.')] += 1
        elif op.name == 'STACK_GLOBAL':
            out[f'{strings[-2]}.{strings[-1]}'] += 1
    return dict(out)


def load_checkpoint(path, expect_md5=None, expect_size=None):
    '''md5 / kích thước (nếu cho) -> GLOBAL trong pickle phải thuộc ALLOWED_GLOBALS -> torch.load(weights_only=True), chỉ cho phép thêm
       argparse.Namespace. -> (đối tượng đã nạp, dict nguồn).'''
    src = dict(tep=os.path.basename(path), byte=os.path.getsize(path))
    if expect_size is not None:
        assert src['byte'] == expect_size, f"{path}: {src['byte']} B != {expect_size} B -> DUNG"
    if expect_md5 is not None:
        src['md5'] = md5(path)
        assert src['md5'] == expect_md5, f"{path}: md5 {src['md5']} != {expect_md5} -> DUNG"
    g = pickle_globals(path)
    if g is not None:
        bad = sorted(set(g) - ALLOWED_GLOBALS)
        assert not bad, f'{path}: GLOBAL la trong pickle {bad} -> DUNG (khong nap)'
        src['global_pickle'] = g
    with torch.serialization.safe_globals([argparse.Namespace]):
        obj = torch.load(path, map_location='cpu', weights_only=True)
    return obj, src


def vit_b16_224():
    '''ViT-B/16 224 = models_vit.vit_base_patch16 của MRM (patch 16, 768, 12 tầng, 12 đầu, mlp 4, qkv_bias, LayerNorm eps 1e-6) = timm
       'vit_base_patch16_224' (BiomedCLIP), không đầu phân loại.'''
    from timm.models.vision_transformer import VisionTransformer
    return VisionTransformer(**config.VIT_B16_224, norm_layer=partial(nn.LayerNorm, eps=1e-6))


def _load_strict(model, state_dict, name):
    '''Nạp CHẶT vào mô hình MỚI (nạp chặt thất bại vẫn có thể đã chép một phần -> mô hình bị bỏ, ném lỗi).'''
    want = set(model.state_dict())
    missing, extra = sorted(want - set(state_dict)), sorted(set(state_dict) - want)
    assert not missing and not extra, f'{name}: thieu {missing[:5]} / thua {extra[:5]} -> DUNG'
    bad_shape = [k for k, v in model.state_dict().items() if tuple(v.shape) != tuple(state_dict[k].shape)]
    assert not bad_shape, f'{name}: sai hinh {bad_shape[:5]} -> DUNG'
    model.load_state_dict(state_dict, strict=True)
    return model


def mrm_encoder_state(checkpoint):
    '''checkpoint MRM -> (state_dict encoder ảnh cho vit_b16_224, báo cáo nhóm khoá). Cổng: đủ khoá encoder; khoá còn lại thuộc MRM_NON_ENCODER.'''
    assert isinstance(checkpoint, dict) and 'model' in checkpoint, 'checkpoint MRM khong co khoa model -> DUNG'
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in checkpoint['model'].items()}
    want = set(vit_b16_224().state_dict())
    enc = {k: v for k, v in sd.items() if k in want}
    rest = collections.Counter(k.split('.')[0] for k in sd if k not in want)
    assert set(rest) <= MRM_NON_ENCODER, f'khoa MRM ngoai encoder la: {sorted(set(rest) - MRM_NON_ENCODER)} -> DUNG'
    args = checkpoint.get('args')
    info = dict(so_khoa_model=len(sd), so_khoa_encoder=len(enc), nhom_ngoai_encoder=dict(rest), epoch=checkpoint.get('epoch'),
                mo_hinh_huan_luyen=getattr(args, 'model', None), co_optimizer='optimizer' in checkpoint)
    return enc, info


BIOMEDCLIP_GROUPS = {'visual.trunk', 'visual.head', 'text', 'logit_scale'}     # nhóm khoá của tệp thật (12/11): trunk 150, chiếu 1, chữ 200, nhiệt độ 1


def biomedclip_trunk_state(state_dict):
    '''state_dict open_clip của BiomedCLIP -> (state_dict trunk cho vit_b16_224, báo cáo). Cổng: có visual.trunk.*; mọi khoá thuộc BIOMEDCLIP_GROUPS.'''
    prefix = 'visual.trunk.'
    trunk = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    groups = collections.Counter(k.split('.')[0] + ('.' + k.split('.')[1] if k.startswith('visual.') else '') for k in state_dict)
    assert trunk, 'khong co khoa visual.trunk.* -> DUNG'
    assert set(groups) <= BIOMEDCLIP_GROUPS, f'nhom khoa BiomedCLIP la: {sorted(set(groups) - BIOMEDCLIP_GROUPS)} -> DUNG'
    return trunk, dict(so_khoa=len(state_dict), so_khoa_trunk=len(trunk), nhom=dict(groups))


@torch.no_grad()
def intermediate_layers(model, x, n=N_BLOCKS, pool='cls'):
    '''timm ViT: n tầng cuối, mỗi tầng qua LayerNorm cuối của mô hình (= get_intermediate_layers(n, return_class_token=True, norm=True) của DINOv2) ->
       tuple n x (patch [B, N, d], token toàn ảnh [B, d]); token toàn ảnh = CLS (pool 'cls') hoặc TB patch (pool 'mean'). Trình tự = forward_features của
       timm (patch_embed -> _pos_embed -> patch_drop -> norm_pre -> blocks -> norm).'''
    assert pool in ('cls', 'mean'), pool
    h = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(x))))
    outs = []
    for i, blk in enumerate(model.blocks):
        h = blk(h)
        if i >= len(model.blocks) - n:
            outs.append(model.norm(h))
    p0 = getattr(model, 'num_prefix_tokens', 1)
    return tuple((o[:, p0:], o[:, p0:].mean(1) if pool == 'mean' else o[:, 0]) for o in outs)


@dataclass
class TimmBackbone(Backbone):
    '''Backbone timm ViT (MRM / BiomedCLIP) theo giao diện của `86`: features(x, with_last) -> (blocks, patch tầng cuối) [, ref]; ref = dict
       x_norm_patchtokens / x_norm_clstoken từ forward_features CHUẨN của timm (để check_features so từng bit).'''
    pool: str = 'cls'

    @property
    def dim(self):
        return int(self.model.embed_dim)

    @property
    def patch_size(self):
        return int(self.model.patch_embed.patch_size[0])

    @torch.no_grad()
    def features(self, x, with_last=False):
        with torch.autocast(self.device, dtype=self.amp, enabled=(self.device == 'cuda')):
            blocks = intermediate_layers(self.model, x, N_BLOCKS, self.pool)
            ref = self.model.forward_features(x) if with_last else None
        if not with_last:
            return blocks, blocks[-1][0]
        p0 = getattr(self.model, 'num_prefix_tokens', 1)
        cls = ref[:, p0:].mean(1) if self.pool == 'mean' else ref[:, 0]
        return blocks, blocks[-1][0], dict(x_norm_patchtokens=ref[:, p0:], x_norm_clstoken=cls)


def load_mrm(path, device='cpu', amp=torch.bfloat16, check_md5=True):
    '''MRM.pth (tệp thật) -> TimmBackbone 'mrm' (pool 'mean'), eval(). Cổng: md5 / kích thước, GLOBAL pickle, weights_only, đủ encoder, nạp chặt.'''
    ck, src = load_checkpoint(path, config.MRM_MD5 if check_md5 else None, config.MRM_SIZE if check_md5 else None)
    enc, info = mrm_encoder_state(ck)
    del ck
    model = _load_strict(vit_b16_224(), enc, 'MRM').eval().to(device)
    src.update(info, repo_commit=config.MRM_REPO_COMMIT)
    return TimmBackbone(name='mrm', model=model, spec=MRM_SPEC, device=device, amp=amp, source=src, kind='timm_vit', pool='mean')


def biomedclip_preprocess(config_path):
    '''preprocess_cfg của open_clip_config.json -> (mean, std); cổng: = hằng số đã ghi (CLIP OpenAI).'''
    with open(config_path, encoding='utf-8') as f:
        cfg = json.load(f)
    mean, std = tuple(cfg['preprocess_cfg']['mean']), tuple(cfg['preprocess_cfg']['std'])
    vc = cfg['model_cfg']['vision_cfg']
    assert (mean, std) == (config.BIOMEDCLIP_MEAN, config.BIOMEDCLIP_STD), f'chuan hoa BiomedCLIP {mean} / {std} khac hang so -> DUNG'
    assert vc['timm_model_name'] == 'vit_base_patch16_224' and vc['timm_pool'] == '' and vc['image_size'] == 224, f'vision_cfg la: {vc} -> DUNG'
    return mean, std


def load_biomedclip(bin_path, config_path, device='cpu', amp=torch.bfloat16, check_md5=True):
    '''open_clip_pytorch_model.bin + open_clip_config.json (tệp thật, commit ghim) -> TimmBackbone 'biomedclip' (pool 'cls'), eval().'''
    if check_md5:
        for p in (bin_path, config_path):
            want = config.BIOMEDCLIP_FILES[os.path.basename(p)]
            assert md5(p) == want, f'{p}: md5 khac {want} -> DUNG'
    biomedclip_preprocess(config_path)
    sd, src = load_checkpoint(bin_path)
    trunk, info = biomedclip_trunk_state(sd)
    del sd
    model = _load_strict(vit_b16_224(), trunk, 'BiomedCLIP').eval().to(device)
    src.update(info, repo=config.BIOMEDCLIP_REPO, commit=config.BIOMEDCLIP_COMMIT, md5_ok=bool(check_md5))
    return TimmBackbone(name='biomedclip', model=model, spec=BIOMEDCLIP_SPEC, device=device, amp=amp, source=src, kind='timm_vit', pool='cls')
