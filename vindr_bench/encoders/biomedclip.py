# -*- coding: utf-8 -*-
"""BiomedCLIP — tháp ảnh (timm ViT-B/16 224) nạp thẳng từ tệp open_clip của đúng revision, KHÔNG qua bộ nạp của open_clip (bộ nạp đó không ghim
được revision và đổi theo phiên bản). Thiết kế như module 88 (external_benchmarks.backbones.load_biomedclip, PLAN_88 §3.1).

Đặc trưng: n block cuối qua `norm` của trunk; vector toàn ảnh = CLS — đặc trưng TRƯỚC phép chiếu 512 chiều (quy ước linear probe của CLIP). Lớp bọc
BiomedCLIPBase của CheXFound trả vector ĐÃ chiếu và không có token patch (không chạy được GLoRI / ML-Decoder) -> không dùng ở giao thức này.
Chuẩn hoá gốc đọc từ open_clip_config.json và so với spec."""
import json

import torch

from ..checks import weights
from . import timm_vit
from .base import EncoderWrapper
from .spec import IMAGENET_MEAN, IMAGENET_STD, SPECS

REPO = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
REVISION = '9f341de24bfb00180f1b847274256e9b65a3a32e'      # main 13/11/2026 (= commit module 88 ghim)
WEIGHT_FILE, CONFIG_FILE = 'open_clip_pytorch_model.bin', 'open_clip_config.json'
FILE_MD5 = {WEIGHT_FILE: '531cdee3c02f0e1ce22c474df82ad618', CONFIG_FILE: '26b72194932e548fe568a47ff5d33ffc'}     # như external_benchmarks.config (88)
TRUNK_PREFIX = 'visual.trunk.'


class BiomedClip(EncoderWrapper):
    def __init__(self, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, revision=REVISION):
        super().__init__(SPECS['biomedclip'], input_mean, input_std)
        import timm
        from huggingface_hub import hf_hub_download
        paths = {f: hf_hub_download(REPO, f, revision=revision) for f in (WEIGHT_FILE, CONFIG_FILE)}
        md5 = {f: timm_vit.md5(p) for f, p in paths.items()}
        assert md5 == FILE_MD5, f'tep BiomedCLIP khac ban da ghim ({md5} != {FILE_MD5}) -> DUNG'
        with open(paths[CONFIG_FILE], encoding='utf-8') as f:
            cfg = json.load(f)
        pre, vc = cfg['preprocess_cfg'], cfg['model_cfg']['vision_cfg']
        assert (tuple(pre['mean']), tuple(pre['std'])) == (self.spec.mean, self.spec.std), f'chuan hoa BiomedCLIP khac spec: {pre} -> DUNG'
        assert vc['timm_model_name'] == 'vit_base_patch16_224' and vc['timm_pool'] == '' and vc['image_size'] == self.spec.image_size, f'vision_cfg: {vc}'
        sd = torch.load(paths[WEIGHT_FILE], map_location='cpu', weights_only=True)
        sd = sd.get('state_dict', sd)
        self.model = timm_vit.vit_b16_224()
        self.model.load_state_dict({k[len(TRUNK_PREFIX):]: v for k, v in sd.items() if k.startswith(TRUNK_PREFIX)}, strict=True)
        self.n_prefix = int(self.model.num_prefix_tokens)
        assert self.n_prefix == 1 and int(self.model.embed_dim) == self.spec.embed_dim, (self.n_prefix, self.model.embed_dim)
        self.nap = weights.record(source=f'{REPO}@{revision}', file_name=WEIGHT_FILE, library=f'timm {timm.__version__} (nap chat, khong qua open_clip)',
                                  model_state=self.model.state_dict(), file_tensors=sd, lib_info=None, prefix=TRUNK_PREFIX, md5=md5,
                                  attention=f'timm fused_attn (SDPA) = {bool(timm.layers.use_fused_attn())}',
                                  kieu_mo_hinh=str(next(self.model.parameters()).dtype))

    def _blocks(self, x, n):
        return timm_vit.last_blocks(self.model, x, n)


def build(input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD):
    return BiomedClip(input_mean, input_std).eval()
