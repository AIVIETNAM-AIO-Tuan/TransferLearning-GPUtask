# -*- coding: utf-8 -*-
"""PubMedCLIP ViT-B/32 (sarahESL/PubMedCLIP) — tệp HF PubMedCLIP_ViT32.pth = {best_epoch, state_dict, optimizer_state_dict, best_loss}; state_dict cùng
khuôn CLIP OpenAI (tinh chỉnh từ clip.load('ViT-B/32')). Tháp ảnh + đặc trưng: clip_openai. Chuẩn hoá: CLIP mặc định (tiền xử lý của clip.load)."""
import torch

from ..checks import safe_load, weights
from . import clip_openai
from .spec import IMAGENET_MEAN, IMAGENET_STD, SPECS

REPO = 'sarahESL/PubMedCLIP'
REVISION = 'c5885a5c0ba8ac8d86341e57c8466e49783874c1'      # main 14/11/2026
WEIGHT_FILE, FILE_MD5 = 'PubMedCLIP_ViT32.pth', '0205eb29095bce53b02e9c2a153b1db0'
TOP_KEYS = {'best_epoch', 'state_dict', 'optimizer_state_dict', 'best_loss'}
CONFIG = dict(input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512)


class PubMedClip(clip_openai.OpenAIClipWrapper):
    def __init__(self, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, revision=REVISION, path=None):
        super().__init__(SPECS['pubmedclip'], input_mean, input_std)
        if path is None:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(REPO, WEIGHT_FILE, revision=revision)
        ck, src = safe_load.load_checkpoint(path, expect_md5=FILE_MD5)
        assert isinstance(ck, dict) and set(ck) == TOP_KEYS, f'PubMedCLIP: khoa ngoai {sorted(ck)[:6]} != {sorted(TOP_KEYS)} -> DUNG'
        sd = ck['state_dict']
        self.visual, cfg = clip_openai.build_visual(sd)
        assert cfg == CONFIG, f'PubMedCLIP: cau hinh {cfg} != {CONFIG} -> DUNG'
        self.nap = weights.record(source=f'{REPO}@{revision}', file_name=WEIGHT_FILE,
                                  library=f'torch {torch.__version__} (lop CLIP OpenAI chep tu CheXzero @ 5c341db)', model_state=self.visual.state_dict(),
                                  file_tensors=sd, lib_info=None, prefix=clip_openai.PREFIX, nguon_tep=src, cau_hinh=cfg, best_epoch=int(ck['best_epoch']),
                                  attention='nn.MultiheadAttention (need_weights=False)')


def build(input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None):
    return PubMedClip(input_mean, input_std, path=path).eval()
