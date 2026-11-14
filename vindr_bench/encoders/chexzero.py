# -*- coding: utf-8 -*-
"""CheXzero (rajpurkarlab/CheXzero) — CLIP ViT-B/32 định dạng OpenAI; MỘT checkpoint (spec.drive_file, chọn trước khi có số VinDr) từ thư mục Drive của ta
(bản chép thư mục công khai trong README của tác giả). Tháp ảnh + đặc trưng: clip_openai. Chuẩn hoá gốc của họ (zero_shot.py @ 5c341db): Normalize(101,48761,
83,43944) trên thang 0–255 = (0,39799 / 0,32721) trên [0, 1] = spec."""
import torch

from ..checks import safe_load, weights
from . import clip_openai, files
from .spec import IMAGENET_MEAN, IMAGENET_STD, SPECS

CONFIG = dict(input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512)


class CheXzero(clip_openai.OpenAIClipWrapper):
    def __init__(self, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None):
        spec = SPECS['chexzero']
        super().__init__(spec, input_mean, input_std)
        path = path or files.weights_path(spec)
        sd, src = safe_load.load_checkpoint(path, expect_md5=spec.file_md5, expect_size=spec.file_size)
        assert isinstance(sd, dict) and clip_openai.PREFIX + 'conv1.weight' in sd, 'CheXzero: khong phai state dict CLIP OpenAI -> DUNG'
        self.visual, cfg = clip_openai.build_visual(sd)
        assert cfg == CONFIG, f'CheXzero: cau hinh {cfg} != {CONFIG} -> DUNG'
        self.nap = weights.record(source=f'Drive {spec.drive_dir} (thu muc cong khai CheXzero)', file_name=spec.drive_file,
                                  library=f'torch {torch.__version__} (lop CLIP OpenAI chep tu CheXzero @ 5c341db)', model_state=self.visual.state_dict(),
                                  file_tensors=sd, lib_info=None, prefix=clip_openai.PREFIX, nguon_tep=src, cau_hinh=cfg,
                                  attention='nn.MultiheadAttention (need_weights=False)')


def build(input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None):
    return CheXzero(input_mean, input_std, path).eval()
