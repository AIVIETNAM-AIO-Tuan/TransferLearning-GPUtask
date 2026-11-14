# -*- coding: utf-8 -*-
"""CheXWorld (LeapLabTHU/CheXWorld) — chexworld_pretrained.tar (checkpoint torch dạng zip, đuôi .tar) từ cxrlt_2024/chexworld (bản Tuấn tải 04/11 theo
README tác giả). Trọng số = ckpt['model'] khoá 'target_encoder.*' (target encoder I-JEPA, --use_target mặc định của tác giả); ViT-B/16 224 KHÔNG CLS.
Chép ý từ 85f (dung_85._nap_chexworld_p2, đã chạy GPU 06/11): timm vit_base_patch16_224 class_token=False, fc_norm=False, nạp CHẶT (tác giả strict=False;
global_pool 'avg' mặc định của timm tạo fc_norm và bỏ `norm` -> khoá norm.* không vào được).

Đặc trưng: n khối cuối qua `norm`; vector toàn ảnh = TB mọi token đầu ra (= FineTuner kiểu 'fc' của tác giả: features.mean(dim=1), không CLS).
Chuẩn hoá ImageNet (timm IMAGENET_DEFAULT trong mã tác giả). 'args' trong pickle -> checks.safe_load."""
import collections

import torch

from ..checks import safe_load, weights
from . import files, timm_vit
from .base import EncoderWrapper
from .spec import IMAGENET_MEAN, IMAGENET_STD, SPECS

TARGET = 'target_encoder.'


class CheXWorld(EncoderWrapper):
    n_prefix = 0

    def __init__(self, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None):
        spec = SPECS['chexworld']
        super().__init__(spec, input_mean, input_std)
        import timm
        path = path or files.weights_path(spec)
        ck, src = safe_load.load_checkpoint(path, expect_md5=spec.file_md5, expect_size=spec.file_size, allow=('namespace',))
        assert isinstance(ck, dict) and 'model' in ck, 'CheXWorld: checkpoint khong co khoa model -> DUNG'
        sd = {k[len(TARGET):]: v for k, v in ck['model'].items() if k.startswith(TARGET)}
        assert sd, 'CheXWorld: khong co khoa target_encoder.* -> DUNG'
        groups = collections.Counter(k.split('.')[0] for k in ck['model'])
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0, class_token=False, global_pool='avg', fc_norm=False,
                                       img_size=224)
        self.model.load_state_dict(sd, strict=True)
        self.n_prefix = int(self.model.num_prefix_tokens)
        assert self.n_prefix == 0, self.n_prefix
        self.nap = weights.record(source='cxrlt_2024/chexworld (README LeapLabTHU/CheXWorld)', file_name=spec.drive_file,
                                  library=f'timm {timm.__version__} (nap chat target_encoder)', model_state=self.model.state_dict(), file_tensors=sd,
                                  lib_info=None, nguon_tep=src, nhom_khoa_model=dict(groups), epoch=ck.get('epoch'),
                                  attention=f'timm fused_attn (SDPA) = {bool(timm.layers.use_fused_attn())}')

    def _blocks(self, x, n):
        return timm_vit.last_blocks(self.model, x, n)


def build(input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None):
    return CheXWorld(input_mean, input_std, path).eval()
