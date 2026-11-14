# -*- coding: utf-8 -*-
"""MRM (RL4M/MRM-pytorch @ 3730ba4) — MRM.pth = checkpoint huấn luyện đầy đủ {'model', 'optimizer', 'epoch', 'scaler', 'args'}; 'model' gồm encoder ảnh
(150 khoá ViT-B/16) + decoder MAE + nhánh chữ BERT. Chỉ nạp encoder ảnh vào timm VisionTransformer = models_vit.vit_base_patch16 của họ (patch 16, 768,
12 tầng, 12 đầu, mlp 4, qkv_bias, LayerNorm eps 1e-6), NẠP CHẶT; khoá còn lại phải thuộc nhóm đã biết. Chép ý từ module 88
(external_benchmarks.backbones.mrm_encoder_state, đã kiểm trên tệp thật 12/11).

Đặc trưng: n khối cuối qua `norm` cuối; vector toàn ảnh = TB token patch đã qua norm (= nhánh report lúc tiền huấn luyện). Chuẩn hoá 0,4978 / 0,2449
(mã fine-tune của họ). 'args' trong pickle là argparse.Namespace -> nạp qua checks.safe_load (Unpickler hạn chế)."""
import collections

import torch

from ..checks import safe_load, weights
from . import files, timm_vit
from .base import EncoderWrapper
from .spec import IMAGENET_MEAN, IMAGENET_STD, SPECS

NON_ENCODER = {'mask_token', 'decoder_pos_embed', 'decoder_embed', 'decoder_blocks', 'decoder_norm', 'decoder_pred', 'bert_encoder', 'bert_mlp'}
N_ENCODER_KEYS = 150


class Mrm(EncoderWrapper):
    def __init__(self, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None):
        spec = SPECS['mrm']
        super().__init__(spec, input_mean, input_std)
        import timm
        path = path or files.weights_path(spec)
        ck, src = safe_load.load_checkpoint(path, expect_md5=spec.file_md5, expect_size=spec.file_size, allow=('namespace',))
        assert isinstance(ck, dict) and 'model' in ck, 'MRM: checkpoint khong co khoa model -> DUNG'
        sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in ck['model'].items()}
        self.model = timm_vit.vit_b16_224()
        want = set(self.model.state_dict())
        enc = {k: v for k, v in sd.items() if k in want}
        rest = collections.Counter(k.split('.')[0] for k in sd if k not in want)
        assert set(rest) <= NON_ENCODER, f'MRM: khoa ngoai encoder la {sorted(set(rest) - NON_ENCODER)} -> DUNG'
        assert set(enc) == want and len(enc) == N_ENCODER_KEYS, f'MRM: encoder thieu {sorted(want - set(enc))[:5]} ({len(enc)} khoa) -> DUNG'
        self.model.load_state_dict(enc, strict=True)
        self.n_prefix = int(self.model.num_prefix_tokens)
        assert self.n_prefix == 1, self.n_prefix
        args = ck.get('args')
        self.nap = weights.record(source='Drive (README RL4M/MRM-pytorch @ 3730ba4)', file_name=spec.drive_file,
                                  library=f'timm {timm.__version__} (nap chat encoder anh)', model_state=self.model.state_dict(), file_tensors=enc,
                                  lib_info=None, nguon_tep=src, nhom_ngoai_encoder=dict(rest), epoch=ck.get('epoch'),
                                  mo_hinh_huan_luyen=getattr(args, 'model', None), attention=f'timm fused_attn (SDPA) = {bool(timm.layers.use_fused_attn())}')

    def _blocks(self, x, n):
        return timm_vit.last_blocks(self.model, x, n)


def build(input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None):
    return Mrm(input_mean, input_std, path).eval()
