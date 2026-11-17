# -*- coding: utf-8 -*-
"""ASG (ASGMVLP/ASGMVLP_CODE @ 110f3f4; "Anatomical Structure-Guided Medical Vision-Language Pre-training", MICCAI 2024) — asg_vit.ckpt = {'state_dict'} của
mô hình tiền huấn luyện đầy đủ: encoder ảnh `img_encoder_q.model.*` (150 khoá ViT-B/16) + hai đầu chiếu của ảnh + nhánh chữ (BERT) + bộ giải mã `irl.*`.
Chỉ nạp encoder ảnh vào timm vit_base_patch16_224 (= lớp VisionTransformer của họ trong models/vit.py: patch 16, 768, 12 tầng, 12 đầu, mlp 4, qkv_bias,
LayerNorm eps 1e-6, pre-norm), NẠP CHẶT; khoá còn lại phải thuộc nhóm đã biết. Hàng THÊM của PLAN_90 (Việc A, Tuấn duyệt 17/11) — spec.EXTRA_ROWS.

Đặc trưng: n khối cuối qua `norm` cuối; vector toàn ảnh = CLS sau norm = `img_feat[:, 0]` của ImageEncoder.forward (TRƯỚC đầu chiếu `global_embed` — đầu
chiếu chỉ phục vụ loss tiền huấn luyện); token patch = `img_feat[:, 1:]`. Chuẩn hoá ImageNet (datasets/pretrain_dataset.py). Tệp chỉ chứa tensor -> nạp
bằng checks.safe_load không cần ngoại lệ pickle."""
import collections

from ..checks import safe_load, weights
from . import files, timm_vit
from .base import EncoderWrapper
from .spec import IMAGENET_MEAN, IMAGENET_STD, SPECS

PREFIX = 'img_encoder_q.model.'
NON_ENCODER = ('img_encoder_q.global_embed.', 'img_encoder_q.local_embed.', 'text_encoder_q.', 'irl.')     # đầu chiếu ảnh, nhánh chữ, bộ giải mã
N_ENCODER_KEYS = 150


class Asg(EncoderWrapper):
    def __init__(self, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None):
        spec = SPECS['asg']
        super().__init__(spec, input_mean, input_std)
        import timm
        path = path or files.weights_path(spec)
        ck, src = safe_load.load_checkpoint(path, expect_md5=spec.file_md5, expect_size=spec.file_size)
        assert isinstance(ck, dict) and set(ck) == {'state_dict'}, f'ASG: khoa cap 1 {sorted(ck) if isinstance(ck, dict) else type(ck)} != [state_dict] -> DUNG'
        sd = ck['state_dict']
        enc = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}
        other = [k for k in sd if not k.startswith(PREFIX)]
        la = sorted(k for k in other if not k.startswith(NON_ENCODER))
        assert not la, f'ASG: khoa ngoai encoder khong thuoc nhom da biet: {la[:5]} -> DUNG'
        rest = collections.Counter(next(p for p in NON_ENCODER if k.startswith(p)).rstrip('.') for k in other)
        self.model = timm_vit.vit_b16_224()
        want = set(self.model.state_dict())
        assert set(enc) == want and len(enc) == N_ENCODER_KEYS, (
            f'ASG: encoder thieu {sorted(want - set(enc))[:5]}, thua {sorted(set(enc) - want)[:5]} ({len(enc)} khoa) -> DUNG')
        self.model.load_state_dict(enc, strict=True)
        self.n_prefix = int(self.model.num_prefix_tokens)
        assert self.n_prefix == 1, self.n_prefix
        self.nap = weights.record(source='Drive (README ASGMVLP/ASGMVLP_CODE @ 110f3f4)', file_name=spec.drive_file,
                                  library=f'timm {timm.__version__} (nap chat encoder anh img_encoder_q.model)', model_state=self.model.state_dict(),
                                  file_tensors=enc, lib_info=None, nguon_tep=src, nhom_ngoai_encoder=dict(rest),
                                  attention=f'timm fused_attn (SDPA) = {bool(timm.layers.use_fused_attn())}')

    def _blocks(self, x, n):
        return timm_vit.last_blocks(self.model, x, n)


def build(input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None):
    return Asg(input_mean, input_std, path).eval()
