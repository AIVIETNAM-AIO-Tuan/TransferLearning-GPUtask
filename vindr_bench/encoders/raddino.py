# -*- coding: utf-8 -*-
"""RAD-DINO (HF `microsoft/rad-dino`, Dinov2Model) — lớp bọc theo giao diện chung (base.EncoderWrapper).

Đặc trưng: hidden_states của n tầng cuối, mỗi tầng QUA layernorm cuối; CLS = token 0. Ảnh 518 (lưới gốc 37 × 37 -> không nội suy vị trí; ở kích
thước khác, đặc trưng thô lệch mạnh giữa các bản transformers — đo 13/11, Event "86j_raddino_kieu_ho lần chạy đầu …"). Chuẩn hoá gốc đọc từ
preprocessor_config.json của đúng revision và so với spec."""
import json

from ..checks import weights
from .base import EncoderWrapper
from .spec import IMAGENET_MEAN, IMAGENET_STD, SPECS

REPO = 'microsoft/rad-dino'
REVISION = '110cbc18d5133582e320b43d53bf5c44e410c936'      # main 13/11/2026
WEIGHT_FILE = 'model.safetensors'
# transformers >= 5.18 đổi tên attention của Dinov2 lúc nạp (đọc wheel 5.17 / 5.18 / 5.22.2; kiểm thật trên Colab 5.22.1 ngày 13/11) -> đưa tên mô
# hình về tên trong tệp hub trước khi so.
RENAME = (('.attention.q_proj.', '.attention.attention.query.'), ('.attention.k_proj.', '.attention.attention.key.'),
          ('.attention.v_proj.', '.attention.attention.value.'), ('.attention.o_proj.', '.attention.output.dense.'))


class RadDino(EncoderWrapper):
    '''attn: None = attention mặc định của bản transformers; 'eager' = ép attention tường minh (phép đo nền ở môi trường mới phải cùng thuật toán
       với transformers 4.36.2 của môi trường CheXFound — bản đó chỉ có eager).'''

    def __init__(self, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, attn=None, revision=REVISION):
        super().__init__(SPECS['raddino'], input_mean, input_std)
        import transformers
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        kw = dict(revision=revision, output_loading_info=True)
        if attn:
            kw['attn_implementation'] = attn
        self.model, info = transformers.AutoModel.from_pretrained(REPO, **kw)
        used = getattr(self.model.config, '_attn_implementation', None)
        assert not attn or used == attn, f'RAD-DINO: attention {used!r} != yeu cau {attn!r} -> DUNG'
        with open(hf_hub_download(REPO, 'preprocessor_config.json', revision=revision), encoding='utf-8') as f:
            pre = json.load(f)
        crop = pre.get('crop_size') or {}
        assert tuple(pre['image_mean']) == self.spec.mean and tuple(pre['image_std']) == self.spec.std and crop.get('height') == crop.get('width') \
            == self.spec.image_size, f'preprocessor_config RAD-DINO khac spec: {pre} -> DUNG'
        cfg = self.model.config
        assert (cfg.patch_size, cfg.hidden_size) == (self.spec.patch_size, self.spec.embed_dim), (cfg.patch_size, cfg.hidden_size)
        self.nap = weights.record(source=f'{REPO}@{revision}', file_name=WEIGHT_FILE, library=f'transformers {transformers.__version__}',
                                  model_state=self.model.state_dict(), file_tensors=load_file(hf_hub_download(REPO, WEIGHT_FILE, revision=revision)),
                                  lib_info=weights.library_info(info), rename=RENAME, attention=used,
                                  kieu_mo_hinh=str(next(self.model.parameters()).dtype))

    def _blocks(self, x, n):
        hidden = self.model(x, output_hidden_states=True).hidden_states[-n:]
        return [self.model.layernorm(h) for h in hidden]


def build(input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, attn=None):
    return RadDino(input_mean, input_std, attn=attn).eval()
