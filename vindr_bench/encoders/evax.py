# -*- coding: utf-8 -*-
"""EVA-X (B) — hustvl/EVA-X: timm Eva (RoPE, SwiGLU) cấu hình eva_x_base_patch16; trọng số MIM eva_x_base_patch16_merged520k_mim.pt (HF MapleF/eva_x @
35ddcd6, md5 ghim). `checkpoint_filter_fn` và lớp `EVA_X` chép NGUYÊN VĂN từ eva_x.py @ eedd642c85339b25682ceb574b644cb9afe88e62 (tests so cây cú pháp).

Đường chính thức (EVA_X.forward_head, pre_logits): global_pool 'avg' của timm Eva -> `norm` = Identity, `fc_norm` = LayerNorm; bộ lọc của họ chuyển
`norm` của MIM thành `fc_norm` -> vector toàn ảnh = fc_norm(TB token patch THÔ) (spec.pool 'norm_of_mean'). Token patch cho GLoRI / ML-Decoder =
fc_norm áp từng token thô (quy ước của ta = đầu ra MIM lúc tiền huấn luyện). Khác tác giả: nạp CHẶT (họ strict=False) — chỉ bỏ bộ đệm RoPE (họ bỏ khoá
'rope' vì là hằng). Tệp có số vô hướng numpy trong pickle -> checks.safe_load (Unpickler hạn chế). Chuẩn hoá 0,49185243 / 0,28509309
(classification/utils/datasets.py: nhánh 'eva')."""
import torch
from timm.models.eva import Eva
from timm.layers import resample_abs_pos_embed, resample_patch_embed

from ..checks import safe_load, weights
from .base import EncoderWrapper
from .spec import IMAGENET_MEAN, IMAGENET_STD, SPECS

REPO = 'MapleF/eva_x'
REVISION = '35ddcd6dab6ca99bbdb6cb45c8d1b093aefbd0ee'      # main 14/11/2026
WEIGHT_FILE, FILE_MD5 = 'eva_x_base_patch16_merged520k_mim.pt', '65a68eb32f2313dc537d4da9142ff962'
# = eva_x_base_patch16 của họ (img 224, patch 16, 768, 12 tầng, 12 đầu, qkv tách, mlp 4·2/3 SwiGLU có scale, RoPE ref 14 × 14); num_classes=0 (không đầu)
BASE_KWARGS = dict(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_fused=False, mlp_ratio=4 * 2 / 3, swiglu_mlp=True,
                   scale_mlp=True, use_rot_pos_emb=True, ref_feat_shape=(14, 14), num_classes=0)


# ==== NGUYÊN VĂN eva_x.py (hustvl/EVA-X @ eedd642) ====
def checkpoint_filter_fn(
        state_dict,
        model,
        interpolation='bicubic',
        antialias=True,
):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    state_dict = state_dict.get('model_ema', state_dict)
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('module', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    # prefix for loading OpenCLIP compatible weights
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    elif 'visual.pos_embed' in state_dict:
        prefix = 'visual.'
    else:
        prefix = ''
    mim_weights = prefix + 'mask_token' in state_dict
    no_qkv = prefix + 'blocks.0.attn.q_proj.weight' in state_dict

    len_prefix = len(prefix)
    for k, v in state_dict.items():
        if prefix:
            if k.startswith(prefix):
                k = k[len_prefix:]
            else:
                continue

        if 'rope' in k:
            # fixed embedding no need to load buffer from checkpoint
            continue

        if 'patch_embed.proj.weight' in k:
            _, _, H, W = model.patch_embed.proj.weight.shape
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )

        k = k.replace('mlp.ffn_ln', 'mlp.norm')
        k = k.replace('attn.inner_attn_ln', 'attn.norm')
        k = k.replace('mlp.w12', 'mlp.fc1')
        k = k.replace('mlp.w1', 'mlp.fc1_g')
        k = k.replace('mlp.w2', 'mlp.fc1_x')
        k = k.replace('mlp.w3', 'mlp.fc2')
        if no_qkv:
            k = k.replace('q_bias', 'q_proj.bias')
            k = k.replace('v_bias', 'v_proj.bias')

        if mim_weights and k in ('mask_token', 'lm_head.weight', 'lm_head.bias', 'norm.weight', 'norm.bias'):
            if k == 'norm.weight' or k == 'norm.bias':
                # try moving norm -> fc norm on fine-tune, probably a better starting point than new init
                k = k.replace('norm', 'fc_norm')
            else:
                # skip pretrain mask token & head weights
                continue

        out_dict[k] = v

    return out_dict

class EVA_X(Eva):
    def __init__(self, **kwargs):
        super(EVA_X, self).__init__(**kwargs)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


# ==== lớp bọc của 89 ====
class EvaX(EncoderWrapper):
    n_prefix = 1

    def __init__(self, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, revision=REVISION, path=None):
        super().__init__(SPECS['evax'], input_mean, input_std)
        import timm
        if path is None:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(REPO, WEIGHT_FILE, revision=revision)
        ck, src = safe_load.load_checkpoint(path, expect_md5=FILE_MD5, allow=('numpy_scalar',))
        self.model = EVA_X(**BASE_KWARGS)
        sd = checkpoint_filter_fn(ck, self.model)
        model_keys = set(self.model.state_dict())
        rope = {k for k in model_keys if 'rope' in k}
        missing, extra = sorted(model_keys - set(sd) - rope), sorted(set(sd) - model_keys)
        assert not missing and not extra, f'EVA-X: thieu {missing[:5]} / thua {extra[:5]} (sau bo loc cua ho) -> DUNG'
        bad = [k for k in sd if tuple(sd[k].shape) != tuple(self.model.state_dict()[k].shape)]
        assert not bad, f'EVA-X: sai hinh {bad[:5]} -> DUNG'
        msg = self.model.load_state_dict(sd, strict=False)
        assert set(msg.missing_keys) <= rope and not msg.unexpected_keys, msg
        self.n_prefix = int(self.model.num_prefix_tokens)
        assert self.n_prefix == 1 and self.model.global_pool == 'avg' and isinstance(self.model.norm, torch.nn.Identity), (
            self.n_prefix, self.model.global_pool, type(self.model.norm).__name__)
        state = {k: v for k, v in self.model.state_dict().items() if k not in rope}
        self.nap = weights.record(source=f'{REPO}@{revision}', file_name=WEIGHT_FILE,
                                  library=f'timm {timm.__version__} Eva (lop EVA_X + bo loc cua ho, nap chat tru bo dem RoPE {len(rope)})',
                                  model_state=state, file_tensors=sd, lib_info=None, nguon_tep=src, bo_dem_rope=sorted(rope),
                                  attention=f'timm fused_attn (SDPA) = {bool(timm.layers.use_fused_attn())}')

    def _blocks(self, x, n):
        '''Token THÔ của n khối cuối (= EVA_X.forward_features trước self.norm — vốn Identity khi global_pool avg).'''
        m = self.model
        x = m.patch_embed(x)
        x, rot_pos_embed = m._pos_embed(x)
        outs = []
        for i, blk in enumerate(m.blocks):
            x = blk(x, rope=rot_pos_embed)
            if i >= len(m.blocks) - n:
                outs.append(x)
        return outs

    def _patch_and_vector(self, h):
        raw = h[:, self.n_prefix:]
        return self.model.fc_norm(raw), self.model.fc_norm(raw.mean(1))


def build(input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None):
    return EvaX(input_mean, input_std, path=path).eval()
