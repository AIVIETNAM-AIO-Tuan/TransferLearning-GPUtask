# -*- coding: utf-8 -*-
"""CheXFound (RPIDIAL/CheXFound) — lớp bọc cho VER2 (PLAN_89 §2, §10 bước 3). Ver1 KHÔNG dùng tệp này: Ver1 chạy mô hình + bộ nạp của chính mã họ
(protocols/chexfound_env, như `86f`).

Mô hình = DinoVisionTransformer của họ (encoders/chexfound_src/dinov2 = chexfound/{models,layers} @ 7e288d0, chép từ module 86), ViT-L/16 + 4 register,
ảnh 512, chuẩn hoá ImageNet. Trọng số = khoá 'teacher' của teacher_checkpoint.pth (cxrlt_2024/chexfound, md5 ghim ở spec), bỏ tiền tố 'module.' /
'backbone.' như load_pretrained_weights của họ. Hai điểm đã gặp ở 86 (PLAN_86 §11.3) và giữ nguyên cách xử lý:

  - `config.yaml` phát hành ghi student.block_chunks = 0 nhưng checkpoint lưu 4 khúc x 6 tầng: dựng theo config rồi nạp strict=False thì 24 tầng GIỮ
    khởi tạo ngẫu nhiên, không báo lỗi -> số khúc lấy theo khoá checkpoint + cổng chặt (không thiếu khoá; khoá thừa chỉ thuộc đầu DINO / iBOT).
  - Không xFormers ở môi trường của ta: attention của mã gốc rơi về nhánh tường minh (ma trận [B, đầu, N, N]) -> thay bằng
    F.scaled_dot_product_attention (cùng phép toán, kernel hợp nhất — như memory_efficient_attention mà tác giả chạy).

Đặc trưng: n tầng cuối qua `norm` = get_intermediate_layers(norm=True) của họ; vector toàn ảnh = CLS; token patch = bỏ CLS + 4 register.
_sdpa_attention_forward / checkpoint_block_chunks / use_sdpa_attention CHÉP NGUYÊN VĂN từ vindr_protocol/backbones.py của 86 (tests so cây cú pháp)."""
import os
import re
import types
import warnings

import torch
import torch.nn.functional as F

from ..checks import safe_load, weights
from . import files
from .base import EncoderWrapper
from .spec import IMAGENET_MEAN, IMAGENET_STD, SPECS

COMMIT = '7e288d06532f24abedb7d00781185c36727d9d69'
# student.* của config.yaml phát hành (md5 dưới đây; tests/test_encoders.py so với tệp thật) — trừ block_chunks (lấy theo checkpoint, xem trên).
CONFIG_MD5 = 'a776cfd0d3e993f293d4434901c84e7e'
STUDENT = dict(arch='vit_large', patch_size=16, layerscale=1e-05, ffn_layer='swiglufused', qkv_bias=True, proj_bias=True, ffn_bias=True,
               num_register_tokens=4, interpolate_antialias=False, interpolate_offset=0.1)
IMG_SIZE = 512                      # crops.global_crops_size của config.yaml
HEAD_PREFIXES = ('dino_head', 'ibot_head')
_XFORMERS_WARNING = r'xFormers is (not available|disabled|available) \((Attention|Block|SwiGLU)\)'


def _sdpa_attention_forward(self, x, attn_bias=None, return_attention=False):
    '''Thay MemEffAttention.forward của mã gốc khi KHÔNG có xFormers: mã gốc rơi về nhánh tường minh (softmax(q k^T / sqrt(d)) v, tạo ma trận
       [B, đầu, N, N]); ở đây cùng phép toán bằng F.scaled_dot_product_attention (kernel hợp nhất, như memory_efficient_attention của xFormers mà
       tác giả chạy). Không có attn_bias / trả attention (mã gốc chỉ dùng hai thứ đó cho nested tensor / bản đồ attention).'''
    if attn_bias is not None or return_attention:
        raise AssertionError('attention SDPA cua 86: khong ho tro attn_bias / return_attention -> DUNG')
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], dropout_p=0.0)
    x = x.transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(x))


def _import_dinov2():
    '''Import mã gốc: XFORMERS_DISABLED trong lúc import -> đường PyTorch thuần (giống nhau ở mọi môi trường); trả biến môi trường về như cũ; chặn
       ĐÚNG ba cảnh báo xFormers lúc import của các tệp layers.'''
    old = os.environ.get('XFORMERS_DISABLED')
    os.environ['XFORMERS_DISABLED'] = '1'
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=_XFORMERS_WARNING)
            from .chexfound_src.dinov2 import vision_transformer
            from .chexfound_src.dinov2.layers.attention import MemEffAttention
    finally:
        if old is None: os.environ.pop('XFORMERS_DISABLED', None)
        else: os.environ['XFORMERS_DISABLED'] = old
    return vision_transformer, MemEffAttention


def build_model(block_chunks):
    '''= build_model(cfg.student, only_teacher=True, img_size=cfg.crops.global_crops_size) của CheXFound với STUDENT; block_chunks theo checkpoint.'''
    vits, _ = _import_dinov2()
    st = STUDENT
    return vits.__dict__[st['arch']](img_size=IMG_SIZE, patch_size=st['patch_size'], init_values=st['layerscale'], ffn_layer=st['ffn_layer'],
                                     block_chunks=int(block_chunks), qkv_bias=st['qkv_bias'], proj_bias=st['proj_bias'], ffn_bias=st['ffn_bias'],
                                     num_register_tokens=st['num_register_tokens'], interpolate_offset=st['interpolate_offset'],
                                     interpolate_antialias=st['interpolate_antialias'])


def checkpoint_block_chunks(checkpoint):
    '''Số khúc tầng (BlockChunk của DINOv2 / FSDP) trong khoá checkpoint: 'blocks.<khúc>.<i>.' -> số khúc khác nhau; 'blocks.<i>.' -> 0.
       Bản công khai (11/11): 4 khúc x 6 tầng, trong khi config.yaml phát hành ghi block_chunks: 0 — dựng theo config đó rồi nạp strict=False như
       load_pretrained_weights của họ thì 24 tầng GIỮ khởi tạo ngẫu nhiên mà không báo lỗi (PLAN_86 §11.3). -> số khúc.'''
    sd = checkpoint['teacher'] if 'teacher' in checkpoint else checkpoint
    keys = [k.replace('module.', '').replace('backbone.', '') for k in sd]
    chunked = {k.split('.')[1] for k in keys if re.match(r'^blocks\.\d+\.\d+\.', k)}
    flat = [k for k in keys if re.match(r'^blocks\.\d+\.[a-z]', k)]
    assert not (chunked and flat), 'checkpoint CheXFound vua co khoa tang chia khuc vua co khoa phang -> DUNG'
    return len(chunked)


def use_sdpa_attention(model):
    '''Gắn _sdpa_attention_forward cho MỌI MemEffAttention của mô hình (mức đối tượng, không sửa lớp). -> số module đã gắn.'''
    _, mem_eff = _import_dinov2()
    n = 0
    for m in model.modules():
        if isinstance(m, mem_eff):
            m.forward = types.MethodType(_sdpa_attention_forward, m)
            n += 1
    assert n > 0, 'khong thay MemEffAttention nao -> DUNG'
    return n


class CheXFound(EncoderWrapper):
    def __init__(self, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None):
        spec = SPECS['chexfound']
        super().__init__(spec, input_mean, input_std)
        path = path or files.weights_path(spec)
        ck, src = safe_load.load_checkpoint(path, expect_md5=spec.file_md5, expect_size=spec.file_size)
        assert isinstance(ck, dict) and 'teacher' in ck, 'CheXFound: checkpoint khong co khoa teacher -> DUNG'
        chunks = checkpoint_block_chunks(ck)
        sd = {k.replace('module.', ''): v for k, v in ck['teacher'].items()}
        sd = {k.replace('backbone.', ''): v for k, v in sd.items()}                       # = load_pretrained_weights(..., 'teacher') của họ
        del ck
        self.model = build_model(chunks)
        msg = self.model.load_state_dict(sd, strict=False)
        extra = sorted({k.split('.')[0] for k in msg.unexpected_keys})
        assert not msg.missing_keys and set(extra) <= set(HEAD_PREFIXES), (
            f'checkpoint CheXFound: thieu khoa {list(msg.missing_keys)[:6]} / khoa thua ngoai dau DINO-iBOT {extra} -> DUNG')
        n_attn = use_sdpa_attention(self.model)
        self.n_prefix = 1 + int(self.model.num_register_tokens)
        assert (self.model.patch_size, self.model.embed_dim, self.n_prefix) == (spec.patch_size, spec.embed_dim, 5), (
            self.model.patch_size, self.model.embed_dim, self.n_prefix)
        body = {k: v for k, v in sd.items() if k.split('.')[0] not in HEAD_PREFIXES}
        self.nap = weights.record(source='cxrlt_2024/chexfound (thu muc Google Drive cong khai o README RPIDIAL/CheXFound)', file_name=spec.drive_file,
                                  library=f'ma goc CheXFound @ {COMMIT[:7]} (encoders/chexfound_src), torch {torch.__version__}',
                                  model_state=self.model.state_dict(), file_tensors=body, lib_info=None, nguon_tep=src,
                                  block_chunks=dict(config_yaml=0, checkpoint=chunks, dung=chunks),
                                  khoa_dau_bo_qua={p: sum(k.startswith(p + '.') for k in sd) for p in extra}, attention=f'SDPA ({n_attn} khoi)')

    def _blocks(self, x, n):
        m = self.model
        outs = m._get_intermediate_layers_chunked(x, n) if m.chunked_blocks else m._get_intermediate_layers_not_chunked(x, n)
        return [m.norm(o) for o in outs]


def build(input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None):
    return CheXFound(input_mean, input_std, path).eval()
