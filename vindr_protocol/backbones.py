"""Backbone đông cứng. RAD-DINO công khai (= ô B của 85f, nhánh raddino, cùng cổng đổi tên khoá Dinov2); CheXFound theo MÃ GỐC của tác giả
(DinoVisionTransformer chép từng byte ở chexfound/dinov2/, dựng theo config.yaml như build_model_for_eval — PLAN_86 §11.3). CheXWorld / B thêm
khi dựng 86b của hàng đó (PLAN_86 §4.2)."""
import hashlib
import os
import re
import types
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from . import features
from .chexfound.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .config import CHEXFOUND_COMMIT, BackboneSpec

_DINOV2_RENAMES = (('.attention.attention.query.', '.attention.q_proj.'), ('.attention.attention.key.', '.attention.k_proj.'),
                   ('.attention.attention.value.', '.attention.v_proj.'), ('.attention.output.dense.', '.attention.o_proj.'))


def rename_dinov2_keys(state_dict, model_keys):
    '''state_dict Dinov2 -> tên khoá của mô hình đang chạy: đã khớp -> giữ nguyên; không -> đổi tên CŨ sang MỚI (chỉ đổi tên). = doi_ten_dinov2.'''
    if set(state_dict) <= set(model_keys): return dict(state_dict)
    out = {}
    for k, v in state_dict.items():
        for a, b in _DINOV2_RENAMES: k = k.replace(a, b)
        out[k] = v
    return out


@dataclass
class Backbone:
    '''kind: 'hf_dinov2' (Dinov2Model của transformers — RAD-DINO) | 'dinov2_original' (DinoVisionTransformer mã gốc — CheXFound) | 'b_luong' (B:
       Dinov2Model của nền notebook B + luồng vùng; parts = hàm / đối tượng của nền — b_from_base).'''
    name: str
    model: torch.nn.Module
    spec: BackboneSpec
    device: str
    amp: torch.dtype
    source: dict = field(default_factory=dict)
    kind: str = 'hf_dinov2'
    parts: dict = None

    @property
    def dim(self):
        return int(self.model.config.hidden_size) if self.kind in ('hf_dinov2', 'b_luong') else int(self.model.embed_dim)

    @property
    def patch_size(self):
        return int(self.model.config.patch_size) if self.kind in ('hf_dinov2', 'b_luong') else int(self.model.patch_size)

    def features(self, x, with_last=False):
        if self.kind == 'b_luong':
            return features.b_luong(self.model, self.parts, x, self.amp, self.device, with_last=with_last)
        fn = features.dinov2_hf if self.kind == 'hf_dinov2' else features.dinov2_original
        return fn(self.model, x, self.amp, self.device, with_last=with_last)


def b_from_base(model, luong, lo, out, neo_value, ref, device, amp, mean, std, image_size=518, source=None):
    '''B (PLAN_86 §14.2) từ ĐỐI TƯỢNG của nền notebook mốc B (Phần 3 skip7 tới C6S, như 85f) — gói KHÔNG chép mã B: model = backbone (Dinov2Model
       đông cứng), luong = LUONG (luồng vùng), lo(img, amp) = shared_lo với kiểu autocast đổi được, out = _out_luong, neo_value = neo của ảnh frontal
       (VinDr: mọi ảnh frontal), ref(x) = tầng cuối đủ token theo đường trích token của 85f (cổng). Chuẩn hoá như RAD-DINO, ảnh 518. -> Backbone.'''
    assert float(neo_value) in (0.0, 1.0), f'neo {neo_value} khong phai 0 / 1 -> DUNG'
    assert not model.training and not luong.training, 'backbone / luong vung cua B dang o che do train (dropout) -> goi .eval() truoc -> DUNG'
    neo = lambda n: torch.full((int(n),), float(neo_value), device=device)
    spec = BackboneSpec('B', int(image_size), (float(mean),) * 3, (float(std),) * 3)
    return Backbone('B', model, spec, device, amp, dict(source or {}, neo=float(neo_value)), kind='b_luong',
                    parts=dict(luong=luong, lo=lo, out=out, neo=neo, ref=ref))


def load_raddino(device, amp):
    '''microsoft/rad-dino công khai, ĐÔNG CỨNG, eval(); chuẩn hoá theo processor (thống kê MIMIC), ảnh 518. Cổng: trọng số from_pretrained = tệp
       model.safetensors trên hub sau khi đổi tên khoá (bắt thay đổi cách đặt tên của transformers).'''
    import transformers.utils.logging as tf_logging
    getattr(tf_logging, 'disable_progress_bar', lambda: None)()          # tắt thanh tiến độ tải / nạp của HF; không đổi số
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoImageProcessor, AutoModel, __version__ as tf_version
    proc = AutoImageProcessor.from_pretrained('microsoft/rad-dino')
    mean, std = float(proc.image_mean[0]), float(proc.image_std[0])
    model = AutoModel.from_pretrained('microsoft/rad-dino')
    sd = model.state_dict()
    hub = rename_dinov2_keys(load_file(hf_hub_download('microsoft/rad-dino', 'model.safetensors')), sd)
    assert set(hub) == set(sd) and all(torch.equal(hub[k].to(sd[k].dtype), sd[k]) for k in sd), (
        f'transformers {tf_version}: doi ten Dinov2 KHAC cach from_pretrained tu doi (tep hub RAD-DINO) -> DUNG')
    del hub, sd
    model = model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    spec = BackboneSpec('raddino', 518, (mean,) * 3, (std,) * 3)
    source = dict(bb='raddino', tep='hub microsoft/rad-dino (ban cong khai)', transformers=tf_version, kich_thuoc=518, chuan_hoa=dict(mean=mean, std=std))
    return Backbone('raddino', model, spec, device, amp, source)


# ==== CheXFound theo mã gốc (PLAN_86 §11.3) ====
CHEXFOUND_DRIVE = ('chexfound',)                        # cxrlt_2024/chexfound — CÙNG tệp với 85f (tải từ thư mục Google Drive công khai của tác giả)
CHEXFOUND_FILES = ('teacher_checkpoint.pth', 'config.yaml')
# md5 bản công khai của tác giả (tải 11/11/2026 từ thư mục Google Drive ở README CheXFound; Claude kiểm bộ nạp trên máy với chính các tệp này)
CHEXFOUND_MD5 = {'teacher_checkpoint.pth': '6976431a96978f27da057e1b9e35dcc4', 'config.yaml': 'a776cfd0d3e993f293d4434901c84e7e'}
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
    '''Import mã gốc (chexfound/dinov2): XFORMERS_DISABLED -> đường PyTorch thuần (giống nhau ở mọi môi trường); chặn ĐÚNG ba cảnh báo xFormers
       lúc import của các tệp layers (không chặn cảnh báo khác).'''
    os.environ['XFORMERS_DISABLED'] = '1'
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=_XFORMERS_WARNING)
        from .chexfound.dinov2 import vision_transformer
        from .chexfound.dinov2.layers.attention import MemEffAttention
    return vision_transformer, MemEffAttention


def build_chexfound(cfg, img_size=None, block_chunks=None):
    '''= build_model(cfg.student, only_teacher=True, img_size=cfg.crops.global_crops_size) của CheXFound (models/__init__.py), trên dict của
       config.yaml. block_chunks khác None -> dùng số này thay student.block_chunks (xem checkpoint_block_chunks). -> DinoVisionTransformer (khởi
       tạo ngẫu nhiên; nạp trọng số bằng load_teacher_weights).'''
    vits, _ = _import_dinov2()
    st = cfg['student']
    arch = str(st['arch']).removesuffix('_memeff')
    assert 'vit' in arch and arch in vits.__dict__, f'config.yaml: arch {arch!r} khong co trong vision_transformer goc -> DUNG'
    return vits.__dict__[arch](img_size=int(img_size or cfg['crops']['global_crops_size']), patch_size=st['patch_size'], init_values=st['layerscale'],
                               ffn_layer=st['ffn_layer'], block_chunks=(st['block_chunks'] if block_chunks is None else int(block_chunks)),
                               qkv_bias=st['qkv_bias'], proj_bias=st['proj_bias'], ffn_bias=st['ffn_bias'],
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


def load_teacher_weights(model, checkpoint):
    '''= load_pretrained_weights(model, ..., "teacher") của CheXFound (lấy khoá 'teacher', bỏ 'module.' và 'backbone.' bằng str.replace, nạp
       strict=False) + CỔNG CHẶT: không thiếu khoá nào của mô hình; khoá thừa CHỈ thuộc đầu DINO / iBOT. -> {so_khoa, thua (số theo tiền tố)}.'''
    sd = checkpoint['teacher'] if 'teacher' in checkpoint else checkpoint
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    sd = {k.replace('backbone.', ''): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    extra = sorted({k.split('.')[0] for k in msg.unexpected_keys})
    assert not msg.missing_keys and set(extra) <= {'dino_head', 'ibot_head'}, (
        f'checkpoint CheXFound: thieu khoa {list(msg.missing_keys)[:6]} / khoa thua ngoai dau DINO-iBOT {extra} -> DUNG')
    return dict(so_khoa=len(sd) - len(msg.unexpected_keys), thua={p: sum(k.startswith(p + '.') for k in msg.unexpected_keys) for p in extra})


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


def _md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''): h.update(chunk)
    return h.hexdigest()


def chexfound_from_files(checkpoint_path, config_path, device, amp, check_md5=True):
    '''CheXFound từ hai tệp trong máy: config.yaml -> build_chexfound -> nạp teacher (weights_only) -> attention SDPA -> eval(), ĐÔNG CỨNG.
       Ảnh = crops.global_crops_size (512), chuẩn hoá ImageNet (mặc định transforms của mã gốc). check_md5: tệp phải là bản công khai đã kiểm.'''
    import yaml
    md5 = {'teacher_checkpoint.pth': _md5(checkpoint_path), 'config.yaml': _md5(config_path)}
    if check_md5:
        assert md5 == CHEXFOUND_MD5, f'tep CheXFound KHAC ban cong khai da kiem 11/11 ({md5} != {CHEXFOUND_MD5}) -> DUNG, bao Claude'
    with open(config_path, encoding='utf-8') as f: cfg = yaml.safe_load(f)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    chunks = checkpoint_block_chunks(checkpoint)
    model = build_chexfound(cfg, block_chunks=chunks)
    report = load_teacher_weights(model, checkpoint)
    del checkpoint
    report['block_chunks'] = dict(config_yaml=cfg['student']['block_chunks'], checkpoint=chunks, dung=chunks)
    n_attn = use_sdpa_attention(model)
    model = model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    size = int(cfg['crops']['global_crops_size'])
    spec = BackboneSpec('chexfound', size, tuple(IMAGENET_DEFAULT_MEAN), tuple(IMAGENET_DEFAULT_STD))
    st = cfg['student']
    source = dict(bb='chexfound', tep=f"cxrlt_2024/{'/'.join(CHEXFOUND_DRIVE)}/teacher_checkpoint.pth (+ config.yaml)", md5=md5,
                  ma=f'CheXFound @ {CHEXFOUND_COMMIT[:7]}: DinoVisionTransformer goc (vindr_protocol/chexfound/dinov2)', attention=f'SDPA ({n_attn} khoi)',
                  kich_thuoc=size, chuan_hoa=dict(mean=list(spec.mean), std=list(spec.std)), nap=report, transformers=None,
                  cau_hinh={k: st[k] for k in ('arch', 'patch_size', 'ffn_layer', 'block_chunks', 'num_register_tokens', 'layerscale',
                                                'interpolate_offset', 'interpolate_antialias')})
    return Backbone('chexfound', model, spec, device, amp, source, kind='dinov2_original')


def load_chexfound(device, amp, kiem_ke, tai, xung_dot, log=print):
    '''CheXFound từ Drive (cxrlt_2024/chexfound, đọc QUA API bằng khối đọc chuẩn của notebook: kiem_ke / tai / xung_dot) -> chexfound_from_files.'''
    kk = kiem_ke(CHEXFOUND_DRIVE)[1]
    paths, ids = {}, {}
    for name in CHEXFOUND_FILES:
        versions = kk.get(name)
        assert versions, f"thieu {name} trong cxrlt_2024/{'/'.join(CHEXFOUND_DRIVE)} -> tai tu Google Drive cua tac gia (README CheXFound). DUNG"
        if xung_dot(versions): log(f"  ! {name}: {len(versions)} ban NOI DUNG KHAC -> dung ban MOI NHAT ({versions[0]['mt']})")
        paths[name], ids[name] = tai(versions[0]), dict(id=versions[0]['id'], so_ban=len(versions))
    bb = chexfound_from_files(paths['teacher_checkpoint.pth'], paths['config.yaml'], device, amp)
    bb.source['drive'] = ids
    return bb
