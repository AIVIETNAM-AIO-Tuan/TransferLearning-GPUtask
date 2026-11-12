# -*- coding: utf-8 -*-
"""86j (PLAN_86 §16): lớp bọc encoder KHÁC CheXFound để chạy bằng linear_glori.py của họ. Mã họ chỉ gọi get_intermediate_layers(x, n,
return_class_token=True) (ModelWithIntermediateLayers; đầu dựng theo hình đầu ra mẫu) -> mỗi lớp bọc chỉ cần hàm đó. Chạy được trong môi trường của họ
(Python 3.9 + torch 2.0 + transformers 4.36.2 + open_clip 2.23.0) lẫn môi trường mới (đặc trưng tham chiếu). Không import gì của vindr_protocol.

Đầu vào = biến đổi của họ (RescaleImage + chuẩn hoá ImageNet), ảnh theo --image-size của lượt.
- raddino_kieu_ho  khuôn lớp bọc HF của chính họ (chexfound/eval/utils.py, vd ViTLargeImagenet21k): hidden_states[-n:] THÔ (không layernorm cuối),
                   CLS = token 0, patch = phần còn lại; không chuẩn hoá lại (ImageNet) — ĐOÁN cách bài chạy RAD-DINO (repo không có lớp bọc RAD-DINO).
- raddino_chuan    n tầng cuối QUA layernorm cuối (= get_intermediate_layers(norm=True) của DINOv2); chuẩn hoá lại ImageNet -> thống kê processor
                   RAD-DINO (0,5307 / 0,2583) bằng phép affine trong lớp bọc — = đặc trưng RAD-DINO của 86b (khác thư viện / độ chính xác).
- biomedclip_ho    BiomedCLIPBase NGUYÊN VĂN của họ (chexfound.eval.utils): trả [(None, vector 512)] — chỉ dùng được cho Linear."""
import torch
import torch.nn as nn

RADDINO_ID = 'microsoft/rad-dino'
RADDINO_REV = '110cbc18d5133582e320b43d53bf5c44e410c936'      # main 13/11 (model.safetensors không đổi từ lần tải lên); preprocessor 0,5307 / 0,2583
RADDINO_MEAN, RADDINO_STD = (0.5307,) * 3, (0.2583,) * 3
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)     # chexfound/data/transforms.py (mặc định biến đổi của họ)
NAMES = ('raddino_kieu_ho', 'raddino_chuan', 'biomedclip_ho')
IMAGE_SIZE = {'raddino_kieu_ho': 512, 'raddino_chuan': 518, 'biomedclip_ho': 224}
HEADS = {'raddino_kieu_ho': ('lin', 'glori'), 'raddino_chuan': ('lin', 'glori'), 'biomedclip_ho': ('lin',)}


class RadDinoHF(nn.Module):
    '''RAD-DINO (HF Dinov2Model). norm_last: n tầng cuối qua layernorm cuối; renorm: ImageNet -> thống kê RAD-DINO trước mô hình.'''

    def __init__(self, norm_last, renorm, revision=RADDINO_REV):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(RADDINO_ID, revision=revision)
        self.norm_last, self.renorm = bool(norm_last), bool(renorm)
        a = torch.tensor([i / r for i, r in zip(IMAGENET_STD, RADDINO_STD)]).view(1, 3, 1, 1)
        b = torch.tensor([(m - rm) / r for m, rm, r in zip(IMAGENET_MEAN, RADDINO_MEAN, RADDINO_STD)]).view(1, 3, 1, 1)
        self.register_buffer('scale', a)                 # x_rd = ((x * std_in + mean_in) - mean_rd) / std_rd = x * scale + shift
        self.register_buffer('shift', b)

    def get_intermediate_layers(self, x, n_last_blocks, return_class_token=True):
        if self.renorm:
            x = x * self.scale.to(x.dtype) + self.shift.to(x.dtype)
        hidden = self.model(x, output_hidden_states=True).hidden_states[-int(n_last_blocks):]
        if self.norm_last:
            hidden = [self.model.layernorm(h) for h in hidden]
        patch, cls = [h[:, 1:] for h in hidden], [h[:, 0] for h in hidden]
        return tuple(zip(patch, cls)) if return_class_token else tuple(patch)

    def forward(self, x):
        return self.get_intermediate_layers(x, 1)[0][1]


def build(name):
    '''-> lớp bọc ở chế độ eval (CPU; tệp chạy chuyển .cuda()). biomedclip_ho cần repo CheXFound trên sys.path (import lớp bọc của họ).'''
    assert name in NAMES, f'encoder {name!r} khong thuoc {NAMES} -> DUNG'
    if name == 'raddino_kieu_ho':
        m = RadDinoHF(norm_last=False, renorm=False)
    elif name == 'raddino_chuan':
        m = RadDinoHF(norm_last=True, renorm=True)
    else:
        from chexfound.eval.utils import BiomedCLIPBase
        m = BiomedCLIPBase()
    return m.eval()
