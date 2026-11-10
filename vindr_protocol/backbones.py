"""Backbone đông cứng. RAD-DINO công khai (= ô B của 85f, nhánh raddino, cùng cổng đổi tên khoá Dinov2); CheXFound / CheXWorld / B thêm
khi dựng 86b (PLAN_86 §4.2)."""
from dataclasses import dataclass, field

import torch

from . import features
from .config import BackboneSpec

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
    name: str
    model: torch.nn.Module
    spec: BackboneSpec
    device: str
    amp: torch.dtype
    source: dict = field(default_factory=dict)

    @property
    def dim(self):
        return int(self.model.config.hidden_size)

    def features(self, x, with_last=False):
        return features.dinov2_hf(self.model, x, self.amp, self.device, with_last=with_last)


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
