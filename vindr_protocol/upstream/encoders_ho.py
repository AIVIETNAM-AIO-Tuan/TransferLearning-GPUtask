# -*- coding: utf-8 -*-
"""86j (PLAN_86 §16): lớp bọc encoder KHÁC CheXFound để chạy bằng linear_glori.py của họ. Mã họ chỉ gọi get_intermediate_layers(x, n,
return_class_token=True) (ModelWithIntermediateLayers; đầu dựng theo hình đầu ra mẫu) -> mỗi lớp bọc chỉ cần hàm đó. Chạy được trong môi trường của họ
(Python 3.9 + torch 2.0 + transformers 4.36.2 + open_clip 2.23.0) lẫn môi trường mới (đặc trưng tham chiếu). Không import gì của vindr_protocol.

Đầu vào = biến đổi của họ (RescaleImage + chuẩn hoá ImageNet), ảnh theo --image-size của lượt.
- raddino_kieu_ho  khuôn lớp bọc HF của chính họ (chexfound/eval/utils.py, vd ViTLargeImagenet21k): hidden_states[-n:] THÔ (không layernorm cuối),
                   CLS = token 0, patch = phần còn lại; không chuẩn hoá lại (ImageNet) — ĐOÁN cách bài chạy RAD-DINO (repo không có lớp bọc RAD-DINO).
- raddino_chuan    n tầng cuối QUA layernorm cuối (= get_intermediate_layers(norm=True) của DINOv2); chuẩn hoá lại ImageNet -> thống kê processor
                   RAD-DINO (0,5307 / 0,2583) bằng phép affine trong lớp bọc — = đặc trưng RAD-DINO của 86b (khác thư viện / độ chính xác).
- biomedclip_ho    BiomedCLIPBase NGUYÊN VĂN của họ (chexfound.eval.utils): trả [(None, vector 512)] — chỉ dùng được cho Linear.

Cổng nạp trọng số (13/11, theo rà soát: bản trước probe_features ghi thieu_khoa = 0 CỐ ĐỊNH — giá trị giả định, không phải kết quả kiểm): build()
gắn `nap` = kết quả kiểm THẬT trong chính tiến trình (probe / lượt học):
  - thong_tin_nap: output_loading_info của transformers (khoá thiếu / thừa / sai hình / lỗi). open_clip không trả danh sách này -> None (BiomedCLIP; bộ nạp
    open_clip của cả 2.23 lẫn 4.x gọi load_checkpoint strict=True — đọc mã 13/11), KHÔNG ghi danh sách rỗng thay cho nó.
  - so_voi_tep: so TỪNG tensor của mô hình đã nạp với tệp trọng số trên đĩa (cùng tệp thư viện tải): bằng từng bit, không thiếu, không thừa — bắt cả lỗi
    mà output_loading_info không báo (transformers 5.x bỏ conversion_errors khỏi output_loading_info; trọng số bị khởi tạo đè sau khi nạp)."""
import collections
import hashlib
import re

import torch
import torch.nn as nn

RADDINO_ID = 'microsoft/rad-dino'
RADDINO_REV = '110cbc18d5133582e320b43d53bf5c44e410c936'      # main 13/11 (model.safetensors không đổi từ lần tải lên); preprocessor 0,5307 / 0,2583
RADDINO_FILE = 'model.safetensors'                             # tệp transformers nạp (repo có safetensors -> ưu tiên)
BIOMEDCLIP_ID = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'     # = chuỗi hf-hub trong BiomedCLIPBase của họ (không ghim revision)
BIOMEDCLIP_FILE = 'open_clip_pytorch_model.bin'                # tệp trọng số DUY NHẤT của repo (13/11: không có safetensors) — open_clip 2.23 / 4.x đều nạp tệp này
RADDINO_MEAN, RADDINO_STD = (0.5307,) * 3, (0.2583,) * 3
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)     # chexfound/data/transforms.py (mặc định biến đổi của họ)
NAMES = ('raddino_kieu_ho', 'raddino_chuan', 'biomedclip_ho')
IMAGE_SIZE = {'raddino_kieu_ho': 512, 'raddino_chuan': 518, 'biomedclip_ho': 224}
HEADS = {'raddino_kieu_ho': ('lin', 'glori'), 'raddino_chuan': ('lin', 'glori'), 'biomedclip_ho': ('lin',)}

# transformers >= 5.18 đổi tên attention của Dinov2 lúc nạp (WeightRenaming trong conversion_mapping.py: attention.attention.query -> attention.q_proj, …;
# đọc wheel 5.17.0 / 5.18.0 / 5.22.2 ngày 13/11), tệp hub giữ tên cũ -> đưa tên mô hình về tên tệp trước khi so. Đổi tên khác chưa biết: so theo giá trị.
_DINOV2_NEW_TO_OLD = (('.attention.q_proj.', '.attention.attention.query.'), ('.attention.k_proj.', '.attention.attention.key.'),
                      ('.attention.v_proj.', '.attention.attention.value.'), ('.attention.o_proj.', '.attention.output.dense.'))
_LIST_MAX = 20


def _old_name(k):
    for a, b in _DINOV2_NEW_TO_OLD:
        k = k.replace(a, b)
    return k


def _digest(t):
    '''Dấu nội dung (hình + sha1 các giá trị) — đổi kiểu số thực sang float64 trước (chính xác với fp16 / bf16 / fp32) -> bằng nhau <=> cùng giá trị.'''
    t = t.detach().cpu()
    t = t.double() if t.is_floating_point() else t
    return tuple(t.shape), hashlib.sha1(t.contiguous().numpy().tobytes()).hexdigest()


def compare_to_file(state, file_tensors, prefix=''):
    '''So TỪNG tensor của mô hình đã nạp (`state` = state_dict) với tệp trọng số (khoá tệp bắt đầu bằng `prefix`, vd 'visual.', bỏ tiền tố trước khi so).
       Ghép theo tên (tên Dinov2 mới đưa về tên cũ); khoá còn lại hai phía ghép theo NỘI DUNG (đa tập dấu) — đổi tên chưa biết vẫn so được giá trị.
       Ghép theo tên: hình phải trùng và giá trị BẰNG từng bit (ép tensor mô hình về kiểu của tệp). -> dict(so_khoa, so_khoa_tep, trung_ten,
       trung_gia_tri_khac_ten, so_thieu, thieu, so_thua, thua, so_khac_hinh, khac_hinh, so_khac_gia_tri, khac_gia_tri, kieu_tep) — danh sách cắt 20.'''
    model = {_old_name(k): v for k, v in state.items()}
    file = {_old_name(k[len(prefix):]): v for k, v in file_tensors.items() if k.startswith(prefix)}
    both = sorted(set(model) & set(file))
    shape = [k for k in both if tuple(model[k].shape) != tuple(file[k].shape)]
    value = [k for k in both if k not in shape and not torch.equal(model[k].detach().cpu().to(file[k].dtype), file[k])]
    rest_m = sorted(set(model) - set(file))
    rest_f = sorted(set(file) - set(model))
    pool = collections.Counter(_digest(file[k]) for k in rest_f)
    by_value, missing = [], []
    for k in rest_m:
        d = _digest(model[k])
        if pool[d] > 0:
            pool[d] -= 1
            by_value.append(k)
        else:
            missing.append(k)
    used = collections.Counter(_digest(model[k]) for k in by_value)
    extra = []
    for k in rest_f:
        d = _digest(file[k])
        if used[d] > 0:
            used[d] -= 1
        else:
            extra.append(k)
    return dict(so_khoa=len(model), so_khoa_tep=len(file), trung_ten=len(both) - len(shape) - len(value), trung_gia_tri_khac_ten=len(by_value),
                so_thieu=len(missing), thieu=missing[:_LIST_MAX], so_thua=len(extra), thua=extra[:_LIST_MAX], so_khac_hinh=len(shape),
                khac_hinh=shape[:_LIST_MAX], so_khac_gia_tri=len(value), khac_gia_tri=value[:_LIST_MAX],
                kieu_tep=sorted({str(v.dtype) for v in file.values()}))


def loading_ok(nap):
    '''Luật cổng nạp: thông tin nạp của thư viện (nếu có) không khoá thiếu / thừa / sai hình / lỗi, VÀ mọi tensor của mô hình = tệp (không thiếu, không
       thừa, không khác hình / giá trị), có ít nhất một tensor.'''
    t, s = nap['thong_tin_nap'], nap['so_voi_tep']
    lib_ok = t is None or not (t['thieu'] or t['thua'] or t['sai_hinh'] or t['loi'])
    return bool(lib_ok and s['so_khoa'] > 0 and s['so_thieu'] == 0 and s['so_thua'] == 0 and s['so_khac_hinh'] == 0 and s['so_khac_gia_tri'] == 0
                and s['trung_ten'] + s['trung_gia_tri_khac_ten'] == s['so_khoa'])


def loading_text(nap):
    '''Một dòng tóm tắt `nap` cho output notebook / log.'''
    t, s = nap['thong_tin_nap'], nap['so_voi_tep']
    lib = ('thu vien khong tra thong tin nap (open_clip: nap strict)' if t is None else
           f"output_loading_info: thieu {len(t['thieu'])} / thua {len(t['thua'])} / sai hinh {len(t['sai_hinh'])} / loi {len(t['loi'])}")
    return (f"nap trong so {'DAT' if nap['dat'] else 'TRUOT'} | {lib} | so voi tep {nap['tep']}: {s['trung_ten'] + s['trung_gia_tri_khac_ten']}/"
            f"{s['so_khoa']} tensor bang tung bit (khac ten cung gia tri {s['trung_gia_tri_khac_ten']}), thieu {s['so_thieu']}, thua {s['so_thua']}, "
            f"khac hinh {s['so_khac_hinh']}, khac gia tri {s['so_khac_gia_tri']} | {nap['thu_vien']}")


def _snapshot(path):
    m = re.search(r'snapshots[\\/]([0-9a-f]{40})[\\/]', path)
    return m.group(1) if m else None


def _strs(x):
    return sorted(str(v) for v in (x or []))


class RadDinoHF(nn.Module):
    '''RAD-DINO (HF Dinov2Model). norm_last: n tầng cuối qua layernorm cuối; renorm: ImageNet -> thống kê RAD-DINO trước mô hình.
       Thuộc tính `nap` = kết quả kiểm nạp trọng số (xem đầu tệp); build() quyết đạt / dừng.'''

    def __init__(self, norm_last, renorm, revision=RADDINO_REV):
        super().__init__()
        import transformers
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        self.model, info = transformers.AutoModel.from_pretrained(RADDINO_ID, revision=revision, output_loading_info=True)
        path = hf_hub_download(RADDINO_ID, RADDINO_FILE, revision=revision)
        self.nap = dict(nguon=f'{RADDINO_ID}@{revision}', tep=RADDINO_FILE, ban_tep=_snapshot(path), thu_vien=f'transformers {transformers.__version__}',
                        kieu_mo_hinh=str(next(self.model.parameters()).dtype),
                        thong_tin_nap=dict(thieu=_strs(info.get('missing_keys')), thua=_strs(info.get('unexpected_keys')),
                                           sai_hinh=_strs(info.get('mismatched_keys')), loi=_strs(info.get('error_msgs'))),
                        so_voi_tep=compare_to_file(self.model.state_dict(), load_file(path)))
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


def _biomedclip_loading(m):
    '''BiomedCLIPBase của họ (m.model = model.visual của open_clip; m.norm là LayerNorm KHÔNG nạp, KHÔNG dùng trong forward / get_intermediate_layers):
       open_clip không trả thông tin nạp -> so TỪNG tensor của m.model với khoá 'visual.*' của tệp open_clip trong bộ đệm HF (cùng tệp open_clip tải).'''
    import importlib.metadata
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(BIOMEDCLIP_ID, BIOMEDCLIP_FILE)
    sd = torch.load(path, map_location='cpu', weights_only=True)
    sd = sd.get('state_dict', sd)
    try:
        oc = importlib.metadata.version('open_clip_torch')
    except Exception as e:
        oc = f'khong ro ({type(e).__name__})'
    return dict(nguon=f'hf-hub:{BIOMEDCLIP_ID}', tep=BIOMEDCLIP_FILE, ban_tep=_snapshot(path), thu_vien=f'open_clip {oc}',
                kieu_mo_hinh=str(next(m.model.parameters()).dtype), thong_tin_nap=None, so_voi_tep=compare_to_file(m.model.state_dict(), sd, prefix='visual.'))


def build(name, check=True):
    '''-> lớp bọc ở chế độ eval (CPU; tệp chạy chuyển .cuda()), thuộc tính `nap` = kết quả kiểm nạp trọng số THẬT + nap['dat'] (loading_ok).
       check=True: trượt -> AssertionError (dừng tiến trình). check=False: trả mô hình kèm nap['dat'] = False để bên gọi ghi lại rồi dừng.
       biomedclip_ho cần repo CheXFound trên sys.path (import lớp bọc của họ).'''
    assert name in NAMES, f'encoder {name!r} khong thuoc {NAMES} -> DUNG'
    if name == 'raddino_kieu_ho':
        m = RadDinoHF(norm_last=False, renorm=False)
    elif name == 'raddino_chuan':
        m = RadDinoHF(norm_last=True, renorm=True)
    else:
        from chexfound.eval.utils import BiomedCLIPBase
        m = BiomedCLIPBase()
        m.nap = _biomedclip_loading(m)
    m.nap['encoder'] = name
    m.nap['dat'] = loading_ok(m.nap)
    if check:
        assert m.nap['dat'], f'{name}: nap trong so KHONG dat: {loading_text(m.nap)} | {m.nap} -> DUNG'
    return m.eval()
