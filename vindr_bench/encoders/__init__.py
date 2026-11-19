"""Chín encoder của giao thức: spec.SPECS = bảng mô tả; build(name) = lớp bọc đã nạp trọng số gốc + bản ghi cổng nạp (`.nap`).

    from vindr_bench import encoders
    enc = encoders.build('raddino')            # eval(), CPU; bên gọi chuyển thiết bị
    outs = enc.get_intermediate_layers(x, 4)   # x: lô ảnh đúng độ phân giải gốc, chuẩn hoá ImageNet (mặc định) -> 4 x (patch, vector toàn ảnh)
"""
from .spec import ABLATION_REF, ABLATION_ROWS, ABLATION_STORE, ALL_ROWS, EXTRA_ROWS, ORDER, SPECS, EncoderSpec          # noqa: F401
from ..checks import weights as _weights

# lớp bọc + kiểm tệp thật. 'chexfound': lớp bọc CHỈ cho Ver2 — Ver1 chạy mô hình + bộ nạp của chính mã họ (notebook/ver1.THEIR_MODEL).
# 'b' (15/11): đông cứng như mọi backbone (Tuấn); Ver1 từ 0.8.0 (khúc ô 64: protocols/chexfound_env.BUILD_KW).
# 'asg' (0.10.0, PLAN_90 Việc A): hàng thêm ngoài 9 hàng đã khoá (spec.EXTRA_ROWS) — encoder có giám sát giải phẫu của người khác.
# 5 hàng Việc B (0.11.0, PLAN_90 §10.2): spec.ABLATION_ROWS, MỘT tệp lớp bọc encoders/vt.py (MODULES); chỉ Ver2 kho kcb.
BUILT = ('b', 'chexfound', 'raddino', 'biomedclip', 'mrm', 'chexzero', 'evax', 'pubmedclip', 'chexworld', 'asg') + ABLATION_ROWS
MODULES = {r: 'vt' for r in ABLATION_ROWS}      # tên hàng -> tệp lớp bọc khi khác tên hàng; build(name, ...) của tệp đó nhận tên hàng


def build(name, check=True, **kw):
    '''-> lớp bọc ở eval() (CPU). check=True: cổng nạp trọng số trượt -> AssertionError; check=False: trả kèm nap['dat'] = False để bên gọi ghi lại rồi
       tự dừng. kw chuyển cho hàm build của encoder (vd input_mean / input_std = chuẩn hoá bên gọi đã áp; raddino: attn='eager').'''
    assert name in SPECS, f'encoder {name!r} khong thuoc {ALL_ROWS + ABLATION_ROWS} -> DUNG'
    if name not in BUILT:
        raise NotImplementedError(f'{name}: lop boc chua dung (spec.status = {SPECS[name].status})')
    import importlib
    mod = importlib.import_module(f'{__name__}.{MODULES.get(name, name)}')
    enc = mod.build(name, **kw) if name in MODULES else mod.build(**kw)
    assert enc.nap is not None and enc.spec.name == name
    enc.nap['encoder'] = name
    if check:
        assert enc.nap['dat'], f'{name}: nap trong so KHONG dat: {_weights.loading_text(enc.nap)} -> DUNG'
    return enc
