"""Chín encoder của giao thức: spec.SPECS = bảng mô tả; build(name) = lớp bọc đã nạp trọng số gốc + bản ghi cổng nạp (`.nap`).

    from vindr_bench import encoders
    enc = encoders.build('raddino')            # eval(), CPU; bên gọi chuyển thiết bị
    outs = enc.get_intermediate_layers(x, 4)   # x: lô ảnh đúng độ phân giải gốc, chuẩn hoá ImageNet (mặc định) -> 4 x (patch, vector toàn ảnh)
"""
from .spec import ALL_ROWS, EXTRA_ROWS, ORDER, SPECS, EncoderSpec          # noqa: F401
from ..checks import weights as _weights

# lớp bọc + kiểm tệp thật. 'chexfound': lớp bọc CHỈ cho Ver2 — Ver1 chạy mô hình + bộ nạp của chính mã họ (notebook/ver1.THEIR_MODEL).
# 'b' (15/11): đông cứng như mọi backbone (Tuấn); Ver1 từ 0.8.0 (khúc ô 64: protocols/chexfound_env.BUILD_KW).
# 'asg' (0.10.0, PLAN_90 Việc A): hàng thêm ngoài 9 hàng đã khoá (spec.EXTRA_ROWS) — encoder có giám sát giải phẫu của người khác.
BUILT = ('b', 'chexfound', 'raddino', 'biomedclip', 'mrm', 'chexzero', 'evax', 'pubmedclip', 'chexworld', 'asg')


def build(name, check=True, **kw):
    '''-> lớp bọc ở eval() (CPU). check=True: cổng nạp trọng số trượt -> AssertionError; check=False: trả kèm nap['dat'] = False để bên gọi ghi lại rồi
       tự dừng. kw chuyển cho hàm build của encoder (vd input_mean / input_std = chuẩn hoá bên gọi đã áp; raddino: attn='eager').'''
    assert name in SPECS, f'encoder {name!r} khong thuoc {ALL_ROWS} -> DUNG'
    if name not in BUILT:
        raise NotImplementedError(f'{name}: lop boc chua dung (spec.status = {SPECS[name].status})')
    import importlib
    enc = importlib.import_module(f'{__name__}.{name}').build(**kw)
    assert enc.nap is not None and enc.spec.name == name
    enc.nap['encoder'] = name
    if check:
        assert enc.nap['dat'], f'{name}: nap trong so KHONG dat: {_weights.loading_text(enc.nap)} -> DUNG'
    return enc
