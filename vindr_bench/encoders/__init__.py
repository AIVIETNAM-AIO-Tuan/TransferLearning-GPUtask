"""Chín encoder của giao thức: spec.SPECS = bảng mô tả; build(name) = lớp bọc đã nạp trọng số gốc + bản ghi cổng nạp (`.nap`).

    from vindr_bench import encoders
    enc = encoders.build('raddino')            # eval(), CPU; bên gọi chuyển thiết bị
    outs = enc.get_intermediate_layers(x, 4)   # x: lô ảnh đúng độ phân giải gốc, chuẩn hoá ImageNet (mặc định) -> 4 x (patch, vector toàn ảnh)
"""
from .spec import ORDER, SPECS, EncoderSpec          # noqa: F401
from ..checks import weights as _weights

BUILT = ('raddino', 'biomedclip')                     # đã có lớp bọc + kiểm tệp thật; các encoder còn lại: bước sau (PLAN_89 §10)


def build(name, check=True, **kw):
    '''-> lớp bọc ở eval() (CPU). check=True: cổng nạp trọng số trượt -> AssertionError; check=False: trả kèm nap['dat'] = False để bên gọi ghi lại rồi
       tự dừng. kw chuyển cho hàm build của encoder (vd input_mean / input_std = chuẩn hoá bên gọi đã áp; raddino: attn='eager').'''
    assert name in SPECS, f'encoder {name!r} khong thuoc {ORDER} -> DUNG'
    if name not in BUILT:
        raise NotImplementedError(f'{name}: lop boc chua dung (spec.status = {SPECS[name].status})')
    import importlib
    enc = importlib.import_module(f'{__name__}.{name}').build(**kw)
    assert enc.nap is not None and enc.spec.name == name
    enc.nap['encoder'] = name
    if check:
        assert enc.nap['dat'], f'{name}: nap trong so KHONG dat: {_weights.loading_text(enc.nap)} -> DUNG'
    return enc
