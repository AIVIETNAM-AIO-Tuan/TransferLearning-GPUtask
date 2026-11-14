# -*- coding: utf-8 -*-
"""Nạp checkpoint AN TOÀN ở cả torch 2.0 (môi trường CheXFound) lẫn torch mới — không bao giờ chạy pickle tuỳ ý.

Một số checkpoint công khai có `argparse.Namespace` trong pickle (MRM: 'args'; CheXWorld) -> `torch.load(weights_only=True)` từ chối. torch >= 2.4 có
`serialization.safe_globals`; torch 2.0 thì không. Cách làm (chép ý của module 88, external_benchmarks.backbones.load_checkpoint):
  1. liệt kê GLOBAL trong data.pkl của tệp zip bằng pickletools (KHÔNG chạy pickle); GLOBAL ngoài danh sách cho phép -> dừng;
  2. torch mới: weights_only=True + safe_globals([argparse.Namespace]); torch 2.0: torch.load với pickle_module có Unpickler.find_class CHỈ trả
     các GLOBAL cho phép (kiểm lại ngay lúc nạp, không chỉ lúc quét).
md5 / kích thước kiểm trước khi nạp nếu được cho. Chỉ cú pháp / API có ở Python 3.9 + torch 2.0."""
import argparse
import hashlib
import io
import os
import pickle
import pickletools
import types
import zipfile

import torch

CORE = {'collections.OrderedDict', 'torch._utils._rebuild_tensor_v2'} | {
    f'torch.{t}Storage' for t in ('Float', 'Half', 'BFloat16', 'Double', 'Long', 'Int', 'Short', 'Char', 'Byte', 'Bool')}
# + cách torch dựng lại tensor / tham số (trong danh sách mặc định của weights_only; checkpoint CheXFound có hai mục cuối)
BASE_ALLOWED = CORE | {'torch._utils._rebuild_parameter', 'torch.Tensor', 'torch._tensor._rebuild_from_type_v2'}
NAMESPACE = 'argparse.Namespace'


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def pickle_globals(path):
    '''GLOBAL trong data.pkl của checkpoint torch dạng zip (KHÔNG chạy pickle). Không phải zip -> None (bên gọi quyết).'''
    if not zipfile.is_zipfile(path):
        return None
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith('data.pkl')]
        assert len(names) == 1, f'{os.path.basename(path)}: {len(names)} data.pkl -> DUNG'
        data = z.read(names[0])
    found, strs = set(), []
    for op, arg, _ in pickletools.genops(io.BytesIO(data)):
        if op.name == 'GLOBAL':
            found.add(arg.replace(' ', '.'))
        elif op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE'):
            strs.append(arg)
        elif op.name == 'STACK_GLOBAL':
            found.add(f'{strs[-2]}.{strs[-1]}' if len(strs) >= 2 else '?')
    return sorted(found)


def _restricted_pickle(allowed):
    '''Mô-đun pickle thay thế cho torch.load ở torch < 2.4: Unpickler.find_class chỉ trả GLOBAL thuộc `allowed`.'''
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            full = f'{module}.{name}'
            if full not in allowed:
                raise pickle.UnpicklingError(f'GLOBAL {full} khong duoc phep -> DUNG')
            return super().find_class(module, name)
    mod = types.ModuleType('vb_restricted_pickle')
    mod.Unpickler, mod.load, mod.loads, mod.__name__ = Unpickler, pickle.load, pickle.loads, 'vb_restricted_pickle'
    return mod


# GLOBAL thêm mà từng checkpoint công khai cần (đã liệt kê trên tệp thật 14/11) — chỉ là lớp dữ liệu, không lớp mô hình / hàm tuỳ ý:
EXTRA = {
    'namespace': {NAMESPACE},                                                     # MRM ('args'), CheXWorld
    'numpy_scalar': {'__builtin__.set', 'builtins.set', '_codecs.encode', 'numpy.core.multiarray.scalar', 'numpy.dtype'},   # EVA-X (số vô hướng numpy)
}


def load_checkpoint(path, expect_md5=None, expect_size=None, allow=()):
    '''-> (đối tượng đã nạp, dict nguồn: tep, byte, md5?, global_pickle, cach_nap). allow: tên nhóm trong EXTRA mà tệp này được dùng.
       GLOBAL chỉ trong BASE_ALLOWED -> weights_only=True; có GLOBAL thêm (đã cho phép) -> Unpickler hạn chế (cùng một đường ở mọi bản torch).'''
    src = dict(tep=os.path.basename(path), byte=os.path.getsize(path))
    if expect_size is not None:
        assert src['byte'] == expect_size, f"{src['tep']}: {src['byte']} B != {expect_size} B -> DUNG"
    if expect_md5 is not None:
        src['md5'] = md5(path)
        assert src['md5'] == expect_md5, f"{src['tep']}: md5 {src['md5']} != {expect_md5} -> DUNG"
    allowed = set(BASE_ALLOWED)
    for a in allow:
        allowed |= EXTRA[a]
    g = pickle_globals(path)
    assert g is not None, f"{src['tep']}: khong phai checkpoint torch dang zip -> DUNG (khong nap)"
    bad = sorted(set(g) - allowed)
    assert not bad, f"{src['tep']}: GLOBAL la trong pickle {bad} -> DUNG (khong nap)"
    src['global_pickle'] = g
    if set(g) <= CORE:
        src['cach_nap'] = 'weights_only=True'
        return torch.load(path, map_location='cpu', weights_only=True), src
    src['cach_nap'] = 'Unpickler han che (chi GLOBAL cho phep: ' + ', '.join(sorted(set(g) - CORE)) + ')'
    return torch.load(path, map_location='cpu', weights_only=False, pickle_module=_restricted_pickle(allowed)), src
