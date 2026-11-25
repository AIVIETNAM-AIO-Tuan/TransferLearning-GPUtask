# -*- coding: utf-8 -*-
"""Một lượt `89` Ver2 trên một bộ ngoài: pipeline.prepare / run của vindr_bench @ c81af62 NGUYÊN VĂN, chỉ đổi thứ bộ dữ liệu buộc đổi (PLAN_93 §4.2):

- MainRun(epochs_phase1 = epochs_phase2 = DatasetSpec.epochs): RSNA 20 (63 / 84 bước mỗi epoch), Shenzhen 100 (2 / 3 bước -> 200 / 300 bước, = script
  Shenzhen của CheXFound); n_seeds = 5, lệch hạt pha 2 = 500.000 như 89. Hạt đầu = công thức của 89 (head_seed + 1e6 x seed + 1e4 x E1 + 1e3 x ô loại + lr):
  với E1 = 100 số hạng 1e4 x E1 = 1e6 -> hạt của seed s trùng hạt seed s + 1 của một lượt E1 = 0 khác — không có lượt nào như vậy trong một notebook; trong lượt,
  hạt các đầu vẫn khác nhau (pipeline tự kiểm ở bước dò lr).
- Đầu đọc ra [B, C] kể cả C = 1 (protocol/heads.keep_class_dim).
- Dòng in: các chữ cố định cho VinDr ("28 nhan", "AUROC 28", "AP 28", "AUROC 21") đổi theo số lớp của bộ (chỉ chữ in, không đổi số)."""
import contextlib

from .. import config
from . import heads as H


def main_run(name, **kw):
    from vindr_bench.protocols.ours import config as C
    e = config.dataset(name).epochs
    return C.MainRun(epochs_phase1=e, epochs_phase2=e, **kw)


def relabel(log, n_cls):
    '''Bọc hàm in: thay nhãn chữ của VinDr (28 nhãn / 21 lớp) bằng số lớp thật của bộ.'''
    lop = 'lop chinh' if n_cls > 1 else 'nhan'
    pairs = (('28 nhan', f'{n_cls} nhan'), ('AUROC 28', 'AUROC' if n_cls == 1 else f'AUROC {n_cls}'), ('AP 28', 'AP' if n_cls == 1 else f'AP {n_cls}'),
             ('AUROC 21', f'AUROC {lop}'))

    def f(s=''):
        for a, b in pairs:
            s = s.replace(a, b)
        return log(s)
    return f


@contextlib.contextmanager
def single_label_ok():
    with H.keep_class_dim():
        yield


def prepare(data, backbone, name, main=None, subset=None, protocol=None):
    '''protocol (0.5.0): None = Protocol() mặc định của 89 (lượt thật); bộ kiểm máy truyền lưới lr nhỏ cho nhanh.'''
    from vindr_bench.protocols.ours import pipeline as PL
    assert data.dataset == name and data.store == config.STORE_KEY, (data.dataset, data.store)
    return PL.prepare(data, backbone, protocol=protocol, main=main or main_run(name), subset=subset)


def run(plan, state=None, save=None, run_tag='', log=print):
    from vindr_bench.protocols.ours import pipeline as PL
    with single_label_ok():
        return PL.run(plan, state=state, save=save, run_tag=run_tag, log=relabel(log, plan.n_cls))


def roundtrip_check(plan, state, arrays):
    from vindr_bench.protocols.ours import pipeline as PL
    with single_label_ok():
        return PL.roundtrip_check(plan, state, arrays)
