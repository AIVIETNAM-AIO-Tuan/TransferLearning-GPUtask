# -*- coding: utf-8 -*-
"""nhanh_c - nhanh C chay theo seed (Tuan duyet 29/10; docs/SPEC_SAU_78Y.md B.2ab; PLAN A.4v).

   Nhanh C = luot goi o cat s42 (78x_jca_v3b_dp_ema_v2_o_s42): Phan 1 B.4 v2 + 5 o luoi phong 518 (khong gradient) + adapter + skip block 7 -> 8
   + dau J doc 6 view luc train; Phan 2 78z 4 dau chuan (doc 1 view, duong skip dong cung); Phan 3 gop trong so ep1-10 (dau quyet dinh
   fus_attn_hr) + tong hop cac seed.

   Moi buoc (buoc/<ten>.py) = mot o notebook DA KIEM, nguyen van + cac dong va 'nhanh C' theo seed; sinh boi experiments/84_nhanh_c/dung_84.py
   (KHONG sua tay - sha256 kiem o buoc()). Notebook goi, moi o mot buoc:

       import nhanh_c as NC; NC.buoc('p1_C0', globals(), seed=43)

   buoc() chay tep trong namespace notebook (globals()) -> nhu chay o: bien xem duoc sau o, %reset cua o chuyen phan xoa dung namespace do."""
import hashlib
import os

from .cau_hinh import PHIEN_BAN, SEEDS, cau_hinh
from .thu_tu import BAM, PHAN

_GOC = os.path.dirname(os.path.abspath(__file__))


def ds_buoc(phan=None):
    '''Ten buoc theo thu tu notebook: phan 1 / 2 / 3, hoac tat ca (phan=None; ke ca o chuyen phan).'''
    if phan is None:
        return [t for k in sorted(PHAN) for t in PHAN[k]]
    return list(PHAN[phan])


def buoc(ten, ns, seed, thu=False):
    '''Chay buoc `ten` trong namespace `ns` (globals() cua notebook) cho `seed`. Dat ns['CAU_HINH_C'] = cau_hinh(seed, thu); namespace dang
       mang cau hinh KHAC (seed / thu) -> dung (khong tron hai luot trong mot phien).'''
    if ten not in BAM:
        raise KeyError(f'nhanh_c: khong co buoc {ten!r} (co {len(BAM)} buoc: {", ".join(sorted(BAM))})')
    if not isinstance(ns, dict):
        raise TypeError('nhanh_c: ns phai la dict globals() cua notebook')
    cfg = cau_hinh(seed, thu)
    cu = ns.get('CAU_HINH_C')
    if cu is not None and cu != cfg:
        raise ValueError(f"nhanh_c: namespace dang mang luot seed {cu.get('seed')} thu={cu.get('thu')} - KHAC seed {seed} thu={thu}. "
                         'Restart session roi chay lai tu dau notebook.')
    p = os.path.join(_GOC, 'buoc', ten + '.py')
    with open(p, 'rb') as f:
        b = f.read()
    if hashlib.sha256(b).hexdigest() != BAM[ten]:
        raise RuntimeError(f'nhanh_c: {p} KHAC ban sinh (sha256) -> tai lai goi dung commit ghim')
    ns['CAU_HINH_C'] = cfg
    exec(compile(b.decode('utf-8'), p, 'exec'), ns)
