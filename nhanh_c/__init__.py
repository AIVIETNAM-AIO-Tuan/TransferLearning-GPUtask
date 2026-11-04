# -*- coding: utf-8 -*-
"""nhanh_c - nhanh C chay theo seed (Tuan duyet 29/10; docs/SPEC_SAU_78Y.md B.2ab; PLAN A.4v).

   Nhanh C = luot goi o cat s42 (78x_jca_v3b_dp_ema_v2_o_s42): Phan 1 B.4 v2 + 5 o luoi phong 518 (khong gradient) + adapter + skip block 7 -> 8
   + dau J doc 6 view luc train; Phan 2 78z 4 dau chuan (doc 1 view, duong skip dong cung); Phan 3 gop trong so ep1-10 (dau quyet dinh
   fus_attn_hr) + tong hop cac seed.

   Moi buoc (buoc/<ten>.py) = mot o notebook DA KIEM, nguyen van + cac dong va 'nhanh C' theo seed; sinh boi experiments/84_nhanh_c/dung_84.py
   (KHONG sua tay - sha256 kiem o buoc()). Notebook goi, moi o mot buoc:

       import nhanh_c as NC; NC.buoc('p1_C0', globals(), seed=43)

   buoc() chay tep trong namespace notebook (globals()) -> nhu chay o: bien xem duoc sau o, %reset cua o chuyen phan xoa dung namespace do.

   Bien the Cg (Tuan 02/11; docs/SPEC_SAU_78Y.md B.2ae): gradient qua embeddings + block 0-7 cua 5 o CHI qua skip; notebook goi
   NC.buoc('g1_C0', globals(), seed=43, bien_the='Cg'). Moi bien the CHI chay buoc trong thu tu cua no (thu_tu.PHAN / PHAN_CG / PHAN_B / PHAN_BG).

   Bien the B (Tuan 02/11; docs/SPEC_SAU_78Y.md B.2ah): cong thuc skip7 s42 (dau J 1 view luc train) theo seed, ghep cap voi C cung seed; notebook
   goi NC.buoc('b1_C0', globals(), seed=43, bien_the='B').

   Bien the Bg (Tuan duyet 04/11; docs/SPEC_SAU_78Y.md B.2ai): cong thuc B + gradient qua embeddings + block 0-7 cua 5 o CHI qua skip (nhu Cg),
   ghep cap voi B cung seed; notebook goi NC.buoc('bg1_C0', globals(), seed=44, bien_the='Bg')."""
import hashlib
import os

from .cau_hinh import BIEN_THE, PHIEN_BAN, SEEDS, cau_hinh
from .thu_tu import BAM, PHAN, PHAN_B, PHAN_BG, PHAN_CG

_GOC = os.path.dirname(os.path.abspath(__file__))
_PHAN_BT = {'C': PHAN, 'Cg': PHAN_CG, 'B': PHAN_B, 'Bg': PHAN_BG}
assert tuple(_PHAN_BT) == BIEN_THE


def ds_buoc(phan=None, bien_the='C'):
    '''Ten buoc theo thu tu notebook cua bien the: phan 1 / 2 / 3, hoac tat ca (phan=None; ke ca o chuyen phan).'''
    if bien_the not in _PHAN_BT:
        raise ValueError(f'nhanh_c: bien the {bien_the!r} khong hop le (can mot trong {BIEN_THE})')
    P = _PHAN_BT[bien_the]
    if phan is None:
        return [t for k in sorted(P) for t in P[k]]
    return list(P[phan])


def buoc(ten, ns, seed, thu=False, bien_the='C'):
    '''Chay buoc `ten` trong namespace `ns` (globals() cua notebook) cho `seed` (+ bien the). Dat ns['CAU_HINH_C'] = cau_hinh(seed, thu,
       bien_the); namespace dang mang cau hinh KHAC (seed / thu / bien the) -> dung (khong tron hai luot trong mot phien). Buoc KHONG thuoc
       thu tu cua bien the -> KeyError (vd p1_C6 cua C trong notebook Cg: thieu luot gradient thu hai).'''
    if ten not in BAM:
        raise KeyError(f'nhanh_c: khong co buoc {ten!r} (co {len(BAM)} buoc: {", ".join(sorted(BAM))})')
    if not isinstance(ns, dict):
        raise TypeError('nhanh_c: ns phai la dict globals() cua notebook')
    if ten not in ds_buoc(bien_the=bien_the):
        raise KeyError(f'nhanh_c: buoc {ten!r} KHONG thuoc bien the {bien_the!r} (thu tu: {", ".join(ds_buoc(bien_the=bien_the))})')
    cfg = cau_hinh(seed, thu, bien_the)
    cu = ns.get('CAU_HINH_C')
    if cu is not None and cu != cfg:
        raise ValueError(f"nhanh_c: namespace dang mang luot seed {cu.get('seed')} thu={cu.get('thu')} bien the {cu.get('bien_the', 'C')} - "
                         f'KHAC seed {seed} thu={thu} bien the {bien_the}. Restart session roi chay lai tu dau notebook.')
    p = os.path.join(_GOC, 'buoc', ten + '.py')
    with open(p, 'rb') as f:
        b = f.read().replace(b'\r\n', b'\n')     # git autocrlf (Windows) doi LF -> CRLF luc checkout: so + chay tren noi dung LF
    if hashlib.sha256(b).hexdigest() != BAM[ten]:
        raise RuntimeError(f'nhanh_c: {p} KHAC ban sinh (sha256) -> tai lai goi dung commit ghim')
    ns['CAU_HINH_C'] = cfg
    exec(compile(b.decode('utf-8'), p, 'exec'), ns)
