# -*- coding: utf-8 -*-
"""Mã forward của mô-đun vùng đặt SAU encoder, chỉ đọc (hàng (1a) / (1b) của PLAN_90 Việc B) — CHỈ phần cần cho cổng "trùng forward đầy đủ" của
lớp bọc encoders/vt.py.

Hai định nghĩa dưới đây CHÉP NGUYÊN VĂN (không sửa chữ nào) phần dưới dấu `# ==== KHOI ====` của experiments/78_chi_anh/khoi/khoi_jca_sau_enc.py —
tệp mà notebook 78x / 78z / 78r của ba lượt vị trí nhúng vào ô. tests/test_vt.py so cây cú pháp từng định nghĩa với tệp nguồn. Tên cần có sẵn
(_ma_hoa_hop, _tuc_luong, LuongVungNeo) lấy từ b_src.py (cũng chép nguyên văn khối 79 / khoi_jca_v2). Chỉ dùng cú pháp / API có ở Python 3.9 + torch 2.0."""
import torch

from .b_src import LuongVungNeo, _ma_hoa_hop, _tuc_luong    # noqa: F401  (tên mà định nghĩa chép nguyên văn đọc lúc chạy)


# ---- experiments/78_chi_anh/khoi/khoi_jca_sau_enc.py (NGUYÊN VĂN): LuongVungSauEnc, ap_sau_enc ----
class LuongVungSauEnc(LuongVungNeo):
    '''Luong vung SAU encoder, chi doc (xem dau tep). buoc: block khac block cuoi -> tra h nguyen; block cuoi -> n_lop_dung lop JCA noi
       tiep cung doc h (mo_ghi False), tra DUNG h da nhan. dau_ra: hop_cac_tang co n_lop_dung phan tu, ghi = [] (khong write-back).'''
    n_lop_dung = None

    def buoc(s, l, h, st, mo_ghi):
        if int(l) != s.tang[-1]: return h
        assert not s.hai_chieu and s.n_lop_dung is not None, 'LuongVungSauEnc: phai mot chieu + da goi ap_sau_enc -> DUNG'
        for lop in list(s.lop)[:s.n_lop_dung]:
            p = s.pos(_ma_hoa_hop(st['hop'].detach(), s.n_pos))
            if s.ckpt and s.training and torch.is_grad_enabled():
                _h, r, hop, ti = _tuc_luong.checkpoint(lop, h, st['r'], st['hop'], p, False, st['neo'], use_reentrant=False)
            else:
                _h, r, hop, ti = lop(h, st['r'], st['hop'], p, False, st['neo'])
            assert ti is None, 'LuongVungSauEnc: lop JCA tra ti le ghi nguoc khi mo_ghi False -> DUNG'
            st['r'], st['hop'] = r, hop
            st['hop_cac_tang'].append(hop)
        return h


def ap_sau_enc(luong, n_lop):
    '''luong = LuongVungNeo MOT CHIEU (hai_chieu False; sau tang_ghi_nguoc + ap_tran_ghi cung duoc - hai phep do khong doi lop cua LUONG)
       -> LuongVungSauEnc chay n_lop lop dau sau block cuoi. -> (so lop chay, so tham so cua cac lop chay + phan dung chung).'''
    assert type(luong) is LuongVungNeo, f'ap_sau_enc: {type(luong).__name__} khong phai LuongVungNeo -> DUNG'
    assert luong.hai_chieu is False, 'ap_sau_enc: luong phai MOT CHIEU (hai_chieu False) -> DUNG'
    n = int(n_lop)
    assert 1 <= n <= len(luong.lop), f'ap_sau_enc: n_lop {n_lop} ngoai 1..{len(luong.lop)} -> DUNG'
    luong.__class__ = LuongVungSauEnc
    luong.n_lop_dung = n
    bo = {id(p) for lop in list(luong.lop)[n:] for p in lop.parameters()}
    return n, sum(p.numel() for p in luong.parameters() if id(p) not in bo)
