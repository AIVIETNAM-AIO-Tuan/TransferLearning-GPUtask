"""Lớp bọc đông cứng của sáu hàng PLAN_92 §13 (backbone 78x CÓ / KHÔNG luồng vùng trên DINOv3, CheXFound, BiomedCLIP).

    rows.py     bảng sáu hàng (lượt 78x, tệp + md5 ghim, epoch đọc, số cổng) + cấu hình encoder tiền huấn luyện
    bb_src.py   mặt nạ encoder + lớp luồng vùng P token đầu — CHÉP NGUYÊN VĂN experiments/78_chi_anh/khoi/khoi_bb_khac.py (phần A + B)
    bb.py       lớp bọc (vindr_bench.encoders.base.EncoderWrapper) + cổng nạp / mặt nạ / forward
    vt_s43.py   (0.3.0) ba hàng SEED 43 (3) / (1b) / (0) của RAD-DINO: lớp bọc vindr_bench.encoders.vt nguyên vẹn, tệp s43, cổng seed 43

    from frozen_probe import encoders as E
    enc = E.build('bb_chexfound_co_x1', chexfound_path=..., token=...)   # eval(), CPU; bên gọi chuyển thiết bị"""
from .rows import ENCODERS, PAIRS, ROWS          # noqa: F401


def build(name, **kw):
    from . import bb
    return bb.build(name, **kw)
