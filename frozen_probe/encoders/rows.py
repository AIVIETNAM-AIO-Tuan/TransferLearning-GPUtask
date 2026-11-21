# -*- coding: utf-8 -*-
"""Sáu hàng của phép kiểm ít nhãn VinDr cho ba cặp mới (PLAN_92 §13; Tuấn 21/11 "được dựng đi"): backbone 78x CÓ / KHÔNG luồng vùng của DINOv3,
CheXFound, BiomedCLIP — đúng tệp `rs_` (trọng số EMA) mà 78r_dac_trung_bb_* đã đọc (20–21/11).

Nguồn của từng số (tests/test_bb.py đối chiếu):
  run / epoch / số epoch / hệ số ghi ngược / khoá cfg_p92  = experiments/78_chi_anh/cau_hinh/cap_bb_p92.json (qua dung_78r_bb.CFG_R)
  md5 rs_ + backbone_ft.pt, số lần cập nhật EMA, "lech pretrain" = output C0 / C2 / C2b của notebook 78r đã chạy (cùng tệp, cùng phép tính)
  cấu hình encoder tiền huấn luyện (PRE)                    = experiments/78_chi_anh/dung_78x_bb.BB (bộ nạp của lượt học)

md5 ghim ở đây là md5 của CHECKPOINT của ta (không phải mã ảnh) — tools/publish.py chỉ cho phép chuỗi 32 hex trong tệp này và chỉ khi nó là một
md5 của PINS bên dưới. KHÔNG chứa ID Drive."""
from dataclasses import dataclass, field
from typing import Optional, Tuple

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
RAD_DINO = 0.5307, 0.2583                 # chuẩn hoá ảnh vào của lượt học (proc.image_mean[0] / image_std[0] của microsoft/rad-dino; = vindr_bench SPECS['raddino'])
SIZE = 518                                # đường ảnh 518 của lượt học; mặt nạ encoder tự thu về S = 512
CFG_FILE = 'backbone_ft.pt'
LUOT = 'B1'                               # khoá 'luot' của mọi rs_ 78x
LUOT_J = 'J_bert_dong'                    # luot_79a / luot_chon của mọi lượt J
CROP_KEYS = ('bo_o7', 'dua_o', 'bo_o11', 'gop_o')

# Encoder tiền huấn luyện (= dung_78x_bb.BB của lượt học; bộ kiểm so từng khoá dùng ở đây)
PRE = {
    'dinov3': dict(ten='dinov3', nguon='hf', hf='facebook/dinov3-vitb16-pretrain-lvd1689m', gated=True, S=512, tang=(8, 9, 10, 11)),
    'chexfound': dict(ten='chexfound', nguon='drive', S=512, tang=(20, 21, 22, 23), mean=IMAGENET[0], std=IMAGENET[1]),
    'biomedclip': dict(ten='biomedclip', nguon='hf', hf='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', S=512, tang=(8, 9, 10, 11),
                       mean=CLIP[0], std=CLIP[1]),
}
ENCODERS = tuple(PRE)
TITLE = dict(dinov3='DINOv3', chexfound='CheXFound', biomedclip='BiomedCLIP')
DIM = dict(dinov3=768, chexfound=1024, biomedclip=768)


@dataclass(frozen=True)
class BBSpec:
    '''Một hàng: giao diện spec mà vindr_bench (lớp bọc, files.fetch, InputSpec) đọc + các số cổng của hàng.'''
    name: str
    encoder: str
    hang: str                              # 'co' (CÓ luồng vùng) | 'khong'
    run: str                               # lượt 78x
    drive_file: str                        # rs_<luot>_ep<E>.pt (trọng số EMA)
    file_md5: str
    ep_bb: int                             # epoch đọc (khoá 'ep' của rs_ = ep_bb - 1)
    epochs: int                            # số epoch của lịch
    ema_n: int                             # số lần cập nhật EMA ghi trong rs_ (78r đã in)
    lech_pre: str                          # max |trọng số rs_ − tiền huấn luyện| trên mọi khoá mặt nạ, 3 chữ số (78r in "lech pretrain")
    cfg_md5: Optional[str] = None          # backbone_ft.pt (CHỈ hàng CÓ: cấu hình luồng vùng)
    g: Optional[float] = None              # hệ số ghi ngược (hàng CÓ)
    rho: Optional[float] = None            # trần ghi ngược (hàng CÓ)
    khoa_p92: Optional[str] = None         # khoá cfg_p92 của backbone_ft.pt (hàng CÓ mới của PLAN_92 §12; None = lượt tháng 10, không có cfg_p92)
    image_size: int = SIZE
    mean: Tuple[float, float, float] = (RAD_DINO[0],) * 3
    std: Tuple[float, float, float] = (RAD_DINO[1],) * 3
    pool: str = 'cls'
    status: str = 'da_dung'
    embed_dim: int = 0                     # chiều token của encoder (đặt ở ROWS = DIM[encoder]; bộ kiểm thay bằng chiều mô hình nhỏ)
    extra_files: Tuple = field(default=())

    @property
    def title(self):
        return f"{TITLE[self.encoder]} {'CO luong vung' + (f' (ghi nguoc x{self.g:g})' if self.g else '') if self.hang == 'co' else 'KHONG luong vung'}"

    @property
    def patch_size(self):
        return 16

    @property
    def tang(self):
        return tuple(PRE[self.encoder]['tang'])

    @property
    def n_patches(self):
        return (int(PRE[self.encoder]['S']) // self.patch_size) ** 2           # 32 x 32 = 1024 (mặt nạ thu 518 -> 512)

    @property
    def drive_dir(self):
        return f'cxrlt_2024/stage4_nb78x/{self.run}'

    @property
    def drive_files(self):
        return [(self.drive_file, self.file_md5, None)] + ([(CFG_FILE, self.cfg_md5, None)] if self.cfg_md5 else [])


ROWS = {s.name: s for s in (
    BBSpec('bb_dinov3_co', 'dinov3', 'co', 'r78x_J_v3b_dp_ema_dinov3_e10_s42', 'rs_J_bert_dong_ep10.pt', 'edb655347fbded2375457ee682169e3f', 10, 10, 8980,
           '0.062', cfg_md5='6a055a665999440f26d48d3623c1e645', g=5.0, rho=0.1, khoa_p92=None, embed_dim=DIM['dinov3']),
    BBSpec('bb_dinov3_khong', 'dinov3', 'khong', 'r78x_B1_dp_ema_dinov3_e10_s42', 'rs_B1_ep10.pt', 'de7f977aa66fe830b88cc3ad4890f20b', 10, 10, 8980, '0.087',
           embed_dim=DIM['dinov3']),
    BBSpec('bb_chexfound_co_x1', 'chexfound', 'co', 'r78x_J_v3b_dp_ema_chexfound_x1_s42', 'rs_J_bert_dong_ep4.pt', 'b32ec8d49cf69610442f7ec24d28d680', 4, 6,
           3592, '0.073', cfg_md5='49799d6d8842d3cd1d50e476bf1272b9', g=1.0, rho=0.1, khoa_p92='chexfound_x1', embed_dim=DIM['chexfound']),
    BBSpec('bb_chexfound_khong', 'chexfound', 'khong', 'r78x_B1_dp_ema_chexfound_s42', 'rs_B1_ep4.pt', '82c3d330793fcf80a107d17c4293399e', 4, 6, 3592, '0.068',
           embed_dim=DIM['chexfound']),
    BBSpec('bb_biomedclip_co', 'biomedclip', 'co', 'r78x_J_v3b_dp_ema_biomedclip_v2_s42', 'rs_J_bert_dong_ep4.pt', '45318441e56bb07c6c5d310d72e665b4', 4, 6,
           3592, '0.042', cfg_md5='8ae12f5d3c7c78a262bd0e99bcc03180', g=5.0, rho=0.1, khoa_p92='biomedclip', embed_dim=DIM['biomedclip']),
    BBSpec('bb_biomedclip_khong', 'biomedclip', 'khong', 'r78x_B1_dp_ema_biomedclip_s42', 'rs_B1_ep4.pt', 'dbedd1eed9bc6d5c54a517772a6f7400', 4, 6, 3592,
           '0.037', embed_dim=DIM['biomedclip']),
)}
PAIRS = {e: (next(n for n, s in ROWS.items() if s.encoder == e and s.hang == 'co'), next(n for n, s in ROWS.items() if s.encoder == e and s.hang == 'khong'))
         for e in ENCODERS}
PINS = frozenset(m for s in ROWS.values() for m in (s.file_md5, s.cfg_md5) if m)
assert len(ROWS) == 6 and all(len(ROWS[a].tang) == 4 and ROWS[a].ep_bb == ROWS[b].ep_bb and ROWS[a].epochs == ROWS[b].epochs for a, b in PAIRS.values())
assert all((s.hang == 'co') == bool(s.cfg_md5 and s.g and s.rho) for s in ROWS.values()) and len(PINS) == 9
assert all(s.embed_dim == DIM[s.encoder] for s in ROWS.values())
