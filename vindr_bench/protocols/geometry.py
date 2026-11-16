# -*- coding: utf-8 -*-
"""Phép thử hình học của Ver2 — đệm vuông viền đen (kho của 85) so với co cạnh ngắn + cắt giữa (PLAN_89 §5.3, §7.4). CHỈ hằng số: một nguồn cho
notebook.ver2 (hàng nào chạy kho cắt giữa), tools/build_notebooks.py (dựng notebook nào), evaluation.collect (lượt nào phải có) và notebook.summary
(thành phần hai họ phép so). Không import gì.

    họ "hình học"          (Tuấn 13/11, khoá trước mọi kết quả): B, RAD-DINO, CheXFound — 28 phép = (a) Δ_E 12 + (b) Δ_B − Δ_X 8 + (c) B − X trên kho cắt
                           giữa 8
    họ "hình học mở rộng"  (Tuấn 16/11, sau khi Claude đọc mã tác giả: 5 encoder này chấm bằng co cạnh ngắn + cắt giữa; đặt TRƯỚC mọi kết quả cắt giữa,
                           SAU khi đã thấy kết quả đệm vuông): BiomedCLIP, MRM, EVA-X, PubMedCLIP, CheXWorld — 60 phép, cùng ngưỡng, max-T RIÊNG
    CheXzero không thêm: tiền xử lý gốc của nó (data_process.preprocess @ 5c341db) = đệm vuông viền đen, như kho của ta.

Δ_E = mAP21(kho cắt giữa) − mAP21(kho đệm vuông), cùng encoder, cùng đầu, cùng giao thức Ver2 (hai lượt học riêng, mỗi bên TB 5 hạt)."""

PAD_STORE = 'kcb'                    # kho đệm vuông KHÔNG cân bằng (85b / 85e _khong_cb) — so cùng kiểu histogram với kho cắt giữa
CROP_STORE = 'cg_kcb'                # kho cắt giữa KHÔNG cân bằng (89k) — lượt của phép thử
CROP_STORE_SPARE = 'cg_cb'           # kho cắt giữa CÓ cân bằng: ghi cùng lượt giải mã, dự phòng (PLAN_89 §5.3: "nếu B tệ đi nhiều" -> Tuấn quyết)
REFERENCE = 'b'
ROWS_PLAN = ('b', 'raddino', 'chexfound')                                  # họ "hình học" (28 phép)
ROWS_EXT = ('biomedclip', 'mrm', 'evax', 'pubmedclip', 'chexworld')        # họ "hình học mở rộng" (60 phép)
ROWS = ROWS_PLAN + ROWS_EXT
NOT_ROWS = ('chexzero',)                                                   # tiền xử lý gốc = đệm vuông
MARGIN = 0.01                        # ngưỡng của (a) "tệ đi nhiều" / "không tệ đi" và của (c) (= luật họ 32)

FAMILIES = {
    'hinh_hoc': dict(rows=ROWS_PLAN, title='ho "hinh hoc" (Tuan 13/11, khoa truoc moi ket qua): B, RAD-DINO, CheXFound'),
    'hinh_hoc_mo_rong': dict(rows=ROWS_EXT, title='ho "hinh hoc mo rong" (Tuan 16/11, dat truoc ket qua cat giua, sau khi thay ket qua dem vuong): '
                                                  'BiomedCLIP, MRM, EVA-X, PubMedCLIP, CheXWorld'),
}

assert REFERENCE in ROWS_PLAN and not set(ROWS) & set(NOT_ROWS) and len(set(ROWS)) == len(ROWS) == 8
