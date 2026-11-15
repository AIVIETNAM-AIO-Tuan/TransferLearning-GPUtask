"""Ver2 — mã + môi trường của ta trên G4 (PLAN_89 §5): giao thức `86b` (PLAN_86 §4.4) cho 4 loại đầu trên lớp bọc chung.

    config.py     hằng số giao thức (lưới 13 lr, 20 + 20 epoch, lô 256, 5 hạt, hạt RNG tách dữ liệu / đầu) + hạt từng đầu
    backbone.py   lớp bọc encoder đông cứng dưới autocast (bf16) -> (4 tầng cuối, patch tầng cuối)
    heads.py      tạo / chạy Linear-1, Linear-4, ML-Decoder, GLoRI
    training.py   một đầu đọc đang học (AdamW + cosine theo bước), bước học, dự đoán, chấm val — chép từ vindr_protocol/training.py của 86
    selection.py  chọn lr theo val (AUROC = luật; AP = nhánh mô tả) + chẩn đoán đường học
    pipeline.py   vòng học song song nhiều đầu trên cùng đặc trưng + ba giai đoạn (dò lr -> thêm hạt -> pha 2 + dự đoán test), xuất / nạp lại giữa giai đoạn
    gates.py      cổng trong notebook: thư viện ghim, GPU, số học bf16 so với fp32, bộ nhớ

Không import gì ở đây (import theo module khi cần); chỉ chạy ở Python của notebook, không chạy trong môi trường thư viện của CheXFound."""
