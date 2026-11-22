"""frozen_probe — phép kiểm ít nhãn trên đặc trưng ĐÔNG CỨNG (module 92; docs/PLAN_92_TONG_QUAT_LUONG_VUNG.md §11 — Tuấn duyệt 19/11).

Câu hỏi: backbone học CÓ luồng vùng (hàng (3) = B.4 v2) có cho đặc trưng đông cứng tách nhãn VinDr tốt hơn hàng KHÔNG luồng vùng (hàng (0)) khi chỉ có
1 % / 10 % / 100 % nhãn train, với đầu đọc KHÔNG chọn siêu tham số theo dữ liệu chấm (k-NN; logistic nghiệm lồi) không?

    config.py      hằng số: hàng, tỉ lệ nhãn, số lần rút, hạt, đầu đọc, luật đặt trước (khoá khi dựng)
    subsets.py     lần rút chỉ số theo tỉ lệ nhãn (chung cho mọi hàng) + md5
    probes.py      k-NN (cosine, TB nhãn k láng giềng — như 78r) + logistic L2 một-đấu-phần-còn-lại giải tới hội tụ (L-BFGS, không lr)
    features.py    trích đặc trưng đông cứng 18.000 ảnh VinDr qua lớp bọc của vindr_bench (GPU) + tệp đặc trưng
    scoring.py     chấm (vindr_bench.evaluation: AP bậc thang / AUPRC hình thang / AUROC, bootstrap theo ảnh), ô = TB các lần rút, họ 6 phép + max-T,
                   hiệu mô tả, bảng
    notebook/      extract.py (ô của 92a, G4) · probe.py (ô của 92b, CPU) · extract_bb.py (ô của 92d, G4) · probe_bb.py (ô của 92e, CPU)
    config_bb.py   (0.2.0, PLAN_92 §13) hằng của phép kiểm cho ba cặp mới (DINOv3, CheXFound x1, BiomedCLIP); giao thức = config.py
    encoders/      (0.2.0) lớp bọc đông cứng 6 hàng CÓ / KHÔNG luồng vùng (rows.py · bb_src.py chép nguyên văn khoi_bb_khac · bb.py)
    config_s43.py  (0.3.0, PLAN_90 §13.8) hằng của phép kiểm SEED 43 cho (3), (1b), (0): ba bảng đọc đặt trước (luật §11.3; xác nhận seed 43; khối VinDr của
                   họ "hơn + không kém" (3) − (1b) với δ đã duyệt); giao thức = config.py
    encoders/vt_s43.py  (0.3.0) ba hàng s43 = lớp bọc s42 của vindr_bench (đăng ký tạm, tệp s43) + cổng seed 43
    notebook/      (0.3.0) extract_s43.py (ô của 92f, G4) · probe_s43.py (ô của 92g, CPU)

Gói phụ thuộc gói `vindr_bench` (module 89) ở commit ghim — lớp bọc, kho ảnh, cổng, hàm chấm KHÔNG chép lại. Không chứa ID Drive, không dữ liệu."""
__version__ = '0.3.0'
