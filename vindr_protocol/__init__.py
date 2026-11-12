"""vindr_protocol — giao thức chung VinDr của module 86 (docs/PLAN_86_VINDR_GIAO_THUC_CHUNG.md), dựa trên mã công khai CheXFound
(RPIDIAL/CheXFound @ 7e288d0; giấy phép MIT ở chexfound/LICENSE).

Mỗi tệp một trách nhiệm:
    config             hằng số giao thức (theo mã CheXFound), cấu hình backbone, cấu hình chạy thử, nhánh sát bài (PaperArms)
    runtime            thiết bị, kiểu autocast, TF32, RAM, số worker
    chexfound          mã CheXFound đã chép: biến đổi ảnh, đầu Linear, đầu GLoRI, mô hình DINOv2 gốc (chexfound/dinov2)
    data               kho ảnh VinDr trên Drive, tập ảnh, luồng mẫu vô hạn, bộ nạp
    backbones          nạp backbone đông cứng: RAD-DINO (HF), CheXFound (mã gốc của tác giả); hàng khác thêm khi dựng 86b
    features           đặc trưng 4 tầng cuối + cổng kiểm (HF Dinov2 / DINOv2 gốc)
    heads              đầu ML-Decoder (của 85f) + tạo / chạy 3 loại đầu đọc
    metrics            AP / AUROC theo ngữ nghĩa torchmetrics 0.10.3, AUPRC hình thang, bản có trọng số cho bootstrap theo ảnh
    training           lượt học của một đầu, bước học, dự đoán, chấm val
    selection          chọn lr, luật khoá epoch, chẩn đoán đường học
    pipeline           vòng học song song, chạy thử pha 1, các phép kiểm, lượt chính (+ nhánh sát bài)
    paper_calibration  ô hiệu chỉnh với Bảng IV bài CheXFound theo quy ước báo của bài (CHỈ mô tả)
    upstream           86e (PLAN_86 §12): chạy NGUYÊN VĂN linear_glori.py của repo CheXFound (clone commit ghim, không chép / sửa tệp của họ):
                       lớp dữ liệu VinDr theo khuôn CheXpert của họ, tệp chạy dưới torchrun, điều phối lượt chạy, chấm + đặt cạnh Bảng IV / 86b;
                       86f: môi trường thư viện của họ (PLAN_86 §13); 86g: ảnh 224 (set_image_size) + so 86f / 86f nửa 1 (§14.1)

Notebook chỉ gọi hàm; khối Drive chuẩn (khoi_doc_drive / khoi_ghi_drive / khoi_ghi_cuoi) vẫn ở ô C0 của notebook và được truyền vào
dưới dạng hàm (kiem_ke, tai, xung_dot, ghi, ghi_json) — gói không chứa ID Drive."""
__version__ = '0.5.0'
