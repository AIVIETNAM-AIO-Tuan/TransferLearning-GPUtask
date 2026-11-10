"""vindr_protocol — giao thức chung VinDr của module 86 (docs/PLAN_86_VINDR_GIAO_THUC_CHUNG.md), dựa trên mã công khai CheXFound
(RPIDIAL/CheXFound @ 7e288d0; giấy phép MIT ở chexfound/LICENSE).

Mỗi tệp một trách nhiệm:
    config      hằng số giao thức (theo mã CheXFound), cấu hình backbone, cấu hình chạy thử
    runtime     thiết bị, kiểu autocast, TF32, RAM, số worker
    chexfound   mã CheXFound đã chép: biến đổi ảnh, đầu Linear, đầu GLoRI
    data        kho ảnh VinDr trên Drive, tập ảnh, luồng mẫu vô hạn, bộ nạp
    backbones   nạp backbone đông cứng (RAD-DINO; hàng khác thêm khi dựng 86b)
    features    đặc trưng 4 tầng cuối + cổng kiểm
    heads       đầu ML-Decoder (của 85f) + tạo / chạy 3 loại đầu đọc
    metrics     AP / AUROC theo ngữ nghĩa torchmetrics 0.10.3
    training    lượt học của một đầu, bước học, dự đoán, chấm val
    selection   chọn lr, luật khoá epoch, kiểm hội tụ
    pipeline    vòng học song song, chạy thử pha 1, các phép kiểm

Notebook chỉ gọi hàm; khối Drive chuẩn (khoi_doc_drive / khoi_ghi_drive / khoi_ghi_cuoi) vẫn ở ô C0 của notebook và được truyền vào
dưới dạng hàm (kiem_ke, tai, xung_dot, ghi, ghi_json) — gói không chứa ID Drive."""
__version__ = '0.1.0'
