"""vindr_bench — giao thức chung 9 encoder × 4 đầu đọc trên VinDr-CXR (module 89; docs/PLAN_89_VINDR_9_ENCODER.md).

Mã sạch viết lại cho module 89 (Tuấn 13/11): mỗi thư mục con một trách nhiệm, notebook chỉ clone gói theo commit ghim rồi GỌI HÀM.
Module 86 (`vindr_protocol`) giữ nguyên làm hồ sơ các notebook đã chạy; logic đã kiểm ở 86 được chép sang đây có ghi nguồn + phép kiểm trùng.

    encoders/                 mỗi encoder một tệp: nạp trọng số gốc + lớp bọc (giao diện chung cho mọi đầu đọc); spec.py = bảng 9 encoder
    checks/                   cổng: weights.py (từng tensor = tệp) · features.py (số học theo nền: tương đối + chặn bền)
    data/                     vindr.py: kho ảnh 518 của 85 (đệm vuông, cân bằng / không cân bằng histogram), nhãn 28 lớp, tách
    heads/                    mldecoder.py: đầu ML-Decoder (lớp của 85f) + bộ chuyển cho giao diện đầu đọc của CheXFound
    protocols/chexfound_env/  Ver1: môi trường của họ, tệp chạy torchrun (tiêm lớp bọc + ML-Decoder), cổng số học, điều phối 4 loại đầu
    evaluation/               metrics.py (ngữ nghĩa torchmetrics 0.10.3, AUPRC hình thang, bootstrap trọng số) · panels.py (28 / 27 / 21 / 7)
    notebook/                 ver1.py: một hàm cho mỗi ô của notebook Ver1
    (bước sau)                protocols/ours (Ver2), kho cắt giữa, lớp bọc 6 encoder còn lại + B, bảng gộp (max-T)

Gói không chứa ID Drive, không chứa dữ liệu. Phần chạy trong tiến trình của họ (encoders, heads, protocols/chexfound_env/{launch, dataset,
probe_features}) chỉ dùng cú pháp / API có ở Python 3.9 + torch 2.0."""
__version__ = '0.2.0'
