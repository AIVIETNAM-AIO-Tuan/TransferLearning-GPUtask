"""Ver1 (PLAN_89 §4): `linear_glori.py` của CheXFound @ 7e288d0 NGUYÊN VĂN, chạy qua torchrun trong môi trường thư viện của họ (Python 3.9, torch 2.0.0
+ CUDA 11.7, xFormers 0.0.18, torchmetrics 0.10.3, transformers 4.36.2, timm 0.9.12) trên A100.

    environment.py   dựng môi trường của họ (micromamba + pip ghim), clone repo họ theo commit ghim (kiểm md5), kiểm import; gói cho Python notebook
    dataset.py       lớp dữ liệu VinDr theo khuôn lớp CheXpert của họ + ghi CSV (chạy trong cả hai Python)
    launch.py        tệp chạy dưới torchrun: đăng ký 'VinDr', ghi mọi lần chấm, thay mô hình bằng lớp bọc, (ML-Decoder) thay setup_glori
    probe_features.py  tiến trình đo đặc trưng 4 tầng của lớp bọc (cổng số học §8 mục 4) + tạo ảnh đầu vào bằng biến đổi của họ
    probes.py        điều phối các tiến trình đo đặc trưng từ notebook
    runner.py        một lượt học / một loại đầu (lưới 9 lr; 5 + 4 cho GLoRI và ML-Decoder), cổng lần chấm, chạy thử, lưu / khôi phục

Không import gì ở đây (launch.py / dataset.py / probe_features.py chạy trong Python 3.9 của môi trường họ)."""
