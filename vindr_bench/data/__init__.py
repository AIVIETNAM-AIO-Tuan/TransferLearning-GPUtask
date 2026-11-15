"""Dữ liệu VinDr-CXR của giao thức (PLAN_89 §1, §6): vindr.py = kho ảnh 518 của module 85 (đệm vuông; cân bằng / không cân bằng histogram) + nhãn
28 lớp + tách 13.500 / 1.500 / 3.000. Nhận các hàm của khối đọc Drive chuẩn làm tham số — gói không gọi Drive, không chứa ID Drive.
transforms.py = chexfound/data/transforms.py của CheXFound @ 7e288d0 từng byte (gốc Meta / DINOv2, Apache 2.0 ở đầu tệp; repo CheXFound MIT —
encoders/chexfound_src/LICENSE). loaders.py = tập ảnh + dòng mẫu vô hạn của vòng học Ver2 (chép từ vindr_protocol/data.py của 86)."""
