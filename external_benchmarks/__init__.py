# -*- coding: utf-8 -*-
"""Gói của module `88` (docs/PLAN_88_SHENZHEN_RSNA.md v3 + §14.13–§14.20): kiểm ngoài VinDr trên Shenzhen (lao) và RSNA Pneumonia, thêm MRM /
BiomedCLIP. README: experiments/88_shenzhen_rsna/README.md.

Tệp theo trách nhiệm (PLAN_88 §6.2; dựng dần — đã có các tệp dưới đây, 12/11):
    config             hằng số đã kiểm: tệp chia Shenzhen (commit + md5), md5 bảng 662 PNG, công thức Shenzhen theo script CheXFound, 9 lr của bài,
                       4 lr kiểm lớp bọc, md5 source CheXFound, checkpoint MRM / BiomedCLIP
    integrity          toàn vẹn dữ liệu trên Drive: md5 + kích thước từng tệp qua Drive API (không tải) vs mốc bản đã kiểm trên máy (Shenzhen từng tệp;
                       RSNA tổng + 256 nhóm + CSV từng tệp)
    shenzhen_manifest  mốc Shenzhen: md5 + kích thước từng PNG (662) + shenzhen_metadata.csv (chỉ mã băm)
    rsna_manifest      mốc RSNA: tổng + 256 nhóm cho DICOM train / test, từng tệp cho 3 CSV, kết quả kiểm bản trên máy (chỉ mã băm / số đếm)
    backbones          nạp MRM, BiomedCLIP (timm ViT-B/16 224) từ tệp thật + cổng; giao diện Backbone của gói `86` (vindr_protocol)
    splits             danh sách chia Shenzhen (BenchmarkTransformers đã ghim) -> CSV đúng khuôn lớp Shenzhen của họ; chuỗi dataset
    datasets           giải mã PNG Shenzhen = mã họ + convert('L') trước NumPy (palette); thống kê sai khác giải mã (chỉ numpy / PIL)
    shenzhen_images    tải 662 PNG Shenzhen từ Drive (sau integrity.check_shenzhen) về thư mục tên gốc; đo giải mã / bộ đệm uint8
    shenzhen_dataset   lớp ShenzhenL = lớp Shenzhen của họ, chỉ đổi giải mã; đăng ký tên (IMPORT mã CheXFound: chỉ tiến trình con / bộ kiểm)
    shenzhen_launch    tệp chạy dưới torchrun: đăng ký ShenzhenL, bộ ghi lần chấm của 86e + ghi chỉ đọc (trạng thái metric, dtype, grad NaN, nạp
                       checkpoint, môi trường, bộ nhớ đỉnh); TUỲ CHỌN dùng chung đầu vào GLoRI (88a đo); rồi main() của linear_glori.py nguyên văn
    shenzhen_upstream  điều phối lượt Shenzhen bằng mã gốc trên hạ tầng 86e (vindr_protocol.upstream.runner) + cổng check_run; đo cho 88a
                       (memory_probe, compare_runs, log_timing), tóm tắt một dòng, pack_logs

Cần gói `vindr_protocol` (experiments/86_vindr_giao_thuc; kiểm với 0.4.1 @ e621f90 — bản đã đẩy) trên sys.path. Gói này KHÔNG sửa gói `86`.
0.3.0 (12/11): ảnh Shenzhen đọc thẳng từ Drive MIMIC_CXR/Shenzhen (Tuấn) — bỏ đường tệp tar của 0.2.0."""

__version__ = '0.3.0'
