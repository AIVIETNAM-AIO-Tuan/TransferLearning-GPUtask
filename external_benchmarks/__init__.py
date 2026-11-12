# -*- coding: utf-8 -*-
"""Gói của module `88` (docs/PLAN_88_SHENZHEN_RSNA.md v3 + §14.13–§14.17): kiểm ngoài VinDr trên Shenzhen (lao) và RSNA Pneumonia, thêm MRM /
BiomedCLIP. README: experiments/88_shenzhen_rsna/README.md.

Tệp theo trách nhiệm (PLAN_88 §6.2; dựng dần — đã có các tệp dưới đây, 12/11):
    config             hằng số đã kiểm: tệp chia Shenzhen (commit + md5), tệp tar ảnh Shenzhen (md5), công thức Shenzhen theo script CheXFound, 9 lr của
                       bài, md5 source CheXFound, checkpoint MRM / BiomedCLIP
    backbones          nạp MRM, BiomedCLIP (timm ViT-B/16 224) từ tệp thật + cổng; giao diện Backbone của gói `86` (vindr_protocol)
    splits             danh sách chia Shenzhen (BenchmarkTransformers đã ghim) -> CSV đúng khuôn lớp Shenzhen của họ; chuỗi dataset
    datasets           giải mã PNG Shenzhen = mã họ + convert('L') trước NumPy (palette); thống kê sai khác giải mã (chỉ numpy / PIL)
    shenzhen_images    tệp tar 662 PNG (dựng tất định trên máy; Colab: tải qua khối đọc Drive, md5 ghim, giải nén an toàn); đo giải mã / bộ đệm uint8
    shenzhen_dataset   lớp ShenzhenL = lớp Shenzhen của họ, chỉ đổi giải mã; đăng ký tên (IMPORT mã CheXFound: chỉ tiến trình con / bộ kiểm)
    shenzhen_launch    tệp chạy dưới torchrun: đăng ký ShenzhenL, bộ ghi lần chấm của 86e + ghi chỉ đọc (trạng thái metric, dtype, grad NaN, nạp
                       checkpoint, môi trường, bộ nhớ đỉnh); TUỲ CHỌN dùng chung đầu vào GLoRI (88a đo); rồi main() của linear_glori.py nguyên văn
    shenzhen_upstream  điều phối lượt Shenzhen bằng mã gốc trên hạ tầng 86e (vindr_protocol.upstream.runner) + cổng check_run; đo cho 88a
                       (memory_probe, compare_runs, log_timing)

Cần gói `vindr_protocol` (experiments/86_vindr_giao_thuc; kiểm với 0.4.1 @ e621f90 — bản đã đẩy) trên sys.path. Gói này KHÔNG sửa gói `86`."""

__version__ = '0.2.0'
