# -*- coding: utf-8 -*-
"""crossdata_bench — bảng ngoài miền của module 93 (docs/PLAN_93_BO_NGOAI_9_ENCODER.md): 9 encoder của `89` × bộ dữ liệu ngoài VinDr / MIMIC, phép kiểm
`89` Ver2 (= giao thức `86b`) KHÔNG đổi phần học.

Gói KHÔNG chép vòng học: lớp dữ liệu ở đây có cùng giao diện với VinDrData của `vindr_bench` (keys / split / indices() / labels / label_names /
main_classes / image_dir / store / sources) nên `vindr_bench.protocols.ours.pipeline.prepare` / `run` (commit ghim) chạy nguyên văn trên nó.

    config.py         commit ghim của hai gói dùng lại (vindr_bench @ c81af62, external_benchmarks @ 3c1ada7), chín hàng, từng bộ dữ liệu (nhãn, epoch,
                      đơn vị bootstrap), vân tay khoá / nhãn từng tách (tính trên máy 22/11 từ nguồn đã ghim md5)
    datasets/         base.py (StoreData = giao diện VinDrData + nạp kho từ Drive + vân tay) · rsna.py (tệp nối RSNA -> người NIH, nhãn CSV, chia 60 / 20 / 20
                      theo người) · shenzhen.py (tệp chia BenchmarkTransformers đã ghim qua external_benchmarks.splits) · jsrt.py (0.3.0: nhãn theo tên tệp
                      đối chiếu metadata, chia phân tầng một lần, cổng toàn vẹn theo mốc) · jsrt_manifest.py (mốc md5 + kích thước từng tệp, sinh từ máy)
    store/            chain.py (giải mã PNG / DICOM -> chuỗi kho 518 đệm vuông của 85, KHÔNG cân bằng histogram — hàm của vindr_bench.data.crop_store) ·
                      build.py (nguồn trên Drive + cổng toàn vẹn của 88, tải nhiều luồng, cổng mẫu, ghi tar + bảng + meta)
    protocol/         heads.py (một nhãn: thêm lại chiều lớp mà `squeeze(-1)` của Linear / GLoRI bỏ khi C = 1 — bọc run_head trong lúc chạy, C >= 2 không đổi)
                      · run.py (MainRun theo bộ, nhãn dòng in theo số lớp, gọi pipeline của 89)
    evaluation/       single.py (AP bậc thang + AUPRC hình thang + AUROC, một nhãn và macro trên tập lớp cố định; hai nửa theo cụm người) ·
                      bootstrap.py (0.2.0: bootstrap theo cụm người NIH / theo ảnh, ô TB 5 hạt, hiệu ghép cặp, họ B − X + khoảng đồng thời max-T, điều
                      khoản trần — hàm số đo / max_t / verdict của vindr_bench)
    summary/          (0.2.0) collect.py (gom lượt 93v từ Drive + cổng từng lượt và giữa các lượt) · tables.py (bảng chữ mỗi bộ) · figures.py (hình mỗi bộ)
    notebook/         store.py (notebook kho 93k) · run.py (notebook lượt 93v) · summary.py (notebook tổng hợp 93t): một hàm cho mỗi ô

Gói không chứa ID Drive, không chứa dữ liệu (chỉ mã băm / số đếm).

0.2.0 (22/11): notebook tổng hợp 93t — mỗi bộ một phần (bảng, luật, hình); không đổi gì của 0.1.0 trừ phần thêm vào config (tên hàng, mức tiếp xúc,
số cụm test, tên lượt tổng hợp).
0.3.0 (23/11): bộ JSRT (chỉ mô tả) — datasets/jsrt.py + jsrt_manifest.py, nhánh JSRT ở store/build.py + notebook/store.py, config (bộ, vân tay, số cụm test,
mức tiếp xúc); RSNA / Shenzhen và mọi hàm của 0.1.0 / 0.2.0 không đổi."""
__version__ = '0.3.0'
