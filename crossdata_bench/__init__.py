# -*- coding: utf-8 -*-
"""crossdata_bench — bảng ngoài miền của module 93 (docs/PLAN_93_BO_NGOAI_9_ENCODER.md): 9 encoder của `89` × bộ dữ liệu ngoài VinDr / MIMIC, phép kiểm
`89` Ver2 (= giao thức `86b`) KHÔNG đổi phần học.

Gói KHÔNG chép vòng học: lớp dữ liệu ở đây có cùng giao diện với VinDrData của `vindr_bench` (keys / split / indices() / labels / label_names /
main_classes / image_dir / store / sources) nên `vindr_bench.protocols.ours.pipeline.prepare` / `run` (commit ghim) chạy nguyên văn trên nó.

    config.py         commit ghim của hai gói dùng lại (vindr_bench @ c81af62, external_benchmarks @ 3c1ada7), chín hàng, từng bộ dữ liệu (nhãn, epoch,
                      đơn vị bootstrap), vân tay khoá / nhãn từng tách (tính trên máy 22/11 từ nguồn đã ghim md5)
    datasets/         base.py (StoreData = giao diện VinDrData + nạp kho từ Drive + vân tay) · rsna.py (tệp nối RSNA -> người NIH, nhãn CSV, chia 60 / 20 / 20
                      theo người) · shenzhen.py (tệp chia BenchmarkTransformers đã ghim qua external_benchmarks.splits) · jsrt.py (0.3.0: nhãn theo tên tệp
                      đối chiếu metadata, chia phân tầng một lần, cổng toàn vẹn theo mốc) · jsrt_manifest.py (mốc md5 + kích thước từng tệp, sinh từ máy) · chexpert.py (0.4.0: khoá theo đuôi đường dẫn — 0.4.1: thêm bố cục batch,
                      đọc / mô tả CSV, nhãn U / trống -> 0, tập học đề xuất theo người, tập lớp >= 10 ca dương ở test; 0.4.2: thư mục val/ = valid) ·
                      chexlocalize.py (0.4.2: bộ test chính thức CheXpert trong CheXlocalize — nhãn so groundtruth.csv GitHub ghim, val so valid CheXpert)
    store/            chain.py (giải mã PNG / DICOM -> chuỗi kho 518 đệm vuông của 85, KHÔNG cân bằng histogram — hàm của vindr_bench.data.crop_store) ·
                      build.py (nguồn trên Drive + cổng toàn vẹn của 88, tải nhiều luồng, cổng mẫu, ghi tar + bảng + meta) · listing.py (0.4.0: liệt kê cây lớn theo
                      lô K thư mục mỗi lần gọi, theo tầng, nhiều luồng, tiếp tục theo tầng; 0.4.1: quyền, tạo thư mục, tải lên kiểm md5, thùng rác) ·
                      zip_remote.py (0.4.0: danh mục zip trên Drive qua Range; 0.4.1: lấy một tệp ra từ xa) ·
                      azure_blob.py (0.4.3: container Azure qua link SAS — quyền / hạn, liệt kê, tải kiểm md5, che link)
    protocol/         heads.py (một nhãn: thêm lại chiều lớp mà `squeeze(-1)` của Linear / GLoRI bỏ khi C = 1 — bọc run_head trong lúc chạy, C >= 2 không đổi)
                      · run.py (MainRun theo bộ, nhãn dòng in theo số lớp, gọi pipeline của 89)
    evaluation/       single.py (AP bậc thang + AUPRC hình thang + AUROC, một nhãn và macro trên tập lớp cố định; hai nửa theo cụm người) ·
                      bootstrap.py (0.2.0: bootstrap theo cụm người NIH / theo ảnh, ô TB 5 hạt, hiệu ghép cặp, họ B − X + khoảng đồng thời max-T, điều
                      khoản trần — hàm số đo / max_t / verdict của vindr_bench)
    summary/          (0.2.0) collect.py (gom lượt 93v từ Drive + cổng từng lượt và giữa các lượt) · tables.py (bảng chữ mỗi bộ) · figures.py (hình mỗi bộ)
    notebook/         store.py (notebook kho 93k) · run.py (notebook lượt 93v) · summary.py (notebook tổng hợp 93t) · store_chexpert.py (0.5.0: kho
                      CheXpert chia phần 93k_kho_chexpert) · sessions.py (0.5.0: phiên, chặn hai phiên, xoá phiên bản cũ của tệp resume) · inventory.py (0.4.0: notebook kiểm kê
                      CheXpert 93i; 0.4.1: + lấy ảnh thiếu từ zip, zip đã khớp -> thùng rác) · chexlocalize.py (0.4.2: notebook kiểm kê CheXlocalize
                      93j, chỉ đọc) · fetch_chexpert.py (0.4.3: notebook 93h tải CheXpert/ của CheXlocalize từ Azure vào Drive): một hàm cho mỗi ô

Gói không chứa ID Drive, không chứa dữ liệu (chỉ mã băm / số đếm).

0.2.0 (22/11): notebook tổng hợp 93t — mỗi bộ một phần (bảng, luật, hình); không đổi gì của 0.1.0 trừ phần thêm vào config (tên hàng, mức tiếp xúc,
số cụm test, tên lượt tổng hợp).
0.3.0 (23/11): bộ JSRT (chỉ mô tả) — datasets/jsrt.py + jsrt_manifest.py, nhánh JSRT ở store/build.py + notebook/store.py, config (bộ, vân tay, số cụm test,
mức tiếp xúc); RSNA / Shenzhen và mọi hàm của 0.1.0 / 0.2.0 không đổi.
0.4.0 (24/11): kiểm kê CheXpert bản đầy đủ (notebook 93i, chỉ đọc) — datasets/chexpert.py, store/listing.py, store/zip_remote.py, notebook/inventory.py;
chưa có kho / lượt CheXpert (config chưa có bộ 'chexpert'); mọi hàm của 0.1.0–0.3.0 không đổi.
0.4.1 (24/11): sửa sau lượt 93i đầu — khoá CheXpert nhận bố cục batch của Stanford AIMI ('… batch 2 (train 1)/patient…', không có thư mục train/; 0.4.0
coi 223.414 ảnh train là "không đúng dạng"); so zip theo khoá / đường dẫn (không theo tên trần); ~1.000 ảnh mẫu theo zip; C7 lấy ảnh thiếu / lệch từ zip
(chỉ khi tài khoản là chủ); C10 chuyển zip đã so khớp vào thùng rác (Tuấn 24/11). Khoá của cột Path CSV không đổi (vân tay = 0.4.0).
0.4.2 (25/11): Tuấn chia CheXpert theo CheXFound -> cần bộ test chính thức (không có trong gói CheXpert của AIMI; nằm trong CheXlocalize, Drive
MIMIC_CXR/CheXlocalize) — datasets/chexlocalize.py + notebook/chexlocalize.py (notebook 93j, chỉ đọc); khoá CheXpert nhận thư mục val/ (= valid);
inventory.csvs thêm tham số `expect` (mặc định như cũ). Khoá + kết quả của 93i không đổi.
0.4.3 (25/11): 93j chưa đạt — thư mục CheXpert/ (ảnh val + test chính thức + 2 CSV) không có trên Drive, không có ở bản Redivis; chỉ còn ở trang AIMI cũ trên
Azure (Tuấn chọn cách 1: "làm cách 1 đi") — store/azure_blob.py (đọc container qua link SAS, chỉ thư viện chuẩn) + notebook/fetch_chexpert.py (notebook 93h:
tải CheXpert/ thẳng vào MyDrive/MIMIC_CXR/CheXlocalize/CheXpert/, kiểm từng tệp); gộp: hạ logger googleapiclient.http, chữ "JSON ở C9" của
inventory.tree. Khoá + kết quả của 93i / 93j không đổi.
0.5.0 (25/11): Tuấn chọn CheXpert (A) — chia đúng CheXFound (train 191.027 / val 202 / test 518 ảnh thẳng) — + 5 nhãn cuộc thi, chạy nhiều phiên Colab
("dựng thật cẩn thận"): datasets/chexpert.table_chexfound (3 CSV md5 ghim; ảnh thẳng theo khoá = luật 'lateral' của CheXFound) + config bộ 'chexpert'
(5 nhãn, cỡ + ca dương test ghim, vân tay theo tiền tố) + kho CHIA PHẦN (datasets/base: shard_*, _fetch_shards, load_store; store/build.chexpert_sources;
notebook/store_chexpert: phần lên Drive ngay qua luu_resume, C5 tải lại kiểm) + protocol/resume (lưu GIỮA giai đoạn vòng học 89 ~30 phút: bản chép
train_parallel thêm điểm bắt đầu / điểm lưu; chạy tiếp = chạy liền) + notebook/run (C2 chặn hai phiên, C5 kiểm chạy tiếp = chạy liền, C6 lưu giữa giai
đoạn + xoá phiên bản cũ) + notebook/sessions; store/chain thêm kiểu 'jpg'. Bộ cũ (RSNA / Shenzhen / JSRT): bảng, kho, ô notebook không đổi."""
__version__ = '0.5.0'
