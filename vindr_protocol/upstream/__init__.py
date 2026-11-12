"""Chạy NGUYÊN mã nguồn CheXFound (RPIDIAL/CheXFound @ 7e288d0) trên VinDr — PLAN_86 §12, notebook 86e_chexfound_ma_goc.

Không sửa tệp nào của tác giả (clone commit ghim, kiểm HEAD + sạch). Ở đây chỉ có phần BẮT BUỘC thêm và phần điều phối:
    vindr_dataset  lớp dữ liệu VinDr theo khuôn lớp CheXpert của họ (họ không công bố lớp VinDr) + ghi CSV từ kho ảnh của 85 / 86
    launch         tệp chạy dưới torchrun: đăng ký tên 'VinDr' (bọc _parse_dataset_str), ghi lại MỌI lần chấm của lượt học (bọc evaluate:
                   preds / target đúng như vào torchmetrics + số họ tính), rồi gọi main() của linear_glori.py nguyên văn
    runner         clone / kiểm repo, tải trọng số, kiểm nạp bằng hàm của họ, chạy torchrun (theo dõi tiến độ, bắt tràn bộ nhớ), đọc kết quả của họ,
                   cổng trên lần chấm (AUROC tính lại = số họ tính; lr họ chọn đúng luật), dự đoán test = lần chấm test cuối của pha 2
    report         chấm dự đoán bằng torchmetrics 0.10.3 (bản họ ghim), bootstrap, đặt cạnh Bảng IV và nhánh sát bài của 86b

Không import gì ở mức gói (launch chạy trong tiến trình torchrun riêng; runner / report import khi gọi)."""
