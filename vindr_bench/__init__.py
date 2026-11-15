"""vindr_bench — giao thức chung 9 encoder × 4 đầu đọc trên VinDr-CXR (module 89; docs/PLAN_89_VINDR_9_ENCODER.md).

Mã sạch viết lại cho module 89 (Tuấn 13/11): mỗi thư mục con một trách nhiệm, notebook chỉ clone gói theo commit ghim rồi GỌI HÀM.
Module 86 (`vindr_protocol`) giữ nguyên làm hồ sơ các notebook đã chạy; logic đã kiểm ở 86 được chép sang đây có ghi nguồn + phép kiểm trùng.

    encoders/                 mỗi encoder một tệp: nạp trọng số gốc + lớp bọc (giao diện chung cho mọi đầu đọc); spec.py = bảng 9 encoder;
                              files.py = tệp trọng số trên Drive của ta (tên cố định, md5 ghim); clip_openai.py = tháp ảnh CLIP OpenAI;
                              chexfound.py + chexfound_src/ = lớp bọc CheXFound cho Ver2 (mã mô hình của họ, MIT / Apache 2.0)
                              b.py + b_src.py = lớp bọc B đông cứng (mã khối 78 / 79 chép nguyên văn; nạp checkpoint lượt 78x skip7 ep5)
    checks/                   cổng: weights.py (từng tensor = tệp) · features.py (số học theo nền) · safe_load.py (nạp checkpoint an toàn, torch 2.0 + mới)
    data/                     vindr.py: kho ảnh 518 của 85 (đệm vuông, cân bằng / không cân bằng histogram), nhãn 28 lớp, tách ·
                              transforms.py (biến đổi ảnh của mã CheXFound) · loaders.py (tập ảnh + dòng mẫu vô hạn cho vòng học của ta)
    heads/                    mldecoder.py: đầu ML-Decoder (lớp của 85f) + bộ chuyển cho giao diện đầu đọc của CheXFound ·
                              linear.py · glori.py (mã công khai của CheXFound, cho Ver2)
    protocols/chexfound_env/  Ver1: môi trường của họ, tệp chạy torchrun (tiêm lớp bọc + ML-Decoder), cổng số học, điều phối 4 loại đầu
    protocols/ours/           Ver2: giao thức `86b` cho 4 loại đầu trên lớp bọc chung (dò lr -> thêm hạt -> pha 2), cổng, khôi phục theo giai đoạn
    evaluation/               metrics.py (ngữ nghĩa torchmetrics 0.10.3, AUPRC hình thang, bootstrap trọng số) · panels.py (28 / 27 / 21 / 7) ·
                              collect.py (gom lượt Ver1 / Ver2 từ Drive) · compare.py (ô, hiệu ghép cặp, max-T) · figures.py (biểu đồ)
    notebook/                 ver1.py · ver2.py · summary.py (notebook tổng hợp): một hàm cho mỗi ô của notebook
    (bước sau)                kho cắt giữa, Ver1 của B

Gói không chứa ID Drive, không chứa dữ liệu. Phần chạy trong tiến trình của họ (encoders, heads/mldecoder, protocols/chexfound_env/{launch, dataset,
probe_features}) chỉ dùng cú pháp / API có ở Python 3.9 + torch 2.0; protocols/ours + notebook/ver2 chỉ chạy ở Python của notebook."""
__version__ = '0.6.0'          # 0.6.0 (15/11): B dong cung nhu moi backbone (encoders/b.py + b_src.py), 2 notebook Ver2 cua B
