"""vindr_bench — giao thức chung 9 encoder × 4 đầu đọc trên VinDr-CXR (module 89; docs/PLAN_89_VINDR_9_ENCODER.md).

Mã sạch viết lại cho module 89 (Tuấn 13/11): mỗi thư mục con một trách nhiệm, notebook chỉ clone gói theo commit ghim rồi GỌI HÀM.
Module 86 (`vindr_protocol`) giữ nguyên làm hồ sơ các notebook đã chạy; logic đã kiểm ở 86 được chép sang đây có ghi nguồn + phép kiểm trùng.

    encoders/                 mỗi encoder một tệp: nạp trọng số gốc + lớp bọc (giao diện chung cho mọi đầu đọc); spec.py = bảng 9 encoder;
                              files.py = tệp trọng số trên Drive của ta (tên cố định, md5 ghim); clip_openai.py = tháp ảnh CLIP OpenAI;
                              chexfound.py + chexfound_src/ = lớp bọc CheXFound cho Ver2 (mã mô hình của họ, MIT / Apache 2.0)
                              b.py + b_src.py = lớp bọc B đông cứng (mã khối 78 / 79 chép nguyên văn; nạp checkpoint lượt 78x skip7 ep5)
                              vt.py + vt_src.py = 5 hàng ablation Việc B của PLAN_90 (B.4 v2 ghi ngược; RAD-DINO tinh chỉnh cho hàng chỉ đọc / không luồng)
    checks/                   cổng: weights.py (từng tensor = tệp) · features.py (số học theo nền) · safe_load.py (nạp checkpoint an toàn, torch 2.0 + mới)
    data/                     vindr.py: kho ảnh 518 của 85 (đệm vuông, cân bằng / không cân bằng histogram) + kho cắt giữa của 89k, nhãn 28 lớp, tách ·
                              crop_store.py: dựng kho cắt giữa (chuỗi giải mã của 85, chỉ đổi hình học) · transforms.py (biến đổi ảnh của mã CheXFound) ·
                              loaders.py (tập ảnh + dòng mẫu vô hạn cho vòng học của ta)
    heads/                    mldecoder.py: đầu ML-Decoder (lớp của 85f) + bộ chuyển cho giao diện đầu đọc của CheXFound ·
                              linear.py · glori.py (mã công khai của CheXFound, cho Ver2)
    protocols/chexfound_env/  Ver1: môi trường của họ, tệp chạy torchrun (tiêm lớp bọc + ML-Decoder), cổng số học, điều phối 4 loại đầu
    protocols/ours/           Ver2: giao thức `86b` cho 4 loại đầu trên lớp bọc chung (dò lr -> thêm hạt -> pha 2), cổng, khôi phục theo giai đoạn
    protocols/geometry.py     phép thử hình học (PLAN_89 §5.3): kho, 8 hàng, thành phần hai họ phép so (28 khoá 13/11 + 60 mở rộng 16/11)
    evaluation/               metrics.py (ngữ nghĩa torchmetrics 0.10.3, AUPRC hình thang, bootstrap trọng số) · panels.py (28 / 27 / 21 / 7) ·
                              collect.py (gom lượt Ver1 / Ver2 + 5 hàng Việc B từ Drive) · compare.py (ô, hiệu ghép cặp, hiệu của hiệu, max-T) · figures.py (biểu đồ)
    notebook/                 ver1.py · ver2.py · summary.py (notebook tổng hợp) · crop_store.py (4 notebook dựng kho cắt giữa): một hàm cho mỗi ô

Gói không chứa ID Drive, không chứa dữ liệu. Phần chạy trong tiến trình của họ (encoders, heads/mldecoder, protocols/chexfound_env/{launch, dataset,
probe_features}) chỉ dùng cú pháp / API có ở Python 3.9 + torch 2.0; protocols/ours + notebook/ver2 chỉ chạy ở Python của notebook."""
__version__ = '0.12.0'         # 0.12.0 (19/11, PLAN_90 Viec B buoc B4 — Tuan chot cach doc 10.2b): 89t them o C7c (notebook/summary.ablation: nam hang; ho 16 phep (3) - X, max-T, phan quyet tung o doc THEO TUNG DAU; bao kem; tach chon lr + cho chon lr mo ta; tep rieng ablation_*), evaluation/collect gom 5 hang Viec B (tham so ablation), figures.family_ci; 0.11.0 (19/11, PLAN_90 Viec B §10.2 — Tuan 'Okay'): 5 hang ablation vi tri mo-dun vung (encoders/vt.py + vt_src.py; spec.ABLATION_ROWS, chi Ver2 kho kcb, KHONG vao ALL_ROWS / ho 32 phep; spec ghim theo md5 khi chua biet kich thuoc); 0.10.0 (17/11, PLAN_90 Viec A — Tuan duyet): hang them ASG (encoders/asg.py; spec.EXTRA_ROWS / ALL_ROWS: 9 hang da khoa giu nguyen, hang them co bang + hinh nhung ho phep so RIENG trong 89t — compare ho_them); 0.9.2 (17/11, Tuan "add those in 89t ... separate"): 89t them o C7b phep thu cat giua RIENG (notebook/summary.crop_test: ba bang — Δ ba so do, tach chon lr luat / cung lr / theo AP, B - X tung kho; ba hinh — figures.delta_modes / lead_shift / cond_diff p21; tep rieng cat_giua_*); 0.9.1 (16/11, Tuan "add plots for AUPRC in 7 findings like RAD-DINO"): 89t them hinh 7 dau hieu RAD-DINO (TB 7 AUPRC hinh thang + AP bac thang; AUPRC tung dau hieu moi dieu kien); 0.9.0 (16/11, Tuan): kho cat giua 8 hang (data/crop_store.py, notebook/crop_store.py, protocols/geometry.py, Ver2 kho cg_kcb, 89t hai ho hinh hoc); PubMedCLIP chuan hoa ImageNet (spec.run_suffix '_imnet'); 0.8.0 (15/11): Ver1 cua B (khuc o 64: protocols/chexfound_env.BUILD_KW; cong so hoc moi truong moi attention eager); 0.7.0: hinh cua 89t = 28 nhan (AP bac thang, AUROC, AUPRC hinh thang; diem, khong CI), nhan tan xa khong chong; 0.6.0: B dong cung nhu moi backbone (encoders/b.py + b_src.py), 2 notebook Ver2 cua B
