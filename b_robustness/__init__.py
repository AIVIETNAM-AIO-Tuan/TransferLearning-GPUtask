"""b_robustness — kiểm thử kiến trúc B, đợt 1 (module 87; docs/PLAN_KIEM_THU_KIEN_TRUC_B.md §15–§17).

Mỗi tệp một trách nhiệm:
    config        seed K3, tên hàng / lượt / thư mục Drive, số quyết định, nguồn đối chiếu, hạt bootstrap
    drive         đọc qua khối Drive chuẩn của notebook (hàm truyền vào) + ghi nguồn từng tệp
    audit         K0 — kiểm nguồn backbone từng seed (đủ cho P1 / đủ cho P2)
    vindr         nhãn VinDr test (85b) + hai nửa; dữ liệu P2 (= ô D của 85f)
    predictions   nạp dự đoán P1 / P2 các hàng (85f, 87c) + cổng
    metrics       AP bậc thang / AUROC / AUPRC hình thang (+ bản có trọng số), tóm tắt, bootstrap tập lớp cố định
    rules         luật P2 (đặt trước, 85g)
    sensitivity   K5 — thống kê lại P2 đã có
    seed_table    K3 — bảng P1 / P2 qua backbone seed
    backbone_b    Phần 3 của gói nhanh_c (module 84; biến thể Bd, riêng s43 biến thể B) + Part3View
    transfer      P1 VinDr (ô C6V của 85f), trích token (ô T), bản ghi + manifest (ô G, M)
    heads         học lại đầu P2 (ô H của 85f)
    pipeline      điều phối từng notebook + ghi tệp bằng hàm của khối Drive chuẩn

Notebook chỉ gọi hàm. Khối Drive chuẩn ở ô C0 của notebook và được truyền vào dưới dạng hàm (kiem_ke, tai, xung_dot, ghi, ghi_json)
— gói không chứa ID Drive."""
__version__ = '0.2.0'   # 0.2.0 (10/11): thêm B s43 (biến thể B) vào K3 — §17
