"""region_loc — phép thử định vị của luồng vùng (module 90; docs/PLAN_90_SO_SANH_GIAI_PHAU.md §11 — Việc C; Tuấn duyệt 17/11).

Hai phép thử trên mô hình ĐÔNG CỨNG, không học:
  C1  giải phẫu  : hộp tầng cuối của luồng vùng so với hộp gold Chest ImaGenome (bác sĩ vẽ)
  C2  tổn thương : bản đồ của mô hình (attention của đầu đọc; điểm theo vùng) so với hộp tổn thương MS-CXR

    config.py     tên thư mục / tệp trên Drive (KHÔNG có ID Drive), sha256 ghim, số đếm ghim, thứ tự 40 nhãn, ngưỡng luật (None = chưa khoá)
    geometry.py   hộp ảnh gốc -> khung ảnh của mô hình (hình học kho 518 của 78g), IoU — chép nguyên văn khối 78
    tables.py     bảng hộp gold / hộp tự động / MS-CXR (KHÔNG BAO GIỜ nạp cột câu report), gắn split CXR-LT, cổng số đếm
    images.py     lấy đúng các ảnh cần từ kho 518 trên Drive, đọc ảnh như đường chấm của 78
    model.py      mô hình cuối (lớp bọc B của gói vindr_bench, commit ghim) + ba đầu có sẵn trong checkpoint: đầu đọc trong lượt, cờ bất thường
                  từng vùng, nhãn theo vùng
    maps.py       bản đồ trên lưới 37 x 37 + mặt nạ hộp + hai số đo "trúng điểm" / "khối lượng trong hộp"
    metrics.py    gộp theo vùng / dấu hiệu, bootstrap theo bệnh nhân
    notebook.py   một hàm cho mỗi ô của notebook `90c_*`

Luật dữ liệu (PLAN_90 §11.2): chỉ đọc toạ độ hộp + tên vùng / tên dấu hiệu; val / test chỉ mở khi `config.THRESHOLDS` đã khoá (bản gói sau).
Gói phụ thuộc gói `vindr_bench` (module 89) ở commit ghim — không chép lại mã mô hình."""
__version__ = '0.2.0'
