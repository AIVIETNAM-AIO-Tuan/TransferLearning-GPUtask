# 84 · Nhánh C — chạy theo seed bằng gói mã `nhanh_c`

Tuấn duyệt 29/10 (Event 2026-10-29 "Tuấn giữ nhánh C …"; `docs/PLAN_SAU_78Y.md` §A.4v, `docs/SPEC_SAU_78Y.md` §B.2ab).

**Nhánh C** = công thức lượt gói ô cắt s42 (`78x_jca_v3b_dp_ema_v2_o_s42`):

- **Phần 1:** B.4 v2 + 5 ô lưới phóng 518 (không gradient) + adapter + skip block 7 → 8 + đầu J đọc 6 view lúc train.
- **Phần 2:** `78z` 4 đầu chuẩn, đọc 1 view, đường skip đông cứng.
- **Phần 3:** gộp trọng số ep1–10. Đầu quyết định là `fus_attn_hr`.

Năm seed gồm s42 (lượt gói đã chạy) và s43–s46 (chạy bằng gói này).

## Tệp

| Tệp | Vai trò | Lên GitHub |
|---|---|---|
| `nhanh_c/__init__.py` | API: `buoc(ten, globals(), seed, thu=False)` chạy một bước trong namespace notebook (như chạy ô); `ds_buoc(phan)`; kiểm sha256 từng tệp bước (chuẩn hoá CRLF → LF) | có |
| `nhanh_c/cau_hinh.py` | cấu hình theo seed (`RUN_TAG`, thư mục `stage4_nb84`, mốc C s42, ngưỡng cổng sức khoẻ) | có |
| `nhanh_c/buoc/*.py`, `nhanh_c/thu_tu.py` | **sinh tự động** bởi `dung_84.py`: mỗi tệp = một ô đã kiểm của 78 + dòng vá "nhanh C"; thứ tự + sha256 | có |
| `dung_84.py` | bộ dựng: sinh `buoc/` từ `78_chi_anh/dung_78x_cat.py` (+ vá theo seed), sinh notebook | không |
| `test_84.py` | kiểm offline (CPU, không vòng train) | không |
| `day_nhanh_c.py` | đẩy **chỉ** `nhanh_c/**/*.py` lên nhánh mồ côi `nhanh-c` (chỉ mục tạm; không đụng `main` / cây làm việc); quét bí mật trước; ghi `da_day.json` | không |
| `da_day.json` | commit đã đẩy + blob từng tệp (notebook ghim commit này) | không |
| `84_nhanh_c_s43.ipynb` … `_s46.ipynb`, `84_nhanh_c_s43_thu.ipynb` | notebook mỗi seed: ô tải gói (ghim commit) + mỗi ô một dòng `NC.buoc(...)` | không |

**Không sửa tay** `nhanh_c/buoc/` hay `thu_tu.py`. Nếu sửa tay, `buoc()` dừng vì sha256 khác bản sinh. Khi cần đổi, làm theo bốn bước:

1. Sửa `dung_84.py` (hoặc `cau_hinh.py` / `__init__.py`).
2. Chạy `python dung_84.py` rồi `python test_84.py`.
3. Chạy `python day_nhanh_c.py` (tạo commit mới trên `nhanh-c`).
4. Chạy `python dung_84.py --notebook` để ghim commit mới vào notebook, rồi chạy lại `test_84.py`. Notebook đã tải lên Colab thì phải tải lại.

**Repo GitHub công khai:**

- Trong gói không có ID Drive, email hay dữ liệu.
- ID thư mục `cxrlt_2024` chỉ nằm trong ô tải gói của notebook, dưới dạng biến môi trường `CXRLT_GOC_ID`.
- Nhánh `main` (đi trước `origin` 46 commit) **không** đẩy.

## Chạy

1. **Bản thử** `84_nhanh_c_s43_thu.ipynb`: G4, Restart session → Run all, khoảng 40 phút.
   - Gửi Claude: ô tải gói, dòng `cong o cat` (`Phần 1 · C4`), các dòng `buoc … VRAM dinh … img/s` cùng dòng epoch và `O CAT` (`Phần 1 · C6`).
2. **Bản chính** `84_nhanh_c_s43.ipynb` … `_s46.ipynb`: mỗi notebook một tài khoản được (chạy song song). G4, Restart session → Run all, khoảng 21 giờ, resume được.
   - Ô chuyển Phần 1 → 2 dừng khi cổng sức khoẻ ra `YEU`: báo Claude. Nếu Tuấn quyết chạy tiếp thì chạy một ô `TIEP_TUC_C = True` rồi chạy lại ô chuyển.
   - **Đừng tắt runtime trước `Phần 3 · C9`.**
   - Ô tải gói khóa cả commit lẫn đường dẫn của module đã import. Chạy lại cùng commit được; đổi commit hoặc đã import ngoài loader thì phải **Restart session**. Cổng dừng trước khi checkout để tránh dùng cấu hình cũ trong bộ nhớ.
3. Ghi kết quả vào `stage4_nb84/r84x_C_s<seed>` (Phần 1), `r84z_C_s<seed>` (Phần 2), `r84z_C_s<seed>_gts` (Phần 3).
   - Bước `TỔNG HỢP` cuối Phần 3 in mọi seed đã xong: TB ± độ lệch chuẩn qua seed, bộ gộp các seed, C − A ở s42.
   - Không đọc nhãn test.

## Kiểm (`python test_84.py`, 67/67 ngày 29/10)

- **Gói và bộ dựng:** gói trên đĩa trùng bộ dựng; chỉ có tệp `.py`; không bí mật.
- **Phần 1:** khác ô đã chạy của lượt gói s42 đúng các dòng "nhanh C".
- **Phần 2 / 3:** khác ô đã chạy của lượt skip7 đúng các ô nhánh C.
- **Dòng cấu hình theo seed:** chạy thật cho s42–s46.
- **Chọn epoch:** đỉnh VAL, bằng nhau → epoch muộn hơn, cùng các ca phải dừng.
- **Cổng sức khoẻ + ô chuyển phần:** chạy thật.
- **Cổng cấu trúc `C2b`:** chạy thật.
- **`C6S` theo seed:** tên tệp đúng; với SEED = 42 trùng tên của lượt cũ.
- **`TỔNG HỢP`:** chạy trên dự đoán giả.
- **`buoc()`:** lỗi đầu vào, trộn seed, sửa tay, CRLF, `%reset` của ô chuyển phần.
- **Notebook:** trùng bộ dựng và ghim đúng commit đã đẩy.
- **Loader:** nạp mới ghi commit vào module; cùng commit chạy lại được; module chưa ghim, commit cũ và đường dẫn chỉ giống tiền tố đều dừng. Các ca dùng gói tạm và Git giả, không gọi mạng.
