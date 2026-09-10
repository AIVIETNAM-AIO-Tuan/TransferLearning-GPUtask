# Ghi chú — vị trí adapter trong `07_screen_deep_ehr` (phát hiện khi đang chạy)

## Vấn đề

`ADAPTER_BLOCKS = (10, 11)` trên backbone 12 block. Thứ tự thực thi trong `forward_tail`:

```
block 10 → adapter[10] → block 11 → adapter[11] → layernorm → ML-Decoder
```

Hai adapter **không tương đương nhau về mặt cơ chế**:

| | sau nó còn gì | bản chất |
|---|---|---|
| `adapter[10]` | **block 11** (self-attention 1370 token) | **deep conditioning THẬT** — output bị xử lý lại, bản đồ attention trở thành hàm của EHR |
| `adapter[11]` | chỉ layernorm → ML-Decoder | **về chức năng là late fusion** — điều biến kênh trên patch cuối, không còn self-attention nào |

Nghĩa là screen đang kiểm **cả hai cơ chế cùng lúc**, và cơ chế *mới* (thứ 06b chưa kiểm)
chỉ có **đúng một block** để thể hiện mình.

## Hệ quả cho việc đọc kết quả

- **`D2 − D1` null** → cả hai cơ chế đều null. Kết luận **vẫn hợp lệ và còn mạnh hơn**, vì
  nó bao trùm cả late fusion (`adapter[11]`) lẫn deep conditioning một-block (`adapter[10]`).
- **`D2 − D1` dương** → **phải quy trách nhiệm trước khi claim**. Gain đến từ `adapter[10]`
  (mới) hay `adapter[11]` (chỉ là late fusion đội lốt)? Nếu chủ yếu từ `[11]` thì nó mâu
  thuẫn với 06b và cần giải thích, chứ không phải bằng chứng cho deep conditioning.

## Cách quy trách nhiệm (chỉ chạy nếu screen dương)

1. **Rẻ, hậu nghiệm — đo biên độ tương đối từng adapter:**
   ```python
   ratio_l = ‖ΔZ_l‖ / ‖Z_l‖        # cho l = 10, 11
   ```
   Nếu `ratio_10 ≈ 0` thì toàn bộ hiệu ứng nằm ở `[11]` ⇒ late fusion, không phải deep.

2. **Chắc chắn hơn — thêm một nhánh:** `D2_only10` (adapter **chỉ** ở block 10).
   `D2_only10 − D1` là hiệu ứng deep conditioning thuần.

## Nếu muốn cho giả thuyết deep conditioning cơ hội công bằng hơn

Đặt adapter **sớm hơn** (block 8–9) ⇒ có **3–4 block** xử lý lại thay vì 1.

Plan chọn block 11–12 (1-indexed) vì lý do ngữ nghĩa — hợp lý, nhưng nó **tối thiểu hóa độ
sâu tái xử lý**, tức đặt giả thuyết vào thế bất lợi nhất. Nếu screen null mà vẫn muốn thử
một lần nữa, chuyển adapter xuống block 8–9 là thí nghiệm tiếp theo **rẻ hơn nhiều** so với
mở pipeline confirmatory đầy đủ.

## Quyết định tại thời điểm ghi chú

**Không đổi cấu hình đang chạy.** Screen ở dạng hiện tại vẫn cho thông tin dùng được
(null bao trùm cả hai cơ chế). Việc quy trách nhiệm chỉ cần thiết nếu kết quả dương.
