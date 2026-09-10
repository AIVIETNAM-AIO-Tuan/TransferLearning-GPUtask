# 09 · Block-select — block 11 vs block 12 làm đầu vào ML-Decoder

**Câu hỏi:** 08 đo block 11 > block 12 (+0.0064) qua linear probe. Lợi thế đó có chuyển sang
ML-Decoder không? **Trả lời: KHÔNG.**

## Thiết lập
- 4 arm, cùng mọi thứ trừ nguồn token, một backbone forward chung, backbone đông cứng.
- Loss/optim/schedule/early-stop/pooling **y hệt H5** để `b12ln` tái lập được.
- Ngưỡng ±0.005 (quy tắc thường trực, có trước thí nghiệm).

## Kết quả — seed 42, early-stop tại ep8 (đỉnh val)

| arm | token vào decoder | best val mAP |
|---|---|---|
| `b12ln` | block 12 + layernorm cuối = `last_hidden_state` (=H5) | 0.4357 |
| `b12` | block 12 thô | 0.4356 |
| `b11` | block 11 thô | 0.4359 |
| `b1112` | block 11 ⊕ 12 (2738 token) | **0.4379** |

**Cổng sanity:** `b12ln` = 0.4357 vs H5 ref 0.4336 → **+0.0021, trong nhiễu**. Pipeline dựng đúng.

## Estimand

| so sánh | Δ | phát biểu |
|---|---|---|
| **`b11 − b12`** (chính) | **+0.0003** | **NULL** — cách ngưỡng 20× |
| `b12ln − b12` (layernorm cuối) | +0.0001 | null ở hội tụ (pilot 1-epoch +0.012 bị LayerNorm arm bù hết) |
| `b1112 − b12ln` (cả hai block) | +0.0022 | dưới ngưỡng +0.005, mới 1 seed |

## Vì sao 08 và 09 mâu thuẫn — 09 đúng

08 đo trên `mean(patch)` + linear. Pooling trung bình + phân loại tuyến tính làm block cuối
(chuyên hóa pretrain) trông tệ. Nhưng ML-Decoder **cross-attend toàn bộ 1369 token** — rút
được thứ cần từ block nào cũng như nhau. **Lợi thế block-11 là giả tạo của phương pháp probe**,
không phải tính chất decoder khai thác được.

## Quyết định: DỪNG

- Câu hỏi chính null (+0.0003, cách ngưỡng 20×). 3 seed chỉ xác nhận null.
- `b1112` +0.0022 dưới ngưỡng; thêm seed chỉ thu hẹp CI quanh +0.0022, không đẩy tới +0.005.
- **Không mở test. Không chạy thêm seed.** Đuổi theo +0.0022 = dịch cột gôn.

## Bối cảnh — null hội tụ thứ ba của dự án

| hướng | phép thử | kết quả |
|---|---|---|
| EHR deep conditioning | H6−H5, shuffle, stacking, D2−D1 (07) | null (4/4) |
| **block-selection** | b11−b12, 09 | **null** |

Khuôn chung: thứ trông hứa hẹn qua proxy rẻ (probe, một metric) không sống sót qua CI ghép
cặp trên mục tiêu thật. `b1112` (+0.0022) là quan sát thật nhưng dưới ngưỡng và đắt hơn ở
inference (2738 token) — không đáng theo đuổi.

## Còn lại trên bàn (chưa chạy)
- Đo lượng tín hiệu bị 04a center-crop vứt đi (15% trục dài) — ~20 phút GPU.
