# REVIEW — nb19 `19_fov_ablation.ipynb` · FOV, hướng lưu trữ, và số phận của `F3`

Ngày: 2026-07-31 · Trạng thái: **letterbox đã bỏ**; còn hai việc CPU

---

## §1 · Giả thuyết "ảnh bị xoay 90°" — **ĐÓNG**

`C5` in 8 ảnh kèm khung đỏ. Cả hai nhóm đều là phim ngực **đúng hướng giải phẫu**; nhóm `ngang`
chỉ là khung lưu rộng hơn.

| view position | % `ngang` | n |
|---|---|---|
| `AP` | **47.7%** | 1,179 |
| `PA` | **21.0%** | 778 |

Chênh **26.7%** ⇒ hướng gắn với **loại phim**. Ba trong bốn ảnh nhóm `ngang` mang nhãn cháy
`PORTABLE` / `SEMI-ERECT` — phim `AP` di động chụp tại giường dùng khung rộng hơn. Lý do thu
nhận, không phải lỗi lưu trữ.

**Hệ quả:** không thêm bước xoay ở tiền xử lý. Phần bị `crop1` cắt khác nhau theo hướng —
`doc` mất trên+dưới (góc sườn hoành → `Pleural_Other`), `ngang` mất trái+phải (sườn bên →
`Fracture`).

---

## §2 · Tái trọng cohort: `ngang` là **25.7%**, không phải `19.4%`

Mẫu `C5` ép 50/50 frontal/lateral; cohort thật là `frontal 67.7% / lateral 32.3%`.

| | `doc` | `ngang` | `vuong` |
|---|---|---|---|
| trong mẫu 50/50 | 74.9% | 19.4% | 5.7% |
| **tái trọng cohort** | **69.4%** | **25.7%** | **5.0%** |

≈ **88,918 ảnh** phim thẳng lưu ngang. Đã chạy, số lấy từ notebook.

---

## §3 · Giả thuyết viền đen — **SAI**

Tôi dự đoán `28.5%` là của **khung** chứ không phải **giải phẫu**, vì ảnh `C5` trông như có dải
đen lớn. `C7` đo trên 1,015 ảnh phân tầng:

| | |
|---|---|
| `khung_là_giải_phẫu` (trung vị) | **99.1%** |
| `mất KHUNG` → `mất GIẢI PHẪU` | `28.5%` → **`25.4%`** |
| `AR` khung → giải phẫu | `1.201` → `1.195` |
| verdict | `ƯU TIÊN` → `ƯU TIÊN` *(không đổi)* |

Khung gần như toàn bộ là giải phẫu. **Giả thuyết bác bỏ.**

### Nhưng `25.4%` chưa bền — đây là việc 1

`C7` chạy ở `thr_frac = 0.03`, tức mọi pixel `> 7.6/255` tính là giải phẫu. Vùng ngoài
collimator sau nén JPEG thường ở `5–20`. Hai dấu hiệu:

- trung vị `khung_là_giải_phẫu` của `frontal` ra **đúng bằng `1.0000`** — trên nửa số ảnh không
  phát hiện viền nào, ngược ấn tượng mắt ở `C5`;
- `25.4%` chỉ hơn ngưỡng `ƯU TIÊN` đúng **`0.4 pp`**.

Kiểm offline xác nhận rủi ro: với viền cường độ `12/255` trên ảnh `max = 250` (tỉ lệ `0.048`),
`thr = 0.03` **bỏ sót hoàn toàn** viền; `thr = 0.06` bắt được.

**`C8` (mới)** giải mã **một lần** mỗi ảnh rồi tính hộp bao ở `thr ∈ {0.03, 0.06, 0.10, 0.15}` —
chi phí vẫn `~1.2 phút`, không nhân lên. Dùng **đúng mẫu** của `C7` (đọc `dicom_id` từ CSV cache)
nên so cặp được. Kèm chẩn đoán chọn ngưỡng thay vì đoán: cường độ vành ngoài `1%` khung so với
`max`. Cell **không** tự đổi verdict — in bảng để đọc.

---

## §4 · **Letterbox bị bỏ** *(chốt 2026-07-31)*

### Số quyết định đã có sẵn trong nb17

`resize518` chính là *"thu hồi crop2, giữ nguyên crop1"* ⇒ cho **độ dốc FOV→AP đo được**:

| bước | FOV | `dMacro` |
|---|---|---|
| `center518 → resize518` | `71.5% → 83.2%` (**+11.8 pp**) | `+0.00066`, CI `[-0.00052, +0.00174]` **chứa 0** |
| còn lại `83.2% → 100%` | **+16.8 pp** | ngoại suy **`+0.00094`** |
| cổng | | `+0.00200` |

Ngoại suy tuyến tính cho **chưa tới nửa cổng**. `Fracture` — nhãn được cho là hưởng lợi nhất —
thực tế **giảm** (`0.1085 → 0.1076`) khi mở FOV.

### Và letterbox là công cụ sai

| phương án | FOV | resolution | padding | train lại |
|---|---|---|---|---|
| hiện tại `center518` | `71.5%` | `100%` | 0 | — |
| nb17 `resize518` *(đã đo)* | `83.2%` | `92.7%` | 0 | không |
| ~~letterbox → `center518`~~ | `92.7%` | **`83.2%`** | **`10.2%`** | **~4 h** |
| 2-crop + `Resize(518)` | **`100%`** | `92.7%` | 0 | không |

Letterbox mua `+29.6%` diện tích bằng cách trả `−16.8%` độ phân giải, biến `~140/1369` patch
token thành viền đen, và lệch phân phối so với pretraining RAD-DINO. Với `Fracture` và
`Pleural_Other` — cấu trúc nhỏ, mảnh — đánh đổi đó có thể **âm**.

2-crop trội hơn trên **mọi** trục. Và scale của nó (`0.203616`) **trùng khít** scale của
`resize518` mà nb17 đã đo (`lệch 0.00%`) — nên thành phần độ phân giải **đã được kiểm và lành
tính**; 2-crop chỉ thêm đúng một biến là FOV.

---

## §5 · `C3` viết lại: shard **giữ trọn FOV** — việc 2

`04a` cell 8 làm hai việc trong một hàm:

```python
s = STORE_SIZE / min(w, h)
im = im.resize(...)                       # (1) shortest-edge -> 559   [khong mat gi]
im = im.crop((l, t, l+559, t+559))        # (2) center-crop vuong      [MAT VINH VIEN]
```

Bước (2) không lấy lại được ở eval-time. `C3` mới dựng shard **chỉ với bước (1)** — ảnh biến
thiên `559 × round(559·AR)`, giữ trọn giải phẫu, đúng tỉ lệ, **không thêm một pixel đen nào**.

Nó là **siêu tập** của shard hiện hành: mọi phương án FOV về sau (2-crop, multi-scale, non-square
input) đều dẫn xuất được ở **eval-time** từ nó, không phải đọc lại `346,578` ảnh gốc lần nữa.
`~2 h` trả một lần.

### Cổng tái tạo — bài học từ smoke test

Cổng đầu tôi đặt `mean |Δ| ≤ 6/255` và `corr ≥ 0.999`. Smoke test **bác bỏ**: với ảnh
`1763×2021` thì `corr = 0.954`, FAIL — dù pipeline hoàn toàn đúng.

Nguyên nhân: JPEG mã hoá lại trên lưới block khác. Offset crop `(672−559)//2 = 56` chia hết cho
`8` ⇒ lưới trùng ⇒ `|Δ| = 0.008`. Offset `(641−559)//2 = 41` ⇒ lệch `1` ⇒ sai số nhảy hàng chục
lần. **`|Δ|` phụ thuộc `offset % 8`, không phụ thuộc tính đúng đắn.**

Cổng đổi sang **tìm dịch chuyển**: quét `±2 px`, vị trí khớp tốt nhất phải đúng `(0,0)`. Đại
lượng này bền với nhiễu JPEG và bắt đúng thứ cần bắt — nếu `04a` dùng `round` thay `floor`, hoặc
resample khác, đỉnh sẽ lệch. `|Δ| ≤ 8/255` và `corr ≥ 0.99` giữ làm chỉ báo.

Kiểm trên 5 kích thước thật của MIMIC-CXR (`offset % 8` từ `0` tới `6`): dịch chuyển tốt nhất
**luôn** `(0,0)`; `|Δ|` từ `0.000` tới `0.653`.

### Chi phí

| | |
|---|---|
| đọc lại `346,578` ảnh gốc từ Drive | **~2 giờ**, CPU |
| dung lượng | **~17 GB** (`+20%` so với shard hiện hành, `AR` trung vị `1.201`) |
| GPU | **0** |

Ghi vào `cxr_559_fullfov_shards`. `cxr_560_shards` **không bị đụng tới**. Có
`rename-on-complete` nên ngắt giữa chừng không hỏng file nào.

**Lưu ý cho loader về sau:** ảnh trong shard này có **kích thước biến thiên** (cạnh ngắn luôn
`559`, cạnh dài `round(559·AR)`, `AR` từ `1.00` tới `1.60`). Không đọc thẳng vào một tensor cố
định được — phải crop/resize trước khi stack.

---

## §6 · Việc còn lại, theo thứ tự

1. **Chạy `C8`** — `~1.2 phút` CPU. Nếu verdict đổi theo ngưỡng thì `25.4%` không bền và phải
   chốt ngưỡng bằng chẩn đoán vành ngoài **trước** khi dùng con số này biện minh cho bất kỳ arm
   FOV nào.
2. **Chạy `C3` với `RUN_FULLFOV = True`** — `~2 h` CPU, `0` GPU. Kiểm Drive còn `≥ 25 GB` trước.

`C2`/`C0` cần chạy lại một lượt vì `SHARD_LB` đã đổi tên thành `SHARD_FULL` (chỉ là config, vài
giây).

Arm 2-crop **chưa** được dựng — không nằm trong phạm vi lần này.
