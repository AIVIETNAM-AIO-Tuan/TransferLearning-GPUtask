# REPORT · `nb35` — Stage 4, sàng lọc trên embedding đông cứng

Arm chuẩn `nosq518 / patch` · val 39,293 ảnh / 23,800 study · lát chọn 39,086 ảnh cắt từ train theo
`subject_id` · RAD-DINO **đông cứng**, không trích lại embedding.

Mọi lựa chọn siêu tham số làm trên **lát cắt 15% của train**. Val chỉ để báo cáo. **Test không bị chạm.**

---

## 0 · Phát hiện lớn nhất là về GIAO THỨC, không phải về mô hình

Hai quy tắc trong kế hoạch ban đầu đều **sai**, và cả hai đã gây kết luận sai trước khi bị bắt.

### 0.1 Không early-stop trong một lần chạy — lấy mô hình ở EPOCH CUỐI

`CosineAnnealingLR` trải đúng trên ngân sách `E`, nên **LR tại epoch được chọn phụ thuộc `E`**:

```
E      ep* chon theo tail    LR tai ep*     val mAP anh
150          139             4.0e-05          0.2081
300          191             8.8e-04          0.2067
600          190             2.3e-03          0.2027
```

Ba con số LR đơn điệu, mAP đơn điệu ngược lại, khớp hoàn hảo. Ngân sách lớn hơn **không** làm mô hình
tệ đi — nó làm **checkpoint ta chọn ít được ủ đi**. Việc ủ LR về 0 là một phần của cái làm nên mô hình
tốt, không phải phụ kiện của lịch.

Bỏ hẳn việc chọn epoch:

```
                       mAP anh    bien do seed (3 seed)
chon epoch theo tail    0.2067        0.0058 .. 0.0072
epoch cuoi              0.2093        0.0003 .. 0.0009
```

**Early-stop mất `~0.003` mAP VÀ thổi phương sai seed lên 8 lần.** Bỏ nó là thắng ròng ở cả hai mặt.

Ca bệnh điển hình: `asl40 @E=300 seed 43` chọn `ep 30/300` (10% ngân sách, LR `2.9e-3` ≈ LR ban đầu)
→ val `0.1914`, thấp nhất mọi lần chạy. Và `ep 34/60` cho **cả bốn arm** ở seed 44 — chính nó tạo ra
biên độ `0.006` che lấp mọi hiệu ứng đang đo.

⇒ `E` là **siêu tham số duy nhất**, quét trên lát cắt bằng **macro**, lấy `E` **nhỏ nhất còn trong
nhiễu của đỉnh**.

### 0.2 Ngưỡng nhiễu phải lấy từ biên độ seed của CHÍNH chỉ số đó

`0.005` suy ra từ khoảng cách hạng 2–4 của giải — đó là **macro mAP mức ảnh** (~0.20). Tail ở thang
~0.06 nên `0.005` là **12% tương đối**, chặt tới mức gạt bỏ đúng thứ ta đang cải thiện. Đo được:

```
macro (anh)  +-0.0007      tail  +-0.0011 .. 0.0023      study  +-0.0007
```

---

## 1 · A1 · Loss — `asl40`, `E=60`

Ở **ngân sách tối ưu của chính mỗi loss** (cố định `E` là thiên vị loss nào hội tụ ở tốc độ đó):

| loss | `E` | mAP ảnh | tail | `sel_macro` trên lát cắt |
|---|---|---|---|---|
| **`asl40`** (γ_neg=4, γ_pos=0) | **60** | **0.2091** | **0.0619** | phẳng từ `E=30` (0.1970 → 0.1983) |
| `bce` | ≥150 | 0.2082 | 0.0600 | **vẫn đang lên** (0.1807 → 0.1970) |
| `bce_pw05` (α=0.5) | 60 | 0.2071 | 0.0582 | — |
| `bce_pw10` (α=1.0) | 150 | 0.2018 | 0.0480 | — |

`asl40@60` `{0.2093, 0.2090}` vs `bce@150` `{0.2081, 0.2082}` — **hai khoảng seed rời nhau**. Tail
`{0.0614, 0.0624}` vs `{0.0600, 0.0599}`, cũng rời nhau. Thắng cả ba trục: macro, tail, và compute
(`bce` chưa hội tụ ở `E=150` ⇒ ≥2.5× đắt hơn).

**Ba con số đã bị bác trong quá trình:**

| ghi ban đầu | thực tế |
|---|---|
| `ASL − BCE = +0.0072` | đo BCE ở `E=60`, **chưa hội tụ**. Ở ngân sách riêng của mỗi loss chỉ còn `+0.0009` |
| `asl40` hơn các biến thể ASL khác | chênh `0.0022`, dưới biên độ seed — **không phân định được** |
| `WASL α=.50` là một arm riêng | `clip [0.5, 3.0]` dưới mất cân bằng **673×** kẹp **15/40 nhãn ở sàn** ⇒ **thoái hoá về ASL** (`0.2079` vs `asl41 0.2078`). Chưa hề được thử đúng nghĩa |

`pos_weight = ((N−n)/n)^α` là bản **không bị kẹp** của cùng ý tưởng. `α=0.5` (1.41…44.9) bắt kịp tốc
độ hội tụ nhưng không bắt kịp tail; `α=1.0` (2.0…2015) cân bằng quá tay — 109 mẫu dương chi phối
gradient, tail tệ nhất bảng.

---

## 2 · Mốc mới

```
                             anh      study     tail    cach vo dich (tail)
32 (BCE @30 epoch)         0.2028    0.2146      —
35 asl40 + epoch cuoi      0.2093    0.2229    0.0620       2.19x
   + EHR (Q_CTX)           0.2111    0.2243    0.0610
vo dich Task 1             0.281       —       0.136
```

`+0.0083` mAP ảnh và `+0.0097` study so với `32` — **không đổi một dòng kiến trúc nào**. Khoảng cách
macro `1.39× → 1.33×`; tail `3.30× → 2.19×`.

---

## 3 · EHR — nhỏ, thật, và toàn bộ đến từ BỐI CẢNH

3 seed, `asl40 @E=60`, epoch cuối:

| tập | cột | mAP ảnh | Δ | tail | Δ |
|---|---|---|---|---|---|
| CXR-only | 768 | 0.2093 | — | 0.0620 | — |
| `Q_FULL` nan→0 | +247 | 0.2113 | **+0.0020** | 0.0584 | **−0.0035** |
| `Q_FULL` nan→trung vị | +247 | 0.2110 | +0.0017 | 0.0575 | −0.0044 |
| **`Q_CTX`** (không giá trị lâm sàng) | +157 | 0.2111 | +0.0018 | **0.0610** | −0.0009 |

Lift macro `+0.0018…+0.0020` là **3× biên độ seed** — thật. Nhưng ba tập **không phân biệt được với
nhau** (`0.2110 / 0.2111 / 0.2113`, chênh trong `0.0003`).

> **90 cột giá trị lâm sàng đóng góp đúng bằng 0 cho macro, và `−0.003` cho tail.**
> Toàn bộ đóng góp của EHR đến từ **bối cảnh nhập viện + missingness + số lần đo + độ mới**.

Hằng số điền (0 hay trung vị) không quan trọng.

---

## 4 · Định tuyến theo bệnh (B2c) — trần đo được trước khi dựng

Nếu mỗi nhãn tự chọn đọc EHR hay không, trần là `mean_j max(AP_cxr_j, AP_ehr_j)`. Oracle chọn trên
val nên phải trừ thiên lệch, đo bằng **hai cặp seed của cùng một arm** (chênh lệch thật = 0):

```
thien lech thuan cua phep max: +0.0006

                oracle   - thien lech   - fusion toan cuc = DU DIA THAT
Q_FULL         +0.0045      +0.0039         +0.0020          +0.0019
Q_CTX          +0.0034      +0.0028         +0.0020          +0.0008
```

Tách theo tier (`Q_FULL`):

| tier | fusion toàn cục | oracle | **dư địa của định tuyến** |
|---|---|---|---|
| head (13) | +0.0065 | +0.0065 | **0.0000** |
| medium (16) | +0.0019 | +0.0047 | **+0.0028** |
| tail (11) | **−0.0032** | +0.0018 | **+0.0050** |

Ở **head** dư địa bằng đúng không — EHR giúp 12/13 nhãn nên fusion toàn cục đã vét sạch. **Toàn bộ giá
trị của disease-aware nằm ở medium và tail**, đúng chỗ fusion toàn cục đang lỗ. Ở tail nó đổi `−0.0032`
thành `+0.0018`, xoay chiều `+0.0050`.

**Hệ quả lật ngược lựa chọn tập đặc trưng:** `Q_FULL` vừa giúp mạnh hơn ở head vừa hại mạnh hơn ở tail
— biên độ lớn hơn về cả hai phía. Không định tuyến thì đó là gánh nặng; **có** định tuyến thì đó là tài
sản (trần cao hơn `+0.0010`).

> **Không định tuyến → `Q_CTX`. Có định tuyến → `Q_FULL`.** Cả hai vào `37` làm **arm quét**.

Danh sách nhãn nhất quán về lâm sàng: EHR nhấc **chấn thương** (Rib Fracture `+0.0220`, Pneumothorax
`+0.0205`, Fracture `+0.0150`, Hydropneumothorax `+0.0157`) và **nhiễm trùng/dịch** (Pneumonia
`+0.0186`, Tuberculosis `+0.0181`, Edema, Infiltration); dìm **tổn thương mạn/cấu trúc** (Kyphosis
`−0.0118`, Emphysema `−0.0110`, Fibrosis `−0.0066`, Pneumomediastinum `−0.0087`) và những nhãn ảnh vốn
đã đủ (Subcutaneous Emphysema `−0.0133` trên nền `0.5236`).

---

## 5 · B1 · Đầu đọc phi tuyến — **null sạch**

`MLP(D → 1024 → GELU → Dropout .2 → 40)` so với `Linear(D → 40)`, cùng nhánh dữ liệu:

```
khong_ehr  -0.0105    zero  -0.0112    median_mask  -0.0084    mask_only  -0.0110
```

Phi tuyến **điểm-theo-điểm** trên đặc trưng **đã global-pool** không mua được gì.

Điều này **không** bác cross-attention ở `37` — đó là **chọn lọc theo không gian** trên 1369 token,
một lớp hàm khác hẳn. Nó chỉ loại được giả thuyết *"đầu đọc tuyến tính là nút thắt"*.

---

## 6 · A5.1 · Mốc tương quan cho `41a`

9 file `p_*.npy` của `32` (3 arm × `cls`/`patch`/`both`), không tốn một giây GPU:

```
khac ARM, cung bieu dien   spearman TB 0.9060  (n=9)   <- DA DANG IT NHAT
khac BIEU DIEN, cung arm   spearman TB 0.5925  (n=9)
toan bo 36 cap                          0.6577  | min 0.3063 | max 0.9708
```

**Ensemble trọng số đều là ÂM ở mọi nhóm:**

```
3 arm x patch     0.2009  vs tot nhat 0.2028   -0.0020
cls + patch       0.1908  vs           0.2028   -0.0120
tat ca 9          0.1939  vs           0.2028   -0.0090
```

Trung bình đều nhau các thành viên **không đều nhau** thì thua. `cls` (`0.1651`) tuy đa dạng nhất
(spearman `0.59`) nhưng yếu hơn `0.0377` và phá mất `0.0120`.

> **Với `41a`, ràng buộc thật là NGANG CHẤT LƯỢNG, không phải đa dạng.** Cổng `CNN ≥ 0.85 ×
> RAD-DINO` (`≈ 0.172`) trong kế hoạch là **quá lỏng**. Kết quả này chỉ bác trung bình trọng số
> **đều**; trọng số học được (chọn trên lát cắt train) chưa ai thử.

---

## 7 · Độ phân giải — chưa ngã ngũ

```
crop518   0.626 mm/px   FOV 71.5%   0.1917
lb518     0.812 mm/px   FOV 100%    0.2002
nosq518   0.812 mm/px   FOV 100%    0.2028
```

`crop518` phân giải cao hơn **1.30×** mà thua `−0.0085` — nhưng mất 28.5% FOV nên **hai biến lẫn nhau**.

Phép thử phân nhóm (tiêu điểm nhỏ vs lan toả) **không kết luận được**, và cách chia nhóm còn **nghiêng
chống lại giả thuyết**: nhóm "tiêu điểm nhỏ" (xương đòn, xương sườn, thành ngực, bờ trung thất) phần
lớn là **ngoại vi**, tức đúng phần mà crop trung tâm cắt đi.

Nhưng ở **13/40 nhãn mà crop không lấy mất tổn thương**, `+1.30×` phân giải cho `+0.006 … +0.010`
(`Atelectasis +0.0104 · Hernia +0.0096 · Pneumothorax +0.0091 · Cardiomegaly +0.0086`).

Cách tách sạch: thêm arm **`lb672`** (`48 × 14 = 672`, `0.812 × 518/672 = 0.626 mm/px` — **đúng bằng
độ phân giải của `crop518`** nhưng FOV 100%). Hoàn thiện ô 2×2, và `lb672 − crop518` phải ra `≈ +0.0111`
(hiệu ứng FOV đã đo) làm **cổng tự kiểm**.

```
                token   GPU so 518   img/s (uoc, tu 04b)   377,110 anh
lb518   0.812mm  1,369     1.0x            ~450              ~14 phut
lb672   0.626mm  2,304     2.2x            ~205              ~31 phut
lb1022  0.406mm  5,329     6.5x             ~69              ~91 phut
```

Dung lượng **không đổi** ở mọi độ phân giải — chỉ lưu `cls` và `mean(patch)`, đều là `[N, 768]`.
`lb1022` **không dùng được cho `37`**: backbone recompute mỗi epoch tăng 6.5× (`20 ph → ~2.2 h`).

---

## 8 · Những gì kết quả này KHÔNG loại trừ

**Đầu đọc.** Mọi số ở đây là probe trên embedding đông cứng **đã global-pool**; `patch` là **trung
bình** 1369 token. ML-Decoder đọc đủ token (`37`) là lớp hàm khác hẳn.

**Cơ chế fusion.** Chỉ **concat** được thử. `Q=patch, K/V=EHR` (`37`) và adapter trong block (`39`)
chưa ai đo trên cohort này.

**Backbone.** RAD-DINO đông cứng, chưa fine-tune (`38`).

**Đích.** Nhãn CXR-LT do NLP trích từ **chính report của phim đó** ⇒ ảnh gần như đủ theo định nghĩa.
Không con số nào ở đây nói gì về tiên lượng, tử vong, chuyển ICU, tái nhập viện.

**Case C.** Câu *"EHR chỉ đang vá bottleneck của pooling?"* **chưa được trả lời**. Bản đo qua `cls`
bị bỏ vì `cls` không xuất hiện ở bất kỳ kiến trúc nào từ `37` trở đi; câu hỏi chuyển thành ô 2×2
`{mean-pool, ML-Decoder} × {có, không EHR}` trong `37`, dùng phép gộp thật thay vì biến thay thế.
