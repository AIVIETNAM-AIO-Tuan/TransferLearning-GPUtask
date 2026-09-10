# 08 · Layer diagnostics — kết quả chạy

Trích từ `08_layer_diagnostics.ipynb` sau lần chạy đầu tiên (A100, subsample phân tầng).
Lưu ở đây vì output trong notebook bị xóa khi build lại từ script.

> ⚠️ Subsample **oversample nhãn hiếm** ⇒ prevalence lệch khỏi tự nhiên.
> Chỉ **hình dạng** đường cong có nghĩa; **không** so giá trị tuyệt đối với 0.4336 của H5.
> Một seed, một lần fit probe mỗi block, **không có CI** — áp quy tắc Δ < 0.005 không phát biểu được.

## C3 · Linear probe từng block → `C3_probe_curve_0.png`

| block | macro | head | medium | tail |
|---|---|---|---|---|
| 1 | 0.3070 | 0.4575 | 0.1326 | 0.1256 |
| 2 | 0.3284 | 0.4836 | 0.1510 | 0.1290 |
| 3 | 0.3447 | 0.5060 | 0.1609 | 0.1350 |
| 4 | 0.3609 | 0.5219 | 0.1784 | 0.1456 |
| 5 | 0.3771 | 0.5389 | 0.1957 | 0.1510 |
| 6 | 0.3847 | 0.5447 | 0.2044 | 0.1664 |
| 7 | 0.3876 | 0.5492 | 0.2050 | 0.1689 |
| 8 | 0.3956 | 0.5548 | 0.2166 | 0.1765 |
| 9 | 0.3978 | 0.5544 | 0.2250 | 0.1656 |
| 10 | 0.4000 | 0.5511 | 0.2330 | 0.1778 |
| **11** | **0.4026** | 0.5543 | 0.2339 | 0.1844 |
| 12 | 0.3962 | 0.5499 | 0.2269 | 0.1667 |

**Mức tăng từng bước (macro):** `+.021 +.016 +.016 +.016 +.008 +.003 +.008 +.002 +.002 +.003 −.006`

- **73%** tổng mức tăng hình thành xong ở **block 5**; block 8→11 chỉ thêm **+0.0070**.
- Phát biểu được: block 1→5 (+0.0701), block 5→8 (+0.0185). **Không** phát biểu được: block 10→11 (+0.0026).
- **Tách theo tier:** head **xong hẳn ở block 8** (8→11 = −0.0005); medium **+0.0173**; tail +0.0079.
  ⇒ bốn block cuối chỉ phục vụ nhãn hiếm/tinh vi.
- **Block 12 tụt −0.0064** — block cuối vứt bỏ thông tin phục vụ nhãn (đặc trưng backbone tự giám sát).

## C4 · Attention distance → `C4_attn_distance_4.png`

Mốc: attention **đều tuyệt đối** trên lưới 37×37 = **19.3 patch**.

| block | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dist (patch) | 13.5 | 10.8 | 12.3 | 8.9 | **8.5** | 8.8 | 10.1 | 11.8 | 14.2 | 15.4 | 16.9 | 17.0 |
| % mức đều | 70 | 56 | 64 | 46 | **44** | 45 | 52 | 61 | 74 | 80 | 88 | 88 |
| head cục bộ nhất | 5.26 | 1.59 | 1.48 | 1.49 | 2.42 | 3.52 | 4.53 | 6.15 | 9.26 | 12.06 | 14.70 | 13.59 |
| **số head cục bộ** (<50% mức đều) | 3 | 5 | 4 | 6 | 8 | **9** | 6 | 4 | 1 | **0** | **0** | **0** |

**Chữ U, không phải tăng dần.** Chọn lọc không gian mạnh nhất ở **block 5** (44% mức đều);
tới block 11–12 attention **gần như trộn đều** (88%).

Dòng cuối là bằng chứng dứt khoát nhất: số head còn xử lý cục bộ **đạt đỉnh ở block 6 (9/12)**
rồi sụp `6 → 4 → 1 → 0 → 0 → 0`. **Từ block 10 trở đi không còn một head cục bộ nào.**
⇒ **lõi "đọc ảnh" là block 5–6**, và năng lực xử lý cục bộ tắt hẳn trước khi tới vị trí adapter của 07.

## C5 · Độ đổi representation giữa 2 block liền kề → `C5_block_delta_1.png`

`0.089 0.056 0.086 0.066 0.068 0.069 0.070 0.083 0.093 0.058 0.093` (block 2→12)

Mạnh nhất block 10 (0.0931), yếu nhất block 3 (0.0562). **Biên độ hẹp, không đơn điệu ⇒ tín hiệu yếu**, không dùng để quyết định.

## C6 / C7 · Bản đồ không gian → `C6_pca_rgb_1.png`, `C7_cls_attn_0.png`

Đọc theo hàng của C7 (CLS → patch):

| block | nhìn thấy gì |
|---|---|
| 1 | nhiễu dạng lưới, texture mức thấp |
| 3 | hiện đường viền lồng ngực |
| 5–9 | **bám đúng trung thất, rốn phổi, vòm hoành** — giải phẫu thật |
| 11–12 | co lại thành vài chấm rời rạc; hàng `Fracture` và `No_Finding` **gần như đen hoàn toàn** |

Các chấm sáng đơn lẻ ở góc trong block 9–12 là **attention sink**, không phải giải phẫu.

## Kết luận — ba phép đo hội tụ

| | block 4–7 | block 10–12 |
|---|---|---|
| C3 thông tin nhãn | đang hình thành (73% xong ở b5) | bão hòa; b12 tụt −0.006 |
| C4 attention | chọn lọc nhất (44% mức đều) | gần đều (88%), hết head cục bộ |
| C7 bản đồ | bám giải phẫu | chấm rời rạc / sink / trống |

**Block 4–7 là nơi ảnh thực sự được đọc. Block 10–12 là nơi biểu diễn được đóng gói cho mục tiêu pretrain.**

### C8 · Phán quyết 4 tiêu chí (output của notebook)

| tiêu chí | block 11 | số liệu |
|---|---|---|
| thông tin đầy đủ | **ĐẠT** | ở 100.0% đỉnh |
| còn khả biến | KHÔNG | block 9→11 chỉ thêm **+0.0048** (ngưỡng 0.005) |
| chọn lọc không gian | KHÔNG | attention ở **88%** mức đều, **0/12** head còn cục bộ |
| độ sâu còn lại | KHÔNG | sau adapter cuối còn **0** block self-attention |

**→ 1/4.** Vị trí đo được nếu kiểm lại deep conditioning: **block 5–8**
(chọn lọc không gian nhất → trước bão hòa).

> `+0.0048` nằm **ngay dưới** ngưỡng 0.005 — biên, không phải cú trượt rõ ràng.
> Giữ nguyên 0.005 vì đó là ngưỡng đã dùng nhất quán ở 06/06b/07; đổi ngưỡng sau khi thấy kết quả là dịch cột gôn.

### Với vị trí adapter của 07 (block 11–12)

Đó là nơi biểu diễn **đầy đủ nhất** nhưng **ít khả biến nhất**, attention gần như đều,
và **không còn độ sâu phía sau** (`adapter[11]` không có self-attention nào sau nó ⇒ về chức năng là late fusion — xem `NOTE_07_adapter_placement.md`).
Tức screen 07 đã kiểm giả thuyết ở **cấu hình bất lợi nhất của nó**.

**GIỚI HẠN — điều này KHÔNG lật null của screen 07.** Phép stacking ở 06b không dùng adapter,
không dùng độ sâu nào, và vẫn null (−0.0027). Phát biểu đúng là "screen đã kiểm ở cấu hình bất lợi nhất",
**không phải** "null là do đặt sai chỗ".

Nhưng nó cho một lời giải cơ học cho bất đối xứng ở 06b:

| 06b | Δ mAP | vì sao |
|---|---|---|
| H2 — EHR vào **CLS** | **+0.0081** (shuffle xác nhận +0.0134) | CLS là pooling toàn cục |
| H6 — EHR vào **patch** | **+0.0003** (p = 0.830) | cuối mạng CLS gần như không đọc patch nữa (C7) |

### Hướng còn lại, không dính EHR

Block 12 tụt −0.0064 so với block 11, mà H5 đang cho ML-Decoder ăn **đúng đầu ra block 12**.

⚠️ **Đây là giả thuyết, chưa phải kết quả.** +0.0064 đo bằng *linear probe* trên `[CLS ‖ mean(patch)]`,
trên subsample lệch prevalence, **1 seed không CI** — không so trực tiếp được với +0.0126 của EHR
(ML-Decoder đầy đủ, val thật, 3 seed ghép cặp, p<0.005). Rất có thể không chuyển sang được.

Lý do đáng ưu tiên là **chi phí** (~1 giờ trích lại + train head), không phải độ lớn.

---

## Phụ lục · Kiểm hướng ảnh → `extra_orientation_0.png`

### B.1 — Tỉ lệ khung hình (KHÔNG trả lời được câu hỏi về hướng giải phẫu)

- `landscape (Columns>Rows)`: frontal **35.8%**, lateral **1.7%** — chênh 21×.
- Chữ in trên chính các phim khổ ngang: `PORTABLE`, `SUPINE`, `AP SEMI UPRT`, `SEMI-ERECT`.
- ⇒ khổ ngang = **chụp tại giường bằng máy di động**, file hợp lệ.

> ⚠️ **Sai lầm phương pháp đã mắc:** tỉ lệ khung hình đo **cái khung**, không đo **giải phẫu bên trong**.
> Có bốn tổ hợp (file dọc/ngang × giải phẫu đứng/ngang); phép đo này chỉ tách được theo cột "file".
> Kết luận "bác bỏ nghi vấn xoay" rút ra từ đây là **overreach** — tấm `Fracture` (`2140×1760`, **file dọc**)
> có giải phẫu **nằm ngang**, đúng ô mà phép đo không thể thấy.

### B.2 — Hướng giải phẫu, đo trên pixel (kết luận đúng)

Score = `boost(profile theo x) − boost(profile theo y)`; `boost` = trung bình 1/3 giữa − trung bình 2/3 ngoài.
Ngực dựng đứng có trung thất sáng chạy dọc giữa ⇒ score > 0.

**Neo kiểm chứng:** tấm `Fracture` (đã xác nhận bằng mắt là nằm ngang) → score **−71.4**, **percentile 0.3%**. Detector hợp lệ.

Phân bố (1500 frontal): p1 **−30.9** · p50 **+54.5** · p99 **+107.4** · `score<0`: **2.40%**

Nhìn 16 tấm score thấp nhất (đáy 1.07%) — **ba** nhóm, không phải hai:

| nhóm | ~tỉ lệ trong đáy | |
|---|---|---|
| **thật sự nằm ngang** | ~4–5 / 16 | score −110.0, −71.7, −57.1, −35.0 |
| phổi trắng xóa, dựng đứng | ~6 / 16 | tràn dịch lớn / đông đặc lan tỏa |
| dải đen lớn (collimation/padding) | ~5–6 / 16 | băng đen chiếm 15–25% khung |

### B.3 — Metadata không có ground truth về hướng pixel

Schema `mimic_cxr_jpg.metadata` chỉ có 13 cột. **Không có** `PatientOrientation` (tag DICOM 0020,0020).
`PatientOrientationCodeSequence_CodeMeaning` là thứ khác và rỗng ở ca đã xác nhận. ⇒ chỉ còn cách nhìn pixel.

**Giả thuyết "decubitus" bị loại bỏ.** Đếm mọi tư thế trông-giống-ảnh-xoay:
`PA LLD 4 · AP LLD 2 · AP RLD 2 · PA RLD 1 · XTABLE LATERAL 2 · RAO 3 · LAO 3 · LPO 1` =
**~18 / 377.000 = 0.005%**. Ảnh nằm ngang trong dataset này **không phải tư thế chụp hợp lệ** — chúng là bất thường thật.
(Ở bản trước tôi từng nêu decubitus như lời giải lành tính; dữ liệu bác bỏ, và điều đó làm mối lo **mạnh lên** chứ không yếu đi.)

### B.4 — Tỉ lệ xoay: **~0.2%** (chấm điểm toàn bộ, không ngoại suy)

Chấm `orient_score` cho **toàn bộ 234.586 ảnh frontal** đã stage (519 giây CPU).
Score lưu ở `layer_diag/orient_scores_frontal.parquet`.

| ngưỡng | số ảnh | % |
|---|---|---|
| sc < −70 | 420 | 0.179% |
| sc < −50 | 926 | 0.395% |
| **sc < −30** | **1.872** | **0.798%** |
| sc < 0 | 4.812 | 2.050% |

**Ước lượng phân tầng toàn dải** (25 ảnh/band, trọng số theo dân số thật) — thay cho phép
điều-kiện-hóa-trên-ngưỡng cũ, vốn giả định mọi ảnh xoay đều có `sc<−30`:

| band | trọng số | đếm nằm ngang | tỉ lệ |
|---|---|---|---|
| `[−∞, −50]` | 0.39% | ~11/25 | 44% |
| `[−50, −30]` | 0.40% | ~7/25 | 28% |
| `[−30, −15]` | 0.44% | ~2/25 | 8% |
| `[−15, 0]` | 0.81% | ~1/25 | 4% |
| `[0, 30]` | **11.9%** | **0/25** | **0%** ← phép thử phủ định |
| `[30, ∞]` | **86.1%** | **0/25** | **0%** ← phép thử phủ định |

```
tỉ lệ xoay = Σ(trọng số × tỉ lệ) ≈ 0.35%   [~0.26–0.48%]   ≈ 820 ảnh
```

**PHÉP THỬ PHỦ ĐỊNH — cốt lõi của độ tin cậy.** Hai band dương (98% dữ liệu) đếm được **0/50**
nằm ngang, đúng như cấu trúc `orient_score` dự đoán (score = năng lượng dải-dọc − dải-ngang ⇒
ngực xoay 90° buộc phải score âm mạnh; ảnh score `+40` không thể là ngực nằm ngang).
⇒ ước lượng **không bỏ sót ảnh xoay ẩn ở score cao** — lỗi điều-kiện-hóa-trên-ngưỡng ở các bản
trước (đáy-16, rồi "25% đồng đều") **được vá và kiểm chứng**.

*Caveat:* số đếm 4 band âm là chấm mắt trên thumbnail (band `[−∞,−50]` dao động 8–15), nên điểm
ước lượng có sai số ±0.1%. Nhưng band dương — nơi quyết định độ tin cậy — được kiểm trực tiếp,
và `P(sc<t)` là điều tra toàn bộ, không lấy mẫu.

**VERDICT: không sửa** — giờ có kiểm chứng. Cận trên ~0.5% vẫn cho tác động macro mAP **< 0.002**,
và **như nhau trên mọi nhánh**. Chi phí sửa là chạy lại `04a → 04b → 05 → 06 → 06b → 07`.

> Con số dịch từ 0.2% (bản trước) lên ~0.35% vì phân tầng **cân đúng trọng số** hai band cực âm
> (tỉ lệ xoay 40–50%), thứ mà cách "25% đồng đều trên sc<−30" đã làm nhòe. Cùng order, cùng verdict.

### B.5 — Chữ ký metadata: có thật, nhưng không dùng làm bộ lọc được

Trên toàn bộ dữ liệu frontal (baseline `sc<−30` = 0.798%):

| trường | n | k | rate | **lift** | **recall** |
|---|---|---|---|---|---|
| `ViewCodeSequence_CodeMeaning IS NULL` | 4.627 | 563 | 12.2% | **15.3** | **30.1%** |
| `ViewPosition` blank | 4.020 | 203 | 5.1% | 6.3 | 10.8% |
| `antero-posterior` | 138.893 | 1.014 | 0.7% | 0.92 | 54.2% |
| `postero-anterior` | 90.816 | 293 | 0.3% | 0.40 | 15.7% |

`view_code IS NULL` là chữ ký thật và ổn định (lift 15.3 toàn bộ vs 13.5 trên mẫu 4.000).
**Nhưng không dùng được:** recall chỉ **30%**, và precision cho "thật sự xoay" = 12.2% × 25% ≈ **3%**.
Gắn cờ 4.627 ảnh để bắt ~140 ca. 97% thứ bị gắn cờ là **phổi trắng xóa** ⇒ dùng làm bộ lọc sẽ
loại ưu tiên bệnh nhân nặng.

> **Ghi lỗi phương pháp:** ở một lượt trung gian, phép đo `vp=BLANK` cho `n=73, k=1` (lift 1.6) và tôi đã
> lấy nó làm bằng chứng **bác bỏ** phép đo trước đó (n=300, lift 8.1), rồi dựng hẳn giả thuyết nhân quân
> "lateral lẫn vào làm nhiễm bẩn". Toàn bộ số liệu đầy đủ (n=4.020, lift **6.33**) cho thấy phép đo đầu
> vốn xấp xỉ đúng, và ô `k=1` kia đơn giản là **nhiễu, không mang thông tin**. Bài học: `k=1` với
> baseline 0.8% không bao giờ là một phép đo.

> 🚫 **Không bao giờ dùng score này làm bộ lọc.** Score thấp có ít nhất hai nguyên nhân không tách được:
> giải phẫu nằm ngang **và** phổi trắng xóa. Lọc theo nó sẽ **xóa ưu tiên bệnh nhân nặng nhất**,
> biến một sai lệch nhỏ về hướng ảnh thành thiên lệch chọn mẫu nghiêm trọng.
> Chỉ dùng để tìm ứng viên đem đi nhìn bằng mắt.

**Nhưng phát hiện được một thứ khác, chắc chắn:** 04a scale theo cạnh ngắn rồi **center-crop thành vuông**
⇒ **mất trung bình 15.0% trục dài** (p50 16.8%, p99 25.2%).
Phim dọc mất đỉnh phổi + góc sườn hoành; phim ngang mất thành ngực hai bên.
Đúng luật của RAD-DINO nên không phải bug, và nhất quán ở mọi thí nghiệm nên không lật kết luận nào —
nhưng là một **trần hiệu năng đã đo được**.

**Hệ quả phụ đáng ghi nhận:** khổ ngang ≈ chụp tại giường ≈ bệnh nhân nặng hơn.
Tỉ lệ khung hình là một proxy cho mức độ nặng, tương quan với `Support Devices` / `Edema` / `Pneumonia`.
Không phải leakage (có sẵn lúc suy luận), nhưng một phần hiệu năng đến từ *bối cảnh chụp* chứ không từ *bệnh lý trên phim*.
