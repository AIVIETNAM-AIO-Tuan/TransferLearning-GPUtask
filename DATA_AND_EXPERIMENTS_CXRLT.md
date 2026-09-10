# CXR-LT 2024 — Hồ sơ dữ liệu và thực nghiệm

> Tài liệu tra cứu: **dữ liệu hiện có gì, ở trạng thái nào, đã chạy thí nghiệm gì, ra số bao nhiêu.**
> Cập nhật: 2026-08-05. Phạm vi: nhánh CXR-LT (`28`–`34`), **không** bao gồm cohort cũ `cohort_v2`.
> Trạng thái sống theo giai đoạn nằm ở `REVIEW_CXRLT_STAGE0.md`; kế hoạch tiếp theo ở
> `PLAN_STAGE4_IMAGE_THEN_EHR.md`.

---

## 0 · Tóm tắt một trang

```
DU LIEU     377,110 anh / 227,835 study / 65,379 benh nhan
            split chinh thuc: train 258,871 | val 39,293 | test 78,946
            40 nhan task-1 (+26 task-2 la tap con, +5 task-3 zero-shot)
            do trai GIUA cac nhan 667x  |  mat can bang TRONG tung nhan  2:1 -> 2006:1
            prevalence TB tren val = 6.17%   <- SAN ngau nhien cua mAP

ANH         375,798 / 377,110 dung duoc (99.65%) — 1,312 bo vi khong xac dinh duoc view
            3 arm da trich embedding: crop518 · lb518 · nosq518   (moi arm cls + patch)

EHR         227,835 study | inpatient 48.67% · ed 34.15% · outpatient 17.18%
            5.2% study KHONG co ban ghi MIMIC-IV nao (khong age/gender)
            3 tang dac trung long nhau: Q_CTX  ⊂  Q_CLIN  ⊂  Q_FULL

MOC GOC     CXR-only  0.2028  |  EHR-only 0.1199  |  EHR+CXR 0.2068     (mAP anh, val)
            head 0.4204 · medium 0.1369 · tail 0.0417
            vo dich CXR-LT Task 1: 0.281 (head 0.567 / medium 0.263 / tail 0.136)
```

**Ba phát hiện chính đến hiện tại**

1. **`patch` thắng `cls` trên 40/40 nhãn** (`+0.0377`) — trục biểu diễn là đòn bẩy lớn nhất đã đo.
2. **Đệm đen không phải thủ phạm** (`+0.0026`, dưới nhiễu) — giả thuyết bị đóng bằng 25 phút GPU.
3. **Nhãn tự động gọi thừa rất nặng** — trên 406 ảnh chuyên gia đọc lại, `Pneumonia` từ 115 xuống
   22 ca, **0 lần thêm**. Trần do nhiễu nhãn là `0.653`; vô địch đạt **81%** của mức đó.

---

## 1 · Dataset

### 1.1 Kích thước và split

```
labels.csv   377,110 anh / 227,835 study / 65,379 benh nhan  x 52 cot (45 nhan + 7 ID)
split        train 258,871 | val 39,293 | test 78,946     <- CHINH THUC, dung thang
nhan         int64 {0,1} TOAN BO — khong -1, khong o rong, tren CA BA split
```

| task | số nhãn | quan hệ | file huấn luyện phát hành |
|---|---|---|---|
| task 1 | **40** | mục tiêu chính | `train_labeled.csv` — **đúng 40 cột này** |
| task 2 | **26** | **tập con thật sự** của task 1 | không có file train riêng |
| task 3 | **5** | zero-shot — **rời hoàn toàn** khỏi task 1 | **không phát hành** |

#### `45` là số cột của FILE CHỦ, không phải số nhãn của bài toán

```
task1 ∩ task3 = {}          -> hai tap ROI NHAU hoan toan
task1 ∪ task3 = 45          -> == so cot nhan cua labels.csv
task2 ⊂ task1               -> khong dong gop cot MOI nao
train_labeled == task1      -> dung 40
```

`labels.csv` là **bảng chủ chứa nhãn của cả ba task**. Task 2 dùng 26 cột đã nằm trong 40 của
task 1, nên chỉ task 1 và task 3 mới góp cột.

#### ⚠️ Bẫy zero-shot — `labels.csv` CÓ nhãn task-3 trên hàng train

```
so ca DUONG trong labels.csv    Bulla  Cardiomyopathy   Hilum  Osteopenia  Scoliosis
train                             463             966  68,700         333      7,455
val                               174             130  10,163          48        971
test                              150             260  20,897         120      2,526
```

Chính việc `train_labeled.csv` **chỉ phát hành 40 cột** là cơ sở của zero-shot. Lấy nhãn từ
`labels.csv` bằng `[c for c in df.columns if c not in ID_COLS]` sẽ vớ đủ **45** và **phá vỡ tiền đề
của Task 3**.

**Trạng thái của ta:** mọi notebook đọc `INV['labels_task1']` (40) ⇒ an toàn. Nhưng
`cxrlt_index.parquet` **giữ cả 45 cột** (Stage 0 cố ý giữ 5 cột kia "để dành"), nên rủi ro vẫn
sống — luôn chọn cột **theo tên từ inventory**, đừng suy ra từ `df.columns`.

**Và task 3 KHÔNG phải "5 nhãn hiếm".** `Hilum` có **68,700** ca dương train — nhiều thứ hai toàn
bộ dataset, chỉ sau `Support Devices` (86,079). Còn `Osteopenia` (333 train / 48 val) thuộc nhóm
hiếm nhất. Tên "zero-shot" nói về **không có nhãn huấn luyện**, không nói về độ hiếm.

### 1.2 Ba cổng toàn vẹn — đều đạt (`28` C4)

| cổng | kết quả |
|---|---|
| bệnh nhân rò rỉ giữa split | **0 / 65,379**; study bị cắt ngang: **0** |
| nhãn hằng số trong cùng study | **0 / 227,835** → làm ở **mức study** hợp lệ |
| `dicom_id` duy nhất | có |

### 1.3 Phân tầng long-tail (theo `n_train`, ngưỡng 1,000 / 10,000)

```
head    13 nhan   10,087 -> 86,079 ca duong
medium  16 nhan    1,169 ->  8,919
tail    11 nhan      129 ->    903
```

#### Hai loại "mất cân bằng" — KHÔNG được gộp làm một

| | định nghĩa | giá trị |
|---|---|---|
| **giữa các nhãn** | nhãn nhiều nhất ÷ nhãn hiếm nhất | **667×** trên train · **506×** trên val |
| **trong từng nhãn** | âm : dương của **chính nhãn đó** | **2 : 1 → 2006 : 1** |

`667×` = `86,079 (Support Devices) / 129 (Lobar Atelectasis)` — nó mô tả **độ trải giữa các nhãn**,
**không** phải đặc tính của từng nhãn. Cái chi phối loss và sàn AP là loại thứ hai:

```
Support Devices     86,079 duong  ->  172,792 : 86,079  =    2.0 : 1   <- gan can bang
Atelectasis         10,131                              =   24.6 : 1
Rib Fracture         1,250                              =  206   : 1
Lobar Atelectasis      129                              = 2006   : 1   <- khac nghiet
```

Bản thân **mức khắc nghiệt** trải qua ~1000 lần. Nói gọn "dataset mất cân bằng 667×" là gộp hai
chuyện khác hẳn nhau.

#### Hệ quả trực tiếp lên cách đọc mAP

**Prevalence trung bình trên val = 6.17%** — 96,941 ca dương trên `40 × 39,293 = 1,571,720` ô.
Đây là **sàn ngẫu nhiên của mAP**.

Nhưng sàn đó **rất không đều giữa các nhãn**: `Support Devices` có sàn **33.49%**, `Clavicle
Fracture` có sàn **0.048%** — chênh nhau **693 lần**. Macro mAP trung bình cộng 40 con số có sàn
lệch nhau như vậy ⇒ **bắt buộc tách theo tier**, và một con số macro đơn lẻ gần như vô nghĩa nếu
không đi kèm sàn.

Đây cũng là lý do **ASL** đáng thử: nó tách riêng cách đối xử với dương và âm, và tham số `γ_neg`
nhắm đúng vào phần âm áp đảo — thứ mà BCE đối xử y hệt phần dương.

### 1.4 Bốn đặc điểm nhãn phải nhớ khi báo cáo

**ID cần chuẩn hoá.** `subject_id` đã là `int64`, nhưng `study_id` là **chuỗi** `'s50414267'`.
`28` C1 cắt tiền tố rồi đối chiếu ngược với `fpath`. Bỏ bước này thì mọi join BigQuery **khớp 0
dòng mà không báo lỗi**.

**20,156 ảnh (5.34%) có vector nhãn toàn 0** — không `Normal`, cũng không finding nào.
`Normal = 0` **không** có nghĩa "có bệnh".

**Nhãn chồng lấn nhưng KHÔNG lồng nhau.** `Rib Fracture → Fracture` chỉ **73.2%**;
`Lobar Atelectasis → Atelectasis` 78.0%; `Round(ed) Atelectasis → Atelectasis` 85.0%. Macro mAP
đếm trùng tín hiệu — phải nói rõ khi báo cáo.

**4 nhãn có <50 ca dương trên val**: `Clavicle Fracture` 19 · `Azygos Lobe` 20 · `Lobar
Atelectasis` 26 · `Round(ed) Atelectasis` 46. AP phương sai rất lớn ⇒ **bắt buộc in `n_pos` cạnh
mọi AP**, và không đọc chênh lệch nhỏ ở nhóm này.

### 1.5 Nhãn được sinh ra thế nào — và vì sao điều đó quyết định

Nhãn CXR-LT do **NLP trích từ chính report của tấm phim đó**. Hệ quả: **ảnh gần như đủ theo định
nghĩa**, và mọi modality thứ hai chỉ còn rất ít đất.

`34` đo được biểu hiện trực tiếp trên 406 ảnh có nhãn chuyên gia:

```
chuyen gia BO 615 nhan   |   THEM 151   ->  ti le 4.1 : 1
Pneumonia     115 -> 22  (bo 93, them 0)     <- NLP goi thua 5x
Infiltration   46 -> 12  (bo 34, them 0)
Lung Lesion    31 ->  6  (bo 26, them 1)
```

---

## 2 · Ảnh — trạng thái hiện tại

### 2.1 Shard trên Drive

| thư mục | số ảnh | tar | dung lượng | nguồn |
|---|---|---|---|---|
| `cxr_560_shards/` | 346,578 | — | ~25–30 GB | `04a` (cohort cũ), crop 559px |
| `cxr_560_shards_cxrlt/` | 30,532 | — | | `28b`, **khớp pixel bit-for-bit** với `04a` |
| `cxr_1022_letterbox_cxrlt/` | **377,110** | **378** | **49.30 GB** | `28c`, letterbox 1022, giữ 100% FOV |

`28c` chạy **3.07 h @ 33.1 img/s**. Ảnh gốc MIMIC-CXR-JPG **đã có sẵn trên Drive** (615 GB) nên
`28b` chỉ là tiền xử lý, không phải tải về.

Toàn bộ 346,578 ảnh của shard cũ **đều nằm trong** CXR-LT (0 ảnh dư) ⇒ tập ảnh `cohort_v2` là **tập
con thật sự** của CXR-LT. Hệ quả: bệnh nhân từng ở TEST cohort cũ nay có thể ở TRAIN của CXR-LT
⇒ **không so được với bất kỳ kết quả R0 nào**.

### 2.2 `view_group` — đã giải xong (`28d` + `28e`)

15,769 ảnh (4.18%) thiếu `ViewPosition`. `ViewCodeSequence_CodeMeaning` **cũng rỗng trên đúng
15,769 ảnh ấy** ⇒ metadata không cứu được. Ảnh `unknown` **đi thành cả study** (11,466 ảnh trong
5,075 study) nên loại-trừ-trong-study gần như vô dụng (cứu được **1** ảnh).

**Kết quả cuối** (`cxrlt_view_resolved.parquet`):

| nguồn | số ảnh |
|---|---|
| `ViewPosition` | 361,341 |
| bộ phân loại `28e` (acc tự tin **0.9967**) | **10,154** |
| metadata BigQuery (`28d`) | 4,302 |
| loại trừ trong study | 1 |
| **bỏ** (không đủ tự tin) | **1,312** |

```
GIU  375,798 / 377,110  (99.65%)      val+test chi mat 7 study (0.004% / 0.012%)
uoc tinh ~34 anh gan sai view (0.33% x 10,154)
```

Bộ phân loại `28e` theo Xue et al.: image profile 400-d + body size ratio 1-d,
**HistGradientBoosting** (tuyến tính chỉ đạt 0.9684 — crop của `04a` xoá mất tỉ lệ khung mà body
size ratio dựa vào). Cache đặc trưng: `cxrlt_view_feat_28e.npz` (~96 MB).

**Phân bố view:** `AP 147,173 · PA 96,161 · LATERAL 82,853 · LL 35,133`.
Trên cohort cũ, **48.3% study có cả hai view** — con số cho CXR-LT **cần xác nhận** (`32` C2 in ra).

### 2.3 Hình học ảnh — đo trên toàn bộ val (`33` C1–C3)

```
dem den (letterbox 1022):  TB 18.88%  | p95 34.69%
phan vi:  5% 0.043 | 25% 0.168 | 50% 0.176 | 75% 0.209 | 95% 0.347
noi dung TB 0.8112   vs   du kien 0.832 tinh doc lap tu 1/AR   -> lech -0.0208, KHOP
huong anh: portrait 74.49% | landscape 25.51%
```

Ba cổng hình học đều đạt (đối xứng · chỉ một trục · khớp 83.2%).

**Bỏ đệm được bao nhiêu** (cạnh dài 1022, cạnh ngắn bội số 14):

```
cach                     dem_tb   patch/anh   chi phi attention
vuong 1022 (hien tai)    14.78%     5329        1.000
5 ro theo phan vi AR      3.03%     4687        0.773
16 ro theo phan vi AR     1.59%     4614        0.750
sap theo AR, batch 64     0.77%     4576        0.737
```

### 2.4 FOV — đo trên CXR-LT

`STORE_SIZE` thật của shard crop là **559 px** (thư mục tên `cxr_560_shards` nhưng **không** phải
560); `MODEL_SIZE` = 518.

**Ba arm khác nhau ở FOV, và hiệu ứng đã đo trực tiếp** (`32` C4, `33` C5):

```
nosq518 - crop518    patch +0.0111   cls +0.0013     <- FOV day du hon crop
lb518   - crop518    patch +0.0085   cls +0.0018
nosq518 - lb518      patch +0.0026   cls -0.0005     <- CO LAP bien DEM: duoi nhieu
```

**FOV thắng độ phân giải ở thang này.** `crop518` có **mm/px tốt hơn** (0.626 vs 0.812) mà vẫn
thua `+0.0111` — nên `lb1022` (0.411 mm/px) **không** phải bước tiếp theo hiển nhiên.

Con số hình học của **chính CXR-LT** nằm ở §2.3 (nội dung 81.12% · đệm 18.88% · portrait 74.49%).

> Phân tích FOV chi tiết ở `19` (mất 28.5% diện tích / 25.4% giải phẫu, giả thuyết xoay 90° đã
> đóng) được đo trên **cohort cũ 346,578 ảnh**, phân bố hướng ảnh khác hẳn CXR-LT (portrait 59.4%
> vs **74.49%**). **Không** chép số đó sang đây; nếu cần thì phải chạy lại `19` trên CXR-LT.

### 2.5 Embedding đã trích (`31`)

```
emb_index.parquet                 thu tu hang, DUNG CHUNG moi arm
                                  sap theo (subject_id, study_id, dicom_id)
emb_<arm>_cls.npy   [377110, 768] float32
emb_<arm>_patch.npy [377110, 768] float32   <- trung binh 1369 patch token
_done_<arm>.json                  danh sach tar da xong (RESUME)
```

| arm | nguồn | model input | FOV | mm/px | đệm | trạng thái |
|---|---|---|---|---|---|---|
| `crop518` | `cxr_560_shards` + `_cxrlt` | 518, CenterCrop | 71.5% | 0.626 | 0% | **đã trích** |
| `lb518` | letterbox 1022 → resize 518 | 518, không crop | 100% | 0.812 | ~18% | **đã trích** |
| `nosq518` | letterbox 1022 → cắt đệm → giữ tỉ lệ | cạnh dài 518, **không vuông** | 100% | 0.812 | **0%** | **đã trích** |
| `lb1022` | letterbox 1022 | 1022, pos-embed nội suy 2× | 100% | 0.411 | ~18% | **chưa chạy** |

**Backbone:** `microsoft/rad-dino` **stock** — patch 14 · huấn luyện ở 518px · hidden 768 · 12 block.
`Dinov2Embeddings.interpolate_pos_encoding` chạy **tự động** cho đầu vào khác 518, và nhận
`height`/`width` **riêng biệt** ⇒ ViT không đòi ảnh vuông (đó là cơ sở của `nosq518`).

`nosq518` dựng bằng cách **dò khung bao vùng khác đen rồi cắt** (shard là 1022×1022 nên tỉ lệ gốc
đã mất), sau đó resize giữ tỉ lệ, cạnh ngắn làm tròn về bội số 14 bằng **`round`** (chấp nhận giãn
≤1.6% một trục để đệm bằng **0 tuyệt đối**).

---

## 3 · EHR — trạng thái hiện tại

### 3.1 `cxrlt_base.parquet` (`29`) — khung định danh + thời gian

```
227,835 study | index_time 2110-01-11 -> 2208-12-08 | truy van quet 0.05 GB / 7s
inpatient 110,892 (48.67%) | ed 77,801 (34.15%) | outpatient 39,142 (17.18%)
co encounter 82.82%  — deu giua ba split (outpatient 17.0-17.6%)
```

| cột | nguồn |
|---|---|
| `study_id`, `subject_id`, `split` | `labels.csv` — split chính thức |
| `index_time` | `MIN(CAST(study_datetime AS TIMESTAMP))` từ `mimic_cxr_jpg.metadata` |
| `cur_hadm`, `cur_edstay` | join `{HOSP}.admissions` / `{ED}.edstays` bao trùm `index_time` |
| `setting` | `inpatient` nếu có `cur_hadm`; `ed` nếu có `cur_edstay`; còn lại `outpatient` |

**Khác `01` ở đúng một chỗ:** **không** áp bộ lọc `has_enc`. Cohort cũ bắt buộc mọi study có
encounter; CXR-LT giữ hết và để `setting='outpatient'` nói lên sự thật.

**Bốn điều phải mang theo:**

`mimiciv_3_1_ed` **KHÔNG tồn tại** — chỉ có `physionet-data.mimiciv_ed`. Cổng khớp-phiên-bản đã
cảnh báo và rơi về đúng lựa chọn duy nhất. `29b` C1-bis kiểm bằng **năm thật** (dựng lại qua
`anchor_year_group`, vì MIMIC-IV dịch ngày **theo từng bệnh nhân**): tỉ lệ `ed` ổn định
**33.7–40.0%** suốt 2009–2017, không sụp ở đầu nào ⇒ `outpatient` **không** bị thổi phồng.

**11,837 study (5.2%) có `subject_id` không nằm trong `hosp.patients`** ⇒ **không age/gender**.
Cohort cũ có 100% vì `01` lọc `in_patients`. Phải xử lý bằng mask + cột chỉ báo thiếu, **không**
điền giá trị trung bình.

**3,871 study khớp CẢ `hadm` LẪN `edstay`** — quy tắc ưu tiên gán `inpatient`, nhưng tại thời điểm
chụp chúng có thể vẫn ở ED.

**Chất lượng join sạch:** 0 study thiếu `index_time` · 0 study khớp >1 đợt nằm viện · 1 study khớp
>1 đợt cấp cứu · 0 lệch `subject_id`.

### 3.2 `cxrlt_ehr.parquet` (`29b`) — bộ đặc trưng

**Ba bất biến giữ tuyệt đối:**

```
BAT BIEN 1 - DINH DANH : moi gia tri join qua subject_id, quy ve dung study_id cua no
BAT BIEN 2 - THOI GIAN : labs/vitals/comorbidity phai thoa  storetime <= index_time
                         (da RESULTED, khong chi duoc chi dinh) — khong mot giot tuong lai
BAT BIEN 3 - Y NGHIA   : moi cua so thoi gian la MOT DAC TRUNG RIENG, khong gop trung binh
                         lab 24h = trang thai CAP ; lab 90d = trang thai MAN
```

**Cấu hình cửa sổ:**

```python
LAB_WINDOWS       = [('24h', 24), ('7d', 24*7), ('90d', 24*90)]   # cua so QUET
LAB_VALUE_WINDOWS = [('24h', 24), ('7d', 24*7)]                   # chi 2 cua so nay cho GIA TRI
MICRO_PAST_D, MICRO_FUTURE_H, VITAL_WINDOW_H, ORG_TOP_K = 7, 72, 24, 30
```

Lab 90d **chỉ cho đếm/recency, không cho giá trị** — theo quyết định "lab ở 90 ngày không có giá
trị lâm sàng".

**Ngoại lệ có căn cứ sinh học — microbiology.** Cửa sổ `[index_time − 7d, index_time + 72h]` tính
trên **giờ lấy bệnh phẩm** (`charttime`), không phải giờ trả kết quả. Vi khuẩn **đã ở trong người
bệnh ngay lúc chụp**; chỉ *phép đo* bị trễ 24–72h vì cần nuôi cấy.

Nhưng **tách bạch hai loại đại lượng**:

| đại lượng | bản chất | cửa sổ tương lai |
|---|---|---|
| **kết quả**: organism mọc ra, dương/âm, kháng sinh đồ | trạng thái **sinh học**, đã tồn tại lúc chụp | **có** — trễ nuôi cấy |
| **hành vi**: "có gửi bệnh phẩm hay không" | **hành động lâm sàng**, có thể do chính tấm phim gây ra | **chặn cứng ở `index_time`** |

Cấy đờm gửi 6h *sau* một phim thấy đông đặc rất có thể được chỉ định **vì** phim đó ⇒ rò rỉ nhãn
qua đường hành vi. Đây là assert quan trọng nhất của cả Stage 1.

**Ba tầng lồng nhau:**

```
Q_CTX   boi canh + missingness + dem do luong + age/gender      (KHONG gia tri lam sang nao)
Q_CLIN  Q_CTX + gia tri labs (24h, 7d) + vitals + comorbidity   (KHONG cot vi sinh nao)
Q_FULL  Q_CLIN + vi sinh t1-t4 + one-hot organism (TOP 30)
```

Khoảng cách `Q_CLIN − Q_CTX` mới là thứ đáng quan tâm: nếu ≈ 0 thì EHR chỉ đang đọc ra *loại bệnh
nhân*, không đọc ra *bệnh*.

> ⚠️ **Cần xác nhận:** số cột cuối cùng. Trong quá trình phát triển nó đi từ `398 → 290` (sau khi
> sửa tên cột thành tiền tố) rồi có chỗ ghi `254`. Chạy `29b` cell cuối để lấy con số thật, rồi
> sửa dòng này.

### 3.3 Độ phủ — bảng quan trọng nhất của nhánh EHR

```
lab 24h co gia tri:   inpatient 91.8%  |  ed 53.9%  |  outpatient 7.6%
```

**MIMIC-IV KHÔNG CÓ bảng vital ngoại trú.** Chỉ có `icu.chartevents` + `ed.vitalsign`/`triage`.
Nên `vital_*_mask` **không** phân biệt "nội trú vs ngoại trú" mà phân biệt **"ICU/ED vs phần còn
lại"**.

**Rủi ro proxy đã được đo, và nhẹ hơn lo ngại.** `vital_*_mask` gần như là biến proxy hoàn hảo cho
`setting`, nhưng `30` cho thấy mức nâng so với prevalence **vẫn giữ trong từng tầng**:

```
inpatient +0.0565  |  ed +0.0579  |  outpatient +0.0484
```

**`outpatient` là HAI quần thể, không phải một:**

```
~27,300 ngoai tru THAT
~11,839 KHONG co ban ghi MIMIC-IV nao  (xac nhan boi co_age = 69.78%)
```

Phải tách `no_ehr` khỏi `outpatient` khi báo cáo — gộp lại là trộn "có dữ liệu nhưng thưa" với
"không có dữ liệu".

### 3.4 Họ đặc trưng và khả năng chịu đựng ngoại trú

| họ | encounter có chặn không | ngoại trú có không |
|---|---|---|
| `age`, `gender` | không — `hosp.patients` | **có**, trừ 5.2% không bản ghi |
| comorbidity | không — neo vào **prior closed admissions** | **có**, nếu từng nhập viện |
| labs | không — `labevents` có hàng `hadm_id` null | **có**, nhưng chỉ 7.6% trong 24h |
| **vitals** | **có** — chỉ ICU + ED | **KHÔNG** |
| microbiology t1–t4 | không chặn cứng | thưa |

---

## 4 · Thí nghiệm đã chạy và kết quả

### 4.1 Bảng notebook

| nb | nội dung | trạng thái |
|---|---|---|
| `28` | kiểm kê, 3 cổng toàn vẹn, chuẩn hoá ID, phân tầng | ✅ |
| `28b` | bù 30,532 ảnh crop còn thiếu | ✅ |
| `28c` | letterbox 1022 toàn bộ 377,110 ảnh | ✅ 3.07 h |
| `28d` | leo thang metadata cho view thiếu | ✅ |
| `28e` | bộ phân loại view theo nội dung ảnh | ✅ |
| `29` | `cxrlt_base.parquet` | ✅ |
| `29b` | bộ đặc trưng EHR, 3 tầng | ✅ |
| `30` | EHR-only, XGBoost mỗi (tier, nhãn) | ✅ |
| `31` | trích embedding, 3 arm | ✅ (`lb1022` chưa) |
| `32` | probe tuyến tính, gộp study, 3 mốc gốc | ✅ |
| `33` | đo viền đen, chấm lại độc lập, cls-vs-patch, lift EHR từng nhãn | ✅ |
| `34` | benchmark Task 2 (26 nhãn), trần nhiễu nhãn | ✅ |

### 4.2 Lưới arm × biểu diễn (`32` C4, val, mAP)

```
muc ANH            cls    patch    both        muc STUDY        cls    patch    both
crop518         0.1638   0.1917  0.1855        crop518       0.1750   0.2023  0.1975
lb518           0.1656   0.2002  0.1974        lb518         0.1769   0.2122  0.2123
nosq518         0.1651   0.2028  0.1974        nosq518       0.1760   0.2146  0.2113

tot nhat: nosq518 / patch  ->  anh 0.2028 | study 0.2146
theo tier: head 0.4204 | medium 0.1369 | tail 0.0417
```

Gộp study lợi `+0.011` đến `+0.015` ở mọi tổ hợp (trọng số `0.7·frontal + 0.3·lateral`).

### 4.3 Ba mốc gốc (`32` C6)

```
CXR-only  0.2028 anh | 0.2146 study
EHR-only  0.1199      | 0.1222
EHR+CXR   0.2068      | 0.2184
  EHR them vao CXR: +0.0040       CXR them vao EHR: +0.0869
```

### 4.4 Trục biểu diễn — `patch` vs `cls` (`33` C6)

```
patch - cls        head              medium            tail
crop518        +0.0222 (13/13)   +0.0416 (15/16)   +0.0145 ( 7/11)
lb518          +0.0173 (12/13)   +0.0493 (16/16)   +0.0333 (11/11)
nosq518        +0.0229 (13/13)   +0.0518 (16/16)   +0.0349 (11/11)
```

Với `nosq518`, `patch` thắng `cls` trên **40/40 nhãn** (thấp nhất `Infarction +0.0002`).
Nhãn hưởng lợi mạnh nhất đều là bệnh **khu trú, ngoại vi**:

```
Subcutaneous Emphysema  +0.2307    Pneumoperitoneum   +0.1086
Hernia                  +0.1276    Kyphosis           +0.0860
                                   Hydropneumothorax  +0.0844
```

Cơ chế khả dĩ: CLS gộp toàn cục làm loãng ổ nhỏ; patch-mean giữ lại.

### 4.5 Trục FOV / đệm (`33` C5, `32` C5-bis)

```
lb518 - crop518        cls +0.0018  |  patch +0.0085
nosq518 - crop518      cls +0.0013  |  patch +0.0111
nosq518 - lb518        cls -0.0005  |  patch +0.0026   <- CO LAP BIEN DEM
```

**Đệm không phải thủ phạm.** `nosq518` và `lb518` cùng 518px, cùng FOV 100%, chỉ khác 18% đệm → 0%;
chênh lệch lớn nhất `+0.0026`, dưới sàn nhiễu 0.005.

Phân tầng val theo tứ phân vị đệm (`33` C5) cũng phẳng:

```
cls   delta theo tang: +0.0012 +0.0018 +0.0001 +0.0023 | do doc +0.0011 | z +0.88
patch delta theo tang: +0.0065 +0.0091 +0.0089 +0.0115 | do doc +0.0050 | z +1.44
```

`z` đo trên **sàn nhiễu hoán vị** (20 lần gán tầng ngẫu nhiên), không phải ngưỡng đoán.

**Lưu ý:** `crop518` có **độ phân giải tốt hơn** (0.626 vs 0.812 mm/px) mà **vẫn thua** — ở thang
này **FOV thắng độ phân giải**, và điều đó hạ hẳn kỳ vọng vào `lb1022`.

### 4.6 EHR theo từng nhãn (`33` C7)

```
tier      n  lift_tb  n_duong
head     13  +0.0068       12
medium   16  +0.0048       10
tail     11  -0.0005        7
macro +0.0040 | 29/40 nhan duong | 11/40 vuot nguong nhieu 0.005
```

**11 nhãn vượt ngưỡng** — và nhóm này **ổn định khi đổi backbone** (từ nền `lb518/patch` sang
`nosq518/patch`):

```
Tuberculosis  +0.0266    Rib Fracture  +0.0172    Lung Opacity  +0.0098
Pneumothorax  +0.0210    Lung Lesion   +0.0143    Edema         +0.0064
Pneumonia     +0.0193    Mass          +0.0140    Infiltration  +0.0060
                         Fracture      +0.0134    Consolidation +0.0054
```

Toàn **nhiễm trùng và chấn thương** — đúng nhóm có liên hệ sinh lý với WBC, vi sinh, bối cảnh nhập
viện. **Tail giờ là `−0.0005`**: EHR không giúp đúng chỗ ta thiếu nhất.

### 4.7 Task 2 — 26 nhãn (`34`)

**Hai tập, hai thang đo, KHÔNG so được với nhau:**

| tập | ảnh | prevalence TB | quan hệ với split của ta |
|---|---|---|---|
| `development_labeled_task2.csv` | **39,293** | 9.13% | **trùng khít** tập val |
| `test_labeled_task2.csv` | **406** | **16.10%** | ⊂ split test, **0 bệnh nhân trùng** với dev |

**14 nhãn bị bỏ so với task 1:** `Adenopathy, Azygos Lobe, Clavicle Fracture, Fissure, Granuloma,
Hydropneumothorax, Infarction, Kyphosis, Lobar Atelectasis, Pulmonary Embolism, Pulmonary
Hypertension, Rib Fracture, Round(ed) Atelectasis, Tuberculosis`.

**Tập 406 ảnh được làm giàu lớp hiếm và gán nhãn lại.** Tách hai hiệu ứng:

```
dev  (tu dong)        0.0913
test406 (tu dong)     0.2049    <- CHON ANH      x2.25
test406 (chuyen gia)  0.1610    <- GAN NHAN LAI  x0.79
```

**Trần do nhiễu nhãn** — lấy chính nhãn tự động làm dự đoán, chấm trên nhãn chuyên gia:

```
AP cua chinh TIN HIEU HUAN LUYEN (macro)   0.6528
vo dich Task 2                             0.5260  = 81% cua muc do
```

Nghĩa là `0.526` **không phải 52.6% của hoàn hảo, mà là 81% của thứ tín hiệu huấn luyện cho phép**.
Trần theo nhãn: `Lung Lesion 0.137` · `Pneumonia 0.191` · `Infiltration 0.261` … `Support Devices
0.940` · `Pleural Thickening 0.957`.

---

### 4.8 Bảng từng nhãn — `per_label_prevalence_baseline.csv`

Sinh bằng `tools_per_label.py`: prevalence đọc **thẳng từ `labels.csv` trong zip** (cả ba split),
AP lấy từ `33` C6/C7 (arm `nosq518`, val 39,293 ảnh).

Cột: `nhan · tier · t2 · n_train · n_val · n_test · prev_train · prev_val · neg_pos · cls · patch ·
ehr · fusion · patch_tru_cls · lift_ehr · boi_so_san`.

**`boi_so_san = patch / prev_val`** — "hơn đoán bừa bao nhiêu lần". Đây là cột đảo ngược hoàn toàn
cách đọc so với mAP thô:

```
tier     n   prev_val    cls   patch    ehr  fusion  patch-cls  lift_ehr  boi_so_san
head    13    0.1707  0.3974  0.4204 0.3109  0.4272    +0.0229   +0.0068       2.73x
medium  16    0.0142  0.0851  0.1369 0.0437  0.1417    +0.0518   +0.0048      10.27x
tail    11    0.0019  0.0068  0.0417 0.0049  0.0412    +0.0349   -0.0005      20.86x
```

**`tail` có bội số sàn 20.9× — cao gấp 7.6 lần `head` (2.73×).** `mAP tail = 0.0417` trông thảm
hại chỉ vì sàn của nó là `0.0019`. Mô hình **có** học được nhóm hiếm; nó chỉ xuất phát từ rất thấp.

Và ngược lại, **`head` không đồng nghĩa với dễ** — bốn nhãn head gần như chỉ hơn đoán bừa:

```
Infiltration               1.47x     Pneumonia      1.55x
Enlarged Cardiomediastinum 1.48x     Lung Opacity   1.79x
```

Toàn tổn thương **lan toả**. Hai nhãn **gần như chưa học được gì**: `Infarction` **1.06×** và
`Lobar Atelectasis` **1.21×**.

Đầu bảng bội số sàn là nhóm **khu trú**: `Pneumoperitoneum` **80.8×** · `Subcutaneous Emphysema`
**49.3×** · `Kyphosis` **42.6×** · `Azygos Lobe` **34.8×** — và phần lớn mức đó đến từ **`patch`**
(`Pneumoperitoneum: patch − cls = +0.1086`).

---

## 5 · Sự cố và bài học (đã sửa, giữ để không lặp)

| # | sự cố | hậu quả nếu bỏ qua | cách sửa |
|---|---|---|---|
| 1 | `study_id` có tiền tố `'s'` | mọi join BigQuery **khớp 0 dòng, không báo lỗi** | cắt tiền tố + đối chiếu ngược với `fpath` (`28` C1) |
| 2 | `sorted(cands)[-1]` chọn `mimiciv_ed` thay vì `mimiciv_3_1_ed` (`'e' > '3'`) | dùng nhầm phiên bản ED | suy phiên bản từ HOSP; hoá ra `3_1_ed` không tồn tại nên fallback là đúng |
| 3 | `LIKE '%hemoglobin%'` khớp **25 itemid** gồm `HbA1c` (%) và `Carboxyhemoglobin` (%) | study có xét nghiệm cuối là HbA1c nhận **5.6** trong cột đơn vị **g/dL** | `= 'hemoglobin'` chính xác + cổng `MAX_ITEMID = 4`. Assert cũ chỉ bắt **thiếu**, không bắt **quá rộng** |
| 4 | regex `lab_\w+_24h` khớp cả `lab_wbc_dh_24h` | cột đếm/recency bị xếp nhầm thành giá trị lâm sàng ⇒ **phá vỡ tách tầng** | đổi `kind` thành **tiền tố**: `dh_lab_wbc_24h`, `n_lab_wbc_24h` |
| 5 | `28c` C4-bis so `_prep_ref` với **chính nó** (`FAST_DECODE=False`) | đèn xanh giả `PSNR 138.1 dB` | mặc định tắt + cổng trên PSNR **tệ nhất** |
| 6 | `31` ghi `_done_<arm>.json` **trước** `flush()` | session chết ⇒ `_done` khai đã xong mà dữ liệu chưa xuống Drive ⇒ **2,330 hàng toàn 0** trong `emb_crop518_cls.npy` (`cls` sai lệch **−0.0272**) | `flush()` **trước** `json.dump`, + cổng đọc **lại từ đĩa** ngay sau mỗi arm |
| 7 | RESUME của `32` khoá theo `(arm, kind)`, **không nhìn file embedding** | embedding sửa rồi nhưng `cxr_only.csv` giữ số cũ mãi mãi | đổi embedding arm nào thì **xoá cả dòng CSV lẫn `p_*.npy`** của arm đó |
| 8 | bảng tra `LOC` dựng bằng cách quét cả ba thư mục shard rồi **ghi đè** | thư mục letterbox chứa cả 377,110 ảnh ⇒ **mọi** `dicom_id` trỏ vào `lb_*.tar` | `loc_for(arm)` chỉ quét thư mục của arm đó |
| 9 | `for m in tar` trực tiếp trên Drive FUSE | **treo cứng** (tar thứ 4 đứng 5 phút) | đọc **nguyên tar vào RAM** rồi `tarfile.open(BytesIO)`, 8 luồng, checkpoint từng tar |
| 10 | AP tự viết tách các điểm **bằng nhau** theo thứ tự sắp xếp | lệch `~1e-7` so với `sklearn` | gộp điểm bằng nhau thành **một ngưỡng** |
| 11 | cổng tái lập dùng **sai số tuyệt đối** `1e-3` | `nosq518` trượt với `1.53e-3` — thực ra là **nhiễu fp16** (`5.57e-04` = đúng epsilon fp16); ngưỡng tuyệt đối còn phụ thuộc biên độ embedding | chấm bằng **sai số tương đối < 5e-3 + cosine > 0.9995** |
| 12 | ngưỡng cứng `0.005` cho độ dốc phân tầng | dry-run bơm suy giảm **hoàn toàn đều** vẫn ra `+0.0117` ⇒ **báo động giả** | đo **sàn nhiễu bằng hoán vị** (20 lần), báo `z` |
| 13 | `os.path.exists()` trả `True` trên mount Drive **đã chết** | tưởng còn sống, lỗi mãi sau mới hiện | verify bằng một lần **đọc thật** |
| 14 | `nan_to_num(0)` **sau** chuẩn hoá | "thiếu" trùng điểm với "bằng trung bình"; lớp tuyến tính **không nhân được** `mask × value` | chưa sửa — nằm trong kế hoạch Stage 4 |

**Bẫy cú pháp lặp lại 6+ lần:** `'''` lồng trong `code(r'''...''')` của các script dựng notebook.
Luôn dùng `r"""..."""` làm delimiter ngoài.

---

## 6 · Artefact trên Drive

```
planB_out/
├── cxr_560_shards/                       346,578 anh crop 559px      (04a)
├── cxr_560_shards_cxrlt/                  30,532 anh crop 559px      (28b)
├── cxr_1022_letterbox_cxrlt/             377,110 anh, 378 tar, 49.3 GB (28c)
└── cxrlt_2024/
    ├── cxrlt_inventory.json              labels_task1, tier head/medium/tail
    ├── cxrlt_index.parquet               dicom_id + 40 nhan + ID
    ├── cxrlt_view_resolved.parquet       dicom_id, view_group, use
    ├── cxrlt_view_feat_28e.npz           cache dac trung view (~96 MB)
    ├── cxrlt_base.parquet                index_time, cur_hadm, cur_edstay, setting
    ├── cxrlt_ehr.parquet                 dac trung EHR muc study
    ├── cxrlt_ehr_spec.json               danh sach cot cua Q_CTX / Q_CLIN / Q_FULL
    ├── cxrlt_ehr_vocab.pkl               vocab organism/specimen (khop CHI tren train)
    ├── embeddings/
    │   ├── emb_index.parquet             thu tu hang dung chung
    │   ├── emb_{crop518,lb518,nosq518}_{cls,patch}.npy
    │   └── _done_{arm}.json
    ├── stage2_ehr_only/                  ket qua nb30
    ├── stage3_baselines/
    │   ├── cxr_only.csv                  mAP moi (arm, kind) + AP tung nhan
    │   ├── three_baselines.csv           CXR-only / EHR-only / EHR+CXR
    │   ├── fov_crop_vs_letterbox.csv
    │   ├── padding_isolation.csv         nosq518 vs lb518 theo tier
    │   ├── nosq_vs_lb_{cls,patch}.csv
    │   └── p_*.npy                       xac suat tren val (9 file)
    ├── stage3_padding/                   ket qua nb33
    │   ├── pad_val.parquet · pad_parts/  do dem tung anh
    │   ├── batching_design.csv · stratified_padding.csv
    │   ├── cls_vs_patch_{arm}.csv · ehr_lift_per_label.csv
    └── task2/                            ket qua nb34
        ├── task2_prevalence.csv · task2_dev.csv · task2_test406.csv
        ├── task2_per_label.csv · task2_label_ceiling.csv
        ├── q_{arm}_{kind}.npz            du doan 26 nhan tren val + test406
        └── _test_touches.jsonl           NHAT KY moi lan cham test
```

---

## 7 · Việc còn mở / cần xác nhận

| # | việc | vì sao quan trọng |
|---|---|---|
| 1 | **số cột thật của `cxrlt_ehr.parquet`** | §3.2 đang ghi ba con số khác nhau (398 / 290 / 254) |
| 2 | **tỉ lệ study val có lateral** (`32` C2 in sẵn) | quyết định trần của mọi cải tiến multi-view; cohort cũ là 48.3% |
| 3 | `zero_rows('crop518')` phải ra **0** sau lần sửa | nếu chưa sạch thì `crop518_cls = 0.1638` cũng chưa đúng |
| 4 | `lb1022` — chưa trích | bằng chứng hiện tại **chống lại** giả thuyết độ phân giải (crop518 mm/px tốt hơn mà vẫn thua) |
| 5 | phân tách `no_ehr` khỏi `outpatient` trong báo cáo `30` | đang trộn "thưa" với "không có" |
| 6 | ASL thay BCE | chưa chạy trên CXR-LT; cohort cũ đo `ASL − BCE = +0.0050`, p=0.029 |

---

## 8 · Ba câu phải ghi cạnh MỌI bảng kết quả

1. **`375,798 / 377,110` ảnh được dùng**; 1,312 ảnh bị loại vì không xác định được view.
2. **val+test mất 7 study** (val 1, test 6 — 0.004% / 0.012%). Split chính thức coi như nguyên vẹn.
3. **10,154 view do mô hình gán**, ước tính **~34 ảnh sai** (0.33% × 10,154).

Và hai câu nữa cho mọi con số mAP:

4. **Sàn ngẫu nhiên trên val là `0.0617`** — mọi mAP phải đọc cạnh nó, và chênh lệch **dưới `0.005`
   là nhiễu** (hạng 2–4 của giải cách nhau 0.004).
5. **Sàn đó không đều**: từ `0.3349` (`Support Devices`) xuống `0.00048` (`Clavicle Fracture`) —
   chênh **693 lần**. Một con số macro đơn lẻ **không đọc được** nếu không tách theo tier.
