# CXR-LT 2024 — trạng thái Stage 0 và các số phải mang theo

Cập nhật: 2026-08-04. Nguồn: `28_cxrlt_inventory.ipynb`, `28b`, `28d`, `28e` đã chạy xong.

---

## 1 · Dataset

```
labels.csv   377,110 anh / 227,835 study / 65,379 benh nhan
split        train 258,871 | val 39,293 | test 78,946      (chinh thuc, dung thang)
nhan         45 cot int64 {0,1} — KHONG -1, KHONG o rong, tren CA BA split
task1        40 nhan  <- muc tieu chinh
task3         5 nhan  zero-shot: Bulla, Cardiomyopathy, Hilum, Osteopenia, Scoliosis
```

**Ba cổng toàn vẹn đều đạt** (`28` C4):

| cổng | kết quả |
|---|---|
| bệnh nhân rò rỉ giữa split | **0 / 65,379**; study bị cắt ngang: **0** |
| nhãn hằng số trong cùng study | **0 / 227,835** → làm ở **mức study** hợp lệ, view pooling 0.7/0.3 dùng được |
| `dicom_id` duy nhất | có |

**ID cần chuẩn hoá:** `subject_id` đã là `int64`, nhưng `study_id` là **chuỗi** `'s50414267'`.
`28` C1 cắt tiền tố `s` → `int64` và đối chiếu ngược với `fpath` để chứng minh không lệch dòng.
Bỏ qua bước này thì mọi join BigQuery **khớp 0 dòng mà không báo lỗi**.

**Phân tầng long-tail** (theo `n_train`, ngưỡng 1,000 / 10,000):

```
head    13 nhan   10,087 -> 86,079 ca duong
medium  16 nhan    1,169 ->  8,919
tail    11 nhan      129 ->    903
mat can bang 667x   (Support Devices 86,079  <->  Lobar Atelectasis 129)
```

---

## 2 · Ảnh

| bộ | số ảnh | chế độ |
|---|---|---|
| `cxr_560_shards/` | 346,578 | crop 559px (từ `04a`) |
| `cxr_560_shards_cxrlt/` | 30,532 | crop 559px (từ `28b`, **khớp pixel bit-for-bit** với chế độ `04a`) |
| **tổng** | **377,110 / 377,110** | |

Ảnh gốc MIMIC-CXR-JPG **đã có sẵn trên Drive**, nên `28b` chỉ là tiền xử lý, không phải tải về.

Toàn bộ 346,578 ảnh của shard cũ đều nằm trong CXR-LT (**0 ảnh dư**) → tập ảnh `cohort_v2` là
**tập con thực sự** của CXR-LT. Hệ quả: bệnh nhân từng ở TEST của cohort cũ nay có thể ở TRAIN của
CXR-LT ⇒ **không so được với bất kỳ kết quả R0 nào**.

---

## 3 · `view_group` — đã giải xong

15,769 ảnh (4.18%) thiếu `ViewPosition`. `ViewCodeSequence_CodeMeaning` **cũng rỗng trên đúng
15,769 ảnh ấy** — hai trường DICOM mất cùng nhau, nên metadata trong CSV không cứu được gì.

Ảnh `unknown` **đi thành cả study**, không rải lẻ: 11,466 ảnh nằm trong 5,075 study (2.26 ảnh/study).
Vì vậy loại-trừ-trong-study gần như vô dụng (cứu được **1** ảnh).

**Kết quả cuối** (`cxrlt_view_resolved.parquet`, ghi bởi `28e` C5):

| nguồn | số ảnh |
|---|---|
| `ViewPosition` | 361,341 |
| `28e classifier (acc_tu_tin 0.9967)` | **10,154** |
| `C1/C2 metadata` (BigQuery, qua `28d`) | 4,302 |
| `C3 loại trừ trong study` | 1 |
| **bỏ** (không đủ tự tin) | **1,312** |

```
GIU  375,798 / 377,110  (99.65%)
```

**Bộ phân loại view** (`28e`) — theo Xue et al., *Chest X-ray Image View Classification*:

* đặc trưng: image profile 400-d + body size ratio 1-d, tiền xử lý §III.A của bài
* mô hình: **HistGradientBoosting** (tuyến tính chỉ đạt 0.9684 — crop của `04a` xoá mất tỉ lệ khung,
  chỗ mà body size ratio dựa vào)
* khớp trên split `train`, chấm trên ảnh **có nhãn** của val/test

```
acc toan bo tap giu rieng   0.9861   (tham khao)
acc tren phan vuot CONF_GATE 0.9967   <- cong cham so nay, vi ta CHI gan phan do
CONF_GATE = 0.99  ->  91.3% tap giu rieng vuot nguong (n=12,140)
```

Cổng chấm trên **phần sẽ được gán**, không phải toàn bộ — chấm toàn bộ là đo nhầm đại lượng, nó bị
kéo xuống bởi chính những ảnh ta sẽ không dùng tới.

Bằng chứng nhóm `unknown` không lệch phân bố nặng: **88.7%** ảnh unknown vượt ngưỡng tự tin, so với
**91.3%** ở tập giữ riêng.

---

## 4 · Ba câu phải ghi cạnh MỌI bảng kết quả

1. **`375,798 / 377,110` ảnh được dùng**; 1,312 ảnh bị loại vì không xác định được view.
2. **val+test mất 7 study** (val 1, test 6 — 0.004% / 0.012%). Split chính thức coi như nguyên vẹn.
3. **10,154 view do mô hình gán**, ước tính **~34 ảnh sai** (0.33% × 10,154).

---

## 5 · Bẫy đã biết

**`28e` C5 đọc chính đầu ra của nó.** `P28D` trỏ vào `cxrlt_view_resolved.parquet` — file mà C5 tự
ghi. Chạy C5 lần hai sẽ báo `lay tu 28d: 14,457 anh` (= 4,303 của `28d` + 10,154 của lần trước) và
`gan 0 anh`. Lần này vô hại vì `nguon_view` giữ nhãn gốc. **Nhưng nó dính**: đổi `CONF_GATE`/
`ACC_GATE` rồi chạy lại thì phép gán cũ vẫn được mang sang dù cổng mới đã bác.
→ **Muốn đo lại từ đầu phải xoá `cxrlt_view_resolved.parquet` trước.**

**`28` C6 đã lỗi thời.** Nó chỉ đọc `cxr_560_shards/`, không biết `cxr_560_shards_cxrlt/`. Chạy lại
`28` bây giờ sẽ ghi `has_shard=False` cho 30,532 ảnh đã có shard. Không dùng cột `has_shard` nữa —
dựng `LOC` từ `_work_list.parquet` của **cả hai** thư mục như `28e` C0 làm.

**`31` C4: `_done` ghi TRƯỚC `flush()` — hàng toàn 0 mà resume không bao giờ sửa.** Đã dính
**hai lần**. `_done_<arm>.json` ghi sau mỗi tar, nhưng `CLS.flush()` chỉ chạy ở `finally` cuối vòng
arm; session chết giữa chừng thì `_done` khai "đã xong" trong khi dữ liệu còn ở page cache, chưa
xuống Drive. Lần hai phát hiện qua cổng C5: `crop518: 63 hàng (trong mẫu `[::37]`) toàn 0` ≈ 0.62%
≈ 2,330 ảnh ≈ 2–3 tar.

Đã sửa: **`flush()` trước `json.dump(done)`**, cộng một cổng đọc **lại từ đĩa** ngay sau mỗi arm
(`np.load(..., mmap_mode='r')` mới, không dùng memmap đang mở) để bắt ngay tại chỗ thay vì ở C5.

Sửa dữ liệu cũ: quét hàng 0 → quy ngược về tar → xoá tar đó khỏi `_done` → chạy lại C4. Khi dựng
bảng `dicom_id -> tar` phải dùng **`loc_for(arm)` chỉ quét thư mục của chính arm đó** — quét cả ba
thư mục rồi ghi đè sẽ làm mọi `dicom_id` trỏ vào `lb_*.tar` (thư mục letterbox chứa cả 377,110 ảnh).

**Cổng tái lập của `31` C5 chấm bằng sai số TUYỆT ĐỐI — sai từ gốc.** `nosq518` trượt với
`1.53e-03` trên ngưỡng `1e-3`, nhưng đó là **nhiễu fp16**: C4 chạy batch 64 còn cell kiểm tra chạy
`embed_blobs`, mà với `nosq` mỗi ảnh một hình dạng nên nó thành **batch 1** — cuBLAS đổi hẳn kernel
(GEMV thay GEMM) nên thứ tự cộng dồn khác. Ngưỡng tuyệt đối còn phụ thuộc biên độ embedding, vốn
khác nhau giữa các arm.

Đã sửa: chấm bằng **sai số tương đối (< 5e-3) + cosine tối thiểu (> 0.9995)**. Lệch hàng hay nhầm
ảnh sẽ làm cosine sụp đổ; nhiễu fp16 thì không đụng tới cosine. Hai đại lượng này tách được đúng
hai thứ cần tách.

**20,156 ảnh (5.34%) có vector nhãn toàn 0** — không `Normal`, cũng không findings nào. `Normal=0`
**không** có nghĩa "có bệnh"; đừng dùng `Normal` làm chỉ báo phủ định.

**Nhãn chồng lấn nhưng KHÔNG lồng nhau**: `Rib Fracture → Fracture` chỉ 73.2% (9,582/13,087),
`Lobar Atelectasis → Atelectasis` 78.0%, `Round(ed) Atelectasis → Atelectasis` 85.0%. Không phải
phân cấp — là các nhãn dư thừa một phần, và nhãn cha không nhất quán.

**4 nhãn có <50 ca dương trên val** (`Azygos Lobe` 20, `Clavicle Fracture` 19, `Lobar Atelectasis`
26, `Round(ed) Atelectasis` 46) → AP phương sai rất lớn, bắt buộc in kèm `n_pos`.

---

## 6 · Stage 1a — `cxrlt_base.parquet` (đã chạy)

```
227,835 study | index_time 2110-01-11 -> 2208-12-08 | truy van quet 0.05 GB trong 7s
inpatient 110,892 (48.67%) | ed 77,801 (34.15%) | outpatient 39,142 (17.18%)
co encounter 82.82%   — deu giua ba split (outpatient 17.0-17.6%)
```

`study_datetime` **có tồn tại** trong `mimic_cxr_jpg.metadata` → dùng thẳng, không phải ghép
`StudyDate`+`StudyTime`.

**Rủi ro số một của plan đã không thành hiện thực.** Chỉ 17% study là ngoại trú, nên EHR-only
không bị pha loãng. Vẫn giữ yêu cầu báo cáo tách theo `setting` — nhưng để đọc cơ chế, không phải
để cứu vãn con số.

### Bốn điều cần mang theo

**`mimiciv_3_1_ed` KHÔNG tồn tại — chỉ có `mimiciv_ed`.** Cổng khớp-phiên-bản đã cảnh báo và rơi
về `physionet-data.mimiciv_ed`. Đây là lựa chọn **đúng và duy nhất** (MIMIC-IV-ED là module riêng,
đánh phiên bản độc lập với MIMIC-IV), nhưng hệ quả cần kiểm: nếu ED module phủ **khoảng thời gian
hẹp hơn** `hosp`, thì study nằm ngoài khoảng đó sẽ không khớp `edstay` và bị gán nhầm thành
`outpatient`. → **Cần kiểm: vẽ tỉ lệ `setting` theo năm của `index_time`.** Nếu `ed` tụt về 0 ở
hai đầu dải thời gian thì một phần `outpatient` là giả.

**11,837 study (5.2%) có `subject_id` không nằm trong `hosp.patients`** → **không có age/gender**.
Cohort cũ có 100% vì `01` lọc `in_patients`; CXR-LT thì không. Phải xử lý bằng mask + cột chỉ báo
thiếu, **không** được điền giá trị trung bình.

**3,871 study khớp CẢ `hadm` LẪN `edstay`.** Quy tắc ưu tiên gán chúng thành `inpatient`, nhưng tại
thời điểm chụp chúng có thể vẫn đang ở ED (bệnh nhân vào ED rồi được nhập viện). Khi báo cáo tách
theo `setting`, nhớ nhóm này nằm trong `inpatient`.

**Chất lượng join sạch:** 0 study thiếu `index_time`, 0 study khớp >1 đợt nằm viện, 1 study khớp
>1 đợt cấp cứu, 0 lệch `subject_id` giữa CXR-LT và `metadata`.

---

## 6-bis · Stage 1b — hai phát hiện khi chạy `29b`

### Module ED phủ đủ dải — con số 17.18% là thật

`29b` C1-bis kiểm bằng **năm thật** (dựng lại qua `anchor_year_group`, **không** dùng năm đã dịch,
vì MIMIC-IV dịch ngày theo từng bệnh nhân):

```
nam that   2009   2010   2011   2012   2013   2014   2015   2016   2017
%ed       39.3%  38.4%  35.0%  37.5%  40.0%  33.7%  37.1%  39.1%  39.2%
```

Ổn định 33.7–40.0%, không sụp ở đầu nào ⇒ `outpatient` **không bị thổi phồng** vì thiếu dữ liệu ED.
`edstays` 205,504 bệnh nhân so với `admissions` 223,452 — chênh lệch hợp lý, không phải giới hạn
phủ sóng.

### ⚠ Nhóm `outpatient` thực ra là HAI quần thể rất khác nhau

Tổng các năm trong bảng trên chỉ 215,996 study — thiếu **~11,839** so với 227,835, đúng bằng nhóm
`subject_id` không có trong `hosp.patients`. Và tỉ lệ ngoại trú **trên phần có năm** chỉ 12.6%,
không phải 17.18%. Chênh lệch khớp nếu toàn bộ 11,839 study kia đều `outpatient` — hợp lý, vì không
có bản ghi bệnh nhân thì cũng không có `admissions`/`edstays`.

| nhóm con | n | có gì |
|---|---|---|
| ngoại trú **thật** | ~27,300 (12.6%) | có trong `hosp.patients` → age/gender, comorbidity, có thể có labs |
| **không có hồ sơ MIMIC-IV** | **~11,839 (5.2%)** | **không có gì cả**, kể cả age/gender |

Đây là bệnh nhân MIMIC-CXR **chưa từng nhập viện trong cửa sổ trích xuất của MIMIC-IV** — có phim
nhưng không có hồ sơ bệnh viện. Với nhánh **ảnh** thì dùng bình thường; chỉ nhánh EHR là rỗng.

**Vì sao đây là bẫy, không chỉ là thiếu dữ liệu:** mọi cột `*_mask` của nhóm này đều bằng 0, nên
chúng thành một **cụm tách biệt hoàn hảo** trong `Q_CTX`. Nếu cụm đó có prevalence bệnh khác phần
còn lại — rất có thể, vì "chưa từng nhập viện" tương quan với "ít bệnh nặng" — thì mô hình nhận
được "tín hiệu" thuần tuý từ **việc dữ liệu có tồn tại hay không**. Đúng cái bẫy proxy đã nêu trong
plan, giờ có con số: **5.2%**.

**Cách làm nó hiện ra thay vì giấu đi:** trong `30` C4, tách thành nhóm thứ tư thay vì gộp vào
`outpatient`. Chỉ báo `age.isna()` tương đương chính xác với "không có trong `hosp.patients`":

```python
va['setting4'] = np.where(va.age.isna(), 'no_ehr', va.setting)   # dung setting4 o C4
```

Nếu AP ở nhóm `no_ehr` cao bất thường thì đó là phân tầng, không phải chẩn đoán.
*(Đã áp dụng trong `29b` C9 — bảng độ phủ nay có dòng `no_ehr` riêng.)*

### Về ý định nội suy giá trị thiếu về sau

Ghi lại để lần sau không phải lập luận lại: **thiếu ở đây có hai loại rất khác nhau**, và chỉ một
loại nội suy được.

| loại | ví dụ | nội suy? |
|---|---|---|
| **thiếu có điều kiện** (MAR trong nhóm) | bệnh nhân nội trú không có troponin trong 24h nhưng có ở 7d | **được** — cùng bệnh nhân, cùng loại đo, chỉ khác thời điểm |
| **vắng do cấu trúc** | `outpatient` không có vitals; `no_ehr` không có age | **không** — MIMIC-IV **không tồn tại** bảng vital ngoại trú; không có gì để nội suy từ đó |

Nội suy loại thứ hai là **bịa ra giá trị cho thứ về mặt định nghĩa không tồn tại**. Tệ hơn: giá trị
bịa đó sẽ được dự đoán từ chính các đặc trưng xác định nhóm, tức nó thành **hàm tất định của
`setting`** — không thêm thông tin, mà lại **xoá mất tín hiệu missingness**.

Hệ quả nặng nhất: nó **phá phép đo `Q_CLIN − Q_CTX`**. Hiện `*_mask` làm missingness tường minh và
đo được. Nếu nội suy, cột "giá trị lâm sàng" của một study ngoại trú trở thành hàm của bối cảnh nó
— `Q_CLIN` bị nhiễm thông tin `Q_CTX`, và hiệu giữa hai tier mất ý nghĩa.

**Nếu nội suy, làm theo ba nguyên tắc:** (1) chỉ trong **cùng tầng `setting`**, (2) **giữ nguyên
cột `*_mask`** để vẫn đo được, (3) **không đụng vào nhóm `no_ehr`**.

Và với `30` thì nội suy nhiều khả năng **không giúp gì**: XGBoost xử lý NaN nguyên bản bằng hướng
đi mặc định học được ở mỗi nút, thường tốt hơn giá trị điền. Nội suy chỉ đáng khi chuyển sang mô
hình **tuyến tính** hoặc **fusion mạng nơ-ron** ở Stage 4 — lúc đó NaN không nạp thẳng vào được.

### ⚠ Lỗi bảng lab đã sửa — `LIKE` quá rộng

`29b` C2 ban đầu dùng `LIKE '%hemoglobin%'`, khớp **25 itemid** gồm cả **`Hemoglobin A1c`** (đái
tháo đường, đơn vị %, giá trị 5–12) và **`Carboxyhemoglobin`** (ngộ độc CO, %) — trong khi
hemoglobin thật là **g/dL, 7–17**. Vì ta lấy **giá trị gần nhất theo canon**, study nào có lần đo
cuối là HbA1c sẽ nhận `5.6` nằm chung cột với `13.2 g/dL`. Rác âm thầm.
`LIKE '%red blood cell%'` cũng khớp `Red Blood Cell Enzyme` / `Fragments`.

Đã sửa thành so khớp chính xác (`= 'hemoglobin'`, `= 'red blood cells'`) và **thêm cổng chặn khi
một canon khớp > 4 itemid**. Assert cũ chỉ bắt được *thiếu* (khớp 0 itemid), không bắt được *quá
rộng* — đó là lỗ hổng của chính thiết kế cổng, không phải của dữ liệu.

---

## 6-quater · `29b` đã chạy — bảng độ phủ là kết quả quan trọng nhất

`cxrlt_ehr.parquet`: **227,835 study × 254 cột**, 36.7 MB. `Q_CTX` 157 ⊂ `Q_CLIN` 212 ⊂
`Q_FULL` 247; vi sinh chỉ còn **35 cột (14%)** so với ~82% ở cohort cũ.

```
             n_study  lab24h  lab90d  vital  comorbid   age   micro
inpatient    110,892   91.8%       —  61.1%     58.9%  100%   80.9%
ed            77,801   53.9%       —  59.9%     63.1%  100%   32.7%
outpatient    39,142    7.6%       —   0.5%     49.0% 69.8%    8.2%
```

**Đây là bằng chứng định lượng cho rủi ro proxy.** Chênh lệch độ phủ theo `setting` cực lớn, nghĩa
là **riêng các cột `*_mask` đã đủ để đoán ra `setting`**. Mà `setting` chắc chắn tương quan với
prevalence bệnh, nên **`Q_CTX` sẽ KHÔNG thấp** — biết trước điều này để không diễn giải nhầm nó
thành "EHR đoán được bệnh". Chỉ hiệu `Q_CLIN − Q_CTX` mới nói lên điều đó.

### Phát hiện mới: MIMIC-IV không có vital cho bệnh nhân nội trú thường

`co_vital` ở `inpatient` chỉ **61.1%**, không phải ~100%. Vì nguồn vital duy nhất là
`icu.chartevents` (ICU) và `ed.vitalsign`/`triage` (cấp cứu) — **không có bảng vital cho khoa
thường**. Nên `vital_*_mask` không phân biệt "nội trú vs ngoại trú" mà phân biệt
**"ICU/ED vs còn lại"** — một trục phân tầng khác và mạnh hơn. Ghi nhớ khi đọc `Q_CTX`.

### Xác nhận nhóm `no_ehr`

`co_age` ở `outpatient` là **69.78%** ⇒ `39,142 × (1 − 0.6978) = 11,829`, khớp gần như tuyệt đối
với 11,837 study không có trong `hosp.patients`. Vậy nhóm "không có hồ sơ" nằm **gần như trọn vẹn**
trong `outpatient`, đúng như §6-bis suy ra. `29b` C9 nay **tách `no_ehr` thành dòng riêng** trong
bảng độ phủ.

### Comorbidity — nhớ đây là tỉ lệ CÓ ĐIỀU KIỆN

58.6% study có tiền sử nhập viện (133,616). Các tỉ lệ dưới đây tính **trên nhóm đó**, không phải
trên toàn cohort: smoking 43.5% · diabetes 34.9% · chf 28.9% · ckd 27.8% · cancer 23.8% ·
pneumonia_hx 23.5% · copd 16.9% · cirrhosis 11.3% · hiv 2.5% · tb 0.19%.

### Lỗi báo cáo đã sửa

`co_lab_90d` in ra **0.00%** ở mọi nhóm — bất khả thi vì 90d là tập cha của 24h. Nguyên nhân: khi
bỏ **giá trị** lab 90 ngày, cell báo cáo vẫn dò cột `lab_*_90d$` (cột giá trị, không còn tồn tại)
nên `DF[[]].any(axis=1)` trả về toàn `False`. Dữ liệu ghi ra vẫn đúng — chỉ dòng thống kê sai. Đã
đổi sang đo bằng cột đếm `n_lab_*_90d`, kèm `assert _N90` để không lặp lại.

---

## 6-ter · Shard ảnh — `28c` đã xong

```
378 tar | 377,110 anh | 49.30 GB | 0 loi | 3.07 gio | 33.1 img/s
```

Nhanh hơn dự đoán (ước 18–22 img/s, thực 33.1) và nhẹ hơn (ước 62 GB, thực 49.3 GB — vùng đệm đen
nén rất tốt). Tốc độ cao hơn cả `28b` ở 559px, xác nhận chẩn đoán cũ: `28b` chậm vì đọc **rải rác**
30k ảnh khắp cây thư mục, còn `28c` đọc **tuần tự** nên Drive chạy đúng tốc độ.

`FAST_DECODE` **không dùng** — và cell đo nó (`C4-bis`) **đo sai**: `_prep` chỉ gọi `draft` khi cờ
bật, mà lúc chạy cờ đang tắt, nên nó so `_prep_ref` với chính nó và ra `PSNR 138 dB, sai lệch 0.0`.
Đèn xanh giả. Không gây hậu quả (đã chạy xong không dùng draft), nhưng nếu tin nó mà chạy lại thì
"kiểm chứng" đó rỗng. Ngoài ra `draft` vốn cũng ít giá trị ở đích 1022: nó chỉ giảm được khi cạnh
ngắn ≥ 2044, mà nhiều ảnh không đạt (`2140x1760 -> 2140x1760`).

---

## 7 · FOV — `19` C7/C8 (đã chạy), kết luận yếu đi

```
mat KHUNG      28.5%
mat GIAI PHAU  25.4%   (thr=0.03)   <- chi 3.1 pp la vien den
```

Giả thuyết "phần bị cắt chủ yếu là viền đen collimation" **sai**: 99.1% phần bị cắt là giải phẫu
thật.

**Nhưng C8 cho thấy con số không bền.** Verdict đổi theo ngưỡng nhị phân hoá:

| thr | mất giải phẫu | verdict |
|---|---|---|
| 0.03 | 25.41% | ƯU TIÊN |
| 0.06 | 25.25% | ƯU TIÊN |
| 0.10 | 25.02% | ƯU TIÊN |
| **0.15** | **24.85%** | **THĂM DÒ** |

Và chẩn đoán vành ngoài nói ngưỡng phải **nằm trên** cường độ viền (trung vị 0.133 × max,
p90 0.493), tức `thr=0.03` đang đếm cả viền thành giải phẫu. Ngưỡng đúng ≥0.15 ⇒ verdict thật là
**THĂM DÒ**, cách ngưỡng quyết định 0.4 pp.

**Đọc đúng:** ~24–25% giải phẫu thật sự bị cắt. Đừng đọc `24.85% < 25%` như một lần đổi hạng —
lệch 0.15 pp trên một ước lượng nhiễu không phải ranh giới thật.

### Vì sao `28c` vẫn đáng ưu tiên: phần bị cắt KHÔNG ngẫu nhiên

```
anh doc  (59.4% phim thang) -> cat TREN + DUOI  = dinh phoi, goc suon hoanh
anh ngang (25.7% cohort)    -> cat TRAI + PHAI  = thanh nguc ben, suon ben
```

Đối chiếu với 40 nhãn task-1, đó đúng là nơi bằng chứng chẩn đoán sống:

| nhãn | `n_train` | tầng | sống ở |
|---|---|---|---|
| `Pleural Effusion` | 66,401 | **head** | **góc sườn hoành** |
| `Pneumothorax` | 13,858 | **head** | **đỉnh phổi** (phim đứng, khí dồn lên đỉnh) |
| `Fracture` / `Rib Fracture` | 11,568 / 8,919 | head / medium | sườn vòng ra hai bên |
| `Pleural Thickening` / `Pleural Other` | 3,272 / 616 | medium / tail | rìa màng phổi |
| `Subcutaneous Emphysema` | 2,046 | medium | thành ngực |
| `Clavicle Fracture` | 168 | tail | xương đòn, sát đỉnh |
| `Hydropneumothorax` | 646 | tail | cả hai |

Hai nhãn **head-tier** có dấu hiệu đặc trưng nằm đúng vùng bị xén. Đây là cơ chế cụ thể, mạnh hơn
lập luận "mất 25% diện tích".

Đánh đổi thật của letterbox: độ phân giải mỗi pixel còn **0.77×** — hại cho nốt nhỏ và gãy xương
kín đáo. Vì có đánh đổi hai chiều nên **phải đo cả hai arm**, và Stage 3 nên báo cáo `crop` vs
`letterbox` **tách theo nhãn**, không chỉ macro: kỳ vọng là letterbox thắng ở nhóm ngoại vi
(`Pleural Effusion`, `Pneumothorax`, `*Fracture`) và thua ở nhóm cần chi tiết (`Nodule`,
`Lung Lesion`, `Granuloma`).

Giả thuyết xoay 90° **đã đóng**: cả hai nhóm hướng đều là phim ngực đúng hướng giải phẫu; ảnh ngang
gắn với phim AP di động (47.7% AP ngang so với 21.0% PA) — lý do thu nhận, không phải lỗi lưu trữ.

---

## 8 · Đối chiếu với CXR-LT 2024 challenge (deep-research, 2026-08-04)

Nguồn chính: overview chính thức [arXiv 2506.07984](https://arxiv.org/html/2506.07984),
*Medical Image Analysis* vol 106 (12/2025). Nghiên cứu chạy **một phần** — 30/102 agent lỗi vì
giới hạn phiên, bước tổng hợp cuối lỗi; 15 luận điểm đã qua kiểm chứng đối kháng, số phiếu ghi kèm.

### Điểm chuẩn — mốc để biết ta có cạnh tranh không *(3-0)*

| | vô địch | mAP | mAUROC | mF1 |
|---|---|---|---|---|
| Task 1 (40 nhãn, test lớn nhãn NLP) | Team A / zguo | **0.281** | 0.847 | 0.289 |
| Task 2 (26 nhãn, test nhãn TAY, 406 ảnh) | Team C / XYPB | 0.526 | 0.833 | 0.499 |
| Task 3 (5 nhãn zero-shot) | Team G / yyge | 0.129 | 0.741 | — |

Task 1 hạng 2–4: **0.279 / 0.277 / 0.277** — sít sao, nên chênh lệch <0.005 là vô nghĩa.
Team A tách tầng: **head 0.567 · medium 0.263 · tail 0.136**.
CXR-LT 2023 đạt 0.349–0.372 nhưng **không so được** (2024 thêm 19 lớp hiếm).
Overview **không có baseline riêng** ngoài các đội dự thi.

### EHR: không đội nào dùng *(3-0)*

Không đội top-9 nào dùng EHR có cấu trúc, dữ liệu bảng, hay metadata DICOM. Bảng 3 xếp modality
thứ hai của mọi đội đa modal là **VL (vision-language)**: báo cáo X-quang, UMLS/PubMedBERT, mô tả
bệnh do GPT-4 sinh, CXR Fact Encoder. Teams A, D, E, F thuần ảnh. Các chữ *EHR / demographics /
labs / vitals* không xuất hiện như đầu vào mô hình.

→ Hướng EHR của ta là **mới**, nhưng cũng nghĩa là **không có mốc đối chiếu**.
*(Lưu ý: đây là kết luận từ việc KHÔNG được nhắc tới trong overview, không phải tuyên bố cấm.)*

### View pooling: các đội top CÓ dùng *(3-0)*

| đội | cách làm |
|---|---|
| D | CheXFusion Transformer hai tầng, gộp đặc trưng đa view mỗi bệnh nhân |
| **F** | **trung bình có trọng số frontal + lateral, thiên về frontal** ← đúng sơ đồ 0.7/0.3 của ta |
| G | gộp đa view bằng DINOv2 + Transformer nhẹ |
| H | hợp nhất dự đoán qua các view lúc test |

→ **Giữ view pooling.** Trọng số thiên frontal có tiền lệ.
Overview **không mô tả** cách xử lý `ViewPosition` thiếu → `28d`/`28e` vẫn không có tiền lệ.

### Long-tail: ASL thắng, các loss khác vắng mặt *(3-0)*

5/9 đội tái cân bằng hàm mất mát, và **tất cả đều dùng weighted asymmetric loss (ASL)**.
8/9 dùng ensemble. **Focal / LDAM / distribution-balanced: không đội top nào dùng.**
Điều chỉnh ngưỡng theo lớp không được bàn. Team A sinh ~100 ảnh tổng hợp mỗi lớp hiếm
bằng mô hình sinh (diffusion/VAE) — hướng ta chưa thử.

### Độ phân giải *(2-1, phiếu yếu hơn)*

Dải 224–1024, phần lớn là ensemble đa độ phân giải. **Team A — vô địch Task 1 — chỉ dùng 224 và
384.** C và D dùng 1024. **Không có ablation nào so center-crop với letterbox**, không có phân
tích theo nhãn ngoại vi.

→ **Củng cố cho `28c`:** nếu vô địch đạt được ở 224px thì chi tiết tần số cao không phải ràng buộc
quyết định. Mà cái giá của letterbox chính là độ phân giải (0.77×) — nên nó rẻ hơn dự tính, và
đánh đổi lấy ~25% FOV càng có lợi. Lý lẽ này mạnh hơn con số 25% đơn thuần.

### Task 2 = tập nhãn TAY

Overview gọi Task 1 là *"large, noisy test set"* còn Task 2 là *"manually annotated gold standard
subset"* — giải thích vì sao `test_labeled_task2.csv` chỉ có **406 dòng**. Cũng giải thích
**20,156 ảnh (5.34%) có vector nhãn toàn 0**: nhãn sinh từ báo cáo, báo cáo không nhắc thì thành 0.

---

## 9 · Việc còn mở

| việc | phần cứng | thời gian | trạng thái |
|---|---|---|---|
| Stage 4 — ASL thay BCE trên `nosq518/patch` | **GPU** | ? | **việc tiếp theo**, xem §16.7 |
| Stage 4 — fine-tune block 9–11 + ML-Decoder | **GPU** | ? | sau ASL |
| `nosq1022` | **GPU** | 1.8–2.5 h | **hoãn** — bằng chứng chống lại giả thuyết độ phân giải (§16.7) |
| `31` arm `nosq518` | GPU | ~25 ph | **đã chạy** — §16 |
| `33` — đo viền đen + chấm lại baseline | CPU | ~10–20 ph | **đã chạy** — kết quả §14 |
| bậc 1 — đầu MLP 2 lớp trên vector nối (EHR×CXR) | CPU | ~15 ph | chưa viết, xem §11 |
| bậc 2 — FiLM `e → (γ, β)` điều biến `h` | CPU | ~30 ph | chưa viết, xem §11 |
| `28e` bậc B — thay `X` bằng embedding RAD-DINO | gần như miễn phí | | tuỳ chọn |
| tách nhóm `no_ehr` trong `30` C4 (xem §6-bis) | 1 dòng | | tuỳ chọn |

Đã xong: `28`, `28b`, `28c`, `28d`, `28e`, `29`, `29b`, `30`, `31` (3 arm), `32`, `33`, `19` C7/C8.

**Ba arm ảnh cho `31`/`32`:**

| arm | nguồn | model input | GPU (A100) |
|---|---|---|---|
| A crop @518 | `cxr_560_shards/` + `_cxrlt/` | 518, CenterCrop | 30 ph |
| B letterbox @518 | hạ từ 1022 lúc nạp | 518, không crop | 30 ph |
| D letterbox @1022 | `cxr_1022_letterbox_cxrlt/` | 1022, pos-embed nội suy 2× | 3.3 h |

`32` phải tính **cả hai mức**: mức ảnh (so được với leaderboard 0.281) và mức study (thiết kế
`0.7·frontal + 0.3·lateral` của ta). Gộp study rồi phát ngược lại cho từng ảnh là ra cả hai, chi
phí gần bằng 0.

`28e` C2 giờ **có cache** (`cxrlt_view_feat_28e.npz`, ~96 MB) — restart session không phải trích
lại 34 phút. Vân tay gồm `N_PROFILE`, `N_SHARD_TR`, số ảnh, số chiều; đổi bất kỳ cái nào thì cache
tự bị bỏ qua.

---

## 10 · Kết quả `31` + `32` — ba mốc gốc, và câu hỏi chúng để lại

### 10.1 Ba mốc gốc (arm `lb518` / `patch`)

```
 mo hinh   mAP_anh  mAP_study    head  medium    tail
CXR-only    0.2002     0.2122  0.4154  0.1352  0.0402
EHR-only    0.1199     0.1222  0.3109  0.0438  0.0049
EHR+CXR     0.2049     0.2169  0.4227  0.1400  0.0420
```

`EHR thêm vào CXR: +0.0048` · `CXR thêm vào EHR: +0.0850`

**Khoảng cách tới vô địch không đều theo tầng** — probe đông cứng bám khá sát ở head (0.4154 vs
0.567 = 1.36×) nhưng thua **3.4 lần** ở tail (0.0402 vs 0.136). Fine-tune và kỹ thuật long-tail
đóng góp chủ yếu **ở đuôi** — đó là nơi Stage 4 nên nhắm.

### 10.2 Lưới arm × biểu diễn

```
mAP muc ANH        cls    patch    both  |  mAP muc STUDY     cls    patch    both
crop518         0.1910   0.1917  0.1977  |  crop518        0.2057   0.2023  0.2098
lb518           0.1656   0.2002  0.1974  |  lb518          0.1769   0.2122  0.2123
```

`32` C4 chọn bằng `R.mAP_anh.idxmax()` — **chỉ trục mức ảnh**, và không nói ra điều đó. Ở mức study
`lb518/both` (0.2123) so với `lb518/patch` (0.2122) chênh **0.0001**, tức hoà trong nhiễu (§8: dưới
0.005 là nhiễu). Muốn chọn theo mức study thì đổi một dòng.

`both` **không** luôn thắng: với `lb518` nó **thua** `patch`. Nối thêm CLS yếu vào 768 chiều dưới
`wd=1e-4` làm probe tuyến tính tệ đi.

### 10.3 Giả thuyết FOV theo giải phẫu — **BÁC BỎ**

`32` C5 nhóm theo giải phẫu, dự đoán `ngoai vi` dương rõ và `chi tiet` âm:

```
nhom        n  delta_tb  n_thang
chi tiet    5   +0.0065        4
khac       26   +0.0097       17
ngoai vi    9   +0.0061        6
```

Ba nhóm **dương và gần bằng nhau**. Cách phân nhóm theo giải phẫu **không giải thích được gì**.

### 10.4 Trục thật là **số ca dương**, không phải giải phẫu

Sắp lại bảng đó theo `n_pos_val`:

| tầng | letterbox thắng | thua |
|---|---|---|
| **tail** (11 nhãn) | **11/11** | 0 |
| medium (16) | 13 | 3 |
| **head** (13) | 2 | **11** |

11/11 cùng dấu ở tail: xác suất ngẫu nhiên ~0.001 → **hiệu ứng hệ thống, không phải nhiễu**. Head
thua đều: `Atelectasis` −0.0104, `Cardiomegaly` −0.0086, `Lung Opacity` −0.0077,
`Pleural Effusion` −0.0061.

Cơ chế khả dĩ: letterbox đổi **độ phân giải** lấy **tính nhất quán khung hình + phủ trọn**. Nhãn có
10,000+ ca dương thừa dữ liệu để trung bình hoá biến thiên đóng khung và hưởng lợi từ độ phân giải;
nhãn 20–130 ca thì không đủ mẫu, nên nhất quán và phủ trọn quý hơn.

### 10.5 ~~Bằng chứng loại trừ độ phân giải~~ — **ĐÃ RÚT LẠI, xem §16.3**

> **RÚT LẠI.** Mục này dựng trên `crop518_cls = 0.1910`, con số tính khi mảng còn ~2,330
> hàng toàn 0. Giá trị sạch là **0.1638**, và khi đó `lb518 − crop518` ở `cls` là **+0.0018**
> chứ không phải `−0.0254`. Đọc **§16** thay cho mục này.

Cùng một thay đổi crop → letterbox, hai cách đọc **cùng một ảnh** đi **ngược chiều**:

```
cls    0.1910 -> 0.1656   -0.0254
patch  0.1917 -> 0.2002   +0.0085
```

Nếu chỉ do mất độ phân giải thì cả hai phải cùng giảm. Chúng không. **Ba cơ chế cạnh tranh**, và
`lb1022` chỉ sửa được một:

| cơ chế | `lb1022` sửa được? |
|---|---|
| **1. độ phân giải** 0.812 → 0.411 mm/px (vượt cả crop 0.626) | **có** |
| **2. đệm đen pha loãng patch-mean** (~17% patch là viền) | **không** — tỉ lệ đệm không đổi theo kích thước |
| **3. lệch phân bố ở CLS** — RAD-DINO pretrain trên ảnh không viền đen | **không**, nội suy pos-embed 2× có thể tệ hơn |

Cơ chế 3 giải thích được dấu ngược chiều: patch-mean chỉ bị pha loãng, còn CLS bị bóp méo **cách
gộp**.

### 10.6 Viền đen KHÔNG phải "padding bị bỏ qua" như trong CNN

Hai chỗ phép so sánh với zero-padding của tích chập bị gãy:

**Hai số 0 nằm ở hai không gian khác nhau.** Zero-padding CNN đệm **feature map** bằng 0, sau chuẩn
hoá — ở đó 0 thật sự là "không đóng góp". Viền đen của ta là **pixel 0 ở ảnh đầu vào**, còn phải đi
qua chuẩn hoá: `(0 − 0.485)/0.229 = −2.12`. Model không bao giờ thấy số 0; cái đến tầng đầu là kích
thích **âm sâu hơn 2 độ lệch chuẩn**.

**RAD-DINO không có padding tích chập nào cả.** Patch embedding là
`Conv2d(3, 768, kernel=14, stride=14, padding=0)` — mỗi ô 14×14 thành **một token thật**, có
position embedding riêng, dự self-attention với mọi token khác **ở cả 12 tầng**. Trong CNN padding
chỉ ảnh hưởng vài pixel biên và tắt dần; trong ViT 233/1369 ô đệm = **17% ngân sách attention** tiêu
cho token rỗng, bơm thẳng vào CLS.

Hệ quả: **đổi giá trị đệm chỉ hạ cường độ, không hạ số lượng** — tile toàn 0 qua `Conv2d` **có bias**
vẫn ra `bias` chứ không ra vector 0, rồi còn cộng position embedding. Bỏ đệm hẳn tốt hơn vá nó.

---

## 11 · `33` — điều tra viền đen (đã viết, chờ chạy)

CPU, **không GPU, không trích lại embedding**. Chỉ đọc shard letterbox + `p_*.npy` đã lưu.

| cell | làm gì |
|---|---|
| C1 | quét bbox nội dung **toàn bộ val** (39,293 ảnh), `Image.draft` giải mã 1/8, cache parquet |
| C2 | cổng hình học: đối xứng · một trục · **khớp 83.2% tính độc lập từ `1/AR`** |
| C3 | mô phỏng đệm còn lại: vuông vs K rổ vs **sắp theo AR + batch liên tiếp** |
| C4 | chấm lại **độc lập** (AP tự viết + gộp `groupby`) rồi đối chiếu từng số với `32` |
| **C5** | **cell quyết định** — phân tầng val theo tứ phân vị đệm, chấm `lb518` vs `crop518` trong từng tầng |
| C6 | `cls` vs `patch` **theo từng nhãn** (trục biểu diễn — `32` chưa từng in) |
| C7 | lift EHR **theo từng nhãn** — `+0.0048` trải mỏng hay tập trung? |

> **CẢNH BÁO (viết sau khi `33` chạy xong):** phán quyết của C5 dưới đây **không dùng được** —
> tầng đệm thấp nhất vẫn còn 9.2% và hình phạt đã bão hoà ngay tại đó, nên đường phẳng mà C5 đo
> được **không** phải bằng chứng phủ định. Xem **§14.2**. Phép thử đúng là `nosq518` ở **§15**.

**C5 quyết định có cần chạy `lb1022` hay không:**

```
do doc AM va z <= -2  -> DEM la thu pham -> resize khong-deu, BO QUA lb1022 (tiet kiem 3h)
|z| < 2 ca hai        -> dem khong giai thich duoc -> chay lb1022
```

`z` đo trên **sàn nhiễu hoán vị** (20 lần gán tầng ngẫu nhiên), không phải ngưỡng đoán. Dry run cho
thấy suy giảm **hoàn toàn đều** vẫn ra độ dốc `+0.0117` — ngưỡng cứng 0.005 sẽ báo động giả. Dry run
xác nhận cả hai chiều: bơm suy giảm phụ thuộc đệm → `z = −7.04` (cls) / `−3.04` (patch); bơm suy
giảm đều → `z = +0.48` / `+0.11`.

**Chi phí:** `33` ~10–20 ph CPU vs `lb1022` 2.3–3.3 h GPU → rẻ hơn ~20–30 lần và có thể **loại bỏ**
lượt chạy GPU đó. `mAP` tuyệt đối giữa các tầng **không so được** (prevalence khác nhau) — chỉ cột
`delta` trong cùng một tầng mới có nghĩa, và `crop518` (không đệm) là mốc so ngay trong tầng đó.

### Cách bỏ đệm nếu C5 kết luận đệm là thủ phạm

`Dinov2Embeddings.interpolate_pos_encoding(embeddings, height, width)` nhận **height và width riêng
biệt** — ViT không đòi ảnh vuông. Resize giữ tỉ lệ, hai chiều chia hết 14:

```
letterbox 1022x1022  -> 73 x 73 = 5329 patch,  ~17% la den
khong deu 1022x854   -> 73 x 61 = 4453 patch,  0% la den   (RE HON, attention theo n^2)
```

Giá phải trả: không xếp batch được ảnh khác kích thước → **sắp toàn bộ theo AR rồi cắt batch từ các
đoạn liên tiếp**, canvas mỗi batch = cạnh ngắn lớn nhất trong batch đó. Không cần chọn số rổ, không
cần biết trước phân bố, ảnh AR lạ tự nằm cạnh ảnh gần nó. Sàn dưới chỉ còn làm tròn bội số 14 =
**1.37% mỗi hàng patch**. Ảnh portrait/landscape phải tách nhóm trước (canvas chuyển vị). `31` ghi
memmap theo chỉ số hàng nên đổi thứ tự xử lý không ảnh hưởng gì.

Phương án vá rẻ hơn nhưng không triệt để: **đệm bằng giá trị trung bình chuẩn hoá** thay vì 0 (sau
chuẩn hoá thành đúng 0 = trung tính tối đa), hoặc **nhân bản pixel biên**. Tránh `reflect` — nó soi
gương giải phẫu, tạo cấu trúc giả trên phim ngực. Che attention bằng `−inf` là can thiệp **lệch phân
bố** lên chính cơ chế đang hỏng, không nên là lựa chọn đầu.

---

## 12 · EHR — CHƯA kết luận được, và vì sao

`+0.0048` **chỉ nói được một câu hẹp**:

> Phần đóng góp **cộng tính và giải mã tuyến tính được** của EHR, ngoài những gì ảnh đã mang, là
> khoảng 0.005.

`32` C6 nối vector rồi cho qua **một lớp tuyến tính**: `W_e·e + W_h·h` — **không có số hạng chéo
nào**. FiLM là `γ(e) ⊙ h + β(e)`, tích giữa một hàm của EHR và đặc trưng ảnh; lớp tuyến tính trên
vector nối **về mặt cấu trúc không biểu diễn được** dạng đó, ở bất kỳ trọng số nào. Phép đo hiện tại
**không nói gì** về điều biến sâu hay distillation.

Lý do thứ hai khiến số đó yếu: nối 768 chiều ảnh với ~250 cột EHR dưới cùng một `wd` thì khối lớn
nuốt trọng số — **đúng hiện tượng CLS làm hỏng `both` ở `lb518`** (§10.2). Ngay cả phần cộng tính
cũng có thể đang bị đo hụt.

**Thang đo, rẻ trước đắt sau** (bậc 1–2 chạy trên embedding đã cache, **không tốn GPU**):

| bậc | biểu diễn được gì mà concat-tuyến-tính không | chi phí |
|---|---|---|
| **0. lift EHR theo từng nhãn** | — chỉ đọc lại `p_fusion_*.npy` | `33` C7 |
| **1. đầu MLP 2 lớp trên vector nối** | tương tác chéo `e × h` xuất hiện lần đầu | ~15 ph CPU |
| **2. FiLM** — `e → MLP → (γ, β)` điều biến `h` | điều biến có cấu trúc, không phải tương tác tuỳ ý | ~30 ph CPU |
| **3. distillation** — teacher EHR (`30`) → student CXR | teacher chỉ dùng lúc học | xem cảnh báo |

Nếu bậc 1 đã nhấc được thì kết luận "dư thừa" chết tại đó, và vấn đề là **dạng hàm** chứ không phải
**nội dung EHR**.

**Cảnh báo cho distillation.** Trên cohort cũ, `A2` và `A2E_full` đều trượt cổng §14, cơ chế đã hiểu
trọn: **soft target bị hấp thụ thành bias mỗi kênh**, số hạng soft chạm sàn entropy ngay epoch 1.
Nếu chạy lại phải **cắm sẵn cổng chẩn đoán đó từ đầu**: theo dõi số hạng soft có tách khỏi sàn
entropy không, và kiểm logit student có khác teacher ngoài một hằng số mỗi kênh không. Không đặt
cổng thì sẽ mất vài giờ để phát hiện lại đúng một thất bại đã biết. Thêm nữa, teacher hiện tại yếu
hơn cohort cũ — EHR-only 0.1199 mAP, **tail 0.0049**, gần như không có gì để dạy ở đuôi.

---

## 13 · ~~Chọn biểu diễn theo từng bệnh~~ — **HẾT ĐỘNG LỰC, xem §17.2**

> Trên dữ liệu sạch, `patch` thắng `cls` ở **mọi tier của mọi arm**, và với `nosq518` là
> **40/40 nhãn**. Không còn nhóm nhãn nào `cls` thắng hệ thống, nên chọn per-label chỉ còn
> là khai thác nhiễu. Phần dưới giữ lại vì cảnh báo phương pháp (chọn trên val rồi báo cáo
> trên val) vẫn đúng cho mọi phép chọn khác.

Nếu chọn per-label **trên val** rồi báo cáo mAP **trên val**, phần cải thiện **một phần là ảo** — đó
là overfit chính tiêu chí chọn. Với 40 nhãn × 2–3 lựa chọn và nhãn tail có phương sai AP rất lớn,
thiên lệch này không nhỏ.

Cách làm sạch: **cắt một lát "selection" từ train** (ví dụ 85/15), khớp probe trên 85%, chọn
`(arm, kind)` cho từng nhãn trên 15%, rồi ghép dự đoán và chấm trên val nguyên vẹn — kèm cột so với
"một cấu hình duy nhất cho cả 40 nhãn" để biết per-label thật sự mua được bao nhiêu. `test` vẫn
không được chạm.

Lưu ý: `both` đã chứa cả `cls` lẫn `patch` nên probe **tự nó có thể học trọng số riêng cho từng
nhãn** — đó là lý do `both` và `patch` hoà nhau ở mức study. Nếu per-label selection thắng `both` rõ
rệt thì cái thắng đó đến từ **chọn cứng**, không phải từ thông tin mới.

---

## 14 · `33` đã chạy — C5 KHÔNG kết luận được, và vì sao

### 14.1 Đo được gì

```
dem den:  trung binh 18.88% dien tich canvas | p95 34.69%
phan vi:  5% 0.043 | 25% 0.168 | 50% 0.176 | 75% 0.209 | 95% 0.347
noi dung trung binh 0.8112  vs  du kien 0.832 doc lap tu 1/AR  -> lech -0.0208, cong 3 DAT
huong anh: portrait 74.49% | landscape 25.51%
```

Ba cổng hình học đều đạt. C4 chấm lại độc lập khớp `32` tới `5.97e-08`.

**Bỏ đệm được bao nhiêu** (cạnh dài 1022, cạnh ngắn bội số 14):

```
cach                     dem_tb   patch/anh   attn
vuong 1022 (hien tai)    14.78%     5329      1.000
5 ro theo phan vi AR      3.03%     4687      0.773
16 ro theo phan vi AR     1.59%     4614      0.750
sap theo AR, batch 64     0.77%     4576      0.737
```

Bỏ đệm vừa sạch vừa **rẻ hơn 26% chi phí attention**.

### 14.2 Phán quyết của C8 SAI — C5 không đủ sức bác bỏ đệm

C8 in `dem khong giai thich duoc -> chay lb1022`. Kết luận đó vượt quá thứ C5 đo được, vì hai lý do:

**Không có tầng nào không đệm.** Bốn tầng có đệm trung bình `0.0919 · 0.1688 · 0.1884 · 0.2916` —
tầng thấp nhất vẫn **9.2%**. C5 so *ít đệm* với *nhiều đệm*, chưa bao giờ so *có* với *không*.

**Và hình phạt đã bão hoà ngay ở tầng thấp nhất:**

```
cls   delta theo tang: -0.0281  -0.0276  -0.0232  -0.0330   | do doc -0.0049 | z -1.35
patch delta theo tang: +0.0065  +0.0091  +0.0089  +0.0115   | do doc +0.0050 | z +1.44
```

Nếu tổn hại tỉ lệ với diện tích đệm, tầng 1 (9.2%) phải chỉ bằng ~1/3 tầng 4 (29.2%), tức khoảng
`−0.010`. Thực tế nó đã là `−0.0281`, **gần như đầy đủ**. Cơ chế **bão hoà** cho ra đúng đường phẳng
mà C5 đo được — nên đường phẳng **không** phải bằng chứng phủ định. C5 được thiết kế với giả định
liều–đáp ứng, mà dải liều sẵn có không chạm tới 0.

**Biến phân tầng còn bị nhiễm.** C2 đo `truc dai bi mat p95 = 0.1484`, mà letterbox **không bao giờ**
cắt trục dài (nó lấp đầy 1022) — phần đó là **collimation**, thứ `crop518` cũng có. Cộng thêm
`max dem = 0.955`. Ít nhất 5% ảnh đang được xếp tầng theo collimation chứ không theo đệm letterbox.

### 14.3 ~~Phân rã FOV theo tier~~ — **ĐÃ RÚT LẠI, xem §16.3**

> **RÚT LẠI.** Mục này dựng trên `crop518_cls = 0.1910`, con số tính khi mảng còn ~2,330
> hàng toàn 0. Giá trị sạch là **0.1638**, và khi đó `lb518 − crop518` ở `cls` là **+0.0018**
> chứ không phải `−0.0254`. Đọc **§16** thay cho mục này.

Từ C6 (`patch − cls` cho từng arm), tách ra thành hiệu ứng FOV trên từng biểu diễn:

```
FOV (lb518 - crop518)      patch       cls
head                     -0.0036    -0.0098
medium                   +0.0108    -0.0330
tail                     +0.0194    -0.0327
```

Kiểm: `patch − cls` từng tier khớp đúng `lb518(patch−cls) − crop518(patch−cls)` = `+0.0061` /
`+0.0437` / `+0.0521`.

**Hai cơ chế tách bạch:**

- **`cls` âm ở CẢ ba tier**, gần đều `−0.033` ở medium/tail. Không phụ thuộc tần suất nhãn ⇒ chữ ký
  của nguyên nhân **cơ học** (dải đen phá cách gộp toàn cục), không phải nguyên nhân kiểu dữ-liệu-ít.
- **`patch` phụ thuộc tần suất rõ rệt** (head âm → tail dương) ⇒ hiệu ứng **FOV** thật, cơ chế khác.

Hệ quả cho cách đọc C6: `lb518` có `patch − cls` dương ở mọi tier (`head +0.0173`, `medium +0.0493`,
`tail +0.0333`) trong khi `crop518` thì tail âm (`−0.0188`, chỉ 1/11 nhãn dương). Nhưng ~63% của
"cải thiện" đó ở tail là do **`cls` sụp**, chỉ 37% là `patch` lên thật.

### 14.4 ~~`lb518/patch` không phải cấu hình tốt nhất ở đuôi~~ — **ĐÃ RÚT LẠI, xem §16.3**

> **RÚT LẠI.** Mục này dựng trên `crop518_cls = 0.1910`, con số tính khi mảng còn ~2,330
> hàng toàn 0. Giá trị sạch là **0.1638**, và khi đó `lb518 − crop518` ở `cls` là **+0.0018**
> chứ không phải `−0.0254`. Đọc **§16** thay cho mục này.

Xét cả bốn ô (2 arm × 2 biểu diễn), `crop518/cls` thắng ở **7/11 nhãn tail**:

| nhãn | n_pos | crop/cls | lb/patch |
|---|---|---|---|
| Kyphosis | 112 | **0.1619** | 0.1217 |
| Hydropneumothorax | 128 | **0.1038** | 0.0906 |
| Azygos Lobe | 20 | **0.0527** | 0.0145 |
| Pulmonary Hypertension | 119 | **0.0247** | 0.0185 |
| Pneumoperitoneum | 54 | 0.0288 | **0.0982** |

Với `n_pos` 19–128 thì từng nhãn chưa vượt nhiễu (~1 độ lệch chuẩn), nhưng 7/11 cùng chiều thì không
bỏ qua được. **Chốt `lb518/patch` làm arm chính là sai ở đuôi** — đúng chỗ đang thua vô địch 3.4 lần.

### 14.5 EHR theo từng nhãn (`33` C7)

```
tier     n  lift_tb  n_duong
head    13  +0.0073       12
medium  16  +0.0048       10
tail    11  +0.0018        7
macro +0.0048 | 29/40 nhan duong | 13/40 vuot nguong nhieu 0.005
```

Không trải mỏng đều: **13 nhãn vượt ngưỡng nhiễu**, dẫn đầu là `Tuberculosis +0.0278`,
`Pneumothorax +0.0223`, `Pneumonia +0.0205`, `Rib Fracture +0.0188`. Nhóm nhiễm trùng/chấn thương —
đúng nhóm có liên hệ sinh lý với WBC, vi sinh, và bối cảnh nhập viện. **Tín hiệu EHR là thật và có
chọn lọc, bị macro trung bình hoá mất.** FiLM nên nhắm nhóm này trước, không phải cả 40 nhãn.

Ngược lại, `Support Devices`, `Atelectasis`, `Cardiomegaly` gần bằng 0 — ảnh đã mang trọn.

---

## 15 · `nosq518` — phép cô lập biến ĐỆM (đã viết, chờ chạy)

`31` có thêm arm `nosq518`, `32` có thêm cell **C5-bis**.

| arm | FOV | mm/px | đệm |
|---|---|---|---|
| `crop518` | 71.5% | 0.626 | 0% |
| `lb518` | 100% | 0.812 | **~18%** |
| **`nosq518`** | **100%** | **0.812** | **0%** |

`nosq518` vs `lb518`: **cùng cạnh dài, cùng FOV, chỉ khác dải đen**. Chênh lệch giữa chúng không thể
quy cho độ phân giải hay tầm nhìn — đây là thứ C5 của `33` không làm được.

**Cách dựng.** Shard letterbox là 1022×1022 nên tỉ lệ gốc đã mất — `prep_nosq` **dò khung bao vùng
khác đen rồi cắt lại**, khôi phục đúng ảnh trước khi đệm. Sau đó resize giữ tỉ lệ, cạnh dài 518, cạnh
ngắn làm tròn về bội số 14 bằng **`round` (không phải `ceil`)**: chấp nhận giãn ≤ 7/430 = **1.6% một
trục** để đệm bằng **0 tuyệt đối**. Lý do phải triệt để: nếu tổn hại của đệm là dạng **bão hoà**
(§14.2) thì 1.6% còn lại vẫn có thể đủ gây ra nó, và phép thử sẽ vô nghĩa.

**Gom batch:** ảnh khác kích thước không xếp chung được, nên gom theo khoá `(H, W)`. Cạnh ngắn đã làm
tròn về bội số 14 nên chỉ có ~30 khoá, mỗi khoá vẫn đủ ảnh để batch đầy. Giải mã cả tar trước rồi
nhóm: ~998 × 518 × 430 uint8 ≈ 220 MB, chấp nhận được.

**Khác biệt phụ duy nhất:** cắt khung bao cũng bỏ luôn collimation dính mép trục dài (~2.4%, đo ở
`33` C2), làm độ phân giải giải phẫu tăng ~2.4%. So với chênh lệch 100% của `lb1022` thì không đáng
kể, nhưng phải ghi ra.

**Chi phí:** ít token hơn (~0.84×), attention rẻ hơn (~0.70×) → **~22–26 phút GPU**, so với 2.3–3.3 h
của `lb1022`.

### Đọc C5-bis

`bu_duoc` = phần `cls` lấy lại được, chia cho phần đã mất vì letterbox (`−0.0098` head / `−0.0330`
medium / `−0.0327` tail).

```
>= 0.6   -> DAI DEN la thu pham chinh -> chuyen han sang nosq518, BO QUA lb1022
<= 0.2   -> khong phai dai den -> chay lb1022 (2.3-3.3h GPU)
0.2-0.6  -> ca hai co che cung gop -> nosq1022 moi la arm dung, nhung dat
```

Cột `lb_tru_crop` in ra để **tự đối chiếu với bảng §14.3** — lệch thì phải đọc lại trước khi kết luận
bất cứ điều gì.

Cell cũng in bảng `nosq518 − lb518` theo từng nhãn cho cả `cls` lẫn `patch`, ghi
`nosq_vs_lb_{cls,patch}.csv` và `padding_isolation.csv`.

---

## 16 · `nosq518` đã chạy — ĐỆM KHÔNG PHẢI THỦ PHẠM, và §10.5/§14.3 phải RÚT LẠI

### 16.1 Lưới sạch (sau khi `crop518` được trích lại và fit lại)

```
mAP muc ANH        cls    patch    both        mAP muc STUDY     cls    patch    both
crop518         0.1638   0.1917  0.1855        crop518        0.1750   0.2023  0.1975
lb518           0.1656   0.2002  0.1974        lb518          0.1769   0.2122  0.2123
nosq518         0.1651   0.2028  0.1974        nosq518        0.1760   0.2146  0.2113

tot nhat: nosq518 / patch — anh 0.2028, study 0.2146
theo tier: head 0.4204 | medium 0.1369 | tail 0.0417
```

### 16.2 Phép cô lập biến ĐỆM — kết quả âm tính, dứt khoát

```
nosq518 - lb518      cls      patch     both
mAP anh           -0.0005   +0.0026   0.0000
mAP study         -0.0009   +0.0024  -0.0010
```

Hai arm **cùng cạnh dài 518, cùng FOV 100%**, chỉ khác **18% đệm → 0%**. Chênh lệch lớn nhất là
`+0.0026`, nằm sâu dưới sàn nhiễu 0.005. **Dải đen không ảnh hưởng đáng kể.**

Đây là kết quả âm tính nhưng có giá trị: nó đóng giả thuyết bằng **~25 phút GPU** thay vì để nó treo
lơ lửng, và nó loại luôn cả cơ chế 2 (pha loãng patch-mean) lẫn cơ chế 3 (lệch phân bố CLS) của
§10.5 trong một lần đo.

### 16.3 PHẢI RÚT LẠI: "letterbox phá `cls`" là ẢO GIÁC do dữ liệu nhiễm

Toàn bộ §10.5 và §14.3 dựng trên `crop518_cls = 0.1910`. Giá trị đó tính khi
`emb_crop518_cls.npy` còn **~2,330 hàng toàn 0**. Giá trị sạch là **0.1638**.

```
                  BAN NHIEM (sai)     BAN SACH (dung)
lb518 - crop518, cls    -0.0254            +0.0018
```

Nghĩa là letterbox **không hề** làm hỏng `cls` — nó còn nhỉnh hơn một chút. Những thứ sau đây **hết
hiệu lực**, không được trích dẫn nữa:

- bảng phân rã FOV × tier ở §14.3 (`cls` âm ở cả ba tier)
- lập luận "`cls` âm không phụ thuộc tần suất nhãn ⇒ chữ ký của nguyên nhân cơ học"
- ba cơ chế cạnh tranh ở §10.5 và bảng "`lb1022` sửa được cái nào"
- §14.4 (`crop518/cls` thắng 7/11 nhãn tail) — dựa trên đúng cột `cls` bị nhiễm

### 16.4 Dấu vết chẩn đoán: hàng 0 chỉ nằm trong file `_cls`

```
crop518    truoc (nhiem)   sau (sach)    hieu
cls           0.1910        0.1638      -0.0272
patch         0.1917        0.1917       0.0000     <- khong doi MOT chu so
both          0.1977        0.1855      -0.0122
```

`patch` bất biến tuyệt đối trong khi `cls` tụt mạnh ⇒ hàng 0 **chỉ có trong `emb_crop518_cls.npy`**,
không có trong `_patch.npy`. Hai file riêng biệt, một cái kịp xuống Drive, một cái không — đúng cơ
chế lỗi ở §5. `both` tụt vì nó nối `cls` vào.

**Chưa giải thích được:** sửa 0.6% số hàng lại làm `cls` **tệ đi**. Giả thuyết: hàng 0 sau chuẩn hoá
train thành một vector hằng số biên độ lớn; probe học "vector đó → điểm cực trị", đẩy ~240 ảnh val
xuống đáy xếp hạng ở mọi nhãn, và nếu chúng phần lớn là âm thì AP **tăng giả**. Chưa kiểm chứng.
Nếu về sau còn nghi ngờ thì đây là chỗ phải đào.

### 16.5 Thứ SỐNG SÓT sau khi dọn

| | crop518 | lb518 | nosq518 |
|---|---|---|---|
| `patch − cls` | +0.0279 | +0.0346 | **+0.0377** |
| vs `crop518` (patch) | — | +0.0085 | **+0.0111** |
| vs `crop518` (cls) | — | +0.0018 | +0.0013 |

**`patch` thắng `cls` khoảng +0.03 ở CẢ BA arm.** Đây là phát hiện vững nhất của Stage 3, không phụ
thuộc cách tiền xử lý. Dùng `patch`, không dùng `cls`.

**FOV đầy đủ có lợi thật:** `+0.0111` ở `patch`. Nhưng lưu ý `crop518` có **độ phân giải tốt hơn**
(0.626 vs 0.812 mm/px) mà vẫn thua — **FOV thắng độ phân giải** ở thang này. Điều đó hạ hẳn kỳ vọng
vào `lb1022`/`nosq1022`.

### 16.6 Arm chuẩn mới: `nosq518`

`nosq518/patch` hơn `lb518/patch` **+0.0026** (dưới nhiễu, không phải thắng thật) nhưng **rẻ hơn 26%
chi phí attention** vì bỏ được ~18% token rỗng. Không có lý do không dùng. Mọi thí nghiệm Stage 4
nên đứng trên `nosq518`.

### 16.7 Còn `nosq1022` không?

Độ phân giải là giả thuyết duy nhất còn lại, nhưng bằng chứng mới **chống lại** nó: tăng FOV mua
được `+0.0111` trong khi độ phân giải tốt hơn của `crop518` không cứu nổi nó. Chi phí ~1.8–2.5 h GPU
(`nosq` rẻ hơn `lb` 26%).

So sánh với khoảng cách còn lại tới vô địch:

```
        ta       vo dich   ty le
head  0.4204     0.567     1.35x
medium 0.1369    0.263     1.92x
tail  0.0417     0.136     3.26x
```

Toàn bộ tiền xử lý (crop → nosq) mua được `+0.0111`. Khoảng cách ở **tail là 3.26×**. Đòn bẩy nằm ở
Stage 4 (ASL, fine-tune block 9–11 + ML-Decoder, kỹ thuật lớp hiếm), không nằm ở độ phân giải.
**Khuyến nghị: hoãn `nosq1022`, sang Stage 4.**

---

## 17 · `33` chạy lại trên dữ liệu sạch — ba kết quả

### 17.1 C5: hình phạt `cls` KHÔNG hề tồn tại

```
                 tang1    tang2    tang3    tang4   | do doc |  z
cls  (lb-crop)  +0.0012  +0.0018  +0.0001  +0.0023  | +0.0011 | +0.88
patch(lb-crop)  +0.0065  +0.0091  +0.0089  +0.0115  | +0.0050 | +1.44
```

Bản nhiễm cho `cls` **âm phẳng −0.028** ở mọi tầng. Bản sạch cho **dương, +0.001 đến +0.002**.
Toàn bộ "letterbox phá CLS" là **artefact 100%** của 2,330 hàng 0 trong `emb_crop518_cls.npy`.

`patch` thì dương thật và tăng nhẹ theo tầng đệm (`+0.0065 → +0.0115`), nhưng `z = +1.44` chưa
vượt sàn nhiễu hoán vị — và dù sao phép cô lập trực tiếp (`nosq518 − lb518 = +0.0026`, §16.2)
đã trả lời dứt điểm rồi.

> **Chữ in ra ở C8 (`=> LAM: chay lb1022`) đã LỖI THỜI.** Nó viết cho kịch bản "chưa có arm cô lập".
> `nosq518` đã chạy và cho kết quả âm tính trực tiếp. Đừng làm theo dòng đó.

### 17.2 C6: `patch` thắng `cls` ở MỌI tier của MỌI arm

```
patch - cls        head              medium            tail
crop518        +0.0222 (13/13)   +0.0416 (15/16)   +0.0145 ( 7/11)
lb518          +0.0173 (12/13)   +0.0493 (16/16)   +0.0333 (11/11)
nosq518        +0.0229 (13/13)   +0.0518 (16/16)   +0.0349 (11/11)
```

Với `nosq518`, `patch` thắng `cls` trên **40/40 nhãn**, không sót một nhãn nào (thấp nhất là
`Infarction +0.0002`).

Điều này **xoá luôn** quan sát ban đầu "dùng patch thì head sụt, tail lên". Bản nhiễm cho
`crop518` tail `−0.0188` với chỉ 1/11 nhãn dương; bản sạch cho `+0.0145` với 7/11. Không có sự
đánh đổi theo tier nào cả — **`patch` tốt hơn `cls`, chấm hết**.

Hệ quả: **§13 (chọn biểu diễn theo từng bệnh) không còn động lực.** Không có nhóm nhãn nào `cls`
thắng một cách hệ thống, nên chọn per-label chỉ còn là khai thác nhiễu.

Nhãn hưởng lợi mạnh nhất từ `patch` (nosq518): `Subcutaneous Emphysema +0.2307`,
`Hernia +0.1276`, `Pneumoperitoneum +0.1086`, `Kyphosis +0.0860`, `Hydropneumothorax +0.0844` —
toàn bệnh lý **khu trú, ngoại vi**. Hợp lý: CLS gộp toàn cục làm loãng ổ nhỏ, patch-mean giữ lại.

### 17.3 C7: nhóm nhãn nhạy EHR ỔN ĐỊNH khi đổi backbone

Ba mốc gốc mới (nền `nosq518/patch`):

```
CXR-only 0.2028 | EHR-only 0.1199 | EHR+CXR 0.2068     EHR them vao CXR: +0.0040
```

```
tier      n  lift_tb  n_duong
head     13  +0.0068       12
medium   16  +0.0048       10
tail     11  -0.0005        7
```

Bảy nhãn dẫn đầu **không đổi** so với nền `lb518/patch` cũ, chỉ đổi thứ tự nhẹ:

| nhãn | nền cũ | nền mới |
|---|---|---|
| Tuberculosis | +0.0278 | +0.0266 |
| Pneumothorax | +0.0223 | +0.0210 |
| Pneumonia | +0.0205 | +0.0193 |
| Rib Fracture | +0.0188 | +0.0172 |
| Lung Lesion | +0.0126 | +0.0143 |
| Mass | +0.0154 | +0.0140 |
| Fracture | +0.0139 | +0.0134 |

Đây là **kiểm chứng độ bền quan trọng**: nhóm nhãn nhạy EHR không phải artefact của một embedding
cụ thể — đổi hẳn cách tiền xử lý ảnh, nó vẫn nguyên. Nhóm này là **nhiễm trùng + chấn thương**,
đúng chỗ WBC / vi sinh / bối cảnh nhập viện có liên hệ sinh lý.

**Nhưng tail giờ là `−0.0005`.** Nền cũ còn `+0.0018` với `Hydropneumothorax +0.0109`,
`Azygos Lobe +0.0060`; nền mới các số đó tụt còn `+0.0028` / `+0.0020`. CXR mạnh lên thì phần EHR
đóng góp ở đuôi **bốc hơi**. EHR **không** giúp được ở chỗ ta thiếu nhất.

Kết luận cho §12: FiLM nếu làm thì **nhắm 7–11 nhãn nhiễm trùng/chấn thương**, không phải cả 40, và
đừng kỳ vọng nó cứu được tail.

### 17.4 Chốt Stage 3

```
arm chuan : nosq518          bieu dien: patch
mAP anh   : 0.2028           mAP study: 0.2146
head 0.4204 | medium 0.1369 | tail 0.0417
```

Mọi thí nghiệm Stage 4 đứng trên mốc này.

---

## 18 · Task 2 (26 nhãn) — `34` đã viết, chờ chạy

**Cấu trúc, đọc thẳng từ zip:**

```
26 nhan  = tap con THAT SU cua 40 nhan Task 1
14 nhan bi bo: Adenopathy, Azygos Lobe, Clavicle Fracture, Fissure, Granuloma,
               Hydropneumothorax, Infarction, Kyphosis, Lobar Atelectasis,
               Pulmonary Embolism, Pulmonary Hypertension, Rib Fracture,
               Round(ed) Atelectasis, Tuberculosis
```

**Hai tập chấm, và chúng KHÔNG so được với nhau:**

| tập | ảnh | prevalence TB | quan hệ với split của ta |
|---|---|---|---|
| `development_labeled_task2.csv` | **39,293** | 9.13% | **trùng khít** tập val ⇒ so được với §16–17 |
| `test_labeled_task2.csv` | **406** | **16.10%** | tập chính thức Task 2, ⊂ split test, **0 bệnh nhân trùng** với dev |

> **SỬA (sau khi chạy C1).** Bảng "làm giàu" ngay dưới gộp nhầm **hai** hiệu ứng. Xem §18.1 —
> chọn ảnh và gán nhãn lại là hai chuyện khác nhau, và `Pneumonia` / `Normal` phải đọc ngược lại.

**Tập 406 ảnh được LÀM GIÀU LỚP HIẾM rất mạnh.**

```
Pneumoperitoneum  43x | Pneumomediastinum 28x | Pleural Other 23x
Fibrosis          13x | Subcutaneous Emph.  9x | Calcification Aorta 8x
Pneumonia        0.3x | Normal            0.7x        (bi lam NGHEO)
```

AP của bộ đoán ngẫu nhiên **bằng đúng prevalence**, nên sàn nhảy từ `0.091` (dev) lên `0.161`
(test406) — **1.76×**. Một phần lớn khoảng cách giữa mốc **0.526** và bất kỳ số nào trên dev là
**cơ học**, không phải chất lượng mô hình. `34` in cột `mAP / sàn ngẫu nhiên` để khử phần đó.

**Chỉ `test406` mới so được với 0.526.** Nhưng 406 ảnh là rất ít — nhãn hiếm nhất có **6** ca dương
(`Lung Lesion`) — nên không được đọc chênh lệch nhỏ giữa các arm ở đó.

**`34` làm gì:** khớp probe **26 đầu ra** trên train, dự đoán một lượt cho cả val lẫn test406
(9 tổ hợp arm × biểu diễn, ~15 phút GPU, có RESUME). Chấm cả hai tập, phân tầng **tính lại** trên
26 nhãn (bỏ 14 nhãn hiếm thì ranh giới dịch), và so `native26` với `slice40` (cắt 26 cột từ dự đoán
40 nhãn của `32`) để trả lời: **14 nhãn kia là nhiệm vụ phụ có ích hay chỉ là nhiễu?** Với một lớp
tuyến tính + BCE, hiệu số này phải gần 0 — nếu không thì có gì đó đáng đào.

**Nhãn của 406 ảnh KHÁC `labels.csv`.** Trên `dev` hai nguồn trùng khít (0 ô lệch), nhưng trên tập
406 ảnh thì không — và đó là **chủ đích, không phải lỗi**. `labels.csv` là nhãn **tự động trích từ
báo cáo**; 406 ảnh của Task 2 được **gán nhãn lại**. Điều này khớp với việc tập nhỏ và giàu lớp
hiếm: chọn ca có tổn thương hiếm rồi gán tay.

Hệ quả: **ta huấn luyện trên nhãn tự động và chấm trên nhãn chuẩn.** Nhà vô địch cũng vậy nên so
sánh vẫn công bằng, nhưng mAP trên `test406` chịu thêm một nguồn sai lệch mà mAP trên `dev` không
có. `34` C1 in mức lệch và **chiều** lệch (thêm hay bỏ tổn thương) thay vì bắt lỗi; `TEST2` là nguồn
chuẩn cho tập đó.

Mỗi lần chấm `test406` được ghi vào `task2/_test_touches.jsonl`. Kết quả đó **chỉ để báo cáo**,
không được dùng để chọn arm / ngưỡng / siêu tham số.

### 18.1 · Tách hai hiệu ứng, và trần do nhiễu nhãn (`34` C1 + C4-bis đã chạy)

**Chuyên gia BỎ nhiều gấp 4 lần số lần THÊM** — `615` bỏ so với `151` thêm trên 766 ô lệch (7.26%).
Nhãn tự động **gọi thừa**; việc gán lại chủ yếu là dọn dương tính giả.

Nên prevalence phải tách làm hai bước, không gộp:

```
dev  (tu dong)        0.0913
test406 (tu dong)     0.2049    <- CHON ANH      x2.25
test406 (chuyen gia)  0.1610    <- GAN NHAN LAI  x0.79
```

Điều này **sửa hai dòng đọc sai** ở bảng §18: `Pneumonia` không hề "bị làm nghèo 0.3×" — nó được
chọn **1.58× thiên về** pneumonia theo nhãn tự động, rồi chuyên gia bỏ **93/115**. `Normal` cũng
vậy: chọn 0.46× nhưng gán lại 1.54×.

**Trần do nhiễu nhãn.** Lấy chính nhãn tự động làm dự đoán, chấm trên nhãn chuyên gia:

```
AP cua chinh TIN HIEU HUAN LUYEN (macro)   0.6528
vo dich Task 2                             0.5260  = 81% cua muc do
```

Nghĩa là **0.526 không phải 52.6% của hoàn hảo, mà là 81% của thứ tín hiệu huấn luyện cho phép**.
Nhiễu nhãn mới là ràng buộc chính của Task 2, không phải thị giác.

Trần theo nhãn lệch nhau rất xa, và ba nhãn thấp nhất đều là **gọi thừa thuần tuý** (cột thêm = 0):

| nhãn | auto → chuyên gia | thêm | trần |
|---|---|---|---|
| Lung Lesion | 31 → 6 | 1 | **0.137** |
| Pneumonia | 115 → 22 | **0** | **0.191** |
| Infiltration | 46 → 12 | **0** | **0.261** |
| Enlarged Cardiomediastinum | 127 → 118 | 44 | 0.474 |
| … | | | |
| Support Devices | 229 → 221 | 4 | 0.940 |
| Pleural Thickening | 21 → 22 | 1 | 0.957 |

Đó là ngôn ngữ báo cáo (`"cannot exclude pneumonia"`, `"possible infiltrate"`) bị NLP đọc thành
dương tính.

**KHÔNG phải trần tuyệt đối.** Mô hình cho điểm liên tục nên vẫn có thể xếp hạng tốt hơn chính nhãn
nhị phân đã dạy nó — nếu các ca dương-tính-giả không có dấu hiệu hình ảnh thật. **Nhãn nào ta vượt
mốc này là nhãn ta nhìn xuyên qua được nhiễu nhãn**; `34` C4-bis đếm đúng con số đó và ghi
`task2_label_ceiling.csv`.

**Hàm ý cho Stage 4:** với Task 2, **làm sạch nhãn có thể là đòn bẩy lớn hơn thị giác**. Ba nhãn
`Lung Lesion` / `Pneumonia` / `Infiltration` không thể vượt `0.26` chừng nào còn học từ nhãn tự động
nguyên bản. Trục nhãn của cohort cũ đã đóng, nhưng đây là biểu hiện **mới, đo được, trên 406 ảnh
chuyên gia đọc** — khác hẳn thứ đã thử trước đây.
