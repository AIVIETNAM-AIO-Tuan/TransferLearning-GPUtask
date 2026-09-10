# REVIEW · Nhãn lật trong cùng một ngày, và câu hỏi ED ↔ hosp có xung đột không

Bối cảnh: timeline của một `subject_id` (12 study trong 11 ngày) do `16_patient_timeline.ipynb`
in ra cho thấy day 0.00 có `Atelectasis, Consolidation, Lung_Lesion, Pleural_Effusion`,
rồi day 0.04 (~58 phút sau) thành `No_Finding` sạch trơn, rồi day 0.60 hiện lại
`Atelectasis + Pleural_Effusion`.

Trạng thái: **đánh giá, chưa sửa gì.** Không file nào bị thay đổi.

---

## Kết luận ngắn

Hai câu hỏi độc lập với nhau, và câu trả lời cho câu thứ hai **không giải thích** câu thứ nhất.

1. **Nhãn lật không liên quan gì tới EHR.** `01` lấy nhãn từ `mimic_cxr_jpg.chexpert`,
   join **chỉ trên `study_id`** (C5, cell 12). Nhánh ED/hosp chỉ sinh ra `setting`,
   `cur_hadm`, `cur_edstay` — không có đường nào chạm vào `y`. Dù join encounter sai
   hoàn toàn thì `Atelectasis` vẫn y nguyên.
2. **ED và hosp không mâu thuẫn nhau.** Chồng lấn `edstays` × `admissions` là *đúng thiết kế*
   với bệnh nhân nhập viện qua ED. Cái đáng lưu ý không phải xung đột dữ liệu, mà là
   `setting` trong `01` được định nghĩa bằng **thứ tự ưu tiên**, nên nhãn `ed`/`inpatient`
   có nghĩa hẹp hơn ta tưởng (§2 bên dưới).

---

## 1 · Vì sao nhãn lật trong 58 phút

Nhãn CheXpert là nhãn của **báo cáo chữ**, không phải của ảnh. Một `study_id` được
CheXpert labeler (NLP) đọc phần findings/impression rồi gán vector. Cùng một lồng ngực,
hai báo cáo khác nhau ⇒ hai vector khác nhau.

Xếp theo mức đóng góp cho chính trace này:

| # | Cơ chế | Vì sao khớp với trace |
|---|---|---|
| 1 | **Phim chụp lại có báo cáo hẹp.** Phim portable follow-up thường chỉ trả lời đúng một câu hỏi ("ống NKQ đúng vị trí, không tràn khí"). Labeler không thấy mention nào ⇒ `No_Finding=1`, còn lại 0. | Tràn dịch màng phổi không rút hết trong 58 phút. **Báo cáo** im lặng, không phải dịch biến mất. |
| 2 | **Báo cáo kiểu so sánh.** "So với phim trước, không thay đổi" — không nêu tên tổn thương ⇒ không có mention dương ⇒ tổn thương biến mất khỏi nhãn. Đây là nguồn nhiễu nhãn ngày-qua-ngày lớn nhất của MIMIC-CXR. | Giải thích trực tiếp kiểu bật/tắt lặp lại ở day 5.63 → 6.61 → 7.63. |
| 3 | **U-Ones (`-1 → 1`), do `01` chọn.** Mọi câu nước đôi ("possible atelectasis", "cannot exclude effusion") bị đẩy thành 1 cứng. | 4 nhãn dương ở day 0.00 rất có thể đến từ **một** câu hedge. Với U-Zeros/U-Ignore, trace này phẳng hơn nhiều. U-Ones khuếch đại biên độ ở **cả hai** phía cú lật. |
| 4 | **`No_Finding` không phải phủ định logic của 12 nhãn kia.** Nó là output riêng, dương khi báo cáo khẳng định phim bình thường. Một phim có thể `No_Finding=0` mà 12 nhãn cũng đều 0. | Cộng thêm: `01` đã **bỏ `Support Devices`**, nên phim kiểm tra ống/catheter mất đúng cái positive thật của nó, để lại hàng trống hoặc `No_Finding`. |
| 5 | **Tư thế/kỹ thuật chụp.** AP portable nằm vs PA đứng: dịch layering rất khó gọi trên phim nằm, và bóng tim bị phóng đại trên AP. | `+Cardiomegaly` day 4.28 → `-Cardiomegaly` day 7.63 ở bệnh nhân nội trú là dấu hiệu kinh điển của artifact chiếu AP, tim không to lên rồi nhỏ lại trong 3 ngày. |
| 6 | **Một phần là thay đổi thật.** Phù phổi/tràn dịch ở bệnh nhân nội trú có thể lên và hết trong 24h khi lợi tiểu. | `+Edema` day 2.44 → `-Edema` day 3.58 hoàn toàn hợp lý về lâm sàng. |

Nói gọn: cú lật **day 0.00 → 0.04 gần như chắc chắn là văn phong báo cáo**, còn các
cú lật cách nhau ≥ 1 ngày thì lẫn lộn giữa tiến triển thật và nhiễu nhãn.

### Vì sao điều này quan trọng với phần đang làm

`12`–`15` (loss screen, DBL weight, disease gate) đều coi nhãn mỗi study là ground truth
độc lập. Nếu tỉ lệ lật giữa hai study cách nhau < 6h là đáng kể, thì một phần trần
hiệu năng của các nhãn đuôi là **nhiễu nhãn**, không phải giới hạn mô hình — và nó
sẽ trông giống hệt "ý tưởng không chạy".

---

## 2 · ED có xung đột với hosp v3.1 không

**Không xung đột về mặt dữ liệu.** Lý do:

- Cùng một hệ thống bệnh viện, cùng `subject_id`. Với bệnh nhân nhập viện qua ED,
  `edstays.intime..outtime` chồng lên `admissions.admittime..dischtime` là **bình thường** —
  ED là cửa trước của lần nhập viện đó, không phải một sự kiện cạnh tranh.
- Date-shift theo từng bệnh nhân là **nhất quán giữa các module** MIMIC-IV hosp / ED và
  MIMIC-CXR cho các subject có mặt ở cả hai. Nên phép join theo khoảng thời gian không
  bị lệch mốc giữa nguồn ảnh và nguồn EHR.

**Cái thật sự đáng lưu ý là định nghĩa `setting`, không phải dữ liệu nguồn.**
`01` C4 (cell 10):

```python
setting = 'inpatient' if cur_hadm.notna() else 'ed' if cur_edstay.notna() else 'outpatient'
```

hadm **thắng** edstay. Hệ quả:

- `setting == 'ed'` thực chất nghĩa là *"đang ở ED và **không** có khoảng admission nào
  phủ mốc này"* — tức ED rồi cho về, chứ không phải "mọi phim chụp ở ED".
- Một phim chụp ngay tại buồng ED nhưng sau `admittime` sẽ bị đóng dấu `inpatient`.
  Trong trace này, `ed → inpatient` ở day 0.60 là **thời điểm `index_time` vượt qua
  `admittime`**, không nhất thiết là lúc bệnh nhân được đẩy lên khoa.

Điều này quan trọng vì chính markdown của `01` C4 nói `setting` vừa là **feature**
vừa là **biến phân tầng**, và §C yêu cầu mọi metric phải báo cáo theo tầng. Một tầng
được định nghĩa bởi ranh giới thời gian thì rò rỉ ngay tại ranh giới đó: nhóm `ed`
bị lấy mất đúng những ca nặng nhất (ca được nhập viện), nên `ed` trong dataset này
lệch về phía nhẹ hơn ED thật.

### Một lỗ hổng kiểm tra được, đã xác nhận

`01` C2 (cell 6) có tính `n_hadm_match`, `n_edstay_match`, `n_distinct_times`, và
markdown C2 viết *"count kept so §A2's assert is checkable"* / *"so §A0's assert is
checkable"*. **Không có cell nào assert hoặc thậm chí in ba cột đó.** Grep toàn notebook
chỉ ra chúng xuất hiện đúng một lần, trong câu SQL. Nghĩa là:

- nếu tồn tại admission chồng lấn (`n_hadm_match > 1`), `rn=1 ORDER BY admittime DESC`
  âm thầm chọn một cái, không ai biết;
- nếu một study có nhiều `study_datetime` khác nhau (`n_distinct_times > 1`),
  `index_time = MIN(...)` âm thầm chọn mốc sớm nhất, cũng không ai biết.

Cả hai đều có thể vô hại, nhưng hiện đang **không được kiểm chứng** dù notebook tự
nói là có.

---

## 3 · Đề xuất kiểm chứng (chưa chạy, chưa sửa gì)

Xếp theo tỉ lệ thông tin thu được / công bỏ ra:

1. **Đọc báo cáo của 2 study đầu.** Một lần nhìn là kết luận được ngay day 0.04 là
   phim kiểm tra ống hay báo cáo so sánh. Cần text report (file `.txt` của MIMIC-CXR,
   hoặc bảng radiology của MIMIC-IV-Note nếu bạn có quyền) — `mimic_cxr_jpg` trên
   BigQuery không có phần chữ.
2. **Đo nhiễu nhãn toàn cohort:** với mọi cặp study liên tiếp cùng bệnh nhân cách nhau
   < 6h, tính tỉ lệ có ít nhất một nhãn đổi, và tỉ lệ `No_Finding` bật lên ngay sau một
   study có ≥ 1 nhãn dương. Đây là ước lượng sàn cho nhiễu nhãn, tính hoàn toàn từ
   `cohort_base.parquet`, không cần ảnh, chạy vài giây.
3. **Bổ sung 3 assert còn thiếu ở `01`** (hoặc chỉ in ra, không cần rebuild cohort):
   phân phối của `n_hadm_match`, `n_edstay_match`, `n_distinct_times`.
4. **Định lượng rò rỉ ranh giới `ed`/`inpatient`:** đếm số study có `cur_edstay` **và**
   `cur_hadm` cùng không-null. Đó chính xác là số phim chụp trong ED nhưng đang được
   gán `inpatient`. Nếu con số này lớn, `setting` nên là 3 mức
   `ed_only / ed_admitted / ward` thay vì 3 mức hiện tại.
5. **Nếu (2) cho tỉ lệ lật cao:** cân nhắc so lại U-Ones với U-Ignore trên riêng các nhãn
   đuôi — nhưng đây là quyết định của bạn, và nó kéo theo phải rebuild `cohort_base`.

Mục 3 và 4 chỉ đọc, không đụng `cohort_v2/`. Mục 5 mới là thay đổi thật sự.

---

## 4 · KẾT QUẢ ĐO (đã chạy trên 211,130 study / 60,798 subject)

### 4.1 · Check A — ranh giới ED/inpatient: có thật, nhưng nhỏ

|  | n_studies | Atelectasis | Cardiomegaly | Pleural_Effusion | No_Finding |
|---|---|---|---|---|---|
| `ed_only` | 77,791 | 14.96 | 9.05 | 11.39 | 53.72 |
| `ed_admitted` | **3,871** | 16.64 | 11.55 | 10.75 | 50.09 |
| `ward` | 106,949 | 35.27 | 34.06 | 40.65 | 15.42 |
| `outpatient` | 22,519 | 12.51 | 16.29 | 17.46 | 39.90 |

- **3,871 study (1.83% cohort)** chụp trong ED nhưng bị dán `inpatient` = 4.7% mọi phim chụp trong một ED stay.
- Khoảng cách prevalence trung bình: `ed_admitted` vs `ed_only` = **0.94 pp**; vs `ward` = **10.44 pp**.
  Tức 3,871 phim đó **giống ED, không giống ward** → precedence dán nhãn sai thật.

**Nhưng hướng ảnh hưởng ngược với dự đoán ở §2.** Dự đoán cũ: nhóm `ed` bị lấy mất các
ca nặng nên lệch nhẹ. Sai. Trong MIMIC-IV `admittime` thường rơi *sau* khi làm xong
workup ở ED, nên hầu hết phim ED chụp **trước** `admittime` và vẫn ở lại nhóm `ed`
(77,791 phim). Chỉ 4.7% (phim chụp lúc boarding, sau `admittime`) bị kéo đi.
Hệ quả thật: **`ward` bị nhiễm 3.6% phim mang đặc tính ED**, còn nhóm `ed` gần như
không bị thiên lệch. Đây là chú thích cho §C, không phải lý do rebuild cohort.

### 4.2 · Check B — sàn nhiễu nhãn: 68% ở 41 phút

| bucket | n_pairs | % pairs | **pct_changed** | mean_gain | mean_loss | median gap |
|---|---|---|---|---|---|---|
| `<1h` | 1,062 | 0.71% | **68.0** | 0.69 | **1.04** | 0.69 h |
| `1-6h` | 9,204 | 6.12% | 80.4 | 0.95 | 1.16 | 3.46 h |
| `6-24h` | 34,278 | 22.80% | 88.7 | 1.28 | 1.23 | 16.1 h |
| `1-7d` | 43,414 | 28.88% | **89.4** | 1.36 | 1.25 | 47.7 h |
| `>7d` | 62,374 | 41.49% | 72.0 | 0.97 | 0.99 | 1994 h |

**Hai phim cách nhau trung vị 41 phút bất đồng ít nhất một nhãn 68% số lần.**
Ở khoảng đó bệnh lý đóng góp gần bằng 0, nên 68% ≈ nhiễu quy trình gán nhãn thuần túy.
Đỉnh là 89.4% ở `1-7d`, tức **phần "thay đổi thật" nhiều nhất chỉ khoảng 21 pp nằm
trên một sàn nhiễu 68 pp** — ~76% biên độ lật đã có sẵn ở 41 phút.

`mean_loss` (1.04) > `mean_gain` (0.69) ở bucket `<1h`: phim thứ hai **mất** nhãn.
Đúng chữ ký của báo cáo follow-up viết hẹp, không phải bệnh thuyên giảm.

`>7d` tụt về 72% không phải vì ít nhiễu hơn — đó là quần thể khác (phim ngoại trú
định kỳ, phần lớn hai đầu đều bình thường nên dễ đồng thuận).

### 4.3 · Check C — 903 artifact chứng minh được

- **collapse** (pos → No Finding sạch): 17,319 cặp = **11.52%** mọi cặp liên tiếp; 1,471 cặp trong <6h; 220 cặp trong <1h.
- **flicker** (pos → No Finding → pos trong ≤24h): **903 triples, 805 subject** (1.3% bệnh nhân). Span ngắn nhất **0.76 h** — ba study trong 46 phút, `A_pos=2 → 0 → C_pos=1`.

Mỗi flicker là artifact 100%: tổn thương không hết rồi quay lại trong 46 phút. 903 là
**cận dưới chứng minh được** của nhiễu nhãn — con số thật là 68% ở §4.2.

### 4.4 · Check D — nhiễu tập trung đúng vào các nhãn đuôi

| nhãn | prevalence % | n_pairs_pos | **unstable %** (<6h) |
|---|---|---|---|
| Pleural_Other | 1.19 | 156 | **82.1** ±6.0 |
| Fracture | 2.17 | 299 | **76.6** ±4.8 |
| Lung_Lesion | 3.22 | 429 | **75.3** ±4.1 |
| Pneumothorax | 5.11 | 2,510 | **45.7** |
| Consolidation | 6.72 | 1,359 | 82.0 |
| Enlarged_Cardiomediastinum | 7.34 | 1,685 | 80.5 |
| Pneumonia | 15.29 | 2,123 | 73.8 |
| Edema | 17.97 | 3,103 | 62.5 |
| Cardiomegaly | 22.54 | 3,536 | 72.6 |
| Lung_Opacity | 24.48 | 4,309 | 67.7 |
| Atelectasis | 25.02 | 4,509 | 67.3 |
| Pleural_Effusion | 26.85 | 4,667 | 54.7 |
| No_Finding | 32.78 | 3,292 | 71.1 |

**Hai chứng cứ nội tại chốt lại rằng đây là nhiễu báo cáo, không phải sinh lý:**

1. **`Fracture` 76.6% và `Lung_Lesion` 75.3%.** Gãy xương sườn và nốt/khối phổi là tổn
   thương **cấu trúc** — vật lý không thể xuất hiện rồi biến mất trong <6 giờ. Nên
   ~76% ở đây gần như toàn bộ là nhiễu gán nhãn. Đây là **chuẩn hiệu chỉnh nội tại**:
   *mức nhiễu thuần cho một nhãn hiếm ≈ 75%.*
2. **`Pneumothorax` 45.7% — ổn định nhất bảng**, dù nó lại chính là tổn thương *dễ thay
   đổi nhanh nhất* về mặt lâm sàng (dẫn lưu màng phổi). Lý do: tràn khí luôn được nêu
   tường minh trong mọi báo cáo phim chụp lại ("không tràn khí") — nó là **câu hỏi**
   của phim đó. Cùng logic cho `Pleural_Effusion` 54.7% và `Edema` 62.5%.

⇒ Độ ổn định tỉ lệ nghịch với **tần suất nhãn được nhắc tới trong báo cáo follow-up**,
không phải với tốc độ biến đổi của bệnh.

### 4.5 · Ý nghĩa cho `12`–`15`

Các nhãn đuôi mà loss screen / DBL weight / disease gate đang tối ưu — `Pleural_Other`,
`Fracture`, `Lung_Lesion`, `Consolidation`, `Enlarged_Cardiomediastinum` — có độ bất ổn
định giám sát **75–82%** trên phim gần như đồng thời.

- Một phần trần AP/F1 nhãn đuôi trong `12`–`15` là **nhiễu nhãn**, không phải giới hạn mô hình.
- Tệ hơn: điểm tối ưu của λ_w / trọng số DBL / ngưỡng gate tinh chỉnh trên các nhãn này
  sẽ **không ổn định giữa các seed/split**, đúng ở những nhãn bất ổn định nhất.
- Với `Pleural_Other` / `Fracture` / `Lung_Lesion`, câu hỏi "loss nào tốt hơn ở đuôi"
  có thể **không phân giải được** bằng dữ liệu này.

### 4.6 · Ba điểm dè dặt bắt buộc nêu

1. **Cặp <6h là quần thể chọn lọc** (ICU, hậu thủ thuật, chấn thương), không đại diện
   toàn cohort — con số bất ổn định có thể bị thổi lên. Phản biện: đó lại chính là
   nhóm phim nội trú nơi các nhãn đuôi của `12`–`15` sinh sống.
2. **`n_pairs_pos` của nhóm đuôi nhỏ** (156/299/429) — CI 95% khoảng ±4–6 pp. Đừng
   trích tới 0.1 pp, nhưng kết luận "rất cao" thì không lung lay.
3. **Chưa khử nhiễu do khác tư thế chụp.** Hai study có thể khác view (chỉ nghiêng vs
   PA+nghiêng), và điều đó *thật sự* làm đổi cái nhìn thấy được. Đây là confounder duy
   nhất có thể kéo con số xuống, và nó **kiểm được**: lọc lại chỉ giữ cặp mà **cả hai**
   study đều có view frontal (`view_map.parquet`), rồi chạy lại 4.4. Rẻ, không cần ảnh.

