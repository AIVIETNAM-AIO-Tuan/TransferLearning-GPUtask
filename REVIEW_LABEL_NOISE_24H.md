# REVIEW — Nhiễu nhãn đo bằng cặp phim ≤24h

| | |
|---|---|
| Ngày | 2026-07-29 |
| Nhánh | Độc lập với `REVIEW_DISEASE_GATE_WITH_TAIL_SELECTIVE_DBL.md` |
| Dữ liệu | val 31,670 study · 10,167 có hàng xóm ≤24h (32.1%) · median gap 11.20 h |
| Model dùng để đo | `B1` (gate+WASL) và `A1` (gate+ASL) của nb14/nb15, đã pool 7:3 |
| Nguồn nhãn | `mimic-cxr-2.0.0-chexpert.csv.gz` = `physionet-data.mimic_cxr_jpg.chexpert` |
| U-policy | **U-Ones** (`-1 → 1`, `blank → 0`), khoá ở `01_cohort_selection.ipynb` C5 |

---

## 0. Kết luận

**Trần của Rare-3 bị nhãn giữ, không phải bị loss hay kiến trúc fusion giữ.**

Cơ chế chi phối: **nhãn theo báo cáo nào liệt kê đầy đủ hơn, và bên nào đầy hơn là ngẫu
nhiên.** `blank → 0` biến sự im lặng của bác sĩ thành khẳng định âm tính. Không phải lỗi
của pipeline trong repo này, không phải lỗi labeler, và **không sửa được bằng U-policy**.

Quy mô: `Rare-3 mAP` bị kẹp trong `[0.074, 0.171]` tuỳ quy ước gán nhãn (`strict = 0.122`).
Biên độ `±0.05` — trong khi nb12–15 tranh nhau ở cổng `±0.002`, tức **~25 lần nhỏ hơn**.

---

## 1. Thiết lập

Với mỗi study val có ≥1 study khác **cùng bệnh nhân** trong vòng 24h, lấy study gần nhất
làm "hàng xóm". Nhãn nhị phân nên `Y_NB = Y` nếu đồng ý, `1 − Y` nếu không.

```
val                31,670 study
có hàng xóm <=24h  10,167  (32.1%)   median gap 11.20 h
forward thêm       12,584 / 51,880 anh  (24.3% val)
```

**Tập audit không đại diện**: bệnh nhân chụp lặp trong 24h lệch nặng về nội trú/ICU.
`strict macro_mAP = 0.4062` ở đây so với ~0.44 trên toàn val — tập này **khó hơn**.
Mọi số tuyệt đối dưới đây **không** chuyển sang được cho bảng val của nb14/nb15.

---

## 2. AUROC — nhiễu nhãn lớn hơn mọi thứ đang đo

`d_AUROC` = AUROC trên study có nhãn nhất quán − AUROC trên toàn tập audit:

| arm | macro d_AUROC | tail d_AUROC | `pct_cont_neg` |
|---|---:|---:|---:|
| `B1` (WASL) | **+0.0713** | +0.0546 | 0.706 |
| `A1` (ASL) | **+0.0710** | +0.0542 | 0.706 |

Chênh lệch lớn nhất giữa hai arm trên 13 nhãn là `0.0024`. **Lựa chọn loss không đụng gì
tới quan hệ giữa model và nhiễu nhãn.**

`pct_cont_neg = 0.706` — study bị gán **âm** nhưng hàng xóm nói dương thì được model chấm
cao hơn hẳn âm sạch (0.5 = giống âm thật). Bất đối xứng theo **cả hai chiều** (dương tranh
chấp bị chấm thấp, âm tranh chấp bị chấm cao) là chữ ký của nhãn không tin cậy, không phải
của độ khó.

### % nhãn **dương** bị tranh chấp (`n_pos_contested / n_pos`)

| nhãn | % | | nhãn | % |
|---|---:|---|---|---:|
| Enlarged_Cardiomediastinum | 75.2 | | No_Finding | 63.1 |
| **Pleural_Other** | **71.4** | | Pneumonia | 61.7 |
| Consolidation | 69.1 | | Cardiomegaly | 54.4 |
| **Fracture** | **67.8** | | Lung_Opacity | 51.4 |
| **Lung_Lesion** | **65.0** | | Atelectasis | 49.8 |
| | | | Edema | 46.5 |
| | | | Pleural_Effusion | 37.3 |
| | | | Pneumothorax | 35.6 |

Rare-3 trung bình ~68%, head ~52%.

---

## 3. AP / AUPRC — ba quy ước gán nhãn, cùng 10,167 study

`strict` = nhãn gốc · `union` = dương nếu study **hoặc** hàng xóm dương · `intersect` =
dương chỉ khi **cả hai** đồng ý.

| | macro mAP | Rare-3 mAP | NonTarget-10 | Tail AP | macro AUPRC | macro lift | Rare-3 lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| intersect | 0.2865 | 0.0737 | 0.3503 | 0.0498 | 0.2845 | 5.880 | **13.41** |
| **strict** | 0.4062 | **0.1220** | 0.4915 | 0.0779 | 0.4053 | 3.400 | 6.82 |
| union | 0.5062 | 0.1714 | 0.6066 | 0.1101 | 0.5056 | 2.815 | 5.73 |

Bootstrap paired ở mức study (1000 resample):

```
union     - strict   rare3 +0.0482 [+0.0341, +0.0634]   macro +0.0994 [+0.0942, +0.1048]
                     tail  +0.0314 [+0.0086, +0.0597]
intersect - strict   rare3 -0.0437 [-0.0640, -0.0194]   macro -0.1181 [-0.1253, -0.1103]
                     tail  -0.0206 [-0.0569, +0.0363]   <- CHUA 0, bo qua
```

### ⚠ Cảnh báo phương pháp — **AP không so ngang được qua prevalence**

Tập `clean/intersect` có prevalence thấp hơn hẳn, nhất là ở nhãn hiếm:

| nhãn | prev strict | prev intersect | tỷ lệ |
|---|---:|---:|---:|
| Pleural_Other | 0.895% | 0.259% | **0.29×** |
| Enlarged_Cardiomediastinum | 10.99% | 3.24% | 0.30× |
| Fracture | 2.02% | 0.667% | 0.33× |
| Lung_Lesion | 2.89% | 1.052% | 0.36× |
| Pleural_Effusion | 40.58% | 35.98% | 0.89× |

`AP` của bộ phân loại ngẫu nhiên **bằng đúng prevalence**. `AP` thô tăng theo prevalence,
`lift = AP/prev` giảm theo prevalence — **không cái nào bất biến**. Ba cột là **ba câu hỏi
khác nhau**, không phải ba ước lượng của cùng một đại lượng.

Đây là lý do audit gốc dùng AUROC là đúng: AUROC bất biến với prevalence.

### Ước lượng chất lượng xếp hạng theo nhóm (`Pleural_Other`, thang `lift`)

| nhóm | lift |
|---|---:|
| dương **đồng thuận** (26 ca) | **19.5** |
| dương theo nhãn gốc (91 ca) | 8.7 |
| âm **bị tranh chấp** (66 ca) | **≈ 5.0** |
| ngẫu nhiên | 1.0 |

(suy từ `(157×7.13 − 91×8.70)/66`; AP không tuyến tính nên là ước lượng thô)

Âm-bị-tranh-chấp được xếp hạng **cao gấp ~5 lần ngẫu nhiên** ⇒ model *có* nhìn thấy dấu
hiệu ở những ca report nói âm.

---

## 4. U-policy — **không phải** nguồn chính

`01_cohort_selection.ipynb` C5: `U_POLICY = 'ones'`, `v.replace(-1, 1)`, `fillna(0)`.
Nhãn thô `-1` **không còn** trong bất kỳ file nào; phải query lại BigQuery.
Assert `U-Ones(RAW) == Y` → `True`, xác nhận nguồn khớp cohort.

### % dương đến từ `-1`

| nhãn | % | | nhãn | % |
|---|---:|---|---|---:|
| **Enlarged_Cardiomediastinum** | **59.4** | | Cardiomegaly | 12.8 |
| **Pneumonia** | **53.0** | | Pneumothorax | 8.4 |
| Edema | 27.0 | | Pleural_Effusion | 7.8 |
| Consolidation | 21.4 | | Fracture | 6.3 |
| Pleural_Other | 17.6 | | Lung_Opacity | 6.1 |
| Atelectasis | 15.4 | | No_Finding | 0.0 |
| Lung_Lesion | 13.6 | | | |

**toàn bộ 17.2% · Rare-3 11.7%**

### % tranh chấp tách theo giá trị gốc

```
certain +1    55.7%  (Rare-3 66.5%)
uncertain -1  66.2%  (Rare-3 75.0%)     ty le chi 1.19x
blank -> 0    12.4%  (Rare-3  1.3%)
```

### Số học quyết định

```
tranh chap trong so:  0.172 x 0.662 + 0.828 x 0.557 = 57.5%
neu bo U-Ones (U-Zeros):                              55.7%
                                              chenh:   1.8 pp
Rare-3: 67.5% -> 66.5%,  chenh 1.0 pp
```

Đổi sang U-Zeros **vứt 17.2% nhãn dương** để đổi lấy **3% giảm nhiễu tương đối**. Trao đổi tệ.

`No_Finding` là minh chứng sạch nhất: **0%** dương từ `-1` (CheXpert không bao giờ đánh dấu
nghi ngờ cho nhãn này) mà vẫn tranh chấp **63.1%**. Không dính dáng gì tới U-policy.

**Nơi U-Ones thật sự gây hại**: chỉ `Enlarged_Cardiomediastinum` (59.4% dương từ `-1`,
tranh chấp 78.3%) và `Pneumonia` (53.0%). Nhưng ngay cả với chúng, `certain +1` đã tranh
chấp 70.7% / 60.3% — bỏ U-Ones không cứu được.

**Với Rare-3, U-policy gần như không liên quan.**

---

## 5. Nhiễu bộ trích NLP — đo bằng NegBio, gần như bằng 0

MIMIC-CXR-JPG ship **hai** bộ nhãn từ **cùng một report**: `chexpert` và `negbio`.
So chúng trên cùng study cô lập được nhiễu của bộ trích.

| nhãn | NLP bất đồng | hàng xóm 24h | tỷ lệ |
|---|---:|---:|---:|
| Cardiomegaly | 3.8% | 33.0% | 0.12 |
| Pneumothorax | 1.1% | 9.2% | 0.12 |
| Lung_Opacity | 1.7% | 33.5% | 0.05 |
| Edema | 1.2% | 24.3% | 0.05 |
| No_Finding | 0.9% | 19.2% | 0.05 |
| Pleural_Effusion | 1.1% | 29.3% | 0.04 |
| Consolidation | 0.6% | 13.7% | 0.04 |
| Atelectasis | 1.0% | 33.7% | 0.03 |
| Fracture | 0.1% | 2.7% | 0.03 |
| Pleural_Other | 0.0% | 1.3% | 0.03 |
| Pneumonia | 0.5% | 19.3% | 0.03 |
| Lung_Lesion | 0.1% | 3.7% | 0.02 |
| Enlarged_Cardiomediastinum | 0.2% | 16.0% | 0.01 |

⇒ **~90–97% bất đồng giữa hai phim là biến thiên report thật**, không phải lỗi labeler.

**Cận dưới, không phải ước lượng đúng**: CheXpert labeler và NegBio đều là hệ luật, cùng
danh sách từ khoá + quy tắc phủ định. Điểm mù chung không lộ ra ở phép so này.

Chi tiết đáng chú ý: `EC` có NLP bất đồng **thấp nhất** (0.2%) dù tỷ lệ `-1` **cao nhất**
(59.4%) — sự do dự nằm trong **câu chữ của bác sĩ**, cả hai labeler đọc ra `-1` giống nhau.

**Hệ quả**: `CheXpert ∩ NegBio` ≈ CheXpert (đồng ý 96–99.9%), **không** cho bộ nhãn sạch hơn.

---

## 6. Phân rã nguyên nhân

### 6.1 Thay từ trong họ đám mờ — **bác bỏ** (trừ Consolidation)

Với các ca dương-bị-tranh-chấp, hàng xóm có dương ở ≥1 nhãn cùng họ không?

| nhãn | contested | base rate | chênh |
|---|---:|---:|---:|
| Consolidation | 73.4% | 65.7% | **+7.7** |
| Pneumonia | 70.5% | 66.6% | +3.9 |
| Edema | 59.8% | 58.8% | +1.0 |
| Lung_Opacity | 58.3% | 58.0% | +0.3 |
| **Atelectasis** | 49.3% | 54.8% | **−5.5** |

Chỉ `Consolidation` ủng hộ. `Atelectasis` **ngược chiều** — khi hàng xóm bỏ nó, hàng xóm bỏ
luôn cả họ.

### 6.2 Bundling — bất đồng đi theo **chùm**

Số nhãn dương khác (bỏ nhãn `j` và `No_Finding`): study − hàng xóm.
Nhóm **đồng thuận** ≈ 0 ở **mọi** nhãn (`−0.06 … +0.04`) ⇒ cặp phim nói chung không lệch.

| nhãn | n | bundling | CI 95% |
|---|---:|---:|---|
| Pneumonia | 1005 | +0.82 | [+0.72, +0.92] |
| Pleural_Effusion | 1541 | +0.75 | [+0.67, +0.83] |
| Atelectasis | 1767 | +0.66 | [+0.59, +0.74] |
| Enlarged_Cardiomediastinum | 840 | +0.64 | [+0.52, +0.76] |
| Edema | 1248 | +0.57 | [+0.47, +0.67] |
| Consolidation | 706 | +0.51 | [+0.38, +0.64] |
| Lung_Opacity | 1731 | +0.48 | [+0.39, +0.56] |
| **Lung_Lesion** | 191 | **+0.46** | **[+0.21, +0.68]** |
| Cardiomegaly | 1708 | +0.41 | [+0.33, +0.49] |
| **Pleural_Other** | 65 | +0.17 | **[−0.34, +0.66]** |
| **Fracture** | 139 | +0.14 | **[−0.17, +0.44]** |
| **Pneumothorax** | 451 | **+0.00** | **[−0.16, +0.17]** |

- `Pneumothorax` — CI loại trừ mọi giá trị ≥ +0.21. Nhãn **duy nhất** thật sự cô lập.
- `Lung_Lesion` — CI loại trừ 0 ⇒ **có** bundling, và nó là Rare-3.
- `Fracture`, `Pleural_Other` — CI phủ cả 0 lẫn +0.66, **không kết luận được**.

⇒ **Không có bằng chứng nào cho thấy Rare-3 hành xử khác phần còn lại.**

`Pleural_Other` với `n=65` có CI `±0.50`. Muốn thu về `±0.15` cần ~720 dương tranh chấp;
toàn val chỉ có ~377 dương. **Không trả lời được bằng cách tiếp cận hàng xóm**, kể cả nới
cửa sổ lên 72h.

### 6.3 Kiểm gương — cơ chế là **đối xứng**

`ep` = (study dương, hàng xóm âm) · `en` = (study âm, hàng xóm dương)

| nhãn | ep | en | **tổng** |
|---|---:|---:|---:|
| Pneumonia | +0.82 | −0.80 | +0.02 |
| Enlarged_Cardiomediastinum | +0.64 | −0.60 | +0.04 |
| Pleural_Effusion | +0.75 | −0.70 | +0.05 |
| Lung_Opacity | +0.48 | −0.41 | +0.06 |
| Atelectasis | +0.66 | −0.60 | +0.07 |
| Cardiomegaly / Edema | +0.41 / +0.57 | −0.32 / −0.47 | +0.09 |
| Consolidation | +0.51 | −0.39 | +0.12 |
| Fracture | +0.14 | −0.19 | −0.04 |
| Lung_Lesion | +0.46 | −0.53 | −0.08 |
| Pneumothorax | +0.00 | +0.13 | +0.14 |
| Pleural_Other | +0.17 | +0.03 | +0.20 |

`tổng ∈ [−0.08, +0.12]` trong khi hai vế riêng là `±0.4 … ±0.8` ⇒ **gương gần như hoàn hảo**.

Nghĩa là: **bên nào liệt kê đầy hơn thì bên đó được gán dương**, và bên nào đầy hơn là
**ngẫu nhiên**. Không phải "hàng xóm bỏ sót" — mà là chọn mẫu theo độ đầy đủ báo cáo.

### 6.4 Chiều thời gian — có, nhưng nhỏ

```
toan bo 10,167 cap:  study 2.21 vs hang xom 2.17  | chenh +0.044
  study nhieu hon 38.0% | bang 25.8% | it hon 36.2%

study SOM hon : +0.121 (n=4,847)
study MUON hon: -0.025 (n=5,319)      chenh theo chieu thoi gian +0.146
```

Độ lệch nền `+0.048` do ghép cặp không hoàn toàn hai chiều
(`(4847×0.121 + 5319×(−0.025))/10166 = +0.0446` ✓ khớp kiểm không điều kiện).
Bất đối xứng thật `±0.073`, SE ≈ 0.028 ⇒ **~3.6 SE**, có thật.

Nhưng: mất cân bằng có hướng là `1.8pp` trên `74.2%` số cặp có chênh lệch
⇒ **~97.6% chênh lệch là KHÔNG có hướng**. Quy ước "no interval change" tồn tại nhưng chỉ
giải thích được vài phần trăm.

### 6.5 Bảng tổng hợp

| nguồn | phần đóng góp | bằng chứng |
|---|---|---|
| **độ đầy đủ báo cáo dao động, KHÔNG hướng** | **~85–95%** | §6.3 gương `tổng ≈ 0`; §6.4 97.6% không hướng |
| nhiễu bộ trích NLP | 1–12% (cận dưới) | §5 NegBio |
| U-Ones | ~3% tương đối | §4 |
| quy ước "no interval change" | ~2.4% | §6.4 |
| thay từ trong họ đám mờ | rất nhỏ, chỉ `Consolidation` | §6.1 |
| bệnh đổi thật | có cho `Pneumothorax`; không tách được cho `Atelectasis`/`Edema` | §6.2 |

---

## 7. Đánh giá bốn hướng can thiệp

| hướng | phần khối lượng nhãn chạm tới | đánh giá |
|---|---:|---|
| U-policy theo từng bệnh | 17.2% | đáng cho `EC`/`Pneumonia`; **vô nghĩa cho Rare-3** |
| label smoothing trên `-1` | 17.2% | vướng ASL: `γ = γ₊y + γ₋(1−y)` vô nghĩa với `y` không nhị phân |
| distillation (U-SelfTrained) | 17.2% | teacher **thừa hưởng** nhiễu 55.7% của nhãn chắc chắn; ô `-1` là chỗ teacher ngoại suy |
| **nhãn mềm / hạ trọng số từ hàng xóm** | nhắm đúng nguồn 85–95%, **nhưng chỉ phủ 32.1%** | hướng duy nhất đánh trúng |

Sạch hơn smoothing: **U-Ignore + mask trong loss** (`01` đã liệt kê sẵn tuỳ chọn
`'ignore'`). Với ASL chỉ là nhân mask rồi chia `mask.sum()` thay vì `numel()`.
Nhưng **eval cũng phải mask**, nếu không thì không so được với bảng cũ.

Nếu suy ra U-policy theo từng bệnh thì **tự suy từ dữ liệu này**, không mượn bảng của bài
CheXpert (tune trên validation set khác, report khác, nhóm bác sĩ khác).

---

## 8. Điều kiện tiên quyết — **bộ nhãn đánh giá cố định**

Mọi can thiệp ở §7 đều đổi prevalence, mà §3 đã chứng minh **AP không so ngang được qua
prevalence**. Trước khi thử bất cứ hướng nào, phải đóng băng **một** bộ nhãn đánh giá dùng
chung cho mọi biến thể.

Ba lựa chọn, đều có giá:

| | phủ | ghi chú |
|---|---|---|
| `intersect` (đồng thuận cặp ≤24h) | 32.1% study | lệch nội trú/ICU; "đồng ý" ≠ "đúng" |
| `CheXpert ∩ NegBio` | 100% | **loại** — hai bộ đồng ý 96–99.9%, không thêm gì (§5) |
| đọc lại thủ công vài trăm study | rất ít | chuẩn thật; **con đường duy nhất cho Rare-3** |

Với `Pleural_Other` (91 dương trong toàn tập audit) thì lựa chọn thứ ba có lẽ là duy nhất.

---

## 9. Hạn chế

- Tập audit **không đại diện** cho val (32.1%, lệch nội trú/ICU, `macro_mAP` .4062 vs ~.44).
- "Đồng ý" ≠ "đúng": hai study cùng đi qua một labeler, có thể cùng dính lỗi hệ thống.
  `intersect` là **cận trên lạc quan**.
- NegBio và CheXpert labeler **không độc lập** ⇒ §5 là cận dưới.
- `union` giả định hàng xóm dương là đúng; `intersect` vứt cả dương thật chỉ được báo cáo
  một lần. Không cái nào là sự thật.
- Cửa sổ 24h lẫn **biến thiên lâm sàng thật** cho nhãn động (`Atelectasis`, `Edema`,
  `Pneumothorax`). Nhưng `Cardiomegaly` (mạn tính) tranh chấp 52.7% ở `certain +1` — không
  thể giải thích bằng bệnh đổi ⇒ kết luận chung vẫn đứng.
- Toàn bộ ở 1 seed, val-only.

---

## 10. Các khẳng định đã rút lại

Ghi lại để không ai suy diễn lại từ đầu:

| khẳng định | trạng thái | bằng chứng bác bỏ |
|---|---|---|
| "`blank → 0` (bỏ sót) là nguồn chính cho nhãn hiếm" | **sai về tỷ lệ** | `blank` chỉ tranh chấp 1.3% ở Rare-3. Đúng ~một nửa về **số lượng tuyệt đối** (65 dương vs 66 âm tranh chấp cho `Pleural_Other`) |
| "thay từ trong họ đám mờ là cơ chế chính" (tôi xếp *tin cậy cao*) | **sai** | §6.1 — `Atelectasis` ngược chiều, `Lung_Opacity` bằng 0 |
| "với Rare-3 bất đồng là cô lập, khác nhóm bundling" | **sai** | §6.2 — `Lung_Lesion` CI `[+0.21,+0.68]` loại trừ 0; `Fracture`/`Pleural_Other` chỉ là thiếu lực |
| "nếu nhiễu NLP lớn thì dùng `CheXpert ∩ NegBio` phủ 100%" | **không áp dụng được** | §5 — nhiễu NLP 1–12%, giao ≈ CheXpert |
| "báo cáo theo dõi ngắn hơn một cách hệ thống" | **sai** | §6.3, §6.4 — cơ chế đối xứng, chỉ ~2.4% có hướng |

---

## 11. Hệ quả cho chương trình nb11–15

`B1 ≡ A1` trong bảng AUROC (chênh chữ số thứ tư) là hệ quả trực tiếp của §0: loss không
đụng tới nút thắt.

nb12 → nb15 đã tranh nhau ở cổng `±0.002` trên `Rare-3`, trong khi quy ước gán nhãn dịch
chính con số đó `±0.05`. **Mọi kết luận so sánh loss/kiến trúc trên Rare-3 đều nằm dưới sàn
nhiễu nhãn** và chỉ nên phát biểu ở dạng tương đối, có paired bootstrap, không bao giờ ở
dạng tuyệt đối.

Hướng tiếp theo có giá trị nhất **không** phải `B2`/`B3` mà là can thiệp vào **nhãn**.
Nhưng phải bắt đầu bằng §8 — đóng băng bộ nhãn đánh giá — nếu không thì lặp lại đúng cái
bẫy prevalence của §3.
