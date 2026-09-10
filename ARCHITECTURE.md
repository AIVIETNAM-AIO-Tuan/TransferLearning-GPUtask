# Kiến trúc: CXR × EHR FiLM Fusion trên MIMIC

> Tài liệu tự chứa — mô tả kiến trúc, dữ liệu và kết quả của pipeline `01`→`07`.
> Số liệu chốt tại lần chạy seed sweep (`06 · C12`, 3 seed).

## 1. Câu hỏi nghiên cứu

**Đặc trưng EHR có bổ sung cho embedding ảnh X-quang ngực không, đặc biệt ở các bệnh hiếm nơi tín hiệu hình ảnh yếu?**

Bài toán: phân loại **đa nhãn** (multi-label), 13 nhãn CheXpert, mất cân bằng nặng (prevalence 1.2% → 32.8%).

## 2. Dữ liệu

**Nguồn:** MIMIC-CXR-JPG (ảnh + nhãn) + MIMIC-IV v3.1 (EHR, qua BigQuery).

| | |
|---|---|
| Cohort | **211,130 study** / 346,578 ảnh |
| Split | subject-level 70/15/15 → train 147,791 · val 31,670 · test 31,669 |
| Nhãn | 13 (bỏ `Support Devices`), uncertain → 1 |
| Setting | inpatient 52.5% · ED 36.8% · outpatient 10.7% |

**Chia split bằng iterative stratification** (Sechidis 2011) ở mức `subject_id` — bắt buộc vì nhãn tích lũy theo bệnh nhân. Prevalence test khớp train tới 3 chữ số kể cả nhãn hiếm nhất.

### Nguyên tắc nhân quả (quan trọng nhất)

Mọi feature EHR gắn với một study **phải khả dụng với bác sĩ tại thời điểm đọc phim** (`index_time`). Cohort cũ bị loại bỏ vì `frac_studies_with_future_organism = 0.448` (join theo `subject_id` không neo thời gian). Cohort mới: **0.0000**.

**Hai đồng hồ khác nhau:**

- `charttime` = lúc lấy mẫu → *nghi ngờ lâm sàng*
- `storetime` = lúc có kết quả → *kết quả*

Khoảng cách này không phải nhiễu — nó là lý do `frac_specimen_sent_before = 78.8%` nhưng `frac_organism_known_before = 43.2%`. Chênh 35 điểm chính là **Tier 2**.

### Feature EHR

**Vi sinh — 4 tier theo độ mới** (deviation: plan gốc chia theo encounter; đổi vì ED/outpatient có `cur_hadm = NULL` nên **không bao giờ** chạm được T1):

| Tier | Điều kiện | Cung cấp |
|---|---|---|
| T1 | kết quả có, < 7 ngày | `org_ids`, `spec_ids`, status, recency |
| T2 | đã gửi mẫu, **chưa có kết quả** | `spec_ids` only — **có org ở đây là leakage** |
| T3 | kết quả có, 7–90 ngày | `org_ids`, recency |
| T4 | ≥ 90 ngày | chỉ đếm tổng, không danh tính |

- Vocab: **311 org**, **82 spec**; 6 token dự trữ (`PAD/UNK/MISSING/NO_GROWTH/CANCELLED/NEGATIVE`)
- `org_name = NULL` ⇒ **NO_GROWTH** (59% dòng thô, 84% specimen sau dedupe) — không tìm chuỗi `"NO GROWTH"`, MIMIC không ghi vậy
- Ragged (nhiều vi khuẩn/study) → `nn.EmbeddingBag(mode='sum')`

**Labs (7):** bun, creatinine, glucose, hemoglobin, platelet, wbc, **rbc** — cửa sổ `[index−24h, index]`, `storetime ≤ index`. Coverage any-lab **66.95%**. (`corr(rbc, hgb) = 0.872` → collinear mạnh nhưng không trùng.)

**Vitals (6):** temp_c, heart_rate, resp_rate, spo2, sbp, dbp — 3 nguồn: `icu.chartevents` (**itemid chính xác**, không LIKE) + `ed.vitalsign` + **`ed.triage`**. Coverage any-vital **68.06%**. Chuẩn hóa về **°C** (ED ghi °F).

**Comorbidity (2):** diabetes, hiv — ICD-9+10, neo `dischtime < index_time` (chỉ đợt **đã xuất viện**; ICD gán lúc xuất viện nên đợt hiện tại = leakage). `como_mask = 61.91%`.

**Hợp đồng value + mask (§A7):** mọi feature liên tục đi kèm mask. Impute median **nhưng không bao giờ bỏ mask** — "bác sĩ không thèm đo" tự nó là biến lâm sàng (MNAR).

## 3. Pipeline

```
01 cohort         → cohort_base.parquet (211k study, 13 nhãn, split)
02 EHR (BigQuery) → cohort_study_level.parquet + ehr_vocab.pkl + class_freq.pkl
03 view           → frontal/lateral resolution (purity ≥ 0.98)
04a preprocess    → 615GB → 16.9GB (347 tar shard, 560px grayscale)
04b LP-FT         → RAD-DINO backbone fine-tuned
05 embeddings     → x_cxr [211130, 768] (CLS, pool 7:3)
06 fusion         → EHR encoder → FiLM → head → 13 logits
07 export         → dataset phẳng (one-hot & embedding)
```

Thứ tự chạy thật: **03 → 01 → 02 → 04a → 04b → 05 → 06 → 07** (số file ≠ thứ tự chạy).

## 4. Kiến trúc model

### 4.1 Nhánh ảnh (04b → 05)

**RAD-DINO ViT-B/14 @ 518²** (1369 patch token), LP-FT:

- **LP** (4 epoch): đóng băng backbone, chỉ train head — tránh gradient ngẫu nhiên từ head random xóa biểu diễn pretrained
- **FT** (12 epoch): mở khóa, LR nhỏ hơn 100× (1e-5)
- **Head: ML-Decoder** (Alibaba-MIIL) — 13 class query cross-attend vào patch token, **không có self-attention** (theo thiết kế)
- **Loss: Distribution-Balanced Loss** (`ResampleLoss`, port verbatim từ `wutong16/DistributionBalancedLoss`)

**Multi-view 7:3** (train ở mức ảnh, đánh giá ở mức study):

```
p_study = ( w_f·mean(p|frontal) + w_l·mean(p|lateral) ) / ( w_f·[∃frontal] + w_l·[∃lateral] )
```

Mẫu số **renormalize** là bắt buộc: 51.6% study chỉ có frontal → nếu dùng tổng có trọng số, một nửa cohort bị nhân 0.7.

**Kết quả 04b:** test macro mAP **0.4199**, AUROC 0.8154. Sweep `w_f` xác nhận **0.7 tối ưu** (nhưng 0.6–0.8 là cao nguyên, không phân biệt được). Train−val gap **+0.042** → không memorize.

**05:** trích **CLS 768-d** mỗi ảnh (backbone đóng băng), pool 7:3 → `x_cxr` mỗi study. Chuẩn hóa bằng stats **train-only**.

> ⚠️ CLS (05) ≠ ML-Decoder-trên-patch (04b). Nên CXR-only ở 06 (~0.39) **thấp hơn** 0.42 của 04b — đây là bản chất, không phải regression.

### 4.2 Nhánh EHR (06)

```
EmbeddingBag(t1_org 16, t1_spec 8, t2_spec 8, t3_org 16)   # ragged, mode=sum
  ⊕ Embedding(setting 4, t3_recency_bin 4, t1_status 4)
  ⊕ labs(7 value ⊕ 7 mask) ⊕ vitals(6 ⊕ 6) ⊕ como(3)
  ⊕ counts, recency, binary flags
  → Linear(d, 256) → LayerNorm → GELU → Dropout(0.2)
  → Linear(256, 128) → LayerNorm → GELU
  → e [128]
```

**`d_e = 128` cố ý nhỏ**: generator phát ra 2×768, nên `e` rộng sẽ overfit nhiễu bệnh nhân.

**LayerNorm, không BatchNorm** — điều kiện hóa FiLM là **per-patient**; BatchNorm khiến γ,β của bệnh nhân này phụ thuộc bệnh nhân khác trong batch.

### 4.3 FiLM fusion

```python
γ = 1 + tanh(W_γ · e)     # tâm tại 1, chặn trong (0,2)
β = W_β · e
h = LayerNorm(γ ⊙ x_cxr + β)
logits = head(h)          # Linear(768,256) → LN → GELU → Dropout → Linear(256,13)
```

**γ, β là activation, không phải parameter** — chúng được sinh ra *per-patient* từ `e`. Nếu γ,β không phụ thuộc EHR thì FiLM thoái hóa thành affine thường và **không truyền tín hiệu EHR nào**. Đây là điểm ngược với γ,β của LayerNorm (là `nn.Parameter`, giống nhau cho mọi mẫu).

**Identity init** (zero-init lớp cuối generator): tại step 0, γ=1, β=0 **với mọi `e`** → fusion khởi động **đúng bằng CXR-only**, chỉ đi chệch ở nơi EHR chứng minh được giá trị. Đây là bảo đảm chống "fusion regression".

**Ba chế độ:**

- `none`: `head(norm(x))` — baseline CXR-only
- `concat`: `head([norm(x) | e])` — chỉ tổ hợp **cộng**
- `film`: `head(norm(γ⊙x+β))` — tương tác **nhân**

> Quyết định thiết kế (plan im lặng): `none` **dùng chung LayerNorm** với film, để identity gate đúng nghĩa đen (`allclose`), không phải xấp xỉ.

### 4.4 Loss

| loss | mô tả |
|---|---|
| **DB Loss** | rebalance weight theo co-occurrence + negative-tolerant regularization. Mặc định của plan. |
| **ASL** | Asymmetric Loss — γ_neg=4, γ_pos=1, asymmetric clip 0.05 |
| **BCE-posweight** | BCE + `pos_weight = (N−n_k)/n_k` |

## 5. Kết quả (3 seed, mean ± CI95)

```
none       0.3916 ± 0.0025    (CXR-only)
concat     0.4042 ± 0.0028
film_db    0.4059 ± 0.0051
film_bce   0.4152 ± 0.0008
film_asl   0.4202 ± 0.0041    ← tốt nhất
D3_labs    0.4045 ± 0.0034    (labs only, không micro)
D3_12      0.4043 ± 0.0037    (+ T1,T2)
```

### Đã xác lập

| so sánh | Δ | p | |
|---|---|---|---|
| concat − none | +0.0126 | 0.000 | ✅ **EHR bổ sung CXR** |
| film − none | +0.0143 | 0.002 | ✅ |
| **film − concat** | +0.0017 | **0.308** | ❌ **FiLM KHÔNG thắng concat** |
| **D3_12 − D3_labs** | −0.0002 | **0.901** | ❌ **micro KHÔNG thêm gì** |
| bce − db | +0.0094 | 0.014 | ✅ |
| asl − db | +0.0144 | 0.001 | ✅ |
| asl − bce | +0.0050 | 0.029 | ✅ |

### Loss × tier

| loss | head | medium | tail |
|---|---|---|---|
| db | 0.5829 ±.0007 | 0.2229 ±.0098 | 0.0814 ±.0153 |
| bce | 0.5931 ±.0004 | 0.2339 ±.0026 | 0.0771 ±.0133 |
| **asl** | **0.5979** ±.0024 | **0.2387** ±.0073 | **0.0843** ±.0054 |

**DB Loss thua ở chính lý do nó được chọn** — plan chọn nó *vì tail* (§B8), nhưng nó không thắng tier nào. Tuy nhiên **tail không phân định được ai cả**: CI chồng lấn (±0.013–0.015), vì tier tail chỉ có **1 nhãn** (Pleural_Other, 379 ca dương).

### Ablation khác

| | kết quả |
|---|---|
| `gating_only` (γ=sigmoid, β=0) | **0.3936** — kém hẳn full FiLM |
| `h_concat_e` | 0.4073 — tweak tốt nhất |
| identity init on/off | 0.4037 / 0.4033 — không đổi *đích* (quan trọng ở *xuất phát*) |
| EmbeddingBag vs one-hot | 0.4037 / 0.4045 — **hòa** (capacity y hệt) |
| **vitals** on/off | 0.4037 / 0.4037 — **không thêm gì** |
| **comorbidity** on/off | 0.4037 / 0.4027 — trong nhiễu |
| setting stratification | Δ dương ở **cả 3 setting** → không phải lối tắt |

## 6. Kết luận chính

1. **EHR bổ sung cho CXR** (+0.013–0.014, p<0.01), gain tương đối lớn nhất ở lớp hiếm (tail +32–34%).
2. **Nguồn tín hiệu là LABS** — không phải vi sinh (p=0.901), không phải vitals (Δ=0.0000), không phải bệnh nền (+0.001).
3. **Kiến trúc fusion không quan trọng** — FiLM ≈ concat (p=0.308). Giả thuyết trung tâm của plan thất bại. Concat đơn giản tốt ngang FiLM cầu kỳ.
4. **Loss quan trọng ngang toàn bộ công sức EHR**: đổi db→asl = **+0.0144**, thêm toàn bộ EHR = **+0.0143**.

## 7. Câu hỏi mở

### ⚠️ Confound chưa giải quyết — ma trận 2×2 thiếu một ô

| | db | asl |
|---|---|---|
| none | 0.3916 ✅ | **❓ chưa đo** |
| film | 0.4059 ✅ | 0.4202 ✅ |

Toàn bộ gain EHR đo tại `loss=db` — **loss tệ nhất trong ba**. Hai con số bằng nhau đáng ngờ: EHR +0.0143, loss swap +0.0144.

- **Nếu `EHR gain @ asl ≈ +0.014`** → cộng tính, kết luận #1 đứng vững.
- **Nếu `≈ 0`** → EHR và ASL vá **cùng một khiếm khuyết**; cái +0.0143 là artifact của baseline yếu, và kết luận #1 sụp.

### Pneumonia — chưa xác lập

Micro cho `+0.0085 ± 0.0041` (p=0.016) trên Pneumonia — nhãn **duy nhất** p<0.05, đúng nhãn nhiễm trùng mà lý luận lâm sàng dự đoán. Nhưng:

- **Trượt Bonferroni** (13 nhãn cần p<0.0038)
- **1/13 nhãn ở p<0.05 đúng bằng tỉ lệ ngẫu nhiên** (0.05×13 ≈ 0.65)
- Giả thuyết được nêu **sau khi** nhìn số → post-hoc

Phép thử sạch: nhiều seed, **chỉ** D3_labs vs D3_12, **chỉ** nhìn Pneumonia (giả thuyết nêu trước → không phải trả thuế Bonferroni).

## 8. Deviation so với plan (có chủ ý)

| # | plan | thực tế | lý do |
|---|---|---|---|
| 1 | tier theo encounter | **tier theo recency** | ED/outpatient có `cur_hadm=NULL` → `has_t1 = 0.000`. Sau khi đổi: ED T1 0.0000 → 0.0695 |
| 2 | `storetime` null → gán end-of-day | **loại bỏ** (0.18%) | bịa giờ tạo artifact không truy vết được |
| 3 | `assert 0.40 ≤ frac_no_micro ≤ 0.60` | **report, assert §A1 cấu trúc** | metric trộn đơn vị, và band trôi theo chính tham số của ta |
| 4 | 6 labs | **7** (+rbc) | có trong dataset cũ; `corr=0.872` không đủ để gọi là trùng |
| 5 | không có vitals/comorbidity | **thêm** | D3 chỉ ra tín hiệu là sinh lý cấp |
| 6 | `fusion=none` = `head(x)` | `head(norm(x))` | để E2.1 identity gate đúng nghĩa đen |

## 9. Bài học kỹ thuật đáng nhớ

- **Port, đừng tự dẫn lại** (§B8). Bản ASL tự viết dùng `p*(1-y)` thay vì `xs_neg = 1-p` → đảo ngược focal ở nhánh âm: easy-negative bị nhồi loss, hard-negative bị bỏ qua **hoàn toàn**. Sụp 0.4036 → 0.2778. Nhánh dương đúng nên bug ẩn.
- **Itemid phải được verify, không hardcode mù.** `LIKE '%heart rate%'` khớp cả `220046 Heart rate Alarm - High`. Coverage **không đổi** (ai có alarm cũng có HR thật) nhưng **giá trị** bị hỏng — query lấy dòng gần nhất, nên ngưỡng alarm 120 lọt vào `vital_heart_rate`. Không assert nào nổ.
- **Kỳ vọng có thể bị bác bỏ có giá trị.** Quy tắc tự đặt "ED phải >0.9" fail ở 58.9% → dẫn tới phát hiện thiếu bảng `ed.triage` (+37.8pp).
- **Cell phải tự chứa.** Mảnh SQL/biến sống ở cell này, cell khác dùng → stale-state trap (dính 4 lần với `MICRO_CTE`/`SX`, 2 lần nữa với `ED`/`ICU`, `vval`).
- **So sánh phải cùng đơn vị.** 59% (dòng) vs 84% (specimen) là *cùng một sự thật*; đem assert cái này bằng cái kia là sai.
