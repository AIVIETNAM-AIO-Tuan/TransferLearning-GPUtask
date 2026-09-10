# KẾ HOẠCH CHI TIẾT V2 — 3-ARM NO-TEXT LABEL-RELIABILITY EXPERIMENT

## 0. Câu hỏi nghiên cứu

Với cùng một mô hình CXR mạnh, liệu thay đổi cách sử dụng nhãn report-derived có giúp:

1. giảm tác hại của việc coi `blank` là âm tính;
2. giảm gradient từ các nhãn mâu thuẫn giữa hai phim gần nhau;
3. cải thiện Rare-3 mà không làm hại NonTarget-10?

Không dùng:

```text
report text
report embedding
expert relabeling
pseudo-label từ external model
WASL / DBL
EHR augmentation
```

---

# 1. Thiết kế tổng quát

## 1.1 Kiến trúc khóa

```text
RAD-DINO patch tokens
→ ML-Decoder 13 labels
→ train ở mức ảnh
→ validation/test pool ở mức study, frontal:lateral = 7:3, có renormalization
```

Khởi tạo cả ba arm từ cùng checkpoint H5.

Để cô lập label policy:

```text
shared frozen RAD-DINO blocks 0–8
separate trainable blocks 9–11 + ML-Decoder cho mỗi arm
epoch 1: decoder frozen nếu cần tái hiện schedule cũ
epoch 2+: decoder mở
```

Không dùng EHR trong screen đầu tiên. Nếu label policy thắng, chạy một confirmatory transfer lên multimodal model sau đó.

## 1.2 Các arm

### N0 — Historical baseline

```text
raw +1 → 1
raw -1 → 1
raw  0 → 0
raw NaN → 0
plain ASL
```

Đây phải tái hiện baseline hiện tại.

### N1 — Partial-label ASL

Đối với 12 pathology labels:

```text
raw +1 → y=1, mask=1
raw  0 → y=0, mask=1
raw -1 → mask=0
raw NaN → mask=0
```

Đối với `No_Finding`:

```text
raw +1 → y=1
mọi trạng thái khác → y=0
mask=1
```

Không dùng `No_Finding` để suy ngược các pathology labels.

### N2 — N1 + temporal reliability

Supervised labels giống N1.

Chỉ can thiệp pair-wise trên Rare-3:

```text
Pleural_Other
Fracture
Lung_Lesion
```

Với cặp phim đủ điều kiện:

```text
known / unknown:
giữ weight của known label = 1
unknown vẫn không có supervised target
dùng prediction phía known làm stop-gradient teacher nhẹ cho phía unknown

known positive / known negative:
hạ weight supervised của cả hai xuống eta = 0.5
không dùng consistency

unknown / unknown:
không loss pair
```

Không đổi nhãn 0↔1.

---

# 2. Prerequisite data audit

## 2.1 Khôi phục raw four-state labels

Nguồn:

```text
mimic-cxr-2.0.0-chexpert.csv.gz
```

Merge bằng `study_id`.

Lưu:

```text
y_raw [N,13] ∈ {-1,0,1,NaN}
y_n0  [N,13] ∈ {0,1}
y_n1  [N,13] ∈ {0,1}
m_n1  [N,13] ∈ {0,1}
```

Assertions:

```text
one row per study_id
no missing study in cohort
column order exactly equals LABELS
U-Ones(y_raw) == existing cohort labels
```

## 2.2 Báo cáo coverage trước khi train

Cho từng label và split:

```text
n(+1)
n(0)
n(-1)
n(blank)
known rate = n(+1 or 0) / N
definite-positive prevalence
definite-negative prevalence
```

Hard block:

```text
n(+1) < 200 trong TRAIN cho một Rare-3 label
→ không được claim class-specific result cho label đó
```

Warning:

```text
known negative rate < 5%
→ N1 có nguy cơ thiếu negative supervision
```

Không thay policy dựa trên VAL.

## 2.3 Không recompute tier

Rare-3, NonTarget-10 và Worst-3 giữ nguyên theo `class_freq.pkl` lịch sử.

Không phân tier lại từ definite-label counts.

---

# 3. Pair manifest cho N2

## 3.1 Chỉ dùng TRAIN subjects

Tuyệt đối không dùng cặp VAL/TEST để tạo training loss.

## 3.2 Xây pair

Trong mỗi subject:

```text
sort studies theo exact index_time
tạo các cặp chronological-adjacent
giữ pair nếu 0 < delta_t <= 24h
```

Không dùng all-pairs để tránh một ICU patient tạo hàng chục cạnh.

Lưu unordered pair:

```text
subject_id
study_a
study_b
delta_hours
hadm_a / hadm_b
setting_a / setting_b
view_signature_a / view_signature_b
raw labels của hai study
```

## 3.3 Pair eligibility cho consistency — sửa đổi V2

### Điều kiện bắt buộc số 1: cùng bệnh nhân

Mọi temporal pair phải thỏa:

```text
subject_id_a == subject_id_b
```

`same setting` không bao giờ được dùng để ghép hai bệnh nhân khác nhau.

Setting chỉ là bối cảnh chăm sóc:

```text
inpatient
ED
outpatient
```

và chỉ có thể dùng như một điều kiện phụ sau khi đã xác nhận cùng bệnh nhân.

### Primary pair eligibility

Một pair được dùng trong N2 primary khi:

```text
cùng subject_id
0 < delta_t <= 24h
hai study là chronological-adjacent
same view signature
timestamp có giờ/phút đủ chính xác
không phải cùng study_id
```

Ngoài ra phải xác nhận cùng một care episode bằng thứ tự ưu tiên:

```text
1. cùng hadm_id, nếu cả hai có hadm_id;
2. với ED, cùng ed stay_id nếu có;
3. với ICU, cùng icu stay_id hoặc cùng hadm_id;
4. nếu không có bất kỳ encounter identifier đáng tin nào:
   loại khỏi primary pair loss.
```

Không dùng `same setting` đơn độc làm proxy cho cùng encounter trong primary analysis.

### Sensitivity-only fallback

Các pair:

```text
cùng subject_id
cùng setting
delta_t <= 12h
nhưng thiếu encounter identifier
```

chỉ được giữ trong một sensitivity analysis riêng. Chúng không tham gia primary N2 training.

`view_signature`:

```text
F
L
FL
```

Không yêu cầu số ảnh giống nhau, chỉ orientation set giống nhau.

Pair khác view, khác encounter hoặc không xác minh được care episode vẫn có thể tồn tại trong audit, nhưng:

```text
pair consistency weight = 0
pair contradiction downweight = 0
```

## 3.4 Subject balancing

Nếu subject có `K` eligible pairs:

```text
base_pair_weight = 1 / K
```

Tổng đóng góp pair của một subject không vượt quá một subject có một pair.

Không để bệnh nhân chụp lặp nhiều thống trị N2.

---

# 4. Label states trong N2

Cho mỗi Rare-3 label:

```text
P = raw +1
N = raw 0
U = raw -1 hoặc blank
```

## 4.1 Bảng xử lý

| Pair state | Supervised weight | Pair loss |
|---|---:|---|
| P / P | 1, 1 | 0 |
| N / N | 1, 1 | 0 |
| P / U | 1, 0 | teacher P → U |
| U / P | 0, 1 | teacher P → U |
| N / U | 1, 0 | teacher N → U |
| U / N | 0, 1 | teacher N → U |
| P / N | 0.5, 0.5 | 0 |
| N / P | 0.5, 0.5 | 0 |
| U / U | 0, 0 | 0 |

Lý do không dùng consistency cho P/N:

```text
có thể là clinical change thật
hai supervised labels xung đột
ép hai prediction giống nhau sẽ tạo mục tiêu không xác định
```

## 4.2 Time weight

\[
w_{\text{time}} = 2^{-\Delta t / 12}.
\]

Tức:

```text
0h  → 1.00
12h → 0.50
24h → 0.25
```

Pair weight cuối:

\[
w_{\text{pair}}=
w_{\text{subject}}
\cdot
w_{\text{time}}
\cdot
w_{\text{view/context}}.
\]

Trong primary experiment:

```text
w_view/context = 1 nếu pair eligible
w_view/context = 0 nếu không eligible
```

---

# 5. Loss

## 5.1 Element-wise ASL

ASL phải trả về tensor `[B,13]` trước reduction.

\[
L_{\text{sup}}
=
\frac{
\sum_{i,c} w_{i,c} \ell^{ASL}_{i,c}
}{
\max(\sum_{i,c}w_{i,c},1)
}.
\]

Arm weights:

```text
N0: w = 1 cho mọi ô
N1: w = m_n1
N2: w = m_n1, sau đó P/N eligible Rare-3 → 0.5
```

Không chia cho `B×13` ở N1/N2.

## 5.2 Pair teacher loss

Với pair known/unknown:

```text
p_teacher = sigmoid(z_known).detach()
p_student = sigmoid(z_unknown)
```

Dùng:

\[
L_{\text{pair}}
=
\frac{
\sum w_{\text{pair}}
(p_{\text{student}}-p_{\text{teacher}})^2
}{
\max(\sum w_{\text{pair}},1)
}.
\]

Chỉ tính trên Rare-3 và pair state P/U hoặc N/U.

Không teacher ở cả hai chiều cùng lúc.

## 5.3 Tổng loss N2

\[
L_{N2}=L_{\text{sup}}+\lambda_{\text{pair}}L_{\text{pair}}.
\]

### Chọn lambda bằng gradient ratio

Không sweep bằng VAL.

Trên một calibration batch TRAIN cố định tại init:

\[
\rho=
\frac{
\|\nabla_\theta \lambda L_{\text{pair}}\|
}{
\|\nabla_\theta L_{\text{sup}}\|
}.
\]

Khóa:

```text
target rho = 0.10
```

Đo trên trainable tail blocks + decoder.

Clamp:

```text
lambda_pair ∈ [0.01, 1.0]
```

Nếu không có eligible pair-label trong calibration batch, dùng nhiều batch cho tới tối đa 20; nếu vẫn không có, N2 không đủ dữ liệu để chạy.

## 5.4 Warm-up

```text
epoch 1:
pair loss = 0
P/N downweight chưa bật

epoch 2:
0.5 × lambda_pair
P/N weight chuyển từ 1 → 0.75

epoch 3+:
full lambda_pair
P/N weight = 0.5
```

Mục tiêu: tránh teacher prediction ngẫu nhiên đầu training.

---

# 6. Data loaders

## 6.1 Main image loader

Cả ba arm dùng:

```text
same image order
same augmentation seed per image
same natural sampler
same effective batch size 256
```

Mọi ảnh của một study nhận cùng label state.

Train supervised ở image level như pipeline cũ.

## 6.2 Pair loader N2

Mỗi item là hai studies.

Collate:

```text
flatten toàn bộ ảnh của study A và B
forward images
pool probability/logit theo study bằng 7:3 renormalized rule
tính pair loss trên study-level prediction
```

Pair batch:

```text
16–32 pairs tùy VRAM
```

Dùng một pair batch mỗi hai main batches.

Không tính thêm supervised loss trên pair loader; nếu không N2 sẽ được oversample supervised ngoài ý muốn.

## 6.3 Multiple images cùng orientation

Trong pair branch:

```text
mean trong orientation trước
sau đó frontal/lateral 7:3
```

Giống validation.

---

# 7. Model schedule

```text
epochs: 8
seed: 42, 1337, 2024
natural sampling
ASL params giống baseline
```

Khởi tạo:

```text
copy cùng H5 state cho N0/N1/N2
assert max parameter difference = 0
```

Schedule:

```text
epoch 1:
tail blocks train theo schedule hiện tại
decoder frozen nếu baseline cũ frozen

epoch 2:
decoder unfreeze

epoch 1–8:
same optimizer, LR, weight decay, scheduler
```

Mỗi arm có optimizer/scheduler riêng.

Log mỗi epoch:

```text
train supervised loss
pair loss
effective known cells
effective positive/negative cells
gradient norms
prediction entropy
Rare-3 positive rate
```

---

# 8. Evaluation — labels phải khóa

## 8.1 Endpoint H — historical

Dùng đúng binary labels cũ:

```text
U-Ones
blank → 0
toàn bộ VAL
```

Metrics:

```text
Macro mAP
Rare-3 mAP
NonTarget-10
Tail AP
Worst-3 fixed
Macro AUROC
Rare-3 AUROC
Recall@K
```

Không đổi endpoint này.

## 8.2 Endpoint K — known-only

Dùng raw labels:

```text
evaluate chỉ tại raw +1 và raw 0
raw -1 / blank masked
```

Tất cả arms dùng cùng một frozen mask và cùng prevalence trong endpoint này.

### Primary metrics

```text
Macro mAP_K
Rare-3 mAP_K
NonTarget-10 mAP_K
per-label AP_K
```

Có thể báo cáo thêm trapezoidal AUPRC_K, nhưng không dùng đồng thời mAP và AUPRC như hai
primary endpoints vì chúng đo gần cùng một đối tượng. `average_precision_score` được khóa
làm primary implementation.

### Secondary metrics

```text
Macro AUROC_K
Rare-3 AUROC_K
per-label AUROC_K
Brier score_K
calibration slope/intercept nếu đủ positives
```

AUROC là robustness diagnostic, không phải mục tiêu tối ưu chính.

Không so AP_K với historical AP vì hai endpoint có label mask và prevalence khác nhau.
Chỉ so các arm với nhau bên trong Endpoint K, nơi mask được giữ cố định.

## 8.3 Endpoint T — temporal diagnostic

Dùng fixed VAL pair manifest, không train từ nó.

Stratify:

```text
same-view/same-context
different-view
0–6h
6–12h
12–24h
```

Report:

```text
AUROC trên known-concordant labels
score of contested positive
score of contested negative
pct_cont_neg
mean |p_a-p_b| cho Rare-3
P/U teacher-direction agreement
```

Tập này là diagnostic, không phải gold standard.

## 8.4 Test

TEST khóa tới khi:

```text
code hoàn tất
arm definitions khóa
checkpoint rule khóa
seed runs hoàn tất
```

Sau đó chạy một lần.

---

# 9. Checkpoint selection

Để tránh đổi endpoint sau khi thấy kết quả, báo cáo hai selection rules.

## Rule S-H — historical compatibility

Chọn epoch có:

```text
highest historical VAL Macro mAP
```

Tie trong 0.0002:

```text
chọn epoch sớm hơn
```

## Rule S-K — noise-aware sensitivity

Chọn epoch có:

```text
highest known-only VAL Macro mAP_K
```

Tie trong 0.0002:

```text
chọn epoch sớm hơn
```

Safeguard:

```text
historical Macro mAP không thấp hơn best của arm quá 0.002
```

AUROC_K không được dùng để chọn checkpoint.

Kết luận chỉ được gọi là robust khi cùng chiều dưới cả S-H và S-K.

---

# 10. Primary contrasts và go/no-go

## 10.1 Contrasts

```text
C1 = N1 - N0
C2 = N2 - N1
C3 = N2 - N0
```

## 10.2 Success gate N1

N1 là positive nếu:

```text
Rare-3 mAP_K tăng ít nhất +0.005
hoặc Macro mAP_K tăng ít nhất +0.003
```

đồng thời:

```text
historical Macro mAP >= N0 -0.002
historical NonTarget-10 >= N0 -0.002
known-only NonTarget-10 mAP_K không có broad harm
```

Cùng chiều ở ít nhất 2/3 seeds và subject-cluster bootstrap không cho thấy material harm.

AUROC_K được báo cáo để xác nhận ranking direction, nhưng không thể tự mình làm arm
đạt success gate.

## 10.3 Success gate N2

N2 là positive nếu:

```text
Rare-3 mAP_K >= N1 +0.005
```

hoặc đạt mechanistic-support gate:

```text
Rare-3 pair disagreement giảm >=10% tương đối
Rare-3 mAP_K không giảm
Rare-3 AUROC_K không giảm
```

Đồng thời:

```text
historical Macro mAP >= N1 -0.002
historical NonTarget-10 >= N1 -0.002
known-only NonTarget-10 mAP_K không có broad harm
```

Pair diagnostic-only improvement không được gọi là classification improvement.

---

# 11. Statistics

```text
3 paired seeds
same initialization mapping
same data order
```

Paired bootstrap:

```text
2,000 resamples
cluster = subject_id
```

Pair metrics cũng bootstrap theo subject, không theo pair hoặc study độc lập.

Report:

```text
mean seed delta
seed SD
cluster-bootstrap CI
```

Không dùng unpaired Welch test.

---

# 12. Edge cases và xử lý bắt buộc

## E1 — batch không có supervised cell

Nếu:

```text
sum(weights) = 0
```

thì:

```text
skip supervised backward cho arm đó
không tạo NaN
log zero-supervision batch
```

Nếu >1% batches xảy ra, dừng và sửa sampler.

## E2 — batch không có eligible pair-label

```text
L_pair = 0
không chia 0
không update lambda
```

## E3 — blank chiếm gần toàn bộ negative labels

Báo cáo effective negative count theo class.

Nếu một Rare-3 class có:

```text
<1,000 known negatives TRAIN
```

N1/N2 không đủ negative supervision để đưa ra claim cho class đó.

Không tự thêm blank-negative sau khi nhìn VAL.

## E4 — `No_Finding`

- Giữ binary baseline cho chính `No_Finding`.
- Không dùng nó để điền pathology negatives.
- Không thêm mutual-exclusion loss.
- Report tỷ lệ model dự đoán `No_Finding` cùng pathology positive.

## E5 — Support Devices đã loại khỏi 13 labels

Không suy `No_Finding` từ 12 pathology labels vì một study có thể abnormal chỉ do device.

## E6 — timestamp thiếu hoặc chỉ có ngày

Không dùng study đó trong pair loss.

Vẫn dùng supervised N0/N1/N2 bình thường.

## E7 — duplicate time / duplicate study

```text
same study_id → loại self-pair
same subject + exact same timestamp → loại khỏi pair loss
```

Có thể giữ trong supervised data nếu là study khác thật.

## E8 — nhiều study trong 24h

Chỉ dùng chronological-adjacent edges.

Normalize tổng pair weight theo subject.

## E9 — khác view composition

Không consistency và không contradiction downweight trong primary N2.

Vẫn giữ để làm diagnostic subgroup.

## E10 — clinical change thật

Giảm rủi ro bằng:

```text
Rare-3 only
time decay
same view
same encounter/setting
no consistency for explicit P/N conflict
```

Không áp pair loss cho:

```text
Edema
Atelectasis
Pneumothorax
```

trong primary experiment.

## E11 — pair teacher sai

Known-side prediction là teacher, không phải raw label value.

Safeguards:

```text
stop-gradient
small rho=0.10
warm-up
same-context pairs
```

Không hard pseudo-label.

## E12 — collapse do consistency

Monitor:

```text
prediction variance
mean entropy
positive prediction rate
|p_a-p_b|
```

Nếu pair disagreement giảm nhưng AUROC và variance cùng giảm mạnh, đó là collapse.

## E13 — arms có khác effective optimization steps

Main supervised optimizer steps phải giống nhau.

Pair loss được cộng vào step N2, không tạo optimizer step riêng.

## E14 — pair loader oversamples rare subjects

Subject-normalized weights và một pair tối đa mỗi subject trong một pair batch.

## E15 — leakage qua VAL pair manifest

TRAIN pair manifest chỉ từ TRAIN.

VAL/TEST pair manifest chỉ dùng evaluation.

## E16 — changing prevalence

Không thay evaluation labels.

Không so AP giữa Endpoint H và Endpoint K.

## E17 — raw label merge mismatch

Hard assertions:

```text
row count unchanged
study_id unique
label order exact
existing N0 labels reproduced exactly
```

## E18 — class with no positives/negatives in bootstrap resample

Skip undefined per-class metric trong resample.

Macro bootstrap dùng only valid classes và report valid-resample fraction.

Nếu valid fraction <95% cho một Rare-3 label, report percentile CI không ổn định và dùng subject-level stratified bootstrap sensitivity.

## E19 — augmentations

Main branches dùng cùng augmentation.

Pair branch dùng weak independent augmentations.

Không dùng aggressive crop có thể xóa vùng pleura hoặc lesion nhỏ.

## E20 — accidental tuning

Khóa trước train:

```text
eta = 0.5
half-life = 12h
window = 24h
Rare-3 only
rho = 0.10
pair every 2 main batches
8 epochs
3 seeds
```

Không thay sau khi xem VAL.

---

# 13. Notebook structure

Đề xuất:

```text
16_label_reliability_3arm.ipynb
```

Cells:

```text
C0  Config + locked decisions
C1  Load cohort and raw labels
C2  Raw label audit + assertions
C3  Build TRAIN/VAL/TEST pair manifests
C4  Image datasets and main loader
C5  Pair dataset and collate
C6  H5 model + three arms
C7  Element-wise ASL
C8  N2 reliability weights and pair loss
C9  Gradient-ratio calibration
C10 Train loop
C11 Historical evaluation
C12 Known-only evaluation
C13 Temporal diagnostics
C14 Cluster bootstrap
C15 Go/no-go summary
C16 Export artifacts
```

Artifacts:

```text
raw_label_state_counts.csv
train_pair_manifest.parquet
val_pair_manifest.parquet
test_pair_manifest.parquet
metrics_by_epoch.json
selected_checkpoints.json
per_label_results.csv
bootstrap_contrasts.csv
```

---

# 14. Interpretation matrix

| Result | Interpretation |
|---|---|
| N1 > N0, N2 ≈ N1 | hard treatment of blank/-1 harms; temporal pairs add little |
| N1 ≈ N0, N2 > N1 | pair-derived reliability is useful; simple partial labeling is insufficient |
| N1 > N0, N2 > N1 | both unknown masking and temporal reliability contribute |
| N1 < N0, N2 recovers | full masking removes too many negatives; pair supervision restores part of signal |
| N1/N2 improve K but not H | possible robustness gain hidden by historical noisy labels; hypothesis-generating only |
| H improves but K does not | likely fitting historical labeling convention, not cleaner ranking |
| pair consistency improves but AUROC falls | oversmoothing/collapse |
| all null | label audit is diagnostically real, but proposed weak-supervision intervention lacks enough coverage or identifiability |

---

# 15. Final recommendation

Run the 3-arm screen first on the image-only H5 architecture.

The most defensible N2 intervention is deliberately conservative:

```text
do not relabel
do not use text
do not use expert labels
do not enforce equality on explicit contradictions
only transfer weak prediction consistency from known to unknown
only for Rare-3, short-gap, same-view, same-context pairs
```

A positive result can then be transferred unchanged to the selected multimodal architecture as a confirmatory experiment.
