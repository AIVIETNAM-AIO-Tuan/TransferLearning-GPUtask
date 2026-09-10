# PLAN V3.1 — Weight-Matched Temporal Label Reliability with Partial RAD-DINO Fine-Tuning

## 0. Quyết định

Không triển khai `PLAN_3ARM_NO_TEXT_LABEL_RELIABILITY_V2.md` nguyên trạng.

Lý do chặn:

1. Rare-3 có khả năng gần như không có `raw 0`, nên mask toàn bộ `blank` sẽ xóa gần hết negative supervision.
2. Endpoint known-only có thể đảo prevalence và có cỡ mẫu quá nhỏ cho Rare-3.
3. Pair-consistency cần differentiable study pooling mới, trong khi code hiện tại chỉ có numpy/pandas pooling.
4. Backbone hiện tại đông cứng toàn bộ; kế hoạch V2 mô tả train blocks 9–11 không khớp code.
5. N1 khác N0 đồng thời ở label semantics và pos/neg balance.

Thiết kế mới giữ câu hỏi Rare-3 và chỉ yêu cầu thay đổi nhỏ trong loss/data weights.

---

# 1. Câu hỏi nghiên cứu mới

Liệu giảm trọng số các `blank → negative` có khả năng là false-negative theo temporal evidence
có tốt hơn:

1. historical hard-negative policy; và
2. giảm trọng số blank một cách đồng đều?

Không dùng:

```text
report text
expert relabeling
hard union/intersection
pair consistency
pseudo-label
EHR
WASL/DBL
```

---

# 2. Kiến trúc và code path

## 2.1 Partial fine-tuning thật sự

```text
RAD-DINO blocks 0–8:
frozen, no_grad, dùng chung một forward

RAD-DINO blocks 9–11:
trainable, bản sao riêng cho R0/R1/R2

final LayerNorm + ML-Decoder:
trainable, bản sao riêng cho R0/R1/R2
```

Đây là partial fine-tuning thật sự:

```python
for p in blocks_0_8.parameters():
    p.requires_grad = False

for arm in arms:
    for p in arm.blocks_9_11.parameters():
        p.requires_grad = True
```

Không được bọc blocks 9–11 trong `torch.no_grad()` và không được chỉ để graph chạy qua
các weight frozen như nb11–13.

## 2.2 Khởi tạo

Mỗi arm lấy cùng một state:

```text
blocks 0–11:
checkpoint supervised LP–FT tốt nhất từ 04b

final norm + ML-Decoder:
checkpoint H5 tương ứng
```

Nếu H5 artifact chỉ chứa `norm.*` và `mld.*`, blocks 9–11 phải được load riêng từ checkpoint
04b. Không được để chúng ở pretrained RAD-DINO nguyên thủy trong khi decoder đến từ H5.

Assertions trước train:

```text
max parameter difference giữa ba arm = 0
same batch, eval mode, FP32:
max absolute logit difference <= 1e-6
```

## 2.3 Forward hiệu quả

```text
images
→ shared blocks 0–8 under no_grad
→ detached shared tokens
→ R0 blocks 9–11 → norm → decoder
→ R1 blocks 9–11 → norm → decoder
→ R2 blocks 9–11 → norm → decoder
```

Ba tail branch được chạy tuần tự trên cùng shared tokens để giảm VRAM. Mỗi arm có optimizer,
scheduler và scaler riêng.

Same batch và augmentation được giữ nguyên giữa các arm.

## 2.4 RNG pairing

Blocks 9–11 có thể chứa dropout hoặc stochastic depth. Để tránh arm delta bị nhiễu bởi
random mask khác nhau:

```text
lưu RNG state trước tail forward
reset cùng RNG state cho R0/R1/R2
```

hoặc khóa dropout/drop-path về 0 nếu điều này khớp protocol fine-tuning đã dùng.

Không được để mỗi arm nhận stochastic-depth realization khác nhau trong cùng batch.

## 2.5 Decoder schedule

Giữ delayed unfreeze như protocol cũ:

```text
epoch 1:
train blocks 9–11
freeze final norm + ML-Decoder

epoch 2+:
train blocks 9–11 + norm + ML-Decoder
```

Lý do:

```text
epoch 1 vẫn có khoảng 21M parameter tail-block được cập nhật
nên không còn tình trạng “không có parameter nào học”
decoder H5 được bảo vệ khỏi gradient shock do label policy mới
```

Nếu smoke test cho thấy epoch 1 làm representation drift quá mạnh trong khi logits không
cải thiện, bỏ delayed unfreeze trong cả ba arm bằng một preregistered fallback; không thay
riêng từng arm.

## 2.6 Learning rates và anchoring

Khởi đầu bảo thủ:

```text
blocks 9–11 LR = 1e-5
norm/decoder LR = 1e-5
weight decay = 1e-2
warmup = 5%
grad clip = 5.0
```

Dùng L2-SP neo về initial 04b/H5 state cho blocks 9–11 và decoder:

\[
L_{\mathrm{total}}
=
L_{\mathrm{weighted\ ASL}}
+
\lambda_{\mathrm{SP}}
\|\theta-\theta_0\|_2^2.
\]

`lambda_SP` phải giống nhau giữa các arm và lấy từ protocol hiện có, không tune riêng theo
label policy.

N0/R0 là baseline mới trong notebook này, không được tuyên bố là tái hiện A0/L0.

---

# 3. Ba arm

## R0 — Historical hard labels

Giữ đúng policy hiện tại:

```text
raw +1    → y=1, weight=1
raw -1    → y=1, weight=1
raw 0     → y=0, weight=1
raw blank → y=0, weight=1
```

## R1 — Uniform blank downweight control

Chỉ thay blank weight:

```text
raw +1    → y=1, weight=1
raw -1    → y=1, weight=1
raw 0     → y=0, weight=1
raw blank → y=0, weight=beta
```

Khóa:

```text
beta = 0.50
```

R1 đo tác động của generic negative downweighting.

## R2 — Temporal reliability, classwise weight-matched

Label mapping giống R1.

Với mỗi label, các blank negatives được chia:

```text
contested blank:
study raw blank, nhưng eligible temporal neighbor raw +1

ordinary blank:
mọi blank còn lại
```

Trọng số:

```text
contested blank → eta = 0.10
ordinary blank  → beta_hi,c
```

Chọn `beta_hi,c` để tổng blank-negative weight của R2 bằng chính xác R1:

\[
\eta N_{\mathrm{cont},c}
+
\beta_{\mathrm{hi},c} N_{\mathrm{ordinary},c}
=
\beta N_{\mathrm{blank},c}.
\]

Do đó:

\[
\beta_{\mathrm{hi},c}
=
\frac{
\beta N_{\mathrm{blank},c}
-
\eta N_{\mathrm{cont},c}
}{
N_{\mathrm{ordinary},c}
}.
\]

Hard assertions:

```text
0 <= beta_hi,c <= 1
total blank weight R2 == R1 within 1e-6
positive and explicit-negative weights unchanged
```

Nếu `beta_hi,c > 1`, giảm `beta` trước khi train bằng một rule khóa theo TRAIN; không clamp
âm thầm.

R2 và R1 có cùng total negative-weight budget theo class. Vì vậy R2−R1 kiểm tra vị trí đặt
trọng số, không còn lẫn generic reweighting.

---

# 4. Eligible temporal evidence

Một contested blank chỉ được tạo khi:

```text
same subject_id
chronological-adjacent studies
0 < delta_t <= 24h
same view signature
same cur_hadm nếu cả hai có cur_hadm
hoặc same cur_edstay nếu cả hai có cur_edstay
exact study timestamp
neighbor raw label == +1
current raw label is blank
```

Không dùng:

```text
raw -1 làm positive anchor
different-patient pair
same setting đơn độc
different encounter
different view signature
explicit 0/+1 conflict
```

Nếu thiếu encounter identifier, giữ pair cho sensitivity audit nhưng không thay weight trong
primary R2.

Pair manifest được dựng offline từ TRAIN. Không cần pair loader trong training.

Mọi ảnh của một study nhận cùng study-label weight.

---

# 5. Bước 0 — bắt buộc trước GPU

## 5.1 Raw-label coverage

In tuyệt đối `+1 / 0 / -1 / blank` cho 13 labels × 3 splits.

## 5.2 Pair funnel

Cho TRAIN:

```text
same-subject adjacent <=24h
+ exact timestamp
+ same view
+ same encounter
```

In số pair và số unique subjects sau từng bước.

## 5.3 Contested blank counts

Cho từng label:

```text
n contested blank study-label cells
n unique studies
n unique subjects
fraction of all blank negatives
beta_hi,c
```

Go/no-go:

```text
<500 unique eligible contested cells cho một label:
không claim class-specific result

<1,000 unique eligible contested cells tổng Rare-3:
R2 chỉ exploratory, không primary
```

Các threshold được khóa trước train.

## 5.4 Timestamp audit

Đếm:

```text
missing index_time
time exactly 00:00:00
duplicate subject/time
```

Không giả định `00:00` luôn date-only; cần đối chiếu metadata nguồn trước khi loại.

---

# 6. Loss

ASL trả về `[B,13]` trước reduction:

\[
L =
\frac{
\sum_{i,c} w_{i,c}\ell^{ASL}_{i,c}
}{
\sum_{i,c}w_{i,c}
}.
\]

Không pair loss, không extra optimizer step.

Hard checks mỗi epoch:

```text
effective positive weight per class
explicit-negative weight per class
blank-negative weight per class
R1/R2 blank-weight equality
```

---

# 6.1 Fine-tuning diagnostics

Log riêng theo arm và epoch:

```text
gradient norm blocks 9 / 10 / 11
gradient norm decoder
parameter drift ||theta-theta0||
representation CKA hoặc cosine drift tại output block 8 và block 11
train–val gap
```

Hard checks:

```text
grad block 9–11 > 0 sau backward
parameter delta block 9–11 > 0 sau optimizer step
blocks 0–8 parameter delta == 0
```

Nếu blocks 9–11 không đổi, đó là implementation failure chứ không phải null result.

Overfitting safeguard:

```text
nếu historical Macro giảm >0.005 trong 2 epoch liên tiếp
và train loss vẫn giảm:
flag representation overfit
```

Checkpoint selection vẫn theo frozen historical rule; không dừng riêng một arm bằng rule
khác.

---

# 7. Evaluation

## Primary endpoint

Historical frozen validation labels, giống codebase:

```text
Macro mAP
Rare-3 mAP
NonTarget-10 mAP
Tail AP
Worst-3 fixed
Recall@K
```

Checkpoint rule giữ nguyên nb12–15:

```text
macro >= best_macro - 0.0015
→ max Tail AP
→ Worst-3
→ Recall@K
→ Macro mAP
```

## Secondary

```text
Macro/Rare-3 AUROC
Macro AUPRC
per-label AP
Brier score
```

## Diagnostic endpoint

Known-only subset:

```text
AUROC_K
AP_K
sample size
prevalence
```

Không dùng K để chọn checkpoint hoặc quyết định arm thắng.

## Temporal diagnostic

Trên fixed VAL audit pairs:

```text
score blank-with-positive-neighbor
score clean historical negatives
pct_cont_neg
same-view versus different-view
0–6h / 6–12h / 12–24h
```

Pair diagnostic không phải ground truth.

---

# 8. Contrasts

```text
D1 = R1 - R0:
generic blank-negative downweighting

D2 = R2 - R1:
temporal targeting, weight-budget matched

D3 = R2 - R0:
net practical effect
```

D2 là contrast quan trọng nhất.

---

# 9. Success gate

R2 được xem là positive khi:

```text
Rare-3 mAP >= R1 +0.002
hoặc
Macro mAP >= R1 +0.002
```

và:

```text
NonTarget-10 >= R1 -0.002
```

Cần:

```text
same direction >=2/3 seeds
subject-cluster bootstrap không cho thấy material harm
temporal diagnostic đi đúng hướng
```

Nếu R1≈R2 nhưng cả hai >R0:

```text
gain là generic reweighting, không phải temporal reliability
```

Nếu R2>R1:

```text
vị trí downweight có giá trị ngoài tổng weight budget
```

---

# 10. Code changes

## Cần viết mới

```text
raw-label merge
offline pair manifest
study-label weight matrix [N_study,13]
element-wise weighted ASL
subject-cluster bootstrap
shared forward blocks 0–8
ba bản sao trainable blocks 9–11
sequential tail forward/backward
RNG-state synchronization giữa arm
04b + H5 dual-source checkpoint loading
```

## Không cần viết

```text
differentiable torch study pooling
pair loader
teacher-student branch
extra forward cho temporal neighbor
```

## Encounter columns

Merge từ `cohort_base.parquet`:

```text
cur_hadm
cur_edstay
```

Không dùng ICU `stay_id` vì chưa có trong cohort hiện tại.

---

# 11. Notebook đề xuất

```text
16_label_reliability_weightmatched.ipynb
```

Cells:

```text
C0  Locked protocol
C1  Load cohort + raw labels
C2  Coverage audit
C3  Merge encounter metadata
C4  Build pair funnel
C5  Build R0/R1/R2 weight matrices
C6  Weight-matching assertions
C7  Shared frozen blocks 0–8
C8  Three trainable blocks 9–11 + H5 heads
C9  Weighted element-wise ASL
C10 Train
C11 Historical evaluation
C12 Temporal diagnostic
C13 Subject-cluster bootstrap
C14 Go/no-go
```

---

# 12. Interpretation

| Result | Kết luận |
|---|---|
| R1 > R0, R2 ≈ R1 | giảm blank-negative nói chung hữu ích; temporal targeting không thêm |
| R1 ≈ R0, R2 > R1 | temporal evidence xác định được các negative đáng ngờ |
| R1 > R0, R2 > R1 | cả generic và targeted reliability cùng hữu ích |
| R1/R2 < R0 | blank negatives vẫn cần thiết hoặc weight quá thấp |
| R2 cải thiện pair diagnostic nhưng không mAP | mechanistic signal chưa chuyển thành benchmark gain |
| tất cả null | pair coverage hoặc signal không đủ, không nên mở rộng consistency model |
