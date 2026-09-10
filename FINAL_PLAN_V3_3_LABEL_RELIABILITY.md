# FINAL DECISIONS V3.3 — Gradient-Matched Temporal Label Reliability

## 0. Phán quyết chung

V3.2 chưa được triển khai nguyên trạng.

Tiếp tục hướng 3-arm, partial fine-tuning blocks 9–11, nhưng sửa:

1. loss denominator;
2. drift policy;
3. CLS convention;
4. final LayerNorm ownership;
5. interpretation of gradient matching;
6. subject concentration handling;
7. seed continuation rule.

Primary estimand:

```text
D2 = R2 temporal targeting − R1-GM uniform control
```

D2 được mô tả chính xác là:

> same initial blank-negative logit-gradient mass, different placement.

Không gọi là “same parameter-gradient magnitude”.

---

# 1. Arm definitions

## R0 — historical

```text
raw +1    → y=1, w=1
raw -1    → y=1, w=1
raw 0     → y=0, w=1
raw blank → y=0, w=1
```

## R2 — temporal targeting

```text
ordinary blank  → w=1
contested blank → w=eta=0.10
```

Các ô khác giữ `w=1`.

## R1-GM — uniform logit-gradient-matched control

Với mỗi Rare-3 class:

\[
\beta_c =
\frac{
\eta G_{\mathrm{cont},c}+G_{\mathrm{ord},c}
}{
G_{\mathrm{cont},c}+G_{\mathrm{ord},c}
}.
\]

Mọi blank của class `c` trong R1 nhận `w=beta_c`.

NonTarget-10 tự động có `beta=1`.

---

# 2. Loss normalization — CHỐT

Dùng element-wise ASL:

\[
L =
\frac{
\sum_{i,c} w_{i,c}\ell_{i,c}
}{
B_{\mathrm{actual}}\times C
}.
\]

Không chia cho `sum(w)`.

Implementation:

```python
loss = (weights * loss_elem).sum() / loss_elem.numel()
```

Lý do:

```text
cùng một ô không bị can thiệp có cùng hệ số loss trực tiếp trong mọi arm;
không tạo artefact +20% cho NonTarget-10 chỉ vì mẫu số của arm khác nhau.
```

AdamW không hoàn toàn scale-invariant vì epsilon, clipping và weight decay. Việc dùng mẫu số
chung được chọn vì giữ coefficient của các ô không bị nhắm giống nhau, không dựa vào một
tuyên bố bất biến tuyệt đối của AdamW.

Mọi gradient calibration dùng cùng denominator này.

---

# 3. Gradient matching — CHỐT LẠI

## 3.1 Primary matching level

Khớp exact negative logit-gradient mass tại initialization:

\[
g_{k,c}
=
\left|
\frac{\partial \ell^{ASL}_{neg}}{\partial z_{k,c}}
\right|.
\]

Đo:

```text
model eval
FP32
không augmentation
toàn bộ TRAIN images
H5-compatible patch-only forward
```

Training image-level nên gradient cũng được tính image-level; có thể aggregate study chỉ để
báo cáo/concentration analysis.

## 3.2 Parameter-gradient check

Parameter-gradient norm được đo trên fixed TRAIN calibration set nhưng chỉ là diagnostic:

```text
ratio R2/R1-GM theo blocks 9–11
ratio theo backbone final LN
ratio theo H5 head
gradient cosine R2 versus R1-GM
```

Không dùng `kappa` để ép parameter-gradient norm bằng nhau.

Lý do:

```text
R2 chọn những examples có feature/Jacobian khác.
Parameter-gradient norm khác do vị trí examples là một phần của targeting mechanism.
Ép norm bằng một scalar sẽ over-control cơ chế và phá classwise logit matching.
```

Do đó bỏ binary search `kappa`; vấn đề `kappa*beta > 1` không còn tồn tại.

Claim được khóa:

```text
gradient matched at the output/logit level at initialization
```

Không claim:

```text
parameter-gradient matched throughout training
```

## 3.3 Gradient drift qua epoch

Dự kiến R2/R1 gradient ratio có thể tăng:

```text
R2 giảm pressure trên contested high-score blanks
→ score nhóm này có thể tăng
→ raw ASL negative gradient của nhóm này tăng
```

Đây là expected treatment dynamics, không phải implementation failure.

Không thay weights động trong training.

---

# 4. Temporal manifest

## 4.1 Primary eligibility

```text
same subject_id
raw current label = blank
raw anchor label = +1
0 < absolute time gap <= 7 days
same view signature
exact/reliable study timestamp
nearest eligible positive anchor per blank cell
```

Dùng TRAIN only để tạo training weights.

Không dùng raw `-1` làm anchor.

## 4.2 Encounter

Encounter không phải primary filter cho structural Rare-3, nhưng lưu:

```text
same cur_hadm
same cur_edstay
different/unknown encounter
```

Báo cáo sensitivity theo encounter.

## 4.3 Direction and time sensitivity

Bắt buộc báo cáo:

```text
positive anchor before blank
positive anchor after blank
<=24h
24–72h
72h–7d
```

Đặc biệt Fracture có thể xuất hiện mới giữa hai study. Nếu hiệu ứng chỉ đến từ
future-positive anchors hoặc 72h–7d, kết luận phải hạ xuống exploratory.

## 4.4 Subject concentration

Primary manifest không tự động cap hoặc reweight sau khi thấy số liệu.

Data sufficiency gate:

```text
>=500 contested cells/class
>=200 unique subjects/class
Rare-3 total >=1,000 cells and >=500 subjects
top 1% subjects <=15% contested initial gradient mass
```

Nếu concentration gate fail:

```text
R2 mất primary status;
không sửa manifest rồi tiếp tục claim cùng estimand.
```

Có thể chạy sensitivity `cap <=3 cells/subject/class`, chọn ba gap nhỏ nhất, nhưng phải ghi
exploratory và không thay primary result.

Lý do không dùng fallback tự động:

```text
cap hoặc subject-normalization thay đổi estimand.
Một data-sufficiency failure không nên được “chữa” bằng can thiệp được chọn sau audit.
```

---

# 5. Backbone and token convention

## 5.1 Blocks

`blocks 9–11` là 0-indexed, ba transformer block cuối.

```text
blocks 0–8:
frozen, shared, no_grad

blocks 9–11:
separate and trainable per arm
```

## 5.2 Two different LayerNorms

Phân biệt rõ:

### A. `backbone.layernorm`

```text
final ViT norm sau block 11
per-arm
thuộc tail-backbone group
trainable từ epoch 1
LR = block LR
clip cùng blocks
```

Lý do: block output thay đổi theo arm; final backbone norm phải đi cùng representation tail.

### B. H5 `norm`

```text
LayerNorm riêng nằm trước ML-Decoder
per-arm
thuộc decoder group
frozen epoch 1
trainable epoch 2–8
```

## 5.3 CLS convention

`nb16` bắt buộc:

```python
patches = backbone.layernorm(h)[:, 1:]
logits = h5_mld(h5_norm(patches))
```

Không đưa CLS vào ML-Decoder.

Lý do:

```text
H5 được train trên 1,369 patch tokens;
nb11/13 dùng patch-only;
giữ CLS ở nb14/15 là một protocol mismatch.
```

Identity gate phải so với một direct H5 reference dùng patch-only, không so hai copies của
cùng implementation.

---

# 6. Optimization

```text
EPOCHS = 8
EFF = 256
natural sampling
same batch/order/augmentation across arms
```

Schedule:

```text
epoch 1:
train blocks 9–11 + backbone.layernorm
freeze H5 norm + ML-Decoder

epoch 2–8:
train all above
```

LR:

```text
blocks 9–11 + backbone.layernorm: 1e-5
H5 norm + decoder: 1e-5
```

Clipping:

```text
blocks + backbone.layernorm: 5.0
H5 norm + decoder: 1.0
```

Weight decay:

```text
wd=1e-2 only for ndim>1
bias and LayerNorm parameters wd=0
```

No L2-SP safeguard claim.

---

# 7. Drift policy — SỬA LỚN

## 7.1 Remove arbitrary hard drift gates

Không dùng các số:

```text
0.5%, 0.75%, 1%, 1.5%
cosine 0.99/0.98
```

để tự động restart hoặc loại checkpoint.

Lý do:

```text
không được hiệu chuẩn;
parameter drift không có quan hệ đơn điệu với model validity;
fine-tuning có mục đích làm representation thay đổi;
một cap sai có thể loại mọi checkpoint tốt.
```

Ước lượng `lr*sqrt(steps)/RMS` chỉ là heuristic; không đủ để đặt threshold. Tương tự,
ngoại suy 200 steps bằng sqrt(t) không đáng tin nếu update có tương quan, warmup và clipping.

## 7.2 Smoke run

Chạy 200 steps R0 để:

```text
xác minh memory/runtime
gradient flow
blocks 0–8 unchanged
blocks 9–11 updated
measure stochasticity
ghi drift/cosine trajectory mô tả
```

Smoke không dùng để đặt checkpoint-eligibility threshold bằng hệ số 2–3×.

## 7.3 Drift remains diagnostic

Mỗi epoch log:

```text
relative parameter drift by block
feature cosine
feature variance/rank proxy
train/val gap
```

Hard failure chỉ cho:

```text
NaN/Inf
optimizer/scaler failure kéo dài
blocks 0–8 thay đổi
không có gradient/update ở blocks 9–11
feature tensor collapse/numerical degeneracy
```

Checkpoint vẫn chọn bằng frozen validation rule, không bằng drift.

Không có delayed-unfreeze fallback dựa trên drift. Schedule epoch 1/epoch 2–8 được khóa trước.

---

# 8. Stochasticity

Chạy hai forward liên tiếp trong `train()` trên cùng input.

```text
max_diff <=1e-7:
không RNG sync

max_diff >1e-7:
in module/config gây stochasticity
reset cùng CPU/CUDA RNG state trước mỗi arm tail forward
```

Frozen shared blocks 0–8 chạy một lần nên vốn đã paired.

---

# 9. Evaluation

Primary historical endpoint:

```text
Macro mAP
Rare-3 mAP
NonTarget-10
Tail AP
Worst-3 fixed
Recall@K
```

Checkpoint rule:

```text
Macro >= arm best Macro - 0.0015
→ max Tail AP
→ Worst-3
→ Recall@K
→ Macro
```

Secondary:

```text
AUROC
AUPRC
per-label AP
calibration
temporal diagnostics
```

Known-only endpoint diagnostic only.

Bootstrap:

```text
2,000 resamples
cluster = subject_id
```

Handle no-positive resamples explicitly and report valid fraction.

---

# 10. Success and interpretation

Primary contrast:

```text
D2 = R2 - R1-GM
```

Gate:

```text
Rare-3 mAP >= +0.002
or Macro mAP >= +0.002
NonTarget-10 >= -0.002
same direction >=2/3 seeds
subject-cluster bootstrap no material harm
data sufficiency passed
```

`+0.002` is defensible because this is a paired comparison on an identical fixed evaluation
label set; it does not estimate the absolute label-noise ceiling.

Interpretation:

```text
R1-GM > R0, R2 ≈ R1-GM:
generic reduction of Rare-3 negative pressure

R2 > R1-GM:
temporal placement adds value beyond equal initial output-gradient mass

R1-GM early peak then collapse:
strong positive/negative rebalance overfits, consistent with prior loss behavior
```

---

# 11. Seed staging

Run seed 42 first for implementation/runtime validation.

Once implementation and pre-GPU gates pass:

```text
run seeds 1337 and 2024 regardless of whether seed 42 delta is positive or negative
```

Do not use seed-42 performance to decide whether positive confirmatory seeds are collected.

Reason:

```text
continue-only-after-positive creates selection bias and upward-biased 3-seed evidence.
```

A run may stop after seed 42 only for:

```text
implementation failure
data-sufficiency failure
numerical instability
gross predefined safety harm
```

If stopped for performance futility, report a one-seed exploratory no-go; do not make a
positive claim.

---

# 12. Remaining implementation decisions

```text
raw labels:
new cohort_v2/cohort_raw_labels.parquet; never overwrite old cohort

timestamp rule:
decide only after 00:00/missing/source-field audit

notebook:
16_label_reliability_gradient_matched.ipynb

artifacts:
DATA_DIR/label_reliability

one seed per run:
SEED config + independent resume/checkpoint path
```

---

# 13. Pre-GPU order

```text
0a raw labels + temporal manifest + concentration
0b exact initial ASL logit-gradient share + beta
0c 200-step smoke for implementation/runtime/stochasticity only
0d deterministic calibration diagnostics
1  lock manifest hashes, beta, code commit, all hyperparameters
2  seed 42 full 8 epochs
3  seeds 1337 and 2024
4  subject-cluster bootstrap and final interpretation
```
