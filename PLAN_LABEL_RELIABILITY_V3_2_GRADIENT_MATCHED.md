# PLAN V3.2 — Gradient-Matched Temporal Label Reliability with Partial Fine-Tuning

## 0. Các quyết định đã chốt

1. Không dùng cell-count weight matching.
2. Giữ ba arm, nhưng thay R1 bằng một uniform control được khớp theo negative logit-gradient tại initialization.
3. Mở RAD-DINO blocks 9–11 thật sự; blocks 0–8 frozen và shared.
4. Dùng cửa sổ ±7 ngày cho Rare-3 structural labels và không giới hạn chronological-adjacent.
5. Mỗi blank study-label cell chỉ dùng nearest eligible positive anchor, không tạo all-pair training samples.
6. Bỏ L2-SP như một safeguard; dùng hard drift eligibility thresholds.
7. EPOCHS = 8.
8. Delayed decoder unfreeze cố định: epoch 1 frozen, epoch 2–8 trainable.
9. Kiểm tra model có stochasticity thật trước khi triển khai RNG synchronization.
10. Historical mAP vẫn là primary endpoint.

---

# 1. Câu hỏi nghiên cứu

Với cùng một mức negative supervision ban đầu, việc giảm trọng số đúng các blank labels có
temporal evidence dương có tốt hơn giảm đồng đều tất cả blank labels hay không?

Không dùng:

```text
report text
expert relabeling
hard relabel
pair-consistency loss
EHR
WASL/DBL
```

---

# 2. Kiến trúc

```text
RAD-DINO blocks 0–8:
frozen, no_grad, shared forward

RAD-DINO blocks 9–11:
trainable, separate copy cho R0/R1/R2

final norm + ML-Decoder:
separate copy cho R0/R1/R2
```

Initialization:

```text
blocks 0–11 ← supervised LP–FT checkpoint 04b
norm + ML-Decoder ← H5 checkpoint
```

Assertions:

```text
all arm parameters identical before training
eval-mode logits identical within 1e-6
grad block 9–11 > 0
parameter change block 9–11 > 0
parameter change block 0–8 == 0
```

Forward:

```text
one shared blocks-0–8 pass
→ sequential R0 tail forward/backward
→ sequential R1 tail forward/backward
→ sequential R2 tail forward/backward
```

---

# 3. Arm definitions

## R0 — Historical

```text
raw +1    → positive, weight 1
raw -1    → positive, weight 1
raw 0     → negative, weight 1
raw blank → negative, weight 1
```

## R2 — Temporal targeted policy

R2 is defined before its control.

```text
ordinary blank  → weight 1.0
contested blank → weight eta = 0.10
```

Positive, uncertain-positive and explicit-negative weights remain 1.

## R1-GM — Uniform gradient-matched control

All blank negatives in class `c` receive the same weight `beta_grad,c`.

`beta_grad,c` is selected so that, at the shared model initialization, R1-GM and R2 have the
same total first-order blank-negative logit-gradient mass for that class.

---

# 4. Step 0b — gradient-share measurement

## 4.1 Exact quantity

Do not use only the ASL modulating factor `(p-clip)^gamma`.

For each TRAIN image and class, calculate the exact unweighted negative loss derivative:

\[
g_{k,c}
=
\left|
\frac{\partial \ell^{ASL}_{neg}(z_{k,c})}
{\partial z_{k,c}}
\right|.
\]

`k` indexes images.

Because training occurs at image level, aggregate images within a study:

\[
q_{i,c}
=
\sum_{k \in study(i)} g_{k,c}.
\]

No augmentation, model in `eval()` mode, FP32, same initialization for all arms.

## 4.2 Contested and ordinary gradient mass

For class `c`:

\[
G_{\mathrm{cont},c}
=
\sum_{i \in contested(c)} q_{i,c},
\]

\[
G_{\mathrm{ord},c}
=
\sum_{i \in ordinary\ blank(c)} q_{i,c}.
\]

R2 initial gradient mass:

\[
G_{R2,c}
=
\eta G_{\mathrm{cont},c}
+
G_{\mathrm{ord},c}.
\]

R1-GM weight:

\[
\beta_{\mathrm{grad},c}
=
\frac{G_{R2,c}}
{G_{\mathrm{cont},c}+G_{\mathrm{ord},c}}.
\]

Hard checks:

```text
0 <= beta_grad,c <= 1
relative logit-gradient mismatch <= 1e-6
```

## 4.3 Parameter-gradient calibration check

Logit-gradient matching is first-order output matching, not exact parameter-gradient matching.

On a fixed deterministic TRAIN calibration loader, calculate blank-negative-only gradients
for all trainable parameters under R1-GM and R2.

Required before official training:

```text
total trainable-parameter gradient-norm ratio R2/R1 ∈ [0.90, 1.10]
decoder gradient-norm ratio R2/R1 ∈ [0.90, 1.10]
```

If not satisfied:

```text
multiply all beta_grad,c by one global scalar kappa
solve kappa by binary search on the calibration loader
repeat until total gradient-norm ratio is within [0.90,1.10]
```

Classwise logit-gradient equality may then become approximate; report both mismatches.

If no `kappa` in `[0.5,2.0]` passes:

```text
D2 cannot be called gradient-matched
do not launch official seed sweep
```

## 4.4 During-training diagnostic

At the end of each epoch, repeat the gradient comparison on the same frozen TRAIN calibration
loader without updating weights.

If R2/R1 parameter-gradient ratio leaves `[0.75,1.25]` for two consecutive epochs:

```text
D2 remains a practical-policy comparison
but the claim “equal gradient magnitude” is withdrawn
```

Do not dynamically modify weights after training begins.

---

# 5. Temporal manifest V3.2

## 5.1 Labels

Primary temporal targeting only for:

```text
Pleural_Other
Fracture
Lung_Lesion
```

## 5.2 Eligibility

A blank study-label cell is contested if there exists a positive anchor satisfying:

```text
same subject_id
raw anchor label == +1
raw current label is blank
absolute time gap > 0 and <= 7 days
same view signature
exact timestamp available
```

No same-patient condition can be relaxed.

The primary 7-day window is justified as a persistence prior for structural findings, not
as proof that both studies have identical disease state.

## 5.3 Do not require chronological adjacency

Search all studies from that subject within ±7 days.

For each blank study-label cell:

```text
choose nearest eligible positive anchor only
mark the blank cell once
```

Thus all longitudinal evidence is searched without creating quadratic pair-loss samples.

## 5.4 Encounter handling

Same encounter is recorded but not required in the primary 7-day structural-label manifest.

Report sensitivity subsets:

```text
same encounter
different/unknown encounter
<=24h
24h–72h
72h–7d
```

If results exist only in different-encounter or 72h–7d pairs, interpret cautiously.

## 5.5 Concentration diagnostics

For each class report:

```text
unique contested cells
unique studies
unique subjects
median cells per subject
top 1% subjects' share of contested gradient mass
```

Go/no-go for a class-specific claim:

```text
>=500 contested study-label cells
>=200 unique subjects
top 1% subjects contribute <=15% of contested gradient mass
```

Rare-3 primary eligibility:

```text
>=1,000 contested cells total
>=500 unique subjects total
```

Otherwise R2 is exploratory.

---

# 6. Optimization

```text
EPOCHS = 8
effective batch = 256
natural sampling
same batch/augmentation/order across arms
```

Schedule:

```text
epoch 1:
train blocks 9–11
freeze final norm + ML-Decoder

epoch 2–8:
train blocks 9–11 + norm + ML-Decoder
```

Learning rates:

```text
blocks 9–11 = 1e-5
norm/decoder = 1e-5
weight decay = 1e-2
warmup = 5%
gradient clip = 5.0
```

## 6.1 L2-SP decision

```text
lambda_SP = 0 in the primary experiment
```

Reason:

```text
1e-5 L2-SP is too small to be cited as protection for 21M trainable parameters under AdamW.
```

No claim of stabilization is made from L2-SP.

---

# 7. Hard drift safeguards

Let:

\[
d_b(t)
=
\frac{\|\theta_b(t)-\theta_b(0)\|_2}
{\|\theta_b(0)\|_2}
\]

for blocks 9, 10 and 11.

Also use a fixed TRAIN probe of 2,048 studies and calculate mean cosine similarity between
initial and current block-11 patch features.

## 7.1 Epoch-1 protocol check

After epoch 1:

```text
total blocks-9–11 relative drift <= 0.005
each individual block drift <= 0.0075
mean block-11 feature cosine >= 0.99
```

If violated:

```text
the delayed-decoder protocol fails
do not continue official runs
restart all arms with decoder trainable from epoch 1
```

This fallback is decided using TRAIN diagnostics only and applies identically to all arms.

## 7.2 Checkpoint eligibility through epoch 8

A checkpoint is eligible only if:

```text
total tail drift <= 0.010
each block drift <= 0.015
feature cosine >= 0.98
```

A checkpoint outside this trust region may be reported diagnostically but cannot be selected.

---

# 8. Stochasticity check

Before implementing RNG synchronization:

```python
model.train()
with torch.no_grad():
    y1 = tail(x.clone())
    y2 = tail(x.clone())
max_diff = (y1-y2).abs().max()
```

Decision:

```text
max_diff <= 1e-7:
tail is deterministic; do not add RNG-sync code

max_diff > 1e-7:
identify dropout/drop-path modules
reset identical CPU/CUDA RNG state before each arm tail forward
```

Also print actual model config:

```text
hidden dropout
attention dropout
drop-path/stochastic-depth probability
```

Do not assume library defaults.

---

# 9. Loss

ASL returns `[B,13]` before reduction:

\[
L
=
\frac{\sum_{i,c}w_{i,c}\ell_{i,c}^{ASL}}
{\sum_{i,c}w_{i,c}}.
\]

Weights are fixed before official training.

Each epoch log:

```text
weighted positive loss
weighted explicit-negative loss
weighted ordinary-blank loss
weighted contested-blank loss
negative logit-gradient mass
trainable-parameter gradient norm
```

---

# 10. Contrasts

```text
D1 = R1-GM - R0:
uniform negative-gradient reduction

D2 = R2 - R1-GM:
temporal placement under matched initial gradient strength

D3 = R2 - R0:
net practical effect
```

Interpretation of D2 is conditional:

```text
strongest when init gradient checks pass
and epoch gradient ratio stays within [0.75,1.25]
```

If the ratio diverges, D2 is still a valid practical comparison but not a pure
gradient-magnitude-controlled contrast.

---

# 11. Evaluation and checkpoint selection

Primary historical frozen labels:

```text
Macro mAP
Rare-3 mAP
NonTarget-10 mAP
Tail AP
Worst-3 fixed
Recall@K
```

Checkpoint rule, unchanged from nb12–15:

```text
Macro >= arm best Macro - 0.0015
→ maximum Tail AP
→ Worst-3
→ Recall@K
→ Macro mAP
```

Secondary:

```text
Macro/Rare-3 AUROC
Macro AUPRC
per-label AP
calibration
temporal diagnostics
```

Known-only metrics remain diagnostic only.

---

# 12. Success gate

R2 succeeds over R1-GM when:

```text
Rare-3 mAP >= +0.002
or Macro mAP >= +0.002
```

with:

```text
NonTarget-10 >= -0.002
same direction in >=2/3 seeds
subject-cluster bootstrap does not show material harm
temporal diagnostic moves in the expected direction
data sufficiency gate passes
initial gradient matching gate passes
```

An increase in AUROC alone does not pass the gate.

If R2≈R1-GM and both improve over R0:

```text
benefit is explained by reducing negative gradient strength, not temporal targeting
```

If R2>R1-GM:

```text
temporal placement adds value beyond a uniform control with matched initial gradient strength
```

---

# 13. Compute staging

Expected cost is materially larger than nb15.

Run:

```text
Stage 1:
seed 42, all 3 arms, full 8 epochs

Continue to seeds 1337 and 2024 only if:
NonTarget-10 harm <=0.002
and D2 Rare-3 or Macro delta is non-negative
and no drift/gradient-matching protocol failure
```

The seed-42 arm definitions and hyperparameters are already fixed; no retuning is allowed
before the confirmatory seeds.

---

# 14. Required pre-GPU outputs

```text
raw_label_counts.csv
temporal_manifest_funnel.csv
contested_cells_by_class.csv
initial_asl_gradient_share.csv
gradient_matched_beta.csv
parameter_gradient_calibration.json
stochasticity_assert.json
drift_protocol_config.json
```
