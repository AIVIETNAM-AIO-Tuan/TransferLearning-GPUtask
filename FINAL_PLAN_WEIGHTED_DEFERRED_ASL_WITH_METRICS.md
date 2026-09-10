# FINAL EXPERIMENTAL PLAN — Weighted ASL & Deferred Weighted ASL

## 0. Mục tiêu

Kiểm tra liệu explicit inter-class reweighting trên nền ASL có cải thiện tail classes mà vẫn giữ được overall performance của architecture hiện tại hay không.

Current reference:

```text
Late unconditional:
macro mAP = 0.4375
tail mAP  = 0.1222

Late Full-EHR:
macro mAP = 0.4408
tail mAP  = 0.1228
```

Current late EHR-specific gain:

```text
macro +0.0033
tail  +0.0006
```

Primary research question:

> Can class-weighted ASL improve tail recognition without materially sacrificing overall performance?

Secondary question:

> Is deferred class weighting better than applying class weights from the beginning?

---

# 1. Freeze everything except loss

Use the **current late Full-EHR architecture** as the fixed model for Stage 1.

Keep unchanged:

```text
RAD-DINO checkpoint
patch-token representation
late adapters at blocks 10 & 11
Full-EHR encoder
current EHR features
ML-Decoder
ASL gamma_pos / gamma_neg / clip
optimizer
learning rates
weight decay
warmup
batch size
gradient clipping
image augmentation
train/validation split
current multi-view evaluation
delayed decoder-unfreeze protocol
```

Do NOT add:

```text
new multi-view architecture
disease-aware gate
time-aware EHR changes
imputation
tail expert
teacher-student
resampling / oversampling
new class sampler
```

The only experimental variable is:

```text
inter-class loss weighting
```

---

# 2. Lock label tiers

Use the existing frozen definition of:

```text
head
medium
tail
```

Do not redefine tiers after seeing results.

For every label save:

```text
label
N_positive_train
prevalence_train
tier
```

All class weights must be calculated using **TRAIN labels only**.

---

# 3. Baseline loss

Baseline:

```text
L0 = current ASL
```

Keep the current implementation and current values of:

```text
gamma_pos
gamma_neg
clip / margin
eps
```

No ASL hyperparameter tuning in this experiment.

---

# 4. Weighted ASL

For class c, weight only the positive ASL term:

\[
L_c^{WASL}
=
-w_c y_c(1-p_c)^{\gamma_+}\log p_c
-
(1-y_c)(p'_c)^{\gamma_-}\log(1-p'_c)
\]

The negative term is kept unchanged.

Rationale:

```text
ASL already suppresses easy-negative dominance.
The new experiment targets inter-class positive-gradient imbalance.
```

---

# 5. Class-weight calculation

Let:

```text
n_c = number of positive TRAIN samples for class c
```

Raw weight:

\[
\tilde w_c=n_c^{-\alpha}
\]

Normalize:

\[
w_c=
\frac{\tilde w_c}
{\frac{1}{C}\sum_j\tilde w_j}
\]

Then clip:

```text
0.5 <= w_c <= 3.0
```

Normalize again so:

```text
mean(w_c) = 1
```

Only two weighting strengths are allowed in Stage 1:

```text
alpha = 0.25
alpha = 0.50
```

Do not introduce more alpha values before Stage-1 analysis.

### Mandatory pre-run table

Print:

| Label | N positive | Prevalence | Tier | w(alpha=.25) | w(alpha=.50) |
|---|---:|---:|---|---:|---:|

Inspect manually before GPU training.

---

# 6. Deferred Weighted ASL

Use hard switching, not a continuous ramp.

If max epochs = 8:

```text
Epoch 1:
plain ASL
decoder frozen as current protocol

Epoch 2–3:
plain ASL
decoder unfrozen

Epoch 4–8:
switch to Weighted ASL
```

Equivalent approximately to:

```text
first 40% optimization:
class weights = 1

remaining 60%:
class weights = target weights
```

### Early stopping rule

Deferred runs must receive at least:

```text
2 complete weighted epochs
```

before they can early-stop.

Final deferred checkpoint must be **after the weighting switch**.

---

# 7. Stage 0 — implementation sanity check

Before training:

## Test A

```text
Weighted ASL with all weights = 1
```

must reproduce current ASL loss to floating-point tolerance.

## Test B

Gradients with:

```text
weights = 1
```

must match current ASL gradients.

## Test C

Artificially increase one tail-class weight.

Expected:

```text
positive loss contribution of that class increases
negative term remains unchanged
```

## Test D

Check class weights:

```text
no NaN
no inf
0.5 <= weight <= 3.0
mean(weight) ≈ 1
```

---

# 8. Stage 1 — one-seed screening

Use:

```text
seed = 42
```

Architecture:

```text
current Late Full-EHR L_E
```

Run exactly five arms:

```text
L0 = ASL

L1 = WASL alpha=0.25
L2 = WASL alpha=0.50

L3 = Deferred WASL alpha=0.25
L4 = Deferred WASL alpha=0.50
```

All arms must use:

```text
same H5 checkpoint
same model initialization logic
same EHR encoder initialization
same decoder initialization
same adapter initialization
same data split
same batch construction/order where feasible
same optimizer configuration
```

Do not add RAL or DB-loss yet.

---

# 9. Final metric set

Every arm must report the following.

## A. AP metrics

```text
Macro mAP
Tail mAP
Worst-3 AP
```

Where:

```text
Macro mAP = mean AP across all 13 labels
Tail mAP  = mean AP across locked tail labels
Worst-3 AP = mean AP of the three lowest-AP labels
```

AP must use the same implementation across all arms.

Recommended:

```python
sklearn.metrics.average_precision_score
```

---

## B. AUPRC

Report:

```text
Macro AUPRC
```

For each label:

1. construct precision-recall curve;
2. integrate the PR curve using trapezoidal AUC;
3. average across labels.

Recommended implementation:

```python
precision, recall, _ = precision_recall_curve(y_true, y_score)
auprc = auc(recall, precision)
```

Important:

```text
AP and trapezoidal AUPRC are not assumed to be numerically identical.
```

Do not silently substitute one for the other.

---

## C. AUROC

Report:

```text
Macro AUROC
Tail AUROC
```

Definitions:

```text
Macro AUROC = mean AUROC across all labels
Tail AUROC  = mean AUROC across locked tail labels
```

---

## D. Tail Recall

Report:

```text
Tail Recall@K
```

No probability threshold is used.

For each tail label c:

```text
K_c = number of positive examples for label c
      in the evaluation set
```

Rank all studies by predicted score for label c.

Then:

\[
Recall@K_c
=
\frac{\text{number of true positives among top }K_c}
{\text{total number of positives for class }c}
\]

Finally:

```text
Tail Recall@K
=
mean Recall@K_c across tail labels
```

This metric is intended to evaluate positive-case retrieval without depending on score calibration or a fixed probability threshold.

---

# 10. Main result table

Use exactly this main table:

| Loss | Macro mAP | Tail mAP | Worst-3 AP | Macro AUPRC | Macro AUROC | Tail AUROC | Tail Recall@K |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASL | | | | | | | |
| WASL .25 | | | | | | | |
| WASL .50 | | | | | | | |
| DWASL .25 | | | | | | | |
| DWASL .50 | | | | | | | |

Also calculate delta versus baseline:

```text
Delta Macro mAP
Delta Tail mAP
Delta Worst-3 AP
Delta Macro AUPRC
Delta Macro AUROC
Delta Tail AUROC
Delta Tail Recall@K
```

---

# 11. Per-label table

For diagnostic analysis save:

| Label | N positive | Tier | AP ASL | AP candidate | Delta AP | AUROC ASL | AUROC candidate |
|---|---:|---|---:|---:|---:|---:|---:|

Also plot:

```text
x = log10(N_positive_train)
y = Delta AP
```

This is diagnostic only, not the primary selection criterion.

---

# 12. Metric priority

The decision hierarchy is:

## Primary

1. Tail mAP
2. Worst-3 AP
3. Tail Recall@K

## Overall safeguard

4. Macro mAP

## Supporting evidence

5. Macro AUPRC
6. Macro AUROC
7. Tail AUROC

Ideal pattern:

```text
Tail mAP       ↑
Worst-3 AP     ↑
Tail Recall@K  ↑
Tail AUROC     ↑ or stable

Macro mAP      stable or ↑
Macro AUPRC    stable or ↑
Macro AUROC    stable or ↑
```

---

# 13. Stage-1 checkpoint selection

Do not use tail-only checkpoint selection.

For each run:

1. find its best validation Macro mAP;
2. retain checkpoints satisfying:

```text
Macro mAP >= best Macro mAP - 0.0015
```

3. among those checkpoints select the checkpoint with highest:

```text
Tail mAP
```

Tie-break order:

```text
1. higher Worst-3 AP
2. higher Tail Recall@K
3. higher Macro mAP
```

For Deferred WASL:

```text
only checkpoints after the weighting switch are eligible
```

Also report the pure best-Macro checkpoint as a secondary sanity check.

---

# 14. Stage-1 go / no-go rule

A weighted candidate is **promising** if:

```text
Delta Tail mAP >= +0.002
```

while:

```text
Delta Macro mAP >= -0.002
```

and:

```text
Worst-3 AP does not materially worsen
```

and preferably:

```text
Tail Recall@K > baseline
```

### Strong signal

```text
Delta Tail mAP >= +0.004
Worst-3 AP ↑
Tail Recall@K ↑
Macro mAP >= baseline
```

### No-go

If all four weighted arms show:

```text
Delta Tail mAP < +0.002
```

or tail gain occurs only with:

```text
Macro mAP loss > 0.002
```

then do not expand the alpha search immediately.

Conclusion:

> simple class reweighting on ASL does not show a favorable tail-performance trade-off under the tested architecture.

---

# 15. Training diagnostics

Log every epoch:

```text
Macro mAP
Tail mAP
Worst-3 AP
Macro AUPRC
Macro AUROC
Tail AUROC
Tail Recall@K

training loss
```

Also log, if practical:

```text
positive loss contribution:
  head
  medium
  tail

negative loss contribution:
  head
  medium
  tail
```

And:

```text
decoder gradient norm
EHR encoder gradient norm
adapter gradient norm
```

Purpose:

> verify that reweighting actually changes gradient allocation toward rare positive labels.

---

# 16. Safety rules

## Weight stability

```text
0.5 <= w <= 3.0
```

## Gradient stability

If gradient norm becomes:

```text
>3x baseline
```

and training becomes unstable:

```text
stop and record instability
```

Do not silently alter LR in that arm.

## No sampler changes

Use:

```text
natural sampling only
```

No tail oversampling.

## No gamma tuning

Keep current ASL:

```text
gamma_pos
gamma_neg
clip
```

fixed.

---

# 17. Stage 2 — multi-seed replication

After Stage 1 select:

```text
best immediate WASL
best Deferred WASL
```

If one family clearly dominates, retain only the strongest candidate plus baseline.

Use seeds:

```text
42
1337
2024
```

At Stage 2 run both:

```text
Late unconditional L_U
Late EHR L_E
```

under each retained loss.

This allows separation of:

```text
general tail-loss effect
EHR-specific tail effect
loss × EHR interaction
```

---

# 18. Stage-2 estimands

For retained loss l:

## Tail loss effect without EHR

\[
D_{tail,U}^{(l)}
=
TailmAP(L_U^{(l)})
-
TailmAP(L_U^{ASL})
\]

## Tail loss effect with EHR

\[
D_{tail,E}^{(l)}
=
TailmAP(L_E^{(l)})
-
TailmAP(L_E^{ASL})
\]

## Tail EHR gain

\[
G_{EHR,tail}^{(l)}
=
TailmAP(L_E^{(l)})
-
TailmAP(L_U^{(l)})
\]

## Macro EHR gain

\[
G_{EHR,macro}^{(l)}
=
MacromAP(L_E^{(l)})
-
MacromAP(L_U^{(l)})
\]

## Loss × EHR tail interaction

\[
I_{loss\times EHR,tail}
=
G_{EHR,tail}^{(l)}
-
G_{EHR,tail}^{ASL}
\]

---

# 19. Stage-2 reporting

Report:

```text
per-seed values
per-seed deltas
3-seed mean
range
```

For final selected candidate use paired subject-level bootstrap for:

```text
Tail mAP delta
Macro mAP delta
Worst-3 AP delta
Tail Recall@K delta
```

Do not rely on training-seed p-values alone.

---

# 20. Key interpretation cases

## Case A — clear success

Example:

```text
Tail mAP      +0.004
Worst-3 AP    +0.005
Tail Recall@K +0.04
Macro mAP     +0.001
```

Interpretation:

> explicit inter-class reweighting improves tail recognition without harming overall performance.

Adopt weighted/deferred ASL as new default for the next architecture experiment.

---

## Case B — tail improves but macro drops

Example:

```text
Tail mAP  +0.006
Macro mAP -0.003
```

Interpretation:

> class weighting contains useful tail signal but is too aggressive.

Do not modify current runs.

Discuss a weaker weighting schedule only in a new experiment.

---

## Case C — no tail improvement

Example:

```text
Tail mAP      ~0 delta
Tail Recall@K ~0 delta
```

Interpretation:

> inter-class positive-gradient allocation is probably not the dominant tail bottleneck.

Then move to other hypotheses such as:

```text
multi-view evidence
missingness/time-aware EHR
tail augmentation
disease-aware fusion
```

---

## Case D — EHR tail gain becomes positive under weighted ASL

Example:

```text
ASL:
G_EHR_tail = +0.0006

Weighted ASL:
G_EHR_tail = +0.0040
```

This is particularly important.

Interpretation:

> EHR contains useful residual signal for rare labels, but under unweighted ASL the rare-label gradients were too weak for the fusion pathway to learn how to exploit that signal.

This would strongly motivate disease-aware EHR fusion next.

---

# 21. Exact first experiment to run

```text
MODEL:
current Late Full-EHR L_E

SEED:
42

MAX EPOCHS:
8

DECODER:
same delayed-unfreeze protocol

LOSS ARMS:
L0 ASL
L1 WASL alpha=.25
L2 WASL alpha=.50
L3 Deferred WASL alpha=.25
L4 Deferred WASL alpha=.50

DEFERRED SWITCH:
40% of training / approximately epoch 4

WEIGHTS:
positive ASL term only

WEIGHT RANGE:
0.5–3.0

SAMPLING:
natural

PRIMARY METRICS:
Tail mAP
Worst-3 AP
Tail Recall@K

OVERALL SAFEGUARD:
Macro mAP

SUPPORTING:
Macro AUPRC
Macro AUROC
Tail AUROC

CHECKPOINT:
Macro-constrained tail selection

NO TEST ACCESS
NO ARCHITECTURE CHANGE
NO SAMPLING CHANGE
NO ASL-GAMMA CHANGE
```

---

# 22. Stop point

After Stage 1:

- return the five-arm result table;
- return per-label AP deltas;
- return class weights;
- return training diagnostics.

Do **not** start new architecture/loss families automatically.

Interpret Stage-1 results first, then decide whether Stage 2 replication is justified.
