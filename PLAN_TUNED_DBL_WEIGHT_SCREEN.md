# PLAN — Tuned Distribution-Balanced Loss Screen

## Context

Notebook 12 đã hoàn tất Stage-1 Weighted / Deferred ASL screen, seed 42.

Selected validation results:

| Loss | Epoch | Macro mAP | Tail mAP | Worst-3 fixed | Tail Recall@K |
|---|---:|---:|---:|---:|---:|
| ASL | 3 | 0.4406 | 0.1210 | 0.1370 | 0.1963 |
| WASL .25 | 2 | 0.4405 | 0.1212 | 0.1378 | 0.1936 |
| **WASL .50** | **2** | **0.4405** | **0.1230** | **0.1388** | **0.1963** |
| DWASL .25 | 4 | 0.4405 | 0.1209 | 0.1375 | 0.1989 |
| DWASL .50 | 4 | 0.4406 | 0.1211 | 0.1378 | 0.2016 |

Only WASL alpha=.50 passed the preregistered "promising" rule.

However, it is fragile:

```text
WASL .50 tail mAP:
ep2  0.1230
ep3  0.1198
ep4  0.1174
ep5  0.1171
ep6  0.1142
ep7  0.1136
ep8  0.1098
```

Thus the current signal is:

> stronger inter-class positive weighting can improve the tail briefly, but the gain overfits rapidly.

Before committing to 3-seed WASL replication, run one compact tuned DB-Loss screen.

---

# 1. Scientific rationale

The previous result:

```text
ASL > DB Loss
```

must be interpreted as:

> Under the previously tested DB-Loss configuration, ASL performed better.

It does NOT establish:

> distribution-aware rebalancing is ineffective for this cohort.

Distribution-Balanced Loss (DBL) was specifically designed for long-tailed multi-label classification and combines:

1. co-occurrence-aware re-balanced weighting;
2. negative-tolerant regularization (NTR).

This is particularly relevant because the current problem is both:

```text
long-tailed
+
multi-label with substantial co-occurrence
```

and because the current WASL arm shows rapid tail overfitting.

---

# 2. Important distinction: what "tuning DBL weights" means

Do NOT initially do:

```text
DBL × external inverse-frequency class weight
```

or:

```text
DBL × WASL n_c^-alpha
```

DBL already computes sample × class weights from label frequencies and co-occurrence.

Adding another inverse-frequency multiplier immediately would:

- double-count imbalance;
- obscure whether improvement came from DBL or external weighting;
- potentially make rare-class gradients excessively large;
- make comparison with the original DBL method difficult.

Also do NOT tune the generic:

```text
loss_weight
```

scalar.

If DBL is the only objective, changing a global `loss_weight` is largely equivalent to changing the effective optimization scale and confounds loss design with learning rate.

### Primary "weight tuning" target

Tune the DBL re-balancing mapping:

\[
\hat r_{k,i}
=
\alpha_{map}
+
\sigma\left[
\beta_{map}(r_{k,i}-\mu)
\right].
\]

The official implementation uses the name:

```python
map_param = {
    'alpha': ...,
    'beta': ...,
    'gamma': ...
}
```

where `map_param['gamma']` corresponds to the paper's threshold/location parameter \(\mu\).

Do not confuse this with focal-loss gamma.

---

# 3. New notebook

Create:

```text
13_dbl_weight_screen.ipynb
```

Do NOT edit notebook 12.

Copy from notebook 12:

```text
data prep
split
RAD-DINO/H5 loading
Full-EHR encoder
late conditional adapters blocks 10–11
ArmModel
forward_shared
optimizer
scheduler
delayed decoder unfreeze
L2-SP
study-level 7:3 pooling
all seven metrics
checkpoint-selection logic
diagnostics
Colab resume machinery
```

Only add/change:

```text
DBL implementation
DBL hyperparameter arms
DBL-specific weight diagnostics
```

---

# 4. Freeze all non-loss variables

Exactly retain notebook 12:

```text
architecture
H5 warm start
late Full-EHR L_E
seed = 42
train / val split
natural sampler
augmentations
batch size / effective batch
optimizer
LR_adapter = 1e-4
LR_decoder = 1e-5
weight decay
scheduler
decoder frozen epoch 1 then unfrozen
L2-SP
gradient clipping
7:3 frontal/lateral study pooling
8 fixed epochs
NO early stopping
```

Test remains untouched.

---

# 5. Mandatory audit of the old DB configuration

Before GPU training, locate the exact old `ResampleLoss(...)` call used in the project.

Print and save:

```text
reweight_func
focal enabled / disabled
focal balance_param
focal gamma
map alpha
map beta
map gamma  (= paper mu)
neg_scale  (= NTR lambda in implementation)
init_bias  (= NTR prior-bias strength)
loss_weight
sampler
class-frequency source
train_num
```

Do not infer these settings from memory.

### Why this matters

The official COCO DB configuration uses approximately:

```python
reweight_func='rebalance'

focal=dict(
    focal=True,
    balance_param=2.0,
    gamma=2
)

map_param=dict(
    alpha=0.1,
    beta=10.0,
    gamma=0.2
)

logit_reg=dict(
    neg_scale=2.0,
    init_bias=0.05
)
```

and the official COCO training configuration also uses a class-aware sampler.

For our controlled experiment:

```text
KEEP natural sampling
```

because changing sampler and loss simultaneously would confound the comparison.

The official configuration is therefore a reference, not something to reproduce blindly.

---

# 6. Stage 0 — DBL implementation checks

## 6.1 Frequency source

Use:

```text
study-level TRAIN counts
```

from the same current train cohort as notebook 12.

Recompute directly from current training studies.

Cross-check against `class_freq.pkl`.

Never use image-level counts because multi-view studies would be counted multiple times.

---

## 6.2 Legacy reproducibility

For a fixed toy batch of logits and multilabel targets:

```text
legacy DB code
vs
new notebook DB code
```

must return the same loss and gradients when given identical configuration.

Recommended tolerance:

```text
loss abs diff < 1e-7
max gradient diff < 1e-6
```

---

## 6.3 Inspect DBL weights

Unlike WASL, DBL's weight is not simply one fixed scalar per class.

The re-balanced weight depends on:

```text
class frequency
+
the other positive labels co-occurring in that study
```

Therefore save distributions, not only a 13-number class-weight table.

For every DBL arm report:

```text
overall:
  min
  p05
  median
  mean
  p95
  max

by tier:
  head
  medium
  tail

for fixed Worst-3:
  Fracture
  Pleural_Other
  Enlarged_Cardiomediastinum
```

Also report mean positive DB weight per class.

---

# 7. Stage 1A — tune only the re-balancing mapping

The first screen should tune the location/strength of DBL's own re-balancing.

Do NOT tune NTR and mapping simultaneously.

## Fixed settings

Unless the legacy audit reveals a strong reason otherwise:

```text
map alpha = 0.1
map beta  = 10

NTR neg_scale = 2
NTR init_bias = 0.05

focal flag / focal gamma / balance_param:
keep fixed across ALL DBL arms
```

If the legacy project uses DB-Focal rather than pure DB, keep that same DB-Focal family for the first screen.

Do not silently switch the family.

## Arms

### A0 — ASL anchor

Retrain one ASL model inside notebook 13.

Purpose:

```text
detect any code-path/data-order regression
provide perfectly paired seed-42 reference
```

Expected selected result should be close to notebook 12:

```text
Macro ~0.4406
Tail  ~0.1210
```

### D0 — legacy DBL

Exact old project DB configuration.

This answers:

> Does the old DB configuration still perform poorly under the current strong patch + deep-EHR architecture?

### D1 — DBL mu=.10

```text
map alpha = .1
map beta  = 10
mu        = .10
```

### D2 — DBL mu=.20

```text
map alpha = .1
map beta  = 10
mu        = .20
```

### D3 — DBL mu=.30

```text
map alpha = .1
map beta  = 10
mu        = .30
```

All other loss settings identical.

If `D0` is exactly identical to one of D1–D3, do not waste compute on the duplicate.

Thus Stage 1A contains:

```text
1 ASL anchor
+
3–4 unique DBL arms
```

---

# 8. Why tune mu first?

For the smoothed rebalancing function:

\[
\hat r
=
\alpha+\sigma[\beta(r-\mu)]
\]

- `mu` determines where a raw co-occurrence-aware weight starts being strongly upweighted;
- `beta` determines how sharp that transition is;
- `alpha` raises the whole lower bound.

Tuning many parameters simultaneously would make interpretation poor.

The original DBL work used:

```text
mu ~ 0.2 for COCO-MLT
mu ~ 0.3 for VOC-MLT
```

so `.10/.20/.30` is a compact local screen, not an arbitrary huge search.

---

# 9. Training protocol

Exactly as notebook 12:

```text
seed: 42
epochs: 8 fixed
early stopping: OFF
```

Do not shorten to 2–3 epochs just because WASL peaked early.

We specifically want to observe:

```text
peak performance
+
post-peak stability / overfitting trajectory
```

All DBL arms should begin from exactly the same model initialization.

Where feasible:

```text
same minibatch order
same augmentation realization
same dropout RNG per step
```

as notebook 12's paired-arm protocol.

No fork is needed because DBL variants differ from epoch 1.

---

# 10. Metrics

Keep exactly the locked metric set:

```text
Macro mAP
Tail mAP
Worst-3 fixed AP
Macro AUPRC
Macro AUROC
Tail AUROC
Tail Recall@K
```

Worst-3 fixed labels remain:

```text
Fracture
Pleural_Other
Enlarged_Cardiomediastinum
```

Do NOT redefine Worst-3 from the DBL arms.

---

# 11. Checkpoint selection

Use the exact notebook-12 rule.

For each arm:

1. find best validation Macro mAP;
2. retain epochs with:

```text
Macro mAP >= best Macro mAP - 0.0015
```

3. among them maximize:

```text
Tail mAP
```

Tie-break:

```text
Worst-3 fixed AP
→ Tail Recall@K
→ Macro mAP
```

No test access.

---

# 12. Extra overfitting/stability diagnostics

Because overfitting is now a central observed failure mode, add:

```text
best_tail
epoch_best_tail
tail_ep8
tail_peak_drop = best_tail - tail_ep8

best_macro
macro_ep8
macro_peak_drop = best_macro - macro_ep8
```

Also report:

```text
Tail mAP by epoch
Worst-3 fixed AP by epoch
Tail Recall@K by epoch
train loss by epoch
```

Current reference:

```text
WASL .50:
Tail peak = 0.1230 @ ep2
Tail ep8  = 0.1098
drop      ≈ 0.0132
```

A DBL configuration with similar selected tail performance but substantially smaller collapse is scientifically attractive.

Do NOT make stability the sole selection criterion; it is a secondary diagnostic.

---

# 13. Go / no-go versus ASL

Same primary gate used in notebook 12:

```text
Delta Tail mAP >= +0.002
Delta Macro mAP >= -0.002
Delta Worst-3 fixed > -0.002
```

Prefer:

```text
Tail Recall@K >= ASL
```

---

# 14. DBL versus current promising WASL .50

Current WASL .50 seed-42 target:

```text
Macro mAP       0.4405
Tail mAP        0.1230
Worst-3 fixed   0.1388
Tail Recall@K   0.1963
```

A DBL arm is clearly more interesting than WASL .50 if:

### Pattern A — higher tail

```text
Tail mAP > 0.1230 by a meaningful margin
Macro remains stable
```

### Pattern B — similar tail, better robustness

For example:

```text
Tail mAP ~0.1225–0.1230
but:
Worst-3 ↑
Tail Recall@K ↑
tail peak→ep8 collapse much smaller
```

This would be important given the severe WASL overfitting.

At seed 42 this remains a screening comparison, not a final superiority claim.

---

# 15. Stage 1B — only if DBL mapping is promising

Do not run this automatically.

If one mapping arm passes the primary gate, freeze its:

```text
map alpha
map beta
mu
focal settings
```

Then tune only NTR negative scaling:

```text
neg_scale lambda ∈ {2, 3, 5}
```

Keep:

```text
init_bias = 0.05
```

fixed initially.

Reason:

- DBL's NTR is specifically designed to reduce negative over-suppression;
- the original paper found the useful lambda range to be dataset-dependent;
- current WASL trajectory shows a strong rare-class generalization-collapse problem.

Do not tune `init_bias/kappa` at the same time.

The original supplementary analysis suggests small changes to the prior-bias strength have relatively small effect, so it is a lower-priority tuning dimension.

---

# 16. What NOT to do yet

Do not simultaneously search:

```text
mu
beta
neg_scale
init_bias
focal gamma
focal balance
LR
sampler
external class weight
```

That would become an unprincipled hyperparameter hunt.

Do not use a class-aware sampler in this screen.

Do not multiply DBL by the WASL weights yet.

Do not change EHR architecture.

Do not change the multi-view rule.

---

# 17. Optional next DBL tuning dimension

Only after the `mu` screen, if needed:

```text
beta = 5 vs 10
```

at the best `mu`.

This tests a smoother versus sharper weight mapping.

Do not run a full:

```text
3 mu × 2 beta × 3 lambda
```

grid.

---

# 18. Decision after DBL screen

## Case A — tuned DBL beats WASL

Then Stage 2 candidates:

```text
ASL
WASL .50
best tuned DBL
```

Run seeds:

```text
42
1337
2024
```

## Case B — DBL beats ASL but not WASL

Keep DBL as a useful comparator.

Replicate only if it offers:

```text
better stability
or better Worst-3 / Recall@K
```

that makes it scientifically different from WASL.

## Case C — tuned DBL still clearly loses

Then the conclusion becomes much stronger:

> A carefully tuned Distribution-Balanced Loss did not outperform ASL/WASL under the current RAD-DINO patch-token + structured-EHR architecture.

At that point deprioritize DBL.

---

# 19. Recommended execution order

```text
1. Create 13_dbl_weight_screen.ipynb from notebook 12.
2. Audit exact legacy ResampleLoss parameters.
3. Recompute study-level TRAIN frequencies.
4. Verify DB implementation on fixed toy batch.
5. Print DB weight distributions.
6. Run A0 ASL anchor + unique D0–D3 arms for 8 epochs.
7. Apply exactly the same checkpoint selection as notebook 12.
8. Compare against ASL and WASL .50.
9. Inspect overfitting trajectory.
10. STOP.
11. Only if DBL is promising, decide whether Stage 1B lambda tuning is justified.
```

---

# 20. Literature anchors

Wu et al., ECCV 2020:
**Distribution-Balanced Loss for Multi-Label Classification in Long-Tailed Datasets**
https://arxiv.org/abs/2007.09654

Official implementation:
https://github.com/wutong16/DistributionBalancedLoss

Official COCO config:
https://raw.githubusercontent.com/wutong16/DistributionBalancedLoss/master/configs/coco/LT_resnet50_pfc_DB.py

CXR-LT 2024 overview:
https://pmc.ncbi.nlm.nih.gov/articles/PMC12306832/

Key methodological point:

> DBL should first be given a fair, controlled tuning of its own co-occurrence-aware rebalancing mechanism. External inverse-frequency weights should not be stacked on top until DBL itself has been understood.
