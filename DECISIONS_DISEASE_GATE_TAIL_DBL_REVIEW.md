# DECISIONS — Disease-Aware Gate and Tail-Selective DBL

Reviewed document:

```text
REVIEW_DISEASE_GATE_WITH_TAIL_SELECTIVE_DBL.md
```

## Overall decision

Proceed in two distinct stages:

```text
Stage A:
logit-level disease-aware EHR residual fusion
plain ASL only

Stage B:
tail-selective DBL auxiliary
only if Stage A demonstrates real EHR-content use
```

Do not retain the old in-backbone CondAdapter in Stage A.

This is a new architecture and must be reported as such.

---

# Q1 — Where does z_IMG come from?

## Decision: choose option (a), logit-level fusion

Use the current strong H5 image model without EHR conditioning:

```text
RAD-DINO patch tokens
→ unconditioned ML-Decoder
→ z_IMG
```

Then add a new EHR residual pathway:

\[
z_c^{MM}
=
z_c^{IMG}
+
g_c\delta_c.
\]

This is preferable to two-forward or dual-decoder alternatives because it gives:

- an explicit image-only fallback;
- exact identity initialization;
- cheap inference/training;
- clear attribution of the EHR residual;
- direct compatibility with tail-selective auxiliary supervision.

The old block-10/11 conditional adapters are not used in the primary Stage-A arms.

They remain an external global-conditioning reference.

---

## Gate inputs

Preferred if ML-Decoder class features are accessible:

\[
g_c
=
\sigma f_g
\left(
\operatorname{sg}(h_c^{IMG}),
q_c,
e,
m
\right),
\]

\[
\delta_c
=
f_\delta(q_c,e,m).
\]

Where:

```text
h_c_IMG:
pre-logit disease-specific decoder feature

q_c:
disease embedding

e:
Full-EHR embedding

m:
EHR missingness/completeness summary
```

If the current ML-Decoder implementation does not expose \(h_c^{IMG}\), use:

```text
sg(z_c_IMG)
+
q_c
+
e
+
m
```

as the minimal gate input.

A scalar image logit is acceptable for the screen, but a class feature is more informative.

---

## Initialization

Use:

```text
delta output layer:
exact zero initialization
```

This guarantees:

\[
z^{MM}=z^{IMG}
\]

at step 0.

### Gate bias

Use:

```text
gate bias = 0
```

for the first screen, giving initial \(g\approx0.5\).

Reason:

- exact identity is already guaranteed by \(\delta=0\);
- a strongly negative gate bias would unnecessarily reduce residual-output gradients;
- gate gradients are already delayed until the residual becomes non-zero.

Protect the image pathway using:

```text
image-only auxiliary loss
L2-SP
short delayed decoder unfreeze
```

rather than an extremely closed initial gate.

Mandatory step-0 test:

```text
max |z_MM - z_IMG| < 1e-6
```

in eval FP32.

---

# Q2 — Which tail target set?

## Decision: use Rare-3 for Stage-B DBL supervision

Lock from TRAIN counts:

```text
Rare-3:
Pleural_Other
Fracture
Lung_Lesion
```

Do not redefine the existing tier labels.

Continue reporting:

```text
original Tail AP:
Pleural_Other only

Rare-3 mAP:
new targeted rare-set metric
```

The original tail metric remains for continuity, but it is not sufficient by itself for go/no-go.

---

## Additional safeguard metric

Add:

```text
NonTarget-10 mAP
```

defined as macro AP over the 10 labels not in Rare-3.

This is more directly informative than relying only on Worst-3:

> Does tail-selective DBL improve its intended targets without causing broad collateral damage?

Also retain:

```text
Worst-3 fixed AP
Enlarged_Cardiomediastinum AP
```

as secondary diagnostics.

The overlap between Rare-3 and Worst-3 is not leakage. All labels are supervised.

However, it means Worst-3 improvement cannot be described as out-of-target generalization.

Therefore report the non-target label separately and use NonTarget-10 as the main collateral-damage safeguard.

---

## Uncertainty

For:

```text
Pleural_Other AP
Rare-3 mAP
```

compute paired study-level bootstrap intervals against the control arm.

At the one-seed screen, do not require a p-value.

Use:

```text
point estimate
95% paired bootstrap CI
per-label deltas
```

to assess whether the effect is driven by one rare label.

---

# Q3 — How strongly should DBL gradients be isolated?

## Decision: use a stricter partially isolated design as the PRIMARY Stage-B arm

This differs from the review's default.

The motivation for Stage B is to retain the DBL tail signal while avoiding the broad degradation caused by global DBL.

Therefore the primary auxiliary branch should not allow DBL to modify:

```text
image decoder
shared EHR encoder
```

Use:

\[
e=EHREncoder(x),
\]

\[
e_T=P_T(\operatorname{sg}(e)),
\]

\[
z_c^{DB}
=
\operatorname{sg}(z_c^{IMG})
+
g_c^T\delta_c^T(e_T),
\qquad c\in T.
\]

Then:

\[
L_{DB,T}
=
DBL(z_T^{DB},y_T).
\]

DBL auxiliary gradients update:

```text
tail-specific projection P_T
tail gate/residual
tail-specific disease parameters
```

but not:

```text
shared image pathway
shared EHR encoder
```

The shared EHR encoder is still trained by the global ASL objective.

This is not a completely frozen EHR system; it only isolates the additional DBL gradient.

---

## Why not use the light version first?

The light version:

```text
sg(z_IMG)
but no sg(e)
```

still lets DBL change the EHR representation used by all 13 diseases.

That reintroduces a plausible collateral-damage pathway—the exact behavior that motivated selective DBL.

Therefore:

```text
strict/partial isolation:
primary

light shared-EHR-gradient version:
follow-up only if the primary is null
```

A useful interpretation:

- strict positive → DBL helps through disease-specific residual specialization;
- strict null but light positive → DBL needs to reshape shared EHR representation;
- light causes broad degradation → confirms the shared-encoder interference hypothesis.

---

# Q4 — Does Stage A exist?

## Decision: build Stage A first

Create:

```text
14_disease_gate.ipynb
```

Stage A uses:

```text
logit-level disease-aware residual architecture
plain ASL
no DBL
```

Primary training arms:

```text
DA-U:
NULL EHR, same architecture and initialization

DA-E:
real matched EHR
```

At the selected DA-E checkpoint evaluate:

```text
REAL
same-setting SHUFFLE
NULL
```

A separately trained shuffle arm is not needed.

Primary estimands:

\[
G_{DA,macro}
=
Macro(DA-E)-Macro(DA-U),
\]

\[
G_{DA,R3}
=
Rare3(DA-E)-Rare3(DA-U).
\]

Proceed to Stage B only if:

```text
DA-E > DA-U
and
REAL > SHUFFLE/NULL
```

on at least macro or the prelocked rare-set outcome without material collateral damage.

If Stage A is null, do not spend GPU on DBL-gated arms.

---

# Stage-A loss

Use:

\[
L_A
=
L_{ASL}(z^{MM},y)
+
\lambda_I L_{ASL}(z^{IMG},y).
\]

Lock:

```text
lambda_I = 0.25
```

for both DA-U and DA-E.

Purpose:

- preserve the strong image-only prediction;
- ensure the fallback remains useful;
- prevent the residual path from compensating for degradation of the image decoder.

No WASL or DBL in Stage A.

---

# Stage-B loss

Only after Stage A passes:

\[
L_B
=
L_{ASL}(z^{MM},y)
+
0.25L_{ASL}(z^{IMG},y)
+
\lambda_T L_{DB,T}.
\]

Do not multiply DBL by the gate.

---

# DBL scaling

Agree with the review:

```text
loss-value matching is not sufficient
gradient ratio is the relevant quantity
```

For Rare-3, remeasure at initialization because the existing values were obtained for one tail column.

Compute:

\[
\rho
=
\frac{
\|\nabla_{\theta_T} \lambda_T L_{DB,T}\|
}{
\|\nabla_{\theta_T} L_{ASL,T}\|
}.
\]

Choose two Stage-B arms by target gradient ratio, not raw lambda:

```text
H1:
rho ≈ 0.25

H2:
rho ≈ 0.50
```

This is deliberately weaker than the previous one-label hybrid, where even \(\lambda=.10\) produced a DB/ASL gradient ratio near 1.11.

Save the resulting numerical lambda values before training.

Do not adapt them using validation results.

---

# Metrics

## Locked main panel

```text
Macro mAP
original Tail AP
Worst-3 fixed AP
Macro AUPRC
Macro AUROC
Tail AUROC
Tail Recall@K
```

## Additional targeted diagnostics

```text
Rare-3 mAP
NonTarget-10 mAP
Enlarged_Cardiomediastinum AP
per-label AP delta
```

---

# Revised go/no-go

## Stage A — disease gate

Promising if:

```text
G_DA_macro >= +0.002
```

or:

```text
G_DA_Rare3 >= +0.002
G_DA_macro >= -0.002
NonTarget-10 >= -0.002
```

and:

```text
REAL > same-setting SHUFFLE/NULL
```

## Stage B — tail-selective DBL

Versus Stage-A DA-E:

```text
Delta Rare-3 mAP >= +0.002
Delta Macro mAP >= -0.002
Delta NonTarget-10 mAP >= -0.0015
```

Prefer:

```text
at least 2 of 3 Rare-3 labels improve
Worst-3 fixed stable or improved
```

The single Pleural_Other AP is supporting evidence, not the sole gate.

---

# Gate-collapse diagnostics

Log per epoch:

```text
mean gate
gate p05 / p50 / p95
mean |g * delta|
mean |g * delta| / mean |z_IMG|
per-label gate statistics
per-label residual contribution
```

At selected checkpoint report under:

```text
REAL
SHUFFLE
NULL
```

Do not interpret a large gate alone as useful EHR reliance.

Useful reliance requires performance and correspondence evidence.

---

# Notebook sequence

```text
14_disease_gate.ipynb
  DA-U
  DA-E
  REAL / SHUFFLE / NULL
  plain ASL only

if Stage A passes:

15_tail_dbl_disease_gate.ipynb
  H0 = selected disease-gate + ASL
  H1 = + isolated Rare-3 DBL, gradient ratio .25
  H2 = + isolated Rare-3 DBL, gradient ratio .50
```

---

# Final decisions

| Question | Decision |
|---|---|
| Q1 | Logit-level fusion; new architecture; use pre-logit class feature when available |
| Gate input | Detached image class feature/logit + disease embedding + EHR + missingness |
| Delta init | Exact zero |
| Gate bias | 0 |
| Q2 | Rare-3 for DBL; keep original tail metric; add NonTarget-10 |
| Q3 | Partially isolated DBL gradient is primary; light shared-EHR version is follow-up |
| Q4 | Build Stage A first; do not combine architecture and DBL in the first screen |
| Macro safeguard | -0.002, not -0.001 |
| DBL strength | Select by gradient ratio .25/.50, not raw lambda |
