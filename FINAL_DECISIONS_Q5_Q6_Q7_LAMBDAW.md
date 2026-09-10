# FINAL DECISIONS — Remaining Questions for Disease-Gate Tail-Loss Design

Date: 2026-07-29

Resolved items:

```text
Q5 — implementation of the isolated auxiliary path
Q6 — exact definition of gradient ratio rho
Q7 — definitions of q_c and missingness summary m
lambda_W — strength of selective WASL arm B2
```

---

# Q5 — How should the "strict" auxiliary path be implemented?

## Decision

Use the **same disease-gate/residual head and the same parameters** as Stage A.

Do not add `P_T` or a second tail head in the first Stage-B screen.

The architecture must be identical across:

```text
B0 = gate + ASL
B1 = gate + global WASL .50
B2 = gate + selective WASL
B3 = gate + selective DBL
```

Only the loss changes.

---

## Main forward

For all classes:

\[
z_c^{MM}
=
z_c^{IMG}
+
g_c\left(
\operatorname{sg}(h_c^{IMG}),q_c,e,m
\right)
\delta_c(q_c,e,m).
\]

The main ASL/WASL objective updates:

```text
image decoder
EHR encoder
gate/residual head
learned disease embeddings
```

according to the selected arm.

---

## Auxiliary isolated forward

For the prelocked Rare-3 classes only:

\[
z_c^{AUX}
=
\operatorname{sg}(z_c^{IMG})
+
g_c\left(
\operatorname{sg}(h_c^{IMG}),q_c,\operatorname{sg}(e),\operatorname{sg}(m)
\right)
\delta_c\left(
q_c,\operatorname{sg}(e),\operatorname{sg}(m)
\right).
\]

Use the **same gate/residual parameters** as the main forward.

The auxiliary WASL/DBL gradient is blocked from:

```text
image decoder
shared EHR encoder
missingness construction
```

but still updates:

```text
gate/residual parameters
Rare-3 disease embeddings q_c
```

---

## Important terminology correction

This is:

```text
image- and EHR-encoder-isolated
```

but it is not necessarily:

```text
fully class-parameter-isolated
```

if the gate/residual MLP weights are shared across all 13 diseases.

Rare-3 auxiliary gradients can still change shared gate/residual parameters and indirectly alter non-target classes.

Therefore:

```text
NonTarget-10 mAP
```

remains mandatory.

Do not claim that the auxiliary gradient updates only tail-specific parameters unless the head is later redesigned with class-specific adapters.

---

## Why no P_T in the first screen?

Adding `P_T` only to B2/B3 would confound:

```text
extra tail capacity
+
selective loss
```

If a tail-specific projection is later tested, it must be present in B0/B1/B2/B3 with identical initialization.

The first screen should avoid this confound.

---

## Numerical equality of main and auxiliary logits

Because `detach()` preserves numerical values:

```text
e and sg(e) are numerically identical
```

the main and auxiliary residuals should be equal if the head is deterministic.

For the first screen:

```text
set gate/residual dropout = 0
```

This ensures:

```text
max |z_AUX - z_MM[T]| < 1e-6
```

before considering gradient paths.

The two forwards differ only in gradient routing.

---

# Q6 — Exact definition of rho

## Decision

Define rho against the **actual global main-loss gradient** on the exact same parameter set that the auxiliary loss can update.

Let:

```text
theta_res =
gate MLP
residual MLP
learned disease embeddings
```

Then:

\[
\rho
=
\frac{
\left\|
\nabla_{\theta_{res}}
\left(
\lambda_T L_{DB,T}^{AUX}
\right)
\right\|_2
}{
\left\|
\nabla_{\theta_{res}}
L_{ASL}(z^{MM},y)
\right\|_2
}.
\]

Do not use tail-only ASL in the denominator.

Do not include `0.25 ASL(z_IMG)` in the denominator because it has zero gradient with respect to `theta_res`.

---

## Calibration procedure

At initialization:

1. Lock a fixed TRAIN calibration subset, for example 16 deterministic minibatches.
2. Disable AMP for gradient calibration.
3. For each batch, separately compute:
   - `g_main = grad_theta_res L_ASL(z_MM)`
   - `g_aux  = grad_theta_res L_DB,T^AUX` at `lambda=1`
4. Aggregate:

\[
R_0
=
\sqrt{
\frac{
\sum_b \|g_{aux,b}\|_2^2
}{
\sum_b \|g_{main,b}\|_2^2
}
}.
\]

5. Set:

\[
\lambda_T
=
\frac{\rho_{target}}{R_0}.
\]

No bisection is needed because gradient magnitude scales linearly with `lambda_T`.

Lock the resulting numerical lambda before training.

---

## Target values

For selective DBL:

```text
primary:
rho = 0.25

optional stronger arm:
rho = 0.50
```

If compute permits only one DBL arm, choose `rho=.25`.

---

# lambda_W — Selective WASL strength for B2

## Decision

Use:

\[
\lambda_W
=
\frac{|T|}{C}
=
\frac{3}{13}
=
0.230769.
\]

This is the primary and only B2 strength in the first screen.

---

## Why this value?

The selective WASL term is:

\[
L_{W,T}^{extra}
=
\frac{1}{|T|}
\sum_{c\in T}
(w_c-1)\ell_c^+.
\]

The main ASL loss averages over all \(C=13\) classes.

Setting:

\[
\lambda_W=\frac{|T|}{C}
\]

makes the incremental positive coefficient for each Rare-3 class exactly match global WASL alpha=.50.

Therefore:

```text
B1:
same WASL emphasis flows through the full network

B2:
same WASL emphasis flows through the isolated residual path
```

This creates the cleanest comparison of gradient routing.

---

## Do not add a second lambda_W arm initially

Do not automatically run:

```text
lambda_W = 6/13 = 0.461538
```

That doubles the positive emphasis and changes both strength and pathway.

Run it only as a declared follow-up if:

```text
B2 is safe but apparently underpowered
```

and not based on validation tuning.

---

# Q7 — Definitions of q_c and m

## q_c decision

Do not use the frozen random ML-Decoder query embedding as the disease representation.

Add:

```python
nn.Embedding(num_embeddings=13, embedding_dim=64)
```

Properties:

```text
trainable
shared identically across arms
normal initialization, std ≈ .02
```

This embedding represents disease identity for the gate/residual module.

The visual evidence still comes from:

```text
h_c_IMG, shape [B,13,768]
```

Thus:

```text
h_c_IMG:
patient-specific visual evidence for disease c

q_c:
learned global identity of disease c
```

---

## m decision

Use an 8-dimensional normalized missingness/completeness vector:

```text
m1 = mean(lab_mask)
m2 = mean(vital_mask)
m3 = comorbidity observation fraction / comorbidity mask
m4 = mean(T1 recency masks)
m5 = T3 recency availability mask
m6 = indicator(no T1 specimen in the allowed window)
m7 = indicator(no T3 specimen / recency bin is none)
m8 = log1p(number of valid microbiology tokens)
     / log1p(max valid token count)
```

All components should lie in:

```text
[0,1]
```

as far as possible.

`m` is not new clinical information. It explicitly exposes EHR reliability already partly encoded inside the EHR encoder.

---

## NULL and shuffle handling

### DA-U / NULL evaluation

Use a true all-missing signature:

```text
observed fractions = 0
availability masks = 0
missing indicators = 1
micro token count = 0
```

Do not represent NULL EHR as an arbitrary all-zero `m`, because zero missing indicators would incorrectly imply data are present.

### SHUFFLE evaluation

Shuffle as one consistent patient package:

```text
EHR values
EHR masks
microbiology tokens
m
```

must all use the same permutation.

Prefer same-setting shuffle.

Do not shuffle `e` while leaving `m` from the original patient.

---

# Final locked architecture for Stage A

\[
u_c=
[
\operatorname{sg}(h_c^{IMG}),
q_c,
e,
m
].
\]

\[
g_c=\sigma(f_g(u_c)).
\]

\[
\delta_c=f_\delta(q_c,e,m).
\]

\[
z_c^{MM}=z_c^{IMG}+g_c\delta_c.
\]

Locked settings:

```text
q_c:
learned 64-d embedding

m:
8-d normalized summary

delta output:
zero-init

gate bias:
0

gate/residual dropout:
0 for the first screen

Stage-A loss:
ASL(z_MM) + .25 ASL(z_IMG)
```

---

# Final answers table

| Item | Decision |
|---|---|
| Q5 | Same head, second detached-input forward; no `P_T`; identical architecture across B0–B3 |
| Q5 caveat | Encoder-isolated, not fully class-parameter-isolated because the residual MLP is shared |
| Q6 denominator | Full main ASL gradient on the same `theta_res`, not tail-only ASL |
| Q6 calibration | Fixed 16-batch TRAIN subset; RMS norm aggregation; lambda = target rho / R0 |
| DBL rho | .25 primary; .50 optional |
| lambda_W | `3/13 = 0.230769` |
| q_c | New learned `Embedding(13,64)` |
| m | Locked normalized 8-d completeness summary |
| NULL | All-missing signature, not naive all-zero indicators |
| SHUFFLE | Shuffle EHR and `m` together |
