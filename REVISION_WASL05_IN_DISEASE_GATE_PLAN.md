# REVISION — Where WASL alpha=0.5 Fits in the Disease-Gate Plan

## 1. Conclusion

WASL alpha=0.5 should not be discarded.

It is currently the strongest practical loss candidate because the seed-42 screen showed approximately:

```text
Macro mAP:
stable versus ASL

Tail AP:
+0.002

Worst-3 fixed:
improved

Tail Recall@K:
approximately stable
```

Its main weakness was severe temporal instability:

```text
Tail AP:
0.1230 at epoch 2
→ 0.1098 at epoch 8
```

Therefore WASL alpha=0.5 should be included in the disease-gate program, but not used as the only loss in Stage A.

---

# 2. Why not use WASL alpha=0.5 in Stage A?

Stage A asks:

> Does the disease-aware EHR gate itself provide useful patient-specific information?

If Stage A changes both:

```text
architecture:
global adapter → disease-aware logit gate

and loss:
ASL → WASL alpha=.50
```

then a positive result cannot be attributed cleanly.

Possible explanations would include:

```text
gate effect
WASL effect
gate × WASL interaction
different overfitting dynamics
```

Therefore Stage A should remain:

```text
DA-U + plain ASL
DA-E + plain ASL
```

This isolates the EHR-content effect of the new architecture.

---

# 3. Why WASL may be more appropriate than DBL after Stage A

The current evidence differs:

## WASL alpha=.50

```text
tail improvement
macro approximately preserved
Worst-3 improved
but rapid overfitting
```

## Global DBL

```text
larger gain for one tail label
but broad degradation across most other labels
macro and Worst-3 decline
```

Thus for practical model improvement:

```text
WASL .50 is the safer first loss extension.
```

DBL remains scientifically interesting because its co-occurrence-aware mechanism produced a distinct tail signal, but it is not currently the best overall loss.

---

# 4. Revised experiment sequence

## Stage A — architecture isolation

Notebook:

```text
14_disease_gate.ipynb
```

Arms:

```text
A0 = DA-U + ASL
A1 = DA-E + ASL
```

At selected A1 checkpoint:

```text
REAL
same-setting SHUFFLE
NULL
```

Proceed only if disease-aware EHR content is mechanistically active.

---

## Stage B — loss interaction with disease gate

Notebook:

```text
15_disease_gate_tail_loss.ipynb
```

Recommended priority:

```text
B0 = disease gate + ASL

B1 = disease gate + global WASL alpha=.50

B2 = disease gate + isolated Rare-3 selective WASL

B3 = disease gate + isolated Rare-3 DBL
     optional / secondary
```

If compute is limited:

```text
run B0, B1, B2 first
```

and defer DBL unless selective WASL is null or DBL is needed as a mechanistic comparator.

---

# 5. Global WASL arm

B1 reproduces the successful notebook-12 loss in the new architecture.

Use exactly the locked WASL formulation:

\[
L_c
=
-w_c y_c(1-p_c)^{\gamma_+}\log p_c
-
(1-y_c)(p'_c)^{\gamma_-}\log(1-p'_c),
\]

with:

```text
alpha = .50
positive term weighted only
same clipping / normalization
same ASL gamma and clip
```

Add the same image-preservation auxiliary:

\[
L_{B1}
=
L_{WASL}(z^{MM},y)
+
0.25L_{ASL}(z^{IMG},y).
\]

Do not weight the image-only auxiliary initially; its role is to preserve the original visual task.

---

# 6. Why add a selective WASL arm?

Global WASL still sends weighted gradients through:

```text
all final disease logits
shared image decoder
shared EHR encoder
```

This may reproduce the rapid overfitting observed previously.

Disease-aware architecture allows a cleaner use of WASL:

> apply only the additional rare-positive emphasis to the rare-disease residual branch.

---

# 7. Selective WASL formulation

Let \(T\) be the prelocked Rare-3 set.

The global ASL objective remains:

\[
L_{MM}=L_{ASL}(z^{MM},y).
\]

For \(c\in T\), define the positive ASL component:

\[
\ell_{c}^{+}
=
-y_c(1-p_c)^{\gamma_+}\log p_c.
\]

WASL alpha=.50 gives fixed class weights \(w_c\).

The extra selective weighting term is:

\[
L_{W,T}^{extra}
=
\frac{1}{|T|}
\sum_{c\in T}
(w_c-1)\ell_c^{+}(z_c^{RES}).
\]

Use:

\[
z_c^{RES}
=
\operatorname{sg}(z_c^{IMG})
+
g_c^T\delta_c^T(P_T(\operatorname{sg}(e))).
\]

Final selective-WASL loss:

\[
L_{B2}
=
L_{ASL}(z^{MM},y)
+
0.25L_{ASL}(z^{IMG},y)
+
\lambda_W L_{W,T}^{extra}.
\]

Important:

```text
only the incremental WASL emphasis is auxiliary
```

Do not add a second full WASL/ASL loss on the same tail logits, which would double-count the baseline objective.

---

# 8. Why selective WASL may be better than selective DBL

Advantages:

1. It is directly grounded in the only promising loss result already observed.
2. It changes only rare positive gradients.
3. It does not alter the negative term.
4. It is simpler to interpret.
5. It avoids DBL's broad head/medium degradation.
6. It can be isolated from shared image and EHR pathways.
7. It tests whether disease-specific residual capacity stabilizes the previous WASL tail gain.

DBL remains useful if:

```text
co-occurrence-aware weighting adds benefit beyond frequency-only WASL.
```

---

# 9. Key scientific interaction

The relevant question is not only:

```text
Does WASL improve tail?
```

It is:

> Does disease-aware residual isolation make the WASL tail gain more stable and less harmful?

Compare the previous global WASL trajectory with B1/B2:

```text
peak Tail AP
epoch of peak
Tail AP at epoch 8
peak-to-epoch8 collapse
Macro and NonTarget-10 degradation
```

A successful disease-gated selective WASL should show:

```text
similar or better Rare-3 gain
smaller peak-to-late collapse
stable NonTarget-10
```

---

# 10. Arm priority

## Minimal Stage-B screen

```text
B0:
Gate + ASL

B1:
Gate + global WASL .50

B2:
Gate + isolated Rare-3 selective WASL
```

## Extended screen

Add:

```text
B3:
Gate + isolated Rare-3 DBL
```

Only if compute and time permit.

This ordering reflects the current evidence:

```text
WASL:
best practical candidate

DBL:
mechanistically interesting but more destructive
```

---

# 11. Selection and metrics

Keep:

```text
Macro mAP
original Tail AP
Rare-3 mAP
NonTarget-10 mAP
Worst-3 fixed
Tail Recall@K
Macro AUPRC
Macro AUROC
Tail AUROC
```

Checkpoint rule remains macro-constrained tail selection.

Add stability:

```text
Rare-3 peak
epoch of Rare-3 peak
Rare-3 at epoch 8
peak-drop
```

---

# 12. Go/no-go for selective WASL

Versus B0:

```text
Delta Rare-3 mAP >= +0.002
Delta Macro mAP >= -0.002
Delta NonTarget-10 mAP >= -0.0015
at least 2 of 3 Rare-3 labels improve
```

Strong:

```text
Rare-3 improves
Worst-3 improves
Macro stable
NonTarget-10 stable
peak-to-epoch8 collapse clearly smaller than previous WASL .50
```

---

# 13. Final recommendation

Use:

```text
plain ASL for Stage A
```

to establish the architecture effect.

Then prioritize:

```text
global WASL alpha=.50
and
isolated Rare-3 selective WASL
```

in Stage B.

Treat isolated Rare-3 DBL as a secondary comparator rather than the primary tail-loss extension.

This revision better reflects the empirical evidence already obtained.
