# Disease-Aware Gate + Tail-Selective DBL

## 1. Kết luận

Có thể kết hợp disease-aware gate với DBL cho tail diseases.

Đây là hướng hợp lý hơn global DBL vì kết quả hiện tại cho thấy:

```text
global DBL:
tail label ↑
nhưng 12/13 labels ↓
macro và Worst-3 ↓
```

Disease-aware residual fusion cho phép DBL tác động chủ yếu lên:

```text
tail disease query
+
tail EHR gate/residual
```

thay vì làm thay đổi toàn bộ shared representation.

Tuy nhiên, nếu:

```text
ASL cho head/medium
DBL cho tail
```

thì về mặt loss nó vẫn là một dạng hybrid. Điểm mới là hybrid được gắn vào kiến trúc disease-aware và cô lập gradient tốt hơn.

---

# 2. Không nên gộp ngay trong experiment đầu tiên

Experiment đầu tiên của disease gate nên dùng:

```text
plain ASL only
```

Lý do:

- cần biết disease-aware fusion tự nó có hiệu quả hay không;
- nếu thay cả architecture và loss cùng lúc, không thể phân biệt gain đến từ gate hay DBL;
- current DBL đã có một tail signal nhưng cũng có broad degradation.

Recommended sequence:

```text
Stage A:
disease-aware gate + ASL

Stage B:
nếu gate dùng EHR thật nhưng tail vẫn yếu,
thêm tail-selective DBL
```

---

# 3. Kiến trúc

For disease c:

\[
z_c^{MM}
=
z_c^{IMG}
+
g_c\delta_c^{EHR}.
\]

Where:

```text
z_img:
strong image-only logit

g_c:
patient × disease gate

delta_c:
disease-specific EHR residual
```

Image-only prediction remains the fallback.

---

# 4. Recommended loss design

## Main multimodal loss

Apply ASL to all final logits:

\[
L_{MM}
=
L_{ASL}(z^{MM},y).
\]

## Image-preservation auxiliary loss

Apply ASL to the image-only logits:

\[
L_{IMG}
=
L_{ASL}(z^{IMG},y).
\]

This discourages the multimodal model from sacrificing the strong image solution.

## Tail-selective DBL auxiliary

Let \(T\) be the prelocked rare-label set.

Construct:

\[
z_c^{RES}
=
\operatorname{sg}(z_c^{IMG})
+
g_c\delta_c^{EHR},
\qquad c\in T,
\]

where `sg` is stop-gradient.

Then:

\[
L_{DB,T}
=
L_{DBL}(z_T^{RES},y_T).
\]

Final loss:

\[
L
=
L_{MM}
+
\lambda_I L_{IMG}
+
\lambda_T L_{DB,T}.
\]

Recommended initial values:

```text
lambda_I = 0.25
lambda_T = 0.10 or 0.25
```

Do not start with a large lambda_T.

---

# 5. Why stop-gradient is useful

Without stop-gradient, tail DBL gradients can still alter:

```text
shared image decoder
shared visual representation
```

and reproduce the same broad degradation observed with global DBL.

Using:

```text
stop_gradient(z_img)
```

inside the DBL auxiliary term means the extra DBL signal primarily updates:

```text
EHR encoder
tail gate
tail residual
tail-specific query/projection parameters
```

The normal ASL loss still jointly trains the final model.

---

# 6. Do not multiply DBL by the gate

Avoid:

\[
L_{tail}=g_c L_{DBL,c}.
\]

This creates a trivial escape route:

```text
model sets g_c → 0
→ DBL penalty disappears
```

Then a closed gate could mean either:

```text
EHR is not useful
```

or:

```text
the model is avoiding the auxiliary loss
```

Therefore:

```text
gate modifies predictions
loss supervises predictions
```

The gate should not directly decide whether the loss is paid.

If a gate-weighted loss is ever studied, use a detached gate, but this is not recommended initially.

---

# 7. Defining the tail set

Current logs suggest the existing `tail_map` may contain only one label.

Before this experiment print:

```text
head labels
medium labels
tail labels
N_positive_train
```

Options:

## Option A — use existing locked tail tier

Best for continuity with the existing protocol.

Caveat:

```text
may supervise only one tail disease
```

## Option B — create a new Rare-3 auxiliary set

Define using TRAIN only:

```text
T = three labels with the fewest positive training studies
```

Lock it before training.

Keep the original Tail mAP as a primary metric and report Rare-3 mAP as an additional diagnostic.

Do not redefine the tail set after validation results.

---

# 8. DBL weight computation

DBL can still use the full multilabel target vector to compute its co-occurrence-aware weights.

Then mask the final loss to \(c\in T\):

```text
compute DBL weights using all 13 labels
calculate unreduced per-label DBL
retain only tail/Rare-3 columns
average over selected labels
```

This preserves DBL's co-occurrence information while preventing direct DBL supervision on head/medium outputs.

---

# 9. Loss-scale matching

DBL and ASL have different numerical scales.

Do not interpret:

```text
DBL loss 0.008
ASL loss 0.044
```

as a direct difference in fit.

Before selecting `lambda_T`, estimate on a fixed training subset at initialization:

\[
s
=
\frac{E[L_{ASL,T}]}
{E[L_{DBL,T}]}.
\]

Use:

\[
L
=
L_{MM}
+
\lambda_I L_{IMG}
+
\lambda_T s L_{DB,T}.
\]

Fix `s` before training.

This makes `lambda_T` easier to interpret and avoids DBL being numerically negligible.

Do not adapt `s` using validation results.

---

# 10. Compact experiment after plain disease-gate screen

Only if disease-aware ASL is promising or mechanistically active:

```text
H0:
DA gate + ASL

H1:
DA gate + ASL
+ scaled tail-DBL, lambda_T=.10

H2:
DA gate + ASL
+ scaled tail-DBL, lambda_T=.25
```

Keep:

```text
same initialization
same data order
same optimizer
same schedule
same warm-up
same tail set
```

Optional stronger control:

```text
H3:
no disease gate
+ same tail-selective DBL
```

This distinguishes:

```text
tail DBL effect
from
gate × DBL interaction
```

---

# 11. Factorial estimands

Let:

```text
A = architecture: global/no gate vs disease gate
B = loss: ASL vs tail-selective DBL
```

Then:

\[
I_{gate\times DBL}
=
[(Gate+DBL)-(Gate+ASL)]
-
[(NoGate+DBL)-(NoGate+ASL)].
\]

A positive tail interaction means:

> DBL is more useful when its gradient can be expressed through a disease-specific residual pathway.

This is a strong scientific result.

---

# 12. Go/no-go

A disease-gate + tail-DBL arm is promising if versus gate + ASL:

```text
Delta Tail or Rare-3 mAP >= +0.002
Delta Macro mAP >= -0.001
Delta Worst-3 fixed >= -0.001
Tail Recall@K stable or improved
```

Strong:

```text
Tail/Rare-3 ↑
Worst-3 ↑
Macro stable or ↑
REAL > SHUFFLE/NULL
```

No-go:

```text
tail gain requires macro loss > .002
or
gate collapses toward zero
or
DBL benefit remains isolated to one label while other rare labels decline
```

---

# 13. Recommended sequence

```text
1. Finish disease-aware gate + plain ASL screen.
2. Verify DA-E > DA-U and REAL > SHUFFLE/NULL.
3. Audit current tail tier and define locked auxiliary rare set.
4. Add scaled tail-only DBL on the disease-aware residual branch.
5. Compare lambda_T=.10 and .25.
6. If positive, replicate across three seeds.
```

## Final recommendation

Yes, disease-aware gating is a good place to reuse the useful part of DBL.

The cleanest version is:

> retain ASL as the global objective and use DBL only as an auxiliary supervision signal for prelocked rare-disease residuals, with the image pathway protected by an image-only auxiliary loss and stop-gradient inside the DBL branch.

This is more targeted than global DBL and more interpretable than multiplying the whole model loss by rare-class weights.
