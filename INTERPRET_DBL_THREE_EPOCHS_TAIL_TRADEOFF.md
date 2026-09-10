# INTERPRETATION — DBL Three-Epoch Results

## 1. Main conclusion

DBL does show a real tail-learning signal.

However, the signal is highly selective:

```text
DBL improves the current tail label
while reducing performance on almost every other label.
```

Thus DBL is not a globally better loss under the current configuration. It is a strong tail-gradient reallocater.

---

## 2. Macro-constrained checkpoint selection after three epochs

Applying the locked checkpoint rule to epochs 1–3:

| Arm | Selected epoch | Macro mAP | Tail mAP | Worst-3 fixed | Tail Recall@K |
|---|---:|---:|---:|---:|---:|
| A0 ASL | 2 | 0.4404 | 0.1215 | 0.1372 | 0.1963 |
| D0 legacy DBL | 3 | 0.4328 | 0.1256 | 0.1326 | 0.1857 |
| D1 | 2 | 0.4383 | 0.1225 | 0.1362 | 0.1804 |
| D2 | 2 | 0.4382 | 0.1228 | 0.1362 | 0.1804 |
| D3 | 2 | 0.4376 | 0.1230 | 0.1360 | 0.1777 |

Deltas versus selected A0:

| Arm | Δ Macro | Δ Tail | Δ Worst-3 | Δ Tail Recall@K |
|---|---:|---:|---:|---:|
| D0 | -0.0076 | **+0.0041** | -0.0046 | -0.0106 |
| D1 | -0.0021 | +0.0010 | -0.0011 | -0.0159 |
| D2 | -0.0022 | +0.0013 | -0.0010 | -0.0159 |
| D3 | -0.0028 | +0.0015 | -0.0012 | -0.0186 |

## 3. The strongest signal is D0

D0 shows:

```text
Tail mAP:
A0 selected  0.1215
D0 selected  0.1256
Delta       +0.0041
```

This is larger than the preregistered strong tail threshold.

But it comes with:

```text
Macro        -0.0076
Worst-3      -0.0046
Recall@K     -0.0106
Head         -0.0099
Medium       -0.0067
```

Therefore D0 is not acceptable as a full-model replacement.

It demonstrates that the tail label can be improved if sufficient optimization mass is redirected toward it.

---

## 4. Per-label evidence

Comparing selected D0 with selected A0:

```text
12 of 13 labels decline.
Only the current tail label improves.
```

The tail-label AP improves by approximately:

```text
+0.00414
```

while many other labels lose between approximately:

```text
-0.005 and -0.014 AP.
```

This is a near-pure redistribution of performance rather than a general representation improvement.

---

## 5. Important tier-definition caveat

In the current log:

```text
tail_map == one element of per_ap
```

for every arm and epoch.

This indicates that the locked tail tier currently contains only one label, or that Tail mAP is effectively being driven by one label.

Therefore the scientifically precise statement is:

> DBL improves the current tail label.

Not yet:

> DBL broadly improves rare diseases.

Worst-3 fixed does not improve, which confirms that the benefit does not generalize across all weak labels.

---

## 6. D1–D3 form a trade-off frontier

Across tuned mapping arms:

```text
D1 → D2 → D3
tail gain increases
macro cost also increases
```

Selected deltas:

```text
D1: Tail +0.0010 | Macro -0.0021
D2: Tail +0.0013 | Macro -0.0022
D3: Tail +0.0015 | Macro -0.0028
```

Thus increasing the DB rebalancing emphasis moves the model along a Pareto frontier:

```text
more tail AP
↔
less overall performance
```

None of D1–D3 currently passes the original promising gate:

```text
Delta Tail >= +0.002
Delta Macro >= -0.002
```

D1 is closest to the macro boundary; D3 is closest to the tail boundary.

---

## 7. Why Tail Recall@K falls despite Tail AP increasing

For D0:

```text
Tail AP       improves
Tail Recall@K declines
```

This is possible because AP and Recall@K inspect different parts of the score ranking.

Interpretation:

- DBL improves the shape/order of some relevant positives across the PR curve;
- but fewer positives appear in the exact top-K region used by Recall@K;
- the tail improvement is therefore not uniformly distributed across the ranked list.

This is not a calibration issue because Recall@K is rank-based.

It is a local-versus-global ranking trade-off.

---

## 8. Gradient-scale observation

At epoch 3:

```text
A0 decoder gradient ≈ 0.075
D0 decoder gradient ≈ 0.020
D1 decoder gradient ≈ 0.045
D2 decoder gradient ≈ 0.042
D3 decoder gradient ≈ 0.039
```

D0 operates with a much smaller effective decoder gradient.

This may contribute to:

- slower head/medium recovery;
- less aggressive fitting of common labels;
- continued tail improvement from epoch 2 to epoch 3.

Do not rescale D0 during the current run. Complete the fixed protocol first.

---

## 9. What to do with the current experiment

Continue all arms to epoch 8.

Questions to inspect:

1. Does D0 tail AP continue increasing?
2. Does D0 macro recover as the learning rate declines?
3. Do D1–D3 overfit the tail after epoch 2, like WASL?
4. Does any DBL arm improve fixed Worst-3 later?
5. Is D0's tail gain stable or transient?

Do not abandon DBL based only on macro after three epochs.

But also do not promote D0 as the final loss based only on its single-label tail gain.

---

## 10. Most useful next method if the pattern persists

If D0 continues to improve the tail label while harming all other labels, the next experiment should not be a larger global DBL grid.

Use a selective hybrid:

\[
L
=
L_{ASL}^{all}
+
\lambda_{tail}L_{DBL}^{tail}
\]

where:

```text
ASL trains all 13 labels.
DBL auxiliary term is applied only to the locked tail label(s).
```

Better isolation with ML-Decoder:

```text
ASL:
update full decoder/adapters/EHR encoder

tail-DBL auxiliary:
update only tail disease query / tail classifier parameters
or detach shared representation for the auxiliary branch
```

This directly tests whether the useful D0 tail signal can be retained without damaging head/medium labels.

Suggested compact screen after current run:

```text
H0 = ASL
H1 = ASL + 0.10 * tail-only DBL
H2 = ASL + 0.25 * tail-only DBL
H3 = ASL + 0.50 * tail-only DBL
```

Do not run this until the full eight-epoch DBL trajectory is reviewed.

---

## 11. Current scientific wording

Recommended:

> Distribution-Balanced Loss produced a clear but highly selective tail-class benefit. The legacy DBL configuration improved the current tail-label AP by approximately 0.004, but this gain was accompanied by broad degradation across the remaining labels, lower fixed Worst-3 AP, and lower Tail Recall@K. Tuned DBL mappings reduced the global cost but also attenuated the tail benefit, revealing a tail–overall performance trade-off.

Avoid:

> DBL improves the long tail.

Because the current evidence is driven by one tail label and fixed Worst-3 performance remains lower.
