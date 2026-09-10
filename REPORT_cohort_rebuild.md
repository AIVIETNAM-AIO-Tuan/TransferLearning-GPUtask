# REPORT — Cohort Rebuild (PLAN_end_to_end Part A)

**Status:** cohort + EHR features built and passing §E1. Model notebooks not started.
**Cohort:** 209,096 studies · **Leakage: zero** (old cohort: 0.448)
**Date:** 2026-07-15

---

## 1. Why we rebuilt

The old cohort (`fusion_dataset_hybrid_*.csv`) was abandoned after diagnostics showed
its construction was unrecoverable. The builder notebook does not exist in the repo,
so its join logic could only be inferred — and what we inferred was broken:

| Failure | Evidence | Root cause |
|---|---|---|
| Future data leaked in | `frac_studies_with_future_organism = 0.448` | micro joined on `subject_id` with **no time anchor** |
| Cartesian product | mean **94.85** micro rows/study, max **1427** | same |
| 59% of microbiology silently deleted | `CANCELLED` appeared as **30.9%** of organisms (raw truth: 1.04%) | `org_name` NULL = NO_GROWTH; inner joins / `dropna` drop NULLs, inflating everything else |
| Organism vocab pruned | `full.csv` had 20 organisms; the real vocab is 617 | one-hot-era pruning |

The `CANCELLED` inflation was the fingerprint: `CANCELLED` is a *string* and survives an
inner join; NO_GROWTH is a *NULL* and does not.

---

## 2. What was built

| Notebook | Covers | State |
|---|---|---|
| `01_cohort_selection.ipynb` | §A0–A2, labels, split, EDA | 27 cells, runs |
| `02_ehr_features.ipynb` | §A3–A8, micro tiers, labs, §E1 | 26 cells, runs, **§E1 passes** |

Everything is rebuilt from **MIMIC-IV v3.1 + MIMIC-CXR-JPG on BigQuery**
(`physionet-data.mimiciv_3_1_hosp`, `..._ed`, `physionet-data.mimic_cxr_jpg`).
The only artefact that cannot come from BigQuery is the 768-d RAD-DINO embedding
(a model output). CheXpert labels, `study_datetime`, and the official split **are**
on BigQuery — an earlier claim to the contrary was wrong.

---

## 3. What the data said that the plan did not expect

**(a) The cohort is inpatient-dominated, not outpatient-dominated.**
Solved from the observed tier×setting table:

```
inpatient 52.0% | ed 37.2% | outpatient 10.8%     (reproduces T1 and T2 exactly)
```

§A2/§A4 assumed the opposite — *"Matching neither → outpatient. Expected, and explains
why Tier 1 is rare (11.2%)"*. That explanation no longer holds. Clinically the data is
credible: MIMIC-CXR is full of portable AP films taken at the bedside. **Consequence:**
§C's requirement to stratify every metric by `setting` becomes load-bearing, not optional.

**(b) 59% (rows) and 84% (specimens) are both correct.**
NO_GROWTH is 59% of *rows* but 84% of *specimens*, because a positive specimen fans out
to ~1 organism × ~12 antibiotics (5.42 rows) while a no-growth specimen has ~1.45 rows.
The **rise** from 59%→84% after de-duplication is the evidence §A5's dedupe ran.
`rows/specimen = 2.08` matches §A5's reference of 2.07.

**(c) Labs cluster by ordering panel — evidence the itemid map is right.**

```
hemoglobin 64.77 | platelet 64.11 | wbc 64.16   -> CBC   (spread 0.66pp)
bun        58.17 | creatinine 58.31 | glucose 57.70 -> BMP (spread 0.61pp)
```

§A7's "71.9% @24h" is an **any-lab** figure and is not comparable to per-lab coverage.

**(d) `org_name = 'NEGATIVE'` is a per-test, not per-specimen, result.**
67,546 rows collapse to **26** specimens — those rows sit on specimens that also grew a
real organism, so `POSITIVE` wins. Reserved id `NEGATIVE=5` will be effectively unused.

---

## 4. Deliberate deviations from PLAN_end_to_end

Each was confirmed with Tuan, not applied silently.

| # | Plan says | We do | Why |
|---|---|---|---|
| 1 | 14 CheXpert labels | **13** — drop `Support Devices` | not a pathology |
| 2 | "confirm U-Ignore" | **U-Ones** (`-1 → 1`) | favours recall on rare labels |
| 3 | outpatient open question | **keep** | dropping selects on encounter type |
| 4 | official MIMIC-CXR split | **regenerate**, subject-grouped, **stratified on tail labels** | cohort was re-selected, so the official balance no longer holds; unstratified splits give `Pleural Other` single-digit positives in val/test |
| 5 | `storetime` date fallback @ end-of-day | **exclude the specimen** (~0.8%, 3,553) | inventing a result time creates a traceless artifact; a null `storetime` supports neither "resulted" nor "pending" |
| 6 | T1 = *same encounter*, unbounded; T3 = *prior encounter* ≤90d | **T1 = `<7d`, T3 = `[7,90)`, any encounter**; `same_enc` becomes a flag | the two tiers sat on different axes and could not be ordered; ED could never reach T1 (see §6) |
| 7 | `assert 0.40 <= frac_studies_with_no_micro <= 0.60` | **report it, assert §A1 structurally instead** | the metric mixed units *and* the band drifts with our own window parameters (see §5) |

Deviation 5 was Tuan's correction and it proved itself: the end-of-day fallback we *kept*
for `charttime` went on to fabricate `ct > st` (a specimen "collected" after its own
result), which then tripped a false leak assert. Now clamped via `LEAST(ct_fallback, st)`.

**Split reproducibility bug found and fixed:** `hash()` on strings is salted per process
(`PYTHONHASHSEED`), so the fallback split was **different on every run** — subjects could
migrate train↔test between runs. Replaced with `hashlib.md5` / seeded RNG. Verified
identical across `PYTHONHASHSEED=1` and `=99`.

---

## 5. Current state — §E1

| Check | Old | Now |
|---|---|---|
| `frac_studies_with_future_organism` | 0.4480 | **0.0000** ✔ |
| T1/T3 organism identity resulted after film | — | **0** ✔ |
| T2 specimen collected after film | — | **0** ✔ |
| T4 aggregate from the future | — | **0** ✔ |
| `org_itemid` NULL (raw rows) | — | **0.590** ✔ (≈0.59 expected) |
| rows/specimen | 94.85/study | **2.08** ✔ (2.07 expected) |
| organisms/study T1 | mean 94.85, max 1427 | **mean 0.34, max 15** ✔ |
| §A1 no-selection-on-EHR | — | **asserted structurally** ✔ (see below) |

### `frac_studies_with_no_micro` — reported, not asserted

The plan's `assert 0.40 <= frac <= 0.60` was **dropped**, for two independent reasons:

1. **The metric mixed units.** T1 counted *specimens* (`t1_status`), T3 counted *organisms*
   (`t3_org_ids`) — so a study whose only culture came back **negative** was counted as
   "never cultured". That is the §A6 error occurring *inside the assert meant to guard
   against it*. Now measured in one unit (`t3_n_specimens == 0`).
2. **The band drifts with our own parameters.** It was calibrated on encounter-based tiers.
   The recency remap correctly captures fresh no-growth results from other encounters, so
   the figure fell `0.4039 → 0.3810` **because it became more accurate** — and the assert
   fired on the improvement. An assert that must be re-tuned whenever `T1_WINDOW_DAYS`
   moves protects nothing.

**What replaced it** — the invariant §A1 actually states, checked structurally:

```python
# a step violates §A1 only if it references EHR *and dropped rows*
offend = [(s, d) for s, d in zip(attr['step'], attr['n_dropped'])
          if EHR_WORDS.search(s) and d > 0]
assert not offend
assert len(df) == len(cohort)      # 02 drops nothing either: every merge is LEFT
```

`01` never queries `microbiologyevents` or `labevents`, so selection-on-EHR is impossible
by construction; this verifies it from the attrition table instead of inferring it from a
drifting proxy. Tested to fire on injected `has micro culture before CXR` and `has a lab
within 24h` filters, and to pass clean. The `n_dropped > 0` condition is load-bearing: 01's
honest step *`5. outpatient KEPT (no EHR-based exclusion)`* names EHR precisely to say it
does **not** filter on it, and a name-only check false-positives on exactly that.

**Tier coverage — reported on the "≥1 organism" definition (C9):**

```
T2: 48.21%   (plan: 48.3%  -- near-exact, measured by an independent route)
T3: 10.97%   (plan: 12-16% -- slightly low)
T4: 29.01%
```

**Definition inconsistency — resolved.** C9 had mixed units: `T1 = (t1_status > 0)` counted
*specimens* (29.25%) while `T3 = (k_t3 > 0)` counted *organisms* (10.97%). C9 now reports
**both**, with the organism-based figure as the headline (that is the unit §E1's 11–16%
references came from — the source query carried `WHERE org_name IS NOT NULL`). The
specimen-based block is kept because T2 has no organism identity at all and is only
definable there, and because NO_GROWTH is a real finding rather than an absence.

---

## 6. T1/T3 remapped by recency, not encounter — **implemented**

```
T1: result_known AND days_before < 7          (any encounter)
T3: result_known AND 7 <= days_before < 90    (any encounter)
T4: result_known AND days_before >= 90        (unchanged)
+ t1_same_encounter as a FLAG, not a filter
+ t1_days_since_last_specimen (always defined) and t1_days_since_last_positive (+ mask)
```

**Rationale:** "same encounter" is an administrative axis; recency is a physiological one.
The current T1 (same encounter, *unbounded*) and T3 (prior encounter, ≤90d) sit on
different axes and cannot be ordered. The remap makes T1/T3/T4 a genuine ordinal ladder
and **dissolves the ED bug** rather than patching it — no `stay_id` join required.

**Measured effect:**

```
T1: 29.25% -> 36.30%  (+7.05pp)    lost 1,493 (2.44% of old T1) | gained 16,241
ED T1: 0.0000 -> 0.0695            outpatient T1: 0.0000 -> 0.0667
```

The change is a **net expansion**, not a lossy cut — the "97.2% coverage" framing measured
only the loss side. ED's ceiling of ~7% is physiology, not residual bug: ED cultures are
sent during the visit and result 24–72h later, so at film time they are *pending* (T2 =
31.2%), which is exactly why Tier 2 exists.

**7d is not a data-derived optimum.** Marginal gain per day decays smoothly with no elbow
(2.00 → 1.06 → 0.69 → 0.53 pp/day across 3→5→7→10→14). Any bound in 5–14 looks similar.
7d must be justified **clinically** (≈ one antibiotic course), positioned just past the knee
— not as an optimum found in the data.

**Both recency features are needed:** 58.80% of T1 studies have only NO_GROWTH/CANCELLED,
so `t1_days_since_last_positive` alone would be NULL for a majority. The two distributions
genuinely differ (median 0.96d any-specimen vs 1.92d positive-only).

---

## 7. Watch items

**(a) `t1_same_encounter` is a strict subset of `setting == inpatient`.** By construction:
`setting=='inpatient'` *is* defined by `cur_hadm`, so the flag can only ever be 1 for
inpatients (`ed 0.0 | outpatient 0.0 | inpatient 0.5619`). It is not merely correlated with
setting — it is nested inside it. Keep it, but it must be independently ablatable, and any
fusion gain attributed to it has to survive `setting` stratification.

**(b) Outpatients are the history-rich group — the opposite of intuition.**

```
t4 coverage:  outpatient 0.6696 | ed 0.5720 | inpatient 0.4664
t3 coverage:  outpatient 0.3098 | ed 0.2325 | inpatient 0.1132
```

Outpatient CXR is follow-up imaging for chronic patients, so their records run deep. This
matters directly for §A4's mechanism argument: **T4 is a utilization signal, and the group
richest in it is exactly the chronically-unwell cohort whose reports run long and mention
more comorbidities** — the shortcut §A4 warns about. Ablation D4 should be read with this
in mind; if T4 carries the gain, this table is why.

**(c) `frac_studies_with_no_micro` now reads ~0.38 and is below §E1's floor — by design.**
The cohort did not change; the tier definitions did, and the metric was fixed. See §5. It
is a diagnostic now, not a gate. Do not "restore" it to 0.40+ by narrowing T1 — that would
be tuning the data to fit a proxy.

**(d) ~36–42% of studies have no lab within 24h** → `lab_*_mask = 0` across the board. The
lab branch is only active on ~60% of the cohort. Legitimate (mostly outpatient) — a
limitations note, not a defect.

---

## 8. Next

1. ~~Implement the §6 remap~~ — **done**; `02` must be re-run to regenerate
   `cohort_study_level.parquet` under the new tier definitions.
2. ~~Measure organism-based T1~~ — **done**; C9 now reports both units.
3. `03_cxr_embeddings.ipynb` — §B1 checkpoint inspection, §B2 extract/cache.
4. `04_raddino_transfer.ipynb` — **MLP head (not linear) + DB loss** (Tuan's change).
5. `05_fusion_model.ipynb` (§B3–B7, E2 identity-start gate) → `06_train_eval.ipynb` → `07_ablations.ipynb`.

**Schema added by the remap** (§A8 grows): `t1_same_encounter`,
`t1_days_since_last_specimen_{value,mask}`, `t1_days_since_last_positive_{value,mask}`,
`t3_days_since_last_positive_{value,mask}`, `t3_n_specimens`. `t3_recency_bin` is now
`{7-30d, 30-90d, none}` — the plan's `0-7d` bucket is structurally empty once T1 owns
`[0,7)`. Recency features go through the **same value+mask contract as the labs** (§A7):
train-only standardization, median-imputed, mask never discarded.

---

## Appendix — process notes

Two classes of error recurred and are worth naming:

**Wrong-granularity comparison** (3×): a row-level constant asserted against a
specimen-level quantity (59% vs 84%); a row-level clamp counted at specimen level (always
0); an any-lab reference compared to per-lab coverage. Every metric must state its unit.

**Stale notebook state** (3×): `MICRO_CTE`/`SX` are Python strings defined in one cell and
interpolated in another. Re-running a consumer after an edit reuses the old string, and
BigQuery then reports an error pointing at a line the file on disk does not contain.
Mitigated by co-locating the fragments in one cell, a `need()` guard that names the stale
fragment, and `run_q()` which dry-runs every query (full parse + name resolution, **0 bytes
billed**) before executing.
