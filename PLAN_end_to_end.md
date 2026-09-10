# PLAN — End-to-End: Cohort → EHR Features → FiLM Fusion → Training

Spec for Claude Code. Multi-label thoracic disease classification (14 CheXpert labels) on MIMIC-CXR + MIMIC-IV, fusing frozen RAD-DINO CXR embeddings with EHR features via FiLM conditioning, trained with Distribution-Balanced Loss.

**Research question:** do EHR features complement CXR embeddings, particularly for rare diseases where the imaging signal is weak?

**Two constraints that govern every decision below:**
1. **Causality** — every EHR feature attached to a study must have been available to a clinician *at the moment the film was read*. No future data, no positive-upper-bound windows.
2. **No selection on EHR** — cohort inclusion may never depend on EHR availability or on label values. The ~50% of studies with no prior micro are *data*, not missingness to filter away.

Supersedes `PLAN_cohort_selection.md` and `PLAN_cxr_ehr_film_fusion.md`.

---

# PART A — COHORT

## A0. Unit of analysis

**One row = one CXR `study_id`** (~227,835 studies from ~377k images, ≈1.66 images/study).

Rationale: images in a study share one acquisition time and one radiology report; CheXpert labels are study-level, so per-image propagation is a known noise source.

- `index_time` := study acquisition timestamp (`mimic_cxr_jpg.metadata`: `StudyDate` + `StudyTime`). Assert one distinct time per `study_id`; if multiple, take min and log.
- Keep `study_id → [dicom_id]` side table (needed for embedding pooling and the view ablation).

## A1. Inclusion / exclusion

Include:
1. Valid `index_time`
2. Has CheXpert labels (`mimic-cxr-2.0.0-chexpert.csv.gz`)
3. ≥1 image with usable `ViewPosition`
4. `subject_id` resolvable in `mimiciv_3_1_hosp.patients`

Exclude: **nothing** based on micro/lab presence or label values.

> Selecting on "has micro before CXR" would condition on the feature itself (collider bias) and inflate the EHR-complementarity Δ. Guarded by an assert in §D2.

Log a CONSORT-style attrition table (n at each step).

## A2. Encounter assignment

`mimic_cxr_jpg.metadata` has **no `hadm_id`** — assign by time:

```
cur_hadm   := hadm_id where admittime <= index_time <= dischtime   (hosp.admissions)
cur_edstay := stay_id  where intime   <= index_time <= outtime     (mimiciv_ed.edstays)
setting    := 'inpatient' if cur_hadm else 'ed' if cur_edstay else 'outpatient'
```

- Matching neither → `outpatient`. Expected, and explains why Tier 1 is rare (11.2%).
- Assert no study matches >1 `hadm_id`; on overlap take latest `admittime` and log.
- **ED is retained.** `microbiologyevents` lives in `hosp` and carries `stay_id`, so ED-ordered cultures are recorded there — ED is not micro-poor. What ED lacks is *labs* (MIMIC-IV-ED has no lab table; ED labs appear in `hosp.labevents` only when the encounter has an `hadm_id`).
- **`setting` is both a feature and a stratification variable.** Known confound: inpatient vs ED have different labeling regimes and different missingness patterns, potentially recoverable from acquisition posture (portable AP vs standing PA). Report all metrics stratified by `setting`.

## A3. Time windows — the core fix

Two clocks in `microbiologyevents`, meaning different things:
- `charttime` = specimen collected → signals **clinical suspicion**
- `storetime` = result available → signals **the result itself**

Nulls: `charttime` ~7.6%, `storetime` ~0.8%. Fallback to `chartdate`/`storedate` **at end-of-day + set `time_imprecise=1`**; treat imprecise rows conservatively at the boundary (an imprecise same-day row is NOT available). Drop rows with no resolvable time.

| Feature block | Window | Clock |
|---|---|---|
| Labs | `[index_time − 24h, index_time]` | `storetime ≤ index_time` |
| Micro Tier 1 (current encounter, resulted) | same `cur_hadm`/`cur_edstay` | `storetime ≤ index_time` |
| Micro Tier 2 (pending) | — | `charttime ≤ index_time < storetime` |
| Micro Tier 3 (recent history) | `[index_time − 90d, index_time)`, prior encounter | `storetime ≤ index_time` |
| Micro Tier 4 (long-term burden) | `> 90d` before | `storetime ≤ index_time` |

**Never** `direction='forward'`; never a positive upper bound.

Lab window rationale: coverage saturates after 24h (59.3% @6h → 67.3% @12h → **71.9% @24h** → 73.4% @48h → 75.8% @7d). Median lab turnaround 0.72–1.25h, so "collected" ≈ "resulted"; still filter on `storetime`.

## A4. Micro feature tiers

All causal. Semantically distinct — do not merge.

### Tier 1 — current encounter, result known (≈11.2%)
Strongest signal, rare (outpatient/ED studies have no `cur_hadm`, so their cultures land in Tier 2/3 instead — Tier-1 rarity is an artifact of encounter assignment, not missing data).
→ `t1_org_ids[]`, `t1_status`, `t1_spec_ids[]`

### Tier 2 — pending at CXR (≈48.3%)
`charttime ≤ index_time < storetime`. Clinician had sent a specimen but had no result. This is the MNAR *suspicion* signal — the largest single block and fully legitimate.
→ `t2_pending` (flag), `t2_spec_ids[]` (blood/sputum/urine — carries suspected source), `t2_n_pending`

**Do NOT leak the result of a pending specimen into any feature.** Only the fact + specimen type.

Evidence for splitting the clocks: `frac_specimen_sent_before = 78.8%` vs `frac_organism_known_before = 43.2%` — a 35-point gap that is entirely legitimate signal.

### Tier 3 — recent history, ≤90d (≈12.4% additional coverage)
Physiological signal: which organism, how recent.
→ `t3_org_ids[]`, `t3_days_since_last_positive`, `t3_recency_bin` ∈ {`0-7d`,`7-30d`,`30-90d`}

### Tier 4 — long-term burden, >90d (aggregate only)
**No `org_ids`.** Aggregate quantities only, explicitly framed as utilization:
```
t4_n_prior_positive_ever       # count, not identity
t4_years_since_first_culture   # record depth
t4_n_prior_encounters          # contact frequency
```

**Why Tiers 3 and 4 are split.** Prior-infection history is genuinely prognostic, but the >90d portion routes through the labels by a different mechanism than an imaging feature does: CheXpert labels are NLP-extracted from reports, and patients with thick histories get longer reports mentioning more comorbidities → more positive labels *independent of what today's film shows*. So `has_prior_positive_culture_3_years_ago` correlates with labels partly because of how that patient gets written up, not because of imaging findings. A clinician uses history to *interpret* the film; a model can use it to *predict the label without looking at the film*. Same feature, two mechanisms — and for rare labels (<2% prevalence, weak imaging signal) the shortcut mechanism wins more easily.

The numbers: `>90d` accounts for **45.4%** of studies with a prior positive (n=44,809, larger than all other bins combined), and `frac_tier3_ONLY_gain` is 32.1% (unlimited) vs 12.4% (90d) — so ~2/3 of the unlimited gain comes from exactly the region where the shortcut is strongest.

Tier 4 keeps the "chronically unwell" signal Tuan wants, but states it *as* utilization rather than dressing it as microbiological evidence — and makes it independently ablatable (§D4). If Tier 4 carries most of the gain, that is itself a reportable finding about the mechanism, not something to hide.

## A5. Micro aggregation

The old join produced mean 94.85 rows/study (max 1427) — merging on `subject_id` with no time anchor is a Cartesian product. Aggregate in two levels:

```
microevent_id            # 1 row = 1 organism × 1 antibiotic
  └─ micro_specimen_id   # 1 physical specimen -> culture status lives HERE
       └─ study          # via subject_id + time window
```

1. **Dedupe to `(micro_specimen_id, org_itemid)`** first. A specimen with 1 organism × 12 antibiotics = 12 rows of the same organism; `distinct_orgs = 718` is inflated at row level. (Raw: 3,988,224 rows / 1,924,289 specimens ≈ 2.07 rows per specimen.)
2. Derive **per-specimen status** (§A6).
3. Aggregate specimens → study as multi-valued sets.

Use **`org_itemid` / `spec_itemid`** (integers) as vocab keys, not strings — avoids case/whitespace variants.

Post-fix assertion: median organisms/study ~0–2, small max. Anywhere near mean 49 ⇒ window not applied.

## A6. Culture status vocabulary (per specimen)

**`org_name` NULL = NO GROWTH.** 2,352,859 / 3,988,224 = **59%** of raw rows. Verified: `org_is_NULL`, `org_itemid_NULL`, and `isolate_num_NULL` are all exactly 2,352,859 — matching the documented behavior ("if no growth is found, the remaining columns will be NULL").

> **Do not search for the string.** MIMIC never writes `"NO GROWTH"` into `org_name` (`text_NO_GROWTH = 1`, `text_NONE = 0`, `text_STERILE = 0`). Counting text finds nothing and leads to "MIMIC has no negative cultures" — which is exactly how 59% of the data disappears unnoticed. **Infer status from NULL, at specimen level.**

A negative culture is a *real diagnostic result* (infection ruled out), not missing data. But because it is encoded as NULL, every NULL-dropping operation deletes it silently — no exception, no warning:
```python
df.merge(micro, on='org_itemid', how='inner')   # NULLs don't match -> gone
micro.dropna(subset=['org_name'])               # gone
micro.groupby('org_name').size()                # pandas dropna=True by default -> gone
```
Use **LEFT joins**, `groupby(..., dropna=False)`, and derive status at `micro_specimen_id` level *before* rolling up to study (once at study level, "no specimen at all" and "specimen, negative" are indistinguishable).

`org_name` also carries non-organism tokens. **Three tiers, not two:**

| Tier | Examples | Handling |
|---|---|---|
| Administrative status | `CANCELLED` (41,504 = 1.04%), `NEGATIVE` (67,546 = 1.7%), `2ND_ISOLATE` | reserved ids |
| Preliminary / non-speciated | `GRAM POSITIVE BACTERIA`, `GRAM NEGATIVE ROD(S)`, `MIXED BACTERIAL FLORA`, `POSITIVE`, `AFB GROWN IN CULTURE...` | own ids — real signal at coarser resolution (AFB ⇒ TB suspicion; MIXED ⇒ likely contaminated; `POSITIVE FOR MRSA` overlaps `STAPH AUREUS COAG +`) |
| True species | `ESCHERICHIA COLI`, `KLEBSIELLA PNEUMONIAE`, … | vocab from **train split only** |

Reserved ids in the organism vocab:
```
PAD=0, UNK=1, MISSING=2, NO_GROWTH=3, CANCELLED=4, NEGATIVE=5,
<preliminary tokens 6..9>, <species from id 10+>
```

Per-specimen status ∈ {`POSITIVE`, `NO_GROWTH`, `CANCELLED`, `PENDING`}; per-study `MISSING` when no specimen exists in-window.

Note on `CANCELLED`: 30.9% in the old join vs **1.04%** raw. The old number was the *fingerprint* of NO_GROWTH vanishing — `CANCELLED` is a string and survives an inner join; NULL doesn't, so everything else got proportionally inflated. It means "specimen sent, culture never resulted" — distinct from both `MISSING` (never suspected) and `NO_GROWTH` (tested, negative).

## A7. Lab features

Six canonical labs: BUN, creatinine, glucose, hemoglobin, platelet, WBC (WBC is the infection-relevant one).

Per lab, in `[index_time − 24h, index_time]`, `storetime ≤ index_time`:
- `lab_{name}_value` — most recent in window; standardize with **train-split** mean/std; winsorize outliers
- `lab_{name}_mask` — 1 if measured, 0 otherwise
- optional: `lab_{name}_delta_h` — hours since measurement

Impute missing (median) **and always keep the mask** — whether a test was *ordered* is itself a clinical variable (MNAR). Never impute-and-discard-the-mask.

## A8. Cohort output schema

One row per `study_id`:
```
study_id, subject_id, index_time, setting, cur_hadm, cur_edstay
labels_0..13                      # CheXpert, chosen U-policy
lab_*_value, lab_*_mask           # 6 labs × 2
t1_org_ids[], t1_spec_ids[], t1_status
t2_pending, t2_spec_ids[], t2_n_pending
t3_org_ids[], t3_days_since_last_positive, t3_recency_bin
t4_n_prior_positive_ever, t4_years_since_first_culture, t4_n_prior_encounters
time_imprecise
split                             # train/val/test
```

Multi-valued `*_ids[]` feed `nn.EmbeddingBag`. **Split grouped by `subject_id`**, zero overlap — required regardless of temporal direction because of cumulative per-patient labels. All vocabs and standardization stats from **train only**.

---

# PART B — MODEL

## B0. Locked architecture decisions

- **`LayerNorm`, not BatchNorm**, throughout the EHR encoder, FiLM generator, fusion, and head. FiLM conditioning is *per-patient*; BatchNorm would make one patient's `γ,β` depend on the other patients in the batch — conceptually wrong. Also: class imbalance makes batch composition swing, and the inpatient/ED confound drifts batch statistics. Standardize continuous labs at preprocessing, not with BN.
- **`GeLU`** (default), matching the ViT lineage. `SiLU` interchangeable; a flag, not a tuning priority.
- **RAD-DINO stays frozen.** Embeddings extracted once and cached.
- Both norm and activation are config flags so they can be A/B-tested.

## B1. Verify the checkpoint first

Inspect `raddino_lpft_mv73_dbml_best.pt`: print `list(torch.load(path, map_location='cpu').keys())` (and one level down for `{'state_dict': ...}` / `{'model': ...}`).
- Expected: **LP-FT fine-tuned RAD-DINO ViT-B/14 encoder + a classification head**. Load encoder, **discard head**, keep 768-d CLS.
- If it turns out to be a precomputed embedding tensor/dict, skip extraction.
- Branch on what `keys()` actually shows — do not assume key names.

## B2. CXR embedding extraction (`extract_cxr.py`)

- Instantiate RAD-DINO ViT-B/14 (patch 14, 518×518), load fine-tuned encoder weights, strip classifier head.
- `model.eval()`, `torch.no_grad()`, autocast. Return **CLS token** (`[768]`) per image.
- Cache to `.npy` (float32, `[N,768]`) + parallel `dicom_id` index parquet. `.npy` over 768 parquet columns at this width.
- **Reuse:** if the new cohort's `dicom_id` set is a *subset* of the old, do NOT re-extract — index the cached `.npy` by `dicom_id`.
- **Study-level pooling:** flag; default mean-pool over frontal images. (Attention pooling interacts with the view ablation — separate decision.)

## B3. Data contract (per study `i`)

| Field | Shape | Notes |
|---|---|---|
| `x_cxr` | `[768]` float | pooled frozen RAD-DINO CLS |
| `t1_org_ids`, `t2_spec_ids`, `t3_org_ids`, … | ragged `[k_i]` long | multi-valued → `EmbeddingBag` (flat index + offsets) |
| `labs` | `[6]` float | standardized, imputed |
| `labs_mask` | `[6]` float | 1 = measured |
| `t4_*` | `[3]` float | aggregate burden |
| `setting` | `[1]` long | inpatient/ed/outpatient |
| `y` | `[14]` float | CheXpert, chosen U-policy |

Special ids in every categorical vocab: `PAD=0`, `UNK=1`, `MISSING=2` (+ §A6 status ids). Vocabs from **train split only**.

## B4. Config + ablation switches

```python
@dataclass
class Cfg:
    # data
    labels: int = 14
    emb_dim_org: int = 16          # organism cardinality high (hundreds)
    emb_dim_spec: int = 8          # specimen cardinality low (dozens)
    ehr_hidden: int = 128          # dim of e
    cxr_dim: int = 768
    # architecture
    norm: str = "layernorm"        # {"layernorm","batchnorm"}
    activation: str = "gelu"       # {"gelu","silu"}
    dropout_ehr: float = 0.3
    # ablation axes
    ehr_cat_encoding: str = "embedding"   # {"embedding","onehot_mlp"}
    fusion: str = "film"                  # {"none","concat","film"}
    film_gamma_mode: str = "full"         # {"full","gating_only"}
    film_identity_init: bool = True       # start == CXR-only
    head_input: str = "h_only"            # {"h_only","h_concat_e"}
    micro_tiers: tuple = (1, 2, 3)        # {(), (1,2), (1,2,3), (1,2,3,4)}
    t3_window_days: int = 90              # {30, 90}
    # loss / opt
    loss: str = "db"                # {"db","asl","bce_posweight"}
    lr: float = 3e-4
    weight_decay: float = 1e-2      # stronger group value on the EHR branch
    epochs: int = 50
    batch_size: int = 256

@dataclass
class Paths:
    data_dir: str      # cohort tables (EHR features + labels + split)
    ehr_table: str     # data_dir/cohort_study_level.parquet
    split_file: str    # subject-level split
    ckpt: str          # .../raddino_lpft_mv73_dbml_best.pt
    cxr_cache: str     # .../cxr_emb.npy
    cxr_index: str     # .../cxr_emb_index.parquet   (dicom_id -> row)
    class_freq: str    # .../class_freq.pkl
```

> The 768-d image features live in `cxr_cache` (.npy), **not** in the CSV/parquet. At train time each cohort row joins to its embedding via `dicom_id`/`study_id`.

## B5. EHR encoder

Categoricals → **`nn.EmbeddingBag`** (`mode="sum"`). A study can carry several specimens/organisms, so this is the dense equivalent of `multi-hot × Linear`; single-valued fields degenerate to a plain lookup; empty → the `MISSING` id.

> `one-hot → Linear` and `EmbeddingBag` have **identical capacity** (one-hot × weight matrix = selecting a row = a lookup). Expect parity, not a jump. The real differences: rare categories (one-hot gives each an independent, barely-trained vector; embeddings share statistics), capacity control (per-field dims, decoupled from hidden size), and clean `MISSING`/`UNK` handling (one-hot's all-zeros conflates "not measured" with "no category"). Speed gains are real but small in absolute terms — the EHR branch is tiny next to RAD-DINO extraction. `ehr_cat_encoding="onehot_mlp"` exists to A/B this, not as a lever.
>
> Never ordinal-encode organism/specimen — they are nominal; integer codes impose a false ordering.

```
in  = concat[ bag_t1_org(16), bag_t1_spec(8), bag_t2_spec(8), bag_t3_org(16),
              t2_pending(1), t2_n_pending(1), t3_days_since(1), t3_recency_bin(emb 4),
              t4_*(3),                          # only if 4 in cfg.micro_tiers
              labs(6), labs_mask(6), setting(emb 4) ]
MLP : Linear(in, 256) -> Norm(256) -> Act -> Dropout
      Linear(256, 128) -> Norm(128) -> Act
out : e  [128]
```

Keep `d_e` small (128): the generator emits 2×768, so a wide `e` overfits and memorizes patient noise — the likely culprit behind the fusion regression.

## B6. FiLM generator + fusion

`γ` and `β` are **not** free parameters — they are *activations* produced per-patient by the generator: `γ = f_γ(e; θ_γ)`, `β = f_β(e; θ_β)`. What is trainable is `θ` (generator + EHR encoder + head). This is the opposite of LayerNorm's `γ,β`, which are `nn.Parameter`, identical for every sample, and carry no external information. If `γ,β` didn't depend on EHR, FiLM would degenerate into a plain affine and feed *no* EHR signal at all — conditioning on EHR is what makes it FiLM. The whole path is differentiable: gradients flow back through `γ ⊙ x + β` into the generator, teaching it to read EHR and emit useful modulation.

**Why FiLM over concat:** `[x_cxr | e] → Linear` can only form *additive* combinations (`w₁·img + w₂·ehr`). FiLM lets EHR change the *per-dimension gain* on the image features — a *multiplicative* interaction. Clinically: for a patient with a positive culture, the generator can amplify the embedding dimensions tied to consolidation and damp the noise. That is the concrete mechanism for "EHR compensates where the imaging signal is weak," rather than hoping an MLP discovers it.

**Identity init is the anti-fusion-gap guarantee:** zero-init the generator's last layer so at step 0 `γ=1, β=0` → fusion starts *exactly at CXR-only*, then diverges only where EHR earns it. Worst case it falls back to CXR-only instead of dragging it down; plain concat has no such property and lets noisy EHR dilute a strong image signal.

```python
class FiLMGenerator(nn.Module):
    def __init__(self, d_e=128, d_cxr=768, gamma_mode="full", identity=True):
        super().__init__()
        self.gamma_mode = gamma_mode
        self.to_gamma = nn.Linear(d_e, d_cxr)
        self.to_beta  = nn.Linear(d_e, d_cxr)
        if identity:                       # zero last layer => gamma=1, beta=0 at t=0
            for m in (self.to_gamma, self.to_beta):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, e):
        g = self.to_gamma(e)
        if self.gamma_mode == "gating_only":
            gamma = torch.sigmoid(g)       # in (0,1); sigmoid(0)=0.5 -> identity init N/A here
            beta  = torch.zeros_like(g)
        else:
            gamma = 1.0 + torch.tanh(g)    # centered at 1, bounded (0,2)
            beta  = self.to_beta(e)
        return gamma, beta

class FiLMFusion(nn.Module):
    def __init__(self, d_cxr=768, Norm=nn.LayerNorm):
        super().__init__(); self.norm = Norm(d_cxr)
    def forward(self, x_cxr, gamma, beta):
        return self.norm(gamma * x_cxr + beta)     # h
```

Document that `gating_only` cannot be identity-init at 1 via `sigmoid(0)=0.5`; use `full` for the identity-preservation ablation.

Optionally regularize `‖γ−1‖`, `‖β‖` toward identity.

## B7. Fusion model + head

- `fusion="none"`: head on `x_cxr` (CXR-only baseline)
- `fusion="concat"`: head on `concat[x_cxr, e]` (naive baseline — the currently-underperforming one)
- `fusion="film"`: `e → (γ,β) → h`, head on `h` (or `concat[h, e]` if `head_input="h_concat_e"`)

Head:
```
in  = h [768]  (or concat[h, e] -> 896)
Linear(in, 256) -> Norm -> Act -> Dropout -> Linear(256, 14)   # logits
```
Sigmoid only at inference/thresholding.

## B8. Distribution-Balanced Loss

DB Loss (Wu et al., ECCV 2020) is purpose-built for **multi-label long-tailed** recognition. Two modifications to BCE: **rebalanced weighting** that accounts for label co-occurrence (standard resampling double-counts images containing multiple labels; this corrects expected vs. observed sampling frequency), and **negative-tolerant regularization (NTR)** that mitigates over-suppression of the many negative labels for tail classes via a higher, class-aware bias/threshold so rare-positive gradients aren't drowned out.

**Do not re-derive the constants — port the official implementation:**
- Reference: `wutong16/DistributionBalancedLoss`, the `ResampleLoss` class with `reweight_func='rebalance'`. Copy the math verbatim; adapt only I/O.
- Required stat: **per-class positive counts `n_k` on the TRAIN split** (and `N − n_k`). Save as `class_freq.pkl`, pass in.
- Starting hyperparameters (verify against source, then tune):
  - `map_param = dict(alpha=0.1, beta=10.0, gamma=0.9)` — rebalance-weight smoothing
  - `focal = dict(focal=True, alpha=0.5, gamma=2.0)`
  - `logit_reg = dict(neg_scale=2.0, init_bias=0.05)` — NTR
- Also implement `asl` and `bce_posweight` behind `cfg.loss` for A/B. Leave a TODO stub for **RAL** (Robust Asymmetric Loss) — it beat ASL on CXR-LT and is robust to label noise.
- Plain Focal Loss is a poor primary choice given NLP-extracted CXR labels: a mislabeled example always looks "hard," so focal persistently amplifies its gradient.

---

# PART C — TRAINING

- Read cached `x_cxr` + cohort table; subject-level split.
- `AdamW`, cosine schedule with warmup. **Parameter groups:** stronger `weight_decay` + dropout on EHR encoder + FiLM generator; light/zero decay on norms and biases.
- Compute `class_freq.pkl` from train targets before the loop.
- Log train/val loss + val metrics per epoch; checkpoint on **best val macro AUPRC (mAP)**.

## Metrics

- **Primary: macro AUPRC / mAP.** AUROC is inflated at <2% prevalence. Macro AUROC auxiliary.
- Per-label AUROC + AUPRC, with **head / medium / tail** breakdown by train frequency.
- **Per-label thresholds** tuned on val (target per-label recall or F-β) — never a global 0.5. Report recall/precision at the operating point.
- **Never compare architectures by F1** — the XGBoost-vs-LogReg gap was diagnosed as a threshold + class-weight artifact, not model quality. Use AUROC/AUPRC.
- **Confound-corrected Δ** for EHR complementarity — not naive `fusion − CXR` (algebraically confounded with the CXR performance level).
- **Stratify by `setting`** (inpatient/ED/outpatient) — checks the fusion isn't learning a setting shortcut.

---

# PART D — ABLATIONS

## D1. Fusion architecture
`A0` CXR-only · `A1` concat · `A2` FiLM · `A3` FiLM identity-init on/off · `A4` gating-only vs full FiLM · `A5` FiLM-only vs FiLM + concat `e` into head

## D2. EHR encoding
`one-hot → MLP` vs `EmbeddingBag`, same masks. Compare AUROC/AUPRC **on rare labels** + wall-clock. Expect small differences (identical capacity) — this is a check, not a lever.

## D3. Micro tiers
`()` (labs only) · `(1,2)` · `(1,2,3)` · `(1,2,3,4)`

## D4. Tier 3/4 — the mechanism test
`A1` T3 ≤90d (org ids) · `A2` A1 + T4 (aggregate burden) · `A3` T3 unlimited (org ids)

**Default `(1,2,3)` with T3 ≤90d; `+T4` optional; `A3` is a diagnostic, not a candidate config.**

Read it like this: if `A3` beats `A1` mainly on rare labels and the gain evaporates when stratified by `setting`/number-of-prior-studies → shortcut. If `A2` ≈ `A3` → the signal is utilization, not microbiology, and `A2` is the honest way to say it.

## D5. Loss
`bce_posweight` · `asl` · `db` (default) · (`ral` — TODO)

Compare on per-label AUPRC + confound-corrected Δ, always subject-level split, thresholds per label.

---

# PART E — ACCEPTANCE TESTS

## E1. Cohort — not done until these pass

Leakage (must be exactly zero):
```python
assert (micro_joined['storetime'] > df['index_time']).sum() == 0
assert (micro_joined['charttime'] > df['index_time']).sum() == 0   # Tier 2 satisfies this by construction
assert frac_studies_with_future_organism == 0.0
```
Old-cohort reference values, all of which must become 0: `frac_collected_after_cxr=0.514`, `frac_resulted_after_cxr=0.672`, `frac_POSITIVE_resulted_after=0.689`, `frac_studies_with_future_organism=0.448`.

No selection-on-EHR:
```python
# ~50% of studies having no prior micro is CORRECT. If this drops, the cohort filtered on EHR presence.
assert 0.40 <= frac_studies_with_no_micro <= 0.60
```

NO_GROWTH preserved:
```python
assert abs(micro_raw['org_itemid'].isna().mean() - 0.59) < 0.02   # at raw row level
```

Sanity:
- `df.groupby('study_id').size().max() == 1`
- median organisms/study ≈ 0–2; mean nowhere near 49
- train/val/test `subject_id` sets disjoint
- tier coverage ≈ T1 11%, T2 48%, T3(90d) 12–16%
- report attrition table + tier coverage + `setting` distribution

## E2. Model

1. **Identity start** — with `fusion="film"`, `film_gamma_mode="full"`, `film_identity_init=True`, step-0 FiLM logits equal CXR-only logits (`allclose`). This is the anti-fusion-gap gate.
2. **Shapes** — fake batch through every `fusion` mode → `[B,14]`.
3. **EmbeddingBag multi-value** — a study with 3 organism ids and one with 0 (→ MISSING) both yield `[dim]`.
4. **DB loss** — finite; per-class rebalance weights larger for rarer classes.
5. **Overfit-one-batch** — loss → ~0 on a single batch.

---

# PART F — BUILD ORDER

**Cohort**
1. Study table + `index_time` + attrition log
2. Encounter assignment + `setting`; verify Tier-1 rarity is explained by outpatient/ED
3. Micro: dedupe → per-specimen status → tiers → aggregate. **Run E1 leakage asserts here (fail fast).**
4. Labs in 24h window + masks
5. Vocabs + standardization on train split; grouped subject-level split
6. Emit A8 schema
7. Re-run the old diagnostic queries; confirm all leakage fractions are 0

**Model**
8. B1 checkpoint inspection → B2 extraction/reuse + cache
9. B5–B7 modules; E2.1 (identity start) is the key gate
10. B8 DB loss port + `class_freq.pkl`; E2.4

**Train**
11. Baselines first: `fusion="none"`, then `"concat"`, then `"film"`
12. Ablation sweep (D1–D5), compared on per-label AUPRC + confound-corrected Δ, stratified by `setting`

---

# Repo layout

```
cohort/
  build_cohort.py     # A0-A2: studies, index_time, encounter, setting
  micro_features.py   # A3-A6: tiers, per-specimen status, aggregation
  lab_features.py     # A7
  emit.py             # A8 schema + vocabs + split
fusion/
  config.py           # Cfg + Paths + ablation flags
  data.py             # dataset, vocab build, standardization, EmbeddingBag collate
  extract_cxr.py      # B1-B2: checkpoint -> cached 768-d embeddings
  models/
    ehr_encoder.py    # B5
    film.py           # B6
    head.py           # B7
    fusion_model.py   # B7
  losses/db_loss.py   # B8
  train.py            # Part C
  eval.py             # metrics, per-label thresholds, confound-corrected Δ
  tests/
    test_cohort.py    # E1
    test_model.py     # E2
```

---

# Open questions for Tuan

1. **Outpatient studies** — keep (they're a large share; dropping them selects on encounter type) or restrict to inpatient+ED? Affects generalizability claims. *(ED: resolved — keep.)*
2. **`t2_spec_ids` granularity** — full `spec_itemid` vocab, or collapse to coarse source (blood/respiratory/urine/other)?
3. **Study-level image pooling** — mean-pool frontal only, or attention over all views? Interacts with the view ablation.
4. **U-policy** — confirm U-Ignore (previously preferred over U-Zeros given the recall priority).
