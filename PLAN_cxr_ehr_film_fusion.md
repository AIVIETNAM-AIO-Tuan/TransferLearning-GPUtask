# PLAN — CXR × EHR FiLM Fusion (frozen RAD-DINO embeddings)

Spec for Claude Code to implement. Multi-label thoracic disease classification (14 CheXpert labels) fusing frozen RAD-DINO CXR embeddings with EHR features, where EHR **modulates** the CXR features via a FiLM gate. Loss = Distribution-Balanced Loss.

---

## 0. Verify first (do not skip)

1. **Inspect the checkpoint** `/content/drive/MyDrive/lpft_out/raddino_lpft_mv73_dbml_best.pt` (Drive-mounted on Colab). Print `list(torch.load(path, map_location='cpu').keys())` (and one level down if it's `{'state_dict': ...}` / `{'model': ...}`).
   - Expected: an **LP-FT fine-tuned RAD-DINO ViT-B/14 encoder + a classification head**. Load the encoder, **discard the head**, keep the 768-d CLS output.
   - If instead it turns out to be a precomputed embedding tensor/dict, skip §5.1 extraction and load directly.
   - Do not assume key names — branch on what `keys()` actually shows.
2. Confirm the **subject-level split** file (train/val/test by `subject_id`, zero leakage — already established) and load it.
3. Confirm EHR join keys: `dicom_id` (image) → `subject_id` (shared with EHR). Confirm whether a study maps to **multiple** culture/specimen rows (it usually does → see §5.2, use `EmbeddingBag`).

---

## 1. Architecture decisions (locked)

- **Normalization: `LayerNorm`** everywhere in the EHR encoder, FiLM generator, fusion, and head. Rationale: FiLM conditioning is per-sample — BatchNorm would couple patients within a batch and is unstable under class imbalance / inpatient-vs-ED batch-composition drift. Standardize continuous labs at preprocessing (train mean/std), not with BN.
- **Activation: `GeLU`** (default), matching the ViT lineage. `SiLU` is interchangeable; wired as a flag, not a tuning priority.
- Both are exposed as config flags (`norm`, `activation`) so they can be A/B-tested.
- **RAD-DINO stays frozen.** Embeddings are extracted once and cached; the fusion model never backprops into the backbone in this plan.

---

## 2. Data contract

Per sample `i`:

| Field | Shape | Notes |
|---|---|---|
| `x_cxr` | `[768]` float | frozen RAD-DINO CLS embedding (cached) |
| `org_ids` | ragged `[k_i]` long | organism category ids for this study (0+ values) |
| `spec_ids` | ragged `[m_i]` long | specimen-type category ids (0+ values) |
| `labs` | `[L]` float | continuous lab values, standardized, missing→imputed |
| `labs_mask` | `[L]` float | 1 = measured, 0 = missing (MNAR-informative) |
| `y` | `[14]` float | multi-label CheXpert targets, per chosen U-policy |

`org_ids` / `spec_ids` are variable-length per sample → batch as flat index tensor + offsets for `nn.EmbeddingBag`.

Special category ids reserved in every categorical vocab: `PAD=0`, `UNK=1` (unseen at inference), `MISSING=2` (not measured). Build vocabs **on the train split only**.

---

## 3. Repo layout

```
fusion/
  config.py           # dataclass config + ablation flags
  data.py             # dataset, vocab build, standardization, EmbeddingBag collate
  extract_cxr.py      # load checkpoint, extract & cache 768-d embeddings
  models/
    ehr_encoder.py    # entity embeddings + MLP -> e
    film.py           # FiLM generator (identity init) + FiLM fusion
    head.py           # classification head
    fusion_model.py   # wires cxr + ehr + film + head
  losses/db_loss.py   # Distribution-Balanced Loss (ported from official repo)
  train.py            # loop, metrics, per-label thresholds, ablation switches
  eval.py             # AUROC/AUPRC per-label, head/med/tail, confound-corrected Δ
  tests/test_sanity.py
```

---

## 4. Config + ablation switches

Every axis we discussed is a flag so ablations are one-line changes:

```python
@dataclass
class Cfg:
    # data
    labels: int = 14
    emb_dim_org: int = 16          # organism cardinality is high (hundreds)
    emb_dim_spec: int = 8          # specimen cardinality is low (dozens)
    ehr_hidden: int = 128          # dim of e
    cxr_dim: int = 768
    # architecture
    norm: str = "layernorm"        # {"layernorm","batchnorm"}  <- test the recommendation
    activation: str = "gelu"       # {"gelu","silu"}
    dropout_ehr: float = 0.3
    # ablation axes
    ehr_cat_encoding: str = "embedding"   # {"embedding","onehot_mlp"}  <- one-hot vs entity-emb study
    fusion: str = "film"                  # {"none","concat","film"}
    film_gamma_mode: str = "full"         # {"full","gating_only"}   gating_only => beta=0, gamma in (0,1)
    film_identity_init: bool = True       # start == CXR-only
    head_input: str = "h_only"            # {"h_only","h_concat_e"}
    # loss / opt
    loss: str = "db"                # {"db","asl","bce_posweight"}
    lr: float = 3e-4
    weight_decay: float = 1e-2      # apply a stronger group value to the EHR branch
    epochs: int = 50
    batch_size: int = 256
```

---

## 5. Modules

### 5.1 CXR embedding extraction (`extract_cxr.py`)
- Instantiate RAD-DINO ViT-B/14 (HF `AutoModel` for `microsoft/rad-dino` architecture, patch 14, 518×518), load fine-tuned encoder weights from the checkpoint (after §0 inspection), strip the classifier head.
- `model.eval()`, `torch.no_grad()`, autocast. For each image return the **CLS token** (`[768]`).
- Cache to a `.npy` (float32, shape `[N,768]`) indexed by row position + a parallel `dicom_id` index (parquet). `.npy` preferred over 768 parquet columns for this width.
- This script runs **once**; the fusion model reads the cache.

### 5.2 EHR feature builder (`data.py`)
- **Categoricals (organism, specimen) → `nn.EmbeddingBag`.** A study can have several cultures/specimens (multi-valued). `EmbeddingBag(vocab, dim, mode="sum")` is exactly the dense equivalent of `multi-hot × Linear` — this is the "keep the features, drop the explicit one-hot" step. If a field is single-valued it degenerates to a plain lookup. A study with no measurement → the single `MISSING` id.
- `ehr_cat_encoding="onehot_mlp"` branch: build the literal multi-hot vector → `Linear`. Provided **only** to A/B against `"embedding"` (they are the same capacity by construction; expect ~parity, small tail/missing differences).
- **Continuous labs:** compute train mean/std, standardize; winsorize extreme values; median-impute missing and emit `labs_mask`. Feed `[labs, labs_mask]` — the mask carries the MNAR "was it ordered" signal.
- Collate function must emit flat index tensors + offsets for each EmbeddingBag field.

### 5.3 EHR encoder (`ehr_encoder.py`)
```
in  = concat[ bag_org(dim=16), bag_spec(dim=8), labs(L), labs_mask(L) ]   # dim = 24 + 2L
MLP : Linear(in, 256) -> Norm(256) -> Act -> Dropout
      Linear(256, 128) -> Norm(128) -> Act
out : e  [128]
```
`Norm`/`Act` resolved from `cfg.norm` / `cfg.activation`.

### 5.4 FiLM generator + fusion (`film.py`)
Generator maps `e -> (gamma, beta)`, each `[768]`. **Identity init** so training starts equal to CXR-only.

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
            gamma = torch.sigmoid(g)       # in (0,1); with zero init starts at 0.5 -> disable identity in this mode
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
Note: `gating_only` cannot also be identity-init at 1 via sigmoid(0)=0.5 — document this; for the identity-preservation ablation use `full`.

### 5.5 Fusion model (`fusion_model.py`)
- `fusion="none"`: head on `x_cxr` (CXR-only baseline).
- `fusion="concat"`: head on `concat[x_cxr, e]` (naive baseline — the currently-underperforming one).
- `fusion="film"`: `e -> (γ,β) -> h`, head on `h` (or `concat[h, e]` if `head_input="h_concat_e"`).

### 5.6 Head (`head.py`)
```
in  = h [768]  (or concat[h, e] -> 896)
Linear(in, 256) -> Norm -> Act -> Dropout -> Linear(256, 14)   # logits
```
Sigmoid only at inference / thresholding.

### 5.7 Distribution-Balanced Loss (`losses/db_loss.py`)
DB Loss (Wu et al., ECCV 2020) = **rebalanced weighting** (corrects the expected-vs-observed sampling frequency under label co-occurrence) **× BCE with Negative-Tolerant Regularization** (class-aware bias/threshold so the dominant negatives don't drown out rare-positive gradients). Do **not** re-derive the constants — port the official implementation and wire our stats:

- Reference: `wutong16/DistributionBalancedLoss` — the `ResampleLoss` class with `reweight_func='rebalance'`. Copy its math verbatim; adapt only I/O.
- Required input stat: **per-class positive counts `n_k` computed on the TRAIN split** (and `N - n_k` for negatives). Save as `class_freq.pkl` and pass in.
- Reference hyperparameters to start from (verify against the source, then tune):
  - `map_param = dict(alpha=0.1, beta=10.0, gamma=0.9)`  — rebalance-weight smoothing.
  - `focal = dict(focal=True, alpha=0.5, gamma=2.0)`.
  - `logit_reg = dict(neg_scale=2.0, init_bias=0.05)` — NTR: `neg_scale` = negative-tolerant scale, `init_bias` sets the class-specific bias from priors.
- Also implement `asl` and `bce_posweight` behind `cfg.loss` so DB can be A/B'd against them (survey note: RAL beat ASL on CXR-LT — leave a TODO stub for RAL as a later option).

---

## 6. Training (`train.py`)

- Read cached `x_cxr` + EHR features; subject-level split.
- Optimizer `AdamW`, cosine schedule w/ warmup. **Parameter groups:** stronger `weight_decay` + `dropout` on the EHR encoder + FiLM generator (they overfit and cause the fusion regression); light/zero decay on norms/biases.
- Compute `class_freq.pkl` from train targets before the loop.
- Log train/val loss + val metrics each epoch; checkpoint on **best val macro AUPRC (mAP)**.

### Metrics (`eval.py`)
- Primary: **macro AUPRC / mAP** (survey: AUROC is inflated under <2% prevalence). Report macro AUROC as auxiliary.
- Per-label AUROC + AUPRC, with **head / medium / tail** breakdown by train frequency.
- **Per-label thresholds** tuned on val (target per-label recall or F-β) — never a single global 0.5. Report recall/precision at the chosen operating point.
- **Confound-corrected Δ** for EHR complementarity — not naive `fusion − CXR` (algebraically confounded with CXR level). Also report the CXR-only vs FiLM gap **stratified by inpatient/ED** to check the fusion isn't just learning a setting shortcut.
- Never compare architectures by F1 (threshold/class-weight artifact) — use AUROC/AUPRC.

---

## 7. Acceptance / sanity tests (`tests/test_sanity.py`)

1. **Identity start:** with `fusion="film"`, `film_gamma_mode="full"`, `film_identity_init=True`, at step 0 the FiLM logits equal the CXR-only logits (assert `allclose`). This is the anti-fusion-gap guarantee.
2. **No leakage:** assert `subject_id` sets of train/val/test are disjoint.
3. **Shapes:** forward a fake batch through every `fusion` mode; assert `[B,14]` logits.
4. **EmbeddingBag multi-value:** a study with 3 organism ids and a study with 0 (→ MISSING) both produce a `[dim]` vector.
5. **DB loss finite & class-aware:** loss is finite; per-class rebalance weights are larger for rarer classes.
6. **Overfit-one-batch:** model drives loss → ~0 on a single batch (wiring check).

---

## 8. Build order

1. §0 checkpoint inspection → §5.1 extraction + cache.
2. §5.2 data + vocabs + standardization + collate; §7.2/§7.4 tests.
3. §5.3–§5.6 modules; §7.1/§7.3 tests (identity start is the key gate).
4. §5.7 DB loss port + `class_freq.pkl`; §7.5 test.
5. §6 train + eval; run baselines `none` and `concat` first, then `film`.
6. Ablation sweep over the §4 flags (esp. `ehr_cat_encoding`, `fusion`, `film_identity_init`, `norm`), compared on per-label AUPRC + confound-corrected Δ.
