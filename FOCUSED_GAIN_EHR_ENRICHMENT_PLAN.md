# FOCUSED PLAN — Enrich EHR for the Current Multimodal Long-Tail Experiment

## Objective

Build one practical stochastic EHR completion module to:

1. fill missing continuous lab/vital positions;
2. create multiple plausible EHR views for existing tail-positive studies;
3. test whether enriched EHR improves the locked CXR–EHR classifier.

This is not a general imputation benchmark.

---

## Chosen method

Use one **GAIN-style generator–discriminator**.

Do not run a broad comparison against RNN, LSTM, Transformer, VAE and multiple GAN variants in the first experiment.

### Generator

Input:

```text
observed normalized labs/vitals
original/input masks
noise at missing positions
setting, demographics, comorbidity and recency context
```

Output:

```text
all 13 continuous lab/vital values
```

Observed entries remain unchanged. Generated values replace only missing entries.

### Discriminator

Output one observed-versus-generated probability for each of the 13 features.

Use a hint vector as in GAIN.

---

## Training population

### Phase 1 — warm start

Use complete and high-completeness rows from TRAIN subjects.

Artificially hide observed entries and reconstruct them.

Do not use a fixed one-epoch warm start. Stop by masked validation reconstruction.

### Phase 2 — all-row adversarial training

Continue on all TRAIN-subject study-level EHR snapshots.

Loss on known values is calculated only for genuinely observed entries that were artificially hidden.

Originally missing entries receive adversarial supervision, not reconstruction targets.

Retain the current index-time temporal extraction rules.

---

## Loss

Generator:

\[
L_G
=
\lambda_{\mathrm{rec}}L_{\mathrm{rec}}
+
\lambda_{\mathrm{adv}}L_{\mathrm{adv}}.
\]

Start with reconstruction dominant:

```text
lambda_rec = 10
lambda_adv = 1
```

Discriminator:

```text
component-wise BCE on mask status
```

A strong reconstruction term prevents the generator from producing population-plausible but patient-inappropriate values.

---

## Minimal validation only

Do not build a full benchmark suite.

Require only:

### 1. Hidden-value reconstruction

On a subject-level validation subset:

```text
Macro normalized MAE at least 10% better than TRAIN median
at least 10/13 features better than median
```

### 2. Plausibility

```text
clinical-range violation below 1%
no obvious collapse to one value
pairwise-correlation distortion small
```

### 3. Stochasticity

For the same incomplete patient and different noise draws:

```text
observed values remain identical
missing values vary within plausible limits
```

### 4. Downstream usefulness

This is the actual deciding test:

```text
baseline:
current median placeholder + original masks

enriched:
GAIN-completed values + original masks + imputed mask
```

Proceed only if downstream Macro/NonTarget-10 are safe and Rare-3/Tail or stability improve.

---

## Dataset enrichment strategy

Do not create arbitrary synthetic patients.

For each existing Rare-3-positive TRAIN study:

```text
same real CXR
same real labels
same observed EHR
different stochastic samples only at originally missing positions
```

Use moderate oversampling:

```text
Rare-3-positive study:
1 original view + 1 stochastic EHR view initially
```

Do not create many copies initially.

For head/non-target studies:

```text
use one imputed realization or the original representation
```

This changes training exposure without pretending that synthetic completions are independent patients.

---

## Controlled downstream arms

Only three arms are needed:

```text
A0:
current EHR representation
median placeholder + masks

A1:
single GAIN completion for all TRAIN studies

A2:
A1 + one extra stochastic completion for Rare-3-positive studies
```

Keep:

```text
same classifier architecture
same loss
same optimizer
same number of update steps where possible
same validation set
```

A1 tests learned imputation.

A2 tests tail-oriented data enrichment.

Do not add WASL/DBL in the same first experiment.

---

## Important safeguards

1. Fit the GAIN model and all normalization only on TRAIN subjects.
2. Preserve original missingness masks after imputation.
3. Add an imputed-mask indicator; do not present generated values as observed.
4. Never alter observed values.
5. Never pair generated EHR with another patient's CXR.
6. Do not condition the first generator directly on target disease labels.
7. Validation and test are never oversampled.
8. At validation, use a fixed seed or average a small fixed number of imputations.

---

## Recommended implementation order

```text
Step 1:
audit complete/high-completeness subset size

Step 2:
train reconstruction-warm-start generator

Step 3:
add component-wise discriminator and GAIN fine-tuning

Step 4:
freeze the generator

Step 5:
generate one completion for all TRAIN rows

Step 6:
generate one extra completion for Rare-3-positive TRAIN rows

Step 7:
run A0/A1/A2 downstream comparison
```

## Final decision

The first experiment is not “which imputation architecture is best?”

It is:

> Does one practical GAIN-based completion and tail-oriented stochastic enrichment improve the current multimodal long-tail classifier without harming the other labels?
