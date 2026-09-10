# LABEL PIPELINE V1
## Disease-aware uncertainty handling and blank-rate routed distillation

**Mục tiêu chính:** cải thiện Macro mAP của mô hình CXR hiện tại bằng cách sửa cách xử lý nhãn `-1` và `blank`, trong khi giữ nguyên kiến trúc, dữ liệu ảnh, split, preprocessing, view pooling và lịch huấn luyện R0.

## 0. Phạm vi khóa

Chỉ thay đổi supervision. Không thay đổi RAD-DINO, patch-only ML-Decoder, resolution, augmentation, subject split, view pooling, EHR, sampler, checkpoint rule hoặc evaluation labels.

Primary endpoint: `Macro mAP` trên historical VAL/TEST labels hiện tại.

Key secondary: `Rare-3 mAP`, `NonTarget-10 mAP`, `Tail AP`, `Worst-3`, per-label AP.

---

## 1. Raw label states

Mỗi study-label cell giữ đúng raw state:

```text
+1     positive mention
 0     explicit negative
-1     uncertain mention
blank  no mention
```

Mọi thống kê và routing được tính ở mức **study**, không phải mức image.

---

## 2. Chính sách `-1`

```python
HARD_U_ONES = {
    "Atelectasis",
    "Edema",
}

SOFT_U_LABELS = {
    "Cardiomegaly",
    "Consolidation",
    "Enlarged_Cardiomediastinum",
    "Fracture",
    "Lung_Lesion",
    "Lung_Opacity",
    "Pleural_Effusion",
    "Pleural_Other",
    "Pneumonia",
    "Pneumothorax",
}
```

Routing:

```text
Atelectasis raw -1 → hard 1
Edema raw -1       → hard 1
10 pathology labels còn lại raw -1 → OOF teacher probability
No Finding raw -1 → mask
```

---

## 3. Chính sách `blank`

Chỉ dùng raw labels của TRAIN.

Với disease `c`:

\[
r^{blank}_c = \frac{N_{TRAIN}(raw_{i,c}=blank)}{N_{TRAIN}(study)}.
\]

Routing khóa một lần:

```text
blank rate <= 30% → blank = hard 0
blank rate > 30%  → blank = OOF teacher probability
No Finding blank  → hard 0
```

Không tính lại threshold theo fold, seed, VAL hoặc TEST.

Lưu:

```text
label
n_train_studies
n_blank
blank_rate
blank_policy = HARD_ZERO / TEACHER_SOFT
```

---

## 4. Bảng policy cuối

| Raw state | Nhóm | Target cuối |
|---|---|---|
| `+1` | tất cả | hard `1` |
| `0` | tất cả | hard `0` |
| `-1` | Atelectasis, Edema | hard `1` |
| `-1` | 10 pathology còn lại | OOF teacher probability |
| `-1` | No Finding | mask |
| `blank` | pathology blank rate ≤30% | hard `0` |
| `blank` | pathology blank rate >30% | OOF teacher probability |
| `blank` | No Finding | hard `0` |

---

## 5. Teacher

Teacher chỉ dùng để sinh soft targets trên TRAIN.

```text
CXR-only
same image architecture as R0
same preprocessing
same study-level view pooling
no EHR
no architecture change
```

Teacher không được dùng ở inference của final model.

### Teacher hard supervision

```text
raw +1 → 1
raw  0 → 0
Atelectasis/Edema raw -1 → 1
blank của disease blank rate <=30% → 0
No Finding +1 → 1
No Finding 0/blank → 0
```

### Teacher masked cells

```text
raw -1 của SOFT_U_LABELS
raw -1 của No Finding
blank của disease blank rate >30%
```

Teacher loss:

\[
L_T = \frac{\sum M^T_{i,c} L_{ASL}(y^T_{i,c},z^T_{i,c})}{B_{actual}\times13}.
\]

Denominator cố định `B_actual × 13`.

---

## 6. OOF teacher generation

### 6.1 Subject folds

Chia TRAIN thành 5 folds theo `subject_id`.

Yêu cầu:

```text
không subject overlap
một fold map dùng cho mọi label
cân bằng multilabel positive/explicit-negative counts tối đa có thể
```

Teacher seed cố định:

```text
TEACHER_SEED = 42
```

Không chọn teacher seed bằng mAP.

### 6.2 Mỗi fold

```text
train teacher trên 4 folds
predict fold còn lại
```

Mỗi soft target phải được tạo bởi teacher chưa từng train trên chính study đó.

### 6.3 Image setup

Dùng locked R0 schedule:

```text
blocks 0–8 frozen
blocks 9–11 + backbone LN + ML-Decoder theo R0
patch-only
same preprocessing
same checkpoint rule
```

Có thể cache output sau block 8 để giảm compute, nhưng không đổi frozen weights hoặc preprocessing.

---

## 7. Teacher probabilities

Teacher tạo image-level scores trên held-out fold, sau đó aggregate về study level bằng cùng locked view pooling.

\[
p^T_{i,c}=\sigma(z^T_{i,c}).
\]

Soft target:

\[
y^{soft}_{i,c}=clip(p^T_{i,c},10^{-4},1-10^{-4}).
\]

Không threshold thành `0/1` và không calibrate bằng VAL/TEST.

Assertions:

```text
mỗi TRAIN study có đúng một OOF prediction
không study bị thiếu/lặp
shape = [n_train_studies, 13]
all finite
0 < p < 1
```

---

## 8. Final label manifest

Mỗi study-label cell lưu:

```text
subject_id
study_id
label
raw_label
blank_rate
blank_policy
target_type = hard / soft / mask
final_target
teacher_probability
target_reason
teacher_fold
```

`target_reason`:

```text
explicit_positive
explicit_negative
uncertain_hard_positive
uncertain_teacher
blank_hard_negative
blank_teacher
no_finding_positive
no_finding_negative
no_finding_uncertain_mask
```

Pseudocode:

```python
def resolve_target(raw, label, blank_rate, teacher_p):
    if raw == 1:
        return 1.0, "hard", "explicit_positive"

    if raw == 0:
        return 0.0, "hard", "explicit_negative"

    if raw == -1:
        if label in {"Atelectasis", "Edema"}:
            return 1.0, "hard", "uncertain_hard_positive"
        if label == "No_Finding":
            return None, "mask", "no_finding_uncertain_mask"
        return teacher_p, "soft", "uncertain_teacher"

    if label == "No_Finding":
        return 0.0, "hard", "no_finding_negative"

    if blank_rate <= 0.30:
        return 0.0, "hard", "blank_hard_negative"

    return teacher_p, "soft", "blank_teacher"
```

---

## 9. Mapping về image training

Vì ontology hiện tại là study-level:

```text
mọi image trong cùng study nhận cùng final target vector
mọi image trong cùng study nhận cùng hard/soft/mask vector
```

Không dùng teacher image-level target riêng theo view.

---

## 10. Final student loss

Tạo:

```text
Y_HARD [B,13]
M_HARD [B,13]
Y_SOFT [B,13]
M_SOFT [B,13]
```

Loss:

\[
L=\frac{\sum M^{hard}L_{ASL}(Y^{hard},z)+\lambda_{soft}\sum M^{soft}L_{BCE}(Y^{soft},z)}{B_{actual}\times13}.
\]

Khóa:

```text
lambda_soft = 0.5
```

Không renormalize theo số hard/soft cells hoặc tổng weight. Masked cells đóng góp đúng 0.

Teacher probabilities là constants; không backpropagate vào teacher.

---

## 11. Final training

Final model giữ nguyên R0:

```text
same initialization
same frozen/trainable blocks
same LR/scheduler
same augmentation
same batch/accumulation
same epochs
same checkpoint rule
```

Không cần teacher ở inference.

---

## 12. Comparison duy nhất

### H0 — Historical

```text
+1 → 1
-1 → 1
0 → 0
blank → 0
```

### H1 — Corrected labels

```text
+1/0 giữ nguyên
-1 disease-aware
blank routed bằng 30%
OOF CXR-teacher soft targets
```

Không thêm EHR, view selection, temporal relabeling hoặc multimodal teacher.

---

## 13. Evaluation

VAL và TEST giữ historical evaluation labels:

```text
+1/-1 → positive
0/blank → negative
```

Primary:

```text
Macro mAP
```

Secondary:

```text
Rare-3 mAP
NonTarget-10 mAP
Tail AP
Worst-3
per-label AP
```

Bootstrap:

```text
2,000 paired subject-cluster resamples
```

Confirmation:

```text
student seeds 42, 1337, 2024
cùng một OOF teacher-label manifest
```

---

## 14. Success gate

```text
Macro delta >= +0.002
OR Rare-3 delta >= +0.002

NonTarget-10 delta >= -0.0015
same direction >= 2/3 student seeds
subject bootstrap no material harm
```

Teacher mAP không phải endpoint chính. Teacher chỉ cần tạo OOF probabilities hợp lệ và không collapse.

---

## 15. Diagnostics bắt buộc

### Raw label summary

```text
n(+1), n(0), n(-1), n(blank), blank rate, blank route
```

### Teacher hard coverage theo label/fold

```text
hard positives
hard negatives
masked cells
positive prevalence
```

Nếu một fold-label không có cả positive và negative hard supervision:

```text
STOP và sửa group-stratified fold assignment
```

### Soft-target summary

Tách `uncertain_teacher` và `blank_teacher`:

```text
count, mean, std
p01, p10, p25, p50, p75, p90, p99
fraction p>0.5
fraction p>0.9
```

### Effective prevalence

\[
\pi^{eff}_c=\frac{\sum hard\_positive+\sum soft\_target}{N_{TRAIN}}.
\]

So sánh historical prevalence, teacher-hard prevalence và final effective prevalence.

Cảnh báo nếu effective prevalence >2× hoặc <0.5× historical prevalence; cảnh báo không tự đổi policy.

### Leakage checks

```text
subject disjoint across teacher folds
VAL/TEST absent from teacher training
VAL/TEST absent from blank-rate calculation
all TRAIN teacher predictions are OOF
```

---

## 16. Artifacts

```text
label_pipeline_v1/
├── raw_train_label_summary.csv
├── blank_policy_by_label.json
├── train_teacher_folds.parquet
├── teacher_supervision_manifest.parquet
├── teacher_oof_train_probs.npy
├── teacher_oof_index.parquet
├── final_train_label_manifest.parquet
├── final_label_summary.csv
├── teacher_soft_target_summary.csv
├── effective_prevalence.csv
├── experiment_manifest.json
└── checkpoints/
    ├── teacher_fold0.pt
    ├── teacher_fold1.pt
    ├── teacher_fold2.pt
    ├── teacher_fold3.pt
    └── teacher_fold4.pt
```

Manifest phải chứa:

```text
raw label file hash
TRAIN/VAL/TEST split hashes
blank threshold = 0.30
blank rate and policy per label
HARD_U_ONES
SOFT_U_LABELS
teacher fold-map hash
teacher seed
teacher checkpoint hashes
lambda_soft = 0.5
student seeds
code commit
```

---

## 17. Execution order

### Stage 0 — CPU label audit

```text
load raw study labels
compute and lock blank rates
create subject folds
create teacher masks
run assertions
```

### Stage 1 — GPU OOF teacher

```text
train 5 fold teachers
predict held-out TRAIN folds
aggregate study probabilities
save OOF predictions
```

### Stage 2 — CPU final label manifest

```text
apply disease-aware -1 policy
apply 30% blank policy
insert OOF soft targets
save hard/soft/mask matrices
run distribution/leakage checks
```

### Stage 3 — GPU corrected-label model

```text
train H1 using locked R0 setup
evaluate historical VAL
confirm with seeds 42/1337/2024
```

### Stage 4 — TEST

Chỉ mở TEST khi H1 vượt locked VAL gate, cùng chiều >=2/3 seeds, và toàn bộ manifests/checkpoints/hashes đã khóa.

---

## 18. Quyết định khóa

```text
-1:
    Atelectasis, Edema → hard 1
    10 pathology labels → OOF CXR-teacher probability
    No Finding → mask

blank:
    blank rate <=30% → hard 0
    blank rate >30%  → OOF CXR-teacher probability
    No Finding       → hard 0

teacher:
    CXR-only
    5-fold subject-level OOF
    trusted cells only

final training:
    hard ASL + 0.5 × soft BCE
    fixed denominator B×13

experiment:
    historical baseline vs one corrected-label model
```
