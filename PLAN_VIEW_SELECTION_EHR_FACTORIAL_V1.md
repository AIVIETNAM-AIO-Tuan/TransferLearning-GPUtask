# PLAN V1 — Disease-Specific View Selection × Selective EHR Residual Fusion

## 0. Mục tiêu

Thí nghiệm trả lời bốn câu hỏi độc lập:

1. Trọng số frontal/lateral học được có tốt hơn quy tắc cố định `0.7/0.3` không?
2. Một trọng số riêng theo từng bệnh có tốt hơn một trọng số chung học được không?
3. EHR còn bổ sung tín hiệu sau khi view pooling đã được tối ưu không?
4. EHR và view selection có bổ sung lẫn nhau, hay EHR chỉ đang bù cho một image-pooling
   bottleneck?

Đây là thí nghiệm study-level, sử dụng prediction đã cache từ image model tốt nhất.
Không fine-tune backbone, ML-Decoder hoặc EHR encoder trong thí nghiệm chính.

---

# 1. Thiết kế tổng thể

Thiết kế xác nhận chính là factorial 2×2:

| | Không EHR | Full EHR residual |
|---|---|---|
| Fixed view `0.7/0.3` | **A0** | **B0** |
| Learned disease-specific view | **A1** | **B1** |

Các contrast:

```text
G_view        = A1 - A0
G_EHR_fixed   = B0 - A0
G_EHR_learned = B1 - A1
G_combined    = B1 - A0

Interaction =
(B1 - A1) - (B0 - A0)
```

Cách đọc interaction:

```text
Interaction ≈ 0:
view selection và EHR gần như cộng tính

Interaction < 0:
EHR gain giảm khi image pooling tốt hơn
→ EHR có thể đang bù cho view-pooling bottleneck

Interaction > 0:
learned view selection làm EHR hữu ích hơn
→ hai modality có tính bổ sung
```

---

# 2. Nguồn model và dữ liệu

## 2.1 Image model

Dùng schedule R0 tốt nhất được khóa từ notebook baseline optimization.

Bắt buộc:

```text
RAD-DINO supervised checkpoint
patch-only ML-Decoder
không CLS
không thay backbone trong notebook này
```

Chạy riêng cho mỗi image seed:

```text
42
1337
2024
```

Mỗi seed xuất prediction cấp ảnh trên VAL và TEST.

## 2.2 Study-level view scores

Với mỗi study và class `c`:

```text
p_F[i,c] = mean probability của tất cả frontal images
p_L[i,c] = mean probability của tất cả lateral images
m_F[i]   = có frontal
m_L[i]   = có lateral
```

Không trộn ảnh khác subject hoặc khác study.

Study chỉ có frontal:

```text
p_view = p_F
```

Study chỉ có lateral:

```text
p_view = p_L
```

Chỉ study có cả F+L mới cung cấp gradient cho tham số view selection.

## 2.3 EHR

Dùng cohort time-safe hiện tại:

```text
mọi feature phải khả dụng tại index_time
split ở subject level
không dùng report text
```

Train hai EHR-only model trên TRAIN:

```text
Q_CTX:
demographics
care setting
missingness indicators
measurement counts/recency

Q_FULL:
toàn bộ Q_CTX
+ lab values
+ vital values
+ comorbidity
+ time-safe microbiology features
```

Khuyến nghị primary EHR learner:

```text
XGBoost hoặc CatBoost
```

Không dùng deep EHR encoder trong primary experiment.

Xuất:

```text
q_CTX[i,c]
q_FULL[i,c]
```

cho VAL và TEST.

---

# 3. View models

## 3.1 A0 — fixed baseline

\[
p^{A0}_{i,c}
=
\frac{
m_{F,i}\,0.7\,p_{F,i,c}
+
m_{L,i}\,0.3\,p_{L,i,c}
}{
m_{F,i}\,0.7+m_{L,i}\,0.3
}.
\]

Mẫu số renormalize là bắt buộc.

## 3.2 AG — learned global weight

Một trọng số frontal chung:

\[
\alpha_0
=
0.5+0.5\sigma(a_0).
\]

\[
p^{AG}_{i,c}
=
\frac{
m_F\alpha_0p_F+
m_L(1-\alpha_0)p_L
}{
m_F\alpha_0+m_L(1-\alpha_0)
}.
\]

Initialization:

```text
alpha_0 = 0.70
```

`AG` là mechanism control, không nằm trong factorial chính.

## 3.3 A1 — learned disease-specific weights

\[
\alpha_c
=
0.5+0.5\sigma(a_0+\Delta_c).
\]

Trong đó:

```text
a0:
global frontal tendency

Delta_c:
disease-specific deviation
```

Initialization:

```text
alpha_c = 0.70 cho cả 13 nhãn
Delta_c = 0
```

Giới hạn primary:

```text
0.50 <= alpha_c <= 1.00
```

Lý do:

```text
frontal là primary radiograph;
audit hiện tại không cho bằng chứng lateral-dominant;
giới hạn giảm nguy cơ rare-label weight chạy tới nghiệm cực đoan do noise.
```

Exploratory sensitivity sau primary:

```text
alpha_c ∈ [0,1]
```

không dùng để cứu primary null result.

## 3.4 Regularization

\[
L_{\mathrm{view}}
=
L_{\mathrm{BCE}}
+
\lambda_g(a_0-\operatorname{logit}(0.4))^2
+
\lambda_d\sum_c\Delta_c^2.
\]

Do mapping `0.5 + 0.5 sigmoid(a)`, `alpha=0.70` tương ứng `sigmoid(a)=0.40`.

Prelocked grids:

```text
lambda_g ∈ {0, 0.01, 0.1, 1.0}
lambda_d ∈ {0.01, 0.1, 1.0, 10.0}
```

Không chọn lambda trên TEST.

---

# 4. View controls

## 4.1 Availability-only control AV

Kiểm tra gain có đến từ chính nội dung lateral hay từ việc bệnh nhân được chỉ định lateral.

\[
z^{AV}_{i,c}
=
\operatorname{logit}(p_{F,i,c})
+
\gamma_c\,\mathbb{1}(m_L=1).
\]

Với L-only study, dùng `logit(p_L)`.

Nếu:

```text
AV ≈ A1 > A0
```

thì gain có thể chủ yếu đến từ acquisition-policy/context signal.

## 4.2 Shuffled-lateral control AS

Trong các study F+L:

```text
giữ frontal prediction và label nguyên
permute toàn bộ lateral prediction vector ở mức subject
chỉ permute trong cùng care setting
```

Dùng 5 permutation seeds được khóa trước.

Fit lại A1 trên mỗi permutation.

Nếu:

```text
A1(real lateral) > AS(shuffled lateral)
```

thì lateral image content có đóng góp.

Nếu:

```text
A1 ≈ AS
```

không được claim learned view selection khai thác nội dung lateral.

---

# 5. EHR residual architecture

## 5.1 Image coefficient được cố định

Sau khi có `p_view`:

\[
z^{IMG}_{i,c}
=
\operatorname{logit}
\left(
\operatorname{clip}(p^{view}_{i,c},10^{-6},1-10^{-6})
\right).
\]

Không học lại hệ số của image score.

## 5.2 Chuẩn hóa EHR score

Với mỗi class và calibration fold:

\[
u^{EHR}_{i,c}
=
\frac{
\operatorname{logit}(q_{i,c})-\mu_c
}{
\sigma_c+\epsilon
}.
\]

`mu` và `sigma` chỉ tính trên phần calibration-training của fold.

## 5.3 Selective residual

\[
z^{FINAL}_{i,c}
=
z^{IMG}_{i,c}
+
\beta_c u^{EHR}_{i,c}.
\]

Primary constraint:

```text
0 <= beta_c <= 1
```

Parameterization:

\[
\beta_c=\sigma(b_c).
\]

Initialization:

```text
b_c = -8
beta_c = sigma(-8) = 3.354e-04
```

> **AMENDMENT 2026-07-31 — §5.3.** Bản gốc ghi `beta_c ≈ 0` cạnh một assertion §8 đòi
> `init logits == image-only`. Hai điều đó **bất khả đồng thời**: `sigma(-8)` không bằng `0`.
>
> Văn bản chuẩn hoá:
>
> ```text
> beta_c được khởi tạo rất gần 0 nhưng KHÔNG bằng 0.
> Initialization phải gần như bảo toàn ranking và AP trong tolerance kỹ thuật;
> KHÔNG yêu cầu bit-for-bit identity với image-only.
> ```
>
> `b_init = -8` **giữ nguyên**. Không hạ xuống `-12`: `dbeta/db = beta(1-beta)` bằng
> `3.35e-04` ở `-8` nhưng chỉ `6.14e-06` ở `-12` — nhỏ hơn ~55 lần, đủ để LBFGS gần như
> không mở được residual branch.

Regularization:

\[
L_{\mathrm{EHR}}
=
L_{\mathrm{BCE}}
+
\lambda_\beta\sum_c\beta_c^2.
\]

Grid:

```text
lambda_beta ∈ {0.01, 0.1, 1.0, 10.0}
```

Không có EHR intercept riêng trong primary model.

Lý do:

```text
image path phải có đường lui chính xác;
residual không được trở thành nơi sửa bias chung không liên quan EHR.
```

## 5.4 Sequential fitting

Primary:

```text
1. fit và khóa alpha
2. freeze alpha
3. fit beta
```

Không fit `alpha` và `beta` jointly trong primary experiment.

Joint fitting chỉ là exploratory follow-up vì nó làm attribution view/EHR kém sạch.

---

# 6. EHR arms và controls

## 6.1 Main factorial arms

```text
A0:
fixed 0.7/0.3, không EHR

A1:
learned alpha_c, không EHR

B0:
fixed 0.7/0.3 + Q_FULL residual

B1:
learned alpha_c + Q_FULL residual
```

## 6.2 Context decomposition

```text
B1_CTX:
A1 + Q_CTX residual
```

Contrasts:

```text
Context gain       = B1_CTX - A1
Clinical increment = B1 - B1_CTX
```

Nếu `B1_CTX` lấy gần hết gain:

```text
EHR chủ yếu là care-setting/missingness context
```

Nếu `B1 > B1_CTX` rõ:

```text
lab/vital/clinical values có residual signal riêng
```

## 6.3 Shuffled-EHR control

```text
B1_SHUF:
A1 + shuffled Q_FULL residual
```

Permutation:

```text
subject-level
trong cùng care setting
trong cùng EHR-completeness quintile
5 fixed permutation seeds
```

Yêu cầu để claim EHR content:

```text
B1 > B1_SHUF
```

`B1 > A1` một mình chưa đủ vì residual branch có thể học cohort/context artefact.

---

# 7. Calibration và validation protocol

## 7.1 TEST không được dùng để lựa chọn

VAL là calibration/development set.

TEST chỉ chạy một lần sau khi khóa:

```text
arm definitions
regularization
permutation seeds
optimization
metric
go/no-go
```

## 7.2 Nested subject-level cross-validation trên VAL

Outer folds:

```text
5 folds
group = subject_id
cùng fold assignment cho mọi arm
```

Inner folds:

```text
4 folds trong outer-train
chọn lambda
```

Mỗi outer fold:

```text
fit AG/A1/AV trên outer-train
fit beta cho B0/B1/B1_CTX trên outer-train
predict outer-held-out
```

Ghép năm fold để tạo OOF-VAL prediction.

Không fit hyperparameter trên OOF-held-out fold.

## 7.3 Final TEST fit

Sau khi OOF design được khóa:

```text
chọn lambda bằng modal value hoặc median log-lambda của 5 outer folds
fit toàn bộ calibration layer trên full VAL
predict TEST một lần
```

EHR-only learners `Q_CTX/Q_FULL` vẫn chỉ được train trên TRAIN.

---

# 8. Optimization

Vì số tham số rất nhỏ:

```text
optimizer: LBFGS hoặc Adam
FP64
deterministic
max 500 update
early stop theo calibration-training loss
```

Không early-stop bằng outer-held-out AP.

Khuyến nghị:

```text
LBFGS
line_search_fn="strong_wolfe"
```

Assertions:

```text
A0 reproduction max_abs_diff <= 1e-8
A1 init logits == A0 logits
B0/B1 init: NEAR-IDENTITY voi paired image-only arm (xem amendment)
single-view study prediction không phụ thuộc alpha
subject folds không chồng lấn
TEST chưa bao giờ xuất hiện trong fit
```

> **AMENDMENT 2026-07-31 — §8.** Assertion `B0/B1 init logits == corresponding image-only
> arm` là **bất khả thi** với `beta = sigma(-8)`. Thay bằng near-identity **khai báo trước**:
>
> ```text
> Spearman toi thieu        >= 0.9999
> max per-label |dAP|       <= 1e-4        (= 5% cua cong thuc dung +0.002)
> ```
>
> Báo cáo **cả bốn** đại lượng, không chỉ giá trị lớn nhất:
>
> ```text
> Macro |dAP| | Rare-3 |dAP| | NonTarget |dAP| | max per-label |dAP|
> ```
>
> **Kiểm đúng cặp baseline.** Nhiễu khởi tạo **không** common-mode giữa các B-arm: `B0` cộng
> residual lên `A0`, `B1` lên `A1`, `B1_CTX` dùng `u_CTX`, `B1_SHUF` dùng `u` đã hoán vị.
> Ranking nền và vector `u` khác nhau nên nhiễu **không** triệt tiêu chính xác trong `B1-B0`
> hay `B1_CTX-B1`. Vì vậy kiểm từng arm với paired baseline của chính nó:
>
> ```text
> B0 init      <-> A0
> B1 init      <-> A1
> B1_CTX init  <-> A1
> B1_SHUF init <-> A1
> ```
>
> **Không** assert `B0 init == B1 init`. Thứ phải giống nhau giữa các arm chỉ là
> `beta_init_B0 == beta_init_B1 == beta_init_B1_CTX == beta_init_B1_SHUF`.
>
> **Scale diagnostic (diagnostic only, không đổi initialization chính).** Chạy lại ở
> `b = -12` và yêu cầu `Spearman >= 0.99999`, `max |dAP| <= 1e-5`, kèm kiểm số học
> `delta12 == (beta12/beta8) * delta8` (`rtol 1e-6`). Nếu `-12` **không** làm nhiễu nhỏ đi
> thì dừng lại và kiểm: sai paired baseline · prob/logit không nhất quán · dùng nhầm fitted
> beta · lệch hàng dự đoán · clipping/standardization khác giữa hai đường code.
>
> **Chẩn đoán `|u|`.** In `p50 / p90 / p99 / p999 / max` của `|u|` cho `Q_FULL`, `Q_CTX` và
> từng nhãn, kèm `max |dz| = beta_init * max|u|`.
>
> Đây là sửa một **mâu thuẫn kỹ thuật trong plan**. **Không** đổi model, dữ liệu,
> hyperparameter huấn luyện, `beta` initialization, endpoint, hay success gate khoa học.

---

# 9. Endpoints

## 9.1 Primary

```text
Macro mAP
```

## 9.2 Key secondary

```text
Rare-3 mAP
NonTarget-10 mAP
Tail/Pleural_Other AP
per-label AP
```

## 9.3 Diagnostics

```text
F+L subset Macro/Rare-3/per-label AP
F-only subset
care-setting subsets
view availability
alpha_c per fold/seed
beta_c per fold/seed
real vs shuffled controls
```

AUROC và calibration là secondary only.

---

# 10. Success gates

## 10.1 Disease-specific view selection

Primary contrast:

```text
A1 - AG
```

Practical contrast:

```text
A1 - A0
```

View selection succeeds when TEST shows:

```text
A1 - AG Macro >= +0.001
same direction in >=2/3 image seeds
NonTarget-10 >= -0.0015
subject-cluster bootstrap does not show material harm
```

A Rare-3-specific success may also be reported when:

```text
Rare-3 >= +0.002
```

Content claim additionally requires:

```text
A1 > AS
```

If `A1 ≈ AV`, describe gain as acquisition-policy/context exploitation, not lateral-content gain.

## 10.2 EHR residual

Primary:

```text
B1 - A1
```

EHR succeeds when:

```text
Macro >= +0.002
OR Rare-3 >= +0.002

NonTarget-10 >= -0.0015
same direction in >=2/3 image seeds
B1 > B1_SHUF
subject-cluster bootstrap no material harm
```

## 10.3 Combined system

```text
B1 - A0 Macro >= +0.002
```

with the same safeguards.

## 10.4 Interaction

No magnitude gate.

Report:

```text
Interaction point estimate
subject-cluster CI
sign across seeds
```

---

# 11. Bootstrap and seed handling

## 11.1 Bootstrap

```text
2,000 resamples
cluster = subject_id
paired across arms
```

For rare labels:

```text
report valid bootstrap fraction
skip resample with no positive
```

## 11.2 Image seeds

Repeat full calibration/test process for:

```text
42
1337
2024
```

Report:

```text
delta per seed
mean delta
same-direction count
subject bootstrap per seed
```

Optional deployment ensemble:

```text
average final probabilities of 3 seeds
```

This is secondary; primary inference is per-seed consistency.

---

# 12. Interpretation matrix

| Result | Interpretation |
|---|---|
| `AG>A0`, `A1≈AG` | 0.7 chưa tối ưu toàn cục; không có disease-specific gain |
| `A1>AG>A0` | disease-specific view selection có giá trị |
| `A1≈AS` | learned weights không khai thác lateral content rõ |
| `A1≈AV>A0` | gain chủ yếu từ lateral-availability/context |
| `B0>A0`, `B1≈A1` | EHR bù cho fixed view pooling; gain mất khi image pooling tốt hơn |
| `B1>A1` và `B1>B1_SHUF` | EHR có residual content sau learned view selection |
| `B1_CTX≈B1` | signal chủ yếu là setting/missingness |
| `B1>B1_CTX` | clinical values thêm signal |
| `A1≈A0`, `B1≈B0` | view selection không quan trọng; EHR độc lập với view rule |
| mọi contrast null | cả view selection và EHR residual dưới ngưỡng thực dụng |

---

# 13. Không được làm sau khi xem TEST

```text
không đổi label subset
không hard-code alpha_Fracture=1
không tắt EHR riêng Pleural_Other
không đổi beta constraint
không đổi lambda
không chọn arm theo per-label TEST result
không thêm patient-specific gate để cứu null
```

Patient-specific view gate chỉ được mở thành experiment mới nếu A1 thắng AG.

Deep EHR fusion chỉ được mở nếu B1 thắng A1 và real > shuffle.

---

# 14. Artifacts

Notebook:

```text
19_view_selection_ehr_factorial.ipynb
```

Directory:

```text
DATA_DIR/view_ehr_factorial/
```

Required files:

```text
manifests/
  study_view_manifest.parquet
  val_subject_folds.parquet
  test_manifest_hash.json

predictions/
  image_view_probs_seed{seed}_{val,test}.npz
  ehr_ctx_probs_{val,test}.npz
  ehr_full_probs_{val,test}.npz

models/
  view_params_seed{seed}.json
  ehr_residual_params_seed{seed}.json

metrics/
  oof_val_metrics.csv
  test_metrics.csv
  per_label_deltas.csv
  factorial_interaction.csv
  bootstrap_subject_2000.csv
  alpha_stability.csv
  beta_stability.csv
  shuffled_controls.csv

experiment_manifest.json
```

`experiment_manifest.json` phải chứa:

```text
image checkpoint hashes
cohort hash
VAL/TEST manifest hashes
fold assignments
regularization grids
permutation seeds
code commit
all success gates
```

---

# 15. Computational staging

## Stage 0 — no/low GPU

```text
cache p_F/p_L
train Q_CTX/Q_FULL on TRAIN
generate VAL/TEST EHR scores
verify hashes
```

## Stage 1 — OOF-VAL development

```text
AG/A1/AV/AS
B0/B1/B1_CTX/B1_SHUF
nested CV
lock all choices
```

## Stage 2 — TEST confirmation

```text
fit calibration layers on full VAL
run TEST once
3 image seeds
subject bootstrap
```

Expected compute:

```text
image forward cache: dominant cost
view/EHR calibration: minutes on CPU
no new backbone training
```

---

# 16. Primary scientific claim permitted by this design

Nếu thành công:

> Disease-specific view selection improves study-level CXR classification beyond both a
> fixed frontal/lateral rule and a learned global view weight. A time-safe EHR residual adds
> further predictive information only when real EHR outperforms shuffled controls. The
> factorial interaction determines whether EHR is complementary to optimized image pooling
> or primarily compensates for a view-pooling bottleneck.

Nếu null:

> Neither disease-specific view weighting nor selective EHR residual fusion produced a
> practically meaningful improvement over the fixed image-only baseline under the locked
> evaluation protocol.
