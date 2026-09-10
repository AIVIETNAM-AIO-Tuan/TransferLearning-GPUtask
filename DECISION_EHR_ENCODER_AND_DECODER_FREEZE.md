# Quyết định kiến trúc — Freeze hay Unfreeze EHR Encoder và ML-Decoder trong Full-EHR Mid-vs-Late Screen

## 0. Kết luận điều hành

Có hai thành phần dễ bị gọi lẫn là “encoder/decoder”:

1. **EHR encoder**: mạng mã hóa labs, microbiology, vitals và comorbidity thành vector `e`.
2. **ML-Decoder**: head đọc patch representation của RAD-DINO để tạo 13 logits.

Trong kế hoạch hiện tại, `EhrEncoder` thực ra đã được thiết kế để **train**:

```text
grad adapter > 0
grad EHR encoder > 0
grad ML-Decoder = 0 khi FREEZE_DECODER=True
```

Vì vậy, câu hỏi freeze/unfreeze trong ảnh chủ yếu là câu hỏi về **ML-Decoder**, không phải EHR encoder.

### Khuyến nghị

```text
EHR encoder: luôn train.
ML-Decoder: warm-start từ H5, freeze ngắn rồi unfreeze với LR thấp.
```

Cụ thể:

```text
epoch 0–1:
    freeze ML-Decoder
    train EHR encoder + adapters

sau warm-up:
    unfreeze ML-Decoder
    decoder LR thấp hơn adapter/encoder 5–20 lần
```

Đây là phương án cân bằng nhất giữa:

- attribution sạch;
- tránh false negative;
- không phá H5 decoder warm-start;
- đủ power để decoder thích nghi với representation mới.

---

# 1. Việc thêm lại microbiology là hợp lý

## 1.1. Vì sao loại microbiology dựa trên thí nghiệm 06 chưa đủ cơ sở

Kết quả:

```text
micro ≈ 0 trong late fusion
```

không tự động suy ra:

```text
micro ≈ 0 trong deep conditioning
```

Notebook mid-vs-late tồn tại chính vì giả thuyết:

\[
\text{deep conditioning} \neq \text{late fusion}
\]

Microbiology, đặc biệt `org_name` và specimen information, có thể cung cấp context gần với chẩn đoán và có khả năng thay đổi cách model diễn giải opacity hoặc consolidation.

Do đó, thêm lại:

```text
labs
microbiology org/spec
vitals
comorbidity
```

là hợp lý cho một phép thử quyết định.

## 1.2. Rủi ro khi dùng Full EHR

Full EHR cũng làm tăng:

- noise;
- sparse categories;
- shortcut risk;
- capacity;
- nguy cơ overfit validation.

Vì vậy phải giữ:

```text
per-tier metrics
real-vs-shuffle diagnostics
micro bag sensitivity gate
timestamp/leakage checks
```

Nếu Full EHR dương nhưng:

```text
real ≈ shuffle
```

thì không được quy gain cho patient-specific EHR.

---

# 2. EHR encoder nên freeze hay unfreeze?

## 2.1. Khuyến nghị: EHR encoder phải train

Nếu `EhrEncoder` được tạo mới hoặc chứa các embedding mới cho microbiology mà bị freeze, model gần như đang dùng:

```text
random/fixed projection của EHR
```

Điều này làm phép thử thiếu power và không hợp lý.

Ngay cả khi warm-start encoder từ 06b, representation đó được học trong một cơ chế fusion khác. Mid-block conditioning cần encoder thích nghi với:

- adapter location;
- full EHR feature mix;
- microbiology bags;
- gradient từ image-conditioned objective.

Do đó:

```python
FREEZE_EHR_ENCODER = False
```

nên là mặc định.

## 2.2. Unconditional arm vẫn phải train encoder

Trong unconditional arm:

```python
e = enc(NULL_BATCH)
```

`NULL_BATCH` hằng giữa mọi bệnh nhân.

Nếu encoder train, nó trở thành:

> learned constant context có cùng architecture và gần cùng parameter capacity với conditional arm.

Điều này là control tốt hơn việc loại hoàn toàn encoder khỏi unconditional arm.

Yêu cầu:

- conditional và unconditional encoder copy từ cùng initialization;
- optimizer riêng;
- cùng LR;
- cùng regularization;
- unconditional output constant giữa sample.

## 2.3. Khi nào mới freeze EHR encoder?

Chỉ nên freeze trong một diagnostic phụ, ví dụ:

> Kiểm tra EHR representation đã học ở 06b có thể được tái sử dụng mà không adaptation hay không.

Không dùng frozen EHR encoder làm main screen.

---

# 3. ML-Decoder frozen — lợi và hại

## 3.1. Lợi ích

### Attribution sạch

Nếu decoder frozen, khác biệt so với H5 chỉ đến từ:

```text
EHR encoder
conditional/unconditional adapter
vị trí adapter
```

Do đó:

\[
G_{\text{mid}}=M_E-M_U
\]

dễ diễn giải hơn.

### Ổn định

- không phá H5 decoder;
- ít parameter train;
- ít overfit validation;
- ít compute và VRAM hơn;
- identity step 0 rõ ràng.

### Phù hợp cho diagnostic

Nếu frozen decoder vẫn dùng được representation đã được điều biến, đó là bằng chứng mạnh rằng EHR effect thực sự đi qua feature representation.

## 3.2. Rủi ro lớn nhất: false negative

Decoder H5 được train trên distribution gốc:

\[
Z_{\text{H5}}
\]

Adapter tạo representation mới:

\[
Z'=Z_{\text{H5}}+\Delta Z
\]

Dù \(\Delta Z\) có chứa signal EHR thật, frozen decoder có thể không biết cách đọc nó.

Khi đó có thể thấy:

```text
real-vs-shuffle representation thay đổi
adapter hoạt động
EHR encoder có gradient
nhưng logits gần như không đổi
```

Đây không chứng minh EHR vô dụng; nó chỉ chứng minh fixed H5 decoder không sử dụng được feature mới.

Với một screening experiment, false negative là loại lỗi tốn kém vì có thể bỏ nhầm một hướng tốt.

---

# 4. ML-Decoder unfrozen — lợi và hại

## 4.1. Lợi ích

### Power cao hơn

Decoder có thể thích nghi với representation đã đổi:

\[
Z'\rightarrow \text{new decision boundary}
\]

Điều này đặc biệt quan trọng khi:

- adapter nằm ở block 5–7;
- còn nhiều transformer block tạo distribution shift;
- Full EHR có microbiology embedding mới;
- EHR signal không align sẵn với H5 decoder axes.

### Giảm nguy cơ false negative

Unfreeze cho cơ chế cơ hội tốt nhất để thể hiện effect thật.

### Warm-start giúp hội tụ nhanh

Decoder không khởi tạo ngẫu nhiên mà bắt đầu từ H5. Với LR nhỏ, nó chỉ cần local adaptation.

## 4.2. Rủi ro

### Trộn hiệu ứng

Gain có thể đến từ:

- continued decoder training;
- decoder capacity;
- adapter;
- EHR conditioning.

Tuy nhiên, paired controls giảm đáng kể confound này:

\[
G_{\text{mid}}=M_E-M_U
\]

vì cả `M_E` và `M_U` đều có decoder trainable với cùng schedule.

### Overfit validation

Mỗi arm có decoder riêng, nên tổng capacity lớn hơn.

Giảm thiểu bằng:

- LR thấp;
- early stopping;
- L2-SP tới H5 weights;
- cùng schedule cho mọi arm;
- không tune trên test.

### Catastrophic forgetting

Nếu decoder LR quá cao, H5 decision boundary có thể bị phá.

Đây là failure mode từng xuất hiện trước đây.

Giảm thiểu bằng:

```text
decoder LR thấp hơn adapter LR 5–20 lần
freeze warm-up
gradient clipping
optional L2-SP
```

### Decoder có thể hấp thụ gain không liên quan EHR

Nếu cả conditional và unconditional cùng tăng, đó là continued-training gain, không phải EHR gain.

Do đó phải báo:

```text
M_E − M_U
L_E − L_U
real − shuffle
```

không chỉ absolute mAP của `M_E`.

---

# 5. Phương án khuyến nghị — Delayed unfreeze

Không chọn hai cực:

```text
freeze decoder suốt toàn run
```

hoặc:

```text
unfreeze decoder từ step 0 với LR cao
```

Dùng staged training:

## Stage 0 — Identity

```text
warm-start decoder từ H5
zero-init adapters
eval FP32
all arms == H5
```

## Stage 1 — Representation warm-up

```text
1 epoch hoặc 10% optimizer steps
decoder frozen
EHR encoder train
adapter train
```

Mục tiêu:

- adapter bắt đầu tạo perturbation có cấu trúc;
- EHR encoder học mapping ổn định;
- tránh decoder chạy theo noise ban đầu.

## Stage 2 — Decoder adaptation

```text
unfreeze decoder
tiếp tục train encoder + adapter + decoder
```

LR đề xuất:

```text
adapter LR      = 1e-4
EHR encoder LR  = 1e-4
ML-Decoder LR   = 5e-6 đến 2e-5
```

Khởi điểm hợp lý:

```text
ML-Decoder LR = 1e-5
```

Tỷ lệ:

```text
decoder LR = 0.1 × adapter LR
```

## Stage 3 — Selection

```text
checkpoint bằng validation study-level macro mAP
patience = 2–3
```

---

# 6. Regularization decoder

## 6.1. L2-SP về H5 decoder

Thêm:

\[
L_{\text{total}}
=
L_{\text{ASL}}
+
\lambda_{\text{SP}}
\|\theta_{\text{dec}}-\theta_{\text{H5}}\|_2^2
\]

Mục đích:

- cho phép local adaptation;
- ngăn decoder trôi quá xa khỏi H5;
- giảm overfit.

Giá trị pilot:

```text
lambda_SP = 1e-5 hoặc 1e-4
```

Không grid-search rộng trong screen.

## 6.2. Gradient clipping

Clip riêng từng arm:

```python
clip_grad_norm_(arm.decoder.parameters(), 1.0)
clip_grad_norm_(arm.adapter.parameters(), 5.0)
clip_grad_norm_(arm.encoder.parameters(), 5.0)
```

Có thể dùng decoder clip thấp hơn vì LR nhỏ và muốn giữ ổn định.

---

# 7. Controls cần giữ khi decoder trainable

## 7.1. Mỗi arm có decoder riêng

Không share một trainable decoder giữa nhiều arm.

Nếu share:

\[
\nabla L_{M_E}+\nabla L_{M_U}
\]

sẽ làm các arm coupling và phá independent comparison.

Mỗi arm phải:

- copy cùng H5 decoder state;
- optimizer riêng;
- scheduler riêng;
- checkpoint riêng.

## 7.2. Giữ D0 frozen reference

D0 dùng H5 nguyên bản, không train.

Báo:

```text
M_U − D0
M_E − D0
L_U − D0
L_E − D0
```

nhưng primary vẫn là paired EHR gains:

```text
G_mid  = M_E − M_U
G_late = L_E − L_U
```

## 7.3. Có thể thêm D0-retrain phụ

Nếu chi phí cho phép:

```text
D0-retrain = H5 decoder continued training, không adapter
```

Nó đo continued-training effect thuần.

Không bắt buộc cho one-day screen nếu đã có unconditional controls.

---

# 8. Diagnostic quyết định

## 8.1. Nếu decoder frozen

Nếu thấy:

```text
representation real-vs-shuffle thay đổi
logit real-vs-shuffle gần 0
```

không được kết luận null.

Phải chạy lại với decoder adaptation.

## 8.2. Nếu decoder unfrozen

Kết luận EHR effect chỉ khi:

```text
G_mid ≥ +0.002
real EHR > shuffled EHR
logit sensitivity > 0
dấu nhất quán qua seed
```

Nếu:

```text
M_E và M_U cùng tăng
G_mid ≈ 0
```

thì gain đến từ decoder adaptation, không phải EHR.

Nếu:

```text
M_E tăng
real ≈ shuffle
```

thì model dùng capacity hoặc distributional shortcut, không dùng patient-specific EHR.

---

# 9. Có nên chạy frozen và unfrozen cùng lúc?

## 9.1. Phương án mạnh nhất nhưng tốn hơn

Chạy một seed pilot:

```text
Regime A: decoder frozen
Regime B: delayed unfreeze
```

Cùng Full EHR và cùng H5.

Chi phí bổ sung khoảng:

```text
4–8 GPU-hours
```

Lợi ích:

- biết frozen có tạo false negative hay không;
- chọn regime trước 3-seed screen;
- không phải suy diễn sau khi run hoàn tất.

## 9.2. Phương án tiết kiệm

Dùng delayed unfreeze làm main regime ngay từ đầu.

Vì screen ưu tiên tránh false negative, đây là lựa chọn thực dụng hơn.

Khuyến nghị cuối:

```text
không cần chạy frozen full 3 seed
dùng delayed unfreeze main
giữ diagnostics để biết decoder có thực sự thích nghi hay không
```

---

# 10. Ảnh hưởng tới estimand

Việc decoder trainable không làm mất tính hợp lệ của:

\[
G_{\text{mid}}=M_E-M_U
\]

và:

\[
I_{\text{location}}
=
(M_E-M_U)-(L_E-L_U)
\]

nếu tất cả arm có:

- same H5 initialization;
- same decoder architecture;
- same LR;
- same warm-up;
- same epoch budget;
- same early-stopping rule.

Unfreezing làm giảm “purity” của comparison với D0, nhưng không phá paired conditional-vs-unconditional comparison.

---

# 11. Compute impact

So với decoder frozen:

```text
delayed unfreeze tăng khoảng 10–25% runtime
```

vì:

- decoder backward được bật;
- optimizer state decoder được lưu;
- validation/checkpoint lớn hơn.

Ước tính one-day screen:

| Regime | GPU-hours |
|---|---:|
| decoder frozen | khoảng 16–26 |
| delayed unfreeze | khoảng 19–32 |
| frozen + unfrozen full comparison | khoảng 30–50 |

Trên một RTX PRO 6000 Blackwell:

```text
delayed unfreeze: khoảng 20–32 giờ
```

Early stopping có thể kéo xuống gần một ngày.

---

# 12. Quyết định cuối cùng cho notebook 11

## Config đề xuất

```python
FULL_EHR = True

FREEZE_EHR_ENCODER = False

DECODER_MODE = "delayed_unfreeze"
DECODER_WARMUP_EPOCHS = 1

LR_ADAPTER = 1e-4
LR_EHR_ENCODER = 1e-4
LR_DECODER = 1e-5

L2SP_DECODER = 1e-5
```

## Gates

```text
EHR encoder grad > 0
adapter grad > 0
decoder grad = 0 trong warm-up
decoder grad > 0 sau unfreeze
identity step 0 = exact
micro bag thay đổi → encoder output thay đổi
NULL_BATCH → constant EHR embedding
```

## Decision

Chạy confirmatory chỉ khi:

```text
G_mid ≥ +0.002
real > shuffle
dấu nhất quán qua seed
```

`I_location` quyết định claim về vị trí:

```text
I_location ≥ +0.002
→ mid tốt hơn late
```

---

# 13. Kết luận

### EHR encoder

Nên **unfreeze/train**. Freeze EHR encoder trong Full-EHR deep-conditioning screen làm giảm power một cách không cần thiết, đặc biệt khi vừa thêm microbiology embeddings.

### ML-Decoder

Không nên freeze suốt toàn bộ run, nhưng cũng không nên unfreeze mạnh ngay từ đầu.

Phương án tốt nhất:

> Warm-start từ H5, freeze trong một warm-up ngắn, rồi unfreeze decoder với LR thấp và regularization về H5.

Cách này:

- giảm nguy cơ false negative;
- vẫn giữ H5 làm anchor;
- cho decoder đủ khả năng đọc representation mới;
- giữ `G_mid` và `I_location` hợp lệ nhờ unconditional controls;
- tránh failure mode decoder bị phá do LR cao.
