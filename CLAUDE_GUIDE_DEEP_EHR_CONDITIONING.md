# Hướng dẫn cho Claude — Rà soát kết luận 06b và thiết kế thí nghiệm EHR-conditioning sâu trong RAD-DINO

## 0. Mục tiêu của tài liệu

Tài liệu này dùng để hướng dẫn Claude:

1. rà soát các kết luận trong `REPORT_frozen_token_fusion.md`;
2. chỉ ra những điểm đang được diễn giải quá mạnh hoặc chưa được chứng minh trực tiếp;
3. sửa lại câu chữ của report theo mức bằng chứng hiện có;
4. thiết kế một thí nghiệm giới hạn, có kiểm soát, để kiểm tra giả thuyết cuối cùng:

> EHR có thể cải thiện mô hình nếu được dùng để **điều kiện hóa quá trình hình thành patch representation bên trong các block sâu của RAD-DINO**, thay vì chỉ fusion sau khi backbone đã tạo xong representation hay không?

Không mở rộng thành một chương trình EHR fusion lớn. Mục tiêu là một phép thử bác bỏ hoặc xác nhận có giới hạn, với tiêu chí dừng được khóa trước.

---

# 1. Bối cảnh thực nghiệm đã có

Các kết quả chính của thí nghiệm 06b:

| Head | Mô tả | Test macro mAP |
|---|---|---:|
| H1 | CLS + MLP | 0.4117 ± 0.0008 |
| H2 | CLS + MLP + concat EHR | 0.4199 ± 0.0027 |
| H3 | CLS + ML-Decoder | 0.4068 ± 0.0010 |
| H4z | CLS + token rỗng | 0.4003 ± 0.0069 |
| H4 | CLS + EHR token | 0.4052 ± 0.0008 |
| H5 | Patch + ML-Decoder | 0.4336 ± 0.0032 |
| H6z | Patch + token rỗng | 0.4330 ± 0.0044 |
| H6 | Patch + EHR token | 0.4339 ± 0.0064 |

Các delta quan trọng:

```text
Representation gain, H5 − H1 : +0.0219, p = 0.002
EHR gain trên CLS, H2 − H1   : +0.0081, p = 0.004
EHR gain trên patch, H6 − H5 : +0.0003, p = 0.830
```

Các diagnostic khác:

```text
H6 real-vs-shuffled EHR:
delta mAP ≈ 0
corr(logit) = 0.999934

Post-hoc linear stacking:
H5 prediction + EHR embedding
delta mAP = −0.0027
```

Các kết quả này là bằng chứng mạnh rằng:

> EHR không còn residual predictive signal dễ khai thác khi được fusion **sau frozen patch representation** cho tác vụ 13 nhãn CheXpert hiện tại.

Tuy nhiên, chúng chưa trực tiếp kiểm tra:

\[
f(CXR \mid EHR)
\]

trong đó EHR can thiệp vào quá trình tạo representation bên trong RAD-DINO.

---

# 2. Những điểm trong report hiện tại chưa hoàn toàn hợp lý

## 2.1. Kết luận “EHR chủ yếu vá bottleneck CLS” đang hơi mạnh

### Vấn đề

Dữ liệu hiện có chứng minh:

- EHR giúp global CLS representation;
- EHR không giúp frozen patch representation qua late fusion;
- EHR không cải thiện prediction của H5 qua linear stacking.

Nhưng chưa chứng minh tuyệt đối rằng toàn bộ gain trên CLS chỉ là “vá bottleneck”.

Có thể còn một phần interaction mà:

- không tồn tại sẵn trong output logits của H5;
- không thể được recovery bằng linear stacker;
- chỉ xuất hiện nếu EHR thay đổi internal image representation.

### Yêu cầu sửa câu chữ

Không viết:

> EHR gain chủ yếu chỉ vá bottleneck của CLS.

Nên viết:

> Kết quả hiện tại nhất quán với giả thuyết rằng phần lớn EHR gain quan sát trên CLS đến từ việc bù cho thông tin bị mất trong global pooling. Giả thuyết này được hỗ trợ mạnh nhưng chưa trực tiếp loại trừ EHR-conditioned feature extraction bên trong backbone.

---

## 2.2. “Ba đường bằng chứng độc lập” không hoàn toàn không chia sẻ giả định

### Vấn đề

Ba phép thử khác nhau về cơ chế:

1. train H6 end-to-end;
2. shuffle EHR trong model đã train;
3. linear stacking hậu nghiệm.

Nhưng chúng vẫn chia sẻ:

- cùng cohort;
- cùng 13 nhãn;
- cùng EHR feature set;
- cùng frozen image backbone;
- cùng test set;
- cùng family của downstream prediction.

### Yêu cầu sửa câu chữ

Không viết:

> Ba cơ chế không chia sẻ giả định nào.

Nên viết:

> Ba phép thử có cơ chế khác nhau và hội tụ về cùng kết quả, dù vẫn chia sẻ dữ liệu, target và frozen image backbone.

---

## 2.3. Linear stacking không phải bằng chứng quyết định chống lại deep fusion

### Vấn đề

Linear stacking kiểm tra gần với:

\[
\hat y = h(\operatorname{logit}_{H5}, e)
\]

Nó trả lời:

> Sau khi H5 đã tạo prediction, EHR embedding có bổ sung signal tuyến tính hoặc đơn giản không?

Nó không kiểm tra:

\[
Z_l = B_l(Z_{l-1}; e)
\]

trong đó EHR thay đổi:

- patch-to-patch interaction;
- feature morphology;
- class-relevant spatial attention;
- information được bảo toàn qua block cuối.

### Yêu cầu sửa câu chữ

Không gọi stacking là:

> đường bằng chứng quyết định loại trừ lỗi kiến trúc.

Nên gọi là:

> bằng chứng mạnh chống lại late fusion và additive post-hoc complementarity, nhưng không trực tiếp kiểm tra conditional feature extraction bên trong backbone.

---

## 2.4. “Model chủ động bỏ qua EHR” là diễn giải quá mức

### Vấn đề

Dữ liệu cho thấy:

- EHR embedding có biến thiên;
- EHR token có biến thiên;
- token nhận attention;
- logits gần như bất biến.

Điều này có thể do:

- cancellation giữa attention heads;
- output projection triệt tiêu contribution;
- residual path lấn át;
- clinical feature không phù hợp với decision boundary;
- model học representation EHR nhưng không dùng để thay đổi prediction.

### Yêu cầu sửa câu chữ

Không viết:

> Model chủ động bỏ qua EHR.

Nên viết:

> Ảnh hưởng của patient-specific EHR bị triệt tiêu hoặc không được truyền hiệu quả tới logits khi frozen patch representation đã được cung cấp.

---

## 2.5. Kết luận “dừng đầu tư vào EHR fusion” cần giới hạn phạm vi

### Vấn đề

Có đủ bằng chứng để dừng:

- mở rộng late fusion;
- thêm nhiều EHR token sau backbone;
- cross-attention chỉ ở output patch;
- mở rộng microbiology, vitals, comorbidity;
- tối ưu thêm FiLM/concat trên frozen representation.

Nhưng chưa có phép thử trực tiếp đối với deep conditioning.

### Yêu cầu sửa câu chữ

Nên viết:

> Không tiếp tục đầu tư rộng vào late fusion hoặc EHR feature engineering cho tác vụ này. Chỉ cân nhắc một thí nghiệm deep-conditioning cuối cùng, được giới hạn trước, để kiểm tra giả thuyết EHR-conditioned feature extraction.

---

## 2.6. Test set đã được sử dụng thích nghi

### Vấn đề

Test set đã được dùng để:

- đánh giá H1–H6;
- xem per-label AP;
- hình thành giả thuyết Fracture/Pneumonia;
- quyết định có nên thử deep fusion;
- xây dựng kết luận hiện tại.

Một deep-fusion experiment mới trên cùng test set không còn là confirmatory proof.

### Yêu cầu

Claude phải ghi rõ:

> Mọi deep-fusion experiment tiếp theo trên cùng test set là exploratory extension.

Để có confirmatory claim, cần:

- một subject-level holdout chưa từng xem;
- hoặc external cohort.

---

# 3. Kết luận được phép giữ nguyên

Claude có thể giữ các kết luận sau:

1. Patch representation mạnh hơn global CLS rõ rệt trong pipeline hiện tại.
2. EHR cải thiện global CLS representation.
3. EHR không cải thiện frozen patch representation qua late EHR token fusion.
4. Shuffle EHR gần như không làm thay đổi H6 logits.
5. Linear stacking không tìm thấy residual EHR gain trên H5.
6. Late fusion trên frozen patch output không còn là hướng ưu tiên.
7. Microbiology, vitals và comorbidity không tạo gain đáng kể ngoài labs trong cohort hiện tại.
8. ASL và image representation là hai đòn bẩy lớn hơn EHR late fusion.

---

# 4. Câu hỏi nghiên cứu mới

## Câu hỏi chính

> EHR-conditioned adapters đặt trong các block cuối của RAD-DINO có cải thiện 13-label CheXpert classification so với một unconditional adapter có cùng số parameter hay không?

## Giả thuyết null

\[
H_0:
mAP(D_{\text{EHR-cond}})
-
mAP(D_{\text{unconditional}})
\le 0
\]

## Giả thuyết nghiên cứu

\[
H_1:
mAP(D_{\text{EHR-cond}})
-
mAP(D_{\text{unconditional}})
> 0
\]

Primary estimand:

\[
\Delta_{\text{deep EHR}}
=
mAP(D2)-mAP(D1)
\]

Không dùng trực tiếp `D2 − H5` làm estimand chính, vì adapter capacity tự nó có thể cải thiện model.

---

# 5. Thiết kế thí nghiệm mới

## 5.1. Các model bắt buộc

| ID | Mô tả | Vai trò |
|---|---|---|
| D0 | H5 patch baseline | baseline hiện tại |
| D1 | Patch + unconditional adapters | parameter-matched control |
| D2 | Patch + EHR-conditioned adapters | target model |
| D2-shuffle | D2 nhưng shuffle EHR ở test | causal diagnostic |
| D2-zero | D2 nhưng zero EHR ở test | dependency diagnostic |

Không thay backbone sang ConvNeXt V2 trong phép thử này.

Không thêm grouped tokens, new EHR features hoặc cross-attention lớn.

---

## 5.2. EHR input

Chỉ dùng nhóm feature đã có bằng chứng mạnh nhất:

```text
7 labs
7 lab masks
setting, chỉ nếu cần
```

Không dùng vòng đầu:

```text
microbiology
vitals
comorbidity
```

Lý do:

- không có gain trong ablation trước;
- tăng capacity và noise;
- làm khó diễn giải null result.

EHR encoder có thể giữ cấu trúc:

```text
input
→ Linear
→ LayerNorm
→ GELU
→ Dropout
→ Linear
→ e[128]
```

Mọi value/mask và train-only normalization phải giữ nguyên notebook 06.

---

# 6. Kiến trúc đề xuất

## 6.1. Vị trí conditioning

Bắt đầu với hai block cuối:

```text
RAD-DINO blocks 1–10: frozen
RAD-DINO block 11: conditional adapter
RAD-DINO block 12: conditional adapter
ML-Decoder
```

Không condition block đầu vì:

- block đầu chủ yếu học texture/edge;
- clinical context hợp lý hơn ở semantic blocks;
- giảm nguy cơ EHR làm méo low-level visual features.

---

## 6.2. Conditional residual adapter

Cho image token representation tại block \(l\):

\[
Z_l \in \mathbb{R}^{B\times N\times768}
\]

EHR embedding:

\[
e \in \mathbb{R}^{B\times128}
\]

Thiết kế:

\[
u_l =
W_{\text{down},l}\operatorname{LN}(Z_l)
\]

\[
\gamma_l(e),\beta_l(e)
=
G_l(e)
\]

\[
\tilde u_l
=
(1+\gamma_l(e))\odot u_l+\beta_l(e)
\]

\[
\Delta Z_l
=
W_{\text{up},l}\operatorname{GELU}(\tilde u_l)
\]

\[
Z_l'
=
Z_l+\alpha_l\Delta Z_l
\]

Trong đó:

```text
adapter bottleneck: 64 hoặc 128
gamma/beta broadcast trên token dimension
alpha_l: trainable scalar
```

---

## 6.3. Identity initialization bắt buộc

Khởi tạo:

```python
nn.init.zeros_(W_up.weight)
nn.init.zeros_(W_up.bias)
alpha_l = 1.0
```

hoặc:

```python
alpha_l = nn.Parameter(torch.tensor(0.0))
W_up dùng init chuẩn
```

Chỉ chọn một trong hai.

Yêu cầu tại step 0:

```python
assert torch.allclose(
    logits_D1_step0,
    logits_D0_step0,
    atol=1e-6,
)
```

Với D2:

```python
assert torch.allclose(
    logits_D2_step0,
    logits_D0_step0,
    atol=1e-6,
)
```

Identity phải đúng ở logit level, không chỉ feature level.

---

# 7. Unconditional control D1

D1 phải có cùng số parameter gần nhất với D2.

Thay vì sinh \(\gamma,\beta\) từ EHR:

```python
gamma = nn.Parameter(torch.zeros(1, adapter_dim))
beta  = nn.Parameter(torch.zeros(1, adapter_dim))
```

Hoặc dùng một learned constant context vector:

```python
e_const = nn.Parameter(torch.zeros(1, 128))
```

sau đó chạy cùng generator `G_l`.

Yêu cầu:

- cùng adapter bottleneck;
- cùng generator architecture;
- cùng decoder;
- cùng optimizer groups;
- cùng LR;
- cùng number of trainable parameters gần nhất có thể;
- khác biệt duy nhất là context có phụ thuộc bệnh nhân hay không.

---

# 8. Training protocol

## 8.1. Stage A — Adapter-only

Train:

```text
EHR encoder
conditional/unconditional adapters
ML-Decoder
```

Freeze:

```text
toàn bộ RAD-DINO blocks
patch embedding
positional embedding
```

Config khởi đầu:

```text
loss          = ASL
epochs        = 12
patience      = 3
effective B   = 256
adapter LR    = 1e-4
EHR LR        = 1e-4
decoder LR    = 1e-4
weight decay  = 1e-2
warmup        = 5%
grad clip     = 5.0
AMP           = bf16
```

Checkpoint selection:

```text
validation study-level macro mAP
```

Không dùng test để chọn epoch hoặc hyperparameter.

---

## 8.2. Stage B — Partial unfreeze

Chỉ chạy nếu Stage A cho tín hiệu dương theo go/no-go rule.

Unfreeze:

```text
block 11
block 12
```

Learning rates:

```text
last two blocks : 1e-6 đến 5e-6
adapters        : 1e-4
EHR encoder     : 1e-4
ML-Decoder      : 1e-4
```

D1 và D2 phải unfreeze cùng block, cùng LR, cùng scheduler.

Không unfreeze toàn bộ backbone trong vòng đầu.

---

# 9. Shared forward và fairness

D0, D1 và D2 không thể dùng hoàn toàn cùng một backbone forward nếu adapters được đặt bên trong block 11–12, vì representation đã khác nhau.

Tuy nhiên:

- blocks 1–10 có thể chạy chung;
- output của block 10 có thể detach hoặc checkpoint;
- sau block 10, tạo ba branch D0/D1/D2.

Pseudo-code:

```python
with torch.no_grad():
    z10 = backbone.forward_to_block10(images)

logits_D0 = forward_blocks_11_12_plain(z10)
logits_D1 = forward_blocks_11_12_unconditional(z10, adapter_D1)
logits_D2 = forward_blocks_11_12_conditional(z10, ehr, adapter_D2)
```

Nếu block 11–12 được unfreeze ở Stage B:

- không detach `z10` nếu cần gradient về block 10; nhưng block 10 vẫn frozen;
- branch-specific block 11–12 phải có state riêng hoặc được khởi tạo từ cùng checkpoint;
- optimizer, scheduler và gradient clipping phải riêng.

Không cộng loss rồi global-clip toàn bộ parameter của các branch.

---

# 10. Sanity checks bắt buộc

## 10.1. Step-0 identity

```python
assert max_abs(logits_D0 - logits_D1) < 1e-6
assert max_abs(logits_D0 - logits_D2) < 1e-6
```

## 10.2. Gradient routing

Stage A:

```python
assert no_grad(blocks_1_to_12)
assert has_grad(adapter_D1)
assert has_grad(adapter_D2)
assert has_grad(ehr_encoder_D2)
assert has_grad(decoder_D1)
assert has_grad(decoder_D2)
```

Stage B:

```python
assert no_grad(blocks_1_to_10)
assert has_grad(blocks_11_to_12_D1)
assert has_grad(blocks_11_to_12_D2)
```

## 10.3. Parameter matching

Export:

```text
trainable_param_D1
trainable_param_D2
relative_difference
```

Mục tiêu:

```text
relative difference < 5%
```

Nếu lớn hơn, Claude phải giải thích hoặc sửa control.

## 10.4. EHR alignment

- cùng study dùng cùng EHR cho tất cả image;
- shuffle ở mức study;
- shuffle trong cùng split;
- ưu tiên shuffle trong cùng setting;
- không dùng future EHR.

---

# 11. Metrics và kiểm định

## Primary metric

```text
test study-level macro mAP
```

## Primary comparison

\[
D2-D1
\]

## Secondary comparisons

```text
D1 − D0: adapter capacity gain
D2 − D0: net deep-conditioning gain
D2_real − D2_shuffle
D2_real − D2_zero
```

## Secondary metrics

- per-label AP;
- macro AUROC;
- head/medium/tail mAP;
- setting-stratified mAP;
- calibration;
- training stability;
- parameter count;
- GPU time.

## Statistical plan

Dùng tối thiểu:

```text
5 seeds
```

Lý do:

- expected effect nhỏ;
- 3 seed không đủ power cho khoảng +0.001 đến +0.004.

Dùng:

- paired seed differences;
- subject-level cluster bootstrap;
- report mean, CI95 và raw per-seed results.

Không dùng batch làm statistical unit.

---

# 12. Go/no-go rule khóa trước

## Stage A

| Kết quả `D2 − D1` | Quyết định |
|---|---|
| `< +0.002`, seed-CI chứa 0 | Dừng deep EHR fusion |
| `+0.002 đến +0.005`, nhất quán seed và real > shuffle | Chạy Stage B |
| `> +0.005` | Chạy Stage B và xem xét mở rộng có giới hạn |
| gain chỉ xuất hiện ở 1–2 nhãn post-hoc | Không claim |

## Stage B

| Kết quả | Quyết định |
|---|---|
| `< +0.002` | Kết luận EHR conditioning không phải hướng ưu tiên |
| `+0.002 đến +0.005` | Chỉ tiếp tục nếu external/holdout validation xác nhận |
| `> +0.005` | Có cơ sở nghiên cứu grouped conditioning hoặc thêm block |

Không thay threshold sau khi xem test.

---

# 13. Diagnostic cơ chế

## 13.1. Representation shift

Đo:

\[
d_l
=
\|Z_l^{\text{real}}-Z_l^{\text{shuffle}}\|_2
\]

tại:

```text
sau block 11
sau block 12
trước ML-Decoder
```

Nếu representation thay đổi nhưng logits không đổi:

> conditioning được học nhưng bị downstream head triệt tiêu.

Nếu representation không thay đổi:

> EHR generator hoặc adapter không truyền tín hiệu.

---

## 13.2. Logit intervention

Đo:

```text
real EHR
zero EHR
shuffle EHR
median EHR
```

Các metric:

```text
mean absolute logit delta
per-label logit delta
corr(logit_real, logit_shuffle)
macro mAP delta
```

---

## 13.3. Adapter magnitude

Log:

```text
||ΔZ_l|| / ||Z_l||
gamma mean/std
beta mean/std
alpha_l
```

Nếu adapter magnitude gần zero trong toàn training:

> model không tìm được ích lợi từ conditioning.

Nếu magnitude lớn nhưng test gain zero:

> conditioning có thể overfit hoặc gây non-useful representation shift.

---

# 14. Những hướng không được làm trong vòng này

Claude không nên:

- đổi sang ConvNeXt V2;
- thêm grouped EHR tokens;
- thêm EHR cross-attention quy mô lớn;
- mở rộng feature EHR;
- thay loss khỏi ASL;
- đổi split;
- chọn nhãn sau khi xem kết quả;
- dùng lại test để tune;
- unfreeze toàn bộ backbone;
- claim confirmatory nếu vẫn dùng test hiện tại.

Mỗi thay đổi trên sẽ tạo thêm confound và làm mất khả năng diễn giải.

---

# 15. Hướng phát triển mới sau deep-conditioning

## Nếu deep-conditioning thất bại

Chuyển trọng tâm sang:

1. image representation;
2. attention pooling;
3. partial image fine-tuning;
4. loss design;
5. calibration;
6. external validation;
7. các target mà EHR có lý do lâm sàng để bổ sung.

Các target phù hợp hơn:

```text
mortality
ICU transfer
length of stay
readmission
treatment response
deterioration
prognosis
```

Không tiếp tục dùng 13 nhãn CheXpert để chứng minh multimodal complementarity nếu image branch đã gần đủ thông tin theo định nghĩa target.

## Nếu deep-conditioning thành công

Chỉ sau khi:

```text
D2 − D1 > +0.005
real > shuffle
gain nhất quán qua seed
```

mới cân nhắc:

- condition thêm block 9–10;
- grouped labs token;
- class-specific conditioning;
- external cohort;
- target-specific analysis được nêu trước.

---

# 16. Deliverables Claude phải tạo

Claude cần tạo hoặc cập nhật:

```text
REPORT_frozen_token_fusion_revised.md
PLAN_deep_ehr_conditioning.md
07_deep_ehr_conditioning.ipynb
```

Trong notebook cần có:

```text
C0  Config và preregistered thresholds
C1  Load cohort/EHR/image manifest
C2  Reuse lab-only EHR featurization
C3  Load RAD-DINO checkpoint
C4  Forward blocks 1–10
C5  D0 baseline branch
C6  D1 unconditional adapter
C7  D2 conditional adapter
C8  Identity and gradient gates
C9  Stage-A training
C10 Validation study-level pooling
C11 Five-seed sweep
C12 Real/zero/shuffle diagnostics
C13 Subject bootstrap
C14 Stage-B optional partial unfreeze
C15 Export metrics and report tables
```

Outputs:

```text
deep_ehr/
├── checkpoints/
├── predictions/
├── metrics/
│   ├── per_seed.csv
│   ├── per_label.csv
│   ├── intervention.csv
│   ├── representation_shift.csv
│   └── subject_bootstrap.csv
└── experiment_manifest.json
```

---

# 17. Kết luận hướng dẫn

Claude phải giữ lập trường cân bằng:

> Kết quả 06b bác bỏ mạnh late EHR fusion trên frozen patch representation, nhưng chưa trực tiếp bác bỏ EHR-conditioned feature extraction bên trong RAD-DINO.

Do đó:

> Chỉ chạy một thí nghiệm deep-conditioning nhỏ, parameter-matched, identity-initialized và preregistered. Nếu `D2 − D1 < +0.002` hoặc seed-CI chứa 0, dừng hướng EHR fusion cho 13 nhãn CheXpert này.

Đây là phép thử cuối nhằm phân biệt:

\[
D(f(CXR),EHR)
\]

với:

\[
D(f(CXR\mid EHR))
\]

mà không mở thêm confound về backbone, loss, EHR feature set hoặc target.
