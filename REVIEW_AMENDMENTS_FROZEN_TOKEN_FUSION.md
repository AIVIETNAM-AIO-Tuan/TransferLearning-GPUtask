# Phản biện các đề xuất chỉnh sửa kế hoạch Frozen Token Fusion

## 1. Đánh giá chung

Các đề xuất đều có cơ sở tốt, nhưng mức độ chấp nhận khác nhau:

| Đề xuất | Đánh giá | Quyết định |
|---|---|---|
| ① Dùng chung backbone forward | Rất hợp lý, nhưng chưa “tương đương tuyệt đối” nếu triển khai không cẩn thận | **Chấp nhận có điều kiện** |
| ② Sửa interaction bị confound | Hoàn toàn đúng và ảnh hưởng trực tiếp đến kết luận nghiên cứu | **Nên sửa bắt buộc** |
| ③ Zero-init EHR token | Hữu ích nhưng chưa tạo identity thật sự | **Chấp nhận một phần, cần thêm control** |
| ④ Đo attention mass | Diagnostic tốt, nhưng attention weight không phải bằng chứng nhân quả | **Nên thêm, nhưng không dùng làm kết luận duy nhất** |

---

# 2. Đề xuất ① — Dùng chung forward cho tất cả pipeline

## 2.1. Phần hợp lý

Đề xuất đúng ở điểm quan trọng nhất: backbone đã frozen và giống nhau, nên cùng một ảnh luôn tạo ra cùng CLS và patch tokens. Tính lại RAD-DINO riêng cho từng head là lãng phí lớn.

Việc dùng một backbone forward rồi phân phối feature cho nhiều head sẽ giảm đáng kể thời gian tính toán:

```python
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = backbone(images)

cls = out.last_hidden_state[:, 0].detach()
patches = out.last_hidden_state[:, 1:].detach()
```

Sau đó từng pipeline xử lý các feature này bằng module riêng.

Cách làm này còn tạo một dạng **common random numbers**:

- cùng batch;
- cùng augmentation;
- cùng data order;
- cùng image feature.

Nhờ đó, nuisance variance giữa các pipeline giảm xuống.

## 2.2. Điểm cần phản biện

Phát biểu:

> “Tương đương toán học tuyệt đối với bốn lần train riêng”

chỉ đúng khi **không có bất kỳ trainable parameter hoặc phép biến đổi nào được dùng chung**.

Mỗi pipeline phải có riêng:

- `EhrEncoder`;
- MLP head hoặc ML-Decoder;
- EHR projector;
- modality embeddings;
- optimizer state;
- scheduler;
- early-stop counter;
- gradient clipping.

Nếu P2 và P4 dùng chung một `EhrEncoder`, gradient sẽ là:

\[
\nabla_{\theta_E}(L_{P2}+L_{P4})
=
\nabla_{\theta_E}L_{P2}
+
\nabla_{\theta_E}L_{P4}
\]

Khi đó hai pipeline không còn độc lập.

## 2.3. Bẫy gradient clipping

Không nên làm:

```python
loss = loss_P1 + loss_P2 + loss_P3 + loss_P4
loss.backward()

clip_grad_norm_(all_trainable_parameters, 5.0)
```

Vì gradient của bốn pipeline sẽ bị scale chung theo global norm. Pipeline có gradient lớn sẽ ảnh hưởng update của pipeline khác.

Phải clip riêng:

```python
clip_grad_norm_(model_P1.parameters(), 5.0)
clip_grad_norm_(model_P2.parameters(), 5.0)
clip_grad_norm_(model_P3.parameters(), 5.0)
clip_grad_norm_(model_P4.parameters(), 5.0)
```

## 2.4. Vòng train được khuyến nghị

Nên backward tuần tự trên feature đã detach:

```python
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = backbone(images)

cls = out.last_hidden_state[:, 0].detach()
patches = out.last_hidden_state[:, 1:].detach()

for experiment in active_experiments:
    experiment.optimizer.zero_grad(set_to_none=True)

    loss = experiment.compute_loss(
        cls=cls,
        patches=patches,
        batch=batch,
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        experiment.trainable_parameters,
        max_norm=5.0,
    )

    experiment.optimizer.step()
    experiment.scheduler.step()
```

Ưu điểm:

- backbone chỉ forward một lần;
- không phải giữ graph của nhiều head cùng lúc;
- gradient clipping riêng;
- optimizer hoàn toàn độc lập;
- head đã early-stop có thể bỏ qua.

## 2.5. “Paired ở mức batch” không phải đơn vị thống kê mới

Shared forward giúp so sánh sạch hơn, nhưng batch không trở thành independent experimental unit.

Không nên dùng hàng nghìn batch để tính p-value.

Các phép kiểm định chính vẫn nên là:

- paired seeds;
- paired subject-level bootstrap;
- bootstrap delta prediction trên cùng subject.

Batch pairing chỉ giúp giảm nuisance variance, không làm sample size thống kê tăng từ vài seed thành hàng nghìn batch.

## 2.6. Batch 96–128 chưa nên khóa trước

Phép tính bộ nhớ chỉ dựa vào tensor patch output chưa tính:

- attention workspace;
- image tensor;
- Q/K/V projections;
- ML-Decoder;
- nhiều EHR encoder/head;
- allocator fragmentation.

Nên chạy auto-probe:

```text
B = 32 → 48 → 64 → 96 → 128
```

Sau đó chọn batch lớn nhất vẫn để lại khoảng 15–20% VRAM dự phòng.

## 2.7. Kết luận

**Chấp nhận đề xuất ①**, nhưng sửa phát biểu thành:

> Shared frozen-backbone forward tạo ra gradient tương đương với các run riêng, với điều kiện mỗi pipeline có trainable modules, optimizer, scheduler và gradient clipping hoàn toàn độc lập.

---

# 3. Đề xuất ② — Interaction hiện tại bị lẫn representation và fusion

## 3.1. Đề xuất này hoàn toàn đúng

Interaction ban đầu:

\[
I=(P4-P3)-(P2-P1)
\]

đang so sánh hai nhánh khác nhau ở nhiều yếu tố:

| Nhánh | Representation | Fusion | Head |
|---|---|---|---|
| `P2 − P1` | CLS | concat | MLP |
| `P4 − P3` | patch tokens | EHR token | ML-Decoder |

Nếu \(I>0\), ta không biết nguyên nhân là:

1. patch representation giúp EHR tương tác tốt hơn;
2. token fusion mạnh hơn concat;
3. ML-Decoder phù hợp với EHR hơn MLP;
4. hoặc cả ba.

Do đó, interaction này chỉ là **pipeline-level interaction**, không phải interaction thuần representation.

## 3.2. Thêm P3′ và P2′ là giải pháp hợp lý

Định nghĩa:

```text
P3′ = CLS only → ML-Decoder, sequence length = 1
P2′ = CLS + EHR token → ML-Decoder, sequence length = 2

P3  = patch tokens → ML-Decoder
P4  = patch tokens + EHR token → ML-Decoder
```

Khi đó:

\[
I_{\text{repr}}
=
(P4-P3)-(P2'-P3')
\]

Giữ cố định:

- fusion mechanism;
- head;
- loss;
- backbone checkpoint;
- training protocol.

Khác biệt chính còn lại là image representation:

- một CLS token;
- 1.369 patch tokens.

## 3.3. Caveat của ML-Decoder với sequence length 1

Khi chỉ có một CLS token:

- mọi class query chỉ có một key;
- attention weight trên token đó luôn bằng 1;
- class-specific discrimination đến từ learned queries, projections và classifier parameters, không đến từ lựa chọn vùng ảnh.

Đây không phải lỗi của control. Nó chính là hệ quả của việc representation chỉ còn một global token.

## 3.4. Nên giữ hai interaction

### Pipeline-level interaction

\[
I_{\text{pipeline}}
=
(P4-P3)-(P2-P1)
\]

Diễn giải:

> Pipeline patch-token + EHR-token đạt EHR gain khác pipeline CLS + concat bao nhiêu?

Đây là so sánh hệ thống hoàn chỉnh.

### Representation-controlled interaction

\[
I_{\text{repr}}
=
(P4-P3)-(P2'-P3')
\]

Diễn giải:

> Khi giữ ML-Decoder và token fusion cố định, EHR gain thay đổi thế nào khi chuyển từ CLS sang patch representation?

Đây nên là headline khoa học chính.

## 3.5. Kết luận

**Chấp nhận hoàn toàn đề xuất ②.**

Nếu mục tiêu là kết luận về bottleneck CLS so với patch representation, bổ sung `P2′` và `P3′` là cần thiết.

---

# 4. Đề xuất ③ — Zero-init EHR projector

## 4.1. Phần hợp lý

Zero-init:

```python
nn.init.zeros_(ehr_proj.weight)
nn.init.zeros_(ehr_proj.bias)
```

giúp EHR token ban đầu không chứa tín hiệu ngẫu nhiên.

Đây là cách khởi tạo tốt và phù hợp với kỷ luật identity-init đã dùng cho FiLM.

## 4.2. Zero-init chưa tạo identity tuyệt đối

Giả sử attention logits của 1.369 patch token là:

\[
s_1,\ldots,s_{1369}
\]

Khi chưa có EHR token:

\[
a_i=
\frac{e^{s_i}}
{\sum_j e^{s_j}}
\]

Khi thêm một zero token có logit \(s_E=0\):

\[
a_i'=
\frac{e^{s_i}}
{\sum_j e^{s_j}+1}
\]

Như vậy mọi patch đều bị giảm trọng số, dù EHR token không chứa thông tin.

Zero input cũng chưa chắc tạo zero effective key/value nếu:

- attention projection có bias;
- `ehr_type` không được zero-init;
- LayerNorm có affine parameter khác zero;
- decoder P3 và P4 không bắt đầu từ cùng state dict.

## 4.3. Sửa tối thiểu

Nên thực hiện:

```python
nn.init.zeros_(ehr_proj.weight)
nn.init.zeros_(ehr_proj.bias)
nn.init.zeros_(ehr_type)
```

Đồng thời:

```python
decoder_P4.load_state_dict(decoder_P3.state_dict())
```

để P3 và P4 có decoder initialization giống nhau.

## 4.4. Control tốt hơn: zero-token pipeline

Thêm:

```text
P4_zero = patch tokens + constant zero token → ML-Decoder
P4_real = patch tokens + projected EHR token → ML-Decoder
```

Khi đó:

### Structural cost của việc thêm token

\[
\Delta_{\text{structure}}
=
P4_{\text{zero}}-P3
\]

### Giá trị nội dung EHR

\[
\Delta_{\text{EHR content}}
=
P4_{\text{real}}-P4_{\text{zero}}
\]

Ví dụ:

```text
P3      = 0.425
P4_zero = 0.423
P4_real = 0.431
```

Ta có thể kết luận:

- thêm token làm mất 0.002;
- nội dung EHR đem lại 0.008;
- net gain P4 − P3 là 0.006.

Nếu không có `P4_zero`, ta chỉ thấy +0.006 và không biết EHR thực chất đóng góp +0.008.

## 4.5. Exact identity cần kiến trúc khác

Muốn identity tuyệt đối, có thể dùng gated residual:

```python
image_logits = decoder(patches)
ehr_delta = ehr_branch(e, patches)

logits = image_logits + alpha * ehr_delta
```

với:

```python
alpha = nn.Parameter(torch.tensor(0.0))
```

Tại initialization:

\[
\text{logits}_{P4}
=
\text{logits}_{P3}
\]

Tuy nhiên, kiến trúc này không còn kiểm tra trực tiếp việc EHR token cạnh tranh cùng patch tokens. Vì vậy, nếu mục tiêu chính là token-level fusion, zero-token control phù hợp hơn.

## 4.6. Kết luận

**Chấp nhận zero-init**, nhưng không gọi nó là identity-init.

Nên sửa phát biểu thành:

> Zero-init giảm perturbation ngẫu nhiên khi thêm EHR token. Để tách structural effect của token insertion khỏi clinical content, chạy thêm constant-zero-token control.

---

# 5. Đề xuất ④ — Đo attention mass trên EHR token

## 5.1. Đây là diagnostic hữu ích

Attention mass giúp trả lời:

- nhãn nào attend nhiều vào EHR;
- EHR token có bị chìm giữa 1.369 patch không;
- attention mass có khác uniform baseline hay không.

Diagnostic này hữu ích khi:

\[
P4-P3\approx0
\]

vì nó giúp phân biệt model không sử dụng EHR token với trường hợp model có sử dụng nhưng EHR không cải thiện prediction.

## 5.2. Không nên diễn giải quá mạnh

Hai kết luận sau chưa chắc chắn:

```text
attention ≈ 1/1370 → fusion hỏng
attention >> 1/1370 → EHR được dùng thật
```

Attention weight không đồng nghĩa với causal importance.

Một token có thể có:

- attention weight nhỏ;
- value vector norm lớn;
- contribution sau output projection vẫn đáng kể.

Ngược lại, attention weight lớn không bảo đảm output cuối phụ thuộc vào token đó.

Ngoài ra:

- cần xem từng attention head;
- không nên chỉ average head;
- residual path ảnh hưởng kết quả;
- output projection ảnh hưởng contribution;
- \(1/1370\) chỉ là uniform reference, không phải ngưỡng sử dụng tuyệt đối.

## 5.3. Nên báo cáo ba lớp diagnostic

### Lớp 1 — Raw attention mass

```python
ehr_attn = attn[..., -1]
```

Báo cáo:

- mean theo label;
- median;
- quantiles;
- per-head distribution;
- so với median attention mass của một patch.

### Lớp 2 — Value-weighted contribution

Đo:

\[
c_E=
\left\|
a_E V_E W_O
\right\|_2
\]

và:

\[
r_E=
\frac{\|a_E V_E W_O\|_2}
{\left\|\sum_j a_jV_jW_O\right\|_2+\epsilon}
\]

Điều này phân biệt:

- attention nhỏ nhưng value mạnh;
- attention lớn nhưng contribution yếu.

### Lớp 3 — Causal intervention

Đây vẫn là bằng chứng mạnh nhất:

```text
real EHR
zero EHR
shuffled EHR
```

Kỳ vọng:

\[
mAP_{\text{real}}
>
mAP_{\text{zero}}
\approx
mAP_{\text{shuffle}}
\]

Nếu attention mass cao nhưng `real ≈ shuffle`, model có thể chỉ sử dụng distributional shortcut hoặc token như learned bias.

## 5.4. Kết luận

**Nên thêm attention diagnostic**, nhưng viết lại:

> Attention mass là mechanistic indicator, không phải bằng chứng nhân quả. Kết luận về giá trị của EHR phải dựa chủ yếu trên real/zero/shuffle intervention và prediction delta.

---

# 6. Các điểm nhỏ

## 6.1. Shuffle phải ở mức study

Mọi ảnh của cùng study phải nhận cùng EHR đã tráo.

Không được shuffle riêng từng ảnh, vì khi đó một study có thể nhận nhiều clinical contexts khác nhau trước multi-view pooling.

Nên shuffle trong strata:

```text
same split
same setting
optionally same EHR-availability pattern
```

Permutation phải ở mức `study_id`.

## 6.2. “Frozen” cần ghi rõ

Ở đây, frozen không có nghĩa là pretrained RAD-DINO nguyên bản.

Cách mô tả chính xác:

> Backbone đã được supervised fine-tune trên 13 nhãn CXR ở bước 04b, sau đó mới frozen.

Điều này không phá within-experiment delta vì mọi model dùng cùng backbone, nhưng ảnh hưởng cách diễn giải external generalization.

## 6.3. `patch_norm`

Nếu thêm `patch_norm`, P3 không còn là bản sao của pipeline 04b.

Tuy nhiên, P3 và P4 cùng dùng nó nên EHR delta vẫn công bằng.

Không nên dùng trực tiếp `0.4199` làm baseline cho P3 vì:

- training protocol khác;
- loss có thể khác;
- checkpoint selection khác;
- `patch_norm` khác;
- backbone state và head initialization có thể khác.

---

# 7. Phiên bản thí nghiệm được khuyến nghị

## 7.1. Bộ sáu head chính

| ID | Representation | Head | EHR fusion | Vai trò |
|---|---|---|---|---|
| H1 | CLS | MLP | none | replication baseline |
| H2 | CLS | MLP | concat | replication EHR gain |
| H3 | CLS token | ML-Decoder | none | representation control |
| H4 | CLS + EHR token | ML-Decoder | token | representation control |
| H5 | Patch tokens | ML-Decoder | none | patch baseline |
| H6 | Patch + EHR token | ML-Decoder | token | target model |

Tất cả dùng một frozen backbone forward nhưng có trainable modules riêng.

## 7.2. Structural controls

Nếu chi phí cho phép, thêm:

```text
H4z = CLS + constant-zero token → ML-Decoder
H6z = patches + constant-zero token → ML-Decoder
```

Khi đó:

### EHR content gain trên CLS

\[
G_{\text{CLS}}
=
H4-H4z
\]

### EHR content gain trên patches

\[
G_{\text{Patch}}
=
H6-H6z
\]

### Interaction thuần hơn

\[
I_{\text{content}}
=
(H6-H6z)-(H4-H4z)
\]

Metric này tốt hơn:

\[
(H6-H5)-(H4-H3)
\]

vì loại thêm ảnh hưởng cơ học của việc tăng sequence length thêm một token.

---

# 8. Quyết định cuối cùng

Kế hoạch nên được sửa theo hướng:

1. **Áp dụng shared backbone forward**, nhưng mỗi pipeline có model, optimizer, scheduler, early stopping và gradient clipping riêng.
2. **Thêm CLS + ML-Decoder controls**, vì interaction ban đầu bị confound.
3. **Zero-init EHR projector**, nhưng không gọi đó là identity-init.
4. **Thêm zero-token control**, ít nhất cho patch pipeline.
5. **Ghi attention mass**, đồng thời đo value-weighted contribution.
6. Giữ `real/zero/shuffle` làm bằng chứng chính.
7. Vẫn dùng paired seeds và subject-level bootstrap; không dùng batch làm đơn vị kiểm định.

Hai phát biểu cần phản bác mạnh:

- Shared forward không tự động “tương đương tuyệt đối” nếu có shared EHR encoder hoặc global gradient clipping.
- Zero-init EHR projection không bảo đảm P4 khởi động đúng bằng P3.

Cả hai vấn đề đều sửa được mà gần như không tăng chi phí backbone.
