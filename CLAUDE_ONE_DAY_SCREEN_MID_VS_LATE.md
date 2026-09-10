# Đề xuất điều chỉnh — Screen Mid-vs-Late trước Confirmatory Full-Pipeline

## 0. Đánh giá chung

Hướng đề xuất là hợp lý:

> Chạy một vòng screen mid-vs-late trên split hiện có, tái sử dụng H5, trong khoảng một ngày. Nếu có tín hiệu đủ lớn thì mới đầu tư vào sealed confirmatory 5-seed; nếu null thì dừng.

Đây là chiến lược **sequential evidence** tốt hơn việc chạy ngay confirmatory 4–6 ngày.

Lợi ích:

- giảm mạnh chi phí tính toán;
- kiểm tra trực tiếp giả thuyết block 5–7;
- không phải retrain 04b/H5 ở vòng screen;
- chỉ trả chi phí confirmatory khi có signal đủ lớn;
- giữ confirmatory holdout chưa được tạo/mở cho tới khi cần.

Tuy nhiên, cần sửa một điểm:

> Không nên dùng riêng \(I_{\text{location}}\) làm cổng duy nhất.

Lý do: mid và late có thể cùng tạo EHR gain tương tự. Khi đó:

\[
I_{\text{location}}\approx0
\]

dù deep conditioning thực sự có ích.

Vì vậy screen phải theo dõi đồng thời:

\[
G_{\text{mid}}=M_E-M_U
\]

\[
G_{\text{late}}=L_E-L_U
\]

\[
I_{\text{location}}=G_{\text{mid}}-G_{\text{late}}
\]

---

# 1. Phạm vi kết luận của vòng screen

Vòng này tái sử dụng H5 hiện có và split hiện có.

Do đó nó trả lời sạch trong phạm vi:

> Với fixed H5 hiện tại, channel-wise EHR conditioning tại block 5–7 có tạo gain lớn hơn conditioning tại block 10–11 hay không?

Nó không phải confirmatory proof ở cấp toàn pipeline vì:

- H5 đã được train trước đó;
- base-model seed variance không được phản ánh;
- split/test hiện tại đã được sử dụng trong các exploratory experiment.

Nếu screen null nhất quán, đây là bằng chứng đủ mạnh để **dừng về mặt kỹ thuật**.

Nếu screen dương, vẫn phải chạy sealed confirmatory full-pipeline trước khi đưa ra claim chính thức.

---

# 2. Arms của vòng screen

Dùng bốn arm:

| Arm | Vị trí adapter | Context |
|---|---|---|
| `M_U` | block 5 và 7 | learned constant |
| `M_E` | block 5 và 7 | patient EHR |
| `L_U` | block 10 và 11 | learned constant |
| `L_E` | block 10 và 11 | patient EHR |

Tất cả phải:

- dùng cùng H5 checkpoint;
- warm-start cùng backbone;
- warm-start cùng ML-Decoder;
- cùng ASL;
- cùng adapter bottleneck;
- cùng số adapter;
- parameter difference giữa unconditional và EHR arm dưới 5%;
- zero-init adapter;
- identity tại step 0.

---

# 3. Identity và fairness

Trong `eval()` và FP32:

```python
assert allclose(H5_logits, M_U_logits, atol=1e-5)
assert allclose(H5_logits, M_E_logits, atol=1e-5)
assert allclose(H5_logits, L_U_logits, atol=1e-5)
assert allclose(H5_logits, L_E_logits, atol=1e-5)
```

Mỗi arm phải có:

- optimizer riêng;
- scheduler riêng;
- early stopping riêng;
- gradient clipping riêng;
- EHR encoder riêng;
- adapter riêng;
- decoder copy riêng nếu decoder tiếp tục train.

Không dùng global gradient clipping cho tất cả arm.

---

# 4. Dữ liệu sử dụng

## 4.1. Training

Dùng existing training split.

## 4.2. Screening evaluation

Dùng existing validation split làm screen set.

Không sử dụng existing test để quyết định:

- vị trí block;
- adapter bottleneck;
- số epoch;
- go/no-go.

Lý do:

> Screen trên validation đủ để quyết định có đáng đầu tư vào confirmatory hay không và tránh thêm adaptive reuse của test.

Existing test có thể giữ nguyên như historical reference, nhưng không dùng làm gate.

---

# 5. Training schedule đề xuất

## 5.1. S0 — smoke test

```text
1 adapter seed
1,000–2,000 study
200–500 optimizer steps
```

Mục tiêu:

- identity;
- gradient routing;
- memory;
- throughput;
- loss giảm;
- real-vs-shuffle representation delta không bằng zero.

Thời gian:

```text
0.25–0.5 GPU-hour
```

## 5.2. S1 — main screen

```text
3 adapter seeds
existing train split
6–8 epochs tối đa
patience = 2
ASL
effective batch = 256
```

Không retrain 04b hoặc H5.

Nếu full train quá chậm, có thể dùng 50% train cho hai epoch đầu; nhưng quyết định cuối nên dựa trên full existing train.

---

# 6. Primary và safeguard estimands

## 6.1. Primary location estimand

\[
I_{\text{location}}
=
(M_E-M_U)-(L_E-L_U)
\]

Ngưỡng thực tiễn:

```text
+0.002 macro mAP
```

## 6.2. Absolute mid-block EHR gain

\[
G_{\text{mid}}=M_E-M_U
\]

Cũng dùng ngưỡng:

```text
+0.002 macro mAP
```

## 6.3. Vì sao cần cả hai

Ví dụ:

```text
G_mid  = +0.004
G_late = +0.003
I      = +0.001
```

Nếu chỉ dùng \(I\), ta sẽ dừng và bỏ lỡ việc deep conditioning thực sự có ích ở cả hai vị trí.

Ngược lại:

```text
G_mid  = +0.001
G_late = −0.002
I      = +0.003
```

Interaction vượt ngưỡng nhưng mid gain vẫn dưới mức thực tiễn. Không nên chạy confirmatory chỉ vì late arm tệ hơn.

Do đó, để claim mid-block cứu được conditioning, cần đồng thời:

```text
G_mid ≥ +0.002
I_location ≥ +0.002
```

---

# 7. Decision matrix

## Trường hợp A — Mid-block có ích và tốt hơn late

```text
G_mid ≥ +0.002
I_location ≥ +0.002
real > shuffle
dấu nhất quán ở ít nhất 2/3 seed
```

Quyết định:

> Xây sealed confirmatory 5-full-pipeline-seed theo kế hoạch chính.

Primary confirmatory estimand:

\[
I_{\text{location}}
\]

## Trường hợp B — Deep conditioning có ích nhưng không phụ thuộc vị trí

```text
G_mid ≥ +0.002
G_late ≥ +0.002
I_location < +0.002
real > shuffle ở cả hai vị trí
```

Quyết định:

> Có thể chạy confirmatory, nhưng câu hỏi phải đổi từ “mid tốt hơn late” sang “deep conditioning có ích hay không”.

Primary confirmatory estimand khi đó:

\[
G_{\text{deep}}=EHR-conditioned-unconditional
\]

Không claim block 5–7 tốt hơn.

## Trường hợp C — Mid-block không cứu được deep conditioning

```text
G_mid < +0.002
real ≈ shuffle
logit sensitivity gần 0
```

Quyết định:

> Dừng hướng deep EHR conditioning.

Kết luận hợp lệ:

> Trong fixed-H5 screen, chuyển channel-wise EHR conditioning từ block 10–11 sang block 5–7 không tạo gain thực tiễn.

Không cần trả thêm 4–6 ngày confirmatory.

## Trường hợp D — Kết quả không ổn định

```text
mean G_mid khoảng +0.001 đến +0.003
dấu khác nhau giữa seed
real-vs-shuffle có signal nhưng mAP không ổn định
```

Quyết định:

> Chạy thêm 2 adapter seed trên cùng H5, tổng 5 seed, trước khi quyết định.

Không retrain full pipeline ở giai đoạn này.

---

# 8. Mechanistic gates

Ngoài mAP, `M_E` phải cho thấy ít nhất một tín hiệu cơ chế.

## 8.1. Representation sensitivity

\[
d_l=
\|Z_l^{real}-Z_l^{shuffle}\|_2
\]

Đo tại:

```text
block 5
block 7
block 9
block 11
block 12
```

## 8.2. Logit sensitivity

```text
mean |logit_real − logit_shuffle|
corr(logit_real, logit_shuffle)
```

## 8.3. Adapter activity

```text
||ΔZ|| / ||Z||
gamma std
beta std
adapter gradient norm
```

Nếu:

```text
G_mid ≈ 0
representation delta ≈ 0
logit delta ≈ 0
```

thì null có diễn giải rõ:

> Adapter không tìm thấy patient-specific signal hữu ích.

Nếu representation thay đổi nhưng logits không đổi:

> EHR conditioning tác động feature nhưng không cải thiện decision boundary.

---

# 9. Statistical treatment của screen

Screen không phải confirmatory analysis.

Do đó:

- báo raw seed deltas;
- báo mean và range;
- có thể báo paired descriptive CI;
- không dùng p-value làm cổng chính;
- không claim significance.

Go/no-go dựa trên:

```text
effect magnitude
direction consistency
real-vs-shuffle behavior
mechanistic activity
```

---

# 10. Compute estimate

Giả định:

```text
reuse H5
backbone frozen
4 adapter arms
3 adapter seeds
mid branch phải backprop qua block 5–12
early stopping 6–8 epochs
RTX PRO 6000 Blackwell
```

## 10.1. GPU-hours

| Thành phần | GPU-hour |
|---|---:|
| Smoke test | 0.25–0.5 |
| Seed 1, four arms | 5–8 |
| Seed 2, four arms | 5–8 |
| Seed 3, four arms | 5–8 |
| Evaluation/diagnostics | 1–2 |
| **Tổng** | **16–26.5 GPU-hours** |

Một GPU:

```text
khoảng 16–27 giờ
```

Hai GPU chạy seed song song:

```text
khoảng 9–15 giờ
```

Estimate phải được cập nhật sau epoch đầu.

## 10.2. VRAM

Khuyến nghị:

```text
64–96 GB
```

Nếu 48 GB:

```text
actual batch 16–24
gradient accumulation để effective batch = 256
gradient checkpointing block 5–12
```

Không giữ graph của bốn arm cùng lúc.

---

# 11. Memory-efficient execution

Chia sẻ forward blocks 1–4:

```python
with torch.no_grad():
    z4 = forward_blocks_1_to_4(images)
```

Sau đó chạy từng arm tuần tự:

```python
for arm in [M_U, M_E, L_U, L_E]:
    arm.optimizer.zero_grad(set_to_none=True)

    logits = arm.forward_from_z4(
        z4=z4,
        ehr=ehr,
    )

    loss = criterion(logits, labels)
    loss.backward()

    clip_grad_norm_(arm.trainable_parameters(), 5.0)
    arm.optimizer.step()
```

Không cộng bốn loss rồi backward chung.

---

# 12. Điều kiện chuyển sang sealed confirmatory

Khuyến nghị strict gate:

```text
G_mid ≥ +0.002
I_location ≥ +0.002
real EHR > shuffled EHR
dấu nhất quán ít nhất 2/3 adapter seed
không có training instability
```

Nếu vượt:

> Chạy sealed confirmatory 5 full-pipeline seed, retrain từ pretrained RAD-DINO sau split mới.

Nếu null:

> Dừng, không xây sealed holdout pipeline.

---

# 13. Mức độ “sạch” của câu trả lời khi screen null

Screen null không phải universal proof rằng mọi deep fusion đều vô dụng.

Nhưng nó cung cấp câu trả lời sạch cho câu hỏi hẹp:

> Với cùng H5, cùng adapter capacity và cùng downstream head, việc chuyển channel-wise EHR conditioning từ block 10–11 sang block 5–7 có tạo gain ít nhất +0.002 hay không?

Nếu:

```text
G_mid < +0.002
I_location < +0.002
real ≈ shuffle
```

thì có thể kết luận thuyết phục:

> Block 5–7 không cứu được channel-wise deep conditioning trong kiến trúc đang xét.

Đây là mức bằng chứng đủ để dừng engineering direction mà không cần chi phí confirmatory 4–6 ngày.

---

# 14. Khuyến nghị cuối cùng

Chấp nhận kế hoạch hai tầng:

## Tầng 1 — khoảng một ngày

```text
reuse existing H5
existing train/validation split
M_U, M_E, L_U, L_E
3 adapter seeds
primary: I_location
safeguard: G_mid
threshold: +0.002
```

## Tầng 2 — chỉ khi vượt gate

```text
sealed subject-disjoint split
retrain từ pretrained RAD-DINO
5 full-pipeline seeds
confirmatory holdout mở một lần
```

Đây là phương án tối ưu giữa:

- độ sạch phương pháp;
- chi phí;
- tốc độ ra quyết định;
- khả năng tránh false negative do chỉ nhìn interaction.
