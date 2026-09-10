# Hướng dẫn cho Claude — Đánh giá và thử nghiệm EHR fusion tại mid-block RAD-DINO

## 0. Kết luận điều hành

Các diagnostic hiện tại tạo ra một lý do **hợp lý hơn trước** để thử EHR conditioning tại block 5–7 thay vì chỉ đặt adapter gần block cuối.

Lý do chính:

- đặc trưng giải phẫu phổi đã bắt đầu xuất hiện rõ ở block 5–7;
- block 5 có attention không gian chọn lọc nhất;
- phần lớn mức tăng của representation hình thành trong 5 block đầu;
- representation gần bão hòa từ khoảng block 9;
- block 11 đạt đỉnh ở linear probe nhưng block 12 giảm;
- conditioning ở block 10–11 gần như không còn nhiều self-attention phía sau để chuyển modulation thành representation hữu ích.

Tuy nhiên, kết luận đúng hiện tại chỉ là:

> Block 5–7 là vị trí can thiệp có cơ sở cơ chế tốt hơn block 10–11 cho một phép thử mới.

Chưa được kết luận:

> Fusion EHR ở block 5–7 chắc chắn sẽ cải thiện mAP.

---

# 1. Kiểm tra tính nhất quán của notebook 09

## 1.1. Bản notebook được tải lên không trích CLS ở block 11

Trong `09_block_select.ipynb`, arm `b11` được định nghĩa là:

```python
p11 = hidden_states[11][:, 1:]
```

Tức là:

```text
patch tokens của block 11
```

không phải:

```text
CLS token của block 11
```

Tương tự:

```python
p12 = hidden_states[12][:, 1:]
p12ln = final_layernorm(hidden_states[12])[:, 1:]
```

Các arm đều được đưa vào ML-Decoder dưới dạng patch-token sequence.

## 1.2. Yêu cầu cho Claude

Trước khi viết report, Claude phải xác minh:

- kết quả `+0.0003` đến từ patch tokens block 11;
- hay một phiên bản notebook khác thực sự dùng CLS block 11.

Nếu kết quả đến từ notebook hiện tại, câu đúng là:

> Dùng patch tokens block 11 thay block-12-final representation chỉ cải thiện khoảng +0.0003.

Không viết:

> Trích CLS từ block 11 cải thiện +0.0003.

Sai khác này không đổi kết luận lớn, nhưng ảnh hưởng mô tả thí nghiệm và khả năng tái lập.

---

# 2. Những gì notebook 08 đã cho thấy

## 2.1. Linear-probe curve

Kết quả diagnostic:

```text
đỉnh linear probe       : block 11, macro mAP khoảng 0.4026
gần bão hòa             : từ block 9
block 10 → 11 → 12      : 0.4000 → 0.4026 → 0.3962
mức tăng hình thành
trong 5 block đầu       : khoảng 73%
```

Diễn giải:

- block 5–7 chưa phải representation cuối cùng;
- nhưng đã chứa phần lớn cấu trúc có ích;
- block 9–11 chủ yếu tinh chỉnh;
- block 12 có thể làm mất một phần thông tin tuyến tính dễ đọc.

## 2.2. Attention distance

Kết quả:

```text
block chọn lọc không gian nhất : block 5
attention distance block 5     : khoảng 44% mức đều
attention distance block 12    : khoảng 88% mức đều
local heads block 10–12        : gần như không còn
```

Điều này cho thấy block cuối có representation đầy đủ nhưng attention đã gần global/uniform.

## 2.3. Ý nghĩa đối với vị trí adapter

Adapter ở block 10–11 gặp ba bất lợi:

1. representation gần bão hòa;
2. attention không gian ít chọn lọc;
3. hầu như không còn đủ block phía sau để chuyển EHR modulation thành feature mới.

Block 5–7 có lợi thế:

1. anatomy đã xuất hiện;
2. attention còn local/selective;
3. còn 5–7 block để biến modulation thành task-specific representation.

---

# 3. Vì sao mid-block fusion có thể tốt hơn late-block fusion

## 3.1. Có điểm tựa giải phẫu

Ở block quá sớm, model chủ yếu học:

```text
edge
texture
contrast
local intensity
```

EHR conditioning ở đó có nguy cơ làm méo low-level visual representation.

Ở block 5–7, representation đã tổ chức thành:

```text
lung fields
mediastinum
pleural boundary
large anatomical regions
```

Đây là giai đoạn hợp lý để clinical context thay đổi cách model diễn giải visual pattern.

## 3.2. Còn đủ computational depth phía sau

Nếu adapter đặt sau block 5 hoặc block 7, các block sau vẫn có thể:

- tái phân phối self-attention;
- kết hợp patch xa nhau;
- chuyển channel modulation thành patch-specific effect;
- tạo feature phù hợp với ML-Decoder.

Nếu adapter đặt sau block 11, gần như chỉ còn block 12 và final norm. Khả năng “điều hướng” representation rất hạn chế.

## 3.3. Block 12 giảm mAP không có nghĩa phải bỏ block 12

Kết quả block 11 chỉ hơn block-12 representation khoảng +0.0003 trong thí nghiệm decoder thực tế.

Do đó, quyết định giữ output architecture cũ là hợp lý.

Điều cần thay đổi không phải:

```text
decoder đọc block nào
```

mà là:

```text
EHR được phép tác động ở đâu trước khi representation đi tới block 12
```

---

# 4. Vì sao bằng chứng hiện tại chưa đủ để kết luận mid-block sẽ thắng

## 4.1. PCA-RGB là visualization, không phải causal evidence

PCA-RGB cho thấy các patch có representation tương đồng, nhưng:

- màu PCA có thể đổi giữa ảnh hoặc block;
- hình phổi xuất hiện không chứng minh feature đó tối ưu cho nhãn;
- anatomy organization khác với clinical complementarity.

## 4.2. CLS attention không phải feature attribution hoàn chỉnh

Attention map cho biết một cơ chế routing, nhưng không chứng minh:

- vùng sáng là vùng quyết định logit;
- thay đổi attention sẽ cải thiện AP;
- EHR có signal cần thiết để điều chỉnh vùng đó.

## 4.3. Patch model hiện đã rất mạnh

Kết quả 06b cho thấy:

```text
H5 patch baseline          ≈ 0.4336
H6 patch + EHR token gain  ≈ +0.0003
real-vs-shuffle EHR        ≈ 0
post-hoc stacking          < 0
```

Do đó prior cho EHR gain vẫn thấp.

Mid-block experiment chỉ hợp lý vì nó kiểm tra một cơ chế chưa được thử trực tiếp:

\[
f(CXR\mid EHR)
\]

chứ không phải vì late-fusion null đã bị phủ định.

---

# 5. Giả thuyết mới

## 5.1. Giả thuyết chính

> Patient-specific EHR conditioning tại block 5 và block 7 tạo EHR gain lớn hơn conditioning tại block 10 và block 11.

## 5.2. Primary quantities

Mid-block EHR gain:

\[
G_{\text{mid}}=mAP(M_E)-mAP(M_U)
\]

Late-block EHR gain:

\[
G_{\text{late}}=mAP(L_E)-mAP(L_U)
\]

Location interaction:

\[
I_{\text{location}}
=
G_{\text{mid}}-G_{\text{late}}
\]

Trong đó:

| Arm | Vị trí | Context |
|---|---|---|
| `M_U` | block 5 và 7 | learned constant |
| `M_E` | block 5 và 7 | patient EHR |
| `L_U` | block 10 và 11 | learned constant |
| `L_E` | block 10 và 11 | patient EHR |

Block numbering phải dùng **1-indexed** trong report.

---

# 6. Vì sao chọn block 5 và 7

Không chèn adapter vào cả block 5, 6 và 7 ở phép thử đầu.

Chọn hai vị trí:

```text
block 5
block 7
```

Lý do:

- block 5 là vị trí attention chọn lọc nhất;
- block 7 nằm trong giai đoạn lung anatomy đã rõ hơn;
- dùng hai adapter giống late control có block 10 và 11;
- giữ số adapter và parameter gần tương đương;
- giảm confound do capacity.

Nếu dùng ba mid adapters nhưng chỉ hai late adapters, location effect bị lẫn với parameter count.

---

# 7. Adapter cho phép thử đầu

## 7.1. Channel-wise conditional residual adapter

\[
u_l=W_{\text{down},l}\operatorname{LN}(Z_l)
\]

\[
\gamma_l(e),\beta_l(e)=G_l(e)
\]

\[
\tilde u_l=(1+\gamma_l(e))\odot u_l+\beta_l(e)
\]

\[
Z_l'=Z_l+W_{\text{up},l}\operatorname{GELU}(\tilde u_l)
\]

Zero-init:

```python
nn.init.zeros_(W_up.weight)
nn.init.zeros_(W_up.bias)
```

## 7.2. Warm-start bắt buộc

Tất cả arm phải warm-start từ cùng new H5 checkpoint trong từng full-pipeline seed:

```text
same backbone
same block weights
same final layernorm
same ML-Decoder
```

Tại step 0:

```python
logits_B == logits_M_U == logits_M_E == logits_L_U == logits_L_E
```

trong `eval()` và FP32.

---

# 8. Giới hạn của channel-wise modulation

\(\gamma,\beta\) được broadcast cho mọi patch:

\[
\gamma_{b,c},\beta_{b,c}
\]

nên không trực tiếp nói:

> tăng patch đáy phổi phải.

Nó chỉ nói:

> với bệnh nhân này, tăng hoặc giảm một nhóm semantic channel.

Tuy vậy, ở block 5–7 vẫn còn nhiều self-attention block phía sau. Vì mỗi patch có content khác nhau, channel recalibration có thể gián tiếp tạo spatial effect.

Nếu kết quả null, kết luận chỉ là:

> Channel-wise EHR conditioning tại mid-block không cải thiện tác vụ.

Không được kết luận mọi spatial EHR conditioning đều vô dụng.

---

# 9. Không dùng spatial gate ngay trong confirmatory test đầu

Một spatial gate mạnh hơn có thể là:

\[
g_{b,i}
=
\sigma\left(
w^\top\tanh(W_zZ_{b,i}+W_ee_b)
\right)
\]

\[
Z'_{b,i}
=
Z_{b,i}+\alpha g_{b,i}U(e_b)
\]

Cơ chế này trực tiếp phụ thuộc:

- patch \(i\);
- EHR của bệnh nhân.

Tuy nhiên, nó:

- có capacity lớn hơn;
- tạo thêm confound;
- khó parameter-match;
- khó diễn giải nếu gain nhỏ.

Do đó:

1. chạy channel-wise adapter trước;
2. spatial gate chỉ là experiment thứ hai nếu mid-block channel conditioner cho mechanistic signal rõ.

---

# 10. Confirmatory design

## 10.1. Split

Dùng sealed subject-disjoint confirmatory split đã thống nhất:

```text
C_train
C_val
C_holdout
```

Bắt đầu lại từ pretrained RAD-DINO trước supervised fine-tuning.

Không dùng checkpoint cũ vì proposed holdout từng nằm trong original TRAIN.

## 10.2. Full-pipeline seed

Mỗi seed:

```text
pretrained RAD-DINO
→ new 04b trên C_train
→ new H5 trên C_train
→ warm-start B/M_U/M_E/L_U/L_E
→ train trên C_train
→ select bằng C_val
→ mở C_holdout một lần
```

## 10.3. Primary claim options

### Phương án A — xác nhận EHR mid-block có ích

Primary:

\[
G_{\text{mid}}=M_E-M_U
\]

Late arms dùng như secondary mechanistic comparison.

### Phương án B — xác nhận mid-block tốt hơn late-block

Primary:

\[
I_{\text{location}}
=
(M_E-M_U)-(L_E-L_U)
\]

Nếu muốn claim trực tiếp:

> block 5–7 tốt hơn block 10–11,

phải chọn phương án B và giữ cả bốn arm trên holdout.

Khuyến nghị chọn **phương án B**, vì đây là câu hỏi mới được đặt ra từ notebook 08.

---

# 11. Development gate trước holdout

Dùng `C_val`, không dùng `C_holdout`.

Sau 3 seed development:

| Kết quả | Hành động |
|---|---|
| `G_mid ≤ 0` và không có real-vs-shuffle effect | dừng |
| `G_mid > 0` nhưng `< +0.002` | dừng vì dưới MPRE |
| `G_mid ≥ +0.002`, dấu nhất quán | hoàn tất 5 full seed |
| `I_location ≥ +0.002` | có cơ sở claim mid ưu thế |
| representation thay đổi nhưng logits không đổi | không mở rộng spatial gate ngay |

Mọi rule phải khóa trước run.

---

# 12. Diagnostics bắt buộc

## 12.1. Real, shuffle và zero EHR

Chạy cho `M_E` và `L_E`:

```text
real EHR
study-level shuffled EHR
zero EHR
median EHR
```

## 12.2. Representation shift theo độ sâu

Đo:

\[
d_l=
\|Z_l^{real}-Z_l^{shuffle}\|_2
\]

tại:

```text
sau block 5
sau block 7
sau block 9
sau block 11
sau block 12
```

## 12.3. Patch-wise shift map

Với mỗi patch:

\[
d_{l,i}
=
\|Z_{l,i}^{real}-Z_{l,i}^{shuffle}\|_2
\]

Vẽ heatmap 37×37.

Đây là diagnostic trực tiếp hơn PCA-RGB để xem EHR tạo spatially heterogeneous change hay không.

## 12.4. Adapter magnitude

Log:

```text
||ΔZ_l|| / ||Z_l||
gamma mean/std
beta mean/std
gradient norm
```

## 12.5. Logit sensitivity

```text
mean |logit_real − logit_shuffle|
corr(logit_real, logit_shuffle)
per-label logit delta
macro mAP delta
```

Attention map chỉ là secondary visualization.

---

# 13. Compute implications

Mid-block conditioning đắt hơn late-block conditioning.

Lý do:

- adapter ở block 5 làm graph phải đi qua block 6–12;
- dù các downstream weight frozen, autograd vẫn phải truyền gradient qua chúng để cập nhật adapter;
- D1 và D2 có branch-specific representation nên không dùng chung toàn bộ forward sau block 5.

## 13.1. Selected mid-only confirmatory

Ước tính mỗi full-pipeline seed:

| Thành phần | GPU-hour |
|---|---:|
| new 04b | 9–12 |
| new H5 | 1.5–2.5 |
| mid D0/M_U/M_E | 5–8 |
| validation + export | 0.5–1 |
| **tổng mỗi seed** | **16–23.5** |

Năm seed:

```text
80–118 GPU-hours
```

Một GPU:

```text
khoảng 3.5–5 ngày
```

## 13.2. Mid-vs-late confirmatory đầy đủ

Nếu giữ cả `M_U/M_E/L_U/L_E`:

| Thành phần | GPU-hour mỗi seed |
|---|---:|
| new 04b | 9–12 |
| new H5 | 1.5–2.5 |
| four adapter arms | 8–12 |
| validation + export | 0.5–1 |
| **tổng** | **19–27.5** |

Năm seed:

```text
95–138 GPU-hours
```

Một GPU:

```text
khoảng 4–6 ngày liên tục
```

Hai GPU chạy theo seed:

```text
khoảng 2–3 ngày
```

## 13.3. Optional partial unfreeze

Nếu unfreeze block 5–7 sau Stage A:

```text
additional 7–10 GPU-hours mỗi seed
additional 35–50 GPU-hours cho 5 seed
```

Không unfreeze trước khi adapter-only experiment cho signal dương.

---

# 14. Memory strategy

Không giữ graph của mọi arm cùng lúc.

Pseudo-code:

```python
with torch.no_grad():
    z4 = forward_blocks_1_to_4(images)

for arm in active_arms:
    optimizer[arm].zero_grad(set_to_none=True)
    logits = forward_branch_from_block5(z4, arm, ehr)
    loss = criterion(logits, y)
    loss.backward()
    clip_grad_norm_(arm.parameters(), 5.0)
    optimizer[arm].step()
```

Ưu điểm:

- blocks 1–4 chỉ forward một lần;
- mỗi arm backward tuần tự;
- VRAM thấp hơn;
- optimizer và clipping độc lập.

Dùng gradient checkpointing cho block 5–12 nếu cần.

Khuyến nghị:

```text
VRAM: 64–96 GB
actual batch: 16–32 cho full four-arm run
effective batch: 256 qua accumulation
```

Auto-probe trước khi khóa.

---

# 15. Go/no-go rule

## Mid EHR effect

| `G_mid` | Quyết định |
|---|---|
| `< +0.002` hoặc CI chứa 0 | dừng |
| `+0.002 đến +0.005` và real > shuffle | effect nhỏ, cần cân nhắc |
| `> +0.005` và nhất quán seed | tiếp tục Stage B |

## Location interaction

| `I_location` | Diễn giải |
|---|---|
| `≤ 0` | mid-block không tốt hơn late-block |
| `0 đến +0.002` | không có ưu thế thực tiễn |
| `> +0.002`, CI loại 0 | bằng chứng mid-block tốt hơn |
| chỉ dương ở một label post-hoc | không claim |

---

# 16. Kết luận được phép sau thí nghiệm

## Nếu mid thắng

Có thể viết:

> Patient-specific channel conditioning có hiệu quả hơn khi được đặt tại giai đoạn representation còn local và khả biến, thay vì gần output nơi representation đã bão hòa.

Không viết:

> EHR đã học đúng vùng bệnh lý,

trừ khi patch-wise real-vs-shuffle shift và causal output effect cùng hỗ trợ.

## Nếu mid vẫn null

Có thể viết:

> Việc chuyển channel-wise EHR conditioning từ late blocks sang mid blocks không phục hồi EHR gain trên strong patch representation.

Không viết:

> Mọi loại EHR fusion đều vô dụng.

---

# 17. Khuyến nghị cuối cùng

Có, block 5–7 là vị trí đáng thử hơn block 10–11 dựa trên diagnostic hiện tại.

Thiết kế khuyến nghị:

```text
mid adapters: block 5 và 7
late controls: block 10 và 11
same adapter count
same bottleneck
same new H5 warm-start
zero-init
unconditional control cho mỗi vị trí
5 full-pipeline seed
sealed holdout
```

Primary estimand nên là:

\[
I_{\text{location}}
=
(M_E-M_U)-(L_E-L_U)
\]

Đây là phép thử trực tiếp và thuyết phục nhất cho câu hỏi:

> EHR có hiệu quả hơn khi can thiệp vào giai đoạn model đang hình thành representation giải phẫu, thay vì khi representation đã gần hoàn tất hay không?
