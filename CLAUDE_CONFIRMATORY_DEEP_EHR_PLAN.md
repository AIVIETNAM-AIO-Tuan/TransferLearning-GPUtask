# Hướng dẫn cho Claude — Confirmatory Deep EHR Conditioning trong RAD-DINO

## 0. Mục tiêu và quyết định nghiên cứu

Thực hiện một thí nghiệm confirmatory có kiểm soát để kiểm tra:

> EHR-conditioned adapters đặt trong các block sâu của RAD-DINO có cải thiện phân loại 13 nhãn CheXpert so với unconditional adapters có cùng capacity hay không?

Primary estimand:

\[
\Delta_{\text{deep EHR}} = mAP(D2)-mAP(D1)
\]

Trong đó:

- **D1**: unconditional adapter, parameter-matched control.
- **D2**: patient-specific EHR-conditioned adapter.

Không dùng `D2 − D0` làm estimand chính vì nó trộn EHR effect với adapter capacity và continued training.

---

# 1. Mức độ confirmatory và giới hạn phải ghi rõ

## 1.1. Confirmatory nội bộ mạnh nhất có thể

Thiết kế phải:

1. tạo một subject-disjoint holdout mới;
2. loại holdout khỏi toàn bộ supervised retraining;
3. bỏ toàn bộ checkpoint 04b/H5 cũ;
4. bắt đầu lại từ pretrained RAD-DINO trước supervised fine-tuning;
5. khóa kiến trúc, hyperparameter, số seed và analysis rule trước khi mở holdout;
6. chỉ đánh giá holdout một lần sau khi toàn bộ quyết định đã cố định.

Cách gọi phù hợp:

> **Internal confirmatory rerun with a sealed, model-isolated holdout.**

Không gọi là external confirmation hoặc fully prospective confirmation.

## 1.2. Caveat lịch sử

Các subject trong original TRAIN từng tham gia các exploratory model cũ. Vì vậy holdout mới:

- không được dùng bởi pipeline confirmatory mới;
- nhưng không hoàn toàn “chưa từng xuất hiện trong toàn bộ lịch sử nghiên cứu”.

Strict confirmation ở cấp dữ liệu chỉ đạt được bằng cohort mới hoặc external cohort.

---

# 2. Split confirmatory mới

## 2.1. Nguồn dữ liệu

Chỉ dùng original TRAIN hiện có, khoảng:

```text
147,791 studies
```

Tạo ba split mới ở mức `subject_id`:

| Split | Target study count | Vai trò |
|---|---:|---|
| `C_train` | khoảng 117,791 | supervised training |
| `C_val` | khoảng 15,000 | early stopping và model selection |
| `C_holdout` | khoảng 15,000 | confirmatory evaluation duy nhất |

Target theo study count nhưng split bắt buộc ở mức subject.

## 2.2. Phương pháp split

Dùng subject-level iterative multilabel stratification.

Yêu cầu:

```python
assert subjects(C_train).isdisjoint(subjects(C_val))
assert subjects(C_train).isdisjoint(subjects(C_holdout))
assert subjects(C_val).isdisjoint(subjects(C_holdout))
```

Kiểm tra trước training:

- prevalence từng nhãn;
- setting distribution;
- frontal/lateral distribution;
- số study trên mỗi subject;
- lab availability;
- rare-label positive count.

Không điều chỉnh split sau khi xem performance.

## 2.3. Niêm phong holdout

`C_holdout` không được dùng để:

- tune LR;
- chọn epoch;
- chọn adapter bottleneck;
- chọn conditioning block;
- chọn EHR feature;
- quyết định số seed;
- chọn label để báo cáo;
- debug model performance.

Tạo:

```text
confirmatory_split_manifest.json
```

Nội dung tối thiểu:

```json
{
  "split_seed": 20260720,
  "split_method": "subject-level iterative multilabel stratification",
  "counts": {
    "C_train": 117791,
    "C_val": 15000,
    "C_holdout": 15000
  },
  "holdout_hash": "...",
  "created_before_training": true
}
```

---

# 3. Không được sử dụng checkpoint cũ

Cấm warm-start confirmatory pipeline từ:

```text
old 04b checkpoint
old H5 checkpoint
old H6 checkpoint
```

Lý do:

- các checkpoint cũ đã train trên original TRAIN;
- proposed holdout được lấy từ original TRAIN;
- dùng checkpoint cũ tạo leakage qua backbone và decoder.

Confirmatory pipeline phải bắt đầu từ:

```text
original pretrained RAD-DINO checkpoint
trước mọi supervised fine-tuning trên 13 nhãn
```

---

# 4. Full retraining pipeline

Mỗi full-pipeline seed gồm:

```text
pretrained RAD-DINO
    ↓
new 04b supervised LP-FT trên C_train
    ↓
new frozen-patch H5 + ML-Decoder + ASL
    ↓
warm-start D0/D1/D2 từ new H5
    ↓
train D1/D2 trên C_train
    ↓
select checkpoint trên C_val
    ↓
evaluate một lần trên C_holdout
```

Để uncertainty phản ánh toàn bộ pipeline, mỗi seed phải retrain:

- new 04b;
- new H5;
- D1;
- D2.

Không dùng một H5 duy nhất cho tất cả seed trong confirmatory analysis chính.

---

# 5. Warm-start ML-Decoder là bắt buộc

## 5.1. Lý do

Zero-init adapter chỉ đảm bảo:

\[
\Delta Z_{\text{adapter}}=0
\]

Nếu D1/D2 có decoder random nhưng D0 dùng H5 decoder đã train:

\[
Decoder_{D1}\neq Decoder_{H5}
\]

nên logits không thể bằng nhau dù adapter đúng.

## 5.2. Thiết kế đúng

Với mỗi seed \(s\):

```python
h5 = load_new_h5_checkpoint(seed=s)

D0.load_backbone(h5.backbone)
D0.load_decoder(h5.decoder)

D1.load_backbone(h5.backbone)
D1.load_decoder(h5.decoder)

D2.load_backbone(h5.backbone)
D2.load_decoder(h5.decoder)

zero_init(D1.adapters)
zero_init(D2.adapters)
```

Block 11–12 của D0/D1/D2 phải khởi tạo từ cùng H5 state.

## 5.3. Identity assertion

Chạy trong:

```text
eval mode
FP32
dropout disabled
same batch
```

```python
with torch.no_grad(), torch.autocast("cuda", enabled=False):
    y0 = D0(batch.float())
    y1 = D1(batch.float())
    y2 = D2(batch.float())

assert torch.allclose(y0, y1, atol=1e-6, rtol=1e-6)
assert torch.allclose(y0, y2, atol=1e-6, rtol=1e-6)
```

Nếu kernel nondeterministic, cho phép:

```text
atol = 1e-5
rtol = 1e-5
```

nhưng phải log `max_abs_diff`.

Đây là identity thật, tốt hơn H6 token insertion vì token rỗng vẫn thay đổi softmax denominator.

---

# 6. Model definitions

## 6.1. D0-reference

```text
new H5 checkpoint
không train thêm
```

Vai trò:

- identity reference;
- regression anchor;
- historical comparison.

## 6.2. D0-retrain

Khuyến nghị thêm:

```text
new H5 warm-start
continued training cùng schedule
không adapter
```

Dùng để tách:

\[
D0_{\text{retrain}}-D0_{\text{reference}}
\]

continued-training effect khỏi adapter effect.

## 6.3. D1 — unconditional adapter

D1 dùng cùng architecture với D2 nhưng context không phụ thuộc bệnh nhân.

Ví dụ:

```python
e_const = nn.Parameter(torch.zeros(1, 128))
```

sau đó chạy qua cùng generator với D2.

Yêu cầu:

- cùng adapter bottleneck;
- cùng generator architecture;
- cùng decoder;
- cùng block;
- cùng optimizer groups;
- cùng LR;
- trainable parameter difference dưới 5%.

## 6.4. D2 — EHR-conditioned adapter

Chỉ dùng feature đã có bằng chứng mạnh nhất:

```text
7 labs
7 lab masks
setting chỉ khi preregistered
```

Không dùng vòng đầu:

```text
microbiology
vitals
comorbidity
```

---

# 7. Adapter architecture

## 7.1. Minimal channel-wise conditioner

Với:

\[
Z_l\in\mathbb{R}^{B\times N\times768}
\]

và EHR embedding:

\[
e\in\mathbb{R}^{B\times128}
\]

Adapter:

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
\Delta Z_l=W_{\text{up},l}\operatorname{GELU}(\tilde u_l)
\]

\[
Z_l'=Z_l+\Delta Z_l
\]

Bottleneck phải khóa trước holdout:

```text
64 hoặc 128
```

## 7.2. Identity initialization

Khuyến nghị:

```python
nn.init.zeros_(W_up.weight)
nn.init.zeros_(W_up.bias)
```

Khi đó:

\[
\Delta Z_l=0
\]

tại step 0.

Không zero-init toàn bộ upstream path theo cách làm gradient chết.

## 7.3. Giới hạn của broadcast gamma/beta

\(\gamma,\beta\) broadcast trên token dimension:

\[
\gamma_{b,c},\beta_{b,c}
\]

không phụ thuộc patch \(i\).

Kiến trúc này trực tiếp kiểm tra:

> patient-specific channel recalibration trong semantic blocks.

Nó không trực tiếp kiểm tra:

> patient-specific spatial highlighting của từng patch.

Spatial effect chỉ có thể xuất hiện gián tiếp qua self-attention của block tiếp theo.

Nếu D2 không hơn D1, kết luận hợp lệ là:

> Channel-wise EHR conditioning ở hai block cuối không cải thiện tác vụ.

Không được kết luận:

> Mọi hình thức deep multimodal fusion đều vô dụng.

---

# 8. Vị trí conditioning

Bắt đầu với:

```text
blocks 1–10: frozen
block 11: adapter
block 12: adapter
ML-Decoder
```

Không condition block đầu vì:

- semantic context hợp lý hơn ở block cuối;
- giảm nguy cơ làm méo edge/texture feature;
- giảm compute;
- giữ phép thử tối thiểu và diễn giải được.

---

# 9. Training protocol

## 9.1. New 04b retraining

Mỗi full seed:

```text
LP  : 4 epochs
FT  : tối đa 12 epochs
selection: C_val study-level macro mAP
```

Bắt đầu từ original pretrained RAD-DINO.

## 9.2. New H5 training

```text
backbone: new 04b checkpoint, frozen
head: ML-Decoder trên patch tokens
loss: ASL
selection: C_val study-level macro mAP
```

New H5 là nguồn warm-start duy nhất của D0/D1/D2 trong cùng seed.

## 9.3. Stage A — adapter-only

Train:

```text
D0-retrain decoder, nếu dùng
D1 adapter + decoder
D2 EHR encoder + adapter + decoder
```

Freeze:

```text
toàn bộ RAD-DINO blocks
```

Config ban đầu:

```text
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
loss          = ASL
```

## 9.4. Stage B — partial unfreeze

Chỉ chạy nếu Stage-A rule cho phép.

Unfreeze:

```text
block 11
block 12
```

LR:

```text
block 11–12 : 1e-6 đến 5e-6
adapter     : 1e-4
EHR encoder : 1e-4
decoder     : 1e-4
```

D1 và D2 phải:

- unfreeze cùng block;
- dùng cùng initialization;
- dùng optimizer/scheduler riêng;
- clip gradient riêng.

---

# 10. Seed, MPRE và power

## 10.1. Minimum practically relevant effect

Khóa:

```text
MPRE = +0.002 macro mAP
```

Nghĩa là gain nhỏ hơn +0.002 không đáng tiếp tục đầu tư.

MPRE không phải minimum detectable effect.

## 10.2. Designed detectable effect

Với variance quan sát ở 06b, thiết kế 5 seed nhiều khả năng nhạy hơn với:

```text
effect khoảng +0.004 trở lên
```

Ghi rõ:

```text
MPRE ≈ +0.002
Designed MDE ≈ +0.004
```

## 10.3. Full-pipeline seeds

Confirmatory chính:

```text
5 full-pipeline seeds
```

Mỗi seed retrain full stack.

## 10.4. Predefined expansion rule

Quyết định mở rộng lên 10 seed chỉ dựa trên `C_val`, trước khi mở `C_holdout`.

| Validation result sau 5 seed | Hành động |
|---|---|
| mean `D2−D1 < +0.001`, dấu không nhất quán | giữ 5 seed, dự kiến no-go |
| mean `+0.001 đến +0.004` | mở rộng lên 10 full seeds |
| mean `> +0.004`, dấu nhất quán | 5 seed có thể đủ |
| instability hoặc failed runs | sửa implementation, không mở holdout |

Rule phải ghi vào manifest trước run đầu.

---

# 11. Confirmatory evaluation

Chỉ mở `C_holdout` sau khi:

- architecture khóa;
- LR khóa;
- số seed khóa;
- Stage A/B decision khóa;
- checkpoint selection hoàn tất trên C_val.

Primary per-seed delta:

\[
\Delta_s=mAP(D2_s)-mAP(D1_s)
\]

Báo cáo:

```text
raw paired deltas
mean delta
t-based CI95 qua full-pipeline seed
paired p-value
subject-level cluster bootstrap
```

Bootstrap theo subject là bổ sung, không thay thế seed variance.

Không claim per-label findings post-hoc.

---

# 12. Causal diagnostics

Chạy trên final D2:

```text
real EHR
shuffle EHR
zero EHR
median EHR
```

Shuffle:

- ở mức study;
- mọi image cùng study nhận cùng shuffled EHR;
- trong cùng split;
- ưu tiên cùng setting.

Metrics:

```text
macro mAP delta
per-label AP delta
mean absolute logit delta
corr(real logits, shuffled logits)
```

Primary evidence vẫn là:

\[
D2-D1
\]

---

# 13. Compute assumptions

Ước tính dựa trên:

```text
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
C_train images: khoảng 190k–200k
C_val images: khoảng 20k–30k
C_holdout images: khoảng 20k–30k
input: 518×518
AMP: bf16
04b empirical throughput: khoảng 157 image/s
frozen forward estimate: khoảng 400 image/s
```

Đây là planning range, không phải guarantee. Sau dry run một epoch phải cập nhật lại.

---

# 14. GPU-hour estimate

## 14.1. Per full-pipeline seed — Stage A

| Thành phần | GPU-hour ước tính |
|---|---:|
| New 04b LP-FT trên C_train | 9–12 |
| New H5 frozen-patch + ASL | 1.5–2.5 |
| D0-retrain/D1/D2 Stage A | 3–5 |
| Validation, diagnostics, export | 0.5–1 |
| **Tổng mỗi seed** | **14–20** |

## 14.2. Năm full seeds

```text
70–100 GPU-hours
```

## 14.3. Stage B optional

Partial unfreeze block 11–12:

```text
4–6 GPU-hours mỗi seed
```

Năm seed:

```text
additional 20–30 GPU-hours
```

Tổng Stage A+B:

```text
90–130 GPU-hours
```

## 14.4. Nếu mở rộng lên 10 full seeds

Stage A:

```text
140–200 GPU-hours
```

Stage A+B:

```text
180–260 GPU-hours
```

---

# 15. Wall-clock estimate

## Một GPU

| Phạm vi | Wall-clock |
|---|---:|
| 5 seed, Stage A | khoảng 3–4.5 ngày chạy liên tục |
| 5 seed, Stage A+B | khoảng 4–5.5 ngày |
| 10 seed, Stage A | khoảng 6–9 ngày |
| 10 seed, Stage A+B | khoảng 8–11 ngày |

## Hai GPU

| Phạm vi | Wall-clock |
|---|---:|
| 5 seed, Stage A | khoảng 1.5–2.5 ngày |
| 5 seed, Stage A+B | khoảng 2–3 ngày |

## Năm GPU

Mỗi GPU chạy một full seed độc lập:

| Phạm vi | Wall-clock |
|---|---:|
| 5 seed, Stage A | khoảng 14–20 giờ |
| 5 seed, Stage A+B | khoảng 18–26 giờ |

Parallel theo seed dễ kiểm soát hơn DDP cho một seed.

---

# 16. GPU memory và batch

Khuyến nghị:

```text
VRAM tối thiểu thực tế: 48 GB
VRAM khuyến nghị: 64–96 GB
```

Trên RTX PRO 6000 Blackwell:

```text
actual batch dự kiến: 48–64
effective batch: 256 qua gradient accumulation
```

Auto-probe:

```text
B = 24 → 32 → 48 → 64 → 96
```

Chọn batch lớn nhất còn:

```text
15–20% VRAM headroom
```

Nếu VRAM 48 GB:

```text
actual batch 24–32
gradient accumulation 8–11 bước
```

---

# 17. CPU, RAM, I/O và storage

## CPU

```text
minimum: 16 cores
recommended: 24–32 cores
```

## RAM

```text
minimum: 64 GB
recommended: 128 GB
```

## Local storage

Existing compressed image shards:

```text
khoảng 16.9 GB
```

Khuyến nghị NVMe scratch:

```text
100–150 GB
sequential read tốt
ít nhất khoảng 1 GB/s thực tế
```

Không cache block-10 hoặc full patch tokens.

## Checkpoint storage

Nếu chỉ lưu:

- best;
- last;
- trainable deltas;
- optimizer state tối thiểu;

thì 5-seed confirmatory run cần khoảng:

```text
15–30 GB artifacts
```

Không lưu full backbone cho mọi epoch.

---

# 18. Go/no-go rule

## Stage A

| Kết quả `D2 − D1` | Quyết định |
|---|---|
| `< +0.002`, seed-CI chứa 0 | dừng deep EHR fusion |
| `+0.002 đến +0.005`, nhất quán seed và real > shuffle | chạy Stage B |
| `> +0.005` | chạy Stage B và xem xét mở rộng có giới hạn |
| gain chỉ ở 1–2 nhãn post-hoc | không claim |

## Stage B

| Kết quả | Quyết định |
|---|---|
| `< +0.002` | EHR conditioning không phải hướng ưu tiên |
| `+0.002 đến +0.005` | cần external validation |
| `> +0.005` | có cơ sở nghiên cứu thêm |

Không thay threshold sau khi mở holdout.

---

# 19. Những hướng không được làm trong vòng này

Không:

- đổi sang ConvNeXt V2;
- thêm grouped EHR tokens;
- thêm EHR cross-attention lớn;
- mở rộng feature EHR;
- đổi loss khỏi ASL;
- đổi split sau khi training;
- chọn label sau khi xem holdout;
- unfreeze toàn bộ backbone;
- dùng holdout để debug;
- claim spatial conditioning nếu chỉ dùng broadcast gamma/beta.

---

# 20. Deliverables Claude phải tạo

```text
CONFIRMATORY_SPLIT_PLAN.md
PLAN_confirmatory_deep_ehr_conditioning.md
07_confirmatory_deep_ehr_conditioning.ipynb
REPORT_confirmatory_deep_ehr.md
```

Notebook:

```text
C0  preregistration và thresholds
C1  create/seal confirmatory split
C2  load pretrained RAD-DINO
C3  retrain new 04b
C4  train new H5
C5  construct D0/D1/D2
C6  warm-start decoder
C7  identity and gradient gates
C8  Stage-A training
C9  validation-only seed decision
C10 optional Stage-B partial unfreeze
C11 seal final checkpoints
C12 open holdout once
C13 real/zero/shuffle diagnostics
C14 paired seed statistics
C15 subject bootstrap
C16 export final report
```

Outputs:

```text
confirmatory_deep_ehr/
├── split/
│   ├── confirmatory_split_manifest.json
│   └── holdout_hash.txt
├── checkpoints/
├── predictions/
├── metrics/
│   ├── per_seed.csv
│   ├── per_label.csv
│   ├── interventions.csv
│   ├── subject_bootstrap.csv
│   └── compute_profile.csv
└── experiment_manifest.json
```

---

# 21. Kết luận hướng dẫn

Claude phải giữ lập trường:

> 06b bác bỏ mạnh late EHR fusion trên frozen patch representation, nhưng chưa trực tiếp bác bỏ patient-specific channel conditioning bên trong RAD-DINO.

Phép thử confirmatory này phải:

- retrain toàn bộ supervised pipeline sau split mới;
- warm-start D1/D2 từ new H5 cùng seed;
- identity-init adapter;
- dùng parameter-matched D1;
- khóa rules trước holdout;
- báo cáo full-pipeline seed variance;
- dừng nếu gain dưới MPRE.

Nếu:

\[
D2-D1<+0.002
\]

hoặc seed-CI chứa 0, kết luận:

> Channel-wise deep EHR conditioning không phải hướng ưu tiên cho 13 nhãn CheXpert trong cohort này.
