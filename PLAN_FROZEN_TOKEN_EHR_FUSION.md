# Kế hoạch thí nghiệm: Frozen CXR Representation × EHR Fusion

## 1. Mục tiêu nghiên cứu

Chạy bốn pipeline:

| ID | Image representation | EHR | Head | Backbone |
|---|---|---|---|---|
| **P1** | CLS | Không | MLP | Frozen |
| **P2** | CLS | Concat | MLP | Frozen |
| **P3** | Patch tokens | Không | ML-Decoder | Frozen |
| **P4** | Patch tokens | EHR token | ML-Decoder | Frozen |

Hai phép so sánh chính:

\[
\Delta_{\text{EHR, CLS}} = P2 - P1
\]

\[
\Delta_{\text{EHR, Patch}} = P4 - P3
\]

Interaction:

\[
I = (P4 - P3) - (P2 - P1)
\]

Interaction này trả lời trực tiếp:

> EHR bổ sung tín hiệu mới trên một image representation mạnh, hay chủ yếu bù lại thông tin đã mất khi ảnh bị nén thành CLS?

Notebook 06 hiện dùng frozen `x_cxr` ở mức study và chỉ huấn luyện EHR encoder, fusion module và head. Vì vậy P1–P2 gần như đã có sẵn, còn P3–P4 cần một image-level pipeline mới.

---

## 2. Nguyên tắc để bốn pipeline có thể so sánh

### 2.1. Dùng cùng một backbone checkpoint

Cả bốn pipeline sử dụng cùng:

```text
RAD-DINO ViT-B/14
checkpoint: backbone sau bước 04b
input: 518 × 518
requires_grad = False
model.eval()
```

Không dùng:

- checkpoint khác nhau giữa CLS và patch;
- backbone pretrained nguyên bản cho một nhánh và backbone fine-tuned cho nhánh khác;
- unfreeze bất kỳ block nào trong bốn thí nghiệm này.

Pipeline hiện tại có RAD-DINO với 1.369 patch token và một CLS token 768 chiều. Bước 04b dùng ML-Decoder trên patch token, còn bước 05 trích CLS rồi pool thành một vector study-level.

### 2.2. Dùng cùng loss

Dùng **ASL** cho cả bốn pipeline:

```python
gamma_neg = 4
gamma_pos = 1
clip = 0.05
```

Không lấy lại kết quả `0.4199` của 04b làm P3 vì số đó được tạo trong một training protocol khác. P3 phải được chạy lại với:

- cùng frozen checkpoint;
- ASL;
- cùng split;
- cùng image preprocessing;
- cùng multi-view evaluation;
- cùng checkpoint-selection rule như P4.

### 2.3. Train ở mức ảnh, đánh giá ở mức study

Mỗi sample trong DataLoader là:

```python
{
    "image": image_tensor,
    "study_id": study_id,
    "subject_id": subject_id,
    "view": frontal_or_lateral,
    "ehr_row": study_level_ehr,
    "y": labels
}
```

EHR và labels được lặp lại cho các ảnh thuộc cùng study.

Sau khi model tạo probability cho từng ảnh, pool về study:

\[
p_{\text{study}} =
\frac{
0.7 \cdot \operatorname{mean}(p_{\text{frontal}})
+
0.3 \cdot \operatorname{mean}(p_{\text{lateral}})
}{
0.7[\exists \text{frontal}]
+
0.3[\exists \text{lateral}]
}
\]

Mẫu số phải renormalize vì nhiều study chỉ có frontal.

Nếu P1–P2 tiếp tục dùng study-level CLS còn P3–P4 dùng image-level prediction, ta vẫn đo được EHR gain trong từng cặp, nhưng việc so trực tiếp P1 với P3 sẽ chứa confound do pooling khác nhau. Vì vậy, bản chính thức nên chạy lại P1–P2 ở image level.

---

## 3. Chuẩn bị dữ liệu

### 3.1. Tạo image manifest

Tạo file:

```text
token_fusion/image_manifest.parquet
```

Mỗi hàng tương ứng một ảnh:

| Cột | Nội dung |
|---|---|
| `dicom_id` | ID ảnh |
| `study_id` | ID study |
| `subject_id` | ID bệnh nhân |
| `split` | train/val/test |
| `view` | frontal/lateral |
| `shard_path` | tar shard chứa ảnh |
| `sample_key` | key trong shard |
| `ehr_row` | row index trong EHR table |
| 13 label columns | ground truth |

Các assertion bắt buộc:

```python
assert manifest.study_id.isin(ehr.study_id).all()

assert set(train_subjects).isdisjoint(val_subjects)
assert set(train_subjects).isdisjoint(test_subjects)
assert set(val_subjects).isdisjoint(test_subjects)

assert manifest.groupby("study_id")["split"].nunique().max() == 1
assert manifest.groupby("study_id")[LABELS].nunique().max().max() == 1
```

Giữ nguyên subject-level split hiện tại.

### 3.2. Giữ nguyên EHR featurization

Tái sử dụng trực tiếp từ notebook 06:

- `as_id_lists`;
- labs value + mask;
- vitals value + mask;
- comorbidity + `como_mask`;
- ragged microbiology;
- `EmbeddingBag`;
- `EhrEncoder`.

EHR encoder vẫn tạo:

```text
e ∈ R^128
```

Không thay đổi feature set giữa P2 và P4. Việc này giúp mọi khác biệt EHR gain xuất phát từ vị trí fusion, không phải từ một EHR encoder khác.

Các EHR feature vẫn phải tuân thủ điều kiện khả dụng tại `index_time`; tuyệt đối không join lại EHR chỉ theo `subject_id`.

---

## 4. Không cache toàn bộ patch tokens

Nếu cache patch tokens cho 346.578 ảnh bằng FP16:

```text
346,578 × 1,369 × 768 × 2 bytes
≈ 679 GiB
```

Do đó, pipeline chính nên:

```python
backbone.eval()

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    tokens = backbone(images)
```

Sau đó detach output và chỉ backpropagate qua head/EHR encoder.

Có thể cache per-image CLS vì nó chỉ khoảng 0,5 GiB FP16, nhưng để bảo đảm preprocessing giống hoàn toàn giữa bốn pipeline, vòng chạy chính nên lấy CLS và patch tokens từ cùng một online forward.

Không adaptive-pool patch tokens trong thí nghiệm đầu tiên. Việc giảm từ \(37 \times 37\) xuống \(8 \times 8\) hoặc \(16 \times 16\) sẽ trở thành một biến kiến trúc mới.

---

## 5. Thiết kế từng pipeline

### P1 — Frozen CLS, CXR-only

#### Data flow

```text
Image
  ↓
Frozen RAD-DINO
  ↓
CLS [B, 768]
  ↓
LayerNorm
  ↓
MLP head
  ↓
13 logits
```

#### Model

```python
class CLSNoneModel(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(768)
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 13),
        )

    def forward(self, batch):
        with torch.no_grad():
            out = self.backbone(batch["image"])
            cls = out.last_hidden_state[:, 0]

        return self.head(self.norm(cls))
```

#### Mục đích

Tạo CXR-only baseline trong cùng image-level protocol với patch pipelines.

Không dùng trực tiếp `none_asl = 0.4097` làm P1 chính thức, vì con số hiện tại dựa trên study-level pooled CLS.

---

### P2 — Frozen CLS + EHR concat

#### Data flow

```text
Image → frozen RAD-DINO → CLS [768] ───────┐
                                           ├─ concat → MLP → logits
EHR → existing EhrEncoder → e [128] ──────┘
```

#### Model

```python
class CLSConcatModel(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone = backbone
        self.ehr_encoder = EhrEncoder(cfg)
        self.norm = nn.LayerNorm(768)

        self.head = nn.Sequential(
            nn.Linear(768 + 128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 13),
        )

    def forward(self, batch):
        with torch.no_grad():
            out = self.backbone(batch["image"])
            cls = out.last_hidden_state[:, 0]

        e = self.ehr_encoder(batch)
        feat = torch.cat([self.norm(cls), e], dim=-1)
        return self.head(feat)
```

#### Mục đích

Đo lại:

\[
\Delta_{\text{EHR, CLS}} = P2 - P1
\]

trong cùng image-level protocol.

Kết quả study-level hiện tại `concat_asl − none_asl = +0.0095` được giữ như replication anchor, không trộn vào phép kiểm định chính.

---

### P3 — Frozen patch tokens + ML-Decoder, CXR-only

#### Data flow

```text
Image
  ↓
Frozen RAD-DINO
  ↓
Patch tokens [B, 1369, 768]
  ↓
ML-Decoder: 13 class queries
  ↓
13 logits
```

#### Token selection

```python
tokens = out.last_hidden_state
patches = tokens[:, 1:, :]       # bỏ CLS
assert patches.shape[1:] == (1369, 768)
```

Không đưa CLS vào ML-Decoder trong bản chính. Nếu vừa dùng CLS vừa dùng patches, ta không còn đo riêng patch representation.

#### Model

```python
class PatchNoneModel(nn.Module):
    def __init__(self, backbone, decoder):
        super().__init__()
        self.backbone = backbone
        self.token_norm = nn.LayerNorm(768)
        self.decoder = decoder

    def forward(self, batch):
        with torch.no_grad():
            out = self.backbone(batch["image"])
            patches = out.last_hidden_state[:, 1:, :]

        patches = self.token_norm(patches)
        return self.decoder(patches)
```

#### Mục đích

Tạo image baseline có khả năng giữ:

- cấu trúc không gian;
- tổn thương nhỏ;
- feature riêng cho từng vùng;
- class-specific attention.

#### Lưu ý về diễn giải

So sánh `P3 − P1` chưa hoàn toàn tách riêng ảnh hưởng của representation vì:

- P1 dùng MLP head;
- P3 dùng ML-Decoder.

Do đó, phép so sánh quan trọng hơn là:

- `P2 − P1`;
- `P4 − P3`.

Nếu cần chứng minh riêng patch tốt hơn CLS, sau bốn pipeline này mới thêm control thứ năm: `CLS → ML-Decoder với sequence length = 1`.

---

### P4 — Frozen patch tokens + EHR token + ML-Decoder

#### Data flow

```text
Image → frozen RAD-DINO → 1369 patch tokens ───┐
                                               ├─ token concat → ML-Decoder
EHR → EhrEncoder → e[128] → projection[768] ──┘
```

#### Tạo EHR token

```python
e = ehr_encoder(batch)              # [B, 128]
ehr_token = ehr_proj(e)             # [B, 768]
ehr_token = ehr_norm(ehr_token)
ehr_token = ehr_token.unsqueeze(1)  # [B, 1, 768]
```

Thêm modality embeddings:

```python
patches = patches + image_type_embedding
ehr_token = ehr_token + ehr_type_embedding
tokens = torch.cat([patches, ehr_token], dim=1)
```

Kết quả:

```text
tokens.shape = [B, 1370, 768]
```

ML-Decoder dùng 13 class queries để lựa chọn thông tin từ cả:

- 1.369 image patch tokens;
- 1 EHR token.

#### Model

```python
class PatchEHRTokenModel(nn.Module):
    def __init__(self, backbone, decoder, cfg):
        super().__init__()
        self.backbone = backbone
        self.ehr_encoder = EhrEncoder(cfg)

        self.ehr_proj = nn.Linear(128, 768)
        self.patch_norm = nn.LayerNorm(768)
        self.ehr_norm = nn.LayerNorm(768)

        self.image_type = nn.Parameter(torch.zeros(1, 1, 768))
        self.ehr_type = nn.Parameter(torch.zeros(1, 1, 768))

        self.ehr_dropout = nn.Dropout(0.2)
        self.decoder = decoder

    def forward(self, batch):
        with torch.no_grad():
            out = self.backbone(batch["image"])
            patches = out.last_hidden_state[:, 1:, :]

        patches = self.patch_norm(patches) + self.image_type

        e = self.ehr_encoder(batch)
        ehr_token = self.ehr_proj(e)
        ehr_token = self.ehr_dropout(self.ehr_norm(ehr_token))
        ehr_token = ehr_token[:, None, :] + self.ehr_type

        tokens = torch.cat([patches, ehr_token], dim=1)
        return self.decoder(tokens)
```

#### Vì sao bắt đầu bằng một EHR token?

Một token là thiết kế tối thiểu:

- ít tăng capacity;
- dễ diễn giải;
- giữ nguyên EHR encoder;
- kiểm tra trực tiếp việc class queries có sử dụng clinical context hay không.

Không bắt đầu ngay bằng 4–8 token cho labs, vitals, microbiology và comorbidity. Nếu P4 một-token có gain, mới phân rã EHR thành grouped tokens trong ablation tiếp theo.

---

## 6. Config đề xuất

```python
@dataclass
class TokenCfg:
    image_repr: str       # {"cls", "patch"}
    ehr_mode: str         # {"none", "concat", "token"}
    head: str             # {"mlp", "mldecoder"}

    freeze_backbone: bool = True
    loss: str = "asl"

    d_cxr: int = 768
    d_e: int = 128
    d_head: int = 256

    max_epochs: int = 12
    patience: int = 3

    effective_batch: int = 256
    lr: float = 1e-3
    wd: float = 1e-2
    warmup_frac: float = 0.05
    grad_clip: float = 5.0

    amp_dtype: str = "bf16"
    seed: int = 42
    tag: str = ""
```

Bốn config:

```python
P1 = TokenCfg(
    image_repr="cls",
    ehr_mode="none",
    head="mlp",
    tag="P1_cls_none_asl"
)

P2 = TokenCfg(
    image_repr="cls",
    ehr_mode="concat",
    head="mlp",
    tag="P2_cls_concat_asl"
)

P3 = TokenCfg(
    image_repr="patch",
    ehr_mode="none",
    head="mldecoder",
    tag="P3_patch_none_asl"
)

P4 = TokenCfg(
    image_repr="patch",
    ehr_mode="token",
    head="mldecoder",
    tag="P4_patch_ehrtok_asl"
)
```

---

## 7. Training protocol

### 7.1. Hai giai đoạn

#### Giai đoạn pilot

Chạy seed 42 cho cả bốn pipeline.

Mục tiêu:

- kiểm tra memory;
- chọn actual batch;
- kiểm tra learning curve;
- phát hiện model đạt best epoch quá sớm hoặc vẫn tăng ở epoch 12;
- kiểm tra EHR token không làm training collapse.

Không báo test result trong giai đoạn chọn hyperparameter. Chỉ dùng validation.

#### Giai đoạn confirmatory

Sau khi khóa training schedule:

```python
SEEDS = [42, 1337, 2024]
```

Nếu interaction gần 0 hoặc CI rộng, mở rộng lên năm seed:

```python
SEEDS = [42, 1337, 2024, 3407, 9001]
```

### 7.2. Batch

Vì actual batch của patch pipeline sẽ nhỏ hơn CLS pipeline:

```text
P1/P2 actual batch: 64–128
P3/P4 actual batch: 16–32
```

Dùng gradient accumulation để tất cả có:

```text
effective batch = 256
```

Ví dụ:

```python
accum_steps = effective_batch // actual_batch
```

Scheduler phải tính theo **optimizer steps**, không phải DataLoader iterations.

### 7.3. Parameters được train

| Module | P1 | P2 | P3 | P4 |
|---|---:|---:|---:|---:|
| RAD-DINO | Frozen | Frozen | Frozen | Frozen |
| EHR encoder | — | Train | — | Train |
| EHR projector | — | — | — | Train |
| MLP head | Train | Train | — | — |
| ML-Decoder | — | — | Train | Train |
| Modality embeddings | — | — | — | Train |

Trong code:

```python
for p in backbone.parameters():
    p.requires_grad = False

assert sum(p.requires_grad for p in backbone.parameters()) == 0
```

### 7.4. Checkpoint selection

- Chọn checkpoint bằng **validation study-level macro mAP**.
- Không chọn bằng image-level mAP.
- Không chọn riêng theo Pneumonia.
- Test chỉ được chạy sau khi checkpoint đã khóa.

---

## 8. Sanity checks trước training

### 8.1. Backbone identity

Cùng một batch ảnh phải tạo cùng image representation trong mọi pipeline:

```python
assert torch.allclose(
    P1.extract_cls(image),
    P2.extract_cls(image),
    atol=1e-6
)

assert torch.allclose(
    P3.extract_patches(image),
    P4.extract_patches(image),
    atol=1e-6
)
```

### 8.2. EHR alignment

```python
for sample in random_samples:
    assert sample.study_id == ehr.iloc[sample.ehr_row].study_id
```

Kiểm tra thêm:

- nhiều ảnh cùng study dùng cùng EHR row;
- không có ảnh test nhận EHR từ train study;
- không có future EHR.

### 8.3. Shape gates

```python
assert cls.shape == (B, 768)
assert patches.shape == (B, 1369, 768)
assert e.shape == (B, 128)
assert ehr_token.shape == (B, 1, 768)
assert fused_tokens.shape == (B, 1370, 768)
assert logits.shape == (B, 13)
```

### 8.4. Gradient gates

Sau một backward pass:

```python
assert all(p.grad is None for p in backbone.parameters())
assert any(p.grad is not None for p in decoder.parameters())
```

Với P4:

```python
assert any(p.grad is not None for p in ehr_encoder.parameters())
assert any(p.grad is not None for p in ehr_proj.parameters())
```

---

## 9. Metrics và kiểm định

### Primary metric

```text
Study-level macro mAP
```

### Secondary metrics

- per-label AP;
- macro AUROC;
- head/medium/tail mAP;
- setting-stratified mAP;
- trainable parameter count;
- GPU memory;
- inference cost.

### Các delta phải báo cáo

```text
P2 − P1: EHR gain trên CLS
P4 − P3: EHR gain trên patch tokens
P3 − P1: tổng khác biệt pipeline ảnh, có head confound
P4 − P2: tổng khác biệt multimodal pipeline
Interaction = (P4 − P3) − (P2 − P1)
```

Nên dùng **paired seeds**, không dùng Welch test độc lập như hai nhóm không liên quan.

Ngoài CI qua seed, chạy cluster bootstrap theo `subject_id` trên test predictions:

```text
2.000 bootstrap samples
sampling unit = subject
primary quantity = delta macro mAP
```

Điều này tránh xem nhiều study của cùng bệnh nhân là các quan sát độc lập.

---

## 10. Diagnostic cho EHR token

Không cần train thêm model, nhưng P2 và P4 phải được đánh giá trong ba chế độ:

| Chế độ | EHR |
|---|---|
| `real` | EHR đúng bệnh nhân |
| `zero` | mask toàn bộ EHR |
| `shuffle` | EHR của bệnh nhân khác trong cùng setting |

Kỳ vọng:

```text
mAP(real EHR) > mAP(shuffled EHR)
mAP(real EHR) > mAP(zero EHR)
```

Đặc biệt với P4:

- nếu `real ≈ shuffle`, token có thể chỉ tăng capacity;
- nếu `real > shuffle` nhưng `real ≈ zero`, kiểm tra implementation shuffle;
- nếu `zero` làm sụp mạnh hơn P3, model đang phụ thuộc quá mức vào EHR;
- nếu `real − zero` tập trung ở labs-related labels, kết quả phù hợp với ablation trước.

---

## 11. Quy tắc diễn giải kết quả

### Trường hợp A

```text
P2 − P1 > 0
P4 − P3 > 0
Interaction ≈ 0
```

Kết luận:

> EHR cung cấp thông tin bổ sung gần như độc lập với loại image representation.

### Trường hợp B

```text
P2 − P1 > 0
P4 − P3 > 0 nhưng nhỏ hơn rõ rệt
Interaction < 0
```

Kết luận:

> EHR vừa có signal độc lập, vừa bù một phần thông tin bị mất trong CLS.

### Trường hợp C

```text
P2 − P1 > 0
P4 − P3 ≈ 0
```

Kết luận:

> Gain EHR hiện tại có khả năng chủ yếu bù cho bottleneck của global CLS; khi patch representation đủ mạnh, EHR không còn bổ sung đáng kể.

### Trường hợp D

```text
P4 − P3 > P2 − P1
Interaction > 0
```

Kết luận:

> EHR có interaction mạnh hơn khi class queries được quyền truy cập đồng thời vào clinical token và localized image tokens. Đây là bằng chứng ủng hộ deep/token-level multimodal fusion.

---

## 12. Cấu trúc notebook đề xuất

Không sửa trực tiếp notebook 06. Tạo:

```text
06b_frozen_token_fusion.ipynb
```

Các cell:

```text
C0  Config + paths
C1  Load cohort, EHR, vocab, labels
C2  Build image manifest
C3  Reuse EHR featurization từ notebook 06
C4  Image-level Dataset + collate
C5  Load frozen RAD-DINO checkpoint
C6  Backbone output adapter: CLS/patch
C7  Reuse EhrEncoder
C8  MLP head + ML-Decoder
C9  Four pipeline model classes
C10 ASL
C11 Study-level multi-view pooling
C12 Sanity gates
C13 Train one config
C14 Pilot seed
C15 Three-seed sweep
C16 Real/zero/shuffle EHR diagnostics
C17 Paired deltas + interaction
C18 Subject-level bootstrap
C19 Export predictions and summary
```

Outputs:

```text
token_fusion/
├── manifests/
│   └── image_manifest.parquet
├── checkpoints/
│   ├── P1_cls_none_asl_s42_best.pt
│   ├── P2_cls_concat_asl_s42_best.pt
│   ├── P3_patch_none_asl_s42_best.pt
│   └── P4_patch_ehrtok_asl_s42_best.pt
├── predictions/
│   ├── image_predictions_*.parquet
│   └── study_predictions_*.parquet
├── metrics/
│   ├── per_seed.json
│   ├── per_label.csv
│   ├── interaction_bootstrap.csv
│   └── ehr_shuffle_diagnostics.csv
└── experiment_manifest.json
```

---

## 13. Thứ tự chạy thực tế

1. Tạo và kiểm tra `image_manifest`.
2. Load frozen backbone và chạy shape/gradient gates.
3. Chạy **P1 seed 42**.
4. Chạy **P2 seed 42** và xác nhận EHR gain có cùng chiều với notebook 06.
5. Chạy **P3 seed 42**.
6. Chạy **P4 seed 42**.
7. Khóa epoch, LR và effective batch.
8. Chạy ba seed cho cả bốn pipeline.
9. Tính `P2 − P1`, `P4 − P3` và interaction.
10. Chạy real/zero/shuffle EHR.
11. Chỉ sau đó mới quyết định có đáng thử:
    - grouped EHR tokens;
    - cross-attention riêng;
    - unfreeze 2–4 block cuối;
    - ConvNeXt V2.

---

## 14. Kết quả quyết định nhất

Con số cần tập trung không phải `P3 − P1`, mà là:

\[
\boxed{\Delta_{\text{EHR,Patch}} = P4 - P3}
\]

và:

\[
\boxed{I = (P4 - P3) - (P2 - P1)}
\]

Hai con số này sẽ cho biết EHR thực sự bổ sung cho localized visual representation, hay chủ yếu đang cứu một global CLS embedding bị nén quá mạnh.
