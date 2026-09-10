# Đề xuất chỉnh sửa — `PLAN_FROZEN_TOKEN_EHR_FUSION.md`

**Đánh giá chung:** thiết kế factorial đúng, câu hỏi đúng, thống kê (paired seeds + cluster bootstrap theo `subject_id`) **chặt hơn** cái đang dùng ở `06/C12` (Welch độc lập). Bốn chỉnh sửa dưới đây, xếp theo mức độ quan trọng.

**Bối cảnh làm thí nghiệm này đáng chạy:**

```
EHR gain @db  : 0.4059 − 0.3916 = +0.0143
EHR gain @asl : 0.4202 − 0.4097 = +0.0105   ← giữ 73%, KHÔNG sụp
interaction   : −0.0038
```

Kết luận trung tâm sống sót dưới loss tốt. Và `+0.0105` xấp xỉ đúng khoảng cách patch↔CLS — nên **Case C (EHR chỉ vá bottleneck CLS) là giả thuyết rất đáng ngờ**, không phải khả năng lý thuyết.

---

## ① Dùng chung forward cho cả 4 pipeline — 30h → 7.5h

**Vấn đề.** Backbone đóng băng và **giống hệt nhau** ở cả 4 pipeline. Cùng ảnh → cùng token, mọi epoch, mọi seed. Chạy 12 run riêng = tính lại y hệt **144 lần**.

Ước tính từ số thật của 04b (157 img/s train → forward-only ~400 img/s):

```
242k ảnh train / 400 img/s = 10.1 ph/epoch  (+2.2 ph val)
× 12 epoch                 = 2.5 h/run
× 4 pipeline × 3 seed      ≈ 30 h        ← gấp 2.3× toàn bộ LP-FT của 04b
```

**Sửa.** Train cả 4 head trong **một** vòng lặp:

```python
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    out = backbone(images)
cls     = out.last_hidden_state[:, 0].detach()
patches = out.last_hidden_state[:, 1:].detach()

loss = loss_P1(cls) + loss_P2(cls, e) + loss_P3(patches) + loss_P4(patches, e)
loss.backward()
```

`θ_P1..θ_P4` **rời nhau hoàn toàn** ⇒ `∂(L1+L2+L3+L4)/∂θ_P1 = ∂L1/∂θ_P1`. **Tương đương toán học tuyệt đối** với 4 lần train riêng, không phải xấp xỉ.

Lợi thêm: cùng data order ⇒ **paired ở mức batch**, chặt hơn paired-by-seed.

Yêu cầu: 4 optimizer + 4 scheduler + 4 early-stop counter **riêng** (mỗi pipeline dừng theo val của chính nó, chỉ ngừng cập nhật head đó). Scheduler đếm chung một bộ optimizer step.

**Kết quả: 12 run → 3 run ≈ 7.5h** (thực tế ~4–5h nếu `patience=3` và đỉnh rơi ~ep5 như 06).

**Batch trong plan quá thận trọng.** 04b chạy `B=48` với **full backprop qua backbone**. Forward-only no_grad rẻ hơn nhiều → **`B = 96–128`** khả thi; `patches [128,1369,768]` bf16 chỉ 269 MB.

---

## ② `I` đang đo lẫn hai biến — headline không đo đúng thứ đang tuyên bố

**Vấn đề.**

$$I = (P4-P3) - (P2-P1)$$

được diễn giải là *"EHR bổ sung trên representation mạnh, hay bù cho CLS bị nén"*. Nhưng hai nhánh khác nhau ở **hai** biến:

| | representation | fusion |
|---|---|---|
| P2 − P1 | CLS | **concat** |
| P4 − P3 | patch | **token** |

`I ≠ 0` không phân biệt được **"patch cho EHR nhiều đất diễn hơn"** với **"token-fusion mạnh hơn concat-fusion"**. Đây đúng loại confound plan đã tự bắt ở `P3 − P1` (MLP vs ML-Decoder) — nhưng nó nằm trong chính con số headline mà plan không nêu.

**Sửa (a) — rẻ, chỉ đổi cách phát biểu.** `I` đo *"gain EHR đạt được bởi pipeline patch+token so với pipeline CLS+concat"*. Đây là câu hỏi **pipeline-level**, hợp lệ (không thể token-fuse vào một MLP). Chỉ cần đừng viết nó như câu hỏi thuần representation.

**Sửa (b) — khuyên dùng.** Thêm 2 head giữ fusion cố định:

```python
P3' = CLS + ML-Decoder (seq = 1)                    # cũng chính là control thứ 5 plan đã nghĩ tới
P2' = CLS + EHR token + ML-Decoder (seq = 2)
```

Khi đó **interaction thuần representation**:

$$I_{repr} = (P4-P3) - (P2'-P3')$$

giữ `fusion = token` và `head = ML-Decoder` cố định ở cả hai nhánh.

Vì đã dùng chung forward (①), thêm 2 head gần như **miễn phí** — backbone thống trị chi phí, ML-Decoder trên 1–2 token là không đáng kể.

> Prior: `06` cho thấy **FiLM ≈ concat** (p=0.308) — cơ chế fusion không quan trọng ở mức CLS. Nếu đúng cả ở mức token thì confound này nhỏ. Nhưng ta **không biết**, và nó nằm trong headline.

---

## ③ P4 thiếu identity-init → `P4 − P3 < 0` có thể là nhiễu loạn, không phải EHR

**Vấn đề.** `06` có bảo đảm: FiLM identity-init ⇒ fusion **khởi động đúng bằng CXR-only**, chỉ đi chệch khi EHR chứng minh được giá trị; worst case là thoái lui về baseline chứ không kéo tụt.

**P4 không có tính chất đó.** Thêm một token đổi phân bố attention ngay từ step 0 ⇒ P4 **không xuất phát tại P3**. Nếu `P4 − P3 < 0`, không phân biệt được *"EHR vô dụng"* với *"thêm token làm nhiễu ML-Decoder"*.

**Sửa.**

```python
nn.init.zeros_(self.ehr_proj.weight)
nn.init.zeros_(self.ehr_proj.bias)
```

⇒ `ehr_token = LayerNorm(0) + ehr_type = 0` (LN của vector 0 trả về `beta`, init = 0 ⇒ an toàn, không NaN).

P4 tại init ≈ P3 + một token rỗng. **Không phải identity tuyệt đối** — token 0 vẫn chiếm `exp(0)=1` trong mẫu số softmax nên làm loãng nhẹ các patch — nhưng gần hơn nhiều so với init ngẫu nhiên, và đối xứng với kỷ luật của `06`.

---

## ④ EHR token chiếm 1/1370 attention mass — hãy **đo**, đừng đoán

**Vấn đề.** Một token cạnh tranh với 1369 token khác. Tại init attention ~đều ⇒ EHR token nhận **~0.07%** trọng số. Gradient khởi đầu cực yếu. Rủi ro thật là nó **bị nhấn chìm**.

Diagnostic `real/zero/shuffle` bắt được điều này **về hành vi**. Nhưng có phép đo **cơ chế** trực tiếp hơn, gần như miễn phí:

```python
# ML-Decoder cross-attention: 13 class query × 1370 key
attn = decoder_layer.multihead_attn(..., need_weights=True)[1]   # [B, 13, 1370]
ehr_mass = attn[:, :, -1]                                        # trọng số lên EHR token
print(f'EHR attention mass: {ehr_mass.mean():.4f}  (uniform = {1/1370:.4f})')
print(pd.Series(ehr_mass.mean(0).cpu().numpy(), index=LABELS).sort_values(ascending=False))
```

- `≈ 1/1370` ⇒ class query **không thèm nhìn** EHR token ⇒ fusion hỏng **về cơ chế**, không phải EHR vô dụng.
- `>> 1/1370` ⇒ token được dùng thật, **và biết nhãn nào dùng nhiều nhất** (dự đoán: Pneumonia / Consolidation).

Biến `P4 − P3 ≈ 0` từ kết quả *mơ hồ* thành kết luận *có cơ chế*.

---

## Vài điểm nhỏ

- **`shuffle` phải ở mức study** — mọi ảnh của cùng một study nhận **cùng** EHR đã tráo, nếu không phá vỡ tính nhất quán study-level mà pooling 7:3 giả định.
- **"Frozen" ở đây = "đóng băng sau khi đã FT trên nhãn"** (backbone 04b), không phải pretrained gốc. Gap train−val `+0.042` vẫn nằm trong đó. Cả 4 dùng chung nên **triệt tiêu trong mọi delta** — nhưng phải ghi rõ trong report.
- **`token_norm`/`patch_norm` là tham số train thêm** mà 04b không có. Nhất quán giữa P3/P4 nên triệt tiêu — chỉ nhớ P3 vì thế **không** so được với `0.4199`, đúng như plan đã nói.
- **Scheduler theo optimizer step**, và khi dùng chung forward thì 4 scheduler đếm **một** bộ step chung.

---

## Tổng kết

| # | Chỉnh sửa | Lý do | Chi phí |
|---|---|---|---|
| ① | Dùng chung forward cho 4 (6) head | 30h → 7.5h; paired ở mức batch | sửa vòng train |
| ② | Thêm `P2'`/`P3'` **hoặc** sửa phát biểu về `I` | headline đang lẫn representation với fusion | ~0 (nhờ ①) |
| ③ | Zero-init `ehr_proj` | `P4−P3 < 0` phải là EHR, không phải nhiễu loạn | 2 dòng |
| ④ | Đo attention mass trên EHR token | biến kết quả null thành kết luận có cơ chế | ~10 dòng |

**Giữ nguyên, không cần sửa:** thiết kế 2×2 · cùng backbone/loss/split · không cache patch (679 GiB — tính đúng) · train ảnh, eval study với 7:3 renorm · checkpoint theo val study-level macro mAP · test khóa tới cuối · `real/zero/shuffle` · paired seeds + cluster bootstrap theo subject · một EHR token trước, grouped tokens sau.
