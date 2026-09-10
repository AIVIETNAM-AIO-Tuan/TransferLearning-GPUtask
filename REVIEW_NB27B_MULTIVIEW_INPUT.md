# REVIEW · `27b_multiview_input_export.ipynb` — input ablation multi-view (cohort_v2)

**Ngày:** 2026-08-27 · **Nhánh:** cohort_v2 (01–27), **không** phải nhánh CXR-LT (28+)
**Yêu cầu gốc:** notebook ablation multi-view trên MIMIC-CXR cũ gồm 0.7:0.3, 0.5:0.5,
0.6:0.4, multi-view attention, frontal-only.
**Phạm vi đã chốt lại giữa chừng:** *không* gắn vào RAD-DINO — chỉ **xuất input** để gửi
cho bên thứ ba gắn vào kiến trúc của họ.

---

## 1. Đã giao gì

`27b_multiview_input_export.ipynb` — 19 cell (9 code), CPU thuần, không đụng torch.
Ghi ra `planB_out/mv_input/` + một `.zip` để gửi.

| arm | dạng xuất |
|---|---|
| `P73` (0.7/0.3), `P64` (0.6/0.4), `P55` (0.5/0.5), `FRONTAL`, `MEAN` | vector trọng số per-image `[n_img]`, tổng = 1 trên từng study |
| `ATTN` | bó ragged `pad_idx [N,K]` + `pad_mask` + `pad_is_frontal` + module gated-attention mẫu trong README |

Kèm `mv_pool.py` — module drop-in để gắn vào một model bất kỳ: `MultiViewPool`
(`mode='fixed'|'attn'`), `collate_studies`, `StudyModel`, `FrozenFeatureBackbone`.

**Chỗ khó khi gắn vào kiến trúc không phải công thức pooling — là batching.** Dataloader
thường trả một *ảnh* một sample; multiview cần một *study* một sample với số ảnh thay đổi.
Module chọn **phẳng + segment id** thay vì đệm `[B,K,3,H,W]`: batch 64 study ≈ 105 ảnh
(1.64 ảnh/study), đệm tới `K=8` thì thành 512 ảnh — phí ~79% compute của backbone. Phẳng
thì backbone chạy một lần trên `M` ảnh rồi `index_add_` gom về `[B,d]`, không phí gì.
Đổi arm = đổi vector `w` truyền vào, model không đổi một dòng — đó chính là lý do bundle
xuất trọng số *per-image*: khớp thẳng với batching phẳng, không phải reindex.

Ba chỗ có thể gắn, README ghi cả ba: **embedding** (`backbone → f → pool → head`, câu hỏi
chưa ai đo), **xác suất** (`… → p mỗi ảnh → pool p`, áp đúng `w` đó lên `p`, không đổi
model, nhưng nb17 đã đo và null), **token** (ghép patch token đa view, đắt nhất). Cùng bộ
`w` dùng được ở cả mức embedding lẫn mức xác suất vì cả hai đều là tổ hợp lồi tổng = 1.

`MEAN` (trung bình đều mọi ảnh) **không nằm trong yêu cầu gốc** — thêm vào vì nó là null
đúng của `ATTN`: attention init `w=0` khởi động **bằng** `MEAN`, nên `ATTN − MEAN` là phần
attention thực sự học được, đã tách khỏi phần chỉ do thêm tham số. Tắt bằng
`INCLUDE_MEAN_CONTROL=False`.

## 2. Quyết định thiết kế và lý do

**2.1 Xuất *trọng số*, không xuất embedding.** Cả 5 arm cố định đều có dạng
`x_study = Σ wᵢ·fᵢ` với `w` **chỉ phụ thuộc số ảnh frontal/lateral của study**, không phụ
thuộc backbone. Nên `w` là thứ agnostic đáng giao; bên nhận chạy backbone của họ ra `f`
theo đúng thứ tự manifest rồi nhân một dòng. Bản pool sẵn từ cache CLS của `05` để sau cờ
`EXPORT_EMBEDDINGS` (mặc định `False`) — dùng được nếu họ muốn feature RAD-DINO, bỏ được
mà bundle vẫn đủ.

**2.2 `ATTN` không precompute được, và đó không phải thiếu sót.** Trọng số attention do
model sinh từ chính `f` nên chỉ tồn tại lúc forward. Thứ duy nhất precompute được là cấu
trúc ragged. README kèm module `GatedAttnPool` (Ilse et al. 2018) dùng đúng bó đó.

**2.3 Score attention phải đọc cờ frontal/lateral.** Có cờ thì `ATTN` biểu diễn được cả 5
arm cố định (học một hằng số theo view) ⇒ siêu tập đúng nghĩa, so sánh mới có nghĩa —
đúng quan hệ `V3 ⊃ V1,V2` của nb17. Không cờ thì nó là một họ khác hẳn và Δ vô nghĩa.
Đã ghi vào README như một ràng buộc, không phải gợi ý.

**2.4 Chuẩn hoá ở mức từng ảnh, không ở mức study.** Với trọng số tổng = 1, pooling là tổ
hợp lồi nên `pool(standardize(f)) = standardize(pool(f))` **chính xác** — mọi arm chung
một thang, không arm nào thắng nhờ scale. Chuẩn hoá sau khi pool thì mỗi arm phải fit
`mean/std` riêng và ta mất khả năng phân biệt "arm tốt hơn" với "arm hợp LR hơn".

**2.5 `FRONTAL` và 366 study lateral-only.** Mẫu số tái chuẩn hoá `w_f·[∃F]+w_l·[∃L]` bằng
0 tại `w_f=1.0`. `04b` xử lý bằng cách *loại* — nhưng thế thì `FRONTAL` được chấm trên tập
study khác 4 arm kia và Δ không còn quy về pooling. Mặc định
`FRONTAL_LATERAL_FALLBACK=True`: giữ N, các study đó dùng lateral mean, và cột
`frontal_is_fallback` đánh dấu chính xác chúng để bên nhận lọc lại theo quy ước `04b` nếu
muốn — **lọc ở cả 5 arm**, README nói rõ điều này.

## 3. Cổng đã cài trong notebook

| cổng | cell | bắt được gì |
|---|---|---|
| `sum(w) == 1` trên mọi study | C3 | arm nào đó lặng lẽ đổi scale một nhóm study |
| `pad_idx` không trỏ sang study khác | C4 | manifest/ragged lệch nhau |
| lateral sống sót > 99% sau cắt `K_MAX` | C4 | cắt theo `dicom_id` ăn mất lateral ⇒ `ATTN` thua giả tạo |
| `P73` tái lập `study_x_cxr.npy` của `05` ≤ 1e-3 | C5 | sai công thức trọng số **hoặc** sai thứ tự manifest — bắt cả hai cùng lúc |
| round-trip: nạp lại từ đĩa, dựng lại 5 arm | C7 | lỗi serialise, lệch trật tự hàng, README viết sai |
| ragged pool đều == arm `MEAN` | C7 | bó ragged lệch khỏi các arm trọng số |
| `mv_pool` `fixed(w=1/K)` == mean; `attn`@init == mean; `segment_softmax` tổng = 1 | C6-bis | module gửi đi bị hỏng — chạy thật, không chỉ import |

Ngoài ra công thức `arm_weights` đã được smoke-test riêng trên ragged tổng hợp (4000
study, cả 3 nhóm composition kể cả lateral-only) đối chiếu công thức gốc `05`:
sai số ≤ 2.4e-7 cho cả 3 tỉ lệ, cả hai nhánh fallback, `MEAN`, và bó ragged. **Chưa
chạy trên dữ liệu thật** — xem mục 5.

## 4. ⚠ Điều cần nói với bên nhận (và với chính chúng ta)

**nb17 C6 đã đo trục này ở mức xác suất và kết quả là null:**

```
V3_learned_pooler - moc   dMacro +0.00047  CI95 [-0.00096, +0.00199]  chua 0
-> POOLING KHONG phai nut that. KHONG dau tu attention multi-view trong model.
```

Đó là **mức xác suất** (model đóng băng, chỉ đổi cách trộn đầu ra). Ablation mà bundle này
phục vụ là **mức embedding** (pool trước head, head train lại) — câu hỏi khác, chưa ai đo.
Nhưng nb17 là bằng chứng trước rằng dư địa ở trục này rất mỏng, và nó nên là **kỳ vọng cơ
sở** chứ không phải thứ bỏ qua. Đã ghi vào mục cuối README.

Hệ quả thực hành, cũng đã ghi vào README: **đừng so arm bằng một seed.** Chênh lệch giữa
các cách pooling ở bộ này nhỏ hơn nhiễu train của một seed đơn lẻ. Tối thiểu 3 seed, và Δ
phải lớn hơn độ lệch chuẩn giữa các seed trước khi gọi là phát hiện.

## 5. Câu hỏi chặn — cần Tuan trả lời

**Q1 — Bên nhận cần feature hay cần ảnh?** Bundle hiện giao *trọng số + cấu trúc*, giả
định họ tự chạy backbone ra feature per-image. Nếu họ mong nhận **ảnh đã xếp theo study**
(tar shard + manifest) thì phải xuất thêm — `04a` đã có `cxr_560_shards`, thêm một cell
lọc theo manifest là xong, nhưng đó là ~17 GB chứ không phải ~50 MB. *Chưa làm, chờ trả
lời.*

**Q2 — Có được gửi label ra ngoài không?** `labels.npz` chứa `Y [N,13]` dẫn xuất từ
MIMIC-CXR. MIMIC là dữ liệu có DUA (PhysioNet credentialed); gửi nhãn + `study_id` cho
người chưa ký DUA là vi phạm. Notebook **vẫn xuất** vì đó là yêu cầu, nhưng nếu bên nhận
chưa có credential PhysioNet thì phải bỏ `labels.npz` và `study_id` khỏi bundle trước khi
gửi. *Tôi không biết trạng thái credential của họ — cần xác nhận trước khi gửi file đi.*

**Q3 — `EXPORT_EMBEDDINGS`?** Mặc định `False`. Bật thì bundle từ ~50 MB lên ~1.7 GB và
dính chặt vào RAD-DINO. Chỉ bật nếu họ nói rõ là muốn dùng feature của mình.

## 6. Chưa làm

- **Chưa chạy notebook trên dữ liệu thật.** Cần Colab + Drive; các cổng ở mục 3 chỉ chứng
  minh được khi chạy. Cổng đắt nhất (`P73` tái lập `study_x_cxr.npy` của `05`) cần
  `EXPORT_EMBEDDINGS=True` ít nhất một lần — **nên chạy một lần như vậy để nghiệm thu**,
  rồi mới tắt đi và xuất bản gửi đi.
- **Không có phần train/đánh giá.** Theo đúng phạm vi đã chốt lại: bên nhận gắn vào kiến
  trúc của họ. Nếu sau này muốn tự chạy ablation ở phía mình thì đó là notebook khác.