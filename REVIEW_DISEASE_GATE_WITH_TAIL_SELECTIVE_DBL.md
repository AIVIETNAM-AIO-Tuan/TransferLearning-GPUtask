# REVIEW — Disease-Aware Gate + Tail-Selective DBL

| | |
|---|---|
| Ngày | 2026-07-29 |
| Tài liệu được review | `DISEASE_GATE_WITH_TAIL_SELECTIVE_DBL.md` |
| Quyết định của Tuấn | `DECISIONS_DISEASE_GATE_TAIL_DBL_REVIEW.md` (2026-07-29) |
| Sửa đổi sau đó | `REVISION_WASL05_IN_DISEASE_GATE_PLAN.md` (2026-07-29) — chốt Q8, đổi thứ tự ưu tiên Stage B |
| Chốt cuối | `FINAL_DECISIONS_Q5_Q6_Q7_LAMBDAW.md` (2026-07-29) — Q5, Q6, Q7, `λ_W` |
| Trạng thái | **ĐÓNG.** Stage A trượt cổng (§7). Stage B chạy đủ 8 epoch, **cũng trượt** và **không** chữa được cú sập của WASL (§8). `B2`/`B3` mất cơ sở. Nhánh loss khép lại. |
| Nhánh tiếp theo | `REVIEW_LABEL_NOISE_24H.md` — nhiễu nhãn dịch Rare-3 `±0.05`, lớn hơn ~25 lần thứ nhánh loss đang tranh |
| Notebook liên quan | `11_midvslate_screen.ipynb`, `12_loss_screen.ipynb`, `13_dbl_weight_screen.ipynb` |
| Notebook sẽ dựng | `14_disease_gate.ipynb` (Stage A: `A0`=DA-U+ASL, `A1`=DA-E+ASL) → `15_disease_gate_tail_loss.ipynb` (Stage B: `B0` ASL, `B1` global WASL .50, `B2` selective WASL, `B3` selective DBL — có điều kiện) |

---

## 0. Kết luận ngắn

Thiết kế **về nguyên tắc hợp lý**. Hướng "ASL là mục tiêu toàn cục, DBL chỉ là auxiliary
trên nhánh residual của nhãn hiếm đã khoá, nhánh ảnh được bảo vệ bằng stop-gradient" tránh
đúng cái đã hỏng ở global DBL.

Ba vấn đề cần giải quyết trước khi code:

1. **Chặn**: kiến trúc hiện tại không có `z^IMG` tách rời được (§1 dưới).
2. **Đo lường**: tier tail chỉ có 1 nhãn, ngưỡng go/no-go `+0.002` nằm tại sàn nhiễu (§2).
3. **Tài liệu nói mạnh hơn thực tế**: `sg(z^IMG)` không cô lập EHR encoder (§3).

---

## 1. Bối cảnh (state of work tính đến 2026-07-29)

### Kiến trúc đang dùng (nb11 → nb13)

```
forward_shared(x)  : RAD-DINO block 0-8, no_grad, dùng chung mọi arm
forward_tail       : block 9,10,11 (0-idx); CondAdapter tại block 9,10 (= 10,11 1-idx)
CondAdapter        : Z + up(gelu((1+g)·down(ln Z) + b)),  g,b = gen(e),  up/gen zero-init
head               : LayerNorm → MLDecoder(13 class), warm-start từ H5 (06b)
decoder            : delayed unfreeze (ep1 frozen), L2-SP neo về H5
optimizer          : AdamW, LR adapter/enc 1e-4, decoder 1e-5, wd 1e-2, cosine + 5% warmup
batch              : EFF 256, natural sampling
eval               : VAL only, study-level pool frontal/lateral 7:3
```

### Nhãn & tier (khoá, `class_freq.pkl`, 147,791 study train)

| Label | N pos train | Prevalence | Tier |
|---|---:|---:|---|
| No_Finding | 48,443 | 32.78% | head |
| Pleural_Effusion | 39,675 | 26.85% | head |
| Atelectasis | 36,972 | 25.02% | head |
| Lung_Opacity | 36,179 | 24.48% | head |
| Cardiomegaly | 33,305 | 22.54% | head |
| Edema | 26,553 | 17.97% | head |
| Pneumonia | 22,603 | 15.29% | head |
| Enlarged_Cardiomediastinum | 10,848 | 7.34% | medium |
| Consolidation | 9,934 | 6.72% | medium |
| Pneumothorax | 7,555 | 5.11% | medium |
| Lung_Lesion | 4,761 | 3.22% | medium |
| Fracture | 3,212 | 2.17% | medium |
| **Pleural_Other** | **1,763** | **1.19%** | **tail** |

`Worst-3 fixed` (khoá cho metric): `Fracture`, `Pleural_Other`, `Enlarged_Cardiomediastinum`.
Val: 31,670 study ⇒ `Pleural_Other` có ~377 dương.

### Kết quả đã có

- **nb11**: EHR-specific gain ở late fusion — macro **+0.0033**, tail **+0.0006**.
- **nb12** (chọn theo §13): ASL ep3 `0.4406 / 0.1210 / 0.1370 / 0.1963`;
  WASL .50 ep2 `0.4405 / 0.1230 / 0.1388 / 0.1963` (macro/tail/worst3f/recall@K).
  WASL .50 overfit nhanh: tail 0.1230@ep2 → 0.1098@ep8.
- **nb13** ep1 (warm, decoder frozen): mọi arm DBL thua ASL trên macro (−0.008…−0.015)
  và tail, nhưng **Recall@K cao hơn** (+0.005…+0.024), tăng đơn điệu theo mu trong
  nhóm .10/.20/.30 rồi rơi ở .90.
- **Đo được ở nb13**: mu **cao** = re-balance tail/head **mạnh** hơn ở cohort này
  (ngược giả định của plan DBL). Đỉnh tail/head quanh mu≈0.7.
- **Hybrid C14** (`ASL + λ·tail-only DBL`, ngoài prereg): λ=0.25 với chuẩn hoá `mean` trên
  cột tail ⇒ tỷ lệ độ lớn gradient DB/ASL trên cột tail = **2.77** (λ=0.10 → 1.11,
  λ=0.50 → 5.54, λ=1.0 → 11.09).

---

## 2. Bốn câu hỏi chặn

### Q1 — `z^IMG` lấy từ đâu? (chặn thật sự)

Công thức §3 của tài liệu là `z_c^MM = z_c^IMG + g_c·δ_c^EHR`. Kiến trúc hiện tại **không có**
đại lượng nào đóng vai `z^IMG`: EHR được tiêm **bên trong backbone** ở block 10–11 rồi trộn
tiếp qua 2 transformer block và cả ML-Decoder.

| | cách | hệ quả |
|---|---|---|
| **(a)** | Fusion **mức logit**: `z^IMG` = decoder trên token không điều kiện; `δ = W_δ·e` → 13 logit; `g = σ(W_g·[e, sg(z^IMG)])` | Đúng nguyên văn §3. Nhưng là **kiến trúc mới**, bỏ adapter-in-backbone mà nb11 đã chọn. Rẻ, tường minh, dễ diễn giải. |
| **(b)** | Giữ adapter, **hai forward** (một với `e` thật, một với `NULL_BATCH` = arm `L_U` nb11) | Giữ kiến trúc đã validate. Nhưng `z^IMG` không cố định (cũng đang học), ~2× compute phần đuôi, và `g·δ` không tồn tại tường minh ⇒ §3–§5 không áp được. |
| **(c)** | Adapter giữ nguyên **+** thêm decoder head thứ hai trên token chưa điều kiện | Có `z^IMG` thật, nhưng +~8M tham số, hai decoder dễ trôi lệch. |

**Đề xuất: (a)**, và ghi rõ trong báo cáo rằng đây là kiến trúc **khác** nb11–13, không phải
"thêm gate vào cái cũ".

Cần chốt thêm:
- `g_c` có nhìn `sg(z^IMG)` không? **Đề xuất: có** — đó mới đúng nghĩa disease-aware
  (EHR chỉ can thiệp khi ảnh mơ hồ).
- `δ` **zero-init** để bước 0 là identity, giống cách `CondAdapter` đang làm.
- `g` dùng sigmoid; init bias để `g` khởi đầu ở đâu (gần 0 hay gần 1)?

> **Quyết định (2026-07-29): (a) fusion mức logit.** Bỏ hẳn CondAdapter block 10/11 ở arm
> chính Stage A; adapter cũ chỉ còn là tham chiếu global-conditioning bên ngoài. Đây là
> **kiến trúc mới** và phải báo cáo đúng như vậy.
> - `z_IMG` = RAD-DINO patch token → ML-Decoder **không điều kiện** (= H5).
> - Gate: `g_c = σ f_g(sg(h_c^IMG), q_c, e, m)`; residual `δ_c = f_δ(q_c, e, m)`.
>   Dùng **pre-logit class feature** `h_c^IMG` nếu ML-Decoder cho truy cập, không thì lùi
>   về `sg(z_c^IMG)` vô hướng.
> - `δ` output layer **zero-init chính xác**; **gate bias = 0** (⇒ `g ≈ 0.5` tại init).
>   Lý do: identity đã được `δ=0` bảo đảm; bias âm mạnh sẽ bóp gradient của residual vô ích.
> - Bảo vệ nhánh ảnh bằng `L_IMG` + L2-SP + delayed decoder unfreeze, **không** bằng gate đóng.
> - Test bắt buộc bước 0: `max|z_MM − z_IMG| < 1e-6` ở eval FP32.
>
> **Đã xác minh (2026-07-29)** — `h_c^IMG` **truy cập được**. Trong `ml_decoder.py`, tensor
> `h = self.decoder(tgt, ...).transpose(0,1)` có shape `[B, embed_len_decoder, 768]`; với
> `num_of_groups = num_classes = 13` thì `embed_len_decoder = 13`, `duplicate_factor = 1`
> ⇒ **1:1 class ↔ query**, `h[:, c, :]` đúng là feature 768-d của bệnh `c` ngay trước
> `group_fc`. `forward()` chỉ trả logits nên phải patch/tái hiện thân hàm để lấy thêm `h`.
> Notebook sẽ `assert h.shape == (B, 13, 768)` lúc chạy.
>
> **Cảnh báo kèm theo**: `query_embed.requires_grad_(False)` — `q_c` của ML-Decoder là vector
> ngẫu nhiên **đóng băng**, chỉ mang danh tính lớp chứ không mang ngữ nghĩa học được.
> Xem Q7.

---

### Q2 — Tập `T`: Option A (tail hiện tại) hay Rare-3?

Tier tail có **đúng 1 nhãn** (`Pleural_Other`, ~377 dương val). Sai số lấy mẫu của AP một nhãn:

```
SE ≈ AP·√((1−AP)/n_pos) ≈ 0.12·√(0.88/377) ≈ 0.006
so sánh paired (cùng study, score tương quan)  ⇒  ~0.002–0.003
```

Cổng §12 đòi `ΔTail ≥ +0.002` — **ngưỡng nằm ngay tại sàn nhiễu**. Một arm có thể qua cổng
do may. Vấn đề này áp cho cả nb12 và nb13 chứ không riêng thí nghiệm này.

`Rare-3` = `{Pleural_Other 1763, Fracture 3212, Lung_Lesion 4761}` — ~5× số dương, trung bình
3 nhãn ⇒ giảm nhiễu đáng kể.

**Chồng lấn cần xử lý**: `Worst-3 fixed` = `{Fracture, Pleural_Other, Enlarged_Cardiomediastinum}`.
Nếu `T` = Rare-3 thì **2/3 nhãn của metric Worst-3 được DBL giám sát trực tiếp**. Không phải
rò rỉ test, nhưng "Worst-3 tăng" một phần chỉ là "đã tối ưu đúng hai nhãn đó mạnh hơn".

| | phương án |
|---|---|
| **B1** *(đề xuất)* | `T` = Rare-3, giữ cả hai metric, khi báo cáo **tách riêng `Enlarged_Cardiomediastinum`** — nhãn duy nhất trong Worst-3 **không** được supervise ⇒ chứng cứ generalization |
| B2 | Đổi metric chính sang Rare-3 mAP, hạ Worst-3 xuống phụ |
| B3 | Chọn `T` rời khỏi Worst-3 — nhưng thế `T` không còn là nhãn hiếm nhất, mất ý nghĩa |

> **Quyết định (2026-07-29): Rare-3**, khoá từ TRAIN counts =
> `{Pleural_Other, Fracture, Lung_Lesion}`. **Không** định nghĩa lại tier cũ.
> - Vẫn báo cáo `Tail AP` gốc (chỉ `Pleural_Other`) để liên tục, nhưng **một mình nó không
>   đủ** làm go/no-go.
> - **Thêm `NonTarget-10 mAP`** = macro AP trên 10 nhãn ngoài Rare-3 → đây mới là safeguard
>   chính chống collateral damage, thay vì dựa vào Worst-3.
> - Giữ `Worst-3 fixed AP` và `Enlarged_Cardiomediastinum AP` làm chẩn đoán phụ.
> - Chồng lấn Rare-3 ∩ Worst-3 **không phải leakage** (mọi nhãn đều được supervise), nhưng
>   nghĩa là Worst-3 tăng **không được mô tả** là generalization ngoài mục tiêu.
> - `Pleural_Other AP` và `Rare-3 mAP` phải kèm **paired study-level bootstrap CI 95%** so
>   với arm control. Ở screen 1 seed **không** đòi p-value; dùng point estimate + CI +
>   delta per-label để biết hiệu ứng có bị một nhãn kéo hay không.

---

### Q3 — Mức cô lập gradient: `sg(z^IMG)` **không** đủ

§5 khẳng định stop-gradient khiến DBL chỉ cập nhật "EHR encoder, tail gate, tail residual".
Nhưng **EHR encoder dùng chung cho cả 13 bệnh** — gradient DBL vào encoder sẽ đổi biểu diễn
`e` mà head/medium cũng dùng. Cô lập chỉ **một phần**: chặn được nhánh ảnh (quan trọng nhất),
không chặn được nhánh EHR.

| | mức | công thức | tính chất |
|---|---|---|---|
| **nhẹ** | đúng như tài liệu | `z^RES = sg(z^IMG) + g_c·δ_c` | DBL **được** cải thiện biểu diễn EHR cho nhãn hiếm — có thể là điều ta muốn, nhưng cũng chính là đường global DBL đã dùng để phá |
| **chặt** | control | `z^RES = sg(z^IMG) + g_c·δ_c(sg(e))` | DBL chỉ cập nhật head/gate riêng của nhãn hiếm. Sạch nhân quả, nhưng DBL không cải thiện được `e` |

**Đề xuất**: mức **nhẹ** làm arm chính (đúng tài liệu), mức **chặt** làm arm control **nếu**
mức nhẹ tái hiện broad degradation. Đây chính là giả thuyết §5 đang ngầm định — nên kiểm chứng
chứ đừng giả định.

> **Quyết định (2026-07-29): NGƯỢC đề xuất của review — lấy mức CHẶT làm arm CHÍNH.**
> Lý do: động cơ của Stage B là giữ tín hiệu tail của DBL mà **tránh** broad degradation;
> mức nhẹ vẫn để DBL sửa biểu diễn EHR dùng chung cho cả 13 bệnh, tức mở lại đúng đường
> gây hại. Mức nhẹ chỉ chạy **follow-up nếu mức chặt cho null**.
>
> Cách đọc kết quả đã khoá:
> - chặt dương → DBL giúp qua **chuyên biệt hoá residual theo bệnh**;
> - chặt null nhưng nhẹ dương → DBL **cần** định hình lại biểu diễn EHR dùng chung;
> - nhẹ gây broad degradation → xác nhận giả thuyết **nhiễu qua shared encoder**.
>
> ⚠️ Cách hiện thực cụ thể của mức chặt còn mơ hồ — xem **Q5** ở §6.

---

### Q4 — Stage A đã tồn tại chưa? (chưa)

§13 bước 1–2 yêu cầu hoàn thành "disease-aware gate + plain ASL screen, xác minh
`DA-E > DA-U` và `REAL > SHUFFLE`" trước. Trong repo **chưa có notebook disease-gate nào**:
nb11 = mid-vs-late adapter, nb12 = loss, nb13 = DBL. Tài liệu này mô tả Stage B của một
Stage A chưa tồn tại.

| | phương án |
|---|---|
| **(i)** *(đề xuất)* | Dựng Stage A trước — `14_disease_gate.ipynb`: kiến trúc gate + ASL thuần, arm `DA-U` / `DA-E` / `SHUFFLE` theo cổng kép của nb11. Đúng §2. |
| (ii) | Dựng thẳng notebook có cả gate lẫn tail-DBL, `λ_T ∈ {0, .10, .25}` — `λ_T=0` chính là Stage A nằm sẵn trong cùng run. Hiệu quả hơn nhưng vi phạm §2: nếu `λ_T=0` null thì hai arm kia lãng phí GPU mà vẫn phải chạy hết. |

> **Quyết định (2026-07-29): (i) dựng Stage A trước.** `14_disease_gate.ipynb`, kiến trúc
> residual mức logit, **ASL thuần, không DBL**.
> - Arm train: `DA-U` (NULL EHR, cùng kiến trúc + cùng init) và `DA-E` (EHR thật khớp bệnh nhân).
> - Tại checkpoint `DA-E` được chọn, **chỉ đánh giá** (không train thêm) dưới `REAL` /
>   `same-setting SHUFFLE` / `NULL`. **Không cần** arm shuffle train riêng.
> - Estimand: `G_DA_macro = Macro(DA-E) − Macro(DA-U)`, `G_DA_R3 = Rare3(DA-E) − Rare3(DA-U)`.
> - Loss Stage A: `L_A = ASL(z_MM) + 0.25·ASL(z_IMG)`, `λ_I = 0.25` **khoá** cho cả hai arm.
> - Chỉ sang Stage B nếu `DA-E > DA-U` **và** `REAL > SHUFFLE/NULL` trên macro hoặc rare-set
>   đã khoá, không kèm collateral damage đáng kể. Stage A null ⇒ **không** tiêu GPU cho DBL.
> - Macro safeguard nới về **−0.002** (không phải −0.001), đúng đề xuất §3.2.
> - Stage B chọn `λ_T` theo **tỷ lệ gradient** `ρ ≈ 0.25` (H1) và `ρ ≈ 0.50` (H2), không
>   theo λ thô; đo lại tại init cho Rare-3 vì số cũ đo trên 1 cột. Chốt λ số trước khi train,
>   **không** chỉnh theo kết quả val.

---

## 3. Ghi chú không chặn

### 3.1 Headroom rất mỏng

nb11 đo được EHR-specific gain ở late: macro **+0.0033**, tail **+0.0006**. Cổng §12 đòi thêm
`ΔTail ≥ +0.002` từ auxiliary DBL — **hơn 3× toàn bộ hiệu ứng EHR trên tail đã đo được**.

Không phải lý do để không làm, nhưng nên đặt kỳ vọng đúng: nhiều khả năng kết quả là null, và
giá trị khoa học nằm ở chỗ **đo được `I_gate×DBL`** (§11) chứ không ở chỗ thắng cổng.

### 3.2 Ngưỡng `ΔMacro ≥ −0.001` của §12 quá chặt

Chặt hơn nb12/nb13 (`−0.002`) và nằm dưới nhiễu seed đã quan sát (~0.0005–0.001). Đề xuất nới
về `−0.002` cho nhất quán, hoặc chấp nhận safeguard này sẽ chặn cả arm thực ra vô hại.

### 3.3 `s` của §9 — đã có số đo tương đương

§9 muốn `λ_T` "dễ diễn giải" qua `s = E[L_ASL,T] / E[L_DBL,T]` đo tại init. Nhưng thứ quyết
định hành vi tối ưu là **tỷ lệ độ lớn gradient**, không phải tỷ lệ giá trị loss.

Số đã đo (cell C14a của nb13, cột `Pleural_Other`):

| λ | ‖grad từ DB‖ / ‖grad từ ASL‖ |
|---:|---:|
| 0.10 | 1.11 |
| 0.25 | 2.77 |
| 0.50 | 5.54 |
| 1.00 | 11.09 |

Nguyên nhân: DBL chỉ áp lên `|T|` cột nên hệ số thực trên cột đó là `λ·C/|T|`
(= 0.25 × 13 = 3.25 khi `|T|=1`).

Khi dựng notebook sẽ đo `s` **đúng theo §9** *và* in thêm tỷ lệ gradient, để `λ_T` được chọn
dựa trên đại lượng đúng.

### 3.4 §8 (tính weight trên đủ 13 nhãn rồi mask cột) là đúng và đã hiện thực

Cách này giữ được thông tin co-occurrence của DBL. Đã cài trong `HybridASLTailDB` của nb13
(cell C14a) bằng vector hệ số `mask` = 0 ngoài `T` — Stage-0 `H3` xác nhận gradient ở cột
ngoài `T` lệch **chính xác 0.000e+00** so với ASL thuần.

### 3.5 §6 (không nhân DBL với gate) là đúng

Lý do trong tài liệu chính xác: `g→0` sẽ xoá auxiliary loss, tạo đường thoát. Cần thêm
**diagnostic chống gate-collapse**: log `mean|g|`, `mean|g·δ| / mean|z^IMG|` mỗi epoch, và
control `REAL vs SHUFFLE` (nb11 đã có `shuffle_perm`).

---

## 4. Mặc định nếu không có chỉ đạo khác

~~Đã được thay bằng quyết định thật ở §2.~~ Giữ lại để đối chiếu: đề xuất ban đầu của review
là **Q3 → mức nhẹ làm arm chính**; Tuấn chọn **ngược lại** (mức chặt làm chính). Ba câu còn
lại trùng đề xuất.

---

## 5. Câu hỏi phát sinh từ quyết định (Q5–Q7)

### Q5 — Mức "chặt" hiện thực thế nào? Và `B0` có cùng kiến trúc với `B1/B2/B3` không? *(chặn Stage B, chưa chặn Stage A)*

> **Cập nhật sau `REVISION`**: câu hỏi này **rộng hơn** ban đầu. Cấu trúc
> `z_c^RES = sg(z_c^IMG) + g_c^T·δ_c^T(P_T(sg(e)))` giờ xuất hiện ở **cả `B2` (selective
> WASL)** lẫn `B3` (selective DBL) — `REVISION` §7 dùng lại nguyên văn. Nên nó chặn arm
> chính của Stage B chứ không chỉ arm thứ cấp.

Quyết định Q3 viết:

```
e_T   = P_T(sg(e))
z_c^DB = sg(z_c^IMG) + g_c^T·δ_c^T(e_T),   c ∈ T
```

Chỗ mơ hồ: nhánh tail `(P_T, g^T, δ^T)` có phải **chính** nhánh sinh ra `z^MM` cho các nhãn
`c ∈ T`, hay là một head song song?

| đọc | hệ quả |
|---|---|
| **head song song**, không dùng trong `z^MM` | Nhánh tail chỉ được DBL train và **không bao giờ ảnh hưởng dự đoán** ⇒ vô nghĩa |
| **thay nhánh chung cho `c ∈ T`** | Nhãn tail **không bao giờ** định hình được `e`, kể cả dưới ASL. Và `H0` (= Stage-A DA-E, không có `P_T`) khác kiến trúc với `H1/H2` ⇒ so sánh Stage B lẫn lộn *"thêm DBL"* với *"nhánh tail bị tách khỏi encoder + thêm tham số"* |

**Đề xuất (khớp với chính câu của Tuấn "it only isolates the additional DBL gradient")** —
giữ nguyên head residual của Stage A, và tính số hạng DBL bằng **một forward thứ hai của
cùng head đó với input đã detach**:

```
z_c^MM = z_c^IMG      + g_c(sg(h_c^IMG), q_c,  e,     m    ) · δ_c(q_c,  e,     m    )      # ASL
z_c^DB = sg(z_c^IMG)  + g_c(sg(h_c^IMG), q_c, sg(e), sg(m)) · δ_c(q_c, sg(e), sg(m))      # DBL, c ∈ T
```

**Cùng một bộ tham số `θ_res`** ở cả hai dòng. Tính chất:

- ASL cập nhật `θ_res` **và** EHR encoder (qua `e` sống) **và** image head;
- DBL cập nhật **chỉ** `θ_res` — không chạm encoder, không chạm nhánh ảnh;
- `H0` ≡ Stage-A `DA-E` **chính xác về kiến trúc** ⇒ `H0/H1/H2` chỉ khác nhau ở `λ_T`;
- chi phí thêm: một forward MLP nhỏ trên `|T| = 3` lớp, không đáng kể.

Nếu vẫn muốn `P_T` (thêm sức chứa riêng cho tail) thì **`P_T` phải có mặt cả trong `H0`**
với `λ_T = 0`, nếu không so sánh sẽ bị nhiễu.

> **Quyết định:** _(chưa trả lời)_

---

### Q6 — Mẫu số của `ρ` định nghĩa chính xác ra sao? *(chặn Stage B)*

```
ρ = ‖∇_{θ_T} λ_T·L_DB,T‖ / ‖∇_{θ_T} L_ASL,T‖
```

`L_ASL,T` là ASL **giới hạn ở cột tail**, hay ASL đầy đủ? Khác nhau đúng hệ số `C/|T| = 13/3`
— đủ để `ρ = 0.25` biến thành `ρ ≈ 1.08`.

**Đề xuất**: định nghĩa `ρ` bằng **đúng các số hạng có thật trong tổng loss**, để nó đọc
được trực tiếp là "DBL đóng góp bao nhiêu phần gradient so với ASL trên cùng bộ tham số":

```
ρ = ‖∇_{θ_res} ( λ_T · L_DB,T ) ‖  /  ‖∇_{θ_res} L_ASL(z^MM) ‖
```

Đo tại **init**, trên một subset train cố định, `θ_res` = tham số của head gate+residual.
Rồi bisection trên `λ_T` để đạt `ρ ∈ {0.25, 0.50}`. Chốt số `λ_T` vào prereg trước khi train.

> **Quyết định:** _(chưa trả lời)_

---

### Q7 — `q_c` và `m` định nghĩa cụ thể ra sao? *(chặn Stage A, nhẹ)*

**`q_c`** — `query_embed` của ML-Decoder là `requires_grad_(False)`, tức vector ngẫu nhiên
**đóng băng**, chỉ mang danh tính lớp. Ba lựa chọn:

| | |
|---|---|
| dùng `query_embed.weight[c]` (768-d, đóng băng) | không thêm tham số, nhưng không học được gì |
| **thêm `nn.Embedding(13, 64)` học được của riêng ta** *(đề xuất)* | rẻ (832 tham số), gate/residual học được đặc trưng riêng từng bệnh |
| cả hai, nối lại | thừa |

**`m`** (tóm tắt missingness) — lưu ý `e` **đã** chứa toàn bộ mask (EhrEncoder nối
`lab_mask`, `vit_mask`, `como_mask`, `rec1_m`, `rec3_m`…), nên `m` là tín hiệu **tường minh
hoá** chứ không phải thông tin mới. Đề xuất 8 chiều:

```
mean(lab_mask)            # 7 labs
mean(vit_mask)            # 6 vitals
como_mask
mean(rec1_m)              # 2 recency T1
rec3_m
1[t1_status == 0]         # MISSING: khong co specimen trong cua so
1[t3_recency_bin == 'none']
log1p(so token micro that) / log1p(max)   # 0 neu chi co token MISSING
```

> **Quyết định:** _(chưa trả lời)_

---

### Q8 — "ASL" mặc định có nghĩa là WASL α=.50? *(chặn Stage A)*

Chỉ thị của Tuấn (2026-07-29, sau khi chốt Q1–Q4):

> *"à mà ASL nhớ sử dụng WASL@0.5 in case trong plan không nói"*

**Xung đột cần gỡ**: `DECISIONS_DISEASE_GATE_TAIL_DBL_REVIEW.md` mục *Stage-A loss* viết
rõ **"No WASL or DBL in Stage A"** — tức plan **có** nói. Nên "in case plan không nói"
hoặc là không áp cho Stage A, hoặc là chỉ thị mới **ghi đè** dòng đó.

Ba chỗ chữ "ASL" xuất hiện, cần chốt riêng từng chỗ:

| # | vị trí | plan nói gì | đề xuất |
|---|---|---|---|
| 1 | `L_MM = ASL(z^MM)` — mục tiêu chính | "No WASL in Stage A" (**có** nói) | ⚠️ cần Tuấn xác nhận ghi đè |
| 2 | `λ_I·ASL(z^IMG)` — bảo tồn nhánh ảnh | như trên | **giữ ASL thuần** — xem lý do dưới |
| 3 | mẫu số của `ρ` ở Stage B (Q6) | không nói | theo bất kỳ loss nào chọn ở #1 |

**Vì sao đề xuất giữ ASL thuần cho `L_IMG` (#2)**: `z_IMG` tại init **đúng bằng** H5, mà H5
được train dưới ASL (`PREREG['loss'] = 'ASL'` của nb11/06b). `L_IMG` là số hạng **bảo tồn** —
dùng WASL sẽ kéo nhánh ảnh rời khỏi chính cái optimum mà nó đang neo vào, đi ngược mục đích
"preserve the strong image-only prediction".

**Ba hệ quả nếu #1 dùng WASL .50** — không phải lý do phản đối, nhưng phải biết trước:

1. **Estimand chính vẫn hợp lệ.** `G_DA = Macro(DA-E) − Macro(DA-U)` là contrast *trong cùng
   notebook*, hai arm cùng loss ⇒ không bị ảnh hưởng. Chỉ **mất so sánh tuyệt đối** với
   nb11/nb12 (0.4406 / 0.1210), phải nói rõ trong báo cáo.

2. **Stage B sẽ chồng hai cơ chế re-weighting nhãn hiếm.** WASL .50 cho `Pleural_Other`
   `w = 2.4636`; cộng thêm tail-DBL trên Rare-3 là đúng thứ mà
   `PLAN_TUNED_DBL_WEIGHT_SCREEN.md` §2 cấm: *"Do NOT do DBL × WASL n_c^-alpha — double-count
   imbalance; obscure whether improvement came from DBL or external weighting"*.
   ⇒ Nếu #1 = WASL thì hoặc chấp nhận vi phạm §2, hoặc Stage B phải quay lại ASL thuần cho
   `L_MM` (lúc đó H0 ≠ arm Stage A được chọn).

3. **WASL .50 đạt đỉnh rất sớm rồi tụt.** nb12: tail 0.1230@ep2 → 0.1098@ep8 (sụt 0.0132),
   macro cũng đỉnh ~ep2–3. Mà nhánh residual của disease gate **khởi đầu từ 0** và cần vài
   epoch mới học được gì. Rủi ro thật: base loss kéo model đạt đỉnh ở ep2 trong khi gate
   chưa kịp hoạt động ⇒ checkpoint được chọn rơi vào lúc gate còn ~tắt, và Stage A cho null
   vì lý do lịch trình chứ không phải vì gate vô dụng.
   Giảm thiểu: log `mean|g·δ| / mean|z_IMG|` theo epoch (đã có trong gate-collapse
   diagnostics) để biết checkpoint được chọn có nằm trong vùng gate đã hoạt động hay không.

**Đề xuất của tôi**: Stage A chạy `L_MM` = **ASL thuần** đúng như `DECISIONS` đã khoá (một
biến một lúc: chỉ đổi kiến trúc), rồi nếu Stage A đạt thì đưa WASL .50 vào như **một arm loss
riêng ở Stage B** cùng với tail-DBL — lúc đó mới so được ba thứ tách bạch.

> **Quyết định (2026-07-29): `REVISION_WASL05_IN_DISEASE_GATE_PLAN.md`** — đúng theo đề xuất
> trên, và mở rộng thêm.
>
> - **Stage A giữ ASL thuần** cho cả `A0 = DA-U + ASL` và `A1 = DA-E + ASL`. Lý do trùng
>   phân tích ở trên: đổi cả kiến trúc lẫn loss thì kết quả dương không quy được cho ai
>   (gate / WASL / tương tác / động lực overfit khác nhau).
> - **WASL .50 vào Stage B**, và được **ưu tiên hơn DBL** vì bằng chứng thực nghiệm:
>   WASL .50 cải thiện tail + giữ macro + Worst-3 tăng (chỉ yếu ở chỗ overfit nhanh),
>   còn global DBL nâng một nhãn tail nhưng phá rộng.
> - Notebook Stage B đổi tên: **`15_disease_gate_tail_loss.ipynb`** (không phải
>   `15_tail_dbl_disease_gate.ipynb`), arm:
>
>   | arm | loss |
>   |---|---|
>   | `B0` | gate + ASL |
>   | `B1` | gate + **global WASL .50** — tái hiện loss thắng của nb12 trong kiến trúc mới |
>   | `B2` | gate + **isolated Rare-3 selective WASL** |
>   | `B3` | gate + isolated Rare-3 DBL — **thứ cấp**, chỉ chạy nếu còn compute hoặc B2 null |
>
> - `L_IMG` **không** đổi sang WASL: *"Do not weight the image-only auxiliary initially;
>   its role is to preserve the original visual task"* — trùng lý do #2 ở bảng trên.
> - Câu hỏi khoa học của Stage B được phát biểu lại: không phải *"WASL có nâng tail không"*
>   (đã biết là có) mà **"cô lập residual theo bệnh có làm gain tail của WASL bền hơn và
>   ít hại hơn không"** — so `peak / epoch of peak / ep8 / peak-drop` với quỹ đạo WASL toàn
>   cục của nb12 (0.1230@ep2 → 0.1098@ep8, sụt 0.0132).

#### Mốc số cho `λ_W` của arm B2 — nên khoá trước khi train

`REVISION` §7 định nghĩa số hạng **gia số**:

```
L_W,T^extra = (1/|T|) Σ_{c∈T} (w_c − 1)·ℓ_c^+(z_c^RES)
L_B2        = ASL(z^MM) + 0.25·ASL(z^IMG) + λ_W · L_W,T^extra
```

nhưng **không nói `λ_W` bằng bao nhiêu**. Có một giá trị neo tự nhiên nên dùng:

`ASL(z^MM)` lấy mean trên `B×C` nên hệ số của ô dương lớp `c` là `1/(B·C)`; số hạng gia số
lấy mean trên `B×|T|` nên hệ số là `λ_W(w_c−1)/(B·|T|)`. Trọng số dương hiệu dụng của lớp
`c ∈ T` so với baseline:

```
1 + λ_W·(w_c − 1)·C/|T|
```

Global WASL .50 cho đúng `w_c`, tức `1 + (w_c − 1)`. Cân bằng hai vế:

```
λ_W = |T| / C = 3/13 ≈ 0.2308
```

⇒ **`λ_W = 0.2308` làm B2 phát ra đúng lượng nhấn dương như B1 trên Rare-3**, khác nhau
*chỉ* ở chỗ gradient đi qua nhánh residual đã cô lập thay vì qua toàn bộ mạng. Đó chính là
so sánh mà §9 của `REVISION` muốn — cùng emphasis, khác đường dẫn gradient.

`w_c` (α=.50, từ nb12): `Pleural_Other 2.4636`, `Fracture 1.8252`, `Lung_Lesion 1.4992`
⇒ `(w_c − 1)` = `1.4636 / 0.8252 / 0.4992`, đều dương nên đây là "thêm nhấn", không đảo dấu.

**Đề xuất**: `B2` dùng `λ_W = |T|/C = 0.2308` (matched), và nếu muốn arm thứ hai thì
`λ_W = 2|T|/C = 0.4615`. Ghi số vào prereg trước khi train, không chỉnh theo val.

Lưu ý kỹ thuật thuận lợi: `z_c^RES` và `z_c^MM` **bằng nhau về giá trị** (chỉ khác gradient),
nên `ℓ_c^+` trong số hạng gia số được đánh giá tại đúng xác suất của loss chính — sạch.

---

### Ghi chú kèm: một confound của chính Stage A

Bỏ CondAdapter đổi lấy fusion mức logit làm **cơ chế fusion yếu đi**: EHR chỉ còn dịch được
logit, không còn điều biến attention thị giác. nb11 đo gain của adapter-late là macro
**+0.0033**. Nếu Stage A cho null, kết luận đúng phải là:

> *fusion mức logit có gate theo bệnh* không đủ,

**chứ không phải** *"gating theo bệnh vô dụng"* — vì gating hiện thực bằng adapter chưa được thử.

Ghi câu này vào báo cáo Stage A ngay từ đầu để tránh kết luận quá rộng về sau.

### Ghi chú kèm: tin tốt về chi phí

Bỏ adapter ⇒ **toàn bộ backbone đóng băng và `no_grad`**. `forward_shared` phủ cả 12 block
và dùng chung cho `DA-U`/`DA-E`; mỗi arm chỉ còn decoder + EHR encoder + gate/residual.
Rẻ hơn hẳn nb11–13 (vốn backprop qua 3 block cuối). `z_IMG` tại init **đúng bằng** đầu ra
H5 (`D0` anchor của nb13) — mốc kiểm tra chắc chắn.

---

## 6. Nhật ký quyết định

| Ngày | Câu hỏi | Quyết định | Ghi chú |
|---|---|---|---|
| 2026-07-29 | Q1 `z^IMG` | (a) fusion mức logit, kiến trúc mới, bỏ CondAdapter ở arm chính | Đã xác minh `h_c^IMG` truy cập được → dùng bản "preferred" |
| 2026-07-29 | Q1 gate init | `δ` zero-init, gate bias = 0 (`g≈0.5`) | Test bước 0 `<1e-6` |
| 2026-07-29 | Q2 tập `T` | Rare-3 `{Pleural_Other, Fracture, Lung_Lesion}` + `NonTarget-10` làm safeguard chính | Thêm bootstrap CI paired |
| 2026-07-29 | Q3 cô lập | **Mức chặt là chính** (ngược đề xuất review), mức nhẹ là follow-up | Cách hiện thực → Q5 |
| 2026-07-29 | Q4 trình tự | Stage A trước (`14_disease_gate.ipynb`), ASL thuần | Stage A null ⇒ dừng |
| 2026-07-29 | Macro safeguard | −0.002 | Theo §3.2 của review |
| 2026-07-29 | Độ mạnh DBL | Chọn theo `ρ ∈ {.25, .50}`, không theo λ thô | Định nghĩa `ρ` → Q6 |
| 2026-07-29 | Q8 nền ASL hay WASL .50 | **Stage A = ASL thuần**; WASL .50 vào Stage B và **ưu tiên hơn DBL** | `REVISION_WASL05_IN_DISEASE_GATE_PLAN.md` |
| 2026-07-29 | Thứ tự arm Stage B | `B0` ASL → `B1` global WASL .50 → `B2` selective WASL → `B3` selective DBL (thứ cấp) | Notebook đổi tên `15_disease_gate_tail_loss.ipynb` |
| 2026-07-29 | `L_IMG` | Giữ ASL thuần, không weight | Vai trò bảo tồn nhiệm vụ thị giác gốc |
| 2026-07-29 | Q5 cô lập — hiện thực | **Cùng head, cùng tham số, forward thứ hai với input detach. KHÔNG `P_T`.** Kiến trúc `B0`–`B3` giống hệt, chỉ khác loss | Đúng đề xuất của review. Kèm đính chính: đây là *encoder-isolated*, **không** phải *class-parameter-isolated* (MLP gate/residual dùng chung 13 bệnh) ⇒ `NonTarget-10` là **bắt buộc** |
| 2026-07-29 | Q6 mẫu số `ρ` | Gradient của **ASL chính đầy đủ** trên cùng `θ_res`; **không** dùng tail-only ASL; **không** tính `0.25·ASL(z_IMG)` (grad = 0 theo `θ_res`) | Calib: 16 minibatch TRAIN cố định, tắt AMP, `R₀ = √(Σ‖g_aux‖² / Σ‖g_main‖²)`, `λ_T = ρ_target/R₀`. Tuyến tính nên **không cần bisection** |
| 2026-07-29 | `ρ` mục tiêu (DBL `B3`) | `.25` chính, `.50` tuỳ chọn | Chỉ đủ compute cho một arm thì lấy `.25` |
| 2026-07-29 | `λ_W` cho `B2` | **`3/13 = 0.230769`**, là cường độ **duy nhất** ở screen đầu | Đúng mốc neo tôi đề xuất. **Không** tự chạy `6/13`; chỉ follow-up nếu `B2` an toàn nhưng thiếu lực, và không dựa trên tuning val |
| 2026-07-29 | Q7 `q_c` | `nn.Embedding(13, 64)` **học được**, init normal std≈.02 — **không** dùng `query_embed` đóng băng | `h_c^IMG` = bằng chứng thị giác riêng bệnh nhân; `q_c` = danh tính bệnh học được |
| 2026-07-29 | Q7 `m` | Vector 8 chiều chuẩn hoá `[0,1]` | NULL phải là **chữ ký all-missing** (`m6=m7=1`), không phải all-zero ngây thơ; SHUFFLE hoán vị `e` và `m` **cùng một permutation** |
| 2026-07-29 | Dropout gate/residual | **0** ở screen đầu | Để `max\|z_AUX − z_MM[T]\| < 1e-6` |

---

## 7. Kết quả Stage A + deviation sang Stage B (2026-07-29)

### Chạy thực tế

`14_disease_gate.ipynb`, seed 42, **dừng ở ep5/8** (quyết định của Tuấn: trị tuyệt đối đã trôi
xuống đều, không đáng tốn thêm GPU). Dừng sớm **không** đổi kết luận khoá — checkpoint được
§11 chọn là ep2 dù chạy tới ep5 hay ep8, vì macro chỉ đi xuống sau ep3.

| ep | `G_DA` macro | `G_DA` R3 | `G_DA` NT10 | A1 res/z | A0 res/z | A1 gate | A0 gate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.0008 | +0.0002 | +0.0009 | .0803 | .0719 | .275 | .298 |
| 2 | +0.0011 | +0.0007 | +0.0012 | .0530 | **.0001** | .252 | .202 |
| 3 | +0.0014 | +0.0018 | +0.0014 | .0578 | .0006 | .263 | .185 |
| 4 | +0.0016 | +0.0022 | +0.0015 | .0668 | .0001 | .273 | .178 |
| 5 | +0.0017 | **+0.0023** | +0.0014 | **.0805** | .0002 | **.302** | .172 |

### Kết quả khoá — TRƯỢT

Tại checkpoint §11 chọn (**cả hai arm đều ep2**): `G_DA_macro +0.0011`, `G_DA_R3 +0.0007`,
`G_DA_NT10 +0.0012`. Đường 1 (macro ≥ +0.002) trượt; đường 2 (R3 ≥ +0.002) trượt.

### Nhưng ba bằng chứng cơ chế đều dương

1. **Correspondence**: `REAL − SHUFFLE macro = +0.0028` > `REAL − NULL = +0.0012`
   ⇒ SHUFFLE **tệ hơn NULL 0.0016**. EHR của bệnh nhân khác gây hại hơn là không có EHR
   ⇒ model điều kiện hoá theo **nội dung**, không phải dùng EHR như hằng số/capacity.
2. **Bootstrap paired loại trừ 0**: `Rare-3 +0.0007 [+0.0001, +0.0013]`. Hiệu ứng thật,
   nhưng **cận trên CI vẫn dưới +0.002** — không phải "thiếu lực", mà là "nhỏ hơn thanh xà".
3. **Dissociation residual**: `A0` (EHR hằng) **tự tắt** residual ngay khi image head mở khoá
   (res/z → .0001), `A1` giữ và **tăng** (.0530 → .0805), gate mở rộng dần. Residual của `A1`
   sống được vì mang thông tin riêng bệnh nhân mà image head không hấp thụ được.
4. **Artifact chọn checkpoint**: `G_DA_R3` theo epoch tăng đơn điệu tới +0.0023 @ep5, nhưng
   §11 bốc ep2 (macro đỉnh sớm) nơi nó chỉ +0.0007. Gate phát triển **muộn hơn** macro.

### Quyết định: sang Stage B thu gọn — DEVIATION có ghi nhận

> **Quyết định (2026-07-29, Tuấn):** chạy `B1`. Đây là **chệch khỏi prereg** ("Stage A null ⇒
> không tiêu GPU cho Stage B"), dựa trên 4 điểm trên.

`15_disease_gate_wasl.ipynb`, 2 arm × 8 epoch:

| arm | EHR | loss |
|---|---|---|
| `B1` | thật | `WASL.50(z^MM) + 0.25·ASL(z^IMG)` |
| `B1u` | NULL | `WASL.50(z^MM) + 0.25·ASL(z^IMG)` — **control giai thừa** |

`B0` **không train lại**: trùng khít `A1` của nb14, đọc thẳng từ `disease_gate/metrics/epoch_log.json`.
`BATCH` bị ép bằng nb14 để giữ tính paired.

**Vì sao B1 mà không B2/B3**: `B1` (WASL toàn cục, gradient qua toàn mạng) có cơ sở **độc lập
với độ mạnh của gate` — nó hỏi "kiến trúc image-fallback (`L_IMG` + L2-SP) có chữa được cú sập
của WASL .50 ở nb12 không (tail .1230@ep2 → .1098@ep8, sụt .0132)". `B2`/`B3` định tuyến phần
nhấn thêm **chỉ** qua nhánh residual, mà Stage A vừa đo được nhánh đó chỉ dịch R3 ~+0.0007
(CI ≤ +0.0013) ⇒ **trần bị chặn bởi chính con số đó**. Hoãn.

### Estimand Stage B

```
Delta_WASL   = B1 - A1          I_loss x EHR = (B1-B1u) - (A1-A0)
G_DA|WASL    = B1 - B1u         G_DA|ASL     = A1 - A0  (da co)
```

`I_loss×EHR > 0` ⇒ EHR hữu ích hơn khi gradient nhãn hiếm đủ mạnh để pathway fusion học cách
khai thác — đúng Case D của `FINAL_PLAN_WEIGHTED_DEFERRED_ASL_WITH_METRICS.md` §20.

### Hạn chế phải ghi vào mọi báo cáo dùng kết quả này

- 1 seed (42), val-only.
- Stage A **trượt** cổng preregistered; Stage B là deviation.
- Hiệu ứng gate nhỏ hơn ngưỡng thực dụng đã khoá (+0.002).
- Stage A dừng ở ep5/8 nên `Delta_WASL` paired chỉ định nghĩa được cho ep1–5.

---

## 8. Kết quả Stage B — `15_disease_gate_wasl.ipynb` (2026-07-29)

Chạy đủ **8 epoch**, 2 arm. `B0` = `A1` của nb14 (không train lại). Cả bốn arm đều được
§11 chọn **ep2** ⇒ so sánh paired sạch.

### 8.1 Quỹ đạo

| ep | B1 mAP | B1 R3 | B1 tail | B1 g | B1 res/z | B1u mAP | B1u R3 | B1u res/z |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | .4359 | .1478 | .1218 | .381 | .1856 | .4356 | .1478 | .1833 |
| 2 | .4398 | **.1521** | **.1282** | .436 | .1942 | .4388 | .1521 | .1775 |
| 3 | **.4401** | .1505 | .1261 | .421 | .1892 | .4386 | .1505 | .1642 |
| 4 | .4398 | .1489 | .1228 | .401 | .1835 | .4381 | .1488 | .1560 |
| 5 | .4398 | .1483 | .1219 | .403 | .1869 | .4378 | .1482 | .1526 |
| 6 | .4393 | .1464 | .1173 | .399 | .1882 | .4372 | .1463 | .1499 |
| 7 | .4390 | .1465 | .1191 | .400 | .1884 | .4369 | .1465 | .1460 |
| 8 | .4384 | .1449 | .1154 | .399 | .1919 | .4362 | .1447 | .1490 |

### 8.2 Câu hỏi trung tâm (`REVISION` §9) — **KHÔNG**

```
nb12 WASL .50 (kien truc CU, adapter 10/11):  tail .1230 @ep2 -> .1098 @ep8   sut 0.0132
B1            (kien truc MOI, L_IMG + L2-SP): tail .1282 @ep2 -> .1154 @ep8   sut 0.0128
B1u           (kien truc MOI, NULL EHR):      tail .1280 @ep2 -> .1153 @ep8   sut 0.0127
```

**Sụt y hệt nhau** (chênh 0.0004). `L_IMG` + L2-SP + nhánh ảnh tường minh **không** làm
WASL bền hơn chút nào. `B1u` sụt bằng `B1` ⇒ cú sập là tính chất của **WASL × kiến trúc
này**, không liên quan đường dẫn EHR.

Cái kiến trúc mới cho là **dịch mức lên**: peak `+0.0052`, ep8 `+0.0056`. Đường cong nâng
song song rồi sập với đúng tốc độ cũ — **level shift, không phải stability fix**.

Sụt của các metric khác: `B1` R3 `.1521 → .1449` (0.0072), macro `.4401 → .4384` (0.0017).

### 8.3 Go/no-go — **trượt**

Checkpoint chọn: `B1` best macro `.4401@ep3`, dung sai `.0015` → ứng viên {ep2…ep7} → max
tail → **ep2**. `A1` cũng ep2 (nb14).

| tại ep2 | B1 | A1 | Δ | cổng |
|---|---:|---:|---:|---|
| Macro mAP | .4398 | .4400 | −0.0002 | ≥ −0.002 ✔ |
| **Rare-3 mAP** | **.1521** | **.1530** | **−0.0009** | ≥ +0.002 ✘ |
| NonTarget-10 | .5262 | .5261 | +0.0001 | ≥ −0.0015 ✔ |
| Tail AP | .1282 | .1282 | 0.0000 | |

`dWASL` trên R3 âm ở **mọi** epoch có đối chiếu: `−0.0004 / −0.0008 / −0.0025 / −0.0030 / −0.0025`.

### 8.4 Factorial

Tại checkpoint chọn (ep2):

```
Delta_WASL  = B1 - A1    macro -0.0002  R3 -0.0009  NT10 +0.0001
G_DA|WASL   = B1 - B1u   macro +0.0010  R3  0.0000  NT10 +0.0014
G_DA|ASL    = A1 - A0    macro +0.0011  R3 +0.0007  NT10 +0.0012
I_loss x EHR             macro -0.0001  R3 -0.0007  NT10 +0.0002
```

Nhưng `G_DA|ASL` trên R3 **tăng theo epoch** (+0.0002 → +0.0023 @ep5) còn `G_DA|WASL` thì
**đứng yên ở 0** (+0.0000 … +0.0001 suốt 8 epoch). Tại epoch chung cuối cùng (ep5):

```
I_loss x EHR (R3) = +0.0001 - 0.0023 = -0.0022
```

⇒ **WASL xoá sạch lợi ích riêng của nội dung EHR trên nhãn hiếm**, trong khi vẫn giữ nguyên
nó ở macro (`G_DA|WASL macro` ổn định `+0.0020…+0.0022` ở ep5–8). Ngược hoàn toàn Case D.

### 8.5 Cơ chế — nhánh residual bị **trưng dụng**

| | arm EHR thật | arm NULL | tỷ lệ NULL/thật |
|---|---:|---:|---:|
| **WASL** res/z (ep8) | .1919 | .1490 | **78%** |
| **WASL** gate (ep8) | .399 | .383 | **96%** |
| **ASL** res/z (nb14 ep5) | .0805 | .0002 | **0.25%** |
| **ASL** gate (nb14 ep5) | .302 | .172 | 57% |

Dưới ASL, arm NULL **tắt** residual ngay khi image head mở khoá — nó vô dụng khi EHR là
hằng số. Dưới WASL, arm NULL giữ residual gần bằng arm có EHR thật, gate mở như nhau.
**78% độ lớn residual của `B1` tái tạo được mà không cần một chút EHR nào.**

Giải thích: WASL đổi điểm vận hành tối ưu của từng lớp (nhấn dương nhãn hiếm ⇒ dịch bias
tối ưu). Residual là chỗ **rẻ nhất** để đặt bias đó — rẻ hơn dịch image head vốn bị L2-SP
neo và `L_IMG` kéo. Nhánh fusion bị chiếm dụng cho việc không liên quan EHR.

### 8.6 Diễn giải — phân biệt hai loại bằng chứng

Các delta `±0.002` trên Rare-3 nằm **dưới sàn nhiễu nhãn `±0.05`** đo được ở
`REVIEW_LABEL_NOISE_24H.md` §3. Câu *"B1 thua A1 0.0009"* **không có ý nghĩa tuyệt đối**.

Nhưng ba bằng chứng sau **không phụ thuộc chất lượng nhãn**:

1. `res/z` 78% vs 0.25% — quan sát về **hành vi tối ưu hoá**, không phải về nhãn;
2. gate 96% giống nhau giữa arm có/không EHR dưới WASL;
3. sụt `0.0128` vs `0.0132` — cùng một bộ nhãn ⇒ nhiễu triệt tiêu.

Ba cái đó vững ở mức 1 seed. Các delta metric thì không.

### 8.7 Hệ quả — đóng nhánh loss

**`B2`/`B3` không còn cơ sở.** Chúng định tuyến phần nhấn thêm **chỉ qua nhánh residual** —
mà §8.5 vừa đo được nhánh đó, dưới WASL, đã bị chiếm dụng 78% cho hiệu chỉnh bias không
liên quan EHR. `B2` sẽ làm tranh chấp nặng hơn, không nhẹ đi.

Kết luận nhánh loss (nb12 → nb15):

> Trên kiến trúc RAD-DINO patch-token + EHR fusion này, **không** hàm loss nào trong họ đã
> thử (ASL, WASL α=.25/.50, deferred WASL, DB-Focal μ ∈ {.10,.20,.30,.90}, ASL+tail-DBL
> hybrid, gate+WASL) cho cải thiện tail vượt ngưỡng thực dụng `+0.002` mà giữ được macro.
> Đồng thời nhiễu nhãn dịch chính con số đó `±0.05`, tức **~25 lần lớn hơn** thứ đang tranh.

Hướng tiếp theo có giá trị: can thiệp vào **nhãn**, bắt đầu bằng việc đóng băng bộ nhãn
đánh giá (`REVIEW_LABEL_NOISE_24H.md` §8).
