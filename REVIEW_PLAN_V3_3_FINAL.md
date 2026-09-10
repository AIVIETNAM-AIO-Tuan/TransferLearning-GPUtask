# REVIEW — `FINAL_PLAN_V3_3_LABEL_RELIABILITY.md`

| | |
|---|---|
| Ngày | 2026-07-29 |
| Thay thế | `REVIEW_PLAN_V3_2_GRADIENT_MATCHED.md` |
| Trạng thái | **SẴN SÀNG HIỆN THỰC.** 15/15 mục của review V3.2 đã chốt; **5/5 thắc mắc còn lại đã chốt** (2026-07-29, xem §2). Không còn mục nào treo. |

---

## 0. Kết luận

V3.3 chốt **toàn bộ 15 mục** của review V3.2, và **sửa lại 4 khuyến nghị của tôi** theo hướng
đúng hơn (§1). Nó cũng **bắt được một lỗi tôi đã bỏ sót** trong review V3.2 (§1.5).

Thiết kế đã sẵn sàng để hiện thực. Không còn mục nào chặn.

Hai điều cần nhắc lại, không phải thắc mắc:

- **Chi phí giờ cam kết toàn bộ.** §11 bỏ staging theo hiệu năng (đúng), nên 3 seed × 8 epoch
  × 3 arm là cam kết trước ⇒ ~**12–15× nb15**. Cần biết trước khi bắt đầu.
- **Bước 0a có thể kết thúc thí nghiệm.** §4.4 nói data-sufficiency fail ⇒ R2 mất primary
  status, **không** được sửa manifest rồi tiếp tục. Đó là thiết kế đúng, nhưng nghĩa là rủi ro
  lớn nhất nằm ở bước rẻ nhất. Ngoại suy của tôi cho cửa sổ 24h là ~150–430 cell/class; cửa sổ
  ±7 ngày sẽ nhân lên nhưng chưa biết bao nhiêu.

---

## 1. Bốn chỗ V3.3 sửa tôi — ghi nhận

### 1.1 AdamW **không** bất biến với thang gradient (§2 của plan)

Tôi lập luận: chia hằng số an toàn vì *"AdamW bất biến với thang gradient toàn cục"*.

Plan sửa đúng: *"AdamW không hoàn toàn scale-invariant vì epsilon, clipping và weight decay.
Việc dùng mẫu số chung được chọn vì giữ coefficient của các ô không bị nhắm giống nhau, không
dựa vào một tuyên bố bất biến tuyệt đối của AdamW."*

`eps` phá bất biến ở gradient nhỏ, `clip` phá ở gradient lớn, `wd` là một số hạng độc lập với
gradient. Lý do đúng để chia hằng số là **hệ số của ô không bị can thiệp bằng nhau**, không
phải bất biến của optimizer. Chấp nhận.

### 1.2 Ngưỡng drift: nên **bỏ**, không phải hiệu chuẩn (§7 của plan)

Tôi đề xuất đo smoke 200 step rồi đặt ngưỡng ở 2–3× giá trị ngoại suy.

Plan bỏ hẳn ngưỡng, với hai lý do mạnh hơn:

1. *"parameter drift không có quan hệ đơn điệu với model validity; fine-tuning có mục đích làm
   representation thay đổi"* — gating checkpoint theo drift là **sai về nguyên tắc**, không chỉ
   sai hiệu chuẩn.
2. *"ngoại suy 200 steps bằng sqrt(t) không đáng tin nếu update có tương quan, warmup và
   clipping"* — đúng, ước lượng `lr·√steps/RMS` của tôi giả định random walk thuần, mà warmup
   5% và clip 5.0 phá giả định đó.

Hard failure giờ chỉ còn các lỗi **hiện thực** (NaN/Inf, blocks 0–8 đổi, không có update ở
9–11, feature collapse) — đúng phạm trù. Drift thành diagnostic. Tốt hơn đề xuất của tôi.

### 1.3 Concentration fail: **không** fallback tự động (§4.4 của plan)

Tôi đề xuất `cap ≤ 3 cell/subject` làm fallback.

Plan từ chối: *"cap hoặc subject-normalization thay đổi estimand. Một data-sufficiency failure
không nên được 'chữa' bằng can thiệp được chọn sau audit."*

Đúng — fallback tôi đề xuất là **đổi estimand hậu nghiệm**. Plan cho phép chạy `cap ≤ 3` như
sensitivity nhưng bắt ghi exploratory và không thay primary. Chặt hơn.

### 1.4 Bỏ `κ` thay vì định rõ cách clamp (§3.2 của plan)

Tôi hỏi "clamp hay fail khi `κ·β > 1`".

Plan bỏ `κ` hoàn toàn: *"R2 chọn những examples có feature/Jacobian khác. Parameter-gradient
norm khác do vị trí examples là một phần của targeting mechanism. Ép norm bằng một scalar sẽ
over-control cơ chế."*

Đây là lập luận tôi không nghĩ tới: ép parameter-gradient bằng nhau sẽ **triệt tiêu một phần
chính cơ chế đang thử**. Khớp ở mức logit và để parameter-gradient làm diagnostic là mức đúng.
Vấn đề `κ·β > 1` tự tan.

Và claim được khoá chính xác: *"gradient matched at the output/logit level at initialization"*,
**không** phải *"parameter-gradient matched throughout training"*.

### 1.5 ⚠ Lỗi **tôi đã bỏ sót** — staging theo hiệu năng gây selection bias (§11 của plan)

V3.2 §13 viết: *"Continue to seeds 1337 and 2024 **only if** ... D2 Rare-3 or Macro delta is
non-negative"*.

**Tôi đọc qua và không phản đối** — chỉ nhận xét là "staging đúng". Sai.

V3.3 §11 bắt được: *"continue-only-after-positive creates selection bias and upward-biased
3-seed evidence"*. Nếu chỉ thu seed xác nhận khi seed đầu dương, bằng chứng 3-seed bị lệch lên
theo cấu trúc — đúng dạng lỗi mà cả chương trình này đang cố tránh.

V3.3 chạy cả ba seed **bất kể** dấu của seed 42, chỉ dừng vì lỗi hiện thực / thiếu dữ liệu /
bất ổn số học / harm đã định trước. Và nếu dừng vì futility thì *"report a one-seed exploratory
no-go; do not make a positive claim"*. Đúng.

---

## 2. Năm thắc mắc còn lại — đều nhỏ

### Q1. §5.3 "direct H5 reference" vận hành thế nào? *(đáng hỏi nhất)*

§5.3 yêu cầu: *"Identity gate phải so với một direct H5 reference dùng patch-only, không so
hai copies của cùng implementation."*

Đúng vấn đề — nhưng như viết thì vẫn có thể rơi lại đúng bẫy `G2`: nếu "direct H5 reference"
cũng là `norm`/`mld` từ H5 áp lên `backbone.layernorm(h)[:, 1:]` thì nó **là** cùng một
implementation, chỉ khác instance.

Ba cách vận hành, cần chốt một:

| | cách | mạnh yếu |
|---|---|---|
| **(a)** | So với `ArmModel(plain=True)` của **nb13** (đã verify ở nb13·G2 với `2.98e-07`) trên cùng batch | Reference **thật** từ notebook khác, đã dùng để train H5-lineage. Cần import/tái hiện class của nb13 |
| **(b)** | Assert `patches.shape[1] == n_patch` (CLS đã bị bỏ) + so logit với nb13 nếu có snapshot | Rẻ, bắt đúng lỗi CLS, nhưng không phải reference độc lập |
| **(c)** | So với prediction đã lưu của 06b (nếu `token_fusion/predictions/*` còn) | Reference độc lập nhất |

Đề xuất: **(a)**, và nếu không tiện thì **(b)** kèm ghi rõ giới hạn.

> **Quyết định (2026-07-29): (a)** — `ArmModel(plain=True)` **nguyên bản từ nb13**, kèm 5 điều kiện:
>
> 1. khởi tạo reference **trực tiếp từ code/class của nb13**, KHÔNG copy `forward` sang nb16;
> 2. load backbone 04b và H5 `norm`/`mld` **độc lập**;
> 3. reference và candidate **không dùng chung** object/module;
> 4. `eval()`, FP32, cùng một batch, không augmentation;
> 5. so sánh **ba tầng**, không chỉ logit:
>    - output sau **block 11**
>    - `backbone.layernorm(h)[:, 1:]`
>    - logits sau **H5 head**
>
> Ngưỡng: `max_abs_diff(logits) ≤ 1e-6`.
>
> **Vì sao ba tầng tốt hơn đề xuất của tôi**: tôi chỉ nói so logit. So lần lượt ba tầng
> **định vị** được lỗi nằm ở tail backbone, final LayerNorm, chỗ cắt CLS, hay H5 head — thay
> vì chỉ biết "có lệch". Và hai code path độc lập cùng cho một kết quả tại init mới đúng là
> thoát bẫy `G2`.

### Q2. §4.3 hướng anchor cho `Fracture` — pre-register hay hậu kiểm?

§4.3 đã bắt đúng: *"Fracture có thể xuất hiện mới giữa hai study"*, và yêu cầu báo cáo tách
`anchor before blank` / `anchor after blank`.

Nhưng persistence prior chỉ biện minh **một chiều**:

| chiều | `Fracture` | `Lung_Lesion` / `Pleural_Other` |
|---|---|---|
| anchor **trước** blank (gãy thấy ở `t`, blank ở `t+3d`) | hợp lý — gãy không lành trong 3 ngày ⇒ report bỏ sót | hợp lý |
| anchor **sau** blank (gãy thấy ở `t+3d`, blank ở `t`) | **kém tin** — có thể gãy mới (chấn thương, CPR, bệnh lý) | vẫn hợp lý (mạn tính, có trước) |

Plan chọn: giữ cả hai chiều trong primary, báo cáo tách, và *"nếu hiệu ứng chỉ đến từ
future-positive anchors ... hạ xuống exploratory"* — hậu kiểm.

Lựa chọn khác: **pre-register `Fracture` chỉ dùng backward anchor**. Sạch hơn về nhân quả,
nhưng mất ~một nửa cell của `Fracture` — mà §4.4 đòi ≥500 cell/class, nên có thể đẩy nó xuống
dưới ngưỡng.

Đề xuất: **giữ như plan** (cả hai chiều + báo cáo tách), vì cắt một nửa cell có rủi ro trip
sufficiency gate. Nhưng ghi rõ vào prereg rằng nếu `Fracture` chỉ dương nhờ forward anchor thì
nhãn đó exploratory — plan đã ghi.

> **Quyết định (2026-07-29): giữ bidirectional trong primary**, nhưng **preregister riêng cho
> `Fracture`** một quy tắc về **độ mạnh của claim**:
>
> | hướng | diễn giải cho phép |
> |---|---|
> | past-positive anchor → blank | persistence/omission — **tương đối mạnh** |
> | blank → future-positive anchor | có thể là fracture **mới** — **yếu hơn** |
>
> - **Bắt buộc** báo cáo `D2` tách theo hai hướng.
> - Nếu gain của `Fracture` **chỉ** xuất hiện ở future-positive anchor: **không được** claim
>   phương pháp sửa omission; chỉ được gọi là **temporal association**.
> - Sensitivity backward-only cho `Fracture` phải báo cáo, nhưng **không thay** primary.
>
> Đây là một tầng chặt hơn đề xuất của tôi: tôi chỉ nói "hạ xuống exploratory". Quy tắc này
> gắn **độ mạnh của phát biểu** với **hướng bằng chứng**, chứ không chỉ gắn nhãn exploratory.

### Q3. §7.3 "feature tensor collapse / numerical degeneracy" — tiêu chí?

§7.1 bỏ mọi ngưỡng tuỳ tiện (đúng), nhưng §7.3 vẫn liệt `feature tensor collapse` là **hard
failure** mà không định nghĩa.

Cần chốt một trong hai:

```
(a) mot nguong cu the, vi du:  mean feature variance < 1e-4 x gia tri init
                               HOAC effective rank < 10% init
(b) khong tu dong — log ra roi nguoi doc quyet dinh
```

Đề xuất **(b)**, để nhất quán với tinh thần §7.1 (không đặt ngưỡng chưa hiệu chuẩn). Nhưng khi
đó phải bỏ chữ "hard failure" cho mục này.

> **Quyết định (2026-07-29): bỏ khỏi hard failure.** Hard failure chỉ gồm các trạng thái **xác
> định khách quan**:
>
> ```
> NaN / Inf
> shape sai
> blocks frozen bi thay doi
> blocks trainable khong update
> optimizer / scaler hong
> output tro thanh HANG SO do loi so hoc
> ```
>
> `feature variance`, `effective rank`, `cosine` giảm ⇒ **diagnostics**. Không có threshold đã
> hiệu chuẩn thì **không được** dùng chúng để dừng run hoặc loại checkpoint.
>
> Chặt hơn đề xuất của tôi: tôi chỉ nói "bỏ mục này". Đây là một **danh sách dương** đầy đủ
> cho hard failure, nên về sau không ai thêm tiêu chí mềm vào đó được.

### Q4. §10 thiếu một dòng trong interpretation matrix

Ba dòng hiện có đều giả định `R1-GM > R0`. Thiếu trường hợp **cả hai null**:

```
D1 ~ 0 va D2 ~ 0:
  ap luc blank-negative KHONG phai nut that cho Rare-3;
  ket luan nhat quan voi nhanh loss (nb12-15) — nut that nam o NOI DUNG nhan,
  khong o cach can nang gradient. Khong mo rong huong reweighting nua.
```

Đây là kết cục **có xác suất cao nhất** theo bằng chứng hiện có, nên nó đáng có một dòng viết
trước, để không bị diễn giải lại sau khi thấy số.

> **Quyết định (2026-07-29): thêm BA dòng**, không chỉ một:
>
> | kết quả | diễn giải khoá trước |
> |---|---|
> | `D1 ≈ 0` **và** `D2 ≈ 0` | giảm uniform negative pressure và temporal targeting **đều không** tạo lợi ích đo được. Label instability **có thể tồn tại**, nhưng intervention hiện tại không chuyển nó thành gain |
> | `D1 < 0` **và** `D2 ≈ 0` | giảm negative pressure **gây hại**; temporal targeting không cứu được |
> | `D1 ≈ 0` **nhưng** `D2 > 0` | generic reweighting vô ích, nhưng **vị trí** temporal targeting có giá trị |
>
> Hai dòng sau tôi không đề xuất. Dòng `D1 ≈ 0, D2 > 0` đặc biệt quan trọng vì nó là kết cục
> **mạnh nhất** cho giả thuyết — targeting có giá trị mà không cần đến việc hạ áp lực âm nói
> chung — và nếu không viết trước thì rất dễ bị đọc thành "reweighting có tác dụng".

### Q5. `η = 0.10` khoá mà không có biện minh, và không có arm sensitivity

`η` quyết định đồng thời cường độ can thiệp của R2 **và** giá trị `β_c` của control. Nó được
khoá ở `0.10` không kèm dẫn giải.

Khoá trước là tốt. Nhưng hệ quả: nếu `D2` null, ta **không phân biệt được** "targeting không có
giá trị" với "`η = 0.10` quá yếu". Thêm một arm `η = 0` (bỏ hẳn contested) sẽ tách được, nhưng
+33% chi phí.

Đề xuất: **giữ một `η`**, và ghi hạn chế này vào prereg để một `D2` null không bị đọc rộng hơn
mức nó cho phép.

> **Quyết định (2026-07-29): giữ `η = 0.10`**, kèm biện minh khoá trước:
>
> 1. giảm **90%** áp lực âm trên contested blank;
> 2. đủ mạnh để tạo contrast **đo được**;
> 3. **không** hard-mask hoàn toàn ⇒ vẫn giữ một phần supervision khi temporal evidence **sai**;
> 4. **`R1-GM` tự điều chỉnh** để khớp initial logit-gradient mass của **chính mức can thiệp
>    này**.
>
> Và ghi rõ hạn chế: `D2 ≈ 0` chỉ chứng minh temporal targeting không hữu ích **tại `η = 0.10`**,
> **không** chứng minh mọi mức targeting đều vô ích.
>
> **Điểm 4 là chỗ tôi chưa nói được rõ**: vì `β_c` được **suy ra từ** `η`, control luôn tự khớp
> với bất kỳ `η` nào chọn. Nên tính tuỳ ý của `η` **không** làm hỏng **tính hợp lệ** của `D2` —
> nó chỉ giới hạn **tính tổng quát** của kết luận qua các mức `η`. Đó là một phân biệt quan
> trọng mà đề xuất "ghi hạn chế" của tôi để lộn.

---

## 3. Xác nhận: 15/15 mục của review V3.2 đã chốt

| # | mục | V3.3 chốt ở |
|---|---|---|
| 1 | chuẩn hoá `B×C` hằng số | §2 — kèm lý do đúng hơn của tôi (§1.1) |
| 2 | ngưỡng drift | §7 — **bỏ hẳn**, tốt hơn hiệu chuẩn (§1.2) |
| 3 | CLS convention | §5.3 — `patches = backbone.layernorm(h)[:, 1:]` |
| 4 | `backbone.layernorm` | §5.2 — per-arm, **tail-backbone group**, trainable từ ep1, block LR, clip 5.0 (chặt hơn đề xuất của tôi là gộp vào decoder group) |
| 5 | concentration fallback | §4.4 — **không** fallback tự động (§1.3) |
| 6 | `κ·β > 1` | §3.2 — bỏ `κ` (§1.4) |
| 7 | blocks 9–11 0-indexed | §5.1 |
| 8 | grad clip split | §6 — 5.0 blocks+backbone.LN / 1.0 H5 norm+decoder |
| 9 | wd chỉ `ndim>1` | §6 — kèm "bias and LayerNorm wd=0" tường minh |
| 10 | cache RAW file mới | §12 |
| 11 | timestamp rule sau audit | §12 |
| 12 | tên notebook / thư mục | §12 |
| 13 | bootstrap 2000 / subject | §9 — kèm "handle no-positive resamples, report valid fraction" |
| 14 | temporal diagnostic | §4.3 — kèm **hướng anchor** (tôi không nghĩ tới, xem Q2) |
| 15 | một seed mỗi run | §12 |

Và một mục **ngoài** danh sách 15: §11 sửa selection bias của V3.2 §13 mà tôi đã bỏ sót (§1.5).

---

## 4. Nhật ký quyết định

| Ngày | Mục | Quyết định | Ghi chú |
|---|---|---|---|
| 2026-07-29 | Q1 "direct H5 reference" | `ArmModel(plain=True)` **nguyên bản nb13**, code path độc lập, so **ba tầng** (block 11 → `layernorm[:,1:]` → logits), `≤1e-6` | Ba tầng **định vị** được lỗi, chặt hơn đề xuất chỉ-so-logit của tôi |
| 2026-07-29 | Q2 hướng anchor `Fracture` | Giữ **bidirectional** trong primary + preregister quy tắc **độ mạnh claim** theo hướng | Future-positive only ⇒ chỉ được gọi **temporal association**, không phải "sửa omission" |
| 2026-07-29 | Q3 "feature collapse" | **Bỏ** khỏi hard failure. Hard failure = **danh sách dương 6 mục** | Diagnostics không có threshold hiệu chuẩn ⇒ không được dừng run / loại checkpoint |
| 2026-07-29 | Q4 kịch bản null | Thêm **ba** dòng: `D1≈0 & D2≈0`, `D1<0 & D2≈0`, `D1≈0 & D2>0` | Hai dòng sau tôi không đề xuất; dòng thứ ba là kết cục **mạnh nhất** cho giả thuyết |
| 2026-07-29 | Q5 `η = 0.10` | Giữ, kèm **4 điểm biện minh** khoá trước + hạn chế diễn giải | Điểm 4 (`β_c` suy từ `η` ⇒ control tự khớp) tách **tính hợp lệ** khỏi **tính tổng quát** — chỗ tôi để lộn |

---

## 5. Trạng thái toàn bộ chuỗi review

| file | trạng thái |
|---|---|
| `REVIEW_LABEL_NOISE_24H.md` | đóng — 11 mục, nguồn của toàn bộ hướng này |
| `REVIEW_DISEASE_GATE_WITH_TAIL_SELECTIVE_DBL.md` | đóng ở §8.7 — **còn 2 câu cần sửa** theo lỗi CLS (`REVIEW_PLAN_V3_2` §3-bis) |
| `REVIEW_PLAN_3ARM_LABEL_RELIABILITY.md` (V2) | bị thay |
| `REVIEW_PLAN_V3_1_PARTIAL_FT.md` | bị thay |
| `REVIEW_PLAN_V3_2_GRADIENT_MATCHED.md` | bị thay — nhưng **§3-bis (lỗi CLS) vẫn là nguồn duy nhất** ghi lỗi đó |
| **`REVIEW_PLAN_V3_3_FINAL.md`** | **hiện hành, không còn mục treo** |

Bước tiếp theo theo `FINAL_PLAN_V3_3` §13: `0a → 0b → 0c → 0d → lock → seed 42`.
