# REVIEW — `16_label_reliability_gradient_matched.ipynb` (phần tiền-GPU)

| | |
|---|---|
| Ngày | 2026-07-30 |
| Nguồn | `FINAL_PLAN_V3_3_LABEL_RELIABILITY.md` §13 bước `0a`–`0d` |
| Quyết định | `REVIEW_PLAN_V3_3_FINAL.md` (5/5 chốt) · `REVIEW_PLAN_V3_2_GRADIENT_MATCHED.md` §3-bis (lỗi CLS) |
| Trạng thái | **`0a`–`0d` xong, GO.** 32 cell (15 code / 17 md). Đã thêm `C10` lock · `C11` eval · `C12` train seed 42 · `C13` `D1`/`D2` + bootstrap. |
| Chặn | **không còn** — cả hai mục §3 đã chốt |

---

## 0. Vì sao dừng ở tiền-GPU

`FINAL_PLAN_V3_3` §4.4: nếu data-sufficiency gate fail thì `R2` **mất primary status**, và
**không** được sửa manifest rồi tiếp tục claim cùng estimand.

Nghĩa là **bước rẻ nhất mang rủi ro lớn nhất**. `C4` (pandas, không GPU) có thể kết thúc thí
nghiệm trước khi cần một dòng train code nào.

---

## 1. Nội dung notebook

| cell | bước plan | làm gì | GPU |
|---|---|---|---|
| `C0` | — | prereg khoá → `preregistration.json` | không |
| `C0b` | — | **stage ảnh** từ `cxr_560_shards/*.tar` → `/content/cxr_local` (verbatim nb13) | không |
| `C1` | `0a` | cohort + **nhãn thô 4 trạng thái** từ BigQuery → `cohort_v2/cohort_raw_labels.parquet` | không |
| `C2` | `0a` | coverage `+1/0/-1/blank` × 13 × 3 split + timestamp audit | không |
| `C3` | `0a` | merge `cur_hadm` / `cur_edstay` từ `cohort_base.parquet` | không |
| `C4` | `0a` | temporal manifest ±7d + funnel + concentration + **sufficiency gate** | không |
| `C5` | — | dataset/loader (**không EHR**) | nhẹ |
| `C6` | — | model: 0–8 frozen shared · 9–11 trainable bản sao riêng · H5 head **patch-only** | có |
| `C7` | `0c` | **identity gate ba tầng** vs nb13 + stochasticity | có |
| `C8` | `0b`+`0d` | initial ASL logit-gradient share → `β_c` + concentration gate theo gradient mass | **có, nặng** |
| `C9` | `0c`+`0d` | weight matrix + smoke 200 step + runtime/VRAM | **có, nặng** |

---

## 1-bis. Sửa sau lần chạy đầu (2026-07-30)

### Thiếu cell staging ảnh

`C1` báo `AssertionError: 346578 anh thieu`. Nguyên nhân: nb16 thiếu cell stage shard mà nb11–15
đều có — `/content/cxr_local` còn rỗng nên `os.path.exists` fail toàn bộ. Đã thêm **`C0b`**
verbatim từ nb13 cell 4 (có cờ `.done_<shard>` nên chạy lại là tiếp tục).

### ⚠ `VSIG` tính sai nguồn — lỗi thật, không chỉ do chưa stage

Bản đầu tôi tính view signature từ `man`, tức **sau khi** lọc theo file ảnh có sẵn ở
`/content`. Sai: view signature là thuộc tính của **study**, không phụ thuộc ảnh nào đã stage.

Hệ quả nếu để nguyên: study chưa stage nhận `VSIG = ''`, và điều kiện same-view
`VSIG[b] == VSIG[a]` cho `'' == ''` → **True** ⇒ pair sai. Đã kiểm chứng bằng ca dựng tay:

| trạng thái | không guard | có guard |
|---|---|---|
| `VSIG` rỗng cả hai (chưa stage) | contested = **True** ← sai | contested = False |
| `VSIG` đầy đủ | contested = True | contested = True |

Hai sửa:

1. `VSIG` tính từ `vm` **đầy đủ**, không từ `man`; thêm assert `<2%` study thiếu `view_group`.
2. `C4` chặn tường minh `VSIG != ''` ở **cả hai** phía của pair.

Điểm đáng nói: assert `<1%` ảnh vắng ở `C1` **che** lỗi này chứ không sửa nó. Nếu shard chỉ
thiếu 0.5% thì `C1` vẫn pass, còn `C4` vẫn âm thầm pair sai đúng số study đó — và cỡ mẫu là
chính cái gate §4.4 đang xét. Lỗi kiểu này không tự lộ ra trong output.

### `asl_neg_grad` gọi `torch.autograd.grad` bên trong `torch.no_grad()`

`C9` báo `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`.

Nguyên nhân: `torch.no_grad()` **thắng** `requires_grad_(True)`. Bên trong nó,
`asl_neg_elem(z)` không dựng graph dù `z.requires_grad = True`, nên `torch.autograd.grad`
không có gì để đi ngược.

Vì sao `C8` chạy được mà `C9` chết: hai chỗ gọi khác ngữ cảnh.

```python
# C8 — NGOAI no_grad  -> OK
with torch.no_grad(): z = _A(forward_shared(b['image']))
g = asl_neg_grad(z)

# C9 — TRONG no_grad  -> RuntimeError
with torch.no_grad():
    gm[a] = float((asl_neg_grad(zz)*...).sum())
```

Sửa **trong hàm** bằng `torch.enable_grad()` chứ không sửa chỗ gọi — để hàm đúng ở mọi ngữ
cảnh, thay vì đúng nhờ may mắn của call site. Đã kiểm: gọi trong và ngoài `no_grad()` cho kết
quả **bit-for-bit** giống nhau. Thêm assert ngay tại `C8` để lần sau lỗi này lộ ở cell rẻ.

Đây là lỗi thầm: nếu `C9` tình cờ không bọc `no_grad()`, cả hai đường đều chạy và không ai
biết hàm phụ thuộc ngữ cảnh.

### Chốt 2026-07-30 sau khi đọc kết quả `C8`/`C9`

| | cũ | mới |
|---|---|---|
| hiệu chuẩn `β` | `eval_tf` (CenterCrop) | **`train_tf` pooled 2 aug seed `(42, 1337)`**; `β_eval` thành sensitivity |
| probe batch | `(4…32)`, chọn theo VRAM | **`(32, 64, 128, 256)`**, đo `s/step` thật, chọn theo **img/s** dưới 80% VRAM |
| số được claim | `step 0` (1 batch) | **pooled 200 batch + bootstrap CI**; single batch chỉ là sanity check |
| khối lượng khớp | mọi ô `y=0` của Rare-3 | thêm bản **`blank-only`** — đúng tập mà `β` khớp, phép thử sắc hơn |
| concentration | chỉ gộp Rare-3 | thêm **per-class** làm diagnostic (gate khoá trước vẫn là số gộp) |
| `cell_share` | mẫu số toàn cohort | **mẫu số TRAIN-only** |
| runtime | chỉ train | **cộng cả val pass** đo thật + dung lượng snapshot |
| `η` | `0.10` | **giữ `0.10`** |
| số seed | — | **cam kết 3 seed** sau khi khoá calibration + batch |

Lý do đổi hiệu chuẩn: `β` đo trên center crop nhưng **áp lên** training chạy
RandomResizedCrop 0.85–1.0. `Fracture` và `Pleural_Other` là tổn thương **ngoại vi** — crop
xuống 85% có thể cắt mất chúng thường xuyên hơn tổn thương trung tâm, tức tăng cường tác động
lên ô contested **khác** ô blank thường. Hợp lệ về prereg vì `β` **chưa** khoá (plan §13 đặt
lock ở bước 1, sau `0a`–`0d`).

Hai seed tăng cường phục vụ hai việc khác nhau: pooled mass giảm nhiễu, và **hiệu số `β` giữa
hai seed** cho biết `β` nhạy thế nào với chính ngẫu nhiên tăng cường. Nếu hiệu số đó lớn hơn
khoảng cách `β_train` ↔ `β_eval` thì tranh luận eval-vs-train là moot ngay từ đầu — `C8` in cả
hai để so trực tiếp.

`β` pooled là **khớp mass gộp**, không phải trung bình hai `β` riêng. Hai thứ chỉ trùng nhau khi
hai lượt có cùng tổng blank mass; đã kiểm bằng ca dựng tay (tổng `100` vs `400` → pooled
`0.9370` vs mean `0.8594`).

### Tăng cường phải tái lập được — kiểm bằng số, không bằng lập luận

`β` sẽ vào prereg nên hai lượt `train_tf` phải cho đúng cùng chuỗi crop mỗi lần chạy. Dùng
`worker_init_fn` + `shuffle=False`, rồi **assert bằng số**: 3 batch đầu chạy hai lần phải
bit-for-bit; và hai seed khác nhau phải cho ảnh **khác** nhau. Fail thì tự hạ về `workers=0`.

Lập luận "worker seeding là tất định" không đủ — nó phụ thuộc số worker, phiên bản torch, và
`persistent_workers`.

### `asl_neg_grad` gọi `torch.autograd.grad` bên trong `torch.no_grad()`

`C9` báo `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`.

Nguyên nhân: `torch.no_grad()` **thắng** `requires_grad_(True)`. Bên trong nó,
`asl_neg_elem(z)` không dựng graph dù `z.requires_grad = True`, nên `torch.autograd.grad`
không có gì để đi ngược.

Vì sao `C8` chạy được mà `C9` chết: hai chỗ gọi khác ngữ cảnh.

```python
# C8 — NGOAI no_grad  -> OK
with torch.no_grad(): z = _A(forward_shared(b['image']))
g = asl_neg_grad(z)

# C9 — TRONG no_grad  -> RuntimeError
with torch.no_grad():
    gm[a] = float((asl_neg_grad(zz)*...).sum())
```

Sửa **trong hàm** bằng `torch.enable_grad()` chứ không sửa chỗ gọi — để hàm đúng ở mọi ngữ
cảnh, thay vì đúng nhờ may mắn của call site. Đã kiểm: gọi trong và ngoài `no_grad()` cho kết
quả **bit-for-bit** giống nhau. Thêm assert ngay tại `C8` để lần sau lỗi này lộ ở cell rẻ.

Đây là lỗi thầm: nếu `C9` tình cờ không bọc `no_grad()`, cả hai đường đều chạy và không ai
biết hàm phụ thuộc ngữ cảnh.

### ⚠ Tôi dán nhãn sai số nào được claim

Bản đầu in `step 0: R1GM/R2 = 0.9456` kèm chú `<- day moi la cai duoc CLAIM`. Sai. Một batch
32 ảnh chứa ~5–6 ô contested, mà `g` trải hơn 5 bậc ⇒ tổng của ~5 số hạng đuôi nặng gần như
thuần nhiễu.

Số đúng là **pooled trên 200 batch** (`0.9934` ở lần chạy đầu, ~6,400 ảnh, ~1,100 ô contested),
và giờ có bootstrap CI. Drift trong 200 step chỉ `3.6e-4` tương đối nên vẫn coi là init.

### Cỡ can thiệp — con số nên đọc trước khi cam kết ~40 giờ

Từ `C8` lần chạy đầu: tổng negative-gradient mass của ô blank 13 nhãn = **115,381**; Rare-3
chiếm **8,208 (7.1%)**; `R2` xoá `(1−η)·G_cont = 966`.

$$rac{966}{115{,}381}=\mathbf{0.84\%}$$

Toàn bộ thí nghiệm dịch **0.84%** khối lượng gradient âm, trong khi nhiễu nhãn dịch Rare-3 mAP
**±0.05** và cổng đang xét là **+0.002**. Không phải lý do dừng — nhưng nó làm kịch bản
`D1 ≈ 0 & D2 ≈ 0` thành kết cục xác suất cao nhất, đúng như Q4 đã dự liệu trước. `C8` in lại
con số này mỗi lần chạy.

---

## 2. Bốn chỗ khác nb11–15

### 2.1 Patch-only — sửa lỗi CLS

`REVIEW_PLAN_V3_2` §3-bis: nb14/15 đưa **cả CLS** (1,370 token) vào ML-Decoder, còn nb11/13 và
H5 dùng `h[:, 1:]` (1,369). nb16 dùng patch-only.

`C7` có **kiểm chứng nghịch**: cố tình giữ CLS rồi assert logits phải lệch `> 1e-4`. Nếu không
lệch thì bản thân cái cổng đó vô dụng — phải biết cổng có khả năng bắt lỗi trước khi tin nó pass.

### 2.2 Identity gate **ba tầng**, code path độc lập

Theo `REVIEW_PLAN_V3_3_FINAL` §2·Q1. Reference là `ArmModel(plain=True)` **verbatim từ nb13**
(kèm `_r_forward_shared` / `_r_forward_tail` / instance backbone riêng). Lấy tầng trung gian bằng
**forward hook** nên `_r_forward_tail` không phải sửa.

| tầng | so | lệch ở đây ⇒ lỗi ở |
|---|---|---|
| `L0` | token sau block 8 | `forward_shared` |
| `L1` | token sau block 11, *trước* final LN | bản sao block 9–11 / thứ tự chạy |
| `L2` | `layernorm(h)[:, 1:]` | final LN hoặc **chỗ cắt CLS** |
| `L3` | logits | H5 `norm` / `mld` |

Thêm assert `module dùng chung REF ∩ CANDIDATE == ∅` — đúng điều kiện 3 anh đặt ra, và đúng chỗ
cổng `G2` của nb14 hỏng (nó so hai instance của **cùng** một class).

### 2.3 `backbone.layernorm` thuộc nhóm **tail**, không phải head

Nó là LayerNorm cuối của backbone. Nhóm tail → lr `1e-5`, clip `5.0`, **trainable từ ep1**. H5
`norm` là LayerNorm riêng của head → frozen ở ep1 cùng decoder. `C9` assert cả hai.

### 2.4 Chuẩn hoá loss bằng **hằng số**

`L = Σ(w·ℓ)/(B·C)`. `C9` assert `CellWASL(w=1)` **bit-for-bit** bằng `ASLoss().mean()` của
nb11–15 ⇒ hằng số chuẩn hoá không đổi, `lr = 1e-5` vẫn ở đúng thang cũ.

Không chia `Σw`: nếu chia thì 10 nhãn NonTarget nhận gradient khác nhau giữa các arm chỉ vì mẫu
số khác — hỏng đúng cái mà control cần giữ. Có assert `NonTarget-10 weight == 1` ở **cả ba** arm.

---

## 3. ⚠ Hai mục cần anh quyết **trước** khi viết train loop

### 3.1 ~~Rule cho `index_time == 00:00:00`~~ — **ĐÃ CHỐT: không loại gì**

```
missing index_time     0  (0.000%)
time == 00:00:00       1  (0.000%)
duplicate subject+time 6  (0.003%)
```

Lý do mạnh hơn "nó nhỏ": cohort ≈ **211,000** study, nên số study ở **đúng một giây** bất kỳ dưới
phân bố đều là `211,000 / 86,400 ≈ 2.4`. Quan sát `1` — **thấp hơn** mức ngẫu nhiên đó, mà giờ
chụp X-quang lại dồn vào ban ngày nên mật độ nửa đêm còn thấp hơn nữa. **Không có phần vượt nào
để giải thích** ⇒ giả thuyết "date-only bị điền thành `00:00`" chết tại đây. `missing = 0` xác
nhận từ phía khác: không có bản ghi nào thiếu `StudyTime` để mà bị điền bù.

Người dùng xác nhận `00:00:00` là 0 giờ đêm hợp lệ (2026-07-30). `6` dup bị `d > 0` loại tự động.

Không sửa code: `TS_OK = (~T.isna())` cho tất cả `True`, và đó giờ là hành vi **đúng**.

Tôi đã viết tiêu chí quyết định **sai** ở `C2` (`midnight < ~1% → loại`). Câu hỏi không phải
"`00:00` có nhiều không" mà là "**bao nhiêu trong số đó là date-only**" — và phép so với
`N/86400` trả lời được, còn ngưỡng bốc thì không.

### 3.1-cũ (giữ để đối chiếu)

Hiện `TS_OK` chỉ loại `NaT`. Study có giờ `00:00:00` **vẫn được** làm anchor.

`C2` in ra ba số: `missing`, `midnight`, `duplicate subject+time`. Hai đường ra:

| nếu | nên |
|---|---|
| `midnight` ≲ 1% | loại chúng khỏi manifest — mất ít cell, an toàn |
| `midnight` nhiều | **giữ**, vì `00:00` có thể là giờ chụp thật; ghi nhận là nguồn nhiễu |

Không tự quyết được vì nó đổi cỡ mẫu, mà cỡ mẫu lại là cái gate §4.4 đang xét.

### 3.2 Chi phí — cần cam kết trước

Đây là notebook **đầu tiên** train block backbone. Blocks 9–11 trainable ⇒ phải giữ activation
cho backward qua 3 block × 1,370 token × **3 arm**. Đắt hơn nb15 rất nhiều.

`C9` in `hours_per_seed` đo thật. Nếu `× 3 seed` vượt hạn mức Colab thì **hai lựa chọn không
tương đương**:

| cách | hệ quả |
|---|---|
| giảm `BATCH`, tăng `accum` | chỉ chậm hơn. **Không** đổi thiết kế |
| chạy tuần tự từng arm | **mất tính paired trên cùng batch** — đây là **đổi thiết kế**, không phải đổi tham số |

`C9` cũng bỏ `EFF = 256` nếu `BATCH × ACCUM` không đạt — in ra để so.

---

## 4. Số đã kiểm được trước khi chạy (smoke ngoài Colab)

Toàn bộ logic không cần GPU đã test bằng ca dựng tay:

| kiểm | kết quả |
|---|---|
| funnel: quá 7d / khác view / khác subject → **không** contested | OK |
| 1 ô blank + 3 anchor → đúng **1** cell (anchor gần nhất, không mẫu bậc hai) | OK |
| hướng `past` / `future` | OK |
| `g` autograd == dạng đóng | `max|diff| 6.7e-16` |
| `g` tại `p = clip = 0.05` | **đúng 0** |
| **`g(0.90)/g(0.10)`** | **252,604×** (~5.4 bậc) |
| `β_c ∈ [η, 1]`, `G_cont = 0 ⇒ β = 1` | OK, 500 trial random |
| `β_c·(G_cont+G_ord) == η·G_cont + G_ord` từng nhãn | `1e-9` |
| `NonTarget-10 weight == 1` cả ba arm | OK |
| `CellWASL(w=1) == ASLoss` | **bit-for-bit** |
| đổi `w` một nhãn không đổi gradient 12 nhãn còn lại | OK |
| U-Ones: `-1 → 1` ⇒ ô uncertain là **positive**, ngoài bài toán reweight blank | OK |

### `g(0.90)/g(0.10) = 2.5×10⁵` — con số này quan trọng hơn tôi tưởng

Tôi đã viết "lệch ~3 bậc" trong bản đầu; đo được là **~5.4 bậc**. Nó làm luận điểm mạnh hơn
nhiều: khớp theo **số ô** (V3.1) lệch khỏi khớp gradient tới mức không thể coi là gần đúng. Và
vì `g = 0` tại `p ≤ 0.05`, một phần ô blank **không đóng góp gradient nào** — nên `cell_share` và
`cont_share` về nguyên tắc là hai đại lượng khác nhau, không phải hai cách đo cùng một thứ.

`C8` in cả hai cạnh nhau kèm tỉ lệ để thấy trực tiếp.

---

## 5. Đọc kết quả theo thứ tự nào

| # | file | chi phối |
|---|---|---|
| 1 | `metrics/sufficiency_gate.json` | `gate_pass` — `false` ⇒ `R2` mất primary status |
| 2 | `metrics/temporal_manifest_funnel.csv` | bước nào ăn hết cell |
| 3 | `metrics/contested_cells_by_class.csv` | `past` vs `future` của `Fracture` |
| 4 | `metrics/initial_gradient_share.csv` | `β_c`, `cont_share` vs `cell_share` |
| 5 | `metrics/smoke_200step.json` | `hours_per_seed × 3` |
| 6 | `metrics/timestamp_audit.json` | rule §3.1 ở trên |
| 7 | `metrics/identity_gate.json` | ba tầng ≤ `1e-6` |

### Ba đường ra

- **A · `gate_pass = true`** → khoá manifest hash + `β_c` + commit → viết train loop. Cam kết
  `3 seed × 8 epoch × 3 arm` **trước**, chạy cả ba seed **bất kể** dấu delta của seed 42
  (§11 — chính lỗi selection bias mà V3.2 §13 mắc phải).
- **B · fail vì cell/subject** → `R2` mất primary. Được chạy `cap ≤ 3 cell/subject` như
  sensitivity nhưng phải ghi **exploratory**.
- **C · fail vì concentration** → như B. Top 1% subject chiếm quá `15%` contested gradient mass
  nghĩa là `R2` đang học một nhóm nhỏ bệnh nhân, không phải một hiện tượng nhãn.

---

## 5-bis. `C10`–`C13` (thêm 2026-07-30)

| cell | làm gì |
|---|---|
| `C10` | **LOCK** — hash 6 artifact + `β` + batch + timestamp rule + hyperparameter → `locked_config.json`. Chạy lại là **so lại và raise nếu lệch** |
| `C11` | eval §9: `Macro`/`Rare-3`/`NonTarget-10`/`Tail`/`Worst-3`/`Recall@K` + checkpoint rule + diagnostics (drift theo block, feature cos/var/rank, param-grad ratio) |
| `C12` | train seed 42, 8 epoch × 3 arm, resume theo epoch có guard **hash lock** |
| `C13` | `D1`/`D2` + subject-cluster bootstrap + gate §10 + bảng diễn giải |

### Ba chỗ dễ sai mà tôi chặn bằng test

**`Tail AP ≠ Rare-3 mAP`.** Checkpoint rule §9 dùng **`Tail AP`** (tier `tail` = chỉ
`Pleural_Other`), còn gate §10 dùng **`Rare-3 mAP`** (ba nhãn). Dùng lẫn là đổi tiêu chí.

**`grad_stats` phải gọi TRƯỚC `clip`.** `clip_grad_norm_` sửa `.grad` in-place; đo sau clip thì
mọi arm đều ra `≤ 5.0` và diagnostic §3.2 thành vô nghĩa. Đã kiểm: `31.62 → 1.00`.

**Bootstrap theo subject, không theo study.** Đã đo trên dữ liệu dựng tay có tương quan trong
subject: CI theo study `0.102` vs theo subject `0.151` — theo study **hẹp giả 1.5×**.

### Resume có guard hash lock

`_resume_seed{SEED}.pt` lưu `lock_sig`. Nếu resume được tạo dưới một lock khác thì `C12` raise
thay vì tiếp tục — nếu không, kết quả sẽ trộn hai cấu hình mà không ai biết.

### `C9` giờ đọc cache

Session mới không chạy lại probe và smoke. Không chỉ để tiết kiệm 15 phút: **chạy lại smoke sẽ
ghi đè `smoke_200step.json` bằng một lượt lấy mẫu khác**, tức đổi chính con số `C10` đã khoá.
`FORCE_SMOKE = True` mới chạy lại, và cell nói rõ hệ quả.

### `§3.3` — `R2/R1GM` gradient ratio **được phép tăng**

Plan nói trước: `R2` giảm áp lực lên contested high-score blank ⇒ điểm nhóm đó tăng ⇒ raw ASL
negative gradient tăng. Đó là **động lực điều trị mong đợi**, không phải lỗi hiện thực. `C13` in
ratio theo epoch kèm dòng nhắc này. Không đổi weight động.

---

## 5-ter. AMENDMENT 2026-07-31 — hoãn seed 1337 / 2024

Seed 42 đã chạy đủ 8 epoch. Kết quả:

| | |
|---|---|
| `D1` (Rare-3 mAP) | `+0.0001…+0.0003` mọi epoch; CI95 `±0.00025` ⇒ **null kết luận được**, dưới cổng `+0.002` **tám lần** |
| `D2` (Rare-3 mAP) | `+0.0003, +0.0003, −0.0003, −0.0004, −0.0004, −0.0003, −0.0002, 0.0000` — **âm 5/8 epoch** |
| `D2` (Tail) | `+0.0008…+0.0026`, dương 8/8, nhưng **0/3 epoch** đầu có CI loại 0 |
| Cổng §10 | fail: `Rare-3 +0.0003` tại điểm tốt nhất, `Macro 0.0000` |
| Dự đoán §3.3 (hiệu ứng lớn dần theo epoch) | **không xảy ra** |

**Quyết định: hoãn seed 1337 và 2024.**

**Lý do — đọc kỹ, đây là chỗ dễ bị hiểu nhầm thành selection bias.** §11 cấm quyết định thu
thập seed xác nhận **dựa trên dấu** của delta seed 42, vì "chỉ tiếp khi dương" làm bằng chứng
3-seed lệch lên. Việc hoãn ở đây **không** dựa trên dấu:

- Lập luận là **thứ tự ưu tiên tài nguyên**: toàn bộ phần quyết định của kế hoạch mới
  (`sleepy-puzzling-harbor.md`) chỉ tốn **~40 phút GPU**, trong khi 2 seed còn lại tốn **25 giờ**.
  Chạy phần rẻ trước là đúng thứ tự bất kể seed 42 ra dấu gì.
- Hoãn khi kết quả **null** không tạo lệch **lên**; nó chỉ làm kết luận **yếu đi**.

Trong thời gian hoãn, nb16 đứng ở **one-seed exploratory no-go**. Theo §11:

> *If stopped for performance futility, report a one-seed exploratory no-go; do not make a
> positive claim.*

⇒ **Không** được claim dương từ nb16 dưới bất kỳ hình thức nào cho tới khi có ít nhất một seed
nhân bản.

### Vấn đề power — lẽ ra phải tính ở `0a`–`0d`

Đo được từ `C15`: CI lấy mẫu val trên `D2` Rare-3 là **`±0.0018`**, trong khi cổng là
**`+0.002`**. Hai điều đi kèm:

1. **Nhiễu lấy mẫu val là đồng pha giữa các seed** — ba seed đánh giá trên **cùng một tập val**,
   nên chạy thêm seed giảm phương sai *giữa seed* mà **không** co được `±0.0018`.
2. **Nhánh `Macro` của cổng gần như không với tới được**: can thiệp chạm 3/13 nhãn nên
   `ΔMacro = (3/13)·ΔRare-3`; muốn `ΔMacro ≥ 0.002` phải có `ΔRare-3 ≥ 0.0087`, **khó gấp 4.3×**
   nhánh Rare-3. Nó là điều kiện chặt hơn, không phải lối đi thay thế.

⇒ Thiết kế thiếu power so với chính cổng của nó. Đây là phép tính **chỉ cần cỡ val + prevalence**,
tức làm được ở bước tiền-GPU, và tôi đã không đưa vào `0a`–`0d`. Ghi lại như một thiếu sót của
quy trình, không phải của kết quả.

---

## 6. Khoá trong prereg, không đổi sau khi thấy kết quả

- claim duy nhất được phép: *"gradient matched at the output/logit level at initialization"*
- **cấm** claim: *"parameter-gradient matched throughout training"*
- `η = 0.10` — `D2 ≈ 0` chỉ chứng minh targeting **không hữu ích tại `η = 0.10`**
- `Fracture` chỉ dương nhờ future-positive anchor ⇒ chỉ được gọi **temporal association**
- historical frozen label là primary; known-only là **diagnostic only**
- hard failure = danh sách **dương** 6 mục; drift/cosine/rank là **diagnostic**, không gate
- `cohort_raw_labels.parquet` là **file mới** — không ghi đè cohort cũ
- **TEST không bao giờ chạm**

---

## 7. Nhật ký quyết định

| Ngày | Mục | Quyết định | Ghi chú |
|---|---|---|---|
| 2026-07-30 | phạm vi nb16 lần này | chỉ `0a`–`0d`, **không** train loop | §4.4 có thể kết thúc thí nghiệm ở `C4` |
| 2026-07-30 | thiếu cell staging ảnh | thêm `C0b` verbatim nb13 cell 4 | `C1` fail `346,578 anh thieu` |
| 2026-07-30 | `VSIG` tính từ `man` (đã lọc file local) | đổi sang `vm` đầy đủ + chặn `VSIG != ''` ở `C4` | `'' == ''` gây pair sai; assert `<1%` chỉ **che** lỗi |
| 2026-07-30 | `asl_neg_grad` chết trong `no_grad()` | bọc `torch.enable_grad()` **trong hàm** + assert ở `C8` | sửa call site thì hàm vẫn phụ thuộc ngữ cảnh |
| 2026-07-30 | cột `cell_share` ở `C8` lệch mẫu số | sửa sang mẫu số TRAIN-only | numerator train-only / denominator toàn cohort ⇒ tỉ lệ thổi ~1.5× |
| 2026-07-30 | `β_eval` hay `β_train` | **`β_train` pooled 2 aug seed**; `β_eval` giữ làm sensitivity | crop 85% có thể cắt tổn thương ngoại vi ⇒ tăng cường tác động lệch lên ô contested |
| 2026-07-30 | mở probe batch | **có** — `32/64/128/256`, chọn theo throughput dưới 80% VRAM | lần đầu dừng ở 32 chỉ vì hết danh sách, peak mới 14% |
| 2026-07-30 | đổi `η` vì cỡ can thiệp 0.84% | **không** — giữ `η = 0.10` | ghi cỡ can thiệp vào báo cáo thay vì đổi thiết kế |
| 2026-07-30 | số seed | **cam kết 3 seed** sau khi khoá calibration + batch | chạy cả 3 bất kể dấu delta seed 42 (§11) |
| 2026-07-30 | số nào được claim | **pooled 200 batch + CI**; single batch chỉ sanity | 1 batch ≈ 5 ô contested, `g` trải >5 bậc ⇒ thuần nhiễu |
| 2026-07-30 | runtime dự toán | tạm **38–42 h** cho tới khi có benchmark batch/val mới | `C9` giờ đo cả val pass |
| 2026-07-31 | seed 1337 / 2024 | **HOÃN** (§5-ter) | lý do là **thứ tự ưu tiên tài nguyên**, KHÔNG phải dấu của delta — §11 chỉ cấm điều kiện hoá theo dấu |
| 2026-07-31 | trạng thái nb16 | **one-seed exploratory NO-GO** | không được claim dương cho tới khi có seed nhân bản |
| 2026-07-30 | 3.1 rule `00:00:00` | **không loại gì** | `1` midnight < mức ngẫu nhiên `~2.4`; `0` missing; người dùng xác nhận là giờ thật |
| 2026-07-30 | 3.2 chi phí | **cam kết 3 seed**, `~38 h` train + `~3 h` staging | Pro+ 24 h ⇒ 1 seed/session vừa gọn; không cần đổi thiết kế |
| 2026-07-30 | `BATCH` | **128** (throughput `48.0 img/s`, VRAM 34.6%) | throughput **phẳng** `46.4→48.0` qua `32/64/128/256` ⇒ batching chỉ cho 3% |
| 2026-07-30 | scheduler | tail warmup+cosine · head **constant** | thừa từ nb12/13 — tránh head vừa mở đã bị cosine kéo xuống |
| 2026-07-30 | val loss | tính với `w = 1` cho mọi arm | `IS_BLANK` phủ cả val nên val loss theo weight arm sẽ khác nhau vì lý do không liên quan |
| 2026-07-30 | bootstrap cluster | `subject_id`, 2000 resample, báo `valid_fraction` | theo study cho CI **hẹp giả** (đã kiểm: `0.102` vs `0.151`) |
