# REVIEW — nb20 `20_view_selection_ehr_factorial.ipynb`

Plan: `PLAN_VIEW_SELECTION_EHR_FACTORIAL_V1.md` · Ngày: 2026-07-31

> **TRẠNG THÁI: Stage 1 (OOF-VAL) đã chạy xong. Cả ba cổng §10 FAIL. TEST KHÔNG chạy.**
> Kết quả và lý do ở **§7**–**§8**.

---

## §1 · Ba mục chặn

| # | mục | trạng thái |
|---|---|---|
| 1 | §10 đòi *"cùng chiều ≥ 2/3 image seed"* | **chỉ có seed 42** — xem §2 |
| 2 | TEST | **chưa từng chạm**, khoá sau `RUN_TEST = False` |
| 3 | §14 đặt tên `19_` | dùng `20_` vì `19_fov_ablation.ipynb` đã tồn tại |

---

## §2 · `p_F`/`p_L`: chỉ `seed 42 / VAL` có sẵn — và **`RUN_IMAGE_FORWARD` không cứu được**

| seed / split | thiếu gì | bật `RUN_IMAGE_FORWARD = True` có đủ? |
|---|---|---|
| `42 / VAL` | — (nb17 đã cache `perimg_val_R0_ep2_center518.npy`) | — |
| `42 / TEST` | chỉ thiếu lượt forward | **có** — ~6 phút GPU (nhưng TEST đang khoá) |
| `1337 / *`, `2024 / *` | **không có checkpoint `R0_s{seed}_ep2.pt`** | **KHÔNG** |

nb16 chỉ train seed 42. Seed `1337`/`2024` đã hoãn *(lý do: thứ tự ưu tiên tài nguyên, **không**
phải dấu của delta seed 42 — xem `REVIEW_NB16_PREGPU.md` §5-ter)*. Mỗi seed cần **~12.5 h GPU**
ở nb16 trước khi forward có ý nghĩa.

### Đã sửa

`C2` bản đầu in `"can RUN_IMAGE_FORWARD = True"` cho **cả 5** dòng thiếu — sai cho 4 dòng. Nay
`C2` kiểm sự tồn tại của `{label_reliability}/snapshots/R0_s{seed}_ep2.pt` và in đúng hai thông
điệp khác nhau.

---

## §3 · ⚠ `Q_CTX` bị nhiễm **kết quả lâm sàng** — đã sửa

### Đặc tả (§2.3)

```
Q_CTX  = demographics + care setting + missingness + measurement counts/recency
Q_FULL = Q_CTX + lab values + vital values + comorbidity + micro time-safe + clinical khac
```

Q_CTX cố ý **không** chứa kết quả — đó là điều làm cho `B1_CTX` vs `B1` tách được *shortcut bối
cảnh chăm sóc* khỏi *tín hiệu lâm sàng*.

### Lỗi

Bản đầu của `C3` để **5 feature mang kết quả** trong `Q_CTX`. Định nghĩa gốc ở
`02_ehr_features.ipynb`:

| feature | định nghĩa | bản chất |
|---|---|---|
| `t1_status` | `np.select([t1_any_positive, t1_any_cancelled, t1_any_no_growth], [1,2,3])` | `1` = **cấy DƯƠNG TÍNH** |
| `t4_n_prior_positive_ever` | `COUNTIF(status='POSITIVE')` | đếm kết quả dương |
| `t3_recency_bin` | `pd.cut(t3_days_since_last_positive, ...)` | dẫn từ dương tính |
| `t1_days_since_last_positive_mask` | `MIN(IF(status='POSITIVE', days_before, NULL))` | mask = **đã từng dương** |
| `t3_days_since_last_positive_mask` | như trên | như trên |

### Vì sao nghiêm trọng

Sai **đúng chiều tệ nhất cho câu hỏi đang mở**. Nếu `Q_CTX` chứa kết quả cấy máu thì `B1_CTX`
mạnh lên vì lý do **lâm sàng thật**, và kết quả `B1_CTX ≈ B1` sẽ bị đọc thành *"phần tăng chỉ là
shortcut bối cảnh"*. Tức là **thổi phồng phần shortcut** — đúng câu hỏi nb18 để lại về
`Fracture +0.0118` (nghi ED→chấn thương).

### Sửa

Năm feature trên chuyển sang `Q_FULL` với tiền tố `res_`. `Q_CTX` giữ lại:

| nhóm | giữ trong CTX | lý do |
|---|---|---|
| `demo_*` | nếu cohort có | §2.3 |
| `m_lab_*`, `m_vit_*` | mask "có được **đo** không" | đúng spec `has_hemoglobin`… |
| `m_rec_t1_days_since_last_specimen` | "đã từng **lấy mẫu**" | hành vi đặt lệnh, không phải kết quả |
| `cnt_t2_n_pending`, `cnt_t3_n_specimens`, `cnt_t4_n_prior_encounters`, `cnt_t4_years_since_first_culture` | đếm **đo lường** / mức sử dụng y tế | không phải kết quả |
| `bin_t2_pending`, `bin_t2_time_imprecise`, `bin_t1_same_encounter` | đặt lệnh / chất lượng bản ghi | — |
| `ctx_setting` | inpatient / ED / outpatient | **nơi** chăm sóc |

Giá trị `t1/t3_days_since_last_positive` vốn đã ở `FULL`; nay mask của chúng cũng vậy.

### Chốt chặn trong code

```python
_FORBID_CTX = {'ctx_status', 'ctx_recency_bin', 'cnt_t4_n_prior_positive_ever',
               'm_rec_t1_days_since_last_positive', 'm_rec_t3_days_since_last_positive'}
assert not (_FORBID_CTX & set(NM_C))
assert set(NM_C).isdisjoint(NM_F)
```

Assert theo **tên chính xác**, không theo chuỗi con — tránh báo động giả với tên xét nghiệm kiểu
`inorganic phosphate`. Kèm một cảnh báo mềm theo chuỗi con để bắt trường hợp phát sinh.

Kiểm offline 5/5 với `ehr_f` tổng hợp: CTX không còn tên cấm; CTX ∩ FULL rỗng; cả ba giá trị
recency vẫn ở FULL; `COMPLETE` (độ đầy EHR, dùng cho phân tầng hoán vị §6.3) nay tính trên mask
missingness của lab/vital + 1 mask lấy mẫu, thay vì lẫn 3 mask recency.

---

## §4 · Các sai lệch khác so với plan *(ghi trong `experiment_manifest.json`)*

| sai lệch | ghi chú |
|---|---|
| demographics | `C3` dò `anchor_age`/`age`/`gender`/`sex`; nếu cohort không có thì `Q_CTX` thiếu mục này so với §2.3 và cell in cảnh báo |
| `β` init | `σ(−8) = 3.354e-04`, không bằng `0` — xem §4-bis |
| `λ_β` | chọn bằng **macro mAP** trên inner-held-out; §7.2 chỉ cấm fit trên outer-held-out, không chỉ định chỉ số |
| tên notebook | `20_` thay vì `19_` |

### Một lệch **nội tại của plan**, không phải của hiện thực

§8 tối ưu **BCE**; §9.1 chấm bằng **Macro mAP**. Hai thứ không đồng hướng ở nhãn hiếm. nb17 `V1`
tối ưu **thẳng AP** bằng lưới và được `+0.00017`; `A1` (BCE) có thể **kém hơn** trên chính
endpoint. Notebook làm theo plan và báo cáo cả hai để đọc được chênh lệch.

---

## §4-bis · **AMENDMENT 2026-07-31** — cổng §8 `init == image-only`

### Sự việc

`C6` chạy xong `B0`/`B1`/`B1_CTX`/`B1_SHUF` rồi FAIL ở cổng §8:

```text
Spearman min 0.999999   -> PASS (>= 0.9999)
|dAP| max   6.03e-05    -> FAIL (nguong 1e-5 toi dat)
```

Vế Spearman **đạt**; vế `|ΔAP|` fail vì ngưỡng `1e-5` là tôi đặt tuỳ tiện.

### Nguyên nhân: mâu thuẫn nội tại của plan

Plan §5.3 ghi `b_c = -8` ⇒ `beta_c ≈ 0`, còn §8 assert `init logits == image-only`. Với
`σ(−8) = 3.354e-04` thì **không thể cùng đúng**.

AP là chỉ số **thứ hạng**. Ở `n ≈ 15,000`, khoảng cách logit kề nhau cỡ `3.5e-4` — **cùng bậc**
với nhiễu `β·u`. Nên nhiều cặp kề bị hoán vị: Spearman gần như không đổi (hoán vị kề), AP dịch
một chút. `6.03e-05` là **hệ quả tất yếu**, không phải lỗi.

### Sửa lập luận sai của tôi

Tôi đã đề xuất rằng nhiễu init là **common-mode** và triệt tiêu trong `B1 − B0`. **Sai.** Các
arm có baseline và vector `u` khác nhau:

| arm | residual cộng lên | dùng `u` |
|---|---|---|
| `B0` | `A0` | `u_FULL` |
| `B1` | `A1` | `u_FULL` |
| `B1_CTX` | `A1` | `u_CTX` |
| `B1_SHUF` | `A1` | `u` đã hoán vị |

Ranking nền khác nhau ⇒ **không** triệt tiêu chính xác. Nhiễu phải bị chặn bằng **ngưỡng tuyệt
đối trên từng arm**, không được dựa vào triệt tiêu.

### Nội dung amendment *(đã ghi vào plan §5.3 và §8, và vào `experiment_manifest.json`)*

**Giữ `b_init = -8`.** Không hạ xuống `-12`: `dβ/db = β(1−β)` bằng `3.35e-04` ở `−8` nhưng chỉ
`6.14e-06` ở `−12` — nhỏ hơn ~55 lần, đủ để LBFGS gần như không mở được residual branch.

**Thay exact-identity bằng near-identity khai báo trước:**

```text
Spearman toi thieu   >= 0.9999
max per-label |dAP|  <= 1e-4      (= 5% cua cong thuc dung +0.002)
```

Quan sát `6.03e-5` = **`3.0%`** của cổng `+0.002` ⇒ đạt near-identity sau amendment. **Không cần
train lại** `B0`/`B1`/`B1_CTX`/`B1_SHUF`.

**Báo cáo cả bốn** `Macro / Rare-3 / NonTarget / max per-label`, không chỉ giá trị lớn nhất.

**Kiểm đúng cặp baseline:** `B0↔A0`, `B1↔A1`, `B1_CTX↔A1`, `B1_SHUF↔A1`. **Không** assert
`B0 init == B1 init`. Thứ phải giống nhau chỉ là `beta_init` — được assert riêng.

**Chẩn đoán `|u|`:** `p50/p90/p99/p999/max` cho `Q_FULL`, `Q_CTX`, và từng nhãn, kèm
`max|Δz| = β_init·max|u|`. Mô phỏng với `u ~ N(0,1)` chỉ cho `|ΔAP| ≈ 2–6 × 10⁻⁶`, nhỏ hơn quan
sát ~10 lần — giả thuyết là `u` có **đuôi nặng** (`logit(q)` của XGBoost khi `q` sát `0`/`1`).
Bảng này **đo** thay vì giả định.

**Scale diagnostic `b = −12`** *(diagnostic only)*: `Spearman ≥ 0.99999`, `max|ΔAP| ≤ 1e-5`, kèm
kiểm số học `delta12 == (β12/β8)·delta8` (`rtol 1e-6`). Nếu `−12` **không** làm nhiễu nhỏ đi thì
dừng và kiểm 5 khả năng: sai paired baseline · prob/logit không nhất quán · dùng nhầm fitted beta
· lệch hàng dự đoán · clipping/standardization khác đường.

> **Cảnh báo trước khi chạy:** `|ΔAP|` **không** co tuyến tính theo `β` — nó là hàm bậc thang của
> số cặp bị hoán vị. Mô phỏng cho thấy `β` giảm `55×` chỉ làm `|ΔAP|` giảm `~6×`. Nếu thực tế
> cũng vậy thì `6.03e-5 / 6 ≈ 1.0e-5`, **sát đúng** ngưỡng `DAP_SCALE_MAX = 1e-5`. Thông điệp
> assert đã tách sẵn hai nguyên nhân — *"không co"* (lỗi thật, dừng lại) vs *"co đúng tỉ lệ nhưng
> vẫn trên ngưỡng"* (sàn lượng tử hoá của AP, chỉ cần chỉnh ngưỡng scale và ghi amendment).

### Một cổng bị sửa sau smoke test

Bản đầu tôi assert `|LG(P_init) − LG(P_img) − β·u| < 1e-9`. Smoke test cho thấy nó **báo động
giả**: `LG` kẹp ở `[1e-6, 1−1e-6]`, nên ở các mức bão hoà đẳng thức vỡ — residual đúng bằng
`β·max|u|`. Đó là giả tạo của **phép kiểm**, không phải của đường model (`nested_ehr` trả về
**xác suất**, không `LG` lại).

Đổi sang: `residual ≤ β_init·max|u|`. Kiểm negative xác nhận nó vẫn bắt được lỗi thật — dùng nhầm
`β = 0.3` (fitted) thay vì init cho residual `8.57e-01` ≫ ngưỡng `9.59e-04`.

Cell cũng in tỉ lệ mức chạm trần `LG` để biết bao hoà ảnh hưởng bao nhiêu.

### Phạm vi

**Không** đổi model, dữ liệu, hyperparameter huấn luyện, `β` initialization, endpoint, hay
success gate khoa học. Đây là sửa một **mâu thuẫn kỹ thuật trong plan**.

---

## §5 · Kỳ vọng trước khi chạy

**Nhánh view nhiều khả năng null.** nb17 `V3` đã chạy một **siêu tập** của `A1` (pooler học được,
8 hệ số/nhãn, cross-fit) → macro `+0.00047`, CI chứa 0. `A1` hẹp hơn (một `α_c`), khó vượt hơn.
Giá trị thật của nhánh này là **attribution** mà nb17 không có: `AG` (chung vs theo bệnh), `AV`
(có lateral vs nội dung lateral), `AS` (lateral đã xáo).

**Nhánh EHR là phần đáng giá.** `B1_CTX` vs `B1` và `B1_SHUF` là hai phép thử tách được nghi vấn
còn treo của nb18 — và §3 ở trên là điều kiện để chúng đọc được.

---

## §6 · Chạy

| bước | phần cứng | thời gian |
|---|---|---|
| `C0`–`C2` | CPU (tái dùng cache nb17) | vài phút |
| `C3` (2 × 13 XGBoost) | CPU | **8–15 phút** |
| `C4`–`C7` (`A1`, `AS`, `B*`, bootstrap) | CPU | `A1` **3–6 ph** · `AS` **15–30 ph** |
| `C8` TEST | GPU ~6 ph/seed | **khoá** — `RUN_TEST = False` |

Không cần GPU cho toàn bộ Stage 0 + Stage 1.

---

## §7 · Kết quả OOF-VAL *(seed 42, 31,670 study, 8,956 subject)*

### 7.1 · Bảng arm

| arm | macro | rare3 | nontgt | tail | worst3 |
|---|---|---|---|---|---|
| `A0` | 0.4365 | 0.1467 | 0.5234 | 0.1191 | 0.1339 |
| `AG` | 0.4365 | 0.1468 | 0.5234 | 0.1192 | 0.1339 |
| `A1` | 0.4365 | 0.1468 | 0.5234 | 0.1192 | 0.1339 |
| `AV` | 0.4355 | **0.1512** | 0.5208 | 0.1224 | 0.1374 |
| `AS` | 0.4355 | **0.1512** | 0.5208 | 0.1223 | 0.1374 |
| `B0` | 0.4374 | 0.1467 | 0.5246 | 0.1191 | 0.1339 |
| `B1` | **0.4375** | 0.1468 | 0.5246 | 0.1192 | 0.1339 |
| `B1_CTX` | 0.4369 | 0.1468 | 0.5239 | 0.1192 | 0.1339 |
| `B1_SHUF` | 0.4367 | 0.1468 | 0.5237 | 0.1192 | 0.1339 |

### 7.2 · Cổng §10 — **cả ba FAIL**

| cổng | quan sát | ngưỡng | |
|---|---|---|---|
| `view_primary` `A1−AG` macro | `+0.00001` | `≥ +0.001` | **FAIL** — bằng 1/100 cổng |
| `ehr_primary` `B1−A1` macro / rare3 | `+0.00095` / `+0.00000` | `≥ +0.002` | **FAIL** |
| `combined` `B1−A0` macro | `+0.00098` | `≥ +0.002` | **FAIL** |
| `2/3 seed` | 1 seed | ≥ 2/3 | **không đánh giá được** |

### 7.3 · Nhánh view: null **dứt khoát**, và cơ chế đã rõ

`α` học được nằm trong `0.7035–0.7044` cho **cả 13 nhãn**, `sd` qua fold `0.0002–0.0004`. Init là
`0.70` ⇒ `α` **không nhúc nhích**. Không có nghiệm nào chạm biên (`0.5`: 0.0% · `1.0`: 0.0%).

Quan trọng: inner-CV chọn `λ_g = 0.01` thay vì `0` ở **mọi** fold, dù `0` **có** trong lưới. Tức
dữ liệu tự nói rằng co về init tốt hơn để `α` tự do. **Null thật, không phải regularizer ép** —
đây là điều phân biệt kết luận này với một artifact.

Kết hợp với nb17 `V3` (siêu tập của `A1`, 8 hệ số/nhãn, `+0.00047` CI chứa 0): hai thiết kế độc
lập, hai null. **Đóng nhánh view-pooling.**

### 7.4 · Nhánh EHR: tín hiệu **thật nhưng nhỏ** — và trả lời được nghi vấn của nb18

```text
B1 - B1_SHUF       +0.00075  CI [+0.00025, +0.00126]  LOAI 0
B1 - B1_CTX        +0.00060  CI [+0.00012, +0.00106]  LOAI 0
B1_CTX - A1        +0.00035  CI [+0.00027, +0.00044]  LOAI 0
```

Phần **lâm sàng** chiếm `~63%`, **bối cảnh chăm sóc** `~37%`. Nghi vấn còn treo của nb18 —
*"`Fracture +0.0118` có phải chỉ là shortcut ED→chấn thương"* — được trả lời: **shortcut là thiểu
số**. Phép thử này chỉ đọc được nhờ §3 đã dọn kết quả cấy máu ra khỏi `Q_CTX`.

`B1 > B1_SHUF` với CI loại 0 ⇒ nội dung EHR là thật, không phải artifact của thủ tục.

**Nhưng `β` học được bằng đúng `0.0000`** cho toàn bộ nhãn `medium`/`tail`:

| nhãn | `β` |
|---|---|
| `No_Finding` | 0.2554 |
| `Pleural_Effusion` | 0.1265 |
| `Cardiomegaly` | 0.1072 |
| `Lung_Opacity` · `Atelectasis` · `Pneumonia` · `Edema` | 0.095 → 0.065 |
| `Fracture` · `Lung_Lesion` · `Pleural_Other` · `Enlarged_Card.` · `Consolidation` · `Pneumothorax` | **0.0000** |

EHR chỉ giúp 7 nhãn **head** — nơi ảnh vốn đã mạnh. Optimizer **chủ động đóng** residual ở đúng
những nhãn nb18 báo có lợi. Hai notebook mâu thuẫn nhau ở đây; nb20 có cross-fit chặt hơn và
control hoán vị, nb18 thì không.

Trần EHR-only (`C3`) cho thấy pipeline đặc trưng **không** hỏng: `AUROC_full` trung bình `0.7113`
(`Pneumothorax 0.8054`, `No_Finding 0.7842`), `Q_CTX` `0.6780`. EHR biết nhiều thứ — nhưng **ảnh
đã biết gần hết**.

### 7.5 · `AV ≈ AS` — kết quả bất ngờ nhất

`AV` (chỉ frontal + chỉ báo có lateral) và `AS` (lateral **đã xáo**) cho **cùng một kết quả**,
trùng tới 4 chữ số trên 5 chỉ số. Và trên `rare3` cả hai **thắng** `A1`:

```text
view_availability  A1 - AV  rare3  -0.00432  CI [-0.00775, -0.00032]  LOAI 0
view_content       A1 - AS  rare3  -0.00431  CI [-0.00782, -0.00028]  LOAI 0
```

Nói thẳng: **gộp lateral vào pool đang làm hại Rare-3 khoảng `+0.0043`**, và cái giá là `nontgt`
giảm `0.0026`. Trên macro thì hoà (`−0.0010`).

Đây là hướng **ngược** với dấu mà contrast được đăng ký trước kỳ vọng. Không được claim từ đây —
nhưng nó là ứng viên rõ nhất cho một thí nghiệm **đăng ký mới**.

### 7.6 · Interaction: cộng tính

`(B1−A1) − (B0−A0)` = `+0.00001` macro, `−0.00000` rare3. View và EHR **không** tương tác. Kết
hợp với 7.3, `B1−A1 ≈ B0−A0`: phần EHR đóng góp không phụ thuộc arm view.

### 7.7 · ⚠ Hai câu tự-diễn-giải của `C7` **quá rộng rãi**

Cell in:

```text
A1-AS +0.00104 -> co the claim lateral CONTENT
A1-AV +0.00102 -> A1 khac AV
```

**Hai dòng này là cổng ngôn từ** — chúng quyết định được phép viết phát biểu nào:

> **A.** *"Model khai thác **nội dung** phim nghiêng"* — kết quả thị giác.
>
> **B.** *"Model khai thác **việc có** phim nghiêng"* — **không** phải thị giác, mà là rò rỉ quy
> trình chăm sóc. Ai được chụp thêm phim nghiêng tương quan với luồng ngoại trú/PA, thể trạng,
> mức nghi ngờ lâm sàng. Viết nhầm B thành A là sai nghiêm trọng.

Logic cài vào cell: `A1 − AS > 0` ⇒ được claim **A**; `A1 − AV ≈ 0` ⇒ chỉ được claim **B**.

**Cả hai câu in ra đều đọc sai**, vì cell chỉ kiểm **dấu** — bỏ qua CI và bỏ qua `rare3`.

Câu thứ nhất: trên **macro** thì `CI [-0.00005, +0.00222]` **chứa 0** ⇒ chưa được claim gì; trên
**rare3** thì `−0.00431`, **ngược dấu và loại 0** ⇒ lateral content đang **làm hại**.

Câu thứ hai bỏ sót so sánh thật sự mang thông tin: **`AV` vs `AS`**, không phải `A1` vs `AV`.

**Lưu ý về `AV ≈ AS`.** Việc chúng trùng nhau **không** bác bỏ giá trị của `AS` như một control —
nó chính là điều kỳ vọng nếu `α_AS` sập về `1.0` (bỏ hẳn lateral), khiến cả hai arm quy về
frontal-only. Khi đó `A1 − AS` **vẫn** là phép đo hợp lệ cho *"pool lateral thật so với không
pool"*. Vấn đề của dòng in ra là **CI và dấu trên `rare3`**, không phải tính hợp lệ của control.
Nhưng vì lý do đó, §7.8 phải được chạy trước.

Câu đúng để viết:

> Trên macro, `A1 − AS` dương nhưng CI chứa 0 — **chưa đủ** để claim lateral content. Trên
> Rare-3, hướng **ngược lại** và loại 0: bỏ nội dung lateral (`AV`) **tốt hơn** `A1` khoảng
> `+0.0043`.

### 7.8 · Phải kiểm trước khi tin 7.5

`AV` và `AS` trùng tới 4 chữ số trên 5 chỉ số, qua **hai đường code khác nhau** — đáng nghi.

Giải thích tự nhiên: lateral đã xáo là nhiễu thuần ⇒ `α_AS → 1.0` (bỏ hẳn lateral) ⇒ thu về
frontal-only ⇒ trùng `AV`. Nhưng `C5` **chỉ in `α` cho arm `A1`**, không in cho `AS`.

**Đề xuất (chưa áp dụng):** in `α` của arm `AS` và tỉ lệ chạm biên `1.0`, dùng lại `VIEW_PAR` —
vài giây CPU. `α_AS ≈ 1.0` ⇒ giải thích được. `α_AS ≈ 0.70` ⇒ có bug phải tìm trước khi dùng 7.5.

---

## §8 · Quyết định TEST: **KHÔNG chạy**

### Lý do

1. **Cả ba cổng §10 FAIL trên VAL.** §7.1 của plan định nghĩa TEST là xác nhận **một lần** sau
   khi khoá. Chạy nó trên một null là tiêu suất một-lần để xác nhận điều đã biết.
2. **Cổng `2/3 seed` không đánh giá được** — chỉ có seed 42, và `1337`/`2024` cần `12.5 h` GPU mỗi
   seed ở nb16.
3. Chạy TEST mở ra cám dỗ nhìn per-label TEST — đúng thứ **§13 cấm hành động theo**. Rủi ro lớn
   hơn giá trị.

`C8` giữ `RUN_TEST = False`. `test_touched: false` trong manifest. Hash TEST manifest
`0f1abff04c75a2cb…` chưa dùng ở bất kỳ bước fit nào.

### Điều gì sẽ mở khoá TEST

Một arm vượt `+0.002` trên VAL với **≥ 2 seed**. Hiện chưa có ứng viên nào từ nb20.

### Việc còn lại theo thứ tự

1. **Kiểm `α_AS`** (§7.8) — vài giây CPU. Điều kiện để 7.5 đọc được.
2. Nếu 7.5 đứng vững: **đăng ký một thí nghiệm mới** cho hướng *"bỏ lateral content, giữ chỉ báo
   availability, tối ưu cho Rare-3"*. Contrast chính `AV − A0` trên `rare3`, không phải macro.
3. `CODE_COMMIT` vẫn `None` trong manifest.

### Đã đóng sau notebook này

| nhánh | trạng thái |
|---|---|
| view-pooling học được (`α_c` theo bệnh) | **đóng** — nb17 `V3` và nb20 `A1` cùng null, `α` không dịch khỏi init |
| EHR residual trên kiến trúc hiện tại | **đóng ở mức này** — `+0.00095` macro, `β = 0` cho mọi nhãn rare |
| nghi vấn shortcut care-context của nb18 | **trả lời**: lâm sàng `63%`, bối cảnh `37%` |
