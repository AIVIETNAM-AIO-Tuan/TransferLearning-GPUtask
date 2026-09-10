# REVIEW — `LABEL_PIPELINE_V1_DETAILED_PLAN.md`

**Trạng thái:** đã triển khai đầy đủ thành `21_label_pipeline_v1.ipynb` (25 cell, Stage 0→3).
Toàn bộ logic CPU (C2/C3/C4/C8/C9/C10) đã chạy thử end-to-end trên cohort tổng hợp dựng đúng
marginal thật của `cohort_v2` — pass hết assertion.

**Phạm vi dữ liệu:** chỉ cohort đã select (`cohort_v2/cohort_study_level.parquet`, 211,130 study —
train 147,791 / val 31,670 / test 31,669). Nhãn thô CheXpert chỉ được *reindex* lên đúng
`study_id` của cohort; **không** đụng tới phần MIMIC-CXR ngoài cohort. Blank rate tính trên
147,791 study TRAIN **của cohort**, không phải trên toàn bộ CXR.

---

## 1. Những gì plan nói rõ → đã code y nguyên

| Plan | Cell | Ghi chú |
|---|---|---|
| §1 4 raw state | C1 | `STATE ∈ {1, 0, -1, 2}`, assert `U-Ones(raw) == nhãn cohort` |
| §2 `-1` disease-aware | C4/C8 | Atelectasis+Edema → hard 1 · 10 nhãn → teacher soft · No_Finding → mask |
| §3 blank routed 30% | C2 | TRAIN-only, khoá một lần, ghi `blank_policy_by_label.json` |
| §5 teacher supervision + loss | C4/C6 | `CellWASL(x, y, w=M_T)` ⇒ đúng `Σ M·ASL / (B×13)` |
| §6 5-fold theo subject | C3 | iterative stratification, subject-disjoint, seed 42 |
| §7 OOF prob | C7 | pool `0.7·F + 0.3·L` (view pooling khoá của R0), clip `[1e-4, 1-1e-4]`, 4 assert |
| §8 manifest | C8 | `resolve_target` vector hoá, long-format parquet + 4 ma trận `.npy` |
| §10 student loss | C10 | `ASL(hard) + 0.5·BCE(soft)`, mẫu số `B×13`, test masked-grad == 0 |
| §15 diagnostics | C2/C4/C9 | raw summary · teacher coverage/fold · soft summary · effective prevalence · leakage |
| §16 artifacts | tất cả | đủ 11 file + 5 checkpoint, `experiment_manifest.json` có hash |

Không đổi: RAD-DINO, blocks 0–8 frozen, 9–11 + backbone LN + ML-Decoder patch-only, 518px,
augmentation, subject split, view pooling 0.7/0.3, ASL `(γ+ 1, γ- 4, clip 0.05)`, LR 1e-5,
EFF 256, 8 epoch, head-warmup 1 epoch, evaluation labels.

---

## 2. 🔴 PHÁT HIỆN CHẶN — ngưỡng blank 30% **không phân loại được gì**

Blank rate thật trên TRAIN của cohort (tính từ số liệu đã in ở nb16 C2):

| label | n_blank | **blank_rate** | route theo plan | teacher hard cells | %TRAIN | **%positive trong hard** | prevalence hist. |
|---|---:|---:|---|---:|---:|---:|---:|
| Pleural_Other | 145,945 | **0.9875** | TEACHER_SOFT | 1,383 | 0.9% | **94.0%** | 0.0119 |
| Fracture | 143,993 | **0.9743** | TEACHER_SOFT | 3,451 | 2.3% | **83.0%** | 0.0217 |
| Lung_Lesion | 142,518 | **0.9643** | TEACHER_SOFT | 4,564 | 3.1% | **88.8%** | 0.0322 |
| Enlarged_Cardiomediastinum | 133,619 | **0.9041** | TEACHER_SOFT | 8,091 | 5.5% | 58.9% | 0.0734 |
| Consolidation | 132,668 | **0.8977** | TEACHER_SOFT | 12,306 | 8.3% | 57.8% | 0.0672 |
| Pneumothorax | 112,431 | **0.7607** | TEACHER_SOFT | 34,598 | 23.4% | 19.6% | 0.0511 |
| Pneumonia | 110,073 | **0.7448** | TEACHER_SOFT | 25,751 | 17.4% | 41.3% | 0.1529 |
| Atelectasis | 109,822 | **0.7431** | TEACHER_SOFT | 37,969 | 25.7% | **97.4%** | 0.2502 |
| Lung_Opacity | 109,684 | **0.7422** | TEACHER_SOFT | 35,676 | 24.1% | **94.6%** | 0.2448 |
| Edema | 104,976 | **0.7103** | TEACHER_SOFT | 42,815 | 29.0% | 62.0% | 0.1797 |
| Cardiomegaly | 104,474 | **0.7069** | TEACHER_SOFT | 39,502 | 26.7% | 74.7% | 0.2254 |
| Pleural_Effusion | 91,081 | **0.6163** | TEACHER_SOFT | 52,901 | 35.8% | 67.8% | 0.2685 |
| No_Finding | 99,348 | 0.6722 | HARD_ZERO *(cưỡng bức §3)* | 147,791 | 100% | 32.8% | 0.3278 |

**Blank rate thấp nhất là 0.6163 (Pleural_Effusion).** Ngưỡng 0.30 vì thế không phải một router —
nó là hằng số: **12/12 nhãn pathology đều rơi vào `TEACHER_SOFT`**. Mọi ngưỡng trong khoảng
`(0.30, 0.616]` cho ra kết quả **hoàn toàn giống nhau**. Nhánh `blank_rate <= 30% → hard 0` trong
plan §3/§4/§18 là code chết trên dataset này.

### Ba hệ quả kéo theo

**(a) Teacher mất gần hết negative.** Vì blank bị mask ở teacher (§5), teacher chỉ còn học trên
mention tường minh. Với Pleural_Other: 1,383 cell hard / 147,791 study = **0.9% coverage, 94% là
positive**. Atelectasis 97.4% positive, Lung_Opacity 94.6%. ASL trên một tập gần như toàn positive
sẽ đẩy `p → 1` cho *mọi* study. Plan §14 nói "teacher chỉ cần không collapse" — đây chính là kịch
bản collapse, và nó dự đoán được trước khi chạy chứ không phải rủi ro giả định.

**(b) 75% supervision biến thành soft.** 1,441,284 blank cell + 33,201 uncertain cell =
**1,474,485 / 1,921,283 = 76.7%** số cell TRAIN lấy target từ teacher. Đây không còn là "sửa cách xử
lý nhãn" ở rìa — nó thay gần như toàn bộ tín hiệu huấn luyện.

**(c) Effective prevalence gần như chắc chắn vỡ trần.** `π_eff = (Σ hard_pos + Σ soft)/N_TRAIN`.
Nếu teacher trả trung bình `p̄` trên blank của Pleural_Other thì
`π_eff ≈ 0.0088 + 0.9875·p̄`. Chỉ cần `p̄ = 0.05` là đã `π_eff ≈ 0.058` = **4.9× historical**, vượt
ngưỡng cảnh báo `>2×` của §15. Với `p̄` thực tế (teacher lệch positive) con số sẽ tệ hơn nhiều.
Plan §15 quy định "cảnh báo không tự đổi policy" → notebook chỉ in cảnh báo và chạy tiếp.

### Ước tính chi phí nếu chạy nguyên trạng

Đo trên chính máy đã chạy nb16 — **NVIDIA RTX PRO 6000 Blackwell, 102 GB VRAM** (47.4 img/s cho
3 arm @B=128, tức ~142 img/s cho 1 arm):

* Stage 0 (C1–C4): **CPU thuần, ~1–2 phút**, không cần GPU.
* Stage 1 (C7, 5 teacher × 8 epoch × ~194k ảnh): **≈ 15–16 h GPU**, peak VRAM ~13 GB (1 arm).
* Stage 3 (student 3 seed × 8 epoch): **≈ 13–14 h GPU**.
* Tổng **≈ 30 h GPU** cho một kết quả mà cơ chế hỏng đã nhìn thấy từ Stage 0.

---

## 2-bis. Trong phần nhãn **có tồn tại**, bao nhiêu là 1 thật / 0 thật?

Chỉ tính TRAIN của cohort. `n_mention = n(+1) + n(0) + n(-1)`; `prev_MAR` = prevalence sẽ ra nếu
blank là *missing-at-random* (tức blank có cùng tỉ lệ 1/0 như phần được nhắc tới).

| label | n_mention | %cohort | **%+1** | **%0 (0 thật)** | %-1 | pos:neg | prev hist. | prev nếu MAR | blow-up |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Pleural_Other** | 1,846 | 1.25% | **70.4%** | **4.5%** | 25.1% | **21.2 : 1** | 0.0119 | 0.955 | **80×** |
| **Fracture** | 3,798 | 2.57% | **75.4%** | **15.4%** | 9.1% | 5.5 : 1 | 0.0217 | 0.846 | **39×** |
| **Lung_Lesion** | 5,273 | 3.57% | **76.8%** | **9.7%** | 13.4% | 9.3 : 1 | 0.0322 | 0.903 | **28×** |
| Enlarged_Cardiomediastinum | 14,172 | 9.59% | 33.6% | 23.5% | 42.9% | 3.3 : 1 | 0.0734 | 0.766 | 10× |
| Consolidation | 15,123 | 10.23% | 47.1% | 34.3% | 18.6% | 1.9 : 1 | 0.0672 | 0.657 | 9.8× |
| Pneumothorax | 35,360 | 23.93% | 19.2% | **78.6%** | 2.2% | 0.3 : 1 | 0.0511 | 0.214 | 4.2× |
| Pneumonia | 37,718 | 25.52% | 28.2% | 40.1% | 31.7% | 1.5 : 1 | 0.1529 | 0.599 | 3.9× |
| Atelectasis | 37,969 | 25.69% | 79.5% | **2.6%** | 17.9% | 37.1 : 1 | 0.2502 | 0.974 | 3.9× |
| Lung_Opacity | 38,107 | 25.78% | 88.6% | 5.1% | 6.4% | 18.8 : 1 | 0.2448 | 0.949 | 3.9× |
| Edema | 42,815 | 28.97% | 41.6% | 38.0% | 20.4% | 1.6 : 1 | 0.1797 | 0.620 | 3.5× |
| Cardiomegaly | 43,317 | 29.31% | 68.1% | 23.1% | 8.8% | 3.3 : 1 | 0.2254 | 0.769 | 3.4× |
| Pleural_Effusion | 56,710 | 38.37% | 63.2% | 30.0% | 6.7% | 2.3 : 1 | 0.2685 | 0.700 | 2.6× |
| No_Finding | 48,443 | 32.78% | 100% | 0% | 0% | — | 0.3278 | 1.000 | 3.1× |

### Đọc bảng này

**Với Pleural_Other, phần nhãn tồn tại gần như toàn là 1 thật.** 1,846 mention = 1.25% cohort;
trong đó 70.4% là `+1`, **chỉ 4.5% (83 study) là `0` thật**, 25.1% là `-1`. Tỉ lệ positive : explicit
negative = **21 : 1**. Fracture 5.5:1, Lung_Lesion 9.3:1, Atelectasis 37:1, Lung_Opacity 19:1.

Đây là **reporting bias kinh điển**: bác sĩ chỉ ghi các dấu hiệu hiếm khi *có*; không ai viết
"không thấy tổn thương màng phổi khác". Kênh nhãn của các nhãn hiếm vì thế **hầu như không mang
bằng chứng negative nào** — toàn bộ negative phải đến từ blank.

**Cột `blow-up` là phản chứng trực tiếp cho giả thiết "blank đáng ngờ".** Nếu blank thật sự mơ hồ
theo kiểu missing-at-random, prevalence Pleural_Other sẽ là **95.5%** — tức 95 trên 100 phim ngực có
tổn thương màng phổi khác. Prevalence lâm sàng thực tế của nhãn này ~1%. Không có ngưỡng blank nào
cứu được điều đó: giả thiết MAR sai với **cả 12/12 nhãn**, nhẹ nhất cũng đã sai 2.6× (Pleural_Effusion).
Nói cách khác **blank ≈ 0 thật** là giả định đúng, và đó chính là lý do U-Ones lịch sử cho AP dùng được.

Ngoại lệ duy nhất đáng chú ý là **Pneumothorax**: 78.6% mention là `0` tường minh (pos:neg = 0.3:1) —
bác sĩ có thói quen ghi rõ "no pneumothorax". Đây là nhãn duy nhất mà kênh nhãn tự nó đủ negative;
cũng là nhãn mà blank ít mang thông tin nhất.

### Điều bảng này **không** trả lời được

Không có ground truth nào trong file nhãn cho biết *bao nhiêu blank thật sự là positive bị bỏ sót*.
Bằng chứng duy nhất là kiểm tra prevalence ở trên, và nó chỉ nói được "blank không thể mang tỉ lệ
positive giống mention". Muốn ước lượng thẳng tỉ lệ omission thì cần nguồn ngoài kênh nhãn — đọc
report text, hoặc anchor thời gian như R2 của nb16 (nhánh đó đã đo: 4.5–7.9% blank của Rare-3 có
anchor `+1` trong cùng subject ±7 ngày, tức omission thật có tồn tại nhưng ở mức **vài phần trăm**,
không phải 70–95%).

**Hệ quả cho Q1:** teacher trong plan chỉ học trên mention, tức nó học `P(y=1 | có nhắc tới)` ≈
0.70–0.95 cho nhóm hiếm, rồi con số đó bị đóng dấu lên 97% cell blank. Blow-up 28–80× của Rare-3
trong bảng trên chính là mức lệch prevalence sẽ xảy ra. Con số 4.5–7.9% từ R2 mới là bậc độ lớn
đúng của phần blank cần nghi ngờ.

---

## 3. Chỗ plan chưa nói rõ — tôi đã tự quyết, cần anh xác nhận

| # | Điểm mờ | Quyết định trong nb21 | Lý do |
|---|---|---|---|
| 1 | §6.3 "same checkpoint rule" nhưng rule của R0 (`select_ckpt`) cần metric VAL. Dùng fold held-out để chọn checkpoint rồi lại predict chính fold đó = rò rỉ. | `TEACHER_CKPT_RULE = 'fixed_last_epoch'`: train đủ 8 epoch, lấy epoch cuối, **không** chọn bằng held-out fold. | Giữ OOF sạch tuyệt đối. Phương án thay thế (cắt inner-val từ 4 fold) làm teacher mất 20% dữ liệu và lệch khỏi "train trên 4 folds" của §6.2. |
| 2 | §8 pseudocode chặn `raw==1/0` **trước** nhánh `No_Finding`, nên `no_finding_positive` không bao giờ reachable; §5 lại mô tả "No Finding +1 → 1". | Theo đúng pseudocode: No_Finding `+1` → `explicit_positive`, `0` → `explicit_negative`, chỉ blank → `no_finding_negative`. | Pseudocode là artifact chuẩn tắc. `final_target` giống nhau ở cả hai cách; chỉ khác `target_reason`. Cohort có 0 cell `No_Finding == 0` nên khác biệt là hình thức. |
| 3 | §6.1 "cân bằng multilabel positive/explicit-negative counts" không nói thuật toán. | Iterative stratification ở mức group=subject, cột hiếm nhất rải trước, tie-break bằng fold thiếu study nhất. Ma trận cân bằng = `[positives ‖ explicit_negatives]` (26 cột). | Pleural_Other chỉ có **83 explicit negative** trên toàn TRAIN → ~17/fold. Nếu không rải cột này trước thì gate §15 (fold-label phải có cả pos lẫn neg) rất dễ fail. Dry-run xác nhận: fold size 5913/5911/5911/5912/5911, gate fail = 0. |
| 4 | §7 "mỗi TRAIN study có đúng một OOF prediction" nhưng study không có ảnh nào staged thì không thể predict. | Assert cứng ở C5, STOP kèm danh sách `study_id` nếu thiếu. | Không tự bịa fallback. Plan không cho phép study nào thiếu target. |
| 5 | §5 mẫu số cố định `B_actual × 13` **kể cả cell masked**. | Giữ nguyên. | Đúng plan, và giống hằng số chuẩn hoá của nb11–16 nên LR không đổi nghĩa. Lưu ý: với Pleural_Other (0.9% cell có mask=1) gradient hiệu dụng nhỏ hơn ~100× so với chuẩn hoá theo số cell — đây là hệ quả có chủ đích của plan, không phải bug. |
| 6 | §9 map study→image. | Mọi ảnh trong cùng study nhận cùng vector `final_target` và cùng `hard/soft/mask`. | Đúng §9. |

---

## 4. Câu hỏi chặn

**Q1 — Ngưỡng blank.** Vì 12/12 nhãn đều `> 0.30`, nhánh `HARD_ZERO` không bao giờ chạy. Anh muốn:

* **(A)** giữ nguyên 0.30 như đã khoá, chạy đủ Stage 1–3, ghi nhận cảnh báo prevalence và báo cáo
  kết quả dù xấu (đúng tinh thần prereg, tốn ~30 h GPU);
* **(B)** bỏ nhánh blank-teacher, chỉ giữ `-1` disease-aware (Atelectasis/Edema → 1, 10 nhãn →
  teacher soft, No_Finding → mask), blank vẫn → hard 0. Teacher lúc đó có coverage đầy đủ như R0,
  soft target chỉ chạm 33,201 cell (1.7%) — can thiệp đúng chỗ nhãn thật sự mơ hồ;
* **(C)** đặt ngưỡng > 0.616 để nó thực sự phân loại (vd 0.90 → chỉ Consolidation/Enl.Card/Fracture/
  Lung_Lesion/Pleural_Other đi soft) — nhưng như vậy lại đẩy đúng nhóm hiếm nhất vào tay teacher yếu nhất.

Tôi nghiêng về **(B)**, và nếu vẫn muốn kiểm chứng nhánh blank thì chạy (B) trước rồi mới (A) như một
arm thứ hai — vì (B) rẻ hơn, gần với "chỉ sửa supervision" hơn, và không phá prior prevalence.

> **Trạng thái Q1: ĐÃ CHỐT (2026-07-31)** — giữ nguyên chiến lược `-1` của plan; **đổi router
> của `blank`** từ `blank_rate` sang **tỉ lệ `0` thật trong số mention**. Xem §4-bis.

**Q2 — Checkpoint rule của teacher.** ✅ **ĐÃ CHỐT: `fixed_last_epoch`** — train đủ 8 epoch trên
4 fold, lấy epoch cuối, không chọn checkpoint bằng bất kỳ metric nào. Đây đã là mặc định trong nb21,
không cần sửa code.

**Q3 — Thứ tự chạy.** Stage 0 là CPU thuần ~2 phút và in ra toàn bộ bảng ở §2 trên dữ liệu thật.
Chạy C1–C4 trước rồi quyết Q1 dựa trên số thật, hay chạy thẳng một mạch?

---

## 4-bis. QUYẾT ĐỊNH ĐÃ CHỐT — router mới + ma trận arm

### Router `blank` (thay §3 của plan)

```
neg_share = n(raw==0) / n(mention),   mention = n(+1) + n(0) + n(-1)
neg_share <= 0.30 -> blank = hard 0        (kênh nhãn không đủ negative thật -> cấm ngoại suy)
neg_share >  0.30 -> blank = teacher soft
No_Finding        -> hard 0                (giữ nguyên plan §3)
```

`-1` **giữ nguyên** disease-aware của plan ở mọi arm. `blank_rate` vẫn được tính và lưu đầy đủ
(§8 yêu cầu), chỉ không còn điều hướng.

| label | neg_share | route mới | route theo plan | teacher hard cells | %cov | %pos trong hard |
|---|---:|---|---|---:|---:|---:|
| Pneumothorax | 0.7863 | TEACHER_SOFT | TEACHER_SOFT | 34,598 | 23.4% | 19.6% |
| Pneumonia | 0.4007 | TEACHER_SOFT | TEACHER_SOFT | 25,751 | 17.4% | 41.3% |
| Edema | 0.3798 | TEACHER_SOFT | TEACHER_SOFT | 42,815 | 29.0% | 62.0% |
| Consolidation | 0.3431 | TEACHER_SOFT | TEACHER_SOFT | 12,306 | 8.3% | 57.8% |
| **Pleural_Effusion** | **0.3004** | TEACHER_SOFT | TEACHER_SOFT | 52,901 | 35.8% | 67.8% |
| Enlarged_Cardiomediastinum | 0.2345 | **HARD_ZERO** | TEACHER_SOFT | 141,710 | **95.9%** | 3.4% |
| Cardiomegaly | 0.2311 | **HARD_ZERO** | TEACHER_SOFT | 143,976 | **97.4%** | 20.5% |
| Fracture | 0.1543 | **HARD_ZERO** | TEACHER_SOFT | 147,444 | **99.8%** | 1.9% |
| Lung_Lesion | 0.0971 | **HARD_ZERO** | TEACHER_SOFT | 147,082 | **99.5%** | 2.8% |
| Lung_Opacity | 0.0506 | **HARD_ZERO** | TEACHER_SOFT | 145,360 | **98.4%** | 23.2% |
| Pleural_Other | 0.0450 | **HARD_ZERO** | TEACHER_SOFT | 147,328 | **99.7%** | 0.9% |
| Atelectasis | 0.0263 | **HARD_ZERO** | TEACHER_SOFT | 147,791 | **100%** | 25.0% |
| No_Finding | 0.0000 | HARD_ZERO | HARD_ZERO | 147,791 | 100% | 32.8% |

8/13 nhãn đổi route. Hiệu quả:

* soft cell **1,474,485 (76.7%) → 584,430 (30.4%)** — trong đó 551,229 từ blank + 33,201 từ `-1`
* gradient share của soft (theo số cell, λ=0.5) **62.3% → 17.9%**
* Rare-3 (Pleural_Other/Fracture/Lung_Lesion) chuyển hết sang HARD_ZERO ⇒ teacher có
  **99.5–99.8% coverage** trên đúng ba kênh yếu nhất, thay vì 0.9–3.1%
* 5 nhãn còn đi soft đều có mention cân bằng hai lớp ⇒ teacher không còn ngoại suy mù

> ⚠️ **Pleural_Effusion sát ngưỡng: `neg_share = 0.3004`, chỉ hơn 0.30 đúng 0.04 điểm phần trăm.**
> Khoảng **34 study** đổi trạng thái là lật route của một nhãn head có 91,081 cell blank. C2 in cảnh
> báo này tự động (`|neg_share − 0.30| < 0.02`). Nếu muốn route ổn định thì phải chốt tay, không nên
> để ngưỡng tự quyết.

### Ma trận arm — `BLANK_ROUTE_MODE` ở C0

Đổi **duy nhất một hằng số**; artifacts của mỗi arm ghi vào `label_pipeline_v1/<ARM_TAG>/` riêng.

| arm | `BLANK_ROUTE_MODE` | blank | `-1` | soft cells | grad share soft |
|---|---|---|---|---:|---:|
| `H0` | *(nb16 R0, đã có)* | hard 0 | **tất cả → 1** | 0 | 0% |
| `A0_allone` | `all_one` | **hard 1** | disease-aware | 33,201 | 1.2% |
| `A2_allzero` | `all_hard_zero` | hard 0 | disease-aware | 33,201 | 1.2% |
| `A3_negshare` | `neg_share` | routed | disease-aware | 584,430 | 17.9% |
| `A1_allsoft` | `all_soft` | teacher soft | disease-aware | 1,474,485 | 62.3% |
| `A1_planrate` | `plan_blank_rate` | teacher soft | disease-aware | 1,474,485 | 62.3% |

`all_soft` và `plan_blank_rate` cho ra **label set giống hệt nhau** trên cohort này — bằng chứng
trực tiếp rằng router `blank_rate <= 0.30` của plan chính là "all soft".

**Ba hướng anh chốt = `A0_allone` (hướng 1) · `A2_allzero` (hướng 2) · `A3_negshare` (hướng 3).**
`A1_allsoft` giữ lại như bản tái lập plan gốc, chạy hay không tuỳ anh.

**Bậc thang so sánh (mỗi bước chỉ đổi một thứ):**

```
H0            -> A2_allzero    : hieu ung cua rieng chinh sach -1
A2_allzero    -> A3_negshare   : hieu ung cua rieng chinh sach blank
A3_negshare   -> A1_allsoft    : mo rong blank soft ra toan bo 12 nhan
A2_allzero    -> A0_allone     : nhanh doi chung suy bien
```

> ⚠️ **`A2_allzero` KHÔNG bằng `H0`.** Vì `-1` giữ disease-aware ở mọi arm, `A2_allzero` =
> `blank → 0` **+** `-1` disease-aware, còn `H0` = `blank → 0` **+** `-1 → 1`. Nên `A2_allzero`
> **vẫn cần teacher** (cho 33,201 cell `-1`) và **vẫn cần một lượt train student mới** — không tái
> dùng được checkpoint R0. `H0` là mốc tham chiếu miễn phí, không phải một trong ba hướng.

### Teacher — ĐÃ CHỐT: chung cho A0/A2/A3, riêng cho A1

Teacher được **tách khỏi arm** (`TEACHER_MODE` ở C0), cache theo `TEACHER_TAG`:

| teacher | `TEACHER_MODE` | dùng cho | blank ở teacher | hard cells | coverage |
|---|---|---|---|---|---:|
| `T_blank0` | `shared_blank0` | A0, A2, A3 | hard 0 mọi nhãn | 1,919,461 | 96–100%/kênh |
| `T_masked_A1_allsoft` | `arm_masked` | A1 | mask (đúng plan §5) | 446,798 | 0.9–36%/kênh |

`TEACHER_MODE='auto'` tự chọn: `all_soft`/`plan_blank_rate` → `arm_masked`, còn lại → `shared_blank0`.
Artifacts teacher nằm ở `label_pipeline_v1/_teachers/<TAG>/`, arm nào chạy sau sẽ **dùng lại cache**,
không train lại. C4 khoá `sha256` của fold-map trong thư mục teacher và assert khi tái dùng — fold map
khác thì dừng, không cho trộn OOF của hai phân hoạch.

Chi phí: **~30–32 h GPU cho teacher** (15–16 h × 2) thay vì 60 h nếu mỗi arm một teacher.
Lợi thêm: A0/A2/A3 dùng **cùng một bộ soft target cho cell `-1`**, nên chênh lệch giữa ba arm đó
thuần tuý do chính sách `blank`, không lẫn nhiễu teacher.

> ⚠️ **Điểm đánh đổi nằm ở A3, không phải A1.** `T_blank0` đã được dạy `blank = 0` trên cả 13 kênh,
> nên soft target nó sinh ra cho blank của A3 sẽ **lệch về 0**. Nó vẫn phát hiện được omission
> (cơ chế là "model bất đồng với nhãn" — ảnh trông giống effusion thì prob vẫn cao dù report không
> nhắc), nhưng yếu hơn teacher chưa mang prior đó. Nếu muốn A3 hoàn toàn không mang giả định
> `blank = 0`, đặt `TEACHER_MODE = 'arm_masked'` khi chạy A3 — tốn thêm một teacher (~15 h).
> Với A3, teacher masked **không** rơi vào bẫy của §2: 5 nhãn đi soft đều có mention cân bằng
> (19.6–67.8% positive, coverage 8.3–35.8%), không phải trường hợp Pleural_Other 94%/0.9%.

---

## 5. Cách chạy

```
C0–C4   CPU, ~2 phút          -> raw_train_label_summary.csv, blank_policy_by_label.json,
                                 train_teacher_folds.parquet, teacher_supervision_manifest.parquet
                                 + bảng "CONG QUYET DINH TRUOC GPU"   <- dừng đọc ở đây
C5–C6   GPU, ~10 phút         -> stage ảnh + identity gate G0
C7      GPU, ~15–16 h         -> 5 teacher fold + teacher_oof_train_probs.npy  (resume theo epoch)
C8–C9   CPU, ~2 phút          -> final_train_label_manifest.parquet + Y/M matrices + diagnostics
C10     CPU, ~10 giây         -> student_loss + experiment_manifest.json
Stage 3 GPU, ~13–14 h         -> train H1 seed 42/1337/2024 bằng vòng train nb16 C12
```

C7 có resume theo epoch (`_resume_fold{k}.pt`) và cache theo fold (`oof_fold{k}.npz`) nên chạy rải
nhiều session Colab được. Artifacts ghi vào `planB_out/label_pipeline_v1/` — **không đè** bất kỳ file
nào của nb16/nb17/nb20.

---

## 6. Đã kiểm chứng

* 13/13 code cell parse sạch (cell C6 có `!wget` — magic của IPython, giống nb16).
* Dry-run thật các cell CPU trên cohort tổng hợp dựng đúng marginal `cohort_v2`:
  fold subject-disjoint ✓ · gate §15 fail = 0 ✓ · `mask/hard/soft` không chồng lấn ✓ ·
  6/6 leakage check ✓ · gradient trên cell masked = 0.00e+00 ✓ · `target_reason` khớp pseudocode §8 ✓.
* Cảnh báo effective prevalence bắn đúng như dự đoán ở §2(c).
