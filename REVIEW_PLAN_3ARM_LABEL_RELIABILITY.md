# REVIEW — `PLAN_3ARM_NO_TEXT_LABEL_RELIABILITY_V2.md`

| | |
|---|---|
| Ngày | 2026-07-29 |
| Kế hoạch được review | `PLAN_3ARM_NO_TEXT_LABEL_RELIABILITY_V2.md` |
| Bối cảnh | `REVIEW_LABEL_NOISE_24H.md` (đo nhiễu nhãn) · `REVIEW_DISEASE_GATE_WITH_TAIL_SELECTIVE_DBL.md` (nhánh loss, đã đóng) |
| Trạng thái | **Chờ một phép đo chặn** — §1 dưới có thể giết thiết kế trước khi viết code |

---

## 0. Kết luận

Thiết kế chặt chẽ về mặt phương pháp — nó đã tránh được đúng những bẫy mà nhánh loss đã vấp
(§6 "Điều kế hoạch làm đúng"). Nhưng **ba giả thiết định lượng chưa được kiểm**, và nếu chúng
sai thì N1/N2 không thể phát biểu gì về Rare-3 — tức mất mục tiêu chính.

**Phải chạy §2.2 (coverage audit) TRƯỚC KHI viết bất cứ dòng code train nào.** Nó rẻ (chỉ đọc
bảng `chexpert`) và có thể kết luận rằng thiết kế không khả thi.

Thứ tự nghiêm trọng:

| # | vấn đề | mức |
|---|---|---|
| §1 | Rare-3 gần như **không có `raw 0`** (known negative) | **có thể giết thiết kế** |
| §2 | Endpoint K **đảo prevalence** 1.2% → ~80% cho Rare-3 | **nghiêm trọng** |
| §3 | `C1 = N1 − N0` lẫn hai can thiệp, không chỉ một | **confound thật** |
| §4 | 7 chỗ lệch với code hiện tại | sửa được |
| §5 | Cơ sở dữ liệu của pair loss rất mỏng | cần đo |

---

## 1. ⛔ Rare-3 gần như không có `raw 0` — N1 mất nền negative

### Bằng chứng đã có

Trong bảng phân rã U-policy (`REVIEW_LABEL_NOISE_24H.md` §4), cột `explicit 0` trả **NaN**
cho `Pleural_Other`, `Fracture`, `Lung_Lesion` — nghĩa là `n(raw == 0) < 30` trong **10,167**
study audit (ngưỡng in của tôi là `n >= 30`).

Ngoại suy sang TRAIN (147,791 study, tỷ lệ ×14.5):

```
n(raw == 0) < ~435  cho tung nhan Rare-3
```

Chính kế hoạch, §E3, đặt hard-block:

> `<1,000 known negatives TRAIN` → N1/N2 không đủ negative supervision để đưa ra claim

⇒ **Cả ba nhãn Rare-3 gần như chắc chắn trip E3.**

### Vì sao — và đây là tính chất cấu trúc, không phải lỗi dữ liệu

`raw = 0` nghĩa là report **nhắc tới** phát hiện **và phủ định** nó ("no pneumothorax").
Bác sĩ chỉ phủ định tường minh những thứ **cần loại trừ về mặt lâm sàng**: tràn khí, tràn
dịch, đông đặc, viêm phổi. Không ai viết *"không dày màng phổi"* hay *"không gãy xương sườn"*.

Với phát hiện hiếm/tình cờ, **vắng mặt = im lặng**, và im lặng là `blank`.

Đối chiếu trong cùng bảng: `Pneumothorax` có `explicit 0` = 7.7% (n ≥ 30, báo được) — đúng
là nhãn mà ICU report luôn phủ định tường minh. Cơ chế nhất quán.

### Hệ quả cho N1

N1 mask mọi ô không phải `+1` hoặc `0`. Với `Pleural_Other` trong tập audit:

```
raw +1        91
raw  0       <30
bi mask   ~10,046   (98.8%)
```

Model học ngưỡng cho `Pleural_Other` từ ~120 ô, trong đó **<30 âm tính**. Không đủ để định
vị bất cứ ngưỡng nào.

Nghịch lý: N1 được thiết kế để **sửa** tác hại của `blank → 0`, nhưng với đúng ba nhãn ta
quan tâm, `blank` **là toàn bộ** thông tin âm tính đang có. Bỏ nó đi thì không còn gì.

### Việc phải làm trước

Chạy §2.2 nhưng **trên TRAIN, đủ 13 nhãn, và in `n(raw==0)` tuyệt đối**, không chỉ tỷ lệ:

```python
for j, l in enumerate(LABELS):
    for sp in ('train','val','test'):
        m = (ehr.split == sp).to_numpy()
        v = RAW_ALL[m, j]
        print(f'{l:<28} {sp:<6} +1 {int((v==1).sum()):6d} | 0 {int((v==0).sum()):6d} '
              f'| -1 {int((v==-1).sum()):6d} | blank {int(np.isnan(v).sum()):6d} '
              f'| known {(~np.isnan(v) & (v!=-1)).mean():.1%}')
```

Nếu `n(raw==0) < 1000` ở TRAIN cho Rare-3 — điều tôi dự đoán — thì có **ba lựa chọn**, cần
chốt trước khi code:

| | phương án | đánh giá |
|---|---|---|
| **A** | Đổi mục tiêu N1/N2 sang nhóm nhãn **có** đủ known-negative (`Pneumothorax`, `Consolidation`, `Pleural_Effusion`, `Pneumonia`, `Edema`) | Khả thi ngay, nhưng **không** trả lời được câu hỏi về Rare-3 |
| **B** | N1 mask **chỉ `raw -1`**, giữ `blank → 0` | Giữ nền negative, nhưng chỉ chạm 17.2% khối lượng — mà §4 của review nhiễu nhãn đã đo U-policy chỉ đóng góp ~3% |
| **C** | Bỏ N1, giữ N2 (pair) làm can thiệp duy nhất | Nhưng §5 dưới cho thấy pair cũng mỏng |

Tôi nghiêng về **A**, và đổi tên mục tiêu: thí nghiệm trở thành *"partial-label có giúp nhóm
nhãn được phủ định tường minh không"* — một câu hỏi hợp lệ và trả lời được, chỉ khác câu hỏi
ban đầu.

---

## 2. ⚠ Endpoint K đảo prevalence — `mAP_K` của Rare-3 không còn là metric bệnh hiếm

§8.2 định nghĩa Endpoint K = chỉ evaluate tại `raw +1` và `raw 0`.

Với `Pleural_Other` trên VAL (31,670 study):

| | strict (historical) | Endpoint K |
|---|---:|---:|
| dương | 377 | 377 |
| âm | 31,293 | **~93** |
| tổng | 31,670 | **~470** |
| **prevalence** | **1.19%** | **~80%** |

Ba hệ quả:

**Bộ phân loại ngẫu nhiên được `AP_K ≈ 0.80`.** Không còn dư địa phân biệt. `AP` ở prevalence
80% đo một thứ hoàn toàn khác `AP` ở 1.2%.

**n ≈ 470 study.** Cổng §10.2 đòi `Rare-3 mAP_K` tăng `+0.005`. Với `n ≈ 470` và baseline
~0.8–0.9, sai số lấy mẫu của `AP` đã lớn hơn `0.005` nhiều lần. Cổng nằm dưới sàn nhiễu —
đúng cái bẫy mà nhánh loss đã vấp, chỉ đổi trục.

**Tập K khác nhau theo nhãn.** `macro mAP_K` trung bình các `AP` tính trên **các tập study
khác nhau** (mỗi nhãn một mask). Hợp lệ về kỹ thuật nhưng phải ghi rõ, và không được hiểu
như "macro mAP trên một tập chung".

Kế hoạch **đã** chặn đúng một nửa vấn đề (§8.2: *"Không so AP_K với historical AP"*, E16).
Nửa còn lại chưa: **`AP_K` của Rare-3 tự nó cũng không dùng được làm primary endpoint.**

Đề xuất: với Rare-3, dùng **AUROC_K** làm primary (bất biến với prevalence — đúng lý do
`REVIEW_LABEL_NOISE_24H.md` §3 nêu), và hạ `AP_K` xuống secondary. Điều này **ngược** §8.2
hiện tại (*"AUROC là robustness diagnostic, không phải mục tiêu tối ưu chính"*) — cần chốt.

---

## 3. `C1 = N1 − N0` lẫn **hai** can thiệp

N1 khác N0 ở hai điều cùng lúc, không phải một:

1. **không supervise ô unknown** (ý định);
2. **cân bằng dương/âm của objective đổi hẳn** (tác dụng phụ).

ASL có `γ_neg = 4` và `clip = 0.05`; ở prevalence 1–40%, **phần lớn khối lượng loss nằm ở ô
âm**. N1 xoá ~85% ô âm (blank) ⇒ objective của N1 nghiêng về dương mạnh hơn N0 rất nhiều,
dù công thức ASL không đổi.

Nói cách khác N1 **vô tình là một dạng re-weighting dương** — đúng họ can thiệp mà nb12–15
vừa chứng minh là không hiệu quả.

Nếu `C1` dương, ta **không phân biệt được** "bỏ nhãn giả" với "nhấn dương". Cần một arm
control: **N0′ = N0 nhưng hạ trọng số ô âm sao cho tỷ lệ pos/neg khớp N1**. Nếu N0′ ≈ N1 thì
lợi ích của N1 chỉ là re-weighting, không phải partial labeling.

§5.1 đã đúng về chuẩn hoá (`Σwℓ / Σw` là weighted mean nên **thang loss giữ được**) — vấn đề
không phải thang, mà là **thành phần**.

---

## 4. Bảy chỗ lệch với code hiện tại

### 4.1 ⛔ "separate trainable blocks 9–11" — **không notebook nào đang train block backbone**

§1.1 viết: *"shared frozen RAD-DINO blocks 0–8 / separate trainable blocks 9–11 + ML-Decoder"*.

Thực tế trong repo, **mọi** notebook đều có:

```python
for p in backbone.parameters(): p.requires_grad = False
```

- nb11/12/13: block 9–11 chạy **có** graph nhưng **weight đông cứng** — grad chỉ để chảy về
  `CondAdapter` đặt tại block 9,10. Backbone không học.
- nb14/15: bỏ adapter ⇒ **toàn bộ 12 block `no_grad`**, chỉ decoder + gate học.

Nếu kế hoạch thật sự muốn mở block 9–11:

```
+3 x 7.1M = ~21M tham so hoc / arm  (hien tai: 6.1M chi rieng decoder)
x 3 arm x 3 seed x 8 epoch = 72 arm-epoch, moi cai backprop qua 3 transformer block
                             tren ~1,370 token
```

Nặng hơn nb15 (16 arm-epoch, backbone đông cứng) **rất nhiều**. Và nó phá tương đương với
baseline H5 vì H5 checkpoint chỉ chứa `norm.*` + `mld.*`, không chứa backbone.

Nếu **không** muốn mở block: §1.1 và §7 tự mâu thuẫn — ep1 đóng băng decoder mà backbone
cũng đông cứng thì **không có tham số nào học ở ep1**.

**Cần chốt rõ.** Tôi đề xuất giữ backbone đông cứng hoàn toàn (như nb14/15) và bỏ câu
"decoder frozen ở ep1" — vì không còn EHR encoder/gate để học trong lúc đó.

### 4.2 N0 không có baseline nào để "tái hiện"

§1.2: *"Đây phải tái hiện baseline hiện tại"*. Nhưng repo **không có** run image-only + ASL
8 epoch nào:

| có sẵn | là gì |
|---|---|
| nb12 `L0` | adapter block 10/11 + **EHR** + ASL |
| nb14 `A0` | disease gate + **NULL EHR** + ASL (vẫn có gate/residual) |
| nb14 `A1` | disease gate + EHR thật + ASL |
| `D0` / H5 | head chưa train, chỉ là anchor |

Gần nhất là nb14 `A0`, nhưng nó có nhánh gate. Nên N0 **định nghĩa một baseline mới**, không
tái hiện gì. Phải sửa câu đó, và không được kỳ vọng số của N0 khớp bảng cũ.

### 4.3 `pool_study` **không** khả vi — pair loss cần bản torch

§6.2 yêu cầu pool study-level 7:3 **trong lúc train** để tính pair loss. Hàm hiện có:

```python
def pool_study(p_img, split):
    sub = man.iloc[IDX[split]]; d = pd.DataFrame(p_img, columns=LABELS)
    ... groupby('study_id').mean() ... np.nan_to_num ...
```

numpy + pandas, **không có gradient**, và bị neo vào `IDX[split]`. Phải viết lại bằng torch
(`index_add_` hoặc `scatter_add_` theo study index, rồi renormalize `W_F·hf + W_L·hl`).
Không khó nhưng **không** phải "giống validation" theo nghĩa tái dùng code.

Cũng phải chốt: pool trên **probability** (như validation) hay logit? §5.2 dùng
`p_teacher = sigmoid(z_known)` ⇒ nhất quán nhất là pool **probability**. Ghi rõ.

### 4.4 `hadm_id` / `edstay_id` không nằm trong file mà mọi notebook đang load

§3.3 cần `hadm_id`, ED `stay_id`, ICU `stay_id`.

- `cur_hadm`, `cur_edstay` có trong **`cohort_base.parquet`** (`base_cols` của 01·C18);
- mọi notebook nb11–15 load **`cohort_study_level.parquet`** (từ 02), và C1 của chúng
  không tham chiếu hai cột đó;
- **ICU `stay_id` không tồn tại** trong cohort ⇒ điều kiện 3 của §3.3 phải rơi về `hadm_id`.

Cần merge thêm `cohort_base.parquet`, và sửa §3.3 bỏ ICU stay_id.

### 4.5 `index_time` — đã xác minh là **per-study**, dùng được

Kiểm chứng: bản dựng lại pairing của tôi dùng `ehr.index_time` cho ra `median gap 11.20 h`,
**trùng khít** audit gốc. Nếu `index_time` là thời điểm nhập viện thì mọi study trong một
lần nhập sẽ cùng timestamp và gap = 0. Nên nó là thời điểm chụp. §3.2 dùng được.

Nhưng §E6 (*"timestamp chỉ có ngày → loại"*) cần một phép đếm: bao nhiêu study có phần
giờ/phút bằng 00:00? Chưa đo.

### 4.6 Bootstrap cụm theo `subject_id` — code hiện tại resample **study**

§11 yêu cầu `cluster = subject_id`. Hàm `paired_bootstrap` tôi viết cho nb14/15 resample
**study** độc lập. Phải viết lại: resample **subject**, rồi lấy toàn bộ study của subject
được chọn. `ehr` có `subject_id` nên làm được, chỉ là code mới.

Lưu ý phụ: bootstrap cụm cho `AP` của nhãn hiếm dễ gặp resample **không có ca dương** —
§E18 đã lường, tốt.

### 4.7 Luật chọn checkpoint khác nb12–15 ⇒ không so được với bảng cũ

§9 dùng "highest VAL Macro mAP". nb12–15 dùng "macro-constrained tail selection"
(`macro ≥ best − 0.0015` → max tail). Đây là **luật khác**, nên số của N0 không so trực tiếp
được với `L0`/`A1`. Không sai, chỉ cần ghi rõ.

---

## 5. Cơ sở dữ liệu của pair loss rất mỏng — cần đo trước

Ước lượng thô từ số val đã biết (10,167/31,670 = 32.1% study có hàng xóm ≤24h):

```
TRAIN 147,791 study  ->  ~47,000 study co hang xom  ->  ~23,500 cap ke-thoi-gian
sau loc same-view-signature + same-encounter        ->  ? (CHUA DO, doan 40-70%)
```

Số cặp có **state P/U hoặc N/U** cho từng nhãn Rare-3 (điều kiện để pair loss kích hoạt):

| nhãn | prevalence | cặp có ≥1 bên `raw +1` | sau lọc view/encounter |
|---|---:|---:|---:|
| Pleural_Other | 1.19% | ~560 | **~250–400** |
| Fracture | 2.17% | ~1,020 | ~450–700 |
| Lung_Lesion | 3.22% | ~1,510 | ~650–1,050 |

Và state `N/U` gần như **không tồn tại** vì `raw 0` cực hiếm (§1) ⇒ pair loss thực tế chỉ
chạy ở state **P/U**.

Với `ρ = 0.10` (gradient pair = 10% supervised) và vài trăm instance rải qua 8 epoch, hiệu
ứng kỳ vọng rất nhỏ. §5.3 có cơ chế phát hiện ("nếu 20 batch không có eligible pair thì N2
không đủ dữ liệu") nhưng ngưỡng đó **quá lỏng**: một pair batch 32 cặp sẽ có vài cặp eligible,
nên calibration chạy được trong khi tổng tín hiệu vẫn không đáng kể.

Đề xuất thêm một hard-block định lượng: **`< 500 eligible P/U pair` trong TRAIN cho một nhãn
Rare-3 ⇒ không claim cho nhãn đó**, song song với E3.

---

## 6. Điều kế hoạch làm **đúng** — đáng ghi nhận

Kế hoạch đã hấp thu chính xác các bài học từ nhánh loss:

- **E16 + §8.2**: không so `AP` giữa Endpoint H và K, mask cố định cho mọi arm — đúng bài học
  prevalence của `REVIEW_LABEL_NOISE_24H.md` §3.
- **§5.1**: `Σwℓ / Σw` (weighted mean) chứ không chia `B×13` — giữ được thang loss giữa arm.
- **§3.3 điều kiện bắt buộc số 1**: `subject_id_a == subject_id_b`. Bản V1 để ngỏ chỗ này;
  V2 đã đóng.
- **§E10**: loại `Edema`, `Atelectasis`, `Pneumothorax` khỏi pair loss — đúng, đó là ba nhãn
  **đổi thật** trong 11h.
- **§5.3 ρ-calibration** thay vì sweep λ trên VAL — đúng phương pháp, và giống cách tôi đã
  làm cho `λ_W` ở nb15.
- **§2.3**: không recompute tier — giữ Rare-3/NonTarget-10/Worst-3 khoá.
- **§E20**: khoá toàn bộ hyperparameter trước train.
- **§9**: báo cáo **hai** luật chọn checkpoint và chỉ gọi kết quả robust khi cùng chiều.
- **§4.1**: không dùng consistency cho state `P/N` — đúng, đó có thể là clinical change thật
  và ép hai prediction bằng nhau sẽ tạo mục tiêu vô định.
- **§E12**: monitor collapse bằng variance/entropy — cần thiết cho mọi consistency loss.

---

## 7. Thứ tự đề xuất

```
BUOC 0  (re, khong GPU, khong code train)
  - dem raw four-state cho DU 13 nhan x 3 split, in SO TUYET DOI
  - dem cap: adjacent <=24h, sau tung buoc loc (view / encounter / timestamp precision)
  - dem eligible P/U pair cho tung nhan Rare-3
  -> quyet dinh thiet ke con song hay khong

BUOC 1  chot 5 diem con mo:
  - backbone block 9-11 co train khong (§4.1)
  - Rare-3 hay nhom co known-negative (§1 phuong an A/B/C)
  - primary endpoint cho Rare-3: AUROC_K hay AP_K (§2)
  - co them arm N0' de tach confound pos/neg (§3)
  - N0 la baseline moi, khong "tai hien" (§4.2)

BUOC 2  moi viet notebook
```

**Bước 0 có thể kết luận rằng không nên chạy thí nghiệm này ở dạng hiện tại.** Đó là kết quả
tốt — nó rẻ hơn nhiều so với phát hiện điều đó sau 72 arm-epoch.

---

## 8. Nhật ký quyết định

| Ngày | Mục | Quyết định | Ghi chú |
|---|---|---|---|
| | §1 Rare-3 known-negative | _(chưa trả lời)_ | **Chặn** — cần bước 0 |
| | §2 primary endpoint Rare-3 | _(chưa trả lời)_ | AUROC_K vs AP_K |
| | §3 arm `N0'` tách confound | _(chưa trả lời)_ | |
| | §4.1 train block 9–11 | _(chưa trả lời)_ | **Chặn** — quyết định chi phí |
| | §4.2 N0 = baseline mới | _(chưa trả lời)_ | chỉ là sửa câu chữ |
