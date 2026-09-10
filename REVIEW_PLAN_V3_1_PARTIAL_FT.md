# REVIEW — `PLAN_LABEL_RELIABILITY_WEIGHTMATCHED_V3_1_PARTIAL_FT.md`

| | |
|---|---|
| Ngày | 2026-07-29 |
| Thay thế | `REVIEW_PLAN_3ARM_LABEL_RELIABILITY.md` (review V2) |
| Bối cảnh | `REVIEW_LABEL_NOISE_24H.md` · `REVIEW_DISEASE_GATE_WITH_TAIL_SELECTIVE_DBL.md` |
| Trạng thái | **Một lỗ phương pháp trong §3 cần sửa trước khi code.** Còn lại là hoàn thiện. |

---

## 0. Kết luận

V3.1 gỡ được **cả 5 chặn** của V2 (§7 dưới ghi nhận chi tiết). Thiết kế đã khả thi.

Còn **một lỗ trong lập luận trung tâm**: §3 khẳng định

> *"R2 và R1 có cùng total negative-weight budget theo class. Vì vậy R2−R1 kiểm tra vị trí
> đặt trọng số, không còn lẫn generic reweighting."*

**Khẳng định này không đúng với ASL.** Khớp *weight* không khớp *gradient*, và với `γ_neg = 4`
độ lệch giữa hai thứ đó có thể lên tới **hai bậc độ lớn**. Chi tiết §1.

Điều này đồng thời **sửa một ước lượng sai của tôi**: ở review V2 tôi tính can thiệp kiểu này
chỉ chạm ~0.3% và kết luận nó quá nhỏ. Sai — theo cell-count thì đúng 0.3%, nhưng theo
**gradient** thì có thể là hàng chục phần trăm. R2 là can thiệp **lớn**, không nhỏ.

Thứ tự nghiêm trọng:

| # | vấn đề | mức |
|---|---|---|
| §1 | weight-matching **không** khớp gradient ⇒ `D2` vẫn lẫn confound | **lỗ phương pháp** |
| §2 | số cell eligible có thể trip ngưỡng 500/1,000 của chính plan | cần Bước 0 |
| §3 | `λ_SP = 1e-5` trên AdamW **gần như trơ** — không phải neo thật | hiểu sai công cụ |
| §4 | thiếu số epoch; tiêu chí fallback §2.5 không định lượng | lỗ hổng prereg |
| §5 | chi phí ~5–8× nb15 | cần biết trước |

---

## 1. ⛔ Weight-matching không khớp gradient — `D2` vẫn còn confound

### Cơ chế: `clip` + `γ_neg = 4` làm ô âm điểm thấp gần như vô hình

ASL trong repo (verbatim nb11–15):

```python
xn = 1 - xp
if clip: xn = (xn + clip).clamp(max=1)     # clip = 0.05
pt  = xp*y + xn*(1-y)                       # y=0  ->  pt = xn
mod = (1-pt)**g                             # g = gn = 4  khi y=0
```

Với ô **âm** (`y=0`): `1 − pt = max(p − 0.05, 0)`, nên

```
mod_neg = max(p - 0.05, 0)^4
```

Hàm này **cực lồi**. Đóng góp gradient của một ô âm tỷ lệ với `w · (p−0.05)⁴`.

### Hệ quả: `contested blank` mang phần gradient lớn không tương xứng

Ta đã đo `pct_cont_neg = 0.706` — contested blank được model chấm **cao hơn hẳn** âm sạch.
Mà `(p−0.05)⁴` khuếch đại đúng nhóm điểm cao.

Ước lượng cho `Pleural_Other` (giá trị `p` là **phỏng đoán**, chờ đo — nhưng kết luận bền
với sai số vì độ lồi của `p⁴`):

| nhóm | n | `p` giả định | `(p−0.05)⁴` | tổng |
|---|---:|---:|---:|---:|
| ordinary blank | 10,010 | ~0.10 | 6.3e-6 | 0.063 |
| **contested blank** | **66** | **~0.30** | **3.9e-3** | **0.257** |

⇒ contested blank chiếm **~0.65% số cell** nhưng **~80% khối lượng gradient âm**.

### Vì thế phép khớp weight không làm được điều nó tuyên bố

```
f = N_cont / N_blank ~ 0.002-0.006  (TRAIN, sau khi tinh ca do phu hang xom 32%)
beta_hi = (beta - eta*f)/(1-f) = (0.5 - 0.1f)/(1-f) ~ 0.5008 ... 0.5024
```

- **Weight**: R2 lấy `0.4 × N_cont` đơn vị weight khỏi contested rồi trả `+0.0027` cho mỗi
  ordinary blank ⇒ tổng khớp **chính xác** ✔
- **Gradient**: phần lấy đi nằm ở nhóm mang ~80% gradient; phần trả lại rơi vào nhóm mang
  ~20% gradient và chỉ tăng 0.27% weight ⇒ trả lại được ~0.05%.

**Ròng: R2 xoá cỡ 60–65% gradient âm của nhãn Rare-3 so với R1, trong khi weight budget ghi
là bằng nhau.**

Nên `D2 = R2 − R1` **không** cô lập "vị trí đặt trọng số". Nó bị chi phối bởi "ít gradient âm
hơn nhiều" — chính là confound mà weight-matching được thiết kế để loại.

### Hai tin: một xấu một tốt

**Tốt**: R2 là can thiệp **mạnh**, không phải 0.3% như tôi tính sai ở review V2. Nếu temporal
targeting có tác dụng thì nó **đủ lực để thấy được**.

**Xấu**: kết quả dương của `D2` sẽ không quy được cho targeting.

### Cách sửa — khớp theo **gradient share** đo tại init

Thay §3 bằng: chọn `β_hi,c` sao cho **tổng đóng góp gradient âm** của R2 bằng R1, đo **một
lần tại init**, rồi khoá số.

```python
# do MOT LAN tai init, tren TRAIN (hoac mau lon), model = 04b+H5 chua train
@torch.no_grad()
def neg_grad_mass(p, clip=0.05, gn=4):
    """ty le voi |d(ASL_neg)/dz| cho o am."""
    m = np.clip(p - clip, 0, None) ** gn
    xn = np.clip(1 - p + clip, None, 1)
    return -np.log(np.clip(xn, 1e-12, None)) * m          # = l_neg; dung lam proxy khoi luong

for j, l in enumerate(LABELS):
    blank = (RAW[:, j] != RAW[:, j]) & train_mask          # NaN = blank
    cont  = blank & contested_mask[:, j]
    ord_  = blank & ~contested_mask[:, j]
    Gc, Go = neg_grad_mass(P0[cont, j]).sum(), neg_grad_mass(P0[ord_, j]).sum()
    # R1: beta*(Gc+Go)     R2: eta*Gc + beta_hi*Go     -> khop:
    beta_hi_grad = (BETA*(Gc+Go) - ETA*Gc) / max(Go, 1e-12)
    print(f'{l:<28} n_cont {int(cont.sum()):5d} ({cont.sum()/blank.sum():.2%} cell) | '
          f'grad share cont {Gc/(Gc+Go):.1%} | beta_hi weight-match {(BETA*blank.sum()-ETA*cont.sum())/ord_.sum():.4f} '
          f'| beta_hi GRAD-match {beta_hi_grad:.4f}')
```

Nếu `grad share cont` lớn (dự đoán: 30–80% cho Rare-3) thì `beta_hi_grad` sẽ **lớn hơn nhiều**
`0.5024` — có thể vượt 1. Khi đó phải hạ `η` hoặc nâng `β`, và §3 hiện tại đã có rule
*"nếu `β_hi > 1`, giảm `β` trước khi train"* — giữ nguyên rule đó, chỉ đổi đại lượng khớp.

**Đề xuất tối thiểu nếu không muốn đổi §3**: giữ weight-matching làm arm chính, nhưng **thêm
`R1′`** — uniform `β′` chọn sao cho **tổng gradient âm** khớp R2. Khi đó `R2 − R1′` mới cô lập
được vị trí. `R1` giữ lại để đo `D1`. Thành 4 arm.

---

## 2. Số cell eligible có thể trip ngưỡng của chính plan

Ngoại suy từ audit VAL (10,167 study có hàng xóm ≤24h trong 31,670):

```
TRAIN 147,791 study  ->  ~47,400 co hang xom <=24h  (32.1%)
```

`n_cont_neg / n_audit` từ bảng audit → nhân `47,400`:

| nhãn | audit | TRAIN (trước lọc) | sau lọc view+encounter (giả định 50%) | ngưỡng §5.3 |
|---|---:|---:|---:|---|
| Pleural_Other | 66 | ~308 | **~154** | < 500 ⇒ **không claim** |
| Fracture | 134 | ~625 | **~312** | < 500 ⇒ **không claim** |
| Lung_Lesion | 184 | ~858 | **~429** | < 500 ⇒ **không claim** |
| **tổng Rare-3** | 384 | ~1,791 | **~895** | < 1,000 ⇒ **R2 chỉ exploratory** |

⇒ Rất có thể **cả ba nhãn** trip ngưỡng 500, và tổng trip ngưỡng 1,000. Đúng tinh thần plan
(có sẵn check), nhưng hệ quả là R2 mất tư cách primary trước khi chạy.

### Ba cách nới, xếp theo mức phòng thủ được

**(a) Cửa sổ theo nhãn — mạnh nhất, và có cơ sở y học.**
R2 chỉ nhắm Rare-3 = `{Pleural_Other, Fracture, Lung_Lesion}` — **cả ba đều là tổn thương
cấu trúc/mạn tính**. Xương sườn gãy không lành trong 7 ngày; nốt phổi không biến mất; dày
màng phổi là mạn. Cửa sổ **7 ngày** cho ba nhãn này là hợp lý về lâm sàng, và sẽ nâng độ phủ
hàng xóm từ 32% lên đáng kể.

Ta còn có bằng chứng nội bộ hậu thuẫn: `Cardiomegaly` (mạn) tranh chấp 52.7% ở `certain +1`
trong khi `Pneumothorax` (cấp) chỉ 35.6% — bất đồng **không** do bệnh đổi ở nhãn mạn.

Ghi rõ: cửa sổ khác nhau theo nhãn phải **khoá trước**, và không áp cho nhãn động
(`Edema`/`Atelectasis`/`Pneumothorax` — §E10 của V2 đã loại chúng).

**(b) Dùng MỌI study khác của bệnh nhân trong cửa sổ, không chỉ chronological-adjacent.**
Câu hỏi thực chất là *"có phim nào khác của bệnh nhân này báo cáo phát hiện đó không"* —
không cần kề nhau. Lo ngại "một bệnh nhân ICU tạo hàng chục cạnh" đã được `w_subject = 1/K`
xử lý. Nới này rẻ và tăng đáng kể số cell.

**(c) Cho `raw -1` của hàng xóm làm evidence yếu.** §4 hiện loại nó. Với Rare-3 thì `-1` rất
ít (`Pleural_Other` <30 ca) nên gần như không giúp. Bỏ qua.

**Đề xuất: (a) + (b).** Cả hai đều chỉ đổi pair manifest offline, không đổi loss.

---

## 3. `λ_SP = 1e-5` trên AdamW gần như trơ — không phải neo thật

§2.6 dùng L2-SP để neo blocks 9–11 + decoder về state 04b/H5, và nói *"lấy từ protocol hiện
có"* (nb12–15 dùng `L2SP_DECODER = 1e-5`).

Vấn đề: AdamW **chuẩn hoá gradient**. Update ≈ `lr · m̂/(√v̂+ε)`. Gradient của L2-SP là
`2λ(θ−θ₀)`; với `λ = 1e-5` và drift `~0.01` thì nó cỡ `2e-7` — nhỏ hơn gradient của loss
nhiều bậc, nên nó đóng góp phần nghìn vào **hướng** update.

Ở nb12–15 điều này vô hại vì chỉ có 6.1M tham số decoder và LR đã rất thấp. Nhưng ở đây nó
được viện dẫn như **cơ chế bảo vệ** cho 21M tham số backbone. **Nó không bảo vệ được gì.**

Cái thật sự giữ blocks 9–11 khỏi trôi là **LR = 1e-5** (đã đặt ở §2.6) và số epoch ít — không
phải L2-SP.

Đề xuất: giữ `λ_SP` như hiện tại **cho nhất quán**, nhưng (a) đừng ghi nó vào phần safeguard,
và (b) thêm một safeguard thật sự đo được: ngưỡng cứng trên `‖θ−θ₀‖/‖θ₀‖` của blocks 9–11 —
nếu vượt, dừng và ghi nhận, giống cách §16 của plan WASL xử lý grad-norm bất ổn.

Ghi chú phụ: `l2sp_term` sẽ tăng ~3.4× vì cộng thêm 21M tham số vào tổng. Con số in ra
**không so được** với nb12–15. Cần chú thích.

---

## 4. Hai lỗ hổng prereg

**Số epoch không được nêu.** §2.5 nói "epoch 1 / epoch 2+", §9 nói "≥2/3 seeds", nhưng tổng
số epoch không có ở đâu. V2 ghi 8. Phải khoá — nếu không thì nó trở thành điểm quyết định
hậu nghiệm.

**Tiêu chí fallback §2.5 không định lượng.** *"Nếu smoke test cho thấy epoch 1 làm
representation drift quá mạnh trong khi logits không cải thiện"* — "quá mạnh" chưa có số.
Cần khoá trước, ví dụ: `cosine drift tại output block 11 > X` **và** `Δ macro mAP < 0.001`.

---

## 5. Chi phí

| | cấu hình | mỗi batch |
|---|---|---|
| nb14/15 | backbone đông cứng, 2 arm | 12 block fwd (no-grad) + 2 × decoder |
| **V3.1** | block 0–8 chung, 3 × (block 9–11 + head) | 9 block fwd + 3 × (3 block fwd+bwd + head) |

Backward ≈ 2× forward ⇒ `3 block fwd+bwd ≈ 9 block-equivalent`. Tổng ≈ `9 + 3×9 = 36` so với
`12` ⇒ **~3× mỗi batch**, và 3 arm thay vì 2.

Với 3 seed × 8 epoch: ≈ **5–8× tổng chi phí nb15**. Trên máy 102 GB thì VRAM không phải vấn
đề (3 arm × 27M tham số + Adam ≈ 1 GB; activation của 3 block có grad trên ~1,370 token sẽ do
`probe_batch` dò), nhưng thời gian thì đáng kể. Cần biết trước.

---

## 6. Điểm nhỏ, sửa nhanh

**§2.2 checkpoint dual-source** — đúng và cần thiết: `H5` chỉ có `norm.*` + `mld.*`, nên
blocks 9–11 phải load từ `BB_CKPT` (04b). Plan đã bắt được chỗ này ✔. Thêm assert:
`set(_bb.keys()) ⊇ {block 9,10,11 params}` để không im lặng bỏ sót.

**§2.4 RNG pairing** — bắt đúng vấn đề. Kiểm tra thực tế: RAD-DINO (`Dinov2Model`) có
`drop_path_rate` mặc định `0.0` và dropout `0.0`, nên có thể **không có** stochastic nào ở
blocks 9–11. Nên kiểm bằng một assert (`torch.equal` hai forward liên tiếp ở `train()` mode)
thay vì cài cơ chế RNG-sync không cần thiết.

**§7 checkpoint rule** — giữ đúng luật nb12–15 ✔, nên so được với bảng cũ. Nhưng lưu ý: với
+21M tham số, đỉnh có thể còn sớm hơn ep2, và luật sẽ lại bốc epoch rất sớm — đúng artifact đã
gặp ở nb14 (§7 của `REVIEW_DISEASE_GATE_...`). Nên log `‖θ−θ₀‖` theo epoch để biết checkpoint
được chọn có nằm trong vùng backbone đã thật sự học chưa.

**§9 success gate `+0.002`** — nằm dưới sàn nhiễu nhãn `±0.05` (`REVIEW_LABEL_NOISE_24H.md`
§3). Không sai, vì đây là so sánh **paired** trên **cùng** bộ nhãn nên nhiễu triệt tiêu. Nhưng
phải ghi rõ trong báo cáo, và bắt buộc kèm subject-cluster bootstrap (§11 đã có ✔).

---

## 7. Điều V3.1 đã làm đúng — ghi nhận

Tất cả 5 chặn của review V2 đã được gỡ, và gỡ đúng cách:

| chặn V2 | V3.1 xử lý |
|---|---|
| Rare-3 không có `raw 0` ⇒ mask xoá hết negative | **Bỏ hẳn masking.** Cả 3 arm giữ `blank` làm âm, chỉ đổi **weight**. Gỡ hoàn toàn, thanh lịch |
| Endpoint K đảo prevalence | Historical thành **primary**, K hạ xuống diagnostic, và §7 ghi *"Không dùng K để chọn checkpoint"* |
| Pair loss cần differentiable pooling | **Bỏ pair loss.** Pair manifest offline ⇒ §10 ghi đúng "không cần torch study pooling" |
| "trainable 9–11" không khớp code | §2.1 làm rõ, kèm cảnh báo *"không được bọc trong `no_grad`"* và §6.1 hard-check `parameter delta > 0` |
| N1 lẫn label semantics với pos/neg balance | **`R1` là control riêng cho generic downweighting** ⇒ `D1` và `D2` tách được (một phần — xem §1) |

Thêm những chỗ tốt khác:

- **§6.1 hard-check "blocks 0–8 parameter delta == 0"** và *"nếu blocks 9–11 không đổi, đó là
  implementation failure chứ không phải null result"* — chính xác cái phân biệt cần có.
- **§5 Bước 0 bắt buộc trước GPU** với ngưỡng go/no-go khoá trước.
- **§5.4** *"Không giả định `00:00` luôn date-only"* — cẩn thận đúng chỗ.
- **§2.6** N0/R0 không tuyên bố tái hiện A0/L0 — sửa đúng lỗi §4.2 của review V2.
- **§4** dùng `cur_hadm`/`cur_edstay`, bỏ ICU `stay_id` — khớp đúng thực tế cohort.
- **§2.5** giải thích được vì sao ep1 không còn "không tham số nào học" (21M tail-block).
- **§12** interpretation matrix có dòng *"R2 cải thiện pair diagnostic nhưng không mAP"* —
  đúng bài học của nb15.

---

## 8. Thứ tự đề xuất

```
BUOC 0a  (re, khong GPU)
  - raw four-state x 13 nhan x 3 split, SO TUYET DOI          (§5.1 plan)
  - pair funnel sau tung buoc loc                              (§5.2 plan)
  - contested blank counts + beta_hi weight-match              (§5.3 plan)

BUOC 0b  (MOI — quyet dinh lo §1)
  - forward 1 luot model init (04b+H5) tren TRAIN
  - do GRADIENT SHARE cua contested blank theo tung nhan
  - tinh beta_hi GRAD-match, kiem <= 1

BUOC 1  chot 4 diem:
  - doi §3 sang grad-matching, HAY them arm R1' (4 arm)        (§1)
  - noi cua so cho Rare-3 (7 ngay?) + dung moi study khong chi adjacent  (§2)
  - so epoch (8?)                                              (§4)
  - tieu chi dinh luong cho fallback §2.5                      (§4)

BUOC 2  moi viet notebook
```

Bước 0b là bước mới và là bước quan trọng nhất — nó quyết định `D2` có diễn giải được hay
không, và nó rẻ (một forward pass, không train).

---

## 9. Nhật ký quyết định

| Ngày | Mục | Quyết định | Ghi chú |
|---|---|---|---|
| | §1 grad-matching vs thêm `R1′` | _(chưa trả lời)_ | **Chặn diễn giải `D2`** |
| | §2 nới cửa sổ / bỏ adjacent-only | _(chưa trả lời)_ | Quyết định R2 có đủ cell hay không |
| | §4 số epoch | _(chưa trả lời)_ | Đề xuất 8, khớp nb12–15 |
| | §4 tiêu chí fallback §2.5 | _(chưa trả lời)_ | |
| | §3 safeguard drift thật thay L2-SP | _(chưa trả lời)_ | |
