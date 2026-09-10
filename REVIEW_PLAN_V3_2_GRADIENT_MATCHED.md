# REVIEW — `PLAN_LABEL_RELIABILITY_V3_2_GRADIENT_MATCHED.md`

| | |
|---|---|
| Ngày | 2026-07-29 |
| Thay thế | `REVIEW_PLAN_V3_1_PARTIAL_FT.md` |
| Trạng thái | **Hai chặn cấu trúc** (§1, §2) — sẽ làm thí nghiệm thất bại vì lý do không liên quan giả thuyết. Còn lại là hoàn thiện. |

---

## 0. Kết luận

V3.2 gỡ đúng cả 5 điểm của review V3.1, và §4.3 (kiểm parameter-gradient chứ không chỉ
logit-gradient) **đi xa hơn** đề xuất của tôi. §7 dưới ghi nhận chi tiết.

Còn hai vấn đề, cả hai đều là **chặn cấu trúc** — chúng làm thí nghiệm thất bại vì cơ chế,
không vì giả thuyết sai:

| # | vấn đề | hệ quả nếu không sửa |
|---|---|---|
| **§1** | Ngưỡng drift ở §7 là **số phỏng đoán chưa hiệu chuẩn**; ước lượng cho thấy chúng bị vượt **ngay sau epoch 1** | §7.1 tự kích hoạt fallback; §7.2 làm **mọi** checkpoint không đủ điều kiện ⇒ không chọn được gì |
| **§2** | §9 chuẩn hoá `L = Σwℓ/Σw` **theo từng arm** ⇒ 10 nhãn non-target nhận gradient lệch ~20% giữa R1-GM và R2 | `NonTarget-10` dịch vì lý do không liên quan label policy; cổng `±0.002` bị phá |
| **§3-bis** | **Lỗi trong nb14/15 do tôi gây ra**: chúng đưa cả token `CLS` vào ML-Decoder, còn nb11/13 và H5 thì không | Không phá contrast nội bộ nb14/15, nhưng câu *"`z_IMG` = đầu ra H5"* không chặt. Phải chốt quy ước cho nb16 |

Cả ba sửa được rẻ, và phải sửa **trước** khi chạy.

Toàn bộ **15 mục** cần chốt nằm ở §9 — không chỉ ba mục chặn.

---

## 1. ⛔ Ngưỡng drift ở §7 chưa hiệu chuẩn — ước lượng cho thấy sẽ vượt ngay ep1

### Ước lượng có dạng đóng, và độc lập với số tham số

Với AdamW, update mỗi bước ≈ `lr · m̂/(√v̂+ε)`, tức **mỗi tham số dịch cỡ `lr`** bất kể độ lớn
gradient (đó là bản chất chuẩn hoá của Adam). Nếu các bước không kết hợp (random walk):

```
||Δθ|| ~ lr · sqrt(steps) · sqrt(P)
||θ₀|| ~ RMS(θ₀) · sqrt(P)

drift tuong doi = ||Δθ||/||θ₀|| ~ lr · sqrt(steps) / RMS(θ₀)
```

`sqrt(P)` **triệt tiêu** — nên ước lượng không phụ thuộc việc mở 21M hay 85M tham số.

Thay số của plan: `lr = 1e-5`, `steps/epoch ≈ 950` (nb14 in ra `948`), `RMS(θ₀) ≈ 0.03`
(Linear với `fan_in = 768` khởi tạo `std ≈ 1/√768 ≈ 0.036`):

```
sau 1 epoch:  1e-5 x sqrt(950) / 0.03  =  1e-5 x 30.8 / 0.03  =  0.0103
sau 8 epoch:  1e-5 x sqrt(7600) / 0.03 =  1e-5 x 87.2 / 0.03  =  0.0291
```

Đây là **cận dưới** (random walk thuần). Nếu các bước kết hợp một phần thì lớn hơn — cận trên
coherent là `lr · steps / RMS = 0.32`.

### So với ngưỡng của plan

| ngưỡng | plan | ước lượng | kết quả |
|---|---:|---:|---|
| §7.1 tổng drift sau ep1 | **≤ 0.005** | ~0.010 (cận dưới) | **vượt ~2×** ⇒ fallback tự kích hoạt |
| §7.1 mỗi block sau ep1 | ≤ 0.0075 | ~0.010 | vượt |
| §7.2 tổng drift tới ep8 | **≤ 0.010** | ~0.029 | **vượt ~3×** ⇒ **không checkpoint nào đủ điều kiện** |

§7.2 nói *"A checkpoint outside this trust region ... cannot be selected"*. Nếu mọi epoch đều
vượt, **thí nghiệm không có checkpoint nào chọn được** — thất bại vì ngưỡng, không vì giả thuyết.

Và §7.1 nói vượt thì *"the delayed-decoder protocol fails, do not continue official runs,
restart all arms with decoder trainable from epoch 1"*. Fallback sẽ kích hoạt gần như chắc
chắn, mà nguyên nhân không phải protocol sai.

### Cách sửa — hiệu chuẩn trước, đặt ngưỡng sau

Ngưỡng drift **không thể đoán**; nó phụ thuộc `lr`, số bước, weight decay `1e-2` (cũng kéo θ
về 0), grad clip `5.0`, warmup. Phải **đo** rồi mới khoá:

```python
# smoke ~200 step, MOT arm, khong luu checkpoint
th0 = {n: p.detach().clone() for n, p in tail.named_parameters()}
for step, b in enumerate(loader):
    ...  # train binh thuong
    if step % 50 == 0:
        for blk in (9, 10, 11):
            num = sum(((p - th0[n])**2).sum() for n, p in tail.named_parameters() if f'.{blk}.' in n)
            den = sum((th0[n]**2).sum() for n in th0 if f'.{blk}.' in n)
            print(f'step {step:4d} block {blk} drift {(num/den).sqrt().item():.5f}')
    if step >= 200: break
```

Rồi ngoại suy `drift(t) ≈ drift(200)·√(t/200)` sang ep1 và ep8, và **đặt ngưỡng ở 2–3× giá trị
ngoại suy**. Khoá số vào prereg trước khi chạy chính thức.

Chi phí: vài phút. Rẻ hơn nhiều so với phát hiện sau khi chạy Stage 1 rằng không chọn được
checkpoint nào.

**Ghi chú**: `feature cosine ≥ 0.99` (ep1) và `≥ 0.98` (ep8) cũng chưa hiệu chuẩn, nhưng nó
là đại lượng an toàn hơn — cosine ở output block 11 bền hơn nhiều so với chuẩn tham số, vì
biểu diễn có thể giữ hướng dù weight dịch. Vẫn nên đo trong cùng smoke run.

---

## 2. ⛔ §9 chuẩn hoá theo từng arm ⇒ confound trên 10 nhãn non-target

### Cơ chế

§9: `L = Σ_{i,c} w_{i,c}·ℓ_{i,c} / Σ_{i,c} w_{i,c}` — **mẫu số phụ thuộc arm**.

Gradient tại một ô: `∂L/∂z_ic = w_ic·(∂ℓ_ic/∂z) / Σw`.

Với một ô **non-target** (`w = 1` ở mọi arm), gradient vẫn khác nhau giữa các arm vì `Σw` khác.

### Độ lớn

`β_grad,c = 1` cho 10 nhãn non-target (vì `G_cont = 0` ⇒ `β_grad = G_ord/G_ord = 1`) ✔ —
plan tự động đúng ở đây. Nên chỉ Rare-3 bị downweight.

Với `s ≈ 0.8` (contested gradient share, chờ Bước 0b) thì `β_grad = 1 − 0.9s ≈ 0.28`.
Rare-3 có ~98% ô là blank, nên trên 13 nhãn:

```
R0    Σw = 13 B
R1-GM Σw ~ (13 - 3 x 0.98 x 0.72) B = 10.9 B     -> 0.834 x R0
R2    Σw ~ 13 B - 0.9 x (contested/batch) ~ 13 B  -> ~1.00 x R0
```

⇒ mỗi ô non-target trong **R1-GM** nhận gradient lớn hơn trong **R2** khoảng
`1/0.834 = 1.20×` — **+20%**, hoàn toàn không liên quan label policy.

Cổng `NonTarget-10 ≥ −0.002` của §12 vì thế bị phá bởi một artefact chuẩn hoá.

### Cách sửa — chia cho **hằng số**, và AdamW làm việc này an toàn

```
L = Σ w·ℓ / (B x 13)        # hang so, giong nhau moi arm
```

Hệ quả:

- ô non-target nhận gradient **bit-identical** giữa ba arm ⇒ khác biệt `NonTarget-10` chỉ có
  thể đến qua tham số dùng chung, đúng thứ ta muốn đo;
- tổng độ lớn gradient khác nhau giữa arm — **nhưng AdamW bất biến với thang gradient toàn
  cục** (`m̂/√v̂` không đổi khi nhân gradient với hằng số), nên chênh lệch này bị hấp thụ.

Ngược lại, chia theo `Σw` tạo méo **tương đối** (tỷ phần của ô không bị nhắm thay đổi) — và
Adam **không** hấp thụ méo tương đối.

Lưu ý: V2 §5.1 từng cấm chia cho `B×13` — nhưng bối cảnh khác. V2 dùng **mask** (`w = 0`),
nên `Σw` co lại rất mạnh và chia hằng số sẽ làm loss teo. V3.2 **không mask** (mọi `w > 0`,
thay đổi ôn hoà), nên chia hằng số là lựa chọn đúng.

---

## 3. §4.3 — phép kiểm parameter-gradient có bẫy mẫu số

§4.3 yêu cầu tính *"blank-negative-only gradients"* cho R1-GM và R2 rồi so tỷ lệ chuẩn.

Nếu hiện thực bằng `L_blank = Σw·ℓ/Σw` **chỉ trên ô blank**, thì hai arm có `Σw` khác nhau
(`β_grad·N_blank` vs `η·N_cont + N_ord`) và tỷ lệ chuẩn gradient sẽ bị mẫu số chi phối chứ
không phải phần tử tử.

Phải chốt một trong hai:

```
(a) so gradient KHONG chuan hoa:  L_blank = Σ w·ℓ   (khong chia)
(b) dung CUNG mot mau so cho ca hai arm (vi du B x 13, khop §2 o tren)
```

Với sửa ở §2 (chia hằng số) thì `(b)` là tự động — hai chỗ nhất quán.

Thêm một chỗ chưa định rõ: §4.3 cho phép nhân toàn bộ `β_grad,c` với `κ`. Nếu `κ > 1` và một
`β_grad,c` đã gần 1 thì tích vượt 1, phá hard check `0 ≤ β_grad ≤ 1`. Cần ghi rõ xử lý
(clamp và báo, hay coi là fail).

---

## 3-bis. ⛔ Lỗi tôi gây ra ở nb14/15 — token `CLS` vào ML-Decoder

Phát hiện khi rà code để trả lời câu hỏi về `backbone.layernorm`:

```
11_midvslate_screen.ipynb   return self.mld(self.norm(h[:, 1:]))   <- BO CLS  (1,369 token)
13_dbl_weight_screen.ipynb  return self.mld(self.norm(h[:, 1:]))   <- BO CLS
14_disease_gate.ipynb       z = s.mld(s.norm(tok))                 <- GIU CLS (1,370 token)
15_disease_gate_wasl.ipynb  z = s.mld(s.norm(tok))                 <- GIU CLS
```

`forward_img_tokens` của nb14 trả `backbone.layernorm(h)` với shape `[B, 1+Npatch, 768]`, và
`ImageHead.forward` đưa **nguyên** tensor đó vào decoder. nb11/13 thì cắt `h[:, 1:]`.

H5 được train ở 06b theo quy ước nb11 (`C1` của nb11 ghi *"verbatim 06b"*), tức trên **1,369
patch token**. nb14/15 nạp H5 rồi cho nó ăn **1,370** token.

**Cổng `G2` không bắt được** vì nó so `_M0.img(_tok)` với `D0(_tok)` — hai instance của **cùng
một** `ImageHead`, cùng quy ước sai. `G2 = 2.98e-07` chỉ chứng minh "head của arm bằng một bản
copy của chính nó", không chứng minh "bằng H5 như H5 được train".

### Ảnh hưởng

**Không phá contrast nội bộ.** Mọi arm trong nb14 (`A0`/`A1`) và nb15 (`B1`/`B1u`) dùng cùng
quy ước, nên `G_DA`, `Delta_WASL`, `I_loss×EHR` và toàn bộ phát hiện cơ chế (`res/z` 78% vs
0.25%, gate 96% giống nhau, sụt `0.0128` vs `0.0132`) **vẫn đứng**.

**Phá hai câu trong báo cáo cũ:**
- *"`z_IMG` tại init đúng bằng đầu ra H5"* (nb14 md C5, và §8 của
  `REVIEW_DISEASE_GATE_WITH_TAIL_SELECTIVE_DBL.md`) — chỉ đúng tới mức "bằng H5-head chạy trên
  token có CLS", không phải "bằng H5 như đã train";
- là **một lý do nữa** để absolute number của nb14/15 không so trực tiếp với nb12 (bên cạnh
  việc kiến trúc khác).

**Độ lớn**: thêm một memory token trên 1,370 trong cross-attention. `CLS` của model kiểu DINOv2
là tóm tắt toàn cục, nên nó bổ sung ngữ cảnh — có thể nhẹ nhàng có lợi hoặc có hại, chưa đo.

### Cần chốt cho nb16

Dùng `h[:, 1:]` (khớp H5 + nb11/13) hay giữ `CLS` (khớp nb14/15)? Xem §9 mục 3.

---

## 4. Ba điểm cần bổ sung, không chặn

### 4.1 Tỷ lệ gradient sẽ trôi theo **một chiều đoán được**

§4.4 theo dõi tỷ lệ `R2/R1` và rút lại tuyên bố nếu ra ngoài `[0.75, 1.25]` hai epoch liền.
Tốt và trung thực. Nhưng chiều trôi **đoán trước được**:

- R2 bỏ áp lực đẩy xuống ở contested blank ⇒ `p` của chúng **tăng** ⇒ `(p−0.05)⁴` tăng ⇒ khối
  lượng gradient của nhóm đó tăng (dù bị `η = 0.10` giảm chấn);
- R1-GM dàn mỏng ⇒ ô điểm cao vẫn giữ weight `0.28` và tiếp tục bị đẩy xuống ⇒ `p` giảm ⇒
  khối lượng gradient giảm.

⇒ **`R2/R1` sẽ tăng theo epoch.** Nên ngưỡng bị vượt sẽ là cận **trên** (`> 1.25`), và điều đó
là hành vi bình thường của thiết kế, không phải sự cố. Nên ghi trước để không đọc nhầm.

### 4.2 §5.5 có go/no-go nhưng không có fallback

Cửa sổ ±7 ngày **và** bỏ yêu cầu adjacency (§5.3) làm một bệnh nhân có 1 study dương đánh dấu
**mọi** blank study khác của họ trong 7 ngày. Bệnh nhân ICU chụp 10 phim/tuần với 1 dương →
9 contested cell từ một anchor.

§5.5 bắt được bằng `top 1% subjects ≤ 15% contested gradient mass` ✔. Nhưng **nếu fail thì
làm gì** chưa nói. V3.1 có `w_subject = 1/K`; V3.2 bỏ đi (vì không còn pair loss).

Đề xuất khoá trước một trong hai:

```
(a) cap so contested cell moi subject (vi du <= 3, chon nearest anchor)
(b) chia weight reduction cho so cell cua subject:  eta_eff = 1 - (1-eta)/K_subject
```

### 4.3 Kỳ vọng tiên nghiệm cho `D1` — đây là can thiệp **lớn**

Nếu `s ≈ 0.8` thì `β_grad ≈ 0.28`: R1-GM **xoá ~72%** khối lượng gradient blank-negative của
Rare-3. Đó không phải tinh chỉnh, mà là thay đổi lớn về cân bằng dương/âm.

Điểm đáng nói: nb12–15 đã thử cạn **trục cân bằng theo lớp** (WASL, DBL) nhưng **chưa bao giờ**
thử **trục cân bằng dương/âm**. Nên `D1` có giá trị khoa học độc lập, kể cả khi `D2` null.

Nhưng cũng nên ghi kỳ vọng: giảm mạnh gradient âm ≈ nâng dương tương đối, và nb12–15 cho thấy
họ can thiệp đó thường cho đỉnh sớm + sập nhanh, macro phẳng hoặc giảm nhẹ. Ghi trước để một
`D1` dạng đó được đọc đúng.

---

## 5. Chi phí — con số cụ thể

§13 nói *"materially larger than nb15"*. Định lượng:

```
moi batch:  9 block fwd (no-grad, dung chung)
          + 3 x (3 block fwd+bwd  +  head fwd+bwd)
bwd ~ 2x fwd  =>  3 block fwd+bwd ~ 9 block-equivalent
tong ~ 9 + 3x9 = 36 block-equivalent

nb14/15:  12 block fwd (no-grad) + 2 x head  ~ 12 block-equivalent
```

⇒ **~3× mỗi batch**, và 3 arm thay vì 2.

```
Stage 1 (seed 42, 8 epoch, 3 arm)  ~  4-5 x mot run nb15
+ 2 seed xac nhan                  ~  3 x nua
tong toi da                        ~  12-15 x nb15
```

VRAM không phải vấn đề trên máy 102 GB (3 arm × 27M tham số + Adam ≈ 1 GB; activation của 3
block có grad trên ~1,370 token sẽ do `probe_batch` dò). Thời gian mới là ràng buộc.

§13 chia stage là đúng ✔ — nhưng nên thêm điều kiện dừng sớm dựa trên §1 (nếu drift/ngưỡng
fail thì dừng ngay ở smoke, không vào Stage 1).

---

## 6. Điểm nhỏ

**§4.1 đo ở `eval()` + không augmentation**, còn train ở `train()` + có augmentation. `s` đo
được có thể lệch khỏi `s` lúc train. Là hiệu chuẩn một lần nên chấp nhận được, plan cũng nói
rõ ✔. Chỉ nên ghi chú rằng `s` là giá trị tại init, không phải trung bình theo training.

**`β_grad ∈ [0,1]` luôn thoả** — vì `β_grad = 1 − s(1−η)` với `s ∈ [0,1]`, `η ∈ [0,1]`. Nên
hard check ở §4.2 không bao giờ fail. Đây là **cải tiến thật** so với V3.1 (nơi `β_hi` có thể
vượt 1 và cần rule hạ `β`). Đáng ghi nhận.

**§11 checkpoint rule** giữ nguyên nb12–15 ✔ nên so được với bảng cũ. Nhưng với +21M tham số,
đỉnh có thể còn sớm hơn ep2 — nên log `d_b(t)` theo epoch để biết checkpoint được chọn có nằm
trong vùng backbone đã thật sự học chưa (cùng artefact đã gặp ở nb14 §7).

**§12 cổng `+0.002`** nằm dưới sàn nhiễu nhãn `±0.05`. Không sai vì đây là so sánh **paired**
trên **cùng** bộ nhãn (nhiễu triệt tiêu), nhưng phải ghi rõ và bắt buộc kèm subject-cluster
bootstrap ✔ (đã có).

---

## 7. Điều V3.2 làm đúng — ghi nhận

Cả 5 điểm của review V3.1 đã gỡ, và gỡ tốt:

| điểm V3.1 | V3.2 xử lý |
|---|---|
| weight-matching không khớp gradient | **§4 gradient matching**, dùng `|∂ℓ/∂z|` **chính xác** chứ không chỉ modulator `(p−clip)^γ` — đúng hơn đề xuất của tôi |
| số cell eligible trip ngưỡng | **§5.2 cửa sổ ±7 ngày** cho Rare-3 structural + **§5.3 bỏ adjacency**, nearest anchor only ⇒ không tạo mẫu bậc hai |
| `λ_SP = 1e-5` trơ dưới AdamW | **§6.1 `λ_SP = 0`**, thay bằng hard drift threshold, và ghi rõ *"No claim of stabilization is made from L2-SP"* |
| thiếu số epoch | **§0.7 `EPOCHS = 8`** |
| tiêu chí fallback không định lượng | **§7.1** có ba ngưỡng số cụ thể (dù §1 trên nói chúng chưa hiệu chuẩn) |

Và ba chỗ **vượt** đề xuất của tôi:

**§4.3 parameter-gradient calibration.** Tôi chỉ đề xuất khớp logit-gradient. Plan nhận ra
logit-gradient matching là first-order output matching, **không** phải parameter-gradient
matching, và thêm một tầng kiểm với `κ` binary search. Đúng và cần thiết.

**§4.4 rút lại tuyên bố có điều kiện.** *"D2 remains a practical-policy comparison but the
claim 'equal gradient magnitude' is withdrawn"* — phân biệt đúng giữa "so sánh vẫn hợp lệ" và
"so sánh có kiểm soát". Kèm *"Do not dynamically modify weights after training begins"*.

**§3 định nghĩa R2 trước control.** Đảo thứ tự cho phép `β_grad` suy ra từ R2, và nhờ đó
`β_grad ∈ [0,1]` tự động thoả. Sạch hơn V3.1.

Thêm: **§8** kiểm stochasticity thật trước khi viết RNG-sync (đúng đề xuất), **§5.5**
concentration diagnostics, **§5.4** sensitivity subset theo encounter và theo khoảng thời gian,
**§13** staging theo seed, **§14** danh sách artifact pre-GPU.

---

## 8. Thứ tự đề xuất

```
BUOC 0a  (re, khong GPU)  — nhu plan §5, §14
  raw_label_counts / temporal_manifest_funnel / contested_cells_by_class

BUOC 0b  (mot forward pass) — nhu plan §4.1-4.2
  initial_asl_gradient_share / gradient_matched_beta

BUOC 0c  (MOI — bat buoc, ~vai phut GPU)
  smoke 200 step tren MOT arm
  -> do d_b(t) va feature cosine
  -> ngoai suy sang ep1 / ep8
  -> DAT ngưỡng §7 o 2-3x gia tri ngoai suy, KHOA vao prereg
  -> kiem stochasticity (§8) trong cung run

BUOC 1  chot 15 muc o §9 (6 can can + 9 mac dinh)
        3 muc CHAN:  #1 chuan hoa hang so  |  #2 nguong drift tu 0c  |  #3 quy uoc CLS

BUOC 2  §4.3 parameter-gradient calibration (dung mau so thong nhat, §3 review)

BUOC 3  Stage 1 = seed 42, 3 arm, 8 epoch
```

Bước 0c là bước mới và là bước rẻ nhất trong toàn bộ kế hoạch. Không có nó, §7 sẽ chặn thí
nghiệm bằng những con số chưa ai kiểm.

---

## 9. TOÀN BỘ mục cần chốt trước khi dựng notebook

### 9.1 Cần cân — 6 mục

| # | mục | đề xuất | quyết định |
|---|---|---|---|
| **1** | Chuẩn hoá loss: chia `B×13` hằng số thay `Σw` (§2) | **đổi** | _(chưa)_ |
| **2** | Ngưỡng drift §7 của plan: thêm **Bước 0c** (smoke 200 step) rồi khoá ở 2–3× giá trị ngoại suy (§1) | **thêm** | _(chưa)_ |
| **3** | Token `CLS` cho nb16: `h[:, 1:]` (khớp H5 + nb11/13) hay giữ `CLS` (khớp nb14/15)? (§3-bis) | **`h[:, 1:]`** | _(chưa)_ |
| **4** | `backbone.layernorm` (LN cuối, sau block 11) **buộc phải per-arm** vì nhận output block 11 per-arm. Trainable hay frozen? Nhóm LR nào? Freeze ở ep1 cùng decoder? | **per-arm, trainable, nhóm decoder, freeze ở ep1 cùng decoder** | _(chưa)_ |
| **5** | Fallback khi §5.5 concentration fail (§4.2) | **cap ≤ 3 contested cell/subject** (nearest anchor) — không đổi công thức gradient-matching | _(chưa)_ |
| **6** | `κ·β_grad,c > 1` (§3) | **coi là fail**, không clamp âm thầm | _(chưa)_ |

### 9.2 Mặc định, nói nếu muốn khác — 9 mục

| # | mục | mặc định | quyết định |
|---|---|---|---|
| 7 | "blocks 9–11" | **0-indexed** = ba block cuối (0–8 frozen + 9–11 train = 12) | _(chưa)_ |
| 8 | Grad clip | giữ split nb11–15: **1.0** decoder / **5.0** blocks (plan ghi 5.0 toàn bộ) | _(chưa)_ |
| 9 | Weight decay | chỉ áp cho `ndim > 1`, như nb11–15 | _(chưa)_ |
| 10 | Cache `RAW` bốn trạng thái | ghi **file mới** `cohort_v2/cohort_raw_labels.parquet`, không đụng file cũ | _(chưa)_ |
| 11 | "exact timestamp available" (§5.2 plan) | quyết định **sau Bước 0a** — đếm số study có `00:00:00` trước rồi mới đặt rule | _(chưa)_ |
| 12 | Notebook / thư mục | `16_label_reliability_gradient_matched.ipynb` · `DATA_DIR/label_reliability` | _(chưa)_ |
| 13 | Bootstrap | `n_boot = 2000`, cluster = `subject_id` (V2 §11 ghi 2000; V3.2 không nhắc lại) | _(chưa)_ |
| 14 | Temporal diagnostic VAL | dùng lại danh sách V3.1 §7 + thêm bin `24h–7d` cho cửa sổ ±7 ngày | _(chưa)_ |
| 15 | Seed | một seed mỗi lần (`SEED` + resume riêng) theo staging §13 | _(chưa)_ |

### 9.3 Đã tự trả lời — không cần hỏi

| câu hỏi | trả lời | bằng chứng |
|---|---|---|
| Anchor search ±7 ngày có rò rỉ cross-split không? | **Không thể** | `01·C14`: `stratified_group_split` rồi `cohort['split'] = cohort['subject_id'].map(m)` ⇒ split ở mức **subject**; cùng bệnh nhân ⇒ cùng split |
| `β_grad,c` có cần rule xử lý khi vượt `[0,1]`? | **Không** | `β_grad = 1 − s(1−η)` với `s ∈ [0,1]`, `η ∈ [0,1]` ⇒ luôn thoả. Cải tiến thật so với V3.1 |
| `β_grad,c` cho 10 nhãn non-target? | **= 1 tự động** | `G_cont = 0` ⇒ `β_grad = G_ord/G_ord = 1`. Plan đúng, không cần điều kiện thêm |
| `view_signature` lấy từ đâu? | `view_map.parquet` → `view_group ∈ {frontal, lateral}` → signature `F`/`L`/`FL` | nb11–15 đã filter đúng cột này |
| `cur_hadm`/`cur_edstay` có sẵn không? | Có, trong `cohort_base.parquet` (không phải `cohort_study_level.parquet`) — cần merge | 01·C18 `base_cols` |
