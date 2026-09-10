# CXR-LT 2024 — Stage 4: nâng baseline ảnh, rồi fusion EHR đúng cơ chế

## Context

Stage 0–3 đã xong. Mốc gốc trên CXR-LT (40 nhãn, val 39,293 ảnh), probe tuyến tính trên
RAD-DINO đông cứng:

```
arm chuan  nosq518 / patch       mAP anh 0.2028 | study 0.2146
                                 head 0.4204 | medium 0.1369 | tail 0.0417
EHR-only                         0.1199
EHR + CXR (concat -> tuyen tinh) 0.2068      EHR them: +0.0040
vo dich Task 1 (fine-tune + ens) 0.281       head 0.567 | medium 0.263 | tail 0.136
```

Khoảng cách tới vô địch **không đều**: head 1.35× · medium 1.92× · **tail 3.26×**.

Mục tiêu do người dùng chốt: **cả hai — nâng mAP trước, rồi dùng baseline mạnh đó làm nền cho
các claim phương pháp luận**. Bài toán trung tâm vẫn là **CXR + structured EHR**; không bỏ
nhánh EHR.

---

## 1 · Trạng thái bằng chứng — cái gì đóng, cái gì còn mở

### 1.1 Đã đo trên CXR-LT (cohort hiện tại)

| | |
|---|---|
| `patch` > `cls` | **+0.0377**, 40/40 nhãn (`nosq518`) |
| FOV đầy đủ > crop | **+0.0111** ở `patch` (`nosq518` − `crop518`) |
| bỏ đệm đen (18% → 0%) | **+0.0026** — dưới nhiễu; đệm **không phải** thủ phạm |
| EHR trên `patch`, concat → tuyến tính | **+0.0040**; 11/40 nhãn vượt ngưỡng nhiễu |
| nhãn nhạy EHR (ổn định khi đổi backbone) | Tuberculosis · Pneumothorax · Pneumonia · Rib Fracture · Lung Lesion · Mass · Fracture — **nhiễm trùng + chấn thương** |
| EHR ở **tail** | **−0.0005** — không giúp đúng chỗ ta thiếu nhất |
| trần do nhiễu nhãn (Task 2, 406 ảnh chuyên gia) | **0.6528**; vô địch đạt **81%** của mức đó |

*(Hai dòng `EHR` ở trên đo dưới BCE@30 epoch và đã bị `35` thay — xem 1.1-bis.)*

### 1.1-bis · Đã đo ở `35` (giao thức epoch-cuối, `asl40`, 3 seed)

| | |
|---|---|
| **loss** | `asl40` (γ_neg=4, γ_pos=0) @ `E=60` thắng `bce@150`: macro **+0.0009**, tail **+0.0019**, hai khoảng seed **rời nhau** ở cả hai; và rẻ hơn **≥2.5×** vì BCE chưa hội tụ kể cả ở `E=150` |
| | `pos_weight α=0.5` bắt kịp **tốc độ** nhưng không bắt kịp tail; `α=1.0` kém rõ (tail `0.0474`) |
| **mốc mới** | CXR-only `anh 0.2093 · study 0.2229 · tail 0.0620` — hơn `32` **+0.0083** anh mà **không đổi một dòng kiến trúc** |
| **EHR** | **+0.0020** macro (3× biên độ seed, thật) nhưng **−0.0035** tail với `Q_FULL` |
| | 90 cột **giá trị lâm sàng** đóng góp **đúng bằng 0** cho macro (`Q_CTX 0.2111` vs `Q_FULL 0.2113`) và `−0.003` cho tail. Toàn bộ phần macro đến từ **bối cảnh + missingness** |
| **định tuyến theo bệnh** | trần sau khi trừ thiên lệch oracle (`+0.0006`): `Q_FULL` **+0.0039**, tức dư địa thật của B2c = **+0.0019** macro |
| | dồn hết vào **medium `+0.0028`** và **tail `+0.0050`**; ở head **bằng 0** (EHR đã giúp 12/13 nhãn nên không còn gì để định tuyến) |
| **đầu đọc phi tuyến** | **null sạch**: `MLP − linear = −0.008 … −0.011`. Phi tuyến điểm-theo-điểm trên đặc trưng đã gộp không mua được gì |
| **mốc corr (A5.1)** | same-backbone spearman **0.9060** (khác tiền xử lý) / **0.5925** (khác biểu diễn). Ensemble trọng số đều **ÂM** (`−0.0020`) vì thành viên không đều ⇒ với `41a`, ràng buộc là **ngang chất lượng**, không phải đa dạng |
| **độ phân giải** | chưa ngã ngũ. `crop518` (+1.30× phân giải, −28.5% FOV) thua `−0.0085` **toàn cục** nhưng thắng `+0.006…+0.010` ở 13 nhãn mà crop không cắt mất tổn thương. `lb672` (≈31 ph) hoàn thiện ô 2×2 |

**Hai hệ quả cho `37`:** `Q_CTX` vs `Q_FULL` là **arm quét**, không phải hằng số — không định tuyến
thì `Q_CTX`, có định tuyến thì `Q_FULL`. Và B2c phải chấm bằng **tail/medium**, không phải macro.

### 1.2 Đã đo trên cohort CŨ (211,130 study / **13 nhãn** CheXpert / tail chỉ 1 nhãn)

Xếp hạng đòn bẩy, cùng thang:

```
representation (CLS -> patch)   +0.0219  p=0.002
loss  (DB Loss -> ASL)          +0.0144  p=0.001
EHR tren CLS                    +0.0081  p=0.004
EHR tren PATCH (late fusion)    +0.0003  p=0.830   <- ba co che hoi tu ve 0
vi sinh, ngoai labs             -0.0002  p=0.901
```

Ba phép thử hội tụ cho late fusion trên patch: token end-to-end `+0.0003` · tráo EHR sang bệnh
nhân khác `corr(logit) = 0.999934` · linear stacking `−0.0027`.

**Ba con số tôi đã nói SAI trong quá trình thảo luận — đã sửa:**

| tôi nói | thực tế |
|---|---|
| "ASL hơn BCE +0.0144" | `+0.0144` là **ASL vs DB Loss**. **ASL − BCE = +0.0050**, p=0.029 |
| "multi-view per-disease là ý tưởng mới" | **đã đo ở `17` C7/V1**: Δmacro **+0.00017**, CI [−0.00072, +0.00100] **chứa 0**, tail **xấu đi −0.0016** |
| "cross-attention Q=patch, K/V=EHR nằm trong lớp đã bị bác" | **khái quát quá tay.** `06b` bác bố cục `H6` = `ML-Decoder(Q=40 query lớp, K/V=[1369 patch ; 1 token EHR])`, nơi EHR **giành attention với 1369 patch** và chết vì băng thông. Bố cục `Q=patch, K/V=EHR` có softmax **chỉ trên K phần tử EHR** ⇒ **không thể bị nhấn chìm**. Chưa ai đo — xem §3.2a |

### 1.3 Cơ chế EHR **CÒN SỐNG** — và là cơ chế duy nhất chưa ai đo xong

```
11_midvslate_screen (do dang, 2/3 seed, khong CI):
  G_mid  = M_E - M_U = +0.0036     adapter EHR trong block 5,7
  G_late = L_E - L_U = +0.0032     adapter EHR trong block 10,11
  I_location         =  ~0         -> VI TRI block khong quan trong
07_screen_deep_ehr: KHONG CO OUTPUT NAO; khong report nao ghi ket qua
```

**`+0.003` gấp 10 lần late fusion trên patch (`+0.0003`).** Khác biệt cơ chế: adapter nằm **bên
trong** backbone nên EHR thay đổi *cách representation hình thành* (`Z_l = B_l(Z_{l-1}; e)`),
không phải điều biến output đã đông cứng.

`REPORT_frozen_token_fusion_revised.md` §5 nêu rõ đây là thứ kết quả `06b` **không loại trừ**,
và `CLAUDE_GUIDE_DEEP_EHR_CONDITIONING.md` giới hạn phạm vi kết luận cho **"13 nhãn CheXpert
này"** — không tự động chuyển sang CXR-LT (40 nhãn, tail sâu gấp bội).

### 1.4 Lý do nguyên tắc khiến EHR gain nhỏ trên **loại nhãn này**

Nhãn CheXpert **và** nhãn CXR-LT đều do NLP trích từ **chính report của phim đó** ⇒ ảnh gần như
đủ theo định nghĩa. `34` đo được biểu hiện trực tiếp: nhãn tự động gọi thừa `Pneumonia` **5×**
(115 → 22 ca sau khi chuyên gia đọc lại, **0** lần thêm).

Hệ quả: đừng kỳ vọng EHR mua được nhiều trên trục nhãn này; giá trị của nhánh EHR nằm ở **đo cho
sạch, phát biểu đúng phạm vi**, cộng với cơ chế deep conditioning chưa ai đóng.

### 1.5 MedFuse (Hayat et al., MLHC 2022) — cùng hai CSDL, **ngược chiều**, và số phải chuẩn hoá

MedFuse dùng **đúng MIMIC-IV + MIMIC-CXR** như ta, đạt AUPRC `0.481` (phenotyping 25 nhãn) và
`0.594` (tử vong nội viện). Nhìn thô thì hơn hẳn `0.2028` của ta — nhưng **AUPRC và mAP là cùng
một đại lượng**, và sàn ngẫu nhiên của nó **bằng đúng prevalence**:

```
                        prevalence TB   AUPRC/mAP   BOI SO SAN
MedFuse phenotyping        22.2%          0.481       2.16x
MedFuse tu vong noi vien   12.4%          0.594       4.78x
TA, CXR-LT 40 nhan          6.17%         0.2028      3.29x     <- 96,941 duong / (40 x 39,293)
vo dich CXR-LT              6.17%         0.281       4.55x
```

Nhãn của họ **phổ biến gấp 3.6 lần** nhãn của ta. Chuẩn hoá theo sàn thì **ta đang trên MedFuse
phenotyping** (3.29× vs 2.16×), và vô địch CXR-LT (4.55×) xấp xỉ MedFuse tử vong (4.78×).

**Ba khác biệt cấu trúc quyết định, không phải khác biệt chất lượng:**

| | MedFuse | ta |
|---|---|---|
| modality **chính** | **EHR** — `y_fusion = y_ehr`, chuỗi thời gian *luôn có mặt* | **CXR** |
| modality **phụ** | CXR | EHR |
| nguồn nhãn | **ICD → CCS phenotype**, và tử vong — **KHÔNG** trích từ report của phim | NLP trích từ **chính report của phim đó** |
| đóng góp của modality phụ | `0.453 → 0.481` | `0.2028 → 0.2068` |

```
                          tuyet doi   /baseline   /san (prevalence)
MedFuse  0.453 -> 0.481    +0.028      +6.2%        0.126
ta       0.2028 -> 0.2068  +0.0040     +2.0%        0.065
ti le                       7.0x        3.1x         1.9x
```

**KHÔNG được đọc đây là bất đối xứng theo modality.** Hai thiết lập khác nhau ở **nguồn gốc của
nhãn**, không phải ở chiều fusion. Nguyên tắc đúng là **đối xứng**:

> Nhãn trích từ report của modality X ⇒ **X gần như đủ theo định nghĩa** ⇒ modality thứ hai còn
> rất ít đất. Tổng quát hơn: **mỗi modality đóng góp cho đúng những nhãn mà bằng chứng của nhãn đó
> thật sự nằm trong nó.**

Table 4 của chính MedFuse ủng hộ cách đọc này — ảnh **không** giúp đều:

```
Pleurisy; pneumothorax; pulmonary collapse   0.152 -> 0.212   +0.060
Pneumonia                                    0.416 -> 0.453   +0.037
Other lower respiratory disease              0.194 -> 0.176   -0.018
Respiratory failure; insufficiency; arrest   0.678 -> 0.653   -0.025
Figure 4:  mixed +0.107 | chronic +0.025 | acute +0.001
```

Và nó khớp với `33` C7 của ta: EHR chỉ nhấc được **nhiễm trùng + chấn thương** — đúng nhóm mà xét
nghiệm / vi sinh / bối cảnh nhập viện mang thông tin **không có trên phim**. 29 nhãn còn lại không
nhích, vì bằng chứng của chúng nằm trọn trong ảnh.

**Hệ quả thiết kế:** đừng kỳ vọng fusion nhấc *macro*. Kỳ vọng đúng là **nhấc đúng nhóm nhãn có
bằng chứng ngoài ảnh**, và báo cáo tách nhóm đó ra — nếu gộp vào macro 40 nhãn thì 29 nhãn trơ sẽ
pha loãng mọi hiệu ứng thật xuống dưới sàn nhiễu.

### 1.6 Bài học kiến trúc từ MedFuse — **ngược** với hướng ta đang thiết kế

Bảng 2 của họ, cùng test set, phenotyping AUPRC:

```
MMTM      0.428   <- module fusion CHUYEN DUNG
DAFT      0.434   <- module fusion CHUYEN DUNG
Early     0.453
Joint     0.446
Unified   0.461
MedFuse   0.481   <- mot LSTM tren chuoi HAI phan tu [v_ehr, v_cxr]
```

**Hai module fusion tinh vi nhất lại xếp cuối.** Và toàn bộ khoảng cách của MedFuse đến từ **chiến
lược huấn luyện**, không phải kiến trúc:

```
MedFuse (RI, encoder khoi tao ngau nhien)   0.452
MedFuse (PT, encoder pre-train rieng)       0.466   +0.014
MedFuse (OPTIMAL, chinh ti le mau uni-modal) 0.481  +0.015
```

Cộng với Khader (early → cross-attention chỉ `+0.033` AUC) và `06` của ta (**FiLM ≈ concat,
p=0.308**), đây là **nguồn bằng chứng độc lập thứ ba** nói cùng một chuyện: **lợi ích nằm ở chiến
lược dữ liệu/huấn luyện, không nằm ở độ tinh vi của khối fusion.**

**Hệ quả cho kế hoạch:** giữ B2a/B2b/B2c (chúng rẻ và đo được cơ chế), nhưng **đừng leo thang kiến
trúc** khi chưa khai thác hai thứ MedFuse đo được là có tác dụng: **pre-train encoder riêng từng
modality** trước khi fusion (`+0.014`), và **chỉnh tỉ lệ mẫu uni-modal khi fine-tune** (`+0.015`).
Cả hai đều rẻ và ta chưa làm cái nào.

---

## 2 · PHẦN A — Nâng baseline ảnh (làm TRƯỚC)

### A1 · ASL thay BCE — `nb35` · **XONG**

> **Ket qua:** `asl40` (gn=4, gp=0) @ `E=60` — thang `bce@150` ca macro (+0.0009) lan tail (+0.0019),
> hai khoang seed roi nhau, va re hon >=2.5x. `WASL alpha=.50` **thoai hoa ve ASL** (15/40 nhan kep
> san cua clip [0.5,3.0] duoi mat can bang 673x). `pos_weight` la ban khong bi kep cua cung y tuong:
> `alpha=0.5` bat kip toc do nhung khong bat kip tail; `alpha=1.0` kem ro.
> **`+0.0072` ghi o ban dau la SAI** — no do BCE o mot ngan sach chua hoi tu, khong do loss.

**Vì sao.** Probe hiện dùng `BCEWithLogitsLoss`. Cohort cũ: **ASL − BCE = +0.0050** (p=0.029),
ASL thắng **cả ba tier** (head 0.5979 / medium 0.2387 / tail 0.0843 vs BCE 0.5931 / 0.2339 /
0.0771). Tail ta thua 3.26× nên đúng trục.

```python
class ASL(nn.Module):                       # Ben-Baruch et al. 2021
    def __init__(s, g_neg=4, g_pos=0, clip=0.05):
        super().__init__(); s.gn, s.gp, s.clip = g_neg, g_pos, clip
    def forward(s, x, y):
        p  = torch.sigmoid(x)
        pm = (p - s.clip).clamp(min=0)      # probability shifting: bo negative de
        L  = y*torch.log(p.clamp(min=1e-8))*(1-p)**s.gp \
           + (1-y)*torch.log((1-pm).clamp(min=1e-8))*pm**s.gn
        return -L.mean()
```

Quét `(g_neg, g_pos) ∈ {(2,0), (4,0), (4,1)}` + WASL `α=.50`.

**Cổng khoá trước:** ASL − BCE ≥ **+0.003** mAP ảnh (nửa hiệu ứng cũ, vì 40 nhãn ≠ 13 nhãn).
Báo cáo **tách theo tier** — kỳ vọng chính ở tail.

**Bẫy đã biết:** WASL .50 trên cohort cũ **overfit rất nhanh** — tail `0.1230@ep2 → 0.1098@ep8`
(**−0.0132**). Bắt buộc early-stop theo **tail mAP**, không theo macro.

**Chi phí:** ~20 ph, embedding đã cache, **không trích lại**.

---

### A2 · Kiem `Case C` — **CHUYEN sang `37` thanh o 2x2**

> `cls` chi la bien thay the cho "phep gop te". Tu `37` tro di khong gi dung `cls`: ML-Decoder doc
> 1369 patch, `Q=patch K/V=EHR` cung patch, tang 2 cua B2c la `[patch ; ehr]`. Nen do `gain_cls`
> tao ra mot con so khong notebook nao hanh dong duoc theo.
> Cau hoi Case C tra loi TRUC TIEP hon o `37`, bang phep gop THAT:
> ```
>                        khong EHR   co EHR
> mean-pool + linear      0.2093     0.2111    <- da co o 35
> ML-Decoder(1369)          ?          ?       <- 37
> ```
> Lift EHR **sut** khi len ML-Decoder => Case C dung. **Giu nguyen** => EHR doc lap voi phep gop.
> Bon arm dung chung mot forward backbone nen gan nhu mien phi.

### A2-cu · Kiem `Case C` trên CXR-LT — `nb35` *(khong chay)*

**Vì sao.** Cohort cũ: EHR@CLS `+0.0081` nhưng EHR@patch `+0.0003` ⇒ phần lớn gain của EHR chỉ
**bù bottleneck của global pooling**. Ta đã có `+0.0040` trên patch nhưng **chưa có** số trên
`cls`, nên chưa biết kết luận đó có lặp lại trên 40 nhãn không.

```
gain_cls   = EHR+CXR(nosq518/cls)   - CXR-only(nosq518/cls)      <- CAN DO
gain_patch = EHR+CXR(nosq518/patch) - CXR-only(nosq518/patch)    = +0.0040 (da co)
```

`gain_cls >> gain_patch` ⇒ Case C lặp lại, EHR chỉ vá pooling.
`gain_cls ≈ gain_patch` ⇒ trên 40 nhãn EHR mang thông tin **độc lập** — **khác** cohort cũ, và
đó tự nó là phát hiện.

**Chi phí:** ~3 phút. Phép đo rẻ nhất có sức nặng lớn nhất trong kế hoạch.

---

### A3 · Đầu đọc ML-Decoder trên **đủ 1369 patch token** — `nb37`

**Vì sao.** `patch` hiện là **trung bình 1369 token** — một vector cho mọi nhãn, mọi vị trí. Cohort
cũ: thay bằng ML-Decoder đọc **toàn bộ patch** cho **+0.0219** (`H5 0.4336` vs `H1 0.4117`). Trên
CXR-LT trục representation còn lớn hơn (`patch − cls = +0.0377`, 40/40 nhãn).

**KHÔNG gộp lưới. Không cache. Làm đúng như `06b` đã làm.**

```python
@torch.no_grad()
def forward_backbone(images):            # 06b C7 — mot forward DUNG CHUNG cho moi arm
    h = backbone(pixel_values=images).last_hidden_state      # [B, 1+1369, 768]
    return h[:, 0].detach().float(), h[:, 1:].detach().float()
```

Ảnh stage xuống đĩa local (`06b` C1), backbone đông cứng chạy **trong vòng lặp**. VRAM
`[128, 1369, 768]` bf16 = **269 MB** — không phải ràng buộc. ~20 ph/epoch.

> **Đã bỏ `nb36` (gộp lưới `6×6` rồi cache).** Nó chỉ là cách lách chuyện không cache nổi 1369
> token (`258,871 × 1369 × 768 × 2 B = 544 GB`), và nó **phá đúng thứ ta cần**: ô `6×6` rộng ~68mm
> trong khi đường gãy xương đòn rộng 1–2mm. Ô lớn nhất còn cache được (`10×10`, ~42mm) vẫn lớn hơn
> 20 lần. `06b` đã chứng minh không cần cache.

**Và lý lẽ "cache để sàng lọc nhiều biến thể cho nhanh" cũng không đứng được** — `AMENDMENTS` ① đã
chứng minh: tham số các arm **rời nhau** ⇒

```
d(L1 + L2 + ... + Ln) / d(theta_i)  =  dLi / d(theta_i)
```

⇒ huấn luyện `n` đầu đọc trong **một** vòng lặp là **tương đương toán học tuyệt đối** với `n` lần
chạy riêng, không phải xấp xỉ. `06b` chạy **8 head trên một forward**: 7.5 h thay vì 30 h. Nên quét
`{cls, patch} × {MLP, ML-Decoder} × K ∈ {1,8,250} × {có, không} EHR` trong `nb37` **gần như miễn phí
so với chạy một arm** — không cần representation thay thế nào.

**Một thứ nhỏ vẫn đáng thêm vào `31`, và nó KHÔNG cần lưới:** lưu thêm **`patch_max`** — global
max-pool trên 1369 token, cùng kích thước `[N, 768]` như `patch` hiện tại. `max` **không tuyến
tính** nên giữ được đỉnh cục bộ mà trung bình pha loãng; nó kiểm thẳng câu *"trung bình có phải là
phép gộp sai không"* ở **mức probe**, gần như miễn phí (một phép reduce nữa trong cùng forward).
Với `nb37` thì không cần — attention của ML-Decoder **chính là** phép gộp học được, và nó bao trùm
mọi phép gộp thủ công.

**Đơn giản hoá được biện minh bằng số của chính ta:** dùng **`lb518`** (vuông, DataLoader tầm
thường) thay `nosq518` (kích thước thay đổi từng ảnh, phải gom batch theo hình dạng). Chênh lệch đo
được chỉ `+0.0026`, dưới nhiễu. Lấy lại sau nếu `nb37` cho kết quả đáng kể.

**Cổng:** ML-Decoder(1369) − mean-linear ≥ **+0.010** mAP ảnh (nửa hiệu ứng cũ `+0.0219`).

---

### A4 · Partial unfreeze / LP-FT — `nb38`

**Vì sao.** Đòn bẩy lớn nhất, đã đo: LP `0.3191` → FT `0.4201` = **+0.101**. Ta đang ở LP
`0.2028`. Đây cũng là thứ mở đường cho **Joint Fusion Type I** ở phần B.

**Cấu hình đã biết chạy được (`04b`):** `BATCH=48 @518² bf16`; FT **~27 ph/epoch ≈ 150 img/s**,
LP ~10 ph/epoch ≈ 400 img/s. Ước tính CXR-LT (258,871 ảnh): **~29 ph/epoch**.
Adapter-only (`no_grad` phần dưới) chạy được `BATCH 96` chỉ tốn **15.7 GB**.

#### Bao nhiêu epoch — **không phải ngân sách, mà là điểm dừng phải ĐO**

`04b` đã chạy **đủ 12 epoch** trên cohort cũ và đường cong là:

```
0.3790  0.4035  0.4101  0.4162  0.4178  0.4201  0.4200  0.4187  0.4167  0.4165  0.4164  0.4164
                                          ep5 ^^^^^^   roi TROI XUONG lien tuc toi ep12
```

Nên **chạy dài hơn với CÙNG cấu hình sẽ tệ hơn**, không phải tốt hơn — `ep12` thua `ep5` **0.0037**.
Con số "5–6 epoch" là **đỉnh quan sát được**, không phải giới hạn tài nguyên.

**Nhưng đường cong đó có thể KHÔNG chuyển sang CXR-LT, vì một lý do cụ thể.** Cohort cũ có **đúng
một** nhãn tail (prevalence 0.0119); CXR-LT có **11 nhãn tail**, hiếm nhất 129 ca dương train. Với
`BATCH 48` thì một epoch có 5,393 batch nhưng `Lobar Atelectasis` chỉ xuất hiện trong ~129 batch —
**lớp hiếm nhận rất ít tín hiệu gradient mỗi epoch**.

⇒ **`ep5` của cohort cũ là đỉnh do HEAD quyết định.** Trên CXR-LT, tail hoàn toàn có thể **vẫn đang
lên** khi head đã bắt đầu overfit. Ta không biết, và **phải đo**.

**Ba việc để câu "bao nhiêu epoch" được trả lời bằng dữ liệu:**

1. **Ghi đường cong val THEO TIER mỗi epoch**, không chỉ macro. Đây là thay đổi rẻ nhất và có giá
   trị nhất — nó biến một giả định thành một biểu đồ.
2. **Early-stop theo `tail mAP`, không theo macro.** Cùng lý do đã áp cho WASL ở A1. Nếu head peak
   ep5 mà tail peak ep9 thì dừng theo macro là vứt bỏ phần tail.
3. **Chạy đủ 12 epoch ở lần đầu** (`PATIENCE` lớn), lưu checkpoint mỗi epoch. Tốn ~6 h thay vì 3 h,
   nhưng ta **chỉ phải trả một lần** để biết hình dạng đường cong trên cohort này — và mọi thí
   nghiệm sau đó dùng lại kết luận đó.

#### Và nếu ba tier peak ở ba epoch khác nhau — đừng chọn, hãy ENSEMBLE

`17` đã đo trên cohort cũ: ensemble ba checkpoint **của cùng một lần chạy** cho

```
macro  +0.00112  CI [+0.00027, +0.00197]  loai 0
worst3 +0.00345  CI [+0.00035, +0.00645]  loai 0
tail   +0.00465  CI [-0.00230, +0.01213]  chua 0
```

**Mức tăng lớn nhất nằm ở `tail` và `worst3`** — đúng nhóm mà "chọn một epoch" gây thiệt hại nhất.
Nên câu hỏi *"epoch nào"* tự tan thành *"trung bình xác suất của các epoch quanh đỉnh"* (ví dụ
`{4, 6, 8}`), dùng lại đúng máy móc của A5 và **không tốn thêm một giây GPU nào** — checkpoint đã
lưu sẵn.

**Cổng:** FT − LP ≥ **+0.03** mAP ảnh. Báo cáo **tách theo tier**, và kèm epoch đạt đỉnh của **từng
tier**.

---

### A5 · Ensemble — kiểm MIỄN PHÍ trước, rồi mới thêm một CNN — `nb35` → `nb41`

**Vô địch CXR-LT dùng fine-tune + ensemble**, nên đây là phần của công thức, không phải phụ kiện.

**Bằng chứng sẵn có (cohort cũ, `17`).** Ensemble `mean_prob` của **ba checkpoint cùng một mô hình**:

```
single R0_ep2   macro 0.4365 | worst3 0.1339 | tail 0.1191
ENSEMBLE        macro 0.4376 | worst3 0.1373 | tail 0.1238
delta (paired, 2000 resample, cluster=subject)
  macro  +0.00112  CI [+0.00027, +0.00197]  loai 0
  worst3 +0.00345  CI [+0.00035, +0.00645]  loai 0
  tail   +0.00465  CI [-0.00230, +0.01213]  chua 0
```

Nhỏ nhưng **thật** ở macro và worst3. Và đó là trường hợp **tệ nhất** cho ensemble: ba checkpoint
của cùng một mô hình ⇒ tương quan cực cao. Đa dạng **kiến trúc** phải cho nhiều hơn.

#### A5.1 · Kiểm miễn phí, làm ngay trong `nb35`

Ta **đã có 9 file `p_*.npy`** trên val (3 arm × 3 biểu diễn) từ `32`. Trung bình xác suất giữa
chúng **không tốn một giây GPU nào**, và nó trả lời câu hỏi trước khi tiêu 3 giờ:

```
1. mAP cua ensemble {crop518, lb518, nosq518} x {patch}     vs  nosq518/patch don le
2. mAP cua ensemble {cls, patch} tren cung mot arm
3. TUONG QUAN du doan giua cac thanh vien, tung nhan
```

Điểm 3 là chẩn đoán quyết định: **ensemble chỉ mua được khi các thành viên KHÔNG tương quan.**

```
corr > 0.95  -> ensemble mua duoc ~0; ConvNeXt cung se kho cuu neu no chi la "them mot CNN"
corr < 0.90  -> con dat; da dang kien truc dang dau tu
```

#### A5.2 · Vì sao thêm một **CNN** chứ không phải một ViT nữa

Nó mang lại **đúng thứ RAD-DINO không có**: **phân cấp stride 4/8/16/32**. Đây chính là câu FPN ở
A6 — ViT thuần không có pyramid nên FPN không bám vào đâu được, nhưng ConvNeXt **có sẵn**. Ensemble
lấy được đa tỉ lệ mà **không** phải gá FPN lên một ViT.

#### A5.2a · Lấy trọng số khởi tạo ở đâu — **ImageNet, không phải CXR**

**Có ConvNeXt pretrain sẵn trên X-quang không?** Theo hiểu biết hiện tại: **không có checkpoint
ConvNeXt công khai nào pretrain trên CXR**. Các mô hình CXR công khai đều là kiến trúc khác:

```
TorchXRayVision  DenseNet-121 / ResNet-50   (-all, -mimic_ch, -mimic_nb, -chex, -nih, -pc, -rsna)
RAD-DINO         ViT-B/14   (DINOv2, tu giam sat)      <- ta dang dung
BiomedCLIP       ViT-B/16
CheXzero         ViT (CLIP)
Ark / Ark+       Swin
```

⚠️ **Cần xác nhận trước khi chốt** — hệ sinh thái CXR đổi nhanh; kiểm `timm.list_models()` và
HuggingFace Hub trên Colab.

**Và nếu có thì cũng KHÔNG nên dùng — hai lý do độc lập.**

**(1) Rò rỉ nhãn test.** `torchxrayvision` `-all` / `-mimic_*` được **huấn luyện CÓ GIÁM SÁT trên
nhãn MIMIC-CXR**. Split test của CXR-LT là **tập con của MIMIC-CXR** ⇒ mô hình đó **đã thấy ảnh test
của ta kèm nhãn**. Dùng làm thành viên ensemble là rò rỉ trực tiếp.

RAD-DINO **khác hẳn**: DINOv2 **tự giám sát, không nhãn** — nó học biểu diễn chứ không ghi nhớ nhãn,
và nó là backbone mặc định của chính giải đấu.

**(2) Đa dạng — và đây là chỗ trực giác "pretrain X-quang sẽ nhẹ hơn" đi ngược mục tiêu.** Mục đích
của ConvNeXt ở đây **không phải** để mạnh một mình, mà để **ít tương quan** với RAD-DINO. Một CNN
pretrain trên CXR chia sẻ **cùng miền dữ liệu** với RAD-DINO ⇒ tương quan **cao hơn** ⇒ ensemble mua
được **ít hơn**. ImageNet cho đa dạng ở **cả hai trục**: kiến trúc **và** miền pretrain.

#### A5.2b · Kéo về và fine-tune — **đúng quy trình DenseNet**, một dòng `timm`

```python
import timm
# fine-tune truc tiep
m = timm.create_model('convnext_base.fb_in22k_ft_in1k',
                      pretrained=True, num_classes=40, in_chans=1)
# hoac trich dac trung cho probe (buoc 1 cua A5.2c)
m = timm.create_model('convnext_base.fb_in22k_ft_in1k',
                      pretrained=True, num_classes=0, in_chans=1)   # -> [B, 1024] da global-pool
```

`in_chans=1` để `timm` **tự gộp trọng số stem RGB thành 1 kênh** — sạch hơn nhân bản 3 kênh như
`31` đang làm cho RAD-DINO, và tiết kiệm compute ở stem.

#### A5.2c · Chọn backbone nào — bảng chi phí ở **518px**

```
RAD-DINO ViT-B/14  @518   1,370 token   ~151 GMAC/anh      (moc so sanh)
```

| backbone | tên `timm` | GMAC @518 | so RAD-DINO | IN-1k top-1 * | ghi chú |
|---|---|---|---|---|---|
| **ConvNeXtV2-N** | `convnextv2_nano.fcmae_ft_in22k_in1k` | **~13** | **11.6× rẻ** | ~82.0 | **rẻ hơn DenseNet mà mạnh hơn ~6pp** |
| **DenseNet-121** | `densenet121.ra_in1k` | ~15 | 9.8× rẻ | ~75.6 | CNN chuẩn của CXR (CheXNet, TorchXRayVision, nhánh CNN của MedFusionNet) |
| EfficientNetV2-S | `tf_efficientnetv2_s.in21k_ft_in1k` | ~16 | 9.4× rẻ | ~83.9 | |
| ResNet-50 | `resnet50.a1_in1k` | ~22 | 6.9× rẻ | ~80.4 | đối chứng kinh điển |
| ConvNeXtV2-T | `convnextv2_tiny.fcmae_ft_in22k_in1k` | ~24 | 6.3× rẻ | ~84.9 | |
| ConvNeXt-S (V1) | `convnext_small.fb_in22k_ft_in1k` | ~46 | 3.2× rẻ | ~84.6 | |
| ConvNeXtV2-B | `convnextv2_base.fcmae_ft_in22k_in1k` | ~82 | 1.8× rẻ | ~86.8 | mạnh nhất trong nhóm |
| ConvNeXt-B (V1) | `convnext_base.fb_in22k_ft_in1k` | ~82 | 1.8× rẻ | ~85.8 | V2 hơn ~1pp ở cùng chi phí |

`*` số IN-1k lấy từ trí nhớ, **cần xác nhận** bằng `timm.list_models(pretrained=True)` + model card.

> **Sửa số:** lượt trước tôi ghi ConvNeXt-B @518 ≈ 41 GMAC — **sai**. Đúng là `15.4 GMAC @224 ×
> (518/224)² = 82 GMAC`. Nên ConvNeXt-B chỉ rẻ hơn **1.8×**, không phải 3.7×.

#### V2 khác V1 ở đâu, và điều đó có ý nghĩa gì ở đây

```
FCMAE  masked autoencoder TOAN TICH CHAP, tu giam sat (sparse conv, CHI o giai doan pretrain)
GRN    Global Response Normalization trong block — chong sup dac trung khi huan luyen bang MAE
       (thay vai tro cua LayerScale o V1).  FLOPs them ~ khong dang ke
```

⇒ **chi phí fine-tune của V2 y hệt V1 ở cùng kích thước**; sparse conv không xuất hiện ở đường
fine-tune. Nên V2 là **nâng cấp gần như miễn phí** (~+1pp IN-1k ở cùng GMAC).

Hai lưu ý thật:

**Chọn biến thể `.fcmae_ft_in22k_in1k` làm mặc định.** Bản `.fcmae` thuần chỉ có trọng số tự giám
sát, không head — nó **probe kém hơn** nhưng fine-tune tốt hơn *khi đủ dài* (công thức gốc của
ConvNeXt V2 là 100+ epoch trên ImageNet). **Đừng suy đường cong của ConvNeXt từ đường cong của RAD-DINO** — khác kiến trúc, khác quy trình
pretrain. `04b` chỉ nói về ViT trên 13 nhãn cohort cũ. Câu "bao nhiêu epoch cho CNN" phải trả lời
bằng **đường cong của chính CNN đó**, ghi theo tier, đúng như A4 làm cho RAD-DINO.

**V2 kích thước nhỏ là thứ V1 không có.** `convnextv2_nano` ~13 GMAC @518 — **rẻ hơn DenseNet-121**
(~15) mà IN-1k cao hơn khoảng **6 điểm**. Đây là lý do tôi đổi đề xuất.

#### Đề xuất: **đừng chọn bằng lý lẽ, chọn bằng phép đo** — cả ba trong MỘT lượt quét

Ta không có bằng chứng nào trên CXR để phân định giữa chúng; mọi con số ở trên là **ImageNet**.
Nhưng bước probe (A5.2e) **đọc ảnh mới là phần đắt**, còn chạy thêm một backbone trên cùng batch
gần như miễn phí — đúng lập luận ① của `AMENDMENTS` (một lượt đọc, nhiều nhánh).

```
MOT luot quet shard lb518  ->  trich dac trung cho CA BA:
    convnextv2_nano.fcmae_ft_in22k_in1k   ~13 GMAC
    densenet121.ra_in1k                   ~15 GMAC   <- moc "da duoc dung tren CXR"
    convnextv2_tiny.fcmae_ft_in22k_in1k   ~24 GMAC   <- xem tang kich thuoc co giup khong
tong ~52 GMAC/anh, van RE HON mot minh RAD-DINO (151)
```

Rồi chọn theo **hai** số đo được, không theo IN-1k:

```
1. mAP probe            -> con manh den dau tren DUNG bai toan nay
2. corr voi RAD-DINO    -> con DOC LAP den dau        <- day moi la thu quyet dinh ensemble
```

Chỉ **một** con trong ba đi tiếp sang fine-tune. Nếu **cả ba** đều `corr > 0.90` thì hướng ensemble
CNN đóng lại, và ta tiết kiệm được cả buổi mà không phải fine-tune con nào.

**Chạy ở 518, không phải 224/384.** CNN là toàn tích chập + global pool nên nhận **mọi kích thước**
— không cần phẫu thuật pos-embed như ViT. Ở 518 thì **tái dùng nguyên đường ống ảnh của `31`**
(cùng shard, cùng tiền xử lý), và bỏ được một biến gây nhiễu khi so sánh.

**Cảnh báo bộ nhớ:** ở 518² các tầng đầu của CNN có activation rất lớn (stem stride-4 ⇒ 129×129).
DenseNet-121 còn nặng thêm vì **nối chồng** trong dense block. `BATCH 48` của `04b` có thể phải hạ
— dò batch trước, đừng giả định.

#### A5.2d · Ước tính thời gian

`04b` đo **RAD-DINO** fine-tune ~150 img/s (`BATCH 48`, bf16). Bảng dưới **ngoại suy theo tỉ lệ
GMAC** sang kiến trúc khác — đó là ước lượng compute thuần, **không** phải số đo, và CNN có đặc
tính bộ nhớ/bandwidth khác hẳn ViT. Dò batch và đo img/s thật trước khi tin:

```
RAD-DINO        ~29 ph/epoch  x 5 epoch (dinh o ep5)  ~= 3 h
ConvNeXtV2-B    ~16 ph/epoch                          ~= 1.4 h
ConvNeXtV2-T    ~5  ph/epoch                          ~= 0.5 h
ConvNeXtV2-N    ~3  ph/epoch                          ~= 0.3 h
DenseNet-121    ~3  ph/epoch                          ~= 0.3 h
```

Với mọi con dưới ~25 GMAC thì **đọc ảnh là nút thắt, không phải GPU** — các con số trên sẽ **bằng
nhau trên thực tế**, và đó là lý do nữa để quét cả ba trong một lượt. Stage ảnh xuống đĩa local
trước, đúng cách `06b` C1 đã làm.

#### A5.2e · `nb41a` — probe 3 CNN trong MỘT lượt quét, đo corr THẬT *(đã chốt)*

Probe **không** dùng làm thành viên ensemble, nhưng nó trả lời **hai** câu trước khi tiêu giờ GPU
fine-tune: *khoảng cách miền bao xa* và — quan trọng hơn — ***CNN có thật sự độc lập với RAD-DINO
không***.

**Một lượt quét shard `lb518`, ba backbone dùng chung phần giải mã ảnh** (đọc ảnh mới là phần đắt):

```python
import timm
NAMES = ['convnextv2_nano.fcmae_ft_in22k_in1k',   # ~13 GMAC @518, feat 640
         'densenet121.ra_in1k',                    # ~15 GMAC,      feat 1024
         'convnextv2_tiny.fcmae_ft_in22k_in1k']    # ~24 GMAC,      feat 768
M = [timm.create_model(n, pretrained=True, num_classes=0, in_chans=1).eval().cuda()
     for n in NAMES]                               # num_classes=0 -> tra ve dac trung DA global-pool
```

`in_chans=1` để `timm` gộp trọng số stem RGB → 1 kênh; chuẩn hoá bằng `mean/std` của **chính model
đó** (`timm.data.resolve_model_data_config`), lấy trung bình 3 kênh.

**Dung lượng:** `377,110 × (640 + 1024 + 768) × 2 B (fp16) ≈ 1.8 GB` — không đáng kể.
**Chi phí:** tổng `~52 GMAC/ảnh`, **vẫn rẻ hơn một mình RAD-DINO (151)**; nút thắt là đọc 49 GB
shard ⇒ dùng **đọc bulk + 8 luồng + checkpoint từng tar** (§4). Ước tính **~40 phút**.

Tái dùng **nguyên khung `31`** (memmap, `_done_*.json`, cổng hàng-0, cổng tái lập cosine), chỉ đổi
phần nạp model và vòng lặp thành 3 model trên cùng batch.

**Rồi probe tuyến tính từng cái** (dùng `fit_probe` của `32`) và đo **hai** số:

```
1. mAP probe        -> khoang cach mien
2. corr voi RAD-DINO (spearman tren hang, TB 40 nhan)
     doi chieu voi MOC same-backbone do o A5.1 (nb35)
     -> "0.70 la cao hay thap" tra loi duoc, thay vi phong doan
```

**Chỉ MỘT con trong ba đi tiếp sang fine-tune.** Nếu **cả ba** đều có corr ≥ mốc same-backbone thì
kiến trúc CNN **không** mang lại đa dạng ở bài toán này, hướng ensemble CNN đóng, và ta tiết kiệm
được toàn bộ phần fine-tune.

```
mAP_probe >= 0.15  -> khoang cach mien nho, fine-tune dong lai duoc
mAP_probe <  0.10  -> qua xa; can nhieu epoch hon HOAC bo backbone do
                      (dung suy so epoch tu duong cong RAD-DINO — xem A5.2c)
```

#### A5.2f · Cổng khoá trước cho `nb41b` (fine-tune + ensemble)

```
1. ConvNeXt don le >= 0.85 x mAP cua RAD-DINO don le   -> du manh de vao ensemble
                    < 0.85                             -> KHONG ensemble; se keo tut
2. corr(du doan) < 0.90                                -> co da dang that
3. ensemble - thanh vien manh nhat >= +0.005           -> vuot san nhieu
```

Điều kiện 1 và 2 phải **cùng** đạt. Một mô hình yếu-nhưng-khác-biệt và một mô hình mạnh-nhưng-trùng
lặp đều vô dụng theo hai kiểu khác nhau.

**Trọng số:** dùng **đều nhau**, hoặc chọn trên **lát cắt train** (§4). Chọn trọng số per-label trên
val rồi báo cáo trên val là đúng cái bẫy §13 — và với 19–129 ca dương ở tail thì nó sẽ overfit chắc
chắn.

**Thứ tự:** chạy **sau A4**. Ensemble hai mô hình đã fine-tune mới có nghĩa; ensemble hai probe yếu
thì chỉ đo được nhiễu.

---

### A6 · Việc KHÔNG nên làm (đã có bằng chứng âm)

| ý tưởng | bằng chứng |
|---|---|
| **multi-view `w_l` theo bệnh** | `17` C7/V1: Δmacro **+0.00017**, CI [−0.00072, +0.00100] **chứa 0**, tail **−0.0016**. Trần cấu trúc: chỉ **48.3%** study có cả hai view; toàn dải sweep `w_f` rộng **0.0019** và được chọn trên **TEST** |
| **FPN đa tỉ lệ bằng avg-pool** | mức thô là tổ hợp tuyến tính của mức mịn ⇒ **vô thông tin**. RAD-DINO là ViT thuần — 12 block **cùng một lưới**, không có phân cấp stride để FPN bám vào (ViTDet: top-down + lateral **không cần thiết** cho ViT thuần). FPN của MedFusionNet đặt trên **DenseNet**, có phân cấp thật. → muốn đa tỉ lệ thì lấy qua **ConvNeXt ở A5.2**, đừng gá FPN lên ViT |
| **gộp nhiều block (b11 ⊕ b12)** | `08` probe: b11 0.4026 > b12 0.3962, nhưng qua ML-Decoder chỉ còn **+0.0003** — dưới nhiễu ±0.005. `09` **không có output nào** |
| **mở rộng micro / vitals / comorbidity** | `06`: D3_12 − D3_labs = **−0.0002** (p=0.901); vitals 0.4037/0.4037; como 0.4037/0.4027 |
| **imputer / generator cho EHR thiếu** | missingness của ta là **cấu trúc**, không phải MAR: MIMIC-IV **không có bảng vital ngoại trú**; 11,839 study **không có bản ghi nào**. Generator chỉ huấn luyện được ở ICU/ED ⇒ áp lên ngoại trú là ngoại suy ngoài miền, và **xoá mất tín hiệu** "bệnh nhân này không nằm ICU" |

**Một việc rẻ vẫn nên làm:** in tỉ lệ study val có lateral trên CXR-LT (`32` C2 đã có sẵn dòng
đó). Nếu khác hẳn 48.3% của cohort cũ thì trần của dòng đầu bảng trên đổi.

---

## 3 · PHẦN B — Fusion EHR, làm đúng cơ chế còn sống

### B1 · Đầu đọc phi tuyến: Early Type II → **Joint Type II** — `nb35` · **XONG, NULL**

> **Ket qua:** `MLP - linear = -0.008 .. -0.011` tren ca bon nhanh du lieu. Phi tuyen
> **diem-theo-diem** tren dac trung DA GOP khong mua duoc gi. Dieu nay **khong** bac cross-attention
> o `37` — do la **chon loc theo khong gian**, mot lop ham khac han.
> Ba nhanh missingness: `Q_CTX (157 cot, KHONG mot gia tri lam sang nao)` bang dung `Q_FULL (247)`
> o macro (0.2111 vs 0.2113) va **hon** o tail (-0.0009 vs -0.0035). Hang so dien (0 hay trung vi)
> khong quan trong.

**Vì sao.** Theo Huang et al. (npj Digit Med 2020, Table 2), setup hiện tại là **Early Fusion
Type II**. Table 2 ghi Early **có** ✓ ở *"model interactions between features from different
modalities"* — nhưng ✓ đó thuộc về **điểm hợp nhất**, còn `32` C6 đặt lên trên một **lớp tuyến
tính**: `W_e·e + W_h·h`, **không một số hạng chéo nào**. Ta đã ném đi đúng khả năng mà bảng ghi
công cho early fusion.

Thêm **bộ mã hoá EHR có gradient** ⇒ thành **Joint Fusion Type II** theo đúng định nghĩa của
survey (loss lan ngược vào bộ trích đặc trưng của một modality).

**Và nó sửa luôn bài toán missingness.** `29b` đã có cột `*_mask`, nhưng `32` C6 làm
`nan_to_num(0)` **sau chuẩn hoá** ⇒ "thiếu" trùng điểm với "bằng trung bình". Để phân biệt, mô
hình phải học tích `mask × value` — **lớp tuyến tính không nhân được hai đặc trưng**.
Missingness và interaction là **cùng một vấn đề** ở giai đoạn này.

Ba nhánh xử lý thiếu, cùng split cùng seed:

```
zero         nan -> 0 sau chuan hoa           (hien tai)
median+mask  trung vi train + mask ro rang
mask-only    VUT HET gia tri, chi giu mask    <- = Q_CTX cua nb30
```

**Prior làm nguội kỳ vọng:** `06` đã đo **FiLM ≈ concat, p = 0.308** ở mức CLS ⇒ *cơ chế* fusion
không quan trọng ở mức đó. FiLM là bước **xác nhận**, không phải bước kỳ vọng; cái đáng kỳ vọng
là **phi tuyến**, vì mốc của ta hiện là tuyến tính còn `06` đã là MLP.

**Cổng:** MLP − linear ≥ **+0.003**. **Chi phí:** ~30 ph, embedding đã cache.

---

### B2a · Cross-attention `Q = patch`, `K/V = EHR` — arm chính, **KHÁC** thứ `06b` đã bác — `nb37`

**Vì sao nó không phải thứ đã bị bác.** Chẩn đoán của `06b` chỉ đúng một cơ chế thất bại rất cụ
thể, và bố cục này sửa thẳng vào đó:

| | `06b` H6 (đã bác) | bố cục này (chưa ai đo) |
|---|---|---|
| sắp xếp | `Q = 40 query lớp`, `K/V = [1369 patch ; 1 token EHR]` | `Q = 1369 patch`, `K/V = K token EHR` |
| softmax chạy trên | **1370 key** — EHR giành với patch | **K key EHR** — không có patch nào để giành |
| khối lượng attention EHR tại init | **~0.07%** | **100% của head đó** |
| truyền token → logit (đo được) | **0.0092** (so với 0.25 ở H4) | — |
| bản chất phép toán | thêm một key vào chuỗi | **FiLM biến thiên theo không gian** |

`06` đã đo **FiLM ≈ concat, p = 0.308** — nhưng đó là FiLM **toàn cục**, một cặp `(γ, β)` cho cả
ảnh. Bản không gian cho **mỗi patch một điều biến riêng**; nó là phép toán giàu hơn hẳn và **prior
cũ không bác nó**.

**Kiến trúc.**

```
patch [B, N, 768]                  N = 1369, DU patch token (tinh lai trong vong lap, mau 06b)
ehr   [B, K, 768]                  K in {1, 8, 250}  <- bien quet, xem B2b

patch = patch + Proj( CrossAttn(Q=patch, K=ehr, V=ehr) )     # Proj: ZERO-INIT
                                                             # => khoi dong DUNG BANG CXR-only
logits = ML-Decoder(Q = 40 query lop, K/V = [patch ; ehr])   # tang 2, xem B2c
```

**Chi phí attention:** `1369 × K` — **tuyến tính** theo số patch, không phải `1369² = 1.87M` của
self-attention. Ngay cả `K = 250` cũng chỉ **18%** một lớp self-attention.

**Cổng khoá trước.** So với **cùng đầu đọc, không có nhánh EHR** (`ehr_zero`, tức
`CrossAttn` nhận token rỗng — đối chứng kiểu `H6z` của `06b`, không phải so với arm khác):

```
(Q=patch,K/V=EHR) - (ehr_zero) >= +0.003   -> co hieu ung, di tiep nb39
                               <  +0.003   -> ket hop voi B2b K=1 de quy trach nhiem
```

**Bắt buộc kèm phép tráo nhân quả.** `06b` đo `corr(logit) = 0.999934` khi tráo EHR sang bệnh
nhân khác — đó là bằng chứng mạnh nhất của họ. Arm này **phải** báo cùng con số đó; mAP tăng mà
`corr` vẫn ~1.0 thì mức tăng **không đến từ EHR**.

---

### B2b · Số token EHR là biến quét, `K = 1` là control — `nb37`

```
K = 1     softmax tren 1 phan tu = 1.0 -> moi patch nhan CUNG mot vector
          => suy bien thanh BIAS TOAN CUC.  CONTROL: neu K=8 khong hon K=1
             thi phep chon theo khong gian khong mua duoc gi, va chi can FiLM toan cuc.
K = 8     mot token moi HO (lab24h, lab7d, vitals, como, vi sinh, ngu canh, ...)
          -> chan doan doc duoc: "patch dinh phoi don bao nhieu len token vi sinh"
K = 250   mot token moi COT, kieu FT-Transformer: token_j = emb_j * value_j
          -> cot thieu = KHONG co token (bo luon *_mask): missingness triet de nhat
```

Tham số nặng là 4 phép chiếu `4 × 768² = 2.36M`, **không phụ thuộc K** ⇒ compute không phân định;
cái phân định là **missingness**, **dư thừa** (nhiều cột `29b` tương quan cao), và **tính đọc
được** của chẩn đoán.

**Bộ mã hoá cho nhánh `K = 250`: FT-Transformer, KHÔNG phải TabTransformer.**

`TabTransformer` (Huang et al. 2020) chỉ đưa đặc trưng **phân loại** qua transformer; đặc trưng
**liên tục đi vòng** (chỉ layer-norm rồi nối ở cuối). EHR của `29b` **chủ yếu là liên tục** — giá
trị lab 24h/7d, vitals, đếm, recency theo giờ, tuổi ⇒ dùng bản gốc là đẩy gần hết dữ liệu ra ngoài
attention.

`FT-Transformer` (Gorishniy et al. 2021) token hoá **cả hai loại**:

```
so     :  T_j = b_j + x_j * W_j        (element-wise, moi cot mot vector rieng)
phan loai: T_j = b_j + Embedding[x_j]
thieu   :  KHONG sinh token            -> bo luon cac cot *_mask
           -> self-attention tren so token thay doi tung mau
[CLS] + T_1..T_250 -> N lop transformer -> token dau ra lam K/V cho B2a
```

Cấu hình: `d_ehr = 192`, `3 lớp`, `4 head`, rồi chiếu `192 → 768` cho cross-attention. Chi phí
`250² × 192 × 3 ≈ 36M MAC/mẫu` — vài giây/epoch, không phải ràng buộc.

**Biện minh là KIẾN TRÚC, không phải độ chính xác.** Gorishniy et al. tự kết luận GBDT vẫn thắng
deep tabular trên nhiều bộ dữ liệu. Lý do dùng FT-Transformer ở đây là nó cho **EHR ở dạng token**
để patch cross-attend được — XGBoost của `30` không làm được điều đó. **Đừng** đem nó so với
XGBoost mà kỳ vọng thắng.

**Nhưng PHẢI có cổng chặn với XGBoost.** `06b` đã lập kỷ luật này: encoder của `H2` được xác nhận
bằng shuffle (`+0.0134`) để loại phản biện *"H6 thất bại chỉ vì encoder của nó kém"*.

```
FT-Transformer EHR-only mAP  vs  XGBoost cua nb30 = 0.1199
  trong nhieu (+-0.005)  -> encoder dat, moi so fusion phia sau doc duoc
  kem han                -> ENCODER la nut that; moi ket luan fusion bi confound
```

Chạy cổng này **trước** mọi arm fusion, và nó cũng chính là bước **pre-train encoder riêng** mà
MedFuse đo được là đáng `+0.014` (B2e) — một công đôi việc.

---

### B2c · Định tuyến theo bệnh — hai tầng, và đừng hard-code — `nb37`

ML-Decoder cho định tuyến theo bệnh **miễn phí**: mỗi query lớp có phân bố attention riêng. Nhưng
nếu EHR **chỉ** là K/V ở tầng patch thì query lớp **không hề thấy** token EHR — định tuyến xảy ra
ở tầng patch, mà patch không biết nhãn nào đang hỏi. Cần **hai tầng**:

```
tang 1 KHONG GIAN : Q = patch,        K/V = EHR       "boi canh khien VUNG NAO dang ngo"
tang 2 NGU NGHIA  : Q = 40 query lop, K/V = [patch ; token EHR]
                                                      "NHAN NAY co quan tam EHR khong"
```

Tầng 2 gần như miễn phí (K đi từ 1369 lên 1369+K) và nó **khôi phục** rủi ro nhấn chìm của `06b`
— nhưng lần này **đo được**, và EHR đã được bơm vào patch ở tầng 1 rồi nên không phụ thuộc hoàn
toàn vào tầng 2.

**Đừng** nhét danh sách 11 nhãn nhạy EHR của `33` C7 vào kiến trúc — nó đo trên **val**,
hard-code là rò rỉ val vào thiết kế. Dùng làm **kiểm chứng**: so hạng giữa khối lượng attention
EHR mà mô hình **tự học** với lift đã đo. Khớp ⇒ cơ chế chạy đúng thiết kế; không khớp ⇒ mô hình
không tìm được đường định tuyến, và đó là chẩn đoán chứ không phải một con số mAP.

---

### B2d · Deep EHR conditioning — cơ chế bổ sung, chưa ai đóng — `nb39`

Chạy **sau** B2a. Nếu B2a đã dương thì đây trả lời câu tiếp theo: EHR còn mua thêm được gì khi
được phép can thiệp **trong lúc** representation hình thành, chứ không chỉ sau khi nó xong.

**Kiến trúc.** Adapter FiLM **trong** block, phần còn lại đông cứng:

```
Z_l -> Z_l * (1 + gamma(e)) + beta(e) -> block l+1 -> ... -> ML-Decoder
gamma, beta : MLP(e) -> 768,  ZERO-INIT  =>  khoi dong DUNG BANG CXR-only
```

`zero-init` bắt buộc (kỷ luật `AMENDMENTS` ③): không có nó thì `gain < 0` không phân biệt được
*"EHR vô dụng"* với *"adapter làm nhiễu"*.

**Đặt ở đâu.** `11` đã đo `I_location ≈ 0` (block 5,7 vs 10,11) ⇒ vị trí không quan trọng; dùng
**block 8–9** theo `NOTE_07` để còn 3–4 block xử lý lại. **Tránh block 11–12**: theo `NOTE_07`,
adapter ở block cuối **về chức năng là late fusion đội lốt** — thứ đã bị bác.

**Giữ nguyên `K` và hai tầng của B2b/B2c** — adapter trong block **cộng thêm** vào chúng, không
thay thế.

**BA arm, không phải hai — đây là chỗ dễ tự lừa nhất.** `11` đã lập kỷ luật này: bốn arm
`M_U / M_E / L_U / L_E` với adapter **khớp tham số** (`234,368 params` mỗi arm), để `G = *_E − *_U`
đo **riêng nội dung EHR**, tách khỏi phần chỉ do thêm tham số:

```
D1     khong adapter                                  (= B2a tot nhat)
D2_U   CO adapter, dau vao la VECTOR HANG SO hoc duoc (khong EHR)   <- khop tham so
D2_E   CO adapter, dau vao la EHR

capacity = D2_U - D1     <- phan CHI do them tham so, khong lien quan EHR
G_deep   = D2_E - D2_U   <- phan THAT SU do noi dung EHR   <-- day moi la so can doc
```

Nếu chỉ so `D2_E − D1` thì một adapter 234k tham số **tự nó** có thể nhấc mAP mà chẳng cần EHR, và
ta sẽ ghi công nhầm. `06b` đã dựng đúng cặp này ở mức token (`H4z`/`H4`, `H6z`/`H6`) và nó **chính
là thứ cứu kết luận khỏi sai** — `H6z 0.4330` vs `H6 0.4339` cho thấy token EHR không thêm gì, trong
khi so với `H5 0.4336` thì trông như có.

**Cổng chặn phải đặt trên `G_deep`, không phải `D2_E − D1`.**

**Đo cơ chế, đừng đoán** (kỷ luật `AMENDMENTS` ④):

```python
ratio_l  = ‖ΔZ_l‖ / ‖Z_l‖                  # bien do adapter tung block
attn_ehr = attn[:, :, -K:].mean()           # so voi muc deu 1/(N+K)
d_logit  = ‖logit(e) - logit(e_shuffled)‖   # phep trao nhan qua — 06b dat 0.00094
```

**Cổng khoá trước — đặt trên `G_deep = D2_E − D2_U`:**

```
G_deep <  +0.002          -> DUNG huong EHR fusion cho 40 nhan CXR-LT
          +0.002..+0.004  -> vung xam: can seed thu 3 + CI
       >  +0.004          -> mo pipeline confirmatory
seed-CI chua 0            -> DUNG bat ke do lon
```

Báo cáo `capacity = D2_U − D1` **cạnh** `G_deep`. Nếu `capacity` lớn mà `G_deep ≈ 0` thì kết luận
là *"thêm adapter có ích, nhưng EHR không"* — một phát biểu hoàn toàn khác với *"deep conditioning
có tác dụng"*, và chỉ có cặp arm này mới phân biệt được.

**Thiết kế adapter.** `11` dùng bottleneck khớp tham số; giữ nguyên hình dạng đó để so được với số
cũ (`G_mid +0.0036 / G_late +0.0032`). `D2_U` dùng **vector hằng số học được** thay cho `e` — đúng
kiểu `H4z`/`H6z` của `06b`, không phải vector 0 (vector 0 qua LayerNorm vẫn ra `beta`, nên nó
*gần* identity chứ không bằng).

**Chi phí.** Adapter cần gradient qua backbone từ block 8 trở lên ⇒ tính lại trong vòng lặp.
`BATCH 96` (adapter-only, `no_grad` block 0–7) ≈ 16 GB VRAM. ~25 ph/epoch × 8 epoch × 3 seed,
**mọi arm dùng chung một forward** ⇒ **~10 h**. Khoản đắt nhất kế hoạch — chỉ chạy sau khi A1–A4
xong và B1 chưa đóng được hướng.

---

### B2e · Hai thứ MedFuse đo được là có tác dụng, và ta **chưa làm cái nào** — `nb37`

Rẻ hơn mọi thay đổi kiến trúc, và cộng lại `+0.029` AUPRC trong tay MedFuse:

**Pre-train encoder riêng từng modality trước khi fusion** (`+0.014`). `f_ehr` = FT-Transformer của
B2b, pre-train bằng `L_ehr = BCE(y, g_ehr(f_ehr(x)))` trên **toàn bộ** train (kể cả study không
dùng được ảnh — đây là chỗ 11,839 study không có ảnh dùng được vẫn đóng góp), rồi bỏ `g_ehr`, giữ
`f_ehr`. Bước này **đồng thời là cổng chặn XGBoost** ở B2b: pre-train xong thì có ngay mAP EHR-only
để so với `0.1199`.

**Chỉnh tỉ lệ mẫu uni-modal khi fine-tune** (`+0.015`). MedFuse tìm được **10%** cho tử vong và
**20%** cho phenotyping là tối ưu — tức trộn thêm mẫu **chỉ-EHR** vào lúc fine-tune fusion. Với ta
tương đương: trộn thêm study **không có EHR** (11,839 study không bản ghi + ngoại trú) để nhánh
ảnh không bị nhánh EHR kéo lệch. Quét `p_uni ∈ {0, 10%, 20%, 40%}`.

Cả hai chạy trong cùng `nb37`, dùng chung forward backbone ⇒ gần như miễn phí.

---

### B3 · Nhiệm vụ THỨ HAI: phenotyping / tử vong — nơi fusion thật sự có đất — `nb40` *(tuỳ chọn)*

**Không** thay CXR-LT; **thêm** một target chạy trên **cùng hạ tầng**. Lý do nằm ở §1.5: nhãn
CXR-LT sinh từ chính report của phim nên **ảnh gần như đủ theo định nghĩa** — 29/40 nhãn trơ với
EHR không phải vì fusion yếu mà vì **không còn gì để thêm**. Nhãn CCS/tử vong **không** sinh từ
modality nào trong hai cái, nên cả hai đều còn đất, và máy móc fusion ở B2a–B2e mới có chỗ thể
hiện.

Đã có sẵn mọi thứ để dựng: cùng MIMIC-IV + MIMIC-CXR · nhãn = **25 CCS phenotype từ ICD** (ánh xạ
ICD-10 → ICD-9 → CCS) và **tử vong nội viện 48h** · **mốc công bố** để đối chiếu (`0.481` / `0.594`
AUPRC) · code công khai `github.com/nyuad-cai/MedFuse`.

Vẫn đúng đề bài **CXR + structured EHR** — chỉ đổi **đích**, không đổi bài toán. Và nó biến kết quả
âm ở CXR-LT từ "hướng thất bại" thành **"đã định vị được ranh giới: EHR không giúp nhãn sinh từ
report, giúp rõ nhãn lâm sàng"** — phát biểu mạnh hơn hẳn.

**Chi phí:** ~1 notebook dựng nhãn (BigQuery, giống `29`), rồi tái dùng `f_cxr` đã có.

---

### B4 · Nếu B2 null: phát biểu kết quả âm cho ĐÚNG phạm vi

Kết quả âm là **đóng góp thật**, nhưng chỉ khi phát biểu đúng.
`REPORT_frozen_token_fusion_revised.md` §5 và §7 là mẫu có sẵn trong repo.

Phải nêu rõ nó **không** loại trừ: (a) **target ngoài-ảnh** — nhãn CXR-LT do NLP trích từ chính
report của phim nên ảnh gần như đủ theo định nghĩa; kết luận **không nói gì** về tiên lượng, tử
vong, chuyển ICU, tái nhập viện; (b) cohort / backbone khác; (c) tương tác phi tuyến ngoài lớp
hàm đã thử.

---

## 4 · Kỷ luật xuyên suốt

**Cổng khoá trước.** Mọi thí nghiệm nêu ngưỡng **trước khi chạy** — repo đã có truyền thống này
(`11` khoá `G ≥ +0.002 ∧ I ≥ +0.002`; `20` khoá `+0.002`; `25` khoá `+0.01`).

**Ngưỡng nhiễu lấy từ BIÊN ĐỘ SEED của chính chỉ số đó — không dùng một con số chung.**
*(sửa sau `35`; bản cũ ghi "sàn nhiễu 0.005" không kèm điều kiện và đã gạt nhầm một hiệu ứng thật)*

`0.005` suy ra từ khoảng cách hạng 2–4 của giải, mà đó là **macro mAP mức ảnh** (~0.20). Tail nằm ở
thang ~0.06 nên `0.005` ở đó là **12% tương đối** — chặt tới mức loại bỏ mọi thứ ta đang cố cải thiện.
Biên độ đo được trên 3 seed, giao thức epoch-cuối:

```
macro (anh)   +-0.0007      tail   +-0.0011 .. 0.0023      study   +-0.0007
```

Chỉ đọc là thắng khi chênh lệch lớn hơn **biên độ seed của chính chỉ số đó**, và tốt nhất là khi hai
khoảng seed **rời nhau**.

**KHÔNG early-stop trong một lần chạy. Lấy mô hình ở EPOCH CUỐI.**
*(sửa sau `35`; bản cũ ghi "early-stop theo tail mAP" và đó là một lỗi tốn kém)*

`CosineAnnealingLR` trải đúng trên ngân sách `E`, nên LR **tại epoch được chọn** phụ thuộc `E`. Chọn
epoch trên lát cắt hay rơi vào 65% ngân sách (LR `9.0e-4`) thay vì gần cuối (LR `5.4e-5`) ⇒ nhận một
nghiệm **chưa được ủ**. Đo được ở `35`:

```
                       mAP anh   bien do seed (3 seed)
chon epoch theo tail    0.2067       0.0058 .. 0.0072
epoch cuoi              0.2093       0.0003 .. 0.0009
```

Early-stop **mất `~0.003` mAP VÀ thổi phương sai lên 8 lần**. Việc ủ LR về 0 là một phần của cái làm
nên mô hình tốt, không phải phụ kiện của lịch.

⇒ `E` là **siêu tham số duy nhất**, quét trên lát cắt bằng **macro**, lấy `E` **nhỏ nhất còn nằm trong
nhiễu của đỉnh**. Áp cho A1, A4 (fine-tune), A5, và mọi arm của `37`/`39`.

**Không dùng checkpoint đã huấn luyện CÓ GIÁM SÁT trên MIMIC-CXR.** Split test của CXR-LT là tập
con của MIMIC-CXR, nên mọi mô hình học có nhãn trên MIMIC-CXR (`torchxrayvision` `-all` /
`-mimic_ch` / `-mimic_nb`, CheXNet…) **đã thấy ảnh test của ta kèm nhãn**. RAD-DINO được miễn vì nó
tự giám sát (DINOv2, không nhãn) và là backbone mặc định của giải. Xem A5.2a.

**Bảo vệ test.** Val cho mọi lựa chọn; test chạm **một lần ở cuối**; mọi lần chạm ghi vào
`task2/_test_touches.jsonl`. `REPORT_frozen_token_fusion_revised.md` §7.1 là bài học: test
cohort cũ **đã bị dùng thích nghi** nên mọi thứ sau đó chỉ còn là exploratory.

**Chọn per-label phải có lát cắt riêng.** Chọn trên val rồi báo cáo trên val là overfit tiêu chí
chọn. Cắt 85/15 từ **train**, chọn trên 15%, báo cáo trên val nguyên vẹn.

**RESUME khoá theo TÊN, không theo NỘI DUNG.** Đã dính hai lần: `32` không biết embedding
`crop518` đã đổi (`cls` sai lệch **−0.0272**); `31` ghi `_done` trước `flush()` để lại **2,330
hàng 0**. Đổi embedding của arm nào thì xoá **cả** dòng CSV **lẫn** file `.npy` của arm đó.

**Đọc shard trên Drive.** Đọc nguyên tar vào RAM rồi mở `tarfile` trên `BytesIO`; 8 luồng;
checkpoint từng tar. Lặp `for m in tar` thẳng trên mount **treo cứng** (đã dính ở `33`).

---

## 5 · Thứ tự thực thi

| # | notebook | nội dung | phần cứng | thời gian | cổng |
|---|---|---|---|---|---|
| 1 | ~~**`35`**~~ **XONG** | A1 loss (→`asl40`@60) · B1 đầu đọc (null) · A5.1 mốc corr (0.9060) · trần B2c (+0.0019). A2 chuyển sang `37`. Giao thức: **epoch-cuối**, ngưỡng theo biên độ seed | CPU / GPU nhẹ | ~1 h | ASL ≥ +0.003 · MLP ≥ +0.003 |
| 2 | **`37`** | A3 ML-Decoder trên **đủ 1369 token** **+** B2a `Q=patch, K/V=EHR` **+** B2b quét `K ∈ {1,8,250}` **+** B2c hai tầng **+** B2e — tất cả trên **một forward dùng chung**, mẫu `06b` | GPU | **~4 h** | ảnh ≥ +0.010 · EHR ≥ +0.003 so `ehr_zero` |
| 3 | **`38`** | partial unfreeze / LP-FT — **chạy đủ 12 epoch một lần**, lưu checkpoint mỗi epoch, ghi đường cong **theo tier** | GPU | **~6 h** (một lần duy nhất) | FT − LP ≥ +0.03, tách theo tier |
| 4 | **`39`** | B2d deep conditioning, adapter block 8–9, **cộng thêm** lên `37` | GPU | **~10 h** | `G_deep = D2_E − D2_U ≥ +0.002` **và** seed-CI loại 0 |
| 5a | **`41a`** | A5.2e — **một lượt quét, 3 backbone**: `convnextv2_nano` · `densenet121` · `convnextv2_tiny` → probe tuyến tính → **đo corr thật với RAD-DINO** | GPU | **~40 ph** | probe ≥ 0.15 · corr **thấp hơn mốc same-backbone của A5.1** |
| 5b | **`41b`** | A5.2f — fine-tune con thắng → ensemble với RAD-DINO | GPU | ~0.5–1.5 h | CNN ≥ 0.85× RAD-DINO · ensemble ≥ +0.005 |
| 6 | **`40`** *(tuỳ chọn)* | B3 target thứ hai: 25 CCS phenotype + tử vong 48h, đối chiếu MedFuse | BigQuery + GPU | ~1 nb + tái dùng | so với `0.481` / `0.594` AUPRC |

**Đã bỏ `nb36`** (trích token vùng `6×6` rồi cache). `06b` chạy ML-Decoder trên **đủ 1369 token**
bằng cách tính lại backbone trong vòng lặp — không cần cache, nên không cần gộp lưới, nên không mất
chi tiết mảnh. Xem A3.

**A5.1 nằm trong `nb35`** — trung bình 9 file `p_*.npy` đã có, cộng tương quan từng nhãn. Nó **không**
phải phép thử cho CNN (cả 9 đều là RAD-DINO) mà là **mốc đối chiếu**: cận dưới của đa dạng. Có nó
thì con số corr của CNN ở `41a` mới đọc được — `0.70` là cao hay thấp phụ thuộc mốc này.

Dùng **trung bình HẠNG** song song với trung bình xác suất: AP là chỉ số **xếp hạng**, và ba arm
huấn luyện riêng nên rất có thể lệch calibration.

`37` là notebook trung tâm: nó chạy **cả** đòn bẩy ảnh (ML-Decoder 1369 token) **lẫn** arm EHR chính
(`Q=patch, K/V=EHR`) **lẫn** B2e (pre-train `f_ehr` riêng · quét `p_uni ∈ {0,10,20,40}%`) trên **một
forward backbone dùng chung** — theo lập luận ① của `AMENDMENTS`, tham số các arm rời nhau nên đây là
**tương đương toán học tuyệt đối** với chạy riêng từng arm, không phải xấp xỉ. Quét cả
`K ∈ {1, 8, 250}` × `{có, không}` nhánh EHR gần như miễn phí so với chạy một arm.

Bước 1 rẻ và định hướng: **A2 cho biết trên 40 nhãn thì EHR còn thông tin độc lập hay chỉ vá
bottleneck của pooling** — và nó đọc được trước khi tiêu 4 giờ cho `37`.

Bước 2 và 3 giao nhau — nếu `38` chạy trước thì `37` nên đọc backbone **đã fine-tune**, đúng như
`06b` đã làm (*"frozen" = đã supervised fine-tune ở `04b` rồi mới đóng băng*).

---

## 6 · Verification

**Mỗi notebook** giữ khuôn của `31`–`34`: cổng `assert` nêu trước · RESUME · một cell **chấm lại
độc lập** (AP tự viết + gộp study bằng `groupby`) đối chiếu cài đặt chính — `33` C4 khớp `32` tới
**5.06e-09**, đó là mức nên giữ.

**Dry-run trước khi deploy.** Chạy thật trên dữ liệu giả **có sự thật cài sẵn**, với **cả nhánh
dương lẫn nhánh âm** — `33` dry-run bơm suy giảm-theo-đệm (`z = −7.04`) **và** suy giảm-đều
(`z = +0.48`) để chứng minh cổng không báo động giả.

**Kiểm chứng chéo bắt buộc cho mọi arm có EHR (`37` và `39`):** ba phép đo cơ chế (`ratio_l`,
`attn_ehr`, phép tráo nhân quả) phải **cùng chiều** với mức tăng mAP. Nếu mAP tăng mà
`corr(logit)` khi tráo EHR sang bệnh nhân khác vẫn ~1.0 (mốc `06b`: **0.999934**) thì mức tăng
**không đến từ EHR** — đúng cái bẫy `06b` đã bắt được, và là lý do arm `ehr_zero` phải luôn chạy
song song.

**Ghi kết quả vào markdown trong repo**, không chỉ trong chat: `REVIEW_CXRLT_STAGE0.md` cho trạng
thái sống, và một `REPORT_*.md` riêng cho mỗi kết luận có cổng — theo mẫu
`REPORT_frozen_token_fusion_revised.md` (có §"những gì kết quả KHÔNG loại trừ" và §caveat).

---

## 7 · Sản phẩm đầu tiên khi được duyệt

Ghi kế hoạch này thành **`PLAN_STAGE4_IMAGE_THEN_EHR.md`** trong thư mục dự án (theo quy ước
`REVIEW_*.md` / `PLAN_*.md` đã dùng), rồi bắt đầu từ `nb35`.
