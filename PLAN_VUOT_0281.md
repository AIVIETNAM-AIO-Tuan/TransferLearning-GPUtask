# KẾ HOẠCH VƯỢT `0.281` — từ `0.2625` trên test

*Lập 2026-09-06. Nối tiếp `CHECKLIST_STAGE5_OPEN.md`.*

---

## 0 · Điểm xuất phát và khoảng cách

```
TEST hien tai   0.2625   (duong ong 0.2583 + cnn, w tier 0.40/0.40/0.00)
  head          0.4827
  medium        0.1977
  tail          0.0966
vo dich         0.281
CAN            +0.0185
```

Quy đổi: cần `+0.057` ở head, **hoặc** `+0.046` ở medium, **hoặc** `+0.067` ở tail — **hoặc**, thực
tế nhất, cộng dồn nhiều nguồn.

> **Đính chính (Tuấn hỏi, 2026-09-08).** Bản đầu tôi ghi tail `+0.067` là *"không khả thi"*. **Sai,
> và tôi rút lại.** Nó là `+69%` tương đối — lớn hơn mọi can thiệp đơn lẻ ta từng đo, nhưng
> **không** vượt tầm khi cộng dồn, và **cả một mảng phương pháp long-tail ta chưa động tới**
> (xem §4.4). Riêng PadChest, nếu 4 nhãn tail trùng tên gấp đôi AP thì tail đã đi
> `0.0966 → ~0.132` (`+37%`).

### 0.1 · ĐÃ XÁC MINH bằng paper chính thức (arXiv:2506.07984, Lin et al. 2025)

`paper/2506.07984v1.pdf` — *"CXR-LT 2024: A MICCAI challenge on long-tailed, multi-label, and
zero-shot disease classification from chest X-ray"*.

**Thiết lập khớp tuyệt đối với ta** (Table 1):

```
Train 258,871 | Development 39,293 | Test 78,946 | 40 nhan
chia theo BENH NHAN, ti le 70/10/20, RIENG cho CXR-LT 2024
Metric: mAP = "macro-averaged AP across classes"; hoa thi xet mAUROC
```

⇒ **`0.2625` của ta so trực tiếp được với bảng xếp hạng.**

**Bảng xếp hạng Task 1** (Table 4 + Table 5):

```
Hang Doi  mAP     Head(>10%)  Medium(1-10%)  Tail(<1%)
  1   A   0.281     0.567        0.263        0.136
  2   B   0.279     0.569        0.260        0.133
  3   C   0.277     0.570        0.253        0.136
  4   D   0.277     0.568        0.264        0.125
  5       0.273
  6       0.269
 TA       0.2625                                       -> khoang hang 7-8 / 17 doi
```

### 0.2 · ⚠️ ĐỊNH NGHĨA TIER CỦA TA KHÁC CỦA GIẢI — mọi so sánh tier cũ đều lệch

Paper §3.3: *"we split the 40 target classes into 'head' (>10%), 'medium' (1%–10%), and 'tail'
(<1%) ... based on their prevalence in the **training set**, consisting of **9, 14, and 16**
categories."*

```
GIAI   head  9 nhan (>10%)  | medium 14 (1-10%) | tail 16 (<1%)
TA     head 13              | medium 16         | tail 11
```

**Hệ quả:** `tail 0.0966` của ta **không** so được với `0.136` của đội A — tail của họ rộng hơn
(16 nhãn, gồm cả nhãn ta xếp vào medium). Mọi phát biểu tier trong `JOURNAL`,
`CHECKLIST_STAGE5_OPEN.md` và bản đầu của file này đều dùng thang riêng của ta.

```
[ ] tinh lai 3 tier theo prev(TRAIN): >10% / 1-10% / <1%
[ ] xac nhan ra dung 9 / 14 / 16 nhan
[ ] doc lai head/medium/tail cua ta tren thang do  ->  moi biet khoang cach THAT o tung tier
```

**Làm việc này trước mọi thứ khác** — nó quyết định đổ công vào tier nào.

---

## 0.3 · Các đội đầu bảng làm gì — trích thẳng từ paper (Table 3 + §4.1)

```
Doi A  0.281  ConvNeXt-S/B/T/V2-B, ENSEMBLE 12 mo hinh, 224+384, ImageNet, VL
              DIFFUSION sinh 100 anh MOI cho MOI lop hiem (prompt + U-Net khu nhieu + VAE)
              -> TAIL TOT NHAT (0.136) va HANG 1 tong the
Doi B  0.279  EfficientNetV2-L + PubMedBERT (UMLS), 512, ENS + LRW + VL
              weighted asymmetric loss · TTA (resize, crop, LAT)
              ngoai: ChestX-ray14, CheXpert, VinDr-CXR, BRAX
Doi C  0.277  ConvNeXt-S + EfficientNetV2-S, **1024**, ENS + LRW + VL
              CLEFT/MaMA, da goc nhin + da ti le, Symmetric Local Cross-Attention
              -> HEAD tot nhat (0.570) va HANG 1 Task 2 (0.526)
Doi D  0.277  ConvNeXt-S, **1024**, LRW + VL
              ML-Decoder + Noisy Student self-training + CheXFusion (transformer da goc nhin)
              -> MEDIUM tot nhat (0.264)
Doi E         ViT-L DINOv2 pretrain tren **710,000 anh CXR**
              (MIMIC + CheXpert + PadChest + NIH + BRAX), dau doc ML-Decoder,
              ensemble da goc nhin / da do phan giai
```

**Đội E gần như là kiến trúc của ta** (DINOv2 + ML-Decoder) nhưng `ViT-L` và pretrain trên 710k ảnh
CXR; ta dùng `ViT-B` RAD-DINO.

### Chủ đề chung của 9 đội đầu — và ta đang thiếu gì

| Chủ đề | Bao nhiêu đội | Ta có? |
|---|---|---|
| **Loss re-weighting** (weighted asymmetric loss) | **5/9** (B, C, D, F, G) | ❌ ta dùng `ASL` **trần**, `gn=4/gp=0`, **không trọng số lớp** |
| **Ensemble** | **8/9** | ✅ nhưng chỉ 2 kiến trúc (A dùng 12 mô hình, I dùng 21) |
| **Vision-language** (ảnh + report) | **8/9** | ❌ hoàn toàn không dùng text — ta dùng EHR thay thế |
| **Pretrain quy mô lớn trên CXR ngoài** | 8/9 | ⚠️ chỉ RAD-DINO + PadChest cho CNN |
| **Đa góc nhìn** (frontal + lateral) | C, D, F, H | ⚠️ chỉ trọng số `0.7/0.3`, không có module gộp |
| **Độ phân giải 1024** | C, D | ❌ ta ở 518 |
| **Sinh ảnh tổng hợp cho lớp hiếm** | A | ❌ |

Paper kết luận thẳng về tail (§3.3):

> *"Team A used pretrained large generative models to generate new images for these tail cases,
> while Teams B and C applied loss reweighting, indicating that **both approaches can improve
> performance in the 'tail' group**."*

⇒ **Bằng chứng trực tiếp bác bỏ câu "tail không nhấc được" của tôi.** Hai cơ chế đã được chứng minh
trên đúng bộ dữ liệu này, và ta chưa dùng cái nào.

---

## 1 · Ngân sách kỳ vọng — trả lời thẳng "có vượt được không"

> **CẬP NHẬT 2026-09-08 sau `nb67`.** Bậc 0 **ĐÃ ĐO: `+0.0000`** — mọi phép quét 0-GPU vô ích, mặt
> phẳng `w` phẳng, cấu hình khoá từ `61` đã tối ưu cục bộ. Ước `+0.004` của tôi **sai**. Đổi lại,
> `nb67` chỉ ra khoảng cách `medium` (71% tổng) tập trung vào **7 nhãn**, phần lớn **nhãn sạch**
> (`prec 0.958–1.000`) và là **cấu trúc mảnh dưới `9.46mm`/patch** ⇒ **độ phân giải lên số 1**
> (§5.5 mới). Xem JOURNAL 2026-09-08 (6).

| Bậc | Việc | Chi phí | Kỳ vọng | Độ chắc |
|---|---|---|---|---|
| ~~**0**~~ | ~~Quét trên val~~ | 0 GPU | **`+0.0000` ĐÃ ĐO — ĐÓNG** | — |
| **1** | TTA không lật · che top-k patch · `nb66` | ~3h | **+0.003** | vừa |
| **2** | PadChest cho 4 nhãn tail · EHR mở rộng | ~12h | **+0.005** | vừa |
| **2b** | **Weighted asymmetric loss** (§4.4, 5/9 đội đầu bảng dùng) | ~13h | **+0.004** | cao |
| **3** | **K-fold, fine-tune backbone theo từng fold** | ~35–58h | **+0.010** | vừa |
| **3b** | **Sinh ảnh diffusion cho 16 nhãn tail** (§4.5, nước đi của đội A) | ~30–60h | **+0.008** | thấp–vừa |
| | **cộng thô** | | **+0.034** | |
| | trừ chồng lấn (~30%) | | **+0.024** | |
| | **dự kiến tới** | | **≈ 0.286** | |

**Với `2b` và `3b`, kế hoạch chuyển từ "ngang vô địch" sang "vượt được, nếu cả hai đều ăn".**
Nhưng `3b` là mục **kém chắc chắn nhất** trong bảng — nó phụ thuộc việc có tồn tại bộ sinh không
dính MIMIC-CXR hay không (§4.5, cạm bẫy 1). Nếu phải tự huấn luyện bộ sinh từ đầu thì chi phí gấp
đôi và kỳ vọng không đổi.

**Kết luận thẳng: kế hoạch này đưa ta tới NGANG vô địch, chưa chắc VƯỢT.**

Muốn vượt rõ ràng phải thêm bậc 4. Tôi ghi vậy thay vì tô hồng, vì nếu chỉ chạy bậc 0–3 rồi dừng
thì khả năng cao dừng ở `0.279` và mất công.

---

## 2 · Bậc 0 — miễn phí, làm ngay (0 GPU, ~1h)

Mở ra nhờ nhận định đúng của Tuấn: **val giờ là tập chọn hợp lệ** (độc lập với test; thiên lệch của
nó chỉ hỏng vai trò *thước đo*, không hỏng vai trò *tập chọn*).

```
0.1  trung binh HANG thay vi xac suat        AP la chi so xep hang; thanh vien lech calibration
0.2  quet lai w theo tier (ke ca w_tail)     0.40/0.40/0 khoa tu hoi val con la thuoc do
0.3  trong so theo EPOCH                     p_ep1 (0.245) dang can NGANG p_ep8 (0.258)
0.4  cnn_ms lam thanh vien thu 3, w theo tier
```

**Ba cơ chế chống overfit val, gắn vào cùng cell:**

```
a) chon ROBUST: bootstrap val 300 lan, lay gia tri thang NHIEU LAN NHAT (khong phai argmax mot lan)
b) co ngot ve prior:  w~ = (n*w_val + k*0.40) / (n + k)
c) chiet khau theo tier — dung SO DA DO:   head x1.00 | medium x0.97 | tail x0.81
   (tu chuyen giao val->test: +0.0001 / -0.0055 / -0.0227)
```

---

## 3 · Bậc 1 — một lượt suy luận, không huấn luyện (~3h)

```
1.1  nb66   cascade / do phan giai, 3 nhanh A/B/C
            (A+B)-(A+C) < 0.0007  ->  DONG han huong do phan giai, khoi ban nua
1.2  TTA khong lat: xoay +-5, crop 0.9-1.0        +0.002 ~ +0.005
1.3  chay lai nb65 tren trong so 64/r2a           moi so hien tai la SO BO (checkpoint ep1)
1.4  che top-k% patch luc suy luan, k chon tren val   +0.0008
```

`1.1` là phép **quyết định**: nó đóng hoặc mở cả hướng độ phân giải với giá vài chục phút.

---

## 4 · Bậc 2 — nhắm thẳng vào TAIL (~12h)

Tail = 11/40 nhãn (27.5% macro) và hiện nhận **ZERO** từ ensemble (`w_tail = 0`).

### 4.1 · PadChest làm nguồn nhãn bổ sung ⭐ đòn bẩy tail lớn nhất

Ghi từ `2026-08-14`, **chưa bao giờ dùng**. Ta mới dùng PadChest làm *pretrain*
(`convnextv2_padchest.pt`), chưa dùng làm **nhãn**.

```
Azygos Lobe          176 anh    <- 1 trong 3 nhan cnn_ms pha
Hydropneumothorax    549 anh    <- 1 trong 3 nhan cnn_ms pha
Pneumoperitoneum     454 anh
Kyphosis             669 anh
```

Nhãn hiếm nhất của ta có `~150` ca dương train ⇒ thêm `176–669` ca là **gấp 2–6 lần**.
Sạch với test: PadChest = Hospital San Juan (Alicante), MIMIC-CXR = Beth Israel (Boston), không
chung bệnh nhân; `Removal.csv` của ban tổ chức đã loại dev/test.

**Cách làm:** co-training — mỗi batch trộn ảnh MIMIC (40 nhãn) và ảnh PadChest (chỉ 4 nhãn kia,
mask phần còn lại khỏi loss). `~8h`.

**Cổng:** AP của 4 nhãn đó tăng `≥ 20%` tương đối trên val, và **3 nhãn tail còn lại không tụt**.

### 4.2 · EHR mở rộng — thủ thuật / thở máy / kiểu nhập viện

Hai nhãn `Hydropneumothorax` (bội số EHR `4.2×`) và `Pneumomediastinum` (`4.8×`) có bội số cao nhất
nhóm tail, nhưng EHR của ta thiếu đúng loại đặc trưng: `proc_chest_tube_*`, `proc_thoracentesis_*`,
`proc_intubation_*`, `vent_mech`, `vent_peep_max_24h`, `proc_noi_soi_7d`, `adm_type`,
`dx_copd_ever`. ~15 cột, cùng dạng `lab_*` ⇒ cắm thẳng vào `ColTokBag`, **không đổi kiến trúc**.

```
bac 1  BigQuery dung dac trung     ~2-3h   0 GPU
bac 2  probe XGB EHR-only theo nhan ~30ph  CONG: boi so tang >= 1.5x
bac 3  neu qua cong: huan luyen lai         ~13h
```

**Bắt buộc:** phép tráo ảnh sang bệnh nhân khác (mẫu `06b`). `dẫn lưu → hydropneumothorax` là
**đường tắt**; nếu AP không đổi khi tráo ảnh thì mô hình đang đọc EHR chứ không đọc phim.

### 4.3 · `cnn_ms` ở tail, loại 3 nhãn theo CƠ CHẾ

`cnn_ms` (đa tỉ lệ 10mm) nâng nhiều nhãn tail, chỉ 3 nhãn phá. Hai chẩn đoán độc lập trỏ cùng chỗ:

```
                     boi so EHR   giu@k=5%   ban chat
Azygos Lobe            1.4x        1.025     BIEN THE GIAI PHAU — khong phai benh
Hydropneumothorax      4.2x        0.880     co hoc / do can thiep
Pneumomediastinum      4.8x        0.845     vet sang MANH doc bo trung that — can CA DUONG
```

Loại 3 nhãn theo **cơ chế**, không theo số đo (per-label ở tail không chuyển giao: `|Δr| = 0.416`,
val→test `−19%`). **Khoá danh sách 3 nhãn TRƯỚC khi đọc test.**

### 4.4 · Long-tail: mảng phương pháp ta CHƯA động tới

Mọi hướng ta đã đóng (deep conditioning, mở băng, cRT, đa tỉ lệ, độ phân giải) đều tấn công
**kiến trúc**. Nhóm dưới tấn công **loss, lấy mẫu, và dữ liệu** — **không cái nào bị kết quả âm của
ta bác.**

#### Bộ lọc phải áp trước: cái gì là no-op với macro-AP

macro-AP tính **từng nhãn** rồi mới trung bình ⇒ mọi phép biến đổi **đơn điệu theo từng nhãn**
**không đổi AP của nhãn đó**:

```
KHONG DUNG DUOC (no-op)          logit adjustment tau*log(prior) · balanced softmax hau ky
                                 chinh nguong · hieu chinh prior · moi phep rescale don dieu
DUNG DUOC (doi dac trung/hang)   loss trong luc huan luyen · lay mau · margin · du lieu · chuyen gia
```

Phần lớn văn liệu long-tail nhắm **cân bằng quyết định** (accuracy/F1 giữa lớp), không nhắm **xếp
hạng trong từng nhãn**. Lọc bằng tiêu chí này trước khi đọc bất kỳ bài nào.

#### Còn mở, xếp theo lợi/chi phí

| # | Việc | Chi phí | Vì sao chưa ai làm |
|---|---|---|---|
| 4.4.1 | **Loss thay thế AP trực tiếp** — Smooth-AP (Brown+ ECCV'20), AP-Loss | ~13h | Ta đang dùng `ASL` — một loss **phân loại** — để tối ưu một chỉ số **xếp hạng**. Tối ưu thẳng surrogate của mAP là nước đi hiển nhiên **chưa bao giờ đánh**. |
| 4.4.2 | **Lấy mẫu theo lớp** — oversample ảnh chứa nhãn hiếm | ~1h sửa loader + 13h | Loader hiện là `shuffle` **đều**. Đa nhãn nên phải cẩn thận (một ảnh nhiều nhãn) nhưng có sơ đồ chuẩn. |
| 4.4.3 | **Quét `gamma` ASL theo TIER** | ~13h | `gn=4 / gp=0` cố định cho cả 40 nhãn, chưa bao giờ quét theo tier. |
| 4.4.4 | **Chuyên gia TAIL** (RIDE-style) — mô hình riêng cho 11 nhãn tail, trộn ở tail | ~13h | **Khớp đúng bộ máy `w` theo tier đã có**, và tail hiện nhận **ZERO** từ ensemble. |
| 4.4.5 | **LDAM + DRW** (Cao+ NeurIPS'19) | ~13h | Margin theo phân phối nhãn — đổi hình học biên quyết định, không phải rescale. |

#### Dữ liệu ảnh ngoài — đòn bẩy lớn nhất cho tail

```
PadChest      §4.1, DA trong ke hoach: 4 nhan tail, gap 2-6x ca duong
VinDr-CXR     benh vien Viet Nam (108, HMUH) — KHONG trung MIMIC.
              22 phat hien cuc bo CO BBOX -> vua them nhan, vua mo duong copy-paste ton thuong
NIH ChestX-ray14   NIH Clinical Center — khong trung MIMIC
sinh anh duong tong hop (RoentGen / diffusion CXR)   dau co nhat, nghien cuu dang song
```

⚠️ **Phải xác minh trước khi dùng:** số ảnh / số nhãn / giấy phép của `VinDr-CXR` và
`NIH ChestX-ray14`, và **danh sách nhãn trùng với 40 nhãn của ta**. Quy tắc rò rỉ vẫn giữ: cấm mọi
checkpoint đã huấn luyện **có giám sát trên nhãn MIMIC-CXR**; huấn luyện **mô hình của ta** trên dữ
liệu ngoài thì hợp lệ.

#### Số học của riêng nhánh tail

```
PadChest mot minh, neu 4/11 nhan tail GAP DOI AP:
    tail 0.0966 -> ~0.132   (+37%)   ->  macro +0.010
cong them chuyen gia tail + loss AP + lay mau theo lop:
    0.164 KHONG nam ngoai tam — no chi khong dat duoc bang MOT don bay
```

### 4.5 · SINH ẢNH bằng diffusion cho 16 nhãn tail ⭐ nước đi thắng giải của đội A

Đội A — **hạng 1 tổng thể (`0.281`) và tail tốt nhất (`0.136`)** — sinh **100 ảnh mới cho mỗi lớp
hiếm** bằng conditional denoising U-Net + VAE decoder, prompt mô tả **nhiều bệnh đồng mắc**. Paper
kết luận thẳng: sinh ảnh **và** loss reweighting đều nâng được nhóm tail.

#### Quy mô: mục tiêu "kéo lên medium" lớn hơn thứ đội A làm rất nhiều

```
train 258,871 anh  ->  1% prevalence = 2,589 ca duong/nhan
tail hien tai:  Lobar Atelectasis 0.050% (~129 ca)  ...  Lung Lesion 0.903% (~2,337)
de MOI nhan tail dat 1%:  can ~25,000 - 30,000 anh tong hop
doi A da lam:             100 anh/lop x 16 = ~1,600   -> NHO HON 18 LAN
```

⇒ **"Bao nhiêu là đủ" phải ĐO, không chốt trước ở 1%.** Dựng **đường cong quy mô** trên val:
`{0, 100, 300, 1000, 3000} ảnh/nhãn` rồi xem nó bão hoà ở đâu.

#### Thiết kế đã chốt (Tuấn, 2026-09-08): **TỰ huấn luyện bộ sinh, không lấy mô hình CXR ngoài**

Điều này gỡ sạch vấn đề rò rỉ, và không phát sinh nguyên tắc mới.

```
nen        Stable Diffusion, trong so LAION (ANH TU NHIEN)
           MIMIC-CXR nam sau tuong credentialed cua PhysioNet, khong co tren web mo
           -> CUNG LOAI voi ImageNet (ConvNeXt) va DINOv2 (RAD-DINO) ta DA chap nhan
fine-tune  TRAIN cua ta 258,871  +  PadChest ~160,000   =  ~420,000 anh CXR
           KHONG mot anh dev/test nao
doi chieu  RoentGen fine-tune tren ~60k cap -> ta co gap 7 lan, thua du lieu
```

> **CẤM tuyệt đối:** mọi mô hình sinh công khai đã fine-tune trên MIMIC-CXR (RoentGen…). Nó **đã
> thấy ảnh test kèm report**, và ghi nhớ trong diffusion là hiện tượng đã được ghi nhận. Cùng lỗ
> hổng với checkpoint phân loại có giám sát ⇒ cùng lệnh cấm.

#### Điều kiện hoá: theo VECTOR NHÃN, không phải theo một nhãn

Đội A dùng prompt có **nhiều bệnh đồng mắc**
(*"Round(ed) Atelectasis, Pneumothorax, Pleural Effusion, Lung Opacity, and Atelectasis"*).
Sinh ảnh chỉ-một-nhãn rồi gán one-hot là **tự bơm nhiễu nhãn** vào đúng nhóm đang yếu.

```
1. lay phan phoi DONG MAC thuc nghiem tu TRAIN:  p(y | y_hiem = 1)
2. lay mau mot vector 40 chieu tu phan phoi do
3. dung prompt tu cac ten nhan bat trong vector   "chest x-ray, <ten1>, <ten2>, ..."
4. anh sinh mang DUNG vector 40 chieu do lam nhan
```

Tái dùng nguyên bộ máy điều kiện hoá văn bản của SD, không phải sửa kiến trúc.

#### ⚠️ Rủi ro THẬT — và nó nhắm đúng nhóm nhãn ta cần nhất

```
SD chay o latent 64x64 cho anh 512  ->  moi o latent ~5.5mm mo
Clavicle Fracture: duong gay 1-2mm  ->  DUOI mot o latent
```

Nếu bộ sinh không dựng nổi chi tiết đó, ảnh "dương" **không thật sự chứa bệnh** và ta đang huấn
luyện trên **nhãn sai**. Nhìn mắt thường không phân biệt được "ảnh đẹp" với "ảnh có bệnh".

**Cổng dứt khoát — huấn luyện CHỈ trên ảnh sinh, chấm trên val THẬT:**

```
AP gan 0 cho nhan nao  ->  bo sinh KHONG dung duoc phat hien do  ->  LOAI nhan do khoi §4.5
AP kha                 ->  bo sinh dung duoc  ->  dung anh sinh cho nhan do
```

Rẻ (một lượt huấn luyện đầu đọc), và nó lọc theo **từng nhãn** thay vì chấp nhận/loại cả hướng.

#### Cơ chế thật của ảnh sinh — đính chính bản đầu

Bản đầu tôi viết *"sinh từ 129 ca dương thì chỉ ra 2,460 mẫu của sai số ước lượng"*. Đúng về mặt
thông tin nhưng **bỏ sót cơ chế chính**:

```
129 ca duong that  ->  mo hinh de bam vao dac diem RIENG cua 129 benh nhan do
                       (voc dang, tu the, thiet bi, may chup)
bo sinh hoc TIEN NGHIEM giai phau tu 420,000 anh, roi DAT phat hien hiem len
muon vong giai phau KHAC NHAU
                   ->  BE GAY tuong quan gia giua "phat hien" va "benh nhan cu the"
```

Không phải thêm thông tin về **bệnh**, mà **gỡ tương quan giả** — với 129 mẫu thì tương quan giả
mới là kẻ thù chính. Cơ chế này thật.

Cộng thêm: PadChest mang **thông tin mới thật** cho 4/16 nhãn tail —
`Azygos Lobe 176` · `Hydropneumothorax 549` · `Pneumoperitoneum 454` · `Kyphosis 669`.

#### Đối chứng BẮT BUỘC

```
a) nen                                        <- moc
b) OVERSAMPLE anh THAT cua nhan hiem, CUNG so luong hieu dung
c) anh TONG HOP
neu (c) ~ (b)  ->  diffusion khong them gi ngoai CAN BANG, ma can bang thi MIEN PHI
```

Không có `(b)` thì kết quả dương **không đọc được** — đúng bài học `06b` (`H4z`/`H4`).

Thêm một đối chứng rẻ và rất đáng: **cho bác sĩ / chính ta xem lẫn ảnh thật và ảnh sinh** rồi đoán
cái nào là cái nào. Nếu phân biệt được dễ dàng thì mô hình cũng phân biệt được, và ảnh sinh chỉ dạy
nó nhận ra "ảnh giả" chứ không dạy bệnh.

#### Bậc thang thực thi

```
bac 0  chuan bi du lieu                        ~4h     gop TRAIN + PadChest thanh mot kho anh
                                                       + dung prompt tu vector nhan
bac 1  fine-tune SD tren ~420,000 anh CXR      ~10-30h @512, bf16, VAE dong cung
                                                       (RTX PRO 6000 102GB: batch lon duoc)
bac 2  CONG CHAT LUONG: huan luyen dau doc     ~2h     CHI tren anh sinh -> cham val THAT
       CHI tren anh sinh                               loai tung nhan co AP ~ 0
bac 3  sinh {100, 300, 1000} anh/nhan          ~4-12h  chi cho cac nhan QUA cong bac 2
bac 4  huan luyen lai 64 voi (a)(b)(c)         3 x 13h duong cong quy mo tren val
```

**Đối chứng bắt buộc ở bậc 4** — không có `(b)` thì kết quả dương không đọc được:

```
a) nen  ·  b) OVERSAMPLE anh THAT cung so luong hieu dung  ·  c) anh TONG HOP
neu (c) ~ (b)  ->  diffusion khong them gi ngoai CAN BANG, ma can bang thi MIEN PHI
```

**Cổng khoá trước:** `(c) − (b) ≥ +0.005` ở tail trên val, **và** đường cong quy mô còn đang lên tại
điểm cuối. Nếu `(c) ≈ (b)` ⇒ đóng hướng, chuyển ngân sách sang §4.1 (PadChest) và §4.4.1 (loss AP).

**Kỳ vọng:** `tail +0.01 ~ +0.03` ⇒ `macro +0.004 ~ +0.012`.

**Rủi ro lớn nhất giờ không còn là rò rỉ mà là ĐỘ TRUNG THỰC CHI TIẾT NHỎ** — và cổng bậc 2 đo
thẳng nó, trước khi tiêu giờ nào cho bậc 3-4.

**Rẻ hơn nhiều và nên làm TRƯỚC:** §4.1 PadChest cho 4 nhãn tail là **ảnh THẬT**, `176–669` ca dương
mỗi nhãn, `~8h`, không có cạm bẫy nào ở trên. Ảnh thật từ bệnh viện khác mang thông tin mới **thật**;
ảnh sinh thì không, trừ khi bộ sinh mang nó từ pretrain ngoài.

---

## 5 · Bậc 3 — K-fold với backbone riêng từng fold (~35–58h) ⭐ lớn nhất

### Vì sao k-fold thay vì "thêm seed"

Kết quả `+0.0008` từ thêm seed **không** đóng hướng ensemble — các seed của ta dùng chung một
backbone `38a` **đông cứng** (86M tham số y hệt), chỉ `7.3M` đầu đọc là khác ⇒ tương quan cao là
tất nhiên. Fine-tune backbone theo từng fold cho `K` backbone **khác nhau thật**.

### Nó mua hai thứ với cùng một số tiền

```
1. ENSEMBLE K backbone doc lap      = dung cong thuc vo dich ("fine-tune + ensemble")
2. TAP CHON sach, lon gap 6.6 lan   train OOF 258,871 anh  vs  val 39,293
                                    nhieu uoc luong tail giam ~2.6x (can bac hai)
```

Cho cấu trúc ba tầng sạch: **train-OOF chọn → val xác nhận → test báo cáo.**

### Cạm bẫy phải tránh

Backbone `38a` hiện đông cứng đã fine-tune **có giám sát trên nhãn CXR-LT của TOÀN BỘ train**. Nếu
chỉ k-fold phần đầu đọc thì "out-of-fold" **không thật sự out-of-fold** và ước lượng bị thổi lên.
⇒ **bắt buộc fine-tune backbone riêng từng fold.**

### Chi phí (theo `108 img/s` đo được ở `64`)

```
K=5   moi fold thay 80% train   38a ~6h + dau doc ~5.5h = 11.5h/fold  ->  ~58h
K=3   moi fold thay 67% train                                          ->  ~35h
```

**Triển khai cuối: dùng luôn ensemble K fold làm bài nộp.** Đừng huấn luyện lại trên 100% — bản đó
không có ước lượng held-out nào, còn ensemble K fold vừa mạnh hơn vừa có OOF.

---

## 5.5 · ĐỘ PHÂN GIẢI — hướng số 1 sau `nb67` ⭐

### Bằng chứng: khoảng cách `medium` là bài toán **thị giác**, không phải nhãn

`medium` gánh **71%** khoảng cách, và nó nằm ở **7 nhãn**:

```
Infiltration         0.0644   3.897%   prec 0.261   <- NHAN BAN, loai khoi phan tich nay
Tortuous Aorta       0.0742   1.289%   prec 0.958   <- nhan SACH
Pleural Thickening   0.1248   1.264%   prec 1.000   <- nhan SACH
Adenopathy           0.1349   1.317%   chua co so
Calcification Aorta  0.1492   1.637%   prec 1.000   <- nhan SACH
Fissure              0.1528   1.083%   chua co so
Granuloma            0.1762   1.145%   chua co so
                     TB ~0.125   |   8 nhan medium con lai TB ~0.36
```

Ba nhãn `prec 0.958–1.000` mà AP dưới `0.15` ⇒ **không phải nhiễu nhãn**. Và kích thước phát hiện
nằm **dưới** đơn vị không gian của ta:

```
Fissure ~1mm · Pleural Thickening vai mm · Calcification Aorta mm · Granuloma mm
patch RAD-DINO @518 = 9.46mm
```

Nếu 7 nhãn này lên trung vị medium (`~0.25`): medium `0.2286 → ~0.287` — **vượt vô địch `0.263`**,
macro **`+0.022`**. Đây là đòn bẩy đơn lẻ lớn nhất còn lại.

### `nb51` KHÔNG đóng được hướng này

```
51      doi do phan giai o SUY LUAN, tren backbone da fine-tune o 518
        -> lech phan phoi (noi suy pos-embed) + pha loang attention   -> -0.0028
doi C/D HUAN LUYEN o 1024 tu dau, khong co lech nao                   -> hang 3-4
```

`51` đóng *"đổi độ phân giải lúc suy luận"*. Nó **không** nói gì về *"huấn luyện ở độ phân giải cao"*.

### Chi phí — phải nói trước, nó đắt

```
518px  ->  37x37 =  1,369 token | attention  1.0x | 9.46 mm/patch | ~108 img/s | ~7h/seed
672px  ->  48x48 =  2,304 token | attention  2.8x | 7.29 mm/patch
768px  ->  54x54 =  2,916 token | attention  4.5x | 6.38 mm/patch | ~40 img/s  | ~18h/seed
1024px ->  73x73 =  5,329 token | attention 15.2x | 4.79 mm/patch | ~17 img/s  | ~42h/seed
```

Backbone đông cứng nên chỉ tốn **forward**, nhưng self-attention bậc hai vẫn kéo mạnh. Và để làm
đúng thì phải **fine-tune lại `38a` ở độ phân giải mới** — nếu không lại dính đúng lệch phân phối
mà `51` đo được. Cộng thêm chừng đó nữa.

### Bậc thang

```
5.5.0  nb66 (da dung)                      ~1h      phong to vung attention co giup khong
5.5.1  fine-tune 38a o 768                 ~10h     bo lech phan phoi
5.5.2  huan luyen dau doc o 768, 1 seed    ~18h     do tren val, so voi 518
       CONG: 7 nhan medium tren len >= +0.03 TB, va macro khong tut
5.5.3  neu qua cong: seed thu 2 + can nhac 1024     ~40h+
```

**Đừng nhảy thẳng 1024.** `768` cho `6.38mm/patch` (từ `9.46`) với `4.5×` chi phí; `1024` cho
`4.79mm` với `15.2×`. Đo ở `768` trước, và chỉ leo tiếp nếu đường cong còn dốc.

---

## 6 · Bậc 4 — nếu bậc 0–3 vẫn dừng ở `~0.279`

```
4.1  thanh vien CNN LON hon      convnextv2_nano (13 GMAC, IN-1k 82.0)
                                 -> tiny (24, 84.9) hoac base (82, 86.8)
                                 CNN dang la thanh vien YEU (0.2463 vs duong ong 0.2663);
                                 manh len thi w toi uu tang va blend cho nhieu hon
                                 ~8h  ·  +0.004
4.2  K=5 thay vi K=3             +23h  ·  +0.003
4.3  them kien truc thu 3        Swin / EfficientNetV2 lam thanh vien doc lap
                                 ~10h  ·  +0.003
4.4  do phan giai cao hon        CHI khi nb66 duong
```

---

## 7 · Kỷ luật — cái gì được phép, cái gì không

```
LAT CHON   da gop vao train (R2a), khong con
VAL        TAP CHON hop le — doc lap voi test
TEST       da cham 2 lan (2026-08-21, 2026-09-06), ghi _test_touches_nb61.jsonl
```

**Điều duy nhất còn cấm: chọn TRÊN test.** Giữ sạch bằng cách: chốt danh sách trước → chọn hết trên
val (hoặc train-OOF khi có k-fold) → **đọc test một lượt** → báo cáo **tất cả**, kể cả cái thua.

**Ngân sách đọc test còn lại: 2 lần.**

```
lan 3  sau bac 0+1+2   (cac cai tien re, khong huan luyen lai backbone)
lan 4  sau bac 3       (k-fold) — day la con so NOP
```

**Sàn nhiễu:** `0.0007` lặp lại · `0.005` mức công bố.

**Quy tắc lưu file:** tên duy nhất thì sống; ghi đè cùng tên qua Drive FUSE thì chỉ lần đầu xuống
đĩa. (Mua bằng việc mất trắng trọng số `53/r3`.)

**Quy tắc báo cáo:** số đối chiếu phải đọc từ log/file, không hardcode từ trí nhớ.

---

## 8 · Rủi ro — cái gì có thể làm hỏng kế hoạch

| Rủi ro | Dấu hiệu sớm | Ứng phó |
|---|---|---|
| `0.281` không phải thứ ta nghĩ | Mục 0.1, ~30 phút | Làm **trước tiên** |
| Gain cộng dồn ít hơn tổng (chồng lấn) | Bậc 0 cho `< +0.002` | Nhảy thẳng bậc 3, bỏ bậc 1 |
| K-fold: mỗi mô hình yếu hơn vì ít dữ liệu | Fold đơn `< 0.2500` trên OOF | Chuyển `K=3 → K=5` |
| PadChest lệch miền (máy khác, dân số khác) | 4 nhãn không nhích trên val | Đóng `4.1`, dồn vào `4.2` |
| EHR thành đường tắt | Tráo ảnh mà AP không đổi | Báo cáo đúng như vậy, **không** giấu |
| Hết thời gian | — | Bậc 0+1 luôn xong trong một ngày; bậc 3 cần 2 đêm GPU |

---

## 9 · Thứ tự thực thi

```
NGAY 1    0.1 xac minh con so 0.281                      30 phut, doc tai lieu
          bac 0 (4 phep quet + 3 co che chong overfit)   ~1h, 0 GPU
          bac 1.1 nb66                                   ~1h
          bac 1.2 TTA · 1.3 chay lai nb65                ~2h
          -> CHAM TEST lan 3, bao cao ca bang

NGAY 2    bac 2b WEIGHTED asymmetric loss                 ~13h  <- 5/9 doi dau bang dung, ta chua
          bac 4.5 bac 0: kiem co bo sinh nao SACH MIMIC   ~2h, 0 GPU (doc tai lieu/giay phep)

NGAY 3    bac 2.1 PadChest co-training                    ~8h   anh THAT, khong cam bay
          bac 2.2 EHR mo rong bac 1-2 (song song, 0 GPU)  ~3h

NGAY 4-6  bac 3 k-fold K=3 (nang len 5 neu con gio)       ~35-58h
          -> CHAM TEST lan 4

NGAY 7+   bac 3b sinh anh diffusion (§4.5)                ~30-60h
          CHI chay neu bac 4.5-bac 0 tim duoc bo sinh sach,
          va CHI sau khi doi chung (b) oversample da chay
```

**Vì sao `2b` lên trước mọi thứ khác trong nhóm huấn luyện:** ta **đã có** `ASL` trong mã, thêm
trọng số lớp là vài dòng, và **5/9 đội đầu bảng dùng nó**. Đó là thứ rẻ nhất mà ta bỏ sót lâu nhất.

**Vì sao `3b` (sinh ảnh) xếp cuối dù nó là nước đi của đội hạng 1:** ba cạm bẫy ở §4.5 phải được gỡ
trước, và `§4.1` PadChest cho **ảnh THẬT** với `176–669` ca dương mỗi nhãn ở `~8h` — rẻ hơn nhiều
và không có cạm bẫy nào.

Bậc 0 và 1 xong trong một ngày và **không huấn luyện lại gì** — làm trước để biết nền thật trước
khi cam kết 35–58 giờ.
