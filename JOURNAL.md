# JOURNAL — nhật ký kỹ thuật CXR-LT 2024

> **⚠ 2026-09-10: file này đã bị XOÁ TRẮNG do lỗi script của Claude** (`io.open(J,'w').write(io.open(J).read()+a)`
> — chế độ `'w'` truncate file *trước* khi biểu thức đọc được đánh giá). Mất ~183 KB nhật ký chi tiết.
> Repo **không có git** nên không rollback được.
>
> Bản dưới là **TÁI DỰNG** từ các nguồn còn nguyên: `Event.md`, `PLAN_VUOT_0281.md`,
> `CHECKLIST_STAGE5_OPEN.md`, `PLAN_STAGE5_BREAK_0281.md`, và **output còn lưu trong 12 notebook**.
> Số liệu là thật (đọc lại từ output); phần **diễn giải cơ chế chi tiết theo ngày đã mất**.
>
> Đã khởi tạo git ngay sau sự cố — xem mục cuối.

---

## Mốc số hiện tại

```
TEST (chinh thuc, 78,946 anh, macro-mAP tren 40 nhan)
  duong ong 64/r2a          0.2583
  + cnn_full 64b/r2a        0.2625     +0.0042  CI95 [+0.0033, +0.0051]  LOAI 0
  vo dich (doi A, Table 4)  0.281      cach 0.0185   -> khoang hang 7-8 / 17 doi

VAL  duong ong 0.2663 | + cnn 0.2709

Tier CHINH THUC (paper §3.3, prevalence tren train >10% / 1-10% / <1% -> 9/15/16 nhan)
             val      test    doi A     dong gop vao khoang cach
  head     0.5651   0.5663    0.567     +0.0002    1%   <- DA NGANG vo dich
  medium   0.2342   0.2286    0.263     +0.0129   70%
  tail     0.1399   0.1234    0.136     +0.0050   27%

Chuyen giao val -> test:  head 1.002 | medium 0.976 | tail 0.882
```

**Cấu hình triển khai:** ensemble 20 file `t1_prior_both` (`64/r2a`) ⊕ 16 file `cnn_full`
(`64b/r2a`), trộn theo tier **cũ** (13/16/11) với `head 0.40 | medium 0.40 | tail 0.00`, cổng 0.85.

---

## Kỷ luật đang áp

```
LAT CHON   da gop vao train (R2a), khong con ton tai
VAL        TAP CHON hop le — doc lap voi test; thien lech cua no chi hong vai tro THUOC DO
TEST       da cham 2 lan, ghi _test_touches_nb61.jsonl. Ngan sach con 2 lan.
           Dieu duy nhat con CAM: chon TREN test.
San nhieu  0.0007 (lap lai that) | 0.005 (muc cong bo, hang 2-4 cach nhau 0.004)
```

**Lưu file trên Drive FUSE:** tên file **duy nhất** thì sống; **ghi đè** cùng tên thì chỉ lần ghi
**đầu tiên** xuống được đĩa. Mua bằng việc mất trắng trọng số `53/r3` epoch 2–10 và `epoch_log.json`
của `53/r3` + `41e/r1` (chỉ còn ep1).

**Báo cáo:** số đối chiếu phải **đọc từ log/file**, không hardcode từ trí nhớ. Hai hằng số gõ tay đã
làm cả hai lượt R2a *trông như* sập trong khi thực tế đều tăng.

**Dựng notebook (6 bước, sau khi 3 notebook liên tiếp vỡ lúc chạy):**
```
1. ast.parse tung cell
2. tools kiem_thutu_ast.py     thu tu dinh nghia theo DONG, xuyen cell (ast.parse KHONG bat)
3. ten mo coi                   dung nhung khong bao gio gan
4. grep ten con sot             518, thu muc cu, RUN_TAG cu
5. mo phong I/O                 du lieu gia, xac nhan byte
6. CHAY THAT cell 0-GPU         C0 + bao cao, gia lap torch.cuda TUNG HAM
```

**DataLoader:** mọi tham số là **hàm của độ phân giải**, không phải hằng số —
`RAM = bs × 2 ảnh × MB_ảnh × NW × prefetch`, không có model trong công thức.

---

## R2a — gộp lát chọn vào train (74% → 87%)

`nb64` (đường ống `tpb` + `mldec`) và `nb64b` (`cnn_full`): `TF_ = TR`, `pte_` **mọi epoch**
(bỏ luật last-3), **không** lưu `psl_`, trọng số lưu tên riêng từng epoch.

```
ensemble moi epoch x moi seed, tren VAL
                       n file   macro     head   medium     tail
53/r3  duong ong cu       20   0.2641   0.4758   0.1942   0.1155
64/r2a duong ong R2a      20   0.2663   0.4775   0.1958   0.1193
41e/r1 cnn cu             16   0.2435   0.4667   0.1788   0.0739
64b/r2a cnn R2a           16   0.2463   0.4695   0.1808   0.0779
                                        (tier CU 13/16/11)

duong ong  +0.0022 macro | +0.0039 tail (+3.3%)
cnn_full   +0.0028 macro | +0.0040 tail (+5.4%)
8/8 phep so DUONG, tail huong loi nhieu nhat o ca hai
```

Cơ chế: `+17.8%` dữ liệu là `+17.8%` **ca dương**; nhãn tail hiếm nhất chỉ có `~150` ca dương trong
train nên thêm vài chục ca là đáng kể, còn head đã bão hoà.

**Mối lo bị bác bỏ:** tôi cảnh báo đường ống mạnh lên sẽ kéo `r = AP_cnn/AP_ống` xuống, làm `w=0.40`
thành quá nặng. Đo thật: `head 0.981→0.983 · medium 0.921→0.923 · tail 0.640→0.653` — gần **không
đổi**. Hai thành viên tăng cùng mức tương đối ⇒ `w` khoá vẫn đúng bối cảnh, cổng `0.85` vẫn cho
cùng quyết định.

**Chạm test lần 2** (độ sâu đủ epoch cả val lẫn test): `0.2583 → 0.2625`, bootstrap ghép cặp theo
bệnh nhân `B=300`, `+0.0042 CI95 [+0.0033, +0.0051]`, **loại 0**.

---

## Paper chính thức CXR-LT 2024 — `paper/2506.07984v1.pdf`

Lin et al., arXiv:2506.07984. Thiết lập **khớp tuyệt đối** với ta:

```
Train 258,871 | Development 39,293 | Test 78,946 | 40 nhan
chia theo BENH NHAN 70/10/20, RIENG cho CXR-LT 2024
Metric: mAP = macro-averaged AP across classes; hoa thi xet mAUROC
```

**Bảng xếp hạng Task 1** (Table 4 + 5):

```
Hang Doi  mAP     Head(>10%)  Medium(1-10%)  Tail(<1%)
  1   A   0.281     0.567        0.263        0.136
  2   B   0.279     0.569        0.260        0.133
  3   C   0.277     0.570        0.253        0.136
  4   D   0.277     0.568        0.264        0.125
  5       0.273 · 6  0.269
 TA       0.2625
```

**⚠ Định nghĩa tier của ta LỆCH:** `nb28` chia theo **số ca** → 13/16/11; giải chia theo **tỉ lệ**
(`>10% / 1-10% / <1%`) → **9/15/16** (paper ghi 9/14/16, tổng 39 — lỗi đánh máy). Mọi so sánh tier
trước 2026-09-08 dùng thang riêng. Đã đồng bộ vào `61/65/66/67/68/68b`, giữ `TIER_CU` để đối chiếu.

Kiểm chéo: 9 nhãn đầu bảng của `nb28` là `Support Devices 0.3325` … `Enlarged Cardiomediastinum
0.1145` — **đúng 9 nhãn trên 10%**, khớp paper.

### Tuân thủ nhãn — sạch

```
task1 (40) + task3 (5) = 45 nhan cua labels.csv  [nb28 co assert]
task3 zero-shot: Bulla, Cardiomyopathy, Hilum, Osteopenia, Scoliosis  -> LAB40 KHONG chua
8 notebook khong doc INV['labels_task1'] deu KHONG cham nhan CXR-LT
```

⚠ `cxrlt_index.parquet` giữ **cả 45 cột** — notebook mới lấy nhãn bằng
`[c for c in df.columns if c not in ID_COLS]` sẽ vớ đủ 45. **Luôn chọn theo tên từ inventory.**

**Nhãn dev/test là CHÍNH THỨC** (`labels.csv` phủ cả 377,110 ảnh, phát hành sau khi thi đóng
06/09/2024) ⇒ `0.2625` so trực tiếp được với `0.281`. Nhưng ta **giữ nhãn test** ⇒ đánh giá **hồi
cứu**, không phải nộp mù; thứ thay cho tính mù là kỷ luật chạm test. **Phải nêu trong bài.**

### Các đội đầu bảng làm gì (Table 3 + §4.1)

| Chủ đề | Số đội | Ta |
|---|---|---|
| **Loss re-weighting** (weighted asymmetric loss) | **5/9** (B,C,D,F,G) | ❌ `ASL` trần `gn=4/gp=0` |
| Ensemble | 8/9 | ✅ nhưng 2 kiến trúc (A: 12 mô hình, I: 21) |
| **Vision-language** (ảnh + report) | **8/9** | ❌ không dùng text; ta dùng EHR thay |
| Pretrain lớn trên CXR ngoài | 8/9 | ⚠ RAD-DINO + PadChest |
| Đa góc nhìn | C,D,F,H | ⚠ chỉ trọng số 0.7/0.3 |
| Độ phân giải **1024** | C,D | ❌ ta 518 |
| **Sinh ảnh tổng hợp cho lớp hiếm** | A | ❌ |

Đội **E** gần đúng kiến trúc của ta (DINOv2 + ML-Decoder) nhưng `ViT-L` pretrain trên **710,000 ảnh
CXR**. Paper §3.3 nói thẳng: *"Team A used pretrained large generative models to generate new images
for these tail cases, while Teams B and C applied loss reweighting, indicating that **both approaches
can improve performance in the 'tail' group**."*

---

## `nb67` — chẩn đoán sau R2a (0 GPU)

**Quét ensemble rẻ: CẠN.** Ước `+0.004` của tôi là **sai**, số thật `0.0000`.

```
khoa 0.40/0.40/0.00 (tier CU)   0.2709   <- van TOT NHAT
w quet tren val (tier moi)      0.2709   (head 0.45 | medium 0.40 | tail 0.30)
w ROBUST (bootstrap 60 lan)     0.2709
w CO NGOT ve 0.40               0.2705
3 chieu (+cnn_ms)               0.2693   <- TE hon, lan thu HAI
trung binh HANG   pipe -0.0019 | cnn_full -0.0075 | cnn_ms -0.0053
trong so epoch    `deu` tot nhat
```

**Khoảng cách `medium` (71%) tập trung vào 7 nhãn:**

```
Infiltration         0.0644   3.897%   prec 0.261   <- NHAN BAN
Tortuous Aorta       0.0742   1.289%   prec 0.958   <- nhan SACH
Pleural Thickening   0.1248   1.264%   prec 1.000   <- nhan SACH
Adenopathy           0.1349   1.317%   chua co so
Calcification Aorta  0.1492   1.637%   prec 1.000   <- nhan SACH
Fissure              0.1528   1.083%   chua co so
Granuloma            0.1762   1.145%   chua co so
                     TB ~0.125  |  8 nhan medium con lai TB ~0.36
```

Nếu 7 nhãn này lên trung vị medium (`~0.25`): medium `0.2286 → ~0.287` — **vượt vô địch `0.263`**,
macro `+0.022`.

**Ba nhãn `prec 0.958–1.000` mà AP < 0.15 ⇒ KHÔNG phải nhiễu nhãn**, mà là cấu trúc mảnh dưới
`9.46mm`/patch: `Fissure ~1mm · Pleural Thickening vài mm · Calcification Aorta mm · Granuloma mm`.

**Nhóm nhãn bẩn (Table 8, tập vàng 406 ảnh):** `Lung Lesion 0.161 · Pneumonia 0.191 ·
Infiltration 0.261 · Mass 0.513 · Pneumothorax 0.563 · Enlarged Cardiomediastinum 0.583`
(trung bình toàn bảng 0.711; GPT-4o 0.786).

⚠ **ĐỪNG gắn lại nhãn bằng LLM** — AP chấm trên nhãn test **cũng bẩn y hệt**, và lỗi parser là **hệ
thống** chứ không ngẫu nhiên ⇒ `P(y'=1|x)` không đơn điệu theo `P(y=1|x)`, dạy mô hình đoán đúng sự
thật sẽ kéo nó **xa** khỏi thước đo. Nhóm A nên **hạ trọng số loss**.

---

## Hướng ĐỘ PHÂN GIẢI — `nb68` (ViT @1022)

Fine-tune backbone RAD-DINO ở **đúng** 1022 (bậc A 3 epoch từ `38a/a2_E3`, LP-FT y hệt `38a`), rồi
đông cứng và huấn luyện đầu đọc (bậc B, E=6, 1 seed).

```
             64 @518   68 @1022    chenh
macro         0.2645     0.2590   -0.0055
head          0.5593     0.5545   -0.0048
medium        0.2227     0.2256   +0.0029   <- LEN
tail          0.1379     0.1241   -0.0138   <- XUONG manh
7 nhan medium (bo Infiltration)  +0.0081   (cong >= +0.03: TRUOT)

HAI  Clavicle Fracture -0.0916 (prev 0.065%) · Hydropneumothorax -0.0383 · Pneumoperitoneum -0.0368
LOI  Granuloma +0.0484 (1.145%) · Pneumothorax +0.0124 · Emphysema +0.0095 · Consolidation +0.0092
```

**Mọi nhãn hại nặng là tail, mọi nhãn lợi là medium.** Cơ chế: `40` query phải học phân bố attention
trên `5,329` khoá thay vì `1,369`; nhãn `26,000` ca dương học được, nhãn `129` ca dương thì **pha
loãng thống kê**. Chỉ dấu phụ: `G_ctx` tụt `+0.0115 → +0.0029`, `ratio_img` tăng `0.19 → 0.27`.

**Hai phép đo trước KHÔNG đóng được hướng này:**
```
51  doi do phan giai o SUY LUAN, dau doc van huan luyen o 518 -> lech phan phoi  -> -0.0028
66  crop hop bao roi phong to — hop bao phu 79% anh, phong dai THUC chi 1.06x    -> +0.0003 VO NGHIA
```
`66` có cơ chế tự vệ (in diện tích hộp bao, cảnh báo >0.60) và nó **đã bật**; tôi vẫn đọc nhầm dòng
`"CASCADE DONG"` — đã rút lại. Sâu hơn: **crop bao nhiêu cũng không đủ phóng đại** vì phát hiện trải
khắp lồng ngực; muốn nhiều pixel/mm chỉ có cách tăng kích thước đầu vào.

Nguyên nhân hộp bao rộng: tôi lấy **trung bình attention qua 29 query** head+medium, trong khi `65`
đo độ tập trung **theo từng nhãn** — mỗi nhãn tập trung nhưng ở **chỗ khác nhau**.

### Giả thuyết của Tuấn: CNN tận dụng độ phân giải tốt hơn ViT — paper xác nhận

```
ViT-B/14  luoi PHANG, attention toan cuc; them pixel = them KHOA phai HOC
          518 -> 1,369 khoa  |  1022 -> 5,329 khoa
ConvNeXt  PHAN CAP stride 4/8/16/32, tich chap CHIA SE trong so
          518 -> 16x16 = 256 khoa  |  1022 -> 31x31 = 961 khoa   (do bang timm)
          1022px stride-4 = 1.37mm  <- duong gay xuong don 1-2mm NAM TRONG TAM
```

Table 3: **mọi đội dùng `1024` đều dùng CNN; cả hai đội ViT-L đều ở `≤512`.** Không đội nào chạy ViT
ở 1024.

⚠ Caveat: bậc A chỉ 3 epoch, 1 seed, E=6 ⇒ một phần tụt tail **có thể là thiếu huấn luyện**. Phát
biểu an toàn: *"ở cùng ngân sách, ViT không đổi được độ phân giải thành lợi ích; CNN thì chưa thử"*.

---

## `nb65` — attention khu trú (SƠ BỘ, checkpoint chỉ epoch 1)

```
k giu      head   medium   tail
  1%      0.972   0.908   0.862        14/1369 patch
  5%      0.988   0.964   0.963        CONG >=0.90 QUA o ca ba tier
 20%      0.997   0.996   1.029        <- tail TOT HON luot day du
do tap trung (top-5%; deu = 0.05):  head 0.361 | medium 0.275 | tail 0.222
```

Cổng qua dứt khoát — nhưng nó **rút mất động cơ** của cascade thay vì mở đường: mô hình **không mất
dấu vị trí**. Giả thuyết duy nhất còn sống là **độ phân giải**.

Ba nhãn quen thuộc: `Azygos Lobe 1.025` (tốt hơn khi che) · `Hydropneumothorax 0.880` ·
`Pneumomediastinum 0.845` (tập trung nhất `0.300` nhưng mất nhiều nhất — khí trung thất là **vệt
sáng mảnh chạy dọc bờ trung thất**, cần cả đường).

---

## `cnn_ms` đo lại tử tế — marginal, chưa đóng được

```
w theo tier            +0.0006   (head w* 0.05 | medium 0.05 | tail 0.25)
tail tru 3 nhan        +0.0004   ·  cau truc manh +0.0005  ·  hop 2a+2b +0.0001
phan CHAP uoc          +0.0007   (cnn_ms van o 74% du lieu — phep do BAT DOI XUNG:
                                  DUONG thi hop le va la can duoi; AM chi doc duoc neu DU LON)
=> toi da ~+0.0013, ngang/duoi san nhieu
```

**Đối chứng bắt được sai lầm của tôi:** nhóm "3 nhãn xấu" phải sửa thành **2**.
```
Azygos Lobe -0.0037 (loai DUNG) · Pneumomediastinum -0.0031 (DUNG) · Hydropneumothorax +0.0147 (SAI)
```
`Clavicle Fracture +0.0255 (+26%)` — cơ chế thang-mịn hoạt động **đúng chỗ đã dự đoán**.
`Hydropneumothorax` lên ở **cả** `cnn_ms (+0.0147)` **lẫn** `nb66 nhánh B (+0.0119)` — hai đường
không dùng chung máy móc.

`64c` (R2a cho `cnn_ms`) **chưa từng được ghi ra đĩa** (build trượt anchor); đã hạ ưu tiên.

---

## Hiện vật thống kê — bài học đắt nhất

```
Spearman(r, delta)   r_val vs delta_val   +0.598  p<1e-4    <- CUNG mau
                     r_sel vs delta_val   +0.136  p=0.40    <- KHAC mau, con so THAT
```

`r_j = AP_cnn_j / AP_ống_j` và `delta_j = AP_blend_j − AP_ống_j` **cùng chứa `AP_ống_j`**. Đo cả hai
trên cùng tập val thì nhiễu của `AP_ống_j` vào cả hai vế và **tự sinh** tương quan.

**Rút lại toàn bộ chương trình `w = f(r)`** (tuyến tính, isotonic, co ngót). Phát hiện được nhờ đưa
vào biến `rho` (tương quan dự đoán, **không dùng nhãn**, cỡ mẫu 39,293 thay vì 19–154 ca dương): độ
tin `1.000` nhưng `Spearman(rho, delta) = +0.017`. Chính sự tương phản *biến bẩn +0.598 / biến sạch
+0.017* làm lộ hiện vật.

```
r THEO NHAN   spearman(r_sel, r_val)  head 0.62 | medium 0.64 | tail 0.22
              |r_sel - r_val| TB      head 0.02 | medium 0.08 | tail 0.42  (thang rong 1.1)
ti le THEO TIER (gop 11-16 nhan)      tail: sel 0.68 vs val 0.64  KHOP
```

⇒ Cổng `0.85` đứng vững, và lý lẽ của nó là **đo lường được**, không phải "ít tham số thì an toàn".

---

## Hướng đã ĐÓNG (đừng làm lại)

| Hướng | Bằng chứng |
|---|---|
| Deep EHR conditioning | `56`: `G_deep` **âm**, ổn định qua seed |
| Mở băng block 8–11 | `57`: `G_ft` `−0.007..−0.010`, 2 seed đồng thuận |
| cRT / decoupled classifier | `59`: classifier **đã hội tụ** |
| Đa tỉ lệ `cnn_ms` làm hướng chính | `41f` trượt cổng; `62`: stage 3 **gần rank-1** (hạng hiệu dụng 1.1/320 kênh) ⇒ khoá gần cộng tuyến |
| Độ phân giải ĐẦU VÀO ở suy luận | `51`: `lb672` → `−0.0028` |
| Thêm seed của cùng backbone đông cứng | `+0.0008` < sàn nhiễu. ⚠ **Không** đóng ensemble backbone fine-tune ĐỘC LẬP |
| Neural collapse | `62` C5 |
| Toàn bộ `w = f(r)` | Hiện vật nhiễu chung |
| `rho` làm biến điều kiện | Đo được (`1.000`) nhưng **vô thông tin** (`+0.017`) |
| Quét ensemble 0-GPU | `67`: `+0.0000` đo được |
| ViT @1022 làm đường ống chính | `68`: `−0.0055` macro, tail `−0.0138` |

---

## Còn mở — xếp theo lợi/chi phí

```
68b   ConvNeXtV2 @1022        ~17h (thu gon)   DANG CHO CHAY — phep thu truc tiep gia thuyet CNN
4.1   PadChest lam NGUON NHAN ~8h    Azygos Lobe 176 · Hydropneumothorax 549 ·
      cho 4 nhan tail                Pneumoperitoneum 454 · Kyphosis 669
                                     nhan hiem nhat cua ta ~150 ca duong -> gap 2-6 lan
4.4   weighted asymmetric loss ~13h  5/9 doi dau bang dung, ta CHUA
4.2   EHR mo rong (thu thuat/tho may) ~3.5h bac 1-2, 0 GPU
                                     Hydropneumothorax 4.2x · Pneumomediastinum 4.8x
4.4.1 loss thay the AP (Smooth-AP)   ~13h  ta dung loss PHAN LOAI cho chi so XEP HANG
4.4.4 chuyen gia TAIL (RIDE-style)   ~13h  khop bo may w-theo-tier; tail hien nhan ZERO
5     K-fold, backbone tung fold     ~35-58h  ensemble + tap chon sach gap 6.6 lan
4.5   sinh anh diffusion             ~30-60h  TU huan luyen (SD/LAION -> train + PadChest)
3     nb40 nhiem vu thu hai          ~1 nb    bien ket qua AM thanh phat bieu CO BIEN
```

**Bộ lọc bắt buộc cho long-tail:** macro-AP tính **từng nhãn** rồi trung bình ⇒ mọi biến đổi **đơn
điệu theo từng nhãn** là **no-op** (logit adjustment, balanced softmax hậu kỳ, chỉnh ngưỡng). Chỉ
loss-trong-huấn-luyện, lấy mẫu, margin, dữ liệu, chuyên gia mới đổi được.

---

## 2026-09-10 · `68b` TRÀN RAM 176 GB — không phải model, là hàng đợi ảnh của DataLoader

```
model  ConvNeXtV2-nano 15M tham so  ->  VRAM, ~60 MB
RAM    tensor anh DA GIAI MA trong hang doi prefetch cua worker
       RAM = bs x 2 anh x MB_anh x NW x prefetch      (khong co model trong cong thuc)

 518px:  3.22 MB/anh  ->  6.44 MB/mau
1022px: 12.53 MB/anh  -> 25.07 MB/mau      (x3.9)
```

Thủ phạm: `predict(idx, bs=128)` + `NW = min(16, cpu)` + `prefetch_factor=4`, và `_DLC` cache giữ
eval loader (val **và** test) sống vĩnh viễn với `persistent_workers=True`.

```
                                       518px      1022px
eval bs=128 x NW16 x prefetch4         52.7 GB   205.6 GB
tong (train + 2 eval + chuan doan)    136.6 GB   532.6 GB   <- tran 176 GB
```

**`nb68` (ViT) không dính** vì dòng dõi `53` dùng `nw=0` cho eval; chỉ train loader có worker
(`25.7 GB`). `nb68b` dòng dõi `41e` dùng `_DLC` cache **có worker** — tối ưu đúng ở 518, lỗi chí
mạng ở 1022.

**Sửa:** ngân sách theo độ phân giải + `assert _RAM_GB < 60`. `532.6 GB → 18.4 GB`. `BS_EVAL` chỉ
đổi tốc độ, **không đổi kết quả** (`predict` ghi theo `b['pos']`).

---

## 2026-09-10 · MẤT JOURNAL và biện pháp

Script log dùng `io.open(J,'w').write(io.open(J).read()+a)` — `'w'` truncate **trước** khi biểu thức
đọc được đánh giá. Mất ~183 KB. Repo **không có git** ⇒ không rollback được.

**Đã làm:** khởi tạo git + commit toàn bộ repo; tái dựng JOURNAL từ `Event.md`,
`PLAN_VUOT_0281.md`, `CHECKLIST_STAGE5_OPEN.md`, `PLAN_STAGE5_BREAK_0281.md` và output còn lưu
trong 12 notebook.

**Quy tắc từ nay:** ghi thêm vào file **luôn** dùng chế độ `'a'` (append), hoặc đọc xong đóng file
rồi mới mở ghi — **không bao giờ** lồng `read()` bên trong `open(...,'w').write(...)`.
