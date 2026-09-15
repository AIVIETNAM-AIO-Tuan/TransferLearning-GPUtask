<!-- Ban CHINH THUC cua ke hoach nay nam o day, trong repo.
     Ban duyet goc: ~/.claude/plans/d-c-v-y-th-scalable-quokka.md (15/09, giu lam lich su).
     Sua thi sua file NAY. -->

# Sinh ảnh cho nhãn thiểu số bằng SỬA ẢNH THẬT (img2img) — kế hoạch vòng 1

## Context

Test hiện tại `0.2656`, cách vô địch `0.0154`. Phân rã trên thang chính thức: **medium giữ
`+0.0107` (70%)**, tail `+0.0047` (30%), head ta đã dẫn (`0.5690` vs `0.5670`).

Hướng này thay `PLAN_VUOT_0281.md` §4.5 (chỉ nhắm tail, điều kiện hoá theo vector nhãn đồng mắc).
Tuấn đổi hướng (15/09): sinh cho **cả medium lẫn tail**, chuyên biệt hoá theo nhãn.

**Số liệu nền** (nb28, train 258.871): mất cân bằng `86.079 / 129 = 667×`, trung bình **2,47 nhãn
mỗi ảnh**. 31 nhãn thiểu số = 15 medium (`2.803`–`15.371` ca dương) + 16 tail (`129`–`2.338`).

**Report text**: `mimic-cxr-reports\files\` có sẵn **227.835 file `.txt`** (đủ bộ, untracked từ
17/07, chưa một dòng code nào trong dự án dùng). Tuấn đang chuẩn bị. **Không dùng ở vòng 1.**

### Quyết định thiết kế (15/09, sau hai vòng rà soát)

- **Bỏ text→image.** Chỉ **sửa ảnh thật** (img2img/SDEdit). Ảnh gốc **giảm gánh nặng** dựng giải
  phẫu — ⚠ không phải "giải phẫu miễn phí": sau khi thêm nhiễu tới `t = s·T` rồi khử, giải phẫu
  lẫn bệnh lý **đều có thể trôi**. Mức trôi là thứ phải **đo**, không phải thứ kiến trúc bảo đảm.
- **Text chỉ điều hướng**, không dựng ảnh.
- **Chỉ P→P**, tách hai nhánh:

  | Nhánh | Prompt | Nhãn ảnh sinh | EHR + phim trước |
  |---|---|---|---|
  | **P-giữ** | chỉ prompt dương | nhãn gốc là **nhãn dự kiến** — mức bảo toàn phải đo | kế thừa, sau khi đo trôi |
  | **P-xoá** | thêm prompt âm head | **ẩn số**, phải đo | kiểm riêng từng trường |

  ⚠ "P-giữ" là **mục tiêu, không phải bảo đảm**. Không dùng prompt âm chỉ nghĩa là ta không *chủ
  động* yêu cầu xoá; img2img vẫn có thể làm mất một nốt nhỏ, đổi mức tràn dịch, làm mờ thiết bị,
  hay thêm một đám mờ không có ở ảnh gốc. Prompt dương chỉ nhắc một bệnh nên 39 nhãn kia không
  được ràng buộc gì.

  ⚠ P-xoá **đã là chỉnh sửa phản thực**, không còn là biến thể bảo toàn. Không dùng kết quả nhánh
  này để tự động gán one-hot.

  ⚠ **Cái giá của việc tách nhánh, ghi thẳng:** P-giữ sạch về đo lường và cũng **ít khả năng ăn
  tiền nhất** — không thêm tổ hợp nhãn mới, không thêm giải phẫu mới. **Giả thuyết trung tâm nằm ở
  P-xoá**, mà P-xoá lại là nhánh không xác minh được nhãn đầy đủ. Dự đoán ghi **trước** khi chạy:

  ```
  P-giu  ->  du doan AP3 ~ APdup
  P-xoa  ->  day moi la cho co the co chenh lech
  neu P-giu LAI an tien  ->  KHONG phai gia thuyet ban dau dung, ma la ket qua KHAC, giai thich rieng
  ```

- **Caption lúc huấn luyện = TOÀN BỘ nhãn dương của ảnh** (xem Bước 2). Đây là sửa lỗi gốc: caption
  một-nhãn làm cơ chế prompt âm **không thể hoạt động về nguyên tắc**.
- **Bộ nén: chọn, không tự huấn luyện.** `MedVAE` (`arXiv:2502.14753`) đã có sẵn, huấn luyện trên
  1M ảnh y tế **chỉ bằng ảnh** — cùng ô với RAD-DINO.
- **Giữ lệnh cấm RoentGen**, có ba trích dẫn từ chính bài báo của nó (Bước 2).

---

## Quy tắc quyết định — chốt TRƯỚC khi chạy

Kế hoạch này diễn giải kết quả một cách thận trọng **nhưng phải ra được quyết định**. Cách đọc yếu
đi mà không kèm hành động cứng thì "cần kiểm tiếp" trở thành cách để không bao giờ dừng.

### Hai đợt ngân sách

```
DOT 1  ->  Buoc 0 + 1 + 2 + 3        ~20h    -> bang quyet dinh duoi day
DOT 2  ->  Buoc 4 + 5                ~50-60h -> CHI mo neu Dot 1 cho phep
```

Đợt 2 **không tự động** được cấp. Mọi khả năng khác — đổi bộ nén, tăng độ phân giải, cắt vùng,
thêm report, N→P — nằm ở mục **Đề xuất vòng sau**, **không phải nhánh cứu hộ** khi vòng này hỏng.

### Ngưỡng `δ` — chốt cuối pilot, trước khi chạy đủ 31 nhãn

```
sigma  := do lech chuan cua AP1 qua R=5 lan lay lai be am va mau nguon   (do trong pilot)
delta  := max(2·sigma, 0.01)                                            (chot, khong sua sau)
neu 2·sigma > 0.05  ->  THUOC HONG. Sua dau do hoac dung. KHONG nang delta de cho qua.
```

`δ` và mốc `8/31` là **tiêu chí ra quyết định của dự án**, không phải bảo đảm thống kê. Ghi rõ như
vậy trong bài.

### Bảng hành động

| Kết quả | Hành động đã định trước |
|---|---|
| `AP2 < AP1` ở một nhãn | Không đưa nhãn đó vào Bước 4 **ở cấu hình hiện tại**; lưu kết quả |
| `AP3 − APdup ≤ δ` ở `≥ 24/31` nhãn | **Dừng cấu hình**, không mở Đợt 2 |
| `≥ 8/31` nhãn có `AP3 − APdup > δ` **và** khớp trên tập xác nhận | Mở Đợt 2 **chỉ cho** những nhãn đó |
| Mọi kết quả còn lại, kể cả pilot tốt nhưng xác nhận không đạt | **Dừng cấu hình trong ngân sách hiện tại** |
| `\|APdup − AP1\|` lớn hơn `δ` | Thước hỏng — sửa đầu dò trước, chưa đọc gì khác |

⚠ **Tập xác nhận chạy MỘT lần** theo thiết kế đã chốt. Dùng nó để sửa cấu hình rồi thử lại thì nó
không còn làm đúng vai trò xác nhận.

---

## Bước 0 · Chọn bộ nén — **~5h**

### Vai trò của VAE

Trong latent diffusion, **VAE không sinh ảnh — nó nén**. Diffusion nhiều bước chạy ở latent
`128×128` thay vì pixel `1024×1024` (rẻ hơn 64 lần).

```
x (1024×1024)  --encode-->  z (128×128×4)      [nen 48 lan]
                               │
                               ▼   DIFFUSION nhieu buoc chay o DAY
                             z'
z' --decode-->  x' (1024×1024)
```

⚠ Lo ngại "VAE một bước phá rồi dựng lại nên kém diffusion" đúng với **VAE làm bộ sinh** — lấy
`z ~ N(0,I)` rồi giải mã, ra ảnh mờ. Ở đây **`z` luôn đến từ ảnh thật**; VAE của SD đặt trọng số
KL `1e-6` nên gần như autoencoder tất định. Diffusion vẫn là bộ sinh, sinh trong không gian nén.

⚠ **Không suy ra ngưỡng phát hiện từ `mm/ô`.** Latent không phải ảnh thu nhỏ bằng trung bình ô:
chi tiết được mã hoá qua nhiều kênh và receptive field chồng lấp, và **chi tiết dưới một ô vẫn phụ
thuộc điều kiện bệnh lý** qua giá trị latent. Điều còn đứng được là **giới hạn dung lượng**, đếm
được: `128×128×4 = 65.536` số dựng `≈ 1,05` triệu pixel, nên ảnh sinh nằm trên một đa tạp chiều
`≤ 65.536`, ảnh thật thì không. Đa tạp đó có chứa đúng kết cấu tổn thương hay không là **câu hỏi
thực nghiệm**, trả lời ở Bước 3.

### Thứ tài liệu đã trả lời, khỏi chạy lại

**`arXiv:2210.04133`** (Chambon et al.) đã chạy phép đo này trên MIMIC-CXR với VAE gốc của SD:

| Nhãn | AUC gốc | Sau vòng VAE | Δ |
|---|---|---|---|
| Pleural Effusion · Edema · Cardiomegaly · Consolidation | — | — | **+0,002 … +0,004** |
| Pneumonia | 0,753 | 0,739 | **−0,014** |
| Pneumothorax | 0,811 | 0,791 | **−0,020** |
| Lung Lesion | 0,766 | 0,728 | **−0,038** |

Bác sĩ đọc phim: *"blurred device components, cerclages and **wires that couldn't be traced
reliably**, or **blurred rib contours**."* Đo ở **512**; ở 1024 có gấp bốn số ô cho cùng giải phẫu.

### Ứng viên bộ nén

**MedVAE** (`arXiv:2502.14753`, Stanford, `StanfordMIMI/MedVAE`): huấn luyện trên **1.021.356 ảnh
y tế** / 19 bộ, gồm MIMIC-CXR và CANDID-PTX. **Chỉ ảnh, không nhãn, không report** — hàm mất mát
dùng BiomedCLIP nhưng so `b(x)` với `b(x̂)`, hai ảnh, không văn bản nào đi vào. **Cùng ô với
RAD-DINO** mà dự án đã chấp nhận. Phải **nêu rõ trong bài**.

⚠ **Ký hiệu**: `f` của MedVAE là hệ số giảm **diện tích**; `f=64` ⇒ stride tuyến tính `8` (tương
đương `f=8` của SD); `f=16` ⇒ stride `4`.

| Biến thể | Latent @1024 | mm/ô | Hình dạng | PSNR CXR |
|---|---|---|---|---|
| SD-VAE gốc (stride 8, C=4) — **đường cơ sở** | 128×128×4 | 3,29 | gốc của SDXL | 36,37 |
| MedVAE `f=64, C=4` (stride 8) — **ứng viên** | 128×128×4 | 3,29 | trùng hình dạng | 38,88 |
| MedVAE `f=16, C=3` (stride 4) — **chỉ chẩn đoán** | 256×256×3 | 1,64 | khác hình dạng | 43,55 |

⚠ **Trùng hình dạng KHÔNG phải thay thẳng.** `shape(z_SDXL) = shape(z_MedVAE)` không kéo theo
`z_SDXL ≈ z_MedVAE`. UNet của SDXL học khử nhiễu trong **không gian latent cụ thể** của VAE gốc;
đổi encoder thì LoRA phải học đồng thời **miền CXR lẫn cách diễn giải latent mới**.

⚠ **Dùng `z`, không dùng `z̄`.** MedVAE có lớp chiếu ở Stage 2 nhưng `z̄` chỉ phục vụ *đánh giá
chất lượng latent*; đường giải mã chạy trên `z` gốc. Ghim rõ trong mã.

### 0a · Tái tạo (~3h, không huấn luyện)

`decode(encode(x))` trên ~2.000 ảnh **val thật**, chấm **đầu đọc hiện tại**, so AP gốc **theo từng
nhãn**. Ảnh xám: lặp 3 kênh vào, trung bình ra.

Đây là **cổng đủ tốt**, không phải cuộc thi: ứng viên nào mất không đáng kể thì đi tiếp sang 0c.
`f=16` chạy ở đây **chỉ để chẩn đoán** — trả lời *"mất mát do nén quá thô hay do bản thân bộ
nén"* — và kết quả của nó là **đầu vào cho vòng đề xuất sau**, không phải lối thoát của vòng này.

### Encoder và decoder KHÔNG đối xứng — ba lý do đều chỉ về decoder

```
1. THONG TIN   encoder VUT BO, decoder KET XUAT. Cai da vut thi decoder khong dung lai duoc;
               nhung neu thong tin CON trong z thi decoder tot hon se ket xuat tot hon.
2. THUC NGHIEM arXiv:2605.17087 tren MIMIC-CXR long-tail: phan loai tren anh qua vong nen .261
               vs anh that .263 (p = 0,375)  ->  encoder KHONG pha huy thong tin phan biet.
3. KIEN TRUC   z  = E(x)  -> huan luyen E thi z DOI CHO   -> UNet phai hoc lai tu dau
               x' = D(z)  -> huan luyen D thi z KHONG DOI -> UNet khong he bi anh huong
```

⚠ **Lý do 3 kéo theo một hệ quả lịch trình tôi từng ghi sai.** UNet **không bao giờ nhìn thấy `x`**
— nó chỉ làm việc với `z`. Nên **tinh chỉnh decoder nằm NGOÀI đường găng**: chạy được sau Bước 2,
thậm chí sau Bước 5, và thay decoder lúc nào cũng hợp lệ mà không phải huấn luyện lại UNet. Ràng
buộc *"chốt bộ nén trước khi huấn luyện UNet"* ở Bước 2 **chỉ áp cho encoder**.

⚠ Điều kiện duy nhất làm nguyên tắc này hỏng: **encoder thật sự phá huỷ tín hiệu**, khi đó decoder
sâu bao nhiêu cũng vô ích. Đó chính xác là thứ `0a` đo, và là lý do `0a` vẫn đứng trước.

**Bằng chứng trong chính dự án này**, cùng chiều:

```
mo bang backbone RAD-DINO (nb57)      :  G_ft  s42 −0.0071  |  s43 −0.0103   -> DA DONG
thay dau doc bang ML-Decoder (H5/H1)  :  0.4336  vs  0.4117                  -> +0.0219
```

Đầu tư phía decoder thắng đo được; đầu tư phía encoder thua đo được, trên hai seed.

### 0c · Thích nghi (~2h) — nửa còn lại, và là nơi quyết định

LoRA ngắn (`~1h`) trên từng latent ứng viên, **cùng learning rate, cùng rank, cùng số bước**.

**Đặt LoRA ở đâu trong UNet — đo, đừng khẳng định.** Ta làm img2img ở `s = 0,4–0,6` nên `z_t` đã
gần một latent CXR thật; việc phải làm là **tinh chỉnh lại**, không phải nhận dạng từ đầu — đó là
việc của nhánh đi lên. Và up blocks nhận **skip connection** từ down blocks, nên đông cứng down
blocks rồi cho up blocks thích nghi tức là bắt chúng **diễn giải lại** đặc trưng LAION thành đầu ra
X-quang, có chỗ để điều chỉnh thay vì phải suy ra ngay.

⚠ Đây là **lập luận**, không phải bằng chứng — tôi không có kết quả công bố chắc chắn cho
*"up > down"* trong thích nghi miền của diffusion. Quét ba cấu hình trong cùng `2h` của 0c:

```
toan bo attention  |  chi up blocks  |  chi cross-attention
```

⚠ **KHÔNG so bằng loss khuếch tán.** Loss là MSE trên nhiễu dự đoán *trong latent*; mỗi VAE có
thang và thống kê riêng, và kể cả sau chuẩn hoá phương sai thì **độ khó bài toán khử nhiễu cũng
khác nhau**. Loss thấp hơn có thể chỉ nghĩa là latent dễ đoán hơn. Cùng kiểu lỗi với cổng hỏng cũ:
đo trong không gian không so được.

**Đo trong không gian dùng chung:**

```
sinh ~200 anh cho 3-4 nhan o MOI latent, s = 0,5
chay DUNG bo dau do cua Buoc 3 (AP1 / AP2 / AP3 / APdup)
cham tren val THAT -> cung don vi, so duoc     + 20 anh mat thuong
```

⚠ **Phạm vi kết luận.** Nếu MedVAE thua sau `~1h`, kết luận đúng là *"MedVAE không hiệu quả hơn
**trong ngân sách thích nghi pilot hiện tại**"* — **không phải** *"UNet không học nổi latent
MedVAE"*. Cùng LR và rank là đường cơ sở công bằng về **cấu hình**, chưa phải cấu hình tốt nhất cho
từng latent. Với ngân sách hạn chế, chọn phương án học nhanh hơn là hợp lý; chỉ cần giới hạn đúng
phạm vi câu kết luận.

**Cả hai ứng viên đều hỏng ⇒ dừng vòng này**, ghi lại. Không tự huấn luyện bộ nén ở vòng này.

**Notebook:** `experiments/70_sinh_anh/70a_chuan_bi.ipynb` C0–C4.

---

## Bước 1 · Chuẩn bị dữ liệu — **~4h CPU, song song Bước 0**

**Kho ảnh.** `cxr_1022_letterbox_cxrlt/` (378 tar, 49,3 GB, 377.110 ảnh, 100% FOV).
Lọc `split == 'train'` qua `cxrlt_index.parquet` → **258.871 ảnh**. **Không một ảnh dev/test nào.**

**Giữ nguyên letterbox, KHÔNG cắt viền.** Với img2img viền đến thẳng từ ảnh gốc — miễn phí, và ảnh
sinh khớp luôn với loader đang chạy.

**Chọn ảnh gốc.** Với mỗi nhãn `j`, xếp ảnh dương theo **số nhãn dương tăng dần**, ưu tiên đầu
bảng: ảnh mà `j` gần như là nhãn duy nhất thì prompt âm phải xoá ít hơn, EHR kế thừa ít mâu thuẫn.

**Dựng prompt suy luận.** Với mỗi `j`, tính `p(y_head | y_j = 1)` trên train lấy ba nhãn head đồng
mắc nhất.

```
prompt duong : "chest radiograph, <ten benh j>"
prompt am    : "<head_1>, <head_2>, <head_3>"      (chi nhanh P-xoa)
```

**Chẩn đoán lối tắt head (~1h, 0 GPU).** Với mỗi `j`, hồi quy logistic từ **9 dự đoán head của
chính mô hình** ra `j`, cross-fit trên val, so `AP_head_only(j)` với `AP_model(j)`.

```
ty so ~ 1  ->  chi bao: dang nhan j nao co the doc qua head  ->  uu tien thu P-xoa truoc
ty so ~ 0  ->  chi bao nguoc lai
```

⚠ **Cả hai chiều đều chưa chứng minh gì.** Tỉ số cao không chứng minh mô hình dùng lối tắt (`j` có
thể vừa đồng mắc thật vừa được đọc đúng qua đặc trưng riêng); tỉ số thấp cũng không chứng minh mô
hình đọc đặc trưng thật. Đây **chỉ là thứ tự ưu tiên** cho pilot Bước 3.

Dữ liệu có sẵn: `stage4_nb64/r2a`, `stage4_nb64b/r2a`, `stage4_nb68b/r1022_E8`. Tái dùng `ap_cols`
và `_doc14` trong `61_blend3.ipynb` C14.

---

## Bước 2 · Tinh chỉnh UNet — **ước ~10h, ĐO trước khi chốt**

**Nền:** SDXL base, trọng số LAION (ảnh tự nhiên). Cùng loại với ImageNet cho ConvNeXt và DINOv2
cho RAD-DINO.

### Caption lúc huấn luyện — sửa lỗi gốc

⚠ Nếu ảnh có ba bệnh mà caption chỉ nêu một, mô hình vẫn được tối ưu để **tái tạo cả ảnh ba bệnh**.
Nó học rằng *"caption nêu `j`"* tương thích với *"ảnh có `j` + `h₁` + `h₂`"* — **sự vắng mặt của
một từ không mang thông tin gì**. Mà cơ chế prompt âm ở suy luận **hoàn toàn dựa vào** việc vắng
mặt có mang thông tin. Với caption một-nhãn, **P-xoá không có đường dẫn cơ chế nào để hoạt động.**

```
caption  :  "chest radiograph, <tat ca nhan duong cua anh do>, <view PA|AP|LAT>"
              thu tu nhan XAO ngau nhien de tranh hoc theo vi tri
text dropout : 10%  -> de CFG co nhanh khong dieu kien
```

Nhãn lấy từ `cxrlt_index.parquet` (nhãn train, không rò rỉ). **Không dùng report ở vòng 1.**

### Vì sao KHÔNG dùng RoentGen (ba câu từ `arXiv:2211.12737`)

1. *"Nine subgroups (**P10-P18**, excluding the studies used in the official MIMIC-CXR test set)
   were used for training and development."* — p10–p18 chứa ~90% tập test của ta; bản phát hành
   huấn luyện trên `175.622` ảnh trong đó ⇒ **≥ ~36.000 trong `78.946` ảnh test của ta (≈46%) nằm
   trong tập huấn luyện của nó, kèm report** — mà nhãn CXR-LT được trích từ report. Rơi đúng ô "có
   giám sát trên ảnh + văn bản" mà dự án đã cấm (`PLAN_STAGE4…:439`, `:970`).
2. *"At an image resolution of **512x512** px…"* — ô latent `6,57mm`.
3. *"**the VAE was kept frozen**"* — dùng nguyên VAE gốc SD v1.4, **không có bộ nén CXR để mượn**.

⚠ Ranh giới nhất quán toàn dự án: **tự giám sát trên ảnh ⇒ nhận** (RAD-DINO, MedVAE); **có giám
sát trên ảnh + nhãn/report ⇒ cấm** (`torchxrayvision -mimic_*`, RoentGen).

**Phép trừ:** RoentGen **chỉ** tiết kiệm Bước 2 — không tiết kiệm Bước 0 (bộ nén của nó là bộ nén
gốc), cũng không tiết kiệm Bước 1, 3, 4, 5. Khoảng **13%** tổng, đổi lấy trần 512, rò rỉ ~36.000
ảnh test, và một con số cuối không bảo vệ được.

### Kiến trúc

```
anh that x (1022 letterbox)
   │
   ▼  VAE encoder                       <- bo nen CHON o Buoc 0, DONG CUNG tu day
latent z0 (128×128×4)
   │
   ▼  them nhieu toi buoc t = s·T       <- s = strength, chon luc suy luan
z_t
   │
   ▼  UNet khu nhieu + cross-attn text  <- CHI HOC KHOI NAY (LoRA)
z0'
   │
   ▼  VAE decoder                       <- bo nen CHON o Buoc 0, DONG CUNG tu day
anh sinh
```

⚠ **ENCODER chốt xong trước khi huấn luyện UNet.** Đổi encoder sau Bước 2 là đổi không gian latent
dưới chân UNet ⇒ huấn luyện lại từ đầu. Đó là lý do `0a`/`0c` đứng trước.
**Decoder thì KHÔNG** — UNet không bao giờ nhìn thấy `x`, nên thay decoder lúc nào cũng hợp lệ, kể
cả sau Bước 5 (xem Bước 0).

**Text encoder đông cứng.** Mục tiêu khuếch tán chuẩn trên 258.871 ảnh train.

⚠ **Lấy mẫu `t` lệch về giữa.** Không bao giờ suy luận từ nhiễu thuần nên không cần UNet giỏi ở
`t` gần `T`.

**Kiểm mắt thường (~15 phút):** 20 ảnh ở `s ∈ {0,3 · 0,5 · 0,7}`. `s = 0,5` đã phá giải phẫu thì
dừng, không tiêu tiếp vào Bước 3.

### Benchmark chi phí (~10 phút, chạy TRƯỚC)

Mọi con số giờ trong file này là **ước tính chưa đo**. Trước khi cam kết Đợt 1:

```
train  :  hen gio 200 buoc train that  ->  ngoai suy ra tong
sinh   :  hen gio 100 lan img2img      ->  chot so buoc khu nhieu, batch size, do phan giai truoc
```

Time-box `10 phút`. Đủ để lập ngân sách, **không biến thành một nhánh nghiên cứu**.

**Thang mm/px** theo letterbox thật: `512 → 0,822 mm/px`; `1024 → 0,411 mm/px` (ô latent `3,29mm`).
⚠ `68b` ghi `0.342 mm/px @1022` là thang **crop** nhưng nạp shard **letterbox** — đúng là `0.411`;
hệ quả stride-4 là `1,64mm` chứ không phải `1,37mm`.

---

## Bước 3 · Cổng chuyển giao — **pilot ~1h + đủ ~4h**

⚠ **Thiết kế đầu tiên của cổng này HỎNG.** Bản cũ huấn luyện đầu đọc *chỉ trên ảnh sinh* với cách
gán **mask**: nhãn `j` chỉ nhận loss từ ảnh sinh của chính `j`, mọi ảnh đó đều `y_j = 1` ⇒ **không
mẫu nào đóng góp `y_j = 0`**. Bộ tối ưu hạ loss bằng cách đẩy `p_j → 1` cho mọi ảnh; `AP` thấp mà
không nói gì về bộ sinh. Nhánh one-hot không suy biến nhưng bể âm sai phân bố.

### Bốn nhánh, bể âm THẬT dùng chung

Với **mỗi nhãn `j`** và mỗi cấu hình `(nhánh sinh, strength)`:

```
be am   :  R anh THAT tu train, y_j = 0    <- CO DINH, dung chung cho moi nhanh cua nhan j

AP1     :  k anh THAT duong voi j
AP2     :  k anh SINH tu DUNG k anh goc do
AP3     :  k THAT + k SINH                      (2k duong)
APdup   :  k THAT + ban LAP cua chinh k anh do  (2k duong)   <- khong can trich dac trung moi
```

Cùng backbone đông cứng (đặc trưng tính **một lần**), cùng bể âm, cùng ngân sách. Đầu dò **nhị
phân** cho riêng `j` — vài giây mỗi nhãn. Chấm `AP(j)` trên **val-chọn**.

**Vì sao cần `APdup`.** Tôi từng cho rằng nhân đôi mẫu dương không đổi `AP` của đầu dò tuyến tính.
**Sai.** Kết quả "đổi trọng số lớp chỉ dịch intercept" đúng dưới ba điều kiện — mô hình đúng đặc
tả, không phạt, mẫu lớn — và ta vi phạm cả ba. Cơ chế rõ nhất là phạt `L2`: số hạng phạt `λ‖w‖²`
**cố định** còn số hạng dữ liệu **tăng theo tổng trọng số**, nên nhân đôi mẫu dương làm giảm mức
phạt *tương đối* lên các hướng do mẫu dương chi phối ⇒ `w` **đổi hướng**, không chỉ đổi `b`.

Nên phép so đúng là:

```
Delta_k  =  AP3_k  −  APdup_k        (KHONG phai AP3 − AP1)
```

Nó kiểm soát luôn chuẩn hoá loss, mức phạt, và số mẫu hiệu dụng. `APdup` còn **hiệu chuẩn chính
cái thước**: `|APdup − AP1|` lớn nghĩa là đầu dò không ổn định và `Δ` chỉ là nhiễu.

### Cách đọc — yếu, và đã gắn hành động

| Đọc được | Kết luận cho phép |
|---|---|
| `AP2 < AP1` | ảnh sinh chuyển giao **kém hơn** ảnh thật *trong phép đo này* — có thể do khoảng cách miền **hoặc** nhãn sai, chưa xác định |
| `AP3 − APdup ≤ δ` | **chưa thấy** lợi ích bổ sung ở cấu hình đang thử — không chứng minh ảnh sinh là bản sao |
| `AP3 − APdup > δ`, khớp trên tập xác nhận | **có bằng chứng** ảnh sinh bổ sung thông tin hữu ích cho đầu dò này |

Hành động tương ứng ở **Bảng hành động** đầu file. Cách đọc yếu + hành động cứng.

### Tập xác nhận

Val (`39.293` ảnh) tách **theo bệnh nhân** làm hai nửa: **val-chọn** để quét `31 × 2 × 2` cấu
hình, **val-xác-nhận** chạy **một lần** trên cấu hình đã chốt. Với `124` ô quét trên cùng một tập,
một phần mức tăng chắc chắn đến từ việc thử nhiều lựa chọn. Tách cục bộ cho Bước 3, không đụng
cách chia val ở các hướng khác.

### Pilot trước (Tuấn chốt: pilot + xác nhận trong ngân sách cố định)

**6 nhãn** trải phổ (2 medium lớn · 2 medium cấu trúc mảnh · 2 tail), đủ bốn thước đo, `~45 phút`.
Pilot **không** quyết định số phận của hướng — nó **hiệu chuẩn thước** và chốt `δ`:

```
sigma on dinh, |APdup − AP1| nho  ->  chot delta, chay du 31 nhan
2·sigma > 0.05                    ->  thuoc hong: sua dau do, hoac dung. KHONG nang delta.
```

### Sinh gì

`31 nhãn × 2 nhánh × 2 strength {0,4 · 0,6} × k=100` ≈ **12.400 ảnh, ~3,5h** (ước, đo ở Bước 2).
Nhãn có dưới 100 ca dương thì `k = n_dương`. `AP1`/`APdup` không cần sinh gì.

### Hai phép đo kèm, chạy cho CẢ HAI nhánh

**Trôi nhãn 40 chiều.** Chấm mô hình hiện tại trên ảnh gốc và ảnh sinh, so toàn bộ 40 xác suất.
Đây là thứ biến "P-giữ" từ tên gọi thành phép đo, và là thứ duy nhất nói được gì về 39 nhãn mà đầu
dò nhị phân không chạm tới.

```
P-giu : troi lon  ->  "ke thua nguyen nhan goc" KHONG hop le  ->  phai gan lai nhan truoc khi dung
P-xoa : p_head tut manh  ->  xoa co tac dung        p_head khong doi  ->  P-xoa chi la P-giu ton gio
```

⚠ Dùng chính mô hình của ta để kiểm ít vòng vo hơn ở **nhãn head** (nhóm ta dự đoán tốt nhất,
`0.5690`) so với nhãn hiếm. Vẫn là **bằng chứng một phần**, không chứng minh 39 nhãn kia đều âm.

**Phổ công suất — ba chiều, cùng ảnh gốc, cùng vùng, cùng tiền xử lý.**

```
that   ⟷  qua VAE (encode-decode)  ⟷  qua diffusion
```

Hai chiều thì không tách được lỗi bộ nén khỏi lỗi bộ khử nhiễu. ⚠ Phổ toàn ảnh đơn lẻ **không**
tách được sai khác do tổn thương khỏi sai khác do nhiễu/phơi sáng/letterbox — nên phải so trên
cùng vùng giải phẫu. Và đầu dò thật ⟷ sinh học tốt chỉ chứng minh **đặc điểm phân biệt tồn tại
trong biểu diễn**; chưa chứng minh mô hình bệnh đang dựa vào nó.

### Rủi ro đã ghi

⚠ **Cửa sổ `strength` có thể rỗng.** Thấp quá ⇒ ảnh sinh gần như bản sao (`AP3 ≈ APdup`). Cao quá ⇒
phá giải phẫu (`AP2 << AP1`). Quét hai giá trị là để đo cửa sổ này.

⚠ **Learnability gap.** `arXiv:2605.17087` chạy trên **MIMIC-CXR long-tail 19 nhãn**, trích dẫn
thẳng CXR-LT 2024:

```
tai tao KHONG mat gi        : phan loai qua vong VAE .261  vs  anh that .263   (p = 0,375)
NHUNG hoc TU latent thi mat : 6-13 diem tren MIMIC-LT, p = 9,5e-7 tren 20 cap AE×data
tinh chinh AE trong mien    : PSNR 29,9 -> 38,9 dB, phan loai .370 -> .374
```

Nút thắt của họ là **cấu trúc** không gian latent, không phải độ trung thực bộ nén. Đây là lý do
không tiêu thêm giờ vào bộ nén sau Bước 0. ⚠ Nhưng nếu vòng này hỏng, bài đó là **một giả thuyết
cần kiểm**, không phải lời giải thích sẵn có để dán vào.

---

## Bước 4 · Đường cong quy mô — **~10–20h, Đợt 2**

Chỉ cho nhãn qua Bảng hành động. Sinh `{250, 1000}` ảnh/nhãn, dựng đường cong trên val-chọn.

⚠ **Trần của P→P**: tăng đa dạng **hình thái**, không tăng đa dạng **giải phẫu**. Nhãn hiếm nhất có
`129` ca dương ⇒ số bệnh nhân gốc trần ở `129`.

⚠ **Bão hoà sớm ở tail KHÔNG chứng minh chạm trần giải phẫu** — cũng có thể là đầu dò hết dung
lượng, `k` quá nhỏ, hay bộ sinh chỉ tạo biến thể nông. Tách bằng cách so đường cong tail với đường
cong medium (`2.803`–`15.371` gốc): medium cũng bão hoà ở cùng số ảnh thì nguyên nhân **không phải**
số bệnh nhân gốc.

⚠ **Đường cong có thể ĐI XUỐNG, không chỉ bão hoà.** FreeTumor (`Nat Commun 16:11053`) đo được:
khi mở rộng quy mô cho `SynTumor` — một bộ sinh không tổng quát hoá — Dice trên `LiTS` **tụt từ
`60,2` xuống `52,8`**. Một bộ sinh kém thì **càng nhiều càng hại**, không phải càng nhiều càng
trung tính. Vì vậy Bước 4 phải đọc dấu của độ dốc, không chỉ đọc chỗ bão hoà; xuống dốc là tín
hiệu **dừng**, không phải tín hiệu sinh thêm.

---

## Bước 5 · Huấn luyện lại đường ống — **3 × 13h, Đợt 2**

**EHR và phim trước: là hệ quả của nhánh và của phép đo trôi nhãn, không phải quyết định riêng.**

| Nhánh | Nhãn | EHR + phim trước |
|---|---|---|
| **P-giữ**, trôi nhỏ | kế thừa nhãn gốc | kế thừa |
| **P-giữ**, trôi lớn | gán lại theo đo, hoặc loại | kế thừa các trường không liên quan nhãn bị trôi |
| **P-xoá** | ẩn số đã đo ở Bước 3 | rà `lab_*` / `vital_*` / `como_*` liên quan `h`, hoặc **không kế thừa** |

⚠ **Mask loss của `h` không sửa được mâu thuẫn ở tầng đầu vào**, nó chỉ bỏ phần giám sát đó đi.

**Quản lý xuất xứ** (`§12.2`, `§17` của tài liệu tham chiếu). Mỗi ảnh sinh ghi: ảnh gốc, bệnh nhân
gốc, nhãn đích, prompt, strength, CFG, seed, checkpoint. Dùng để **không dồn nhiều bản sinh cùng
một ảnh gốc vào một lô**, và gỡ hàng loạt khi một prompt hỏng.

⚠ **Sinh lại mỗi epoch, KHÔNG dùng bộ ảnh sinh cố định** (FreeTumor, `Nat Commun 16:11053`):
*"we regenerate tumor masks for each epoch without a synthetic cache, since **reusing the same
synthetic cache could leak information and inflate accuracy**"*. Đổi seed mỗi epoch. Ngoại tuyến
chỉ dùng cho **cổng Bước 3** (cần tái lập chính xác); huấn luyện thật thì trực tuyến.

Ba nhánh, **bắt buộc đủ ba**:

```
a) nen — cau hinh hien tai
b) OVERSAMPLE anh THAT cua nhan thieu so, CUNG so luong hieu dung
c) anh TONG HOP
```

**Không có `(b)` thì kết quả dương không đọc được.** `(c) ≈ (b)` ⇒ diffusion không thêm gì ngoài
cân bằng lớp, mà cân bằng thì miễn phí. Bài học `06b` (`H4z`/`H4`). `APdup` ở Bước 3 là phiên bản
đầu-dò của đối chứng này; **không thay thế nó** — ở đây backbone được huấn luyện nên động lực
gradient khác hẳn.

**Cổng cuối:** `(c) − (b) ≥ +0.005` macro trên val, **và** đường cong Bước 4 còn đang lên.

---

## Đề xuất vòng sau — KHÔNG phải nhánh cứu hộ

Ghi lại để không mất, và để **không tự động được cấp ngân sách** khi vòng 1 hỏng. Mỗi mục là một
đề xuất riêng, phải đứng được bằng lập luận của chính nó.

| Mục | Nội dung | Điều kiện đáng xét lại |
|---|---|---|
| **Cắt vùng** | Cắt ô `1024×1024` từ bản `2044` → `1,64mm/ô`, cùng hình dạng latent, cùng chi phí. "Cách rẻ để có độ phân giải là nhìn ít đi." Phải giải quyết: chọn vị trí cắt, ghép biên, resize cuối | 0a cho thấy mất mát tập trung ở nhãn cấu trúc mảnh |
| **Tự tinh chỉnh decoder** | `~3h`, encoder `no_grad` tuyệt đối, giữ LPIPS + phân biệt theo mảng (bỏ đối kháng thì ảnh ra mờ). **Nằm ngoài đường găng** — chạy được cả sau Bước 5 mà không phải huấn luyện lại UNet | ⚠ `arXiv:2605.17087`: PSNR `29,9→38,9 dB` mà phân loại chỉ `.370→.374`. Xét lại nếu `0a` cho thấy mất mát khu trú ở vài nhãn cụ thể |
| **Latent lớn hơn** | MedVAE `f=16` (phá hình dạng) hoặc đầu vào 2048 (giữ hình dạng, một phần tiên nghiệm). Cả hai `~80h`: SDXL chạy self-attention ở hai tầng dưới, chi phí theo **bình phương** số ô — `128→256` là gấp bốn ô, gấp **mười sáu** ở tầng nặng nhất | chẩn đoán `f=16` ở 0a cứu được nhiều nhãn |
| **Report + N→P** | Ngân hàng prompt từ `227.835` file; N→P phá trần đa dạng giải phẫu. ⚠ N→P **không kế thừa** các trường EHR liên quan bệnh vừa thêm (các trường khác thì xét riêng). Ta đã **đóng** hướng gắn lại nhãn bằng LLM — tách report thành từng finding là việc cùng họ, `10–20h` người + audit tay | Bước 5 vòng 1 qua cổng |
| **Sửa ảnh CÓ MẶT NẠ** | Thay SDEdit toàn ảnh bằng **inpainting có mặt nạ giải phẫu**. Pixel ngoài mặt nạ giữ nguyên bit ⇒ trôi nhãn và trôi giải phẫu thành **bảo đảm kiến trúc**, không phải phép đo; cắt quanh mặt nạ ở độ phân giải gốc thì được luôn `1,64mm/ô`. Nguồn: FreeTumor pt.(1), `x̂ = (1−M)⊗x + M⊗[x − tanh(G(x))⊗g(x)]`. Phải giải quyết: dựng mặt nạ giải phẫu cho từng nhãn, ghép biên | Tuấn đã chọn **giữ kiến trúc cũ** vòng 1 (15/09). Xét lại nếu trôi nhãn 40 chiều ở Bước 3 lớn |
| **N→P làm nhánh chính** | Toàn bộ lợi ích của FreeTumor đến từ vẽ tổn thương lên **cơ quan lành**, và tăng theo **quy mô ngữ cảnh** chứ không theo chất lượng ảnh. Nhãn hiếm nhất của ta có `258.742` nguồn N→P so với `129` nguồn P→P | Cùng điều kiện trên; và chỉ có nghĩa **sau** khi có mặt nạ, vì không mặt nạ thì nhãn của N→P không kiểm được |
| **PadChest** | Số liệu đã đo, giữ ở cuối file | — |

---

## Nghiệm thu

| Bước | Chạy gì | Đọc gì |
|---|---|---|
| 0a | `70a` C0–C3 | AP val sau vòng nén, **theo từng nhãn**, ba ứng viên |
| 0c | `70a` C4 | `AP1/AP2/AP3/APdup` trên 3–4 nhãn ở mỗi latent + 20 ảnh mắt thường |
| 1 | `70a` C5–C8 | số ảnh lọc được; bảng `p(y_head\|y_j)`; thứ tự ưu tiên lối tắt head |
| 2 | `70b` C0–C3 | benchmark giờ; mắt thường 20 ảnh ở `s ∈ {0,3 · 0,5 · 0,7}` |
| 3-pilot | `70b` C4 | `σ`, `\|APdup − AP1\|` → **chốt `δ`** |
| 3 | `70b` C5–C6 | bảng `31 × 2 nhánh × 2 strength`, mỗi ô `AP1/AP2/AP3/APdup` trên val-chọn |
| 3b | `70b` C7 | trôi nhãn 40 chiều (cả hai nhánh); phổ ba chiều cùng vùng |
| 3c | `70b` C8 | **val-xác-nhận, chạy một lần** trên cấu hình đã chốt |
| 4 | `70c` C0 | đường cong `{250, 1000}`, tail **và** medium |
| 5 | `70c` C1–C3 | `(a)(b)(c)` trên val, rồi gộp vào lượt đọc test |

**Đối chứng người** — theo quy trình của FreeTumor (`Nat Commun 16:11053`), gọn lại cho ta:

```
tron 50/50 anh that va anh sinh, thu tu ngau nhien, KHONG noi ty le
bao truoc nhan dich va vi tri can xem       ->  bao cao do NHAY, do DAC HIEU, do CHINH XAC
```

Mốc của họ để so: 13 bác sĩ đạt **`51,1%` độ nhạy / `60,8%` độ chính xác** — gần mức đoán mò; bác
sĩ trẻ chỉ `41,5%`. ⚠ Đây là **cảnh báo sớm**, không phải bằng chứng: phân biệt được dễ chỉ nói có
một dấu hiệu phân biệt tồn tại, chưa nói mô hình bệnh có bám vào nó không.

⚠ **Không dùng FID hay bất kỳ chỉ số chất lượng ảnh nào làm cổng.** FreeTumor: *"many synthetic
tumors, despite with low FIDs, still present with unrealistic characteristics in the views of
radiologists… rendering conventional fidelity metrics unreliable."*

**Bốn kết quả này là chỉ báo định hướng, KHÔNG phải lời giải thích mặc định:**

```
ty so loi tat head        -> ca hai chieu deu chua chung minh gi
bao hoa som               -> chua chung minh cham tran giai phau
AP2 thap                  -> chua tach duoc khoang cach mien khoi nhan sai
dau do that/sinh hoc tot  -> chua chung minh classifier benh dua vao dau hieu do
```

**Mọi lựa chọn chốt trên val.** Đọc test một thể cho cả bảng cùng các hướng khác.

---

## Chi phí và điểm thoát

⚠ Mọi con số dưới đây là **ước tính chưa đo**. Benchmark `10 phút` ở Bước 2 chốt lại trước khi cam
kết Đợt 1.

```
DOT 1                                                   <- ngan sach co dinh
  Buoc 0a  ~3h   tai tao
  Buoc 0c  ~2h   thich nghi     <- DUNG neu ca hai ung vien hong
  Buoc 1   ~4h   CPU, song song
  Buoc 2  ~10h                  <- DUNG o phut 15 neu s=0,5 pha giai phau
  Buoc 3   ~1h pilot -> chot delta
          +~4h du 31 nhan + xac nhan
                 ------
                  ~20h  ->  BANG HANH DONG quyet dinh mo Dot 2 hay dung

DOT 2   (chi mo neu Dot 1 cho phep)
  Buoc 4  ~10-20h
  Buoc 5  ~39h
```

**Trần lý thuyết:** đóng trọn medium `+0.0107` và tail `+0.0047` là `+0.0154`, đúng bằng khoảng
cách tới vô địch. Thực tế sẽ thấp hơn nhiều.

---

## Đã đối chiếu với `plan_cxr_lt_pathology_guided_diffusion.md` (15/09)

Tài liệu đó viết trên nền dự án **giai đoạn 2** (`0.4336` dev / tier `0.6136–0.2555–0.1227` là
thang cũ, đầu đọc đông cứng); Phase 0 của nó ta đã xong từ lâu.

**Lấy vào:** sửa ảnh thật thay vì sinh từ nhiễu (quyết định lớn nhất); quản lý xuất xứ và lấy mẫu
theo lô (`§12.2`, `§17`); phân tích thất bại ba cấp (`§15`).

**Không lấy:** cổng "frozen classifier chấm ảnh sinh" (`§11.1` — vòng luẩn quẩn); `200 candidate`
mỗi ảnh gốc (hàng triệu ảnh; chính tài liệu cũng đã lùi); ngân hàng prompt làm trước generator.

**Bổ sung thứ tài liệu thiếu:** chọn bộ nén bằng hai phép kiểm (Bước 0 — tài liệu `§6.1` phân biệt
"generative latent" với "semantic feature" nhưng **không hỏi latent của LAION có giữ nổi chi tiết
X-quang không**); cổng bốn nhánh có bể âm thật và `APdup` (Bước 3); đối chứng `(b)` (Bước 5);
caption đủ nhãn (Bước 2).

---

## PadChest — HOÃN, số liệu đã đo (15/09)

`CXRLT_2026_training.csv` (142.928 dòng, one-hot 30 cột) nằm ở **`CXR-LT 2026/`**, không phải trong
`PadChest_Full/` — lý do `54` đếm ra 0. Khớp tên phải chuẩn hoá hoa thường + bảng đồng nghĩa;
**19/30 lớp** ánh xạ được.

| PadChest → CXR-LT | Pad | Ta | Tỉ lệ | Tier |
|---|---|---|---|---|
| Kyphosis | 3.576 | 778 | 4,60× | tail |
| aortic elongation → Tortuous Aorta | 9.822 | 3.336 | 2,94× | **medium yếu** |
| azygos lobe → Azygos Lobe | 404 | 199 | 2,03× | tail |
| pleural thickening → Pleural Thickening | 4.397 | 3.272 | 1,34× | **medium yếu** |
| Nodule | 9.159 | 7.531 | 1,22× | medium |
| Hernia · Calcif. Aorta · Emphysema · Fracture | — | — | 0,34–0,48× | medium |
| 5 nhãn head | — | — | 0,12–1,41× | head |
| **Hydropneumothorax 45 · Pneumoperitoneum 61** | — | — | **0,07–0,12×** | tail — **chết** |

**Kết luận:** PadChest là đòn bẩy **medium**, không phải tail — trái hẳn giả định ban đầu.

**Bốn điều phải xử khi quay lại:**
1. `aortic elongation → Tortuous Aorta` và `aortic atheromatosis → Calcification of the Aorta` là
   ánh xạ **ngữ nghĩa**, chưa kiểm.
2. Ảnh PadChest chỉ có 19/40 nhãn; 21 nhãn còn lại **phải mask**, không được ghi `0`.
3. Chỉ dùng cho **RAD-DINO**; `convnextv2_padchest.pt` do ta pretrain trên chính PadChest ở `41c`.
   PadChest **không có EHR và phim trước**.
4. `padchest_518_ready/` chỉ có bản 518; muốn dùng ở 1022 phải đọc lại 964 GB PNG gốc (~3,2h I/O).

**Giấy phép (Tuấn, 15/09):** dùng nhãn CXR-LT 2026, **nêu rõ trong bài** nguồn và ngày; ảnh
PadChest không chứa bệnh nhân MIMIC nên không rò rỉ test 2024.

---

## Việc dọn kèm theo

- Sửa chú thích `mm/px` trong `68b` C0: `0.342` → `0.411` (letterbox), stride-4 `1.37` → `1.64`.
- `docs/BAO_CAO_TIEN_DO.tex` còn `0.2625` / `0.0185` → `0.2656` / `0.0154`.
- `PLAN_VUOT_0281.md` §4.5 đánh dấu **đã thay thế** (không xoá, CLAUDE.md §8).
- `54_padchest_inventory.ipynb`: sửa `PZ` trỏ lên thư mục cha, thêm chuẩn hoá tên + đồng nghĩa.
- `mimic-cxr-reports/` untracked — thêm vào `.gitignore` (227.835 file).
- `Event.md`: bỏ text→image; chọn img2img P→P hai nhánh; nhận MedVAE và giữ lệnh cấm RoentGen kèm
  ba trích dẫn; **cổng Bước 3 phiên bản đầu suy biến vì thiếu mẫu âm — ghi như một sai lầm thiết
  kế đã sửa, không xoá**.
- `PLAN_B_rad_dino.md:7` nói RAD-DINO pretrain "bao gồm MIMIC" nhưng không có số hay trích dẫn.
  Bổ sung — bài viết sẽ phải nêu rõ cả RAD-DINO lẫn MedVAE.
- `Event.md`: ghi việc đối chiếu FreeTumor (`Nat Commun 16:11053`, CT chứ không phải X-quang) —
  lấy bốn điểm phương pháp (sinh lại mỗi epoch, bác FID, quy trình đối chứng người, cảnh báo đường
  cong đi xuống); **Tuấn giữ kiến trúc SDEdit toàn ảnh và thứ tự P→P trước** (15/09), phương án mặt
  nạ và N→P ghi vào đề xuất vòng sau.
