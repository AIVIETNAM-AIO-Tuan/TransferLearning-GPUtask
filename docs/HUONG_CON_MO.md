# Hướng còn mở — cập nhật 2026-09-15

> Thay cho `CHECKLIST_STAGE5_OPEN.md` (06/09). File cũ giữ lại làm lịch sử, đừng đọc để quyết định.
> Lớp quyết định nằm ở `Event.md`; file này là danh sách việc.

## 0 · Đang ở đâu

```
TEST (78,946 anh, macro-mAP 40 nhan, thang tier CHINH THUC 9/15/16)
                         macro     head   medium     tail
  ta (D, 15/09)         0.2656   0.5690   0.2344   0.1243
  vo dich (doi A)       0.2810   0.5670   0.2630   0.1360
  chenh                -0.0154   +0.0020  -0.0286  -0.0117
```

**Cấu hình triển khai:** đường ống `64/r2a` (20 file) ⊕ `cnn518` `64b/r2a` (16) ⊕ `cnn1022`
`68b/r1022_E8` (16), đặc trưng XGB đã khôi phục cho test, trộn theo thang tier **cũ**
(13/16/11) với `w1 = 0.15 / w2 = 0.35`, tail `0`.

**Khoảng cách còn lại phân bổ ở đâu:**

| Tier | n nhãn | Đóng góp vào `0.0154` | Phần |
|---|---|---|---|
| head | 9 | `−0.0005` | ta đang **dẫn** |
| **medium** | 15 | **`+0.0107`** | **70%** |
| tail | 16 | `+0.0047` | 30% |

Mọi ưu tiên dưới đây xếp theo việc nó có chạm vào **medium** hay không.

---

## 1 · Nhắm vào MEDIUM (70% khoảng cách)

### 1.1 · Hàm mất mát có trọng số theo nhãn — **ưu tiên cao nhất**

Thước đo cho mỗi nhãn trọng số bằng nhau, hàm mất mát thì không: một bộ tham số bất đối
xứng duy nhất đang dùng chung cho cả 40 nhãn. Và **không sửa được sau khi huấn luyện** —
AP chỉ phụ thuộc thứ hạng trong từng nhãn, nên mọi hiệu chỉnh đơn điệu theo nhãn đều là
phép không, đúng như `67` đo ra `+0.0000`.

**Đã thử một lần và thoái hoá.** `nb35`:

```
asl40   ASL gn=4 gp=0      mAP 0.2083   tail 0.0614
wasl    WASL alpha=.50     mAP 0.2079   tail 0.0605
```

Ghi chú trong mã: `wasl bo (cham san 15/40 -> thoai hoa ve ASL)`. Trọng số bị kẹp
`[0.5, 3.0]` và ép trung bình 1 ⇒ **15/40 nhãn chạm sàn, 1 chạm trần**. Phân bố từ 26.000
xuống 129 ca dương không nhét vừa khoảng đó. Thêm nữa `nb35` chỉ là **sàng trên embedding
đông cứng với đầu đọc tuyến tính**, không phải đường ống thật.

**Việc:** bỏ kẹp, dùng `w_k ∝ (1 / n_pos_k)^β`, quét `β` trên val. `β = 0` là ASL hiện tại.
Chạy trên đường ống thật. `~13h/seed`.

### 1.2 · Hạ trọng số loss cho 6 nhãn bẩn

`67` đã xác định nhóm này từ tập vàng 406 ảnh của paper (Table 8):

```
Lung Lesion 0.161 · Pneumonia 0.191 · Infiltration 0.261
Mass 0.513 · Pneumothorax 0.563 · Enlarged Cardiomediastinum 0.583
(trung binh toan bang 0.711)
```

`Infiltration` là **nhãn medium tệ nhất của ta** (`AP 0.0644`, prevalence 3.9%). Đã xác định
từ 08/09 nhưng **chưa bao giờ hiện thực**. Ghép được vào 1.1 thành một lượt chạy.

⚠ **Không gắn lại nhãn bằng LLM** — nhãn test bẩn cùng một kiểu, lỗi parser là hệ thống.

### 1.3 · Loss xấp xỉ AP (Smooth-AP)

Ta đang dùng loss **phân loại** để tối ưu một chỉ số **xếp hạng**. Chưa thử lần nào. `~13h`.

### 1.4 · Thành viên CNN lớn hơn

`convnextv2_nano` (~13 GMAC, IN-1k 82.0) → `tiny` (~24, 84.9) hoặc `base` (~82, 86.8).
Độ phân giải 1022 đã chứng minh có ăn ở mức ensemble; kiến trúc to hơn là trục còn lại
chưa đụng. `~8h`.

---

## 2 · Nhắm vào TAIL (30% khoảng cách)

### 2.1 · PadChest làm **nguồn nhãn**, không chỉ pretrain

Bệnh viện khác (Alicante vs Boston) ⇒ không chung bệnh nhân, không rò rỉ test.
`Removal.csv` của ban tổ chức đã loại sẵn dev/test.

```
Azygos Lobe          176 anh
Hydropneumothorax    549 anh
Pneumoperitoneum     454 anh
Kyphosis             669 anh
```

Nhãn hiếm nhất của ta có ~150 ca dương ⇒ thêm được **gấp 2–6 lần**. Trọng số
`convnextv2_padchest.pt` đã có và đang dùng làm khởi tạo, nhưng **chưa dùng PadChest như
nguồn nhãn bổ sung**. `~8h`.

### 2.2 · Chuyên gia TAIL kiểu RIDE

Khớp sẵn với bộ máy trộn theo tier. Tail hiện nhận trọng số `0` từ ensemble, tức 16 nhãn
không hưởng gì từ hai thành viên CNN. `~13h`.

### 2.3 · Sinh ảnh diffusion — **tự huấn luyện**

Nước đi của đội A (hạng 1). Mô hình sinh công khai như RoentGen đã fine-tune trên
MIMIC-CXR nên đã thấy ảnh test ⇒ phải tự huấn luyện: nền SD/LAION → fine-tune trên train
của ta + PadChest. `30–60h`, độ chắc thấp–vừa. Xếp sau 2.1 vì đó là ảnh thật.

---

## 3 · Không nhắm tier cụ thể — đòn bẩy chung

### 3.1 · K-fold, fine-tune backbone theo từng fold ⭐ lớn nhất

Mọi seed hiện tại **dùng chung một backbone 86M đông cứng**, chỉ khác 7.3M đầu đọc. Đó là
lý do thêm seed chỉ cho `+0.0008`. Chạy `38a` với seed/augment khác cho 86M tham số **khác
nhau thật** — loại đa dạng chưa từng đo, và đúng công thức vô địch. `35–58h`.

### 3.2 · Chạy lại `38a` trên 87% dữ liệu

Backbone hiện tại fine-tune trên `TF_` **cũ** (74%). `R2a` cho `+0.0022` chỉ ở tầng đầu đọc
7.3M; ở backbone 86M, vốn đói dữ liệu hơn, có thể nhiều hơn. `~6h + 13h`.

### 3.3 · TTA không lật

Xoay ±5°, crop 0.9–1.0. **Cấm hflip** — ràng buộc y tế. `~1h`, chưa bao giờ chạy.

### 3.4 · Vision-language

**8/9 đội đầu bảng** dùng ảnh + report. Ta dùng EHR thay. Chưa thử lần nào. Đây là khác
biệt lớn nhất còn lại so với nhóm dẫn đầu, và cũng là thứ đắt nhất để bắt kịp.

---

## 4 · Nhánh EHR / dữ liệu — rẻ, chưa ai chạy

### 4.1 · EHR mở rộng: thủ thuật, thở máy, kiểu nhập viện

`Hydropneumothorax` và `Pneumomediastinum` có **bội số EHR cao nhất nhóm tail**
(`4.2×`, `4.8×`) nhưng EHR của ta thiếu đúng loại đặc trưng:

```
DANG CO   lab_* · vital_* · micro_* · dac trung XGB
THIEU     proc_chest_tube_{24h,7d,ever} · proc_thoracentesis · proc_intubation
          vent_mech_dang_hoat_dong · vent_peep_max_24h (proxy barotrauma)
          proc_noi_soi_7d · adm_type · dx_copd_ever · dx_hen_ever
```

~15 cột, **cùng dạng** `lab_*`/`vital_*` ⇒ cắm thẳng vào `ColTokBag`, **không đổi kiến trúc**.
Bậc 1–2 (BigQuery + probe XGB) `~3.5h`, **không cần GPU**.

⚠ **Bắt buộc đối chứng:** dẫn lưu màng phổi → hydropneumothorax là **đường tắt**. Phải chạy
phép tráo ảnh sang bệnh nhân khác, giữ EHR. Nếu AP gần như không đổi thì mô hình đang đọc
EHR chứ không đọc phim, và phải báo cáo đúng vậy.

### 4.2 · Mấy thứ lặt vặt, CPU vài phút

- `EHRColTok` vẫn phát token cho cột **vắng mặt** (`T_j = b_j`). Kế hoạch gốc nói *"thiếu:
  KHÔNG sinh token"* — chưa làm, chưa đo. Đây là cách xử lý missingness triệt để nhất.
- **28,7% ảnh không có phim trước** — cần token *"không có tiền sử"*, chưa đo.
- **18 cột `lab_*_7d` trùng lặp 91%** — dọn rẻ hơn thêm đặc trưng xu hướng.
- `Q_CTX` vs `Q_CTX + como` — tách 11 cột khỏi khối 90.
- Nhánh dọc `D = [patch_cur ; patch_cur − patch_prior]` — thiết kế đã chốt, chưa dựng.

---

## 5 · Cho BÀI VIẾT, không cho mAP

### 5.1 · `nb40` — nhiệm vụ thứ hai

25 CCS phenotype (ICD→CCS) + tử vong nội viện 48h, tái dùng `f_cxr`. Mốc MedFuse:
`0.481` / `0.594` AUPRC; EHR ở đó đóng `+0.028`, còn ta chỉ `+0.004` trên CXR-LT.

Nó biến kết quả yếu của EHR ở CXR-LT từ *"hướng thất bại"* thành **"đã định vị được ranh
giới: EHR không giúp nhãn sinh từ report, giúp rõ nhãn lâm sàng"**. `~1 notebook`.

### 5.2 · Nợ tài liệu

- `REPORT_STAGE4_NB38A.md` chưa tồn tại. Viết phải kèm điều kiện *"ở ngân sách đầu đọc bằng nhau"*.
- **Limitations**: trần cấu trúc `9.46mm/patch` ở 518 (`1.37mm` ở 1022 với CNN), kèm trích
  Muthyala et al. — biến "tail thấp" từ thất bại thành **giới hạn có tài liệu**.
- Ghi rõ: **`Azygos Lobe` là biến thể giải phẫu bẩm sinh, không phải bệnh** — không nên bị
  chấm chung thang với nhãn bệnh lý.
- Ghi rõ: đánh giá **hồi cứu**, ta giữ nhãn test, lựa chọn làm trên val.
- `docs/BAO_CAO_TIEN_DO.tex` còn số cũ `0.2625` / khoảng cách `0.0185` — cập nhật lên
  `0.2656` / `0.0154`, và bảng "hướng còn mở" trong đó nay trỏ về file này.

---

## 6 · ĐÃ ĐÓNG — đừng làm lại

| Hướng | Bằng chứng |
|---|---|
| Đưa EHR vào sâu trong backbone | `56`: chênh lệch **âm**, ổn định qua seed |
| Mở khoá block 8–11 | `57`: `−0.007` đến `−0.010`, 2 seed đồng thuận |
| Huấn luyện lại riêng tầng phân loại (cRT) | `59`: tầng này **đã hội tụ** |
| Ghép đặc trưng đa tỉ lệ (`cnn_ms`) | `41f` trượt cổng; `62`: tầng 3 **gần rank-1** (hạng hiệu dụng 1.1/320) |
| Tăng độ phân giải chỉ ở **suy luận** | `51`: `−0.0028`, lệch phân phối |
| Thêm seed của **cùng** backbone đông cứng | `+0.0008` < sàn nhiễu `0.0007`. **Không** đóng 3.1 |
| Hiện tượng co cụm biểu diễn lớp | `62` C5 |
| Trọng số trộn phụ thuộc chất lượng nhãn (`w = f(r)`) | Tương quan giả do nhiễu dùng chung: `+0.598` cùng mẫu → `+0.136` `p=0.40` khác mẫu |
| `rho` làm biến điều kiện | Đo rất ổn định nhưng **vô thông tin** (`+0.017`) |
| Bốn cách trộn rẻ, 0 GPU | `67`: tất cả `+0.0000` |
| Cắt vùng nghi ngờ rồi phóng to | `66`: hộp bao chiếm **79%** ảnh ⇒ phóng đại thực `1.06×` |
| Định vị vùng mô hình đang nhìn | `65`: giữ 5% vùng vẫn đạt 96–99% ⇒ mô hình **không** nhìn nhầm chỗ |
| **ViT** @1022 làm đường ống chính | `68`: macro `−0.0055`, tail `−0.0138` |
| Gắn lại nhãn bằng mô hình ngôn ngữ | `67`: nhãn test bẩn **cùng kiểu**, lỗi parser hệ thống |
| Multi-view `w_l` theo bệnh | `17`: `+0.00017`, CI chứa 0 |
| FPN đa tỉ lệ trên ViT thuần | Không có phân cấp stride để bám |

**Deferred reweighting (DRW)**: chưa chạy trên CXR-LT, và **không xếp ưu tiên**. Kết quả ở
giai đoạn cũ 13 nhãn: macro `−0.0076`, tail `+0.0041` — nó **phân bổ lại** gradient chứ
không tạo thêm năng lực. Với ta thì đổi sai chiều, vì medium giữ 70% khoảng cách còn tail 30%.

---

## 7 · Thứ tự đề xuất

```
1. 4.1 bac 1-2   EHR mo rong, probe XGB      ~3.5h   0 GPU, chay song song moi thu khac
2. 1.1 + 1.2     loss co trong so + ha nhan ban   ~13h/seed   nham THANG vao medium
3. 2.1           PadChest lam nguon nhan     ~8h     don bay tail lon nhat
4. 3.3           TTA khong lat               ~1h     re nhat trong danh sach
5. 3.1           K-fold + backbone tung fold ~35-58h don bay macro lon nhat
6. 5.1           nb40                        ~1 nb   cho bai viet
```

Chốt hết trên val, rồi **đọc test một thể** cho cả bảng.
