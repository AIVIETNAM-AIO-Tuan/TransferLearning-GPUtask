# CHECKLIST — hướng còn mở, hướng đã đóng

*Lập 2026-09-06, sau khi `R2a` (`64`/`64b`) xong và chạm test lần 2.*
*Nguồn: JOURNAL.md, PLAN_STAGE4/STAGE5, và các phiên thảo luận chưa kịp ghi.*

---

## 0 · Đang ở đâu

```
                  VAL      TEST
duong ong       0.2663   0.2583
+ cnn (w tier)  0.2709   0.2625     d test +0.0042  CI95 [+0.0033, +0.0051]  LOAI 0
  head          0.4826   0.4827
  medium        0.2032   0.1977
  tail          0.1193   0.0966

vo dich CXR-LT 2024   0.281      cach 0.0185
tran nhieu nhan       0.6528     ta o 40% cua no  -> con RAT nhieu du dia, chi la dat
```

**Cấu hình triển khai:** ensemble `20` file `t1_prior_both` (`64/r2a`) ⊕ `16` file `cnn_full`
(`64b/r2a`), trộn theo tier `head 0.40 | medium 0.40 | tail 0.00` (cổng `0.85`).

### Chuyển giao val → test, đo được (dùng cái này để quyết tin val tới đâu)

```
head    +0.0001   hoan hao
medium  -0.0055   -2.7%
tail    -0.0227   -19%     <- gan nhu TOAN BO khoang cach nam o day
```

⇒ **Chọn trên val ở head + medium (29/40 nhãn, 72.5% macro) thì tin được. Ở tail thì không.**
Con số tail thật của ta là **`0.0966`**, không phải `0.1193`.

---

## 1 · Kỷ luật hiện hành

```
LAT CHON  da gop vao train (R2a) — khong con ton tai
VAL       gio la TAP CHON hop le (doc lap voi test; thien lech cua no khong anh huong vai tro nay)
TEST      da cham 2 lan (2026-08-21 x1, 2026-09-06 x1), deu ghi _test_touches_nb61.jsonl
```

**Điều duy nhất còn cấm: chọn TRÊN test.** Cách giữ sạch — chốt danh sách thí nghiệm trước, chọn
hết trên val, đọc test **một lượt** và báo cáo **tất cả**, kể cả cái thua.

**Sàn nhiễu:** `0.0007` (lặp lại thật, đo từ bản sao `41e`) · `0.005` (mức công bố, khoảng cách
hạng 2–4 của giải).

**Quy tắc lưu file (mua bằng việc mất trắng trọng số `53/r3`):**
> Tên file **duy nhất** thì sống. **Ghi đè** cùng tên qua Drive FUSE thì chỉ lần ghi **đầu tiên**
> xuống được đĩa. Áp cho `resume.pt`, `epoch_log.json`, mọi artifact ghi lại mỗi epoch.

**Quy tắc báo cáo:** số đối chiếu phải **đọc từ log/file**, không hardcode từ trí nhớ. Hai hằng số
gõ tay đã làm cả hai lượt `R2a` trông như sập trong khi thực tế đều tăng.

---

## 2 · MỞ — rẻ, 0 GPU, dự đoán đã có sẵn

Tất cả chạy trong `nb61`, vài phút mỗi cái. Chọn trên val, gộp vào một lượt đọc test.

| # | Việc | Kỳ vọng | Ghi chú |
|---|---|---|---|
| 2.1 | **Trung bình HẠNG** thay vì trung bình xác suất | `+0.000 ~ +0.003` | AP là chỉ số **xếp hạng**; các thành viên huấn luyện riêng nên calibration lệch. `PLAN_STAGE4 §5` đã ghi *"dùng trung bình HẠNG song song"* — chưa bao giờ thử. |
| 2.2 | **Quét lại `w` theo tier trên val** | ? | `0.40/0.40/0.00` khoá từ khi val còn là thước đo. Giờ val là tập chọn ⇒ quét được thật, kể cả `w_tail`. |
| 2.3 | **Trọng số theo EPOCH** trong ensemble | `+0.001 ~ +0.003` | Hiện `p_ep1` (macro 0.245) cân **ngang** `p_ep8` (0.258). Cân theo chất lượng là miễn phí. |
| 2.4 | **`cnn_ms` (`41f`) làm thành viên thứ 3, `w` theo tier** | ? | 3 chiều từng bị loại (`+0.0003`, CI chứa 0) nhưng đó là `w` **CHUNG** chọn trên lát chọn. Per-tier trên val là phép khác. **Xem mục 4.1.** |

---

## 3 · MỞ — một lượt suy luận, không huấn luyện

| # | Việc | Chi phí | Kỳ vọng | Cổng |
|---|---|---|---|---|
| 3.1 | **`nb66`** — cascade/độ phân giải, 3 nhánh A/B/C | ~1h + stage 1022 | ? | `(A+B)−(A+C) < 0.0007` ⇒ **đóng hẳn hướng độ phân giải** |
| 3.2 | **TTA không lật** (xoay ±5°, crop 0.9–1.0) | ~1h | `+0.002 ~ +0.005` | Ràng buộc Tuấn: **cấm hflip** |
| 3.3 | **Che top-k% patch lúc suy luận** | ~1h | `+0.0008` macro | `nb65` đo `k=20%` cho tail `1.029`. **Số đó từ checkpoint epoch 1** — phải đo lại trên `64/r2a`. `k` chọn trên val (hợp lệ). |
| 3.4 | **Chạy lại `nb65`** trên trọng số `64/r2a` | ~1h | — | Kết quả hiện tại toàn bộ là **sơ bộ** |

---

## 4 · MỞ — nhắm thẳng vào TAIL (11 nhãn, 27.5% macro, hiện nhận **ZERO** từ ensemble)

### 4.1 · `cnn_ms` cải thiện nhiều nhãn tail, chỉ 3 nhãn phá

Đo ở `61` C7 (thang 10mm vs 21mm): `cnn_ms` nâng **khá nhiều** nhãn tail, nhưng 3 nhãn tụt nặng:

```
                     boi so EHR-only   giu@k=5% (nb65)   ban chat
Azygos Lobe              1.4x              1.025         BIEN THE GIAI PHAU bam sinh (~1% dan so)
Hydropneumothorax        4.2x              0.880         co hoc / do can thiep (dan luu, chan thuong)
Pneumomediastinum        4.8x              0.845         barotrauma, tho may, vo thuc quan, noi soi
(doi chieu: Edema 2.6x · Support Devices 2.2x)
```

**Hai chẩn đoán độc lập trỏ cùng một chỗ:** `Pneumomediastinum` **tập trung attention nhất**
(`0.300` vs tail TB `0.222`) nhưng **mất nhiều nhất khi che** (`0.845`) — khí trung thất hiện thành
**vệt sáng mảnh chạy dọc bờ trung thất**, cần **cả đường** chứ không phải khối gọn. Đó là lý do
đa tỉ lệ 10mm phá nó, và cũng là lý do crop/cascade sẽ phá nó.

`Azygos Lobe` **không phải bệnh** ⇒ EHR không thể biết (`1.4×` ≈ sàn), và nó là nhãn duy nhất
**tốt lên** khi che (`1.025`).

**Việc:** trộn `cnn_ms` ở tail với `w` theo tier, **loại 3 nhãn kia theo CƠ CHẾ** (không theo số đo
— per-label ở tail không chuyển giao, `|Δr| = 0.416` và val→test `−19%`).
**Rủi ro:** loại nhãn theo cơ chế vẫn là một lựa chọn; phải khoá danh sách 3 nhãn **trước** khi đọc test.

### 4.2 · EHR mở rộng — thủ thuật / thở máy / kiểu nhập viện

`Hydropneumothorax` và `Pneumomediastinum` có **bội số EHR cao nhất nhóm tail** nhưng EHR của ta
thiếu đúng loại đặc trưng:

```
DANG CO   lab_* · vital_* · micro_* · dac trung XGB
THIEU     proc_chest_tube_{24h,7d,ever}   procedureevents + procedures_icd
          proc_thoracentesis_{24h,7d}
          proc_intubation_{24h,7d}
          vent_mech_dang_hoat_dong        chartevents / bang ventilation dan xuat
          vent_peep_max_24h               proxy barotrauma
          proc_noi_soi_7d                 EGD / phe quan
          adm_type                        admissions (EMERGENCY/URGENT/ELECTIVE)
          dx_copd_ever / dx_hen_ever      diagnoses_icd truoc moc
```

~15 cột, **cùng dạng** `lab_*`/`vital_*` ⇒ cắm thẳng vào `ColTokBag`, **không đổi kiến trúc**.

```
bac 1  BigQuery dung dac trung        ~2-3h   theo mau 29c
bac 2  PROBE XGB EHR-only theo nhan   ~30ph   0 GPU
       CONG: boi so Hydro & Pneumomed tang >= 1.5x so hien tai (4.2x / 4.8x)
bac 3  neu qua cong: huan luyen lai tpb voi EHR mo rong   ~13h/seed
```

**Bắt buộc đối chứng:** `dẫn lưu màng phổi → hydropneumothorax` là **đường tắt** — mô hình có thể
học nó mà không nhìn ảnh. Chạy phép **tráo ảnh sang bệnh nhân khác, giữ EHR** (mẫu `06b`). Nếu AP
hai nhãn đó gần như không đổi ⇒ mô hình đang đọc EHR chứ không đọc phim, phải báo cáo đúng vậy.

**Kỳ vọng thật:** `~+0.002` macro. Nhỏ, nhưng nó **gỡ đúng chỗ đánh đổi** và là minh chứng sạch
nhất cho luận điểm dự án.

### 4.3 · PadChest — giám sát ngoài cho đúng nhãn tail ta yếu

Ghi từ `2026-08-14`, **chưa bao giờ dùng**. PadChest = Hospital San Juan (Alicante), MIMIC-CXR =
Beth Israel (Boston) ⇒ **không chung bệnh nhân, không rò rỉ test**. `Removal.csv` của ban tổ chức
đã loại sẵn dev/test.

**17/30 nhãn trùng, trong đó 4 là nhãn TAIL:**

```
Azygos Lobe          176 anh    <- 1 trong 3 nhan cnn_ms pha
Hydropneumothorax    549 anh    <- 1 trong 3 nhan cnn_ms pha
Pneumoperitoneum     454 anh
Kyphosis             669 anh
```

Trọng số `convnextv2_padchest.pt` **đã có** và đang dùng làm khởi tạo cho `64b`. Nhưng ta mới dùng
nó như *pretrain*, **chưa** dùng PadChest như **nguồn nhãn bổ sung** cho 4 nhãn tail đó.

**Việc:** huấn luyện chung (co-training) hoặc pretrain có giám sát trên tập con 4 nhãn tail, rồi
fine-tune. Với nhãn hiếm nhất train có `129` ca dương, thêm `176–669` ca là **gấp 2–6 lần**.
**Đây là đòn bẩy tail lớn nhất trong danh sách.** Chi phí: 1 notebook + ~8h.

---

## 5 · MỞ — đắt, nhưng đây mới là chỗ có `0.0185`

### 5.1 · Ensemble các backbone fine-tune ĐỘC LẬP ⭐ lớn nhất

**Kết quả `+0.0008` KHÔNG đóng hướng này** — phạm vi của nó hẹp hơn tôi từng nói:

```
cac seed cua ta dung CHUNG mot backbone 38a DONG CUNG (86M tham so Y HET NHAU)
-> chi 7.3M dau doc la khac  ->  tuong quan cao la tat nhien
```

Chạy `38a` với seed/augment khác cho **86M tham số khác nhau thật**. Đây là loại đa dạng ta
**chưa từng đo**, và đúng công thức vô địch (*"fine-tune + ensemble"*).

`~40–60h` · kỳ vọng `+0.008 ~ +0.015`

### 5.2 · Chạy lại `38a` trên 87% dữ liệu

Backbone hiện tại fine-tune trên `TF_` cũ (74%). `R2a` cho `+0.0022` chỉ ở tầng đầu đọc (7.3M);
ở backbone (86M, đói dữ liệu hơn) có thể nhiều hơn. `~6h + 13h` · `+0.005 ~ +0.010`

### 5.3 · Thành viên CNN lớn hơn

`convnextv2_nano` (~13 GMAC, IN-1k 82.0) → `tiny` (~24, 84.9) hoặc `base` (~82, 86.8). CNN hiện
yếu hơn hẳn đường ống (`0.2463` vs `0.2663`); mạnh lên thì `w` tối ưu tăng và blend cho nhiều hơn.
`~8h` · `+0.004`

---

## 6 · MỞ — nhánh EHR/dữ liệu, rẻ, chưa ai chạy

Từ mục **"Còn treo"** của JOURNAL:

- [ ] **`Q_CTX` vs `Q_CTX + como`** — tách 11 cột khỏi khối 90. CPU vài phút.
- [ ] **`MICRO_FUTURE_H {0,24,48,72}`** — `29b` ghi là quét cho Stage 2, chưa ai quét.
      ⚠️ Cửa sổ `+72h` là thông tin **không có ở thời điểm suy luận thực tế**.
- [ ] **`EHRColTok` vẫn phát token cho cột VẮNG MẶT** (`T_j = b_j`). `PLAN §B2b` nói *"thiếu: KHÔNG
      sinh token"* — chưa làm, chưa đo. Đây là cách xử lý missingness triệt để nhất.
- [ ] **28.7% ảnh không có phim trước** — cần token *"không có tiền sử"*, chưa đo.
- [ ] **18 cột `lab_*_7d` trùng lặp 91%** (`42` C2) — dọn rẻ hơn thêm đặc trưng xu hướng.
- [ ] **Comorbidity vẫn là cờ one-hot** — đo trước, đừng dựng trước (đuôi ICD phần lớn không liên
      quan ngực).
- [ ] **Longitudinal arm `D` = `[patch_cur ; patch_cur − patch_prior]`** — thiết kế đã chốt, chưa
      dựng. `43` cho `+0.0026` (1.5× nhiễu).

---

## 7 · MỞ — nhiệm vụ thứ hai (giá trị cho BÀI VIẾT, không cho mAP)

**`nb40`** — 25 CCS phenotype (ICD→CCS) + tử vong nội viện 48h, tái dùng `f_cxr`.
Mốc MedFuse: `0.481` / `0.594` AUPRC; EHR ở đó đóng `+0.028` (ta chỉ `+0.004` trên CXR-LT).

Nó biến kết quả âm ở CXR-LT từ *"hướng thất bại"* thành **"đã định vị được ranh giới: EHR không
giúp nhãn sinh từ report, giúp rõ nhãn lâm sàng"**. `~1 notebook` + tái dùng.

---

## 8 · ĐÃ ĐÓNG — đừng làm lại

| Hướng | Bằng chứng đóng |
|---|---|
| Deep EHR conditioning (adapter trong block) | `56`: `G_deep` **âm**, ổn định qua seed |
| Mở băng block 8–11 | `57`: `G_ft` `−0.007..−0.010`, 2 seed đồng thuận, đối chứng khoẻ |
| cRT / decoupled classifier | `59`: classifier **đã hội tụ**, không còn gì để tách |
| Đa tỉ lệ `cnn_ms` làm hướng chính | `41f`: cổng chính **trượt**; `62` giải thích cơ chế — stage 3 **gần rank-1** (hạng hiệu dụng `1.1`/320 kênh) ⇒ 1024 khoá gần cộng tuyến, pha loãng attention |
| Độ phân giải ĐẦU VÀO | `51`: `lb672` thêm token → `−0.0028` |
| Thêm seed của cùng backbone đông cứng | `+0.0008`, dưới sàn nhiễu `0.0007`. ⚠️ **Không** đóng mục 5.1 |
| Neural collapse | `62` C5: bác |
| Toàn bộ chương trình `w = f(r)` | Hiện vật nhiễu chung: `+0.598` (cùng mẫu) → `+0.136 p=0.40` (khác mẫu) |
| `rho` (tương quan dự đoán) làm biến điều kiện | Đo được (tin cậy `1.000`) nhưng **vô thông tin** (`+0.017`) |
| Phép thử ngưỡng kích thước bằng đốm tổng hợp | Muthyala et al. đã làm, kỹ hơn, **có RAD-DINO** ⇒ chỉ trích dẫn |
| Multi-view `w_l` theo bệnh | `17` C7/V1: `+0.00017`, CI chứa 0, tail `−0.0016` |
| FPN đa tỉ lệ trên ViT thuần | Không có phân cấp stride để bám |
| `#4` chưng cất nhãn mềm | **Hạ xuống**, không đóng: ta ở 40% trần, trần ràng buộc **cả vô địch lẫn ta như nhau** ⇒ chưa phải chỗ nghẽn |

---

## 9 · Nợ tài liệu

- [ ] `REPORT_STAGE4_NB38A.md` — **chưa tồn tại**. Khi viết phải kèm điều kiện *"ở ngân sách đầu
      đọc bằng nhau"*.
- [ ] `REPORT` cho `41e` / `41f` / `61` / `64` / `64b`
- [ ] **Limitations**: trần cấu trúc `9.5mm/patch` (518px ÷ ~350mm = 0.676 mm/px; patch 14px), kèm
      trích Muthyala et al. arXiv:2606.11606 — biến "tail thấp" từ thất bại thành **giới hạn có
      tài liệu**.
- [ ] Ghi rõ trong bài: **`Azygos Lobe` là biến thể giải phẫu bẩm sinh, không phải bệnh** — nó
      không nên bị chấm chung thang với nhãn bệnh lý.
- [ ] `38b s43`: đề xuất **dừng** (mọi arm EHR của nó chạy trên biểu diễn vừa thay). Tuấn chưa chốt.

---

## 10 · Thứ tự đề xuất

```
1. muc 2 (2.1-2.4)      vai phut, 0 GPU        gop thanh MOT cell quet tren val
2. muc 3.1 (nb66)       ~1h                    mo/dong huong do phan giai
3. muc 3.2 + 3.4        ~2h                    TTA + chay lai 65 tren trong so that
4. muc 4.3 (PadChest)   ~8h                    don bay TAIL lon nhat
5. muc 5.1              ~40-60h                don bay MACRO lon nhat
6. muc 4.2 (EHR mo rong) song song, 0 GPU cho bac 1-2
```

Chốt cả `1–3` trên val, rồi **đọc test một lượt** cho toàn bảng.
