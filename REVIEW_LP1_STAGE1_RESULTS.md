# REVIEW — Kết quả Stage 0–2 của `21_label_pipeline_v1.ipynb`

Arm `A3_negshare` · teacher `T_blank0` (`shared_blank0`) · 5 fold × 8 epoch · seed 42 ·
`fixed_last_epoch` · `BATCH=64 × ACCUM=4`. Chạy ~22.4 h GPU.

---

## 1. Kết luận ngắn

**Pipeline sạch về cơ học.** 6/6 leakage check pass, gate §15 = 0 fold-label thiếu lớp,
`hard/soft/mask` không chồng lấn, gradient trên masked cell = `0.00e+00`, manifest cộng đúng
`1,921,283 = 147,791 × 13`, hash đã khoá.

**Teacher không collapse** — có tán xạ, và xếp hạng có nghĩa.

**Nhưng teacher lệch hiệu chuẩn lên trên rất mạnh, và điều đó phá đúng nhánh `blank`.** Xác suất
trung bình cao hơn prevalence thật **1.6×–25×**. Plan §7 cấm calibrate nên con số lệch đó đi thẳng
vào BCE làm soft target. Hệ quả: 4/13 nhãn vượt trần cảnh báo `>2×` effective prevalence, và cả 4
đều nằm trong 5 nhãn được route sang `TEACHER_SOFT`.

**Giá trị thật nằm ở nhánh `-1`, không phải nhánh `blank`.**

---

## 2. Teacher lệch hiệu chuẩn — bằng chứng

`T_blank0` train với `blank → hard 0` mọi nhãn, nên prevalence trong tập train của nó = cột
`prev_T`. So với xác suất OOF nó sinh ra:

| label | prev_T | OOF mean | **mean/prev** | frac>0.5 | **frac/prev** |
|---|---:|---:|---:|---:|---:|
| Pleural_Other | 0.0088 | 0.2218 | **25.2×** | 0.0091 | 1.03 |
| Fracture | 0.0194 | 0.2989 | **15.4×** | 0.0155 | 0.80 |
| Lung_Lesion | 0.0275 | 0.3134 | **11.4×** | 0.0367 | 1.33 |
| Enlarged_Cardiomediastinum | 0.0336 | 0.3518 | **10.5×** | 0.0280 | 0.83 |
| Consolidation | 0.0491 | 0.3491 | **7.1×** | 0.0974 | 1.98 |
| Pneumothorax | 0.0462 | 0.2995 | **6.5×** | 0.0670 | 1.45 |
| Pneumonia | 0.0783 | 0.4105 | **5.2×** | 0.1641 | 2.10 |
| Edema | 0.1797 | 0.4317 | 2.4× | 0.3453 | 1.92 |
| Cardiomegaly | 0.2048 | 0.4704 | 2.3× | 0.4646 | 2.27 |
| Lung_Opacity | 0.2322 | 0.5028 | 2.2× | 0.5448 | 2.35 |
| Atelectasis | 0.2502 | 0.4922 | 2.0× | 0.5078 | 2.03 |
| Pleural_Effusion | 0.2491 | 0.4602 | 1.9× | 0.4058 | 1.63 |
| No_Finding | 0.3278 | 0.5383 | 1.6× | 0.5096 | 1.55 |

Đọc hai cột cuối cùng lúc: **`frac>0.5` bám prevalence khá tốt (0.80–2.35×) trong khi `mean` lệch
tới 25×**. Đó là chữ ký của một loss **xếp hạng chứ không hiệu chuẩn**.

### Nguyên nhân cơ chế

`ASL_CLIP = 0.05` (probability shifting) đặt gradient của mọi negative có `p < 0.05` về **đúng 0**;
`γ⁻ = 4` xoá tiếp gradient của negative dễ. Model vì thế **không có áp lực nào** đẩy xác suất xuống
gần 0 — chúng dồn đống ở dải 0.2–0.5. Đây là hành vi thiết kế của ASL, không phải bug: nó tối ưu
thứ hạng cho mAP, và mAP thì bất biến với phép biến đổi đơn điệu.

Vấn đề là **BCE thì không bất biến**. Plan §10 dùng `BCE(y_soft, z)` — nó quan tâm giá trị tuyệt
đối. Ta đang nạp một đại lượng chỉ có nghĩa về thứ hạng vào một hàm loss đòi giá trị.

---

## 3. Hệ quả — effective prevalence

| label | historical | effective | ratio | blank→soft? |
|---|---:|---:|---:|:--:|
| Consolidation | 0.0672 | 0.3629 | **5.40×** ⚠ | ✔ |
| Pneumothorax | 0.0511 | 0.2504 | **4.90×** ⚠ | ✔ |
| Pneumonia | 0.1529 | 0.4043 | **2.64×** ⚠ | ✔ |
| Edema | 0.1797 | 0.4505 | **2.51×** ⚠ | ✔ |
| Pleural_Effusion | 0.2685 | 0.4848 | 1.81× | ✔ |
| *8 nhãn còn lại* | — | — | **0.83–1.00** | ✘ |

Bốn nhãn bắn cảnh báo, nhãn thứ năm ở 1.81× — **ngay sát dưới ngưỡng, chỉ vì Pleural_Effusion có
blank rate thấp nhất (0.616)**. Nói cách khác **cả 5/5 nhãn đi soft đều bị méo**, chỉ có một cái
lọt lưới cảnh báo.

Ngược lại, **8 nhãn `HARD_ZERO` gần như không đổi (0.83–1.00)**. Router `neg_share` đã làm đúng
việc của nó: khoanh vùng thiệt hại còn 5 nhãn thay vì 12 như plan gốc. Nếu chạy `A1_allsoft` thì
cả 12 nhãn sẽ có dạng méo như Consolidation.

---

## 4. Nhánh `blank` — gần như self-distillation, không phát hiện được omission

Đây là điểm tôi đã cảnh báo ở `REVIEW_LABEL_PIPELINE_V1.md` §4-bis, và số liệu xác nhận.

`T_blank0` được dạy `blank = 0` trên cả 13 kênh. Nghĩa là nó **kế thừa nguyên giả định blank=0**,
rồi ta lấy chính dự đoán của nó làm soft target cho blank. Bằng chứng:

```
frac_gt_09 trên blank_teacher:  Consolidation 0.0000 · Edema 0.0000 · Pneumonia 0.0000
                                Pneumothorax 0.0001 · Pleural_Effusion 0.0006
```

**Teacher gần như không bao giờ khẳng định một cell blank là positive bị bỏ sót.** Trên 551,229
cell blank đi soft, số cell có `p > 0.9` đếm trên đầu ngón tay. Nó chỉ nói "trong đám blank, mấy
cái này đáng ngờ hơn" — thông tin thứ hạng, không phải phát hiện omission.

Vậy nhánh blank ở arm này ≈ **distill R0 vào chính nó trên 30% số cell**. Kỳ vọng hợp lý: trung
tính, hoặc nhích nhẹ do hiệu ứng self-distillation. Không phải cơ chế "sửa nhãn" như plan hình dung.

Đối chiếu với anchor thời gian của R2 (nb16): chỉ **4.5–7.9%** blank của Rare-3 có anchor `+1`
trong ±7 ngày. Đó mới là bậc độ lớn của omission thật.

---

## 5. Nhánh `-1` — chỗ có giá trị thật

Teacher xử lý `-1` rất hợp lý. So `-1` với chính trung bình của nhãn đó:

| label | OOF mean | `-1` mean | **tỉ số** | `-1` frac>0.5 | n(-1) | % positive lịch sử đến từ `-1` |
|---|---:|---:|---:|---:|---:|---:|
| Pleural_Other | 0.2218 | 0.3667 | 1.65 | 0.0886 | 463 | 26.3% |
| Pneumothorax | 0.2995 | 0.4836 | 1.61 | 0.3976 | 762 | 10.1% |
| Lung_Lesion | 0.3134 | 0.4394 | 1.40 | 0.2652 | 709 | 14.9% |
| Pleural_Effusion | 0.4602 | 0.5966 | 1.30 | 0.7464 | 3,809 | 9.6% |
| Fracture | 0.2989 | 0.3861 | 1.29 | 0.1124 | 347 | 10.8% |
| **Pneumonia** | 0.4105 | 0.5054 | 1.23 | **0.4782** | **11,967** | **52.9%** |
| Consolidation | 0.3491 | 0.4249 | 1.22 | 0.1938 | 2,817 | 28.4% |
| Cardiomegaly | 0.4704 | 0.5535 | 1.18 | 0.7345 | 3,815 | 11.5% |
| **Enlarged_Card.** | 0.3518 | 0.4056 | 1.15 | **0.0613** | **6,081** | **56.1%** |
| Lung_Opacity | 0.5028 | 0.5647 | 1.12 | 0.7927 | 2,431 | 6.7% |

Ba điều đáng chú ý:

**Thứ tự đúng.** `-1` luôn cao hơn trung bình nhãn (1.12–1.65×), và nơi có cả hai thì
`-1 > blank` nhất quán: Pneumothorax 1.82× · Pleural_Effusion 1.62× · Pneumonia 1.29× ·
Consolidation 1.24×. Teacher phân biệt được ba trạng thái, không ra hằng số.

**Hai nhãn quan trọng nhất được sửa mạnh.** `Enlarged_Cardiomediastinum` có 56.1% "positive" lịch
sử đến từ `-1`, và teacher nói **chỉ 6.1%** trong số 6,081 cell đó trông giống positive. `Pneumonia`
có 52.9% từ `-1`, teacher nói **47.8%**. U-Ones đang gán cả hai nhóm này thành 1 hết. Đây là chỗ
relabeling có dư địa thật, và `Enlarged_Cardiomediastinum` đúng là một trong `WORST3_FIXED` của nb16.

**Chỉ chạm 33,201 cell (1.73%)**, gradient share ~0.9%. Can thiệp đúng chỗ, rẻ, không phá prior.

---

## 6. Nhận định tổng

| thành phần | đánh giá |
|---|---|
| Cơ học pipeline | ✅ sạch, đủ assert, hash khoá, tái lập được |
| Teacher không collapse | ✅ có tán xạ, thứ tự `-1` > `blank` > nền |
| Teacher hiệu chuẩn | ❌ lệch 1.6–25×, do ASL clip 0.05 + γ⁻=4 |
| Router `neg_share` | ✅ khoanh thiệt hại từ 12 nhãn xuống 5 |
| Nhánh `-1` → soft | ✅ hợp lý, giá trị tập trung ở Enl.Card + Pneumonia |
| Nhánh `blank` → soft | ⚠ self-distillation, méo prevalence 1.8–5.4×, `frac>0.9 ≈ 0` |

**Dự đoán trước khi train student:** phần gain (nếu có) đến từ `-1`; phần blank hoặc trung tính
hoặc âm ở 5 nhãn bị méo. Vì `Edema`/`Pleural_Effusion`/`Pneumonia` là nhãn head, nếu chúng xấu đi
thì Macro mAP xấu theo — mà Macro là primary endpoint.

---

## 6-bis. Cơ chế gây hại thật — khối lượng gradient, không phải bảng prevalence

Bảng effective prevalence chỉ là *triệu chứng*. Đại lượng quyết định là **student thực sự nhận bao
nhiêu gradient từ đâu**. Ước lượng tại điểm `σ(z) = 0.05` trên ô negative (mức ASL đẩy tới):

| label | hard_pos | n_soft | soft target | `G_pos` | `G_negASL` | `G_soft` | **soft/pos** |
|---|---:|---:|---:|---:|---:|---:|---:|
| Consolidation | 7,117 | 135,485 | 0.343 | 917 | **0** | 19,862 | **21.7×** |
| Pneumonia | 10,636 | 122,040 | 0.403 | 1,370 | **0** | 21,510 | **15.7×** |
| Pneumothorax | 6,793 | 113,193 | 0.267 | 875 | **0** | 12,281 | **14.0×** |
| Edema | 26,553 | 104,976 | 0.381 | 3,420 | **0** | 17,395 | **5.1×** |
| Pleural_Effusion | 35,866 | 94,890 | 0.377 | 4,620 | **0** | 15,519 | **3.4×** |
| Enlarged_Card. | 4,767 | 6,081 | 0.406 | 614 | 0 | 1,081 | 1.8× |
| *6 nhãn còn lại* | — | — | — | — | 0 | — | **0.14–0.44×** |

Ba con số cần đọc kỹ:

```
ASL gradient tren mot o negative de tai p = 0.05  ->  DUNG BANG 0
    vi ASL_CLIP=0.05:  xn = 1 - 0.05 + 0.05 = 1.0  ->  log(1) = 0
ASL gradient tren o negative tai p = 0.20         ->  0.0014   (van gan nhu 0)
BCE·lambda=0.5 tren o soft target 0.2655 @ p=.05  ->  0.1078   (lon gap ~77 lan)
```

**Không có gì cưỡng lại lực kéo lên của BCE.** ASL được thiết kế để *bỏ qua* vùng negative dễ —
đó là lý do nó tốt cho mAP. Nhưng chính vì thế nó không thể bảo vệ vùng đó khi một số hạng khác kéo
ngược.

Hệ quả: với Consolidation, **số hạng soft mang khối lượng gradient gấp 21.7 lần toàn bộ positive
của nhãn đó**. `λ_soft = 0.5` của plan đặt *hệ số*, nhưng trọng số hiệu dụng =
hệ số × số cell × biên độ gradient mỗi cell — và hai thừa số sau lớn hơn nhiều so với plan giả định.
**Số hạng phụ đã trở thành số hạng chính** cho 5 nhãn.

### Một cơ chế hại cụ thể hơn

Với nhãn đi soft, loss yêu cầu hai thứ khác nhau trên hai nhóm cell:

```
o raw == 0  (explicit negative)  ->  ASL keo ve  ~0     (nhung gradient = 0, khong day duoc)
o raw == blank                    ->  BCE keo ve  ~0.27  (gradient lon)
```

Khác biệt hệ thống duy nhất giữa hai nhóm là **bác sĩ có buồn nhắc tới dấu hiệu đó hay không**.
Nên một phần lớn gradient soft đang dạy model dự đoán *mention-status*, không phải bệnh lý. Model
học được (chest tube, hậu phẫu → pneumothorax hay được bàn tới), nhưng đó là **reporting bias được
bơm ngược vào model**, đúng thứ mà cả pipeline này sinh ra để chống lại.

### Điều tôi phải nói rõ là chưa biết

Macro mAP dựa trên **thứ hạng**, nên một phép dịch đơn điệu không làm hỏng nó. Xác suất soft vẫn
mang thông tin thứ hạng (std 0.096–0.162), và thứ hạng đó đến từ teacher train trên tập nhãn gần
như R0. Nên **dấu của tác động lên AP là chưa xác định được từ đây** — có thể trung tính.

Nhưng nhận định "self-distillation nhẹ" ở §4 của tôi là **nói nhẹ đi so với thực tế**. Với
soft/pos = 14–22×, đây không phải một lớp làm mượt — đây là thay đổi mục tiêu huấn luyện ở quy mô
lớn, chỉ là ta chưa biết nó đẩy về hướng nào.

---

## 6-ter. Nhánh `-1`: teacher nói uncertain **gần với 0 hơn là 1**

Hệ số làm giàu = `frac>0.5` của ô `-1` chia cho `frac>0.5` chung của nhãn đó. Cột cuối là ước lượng
xác suất thật sau khi neo lại về prevalence nền (isotonic bảo toàn thứ hạng nên phép neo này hợp lệ):

| label | n(-1) | % positive lịch sử từ `-1` | **làm giàu** | **P(`-1` thật \| đã hiệu chuẩn)** |
|---|---:|---:|---:|---:|
| Pleural_Other | 463 | 26.3% | **9.74×** | 0.086 |
| Fracture | 347 | 10.8% | **7.25×** | 0.141 |
| Lung_Lesion | 709 | 14.9% | **7.23×** | 0.199 |
| Pneumothorax | 762 | 10.1% | **5.93×** | 0.274 |
| Pneumonia | 11,967 | **52.9%** | 2.91× | 0.228 |
| Enlarged_Cardiomediastinum | 6,081 | **56.1%** | 2.19× | **0.074** |
| Consolidation | 2,817 | 28.4% | 1.99× | 0.098 |
| Pleural_Effusion | 3,809 | 9.6% | 1.84× | 0.458 |
| Cardiomegaly | 3,815 | 11.5% | 1.58× | 0.324 |
| Lung_Opacity | 2,431 | 6.7% | 1.46× | 0.338 |

Hai điều rút ra:

**Teacher hành xử đúng về chất.** Hệ số làm giàu > 1 ở cả 10 nhãn — ô `-1` luôn đáng ngờ hơn nền.
Và nhãn càng hiếm thì làm giàu càng mạnh (Pleural_Other 9.7× so với Lung_Opacity 1.5×), đúng trực
giác: với một dấu hiệu hiếm, một lần nhắc mơ hồ là tín hiệu mạnh hơn nhiều so với nền.

**Nhưng về lượng, teacher nói `-1` gần `0` hơn `1`.** 8/10 nhãn có `P(-1 thật) < 0.35`;
`Enlarged_Cardiomediastinum` chỉ **0.074**. U-Ones đang gán **1.000** cho tất cả. Nghĩa là phần lớn
gain khả dĩ của nhánh `-1` có thể lấy được **chỉ bằng cách không gán 1** — và điều đó không cần
teacher nào cả.

Đây là một arm rẻ chưa có trong bộ so sánh: `-1 → 0` (U-Zeros) cho các nhãn mà teacher nói
`P < 0.15` (Enl_Card 0.074, Pleural_Other 0.086, Consolidation 0.098, Fracture 0.141).

---

## 7. Đề xuất — xếp theo giá trị/chi phí

### 7.1 Chạy `A2_allzero` trước, không phải `A3`

`A2` = `blank → hard 0` + `-1` disease-aware. Nó **cô lập đúng phần có giá trị** và bỏ đúng phần
đáng ngờ. Dùng lại `T_blank0` đã train nên **không tốn thêm GPU cho teacher**, chỉ tốn lượt student.

Effective prevalence của `A2` chính là các con số `0.83–1.00` đã in ở trên cho 8 nhãn hard, cộng
`Enl_Card 0.667` và các nhãn `-1` khác — **không nhãn nào vượt 2×**. Sạch.

So sánh `H0 → A2` cho ta hiệu ứng thuần của chính sách `-1`. Nếu `A2` không nhích thì `A3` càng
không có lý do nhích, và ta tiết kiệm được cả một vòng 3 seed.

### 7.2 Nếu vẫn muốn nhánh blank: hiệu chuẩn teacher **trên TRAIN**

Plan §7 cấm calibrate bằng **VAL/TEST**. Nó không cấm hiệu chuẩn bằng chính TRAIN OOF — và điều đó
không rò rỉ gì cả:

```
Với mỗi nhãn c:
  fit isotonic regression trên cac cell HARD cua TRAIN
      x = p_OOF[i,c]     y = Y_HARD[i,c]      (chi cell M_HARD == 1)
  ap dung anh xa do len cac cell SOFT cua chinh nhan do
```

Isotonic **bảo toàn thứ hạng** (đơn điệu) nên không phá thông tin teacher học được, chỉ kéo thang
giá trị về đúng chỗ. Chi phí: **CPU, vài giây**.

Tác động lên đại lượng thật sự quan trọng — tỉ số gradient soft/pos:

| label | soft target trước → sau | `G_soft` trước → sau | **soft/pos trước → sau** |
|---|---|---:|---|
| Consolidation | 0.343 → 0.067 | 19,862 → 1,165 | **21.7× → 1.3×** |
| Pneumothorax | 0.267 → 0.051 | 12,281 → 62 | **14.0× → 0.1×** |
| Pneumonia | 0.403 → 0.153 | 21,510 → 6,279 | **15.7× → 4.6×** |
| Edema | 0.381 → 0.180 | 17,395 → 6,808 | **5.1× → 2.0×** |
| Pleural_Effusion | 0.377 → 0.269 | 15,519 → 10,367 | **3.4× → 2.2×** |

Số hạng soft quay về đúng vai *phụ trợ* — đúng cái mà `λ_soft = 0.5` định nói. **Đây là thay đổi có
đòn bẩy cao nhất trong toàn pipeline**, và nó không vứt đi 22 h GPU đã bỏ ra: teacher giữ nguyên,
chỉ đổi thang đọc đầu ra của nó.

**Vì sao đây không phải rò rỉ.** Isotonic fit trên `{(p_OOF, Y_HARD) : M_HARD == 1}` và chỉ áp lên
các ô `M_SOFT == 1`. Hai tập **rời nhau** — không ô nào tham gia định hình giá trị của chính nó.
Tính OOF của `p` được giữ nguyên vì phép biến đổi là hậu kỳ, đơn điệu, không đụng VAL/TEST.

Cần khoá lại: fit chỉ trên TRAIN, per-label, ghi hash của bảng ánh xạ vào `experiment_manifest.json`,
và ghi rõ là sửa đổi so với plan §7.

### 7.3 Không nên làm

* **Đừng train lại teacher.** Nó không hỏng — nó làm đúng việc ASL bảo nó làm.
* **Đừng đổi `λ_soft`** để bù méo. Đó là vá triệu chứng; gốc là thang xác suất.
* **Đừng chạy `A1_allsoft`** trước khi biết `A2` có nhích không. `A1` đưa cả 12 nhãn vào tình
  trạng của Consolidation, 76.7% cell thành soft.

---

## 7-bis. Stage E0 — EHR-informed teacher: kết quả và **một lỗi thiết kế của cổng**

Chạy `25_ehr_teacher.ipynb` (CPU, ~3 phút — nhanh hơn ước lượng nhiều). Cổng E0 báo
`full: PASS`, `ctx: FAIL`. Nhưng phân rã kỹ thì hai vế của cổng có chất lượng rất khác nhau.

### 7-bis.1 · E0-a đứng vững — EHR có thông tin gia tăng thật

| | macro AUROC trên ô tin cậy held-out |
|---|---:|
| CXR-alone | 0.8391 |
| + EHR Q_CTX | 0.8476 (+0.0085) |
| **+ EHR Q_FULL** | **0.8518 (+0.0127)** |
| + EHR **xáo trộn** | 0.8384 (**−0.0007**) |

Shuffle control cho **−0.0007**, tức thêm một feature bất kỳ vào stacker **không** tự sinh ra lift.
Sau khi trừ control, EHR đóng góp thật **+0.0133 macro AUROC**.

Phân bố đóng góp rất tập trung:

| label | AUROC CXR | +EHR | **EHR thật (trừ shuffle)** | n_trusted | n_neg |
|---|---:|---:|---:|---:|---:|
| **Fracture** | 0.7490 | 0.7936 | **+0.0470** | 3,451 | 586 |
| **Lung_Lesion** | 0.7852 | 0.8287 | **+0.0456** | 4,564 | 512 |
| Pneumonia | 0.8245 | 0.8408 | +0.0164 | 25,751 | 15,115 |
| Pleural_Other | 0.8126 | 0.8229 | +0.0124 | 1,383 | 83 |
| Enlarged_Cardiomediastinum | 0.7830 | 0.7926 | +0.0101 | 8,091 | 3,324 |
| Pleural_Effusion | 0.9515 | 0.9521 | +0.0007 | 52,901 | 17,035 |
| Pneumothorax | 0.8949 | 0.8952 | +0.0004 | 34,598 | 27,805 |

Hai nhãn hưởng lợi nhất — `Fracture`, `Lung_Lesion` — là **hai trong Rare-3**, và cũng là hai nhãn
CXR yếu nhất (0.749 / 0.785). Hợp lý về lâm sàng: tiền sử chấn thương, tiền sử ung thư/comorbidity
không nằm trong phim.

*Dè dặt:* `Fracture` chỉ có **586** negative, `Lung_Lesion` **512**. Vì là so sánh ghép cặp (cùng ô,
thêm feature) nên phương sai của *hiệu* nhỏ hơn nhiều phương sai từng AUROC, nhưng không nên trích
`+0.047` như con số chính xác — coi ±0.02 là hợp lý.

### 7-bis.2 · E0-b **vô nghĩa như đã thiết kế** — 92% độ tán tăng là do hiệu chuẩn

Mốc so sánh của tôi sai: lấy `std_full` so `std_cxr`, trong khi mốc đúng là `std_shuffle`.

| label | std_cxr | **std_shuffle** | std_full | do hiệu chuẩn | **do EHR** |
|---|---:|---:|---:|---:|---:|
| Pneumonia | 0.0976 | **0.2444** | 0.2473 | +0.1468 | **+0.0029** |
| Enlarged_Cardiomediastinum | 0.0574 | **0.1757** | 0.1779 | +0.1183 | **+0.0022** |
| Consolidation | 0.0827 | **0.3097** | 0.3112 | +0.2270 | +0.0015 |
| Pleural_Effusion | 0.1454 | **0.3022** | 0.3022 | +0.1568 | **+0.0000** |
| Edema | 0.1147 | **0.2820** | 0.2822 | +0.1673 | +0.0002 |
| *Fracture* | 0.0912 | 0.1174 | 0.1539 | +0.0262 | *+0.0365* |
| *Lung_Lesion* | 0.1132 | 0.1021 | 0.1246 | −0.0111 | *+0.0225* |

Trên 8 nhãn có std tăng rõ: **hiệu chuẩn 91.7%, EHR 8.3%**.

Cơ chế: stacker là `σ(a·logit p_CXR + b·logit p_EHR + c)`. EHR xáo trộn → `b → 0`, và nó thoái hoá
thành **Platt scaling của p_CXR**. Vì ASL nén xác suất vào dải 0.2–0.5 (§2), hệ số `a > 1` kéo giãn
phân phối. Đúng lỗi lệch thang 1.6–25× đã ghi ở §2 — stacker sửa nó, nhưng đó là công của **hiệu
chuẩn**, không phải EHR.

Hai ngoại lệ `Fracture` / `Lung_Lesion` (EHR đóng 58% / toàn bộ) trùng đúng hai nhãn có lift AUROC
lớn nhất — nhất quán nội tại, củng cố rằng phần EHR là thật ở đúng hai nhãn đó.

### 7-bis.3 · `A2E_full` trộn hai can thiệp, và nó **quay lại gần U-Ones**

Target mới bám sát **tỉ lệ dương trong ô tin cậy** — mốc mà bộ hiệu chuẩn neo vào:

| label | tỉ lệ dương ở ô tin cậy | A2 | **A2E** | cách U-Ones |
|---|---:|---:|---:|---:|
| Lung_Opacity | 0.946 | 0.565 | **0.927** | 0.073 |
| Pleural_Other | 0.940 | 0.367 | **0.921** | 0.080 |
| Lung_Lesion | 0.888 | 0.439 | **0.869** | 0.131 |
| Fracture | 0.830 | 0.386 | **0.767** | 0.233 |
| Pneumonia | 0.413 | 0.505 | 0.596 | 0.404 |
| Pneumothorax | 0.196 | 0.484 | 0.301 | 0.699 |

Ô tin cậy bị reporting bias — với nhãn hiếm nó 83–95% dương (§2-bis của
`REVIEW_LABEL_PIPELINE_V1.md`). Hiệu chuẩn vào quần thể đó kéo target `-1` lên gần 1. **Với 4 nhãn,
`A2E ≈ H0`.** Effective prevalence xác nhận: A2 có Enl_Card 0.667 / Pneumonia 0.738; A2E có
0.800 / 0.786 — **gần U-Ones hơn A2**.

Đây là lựa chọn khoa học chứ không phải lỗi: ô `-1` *là* ô có mention, nên neo vào tỉ lệ dương của
mention là hợp lý thống kê. Nhưng nó cũng có nghĩa ta đang **"cam kết hộ" bác sĩ** ở đúng chỗ họ cố
ý không cam kết. Phải nói rõ trong mọi báo cáo dùng arm này.

### 7-bis.4 · Hệ quả: cần arm control để quy kết

`A2E_full` = **hiệu chuẩn** + **EHR**. Không tách được nếu chỉ chạy một arm.

Đã bổ sung `A2C_platt` vào nb25 (C5): Platt scaling thuần trên `logit p_CXR`, fit OOF trên ô tin
cậy, **không có EHR**. Chi phí CPU vài giây.

Thứ tự chạy hiệu quả:

```
1. A2E_full  (4.2 h)   arm tot nhat kha di
     thua  -> ca nhanh chet, DUNG, khong can control
     thang -> di tiep buoc 2
2. A2C_platt (4.2 h)   quy ket: cong thuoc ve hieu chuan hay EHR?
     A2C ~ A2E  -> cong la cua HIEU CHUAN; EHR khong dang 
     A2C < A2E  -> EHR co dong gop that
```

**Dự đoán ghi trước:** phần lớn tác động (nếu có) đến từ **hiệu chuẩn**, vì nó sửa đúng bệnh đã chẩn
ở §6-bis — target gần-hằng-số. EHR chỉ thêm 8.3% độ tán và +0.0133 AUROC, tác động tập trung ở
`Fracture` / `Lung_Lesion` vốn chỉ có 347 / 709 ô `-1` — quá ít để dịch Macro.

---

## 8. Câu hỏi chốt

**Q1** — Chạy `A2_allzero` trước (rẻ, sạch, cô lập nhánh `-1`), hay giữ nguyên thứ tự và train
student cho `A3_negshare` ngay?

**Q2** — Có thêm bước isotonic calibration trên TRAIN cho teacher probability không? Nếu có thì
`A3` phải sinh lại manifest (CPU, ~2 phút, không đụng GPU) và cần ghi là một sửa đổi so với plan §7.

**Q3** — `Pleural_Effusion` ở `neg_share = 0.3004`, chỉ hơn ngưỡng 0.04 pp, và effective prevalence
của nó là 1.81× — cao nhất trong nhóm không bị flag. Có chốt tay đẩy nó về `HARD_ZERO` không?
Làm vậy thì nhánh blank chỉ còn 4 nhãn và mất 94,890 soft cell.
