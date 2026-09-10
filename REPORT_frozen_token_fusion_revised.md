# Kết luận 06b (bản hiệu chỉnh) — EHR fusion trên frozen CXR representation

> **Bản này thay thế `REPORT_frozen_token_fusion.md`.** Số liệu không đổi; phạm vi các
> phát biểu được thu hẹp lại cho khớp với mức bằng chứng thực có. Các chỗ sửa được ghi
> ở §7.

**Câu hỏi đã kiểm:** khi fusion **sau** khi backbone đã tạo xong representation, EHR có
bổ sung tín hiệu cho ảnh X-quang không — và câu trả lời có phụ thuộc vào việc ảnh được
biểu diễn bằng **một CLS đã pool** hay **1369 patch token** không?

**Trả lời:** EHR bổ sung cho CLS (+0.0081, p=0.004) nhưng **không** bổ sung cho patch
(+0.0003, p=0.830). Kết quả nhất quán với giả thuyết rằng phần lớn gain quan sát được
trên CLS đến từ việc **bù thông tin mất trong global pooling**.

---

## 1. Kết quả

Test macro mAP, 3 seed (42/1337/2024), paired, mean ± CI95:

| head | mô tả | mAP |
|---|---|---|
| H1 | CLS + MLP | 0.4117 ± 0.0008 |
| H2 | CLS + MLP + **concat EHR** | 0.4199 ± 0.0027 |
| H3 | CLS + ML-Decoder (seq 1) | 0.4068 ± 0.0010 |
| H4z | CLS + token rỗng | 0.4003 ± 0.0069 |
| H4 | CLS + **EHR token** | 0.4052 ± 0.0008 |
| H5 | **Patch** + ML-Decoder | 0.4336 ± 0.0032 |
| H6z | Patch + token rỗng | 0.4330 ± 0.0044 |
| H6 | Patch + **EHR token** | 0.4339 ± 0.0064 |

| | Δ | p | |
|---|---|---|---|
| **repr gain** (H5 − H1) | **+0.0219** | 0.002 | ✅ patch mạnh hơn CLS |
| **EHR @ CLS** (H2 − H1) | **+0.0081** | 0.004 | ✅ EHR bổ sung cho CLS |
| **EHR @ patch** (H6 − H5) | **+0.0003** | 0.830 | ❌ không đóng góp |
| `I_content` = (H6−H6z) − (H4−H4z) | −0.0039 | 0.071 | chưa phân biệt được |

---

## 2. Ba phép thử cơ chế khác nhau, cùng kết luận

| # | Phương pháp | Kết quả |
|---|---|---|
| 1 | Train EHR token end-to-end (H6 − H5) | **+0.0003**, p = 0.830 |
| 2 | Can thiệp nhân quả: tráo EHR sang bệnh nhân khác | delta **0.0000**, corr(logit) = **0.999934** |
| 3 | Linear stacking hậu nghiệm (bỏ qua cơ chế token) | **−0.0027** |

**Ba phép thử có cơ chế khác nhau và hội tụ về cùng kết quả — nhưng chúng vẫn chia sẻ
cohort, 13 nhãn, EHR feature set, frozen image backbone và test set.** Do đó chúng củng
cố lẫn nhau về mặt cơ chế fusion, chứ không phải ba quan sát độc lập theo nghĩa thống kê.

### Đường 3 đo gì, và không đo gì

Stacking kiểm tra:

$$\hat{y} = h(\operatorname{logit}_{H5},\, e)$$

Nó trả lời: *"sau khi H5 đã tạo dự đoán, EHR embedding có bổ sung tín hiệu tuyến tính
không?"* → **không**.

Nó **không** kiểm tra:

$$Z_l = B_l(Z_{l-1};\, e)$$

tức trường hợp EHR thay đổi **quá trình hình thành** representation: patch-to-patch
interaction, hình thái feature, attention không gian theo lớp, thông tin được bảo toàn
qua block cuối.

Vì vậy đường 3 là **bằng chứng mạnh chống lại late fusion và additive post-hoc
complementarity**, nhưng **không** trực tiếp kiểm tra conditional feature extraction bên
trong backbone.

Giá trị của nó vẫn lớn ở một điểm: nó dùng encoder của **H2** — đã chứng minh trích được
tín hiệu EHR thật (shuffle delta +0.0134) — nên loại trừ được phản biện *"H6 thất bại chỉ
vì encoder của nó kém"*. Hai encoder độc lập (H2 train cho CLS, H6 train cho patch) đều
không cho tín hiệu giúp được model patch **qua đường late fusion**.

### Chuỗi nhân quả đo từng mắt xích

| | d(e) | d(token) | d(logit) | truyền token→logit |
|---|---|---|---|---|
| H2 (concat, CLS) | 0.326 | — | 0.0862 | — |
| H4 (token, CLS) | 0.260 | 0.188 | 0.0472 | 0.25 |
| H6 (token, patch) | 0.114 | 0.102 | **0.00094** | **0.0092** |

Token của H6 **có** biến thiên theo bệnh nhân (`std = 0.555`, `‖W_proj‖ = 6.16`, gấp 10×
H4) và **có** được chú ý (134× uniform, 9% norm đóng góp) — nhưng logit gần như bất biến.

**Phát biểu đúng:** *ảnh hưởng của EHR đặc thù bệnh nhân bị triệt tiêu hoặc không được
truyền hiệu quả tới logits khi frozen patch representation đã được cung cấp.* Dữ liệu
không phân biệt được giữa: cancellation giữa các attention head · output projection triệt
tiêu đóng góp · residual path lấn át · feature lâm sàng không khớp decision boundary ·
model học representation EHR nhưng không dùng nó để đổi dự đoán.

---

## 3. Kết luận được giữ

1. Patch representation mạnh hơn global CLS rõ rệt trong pipeline hiện tại (+0.0219).
2. EHR cải thiện global CLS representation (+0.0081, xác nhận nhân quả bằng shuffle).
3. EHR **không** cải thiện frozen patch representation qua late EHR-token fusion.
4. Shuffle EHR gần như không đổi logits của H6 (corr 0.999934).
5. Linear stacking không tìm thấy residual EHR gain trên H5.
6. Late fusion trên frozen patch output **không còn là hướng ưu tiên**.
7. Microbiology, vitals, comorbidity không tạo gain đáng kể ngoài labs trong cohort này.
8. ASL và image representation là hai đòn bẩy lớn hơn EHR late fusion.

### Xếp hạng ưu tiên kỹ thuật (cùng thang, cùng cohort)

```
Đổi representation (CLS → patch)  : +0.0219
Đổi loss (DB Loss → ASL)          : +0.0144
Thêm EHR (trên CLS)               : +0.0081
Thêm EHR (trên patch)             : +0.0003
Vi sinh, ngoài labs               : −0.0002
```

---

## 4. Giá trị phương pháp luận

Nhiều công trình multimodal fusion đo EHR gain **trên embedding đã pool** (CLS, global
average pooling). Kết quả này cho thấy con số đó **có thể** phần lớn phản ánh việc bù cho
bottleneck của chính image encoder, thay vì tính bổ sung thật của modality thứ hai.

Đây là một cảnh báo có giá trị — nhưng phát biểu ở dạng *"có thể"*, vì nó dựa trên một
cohort, một target, một backbone.

---

## 5. Những gì kết quả này **KHÔNG** loại trừ

Đây là phần bị viết thiếu trong bản trước.

1. **EHR-conditioned feature extraction bên trong backbone.** Toàn bộ thí nghiệm fusion
   **sau** một backbone đã đóng băng. Trường hợp EHR can thiệp vào các block sâu để thay
   đổi cách representation hình thành **chưa được kiểm tra**.
2. **Tương tác phi tuyến mà linear stacker không phục hồi được.**
3. **EHR cho các target khác.** Nhãn CheXpert được NLP trích từ **chính report của phim
   đó**, nên ảnh gần như theo định nghĩa là đủ. Kết luận này **không** nói gì về tiên
   lượng, tử vong, chuyển ICU, tái nhập viện, đáp ứng điều trị.
4. **Cohort/backbone khác.** Một image encoder yếu hơn có thể vẫn thấy EHR gain thật.

---

## 6. Đề xuất (đã thu hẹp phạm vi)

**Không tiếp tục đầu tư rộng** vào late fusion hoặc EHR feature engineering cho tác vụ
này: thêm token sau backbone, cross-attention chỉ ở output patch, mở rộng microbiology /
vitals / comorbidity, tinh chỉnh thêm FiLM/concat trên frozen representation.

**Chỉ cân nhắc một thí nghiệm deep-conditioning cuối cùng**, được giới hạn và nêu trước,
để kiểm tra giả thuyết EHR-conditioned feature extraction (xem
`PLAN_deep_ehr_conditioning.md`). Nếu `D2 − D1 < +0.002` hoặc seed-CI chứa 0 → dừng hướng
EHR fusion cho 13 nhãn CheXpert.

**Trọng tâm chính vẫn nên là:** image representation (attention pooling, partial
unfreeze), loss design (ASL đã thắng; RAL là TODO từ §B8), calibration, external
validation.

---

## 7. Caveat

1. **Test set đã được dùng thích nghi.** Test đã dùng để đánh giá H1–H6, xem per-label AP,
   hình thành giả thuyết Fracture/Pneumonia, và quyết định có thử deep fusion. **Mọi thí
   nghiệm tiếp theo trên cùng test set là exploratory extension, không phải confirmatory
   proof.** Để có claim confirmatory cần một subject-level holdout chưa từng xem, hoặc
   external cohort.
2. **Chỉ đúng cho 13 nhãn CheXpert** — xem §5.3.
3. **Stacker tuyến tính, một lần fit, không CI.** `−0.0027` gần như chắc chắn là nhiễu của
   stacker, **không** đọc thành "EHR có hại".
4. **`I_content` chưa phân biệt được ở mức seed** (p=0.071). Bootstrap theo subject cho CI
   `[−0.0062, −0.0004]`, nhưng bootstrap chỉ nói *"với đúng 3 model này, khác biệt nhất
   quán giữa các bệnh nhân"* — nó **không** thay thế biến thiên do khởi tạo. Với 3 seed,
   seed-CI là ràng buộc, và nó chưa đủ power.
5. **Fracture (+0.0142) và Pneumonia (+0.0066)** là hai nhãn duy nhất dương trong stacking.
   Chọn nhãn *sau khi* nhìn kết quả, không CI, 2/13 nằm trong tỉ lệ ngẫu nhiên — **không
   claim được**.
6. **"Frozen" = đã supervised fine-tune trên 13 nhãn ở 04b rồi mới đóng băng**, không phải
   RAD-DINO pretrained gốc. Gap train−val `+0.042` nằm trong đó; cả 8 head dùng chung nên
   triệt tiêu trong mọi delta, nhưng ảnh hưởng diễn giải khả năng khái quát.

---

## 8. Các chỗ đã sửa so với bản trước

| Bản trước | Bản này |
|---|---|
| "Ba đường bằng chứng **độc lập**, ba cơ chế **không chia sẻ giả định nào**" | "Ba phép thử **cơ chế khác nhau**, vẫn chia sẻ cohort/target/backbone/test set" |
| Stacking là "đường quyết định **loại trừ lỗi kiến trúc**" | Stacking là bằng chứng mạnh **chống lại late fusion**, không kiểm tra conditional feature extraction |
| "EHR gain **chủ yếu chỉ** vá bottleneck CLS" | "**nhất quán với giả thuyết** rằng phần lớn gain đến từ bù global pooling; chưa loại trừ trực tiếp deep conditioning" |
| "Model **chủ động bỏ qua** EHR" | "Ảnh hưởng bị **triệt tiêu hoặc không truyền hiệu quả** tới logits" |
| "**Dừng đầu tư** vào EHR fusion" | "Không đầu tư rộng vào **late fusion**; cân nhắc **một** thí nghiệm deep-conditioning nêu trước" |
| (không có) | **§7.1 — test set đã dùng thích nghi; thí nghiệm tiếp theo là exploratory** |
| (không có) | **§5 — những gì kết quả KHÔNG loại trừ** |

---

## 9. Artifact

```
token_fusion/
├── manifests/image_manifest.parquet
├── checkpoints/{H1,H2,H3,H4z,H4,H5,H6z,H6}_s{42,1337,2024}_best.pt   # 24 file
├── metrics/
│   ├── ap_test_s{42,1337,2024}.npz
│   ├── attn_{H4,H6}.csv
│   ├── ehr_shuffle_diagnostics.csv
│   └── interaction_bootstrap.csv
└── experiment_manifest.json
```

Notebook: `06b_frozen_token_fusion.ipynb` (41 cell), 8 head trên **một** frozen backbone
forward dùng chung — 8×3 run riêng sẽ tốn ~30 GPU-giờ, cách này ~7.5h.
