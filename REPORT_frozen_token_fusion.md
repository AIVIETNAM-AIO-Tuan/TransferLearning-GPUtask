# Kết luận thí nghiệm 06b — Frozen CXR representation × EHR fusion

**Câu hỏi:** EHR bổ sung **tín hiệu mới** cho ảnh X-quang, hay chủ yếu **vá lại thông tin đã mất** khi ảnh bị nén thành một vector CLS?

**Trả lời: chủ yếu là vá bottleneck của CLS.** Khi biểu diễn ảnh giữ được cấu trúc không gian (patch token), đóng góp của EHR biến mất.

---

## 1. Kết quả chính

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

**Các delta then chốt:**

| | Δ | p | phán quyết |
|---|---|---|---|
| **repr gain** (H5 − H1) | **+0.0219** | 0.002 | ✅ patch mạnh hơn CLS rõ rệt |
| **EHR @ CLS** (H2 − H1) | **+0.0081** | 0.004 | ✅ EHR bổ sung cho CLS |
| **EHR @ patch** (H6 − H5) | **+0.0003** | 0.830 | ❌ không đóng góp gì |
| `I_content` = (H6−H6z) − (H4−H4z) | −0.0039 | 0.071 | chưa phân biệt được (3 seed) |

---

## 2. Ba đường bằng chứng độc lập — cùng một kết luận

Đây là phần quan trọng nhất về mặt phương pháp: kết luận không dựa vào một phép đo.

| # | Phương pháp | Kết quả |
|---|---|---|
| 1 | Train EHR token end-to-end (H6 − H5) | **+0.0003**, p = 0.830 |
| 2 | Bất biến của model đã train: tráo EHR sang bệnh nhân khác | delta **0.0000**, corr(logit) = **0.999934** |
| 3 | Stacking hậu nghiệm — **bỏ qua hoàn toàn cơ chế token** | **−0.0027** |

Ba cơ chế hoàn toàn khác nhau, cùng một câu trả lời: **với patch representation, EHR không thêm gì.**

### Vì sao đường số 3 mang tính quyết định

Phản biện hiển nhiên cho đường 1 và 2 là: *"có lẽ EHR vẫn hữu ích, chỉ là một token chìm giữa 1369 patch nên model không học được"* — tức **lỗi kiến trúc**, không phải kết luận về EHR.

Đường 3 loại bỏ khả năng đó. Nó **không dùng token, không dùng attention, không dùng 1/1370**: lấy encoder của **H2** (đã chứng minh trích được tín hiệu EHR thật — shuffle delta +0.0134), rồi hỏi trực tiếp bằng logistic regression *"cho trước dự đoán của H5, `e` có cải thiện được không?"*. Fit trên val, đánh giá trên test.

Kết quả: **không**.

Và vòng lặp khép kín theo cả hai chiều — ta có **hai** encoder độc lập:
- encoder của H2, train để bổ sung cho **CLS**
- encoder của H6, train để bổ sung cho **patch**

**Không cái nào** cho ra tín hiệu EHR giúp được model patch.

### Chuỗi nhân quả được đo từng mắt xích

| | d(e) | d(token) | d(logit) | truyền token→logit |
|---|---|---|---|---|
| H2 (concat, CLS) | 0.326 | — | 0.0862 | — |
| H4 (token, CLS) | 0.260 | 0.188 | 0.0472 | 0.25 |
| H6 (token, patch) | 0.114 | 0.102 | **0.00094** | **0.0092** |

Token của H6 **có** biến thiên theo bệnh nhân (`std = 0.555`, `‖W_proj‖ = 6.16` — gấp 10× H4), **có** được chú ý (134× uniform, 9% norm đóng góp). Nhưng logit gần như bất biến. Model **chủ động bỏ qua** thông tin EHR khi đã có patch — không phải vì không học được, mà vì không có gì để dùng.

---

## 3. Giá trị thực nghiệm

### 3.1. Kết luận cũ phải viết lại

**Cũ (từ 06):** *"EHR bổ sung cho CXR, +0.013."*

**Mới:** *"EHR bổ sung cho một biểu diễn **CLS đã pool** (+0.0081, p=0.004, xác nhận nhân quả bằng shuffle +0.0134). Gain đó **không sống sót** khi biểu diễn ảnh giữ cấu trúc không gian: với patch token, EHR đóng góp +0.0003 (p=0.83)."*

### 3.2. Đây là cảnh báo phương pháp luận, không chỉ là một số âm

Rất nhiều công trình multimodal fusion đo EHR gain **trên embedding đã pool** (CLS, global average pooling). Kết quả này cho thấy con số đó có thể phần lớn là **đang vá bottleneck của chính image encoder** — chứ không phải bằng chứng về tính bổ sung của modality thứ hai.

Đây là finding **mạnh hơn** một kết quả dương yếu, vì nó đặt câu hỏi vào toàn bộ một lớp thí nghiệm.

### 3.3. Xếp hạng ưu tiên kỹ thuật (cùng thang đo, cùng cohort)

```
Đổi representation (CLS → patch)  : +0.0219   ← lớn nhất
Đổi loss (DB Loss → ASL)          : +0.0144
Thêm EHR (trên CLS)               : +0.0081   ← bốc hơi khi ảnh mạnh
Thêm EHR (trên patch)             : +0.0003
Vi sinh, ngoài labs               : −0.0002
```

**Hàm ý:** đầu tư vào **biểu diễn ảnh** và **hàm loss** cho lợi ích lớn hơn hẳn đầu tư vào fusion EHR. Riêng việc đổi loss (một dòng code) đã ăn đứt toàn bộ pipeline EHR.

### 3.4. Các kết quả phụ đã xác lập

- **Vi sinh không thêm gì ngoài labs** (D3: −0.0002, p=0.901). Toàn bộ pipeline tier T1–T4 chống-leakage không cải thiện dự báo. *Nhưng công đó không phí* — chính nó cho phép khẳng định null này là **thật**, không phải artifact của leak (cohort cũ có `frac_future_organism = 0.448`).
- **Vitals, comorbidity: ≈ 0** (D6 +0.0000, D7 +0.0010).
- **FiLM ≈ concat** (p=0.308) — cơ chế fusion nhân không hơn cộng.
- **ASL > BCE-posweight > DB Loss**, thắng cả ba tier. DB Loss thua ở chính lý do nó được chọn (§B8 chọn nó *vì* tail).

---

## 4. Vì sao đủ thuyết phục để sửa kế hoạch

Bốn lý do, xếp theo sức nặng:

**(1) Ba phương pháp độc lập hội tụ.** Train end-to-end, can thiệp nhân quả (shuffle), và stacking hậu nghiệm — ba cơ chế không chia sẻ giả định nào — đều cho cùng kết luận. Một kết quả null từ một phép đo có thể là lỗi; từ ba thì không.

**(2) Phản biện "lỗi kiến trúc" đã bị loại trừ bằng thiết kế.** Đây là điểm mạnh nhất. Stacking không dùng token/attention, và dùng encoder **đã chứng minh là tốt**. Nếu EHR còn thông tin, phép thử này phải bắt được.

**(3) Có control cấu trúc.** `H4z`/`H6z` (token hằng số) tách **chi phí cơ học của việc chèn token** khỏi **giá trị nội dung lâm sàng**. Không có chúng, `H6 − H5` gộp hai thứ và không diễn giải được.

**(4) Có neo tái hiện.** `H2 − H1 = +0.0081` khớp chiều và độ lớn với `06` (+0.0081→+0.0105 tùy loss). Thí nghiệm mới **tái hiện được** kết quả cũ trước khi bác bỏ cách diễn giải nó — nên sự khác biệt đến từ **representation**, không phải từ pipeline khác.

---

## 5. Caveat — phải ghi vào report

1. **Chỉ đúng cho 13 nhãn CheXpert.** Đây là chẩn đoán **hình ảnh**, và nhãn được NLP trích từ **chính report của phim đó**. Ảnh gần như theo định nghĩa là đủ. EHR vẫn có thể quyết định cho bài toán mà ảnh **không** trả lời được: tiên lượng, đáp ứng điều trị, tử vong, tái nhập viện.
2. **Stacker là tuyến tính, một lần fit, không có CI.** `−0.0027` gần như chắc chắn là nhiễu của stacker, **không** đọc thành "EHR có hại".
3. **`I_content` chưa phân biệt được ở mức seed** (p=0.071). Bootstrap theo subject cho CI `[−0.0062, −0.0004]` (loại trừ 0), nhưng bootstrap chỉ nói *"với đúng 3 model này, khác biệt nhất quán giữa các bệnh nhân"* — nó **không** thay thế được biến thiên do khởi tạo. Với 3 seed, seed-CI mới là ràng buộc, và nó nói chưa đủ power.
4. **Fracture (+0.0142) và Pneumonia (+0.0066)** là hai nhãn duy nhất dương trong stacking. Hấp dẫn về mặt lâm sàng, nhưng **chọn nhãn sau khi nhìn kết quả**, không CI, 2/13 nằm trong tỉ lệ ngẫu nhiên — đúng cái bẫy đã giết giả thuyết Pneumonia ở D3. **Không claim được** nếu chưa có phép thử nêu-trước riêng.
5. **"Frozen" ở đây = đã supervised fine-tune trên 13 nhãn ở 04b rồi mới đóng băng**, không phải RAD-DINO pretrained gốc. Gap train−val `+0.042` vẫn nằm trong đó. Cả 8 head dùng chung nên nó **triệt tiêu trong mọi delta**, nhưng ảnh hưởng cách diễn giải khả năng khái quát ra ngoài.

---

## 6. Đề xuất sửa kế hoạch

**Dừng đầu tư vào EHR fusion cho tác vụ này.** Bằng chứng đã đủ mạnh và đủ nhiều đường.

**Chuyển trọng tâm sang:**

| hướng | cơ sở |
|---|---|
| **Biểu diễn ảnh** (patch, attention pooling, unfreeze 2–4 block cuối, ConvNeXt V2) | +0.0219 — đòn bẩy lớn nhất đã đo được |
| **Hàm loss** (ASL đã thắng; thử RAL — TODO từ §B8) | +0.0144 từ một dòng code |
| **Đổi tác vụ** nếu vẫn muốn theo hướng EHR | Chỉ hợp lý khi nhãn **không** trích từ report của chính phim đó |

**Không nên làm** (trừ khi có lý do mới): grouped EHR token (4–8 token), cross-attention riêng cho EHR, mở rộng feature EHR. Đường bằng chứng số 3 đã cho thấy vấn đề **không** nằm ở cách nhét EHR vào.

---

## 7. Artifact

```
token_fusion/
├── manifests/image_manifest.parquet
├── checkpoints/{H1,H2,H3,H4z,H4,H5,H6z,H6}_s{42,1337,2024}_best.pt   # 24 file
├── metrics/
│   ├── ap_test_s{42,1337,2024}.npz          # per-label AP, 8 head × 3 seed
│   ├── attn_{H4,H6}.csv                     # attention mass + value contribution
│   ├── ehr_shuffle_diagnostics.csv          # real vs shuffle
│   └── interaction_bootstrap.csv            # 2000 subject-level resamples
└── experiment_manifest.json
```

Notebook: `06b_frozen_token_fusion.ipynb` (41 cell). Toàn bộ chạy trên **một** frozen backbone forward dùng chung cho 8 head — 8×3 run riêng sẽ tốn ~30 GPU-giờ, cách này ~7.5h.
