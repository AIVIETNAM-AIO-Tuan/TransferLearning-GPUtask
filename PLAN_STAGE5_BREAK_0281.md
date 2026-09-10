# PLAN STAGE 5 — từ 0.2560 tới trên 0.281 (mốc vô địch CXR-LT Task 1)

*Lập 2026-08-15. Nối tiếp `PLAN_STAGE4_IMAGE_THEN_EHR.md`. Trạng thái sống: `JOURNAL.md`.*

## 0 · Mốc hiện tại và hình dạng khoảng cách

```
kien truc chot   t1_prior_both = ML-Decoder + Tier1(EHR) + Tier1Img(phim truoc)
                 tren backbone `ft` 38a/a2_E3, co LayerScale trong Tier1Img (r3)
MOC HIEN TAI     val 40 nhan 0.2631 | tail 0.1139     (58: ensemble epoch `all` x 2 seed)
  (moc cu 0.2560 / tail 0.1077 la 1 EPOCH DON x 2 seed — noi giam ~0.0138)
test (theo doi)  0.2504
vo dich Task 1   0.281 (test)   head 0.567 | medium 0.263 | tail 0.136
tran nhieu nhan  0.6528 (Task 2, 406 anh chuyen gia); vo dich dat 81% muc do
san nhieu        macro ±0.0007 | tail ±0.0011–0.0023  (tren val 39,293 anh)
                 hieu GHEP CAP o tail: CI nua-rong ~±0.005 -> hieu ung +0.005 la SAT NGUONG
```

Khoảng cách **không đều**: tail `0.1139` vs `0.136` (còn 0.022, đã co 43% nhờ `58`). Vô địch hơn ta ở **mọi** tier ⇒ không phải
một mẹo, mà là **nền tảng biểu diễn** + **xử lý long-tail**.

Quy tắc k-số học vẫn áp: một đòn bẩy chạm `k` nhãn phải nhấc mỗi nhãn `0.0007 × 40/k` mới vượt nhiễu.

---

## 1 · Nhóm RẺ-CHẮC — làm trước, rủi ro gần 0

Cộng lại kỳ vọng **+0.010 … +0.015**. Không cái nào cần ý tưởng mới.

| # | việc | chi phí | bằng chứng |
|---|---|---|---|
| R1 | **Thêm seed cho `t1_prior_both` → ensemble** | ~10 h/seed trên RTX PRO 6000 | `48` đo ensemble seed/checkpoint **+0.0103**, không tốn GPU ngoài giờ huấn luyện |
| R2 | **Lượt chốt trên 100% train** (gộp lại lát chọn, và val nếu nhắm nộp) | 1 lượt | mọi số hiện tại đến từ mô hình chỉ thấy **74%** dữ liệu có nhãn |
| R3 | **TTA lúc suy luận** — KHÔNG lật (ràng buộc lâm sàng của Tuấn: ảnh lật = situs inversus, phi thực tế); chỉ xoay nhỏ ±5° / crop-scale nhẹ, trung bình 3–5 biến thể | ~3–5× thời gian infer | chưa từng thử; kỳ vọng thấp hơn bản có lật |

**R1** — dùng **trung bình hạng** song song trung bình xác suất: AP là chỉ số xếp hạng và các seed
lệch calibration.

**R2** — epoch-cuối protocol tồn tại đúng để làm bước này: `E=10` đã chốt trên lát chọn (`53` đo
bình nguyên ep8–10, chênh 0.0005) nên gộp lát chọn lại **không** phá kỷ luật. Đây cũng là cách xoá
chú thích *"số test là CẬN DƯỚI"* khỏi mọi báo cáo.

---

## 2 · Đang chờ cổng — đừng dựng thêm gì chồng lên

| notebook | câu hỏi | cổng |
|---|---|---|
| `56` | deep conditioning trên kiến trúc chính | `G_deep > +0.004` **và** `tail(CỘNG THÊM) ≥ −0.003` |
| `41c → 41b` | ConvNeXtV2 pretrain PadChest có đủ mạnh **và** đủ khác không | `corr < 0.90` **và** `≥ 0.85 × RAD-DINO`, ensemble `≥ +0.005` |

`41a` đã dạy: đa dạng có (corr 0.52) nhưng thành viên yếu (probe 0.149 vs 0.2042) thì ensemble
**âm** (−0.0035). `41c` là bước sửa **độ mạnh** mà giữ **độ khác**.

---

## 3 · BỐN HƯỚNG ĐỘT PHÁ — nhắm 0.281, không phải +0.005

### #1 · Mở băng block 8–11 — **ĐÃ ĐÓNG 2026-08-18** (`57`)

> ```
> G_ft   s42 -0.0071 | s43 -0.0103      cong < +0.002 -> DONG
> drift  0.1325, bao hoa tu ep5         block THICH NGHI THAT nhung khong giup
> wnorm  1.0090 (>1)                    khong phai bao mon — gia thuyet weight decay SAI
> ```
> Phản biện *"RAD-DINO đã thấy MIMIC"* được xác nhận: đặc trưng `38a` đã đủ thẳng hàng.
> Phần dưới giữ nguyên làm hồ sơ lý lẽ. Sản phẩm phụ đáng giữ: **LayerScale trong `Tier1Img`**
> dập được cú sụp (0/16 epoch-arm, trước đó 3 lần liên tiếp) — đã đưa vào `53` và `57`.



**Lý lẽ.** Đòn bẩy lớn nhất từng đo là fine-tune: `04b` LP 0.3191 → FT 0.4201 = **+0.101**. Ta đã
ăn phần đó ở `38a` — nhưng backbone được fine-tune với **đầu đọc đơn giản**, rồi đông cứng, và toàn
bộ `Tier1` / `Tier1Img` / ML-Decoder học **trên đặc trưng chết**.

```
38a  fine-tune voi dau doc DON GIAN roi dong cung  -> dac trung toi uu cho DAU DOC CU
ta   doc bang ML-Decoder + Tier1 + Tier1Img        -> chua bao gio duoc thich nghi voi cach doc nay
```

Riêng nhánh **phim trước** là thứ `38a` hoàn toàn không biết: đặc trưng chưa từng bị ép mã hoá
*"cái gì thay đổi giữa hai phim"* — `Tier1Img` đang phải tự xoay xở trên đặc trưng tĩnh.

**Phản biện đã nêu và trả lời** *(người dùng, 2026-08-15)*: *"MIMIC nằm trong corpus pretrain của
RAD-DINO rồi, mở ra chắc không ý nghĩa"*. Đúng một nửa — **quen miền** thì đúng (khoảng cách miền
≈ 0, mở băng không mua được gì từ đó), nhưng **thẳng hàng với nhiệm vụ** là chuyện khác: mục tiêu
DINO là bất biến augmentation, không có khái niệm nào về việc tách `Lobar Atelectasis` khỏi
`Atelectasis`. Chính `04b` là phản ví dụ — nó đo +0.101 **trên đúng dataset RAD-DINO đã thấy**.
Nhưng phần lớn +0.101 ấy **ta đã ăn rồi**, nên đòn bẩy còn lại hẹp hơn nhiều: chỉ là phần *"thích
nghi với đầu đọc mới + nhánh phim trước"*.

**Cách làm.** Mở block 8–11 với LR riêng nhỏ 10× trong chính vòng huấn luyện `t1_prior_both`. `56`
đã trả một nửa chi phí (đã tính gradient qua block 8–11 cho hai arm adapter) ⇒ chỉ đổi
`requires_grad` của các block đó thành `True`.

**Đừng chạy chung lượt với `56`** — hai lever cùng đụng block 8–11, gộp lại thì không quy được công.

```
cong    +0.004 macro, doc theo tier, va KHONG duoc lam tut tail
chi phi ~2x luot thuong (~20 h/seed)
rui ro  quen kien thuc pretrain; fine-tune sau hon = tien gan hon NHAN NHIEU (goi thua 5x)
uoc     day la ung vien DUY NHAT con lai co tiem nang +0.01–0.02 mot minh
```

---

### #0 · **Ensemble epoch trong cùng lần chạy** — làm TRƯỚC, 0 giây GPU

Thêm sau `57`. Không phải một hướng đột phá, mà là **điều kiện tiên quyết để đo** #2.

`57` cho thấy `tail` của một epoch đơn dao động `±0.03` giữa các epoch, và **hai seed cho hai quỹ
đạo ngược nhau** ⇒ không chọn được `E` từ dữ liệu. Đo một hiệu ứng `+0.005` của #2 trên nền dao
động `±0.03` là vô vọng; ensemble làm **co nhiễu của chính cái mốc** mà #2 được so vào.

`17` đã đo trên cohort cũ (ensemble 3 checkpoint cùng một lần chạy): macro `+0.00112` CI loại 0,
worst3 `+0.00345` CI loại 0, **tail `+0.00465`** — mức tăng lớn nhất nằm đúng ở tail. Cần đo lại
trên 11 nhãn tail của CXR-LT.

Chi phí: đọc `p_{arm}_{seed}_ep*.npy` đã lưu sẵn trong `stage4_nb53/r3` và `stage4_nb57/r3`. Không
huấn luyện lại. Dùng trung bình **hạng** song song trung bình xác suất (AP là chỉ số xếp hạng).

### #2 · Tách đôi: biểu diễn instance-balanced, classifier class-balanced

**Đây là đòn TẤN CÔNG vào tail, không phải phòng thủ.** (Món phòng thủ là `nb55` gating.)

**Lý lẽ.** Ta đã sửa long-tail bằng **loss** (ASL) nhưng chưa bao giờ bằng **lấy mẫu**. Nhãn hiếm
không thiếu *biểu diễn* — backbone nhìn `Lobar Atelectasis` bằng đúng đôi mắt nhìn mọi nhãn khác.
Nó thiếu **cập nhật classifier**: 129 ca dương / 258,871 ảnh ⇒ `GroupFC` của nhãn đó nhận ~129
gradient dương mỗi epoch, so với hàng chục nghìn của nhãn head. Classifier tail **chưa học xong**,
chứ không phải *không học được*.

Kang et al. (decoupling) đo đi đo lại: giữ biểu diễn học với phân bố **tự nhiên** (điều ta đã có),
rồi huấn luyện lại **riêng classifier** với sampler cân bằng lớp.

**Cách làm.** Đông cứng mọi thứ từ checkpoint `s42.pt`, chỉ huấn luyện lại `GroupFC` (+ query
attention của ML-Decoder) vài epoch với sampler cân bằng lớp.

```
cong    tail +0.005  VA  macro khong tut qua 0.002
        (ve thu hai vi lay mau can bang co tac dung phu da biet: day tail len keo head xuong)
chi phi ~1–2 h tren checkpoint SAN CO
rui ro  gan 0 — khong dung den backbone, khong dung den du lieu moi
```

Vì macro = trung bình 40 nhãn: tail +0.010 × 11/40 ⇒ macro +0.0028. Tail nhích thì macro nhích theo.

---

### #3 · Ensemble dị chủng — **CỔNG BẬC 2 ĐẠT 2026-08-19** (`41d`)

> ```
> 41c probe  +0.0097 (nen 0.2042) | bac 1 +0.0027 | bac 2 `cnn_full` +0.0051 (nen 0.2627, DAT)
> corr 0.7463 (MOC 0.8236) | moc sau blend ~0.2678 | tail -0.0009 (trong nguong)
> ```
> `cnn_full` (CNN + ML-Decoder + Tier1Img + Tier1) vừa mạnh nhất vừa KHÁC nhất.
> **3a CHỐT** (41e): blend +0.0043 CI [+0.0016,+0.0068] LOẠI 0 — vùng xám theo cổng +0.005,
> DÙNG trong ensemble cuối. **3b ĐÓNG** (41f): đa tỉ lệ không cho thành viên mạnh hơn
> (ep-matched 0.2398 < 0.2435); blend tương đương (+0.0050), tail/corr nhỉnh hơn nhưng dưới sàn.
> Còn nợ: blend BA CHIỀU (0 GPU) để chọn 1 hay 2 thành viên CNN cho bản chốt.



**Lý lẽ.** `41a` đã dạy bài học đắt: đa dạng có nhưng thành viên yếu ⇒ ensemble **âm**. Vô địch
CXR-LT dùng **fine-tune + ensemble**, và ensemble của họ gần chắc chắn là dị chủng.

**Thang leo.**

```
41c  (DANG CHAY)  pretrain ConvNeXtV2-nano tren PadChest, 30 nhan, cohort CXR-LT 2026 142,928 anh
                  -> cong: corr voi RAD-DINO + mAP probe
ban day du        hop kho PadChest + CheXpert + NIH (kieu Ark) — corpus ~3x, nhan da dang hon
```

Cả ba kho đều **ngoài MIMIC** ⇒ không rò rỉ (split test CXR-LT là tập con MIMIC-CXR). Đây là kỷ
luật đã khoá ở `PLAN_STAGE4` §4 và không được nới.

**Ra quyết định theo `41c`:**

```
truot vi DO MANH (probe thap)  -> them kho la DUNG THUOC, leo len ban day du
truot vi CORR CAO              -> ca huong DONG, khoi ton cong hop kho
qua ca hai cong                -> 41b: fine-tune + ensemble voi RAD-DINO
```

```
chi phi 1–2 ngay GPU cho ban day du
```

---

### #3-mo-rong (tuỳ chọn, CUỐI CÙNG) · ConvNeXt to hơn làm thành viên

**Điều kiện kích hoạt** — chỉ chạy nếu SAU #4 + R1 + R2a mà độ mạnh thành viên CNN vẫn là ràng
buộc chặt nhất. Lý do xếp cuối (2026-08-20): (1) kích thước chưa từng là nút thắt — `41a` đo
nano ≈ tiny, `41c` cho thấy vấn đề là MIỀN; (2) độ dốc đã đo ~+0.0017 blend cho mỗi +0.023 thành
viên ⇒ `base` lạc quan cũng chỉ ~+0.001 blend, giá ~30 h; (3) anh em cùng nhà không thêm độ khác
(các arm CNN của `41d` corr cao tới mức trộn 3 arm không hơn arm đơn). Nếu chạy: `convnextv2_base`
(KHÔNG `large` — 198M trên 140k ảnh pretrain PadChest là gần chắc overfit; nano đã overfit từ ep5),
pretrain E ngắn + augment nặng hơn.

### #4 · Chưng cất nhãn mềm — hướng DUY NHẤT nâng TRẦN

**Vấn đề gốc.** Nhãn CXR-LT do **NLP trích từ report**, cho cả train / val / **test**. Không phải
chuyên gia đọc — 377,110 ảnh × 40 nhãn ≈ 15 triệu quyết định nhãn. Phần chuyên gia chỉ là tập vàng
**~406 ảnh** (Task 2), và chính nhờ nó `34` đo được: nhãn tự động gọi thừa `Pneumonia` **5×**
(115 ca → chuyên gia giữ 22, thêm 0).

Tức trong train có hàng vạn ảnh mang nhãn cứng `y=1` mà trên phim **không có gì**, và một phần dung
lượng mô hình đang dùng để **học thuộc nhiễu**.

**Ý tưởng.** Nhãn nhiễu **không nhất quán** giữa các ảnh; tín hiệu thật thì nhất quán. Nên một
ensemble đã huấn luyện, khi dự đoán lại chính tập train, cho xác suất **gần sự thật hơn nhãn gốc**
ở đúng các ca dán nhầm:

```
anh X: report ghi "r/o pneumonia" -> NLP dan y=1, nhung phim sach
teacher (ensemble cua ta) du doan p=0.15
target moi = 0.7*y_goc + 0.3*p_teacher = 0.745      <- mem hon, gan su that hon
```

**Cách làm.** (1) ensemble seed sẵn có làm **giáo viên**, infer một lượt trên 258k ảnh train —
chỉ tốn inference; (2) huấn luyện lại **học sinh** y hệt cấu hình, đổi target thành
`λ·y + (1−λ)·p_teacher`; (3) so trên lát chọn.

```
lambda   0.6–0.8  (nhan goc phai con la MO NEO, tranh thien kien tu xac nhan)
cong     +0.004 macro, doc theo tier
chi phi  1 luot suy luan + 1 luot huan luyen lai
```

**Méo mó phải nhớ khi đọc số.** Distillation sửa target lúc *huấn luyện*, nhưng *thước đo* (val)
**vẫn nhiễu**. Mô hình học được sự thật tốt hơn sẽ bị val nhiễu **chấm thấp đi** ở đúng các ca dán
nhầm ⇒ gain đo trên val **thiên lệch xuống**. Nếu vẫn dương qua cổng thì càng đáng tin; nếu lửng lơ
vùng xám thì tập vàng 406 ảnh là trọng tài **tham khảo** (đọc, không quyết — nó thuộc quỹ đạo test).

---

## 4 · Thứ tự thực thi

```
dang chay   56 (deep conditioning)  ·  41c (stage PadChest)
             |
sau do      #2 classifier can bang       ~2 h      rui ro ~0, doc duoc ngay
            R1 them seed                 ~10 h/seed  chay nen, khong can quyet dinh gi
            R3 TTA                       1 cell
            #1 mo block 8–11             ~20 h     ung vien +0.01–0.02 duy nhat con lai
            #3 hop kho + 41b             1–2 ngay GPU
            #4 chung cat nhan mem        1 luot infer + 1 luot huan luyen
             |
luot chot   R2: 100% train (+val neu nop) · ensemble nhieu seed + nhieu backbone · TTA
            -> MOT lan cham test
```

Cộng nhóm rẻ-chắc (+0.010–0.015) với **#1 hoặc #2** thành công là đủ chạm vùng **0.27–0.28**.
Nhưng 0.281 của vô địch đo trên **test** với pipeline nộp bài hoàn chỉnh — muốn so sòng phẳng thì
lượt chốt cuối phải đủ cả ba thành phần ở dòng cuối.

---

## 5 · Kỷ luật giữ nguyên

- **Cổng khoá trước** cho mọi thí nghiệm; sàn nhiễu 0.005 cho mọi so sánh với bảng xếp hạng.
- **Chọn trên lát chọn (15% train theo subject), không bao giờ trên val.** Test chỉ suy luận để
  theo dõi; mọi lần chạm ghi `_test_touches.jsonl`.
- **Đọc theo tier**, không chỉ macro. Nhiều lever nhấc macro mà cắn tail — `39` đo `tail −0.0050`.
- **Đối chứng khớp tham số** (`*_zero` / `*_U` dùng hằng số học được, không phải vector 0).
- **Không dùng checkpoint đã học CÓ GIÁM SÁT trên MIMIC-CXR** (rò rỉ test).
- **Dry-run hai thế giới trước khi deploy.** `41c` hỏng ba lượt liên tiếp vì tôi vá mù — mọi
  notebook khác đều có dry-run, riêng nó thì không. Đã sửa; đừng lặp lại.

## 6 · Đã đóng — đừng quay lại

| hướng | bằng chứng đóng |
|---|---|
| tăng độ phân giải | `51`: `lb672 − lb518 = −0.0028` (âm 9/10 epoch); FOV có giá thật dưới attention (ngoại vi −0.0104) |
| đổi bộ mã hoá EHR (XGB / FT-Transformer) | `53`: `nội dung(xgb) − nội dung(col) = +0.0027` < cổng +0.005 ⇒ encoder **không** phải nút thắt |
| cross-attention thay FiLM ở deep conditioning | `39`: `G_xattn − G_film = −0.0003` ⇒ **hình dạng không quan trọng** |
| multi-view `w_l` theo bệnh | `17` C7/V1: Δmacro +0.00017, CI chứa 0, tail −0.0016 |
| FPN đa tỉ lệ trên ViT | ViT thuần không có phân cấp stride; muốn đa tỉ lệ thì lấy qua ConvNeXt (#3) |
| mở rộng micro / vitals / comorbidity | `06`: D3_12 − D3_labs = −0.0002 (p=0.901) |
| imputer / generator cho EHR thiếu | missingness là **cấu trúc** (MIMIC-IV không có vital ngoại trú), không phải MAR |
