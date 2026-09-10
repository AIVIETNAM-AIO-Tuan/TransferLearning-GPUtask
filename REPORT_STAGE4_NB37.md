# REPORT · `nb37` — ML-Decoder(1369 token) + fusion EHR theo bệnh

RAD-DINO **đông cứng**, ảnh `lb518` stage xuống đĩa local, tăng cường `RandomResizedCrop(.85–1.0)` +
xoay `7°`. Loss `asl40`, **không early-stop, lấy epoch cuối** (kỷ luật `35` §0.1). Val 39,293 ảnh /
23,800 study. Mọi arm chạy trên **một forward backbone dùng chung**.

---

## 0 · Bốn arm ỔN ĐỊNH NHẤT qua các lần chạy

Chỉ 7 arm có mặt ở **cả** v1 và v2 nên chỉ chúng có bằng chứng ổn định. 5 arm còn lại (`col_cat`,
`col_gate`, `t1`, `t1_zero`, `t1t2`) mới chỉ chạy **một** lần.

### Mốc — không tranh hạng, chúng là gốc toạ độ

| arm | v1 `E=12` | v2 `E=8` | vai trò |
|---|---|---|---|
| `mean_lin` | 0.1925 | 0.1893 | mốc của trục **đầu đọc** — `mldec − mean_lin` đo từ đây |
| `mldec` | 0.2312 | 0.2405 | mốc **không EHR** — mọi `arm − mldec` đo từ đây |
| `mldec_gate_zero` | 0.2342 | 0.2389 | đối chứng **khớp tham số** — `G_ehr = gate − gate_zero` |
| `mldec_t1_zero` | — | 0.2383 | đối chứng khớp tham số cho **B2a** |

### Arm EHR — ba arm ỔN ĐỊNH (có mặt ở cả hai lần chạy)

| hạng | arm | v1 | v2 | hạng v1→v2 | so `mldec` (v2) | cơ chế |
|---|---|---|---|---|---|---|
| **1** | `mldec_gate_k1` | **0.2379** | 0.2435 | 1 → 1 | **+0.0030** | 1 token = `MLP(cả 247 cột)` + cổng `g_c`. "Attention" **suy biến** — `W_Q`,`W_K` chết |
| **2** | `mldec_cat` | 0.2350 | 0.2429 | 2 → 2 | +0.0024 | bố cục **gốc của kế hoạch**: `K/V = [1369 patch ; 9 token họ]` |
| **3** | `mldec_gate` | 0.2348 | 0.2426 | 3 → 3 | +0.0021 | attention EHR riêng + `g_c`, 9 họ |

**Thứ hạng 1–2–3 giữ nguyên** qua hai lần chạy khác nhau ở `E`, số arm, cách clip gradient và `LR`
của `g_c`. Và **bố cục `cat` thắng `gate` ở cả hai lần** (`cat` 2 vs `gate` 3).

### Arm chỉ có MỘT phép đo (v2) — cao nhất bảng nhưng chưa lặp lại

| arm | v2 | so `mldec` | ghi chú |
|---|---|---|---|
| `mldec_col_cat` | **0.2437** | **+0.0032** | 247 token cột + 1 token tổng hợp, bố cục `cat`. **Chứa `k1` như trường hợp riêng** — attention chỉ cần dồn hết lên token tổng hợp là tái lập `k1` |
| `mldec_t1t2` | 0.2434 | +0.0029 | B2a + B2c. Nhưng `t1` một mình chỉ `+0.0004` ⇒ giá trị chủ yếu từ **tầng 2** |
| `mldec_t1` | 0.2409 | +0.0004 | B2a một mình. `t1 − t1_zero = +0.0026` (đối chứng xác nhận) |
| `mldec_col_gate` | 0.2402 | −0.0003 | **thất bại**: `corr(logit) = 1.000000`, cổng sập về không |

> **Mang sang `38`: `col_cat`.** Không phải vì `0.2437 > 0.2435` (chênh `0.0002`, vô nghĩa) mà vì nó
> **chứa `k1`** về mặt hàm số, và nó là giao của **bố cục đã lặp lại** (`cat` > `gate`, hai lần) với
> **bộ token tốt nhất** trong lần chạy duy nhất thử nó.

> ⚠ **Chưa có phép đo biên độ seed nào cho `37`.** `35` đo `±0.0007` cho probe tuyến tính; ở đây mỗi
> cấu hình đúng một lần chạy. Khoảng cách giữa 4 arm đầu là `0.0030`, nên **thứ hạng nội bộ chưa
> phân định được** — chỉ có tính lặp lại của thứ tự là đáng tin.

---

## 1 · Kết quả lớn nhất: ML-Decoder thay `mean(1369 token)`

```
mldec - mean_lin      v1 +0.0387    v2 +0.0512       cong dat truoc: +0.010
theo tier (v2)        head +0.0499  medium +0.0493   tail +0.0554
```

Gấp **4–5 lần cổng**, **đều trên cả ba tier**, **lặp lại ở hai lần chạy**. Lớn hơn toàn bộ những gì
`35` đo được cộng lại (loss `+0.0009` · EHR `+0.0020` · phi tuyến `0`).

Nó cũng đóng một câu hỏi mở của `35`: `patch − cls = +0.0377` **không** phải trần của trục biểu diễn.
**Phép trung bình** mới là nút thắt, không phải bản thân patch token.

### Vị trí so với vô địch

```
                        anh      tail     cach vo dich (anh / tail)
32  (BCE @30 epoch)   0.2028     —          1.39x  /  3.30x
35  (asl40, probe)    0.2093   0.0620       1.34x  /  2.19x
37  v2 tot nhat       0.2437   0.0982       1.15x  /  1.39x
vo dich Task 1        0.281    0.136
```

Khoảng cách **tail** từ `3.30×` xuống `1.39×` — mà backbone **vẫn đông cứng, chưa fine-tune lớp nào**.

---

## 2 · EHR: thật, nhỏ, đã xác minh bằng cơ chế

```
G_ehr      = gate - gate_zero    +0.0038      tail +0.0077
dung luong = gate_zero - mldec   -0.0016      <- tham so them vao LAM HAI
```

`dung lượng` **âm** ⇒ toàn bộ phần tăng đến từ **nội dung** EHR. Không còn chỗ ghi công nhầm cho tham
số — đúng cái mà cặp đối chứng `H4z`/`H6z` của `06b` sinh ra để bắt. (v1: `dung lượng +0.0031` tức
83% của `gate − mldec` là do tham số.)

### Phép tráo nhân quả tách sạch thật với giả

Tráo EHR sang bệnh nhân khác, **giữ nguyên nhãn**:

```
mean_lin_ehr 0.9325 · t1 0.9415 · t1t2 0.9508 · cat 0.9583 · col_cat 0.9704 · k1 0.9761
    -> dung EHR THAT
gate 0.9959                          -> gan nhu khong dung
col_gate 1.000000 · t1_zero 1.000000 -> KHONG dung chut nao
moc 06b (H6): 0.999934
```

`col_gate` là arm **duy nhất thua `mldec`** (`−0.0003`), và `corr = 1.000000` giải thích tại sao:
248 token + cổng `g_c` **sập về không**. Nhưng `col_cat` với **cùng** 248 token lại đứng đầu bảng
v2 — nên thiết kế `247+1` chạy được, chỉ **không hợp bố cục `gate`**.

### Case C đã chết

```
lift EHR:   mean-pool -> ML-Decoder
  v1        +0.0021   -> +0.0036
  v2        +0.0021   -> +0.0022
```

Phép gộp tốt hơn **không** nuốt mất đóng góp của EHR. Khớp tỉ số `1.12` của `35`. Đóng qua hai lần chạy.

### B2a (tầng 1, `Q=patch`) có tác dụng

`t1 − t1_zero = +0.0026` với `corr 0.9415` (đối chứng khớp tham số), và `t1t2 − t1 = +0.0025` nên
hai tầng **cộng dồn**, không nuốt nhau. Một phép đo, chưa lặp lại.

---

## 3 · Định tuyến theo bệnh: đúng ĐỘ, sai NHÃN

`LR(g_c) × 20` sửa được phần "độ": `g_c` trải `±0.19` thay vì `±0.03`, `G_ehr` từ `+0.0006` → `+0.0038`.

Nhưng cổng vẫn mở cho **sai nhãn**:

```
rho(lift_cat, lift_35) = +0.478  p=0.002   <- thuoc do HOP LE giua hai che do doc anh
rho(g_c, lift_cat)     = +0.148  p=0.362   <- g_c van khong tuong quan
rho(g_c, lift_35)      = -0.041  p=0.802
```

Ma trận `[40 × 9]` **sụp** về `ctx/demo/mask/n_lab` với gần như **cùng một hàng cho mọi nhãn**;
`lab`, `vital`, `dh_lab` đều `~0.01–0.03`.

> **Cổng giúp bằng cách CHỈNH ĐỘ LỚN của EHR theo từng bệnh, không phải bằng cách CHỌN phần nào của
> EHR cho bệnh nào.** Nửa sau của disease-aware chưa chạy được.

---

## 4 · Ba lỗi đã tìm ra và sửa trong quá trình

**Điểm yên ngựa chết.** `h = h_img + g_c·pe(h_ehr)` với **cả** `g_c = 0` **và** `pe` zero-init cho
`∂L/∂g_c = pe(h_ehr) = 0` và `∂L/∂pe ∝ g_c = 0` — nhánh EHR chết vĩnh viễn. Dry-run bắt được.
Sửa: chỉ `g_c = 0` (mẫu ReZero), `pe` khởi tạo thường.

**`W_Q`, `W_K` chết ở `K=1`.** Softmax trên một key luôn bằng `1.0` bất kể điểm số, nên
`∂a/∂s ≡ 0` và `524,288` tham số nằm nguyên ở init. Xác nhận bằng số: `W_Q/W_K std = 0.02957/0.02959`
khớp tới **5 chữ số** với `0.03125 × exp(−λ·Σlr) = 0.029579` — tức chỉ bị weight decay co lại.
`W_V 0.04505` thì đã di chuyển. Đối chứng: `K=9` cho `W_Q 0.06774`, `W_K 0.06155` (gấp đôi init).
⇒ `mldec_gate_k1` thực chất là **FiLM có cổng theo lớp**, không phải attention; và nó vẫn thắng.

**Clip gradient TOÀN CỤC ghép các arm lại.** `clip_grad_norm_` trên toàn bộ tham số tính một chuẩn
toàn cục rồi chia đều; với `n` arm độc lập chuẩn đó ≈ `√n ×` chuẩn một arm, nên clip kích hoạt thường
hơn và kéo cả `n` arm xuống cùng lúc — phá vỡ khẳng định "tương đương toán học với chạy riêng"
(`AMENDMENTS` ①). Sửa: clip **từng arm**, kèm biến đếm tỉ lệ kích hoạt.

**Cột nhị phân bị z-score.** `29b` one-hot top-K vi khuẩn (`micro_org_*`) + cờ comorbidity ⇒ ~92/247
cột là nhị phân. z-score biến giá trị "có" thành `√((1−p)/p)` — `p = 1/219,785` cho **468**. Probe
tuyến tính chịu được; **attention thì không** (score tỉ lệ `M²`). Sửa: cột nhị phân giữ `0/1`,
`LayerNorm` trên token `col`, kẹp `|z| ≤ 10` làm chặn cuối.

---

## 5 · Hai dự đoán của tôi bị bác

| tôi nói | thực tế |
|---|---|
| `cat` sẽ bị nhấn chìm (EHR chỉ `~0.6%` attention lúc init, đúng bố cục `06b` H6) | `cat` **bằng hoặc hơn** `gate` ở **cả hai** lần chạy, và `corr` cho thấy nó dùng EHR **nhiều hơn** (`0.958` vs `0.996`) |
| `lift_35` là thước sai vì đo ở chế độ mean-pool | **Bác** bởi chính phép đo tôi đề xuất: `rho(lift_cat, lift_35) = +0.478`, `p = 0.002` |

---

## 6 · Những gì kết quả này KHÔNG loại trừ

**Số/loại token EHR.** `K ∈ {1, 9, 248}` đã thử; `K=250` kiểu FT-Transformer thuần chưa. Và mọi
khác biệt giữa chúng nằm trong `0.0035` **không có thang nhiễu** — chưa phân định được.

**Biểu diễn EHR.** One-hot top-K vi khuẩn là biểu diễn sai cho biến phân loại đuôi dài;
**`EmbeddingBag`** (mỗi vi khuẩn một vector, cộng các vector có mặt) là đúng, và cohort cũ đã dùng.
Sửa được thì phải đụng `29b` và dựng lại `cxrlt_ehr.parquet`. Đây là **thay đổi duy nhất còn lại có
khả năng đổi bậc độ lớn** của nhánh EHR.

**Backbone.** RAD-DINO **đông cứng**, chưa fine-tune (`38`). Cohort cũ đo `LP 0.3191 → FT 0.4201 =
+0.101` — lớn hơn mọi thứ ở đây.

**Đích.** Nhãn CXR-LT do NLP trích từ **chính report của phim đó** ⇒ ảnh gần như đủ theo định nghĩa.
Không con số nào ở đây nói gì về tiên lượng, tử vong, chuyển ICU, tái nhập viện.
