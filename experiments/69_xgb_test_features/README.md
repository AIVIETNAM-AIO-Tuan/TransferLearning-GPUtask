# 69 · Khối 40 cột XGB — đo mức phụ thuộc, rồi khôi phục cho TEST

## Giả thuyết

`52a` **không** dự đoán trên test. Nó điền hằng số:

```python
# 52a C2, dòng cuối
OOF[TEST] = _prev[None, :].astype('float32')   # prevalence trên lát khớp
```

Sau logit + chuẩn hoá trong `64` C1, cả **78.946 hàng test giống hệt nhau** ở 40 cột đó.
Nghĩa là trên test, 40 trong 257 cột EHR mang **đúng bằng không thông tin**, trong khi train
và val đều có dự đoán thật.

Đây là bất đối xứng train/val ↔ test, đã nằm trong **mọi con số từ `53` trở đi**.

Lý do được ghi trong `52a` là giữ kỷ luật *"test chưa hề chạm"*. Nhưng mô hình XGB huấn luyện
trên lát khớp; cho nó suy luận trên test **không đọc nhãn test**. Đó là suy luận bình thường,
không phải chọn trên test. Nên đây là lựa chọn quá thận trọng, và cái giá chưa từng được cân.

## Cách đo

| Cell | Làm gì | Chạm test? |
|---|---|---|
| C0–C4 | verbatim `64` (config, EHR, ảnh, kiến trúc, arms) | không |
| **C5** | một lượt backbone trên val, nhiều lượt đầu đọc | **không** |
| **C6** | báo cáo + cổng | **không** |
| C7 | huấn luyện lại 40 XGB, dự đoán test, ghi **file mới** | không (chỉ suy luận) |

C5 khai thác một tính chất: **patch token không phụ thuộc EHR**. Nên backbone chỉ chạy một lần
mỗi lô, còn hai bản đặc trưng chỉ khác nhau ở đầu đọc. Ba đường được chấm:

```
tpb · EHR THẬT          = val hiện tại
tpb · 40 cột HẰNG SỐ    = thứ test đang thực sự nhận
mldec · KHÔNG có EHR    = sàn, không dùng EHR nên miễn nhiễm
```

Chênh lệch giữa hai dòng đầu chính là phần test đang mất.

## Cổng

```
CI95 dưới  >  0.0007  ->  có tín hiệu, chạy C7
CI95 trên  <  0.0007  ->  ĐÓNG, khỏi chạy C7
chồng lấn  0.0007     ->  tăng MAX_CKPT rồi đo lại
```

`0.0007` là sàn nhiễu lặp lại thật, đo từ bản sao `41e`.

## Quan trọng khi chạy C7

- Huấn luyện trên **lát khớp của `52a`** (`TR & ~SL`, ~219.785 ảnh), **không** dùng train R2a.
  Val đã được chấm bằng mô hình huấn luyện trên đúng lát này. Nếu test được chấm bởi mô hình
  khoẻ hơn thì test lại lệch khỏi val theo chiều ngược lại, và val hết vai trò dự báo.
- Ghi ra **file mới** `xgb_oof_co_test.npy`. Không đè `xgb_oof.npy`: mô hình đã huấn luyện với
  giá trị cũ ở hàng train/val, và ghi đè qua Drive FUSE thì chỉ lần ghi đầu xuống được đĩa.
- C7 có cổng khẳng định **chỉ hàng test đổi**, mọi hàng khác giữ nguyên từng bit.
- C7 **không** tính AP của XGB trên test. Làm vậy là một lần chạm; để dành cho lượt đọc test
  cuối, đọc một thể cả bảng.

## Sau C7

Trong `64` C0, trỏ `P_OOF` sang file mới rồi chạy lại C1 và `predict(I_TE)` cho cả 20 file
đường ống. `64b` cũng đọc cùng file nên thành viên CNN cần lượt tương tự.

## Trạng thái kiểm thử

| | |
|---|---|
| `ast.parse` từng cell | sạch |
| `kiem_thutu_ast.py` | 0 vấn đề |
| Dry-run dữ liệu giả, 0 GPU | **8/8 PASS** |
| Chạy thật | **chưa** |

Dry-run chạy thật phần thêm của C1, `head_forward` trên kiến trúc thật, toàn bộ C6, và toàn bộ
C7 với XGB thay bằng stub. Có hai phép kiểm nghịch: C1 phải **nổ** khi giả định hằng số sai, và
`tpb` phải cho kết quả **y hệt** khi `Tier1` còn zero-init (đúng thiết kế) nhưng **khác** sau khi
trọng số được mở ra khỏi 0.

C0/C2/C4 là verbatim `64` đã chạy xong nên không mô phỏng lại.
