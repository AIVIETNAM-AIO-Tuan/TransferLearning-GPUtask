# REVIEW — Tái cấu trúc repo theo CLAUDE.md §4

*2026-09-10. Đề xuất để Tuấn duyệt. **Chưa di chuyển file nào.***

## Hiện trạng vs yêu cầu

```
hien trang (TransferLearning&GPUtask/)        CLAUDE.md §4
  109 notebook .ipynb o goc                     experiments/<huong>/notebooks/
   54 markdown .md o goc                        docs/ + experiments/<huong>/README.md
    3 script .py o goc (kiem_thutu_ast.py...)   tools/ hoac src/
  figs_selected/ results_08/ results_09/         experiments/<huong>/outputs/
  paper/                                         docs/papers/
```

Ba loại lệch, mức rủi ro khác nhau khi di chuyển:

| Loại | Rủi ro gãy | Vì sao |
|---|---|---|
| Notebook `.ipynb` | **Thấp** | Mọi đường dẫn dữ liệu trong notebook là **Drive tuyệt đối** (`/content/drive/MyDrive/MIMIC_CXR/planB_out/...`), không tham chiếu file repo. Di chuyển không đổi hành vi chạy. |
| Markdown | **Vừa** | JOURNAL/PLAN/REVIEW tham chiếu nhau bằng tên (`PLAN_STAGE4 §A5.2f`, `REVIEW_CXRLT_STAGE0.md`) và tham chiếu notebook bằng số (`nb61`). Tên số giữ nguyên nên chỉ gãy link dạng `[x](file.md)`. |
| Script `.py` | **Thấp** | `kiem_thutu_ast.py` nhận đường dẫn tuyệt đối. |

## Đề xuất cấu trúc

```
TransferLearning&GPUtask/
  CLAUDE.md · Event.md · Progress.md · README.md
  tools/
    kiem_thutu_ast.py            kiem thu tu dinh nghia (xuyen cell)
    dryrun_*.py                  harness chay that tren du lieu gia (chuyen tu scratchpad)
  docs/
    JOURNAL.md                   nhat ky ky thuat (giu nguyen ten, chi doi cho)
    plans/    PLAN_*.md
    reviews/  REVIEW_*.md · DECISION*.md · AMENDMENTS*.md
    reports/  REPORT_*.md
    papers/   2506.07984v1.pdf va cac paper khac
  experiments/
    00_cohort_stage0/            28*-34*   inventory, stage anh, view, EHR base, task2
    01_baseline_probe/           35*-37*   ASL, Case C, ML-Decoder 1369 token
    02_backbone_finetune/        38*       38a/b/c/d
    03_ehr_fusion/               39, 45-47, 52*, 53, 55-57   Tier1, prior film, deep cond, XGB
    04_ensemble_epoch/           48, 58, 59, 61, 67          epoch/seed ensemble, blend, quet
    05_cnn_member/               41a-41f, 64b, 64c           ConvNeXt probe -> cnn_full -> cnn_ms
    06_r2a_full_data/            64, 64b                     gop lat chon
    07_resolution/               49, 51, 62, 65, 66, 68, 68b do phan giai, feature map, cascade
    08_synthesis/                (chua co) diffusion §4.5
    _archive/                    notebook cua cohort cu (truoc 2026-08-04): 01-27
    moi thu muc: notebooks/ · outputs/ · README.md (gia thuyet, cong, ket qua, trang thai)
```

Mỗi `README.md` của experiment ghi: giả thuyết, cổng khoá trước, kết quả, trạng thái
(baseline / proposed / ablation / prototype / **deprecated**) — đúng §5.

## Ba việc phải làm KÈM khi di chuyển (không thì vi phạm §2 System-Wide Impact)

1. **Sửa link markdown** `[x](file.md)` sau khi đổi chỗ — grep toàn bộ `*.md` cho `](.*\.md)`.
2. **`64c_R2a_cnn_ms.ipynb` chưa tồn tại** dù JOURNAL nhắc — ghi rõ "chưa dựng" trong README của `05_cnn_member`, đừng để thành file mồ côi ngầm.
3. **Đánh dấu deprecated** thay vì xoá: `41c` C3-bis (kho 518), `55` gating, `39` deep cond, `51` (độ phân giải ở suy luận), `66` (cascade — phép thử mất sức phân giải). Chúng được `Event.md`/`JOURNAL` tham chiếu nên không xoá (§11).

## Thứ tự thực thi an toàn

```
1. git status sach; tag `pre-restructure`
2. git mv (giu lich su) theo bang tren — notebook truoc, markdown sau
3. sua link markdown; chay grep kiem khong con link gay
4. viet README.md cho tung experiments/<huong>
5. graphify update . ; doc GRAPH_REPORT.md
6. Progress.md ghi "Critical Changes: da tai cau truc"
```

Ước ~2–3 giờ, 0 GPU. **Không** làm chung với bất kỳ lượt chạy GPU nào đang mở notebook từ repo.

## Câu hỏi cần Tuấn chốt

- Đồng ý cấu trúc `experiments/<số>_<hướng>/` như trên, hay muốn nhóm khác (theo stage 0–5 của PLAN)?
- `JOURNAL.md` chuyển vào `docs/` hay giữ ở gốc cạnh `Event.md`? (memory rule của tôi trỏ vào JOURNAL theo tên, không theo chỗ — di chuyển được.)
- Notebook cohort cũ 01–27 vào `_archive/` hay giữ trong `experiments/` với nhãn deprecated?
