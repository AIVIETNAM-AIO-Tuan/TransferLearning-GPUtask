# Nâng notebook 11 (mid-vs-late screen) từ labs-only → Full EHR

## Context

Notebook `11_midvslate_screen.ipynb` test giả thuyết "EHR conditioning ở block 5–7 cứu được
deep conditioning" (mà 07 đã null ở block 10–11). Bản hiện tại chỉ dùng **labs** — kế thừa
từ 07. Nhưng loại micro dựa trên "micro≈0 ở 06" là **mâu thuẫn**: micro≈0 đo trong *late
fusion*, còn 11 tồn tại chính vì *deep conditioning ≠ late fusion*. Hơn nữa `org_name` (micro)
là feature **chỉ-đích chẩn đoán** — đúng loại context có thể steer attention — nên nó là
feature *phù hợp nhất* cho cơ chế đang test, không phải kém nhất.

Người dùng chọn **Full EHR** (labs + micro org/spec + vitals + como). Lift **nguyên** máy móc
EHR của `06b` (đã chạy đúng) là vừa quyết định nhất (null với context chỉ-đích = dứt khoát)
vừa an toàn nhất (copy verbatim, không mổ encoder). Decoder giữ **frozen mặc định** + quy tắc
leo thang sang unfrozen nếu C11 cho chữ ký "decoder không dùng được rep".

Nguồn code lift (đã xác minh exact qua exploration): `build_06b.py`
- vocab + data-prep: C1 dòng 191–194, C2 dòng **209–255**
- Dataset/collate/to_dev: C4 dòng **328–380**
- `EhrEncoder` (multi-modal): dòng **456–493** + `Act = nn.GELU`

## Cách tiếp cận — sửa `build_11.py` theo từng cell

Chỉ sửa `build_11.py` (script sinh notebook). Giữ nguyên toàn bộ khung 4-arm / cổng kép /
forward_shared+tail / CondAdapter / train loop / C10-C11 — chỉ thay lớp EHR.

**C1 (cohort + data-prep):** thay khối labs-only bằng data-prep đầy đủ của 06b (verbatim
dòng 191–194 + 209–255). Thêm vào cạnh phần `TIER` (per-tier report) và `man`/`EHR_ROW` đã
có. Kết quả: các mảng `lab_val/mask, vit_val/mask, como_arr, set_id/rec_id/stat_id,
rec1_v/m, rec3_v/m`, dict `binv, cnts, bags` (ragged), hằng `BIN, COUNTS, RAGGED, REC1,
REC3, MICRO_TIERS, ORG_V, SPEC_V, VITALS, COMO_COLS, MISSING`. `man` của 11 đã có `ehr_row`
→ tương thích Dataset 06b.

**C2 (dataset):** thay `ImgDS/collate/loader/to_dev` labs-only bằng bản đầy đủ 06b (dòng
328–380). Điểm quan trọng: ragged bag collate thành tuple `(cat, offsets)` cho EmbeddingBag;
`to_dev` phải giữ tuple (`isinstance(v, tuple)`); `perm` áp cho **mọi** feature qua `e`, `y`
luôn từ `ehr_row` thật (shuffle diagnostic đúng miễn phí). Giữ `shuffle_perm` của 11.

**C4 (encoder):** thay `EhrEncoderLabs` bằng `EhrEncoder` đầy đủ của 06b (dòng 456–493) +
`Act = nn.GELU`. Giữ `CondAdapter` nguyên. Encoder consume các key: `set_id, lab_val,
lab_mask, vit_val, vit_mask, como, stat_id, t1_same_encounter, rec1_v, rec1_m, t2_pending,
t2_time_imprecise, t2_n_pending, rec_id, rec3_v, rec3_m, t3_n_specimens` + 4 ragged
`(t1_org, t1_spec, t2_spec, t3_org)`. Output `d_e=128`.

**C5 (arms) — THAY ĐỔI CHÍNH, mặt lỗi lớn nhất:**
- `ArmModel.enc = EhrEncoder()` (thay EhrEncoderLabs).
- Nhánh `cond`: `e = self.enc(batch)`.
- Nhánh `uncond`: `e = self.enc(NULL_BATCH(B, dev))` — **helper mới** dựng batch "mọi thứ
  missing" (constant, param-matched control). Thay cho `{'lab_val':z,'lab_mask':z}` cũ.
- `build_arms` giữ `_copy_pos` (adapter theo vị trí) + `enc.load_state_dict` (encoder cùng
  key nên load thẳng được). Param-match vẫn giữ (cả 4 dùng cùng EhrEncoder).

**NULL_BATCH(B, dev) — spec chính xác** (dựng "bệnh nhân không có EHR", hằng qua mọi sample):
- dense zeros: `lab_val/mask [B,len(LABS)]`, `vit_val/mask [B,len(VITALS)]`, `como
  [B,len(COMO_COLS)]`, `rec1_v/m [B,len(REC1)]`, `rec3_v/m [B,len(REC3)]`
- scalar-float zeros [B]: `t1_same_encounter, t2_pending, t2_time_imprecise, t2_n_pending,
  t3_n_specimens`
- categorical id = 0 [B] long: `set_id, stat_id, rec_id` (index hợp lệ trong Embedding)
- ragged (cat, offsets): mỗi sample **một token MISSING** → `cat = full(B, MISSING, long)`,
  `offsets = arange(B, long)`. EmbeddingBag trả `emb[MISSING]` — hằng qua sample. (Dùng
  MISSING thay bag rỗng để tránh edge-case EmbeddingBag input rỗng.)
- Kết quả: `enc(NULL_BATCH)` cho `e` **giống hệt mọi sample** ⇒ đúng unconditional control.

**C6 (gates):** giữ identity step-0 (up=0 ⇒ ΔZ=0 ⇒ cả 4 arm == D0, đúng bất kể `e`). Mở
rộng gate "EHR đọc feature": ngoài đổi `lab_val`, thêm **đổi một ragged micro bag** →
`e` phải đổi (chứng minh micro thật sự vào encoder). Thêm gate: `enc(NULL_BATCH)` hằng qua
sample (std≈0 giữa các hàng), và `enc(real) != enc(NULL_BATCH)`.

**C8 (S0), C9 (train), C10 (cổng kép), C11 (diagnostics):** **không đổi logic** — batch giờ
nhiều key hơn nhưng luồng giữ nguyên; `forward_shared` + per-arm backward không phụ thuộc
key EHR. S0 vẫn kiểm grad adapter>0 / grad decoder=0; nay thêm grad **encoder**>0. Shuffle ở
C11 dùng `perm` đã tráo mọi feature. Per-tier đã có sẵn.

## Smoke-test TRƯỚC khi giao (bắt buộc — bản này có ragged bag + NULL_BATCH, nhiều mặt lỗi)

Chạy local với **fake backbone** (transformers/ml_decoder không có local — stub như các
smoke trước), kiểm:
1. `EhrEncoder` forward với batch giả đủ key + ragged `(cat,offsets)` → `[B,128]`, không lỗi.
2. `NULL_BATCH(B)` → `enc` cho `e` **constant** giữa các hàng; `enc(real) != enc(null)`.
3. collate ragged → `(cat, offsets)` đúng; EmbeddingBag nuốt được; `to_dev` giữ tuple.
4. identity step-0: cả 4 arm (MID+LATE) `|logit − D0| = 0` với up zero-init.
5. `build_arms` `_copy_pos` + `enc` copy → 4 arm cùng param count.
6. grad routing: adapter + **encoder** nhận grad, decoder frozen = 0.
7. gate "đổi micro bag → e đổi".

Data-prep column names **chỉ kiểm được trên ehr thật** (không có local) — C1 sẽ assert-fail
rõ ràng nếu thiếu cột; đây là rủi ro tồn dư, giảm bằng cách lift **verbatim** từ 06b (đã chạy).

## Verification (trên Blackwell, sau khi giao)

Thứ tự chạy: `C0 → C0-bis → C1 (data-prep, in `labs N | vitals N | como ... | tiers`) →
C2 → C3 (gate G0 manual==stock) → C4 → C5 (param-match OK) → C6 (identity 0, EHR đọc
labs+micro, NULL_BATCH constant) → C8 (S0: grad adapter/encoder>0, decoder=0)`.

Gửi lại output **C1 + C5 + C6 + C8** để xác nhận trước khi bỏ ~1 ngày vào C9. Rồi
`S0=False → C9 (3 seed × 4 arm) → C10 (G_mid, G_late, I_location, per-tier, decision
matrix) → C11 (real-vs-shuffle, logit sensitivity, adapter activity)`.

Quyết định giữ nguyên preregistration: cổng chạy confirmatory = `G_mid ≥ +0.002`;
`I_location` chọn claim; điều kiện cần `real > shuffle`. **Leo thang decoder:** nếu C11 cho
"rep đổi nhưng logit không đổi" → lật `FREEZE_DECODER=False` chạy lại, không kết luận null.

## Rủi ro & giảm thiểu

- **Schema/column-name** (đã dính 2 lần phiên này): lift verbatim từ 06b đã-chạy; C1 assert
  từng cột; không tự đặt tên.
- **NULL_BATCH sai key/shape** → uncond arm lỗi hoặc `e` không constant: smoke #2 bắt trực tiếp.
- **EmbeddingBag input rỗng**: né bằng MISSING-token-per-sample thay vì bag rỗng.
- **Encoder lớn hơn → OOM**: negligible so với backbone forward; `probe_batch` đã tự dò.
- **Checkpoint phụ thuộc**: `lpft_v2/raddino_mld_db_ft_best.pt` (04b) + `token_fusion/
  checkpoints/H5_s42_best.pt` (H5) — kiểm tồn tại trước (C0/C3/C5 assert).
