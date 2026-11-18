# -*- coding: utf-8 -*-
"""Hằng số của phép thử định vị (PLAN_90 §11). KHÔNG chứa ID Drive: chỉ TÊN thư mục / tệp; notebook tìm qua khối Drive chuẩn.

Tên thư mục trên Drive là giả định cho tới lượt chạy đầu (bộ nhớ dự án: tên gõ trong chat có thể sai) — mỗi tên chỉ khai MỘT lần ở đây."""

FOLDER = 'stage4_nb90'                 # thư mục ghi của module 90 (dưới cxrlt_2024)
ENCODER = 'b'                          # spec của gói vindr_bench: lượt r78x_J_v3b_dp_ema_v2_skip7_s42, ep5, trọng số EMA (= ArrE s42)
SIZE, PATCH, GRID = 518, 14, 37        # ảnh 518, lưới token 37 x 37
LETTERBOX = 1022                       # kho 518 = ảnh letterbox 1022 (28c) thu về 518 -> khung chuẩn hoá = toạ độ letterbox 1022 / 1022 (78g dòng 228)
N_REGIONS, N_LABELS = 36, 40

# ---- tệp trên Drive (tên; sha256 ghim = bản PhysioNet đã kiểm ở máy 17/11) ----
# Chest ImaGenome 1.0.0: bản sao dưới cxrlt_2024 (đọc trước, có thể có cấp '1.0.0') rồi bản gốc dưới My Drive — như 78f.
IMAGENOME_COPY = ('chest-imagenome',)                                                        # tính từ cxrlt_2024
IMAGENOME_ORIGINAL = ('MIMIC_CXR', 'chest-imagenome', 'physionet.org', 'files', 'chest-imagenome', '1.0.0')   # tính từ My Drive
GOLD_BOX = ('gold_dataset', 'gold_bbox_coordinate_annotations_1000images.csv', 'bfd58b0da6bb9fef89a2530c8810db09f4353b95d20c0b087e1b4561dc265a27')
AUTO_BOX = ('gold_dataset', 'auto_bbox_pipeline_coordinates_1000_images.txt', 'ff5d70ebe1424f334c9fdfa5bc483e34faa4c4464ff208684ea935dcc08397e5')
REGION_LIST = ('semantics', 'objects_detectable_by_bbox_pipeline_v1.txt', 'e84d70cf39f02c030b0878933e716f088964b6da8b413b41422a666302d7377b')
# MS-CXR 1.1.0 (Tuấn tải lên 17/11, "cùng tên thư mục"): My Drive/MIMIC_CXR/ms-cxr
MSCXR_DIR = ('MIMIC_CXR', 'ms-cxr')
MSCXR_FILE = ('MS_CXR_Local_Alignment_v1.1.0.csv', 'ba328facb36e92376c716b3a5d0e069b77a438eb1b73657b0e8eda631ee6f394')
MSCXR_COLUMNS = ('dicom_id', 'category_name', 'x', 'y', 'w', 'h', 'image_width', 'image_height')   # KHÔNG có 'label_text' (câu report)
MSCXR_TEXT_COLUMN = 'label_text'
# kích thước ảnh gốc của ảnh gold: metadata MIMIC-CXR-JPG (Columns x Rows), như 78f C3
META_DIR = ('MIMIC_CXR', 'mimic-cxr-jpg-2.1.0.physionet.org')
META_FILE = 'mimic-cxr-2.0.0-metadata.csv.gz'
ORDER_PATH = ('embeddings', 'emb_index.parquet')    # dưới cxrlt_2024: bảng ảnh của mọi notebook 78 (dicom_id, subject_id, study_id, split)
STORE_DIR = ('cxrlt_518_ready',)                    # dưới cxrlt_2024: kho ảnh 518 (các tệp .tar, mỗi ảnh <dicom_id>.jpg)
SPLITS = ('train', 'val', 'test')

# ---- số đếm ghim (đếm ở máy 17/11 từ CSV cuộc thi; Event 2026-11-17 "Tuấn cho phép ngoại lệ …"): (ảnh, bệnh nhân, hộp) theo split CXR-LT ----
GOLD_COUNTS = {'train': (708, 354, 18398), 'val': (86, 43, 2236), 'test': (206, 103, 5355)}
MSCXR_COUNTS = {'train': (701, 563, 966), 'val': (114, 94, 160), 'test': (232, 194, 322)}
N_GOLD_REGIONS = 26                    # gold chỉ có hộp cho 26 / 36 vùng

# ---- 40 nhãn CXR-LT theo thứ tự của mô hình (= INV['labels_task1'] = thứ tự cột của CSV cuộc thi) ----
LAB40 = ('Adenopathy', 'Atelectasis', 'Azygos Lobe', 'Calcification of the Aorta', 'Cardiomegaly', 'Clavicle Fracture', 'Consolidation', 'Edema',
         'Emphysema', 'Enlarged Cardiomediastinum', 'Fibrosis', 'Fissure', 'Fracture', 'Granuloma', 'Hernia', 'Hydropneumothorax', 'Infarction',
         'Infiltration', 'Kyphosis', 'Lobar Atelectasis', 'Lung Lesion', 'Lung Opacity', 'Mass', 'Nodule', 'Normal', 'Pleural Effusion',
         'Pleural Other', 'Pleural Thickening', 'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax', 'Pulmonary Embolism',
         'Pulmonary Hypertension', 'Rib Fracture', 'Round(ed) Atelectasis', 'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta',
         'Tuberculosis')
FINDINGS = ('Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Lung Opacity', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax')   # 8 dấu hiệu MS-CXR
assert len(LAB40) == N_LABELS and all(f in LAB40 for f in FINDINGS)

# ---- cổng (ngưỡng có lý do đo được; không chép từ luật khác) ----
# Toạ độ 224 trong tệp gold phải khớp letterbox 224 của (hộp gốc, W, H của METADATA). Nền ở ĐÚNG điều kiện chạy (kích thước từ metadata MIMIC-CXR-JPG,
# 18.398 hộp gold train): 78f C3 (05/10) — trung vị 0,06 px, 100 % <= 3 px (cổng của 78f: 3 px / 95 %); Colab 18/11 — 98,26 % <= 1 px. Cổng dùng
# dung sai 3 px của 78f + chặn trung vị. (0.1.0 đặt 1 px / 99 % theo số đo ở máy với kích thước DỰNG LẠI từ tệp đệm 224 của chính ImaGenome — lớn nhất
# 0,14 px — không phải điều kiện chạy -> dừng oan ở lượt đầu; sửa ở 0.1.1.) Sai W / H thật (vd đổi chỗ) cho lệch hàng chục px ở phần lớn hộp.
TOL_224_PX, MIN_SHARE_224, MAX_MEDIAN_224_PX = 3.0, 0.99, 0.5
# thứ tự vùng: IoU(hop_tb[k], hộp gold trung bình của tên k) — đo ở máy: nhỏ nhất 0,72; mọi vùng khớp nhất với chính nó
MIN_ORDER_IOU = 0.5
# cổng "đường chấm sống": IoU của luồng vùng trên gold train phải hơn sàn "hộp trung bình" (đo ở máy: 0,394) ít nhất chừng này;
# nhật ký huấn luyện: luồng ~0,78 so với sàn ~0,40 (hở ~0,38) -> 0,20 cách đều hai phía; trượt = sai hình học / thứ tự vùng / trọng số
MIN_GAIN_OVER_MEAN_BOX = 0.20

# ---- bootstrap ----
N_BOOT, BOOT_SEED = 2000, 90117

# ---- luật đọc val / test (PLAN_90 §11.5) — KHOÁ 18/11, Tuấn duyệt "Proposed thresholds" sau lượt định cỡ (Event 2026-11-18 "Kết quả định cỡ `90c` …").
# None = chưa khoá -> chế độ định cỡ (chỉ phần train; gói 0.1.x). Đã khoá -> chế độ val + test, đọc MỘT lần.
# Mọi ngưỡng áp cho CẬN DƯỚI CI95 (bootstrap theo bệnh nhân) của hiệu ghép cặp, trên val + test GỘP; thêm điều kiện ước lượng điểm > 0 ở riêng val VÀ riêng test.
#   c1_luong_tru_san    : IoU luồng vùng − IoU hộp trung bình cố định; 0,30 = 70 % của mức đo trên phần train (+0,423)
#   c2_trung_tru_may_rui: "trúng điểm" − tỉ lệ diện tích hộp; = một nửa mức đo trên phần train (attention +0,454; nhãn-vùng +0,397; cờ bất thường +0,278)
THRESHOLDS = {'c1_luong_tru_san': 0.30, 'c2_trung_tru_may_rui': {'attention': 0.23, 'nhan_vung': 0.20, 'bat_thuong': 0.14}}
# Số của lượt định cỡ (phần TRAIN, ảnh mô hình đã thấy; gói 0.1.1, G4, 18/11) — in cạnh val / test để thấy khoảng cách "đã thấy − chưa thấy" (chỉ mô tả).
TRAIN_REFERENCE = {'c1': {'luong': 0.817, 'hop_tb': 0.394, 'tu_dong': 0.975, 'luong_tru_hop_tb': 0.423, 'ti_le_05': 0.942},
                   'c2': {'dien_tich': 0.098, 'trung': {'attention': 0.552, 'nhan_vung': 0.495, 'bat_thuong': 0.376},
                          'trung_tru_may_rui': {'attention': 0.454, 'nhan_vung': 0.397, 'bat_thuong': 0.278}}}
RUN_TAGS = {'dinh_co': 'r90c_dinh_vi_arre_s42', 'val_test': 'r90c_dinh_vi_arre_s42_vt'}
EVAL_SPLITS = ('val', 'test')
N_THREADS = 6                          # luồng tải kho 518 qua Drive API (mẫu 78r_cat_vung: 6 luồng)
BATCH = 32
