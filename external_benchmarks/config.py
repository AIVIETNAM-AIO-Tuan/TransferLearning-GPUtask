# -*- coding: utf-8 -*-
"""Hằng số của `88` (PLAN_88 v3 §2.1, §3.1, §4.3). Mọi giá trị ở đây đã kiểm trong máy ngày 12/11/2026 (md5 / số đếm ghi kèm)."""

# ---------------- Shenzhen: danh sách chia (PLAN_88 §2.1) ----------------
# BenchmarkTransformers @ commit ghim (giấy phép "Other" -> KHÔNG chép tệp chia vào gói; tải theo commit + kiểm md5 lúc chạy).
SHENZHEN_SPLIT_COMMIT = '68b7f4429c5ea4512c3c8b386989e7105b7f14c4'
SHENZHEN_SPLIT_URL = 'https://raw.githubusercontent.com/jlianglab/BenchmarkTransformers/{commit}/dataset/{name}'
# tách -> (tên tệp của họ, md5, số ảnh). Tách 'val' của CheXFound = tệp 'valid' của BenchmarkTransformers. Số = bài CheXFound IV-B (70:10:20).
SHENZHEN_SPLITS = {
    'train': ('ShenzenCXR_train_data.txt', '4d5020ca7f7ae2b6cde1208c732a227c', 463),
    'val': ('ShenzenCXR_valid_data.txt', '22c347aae8389b47fa8abc1cc44e5271', 65),
    'test': ('ShenzenCXR_test_data.txt', '2dd7fc9c9c9a7ca55e028088fe3b022e', 134),
}
SHENZHEN_N_IMAGES = 662          # 326 bình thường (_0) + 336 lao (_1)

# ---------------- Shenzhen: ảnh trên Drive (PLAN_88 §14.20) ----------------
# Ảnh đọc THẲNG từ MyDrive/MIMIC_CXR/Shenzhen (Tuấn đặt; đường dẫn ở ô notebook) sau khi integrity.check_shenzhen khớp mốc từng tệp (shenzhen_manifest).
# md5 bảng "tên \t md5 \t kích thước \n" của 662 PNG của bản đã kiểm palette trên máy (data/shenzhen/images/images, 12/11/2026) — = shenzhen_manifest.PNG.
SHENZHEN_PNG_TABLE_MD5 = '0a00e9bc49071f992f9164fc4bc80321'

# ---------------- Shenzhen: công thức theo source CheXFound @ 7e288d0 (PLAN_88 §4.3, §14.14–§14.15) ----------------
# scripts/eval/run_eval_shenzhen.sh (+ _glori.sh): lô 64 / GPU x 4 GPU, --val-epochs 100 --epochs 100 --image-size 512 --save-checkpoint-frequency 20
# --eval-period-epochs 20 --val-metric-type binary_auc. Ta: 1 GPU, --batch-size 256, loss của mã GIỮ NGUYÊN (Tuấn 12/11: không chia 4 -> gradient = 4 x TB
# của DDP; AdamW bất biến với tỉ lệ gradient trừ eps 1e-8). Lưới lr = 9 giá trị của bài §IV-A.2 (Tuấn 12/11), truyền qua --learning-rates (mặc định mã: 13).
PAPER_LRS = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3)
SHENZHEN_RECIPE = {
    '--batch-size': 256,
    '--val-epochs': 100,
    '--epochs': 100,
    '--image-size': 512,
    '--save-checkpoint-frequency': 20,
    '--eval-period-epochs': 20,
    '--val-metric-type': 'binary_auc',
}
DATASET_NAME = 'ShenzhenL'       # tên đăng ký của lớp Shenzhen sửa palette (shenzhen_dataset.py); lớp gốc của họ tên 'Shenzhen'
# 88a — kiểm lớp bọc dùng chung đầu vào GLoRI không đổi số trên GPU (PLAN_88 §14.17 (iii)): 4 đầu (vừa bộ nhớ khi không bọc), trải đều lưới 9 lr
EQUIV_LRS = (1e-5, 2e-4, 1e-3, 5e-3)
SCRIPT_BATCH, SCRIPT_GPUS = 64, 4                # script của họ: --batch-size 64 / GPU, CUDA_VISIBLE_DEVICES=0,1,2,3
# md5 (bỏ CR) các tệp source mà lớp bọc 88c dựa vào — đọc từ CheXFound-main (= 7e288d0 từng byte) 12/11. Colab: clone_chexfound đã kiểm HEAD + sạch;
# bản kiểm trên máy (thư mục không phải git) dùng md5 này.
CHEXFOUND_SOURCE_MD5 = {
    'chexfound/data/datasets/shenzhen.py': 'ba67e890792705acfacf955c6bd4aa42',
    'chexfound/data/loaders.py': '5ec14055b9d43f097a02ab5e6dde231b',
    'chexfound/eval/classification/linear_glori.py': '123bf478c4cdaf548084e618facb4f5f',
    'chexfound/eval/classification/utils.py': '3201f4ea557f9b576722887ae8483a0d',
    'chexfound/eval/classification/glori.py': 'd513596c925676ec6c46b42582604c54',
    'chexfound/eval/utils.py': 'd36ee3d67e5334daf4f39e18263e1406',
    'chexfound/eval/metrics.py': '70c3db81f092470eab51fd9570894c6e',
    'scripts/eval/run_eval_shenzhen.sh': 'fad5206f23c5fbcadd1f51e7d4b14475',
    'scripts/eval/run_eval_shenzhen_glori.sh': 'bf19208cdd1b40a707c729e011192a6a',
}

# ---------------- MRM (PLAN_88 §3.1) ----------------
# Tệp: link Google Drive công khai trong README của RL4M/MRM-pytorch @ commit ghim (gói không chứa ID Drive — quy ước gói của dự án; danh tính tệp
# ghim bằng md5 + kích thước). Tải 12/11/2026: 1.815.991.839 B.
MRM_REPO_COMMIT = '3730ba49a678eda870284205178bfe25627aaed8'
MRM_FILE = 'MRM.pth'
MRM_MD5 = 'beabff16078b802273276f92909beed9'
MRM_SIZE = 1815991839
# Chuẩn hoá trong mã fine-tune của họ (NIH_ChestX-ray/utils/data_utils.py dòng 24, 32): Grayscale(3) rồi Normalize([0.4978], [0.2449]).
MRM_MEAN, MRM_STD = (0.4978,) * 3, (0.2449,) * 3

# ---------------- BiomedCLIP (PLAN_88 §3.1) ----------------
BIOMEDCLIP_REPO = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
BIOMEDCLIP_COMMIT = '9f341de24bfb00180f1b847274256e9b65a3a32e'      # giấy phép MIT
BIOMEDCLIP_FILES = {'open_clip_pytorch_model.bin': '531cdee3c02f0e1ce22c474df82ad618', 'open_clip_config.json': '26b72194932e548fe568a47ff5d33ffc'}
BIOMEDCLIP_URL = 'https://huggingface.co/{repo}/resolve/{commit}/{name}'
# preprocess_cfg của open_clip_config.json (= chuẩn hoá CLIP OpenAI)
BIOMEDCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
BIOMEDCLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# ViT-B/16 224 của cả hai (MRM: models_vit.vit_base_patch16; BiomedCLIP: timm 'vit_base_patch16_224' qua open_clip TimmModel)
VIT_B16_224 = dict(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, num_classes=0)
