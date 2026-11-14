# -*- coding: utf-8 -*-
"""Bảng mô tả 9 encoder của giao thức (PLAN_89 §2). CHỈ dữ liệu — không nạp mô hình. Quy tắc của giao thức: mỗi encoder giữ trọng số gốc, norm
cuối, kiểu gộp, chuẩn hoá và độ phân giải GỐC của nó (không nội suy vị trí).

`status`: 'da_dung' = lớp bọc đã viết + kiểm bằng tệp trọng số thật; 'chua_dung' = mới ghi theo nguồn (cột `source_checked` nói đã đọc nguồn nào,
phần chưa kiểm ghi trong `notes`). Gói không chứa ID Drive: tệp ở Drive được nhận dạng bằng đường dẫn tên + md5 + kích thước (encoders/files.py)."""
from dataclasses import dataclass
from typing import Optional, Tuple

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)        # biến đổi của mã CheXFound (đầu vào của mọi lớp bọc ở Ver1)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
POOLS = ('cls', 'mean_patch', 'norm_of_mean')     # norm_of_mean: norm(TB token patch THÔ) — đường chính thức của EVA-X


@dataclass(frozen=True)
class EncoderSpec:
    name: str                              # khoá trong registry
    title: str                             # tên in trong bảng
    arch: str
    image_size: int                        # độ phân giải GỐC (lớp bọc từ chối kích thước khác)
    patch_size: int
    embed_dim: int
    mean: Tuple[float, float, float]       # chuẩn hoá GỐC (thang [0, 1])
    std: Tuple[float, float, float]
    pool: str                              # vector toàn ảnh: 'cls' | 'mean_patch' (TB token patch đã qua norm cuối) | 'norm_of_mean'
    weights: str                           # nguồn trọng số
    license: str
    pretrain_data: str
    status: str                            # 'da_dung' | 'chua_dung'
    source_checked: str                    # nguồn đã đọc + ngày
    notes: str = ''
    drive_dir: Optional[str] = None        # thư mục chứa tệp trọng số trên Drive của ta: 'cxrlt_2024/<...>' hoặc tính từ My Drive (vd 'MIMIC_CXR/MRM');
                                           # None = tải từ Hugging Face. Tên Tuấn báo trong chat = giả định tới lượt chạy đầu (C4 liệt kê qua API, dừng nếu thiếu).
    drive_file: Optional[str] = None       # tên tệp trọng số trong thư mục đó
    file_md5: Optional[str] = None         # md5 + kích thước tệp Drive đã ghim (bản tải từ nguồn công khai của tác giả, kiểm trên máy)
    file_size: Optional[int] = None

    def __post_init__(self):
        assert self.pool in POOLS and self.status in ('da_dung', 'chua_dung'), (self.name, self.pool, self.status)
        assert self.image_size % self.patch_size == 0, (self.name, self.image_size, self.patch_size)
        assert (self.drive_dir is None) == (self.drive_file is None) == (self.file_md5 is None) == (self.file_size is None), self.name

    @property
    def n_patches(self):
        return (self.image_size // self.patch_size) ** 2


_G3 = lambda v: (v,) * 3

SPECS = {s.name: s for s in (
    EncoderSpec('raddino', 'RAD-DINO', 'ViT-B/14 (HF Dinov2Model)', 518, 14, 768, _G3(0.5307), _G3(0.2583), 'cls',
                'HF microsoft/rad-dino, revision ghim, model.safetensors', 'MSRLA (theo model card)',
                'MIMIC-CXR, CheXpert, NIH, PadChest, BRAX (882.775 ảnh); checkpoint công khai được chọn bằng linear probe trên val các tập đánh giá, gồm VinDr',
                'da_dung', 'model card + preprocessor HF; tệp thật 13/11'),
    EncoderSpec('biomedclip', 'BiomedCLIP', 'ViT-B/16 (timm, tháp ảnh của open_clip)', 224, 16, 768, CLIP_MEAN, CLIP_STD, 'cls',
                'HF microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224, open_clip_pytorch_model.bin (khoá visual.*)', 'MIT',
                'PMC-15M (hình + chú thích từ bài báo PubMed Central)', 'da_dung', 'open_clip_config.json + tệp thật 13/11',
                'vector toàn ảnh = CLS sau norm của trunk, TRƯỚC phép chiếu 512 chiều (quy ước linear probe của CLIP); lớp bọc của CheXFound đưa vector đã chiếu'),
    EncoderSpec('chexfound', 'CheXFound', 'ViT-L/16 (DINOv2 của họ, có register)', 512, 16, 1024, IMAGENET_MEAN, IMAGENET_STD, 'cls',
                'Drive tác giả: teacher_checkpoint.pth + config.yaml (md5 đã ghi ở 86)', 'MIT (repo)',
                'CXR-987K, 12 nguồn công khai; VinDr không nằm trong tiền huấn luyện (bài §IV-B)', 'chua_dung', 'mã 7e288d0 + bài arXiv 2502.05142v2; đã chạy ở 86',
                'Ver1: KHÔNG lớp bọc — mô hình + bộ nạp của chính mã họ (setup_and_build_model, --pretrained-weights, block_chunks=4) như 86f; lớp bọc cho Ver2: '
                'bước 3', drive_dir='cxrlt_2024/chexfound', drive_file='teacher_checkpoint.pth', file_md5='6976431a96978f27da057e1b9e35dcc4',
                file_size=1610888895),
    EncoderSpec('mrm', 'MRM', 'ViT-B/16 (timm VisionTransformer = models_vit.vit_base_patch16 của họ)', 224, 16, 768, _G3(0.4978), _G3(0.2449), 'mean_patch',
                'Drive tác giả (README RL4M/MRM-pytorch @ 3730ba4): MRM.pth — checkpoint huấn luyện đầy đủ, dùng 150 khoá encoder ảnh của "model"', 'MIT',
                'MIMIC-CXR-JPG + report', 'da_dung', 'repo RL4M/MRM-pytorch + module 88 (12/11) + tệp thật 14/11',
                'vector toàn ảnh = TB token patch qua norm cuối (= nhánh report lúc tiền huấn luyện; nhánh fine-tune của họ bỏ norm, dùng fc_norm mới)',
                drive_dir='MIMIC_CXR/MRM', drive_file='MRM.pth', file_md5='beabff16078b802273276f92909beed9', file_size=1815991839),
    EncoderSpec('chexzero', 'CheXzero', 'CLIP ViT-B/32 (mã OpenAI trong repo tác giả, QuickGELU)', 224, 32, 768,
                _G3(101.48761 / 255), _G3(83.43944 / 255), 'cls',
                'Drive tác giả (thư mục công khai trong README): 10 tệp best_<lô>_<lr>_original_<bước>_<AUC val>.pt (liệt kê 13/11, AUC 0,854–0,864); '
                'dùng MỘT tệp — AUC val trong tên cao nhất; hoà 0,864 (bước 35000 / 22000) → tệp đứng trước theo tên = '
                'best_64_0.0001_original_35000_0.864.pt (đặt trước khi có số VinDr, không ensemble)', 'MIT',
                'MIMIC-CXR (ảnh + report)', 'da_dung', 'repo rajpurkarlab/CheXzero @ 5c341db (model.py, zero_shot.py) + tệp thật 14/11',
                'state dict OpenAI (302 khoá, trộn fp16 / fp32); chuẩn hoá của họ ghi trên thang 0–255 (101,48761 / 83,43944); tiền xử lý gốc của họ đệm vuông 320 '
                'rồi về 224 — giao thức dùng tăng cường chung; ln_post gốc chỉ áp lên CLS, lớp bọc áp lên mọi token (quy ước chung)',
                drive_dir='MIMIC_CXR/CheXZero', drive_file='best_64_0.0001_original_35000_0.864.pt', file_md5='ea72359f1450ed8a666edbe79a2d5573',
                file_size=353548709),
    EncoderSpec('evax', 'EVA-X (B)', 'ViT-B/16 kiểu EVA-02 (timm Eva: RoPE, SwiGLU)', 224, 16, 768, _G3(0.49185243), _G3(0.28509309), 'norm_of_mean',
                'HF MapleF/eva_x @ 35ddcd6: eva_x_base_patch16_merged520k_mim.pt', 'Apache-2.0 (LICENSE.txt của repo; huy hiệu README ghi MIT)',
                'Merged-520k: NIH ChestX-ray14 + CheXpert + MIMIC-CXR, chỉ ảnh thẳng', 'da_dung',
                'repo hustvl/EVA-X @ eedd642 (eva_x.py, classification/utils/datasets.py) + bài 2405.05237 + tệp thật 14/11',
                'vector toàn ảnh = fc_norm(TB token patch thô) = forward_head(pre_logits) của EVA_X (global_pool avg; bộ lọc của họ chuyển norm của MIM thành '
                'fc_norm); token patch cho GLoRI / ML-Decoder = fc_norm áp từng token (quy ước của ta = đầu ra MIM lúc tiền huấn luyện)'),
    EncoderSpec('pubmedclip', 'PubMedCLIP (ViT-B/32)', 'CLIP ViT-B/32 (định dạng OpenAI)', 224, 32, 768, CLIP_MEAN, CLIP_STD, 'cls',
                'HF sarahESL/PubMedCLIP @ c5885a5: PubMedCLIP_ViT32.pth (khoá state_dict)', 'MIT', 'ROCO (cặp ảnh – chú thích từ PubMed), tinh chỉnh từ CLIP',
                'da_dung', 'repo sarahESL/PubMedCLIP (README) + tệp thật 14/11',
                'tệp = {best_epoch, state_dict, optimizer_state_dict, best_loss}; state_dict cùng khuôn CheXzero (302 khoá OpenAI); chuẩn hoá CLIP mặc định '
                '(tiền xử lý clip.load) — chưa thấy tác giả đổi'),
    EncoderSpec('chexworld', 'CheXWorld', 'ViT-B/16 (target encoder I-JEPA, không có CLS)', 224, 16, 768, IMAGENET_MEAN, IMAGENET_STD, 'mean_patch',
                'Drive tác giả (README LeapLabTHU/CheXWorld): chexworld_pretrained.tar (checkpoint torch dạng zip), khoá model.target_encoder.*',
                'repo không khai giấy phép (GitHub API 14/11)', 'MIMIC-CXR + NIH (train) + CheXpert, ~0,5M ảnh thẳng', 'da_dung',
                'models_vit.py + models_finetune.py (đọc 06/11) + 85f + tệp thật 14/11',
                'vector toàn ảnh = TB token đầu ra (đã qua norm) = đường FineTuner "fc" của tác giả; timm class_token=False, fc_norm=False (global_pool avg mặc '
                'định của timm tạo fc_norm và bỏ norm — bẫy gặp 06/11)',
                drive_dir='cxrlt_2024/chexworld', drive_file='chexworld_pretrained.tar', file_md5='15970f4d0396326f55990c08a965cde9', file_size=733569866),
    EncoderSpec('b', 'B (s42)', 'ViT-B/14 + luồng vùng (của ta, khởi tạo từ RAD-DINO)', 518, 14, 768, _G3(0.5307), _G3(0.2583), 'cls',
                'checkpoint của ta (gói trọng số xuất từ lượt B của Ver2)', 'của ta', 'MIMIC-CXR (CXR-LT) + thừa hưởng RAD-DINO', 'chua_dung', 'PLAN_86 §4.2',
                'CLS / patch lấy SAU luồng vùng + ghi ngược; đầu ML-Decoder đọc thêm token vùng; huấn luyện ở bf16, ảnh cân bằng histogram'),
)}
ORDER = ('b', 'chexfound', 'raddino', 'biomedclip', 'mrm', 'chexzero', 'evax', 'pubmedclip', 'chexworld')
assert set(ORDER) == set(SPECS) and len(ORDER) == 9
