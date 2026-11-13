# -*- coding: utf-8 -*-
"""Bảng mô tả 9 encoder của giao thức (PLAN_89 §2). CHỈ dữ liệu — không nạp mô hình. Quy tắc của giao thức: mỗi encoder giữ trọng số gốc, norm
cuối, kiểu gộp, chuẩn hoá và độ phân giải GỐC của nó (không nội suy vị trí).

`status`: 'da_dung' = lớp bọc đã viết + kiểm bằng tệp trọng số thật; 'chua_dung' = mới ghi theo nguồn (cột `source_checked` nói đã đọc nguồn nào,
phần chưa kiểm ghi trong `notes`). Gói không chứa ID Drive: tệp ở Drive của tác giả được nhận dạng bằng tên + md5 lúc nạp."""
from dataclasses import dataclass
from typing import Optional, Tuple

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)        # biến đổi của mã CheXFound (đầu vào của mọi lớp bọc ở Ver1)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
POOLS = ('cls', 'mean_patch')


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
    pool: str                              # vector toàn ảnh: 'cls' | 'mean_patch' (TB token patch đã qua norm cuối)
    weights: str                           # nguồn trọng số
    license: str
    pretrain_data: str
    status: str                            # 'da_dung' | 'chua_dung'
    source_checked: str                    # nguồn đã đọc + ngày
    notes: str = ''
    drive_dir: Optional[str] = None        # thư mục chứa tệp trọng số trên Drive của ta (TÊN đường dẫn, không phải ID); None = tải từ Hugging Face.
                                           # Tên Tuấn báo trong chat = giả định tới lượt chạy đầu (notebook liệt kê thư mục qua API, dừng nếu không thấy).
    drive_file: Optional[str] = None       # tên tệp trọng số dùng trong thư mục đó

    def __post_init__(self):
        assert self.pool in POOLS and self.status in ('da_dung', 'chua_dung'), (self.name, self.pool, self.status)
        assert self.image_size % self.patch_size == 0, (self.name, self.image_size, self.patch_size)

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
                'CXR-987K, 12 nguồn công khai; VinDr không nằm trong tiền huấn luyện (bài §IV-B)', 'chua_dung', 'mã 7e288d0 + bài arXiv 2502.05142v2; đã chạy ở 86'),
    EncoderSpec('mrm', 'MRM', 'ViT-B/16 (timm)', 224, 16, 768, _G3(0.4978), _G3(0.2449), 'mean_patch',
                'Drive tác giả: MRM.pth (md5 beabff16078b802273276f92909beed9, 1.815.991.839 B — đã kiểm ở module 88)', 'MIT',
                'MIMIC-CXR-JPG + report', 'chua_dung', 'repo RL4M/MRM-pytorch + tệp thật (module 88, 12/11)',
                'tác giả gộp TB patch; lấy norm (không dùng fc_norm của nhánh fine-tune) như module 88'),
    EncoderSpec('chexzero', 'CheXzero', 'CLIP ViT-B/32 (mã OpenAI trong repo tác giả, QuickGELU)', 224, 32, 768,
                _G3(101.48761 / 255), _G3(83.43944 / 255), 'cls',
                'Drive tác giả (thư mục công khai trong README): 10 tệp best_<lô>_<lr>_original_<bước>_<AUC val>.pt (liệt kê 13/11, AUC 0,854–0,864); '
                'dùng MỘT tệp — AUC val trong tên cao nhất; hoà 0,864 (bước 35000 / 22000) → tệp đứng trước theo tên = '
                'best_64_0.0001_original_35000_0.864.pt (đặt trước khi có số VinDr, không ensemble)', 'MIT',
                'MIMIC-CXR (ảnh + report)', 'chua_dung', 'repo rajpurkarlab/CheXzero (zero_shot.py, model.py) 13/11',
                'chuẩn hoá của họ ghi trên thang 0–255 (101,48761 / 83,43944); tiền xử lý gốc của họ đệm vuông 320 rồi về 224 — giao thức dùng tăng cường chung; '
                'ln_post gốc chỉ áp lên CLS, lớp bọc áp lên mọi token (quy ước chung); chưa nạp tệp thật; md5 / kích thước tệp: ghim ở lượt đọc đầu tiên',
                drive_dir='MIMIC_CXR/CheXZero',                    # Tuấn báo 13/11 ("checkpoints trong MIMIC_CXR/CheXZero"); chưa kiểm qua API
                drive_file='best_64_0.0001_original_35000_0.864.pt'),
    EncoderSpec('evax', 'EVA-X (B)', 'ViT-B/16 kiểu EVA-02 (timm Eva: RoPE, SwiGLU)', 224, 16, 768, _G3(0.49185243), _G3(0.28509309), 'mean_patch',
                'HF MapleF/eva_x: eva_x_base_patch16_merged520k_mim.pt', 'Apache-2.0 (LICENSE.txt của repo; huy hiệu README ghi MIT)',
                'Merged-520k: NIH ChestX-ray14 + CheXpert + MIMIC-CXR, chỉ ảnh thẳng', 'chua_dung', 'repo hustvl/EVA-X (eva_x.py, datasets.py) + bài 2405.05237 13/11',
                'README đòi timm 0.9.0, môi trường CheXFound có 0.9.12 — chưa kiểm; tác giả gộp TB patch; trọng số MIM có `norm` (bộ lọc của họ chuyển thành fc_norm)'),
    EncoderSpec('pubmedclip', 'PubMedCLIP (ViT-B/32)', 'CLIP ViT-B/32 (định dạng OpenAI)', 224, 32, 768, CLIP_MEAN, CLIP_STD, 'cls',
                'HF sarahESL/PubMedCLIP (kho của tác giả; README repo cũng dẫn OneDrive)', 'MIT', 'ROCO (cặp ảnh – chú thích từ PubMed), tinh chỉnh từ CLIP',
                'chua_dung', 'repo sarahESL/PubMedCLIP (README) 13/11', 'tên tệp ViT-B/32 và chuẩn hoá (giả định = CLIP) chưa kiểm bằng tệp thật'),
    EncoderSpec('chexworld', 'CheXWorld', 'ViT-B/16 (target encoder, không có CLS)', 224, 16, 768, IMAGENET_MEAN, IMAGENET_STD, 'mean_patch',
                'như 85f (tệp của tác giả trên Drive của ta)', 'theo repo tác giả (chưa tra lại)', 'MIMIC-CXR + NIH (train) + CheXpert, ~0,5M ảnh thẳng',
                'chua_dung', 'PLAN_86 §4.2 + 85f', 'không có CLS → TB patch; tệp .tar là checkpoint zip, global_pool gây fc_norm (bẫy đã gặp 06/11)'),
    EncoderSpec('b', 'B (s42)', 'ViT-B/14 + luồng vùng (của ta, khởi tạo từ RAD-DINO)', 518, 14, 768, _G3(0.5307), _G3(0.2583), 'cls',
                'checkpoint của ta (gói trọng số xuất từ lượt B của Ver2)', 'của ta', 'MIMIC-CXR (CXR-LT) + thừa hưởng RAD-DINO', 'chua_dung', 'PLAN_86 §4.2',
                'CLS / patch lấy SAU luồng vùng + ghi ngược; đầu ML-Decoder đọc thêm token vùng; huấn luyện ở bf16, ảnh cân bằng histogram'),
)}
ORDER = ('b', 'chexfound', 'raddino', 'biomedclip', 'mrm', 'chexzero', 'evax', 'pubmedclip', 'chexworld')
assert set(ORDER) == set(SPECS) and len(ORDER) == 9
