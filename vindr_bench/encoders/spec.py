# -*- coding: utf-8 -*-
"""Bảng mô tả 9 encoder của giao thức (PLAN_89 §2). CHỈ dữ liệu — không nạp mô hình. Quy tắc của giao thức: mỗi encoder giữ trọng số gốc, norm
cuối, kiểu gộp, chuẩn hoá và độ phân giải GỐC của nó (không nội suy vị trí).

`status`: 'da_dung' = lớp bọc đã viết + kiểm bằng tệp trọng số thật; 'chua_dung' = mới ghi theo nguồn (cột `source_checked` nói đã đọc nguồn nào,
phần chưa kiểm ghi trong `notes`). Gói không chứa ID Drive: tệp ở Drive được nhận dạng bằng đường dẫn tên + md5 + kích thước (encoders/files.py)."""
import re
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
    file_size: Optional[int] = None        # 0.11.0: None = ghim theo md5 (kích thước chưa biết lúc dựng — 5 hàng Việc B; kích thước lấy từ Drive, ghi lại)
    extra_files: Tuple[Tuple[str, str, Optional[int]], ...] = ()   # tệp phụ CÙNG thư mục Drive (tên, md5, kích thước | None) — vd cấu hình của B (backbone_ft.pt)
    run_suffix: str = ''                   # hậu tố RUN_TAG khi lớp bọc đổi SAU khi hàng đã chạy (lượt mới không trộn thư mục / tệp resume của lượt cũ):
                                           # '_imnet' = PubMedCLIP chuẩn hoá ImageNet (16/11). Dùng ở notebook.ver1 / ver2.run_tag và evaluation.collect.

    def __post_init__(self):
        assert self.pool in POOLS and self.status in ('da_dung', 'chua_dung'), (self.name, self.pool, self.status)
        assert self.run_suffix == '' or re.fullmatch(r'_[a-z0-9]+', self.run_suffix), (self.name, self.run_suffix)
        assert self.image_size % self.patch_size == 0, (self.name, self.image_size, self.patch_size)
        assert (self.drive_dir is None) == (self.drive_file is None) == (self.file_md5 is None) and (self.drive_file or self.file_size is None), self.name
        assert self.file_size is None or int(self.file_size) > 0, self.name
        assert not self.extra_files or (self.drive_dir and all(len(e) == 3 and e[1] and (e[2] is None or int(e[2]) > 0) for e in self.extra_files)), self.name

    @property
    def drive_files(self):
        '''-> [(tên, md5, kích thước | None)] mọi tệp Drive của hàng: tệp trọng số chính trước, rồi tệp phụ. None = chỉ ghim md5 (0.11.0).'''
        return ([(self.drive_file, self.file_md5, None if self.file_size is None else int(self.file_size))] if self.drive_file else []) + [
            tuple(e) for e in self.extra_files]

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
                'CXR-987K, 12 nguồn công khai; VinDr không nằm trong tiền huấn luyện (bài §IV-B)', 'da_dung',
                'mã 7e288d0 + bài arXiv 2502.05142v2; đã chạy ở 86; tệp thật 15/11',
                'Ver1: KHÔNG lớp bọc — mô hình + bộ nạp của chính mã họ (setup_and_build_model, --pretrained-weights, block_chunks=4) như 86f. Ver2: lớp '
                'bọc encoders/chexfound.py (mã mô hình của họ chép vào chexfound_src, số khúc theo checkpoint, attention SDPA như 86b)',
                drive_dir='cxrlt_2024/chexfound', drive_file='teacher_checkpoint.pth', file_md5='6976431a96978f27da057e1b9e35dcc4',
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
    EncoderSpec('pubmedclip', 'PubMedCLIP (ViT-B/32)', 'CLIP ViT-B/32 (định dạng OpenAI)', 224, 32, 768, IMAGENET_MEAN, IMAGENET_STD, 'cls',
                'HF sarahESL/PubMedCLIP @ c5885a5: PubMedCLIP_ViT32.pth (khoá state_dict)', 'MIT (repo GitHub); thẻ HF ghi "cc"',
                'ROCO (cặp ảnh – chú thích từ PubMed), tinh chỉnh từ CLIP', 'da_dung',
                'repo sarahESL/PubMedCLIP (README) + tệp thật 14/11; mã tinh chỉnh @ bdecf3c (lib/data_transform/transform_wrapper.py, '
                'configs/medclip_roco_all.yaml, lib/config/default.py — một commit 836d8cc) đọc 16/11',
                'tệp = {best_epoch, state_dict, optimizer_state_dict, best_loss} (= khuôn torch.save của main/train.py); state_dict cùng khuôn CheXzero (302 '
                'khoá OpenAI). CHUẨN HOÁ GỐC = ImageNet (16/11): mã tinh chỉnh của tác giả gán cứng Normalize ImageNet (trước 16/11 spec ghi CLIP — lượt '
                'r89v1_pubmedclip_k64 / r89v2_pubmedclip_kcb / _cb là lượt lệch, Event 16/11). Tinh chỉnh ở RandomResizedCrop 250 (conv bước 32 -> lưới 7 x 7, '
                'bỏ 26 px dưới / phải); lớp bọc giữ 224 = đúng lưới 7 x 7, không nội suy vị trí (tỉ lệ lệch ~1,12 lần, ghi trong bảng)',
                run_suffix='_imnet'),
    EncoderSpec('chexworld', 'CheXWorld', 'ViT-B/16 (target encoder I-JEPA, không có CLS)', 224, 16, 768, IMAGENET_MEAN, IMAGENET_STD, 'mean_patch',
                'Drive tác giả (README LeapLabTHU/CheXWorld): chexworld_pretrained.tar (checkpoint torch dạng zip), khoá model.target_encoder.*',
                'repo không khai giấy phép (GitHub API 14/11)', 'MIMIC-CXR + NIH (train) + CheXpert, ~0,5M ảnh thẳng', 'da_dung',
                'models_vit.py + models_finetune.py (đọc 06/11) + 85f + tệp thật 14/11',
                'vector toàn ảnh = TB token đầu ra (đã qua norm) = đường FineTuner "fc" của tác giả; timm class_token=False, fc_norm=False (global_pool avg mặc '
                'định của timm tạo fc_norm và bỏ norm — bẫy gặp 06/11)',
                drive_dir='cxrlt_2024/chexworld', drive_file='chexworld_pretrained.tar', file_md5='15970f4d0396326f55990c08a965cde9', file_size=733569866),
    EncoderSpec('b', 'B (s42)', 'ViT-B/14 + luồng vùng (của ta, khởi tạo từ RAD-DINO)', 518, 14, 768, _G3(0.5307), _G3(0.2583), 'cls',
                'checkpoint của ta: lượt 78x r78x_J_v3b_dp_ema_v2_skip7_s42 ep5 (= mốc B của 85f / 86b_B) — rs_J_bert_dong_ep5.pt (md5 93cf2f85…) + '
                'backbone_ft.pt (cấu hình, md5 192def5e…) trên cxrlt_2024/stage4_nb78x/<lượt>', 'của ta', 'MIMIC-CXR (CXR-LT) + thừa hưởng RAD-DINO', 'da_dung',
                'khối 78 / 79 + ô C2 / C2b / C6S / C7B của 86b_B (15/11); lớp bọc encoders/b.py kiểm checkpoint giả + TỆP THẬT (Tuấn tải 15/11) ở hai môi trường',
                'Tuấn 15/11: đông cứng, dùng NHƯ MỌI backbone — CLS / patch lấy SAU luồng vùng + ghi ngược; ML-Decoder chỉ đọc token patch (85f đọc thêm 36 token '
                'vùng: + 0,0037 [−0,0005; +0,0085] mAP21, không phân biệt); huấn luyện ở bf16, ảnh cân bằng histogram',
                drive_dir='cxrlt_2024/stage4_nb78x/r78x_J_v3b_dp_ema_v2_skip7_s42', drive_file='rs_J_bert_dong_ep5.pt',
                file_md5='93cf2f85cd3c191fbb6cce0304f2de17', file_size=537538740,
                extra_files=(('backbone_ft.pt', '192def5e64e468520754ad5fcb7c83cc', 537515907),)),
    # 0.10.0 (PLAN_90 Việc A, Tuấn duyệt 17/11): encoder có giám sát giải phẫu của người khác — hàng THÊM, không thuộc 9 hàng đã khoá (EXTRA_ROWS bên dưới)
    EncoderSpec('asg', 'ASG (ViT-B/16)', 'ViT-B/16 (lớp ViT của họ = timm vit_base_patch16_224; khởi tạo DeiT-B ImageNet)', 224, 16, 768,
                IMAGENET_MEAN, IMAGENET_STD, 'cls',
                'Drive tác giả (README ASGMVLP/ASGMVLP_CODE @ 110f3f4, mục 01/2026): asg_vit.ckpt — {state_dict} của mô hình tiền huấn luyện đầy đủ, dùng 150 '
                'khoá encoder ảnh img_encoder_q.model.*', 'repo không có LICENSE (README chỉ có huy hiệu "License: Physionet")',
                'MIMIC-CXR (217 nghìn cặp ảnh thẳng – report); vùng lấy từ Faster R-CNN học trên Chest ImaGenome, bộ ba từ RadGraph', 'da_dung',
                'models/vit.py + models/encoder.py + datasets/pretrain_dataset.py @ 110f3f4 (đọc 17/11) + tệp thật 17/11',
                'vector toàn ảnh = CLS sau norm cuối = img_feat[:, 0] của ImageEncoder (TRƯỚC đầu chiếu global_embed, chỉ dùng cho loss tiền huấn luyện); token '
                'patch = img_feat[:, 1:]. Tiền xử lý gốc: co cạnh dài 256 + ĐỆM VUÔNG + cắt 224 -> kho đệm vuông là kiểu ảnh gốc, không chạy kho cắt giữa. So '
                'với B là so HAI HỆ THỐNG (khác khởi tạo, 224 so với 518, mục tiêu) — họ phép so riêng (PLAN_90 §2)',
                drive_dir='cxrlt_2024/asg', drive_file='asg_vit.ckpt', file_md5='d50b724f7e3284d010ea4f4985e765e8', file_size=734485971),
)}

# 0.11.0 (PLAN_90 Việc B §10.1–§10.2; Tuấn duyệt 17/11, dựng VinDr 19/11): năm hàng ablation "vị trí mô-đun vùng" — lớp bọc encoders/vt.py. md5 = bản trên
# Drive mà C0 của 5 notebook 78r_dac_trung_vt_*_s42 đã tải + qua cổng (18/11); kích thước chưa biết -> ghim theo md5. KHÔNG vào bảng 9 encoder / họ 32 phép:
# chỉ kho kcb, bảng + họ 16 phép (3) − X riêng ở 89t.
_VT_DIR = 'cxrlt_2024/stage4_nb78x/'
_VT_SRC = ('khối 78 / 79 (encoders/b_src.py, vt_src.py) + cổng C2b của 78r VT / 78z (đã ĐẠT trên chính các tệp này 18/11); lớp bọc encoders/vt.py kiểm checkpoint '
           'giả + tệp thật (tests/test_vt.py)')
_VT_THAT = '; TỆP THẬT kiểm trên máy 19/11 (md5 = ghim, nạp từng tensor, cổng của hàng, bf16 vs fp32)'
_VT_C4 = '; tệp thật: cổng C4 trong lượt (cùng bộ dựng 78x VT với hàng (1b), tệp thật của (1b) đã kiểm 19/11)'
_VT_DATA = 'MIMIC-CXR (CXR-LT 2024, seed 42, 6 epoch, trọng số EMA ep4) + thừa hưởng RAD-DINO'
_VT_ARCH = 'ViT-B/14 (RAD-DINO tinh chỉnh)'
for _s in (
    EncoderSpec('vt_b4v2', '(3) trong encoder + ghi ngược (B.4 v2)', f'{_VT_ARCH} + luồng vùng trong encoder, ghi ngược (= B không đường ô cắt)', 518, 14, 768,
                _G3(0.5307), _G3(0.2583), 'cls', 'checkpoint của ta: lượt 78x r78x_J_v3b_dp_ema_v2_s42 — rs_J_bert_dong_ep4.pt + backbone_ft.pt', 'của ta', _VT_DATA,
                'da_dung', _VT_SRC + _VT_THAT, 'hàng THAM CHIẾU của họ 16 phép (PLAN_90 §10.2); CLS / patch lấy SAU luồng vùng + ghi ngược (như B)',
                drive_dir=_VT_DIR + 'r78x_J_v3b_dp_ema_v2_s42', drive_file='rs_J_bert_dong_ep4.pt', file_md5='ce771e5b736b97fd08bb62192fd298e0',
                extra_files=(('backbone_ft.pt', 'e1b103a2ef7acdb068e849cfebac0174', None),)),
    EncoderSpec('vt_mot_chieu', '(2) trong encoder, chỉ đọc', f'{_VT_ARCH}; lúc học luồng vùng trong encoder, KHÔNG ghi ngược', 518, 14, 768,
                _G3(0.5307), _G3(0.2583), 'cls', 'checkpoint của ta: lượt 78x r78x_J_v3b_dp_ema_vt_mot_chieu_s42 — rs_J_bert_dong_ep4.pt + backbone_ft.pt', 'của ta',
                _VT_DATA, 'da_dung', _VT_SRC + _VT_C4, 'token ảnh = forward RAD-DINO trọng số tinh chỉnh (luồng chỉ đọc); cổng lúc dựng: forward đầy đủ = token ảnh từng bit',
                drive_dir=_VT_DIR + 'r78x_J_v3b_dp_ema_vt_mot_chieu_s42', drive_file='rs_J_bert_dong_ep4.pt', file_md5='6f66e30a052e332eec9896cfa486cf95',
                extra_files=(('backbone_ft.pt', '6c85dee757585755180533b4ca49a528', None),)),
    EncoderSpec('vt_sau4', '(1a) sau encoder, 4 lớp', f'{_VT_ARCH}; lúc học 4 lớp JCA sau block cuối, chỉ đọc', 518, 14, 768,
                _G3(0.5307), _G3(0.2583), 'cls', 'checkpoint của ta: lượt 78x r78x_J_v3b_dp_ema_vt_sau4_s42 — rs_J_bert_dong_ep4.pt + backbone_ft.pt', 'của ta',
                _VT_DATA, 'da_dung', _VT_SRC + _VT_C4, 'token ảnh = forward RAD-DINO trọng số tinh chỉnh (luồng sau encoder); cổng lúc dựng: forward đầy đủ = token ảnh từng bit',
                drive_dir=_VT_DIR + 'r78x_J_v3b_dp_ema_vt_sau4_s42', drive_file='rs_J_bert_dong_ep4.pt', file_md5='2635d1f0b9c23454216660f9e1787357',
                extra_files=(('backbone_ft.pt', 'e59f3f8fbb54372d3033c2f11260010c', None),)),
    EncoderSpec('vt_sau1', '(1b) sau encoder, 1 lớp', f'{_VT_ARCH}; lúc học 1 lớp JCA sau block cuối, chỉ đọc (kiểu ADPD)', 518, 14, 768,
                _G3(0.5307), _G3(0.2583), 'cls', 'checkpoint của ta: lượt 78x r78x_J_v3b_dp_ema_vt_sau1_s42 — rs_J_bert_dong_ep4.pt + backbone_ft.pt', 'của ta',
                _VT_DATA, 'da_dung', _VT_SRC + _VT_THAT, 'token ảnh = forward RAD-DINO trọng số tinh chỉnh (luồng sau encoder); cổng lúc dựng: forward đầy đủ = token ảnh từng bit',
                drive_dir=_VT_DIR + 'r78x_J_v3b_dp_ema_vt_sau1_s42', drive_file='rs_J_bert_dong_ep4.pt', file_md5='735c00cf28b8e803e4f9248e8f453e46',
                extra_files=(('backbone_ft.pt', '9bfa81f054193ccf0d0daa9d82495ca2', None),)),
    EncoderSpec('vt_khong_luong', '(0) không luồng vùng', f'{_VT_ARCH}; lúc học KHÔNG luồng vùng (B1 + drop-path + EMA)', 518, 14, 768,
                _G3(0.5307), _G3(0.2583), 'cls', 'checkpoint của ta: lượt 78x r78x_B1_dp_ema_s42 — rs_B1_ep4.pt (cấu hình cfg_kl nằm trong tệp)', 'của ta',
                _VT_DATA, 'da_dung', _VT_SRC + _VT_THAT, 'token ảnh = forward RAD-DINO trọng số tinh chỉnh; không luồng vùng -> chỉ cổng cấu hình cfg_kl',
                drive_dir=_VT_DIR + 'r78x_B1_dp_ema_s42', drive_file='rs_B1_ep4.pt', file_md5='679a0cdd48867f97601a520c44694b61'),
):
    SPECS[_s.name] = _s
del _s

ORDER = ('b', 'chexfound', 'raddino', 'biomedclip', 'mrm', 'chexzero', 'evax', 'pubmedclip', 'chexworld')   # 9 hàng của giao thức (khoá 13/11): họ phép so 32
EXTRA_ROWS = ('asg',)                 # hàng thêm SAU khi giao thức đã khoá (0.10.0): bảng + hình có, KHÔNG vào họ 32 phép — họ riêng (PLAN_90 §2)
ALL_ROWS = ORDER + EXTRA_ROWS         # hàng của bảng / hình / họ phép so chính của 89t (KHÔNG gồm hàng ablation)
ABLATION_ROWS = ('vt_b4v2', 'vt_mot_chieu', 'vt_sau4', 'vt_sau1', 'vt_khong_luong')   # 0.11.0: PLAN_90 Việc B — hàng (3) đứng đầu = tham chiếu
ABLATION_REF = 'vt_b4v2'
ABLATION_STORE = 'kcb'                # PLAN_90 §10.2: Ver2, kho đệm vuông không cân bằng — kho duy nhất của hàng ablation
assert set(ALL_ROWS) | set(ABLATION_ROWS) == set(SPECS) and len(ORDER) == 9 and not set(ORDER) & set(EXTRA_ROWS) and not set(ALL_ROWS) & set(ABLATION_ROWS)
assert ABLATION_ROWS[0] == ABLATION_REF
