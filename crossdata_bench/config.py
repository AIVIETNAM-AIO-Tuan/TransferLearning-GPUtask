# -*- coding: utf-8 -*-
"""Hằng số của module 93 (PLAN_93 §2–§5). Mọi số đếm / md5 ở đây tính trên máy 22/11 từ nguồn đã ghim md5 (ghi kèm cách tính); không chứa ID Drive."""
from dataclasses import dataclass

FOLDER = 'stage4_nb93'

# ---- hai gói dùng lại, theo commit ĐÃ ĐẨY (không sửa) ----
PACKAGES = {
    'vindr_bench': dict(sha='c81af62a71dd6606b46f511e7f014b48c8f3dfe3', branch='vindr-bench', version='0.12.0'),
    'external_benchmarks': dict(sha='3c1ada704c4bc771fb643aae833a345699240233', branch='external-benchmarks', version='0.3.0'),
}

# ---- chín hàng = vindr_bench.encoders.spec.ORDER (khoá 13/11); thứ tự CHẠY: ngắn trước (đo thời gian thật ở CheXzero), B cuối (PLAN_93 §7) ----
RUN_ORDER = ('chexzero', 'pubmedclip', 'chexworld', 'biomedclip', 'mrm', 'evax', 'raddino', 'chexfound', 'b')
REFERENCE = 'b'

# ---- chấm / thống kê (PLAN_93 §5) ----
HALF_SEED = 8501                  # hai nửa test (theo cụm người khi bộ có mã người)
BOOT_SEED, N_BOOT = 8502, 2000    # bootstrap của bảng gộp 93t
RULE_DELTA = 0.01                 # "B hơn X": hiệu >= +0,01 VÀ cận dưới đồng thời > 0 VÀ dương ở hai nửa
CEILING = 0.98                    # đối chứng tốt nhất >= 0,98 ở số của luật -> luật ở bộ đó thành mô tả

# ---- chuỗi kho 518 (PLAN_93 §4.2): một kho đệm vuông KHÔNG cân bằng histogram mỗi bộ = kho 'kcb' của 85 ----
STORE_KEY = 'kcb'                 # khoá kho trong vindr_bench.data.vindr.STORE_LABELS ('dem vuong, KHONG can bang histogram') — pipeline.assemble đọc nó


@dataclass(frozen=True)
class DatasetSpec:
    name: str                     # khoá ngắn (tên notebook / RUN_TAG)
    title: str
    label_names: tuple            # cột nhãn
    main_classes: tuple           # lớp của AUROC "lop chinh" trong đường val (một nhãn: chính nhãn đó)
    epochs: int                   # E1 = E2 (PLAN_93 §4.2)
    source: tuple                 # thư mục ảnh gốc trên Drive, tính từ MyDrive (khối đọc chuẩn: _duong)
    unit: str                     # đơn vị bootstrap / hai nửa: 'nguoi' (cột nhom = mã người) | 'anh'
    rule_metric: str              # số của luật B − X
    in_rule: bool                 # False: chỉ mô tả (JSRT)
    sizes: tuple                  # số ảnh (train, val, test) — cổng
    positives: tuple              # số ảnh dương (train, val, test) — cổng (một nhãn)
    exposure: str                 # ghi chú tiếp xúc tiền huấn luyện (PLAN_93 §2)

    @property
    def store_tag(self):
        return f'r93k_kho_{self.name}'

    def run_tag(self, encoder):
        return f'r93v_{self.name}_{encoder}'


DATASETS = {
    # RSNA: 26.684 ảnh Kaggle stage 2 có nhãn = 11.452 người NIH; chia 60 / 20 / 20 theo người, phân tầng "người có >= 1 ảnh dương" (PLAN_88 §14.2 / §14.8);
    # số đếm = chạy lại 22/11 trên tệp nối md5 9a80eed5… + stage_2_train_labels.csv md5 010a01a3… (= chạy thử của 88, 10/11).
    'rsna': DatasetSpec('rsna', 'RSNA Pneumonia (Lung Opacity)', ('Lung Opacity',), ('Lung Opacity',), 20, ('MIMIC_CXR', 'RSNA pneumonia'), 'nguoi',
                        'ap', True, (15932, 5476, 5276), (3553, 1325, 1134),
                        'RSNA nam trong NIH: B / RAD-DINO / CheXFound thay 100 %, CheXWorld 63,8 %, EVA-X co (ti le chua kiem), MRM / CheXzero 0'),
    # Shenzhen: tệp chia BenchmarkTransformers @ 68b7f442 (md5 ghim ở external_benchmarks.config.SHENZHEN_SPLITS), 100 epoch mỗi pha như script CheXFound.
    'shenzhen': DatasetSpec('shenzhen', 'Shenzhen (lao)', ('Tuberculosis',), ('Tuberculosis',), 100, ('MIMIC_CXR', 'Shenzhen'), 'anh', 'ap', True,
                            (463, 65, 134), (235, 33, 68), 'khong hang nao khai Shenzhen trong tien huan luyen (BiomedCLIP / PubMedCLIP chua xac minh)'),
}
SPLITS = ('train', 'val', 'test')

# Vân tay (md5 khoá, md5 nhãn) từng tách (= base.fingerprint_of) — tính trên máy 22/11 bằng datasets.<bộ>.table(...) từ nguồn đã ghim md5 (RSNA: tệp nối +
# stage_2_train_labels.csv; Shenzhen: 3 tệp chia BenchmarkTransformers). Notebook kho (93k) và notebook lượt (93v) đều phải trùng.
FINGERPRINT = {
    'rsna': {'train': ('3ac5fa92241428b1b54ef809bbf2c8ba', 'c561fee265341d5841604d8fec5ed446'),
             'val': ('6c9f409adf21aaa0e970e4add30ae996', '335a1e8fef54fa7797d830ebd2a9b240'),
             'test': ('cf7de86045938af60f1b6a97ff418f85', '9b8c470287600933cd41f5f262f16a38')},
    'shenzhen': {'train': ('de75ffcec7c7d3602539d27ee36d5b67', '3dec2b939223d657e9fc4f725db82dd1'),
                 'val': ('96875143eaba72ba8791a7ed162e1d38', 'cf7739d170f8bee581e3a57ce1e4f24a'),
                 'test': ('8f17dcc7f936e9387402c90f4a4a7721', '00ce6a323b609512326c127688536eb3')},
}


def dataset(name):
    assert name in DATASETS, f'bo du lieu {name!r} khong thuoc {tuple(DATASETS)} -> DUNG'
    return DATASETS[name]
