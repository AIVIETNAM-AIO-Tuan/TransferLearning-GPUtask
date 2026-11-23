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
    # JSRT (0.3.0, 23/11): Drive MyDrive/MIMIC_CXR/JSRT (Tuấn 23/11: đã có sẵn trên Drive; tên xác nhận ở lượt chạy đầu); mốc từng tệp = 247 PNG của máy
    # (datasets/jsrt_manifest) — C3 dừng nếu bản Drive khác từng byte; nhãn theo tên tệp; chia phân tầng một lần (datasets/jsrt.py, hạt 9301); 100 epoch mỗi
    # pha như Shenzhen (1 / 1 bước); CHỈ MÔ TẢ (PLAN_93 §10 điểm 5).
    'jsrt': DatasetSpec('jsrt', 'JSRT (nốt phổi)', ('Nodule',), ('Nodule',), 100, ('MIMIC_CXR', 'JSRT'), 'anh', 'ap', False,
                        (148, 50, 49), (92, 31, 31), 'khong hang nao khai JSRT trong tien huan luyen (CheXFound: bai neu JSRT la bo danh gia ngoai, khong dung anh '
                        'nao; BiomedCLIP / PubMedCLIP chua xac minh)'),
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
    # JSRT: datasets.jsrt.table(read_metadata(jsrt_metadata.csv md5 1da39b35…)) trên máy 23/11 (chạy hai lần trùng; khoá = 247 PNG trên máy)
    'jsrt': {'train': ('ec4799e18245195cc8b6cc8b87d60d87', '320e93bd17ced7c34ec24b7155ec45b9'),
             'val': ('bedcdbc2c0332aaec0a71a55059f0d75', '58671d6fe67e4e032cb9807946560afa'),
             'test': ('adabb1e2a4487e8bb3bdb69c7ff5e7b5', '7ed23b93dbdb0e28c9e23d04c9a57191')},
}


def dataset(name):
    assert name in DATASETS, f'bo du lieu {name!r} khong thuoc {tuple(DATASETS)} -> DUNG'
    return DATASETS[name]


# ---- bảng gộp 93t (0.2.0) ----
SUMMARY_TAG = 'r93t_tong_hop'
TABLE_ORDER = ('b', 'raddino', 'chexfound', 'chexworld', 'biomedclip', 'mrm', 'evax', 'chexzero', 'pubmedclip')     # = vindr_bench.encoders.spec.ORDER
ROW_TITLES = {'b': 'B (của ta)', 'raddino': 'RAD-DINO', 'chexfound': 'CheXFound', 'chexworld': 'CheXWorld', 'biomedclip': 'BiomedCLIP', 'mrm': 'MRM',
              'evax': 'EVA-X', 'chexzero': 'CheXzero', 'pubmedclip': 'PubMedCLIP'}
# Số cụm của tập test (cổng của 93t): RSNA = người NIH (PLAN_88 §14.8, = bảng kho 93k); Shenzhen, JSRT = ảnh.
TEST_CLUSTERS = {'rsna': 2290, 'shenzhen': 134, 'jsrt': 49}
# Mức tiếp xúc tiền huấn luyện (KHÔNG nhãn) với ảnh của bộ — chép từ bảng PLAN_93 §2 (nguồn: hồ sơ 88 / bài báo; chỗ "chưa kiểm" giữ nguyên). Ghi ở mọi bảng.
EXPOSURE = {
    'rsna': {'b': '100 % (thừa hưởng RAD-DINO)', 'raddino': '100 %', 'chexfound': '100 %', 'chexworld': '63,8 %', 'biomedclip': 'chưa xác minh', 'mrm': '0',
             'evax': 'có (tỉ lệ chưa kiểm)', 'chexzero': '0', 'pubmedclip': 'chưa xác minh'},
    'shenzhen': {'b': 'không', 'raddino': 'không', 'chexfound': 'không', 'chexworld': 'không', 'biomedclip': 'chưa xác minh', 'mrm': '0',
                 'evax': 'không (theo danh sách nguồn)', 'chexzero': '0', 'pubmedclip': 'chưa xác minh'},
    # JSRT: CheXFound — bảng II (CXR-987K, 12 nguồn) không có JSRT, §IV-B nêu JSRT là bộ đánh giá ngoài "none of the images … used" (đọc arXiv 2502.05142v2, 23/11)
    'jsrt': {'b': 'không', 'raddino': 'không', 'chexfound': 'không (bài: bộ đánh giá ngoài)', 'chexworld': 'không', 'biomedclip': 'chưa xác minh', 'mrm': '0',
             'evax': 'không (theo danh sách nguồn)', 'chexzero': '0', 'pubmedclip': 'chưa xác minh'},
}
# Bảng VinDr của 89 (chỉ ĐỌC, để đặt cạnh thứ hạng; Ver2, kho đệm vuông không cân bằng = cùng giao thức + cùng kiểu kho): stage4_nb89/r89t_tong_hop
VINDR_SUMMARY = dict(folder=('stage4_nb89', 'r89t_tong_hop'), pattern=r'tong_hop_89_\d{8}_\d{6}\.json', cell='v2|kcb|{enc}|{head}|luat', panel='p21')
