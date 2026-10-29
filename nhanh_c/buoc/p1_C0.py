# ==== nhanh_c buoc p1_C0 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C0 · Cau hinh 78x J v3b DOT 4 — B.4 stochastic depth theo block + EMA trong so — SEED 42 (Tuan duyet 21/10; 6 epoch; THU_NHANH; ghi Drive MOT lan) + du lieu (nhan train / val; KHONG nhan test) + khoi doc / ghi Drive + khoi ====
# 78u (docs/PLAN_SAU_78G.md §6.4-6.7): fine-tune RAD-DINO theo cong thuc 38a a2_E3 (NGUYEN VAN) + MOT loss phu moi luot. Report /
# ImaGenome CHI luc train, CHI benh nhan train (tru 2% giu rieng); cham CHI doc anh; KHONG du doan test (78v lam).
# Dung boi experiments/78_chi_anh/dung_78x.py (nguon: cell 79a CHI DOC qua dung_79.phan_79a + VA_78X_* + VA_78X3_*) — sua o do roi dung lai, KHONG sua tay.
# ---- 78x o cat: CHAN RAC OUTPUT TAI GOC (Tuan 27/10: warning khong anh huong tien trinh -> bo cho output de doc; cach lam nhu
#      CHAN_RAC_81 cua experiments/81_jca_hai_luong/dung_81.py). KHONG doi du lieu / ket qua; canh bao / loi KHAC van in ----
# (1) google_auth_httplib2 "per-request timeout": thong bao cua thu vien trong lenh goi Drive API (khoi Drive CHUAN, khong sua) -> ha DUNG
#     logger do xuong ERROR.
# (2) huggingface_hub (RAD-DINO cong khai): khong tim token ngam, khong bao "unauthenticated"; tat thanh tien do tai / nap trong so cua HF
#     (transformers doc co nay luc import o C2) -> env + logger xuong ERROR + bo DUNG canh bao "Error while fetching HF_TOKEN secret".
#     Thanh tien do train / val cua notebook (tqdm.auto) GIU.
# (3) PyTorch "Converting a tensor with requires_grad=True to a scalar": float() tren tham so trong cong kiem (vd C4b) -> bo DUNG cau do.
# ---- nhanh C (Tuan duyet 29/10; docs/SPEC_SAU_78Y.md B.2ab; PLAN A.4v): cau hinh theo seed do nhanh_c.buoc(ten, globals(), seed) dat vao
#      CAU_HINH_C (seed, RUN_TAG, thu muc Drive stage4_nb84, moc C s42). Cong thuc = Phan 1 luot goi o cat s42 NGUYEN VAN tru dong 'nhanh C' ----
assert isinstance(globals().get('CAU_HINH_C'), dict) and CAU_HINH_C.get('nhanh') == 'C', 'chay qua nhanh_c.buoc(...) (dat CAU_HINH_C) -> DUNG'
LUAT_THU_S44 = False   # nhanh C: luat thu seed 44 cua J v3b (Event 2026-10-16) KHONG ap
# ---- nhanh C (Tuan 29/10 "sao cu de no bao loi nay hoai"): CHAN TAI GOC OSError [Errno 9] Bad file descriptor khi tat worker DataLoader
#      (Python 3.13): luong nap cua multiprocessing.queues.Queue TU bat loi roi goi traceback.print_exc() - KHONG qua threading.excepthook nen
#      bo loc C4a khong chan duoc (cach chan nhu CHAN_RAC_81 cua experiments/81_jca_hai_luong/dung_81.py). Bo qua DUNG EBADF; loi khac VAN in.
#      Dat DAU C0 (truoc moi DataLoader) -> worker fork ve sau thua ke; tham so mac dinh -> van chay sau %reset cua o chuyen phan ----
import errno as _errno_c, multiprocessing.queues as _mpq_c
if not getattr(_mpq_c.Queue._on_queue_feeder_error, '_bo_ebadf_c', False):
    def _feeder_c(e, obj, _goc=_mpq_c.Queue._on_queue_feeder_error, _ebadf=_errno_c.EBADF):
        if isinstance(e, OSError) and e.errno == _ebadf:
            return
        _goc(e, obj)
    _feeder_c._bo_ebadf_c = True
    _mpq_c.Queue._on_queue_feeder_error = staticmethod(_feeder_c)
    del _feeder_c
del _errno_c, _mpq_c
import logging as _logging_rac, os as _os_rac, sys as _sys_rac, warnings as _warnings_rac
for _k_rac, _v_rac in (('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1'), ('HF_HUB_VERBOSITY', 'error'), ('HF_HUB_DISABLE_PROGRESS_BARS', '1')):
    _os_rac.environ.setdefault(_k_rac, _v_rac)
for _k_rac in ('google_auth_httplib2', 'huggingface_hub'):
    _logging_rac.getLogger(_k_rac).setLevel(_logging_rac.ERROR)
_warnings_rac.filterwarnings('ignore', message=r'\s*Error while fetching .HF_TOKEN. secret', category=UserWarning)
_warnings_rac.filterwarnings('ignore', message=r'Converting a tensor with requires_grad=True to a scalar', category=UserWarning)
import huggingface_hub.utils as _hfu_rac
_hfu_rac.disable_progress_bars()
if 'transformers' in _sys_rac.modules:                              # chay lai C0 sau C2: transformers da doc co luc import
    import transformers.utils.logging as _tfl_rac
    _tfl_rac.disable_progress_bar()
    del _tfl_rac
del _logging_rac, _os_rac, _sys_rac, _warnings_rac, _hfu_rac, _k_rac, _v_rac
import os, re, json, math, time, datetime, copy, tarfile, shutil, io as _io
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # 79a: giam phan manh VRAM (dat TRUOC khi CUDA khoi dong -> CHI hieu luc o runtime MOI)
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
try:
    from google.colab import drive
    drive.mount('/content/drive'); DATA_DIR = '/content/drive/MyDrive/MIMIC_CXR/planB_out'
except Exception:
    DATA_DIR = os.environ.get('CXRLT_DATA', '')
    assert DATA_DIR, 'khong phai Colab va chua dat CXRLT_DATA -> gan thang duong dan `planB_out`'
OUT_DIR = os.path.join(DATA_DIR, 'cxrlt_2024'); EMB_DIR = os.path.join(OUT_DIR, 'embeddings')
assert os.path.isdir(OUT_DIR), (
    f'{OUT_DIR} khong ton tai: tai khoan nay CHUA co shortcut MIMIC_CXR trong My Drive -> "Add shortcut to Drive".')

# ---- LUOT (MOT luot moi phien / tai khoan; moi luot MOT thu muc Drive) ----
LUOT_79A = 'J_bert_dong'  # 78x: CO DINH — nen B1 (BiomedBERT dong cung), docs/PLAN_JCA_V2.md §4 (Tuan 12/10, y 3)
CFG_79 = json.loads(r'''{
 "phien_ban": "luong_vung_v3",
 "_ghi_chu": "Huong A module 79 (docs/PLAN_79_PHAT_MINH.md §2, SPEC §5). thap 'hoc' = nhanh B2 cua 78u (BiomedBERT hoc cung, doi chung ghep cap B2); 'dong' = nhanh B1 (doi chung B1). r cua phan moi: gia dinh, KHONG chinh o vong sang loc. hat cong vao SEED.",
 "luot": {
  "J_day_du":      {"thap": "hoc",  "hai_chieu": true,  "neo": true,  "doi_chung": "B2",
                    "phan": {"cau": 0.3, "vung_cau": 0.3, "bat_thuong": 0.1, "hop": 0.1}},
  "J_mot_chieu":   {"thap": "hoc",  "hai_chieu": false, "neo": true,  "doi_chung": "B2",
                    "phan": {"cau": 0.3, "vung_cau": 0.3, "bat_thuong": 0.1, "hop": 0.1}},
  "J_khong_nghia": {"thap": "hoc",  "hai_chieu": true,  "neo": true,  "doi_chung": "B2",
                    "phan": {"cau": 0.3, "hop": 0.1}},
  "J_tu_do":       {"thap": "hoc",  "hai_chieu": true,  "neo": false, "doi_chung": "B2",
                    "phan": {"cau": 0.3, "vung_cau": 0.3, "bat_thuong": 0.1}},
  "J_bert_dong":   {"thap": "dong", "hai_chieu": true,  "neo": true,  "doi_chung": "B1",
                    "phan": {"cau": 0.3, "vung_cau": 0.3, "bat_thuong": 0.1, "hop": 0.1}}
 },
 "R": 36,
 "tang": [8, 9, 10, 11],
 "n_head": 12,
 "ff": 1536,
 "beta": 4.0,
 "t_mem": 0.5,
 "toi_da_vung": 2,
 "hat": {"luong": 7, "phu_vung": 8},
 "khoi_dong": {"epoch": 1, "buoc_thu_nhanh": 50, "lr": 0.001, "keo_dai_neu_giam": 0.05},
 "ckpt_jca": true,
 "ckpt_thap": true,
 "ckpt_bb_tang": [0, 1, 2, 3],
 "_ghi_chu_ckpt": "v3 (09/10, OOM o C5 khi giu batch 144): tinh lai kich hoat block backbone 0-3 luc backward — KHONG doi ket qua (RAD-DINO khong dropout), ~+10% thoi gian; giu batch 144 de ghep cap voi B2."
}''')   # NGUYEN VAN experiments/79_phat_minh/cau_hinh/luong_vung.json
assert LUOT_79A in CFG_79['luot'], f'LUOT_79A {LUOT_79A} khong co trong luong_vung.json'
LUONG_CFG = CFG_79['luot'][LUOT_79A]
LUOT_78U = {'hoc': 'B2', 'dong': 'B1'}[LUONG_CFG['thap']]   # NEN 78u: moi nhanh 78u chay NGUYEN VAN nhanh B2 (thap hoc cung) / B1 (dong)
THAP_HOC = LUOT_78U == 'B2'
CKPT_BB = tuple(int(i) for i in CFG_79['ckpt_bb_tang'])    # 79a: block backbone TINH LAI kich hoat luc backward (VRAM; KHONG doi ket qua)
THU_NHANH = bool(CAU_HINH_C['thu'])   # nhanh C: True = ban thu nhanh (2,048 anh train, 1 epoch, 1,024 anh val) -> RUN_TAG + '_thu'
CKPT_THAP = bool(CFG_79['ckpt_thap']) and THAP_HOC   # 79a: BAT (B2 da 96,5 / 102 GB, them luong) — KHONG doi BATCH / ACCUM
assert LUOT_78U in ('B0', 'B1', 'B2', 'B3', 'B4'), LUOT_78U
RUN_TAG = CAU_HINH_C['run_x'] + ('_thu' if THU_NHANH else '')   # nhanh C: r84x_C_s<seed> (Phan 1 = luot goi o cat s42, theo seed)
LOCAL = '/content/cxrlt_518'                          # kho anh 518 local (C1) — DUNG CHUNG voi 37 / 38a / 78g
READY_X = os.path.join(OUT_DIR, 'cxrlt_518_ready')   # kho do 41c C3-bis tao

# ---- CONG THUC 38a a2_E3 (dong NGUYEN VAN C0 cua 38a) ----
EPOCHS  = 3                # duong cong a1: ft dinh 0.2395@ep3 roi sut ve 0.2231@ep8.
BATCH, ACCUM = 144, 2      # C5 do 128 img/s vs 75 o BATCH 48, VRAM 89.5/95.6 GB
LR_HEAD, LR_MULT_BB = 1e-3, 0.1   # GIU NGUYEN — doi mot bien mot lan
W_FRONTAL, W_LATERAL = 0.7, 0.3
SEED    = 42
SIZE    = 518
WD, WARMUP, CLIP = 1e-2, 0.05, 5.0
NW = min(8, (os.cpu_count() or 2))
D_MLD = 512                                           # dau doc MLDecoder 38a / 37: d 512, ff 1024, 8 head, drop 0.1, query CO DINH
SEED = int(CAU_HINH_C['seed'])                         # nhanh C: seed cua luot (THAY SEED 42 cua 38a o tren)
SEEDS = (SEED,)                                       # seed on dinh (ten tep _s42)
NW_EVAL = min(4, NW)                                  # du doan val / chan doan giu rieng: worker KHONG persistent
EPOCHS = 6                                            # 78x: THAY EPOCHS 3 cua hang so 38a (dong tren, nguyen van) — cosine trai 6 epoch; chon epoch theo luat hai nua o C7 (early stopping kieu a, Tuan 12/10)
_EP_CHAY = 1 if THU_NHANH else EPOCHS                 # lich LR van theo EPOCHS (nhu 78g THU_NHANH)
TXT_MODEL   = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
TXT_REV     = 'e1354b7a3a09615f6aba48dfad4b7a613eef7062'   # sha HuggingFace (2023-11-06), ghim 26/09

# ---- thiet bi (NGUYEN VAN C0 cua 38a): bf16 cho forward backbone ----
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
assert dev == 'cuda', 'KHONG thay GPU -> Runtime > Change runtime type > GPU roi chay lai tu C0.'
assert torch.cuda.is_bf16_supported(), (
    'GPU khong ho tro bf16. Fine-tune fp16 can GradScaler — sua C7 truoc khi chay.')
AMP = torch.bfloat16
_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'thiet bi: {torch.cuda.get_device_name(0)} | VRAM {_vram:.0f} GB | bf16 OK')
_fork = lambda: torch.random.fork_rng(devices=[torch.cuda.current_device()] if dev == 'cuda' else [])
_sts = lambda: [torch.get_rng_state()] + ([torch.cuda.get_rng_state()] if dev == 'cuda' else [])

# ---- DU LIEU (NGUYEN VAN C0 cua 38a, TRU nhan: 78u doc nhan CHI hang train + val) ----
INV   = json.load(open(os.path.join(OUT_DIR, 'cxrlt_inventory.json')))
LAB40 = INV['labels_task1']; C = len(LAB40)
TIER  = {t: np.array([LAB40.index(l) for l in INV['tier'][t]]) for t in ('head', 'medium', 'tail')}

ORDER = pd.read_parquet(os.path.join(EMB_DIR, 'emb_index.parquet'))
SPLIT = ORDER.split.to_numpy(); SUB = ORDER.subject_id.to_numpy()
TR, VA = SPLIT == 'train', SPLIT == 'val'
TE = ~(TR | VA)
# 78u: nhan CHI hang train + val (loc TRONG pyarrow); nhan test KHONG doc. Hang test = NaN: dung nham -> loss / AP NaN -> DUNG.
_YT = pd.read_parquet(os.path.join(OUT_DIR, 'cxrlt_index.parquet'), columns=['dicom_id'] + LAB40,
                      filters=[('dicom_id', 'in', ORDER.dicom_id[TR | VA].tolist())]).set_index('dicom_id')
assert _YT.index.is_unique and len(_YT) == int((TR | VA).sum()), 'nhan train / val thieu hoac trung'
YM = np.full((len(ORDER), C), np.nan, 'float32')
YM[TR | VA] = _YT.loc[ORDER.dicom_id[TR | VA]].to_numpy('float32'); del _YT
assert set(np.unique(YM[TR | VA]).tolist()) <= {0.0, 1.0}, 'nhan train / val khong phai 0 / 1'

P_VIEW = os.path.join(OUT_DIR, 'cxrlt_view_resolved.parquet')
V = ((pd.read_parquet(P_VIEW)[['dicom_id', 'view_group', 'use']]
        .set_index('dicom_id').loc[ORDER.dicom_id].reset_index()) if os.path.exists(P_VIEW)
     else ORDER[['dicom_id', 'view_group']].assign(use=lambda d: d.view_group != 'unknown'))
assert not (set(SUB[TR]) & set(SUB[VA])) and not (set(SUB[TR]) & set(SUB[TE])), 'benh nhan train ro ri sang val / test'
TF_ = TR.copy()                                       # 78u: TOAN BO anh train (38a a2_E3 bo 15% "lat chon")
I_TF, I_VA = np.where(TF_)[0], np.where(VA)[0]
if THU_NHANH:
    _st_tf = ORDER.study_id.to_numpy()[I_TF]          # 78x: THU_NHANH chon TRON study (lo theo study / y B can du anh cung study)
    _st_ch = np.random.default_rng(SEED).choice(np.unique(_st_tf), size=min(1200, len(np.unique(_st_tf))), replace=False)
    I_TF = np.sort(I_TF[np.isin(_st_tf, _st_ch)])
    I_VA = I_VA[:1024]
VA_C = np.zeros(len(ORDER), bool); VA_C[I_VA] = True  # hang val DUOC cham (THU_NHANH: 1,024 dau) — cham diem 38a doi VA -> VA_C
print(f'{len(ORDER):,} anh | train {int(TF_.sum()):,} (dung {len(I_TF):,}) | val {int(VA.sum()):,} (cham {len(I_VA):,}) | '
      f'test {int(TE.sum()):,} (KHONG doc nhan, KHONG du doan)')
print(f'78x J v3b dot 4 dp_ema s{SEED} (nen 79a {LUOT_79A} = 78u {LUOT_78U}; {RUN_TAG}) | E={EPOCHS} (chay {_EP_CHAY}) | BATCH {BATCH}x{ACCUM}={BATCH*ACCUM} | LR dau doc {LR_HEAD} '
      f'| LR backbone {LR_HEAD*LR_MULT_BB} | NW {NW}')

# ---- cham diem (NGUYEN VAN C6 cua 38a, CHI doi VA -> VA_C) + cong ap_cols = sklearn (NGUYEN VAN c5 cua 78g) ----
from sklearn.metrics import average_precision_score
_sid_va = ORDER.study_id.to_numpy()[VA_C]
_uid, _inv = np.unique(_sid_va, return_inverse=True)
_isF = (V.view_group.to_numpy()[VA_C] == 'frontal') & V.use.to_numpy()[VA_C]
_isL = (V.view_group.to_numpy()[VA_C] == 'lateral') & V.use.to_numpy()[VA_C]

def pool_study(p):
    out = np.zeros((len(_uid), p.shape[1]), 'float64'); ws = np.zeros(len(_uid))
    for mask, w in ((_isF, W_FRONTAL), (_isL, W_LATERAL)):
        if not mask.any(): continue
        cnt = np.bincount(_inv[mask], minlength=len(_uid))
        acc = np.stack([np.bincount(_inv[mask], weights=p[mask, j].astype('float64'),
                                    minlength=len(_uid)) for j in range(p.shape[1])], 1)
        has = cnt > 0
        out[has] += w * acc[has] / cnt[has, None]; ws[has] += w
    ok = ws > 0; out[ok] /= ws[ok, None]; return out, ok
_first = np.zeros(len(_uid), int); _first[_inv[::-1]] = np.arange(VA_C.sum())[::-1]
_Ys = YM[VA_C][_first]

def ap_cols(Yc, Sc):
    o = np.full(Yc.shape[1], np.nan)
    for j in range(Yc.shape[1]):
        y, s = Yc[:, j], Sc[:, j]; n1 = y.sum()
        if n1 == 0 or n1 == len(y): continue
        r = np.argsort(-s, kind='stable'); yy, ss = y[r], s[r]
        tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
        last = np.r_[np.diff(ss) != 0, True]; tp, fp = tp[last], fp[last]
        pr, rc = tp / (tp + fp), tp / n1
        o[j] = float((np.diff(np.r_[0.0, rc]) * pr).sum())
    return o
tiers = lambda ap: {t: float(np.nanmean(ap[i])) for t, i in TIER.items()}

def score(p):
    ap_i = ap_cols(YM[VA_C], p)
    ps, ok = pool_study(p)
    return ap_i, ap_cols(_Ys[ok], ps[ok])   # 78x o cat (nhu SCORE_81 cua dung_81.py): AP muc study bang ap_cols (= sklearn, cong 5)
    # -> cot KHONG co study duong = NaN, bi nanmean bo (truoc: sklearn AP 0 + UserWarning 'No positive class' o THU_NHANH); VAL day du:
    # moi nhan deu co study duong -> so KHONG doi
print(f'val {VA_C.sum():,} anh -> {len(_uid):,} study')
_rs6 = np.random.default_rng(6); _yy = (_rs6.random((3000, C)) < 0.2).astype('float32')
_ss6 = _rs6.random((3000, C)).astype('float32')
_dd6 = float(np.nanmax(np.abs(ap_cols(_yy, _ss6) -
        np.array([average_precision_score(_yy[:, j], _ss6[:, j]) for j in range(C)]))))
assert _dd6 < 1e-6, f'ap_cols lech sklearn {_dd6:.2e}'
print(f'cong 5: ap_cols khop sklearn toi {_dd6:.2e}')

# ---- DOC DRIVE (khoi chuan, CHI doc: dau doc 37, dau ra 78f / 78q / 78s, p_ cua B0 o C7) ----
GOC_ID_TAY = os.environ.get('CXRLT_GOC_ID', '')   # nhanh C: ID cxrlt_2024 do o tai goi cua notebook dat (KHONG de trong ma cong khai); rong -> khoi tim theo duong MIMIC_CXR/planB_out/cxrlt_2024
DOC_LOC = '/content/doc78u'
# ==== DOC DRIVE — KHOI CHUAN, CHI DOC. CHEP NGUYEN VAO C0 ====
# Nguon DUY NHAT: experiments/00_ghi_drive/khoi_doc_drive.py. Sua khoi -> sua file do, chay
# test_khoi_doc_drive.py, roi moi chep lai vao notebook. KHONG sua tay trong notebook.
#
# Doc ket qua cua BAT KY notebook nao (ke ca notebook KHAC) QUA API, theo ID, qua MOI ban trung ten
# o MOI tang, kiem md5 may chu khi tai ve. KHONG ghi, KHONG tao, KHONG xoa gi tren Drive. KHONG can
# DRV_DUONG: doc ket qua notebook khac thi KHONG tro DRV_DUONG (thu muc GHI) sang do.
#
# Vi sao khong glob qua FUSE: 23/09 `tim_drv` ghep duong dan tu basename, quet sai cap -> bao
# "THIEU 34/34" cho 74b trong khi du 35 file. Thu muc trung ten (68b: `r1022_E8` x17) lam glob bo
# sot hoac lay nham ban. Cac ham `_md5`, `_q`, `_con`, `_duong`, `_tat_ca` GIONG HET TUNG KY TU ban
# trong khoi_ghi_drive.py (test kiem) -> dan ca hai khoi vao mot notebook van dung.
import os, re, time, io as _io, hashlib as _hl
from google.colab import auth as _auth
_auth.authenticate_user()
from googleapiclient.discovery import build as _build
from googleapiclient.http import MediaIoBaseDownload as _MDL
_API = _build('drive', 'v3', cache_discovery=False)

# ==== HAM ==== (tu day tro xuong kiem offline bang Drive gia — test_khoi_doc_drive.py)
GOC_ID_TAY = globals().get('GOC_ID_TAY') or ''                      # ID cxrlt_2024 (chac nhat)
DOC_LOC    = globals().get('DOC_LOC') or '/content/doc_drive'       # ban tai ve — LOCAL
_FOLDER    = 'application/vnd.google-apps.folder'
_SHORTCUT  = 'application/vnd.google-apps.shortcut'


def _md5(p):
    h = _hl.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def _q(s): return s.replace('\\', '\\\\').replace("'", "\\'")


def _con(parent, ten=None, chi_thu_muc=False):
    """Con cua `parent` — hoi THANG may chu Google theo ID, khong qua FUSE."""
    q = [f"'{parent}' in parents", 'trashed = false']
    if ten is not None: q.append(f"name = '{_q(ten)}'")
    if chi_thu_muc: q.append(f"mimeType = '{_FOLDER}'")
    out, tok = [], None
    while True:
        r = _API.files().list(q=' and '.join(q), pageSize=1000, pageToken=tok,
                              includeItemsFromAllDrives=True, supportsAllDrives=True,
                              fields='nextPageToken, files(id,name,mimeType,md5Checksum,'
                                     'modifiedTime,shortcutDetails)').execute()
        out += r.get('files', []); tok = r.get('nextPageToken')
        if not tok: return out


def _duong(ten_tuple, goc='root'):
    """Di theo ten tu `goc`, DI XUYEN SHORTCUT (MIMIC_CXR la shortcut sang tai khoan chu)."""
    cur = goc
    for i, t in enumerate(ten_tuple):
        ds = []
        for x in _con(cur, t):
            sd = x.get('shortcutDetails') or {}
            if x['mimeType'] == _FOLDER: ds.append((x, x['id']))
            elif x['mimeType'] == _SHORTCUT and sd.get('targetMimeType') == _FOLDER:
                ds.append((x, sd.get('targetId')))
        if not ds and i == 0 and goc == 'root':       # khong co shortcut -> "chia se voi toi"
            r = _API.files().list(q=f"name = '{_q(t)}' and mimeType = '{_FOLDER}' and trashed = false",
                                  includeItemsFromAllDrives=True, supportsAllDrives=True,
                                  fields='files(id,name,mimeType,modifiedTime)').execute()
            ds = [(x, x['id']) for x in r.get('files', [])]
        if not ds: return None
        cur = max(ds, key=lambda d: d[0].get('modifiedTime', ''))[1]
    return cur


def _tat_ca(duong):
    """MOI thu muc khop `duong` (tinh tu cxrlt_2024), qua MOI ban trung ten o MOI tang."""
    tang = [_GOC]
    for t in duong:
        tang = [d['id'] for c in tang for d in _con(c, t, chi_thu_muc=True)]
        if not tang: return []
    return tang


_GOC = GOC_ID_TAY or _duong(('MIMIC_CXR', 'planB_out', 'cxrlt_2024'))
assert _GOC, 'API khong tim thay cxrlt_2024 -> dat GOC_ID_TAY (phan sau /folders/ trong URL) TRUOC khoi nay'
_ten_goc = _API.files().get(fileId=_GOC, fields='name', supportsAllDrives=True).execute()['name']
assert _ten_goc == 'cxrlt_2024', f'GOC tro toi `{_ten_goc}`, khong phai cxrlt_2024 -> sua GOC_ID_TAY'
os.makedirs(DOC_LOC, exist_ok=True)      # LOCAL. Khoi nay KHONG tao gi tren Drive.


def _liet_ke(parent):
    """Nhu `_con` nhung lay THEM size + chu so huu — de bao cao, khong dung de chon."""
    out, tok = [], None
    while True:
        r = _API.files().list(q=f"'{parent}' in parents and trashed = false", pageSize=1000,
                              pageToken=tok, includeItemsFromAllDrives=True, supportsAllDrives=True,
                              fields='nextPageToken, files(id,name,mimeType,md5Checksum,size,'
                                     'modifiedTime,owners(emailAddress))').execute()
        out += r.get('files', []); tok = r.get('nextPageToken')
        if not tok: return out


def kiem_ke(duong, mau=None):
    """Kiem ke QUA API. Tra ve (ds_id_thu_muc, {ten: [ban, ...]}), `ban` moi nhat dung truoc.
       `duong` tinh tu cxrlt_2024, vd ('stage4_nb74b', 'r1022_dbltail'); quet MOI ban trung ten.
       `mau` = regex khop TOAN BO ten (re.fullmatch). ban = dict(id, ten, md5, size, mt, chu, thu_muc)."""
    ids, out = _tat_ca(duong), {}
    for fid in ids:
        for f in _liet_ke(fid):
            if f['mimeType'] == _FOLDER or (mau and not re.fullmatch(mau, f['name'])): continue
            out.setdefault(f['name'], []).append(dict(
                id=f['id'], ten=f['name'], md5=f.get('md5Checksum'), size=int(f.get('size') or 0),
                mt=f.get('modifiedTime', ''), thu_muc=fid,
                chu=((f.get('owners') or [{}])[0]).get('emailAddress', '?')))
    for v in out.values(): v.sort(key=lambda b: b['mt'], reverse=True)
    return ids, out


def xung_dot(ds_ban):
    """True neu cung ten nhung NOI DUNG khac nhau (md5 khac) giua cac ban trung ten."""
    return len({b['md5'] for b in ds_ban}) > 1


def tai(ban):
    """Tai MOT ban (theo ID) ve DOC_LOC, kiem md5 may chu; tra duong dan local. Ten local mang ID ->
       hai file CUNG TEN cua hai luot (vd p_cnn_full_s42_ep8.npy cua 74a va 74b) khong bao gio de nhau.
       Da co ban local dung md5 thi dung lai. Nem loi neu hong ca 3 lan (chi doc — an toan)."""
    lp = os.path.join(DOC_LOC, f"{ban['id']}__{ban['ten']}")
    if ban['md5'] and os.path.exists(lp) and _md5(lp) == ban['md5']: return lp
    loi = None
    for k in range(3):
        try:
            req = _API.files().get_media(fileId=ban['id'], supportsAllDrives=True)
            with _io.FileIO(lp, 'wb') as fh:
                dl = _MDL(fh, req, chunksize=32 << 20); xong = False
                while not xong: _, xong = dl.next_chunk()
            if ban['md5'] is None:
                print(f"  ⚠ {ban['ten']}: Drive khong tra md5 -> chi kiem kich thuoc")
                if os.path.getsize(lp) == ban['size']: return lp
                loi = 'kich thuoc ban tai ve lech ban tren Drive'
            elif _md5(lp) == ban['md5']: return lp
            else: loi = 'md5 ban tai ve lech ban tren Drive'
        except Exception as e:
            loi = e
        if os.path.exists(lp): os.remove(lp)
        time.sleep(3 * (k + 1))
    raise OSError(f"khong tai duoc {ban['ten']} (id {ban['id']}) sau 3 lan: {loi}")


print(f'DOC: Drive cxrlt_2024 (id {_GOC}) QUA API, moi ban trung ten, kiem md5 | tai ve {DOC_LOC}')

# ---- GHI DRIVE (khoi chuan) — moi dau ra vao THU MUC CUA NO ----
DRV_DUONG = (CAU_HINH_C['tm'], RUN_TAG)     # nhanh C: thu muc CUA module 84 (stage4_nb84); 78x / J v3b s42 / C s42 CHI DOC qua API
# ==== GHI DRIVE — KHOI CHUAN, DA KIEM 23/09. CHEP NGUYEN VAO C0, SAU KHI DA CO `RUN_TAG` ====
# Nguon DUY NHAT: experiments/00_ghi_drive/khoi_ghi_drive.py. Sua khoi -> sua file do, chay
# test_khoi_ghi_drive.py, roi moi chep lai vao notebook. KHONG sua tay trong notebook.
#
# Thi nghiem thu_ghi_drive.ipynb (23/09) tren thiet lap THAT: tai khoan chay (Workspace, Editor)
# ghi vao thu muc cua tai khoan chu (Gmail) qua shortcut `MIMIC_CXR`. Ca 5 duong ghi deu len Drive;
# CHI Drive API tu kiem duoc — may chu tra md5 TUNG file.
#
# Moi "loi ghi" ngay 23/09 deu do MA CUA TA, khong do Google:
#   (1) `tim_drv` ghep duong dan tu basename -> quet sai cap -> bao THIEU GIA 34/34 file cua 74b,
#       trong khi file van nguyen tren Drive.           => LUAT 2
#   (2) `ensure_res()` goi `makedirs` qua FUSE moi phien -> de ra `r1022_wasl25` x2; thu muc trung
#       ten lam FUSE nem FileNotFoundError o ep2 cua 74a. => LUAT 3
#   (3) `drive_gate` NEM LOI -> giet 74a du ban local con nguyen.   => LUAT 4
#
# LUAT:
#   1. Ghi = local truoc -> Drive API vao thu muc THEO ID -> so md5 MAY CHU voi md5 local.
#   2. Tim / doi chieu file THEO ID THU MUC qua API. KHONG ghep lai duong dan tu basename.
#   3. KHONG makedirs len Drive qua FUSE. Thieu thu muc -> tao QUA API, dung mot lan.
#   4. KHONG BAO GIO nem loi giua luot train vi tai len hong: vao hang cho, thu lai moi lan ghi sau.
#   5. dong_bo() cuoi luot: doi chieu QUA API bang md5; con thieu thi in lenh tai ve may.
# Doc du lieu DAU VAO lon (anh, checkpoint cu) qua FUSE van duoc — luat chi ap cho GHI ket qua.
import io as _io, hashlib as _hl
from google.colab import auth as _auth
_auth.authenticate_user()
from googleapiclient.discovery import build as _build
from googleapiclient.http import MediaFileUpload as _MFU, MediaIoBaseDownload as _MDL
_API = _build('drive', 'v3', cache_discovery=False)

# ==== HAM ==== (tu day tro xuong kiem offline bang Drive gia — test_khoi_ghi_drive.py)
GOC_ID_TAY = globals().get('GOC_ID_TAY') or ''                      # ID cxrlt_2024 (chac nhat)
# Ket qua cua MOI notebook vao thu muc CUA NO: DRV_DUONG = ('stage4_nb<SO NOTEBOOK>', RUN_TAG), dat
# TRUOC khoi. KHONG co mac dinh: 23/09 74a/74b lot vao `stage4_nb68b` vi thua huong duong dan cua 68b.
assert globals().get('DRV_DUONG'), ('dat DRV_DUONG = ("stage4_nb<SO NOTEBOOK>", RUN_TAG) TRUOC khoi ghi '
                                    '— moi notebook ghi vao thu muc CUA NO')
DRV_DUONG  = tuple(globals()['DRV_DUONG'])                              # tinh tu cxrlt_2024
assert DRV_DUONG[0].startswith('stage4_nb'), f'DRV_DUONG[0] = {DRV_DUONG[0]!r} phai la stage4_nb<SO NOTEBOOK>'
GHI_LOC    = f'/content/ghi_{RUN_TAG}'                               # ban GOC — local
_FOLDER    = 'application/vnd.google-apps.folder'
_SHORTCUT  = 'application/vnd.google-apps.shortcut'


def _md5(p):
    h = _hl.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def _q(s): return s.replace('\\', '\\\\').replace("'", "\\'")


def _con(parent, ten=None, chi_thu_muc=False):
    """Con cua `parent` — hoi THANG may chu Google theo ID, khong qua FUSE."""
    q = [f"'{parent}' in parents", 'trashed = false']
    if ten is not None: q.append(f"name = '{_q(ten)}'")
    if chi_thu_muc: q.append(f"mimeType = '{_FOLDER}'")
    out, tok = [], None
    while True:
        r = _API.files().list(q=' and '.join(q), pageSize=1000, pageToken=tok,
                              includeItemsFromAllDrives=True, supportsAllDrives=True,
                              fields='nextPageToken, files(id,name,mimeType,md5Checksum,'
                                     'modifiedTime,shortcutDetails)').execute()
        out += r.get('files', []); tok = r.get('nextPageToken')
        if not tok: return out


def _duong(ten_tuple, goc='root'):
    """Di theo ten tu `goc`, DI XUYEN SHORTCUT (MIMIC_CXR la shortcut sang tai khoan chu)."""
    cur = goc
    for i, t in enumerate(ten_tuple):
        ds = []
        for x in _con(cur, t):
            sd = x.get('shortcutDetails') or {}
            if x['mimeType'] == _FOLDER: ds.append((x, x['id']))
            elif x['mimeType'] == _SHORTCUT and sd.get('targetMimeType') == _FOLDER:
                ds.append((x, sd.get('targetId')))
        if not ds and i == 0 and goc == 'root':       # khong co shortcut -> "chia se voi toi"
            r = _API.files().list(q=f"name = '{_q(t)}' and mimeType = '{_FOLDER}' and trashed = false",
                                  includeItemsFromAllDrives=True, supportsAllDrives=True,
                                  fields='files(id,name,mimeType,modifiedTime)').execute()
            ds = [(x, x['id']) for x in r.get('files', [])]
        if not ds: return None
        cur = max(ds, key=lambda d: d[0].get('modifiedTime', ''))[1]
    return cur


def _tao(ten, cha):
    return _API.files().create(body={'name': ten, 'mimeType': _FOLDER, 'parents': [cha]},
                               fields='id', supportsAllDrives=True).execute()['id']


def _tat_ca(duong):
    """MOI thu muc khop `duong` (tinh tu cxrlt_2024), qua MOI ban trung ten o MOI tang."""
    tang = [_GOC]
    for t in duong:
        tang = [d['id'] for c in tang for d in _con(c, t, chi_thu_muc=True)]
        if not tang: return []
    return tang


def _dich(duong):
    """(id de GHI, [moi id de DOC]). Trung ten -> ghi vao ban sua gan nhat, doc tu tat ca.
       Chua co -> tao QUA API (luat 3)."""
    ds = _tat_ca(duong)
    if ds:
        if len(ds) > 1:
            print(f'  ⚠ {"/".join(duong)}: x{len(ds)} TRUNG TEN tren Drive -> GHI vao ban sua gan '
                  f'nhat, DOC tu ca {len(ds)} ban')
        tt = {i: _API.files().get(fileId=i, fields='modifiedTime', supportsAllDrives=True)
              .execute().get('modifiedTime', '') for i in ds}
        return max(ds, key=lambda i: tt[i]), ds
    cur = _GOC
    for t in duong:
        c = _con(cur, t, chi_thu_muc=True)
        cur = max(c, key=lambda d: d.get('modifiedTime', ''))['id'] if c else _tao(t, cur)
    return cur, [cur]


_GOC = GOC_ID_TAY or _duong(('MIMIC_CXR', 'planB_out', 'cxrlt_2024'))
assert _GOC, 'API khong tim thay cxrlt_2024 -> dat GOC_ID_TAY (phan sau /folders/ trong URL) TRUOC khoi nay'
_ten_goc = _API.files().get(fileId=_GOC, fields='name', supportsAllDrives=True).execute()['name']
assert _ten_goc == 'cxrlt_2024', f'GOC tro toi `{_ten_goc}`, khong phai cxrlt_2024 -> sua GOC_ID_TAY'
DRV_ID, DRV_TAT_CA = _dich(DRV_DUONG)
DA_GHI, CHO = {}, []                     # ten -> md5 local ; hang cho tai len chua xong
os.makedirs(GHI_LOC, exist_ok=True)      # LOCAL. Day la makedirs DUY NHAT cua khoi nay.


def _tai_len(ten):
    """None neu MAY CHU xac nhan md5 khop local; nguoc lai tra chuoi loi. Khong nem loi."""
    try:
        req = _API.files().create(body={'name': ten, 'parents': [DRV_ID]},
                                  media_body=_MFU(os.path.join(GHI_LOC, ten), resumable=True,
                                                  chunksize=32 << 20),
                                  fields='id,md5Checksum', supportsAllDrives=True)
        r = None
        while r is None: _, r = req.next_chunk()
        h = r.get('md5Checksum') or _API.files().get(
            fileId=r['id'], fields='md5Checksum', supportsAllDrives=True).execute().get('md5Checksum')
        return None if h == DA_GHI[ten] else f'md5 may chu {h} != local {DA_GHI[ten]}'
    except Exception as e:
        return f'{type(e).__name__}: {str(e)[:160]}'


def _da_len(ten):
    """Ban DUNG md5 da nam trong thu muc ghi chua? Lan thu lai khong de ra ban trung ten."""
    try: return any(f.get('md5Checksum') == DA_GHI[ten] for f in _con(DRV_ID, ten))
    except Exception: return False


def thu_lai():
    """Thu lai hang cho. Tu goi o moi lan ghi() va trong drive_gate() cuoi moi epoch."""
    con = []
    for ten in CHO:
        if _da_len(ten) or _tai_len(ten) is None:
            print(f'  ↻ {ten}: da len Drive (may chu xac nhan md5)')
        else:
            con.append(ten)
    CHO[:] = con


def ghi(ten, ham_ghi):
    """Ghi local -> tai len API -> may chu xac nhan md5. KHONG BAO GIO nem loi vi Drive (luat 4).
       `ham_ghi(duong_dan)` nen dung `with open(...)` de file dong va flush truoc khi tinh md5."""
    assert ten not in DA_GHI, f'{ten}: ghi hai lan cung ten trong mot phien — ten phai duy nhat'
    lp = os.path.join(GHI_LOC, ten); ham_ghi(lp)
    DA_GHI[ten] = _md5(lp)
    if CHO: thu_lai()
    loi = _tai_len(ten)
    if loi:
        CHO.append(ten)
        print(f'  !! {ten}: CHUA len Drive ({loi}) -> hang cho, thu lai o lan ghi sau. Ban local con.')
    return lp


put = ghi                                 # ma cu goi put(ten, lambda q: ...) — cung chu ky
PUT_FAIL = CHO                            # ma cu doc PUT_FAIL -> thay dung hang cho


def ghi_json(q, obj):
    """Dung trong ham_ghi: `ghi('x.json', lambda q: ghi_json(q, LOG))`. `with` bao dam file DONG
       va FLUSH truoc khi ghi() tinh md5. `json.dump(obj, open(q, 'w'))` chi dong nho refcount."""
    import json as _js
    with open(q, 'w') as _f: _js.dump(obj, _f, ensure_ascii=False, indent=1)


def tim_moi_noi(ten):
    """Local truoc; khong co thi tai tu Drive QUA API (moi ban trung ten), kiem md5. Dung cho
       resume. Nem loi neu tai hong ca 3 lan — luc do chua train nen an toan."""
    lp = os.path.join(GHI_LOC, ten)
    if os.path.exists(lp): return lp
    ds = [f for fid in DRV_TAT_CA for f in _con(fid, ten) if f.get('md5Checksum')]
    if not ds: return None
    f = max(ds, key=lambda x: x.get('modifiedTime', ''))
    loi = None
    for k in range(3):
        try:
            req = _API.files().get_media(fileId=f['id'], supportsAllDrives=True)
            with _io.FileIO(lp, 'wb') as fh:
                dl = _MDL(fh, req, chunksize=32 << 20); xong = False
                while not xong: _, xong = dl.next_chunk()
            if _md5(lp) == f['md5Checksum']: return lp
            loi = 'md5 ban tai ve lech ban tren Drive'
        except Exception as e:
            loi = e
        if os.path.exists(lp): os.remove(lp)
        time.sleep(3 * (k + 1))
    raise OSError(f'khong tai duoc {ten} tu Drive sau 3 lan: {loi}')


def ten_tren_drive():
    """Ten MOI file trong thu muc ket qua (ca ban trung ten) — hoi API, khong qua FUSE."""
    return {f['name'] for fid in DRV_TAT_CA for f in _con(fid)}


def dong_bo():
    """CUOI luot. Doi chieu MOI file da ghi voi Drive QUA API, theo ID, bang md5 (luat 5).
       Tra ve True khi du. In lenh tai ve may cho file van chua len."""
    xau = []
    for vong in (1, 2, 3):
        tren = {}
        for f in _con(DRV_ID):
            tren.setdefault(f['name'], set()).add(f.get('md5Checksum'))
        xau = [t for t, h in DA_GHI.items() if h not in tren.get(t, set())]
        print(f'--- doi chieu {len(DA_GHI)} file QUA API (vong {vong}): '
              f'{len(DA_GHI) - len(xau)} OK | {len(xau)} chua len ---')
        for t in xau: print(f'   CHUA LEN  {t}')
        if not xau:
            print('=> Drive da co du. May chu xac nhan md5 TUNG file — bang chung, khong phai cache.')
            return True
        CHO[:] = list(dict.fromkeys(CHO + xau)); thu_lai()
    print(f'!! {len(xau)} file van chua len. Ban goc con o {GHI_LOC}. TAI VE MAY truoc khi tat runtime:')
    print('   from google.colab import files')
    for t in xau: print(f'   files.download({os.path.join(GHI_LOC, t)!r})')
    return False


def drive_gate(ep_tag):
    """Ma cu goi o CUOI moi epoch. KHONG nem loi (luat 4): thu lai hang cho, bao neu con."""
    if CHO: thu_lai()
    if CHO: print(f'  ⚠ {ep_tag}: {len(CHO)} file van cho len Drive {CHO[:4]} — ban local con nguyen')


def ensure_res():
    """VO HIEU (luat 3): ban cu makedirs qua FUSE moi phien -> de ra thu muc trung ten."""
    return globals().get('RES41D')


# ---- KIEM TRUOC KHI TRAIN — DUNG NGAY o C0 neu tai khoan nay khong ghi duoc ----
# Tuan doi tai khoan khi het unit Colab, moi tai khoan ghi vao CUNG thu muc (deu Editor). Tai khoan MOI
# co the thieu quyen / het quota / API loi. Luat 4 chi cam nem loi GIUA luot; o C0, truoc gio GPU nao,
# DUNG NGAY moi dung. Ghi thu THAT mot file nho: may chu xac nhan md5 moi tinh la ghi duoc.
def _kiem_truoc():
    ai = _API.about().get(fields='user(emailAddress),storageQuota(limit,usage)').execute()
    tk = ai['user']['emailAddress']; qt = ai.get('storageQuota', {})
    con = (int(qt['limit']) - int(qt.get('usage', 0))) / 1e9 if qt.get('limit') else None
    cap = (_API.files().get(fileId=DRV_ID, fields='capabilities(canAddChildren)', supportsAllDrives=True)
           .execute().get('capabilities') or {})
    assert cap.get('canAddChildren'), (f'{tk} KHONG duoc them file vao cxrlt_2024/{"/".join(DRV_DUONG)} '
                                       f'-> can quyen Editor. DUNG truoc khi train.')
    p = os.path.join(GHI_LOC, '_kiem_ghi.txt')
    with open(p, 'w') as f: f.write(f'kiem ghi {tk} {time.time()}')
    h, loi = _md5(p), None
    try:
        req = _API.files().create(body={'name': '_kiem_ghi.txt', 'parents': [DRV_ID]},
                                  media_body=_MFU(p, resumable=True, chunksize=32 << 20),
                                  fields='id,md5Checksum', supportsAllDrives=True)
        r = None
        while r is None: _, r = req.next_chunk()
        h2 = r.get('md5Checksum') or _API.files().get(
            fileId=r['id'], fields='md5Checksum', supportsAllDrives=True).execute().get('md5Checksum')
        _API.files().update(fileId=r['id'], body={'trashed': True}, supportsAllDrives=True).execute()
        if h2 != h: loi = f'md5 may chu {h2} != local {h}'
    except Exception as e:
        loi = f'{type(e).__name__}: {str(e)[:160]}'
    finally:
        if os.path.exists(p): os.remove(p)
    assert loi is None, f'GHI THU vao Drive THAT BAI tu {tk}: {loi}. DUNG truoc khi train.'
    if con is not None and con < 10:
        print(f'  ⚠ quota {tk} chi con {con:.1f} GB — mot luot train ghi ~2 GB')
    return tk, con


TAI_KHOAN, _QUOTA_CON = _kiem_truoc()
print(f'GHI: {TAI_KHOAN} -> Drive cxrlt_2024/{"/".join(DRV_DUONG)} (id {DRV_ID}) QUA API, kiem md5 tung file'
      f' | ghi thu: DAT' + (f' | doc tu {len(DRV_TAT_CA)} ban trung ten' if len(DRV_TAT_CA) > 1 else ''))

# ---- (sua cua 78l, ra soat 01/10): tep CHI con o LOCAL ma Drive chua co ban cung md5 -> tai len lai QUA khoi ghi chuan ----
# Chay lai C0 (hoac kernel khoi dong lai) xoa DA_GHI / CHO cua phien truoc: tep dang cho tai len bi QUEN du van nam trong
# GHI_LOC. Cell train (sau khi chon epoch chay tiep) va C+ (truoc khi suy luan) goi ham nay cho cac tep phien do KHONG ghi lai.
def tai_len_lai(ten):
    """True neu vua dang ky lai `ten` qua ghi() (tai len + may chu kiem md5 + dong_bo theo doi). Bo qua neu khong co ban
       local, phien nay da ghi ten do (DA_GHI), hoac Drive da co ban CUNG md5. KHONG goi cho tep phien nay se ghi lai:
       ghi() cam ghi hai lan cung ten trong mot phien."""
    lp = os.path.join(GHI_LOC, ten)
    if not os.path.exists(lp) or ten in DA_GHI: return False
    h = _md5(lp)
    try: tren = any(f.get('md5Checksum') == h for fid in DRV_TAT_CA for f in _con(fid, ten))
    except Exception: tren = False                       # hoi API loi -> tai len (ban trung md5 khong gay xung dot)
    if tren: return False
    print(f'  ↻ {ten}: CHI co ban local (hang cho phien truoc da mat) -> tai len lai qua khoi ghi chuan')
    ghi(ten, lambda q: None)                             # tep DA nam dung cho: ghi() chi tinh md5, tai len, kiem
    return True
# ---- khoi bo sung GHI CUOI (NGUYEN VAN experiments/00_ghi_drive/khoi_ghi_cuoi.py; Tuan 17/10) ----
# ==== GHI CUOI — KHOI BO SUNG cho khoi_ghi_drive.py (Tuan 17/10). CHEP NGUYEN VAO C0 NGAY SAU khoi_ghi_drive ====
# Nguon DUY NHAT: experiments/00_ghi_drive/khoi_ghi_cuoi.py; kiem: test_khoi_ghi_cuoi.py (Drive gia). KHONG sua tay trong notebook.
# Tuan 17/10: notebook DUNG MOI ghi Drive MOT lan luc cuoi. Ghi lien tuc (moi epoch tai p_ / nhat ky / rs_ len ngay) de that lac tep
# phai dong bo va ton quota API. Khoi nay DE LAI ghi / put / drive_gate / dong_bo / _tai_len cua khoi chuan; khoi chuan KHONG doi ->
# notebook dung truoc 17/10 dung lai y nguyen.
#
# LUAT (bo sung luat 1-5 cua khoi chuan):
#   6. Giua luot KHONG goi API cho tep ket qua: ghi() chi ghi local + xep hang; drive_gate() khong lam gi.
#   7. Het luot dong_bo(): MOT lan liet ke thu muc + tai len moi tep dang cho chua co (khuc 256 MB, md5 tung tep), roi doi chieu 3 vong
#      nhu khoi chuan. DU -> xoa tep resume. THIEU -> GIU tep resume + in lenh tai ve may.
#   8. Resume = MOT tep `_resume_<RUN_TAG>.pt` tren Drive, GHI DE TAI CHO (cung id) moi lan luu_resume(). Tep resume chua ca tep ket qua
#      NHO dang cho (cho_nho()); tep ket qua LON (> NGUONG_CHO) tai len NGAY, MOT lan, khi luu_resume — khong nhet vao tep resume.
#   9. KHONG nem loi giua luot (luat 4): luu_resume hong -> in canh bao, ban local con, lan luu sau thu lai.
#
# Cach dung trong notebook (bo dung chen; xem memory ghi-drive-mot-lan-cuoi):
#   C0  : khoi_ghi_drive + khoi nay.
#   C6  : dau:  p = tim_moi_noi(TEN_RESUME); neu co -> r = torch.load(p, ...); khoi_phuc_cho(r['cho']); nap trang thai tu r.
#         moi epoch: ghi() cac tep ket qua (p_, nhat ky ...) roi luu_resume(lambda q: torch.save(dict(<trang thai>, cho=cho_nho()), q)).
#         KHONG ghi rs_ tung epoch nua; checkpoint ben sau CAN (vd rs_ ep4 cho frozen reading) -> ghi() nhu tep thuong.
#   C9  : dong_bo().
assert 'DA_GHI' in globals() and 'dong_bo' in globals() and '_con' in globals() and 'DRV_ID' in globals(), (
    'chep khoi_ghi_cuoi NGAY SAU khoi_ghi_drive (can DA_GHI / CHO / _con / DRV_ID cua khoi chuan)')
GHI_CUOI = True
TEN_RESUME = f'_resume_{RUN_TAG}.pt'
NGUONG_CHO = int(globals().get('NGUONG_CHO') or (64 << 20))   # tep dang cho > nguong: tai len khi luu_resume, khong nhet vao tep resume
_KHUC = 256 << 20                                             # khuc tai len: it luot goi hon cho tep lon (khoi chuan: 32 MB)
_RESUME_ID = None
if not getattr(dong_bo, '_ghi_cuoi', False):                  # chay lai C0 KHONG de vong lap dong_bo -> dong_bo
    _dong_bo_chuan = dong_bo


def _tai_len(ten):
    """= _tai_len cua khoi chuan, khuc _KHUC. None neu MAY CHU xac nhan md5 khop local; nguoc lai chuoi loi. Khong nem loi."""
    try:
        req = _API.files().create(body={'name': ten, 'parents': [DRV_ID]},
                                  media_body=_MFU(os.path.join(GHI_LOC, ten), resumable=True, chunksize=_KHUC),
                                  fields='id,md5Checksum', supportsAllDrives=True)
        r = None
        while r is None: _, r = req.next_chunk()
        h = r.get('md5Checksum') or _API.files().get(
            fileId=r['id'], fields='md5Checksum', supportsAllDrives=True).execute().get('md5Checksum')
        return None if h == DA_GHI[ten] else f'md5 may chu {h} != local {DA_GHI[ten]}'
    except Exception as e:
        return f'{type(e).__name__}: {str(e)[:160]}'


def ghi(ten, ham_ghi):
    """Ghi cuoi (luat 6): ghi LOCAL + xep hang, KHONG goi API. Tai len o dong_bo() (hoac luu_resume() neu tep lon)."""
    assert ten not in DA_GHI, f'{ten}: ghi hai lan cung ten trong mot phien — ten phai duy nhat'
    assert ten != TEN_RESUME, f'{ten}: tep resume ghi qua luu_resume(), khong qua ghi()'
    lp = os.path.join(GHI_LOC, ten); ham_ghi(lp)
    DA_GHI[ten] = _md5(lp)
    CHO.append(ten)
    return lp


put = ghi                                 # ma cu goi put(ten, lambda q: ...) — cung chu ky


def drive_gate(ep_tag):
    """Ghi cuoi (luat 6): KHONG goi API giua luot. Ma cu goi cuoi moi epoch -> khong lam gi."""
    return None


def _tai_mot_luot():
    """MOT lan liet ke thu muc ghi + tai len moi tep dang cho chua co tren Drive (cung md5). Tra so tep da tai len."""
    tren = {}
    try:
        for f in _con(DRV_ID): tren.setdefault(f['name'], set()).add(f.get('md5Checksum'))
    except Exception as e:
        print(f'  !! liet ke thu muc ghi hong ({type(e).__name__}) -> tai len khong kiem truoc')
    con, n = [], 0
    for t in CHO:
        if DA_GHI[t] in tren.get(t, set()): continue
        if _tai_len(t) is None: n += 1
        else: con.append(t)
    CHO[:] = con
    return n


def _xoa_resume():
    global _RESUME_ID
    try:
        ds = [f for fid in DRV_TAT_CA for f in _con(fid, TEN_RESUME)]
        for f in ds: _API.files().update(fileId=f['id'], body={'trashed': True}, supportsAllDrives=True).execute()
        _RESUME_ID = None
        if ds: print(f'   xoa {len(ds)} tep {TEN_RESUME} tren Drive (vao thung rac) — luot da du')
    except Exception as e:
        print(f'  !! xoa {TEN_RESUME} hong ({type(e).__name__}: {str(e)[:80]}) — xoa tay trong thu muc ket qua')


def dong_bo():
    """CUOI luot (luat 7): tai len MOT luot moi tep dang cho, roi doi chieu QUA API bang md5 nhu khoi chuan. DU -> xoa tep resume,
       tra True. THIEU -> GIU tep resume, in lenh tai ve may, tra False."""
    n0 = len(CHO)
    n = _tai_mot_luot()
    print(f'--- ghi cuoi: {n0} tep dang cho -> tai len {n} tep MOT luot ({len(CHO)} tep loi -> thu lai khi doi chieu) ---')
    du = _dong_bo_chuan()
    if du: _xoa_resume()
    else: print(f'   GIU {TEN_RESUME} tren Drive: chay lai notebook se tiep tu do va tai lai tep con thieu')
    return du


dong_bo._ghi_cuoi = True


def cho_nho():
    """Cho tep resume (luat 8) -> {'tep': {ten: bytes} cac tep DANG CHO <= NGUONG_CHO, 'da_len': {ten: md5} cac tep da len Drive}.
       Tep LON dang cho: tai len NGAY (MOT lan); hong -> van o hang cho (dong_bo thu lai) nhung KHONG co trong tep resume (canh bao)."""
    for t in [t for t in CHO if os.path.getsize(os.path.join(GHI_LOC, t)) > NGUONG_CHO]:
        if _da_len(t) or _tai_len(t) is None:
            CHO.remove(t)
        else:
            print(f'  !! {t}: tep LON chua len Drive -> KHONG nam trong tep resume; mat runtime truoc dong_bo() se mat tep nay')
    tep = {}
    for t in CHO:
        p = os.path.join(GHI_LOC, t)
        if os.path.getsize(p) <= NGUONG_CHO:
            with open(p, 'rb') as f: tep[t] = f.read()
    return dict(tep=tep, da_len={t: h for t, h in DA_GHI.items() if t not in CHO})


def khoi_phuc_cho(goi):
    """Sau khi nap tep resume: tra cac tep dang cho ve GHI_LOC + hang cho; ghi nho cac tep da len (dong_bo doi chieu ca chung)."""
    goi = goi or {}
    n = 0
    for t, b in (goi.get('tep') or {}).items():
        if t in DA_GHI: continue
        p = os.path.join(GHI_LOC, t)
        with open(p, 'wb') as f: f.write(b)
        DA_GHI[t] = _md5(p); CHO.append(t); n += 1
    for t, h in (goi.get('da_len') or {}).items(): DA_GHI.setdefault(t, h)
    print(f'ghi cuoi: khoi phuc {n} tep dang cho tu {TEN_RESUME} + {len(goi.get("da_len") or {})} tep da len Drive')
    return n


def luu_resume(ham_ghi):
    """MOT tep resume tren Drive, GHI DE TAI CHO (cung id) moi lan goi (luat 8). ham_ghi(duong) ghi trang thai — nen gom cho_nho().
       KHONG nem loi (luat 9). Tra True khi may chu xac nhan md5."""
    global _RESUME_ID
    lp = os.path.join(GHI_LOC, TEN_RESUME); ham_ghi(lp); h = _md5(lp)
    try:
        if _RESUME_ID is None:
            ds = _con(DRV_ID, TEN_RESUME)
            if ds: _RESUME_ID = max(ds, key=lambda f: f.get('modifiedTime', ''))['id']
        m = _MFU(lp, resumable=True, chunksize=_KHUC)
        if _RESUME_ID:
            req = _API.files().update(fileId=_RESUME_ID, media_body=m, fields='id,md5Checksum', supportsAllDrives=True)
        else:
            req = _API.files().create(body={'name': TEN_RESUME, 'parents': [DRV_ID]}, media_body=m, fields='id,md5Checksum',
                                      supportsAllDrives=True)
        r = None
        while r is None: _, r = req.next_chunk()
        _RESUME_ID = r['id']
        h2 = r.get('md5Checksum') or _API.files().get(
            fileId=_RESUME_ID, fields='md5Checksum', supportsAllDrives=True).execute().get('md5Checksum')
        if h2 != h: raise ValueError(f'md5 may chu {h2} != local {h}')
        return True
    except Exception as e:
        print(f'  !! {TEN_RESUME}: CHUA len Drive ({type(e).__name__}: {str(e)[:120]}) -> ban local con, lan luu sau thu lai')
        return False


print(f'GHI CUOI (17/10): tep ket qua chi ghi local, tai len MOT luot o dong_bo(); resume = MOT tep {TEN_RESUME} ghi de moi epoch '
      f'(tep cho <= {NGUONG_CHO >> 20} MB nam trong tep resume)')
# ---- 78x J v3b dot 4: dung luong Drive cua MOT luot (ghi cuoi): rs_ NHE trong so EMA moi epoch (~0,5 GB x 6) + rs_tho ep4 (~0,5 GB) + backbone_ft.pt (~0,9 GB) + khoi dong (~0,2 GB) + p_ / p_tho_ val. Tep resume (~1,6 GB + EMA ~0,4 GB, GHI DE moi epoch) bi xoa khi
#      dong_bo() du. UOC tu kich thuoc tham so, CHUA do. ----
_GB_D4 = 4.7
print(f'78x J v3b dot 4: ghi ~{_GB_D4:.1f} GB len Drive (+ tep resume ~1,6-2 GB ghi de, xoa khi xong)'
      + (f' | ⚠ quota chi con {_QUOTA_CON:.1f} GB' if _QUOTA_CON is not None and _QUOTA_CON < _GB_D4 + 5 else ''))

# ---- khoi (NGUYEN VAN experiments/78_chi_anh/khoi/): ehr_tuong_phan, vung, loss, hop_anh, cau, vung_day ----
def bo_ngau_nhien(x, p, gen):
    '''Dropout voi mat na rut tu `gen` (Generator CUNG thiet bi voi x) — KHONG cham dong ngau nhien chung.'''
    if p <= 0: return x
    giu = torch.rand(x.shape, generator=gen, device=x.device) >= p
    return x * giu.to(x.dtype) / (1.0 - p)


class ThapEHR(nn.Module):
    '''EHR muc study -> z [N, d_chieu]. Vao: x [N, d_in] (da chuan hoa nhu C1) + tui vi sinh (id, so lan) neu co V_mic.
       (Linear -> LayerNorm -> GELU -> Dropout) x 2 lop 512, roi Linear -> 256; phep chieu Linear 256 -> 128.
       Dropout rut mat na tu `gen` — BAT BUOC co `gen` khi .train() (khong bao gio roi ve dong chung).'''
    def __init__(s, d_in, V_mic=None, d_tui=64, d_an=512, d_ra=256, d_chieu=128, p_bo=0.1):
        super().__init__()
        s.p_bo = p_bo
        s.tui = nn.Parameter(torch.randn(V_mic, d_tui) * d_tui ** -0.5) if V_mic else None
        s.ln_tui = nn.LayerNorm(d_tui) if V_mic else None
        s.l1 = nn.Linear(d_in + (d_tui if V_mic else 0), d_an); s.n1 = nn.LayerNorm(d_an)
        s.l2 = nn.Linear(d_an, d_an); s.n2 = nn.LayerNorm(d_an)
        s.l3 = nn.Linear(d_an, d_ra); s.chieu = nn.Linear(d_ra, d_chieu)

    def forward(s, x, mids=None, mcnt=None, gen=None):
        assert gen is not None or not s.training or s.p_bo <= 0, 'ThapEHR.train() phai co generator RIENG (gen)'
        if s.tui is not None:
            x = torch.cat([x, s.ln_tui(F.embedding_bag(mids, s.tui, per_sample_weights=mcnt, mode='sum'))], 1)
        h = F.gelu(s.n1(s.l1(x)))
        if s.training: h = bo_ngau_nhien(h, s.p_bo, gen)
        h = F.gelu(s.n2(s.l2(h)))
        if s.training: h = bo_ngau_nhien(h, s.p_bo, gen)
        return s.chieu(s.l3(h))


class ChieuAnh(nn.Module):
    '''Dac trung anh [B, d_vao] -> (z [B, d_chieu], du doan hoi quy [B, d_hq] hoac None). Linear -> LayerNorm -> GELU
       (lop an 512) -> Linear; dau hoi quy doc lop an. KHONG dropout -> khong rut so ngau nhien nao.'''
    def __init__(s, d_vao, d_an=512, d_chieu=128, d_hq=0):
        super().__init__()
        s.l1 = nn.Linear(d_vao, d_an); s.n1 = nn.LayerNorm(d_an); s.l2 = nn.Linear(d_an, d_chieu)
        s.hq = nn.Linear(d_an, d_hq) if d_hq else None

    def forward(s, f):
        a = F.gelu(s.n1(s.l1(f)))
        return s.l2(a), (s.hq(a) if s.hq is not None else None)


class PhuEHR(nn.Module):
    '''Module loss phu cua MOT arm: chieu anh (+ dau hoi quy neu d_hq), thap EHR + nhiet do hoc duoc neu d_ehr.
       Nhiet do tau = 0.01 + 0.99 * sigmoid(tho): luon trong (0.01, 1), khong ket o bien; khoi tao tau0.'''
    def __init__(s, d_anh, d_ehr=0, V_mic=None, d_hq=0, tau0=0.1):
        super().__init__()
        s.anh = ChieuAnh(d_anh, d_hq=d_hq)
        s.ehr = ThapEHR(d_ehr, V_mic) if d_ehr else None
        s.tho = nn.Parameter(torch.tensor(math.log((tau0 - 0.01) / (1.0 - tau0)))) if d_ehr else None

    def tau(s):
        return 0.01 + 0.99 * torch.sigmoid(s.tho)


def scarf(x, pool, ti_le, gen):
    '''Tang cuong kieu SCARF: moi o (hang, cot) voi xac suat `ti_le` thay bang gia tri CUNG COT cua mot hang NGAU NHIEN
       cua `pool` (phan phoi bien thuc nghiem cua cot do). x [N, D], pool [M, D], cung thiet bi voi `gen`.'''
    if ti_le <= 0: return x
    N, D = x.shape
    m = torch.rand((N, D), generator=gen, device=x.device) < ti_le
    r = torch.randint(pool.shape[0], (N, D), generator=gen, device=x.device)
    return torch.where(m, pool[r, torch.arange(D, device=x.device).expand(N, D)], x)


def lay_am(n_st, K, sub_st, sub_cam, gen):
    '''K chi so study NGAU NHIEN (co hoan lai) trong [0, n_st), BO study cua benh nhan trong `sub_cam` (cua batch).'''
    i = torch.randint(n_st, (K,), generator=gen, device=sub_st.device)
    return i[~torch.isin(sub_st[i], sub_cam)]


@torch.no_grad()
def muc_tieu(xr_neo, xr_ung, duong, loai, alpha, tau_s):
    '''Muc tieu mem moi hang = (1 - alpha) * [duong] / so duong + alpha * softmax(cos(EHR goc neo, EHR goc ung vien) /
       tau_s); o `loai` = 0 (khong duong, khong am). xr_* = EHR GOC (khong tang cuong). Moi hang PHAI co >= 1 duong.'''
    cung = duong.float(); cung = cung / cung.sum(1, keepdim=True)
    if alpha <= 0: return cung
    s = (F.normalize(xr_neo.float(), dim=1) @ F.normalize(xr_ung.float(), dim=1).T) / tau_s
    return (1 - alpha) * cung + alpha * torch.softmax(s.masked_fill(loai, float('-inf')), 1)


def ce_mem(logit, muc, loai):
    '''Cross-entropy voi muc tieu mem, TB theo hang; o `loai` bi bo khoi mau so (logit -inf) va co muc tieu 0.'''
    lp = torch.log_softmax(logit.masked_fill(loai, float('-inf')), 1).masked_fill(loai, 0.0)
    return -(muc * lp).sum(1).mean()


def loss_tuong_phan(za, ze, xr, cs, sub_a, sub_e, n_b, alpha, tau_s, tau):
    '''InfoNCE DOI XUNG voi muc tieu mem (docs/PLAN_EHR_TUONG_PHAN.md §4.3); 0.5 moi chieu.
       za [Ba, k] anh; ze [Ne, k] ung vien EHR = n_b study CUA batch (dung dau, khong trung) roi mau am; xr [Ne, D] EHR
       GOC cua ung vien; cs [Ba] study cua tung anh trong [0, n_b); sub_a [Ba], sub_e [Ne] benh nhan.
       Anh -> EHR: ung vien = ca Ne. EHR -> anh: neo = n_b study cua batch, ung vien = Ba anh (nhieu anh cung study ->
       phan duong chia deu). Ung vien CUNG benh nhan KHAC study bi loai khoi mau so. Tra ve (loss, ti le anh chon dung
       study cua no trong batch + mau am).'''
    za = F.normalize(za.float(), dim=1); ze = F.normalize(ze.float(), dim=1)
    j = torch.arange(ze.shape[0], device=za.device)
    duong = cs[:, None] == j[None, :]
    loai = (sub_a[:, None] == sub_e[None, :]) & ~duong
    s_ae = za @ ze.T
    l_ae = ce_mem(s_ae / tau, muc_tieu(xr[cs], xr, duong, loai, alpha, tau_s), loai)
    duong2 = duong[:, :n_b].T
    loai2 = (sub_e[:n_b, None] == sub_a[None, :]) & ~duong2
    l_ea = ce_mem(ze[:n_b] @ za.T / tau, muc_tieu(xr[:n_b], xr[cs], duong2, loai2, alpha, tau_s), loai2)
    with torch.no_grad():
        dung = float((s_ae.masked_fill(loai, float('-inf')).argmax(1) == cs).float().mean())
    return 0.5 * l_ae + 0.5 * l_ea, dung


def loss_hoi_quy(pred, y, co, nhi_phan):
    '''Hoi quy EHR tu anh (doi chung kieu PaCX): MSE o cot lien tuc, BCE (logit) o cot nhi phan, TB tren moi o CO do.
       pred, y [N, D]; co [N, D] (o co gia tri o EHR goc); nhi_phan [D]. O / cot thieu KHONG tinh.'''
    pred = pred.float(); y = y.float(); m = co.float()
    l = torch.where(nhi_phan[None, :], F.binary_cross_entropy_with_logits(pred, y.clamp(0, 1), reduction='none'),
                    (pred - y) ** 2)
    return (l * m).sum() / m.sum().clamp_min(1.0)


def phu_batch(cfg, mod, f_anh, st, T, gen, K=4096, ti_le_scarf=0.3, p_tui=0.2):
    '''Loss phu cua MOT arm tren MOT micro-batch. Tra ve None neu khong anh nao duoc vao loss phu, nguoc lai dict:
       n_anh, va `tp` (+ `dung`) neu cfg['tp'], `hq` neu cfg['hq'].
       cfg: dict(tp, hq, alpha, tau_s, bien, xao); mod: PhuEHR cua arm; f_anh [B, d] dac trung anh (CO gradient);
       st [B] chi so study (= hang2st[hang], -1 = khong vao loss phu); T: bang muc study tren thiet bi (C3c):
       n_st, sub_st, X[bien], co_tui[bien], mids, mcnt, pi_xao, Y_hq, co_hq, nhi_phan.
       Arm `xao`: ung vien (ke ca study cua chinh anh) mang NOI DUNG EHR cua study pi_xao[.] — chi cap anh <-> EHR doi;
       luat loai cung benh nhan va lay mau am van theo DANH TINH study, nhu arm khong xao.'''
    v = st >= 0
    if not bool(v.any()): return None
    stv = st[v]
    z, hq = mod.anh(f_anh[v].float())
    out = dict(n_anh=int(v.sum()))
    if cfg.get('hq'):
        out['hq'] = loss_hoi_quy(hq, T['Y_hq'][stv], T['co_hq'][stv], T['nhi_phan'])
    if cfg.get('tp'):
        us, cs = torch.unique(stv, return_inverse=True)
        ung = torch.cat([us, lay_am(T['n_st'], K, T['sub_st'], T['sub_st'][us], gen)])
        nd = T['pi_xao'][ung] if cfg.get('xao') else ung
        X = T['X'][cfg['bien']]
        xr = X[nd]
        xin = scarf(xr, X, ti_le_scarf, gen)
        if T['co_tui'][cfg['bien']]:
            giu = (torch.rand(len(nd), generator=gen, device=xr.device) >= p_tui).to(T['mcnt'].dtype)
            ze = mod.ehr(xin, T['mids'][nd], T['mcnt'][nd] * giu[:, None], gen=gen)
        else:
            ze = mod.ehr(xin, gen=gen)
        out['tp'], out['dung'] = loss_tuong_phan(z, ze, xr, cs, T['sub_st'][stv], T['sub_st'][ung], len(us),
                                                 cfg['alpha'], cfg['tau_s'], mod.tau())
    return out


def bang_study(hang_vao, study_hang, sub_hang, n_hang):
    '''Bang muc study cho loss phu. hang_vao: hang (ORDER) DUOC vao loss phu. Tra ve (hang_dd [n_st] = hang DAU cua
       moi study, sub_st [n_st], hang2st [n_hang] = chi so study, -1 neu hang khong vao). Study xep theo study_id.'''
    hang_vao = np.sort(np.asarray(hang_vao, 'int64'))
    _, dau, inv = np.unique(np.asarray(study_hang)[hang_vao], return_index=True, return_inverse=True)
    hang_dd = hang_vao[dau]; sub_st = np.asarray(sub_hang)[hang_dd]
    assert (np.asarray(sub_hang)[hang_vao] == sub_st[inv]).all(), 'mot study thuoc NHIEU benh nhan'
    hang2st = np.full(n_hang, -1, 'int64'); hang2st[hang_vao] = inv
    return hang_dd, sub_st, hang2st


def chon_giu_rieng(sub_train, ti_le, hat):
    '''Nhom benh nhan train GIU RIENG (hat co dinh): vao loss nhan nhu thuong, KHONG vao loss phu — de chan doan.'''
    u = np.unique(np.asarray(sub_train))
    return np.sort(np.random.default_rng(hat).choice(u, size=max(1, int(round(ti_le * len(u)))), replace=False))


def hoan_vi_xao(sub_st, hat):
    '''Hoan vi pi tren n study cho doi chung xao cap: pi[k] != k va sub[pi[k]] != sub[k] voi MOI k. Xep study theo benh
       nhan (thu tu benh nhan ngau nhien, hat co dinh) roi dich vong n // 2 vi tri: study cua moi benh nhan la mot doan
       lien tiep ngan hon n // 2 nen luon roi sang benh nhan khac.'''
    sub_st = np.asarray(sub_st); n = len(sub_st)
    u = np.unique(sub_st); hang = np.random.default_rng(hat).permutation(len(u))[np.searchsorted(u, sub_st)]
    tt = np.lexsort((np.arange(n), hang))
    pi = np.empty(n, 'int64'); pi[tt] = tt[(np.arange(n) + n // 2) % n]
    assert (pi != np.arange(n)).all() and (sub_st[pi] != sub_st).all(), 'hoan vi xao giu study / benh nhan cu'
    assert np.array_equal(np.sort(pi), np.arange(n)), 'pi khong phai hoan vi'
    return pi


def he_so_tang(buoc, tong, ti_le=0.1):
    '''h(t): tang tuyen tinh 0 -> 1 trong `ti_le` dau cua tong so buoc toi uu (buoc 0 -> 0).'''
    return float(min(1.0, max(0.0, buoc / max(1.0, ti_le * tong))))


def chuan_grad(loss, params):
    '''||d loss / d params|| (chuan L2 gop), KHONG cong vao .grad (autograd.grad, giu do thi cho backward sau).'''
    g = [x for x in torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True) if x is not None]
    return float(torch.sqrt(sum((x.float() ** 2).sum() for x in g))) if g else 0.0


@torch.no_grad()
def truy_hoi(za, ze, cs, sub_a, sub_e, n_ung=1000, ks=(1, 5, 10), hat=0):
    '''Chan doan (nhom GIU RIENG): anh -> EHR. Moi anh xep hang study cua no giua n_ung ung vien = no + (n_ung - 1)
       study NGAU NHIEN cua benh nhan KHAC. Tra ve {R@k}: ngau nhien = k / n_ung.'''
    za = F.normalize(za.float(), dim=1); ze = F.normalize(ze.float(), dim=1)
    S = za @ ze.T
    R = torch.rand(S.shape, generator=torch.Generator().manual_seed(hat)).to(S.device)
    R = R.masked_fill(sub_a[:, None] == sub_e[None, :], 2.0)              # cung benh nhan (ke ca chinh no): khong lam am
    k_am = int(min(n_ung - 1, int((R < 2).sum(1).min())))
    chon = R.topk(k_am, dim=1, largest=False).indices
    hang = (S.gather(1, chon) > S.gather(1, cs[:, None])).sum(1)
    return dict({f'R@{k}': float((hang < k).float().mean()) for k in ks}, n_am=k_am)


def probe_r2(X, y, nhom, lam=0.1, k=5, hat=0):
    '''Chan doan: R^2 NGOAI MAU cua hoi quy ridge y ~ X, k fold THEO NHOM (benh nhan); X chuan hoa theo fold train.'''
    X = np.asarray(X, 'float64'); y = np.asarray(y, 'float64'); nhom = np.asarray(nhom)
    u = np.unique(nhom); fold = (np.random.default_rng(hat).permutation(len(u)) % k)[np.searchsorted(u, nhom)]
    du = np.full(len(y), np.nan)
    for i in range(k):
        tr, te = fold != i, fold == i
        if not te.any() or tr.sum() < 2: continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A = (X[tr] - mu) / sd; my = y[tr].mean()
        w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (y[tr] - my))
        du[te] = ((X[te] - mu) / sd) @ w + my
    ok = np.isfinite(du)
    return float(1 - ((y[ok] - du[ok]) ** 2).sum() / max(((y[ok] - y[ok].mean()) ** 2).sum(), 1e-12))

import re
import numpy as np


def doc_ds_vung(txt):
    '''semantics/objects_detectable_by_bbox_pipeline_v1.txt -> danh sach ten vung (giu thu tu tep; moi dong "ten,").'''
    return [l.strip().rstrip(',').strip() for l in txt.splitlines() if l.strip()]


def doc_ds_thuoc_tinh(txt):
    '''semantics/attribute_relations_v1.txt -> danh sach khoa "loai|gia tri" (moi dong "loai|yes|gia tri,").'''
    out = []
    for l in txt.splitlines():
        l = l.strip().rstrip(',')
        if not l: continue
        loai, yn, gt = l.split('|', 2)
        assert yn == 'yes', l
        out.append(f'{loai}|{gt}')
    return out


def dicom_tu_entry(ten):
    '''"scene_graph/<dicom>_SceneGraph.json" -> dicom; ten khac (thu muc, tep khac) -> None.'''
    m = re.fullmatch(r'(?:.*/)?([0-9a-f]{8}(?:-[0-9a-f]{8}){4})_SceneGraph\.json', ten)
    return m.group(1) if m else None


def chon_entry_train(ten_entries, dicom_train):
    '''Chon entry THEO TEN, TRUOC khi mo: chi entry ma dicom thuoc tap train (val / test / ngoai CXR-LT bi bo).'''
    s = set(dicom_train)
    return [n for n in ten_entries if dicom_tu_entry(n) in s]


def doc_scene_graph(j, vung_idx, thay_the=None):
    '''Mot scene graph (dict) -> (hop, co, cau):
       hop: [(vung, ox1, oy1, ox2, oy2, x1, y1, x2, y2)] toa do ANH GOC (original_*) + toa do 224 cua ImaGenome (de kiem
            kich thuoc anh), chi vung co trong `vung_idx`;
       co : [(vung, khoa, thay)] thuoc tinh 'yes' ("loai|gia tri") gan vung (gom nlp|normal / nlp|abnormal), khong lap;
            thay = 1 neu den tu vung CHI co trong report (khong hop) qua `thay_the` {ten vung report: [chi so vung co hop]};
       cau: [(vung, phrase_id, section, text)] cau report CUA CHINH study gan vung CO HOP, khong lap theo (vung, phrase_id).
       Vung ngoai `vung_idx` va ngoai `thay_the` bi bo; cau cua vung thay the KHONG gan vao vung co hop.'''
    hop, co, cau = [], set(), {}
    thay_the = thay_the or {}
    for o in j.get('objects', []):
        v = vung_idx.get(o.get('bbox_name'))
        if v is None: continue
        hop.append((v, float(o['original_x1']), float(o['original_y1']), float(o['original_x2']), float(o['original_y2']),
                    float(o['x1']), float(o['y1']), float(o['x2']), float(o['y2'])))
    for a in j.get('attributes', []):
        ten = a.get('bbox_name')
        v = vung_idx.get(ten)
        dich = [v] if v is not None else list(thay_the.get(ten, []))
        if not dich: continue
        thay = 0 if v is not None else 1
        ph, pid, sec = a.get('phrases', []), a.get('phrase_IDs', []), a.get('sections', [])
        for k, ds in enumerate(a.get('attributes', [])):
            for s in ds:
                loai, yn, gt = s.split('|', 2)
                if yn == 'yes':
                    for d in dich: co.add((d, f'{loai}|{gt}', thay))
            if v is not None and k < len(ph) and k < len(pid):
                cau.setdefault((v, pid[k]), (v, pid[k], sec[k] if k < len(sec) else '', ph[k]))
    # cung (vung, khoa) co ca nguon that va thay the -> giu ban THAT (thay = 0)
    tot = {}
    for d, kk, t in co: tot[(d, kk)] = min(t, tot.get((d, kk), 1))
    return hop, sorted((d, kk, t) for (d, kk), t in tot.items()), list(cau.values())


def mat_na(vung_ds):
    '''Danh sach chi so vung -> mat na bit int64 (vung k -> bit k; can < 63 vung).'''
    m = 0
    for v in vung_ds: m |= 1 << int(v)
    return m


def vung_cua(m, R):
    '''Mat na bit -> danh sach chi so vung (tang dan).'''
    return [k for k in range(R) if (int(m) >> k) & 1]


def hop_sang_letterbox(x1, y1, x2, y2, W, H, SIZE):
    '''Hop toa do ANH GOC (W x H) -> toa do anh letterbox SIZE x SIZE, CUNG HINH HOC voi `_letterbox` cua 28c: scale theo
       canh DAI s = SIZE / max(W, H), nw = max(1, round(W s)), nh = max(1, round(H s)), dan CAN GIUA tai
       ((SIZE - nw) // 2, (SIZE - nh) // 2). Nhan mang numpy (vecto hoa). test_78f so voi chinh ham 28c tren anh that gia.'''
    W = np.asarray(W, 'float64'); H = np.asarray(H, 'float64')
    s = SIZE / np.maximum(W, H)
    nw = np.maximum(1, np.round(W * s)); nh = np.maximum(1, np.round(H * s))
    ox = (SIZE - nw) // 2; oy = (SIZE - nh) // 2
    fx, fy = nw / W, nh / H
    return (ox + np.asarray(x1) * fx, oy + np.asarray(y1) * fy, ox + np.asarray(x2) * fx, oy + np.asarray(y2) * fy)


def iou(a, b):
    '''IoU cua cac hop [n, 4] (x1, y1, x2, y2) tung cap.'''
    a = np.asarray(a, 'float64'); b = np.asarray(b, 'float64')
    iw = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    gi = iw * ih
    dt = lambda h: np.clip(h[:, 2] - h[:, 0], 0, None) * np.clip(h[:, 3] - h[:, 1], 0, None)
    return gi / np.maximum(dt(a) + dt(b) - gi, 1e-9)

import math
import torch, torch.nn as nn, torch.nn.functional as F


def bit_sang_bool(L, R):
    '''Mat na bit int64 [..., C] -> bool [..., C, R] (vung k <-> bit k).'''
    k = torch.arange(R, device=L.device)
    return ((L.unsqueeze(-1) >> k) & 1).bool()


def cung_nhan(Ya, Yc=None):
    '''[Ba, C] (va [Bc, C]) {0, 1} -> [Ba, Bc] bool: hai anh CUNG vector nhan (ca C nhan). Yc = None -> Ya voi chinh no.'''
    Ya = Ya.bool(); Yc = Ya if Yc is None else Yc.bool()
    return (Ya.unsqueeze(1) == Yc.unsqueeze(0)).all(-1)


def trong_so_rep3(Ya, La, R, beta=2.0, Yc=None, Lc=None):
    '''Phuong an 3. Ya [Ba, C] {0, 1}, La [Ba, C] int64 mat na vi tri DA BIET cua neo (0 = chua biet; bit o nhan am bi bo
       qua); Yc / Lc cua ung vien (None = chinh neo). -> W [Ba, Bc] >= 0: 0 neu khac vector nhan; trong nhom
       exp(beta (J - 0.5)) neu co it nhat mot nhan ma CA HAI biet vi tri, nguoc lai 1. Ung vien = neo -> W doi xung,
       W_ii = e^(beta / 2) neu neo biet vi tri cua it nhat mot nhan, nguoc lai 1.'''
    Yc = Ya if Yc is None else Yc; Lc = La if Lc is None else Lc
    Fa = (bit_sang_bool(La, R) & Ya.bool().unsqueeze(-1)).float()                # [Ba, C, R]
    Fc = (bit_sang_bool(Lc, R) & Yc.bool().unsqueeze(-1)).float()                # [Bc, C, R]
    na, nc = Fa.sum(-1), Fc.sum(-1)                                              # so vung moi (anh, nhan)
    giao = torch.einsum('ijr,kjr->ikj', Fa, Fc)                                  # [Ba, Bc, C]
    hop = na.unsqueeze(1) + nc.unsqueeze(0) - giao
    hai = (na > 0).float().unsqueeze(1) * (nc > 0).float().unsqueeze(0)          # ca hai biet vi tri cua nhan j
    so = hai.sum(-1)
    J = (hai * giao / hop.clamp_min(1)).sum(-1) / so.clamp_min(1)
    w = torch.where(so > 0, torch.exp(beta * (J - 0.5)), torch.ones_like(J))
    return w * cung_nhan(Ya, Yc).float()


def trong_so_rep1(Ya, Ea, tau_t=0.05, Yc=None, Ec=None):
    '''Phuong an 1 (doi chung). Ea [Ba, d] vector report goc (78e) cua neo, Ec cua ung vien (None = chinh neo).
       -> W = [cung vector nhan] * exp((cos(E_i, E_k) - 1) / tau_t).'''
    Yc = Ya if Yc is None else Yc; Ec = Ea if Ec is None else Ec
    a = F.normalize(Ea.float(), dim=-1); c = F.normalize(Ec.float(), dim=-1)
    return torch.exp((a @ c.T - 1.0) / tau_t) * cung_nhan(Ya, Yc).float()


def infonce_mem(za, zt, W, co, tau=0.07):
    '''InfoNCE DOI XUNG voi muc tieu mem, ung vien = CHINH lo. za / zt [B, k] (chua chuan hoa); W [B, B] >= 0; co [B] bool:
       anh co muc tieu (hang / cot ngoai `co` bi bo). Anh i -> text: muc tieu W_ik / sum_k W_ik; text k -> anh: W_ik / sum_i W_ik.
       -> loss trung binh tren cac hang hop le (0, van noi do thi, neu khong co hang).'''
    co = co.bool()
    if int(co.sum()) == 0: return (za.sum() + zt.sum()) * 0.0
    a = F.normalize(za[co].float(), dim=-1)
    t = F.normalize(zt[co].float(), dim=-1)
    S = a @ t.T / tau
    Wc = W[co][:, co].float()
    assert bool((Wc.sum(1) > 0).all()), 'hang muc tieu rong (W_ii phai > 0)'
    la = -((Wc / Wc.sum(1, keepdim=True)) * torch.log_softmax(S, 1)).sum(1).mean()
    lt = -((Wc / Wc.sum(0, keepdim=True)) * torch.log_softmax(S, 0)).sum(0).mean()
    return 0.5 * (la + lt)


def infonce_ung_vien(za, zt, W, loai, tau):
    '''InfoNCE DOI XUNG, muc tieu mem, ung vien NGOAI lo. za [B, k] anh (neo); zt [B + K, k] text ung vien — B dau = text cua
       CHINH cac neo (cung thu tu); W [B, B + K] >= 0 (W_ii > 0); loai [B, B + K] bool (cung benh nhan, khac study): bo khoi
       mau so va muc tieu. Anh -> text: B + K ung vien; text -> anh: B text dau, ung vien = B anh. Dung ce_mem (khoi_ehr_tuong_phan).
       -> (loss, nhom@1 = ti le neo co top-1 text voi W > 0).'''
    B = za.shape[0]
    a = F.normalize(za.float(), dim=-1); t = F.normalize(zt.float(), dim=-1)
    W = W.float().masked_fill(loai, 0.0)
    assert bool((W[:, :B].diagonal() > 0).all()), 'text cua chinh neo phai co trong so > 0'
    S = (a @ t.T) / tau
    l_at = ce_mem(S, W / W.sum(1, keepdim=True), loai)
    Wt, lo2 = W[:, :B].T, loai[:, :B].T
    l_ta = ce_mem(S[:, :B].T, Wt / Wt.sum(1, keepdim=True), lo2)
    with torch.no_grad():
        dung = float((W.gather(1, S.masked_fill(loai, float('-inf')).argmax(1, keepdim=True)) > 0).float().mean())
    return 0.5 * (l_at + l_ta), dung


class PhuRep(nn.Module):
    '''Module loss phu `rep` cua MOT arm: chieu anh (ChieuAnh cua khoi_ehr_tuong_phan: TB 40 dau ra query -> 128) + thap text
       (Linear -> LayerNorm -> GELU -> Linear -> 128) + nhiet do hoc duoc tau = 0.01 + 0.99 sigmoid(tho), khoi tao tau0 (nhu
       PhuEHR). Vector text vao DA chuan hoa z-score (78q C4: vi tri nam trong huong phuong sai nho). Khong dropout.'''
    def __init__(s, d_anh, d_text=768, d_an=512, d_chieu=128, tau0=0.1):
        super().__init__()
        s.anh = ChieuAnh(d_anh, d_an, d_chieu)
        s.text = nn.Sequential(nn.Linear(d_text, d_an), nn.LayerNorm(d_an), nn.GELU(), nn.Linear(d_an, d_chieu))
        s.tho = nn.Parameter(torch.tensor(math.log((tau0 - 0.01) / (1.0 - tau0))))

    def tau(s):
        return 0.01 + 0.99 * torch.sigmoid(s.tho)


def attn_truoc_dropout(mha, q, kv):
    '''Attention TRUNG BINH cac head cua nn.MultiheadAttention (batch_first, khong bias_k / add_zero_attn) TINH LAI tu in_proj,
       KHONG dropout. q [B, Lq, d], kv [B, Lk, d] -> [B, Lq, Lk], tong moi hang = 1. O che do eval = trong so `mha` tra ve.'''
    d, h = mha.embed_dim, mha.num_heads; dh = d // h
    W, b = mha.in_proj_weight, mha.in_proj_bias
    Q = F.linear(q, W[:d], None if b is None else b[:d])
    K = F.linear(kv, W[d:2 * d], None if b is None else b[d:2 * d])
    Bq, Lq, Lk = q.shape[0], q.shape[1], kv.shape[1]
    Q = Q.view(Bq, Lq, h, dh).transpose(1, 2); K = K.view(Bq, Lk, h, dh).transpose(1, 2)
    return torch.softmax((Q @ K.transpose(-1, -2)) / math.sqrt(dh), -1).mean(1)


def o_luoi_tu_hop(hop, co, G, o):
    '''Hop [B, R, 4] (x1, y1, x2, y2; pixel cua anh vuong G * o) + co [B, R] -> [B, R, G * G] bool: o luoi (cua so o x o, thu tu
       hang-cot nhu token patch) co TAM nam trong hop, CONG o chua TAM hop (hop nho hon mot o van co it nhat mot o). Vung
       khong co (co = False; toa do co the NaN) -> toan False.'''
    co = co.bool(); hop = torch.nan_to_num(hop.float()) * co.unsqueeze(-1)
    c = (torch.arange(G, device=hop.device, dtype=torch.float32) + 0.5) * o
    cy, cx = torch.meshgrid(c, c, indexing='ij')
    cx, cy = cx.reshape(-1), cy.reshape(-1)                                       # [P]
    x1, y1, x2, y2 = (hop[..., k].unsqueeze(-1) for k in range(4))
    trong = (cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)                       # [B, R, P]
    tx = ((hop[..., 0] + hop[..., 2]) * 0.5 / o).floor().clamp(0, G - 1).long()
    ty = ((hop[..., 1] + hop[..., 3]) * 0.5 / o).floor().clamp(0, G - 1).long()
    tam = torch.zeros_like(trong).scatter_(-1, (ty * G + tx).unsqueeze(-1), True)
    return (trong | tam) & co.unsqueeze(-1)


def mat_na_nhan(bits, trong, R):
    '''bits [B, C] int64 (vung r <-> bit r) + trong [B, R, P] bool -> [B, C, P] bool: o thuoc it nhat mot vung cua nhan.'''
    return torch.einsum('bcr,brp->bcp', bit_sang_bool(bits, R).float(), trong.float()) > 0


def loss_vung(A, M, y, eps=1e-6):
    '''A [B, C, P] attention (tong hang 1); M [B, C, P] bool vung goi y; y [B, C] nhan. Cap (anh, nhan) hop le = nhan DUONG va
       M khong rong. -> (TB -log(khoi attention trong vung + eps), so cap, khoi TB trong vung) hoac (None, 0, None).'''
    khoi = (A * M.to(A.dtype)).sum(-1)                                           # [B, C]
    hl = (y > 0.5) & M.any(-1)
    n = int(hl.sum())
    if n == 0: return None, 0, None
    return -torch.log(khoi[hl] + eps).mean(), n, float(khoi[hl].detach().mean())

import numpy as np, torch


def bang_hop_518(anh, vung, x1, y1, x2, y2, W, H, n_anh, R, S0=1022, S=518):
    '''Hop 78f (anh, vung, toa do CHUAN HOA [0, 1] theo anh goc W[anh] x H[anh]) -> [n_anh, R, 4] float32 toa do pixel cua anh
       vuong S (letterbox S0 roi resize S). Vung khong co hop -> NaN.'''
    a = np.asarray(anh, 'int64'); v = np.asarray(vung, 'int64')
    Wa = np.asarray(W, 'float64')[a]; Ha = np.asarray(H, 'float64')[a]
    bx1, by1, bx2, by2 = hop_sang_letterbox(np.asarray(x1, 'float64') * Wa, np.asarray(y1, 'float64') * Ha,
                                            np.asarray(x2, 'float64') * Wa, np.asarray(y2, 'float64') * Ha, Wa, Ha, S0)
    out = np.full((n_anh, R, 4), np.nan, 'float32')
    out[a, v] = np.stack([bx1, by1, bx2, by2], 1) * (S / S0)
    return out


def bien_doi_kem_hop(img, hop, tf):
    '''Anh PIL + hop [R, 4] (pixel cua anh, NaN = khong co) -> (anh PIL da bien doi, hop [R, 4] float32 sau bien doi (NaN =
       khong co / suy bien < 1 px), co [R] bool). `tf` = torchvision.transforms.v2 nhan (anh, BoundingBoxes) va tra cung kieu.'''
    from torchvision import tv_tensors
    hop = np.asarray(hop, 'float32'); co = np.isfinite(hop).all(1)
    bb = tv_tensors.BoundingBoxes(torch.from_numpy(np.ascontiguousarray(hop[co])).reshape(-1, 4), format='XYXY',
                                  canvas_size=(img.size[1], img.size[0]))
    img2, bb2 = tf(img, bb)
    b = bb2.as_subclass(torch.Tensor).numpy().astype('float32')
    out = np.full_like(hop, np.nan); out[co] = b
    hl = co.copy(); hl[co] = (b[:, 2] - b[:, 0] >= 1.0) & (b[:, 3] - b[:, 1] >= 1.0)
    out[~hl] = np.nan
    return img2, out, hl

def bang_cau(st_dong, ma_dong, n_st):
    '''Cap (study, cau) -> CSR (ptr long [n_st + 1], idx long [nnz]): cau cua study s = idx[ptr[s]:ptr[s + 1]], TANG dan.
       st_dong = chi so study 0..n_st-1, ma_dong = ma_cau (0 <= ma < 2^31): mang numpy song song, moi cap MOT lan (trung ->
       DUNG). Chay mot lan (numpy, CPU, O(nnz log nnz); 3M cap ~0.5 s); ben goi chuyen .to(dev).'''
    st = np.asarray(st_dong); ma = np.asarray(ma_dong)
    assert st.ndim == 1 and st.shape == ma.shape, (st.shape, ma.shape)
    assert np.issubdtype(st.dtype, np.integer) and np.issubdtype(ma.dtype, np.integer), (st.dtype, ma.dtype)
    st = st.astype('int64'); ma = ma.astype('int64'); n_st = int(n_st)
    assert len(st) == 0 or (st.min() >= 0 and st.max() < n_st and ma.min() >= 0 and ma.max() < 2 ** 31), \
        'chi so study ngoai [0, n_st) / ma cau ngoai [0, 2^31) -> DUNG'
    k = np.sort(st * (1 << 31) + ma)                   # khoa ghep: xep theo study roi theo ma cau (1 lan sort, nhanh)
    assert not (k[1:] == k[:-1]).any(), 'cap (study, cau) TRUNG -> DUNG'
    ptr = np.zeros(n_st + 1, 'int64'); ptr[1:] = np.cumsum(np.bincount(k >> 31, minlength=n_st))
    return torch.from_numpy(ptr), torch.from_numpy(k & ((1 << 31) - 1))


def _cau_cua_st(us, ptr, idx):
    '''Cau cua cac study `us` (long [k], TANG dan; -1 = khong co study -> doan rong) -> (seg [T] = vi tri study trong us,
       dau [T] = vi tri dau doan cua phan tu, cau [T]): noi tiep theo thu tu us, trong moi study tang dan. Study >= 0 ma doan
       CSR RONG (khong co cau) -> DUNG: hang cua no phai co st = -1. O(k + T).'''
    if us.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=us.device); return z, z, z
    assert int(us[-1]) < ptr.shape[0] - 1, 'chi so study >= n_st cua bang_cau -> DUNG'
    co = us >= 0; uc = us.clamp_min(0)
    d0 = ptr[uc]; n = torch.where(co, ptr[uc + 1] - d0, torch.zeros_like(d0))
    T, n_rong = torch.stack([n.sum(), (co & (n == 0)).sum()]).tolist()            # MOT lan dong bo cho ca hai so
    assert n_rong == 0, f'{n_rong} study (st >= 0) KHONG co cau -> hang cua no phai co st = -1 (hop_le False) -> DUNG'
    seg = torch.repeat_interleave(torch.arange(us.numel(), device=us.device), n, output_size=T)
    dau = (torch.cumsum(n, 0) - n)[seg]
    return seg, dau, idx[d0[seg] + torch.arange(T, device=us.device) - dau].long()


def lay_cau_lo(st, ptr, idx, toi_da, gen):
    '''Ung vien cau cua lo -> ma_u long [U] TANG dan, khong trung. Voi moi study KHAC NHAU s >= 0 trong `st` (xet theo s TANG
       dan): rut min(toi_da, n_s) cau KHAC NHAU cua s, deu, khong hoan lai (khoa ngau nhien tu `gen` + xep hang trong tung
       doan); hop lai. Rut theo STUDY (hai anh cung study: mot lan rut; thu tu hang khong doi ket qua). st = -1 -> khong gop
       gi; khong hang nao st >= 0 -> tensor rong; study st >= 0 KHONG co cau -> DUNG (phai st = -1). toi_da None -> lay het,
       khong rut so ngau nhien. Khong cham dong ngau nhien chung. O(T log T), T = tong so cau cua cac study trong lo.'''
    st = st.long(); vo = torch.zeros(0, dtype=torch.long, device=st.device)
    us = torch.unique(st); us = us[us >= 0]
    seg, dau, cau = _cau_cua_st(us, ptr, idx)
    T = cau.numel()
    if T == 0: return vo
    if toi_da is None: return torch.unique(cau)
    assert int(toi_da) >= 1 and gen is not None, 'lay_cau_lo: toi_da >= 1 va generator RIENG (gen)'
    khoa = seg * (1 << 31) + torch.randint(1 << 31, (T,), generator=gen, device=st.device)   # theo study, trong study ngau nhien
    o = torch.argsort(khoa, stable=True)          # seg da tang dan -> doan k van o [dau_k, dau_k + n_k) sau khi xep
    hang = torch.arange(T, device=st.device) - dau
    return torch.unique(cau[o[hang < int(toi_da)]])


def ma_tran_duong(st, ptr, idx, ma_u):
    '''P bool [B, U]: P[b, u] = ma_u[u] thuoc TAP CAU DAY DU cua study st[b] (st[b] = -1 -> ca hang False; study st >= 0 KHONG
       co cau -> DUNG, ca khi U = 0). ma_u bat ky thu tu (0 <= ma < 2^31). Khoa ghep (vi tri study trong lo, ma cau) tang ngat
       -> searchsorted. O(T + B U log T).'''
    st = st.long(); B, U = st.shape[0], ma_u.shape[0]
    P0 = torch.zeros(B, U, dtype=torch.bool, device=st.device)
    if B == 0 or not bool((st >= 0).any()): return P0
    us, inv = torch.unique(st, return_inverse=True)
    seg, _, cau = _cau_cua_st(us, ptr, idx)                              # >= 1 cau (study st >= 0 rong -> DUNG o day)
    if U == 0: return P0
    khoa = seg * (1 << 32) + cau
    assert bool((khoa[1:] > khoa[:-1]).all()), 'idx khong tang ngat trong study (dung bang_cau) -> DUNG'
    q = inv[:, None] * (1 << 32) + ma_u.long()[None, :]
    j = torch.searchsorted(khoa, q).clamp_max(khoa.numel() - 1)
    return (khoa[j] == q) & (st >= 0)[:, None]


def mat_na_loai(P, sub, st):
    '''loai bool [B, U] = ~P[b, u] VA co b' cung benh nhan (sub[b'] == sub[b]), KHAC study (st[b'] != st[b]), st[b'] >= 0,
       P[b', u]: cau cua study khac cung benh nhan ma khong thuoc study nay -> KHONG lam am. O(B^2 U) (mot matmul 0/1 float32).'''
    P = P.bool()
    A = (sub[:, None] == sub[None, :]) & (st[:, None] != st[None, :]) & (st[None, :] >= 0)
    with torch.autocast(device_type=P.device.type, enabled=False):
        E = (A.float() @ P.float()) > 0
    return E & ~P


def trong_so_cau(tan_so):
    '''w = 1 / log(1 + tan_so) float32 (cau_hinh dich_78s cau.trong_so): cau rat pho bien (vd cau "binh thuong") bot chi phoi
       gradient. tan_so = so study train chua cau, >= 1 (khong -> DUNG). Vao tensor -> giu thiet bi; numpy -> CPU.'''
    t = tan_so if torch.is_tensor(tan_so) else torch.as_tensor(np.asarray(tan_so))
    assert bool((t >= 1).all()), 'tan_so < 1 -> DUNG'
    return (1.0 / torch.log1p(t.double())).float()


def infonce_cau(z_anh, z_cau, P, loai, w, tau, hop_le=None):
    '''InfoNCE DOI XUNG, DA DUONG, muc cau; ung vien CHI trong lo. z_anh [B, k], z_cau [U, k] (chua chuan hoa); P, loai bool
       [B, U] (ma_tran_duong, mat_na_loai); w [U] (trong_so_cau[ma_u]); tau so / tensor 0-d (NhietDo); hop_le [B] bool (None =
       moi hang): hang False bi bo HAN (khong lam neo, khong lam ung vien; z_anh cua no dat 0 TRUOC khi chuan hoa -> NaN / inf
       o hang do khong lot vao loss, gradient 0). O loai KHONG BAO GIO la duong (duong = P & ~loai).
       S = normalize(z_anh) @ normalize(z_cau).T / tau, float32 (tat autocast).
       - anh -> cau: neo = hang hop le co >= 1 duong; muc tieu ti le P[b, u] w[u]; o loai bi bo (ce_mem); TB deu tren neo.
       - cau -> anh: neo = cau co >= 1 hang hop le duong; muc tieu deu tren cac hang do; ung vien = hang hop le tru loai.T;
         TB co trong so w[u] / TB(w tren cac neo).
       -> (loss = 0.5 (anh->cau + cau->anh), dung1 = tensor 0-d khong grad: ti le neo anh->cau co top-1 (ngoai o loai) la
       duong). Khong neo nao -> loss = +0.0 (van noi do thi, gradient 0), dung1 = nan: ben goi CHI cong dung1 khi huu han, dem
       rieng. Hang / cot KHONG lam neo duoc go mat na truoc log_softmax -> khong NaN. Khong dong bo thiet bi (khong .item()).
       O(B U k).'''
    B, U, dev = z_anh.shape[0], z_cau.shape[0], z_anh.device
    hl = torch.ones(B, dtype=torch.bool, device=dev) if hop_le is None else hop_le.bool()
    za = z_anh.float().masked_fill(~hl[:, None], 0.0)                    # hang KHONG hop le = 0 (NaN / inf o do khong lot)
    if B == 0 or U == 0:
        return (za.sum() + z_cau.float().sum()) * 0.0 + 0.0, torch.full((), float('nan'), device=dev)
    loai = loai.bool()
    Pd = P.bool() & ~loai & hl[:, None]                                  # duong HOP LE
    with torch.autocast(device_type=dev.type, enabled=False):
        S = (F.normalize(za, dim=1) @ F.normalize(z_cau.float(), dim=1).T) / tau
        w = w.float()
        neo1 = Pd.any(1); n1 = neo1.sum()
        Pw = Pd.float() * w[None, :]
        lo1 = loai & neo1[:, None]
        l1 = ce_mem(S, Pw / Pw.sum(1, keepdim=True).clamp_min(1e-12), lo1) * (B / n1.clamp_min(1))
        neo2 = Pd.any(0); a = w * neo2.float()
        P2 = Pd.T.float()
        lo2 = (loai.T | ~hl[None, :]) & neo2[:, None]
        l2 = ce_mem(S.T, P2 / P2.sum(1, keepdim=True).clamp_min(1.0) * a[:, None], lo2) * (U / a.sum().clamp_min(1e-12))
        loss = 0.5 * (l1 + l2) + 0.0                                     # + 0.0: -0.0 -> 0.0 khi khong co neo
    with torch.no_grad():
        top = S.masked_fill(lo1, float('-inf')).argmax(1, keepdim=True)
        dung1 = (Pd.gather(1, top)[:, 0] & neo1).sum().float() / n1.float()
    return loss, dung1


class ThapVector(nn.Module):
    '''Thap cho vector cau BiomedBERT DONG CUNG (78t cau_bert, 78u B1): x [U, d_vao] float32 DA z-score boi ben goi
       (cau_emb_thong_ke) -> Linear -> LayerNorm -> GELU -> Linear -> [U, d_ra] (s.d_ra). Khong dropout: forward khong rut so
       ngau nhien. Khoi tao (nn.Linear) TIEU dong ngau nhien chung -> ben goi tao trong torch.random.fork_rng voi hat rieng.'''
    def __init__(s, d_vao=768, d_an=512, d_ra=256):
        super().__init__()
        s.d_ra = int(d_ra)
        s.mlp = nn.Sequential(nn.Linear(d_vao, d_an), nn.LayerNorm(d_an), nn.GELU(), nn.Linear(d_an, d_ra))

    def forward(s, x):
        return s.mlp(x)


class ThapBert(nn.Module):
    '''Thap van ban HOC CUNG (CLIP; 78t cau_clip, 78u B2): tok [U, L] (cau_token.npy: [CLS] ... [SEP], pad 0 ben phai) -> mask =
       tok != 0 -> mdl(input_ids=tok, attention_mask=mask.long()).last_hidden_state -> TB theo mask (float32) -> Linear ->
       LayerNorm -> GELU -> Linear -> [U, d_ra] (s.d_ra). Cot chi co pad bi cat truoc khi vao mdl (pad KHONG doi ket qua). mdl la
       module con (nhom LR rieng: s.mdl / s.dau; hoc mot phan: mo_bert_tren). mdl.training (dropout) -> BAT BUOC `gen`: forward
       cua mdl chay voi dong toan cuc (CPU / CUDA cua tok) TAM gieo tu gen trong fork_rng roi tra lai nguyen trang -> dong chung
       khong doi. eval: khong can gen; gen truyen vao KHONG bi tieu. Khoi tao s.dau (nn.Linear) TIEU dong ngau nhien chung ->
       ben goi tao trong torch.random.fork_rng voi hat rieng.
       chuan = (TB, DO LECH) [2, d] co dinh (78s cau_emb_thong_ke.npy) -> vector TB theo mask duoc z-score TRUOC s.dau (buffer,
       khong hoc): cung dau vao voi ThapVector tren vector dong cung DA z-score -> cau_clip - cau_bert (78t) va B2 - B1 (78u) CHI
       khac o tinh hoc cung (quyet dinh D-T2, review 07/10: vector BiomedBERT TB tho cos ~0.98 -> khong chuan hoa = them mot yeu
       to thu hai). s.dau LUON chay float32 (tat autocast) nhu ThapVector chay ngoai autocast.'''
    def __init__(s, mdl, d_an=512, d_ra=256, d_mdl=None, chuan=None):
        super().__init__()
        s.mdl = mdl; s.d_ra = int(d_ra)
        d = int(mdl.config.hidden_size if d_mdl is None else d_mdl)
        s.dau = nn.Sequential(nn.Linear(d, d_an), nn.LayerNorm(d_an), nn.GELU(), nn.Linear(d_an, d_ra))
        if chuan is None:
            s.register_buffer('chuan_tb', None); s.register_buffer('chuan_sd', None)
        else:
            tb = torch.as_tensor(np.asarray(chuan[0], dtype='float32')).clone()
            sd = torch.as_tensor(np.asarray(chuan[1], dtype='float32')).clone()
            assert tuple(tb.shape) == (d,) and tuple(sd.shape) == (d,), f'ThapBert: chuan phai la [2, {d}] -> DUNG'
            assert bool(torch.isfinite(tb).all()) and bool((sd > 0).all()), 'ThapBert: chuan khong huu han / do lech <= 0 -> DUNG'
            s.register_buffer('chuan_tb', tb); s.register_buffer('chuan_sd', sd)

    def _dau32(s, x):
        '''z-score co dinh (neu co chuan) roi s.dau, LUON float32 (tat autocast cua thiet bi).'''
        x = x.float()
        if s.chuan_tb is not None: x = (x - s.chuan_tb) / s.chuan_sd
        with torch.autocast(device_type=x.device.type, enabled=False):
            return s.dau(x)

    def forward(s, tok, gen=None):
        tok = tok.long(); dv = tok.device
        if tok.shape[0] == 0 or tok.shape[1] == 0:
            return s._dau32(torch.zeros(tok.shape[0], s.dau[0].in_features, device=dv))
        mask = tok != 0
        L = max(1, int((torch.arange(1, tok.shape[1] + 1, device=dv) * mask.any(0)).max()))
        tok, mask = tok[:, :L], mask[:, :L]
        if s.mdl.training:
            assert gen is not None, 'ThapBert: mdl.train() (dropout) phai co generator RIENG (gen)'
            assert dv.type in ('cpu', 'cuda'), f'ThapBert: thiet bi {dv} chua ho tro fork_rng'
            hat = int(torch.randint(1 << 62, (1,), generator=gen, device=gen.device))
            with torch.random.fork_rng(devices=[dv] if dv.type == 'cuda' else [], device_type=dv.type):
                if dv.type == 'cuda':
                    with torch.cuda.device(dv): torch.cuda.manual_seed(hat)
                else:
                    torch.default_generator.manual_seed(hat)
                h = s.mdl(input_ids=tok, attention_mask=mask.long()).last_hidden_state
        else:
            h = s.mdl(input_ids=tok, attention_mask=mask.long()).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return s._dau32((h.float() * m).sum(1) / m.sum(1).clamp_min(1.0))


def mo_bert_tren(mdl, n_mo):
    '''Hoc cung MOT PHAN bo ma hoa BERT (78t cau_clip, 78u B2; kieu ConVIRT): dong cung MOI tham so cua mdl (embeddings, cac
       tang duoi, pooler neu co) roi MO n_mo tang encoder TREN CUNG (mdl.encoder.layer[so tang - n_mo:]; 0 <= n_mo <= so tang;
       BiomedBERT 12 tang, THAP_MO 6). Dau chieu ThapBert.dau khong thuoc mdl -> van hoc. Chi doi requires_grad (dropout moi
       tang van chay khi .train(), rut tu gen trong ThapBert); goi TRUOC khi tao optimizer; goi lai -> dong lai tu dau. Khong
       rut so ngau nhien. Tra ve mdl.'''
    tang = mdl.encoder.layer; n = len(tang); n_mo = int(n_mo)
    assert 0 <= n_mo <= n, f'mo_bert_tren: n_mo {n_mo} ngoai [0, {n}] -> DUNG'
    mdl.requires_grad_(False); tang[n - n_mo:].requires_grad_(True)
    return mdl


class NhietDo(nn.Module):
    '''Nhiet do hoc duoc tau = 0.01 + 0.99 * sigmoid(tho) (nhu PhuEHR): luon trong (0.01, 1), khoi tao tau0. nd() = nd.tau().
       Khoi tao khong rut so ngau nhien.'''
    def __init__(s, tau0=0.07):
        super().__init__()
        assert 0.01 < tau0 < 1.0, tau0
        s.tho = nn.Parameter(torch.tensor(math.log((tau0 - 0.01) / (1.0 - tau0))))

    def tau(s):
        return 0.01 + 0.99 * torch.sigmoid(s.tho)

    def forward(s):
        return s.tau()


class PhuCau(nn.Module):
    '''Module loss phu `cau` cua MOT arm / luot (78t cau_bert / cau_clip, 78u B1 / B2): .anh = ChieuAnh(d_anh -> d_an ->
       d_chieu) (khoi_ehr_tuong_phan; khong dropout, khong dau hoi quy); .text = thap truyen vao (ThapVector / ThapBert, phai co
       thap.d_ra == d_chieu); .nhiet = NhietDo(tau0). z_anh(f) [B, d_anh] -> [B, d_chieu] (vao .float()); tau() = .nhiet.tau().
       Thap do ben goi tao TRUOC; trong PhuCau chi .anh tieu so ngau nhien (khoi tao) -> tao PhuCau trong torch.random.fork_rng
       voi hat rieng (cung hat -> .anh trung voi ChieuAnh(d_anh, d_an, d_chieu) tao rieng).'''
    def __init__(s, d_anh, thap, d_an=512, d_chieu=256, tau0=0.07):
        super().__init__()
        assert getattr(thap, 'd_ra', None) == d_chieu, f'PhuCau: thap.d_ra {getattr(thap, "d_ra", None)} != d_chieu {d_chieu}'
        s.anh = ChieuAnh(d_anh, d_an=d_an, d_chieu=d_chieu); s.text = thap; s.nhiet = NhietDo(tau0)

    def z_anh(s, f):
        return s.anh(f.float())[0]

    def tau(s):
        return s.nhiet.tau()

class DauDay(nn.Module):
    '''Lop doc 1 x 1 (Linear d -> R + 1 + K) tren MOI token patch (last_hidden_state[:, 1:], SAU LayerNorm cuoi cua backbone) ->
       logit float32 [B, P, R + 1 + K]: R kenh thuoc vung, 1 kenh bat thuong, K kenh ket cau (tach_day). Khong dropout. Bo di
       luc cham. Khoi tao TIEU dong ngau nhien chung -> ben goi tao trong fork_rng.'''
    def __init__(s, d=768, R=36, K=5):
        super().__init__()
        s.R, s.K = int(R), int(K)
        s.lin = nn.Linear(d, s.R + 1 + s.K)

    def forward(s, patch):
        return s.lin(patch.float())


def tach_day(o, R):
    '''logit [..., R + 1 + K] -> (thuoc [..., R], bat [...], ket [..., K]).'''
    return o[..., :R], o[..., R], o[..., R + 1:]


def dich_o_day(hop, co, g, bat_thuong, ket_cau, G, o):
    '''Dich day theo o luoi tu hop DA BIEN DOI. hop [B, R, 4] (x1, y1, x2, y2; pixel cua anh vuong G * o; NaN cho phep o vung
       khong co), co [B, R] bool (vung co hop hop le), g [B] = hang anh trong bang 78s (-1 = anh khong co dich), bat_thuong
       [n_anh, R] int, ket_cau [n_anh, R, K] int (CUNG thiet bi, gia tri -1 / 0 / 1). -> (thuoc [B, P, R], bat [B, P], ket
       [B, P, K]) int8 trong {-1, 0, 1}, P = G * G. Vector hoa; O(B R P K).'''
    g = g.long()
    co = co.bool() & (g >= 0)[:, None]                                  # anh khong dich -> khong vung nao
    tr = o_luoi_tu_hop(hop, co, G, o)                                   # [B, R, P] bool; vung co = False -> toan False
    gi = g.clamp_min(0)
    bt, kc = bat_thuong[gi].long(), ket_cau[gi].long()                  # [B, R], [B, R, K]
    am = torch.full((), -1, dtype=torch.int8, device=tr.device)
    thuoc = torch.where(co[:, None, :], tr.transpose(1, 2).to(torch.int8), am)
    with torch.autocast(device_type=tr.device.type, enabled=False):
        trf = tr.float()
        b1 = torch.einsum('brp,br->bp', trf, (bt == 1).float()) > 0
        b0 = torch.einsum('brp,br->bp', trf, (bt == 0).float()) > 0
        k1 = torch.einsum('brp,brk->bpk', trf, (kc == 1).float()) > 0
        k0 = torch.einsum('brp,brk->bpk', trf, (kc == 0).float()) > 0
    mot, khong = torch.ones_like(am), torch.zeros_like(am)
    bat = torch.where(b1, mot, torch.where(b0, khong, am))
    ket = torch.where(k1, mot, torch.where(k0, khong, am))
    return thuoc, bat, ket


def bce_mat_na(logit, dich):
    '''BCE-with-logits (float32) TB tren o dich >= 0; o dich = -1 gradient DUNG 0. Khong o hop le -> None.'''
    m = dich >= 0
    n = int(m.sum())
    if n == 0: return None
    l = F.binary_cross_entropy_with_logits(logit.float(), dich.clamp_min(0).float(), reduction='none')
    return (l * m.float()).sum() / n


def loss_day(out, dich, R):
    '''out = DauDay(patch) [B, P, R + 1 + K]; dich = dich_o_day(...) -> dict {'thuoc_vung', 'bat_thuong', 'ket_cau': loss} CHI
       gom phan co >= 1 o hop le (dict rong neu khong phan nao).'''
    lg, kq = tach_day(out, R), {}
    for ten, l_, d_ in zip(('thuoc_vung', 'bat_thuong', 'ket_cau'), lg, dich):
        v = bce_mat_na(l_, d_)
        if v is not None: kq[ten] = v
    return kq


def auc_day(s, y):
    '''AUC chinh xac (Mann-Whitney; hang trung binh khi hoa) cua diem s voi nhan y trong {-1, 0, 1} (-1 bo qua); s, y cung so
       phan tu (lam phang). Thieu mot lop -> nan. float64.'''
    s = torch.as_tensor(s).reshape(-1).double(); y = torch.as_tensor(y).reshape(-1).to(s.device)
    m = y >= 0
    s, y = s[m], y[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0: return float('nan')
    _, inv, dem = torch.unique(s, return_inverse=True, return_counts=True)
    cuoi = torch.cumsum(dem, 0)
    hang = (cuoi - dem).double() + (dem.double() + 1.0) / 2.0          # hang trung binh (tu 1) cua moi gia tri
    return float((hang[inv][y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class TichLuyAUC:
    '''AUC tung kenh tich luy qua nhieu lo bang HISTOGRAM logit: n_bin o deu tren [-gioi, gioi] (ngoai -> o bien), mot histogram
       cho duong / am moi kenh. Trung o -> tinh nua (nhu hang trung binh) -> sai so so voi auc_day chi do gop o. Khong luu diem
       tung o luoi; tich luy tren thiet bi cua lo dau. them(logit [..., n_kenh], dich [..., n_kenh]); auc() -> list (nan khi thieu
       mot lop).'''
    def __init__(s, n_kenh, n_bin=4096, gioi=16.0):
        s.n_kenh, s.n_bin, s.gioi = int(n_kenh), int(n_bin), float(gioi)
        s.h = None

    @torch.no_grad()
    def them(s, logit, dich):
        L = logit.detach().float().reshape(-1, s.n_kenh); D = dich.reshape(-1, s.n_kenh).to(L.device)
        if s.h is None: s.h = torch.zeros(2, s.n_kenh * s.n_bin, dtype=torch.float64, device=L.device)
        o = ((L + s.gioi) * (s.n_bin / (2.0 * s.gioi))).floor().clamp(0, s.n_bin - 1).long()
        o = o + torch.arange(s.n_kenh, device=L.device)[None, :] * s.n_bin
        for v in (0, 1):
            m = D == v
            s.h[v] += torch.bincount(o[m], minlength=s.n_kenh * s.n_bin).double()

    def auc(s):
        if s.h is None: return [float('nan')] * s.n_kenh
        h0, h1 = (x.reshape(s.n_kenh, s.n_bin) for x in s.h.cpu())
        n0, n1 = h0.sum(1), h1.sum(1)
        duoi = torch.cumsum(h0, 1) - h0                                 # so am o o THAP hon
        a = (h1 * (duoi + 0.5 * h0)).sum(1) / (n0 * n1).clamp_min(1.0)
        return [float(v) if (p > 0 and q > 0) else float('nan') for v, p, q in zip(a.tolist(), n0.tolist(), n1.tolist())]

# ---- cau hinh trong kho (NGUYEN VAN cau_hinh/): dich_78s.json (cau / vung_day; 78s cung dung), muc_tieu_78q.json (5 nhan phoi) ----
DICH_78S = json.loads(r'''{
 "phien_ban": "dich_v1",
 "_ghi_chu": "Claude viet 07/10 (docs/PLAN_SAU_78G.md muc 5, 6.2, 6.5; spec 78s 2.2 - 2.3) - GIA DINH NGHIEN CUU, Tuan duyet / sua. Dich phu LUC TRAIN (78t / 78u), -1 = mask (khong vao loss); chi anh / report / ImaGenome cua benh nhan TRAIN. T1 thay (nhan duong, theo ngu canh MUC STUDY cua report, khoi_ngu_canh): khang_dinh va nghi_ngo -> 1 (bac si thay / nghi co tren phim nay); phu_dinh, da_het (da het = khong con tren phim nay) va ngoai_chi_dinh (nhan duong chi tu cau hoi lam sang) -> 0; tien_su (co the con / khong), ngoai_so_bo (doc so bo, chua chot), ngoai_khac (vd muc COMPARISON), khong_nhac (tu khoa sot), khong_co_FI, khong_report, khong_tu_khoa -> -1; nhan am -> 0 (nhan khong tu khoa -> -1 moi anh). T2 chac (chi nhan duong khang_dinh / nghi_ngo): khang_dinh 1, nghi_ngo 0; khang_dinh ma ImaGenome gan 'severity|hedge' (thay 0) tren vung vi tri cua nhan o anh thang cung study -> mask (hai parser xung dot); nghi_ngo + hedge = dong y -> giu 0; danh sach xung dot (xung_dot_hedge) la tham so BAT BUOC cua dich_T1_T2. T3 muc_do (0 mild < 1 moderate < 2 severe, muc cao nhat tren cac vung vi_tri 78q): CHI nhan ma bac si X-quang thuong ghi muc do / luong (small / mild / minimal - moderate - large / severe / extensive): kich thuoc tim / trung that / dong mach chu (Cardiomegaly, Enlarged Cardiomediastinum, Tortuous Aorta, Calcification of the Aorta, Hernia), dich / khi (Pleural Effusion, Pneumothorax, Hydropneumothorax, Pneumomediastinum, Pneumoperitoneum, Edema), nhu mo / mang phoi (Atelectasis, Consolidation, Lung Opacity, Emphysema, Fibrosis, Pleural Thickening - Pleural Thickening va Fibrosis cung lay vi tri tu thuoc tinh ImaGenome 'pleural/parenchymal scarring' va cung hay ghi minimal / mild / extensive -> xu ly nhu nhau); loai not / khoi (do bang cm), gay xuong, thiet bi, nhan chan doan (Pneumonia, Tuberculosis) va nhan KHONG BAO GIO co vi tri 78q. vi_tri 78q CHI tu vung CO HOP that (thay == 0, khoi_muc_tieu.mat_na_vi_tri_that): ImaGenome gan 'rib fracture' / 'subcutaneous air' 100% vao chest wall / neck KHONG hop (anh_xa_imagenome.json; 78q C2 05/10: biet vi tri Rib Fracture, Subcutaneous Emphysema 0%) -> cot T3 / T4 cua hai nhan nay luon -1 (dau chet o 78t) -> BO khoi T3_nhan / T4_nhan (sua 07/10 sau ra soat vong 1); Kyphosis, Fissure ... (khong vi tri) cung khong co. 78s C3 nen in so o KHONG mask moi cot T3 / T4 de cot chet hien ngay. GIA DINH: ImaGenome doi small/trace -> mild, large -> severe (chua kiem); muc do o 78f gan theo VUNG khong theo phat hien -> co the la muc cua phat hien khac cung vung (nhieu, test 78t do). T4 ben: CHI nhan co ben co nghia VA co the co vi tri ImaGenome; loai nhan giua / ben co dinh (Azygos Lobe, Cardiomegaly, Enlarged Cardiomediastinum, Tortuous Aorta, Calcification of the Aorta, Kyphosis, Hernia, Pneumomediastinum, Pneumoperitoneum, Pulmonary Hypertension, Support Devices, Normal) nhan khong anh xa ImaGenome (Fissure, Infarction, Pulmonary Embolism, Pleural Other) va nhan chi co vung KHONG hop (Rib Fracture, Subcutaneous Emphysema: vi_tri 78q luon 0, xem T3); Edema / Emphysema giu (phan lon hai_ben, mot ben co y nghia lam sang). Ben lay tu TEN vung cua vi_tri 78q: T4_vung_trai / _phai = moi vung 'left ...' / 'right ...' trong 36 vung co hop TRU 'right atrium' (buong tim, dung cho dau ong thong); chest wall khong co hop (78f doi sang lung); thuoc tinh 'laterality|*' cua ImaGenome KHONG dung (ten vung da mang ben). T5 ong_tb: 12 tubesandlines + 5 device (thu tu tu vung), moi thay, max theo study, lan sang moi anh cua study (ke ca nghieng); 78f bo thuoc tinh tren vung chi co trong report khong co vung thay the (arm, breast) -> T5 co the sot (vd PICC chi gan 'right arm'). T6 ky_thuat (technicalassessment, moi thay) va T7 ket_cau (texture, thay 0) chi anh thang co scene graph, khong lan. nguyen_mau: KMeans k=4, n_init=10, random_state=0, sample_weight = trong so khang dinh w (duoi), fit voi 1 luong OpenMP / BLAS (threadpoolctl; tat dinh bit-for-bit giua cac lan chay cung moi truong - nhieu luong cong theo thu tu khac nhau, lech ~1e-6); < 20 cau khang dinh hoac nhan khong tu khoa -> du phong mo ta lop v1 (meta 'fallback'). TRONG SO cau khang dinh (txt_cau_lop / nguyen_mau_lop; quyet dinh 07/10 sau ra soat vong 1): tinh THEO TUNG LAN XUAT HIEN - w[u] cua nhan j = so study train DUONG muc study voi j ma CHINH lan xuat hien cua cau u trong report do khang dinh j ('khang_dinh' trong khoi_ngu_canh.ngu_canh_cau cua van ban lan do; cau lap cung khoa trong mot report -> bat ky lan nao khang dinh, nhu uu tien khang_dinh cua ngu canh study); chi cau w > 0. Dinh danh cau = khoa 77a (bo dau cau, chu hoa, so thu tu) gop cac bien the vao MOT ma_cau nhung ngu canh NegEx xet tren TUNG bien the ('?' -> nghi_ngo; ';' ':' cat pham vi) -> trong so khong phu thuoc study nao gap truoc (khoi_cau_report: bang_cau_report tra bien_the, trong_so_khang_dinh dung). Vector cua u = text LAN DAU (study tang, I roi F; bien the khac chi lech dau cau / chu hoa). cau: min_tu 2 (so tu [a-z]{2,} cua khoa 77a), max_tok 64; toi_da_moi_study 6 (so cau ung vien toi da lay tu moi study moi buoc, khoi_cau.lay_cau_lo). trong_so w = 1/log(1+tan_so) (khoi_cau.trong_so_cau; tan_so = so study train co cau) vao loss cau khoi_cau.infonce_cau o 78t / 78u: anh -> cau: dich cua hang b ti le P[b,u]*w[u] (cau pho bien nhu 'no pneumothorax' nhe hon, cau hiem nang hon); cau -> anh: moi cau u co >= 1 hang duong la mot neo, loss cac neo lay trung binh co trong so w[u]/mean(w tren cac neo); w deu (moi cau cung tan_so) -> dich deu tren cac cau duong va trung binh deu tren neo = dich deu cua plan (PLAN_SAU_78G muc 3). vung_day (B4): bat_thuong nlp|abnormal (1) / nlp|normal (0) thay 0, ket_cau 5 texture thay 0 (vung binh thuong -> 0), r = he so lambda tuong doi (78u).",
 "T1": {
  "khang_dinh": 1,
  "nghi_ngo": 1,
  "tien_su": -1,
  "da_het": 0,
  "phu_dinh": 0,
  "ngoai_so_bo": -1,
  "ngoai_chi_dinh": 0,
  "ngoai_khac": -1,
  "khong_nhac": -1,
  "khong_co_FI": -1,
  "khong_tu_khoa": -1,
  "khong_report": -1
 },
 "T2": {
  "khang_dinh": 1,
  "nghi_ngo": 0,
  "tien_su": -1,
  "da_het": -1,
  "phu_dinh": -1,
  "ngoai_so_bo": -1,
  "ngoai_chi_dinh": -1,
  "ngoai_khac": -1,
  "khong_nhac": -1,
  "khong_co_FI": -1,
  "khong_tu_khoa": -1,
  "khong_report": -1,
  "hedge_imagenome": "mask_khi_xung_dot"
 },
 "T3_nhan": [
  "Atelectasis",
  "Calcification of the Aorta",
  "Cardiomegaly",
  "Consolidation",
  "Edema",
  "Emphysema",
  "Enlarged Cardiomediastinum",
  "Fibrosis",
  "Hernia",
  "Hydropneumothorax",
  "Lung Opacity",
  "Pleural Effusion",
  "Pleural Thickening",
  "Pneumomediastinum",
  "Pneumoperitoneum",
  "Pneumothorax",
  "Tortuous Aorta"
 ],
 "T3_muc": [
  "mild",
  "moderate",
  "severe"
 ],
 "T4_nhan": [
  "Adenopathy",
  "Atelectasis",
  "Clavicle Fracture",
  "Consolidation",
  "Edema",
  "Emphysema",
  "Fibrosis",
  "Fracture",
  "Granuloma",
  "Hydropneumothorax",
  "Infiltration",
  "Lobar Atelectasis",
  "Lung Lesion",
  "Lung Opacity",
  "Mass",
  "Nodule",
  "Pleural Effusion",
  "Pleural Thickening",
  "Pneumonia",
  "Pneumothorax",
  "Round(ed) Atelectasis",
  "Tuberculosis"
 ],
 "T4_vung_trai": [
  "left lung",
  "left apical zone",
  "left upper lung zone",
  "left mid lung zone",
  "left lower lung zone",
  "left hilar structures",
  "left costophrenic angle",
  "left hemidiaphragm",
  "left clavicle",
  "left cardiac silhouette",
  "left cardiophrenic angle",
  "left upper abdomen"
 ],
 "T4_vung_phai": [
  "right lung",
  "right apical zone",
  "right upper lung zone",
  "right mid lung zone",
  "right lower lung zone",
  "right hilar structures",
  "right costophrenic angle",
  "right hemidiaphragm",
  "right clavicle",
  "right cardiac silhouette",
  "right cardiophrenic angle",
  "right upper abdomen"
 ],
 "T5": [
  "tubesandlines|chest tube",
  "tubesandlines|mediastinal drain",
  "tubesandlines|pigtail catheter",
  "tubesandlines|endotracheal tube",
  "tubesandlines|tracheostomy tube",
  "tubesandlines|picc",
  "tubesandlines|ij line",
  "tubesandlines|chest port",
  "tubesandlines|subclavian line",
  "tubesandlines|swan-ganz catheter",
  "tubesandlines|intra-aortic balloon pump",
  "tubesandlines|enteric tube",
  "device|sternotomy wires",
  "device|cabg grafts",
  "device|aortic graft/repair",
  "device|prosthetic valve",
  "device|cardiac pacer and wires"
 ],
 "T6": [
  "technicalassessment|low lung volumes",
  "technicalassessment|rotated",
  "technicalassessment|artifact",
  "technicalassessment|breast/nipple shadows",
  "technicalassessment|skin fold"
 ],
 "T7": [
  "texture|opacity",
  "texture|alveolar",
  "texture|interstitial",
  "texture|calcified",
  "texture|lucency"
 ],
 "nguyen_mau": {
  "k": 4,
  "toi_thieu": 20,
  "n_init": 10,
  "random_state": 0
 },
 "cau": {
  "min_tu": 2,
  "max_tok": 64,
  "toi_da_moi_study": 6,
  "trong_so": "1/log(1+tan_so)"
 },
 "vung_day": {
  "bat_thuong": [
   "nlp|abnormal",
   "nlp|normal"
  ],
  "ket_cau": [
   "texture|opacity",
   "texture|alveolar",
   "texture|interstitial",
   "texture|calcified",
   "texture|lucency"
  ],
  "r": {
   "thuoc_vung": 0.1,
   "bat_thuong": 0.3,
   "ket_cau": 0.3
  }
 }
}''')
MUC_TIEU = json.loads(r'''{
 "_ghi_chu": "Tuan quyet 05/10 (Event 2026-10-05): dich cua arm rep (78g) = PHUONG AN 3 - co / khong lay tu nhan CXR-LT (nhan dung de cham), vi tri lay tu ImaGenome CHI KHI hai parser dong y (nhan CXR-LT duong + ImaGenome 'yes' tren vung CO HOP THAT, khong qua vung thay the); thieu vi tri = CHUA BIET (trong so trung tinh), khong phai khong co. Phuong an 1 (vector report 78e, cap duong theo vector nhan) chay cung luot lam doi chung. Vung du phong 'hai phoi' cho loss vung khi ImaGenome thieu vi tri cua nhan nam trong phoi. Dung o 78q (du lieu) va khoi_loss.py (78g).",
 "phien_ban": "muc_tieu_v1",
 "rep3": {
  "dinh_dang_cum": "<nhan>: <vung>, <vung>. khi biet vi tri; <nhan>. khi chua biet. Vung theo thu tu 36 vung ImaGenome (semantics/objects_detectable_by_bbox_pipeline_v1.txt).",
  "vector_anh": "trung binh vector BiomedBERT cua cac cum (moi nhan duong mot cum); anh khong co nhan duong -> khong co muc tieu",
  "beta": 2.0,
  "max_tok_cum": 256,
  "_ghi_chu_max_tok": "cum dai nhat co the ~15 vung (78f C5: toi da 15 vung / (anh, nhan)) x ~5 token / ten vung -> ~90 token; 256 du rong. C3 DUNG neu cum nao vuot (khong cat)."
 },
 "vung_du_phong": {
  "nhan": ["Pneumonia", "Infiltration", "Consolidation", "Pneumothorax", "Tuberculosis"],
  "vung": ["right lung", "left lung"]
 },
 "kiem_vi_tri": {
  "nguong_ap_ben": 0.95,
  "_ghi_chu": "Dat truoc khi co ket qua: probe tuyen tinh tu vector cum -> co vung ben trai / ben phai; AP trung binh hai ben >= nguong thi vector BiomedBERT giu vi tri, dung duoc cho phuong an 3; duoi nguong -> 78g can them nhung vung hoc duoc (cho Tuan quyet)."
 }
}''')
print(f"cau hinh: {DICH_78S['phien_ban']} (cau / study / buoc {DICH_78S['cau']['toi_da_moi_study']}, vung_day r "
      f"{DICH_78S['vung_day']['r']}) | muc tieu 78q {MUC_TIEU['phien_ban']}")

# ---- khoi 79 (NGUYEN VAN experiments/79_phat_minh/khoi/khoi_luong_vung.py) ----
import torch.utils.checkpoint as _tuc_luong
from torchvision.ops import generalized_box_iou_loss as _giou_luong


def _out_luong(o):
    '''Dau ra mot block Dinov2: tuple / list (transformers cu) -> phan tu dau; tensor (transformers >= 5) -> giu nguyen.'''
    return o[0] if isinstance(o, (tuple, list)) else o


def _ma_hoa_hop(hop, n=32):
    '''Hop [..., 4] (cx, cy, w, h) chuan hoa -> sin / cos [..., 8n] (nhu DAB-DETR, nhiet do 10000). Khong tham so, khong ngau
       nhien; float32.'''
    t = 10000.0 ** (torch.arange(n, device=hop.device, dtype=torch.float32) / n)
    x = hop.float()[..., None] * (2.0 * math.pi) / t
    return torch.cat([x.sin(), x.cos()], -1).flatten(-2)


def mat_na_hop(hop, luoi=37, t=0.5):
    '''Hop (cx, cy, w, h) chuan hoa [..., R, 4] -> mat na mem [..., R, luoi * luoi] trong (0, 1):
       m = s((x - x1) / t) s((x2 - x) / t) s((y - y1) / t) s((y2 - y) / t), s = sigmoid, (x, y) = tam o luoi, moi do dai theo DON VI
       O LUOI (t = 0.5 o). Thu tu o theo HANG (chi so y * luoi + x), dung thu tu token patch cua Dinov2. float32.'''
    c = torch.arange(luoi, device=hop.device, dtype=torch.float32) + 0.5
    hp = hop.float() * luoi
    x1, x2 = hp[..., 0] - hp[..., 2] / 2, hp[..., 0] + hp[..., 2] / 2
    y1, y2 = hp[..., 1] - hp[..., 3] / 2, hp[..., 1] + hp[..., 3] / 2
    mx = torch.sigmoid((c - x1[..., None]) / t) * torch.sigmoid((x2[..., None] - c) / t)
    my = torch.sigmoid((c - y1[..., None]) / t) * torch.sigmoid((y2[..., None] - c) / t)
    return (my[..., :, None] * mx[..., None, :]).flatten(-2)


class CrossAttnBias(nn.Module):
    '''Cross-attention nhieu dau co thien lech cong: q [B, Lq, d], kv [B, Lk, d], bias [B, 1, Lq, Lk] | None -> [B, Lq, d].
       q / k / v / o = Linear(d, d); F.scaled_dot_product_attention; KHONG dropout. zero_out=True -> o.weight = o.bias = 0 (dau
       ra DUNG 0 luc khoi tao). trong_so(...) = attention TB cac dau [B, Lq, Lk] (chan doan / kiem).'''
    def __init__(s, d, n_head, zero_out=False):
        super().__init__()
        assert d % n_head == 0, (d, n_head)
        s.nh, s.dh = int(n_head), d // n_head
        s.q, s.k, s.v, s.o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        if zero_out:
            nn.init.zeros_(s.o.weight); nn.init.zeros_(s.o.bias)

    def _qkv(s, q, kv):
        B, Lq, d = q.shape; Lk = kv.shape[1]
        Q = s.q(q).view(B, Lq, s.nh, s.dh).transpose(1, 2)
        K = s.k(kv).view(B, Lk, s.nh, s.dh).transpose(1, 2)
        V = s.v(kv).view(B, Lk, s.nh, s.dh).transpose(1, 2)
        return Q, K, V

    def forward(s, q, kv, bias=None):
        Q, K, V = s._qkv(q, kv)
        a = F.scaled_dot_product_attention(Q, K, V, attn_mask=None if bias is None else bias.to(Q.dtype), dropout_p=0.0)
        return s.o(a.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1))

    @torch.no_grad()
    def trong_so(s, q, kv, bias=None):
        Q, K, _ = s._qkv(q, kv)
        S = (Q.float() @ K.float().transpose(-1, -2)) / math.sqrt(s.dh)
        if bias is not None: S = S + bias.float()
        return S.softmax(-1).mean(1)


class LopJCA(nn.Module):
    '''MOT lop JCA sau block l. forward(h [B, 1 + N, d], r [B, R, d], hop [B, R, 4], p [B, R, d] ma hoa vi tri cua hop, mo_ghi)
       -> (h', r', hop', ti): ti = ||dh|| / ||h|| (0-d, khong grad) khi mo_ghi, None khi khong.
         r1 = r + SA(LN(r + p));  r2 = r1 + CA_r(LN(r1 + p), LN(h[:, 1:]), bias_r);  r3 = r2 + FFN(LN(r2))
         mo_ghi: h' = h + CA_h(LN(h), LN(r + p), bias_h)   (r CU: cap nhat dong thoi; CA_h.o khoi tao 0)
         neo: hop' = sigmoid(logit(hop) + MLP_hop(r3))      (lop cuoi MLP_hop khoi tao 0)
       bias_r = -beta (1 - mat_na_hop(hop.detach())) [B, 1, R, N] (None neu khong neo); bias_h = bias_r chuyen vi, them hang 0 cho CLS.'''
    def __init__(s, d, n_head, ff, beta, t_mem, luoi, neo):
        super().__init__()
        s.beta, s.t_mem, s.luoi, s.neo = float(beta), float(t_mem), int(luoi), bool(neo)
        s.ln_sa, s.sa = nn.LayerNorm(d), CrossAttnBias(d, n_head)
        s.ln_r, s.ln_h, s.ca_r = nn.LayerNorm(d), nn.LayerNorm(d), CrossAttnBias(d, n_head)
        s.ln_ff, s.ff = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        s.ln_h2, s.ln_r2, s.ca_h = nn.LayerNorm(d), nn.LayerNorm(d), CrossAttnBias(d, n_head, zero_out=True)
        if s.neo:
            s.mlp_hop = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4))
            nn.init.zeros_(s.mlp_hop[-1].weight); nn.init.zeros_(s.mlp_hop[-1].bias)

    def bias_vung(s, hop):
        if not s.neo: return None
        return (-s.beta * (1.0 - mat_na_hop(hop.detach(), s.luoi, s.t_mem)))[:, None]

    def forward(s, h, r, hop, p, mo_ghi):
        b_r = s.bias_vung(hop)
        x = s.ln_sa(r + p)
        r1 = r + s.sa(x, x)
        r2 = r1 + s.ca_r(s.ln_r(r1 + p), s.ln_h(h[:, 1:]), b_r)
        r3 = r2 + s.ff(s.ln_ff(r2))
        ti = None
        if mo_ghi:
            bh = None if b_r is None else torch.cat(
                [b_r.new_zeros(b_r.shape[0], 1, 1, b_r.shape[2]), b_r.transpose(-1, -2)], 2)
            dh = s.ca_h(s.ln_h2(h), s.ln_r2(r + p), bh)
            with torch.no_grad():
                ti = dh.float().norm() / h.float().norm().clamp_min(1e-12)
            h = h + dh
        if s.neo:
            hop = torch.sigmoid(torch.logit(hop.float().clamp(1e-4, 1.0 - 1e-4)) + s.mlp_hop(r3).float())
        return h, r3, hop, ti


class LuongVung(nn.Module):
    '''Luong vung: R token + MOT LopJCA cho moi block trong `tang`. hop_tb [R, 4] (cx, cy, w, h) chuan hoa = buffer (luu trong
       state_dict). e [R, d] ~ N(0, 0.02); pos = MLP(_ma_hoa_hop) DUNG CHUNG moi lop. ckpt=True: tinh lai kich hoat cua tung lop
       (torch.utils.checkpoint, use_reentrant=False) khi train co grad — khong doi ket qua (khong dropout, khong ngau nhien).
       mo_ghi thuc = mo_ghi and hai_chieu.
         st = bat_dau(B); h = buoc(l, h, st, mo_ghi) sau moi block l (l ngoai tang -> tra h nguyen); dau_ra(st) ->
         dict(r [B, R, d], hop_cac_tang [L, B, R, 4] | None, ghi = list ti tung lop)'''
    def __init__(s, hop_tb, d=768, R=36, tang=(8, 9, 10, 11), n_head=12, ff=1536, beta=4.0, t_mem=0.5, hai_chieu=True,
                 neo=True, luoi=37, ckpt=True, n_pos=32):
        super().__init__()
        hop_tb = torch.as_tensor(np.asarray(hop_tb, dtype='float32'))
        assert tuple(hop_tb.shape) == (int(R), 4) and bool(((hop_tb > 0) & (hop_tb <= 1)).all()), \
            f'hop_tb phai [R, 4] trong (0, 1]: {tuple(hop_tb.shape)}'
        s.R, s.tang, s.hai_chieu, s.ckpt, s.n_pos = int(R), tuple(int(t) for t in tang), bool(hai_chieu), bool(ckpt), int(n_pos)
        assert len(s.tang) >= 1 and list(s.tang) == sorted(set(s.tang)), f'tang phai tang ngat: {s.tang}'
        s._vt = {t: i for i, t in enumerate(s.tang)}
        s.register_buffer('hop_tb', hop_tb.clone())
        s.e = nn.Parameter(torch.randn(s.R, d) * 0.02)
        s.pos = nn.Sequential(nn.Linear(8 * s.n_pos, d), nn.GELU(), nn.Linear(d, d))
        s.lop = nn.ModuleList([LopJCA(d, n_head, ff, beta, t_mem, luoi, neo) for _ in s.tang])

    def bat_dau(s, B):
        return dict(r=s.e[None].expand(int(B), -1, -1), hop=s.hop_tb[None].expand(int(B), -1, -1),
                    hop_cac_tang=[], ghi=[])

    def buoc(s, l, h, st, mo_ghi):
        i = s._vt.get(int(l))
        if i is None: return h
        lop, mg = s.lop[i], bool(mo_ghi and s.hai_chieu)
        p = s.pos(_ma_hoa_hop(st['hop'].detach(), s.n_pos))
        if s.ckpt and s.training and torch.is_grad_enabled():
            h, r, hop, ti = _tuc_luong.checkpoint(lop, h, st['r'], st['hop'], p, mg, use_reentrant=False)
        else:
            h, r, hop, ti = lop(h, st['r'], st['hop'], p, mg)
        st['r'], st['hop'] = r, hop
        st['hop_cac_tang'].append(hop)
        if ti is not None: st['ghi'].append(ti)
        return h

    def dau_ra(s, st):
        return dict(r=st['r'], hop_cac_tang=(torch.stack(st['hop_cac_tang']) if st['hop_cac_tang'] else None), ghi=st['ghi'])


class PhuVung(nn.Module):
    '''Phia ANH cua dich vung (KHONG chua thap van ban — thap la PHU.text cua B1 / B2, dung chung, tranh tham so trung nhom
       optimizer): anh = ChieuAnh(d, d_an, d_chieu) (nhu PhuCau.anh), nhiet = NhietDo(tau0) (RIENG voi cau study), bt = Linear(d, 1).
       z_anh(r) [..., d_chieu]; tau(); bt_logit(r) [...]. Khoi tao TIEU dong chung -> tao trong fork_rng voi hat rieng.'''
    def __init__(s, d=768, d_an=512, d_chieu=256, tau0=0.07):
        super().__init__()
        s.anh = ChieuAnh(d, d_an=d_an, d_chieu=d_chieu); s.nhiet = NhietDo(tau0); s.bt = nn.Linear(d, 1)

    def z_anh(s, r):
        return s.anh(r.float())[0]

    def tau(s):
        return s.nhiet.tau()

    def bt_logit(s, r):
        return s.bt(r.float())[..., 0]


def chay_bb_luong(bb, luong, img, mo_ghi=True, grad_bb=True, ckpt_bb=()):
    '''Forward Dinov2 `bb` (embeddings -> encoder.layer 0..L-1 -> layernorm, = bb(pixel_values).last_hidden_state) voi luong chen
       sau cac block trong luong.tang -> (patch [B, N, d] float32, cls [B, d] float32, luong.dau_ra). Autocast / grad do noi goi
       dat. grad_bb=False: block chay no_grad (luong van hoc) — CHI cho phep khi duong ghi nguoc dong (neu khong, grad cua luong
       qua cac block sau bi cat am tham). Chi goi layer(h): khong phu thuoc ten module con (transformers >= 5.18 doi ten).
       ckpt_bb: chi so block backbone TINH LAI kich hoat luc backward (torch.utils.checkpoint, use_reentrant=False) khi co grad —
       giam VRAM, KHONG doi ket qua (block Dinov2 cua RAD-DINO khong dropout; checkpoint giu nguyen trang thai RNG + autocast).'''
    g = torch.is_grad_enabled(); gb = g and bool(grad_bb); ck = {int(i) for i in ckpt_bb}
    assert grad_bb or not g or not (mo_ghi and luong.hai_chieu), \
        'chay_bb_luong: backbone dong (grad_bb=False) ma duong ghi nguoc mo -> grad cua luong bi cat, DUNG'
    with torch.set_grad_enabled(gb):
        h = bb.embeddings(img)
    st = luong.bat_dau(img.shape[0])
    for l, lop in enumerate(bb.encoder.layer):
        with torch.set_grad_enabled(gb):
            h = _out_luong(_tuc_luong.checkpoint(lop, h, use_reentrant=False) if (gb and l in ck) else lop(h))
        h = luong.buoc(l, h, st, mo_ghi)
    with torch.set_grad_enabled(gb):
        h = bb.layernorm(h)
    return h[:, 1:].float(), h[:, 0].float(), luong.dau_ra(st)


@torch.no_grad()
def upper_luong(bb, luong, h8, mo_ghi=True):
    '''Doc dong cung (79b): h8 = dau ra block luong.tang[0] - 1 (shared_lo cua 78v) -> block tang[0]..L-1 + luong + layernorm
       -> patch [B, N, d] float32. mo_ghi=False -> TRUNG upper_frozen goc cua 78v (cung block, cung layernorm).'''
    st = luong.bat_dau(h8.shape[0]); h = h8
    L = bb.encoder.layer
    for l in range(luong.tang[0], len(L)):
        h = _out_luong(L[l](h))
        h = luong.buoc(l, h, st, mo_ghi)
    return bb.layernorm(h)[:, 1:].float()


def cxcywh_sang_xyxy(b):
    c = b.float()
    return torch.stack([c[..., 0] - c[..., 2] / 2, c[..., 1] - c[..., 3] / 2, c[..., 0] + c[..., 2] / 2, c[..., 1] + c[..., 3] / 2], -1)


def xyxy_sang_cxcywh(b):
    c = b.float()
    return torch.stack([(c[..., 0] + c[..., 2]) / 2, (c[..., 1] + c[..., 3]) / 2, c[..., 2] - c[..., 0], c[..., 3] - c[..., 1]], -1)


def hop_tb(HOP518, SG_HANG, POOL, R, SIZE):
    '''Hop khoi dau cua luong: TB hop HOP518 [n_anh_78f, R, 4] (x1, y1, x2, y2 toa do anh SIZE; NaN = khong co) tren anh POOL
       (CHI train, khong giu rieng) co scene graph (SG_HANG >= 0) -> [R, 4] float32 (cx, cy, w, h) chuan hoa, kep [1e-3, 1].
       Vung khong co mau nao -> hop ca anh (0.5, 0.5, 1, 1). numpy, CPU.'''
    g = np.asarray(SG_HANG)[np.asarray(POOL)]; g = g[g >= 0]
    H = np.asarray(HOP518); out = np.tile(np.array([0.5, 0.5, 1.0, 1.0], 'float64'), (int(R), 1))
    for k in range(int(R)):
        v = H[g, k]; v = v[np.isfinite(v).all(1)]
        if len(v):
            x1, y1, x2, y2 = v.mean(0) / float(SIZE)
            out[k] = ((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1e-3), max(y2 - y1, 1e-3))
    return np.clip(out, 1e-3, 1.0).astype('float32')


def loss_hop(hop_cac_tang, gt_xyxy, co, SIZE):
    '''Hop moi tang: L1 (cx, cy, w, h chuan hoa) + (1 - GIoU), TB tren tang x cap hop le. hop_cac_tang [L, B, R, 4]; gt_xyxy
       [B, R, 4] toa do anh SIZE (DA bien doi cung anh, TF_HOP); co [B, R] bool. Khong cap hop le -> None.'''
    co = co.bool()
    if hop_cac_tang is None or not bool(co.any()): return None
    g = gt_xyxy.float() / float(SIZE)
    p = hop_cac_tang.float()[:, co]
    gx = g[co]
    l1 = (p - xyxy_sang_cxcywh(gx)[None]).abs().sum(-1).mean()
    gi = _giou_luong(cxcywh_sang_xyxy(p).reshape(-1, 4), gx[None].expand(p.shape[0], -1, -1).reshape(-1, 4), reduction='mean')
    return l1 + gi


def loss_bt(logit, y):
    '''Co bat thuong tung vung: BCE tren y in {0, 1}; y = -1 (khong biet) bi bo. Khong phan tu -> None.'''
    m = y >= 0
    if not bool(m.any()): return None
    return F.binary_cross_entropy_with_logits(logit.float()[m], y[m].float())


def cap_vung_lo(rows, SG_HANG_T, POOL_T, PTR_V, R):
    '''Cap (anh trong lo, vung) vao loss cau vung: anh POOL co scene graph, vung co >= 1 cau (CSR PTR_V theo cap = g * R + vung).
       -> (b_idx, vung, cap) long [Np]; khong cap -> tensor rong.'''
    R = int(R)
    g = SG_HANG_T[rows]; hl = (g >= 0) & POOL_T[rows]
    b = torch.nonzero(hl, as_tuple=True)[0]
    vr = torch.arange(R, device=rows.device)
    cap = g[b][:, None] * R + vr[None, :]
    co = PTR_V[cap + 1] > PTR_V[cap]
    return b[:, None].expand(-1, R)[co], vr[None, :].expand(b.numel(), -1)[co], cap[co]


def mat_na_loai_vung(P, sub, anh):
    '''loai bool [Np, U] = ~P VA co cap b' CUNG benh nhan, KHAC anh, P[b', u]: cau vung cua anh khac cung benh nhan (benh man
       tinh cung vung) KHONG lam am. Cau cua VUNG KHAC trong CUNG anh van la am (day token vung phan biet vung).'''
    P = P.bool()
    A = (sub[:, None] == sub[None, :]) & (anh[:, None] != anh[None, :])
    with torch.autocast(device_type=P.device.type, enabled=False):
        E = (A.float() @ P.float()) > 0
    return E & ~P


def dich_vung_cau(cap, sub, anh, PTR_V, IDX_V, toi_da, gen):
    '''Dich InfoNCE cau vung cho cac cap trong lo: ung vien = lay_cau_lo (khoi_cau; "study" = cap, ngau nhien CHI tu `gen`),
       duong = ma_tran_duong (cau trung chu = MOT ung vien, duong voi moi cap co no), loai = mat_na_loai_vung.
       -> (ma_u [U], P [Np, U], loai [Np, U]).'''
    ma_u = lay_cau_lo(cap, PTR_V, IDX_V, toi_da, gen)
    P = ma_tran_duong(cap, PTR_V, IDX_V, ma_u)
    return ma_u, P, mat_na_loai_vung(P, sub, anh)


def token_hoa_cau(TOK, texts, max_tok=64, lo=4096):
    '''DUNG cach 78s lam cau_token: TOK(lo, truncation=True, max_length=max_tok)['input_ids'] -> int32 [n, max_tok], [CLS] ... [SEP],
       dem 0 ben PHAI. Tra token (khong giu van ban) + so cau bi cat.'''
    assert TOK.pad_token_id == 0, f'pad id cua tokenizer = {TOK.pad_token_id} != 0 (ThapBert dem 0 ben phai) -> DUNG'
    out = np.zeros((len(texts), int(max_tok)), 'int32'); n_cat = 0
    for i in range(0, len(texts), int(lo)):
        for r, x in enumerate(TOK(list(texts[i:i + int(lo)]), truncation=True, max_length=int(max_tok))['input_ids']):
            out[i + r, :len(x)] = x; n_cat += int(len(x) >= int(max_tok))
    return out, n_cat


@torch.no_grad()
def iou_cap(a_xyxy, b_xyxy):
    '''IoU tung cap hop [..., 4] (x1, y1, x2, y2) — chan doan.'''
    a, b = a_xyxy.float(), b_xyxy.float()
    iw = (torch.minimum(a[..., 2], b[..., 2]) - torch.maximum(a[..., 0], b[..., 0])).clamp_min(0)
    ih = (torch.minimum(a[..., 3], b[..., 3]) - torch.maximum(a[..., 1], b[..., 1])).clamp_min(0)
    gi = iw * ih
    dt = lambda x: (x[..., 2] - x[..., 0]).clamp_min(0) * (x[..., 3] - x[..., 1]).clamp_min(0)
    return gi / (dt(a) + dt(b) - gi).clamp_min(1e-9)
print(f"79a: khoi luong vung | luot {LUOT_79A} (nen 78u {LUOT_78U}, thap {'HOC' if THAP_HOC else 'DONG'}, doi chung ghep cap "
      f"{LUONG_CFG['doi_chung']}) | cau hinh {CFG_79['phien_ban']} | phan {LUONG_CFG['phan']} | tinh lai block backbone {CKPT_BB} | "
      f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

# ---- 78x: cau hinh J v2 (NGUYEN VAN experiments/78_chi_anh/cau_hinh/jca_v2.json) + anh xa ImaGenome (NGUYEN VAN cau_hinh/anh_xa_imagenome.json) ----
CFG_JV2 = json.loads(r'''{
 "phien_ban": "jca_v2_v1",
 "_ghi_chu": "JCA v2 (docs/PLAN_JCA_V2.md §4 + §4.1; Tuan duyet 12/10). Nen 79a luot J_bert_dong (= 78u B1 + luong vung; khoi / bo dung 79 CHI DOC). r cua phan moi: gia dinh, KHONG chinh o vong sang loc. hat cong vao SEED. mil: 31 nhan do phu >= 20% (kiem_do_phu_vung.py 12/10; 3 nhan thay_the_nhan tinh ca vung thay the). cay: zone con -> phoi (ImaGenome lan con -> phoi 98%); mediastinum KHONG vao cay (43%).",
 "luot_79": "J_bert_dong",
 "epochs": 6,
 "phan_them": {"nhan_vung": 0.1, "nhat_quan": 0.1},
 "view_neo": "frontal",
 "thay_the_nhan": ["Fracture", "Rib Fracture", "Subcutaneous Emphysema"],
 "mil": ["Adenopathy", "Atelectasis", "Calcification of the Aorta", "Cardiomegaly", "Clavicle Fracture", "Consolidation", "Edema",
         "Emphysema", "Enlarged Cardiomediastinum", "Fibrosis", "Fracture", "Granuloma", "Hernia", "Hydropneumothorax",
         "Lobar Atelectasis", "Lung Lesion", "Lung Opacity", "Mass", "Nodule", "Pleural Effusion", "Pleural Thickening",
         "Pneumomediastinum", "Pneumonia", "Pneumoperitoneum", "Pneumothorax", "Pulmonary Hypertension", "Rib Fracture",
         "Round(ed) Atelectasis", "Subcutaneous Emphysema", "Support Devices", "Tortuous Aorta"],
 "cay": {"right lung": ["right apical zone", "right upper lung zone", "right mid lung zone", "right lower lung zone",
                        "right hilar structures", "right costophrenic angle"],
         "left lung": ["left apical zone", "left upper lung zone", "left mid lung zone", "left lower lung zone",
                       "left hilar structures", "left costophrenic angle"]},
 "thu_nhanh_study": 1200,
 "hat": {"doc_vung": 11, "dau_vung": 12}
}''')
ANH_XA_78X = json.loads(r'''{
 "_ghi_chu": "docs/PLAN_CHI_ANH.md Buoc 1c: anh xa thuoc tinh Chest ImaGenome v1.0.0 (semantics/attribute_relations_v1.txt, khoa 'loai|gia tri') -> 40 nhan CXR-LT 2024, va vung giai phau CO DINH cho nhan co vi tri co dinh. Claude viet 04/10 — GIA DINH NGHIEN CUU, Tuan duyet / sua; chi dung cho loss vung LUC TRAIN (78g). Vung goi y cua mot nhan DUONG o mot anh = HOP cac vung co thuoc tinh 'yes' anh xa vao nhan do; khong co thi dung vung co dinh (neu co); khong co ca hai thi KHONG co loss vung cho cap (anh, nhan) do. Ten vung = semantics/objects_detectable_by_bbox_pipeline_v1.txt. 78f kiem: moi khoa co trong tu vung, moi vung co trong 36 vung, moi nhan HOAC co nguon vung (thuoc tinh va / hoac co dinh) HOAC nam trong khong_vung, khong ca hai, du 40 nhan. Khoa tu vung khong anh xa (bronchiectasis, scoliosis, goiter...) khong thuoc 40 nhan.",
 "phien_ban": "anh_xa_v1",
 "thuoc_tinh": {
  "anatomicalfinding|lung opacity": ["Lung Opacity"],
  "anatomicalfinding|airspace opacity": ["Lung Opacity"],
  "anatomicalfinding|consolidation": ["Consolidation", "Lung Opacity"],
  "anatomicalfinding|infiltration": ["Infiltration", "Lung Opacity"],
  "anatomicalfinding|atelectasis": ["Atelectasis", "Round(ed) Atelectasis", "Lung Opacity"],
  "anatomicalfinding|linear/patchy atelectasis": ["Atelectasis", "Lung Opacity"],
  "anatomicalfinding|lobar/segmental collapse": ["Lobar Atelectasis", "Atelectasis", "Lung Opacity"],
  "anatomicalfinding|pulmonary edema/hazy opacity": ["Edema", "Lung Opacity"],
  "anatomicalfinding|vascular congestion": ["Edema"],
  "anatomicalfinding|vascular redistribution": ["Edema"],
  "anatomicalfinding|increased reticular markings/ild pattern": ["Fibrosis"],
  "anatomicalfinding|pleural effusion": ["Pleural Effusion"],
  "anatomicalfinding|costophrenic angle blunting": ["Pleural Effusion"],
  "anatomicalfinding|pleural/parenchymal scarring": ["Pleural Thickening", "Fibrosis"],
  "anatomicalfinding|enlarged cardiac silhouette": ["Cardiomegaly", "Enlarged Cardiomediastinum"],
  "anatomicalfinding|mediastinal widening": ["Enlarged Cardiomediastinum"],
  "anatomicalfinding|enlarged hilum": ["Adenopathy", "Pulmonary Hypertension"],
  "anatomicalfinding|tortuous aorta": ["Tortuous Aorta"],
  "anatomicalfinding|vascular calcification": ["Calcification of the Aorta"],
  "anatomicalfinding|pneumomediastinum": ["Pneumomediastinum"],
  "anatomicalfinding|pneumothorax": ["Pneumothorax"],
  "anatomicalfinding|hydropneumothorax": ["Hydropneumothorax", "Pneumothorax"],
  "anatomicalfinding|lung lesion": ["Lung Lesion"],
  "anatomicalfinding|mass/nodule (not otherwise specified)": ["Mass", "Nodule", "Lung Lesion"],
  "anatomicalfinding|multiple masses/nodules": ["Mass", "Nodule", "Lung Lesion"],
  "anatomicalfinding|calcified nodule": ["Granuloma", "Nodule"],
  "anatomicalfinding|superior mediastinal mass/enlargement": ["Mass", "Enlarged Cardiomediastinum"],
  "anatomicalfinding|rib fracture": ["Rib Fracture", "Fracture"],
  "anatomicalfinding|clavicle fracture": ["Clavicle Fracture", "Fracture"],
  "anatomicalfinding|spinal fracture": ["Fracture"],
  "anatomicalfinding|hyperaeration": ["Emphysema"],
  "anatomicalfinding|cyst/bullae": ["Emphysema"],
  "anatomicalfinding|sub-diaphragmatic air": ["Pneumoperitoneum"],
  "anatomicalfinding|subcutaneous air": ["Subcutaneous Emphysema"],
  "anatomicalfinding|hernia": ["Hernia"],
  "disease|pneumonia": ["Pneumonia"],
  "disease|aspiration": ["Pneumonia"],
  "disease|fluid overload/heart failure": ["Edema"],
  "disease|copd/emphysema": ["Emphysema"],
  "disease|granulomatous disease": ["Granuloma", "Tuberculosis"],
  "disease|interstitial lung disease": ["Fibrosis"],
  "disease|lung cancer": ["Mass", "Lung Lesion"],
  "disease|alveolar hemorrhage": ["Lung Opacity"],
  "tubesandlines|chest tube": ["Support Devices"],
  "tubesandlines|mediastinal drain": ["Support Devices"],
  "tubesandlines|pigtail catheter": ["Support Devices"],
  "tubesandlines|endotracheal tube": ["Support Devices"],
  "tubesandlines|tracheostomy tube": ["Support Devices"],
  "tubesandlines|picc": ["Support Devices"],
  "tubesandlines|ij line": ["Support Devices"],
  "tubesandlines|chest port": ["Support Devices"],
  "tubesandlines|subclavian line": ["Support Devices"],
  "tubesandlines|swan-ganz catheter": ["Support Devices"],
  "tubesandlines|intra-aortic balloon pump": ["Support Devices"],
  "tubesandlines|enteric tube": ["Support Devices"],
  "device|sternotomy wires": ["Support Devices"],
  "device|cabg grafts": ["Support Devices"],
  "device|aortic graft/repair": ["Support Devices"],
  "device|prosthetic valve": ["Support Devices"],
  "device|cardiac pacer and wires": ["Support Devices"]
 },
 "co_dinh": {
  "Azygos Lobe": ["right apical zone", "right upper lung zone"],
  "Calcification of the Aorta": ["aortic arch", "descending aorta"],
  "Cardiomegaly": ["cardiac silhouette"],
  "Clavicle Fracture": ["right clavicle", "left clavicle"],
  "Enlarged Cardiomediastinum": ["mediastinum", "cardiac silhouette"],
  "Hernia": ["cardiac silhouette", "mediastinum"],
  "Kyphosis": ["spine"],
  "Pneumoperitoneum": ["right upper abdomen", "left upper abdomen", "right hemidiaphragm", "left hemidiaphragm"],
  "Tortuous Aorta": ["aortic arch", "descending aorta"],
  "Pneumomediastinum": ["mediastinum", "upper mediastinum"],
  "Adenopathy": ["right hilar structures", "left hilar structures"],
  "Pulmonary Hypertension": ["right hilar structures", "left hilar structures"],
  "Emphysema": ["right lung", "left lung"],
  "Edema": ["right lung", "left lung"]
 },
 "khong_vung": ["Normal", "Fissure", "Infarction", "Pulmonary Embolism", "Pleural Other"],
 "_ghi_chu_vung_thay_the": "ImaGenome gan mot so thuoc tinh vao vung CHI co trong report, KHONG co hop (semantics/objects_extracted_from_reports_v1.txt): xem truoc 04/10 tren 166.475 anh train — 'rib fracture' 100% o right / left chest wall, 'subcutaneous air' o chest wall + neck -> Rib Fracture, Subcutaneous Emphysema 0% co vung. Vung thay the = vung CO hop gan nhat ve giai phau, GIU ben trai / phai. Gia dinh nghien cuu (Tuan duyet); 78f danh dau cot `thay` cho thuoc tinh di qua vung thay the va `chi_thay_the` cho (anh, nhan) chi co vung tu thay the -> 78g co the bo. Cau report cua vung thay the KHONG gan vao vung co hop.",
 "vung_thay_the": {
  "right chest wall": ["right lung"],
  "left chest wall": ["left lung"],
  "neck": ["trachea", "upper mediastinum"],
  "right shoulder": ["right clavicle"],
  "left shoulder": ["left clavicle"]
 }
}''')
assert CFG_JV2['luot_79'] == LUOT_79A and int(CFG_JV2['epochs']) == EPOCHS, 'cau hinh jca_v2 KHAC C0 -> DUNG'

# ---- khoi 78x (NGUYEN VAN experiments/78_chi_anh/khoi/khoi_lo_study.py) — CHUNG J v2 / doi chung ----


class LoTheoStudy(torch.utils.data.Sampler):
    '''Hoan vi theo STUDY: study xao ngau nhien, anh CUNG study LIEN NHAU (thu tu trong study = thu tu trong `st`) -> DataLoader cat lo
       (drop_last) => moi study nam tron trong MOT lo, tru study vat qua bien lo. Tra VI TRI 0..n-1 trong `st` (= vi tri trong idx cua
       DS). Hat: MOT so int64 rut tu dong ngau nhien CHUNG (CPU) moi lan lap, nhu RandomSampler khong generator -> luu / nap rng_cpu
       (resume) va ghep cap hai luot giu nguyen. st: study_id [n] theo vi tri.'''
    def __init__(s, st):
        st = np.asarray(st)
        assert st.ndim == 1 and len(st), 'LoTheoStudy: st phai 1 chieu, khong rong'
        _, inv = np.unique(st, return_inverse=True)
        s.n = int(len(st))
        s.thu_tu = np.argsort(inv, kind='stable')                          # vi tri xep theo study, giu thu tu trong study
        s.ptr = np.concatenate([[0], np.cumsum(np.bincount(inv))]).astype('int64')

    def __len__(s):
        return s.n

    def hoan_vi(s, hat):
        '''Hoan vi vi tri voi hat cho truoc (kiem / chan doan). randperm tren CPU, generator rieng.'''
        g = torch.Generator(); g.manual_seed(int(hat))
        p = torch.randperm(len(s.ptr) - 1, generator=g).numpy()
        return np.concatenate([s.thu_tu[s.ptr[i]:s.ptr[i + 1]] for i in p])

    def __iter__(s):
        return iter(s.hoan_vi(int(torch.empty((), dtype=torch.int64).random_().item())).tolist())


def dau_van_lo(rows, so_lo=3):
    '''Dau van thu tu lo (ghep cap hai luot): sha1 cua hang ORDER `so_lo` lo dau moi epoch -> 12 ky tu hex. rows: list mang / tensor.'''
    import hashlib
    h = hashlib.sha1()
    for r in rows[:so_lo]: h.update(np.asarray(r.cpu() if hasattr(r, 'cpu') else r, 'int64').tobytes())
    return h.hexdigest()[:12]

# ---- khoi 78x (NGUYEN VAN experiments/78_chi_anh/khoi/khoi_jca_v2.py) ----


class LopJCANeo(LopJCA):
    '''LopJCA (79, CHI DOC) + neo THEO MAU. forward(h, r, hop, p, mo_ghi, neo [B] float 0 / 1 | None): neo = 1 -> Y HET LopJCA;
       neo = 0 -> thien lech vung (ca bias_r lan bias_h) = 0 va hop GIU nguyen. neo la DOI SO -> torch.utils.checkpoint tinh lai dung
       mau. KHONG them tham so (LuongVungNeo doi lop cac LopJCA da tao).'''
    _neo = None

    def bias_vung(s, hop):
        b = super().bias_vung(hop)
        if b is None or s._neo is None: return b
        return b * s._neo.to(b.dtype)[:, None, None, None]

    def forward(s, h, r, hop, p, mo_ghi, neo=None):
        s._neo = None if neo is None else neo.float()
        try:
            h2, r2, hop2, ti = super().forward(h, r, hop, p, mo_ghi)
        finally:
            s._neo = None
        if neo is not None and s.neo:
            hop2 = torch.where(neo.bool()[:, None, None], hop2, hop.to(hop2.dtype))
        return h2, r2, hop2, ti


class LuongVungNeo(LuongVung):
    '''LuongVung (79, CHI DOC) voi lop LopJCANeo — DOI LOP cac LopJCA vua tao (khoi tao TRUNG LuongVung cung hat, khong rut them so
       ngau nhien, state_dict cung khoa). dat_neo(neo [B]) NGAY TRUOC moi forward (chay_bb_luong / upper_luong goi bat_dau): bat_dau LAY
       va XOA -> quen dat thi DUNG, khong bao gio dung neo cua lo truoc. Moi mau neo = 1 -> TRUNG LuongVung tung bit.'''
    def __init__(s, hop_tb, **kw):
        super().__init__(hop_tb, **kw)
        for l in s.lop: l.__class__ = LopJCANeo
        s._neo_lo = None

    def dat_neo(s, neo):
        s._neo_lo = neo

    def bat_dau(s, B):
        neo, s._neo_lo = s._neo_lo, None
        assert neo is not None, 'LuongVungNeo: chua dat_neo(neo) truoc forward -> DUNG (khong dung neo cua lo truoc)'
        assert tuple(neo.shape) == (int(B),), f'neo {tuple(neo.shape)} != ({int(B)},)'
        st = super().bat_dau(B)
        st['neo'] = neo.to(st['hop'].device).float()
        return st

    def buoc(s, l, h, st, mo_ghi):
        '''= LuongVung.buoc + truyen st['neo'] vao lop (ca nhanh checkpoint).'''
        i = s._vt.get(int(l))
        if i is None: return h
        lop, mg = s.lop[i], bool(mo_ghi and s.hai_chieu)
        p = s.pos(_ma_hoa_hop(st['hop'].detach(), s.n_pos))
        if s.ckpt and s.training and torch.is_grad_enabled():
            h, r, hop, ti = _tuc_luong.checkpoint(lop, h, st['r'], st['hop'], p, mg, st['neo'], use_reentrant=False)
        else:
            h, r, hop, ti = lop(h, st['r'], st['hop'], p, mg, st['neo'])
        st['r'], st['hop'] = r, hop
        st['hop_cac_tang'].append(hop)
        if ti is not None: st['ghi'].append(ti)
        return h


class DocVung(nn.Module):
    '''Y 1: token vung -> khoa / gia tri THEM cho ML-Decoder: pin(LN(r)) + e_loai + e_vung[k] (pin = phep chieu patch CUA dau doc;
       e_* khoi tao 0). forward(pin, r [B, R, d_in]) -> [B, R, d]. Khong ngau nhien.'''
    def __init__(s, R, d_in, d):
        super().__init__()
        s.ln = nn.LayerNorm(d_in)
        s.e_loai = nn.Parameter(torch.zeros(d))
        s.e_vung = nn.Parameter(torch.zeros(int(R), d))

    def forward(s, pin, r):
        return pin(s.ln(r.float())) + s.e_loai + s.e_vung


def dau_h_v2(m, patch, kv_vung=None):
    '''dau_h cua 78u (MLDecoder 38a) + y 1: K/V = [pin(patch) ; kv_vung] -> (logit, h = 40 dau ra query TRUOC GroupFC, kv = pin(patch)
       CHI phan patch). kv_vung None -> TRUNG dau_h tung bit.'''
    q = m.q.weight.unsqueeze(0).expand(patch.shape[0], -1, -1)
    kv = m.pin(patch)
    h, _ = m.img(q, kv if kv_vung is None else torch.cat([kv, kv_vung.to(kv.dtype)], 1), False)
    return m.out(h), h, kv


class DauVung(nn.Module):
    '''Y 2 + kenh MIL. z(r) [B, R, C] = Linear(LN(r)) = logit "nhan j o vung k". mil(logit, z): logit_j + w_j * max_k z[:, k, j] CHI
       voi j trong mil_idx (w khoi tao 0 -> luc dau TRUNG logit classifier); nhan ngoai mil_idx KHONG doi.'''
    def __init__(s, d_in, C, mil_idx):
        super().__init__()
        s.ln, s.lin = nn.LayerNorm(d_in), nn.Linear(d_in, int(C))
        mi = sorted({int(i) for i in mil_idx})
        assert all(0 <= i < int(C) for i in mi), mi
        s.register_buffer('mil_idx', torch.as_tensor(mi, dtype=torch.long))
        s.w = nn.Parameter(torch.zeros(len(mi)))

    def z(s, r):
        return s.lin(s.ln(r.float()))

    def mil(s, logit, z):
        if not s.mil_idx.numel(): return logit
        return logit.index_add(1, s.mil_idx, (s.w * z[:, :, s.mil_idx].amax(1)).to(logit.dtype))


def mat_na_s(anh, vung, khoa, thay, khoa_nhan, nhan_thay, n_anh, C):
    '''Thuoc tinh 'yes' cua 78f (thuoc_tinh_train.parquet: anh, vung, khoa "loai|gia tri", thay) -> M [n_anh, C] int64 mat na bit tap
       vung S (§4.1): vung CO HOP THAT (thay == 0) cho MOI nhan; them vung THAY THE (thay == 1: chest wall -> phoi cung ben, neck ->
       trachea + upper mediastinum, shoulder -> clavicle) CHI cho nhan trong `nhan_thay` (chi so; Tuan 12/10). khoa_nhan: khoa ->
       [chi so nhan]; khoa khong anh xa bi bo. nhan_thay rong -> = mat_na_vi_tri_that cua khoi_muc_tieu (78q).'''
    anh = np.asarray(anh, 'int64'); vung = np.asarray(vung, 'int64'); khoa = np.asarray(khoa, dtype=object); thay = np.asarray(thay)
    M = np.zeros((int(n_anh), int(C)), 'int64'); nt = {int(j) for j in nhan_thay}
    for k, js in khoa_nhan.items():
        mk = khoa == k
        if not mk.any(): continue
        for j in js:
            g = mk & ((thay == 0) | (int(j) in nt))
            np.bitwise_or.at(M, (anh[g], np.full(int(g.sum()), int(j), 'int64')), np.left_shift(np.int64(1), vung[g]))
    return M


def cay_vung(VUNG, cay):
    '''cay {cha: [con, ...]} (ten vung) -> (CHA [R] int64: chi so cha hoac -1, CON [R] int64: mat na bit cac con). Moi con MOT cha;
       ten ngoai VUNG -> loi.'''
    idx = {v: i for i, v in enumerate(VUNG)}
    R = len(VUNG); CHA = np.full(R, -1, 'int64'); CON = np.zeros(R, 'int64')
    for c, ds in cay.items():
        for x in ds:
            assert CHA[idx[x]] == -1, f'vung {x} co hai cha'
            CHA[idx[x]] = idx[c]; CON[idx[c]] |= np.int64(1) << np.int64(idx[x])
    return CHA, CON


def mo_rong_cha_con(M, CHA, CON):
    '''M [...] int64 mat na bit tap vung S cua (anh, nhan) -> (M1, MC):
       M1 = S | cha(S) (vung CHUA ton thuong -> 1; ImaGenome tu lan con -> phoi ~98%, day la phan con lai);
       MC = cac con cua vung c thuoc M1 KHI S KHONG co con nao cua c (report CHI nhac muc phoi -> chua biet zone nao -> mask).
       S co >= 1 con cua c -> cac con khac cua c giu 0 (report da khu tru).'''
    M = np.asarray(M, 'int64'); M1 = M.copy(); MC = np.zeros_like(M)
    for k in range(len(CHA)):
        if CHA[k] >= 0:
            M1 = np.where(((M >> np.int64(k)) & 1) == 1, M1 | (np.int64(1) << np.int64(CHA[k])), M1)
    for c in range(len(CON)):
        if CON[c]:
            MC = np.where((((M1 >> np.int64(c)) & 1) == 1) & ((M & CON[c]) == 0), MC | CON[c], MC)
    return M1, MC & ~M1


def dich_nhan_vung(y, g, M1_T, MC_T, mil_T, R):
    '''Target y 2 cho MOT lo. y [B, C] {0, 1}; g [B] long: anh 78f mang ngu nghia vung cua STUDY (y A) hoac -1; M1_T / MC_T [n_anh, C]
       int64; mil_T [C] bool -> (T [B, R, C] float, V [B, R, C] bool):
       nhan am -> T 0, V moi vung; duong + M1 = 0 (chua biet vi tri) -> V False; duong + M1 != 0 -> T = 1 tren M1, V False tren MC;
       g < 0 hoac nhan ngoai MIL -> V False.'''
    co = g >= 0
    gg = g.clamp_min(0)
    m1, mc = M1_T[gg], MC_T[gg]
    bit = torch.bitwise_left_shift(torch.ones(int(R), dtype=torch.long, device=y.device), torch.arange(int(R), device=y.device))
    in1 = (m1[:, None, :] & bit[None, :, None]) != 0
    inc = (mc[:, None, :] & bit[None, :, None]) != 0
    duong = (y > 0.5)[:, None, :]
    T = (duong & in1).float()
    Vm = (~duong | ((m1 != 0)[:, None, :] & ~inc)) & co[:, None, None] & mil_T.bool()[None, None, :]
    return T, Vm


def asl_mat_na(x, y, v, gn=4.0, gp=0.0, clip=0.05, eps=1e-8):
    '''ASL (cong thuc NGUYEN VAN lop ASL cua 38a) tung phan tu, TB tren phan tu v = True. Khong phan tu nao -> None.'''
    if not bool(v.any()): return None
    x, y = x.float()[v], y.float()[v]
    xp = torch.sigmoid(x); xn = 1 - xp
    if clip: xn = (xn + clip).clamp(max=1)
    lo = y * torch.log(xp.clamp(min=eps)) + (1 - y) * torch.log(xn.clamp(min=eps))
    pt = xp * y + xn * (1 - y); g = gp * y + gn * (1 - y)
    return -(lo * torch.pow(1 - pt, g)).mean()


def loss_nhat_quan(logit, st, frontal):
    '''Y B: logit [B, C] (sau MIL); st [B] long (study); frontal [B] bool (view_group frontal). Thay = TB sigmoid logit cac anh frontal
       CUNG study trong lo (detach); tro = anh KHONG frontal ma study co frontal trong lo. -> BCE TB (tro x C); khong cap -> None.'''
    fr = frontal.bool()
    assert logit.shape[0] == st.shape[0] == fr.shape[0], f'loss_nhat_quan: lo lech {logit.shape[0]} / {st.shape[0]} / {fr.shape[0]}'
    u, inv = torch.unique(st, return_inverse=True)
    pf = torch.sigmoid(logit.detach().float())
    s = torch.zeros(len(u), logit.shape[1], device=logit.device).index_add_(0, inv[fr], pf[fr])
    n = torch.zeros(len(u), device=logit.device).index_add_(0, inv[fr], torch.ones(int(fr.sum()), device=logit.device))
    tro = ~fr & (n[inv] > 0)
    if not bool(tro.any()): return None
    return F.binary_cross_entropy_with_logits(logit[tro].float(), s[inv[tro]] / n[inv[tro]][:, None])
print(f"78x J v2: cau hinh {CFG_JV2['phien_ban']} | phan them {CFG_JV2['phan_them']} | MIL {len(CFG_JV2['mil'])} nhan | vung thay the "
      f"cho {CFG_JV2['thay_the_nhan']} | neo theo view '{CFG_JV2['view_neo']}' | anh xa {ANH_XA_78X['phien_ban']} | {EPOCHS} epoch")

# ---- 78x J v3: cau hinh (NGUYEN VAN experiments/78_chi_anh/cau_hinh/jca_v3.json) ----
CFG_JV3 = json.loads(r'''{
 "phien_ban": "jca_v3_v1",
 "_ghi_chu": "J v3 (docs/PLAN_JCA_V2.md §5; Tuan 13/10): J v2 lan 2 + lambda x3,23 + y C (nguon nhan) + y D (rang buoc giai phau) + y 5 (ep write-back). he_so_lam = 0,2732 / 0,0846 (cuong do lan 1, chuan hoa duong B1). nguon.bang theo PLAN_79_PHAT_MINH.md §3.2 (ngu canh 78s -> 5 nguon; khong_tu_khoa / khong_report -> -1). trong_so_ce, ghi_nguoc_g: gia dinh, KHONG chinh o vong sang loc.",
 "he_so_lam": 3.23,
 "nguon": {"thu_tu": ["khang_dinh", "nghi_ngo", "chi_cau_hoi", "vang_mat", "khac"],
           "bang": {"khang_dinh": 0, "nghi_ngo": 1, "ngoai_chi_dinh": 2, "phu_dinh": 3, "da_het": 3, "tien_su": 3,
                    "ngoai_so_bo": 4, "ngoai_khac": 4, "khong_nhac": 4, "khong_co_FI": 4},
           "khong_dung": ["khong_tu_khoa", "khong_report"],
           "trong_so_ce": 1.0},
 "ghi_nguoc_g": 5.0
}''')
# ---- khoi 78x (NGUYEN VAN experiments/78_chi_anh/khoi/khoi_jca_v3.py) ----


class NguonNhan(nn.Module):
    '''Y C: offset cho K nguon / lop tren h [B, C, d] (dau ra query cua ML-Decoder, TRUOC GroupFC). delta(h) [B, C, K] = <h_j, w_jk>
       + b_jk (w, b khoi tao 0 -> khong rut so ngau nhien). gop(base [B, C], delta) = base + logsumexp_k(delta) - logsumexp_k(0)
       (CUNG thiet bi / kieu -> TRUNG base tung bit khi delta = 0) = logsumexp_k(base + delta_k) - log K.'''
    def __init__(s, C, d, K=5):
        super().__init__()
        s.K = int(K)
        s.w = nn.Parameter(torch.zeros(int(C), s.K, int(d)))
        s.b = nn.Parameter(torch.zeros(int(C), s.K))

    def delta(s, h):
        return torch.einsum('bcd,ckd->bck', h.float(), s.w) + s.b

    @staticmethod
    def gop(base, delta):
        return base + (torch.logsumexp(delta, -1) - torch.logsumexp(torch.zeros_like(delta), -1)).to(base.dtype)


def loss_nguon(delta, s):
    '''CE "nguon nao": delta [B, C, K], s [B, C] int (nguon 0..K-1 hoac -1). TB tren cap s >= 0; khong cap nao -> None.'''
    m = s >= 0
    if not bool(m.any()): return None
    return F.cross_entropy(delta[m].float(), s[m].long())


def ma_nguon(ngu_canh, bang):
    '''Ngu canh 78s (chuoi) -> chi so nguon int8 theo `bang` {ngu canh: nguon}; ngoai bang (khong_tu_khoa, khong_report, la) -> -1.'''
    return np.array([int(bang.get(str(g), -1)) for g in ngu_canh], 'int8')


def nguon_hang(st_hang, pool, Y, st_nc, nhan_nc, ma_nc, C):
    '''Bang nguon theo hang ORDER [N_O, C] int8: hang POOL, Y = 1, (study, nhan) co trong bang 78s -> ma nguon; con lai -1.
       st_hang [N_O] study cua hang; pool [N_O] bool; Y [N_O, C] (NaN -> khong duong); st_nc / nhan_nc / ma_nc: cot bang 78s.'''
    C = int(C); st_hang = np.asarray(st_hang, 'int64'); Y = np.asarray(Y)
    k = pd.Index(np.asarray(st_nc, 'int64') * C + np.asarray(nhan_nc, 'int64'))
    assert k.is_unique, 'bang ngu canh: (study, nhan) TRUNG'
    out = np.full(Y.shape, -1, 'int8')
    i, j = np.nonzero((Y == 1) & np.asarray(pool, bool)[:, None])
    p = k.get_indexer(st_hang[i] * C + j)
    co = p >= 0
    out[i[co], j[co]] = np.asarray(ma_nc, 'int8')[p[co]]
    return out


class KenhCoDinh(nn.Module):
    '''Y D: logit_j += w_j * max_{k thuoc vung co dinh cua j} z[:, k, j] cho nhan j co vung co dinh (w khoi tao 0 -> luc dau TRUNG
       logit). co_dinh {chi so nhan: [chi so vung, ...]} (moi nhan >= 1 vung); R so vung. z [B, R, C] (DauVung.z).'''
    def __init__(s, co_dinh, R):
        super().__init__()
        idx = sorted(int(j) for j in co_dinh)
        m = torch.zeros(len(idx), int(R), dtype=torch.bool)
        for a, j in enumerate(idx):
            ks = sorted({int(k) for k in co_dinh[j]})
            assert ks and all(0 <= k < int(R) for k in ks), (j, ks)
            m[a, ks] = True
        s.register_buffer('idx', torch.as_tensor(idx, dtype=torch.long))
        s.register_buffer('mat_na', m)
        s.w = nn.Parameter(torch.zeros(len(idx)))

    def forward(s, logit, z):
        if not s.idx.numel(): return logit
        zz = z[:, :, s.idx].float().masked_fill(~s.mat_na.T[None], float('-inf')).amax(1)
        return logit.index_add(1, s.idx, (s.w * zz).to(logit.dtype))


def bit_co_dinh(co_dinh_ten, VUNG, LAB):
    '''anh_xa['co_dinh'] {ten nhan: [ten vung]} -> {chi so nhan: [chi so vung]} (ten la -> KeyError).'''
    vi = {v: i for i, v in enumerate(VUNG)}; li = {l: j for j, l in enumerate(LAB)}
    return {li[l]: [vi[v] for v in vs] for l, vs in co_dinh_ten.items()}


def bo_sung_co_dinh(M, co_dinh):
    '''M [n, C] int64 mat na bit tap vung S cua (anh, nhan); nhan co vung co dinh: o M = 0 (ImaGenome khong co vi tri) -> bit cac vung
       co dinh (vung du phong). Tra BAN SAO; o M != 0 GIU NGUYEN (vi tri ImaGenome uu tien).'''
    M = np.array(M, 'int64', copy=True)
    for j, ks in co_dinh.items():
        b = np.int64(0)
        for k in ks: b |= np.int64(1) << np.int64(int(k))
        M[:, int(j)] = np.where(M[:, int(j)] == 0, b, M[:, int(j)])
    return M


class LinearNhan(nn.Linear):
    '''Y 5: nn.Linear voi dau ra x g (g co dinh, KHONG phai tham so). Dung bang cach DOI LOP ca_h.o cua LopJCA (cung tham so, cung khoa
       state_dict). W_o = 0 -> dau ra 0 (trung tinh); voi Adam, buoc cap nhat W_o ~ LR -> write-back doi ~g lan nhanh hon.'''
    g = 1.0

    def forward(s, x):
        return F.linear(x, s.weight, s.bias) * s.g


def tang_ghi_nguoc(luong, g):
    '''Moi lop JCA cua `luong`: ca_h.o (W_o) -> LinearNhan voi he so g. Tra so lop da doi.'''
    n = 0
    for l in luong.lop:
        l.ca_h.o.__class__ = LinearNhan; l.ca_h.o.g = float(g); n += 1
    return n
print(f"78x J v3: cau hinh {CFG_JV3['phien_ban']} | lambda x{CFG_JV3['he_so_lam']} | nguon nhan {len(CFG_JV3['nguon']['thu_tu'])} nguon, "
      f"CE x{CFG_JV3['nguon']['trong_so_ce']} (CHI dau nguon) | kenh vung co dinh {len(ANH_XA_78X['co_dinh'])} nhan | write-back x"
      f"{CFG_JV3['ghi_nguoc_g']}")

# ---- 78x J v3b: cau hinh (NGUYEN VAN experiments/78_chi_anh/cau_hinh/jca_v3b.json) ----
CFG_JV3B = json.loads(r'''{
 "phien_ban": "jca_v3b_v1",
 "nen": "jca_v3_v1",
 "_ghi_chu": "J v3b (Tuan 16/10; docs/PLAN_SAU_78Y.md §A.3d, SPEC §B.1d): J v3 + MOT thay doi: tran write-back theo anh ||dh|| <= rho ||h|| o moi tang 8-11. rho = 0,10: gia dinh dat truoc (gap 2,5-5 lan muc seed 42 tu dung), KHONG chinh theo ket qua. luat_thu_s44: VAL trong luot ep1 / ep2 toi thieu (= J v3 seed 42 - 0,01), CHI seed 44.",
 "tran_ghi_rho": 0.10,
 "pham_vi": "anh",
 "luat_thu_s44": {"ep1": 0.2345, "ep2": 0.2408}
}''')
# ---- khoi 78x J v3b (NGUYEN VAN experiments/78_chi_anh/khoi/khoi_jca_v3b.py) ----


class LopJCANeoTran(LopJCANeo):
    '''J v3b: LopJCANeo + tran write-back theo anh (xem dau tep). Tra (h', r', hop', ti) nhu lop cha; ti = ||dh sau tran|| / ||h|| tren CA lo
       khi co anh cham tran, con khong = ti cua lop cha (torch.where, KHONG dong bo GPU). Thong ke KHONG grad (tensor tren thiet bi, doc
       NGAY sau forward; checkpoint tinh lai ghi de bang CUNG gia tri): _tk = (ti le tho lon nhat theo anh, so anh cham tran).'''
    rho = None
    _tk = None

    def forward(s, h, r, hop, p, mo_ghi, neo=None):
        h2, r2, hop2, ti = super().forward(h, r, hop, p, mo_ghi, neo)
        if not mo_ghi or s.rho is None:
            return h2, r2, hop2, ti
        B = h.shape[0]
        dh = h2 - h                                                   # phan DA cong vao h (sau lam tron cua phep cong)
        nh = h.float().reshape(B, -1).norm(dim=1)
        nd = dh.float().reshape(B, -1).norm(dim=1)
        tho = nd / nh.clamp_min(1e-12)
        cham = tho > float(s.rho)
        he = torch.where(cham, float(s.rho) * nh / nd.clamp_min(1e-12), torch.ones_like(nd))
        hinh = (B,) + (1,) * (h.dim() - 1)
        m = cham.view(hinh)
        dh_t = he.view(hinh).to(dh.dtype) * dh
        h3 = torch.where(m, h + dh_t, h2)                             # anh duoi tran: h' cua lop cha TUNG BIT
        with torch.no_grad():                                         # ||dh sau tran cua anh b|| = he_b ||dh_b|| -> khong dung tensor lon
            ti3 = (he * nd).square().sum().sqrt() / nh.square().sum().sqrt().clamp_min(1e-12)
            ti = ti3 if ti is None else torch.where(cham.any(), ti3, ti)
            s._tk = (tho.detach().max(), cham.sum())
        return h3, r2, hop2, ti


def ap_tran_ghi(luong, rho):
    '''Moi lop JCA cua `luong` (PHAI la LopJCANeo) -> LopJCANeoTran voi tran rho (> 0). Tra so lop da doi. Khong them tham so / buffer.'''
    rho = float(rho)
    assert rho > 0, f'ap_tran_ghi: rho {rho} phai > 0'
    n = 0
    for l in luong.lop:
        assert type(l) is LopJCANeo, f'ap_tran_ghi: lop {type(l).__name__} khong phai LopJCANeo -> DUNG'
        l.__class__ = LopJCANeoTran; l.rho = rho; n += 1
    return n
assert CFG_JV3B['nen'] == CFG_JV3['phien_ban'], f"jca_v3b dung tren {CFG_JV3B['nen']} != {CFG_JV3['phien_ban']} -> DUNG"
print(f"78x J v3b: cau hinh {CFG_JV3B['phien_ban']} (nen {CFG_JV3B['nen']}) | tran write-back ||dh|| <= {CFG_JV3B['tran_ghi_rho']} ||h|| "
      f"theo {CFG_JV3B['pham_vi']}, moi tang | write-back x{CFG_JV3['ghi_nguoc_g']} GIU NGUYEN")

# ---- 78x J v3b dot 4: cau hinh (NGUYEN VAN experiments/78_chi_anh/cau_hinh/jca_v3b_d4.json) ----
CFG_D4 = json.loads(r'''{
 "phien_ban": "jca_v3b_d4_v1",
 "nen": "jca_v3b_v1",
 "_ghi_chu": "Dot 4 bo trich J v3b (Tuan duyet 21/10; docs/PLAN_SAU_78Y.md §A.4j, SPEC §B.2n; Event 2026-10-21 Tuan duyet dot 4). Moi luot: J v3b s42 + MOT thay doi, 6 epoch, ghep cap r78x_J_v3b_s42 (chep khoi dong luong + lambda), 78z doc backbone ep4. dp_ema = stochastic depth theo block (p_i = dp x i / 11, generator rieng, ngoai checkpoint) + EMA trong so (ema, khoi dong min(ema, (1 + n) / (10 + n))); rs_ nhe = trong so EMA. kd = dich loss nhan chinh = (1 - alpha) x nhan cung + alpha x xac suat giao vien (TB 3 seed J v3b, dau mldec_hr 78y ep1-10, tung anh train); loss phu giu nhan cung.",
 "seed": 42,
 "epochs": 6,
 "ep_bb": 4,
 "ghep_cap": ["stage4_nb78x", "r78x_J_v3b_s42"],
 "dp_ema": {"dp": 0.1, "ema": 0.999, "hat_dp": 41},
 "kd": {"alpha": 0.5,
        "thay": {"duong": "stage4_nb78y", "seeds": [42, 43, 44], "run": "r78y_J_v3b_s{seed}_thay", "tep": "p_thay_mldec_hr_train_s{seed}.npy",
                 "meta": "thay_mldec_hr_train_s{seed}.json", "nguon": "r78y_J_v3b_s{seed}", "dau": "mldec_hr", "epochs": 10}}
}''')
# ---- khoi on dinh fine-tune (NGUYEN VAN experiments/78_chi_anh/khoi/khoi_on_dinh_ft.py) ----
import contextlib


def ti_le_dp(L, p):
    '''Xac suat bo block i (stochastic depth, quy tac tuyen tinh): p x i / (L - 1); block 0 = 0, block cuoi = p.'''
    assert int(L) >= 2 and 0.0 <= float(p) < 1.0, (L, p)
    return [float(p) * i / (int(L) - 1) for i in range(int(L))]


def mat_na_dp(B, ti_le, gen, device):
    '''-> list theo block: None (p_i = 0) hoac tensor [B, 1, 1] float32 = 0 (bo) / 1 / (1 - p_i) (giu). Rut tu `gen` (RIENG), TRUOC vong block.'''
    out = []
    for p in ti_le:
        if p <= 0:
            out.append(None); continue
        giu = (torch.rand(int(B), generator=gen, device=device) >= p).float() / (1.0 - p)
        out.append(giu.view(int(B), 1, 1))
    return out


def chay_bb_luong_dp(bb, luong, img, mo_ghi=True, grad_bb=True, ckpt_bb=(), ti_le=None, gen=None):
    '''= chay_bb_luong (khoi_luong_vung, 79) + stochastic depth theo block khi bb.training, co grad backbone va ti_le khac None.
       -> (patch [B, N, d] float32, cls [B, d] float32, luong.dau_ra). Khong ap -> phep tinh TRUNG chay_bb_luong tung bit.'''
    g = torch.is_grad_enabled(); gb = g and bool(grad_bb); ck = {int(i) for i in ckpt_bb}
    assert grad_bb or not g or not (mo_ghi and luong.hai_chieu), \
        'chay_bb_luong: backbone dong (grad_bb=False) ma duong ghi nguoc mo -> grad cua luong bi cat, DUNG'
    with torch.set_grad_enabled(gb):
        h = bb.embeddings(img)
    st = luong.bat_dau(img.shape[0])
    dp = ti_le is not None and gb and bb.training
    if dp:
        assert gen is not None and len(ti_le) == len(bb.encoder.layer), 'chay_bb_luong_dp: thieu generator / so block lech ti_le -> DUNG'
        m = mat_na_dp(img.shape[0], ti_le, gen, h.device)
    for l, lop in enumerate(bb.encoder.layer):
        with torch.set_grad_enabled(gb):
            hm = _out_luong(_tuc_luong.checkpoint(lop, h, use_reentrant=False) if (gb and l in ck) else lop(h))
            if dp and m[l] is not None:
                hm = (h.float() + (hm.float() - h.float()) * m[l]).to(hm.dtype)
            h = hm
        h = luong.buoc(l, h, st, mo_ghi)
    with torch.set_grad_enabled(gb):
        h = bb.layernorm(h)
    return h[:, 1:].float(), h[:, 0].float(), luong.dau_ra(st)


class EmaTrongSo:
    '''EMA trong so (float32, cung thiet bi voi tham so) cua cac module {ten: nn.Module}. Ten tham so = '<ten module>.<ten trong module>';
       tham so dung chung (cung id) chi lay o module DAU TIEN.'''

    def __init__(s, modules, decay):
        s.modules, s.decay, s.n = dict(modules), float(decay), 0
        assert 0.0 < s.decay < 1.0, s.decay
        thay, s.tham = set(), []
        for t, m in s.modules.items():
            for k, p in m.named_parameters():
                if id(p) in thay: continue
                thay.add(id(p)); s.tham.append((f'{t}.{k}', p))
        s.bong = [p.detach().float().clone() for _, p in s.tham]

    def he_so(s):
        return min(s.decay, (1.0 + s.n) / (10.0 + s.n))

    @torch.no_grad()
    def cap_nhat(s):
        d = s.he_so()
        torch._foreach_lerp_(s.bong, [p.detach().float() for _, p in s.tham], 1.0 - d)   # bong <- d bong + (1 - d) p
        s.n += 1

    @contextlib.contextmanager
    def ap(s):
        '''Trong khoi with: tham so = EMA; ra khoi (ke ca loi): tham so = ban tho TRUOC khi vao, tung bit.'''
        tho = [p.detach().clone() for _, p in s.tham]
        with torch.no_grad():
            for (_, p), b in zip(s.tham, s.bong): p.copy_(b.to(p.dtype))
        try:
            yield s
        finally:
            with torch.no_grad():
                for (_, p), t in zip(s.tham, tho): p.copy_(t)

    def sd_module(s, ten):
        '''state_dict cua module `ten` voi tham so thay bang EMA (dtype cua module), buffer giu nguyen — tensor MOI (khong alias tham so).'''
        m = s.modules[ten]; pre = ten + '.'
        sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
        for (t, _), b in zip(s.tham, s.bong):
            if t.startswith(pre):
                k = t[len(pre):]
                if k in sd: sd[k] = b.to(dtype=sd[k].dtype, device=sd[k].device).clone()
        return sd

    def state_dict(s):
        return dict(decay=s.decay, n=s.n, ten=[t for t, _ in s.tham], bong=[b.detach().cpu() for b in s.bong])

    def load_state_dict(s, sd):
        assert list(sd['ten']) == [t for t, _ in s.tham] and float(sd['decay']) == s.decay, 'EMA: ten / he so tham so KHAC luc luu -> DUNG'
        s.n = int(sd['n'])
        s.bong = [b.to(p.device).float().clone() for b, (_, p) in zip(sd['bong'], s.tham)]
BIEN_THE_D4 = 'dp_ema'                                  # 78x J v3b dot 4 (Tuan duyet 21/10; docs/SPEC_SAU_78Y.md B.2n): luot cua notebook nay
assert CFG_D4['nen'] == CFG_JV3B['phien_ban'] and SEED in tuple(CAU_HINH_C['seeds']) and EPOCHS == int(CFG_D4['epochs']) and (
    BIEN_THE_D4 in CFG_D4), f"jca_v3b_d4: nen {CFG_D4['nen']} / seed {SEED} / epochs {EPOCHS} / luot {BIEN_THE_D4} KHAC cau hinh -> DUNG"
EP_BB_D4 = int(CFG_D4['ep_bb'])                        # 78z dung rs_ epoch nay (nhu J v3b s42)
CFG_D4_CHON = dict(ten=BIEN_THE_D4, phien_ban=CFG_D4['phien_ban'], ghep_cap=list(CFG_D4['ghep_cap']), epochs=EPOCHS, ep_bb=EP_BB_D4, khe=None,
                   dp_ema=(dict(CFG_D4['dp_ema']) if BIEN_THE_D4 == 'dp_ema' else None),
                   kd=(dict(CFG_D4['kd']) if BIEN_THE_D4 == 'kd' else None), trong_so=('EMA' if BIEN_THE_D4 == 'dp_ema' else 'tho'))
print(f"78x J v3b dot 4: luot {BIEN_THE_D4} | {EPOCHS} epoch | 78z dung rs_ ep{EP_BB_D4} (trong so {CFG_D4_CHON['trong_so']}) | ghep cap "
      f"{'/'.join(CFG_D4['ghep_cap'])} (CHEP khoi dong + lambda) | dp_ema {CFG_D4_CHON['dp_ema']} | kd alpha "
      f"{(CFG_D4_CHON['kd'] or {}).get('alpha')}")

# ---- 78x o cat (Tuan duyet 26/10; docs/SPEC_SAU_78Y.md B.2z; PLAN A.4t; Event 2026-10-26 "Tuan duyet dua o cat vao J v3b ..."): B.4 v2 + 5 o luoi
#      (2 x 2 + giua) phong tu anh 518 DA tang cuong, di qua backbone KHONG gradient + adapter duong o + skip block 7 -> block 8 + dau doc 6 view ----
O_CFG = {'ten': 'o_luoi_phong518_skip7_gop6', 'phien_ban': 'o_j_v1', 'n_o': 5, 'k_skip': 8, 'r_adapter': 64, 'khuc': 240, 'hat': {'bo_o7': 61, 'bo_o11': 62}, 'sang_loc': {'moc_run': ['stage4_nb78x', 'r78x_J_v3b_dp_ema_v2_s42'], 'nguong': 0.003}}
SO_O, K_SKIP_O, KHUC_O = int(O_CFG['n_o']), int(O_CFG['k_skip']), int(O_CFG['khuc'])
assert SO_O == 5 and K_SKIP_O == 8 and int(O_CFG['r_adapter']) == 64 and 1 <= KHUC_O <= 1024 and float(O_CFG['sang_loc']['nguong']) == 0.003 \
    and BIEN_THE_D4 == 'dp_ema' and tuple(O_CFG['sang_loc']['moc_run']) == ('stage4_nb78x', 'r78x_J_v3b_dp_ema_v2_s42'), 'o cat: cau hinh lech -> DUNG'
O_CFG_CHON = dict(O_CFG)
print(f"78x o cat: {SO_O} o luoi phong tu anh 518 (khong gradient ve backbone, khuc {KHUC_O} view) | adapter r {O_CFG['r_adapter']} | skip block "
      f"{K_SKIP_O - 1} -> {K_SKIP_O} | dau doc gop {1 + SO_O} view | sang loc vs {O_CFG['sang_loc']['moc_run'][1]} >= +{O_CFG['sang_loc']['nguong']}")

# ---- doan NGUYEN VAN experiments/78_chi_anh/khoi/khoi_cat_vung.py ----


S_1022, S_518, CANH_O = 1022, 518, 518


O_1022 = ((0, 0), (504, 0), (0, 504), (504, 504), (252, 252))     # (x0, y0) goc tren-trai tung o tren anh 1022: 2 x 2 + o giua


assert all(0 <= v and v + CANH_O <= S_1022 for o in O_1022 for v in o)


def luoi_518(dev='cpu'):
    '''Luoi grid_sample [5, 518, 518, 2] (thu tu x, y) cho cac o tren anh 518. Tam pixel dau ra j cua o (x0, y0) -> toa do chuan hoa
       g = 2 (x0 + j + 0,5) / 1022 - 1 (align_corners=False) <=> toa do pixel nguon (x0 + j + 0,5) * 518 / 1022 - 0,5 (Codex muc 44.3).'''
    j = torch.arange(CANH_O, dtype=torch.float64)
    out = []
    for x0, y0 in O_1022:
        gx, gy = 2 * (x0 + j + 0.5) / S_1022 - 1, 2 * (y0 + j + 0.5) / S_1022 - 1
        GY, GX = torch.meshgrid(gy, gx, indexing='ij')
        out.append(torch.stack([GX, GY], -1))
    return torch.stack(out).float().to(dev)


def lay_mau_518(x, luoi):
    '''x [B, 518, 518] (uint8 / float) -> [B, 5, 518, 518] float: bicubic, align_corners=False, padding 'border' (CHUA lam tron).'''
    xf = x.float().unsqueeze(1)
    B = xf.shape[0]
    return torch.stack([F.grid_sample(xf, luoi[k].unsqueeze(0).expand(B, -1, -1, -1), mode='bicubic', padding_mode='border',
                                      align_corners=False)[:, 0] for k in range(luoi.shape[0])], 1)


# ---- doan NGUYEN VAN experiments/78_chi_anh/khoi/khoi_cat_sau.py ----


N_LUOI = S_518 // 14                 # 37 token moi canh (patch 14)


TI_1022 = S_1022 / S_518             # toa do anh 518 -> anh 1022


def hinh_luoi(B, dev='cpu'):
    '''-> [B, 5, 3] long (x0, y0, L) tren anh 1022 cua 5 o luoi (O_1022, canh 518).'''
    g = torch.tensor([[x0, y0, CANH_O] for x0, y0 in O_1022], dtype=torch.long, device=dev)
    return g[None].expand(int(B), -1, -1).clone()


def luoi_con(geo):
    '''geo [B, K, 3] (x0, y0, L tren anh 1022) -> (g [B, K, P, 2] float32, w [B, K, P] float32), P = 37 * 37 * 4, thu tu (i, j, qy, qx):
       diem con q cua token toan anh (i, j) o toa do 1022: p_x = (14 j + 3,5 + 7 qx) 1022 / 518 (y tuong tu); g = 2 (p - x0) / L - 1
       (grid_sample align_corners=False tren luoi token 37 x 37 cua o); w = 1 neu diem con nam trong o.'''
    dev = geo.device
    t = torch.arange(N_LUOI, dtype=torch.float64, device=dev)
    q = torch.tensor([3.5, 10.5], dtype=torch.float64, device=dev)
    pc = (14 * t[:, None] + q[None]) * TI_1022                               # [37, 2]: toa do con theo mot truc
    py = pc[:, None, :, None].expand(N_LUOI, N_LUOI, 2, 2).reshape(-1)       # theo (i, qy)
    px = pc[None, :, None, :].expand(N_LUOI, N_LUOI, 2, 2).reshape(-1)       # theo (j, qx)
    gf = geo.to(torch.float64)
    x0, y0, L = gf[..., 0:1], gf[..., 1:2], gf[..., 2:3]
    dx, dy = px[None, None] - x0, py[None, None] - y0
    w = ((dx >= 0) & (dx < L) & (dy >= 0) & (dy < L)).float()
    g = torch.stack([2 * dx / L - 1, 2 * dy / L - 1], -1).float()
    return g, w


@torch.no_grad()
def lay_mau_con(c, g, w):
    '''Token o c [B, K, N, d] (KHONG CLS; N = 37 * 37) + (g, w) cua luoi_con -> (v [B, N, 4, d] float32 = TB co trong so w qua K o cua gia
       tri bilinear tai 4 diem con cua moi token toan anh, phu [B, N] bool = ca 4 diem con co >= 1 o phu). Khong tham so, khong grad.'''
    B, K, N, d = c.shape
    n = int(round(N ** 0.5))
    assert n * n == N and tuple(g.shape[:2]) == (B, K) and g.shape[2] == 4 * N, (tuple(c.shape), tuple(g.shape))
    tong = torch.zeros(B, 4 * N, d, dtype=torch.float32, device=c.device)
    for k in range(K):
        m = c[:, k].reshape(B, n, n, d).permute(0, 3, 1, 2).float()
        s = F.grid_sample(m, g[:, k][:, None].float(), mode='bilinear', padding_mode='border', align_corners=False)[:, :, 0]
        tong += s.transpose(1, 2) * w[:, k, :, None]
    ws = w.sum(1)
    v = tong / ws.clamp(min=1.0)[..., None]
    return v.reshape(B, N, 4, d), (ws.reshape(B, N, 4) > 0).all(-1)


class DuaVao(nn.Module):
    '''Skip bac thap (cach a): v [B, N, 4, d] (lay_mau_con) -> LayerNorm(d) -> ghep 4 d -> Linear(4 d, d) KHOI TAO 0 -> x phu -> [B, N, d].
       Luc khoi tao dau ra = 0 tung bit -> nhanh skip = nhanh 0.'''
    def __init__(s, d=768):
        super().__init__()
        s.ln = nn.LayerNorm(d)
        s.chieu = nn.Linear(4 * d, d)
        nn.init.zeros_(s.chieu.weight)
        nn.init.zeros_(s.chieu.bias)

    def forward(s, v, phu):
        B, N, q, d = v.shape
        return s.chieu(s.ln(v).reshape(B, N, q * d)) * phu[..., None].to(v.dtype)


# ---- khoi NGUYEN VAN experiments/78_chi_anh/khoi/khoi_fusion_study.py ----


class GopStudy(nn.Module):
    '''Gom 40 dau ra query h [B, Q, d] (mldec_h, TRUOC GroupFC) cua cac anh CUNG study trong lo -> h_study phat lai cho tung anh [B, Q, d].
       kieu 'tb'  : h_S = TB_i (h_i + e_view[v_i]) tren anh con giu;
       kieu 'attn': a_iq = softmax_i(<w_q, LN(h_iq)> + b[v_i, q]) tren anh con giu; h_S,q = sum_i a_iq (h_iq + e_view[v_i]).
       MOI tham so moi khoi tao 0 (LN mac dinh 1 / 0) -> KHONG rut so ngau nhien; luc dau 'tb' = 'attn' = TB deu cac anh cua study; study
       MOT anh -> h_i TUNG BIT (dau fusion luc dau = dau mldec cung khoi tao o study mot anh).
       st [B] long (study cua tung anh), view [B] long (0 frontal / 1 lateral / 2 khac), giu [B] bool (None = giu het; CHI luc train, moi
       study phai con >= 1 anh giu).'''
    def __init__(s, kieu, Q, d, n_view=3):
        super().__init__()
        assert kieu in ('tb', 'attn'), kieu
        s.kieu, s.Q, s.d, s.n_view = kieu, int(Q), int(d), int(n_view)
        s.e_view = nn.Parameter(torch.zeros(s.n_view, s.d))
        if kieu == 'attn':
            s.ln = nn.LayerNorm(s.d)
            s.w = nn.Parameter(torch.zeros(s.Q, s.d))
            s.b = nn.Parameter(torch.zeros(s.n_view, s.Q))

    def forward(s, h, st, view, giu=None):
        B, Q, d = h.shape
        assert Q == s.Q and d == s.d and st.shape == (B,) and view.shape == (B,), (tuple(h.shape), tuple(st.shape), tuple(view.shape))
        _, inv = torch.unique(st, return_inverse=True)
        S = int(inv.max()) + 1
        g = torch.ones(B, dtype=torch.bool, device=h.device) if giu is None else giu.to(h.device).bool()
        hv = h + s.e_view[view].unsqueeze(1).to(h.dtype)
        if s.kieu == 'tb':
            w = g.to(h.dtype).unsqueeze(1).expand(B, Q)
        else:
            sc = (s.ln(h) * s.w).sum(-1) + s.b[view]                                       # [B, Q]
            sc = sc.masked_fill(~g.unsqueeze(1), float('-inf'))
            mx = torch.full((S, Q), float('-inf'), dtype=sc.dtype, device=sc.device).scatter_reduce(
                0, inv.unsqueeze(1).expand(B, Q), sc, reduce='amax', include_self=True)
            w = torch.exp(sc - mx[inv]) * g.unsqueeze(1).to(sc.dtype)
        num = torch.zeros(S, Q, d, dtype=hv.dtype, device=h.device).index_add_(0, inv, w.unsqueeze(-1).to(hv.dtype) * hv)
        den = torch.zeros(S, Q, dtype=w.dtype, device=h.device).index_add_(0, inv, w)
        return (num / den.unsqueeze(-1).to(num.dtype))[inv]


def rut_giu_view(st, gen, p_bo):
    '''Mat na giu anh cho fusion LUC TRAIN: moi anh cua study NHIEU anh bi bo voi xac suat p_bo (doc lap); study bi bo het -> giu anh co so
       rut LON NHAT (moi study con >= 1 anh). Study mot anh: luon giu. So rut CHI tu `gen` (generator rieng, cung thiet bi voi st) -> dong
       ngau nhien chung KHONG doi. st [B] long -> giu [B] bool.'''
    B = st.shape[0]
    u = torch.rand(B, generator=gen, device=st.device)
    _, inv, cnt = torch.unique(st, return_inverse=True, return_counts=True)
    giu = (u >= p_bo) | (cnt[inv] == 1)
    S = cnt.shape[0]
    con = torch.zeros(S, dtype=torch.long, device=st.device).index_add_(0, inv, giu.long())
    mx = torch.full((S,), -1.0, dtype=u.dtype, device=st.device).scatter_reduce(0, inv, u, reduce='amax', include_self=True)
    return giu | ((con[inv] == 0) & (u == mx[inv]))


class LoGomStudy(torch.utils.data.Sampler):
    '''Lo DU DOAN gom TRON study (batch_sampler cua DataLoader): vi tri xep theo study (study theo thu tu xuat hien dau tien trong `st`, anh
       trong study giu thu tu), goi tham lam tung study vao lo <= bs anh; study > bs anh -> MOT lo rieng. Tat dinh (khong rut so). Moi vi
       tri 0..n-1 xuat hien DUNG MOT lan; KHONG study nao bi cat qua hai lo. st [n] = study_id theo vi tri (vi tri trong idx cua DS).'''
    def __init__(s, st, bs):
        st = np.asarray(st)
        assert st.ndim == 1 and len(st) and int(bs) >= 1, 'LoGomStudy: st 1 chieu khong rong, bs >= 1'
        _, dau, inv = np.unique(st, return_index=True, return_inverse=True)
        thu_tu = np.argsort(inv, kind='stable')                                   # vi tri xep theo study, giu thu tu trong study
        ptr = np.concatenate([[0], np.cumsum(np.bincount(inv))])
        s.lo, cur = [], []
        for k in np.argsort(dau, kind='stable'):                                   # study theo thu tu xuat hien dau tien
            pos = thu_tu[ptr[k]:ptr[k + 1]].tolist()
            if cur and len(cur) + len(pos) > int(bs):
                s.lo.append(cur); cur = []
            cur = cur + pos
        if cur: s.lo.append(cur)
        s.n = int(len(st))

    def __len__(s):
        return len(s.lo)

    def __iter__(s):
        return iter([list(x) for x in s.lo])


# ---- khoi NGUYEN VAN experiments/78_chi_anh/khoi/khoi_cat_j.py ----


import contextlib
lay_mau_con_grad = lay_mau_con.__wrapped__        # CUNG phep tinh voi lay_mau_con (khoi_cat_sau, boc @torch.no_grad) nhung CO gradient


def o_phong_518(img, luoi, mean, std):
    '''img [B, 3, 518, 518] float = anh 518 DA chuan hoa ((x / 255 - mean) / std, 3 kenh giong nhau) -> 5 o [B, K, 518, 518] float32 CUNG
       chuan hoa: bicubic tren kenh 0 (lay_mau_518, align_corners=False, padding border), kep ve khoang chuan hoa cua [0, 255] (nhu kep
       uint8 cua cat_518_phong; khong lam tron muc xam).'''
    o = lay_mau_518(img[:, 0], luoi)
    return o.clamp((0.0 - float(mean)) / float(std), (1.0 - float(mean)) / float(std))


@torch.no_grad()
def token_o(bb, img, luoi, mean, std, khuc, k7):
    '''5 o phong (o_phong_518) qua bb (embeddings -> block 0..L-1 -> layernorm) KHONG gradient, chia khuc `khuc` view (autocast do noi goi
       dat) -> (t7 [B, K, N, d] = token patch sau block k7 (0-based), t11 [B, K, N, d] = token patch cuoi sau layernorm); bo CLS.
       Khong qua luong vung, khong drop-path (drop-path B.4 CHI o duong anh toan canh); RAD-DINO dropout = 0 -> khong rut so.'''
    B = int(img.shape[0])
    o = o_phong_518(img, luoi, mean, std)
    K = int(o.shape[1])
    o = o.reshape(B * K, o.shape[2], o.shape[3])
    t7, t11 = [], []
    for i in range(0, B * K, int(khuc)):
        h = bb.embeddings(o[i:i + int(khuc)].unsqueeze(1).expand(-1, 3, -1, -1).contiguous().to(img.dtype))
        for l, lop in enumerate(bb.encoder.layer):
            h = _out_luong(lop(h))
            if l == int(k7): t7.append(h[:, 1:])
        t11.append(bb.layernorm(h)[:, 1:])
    t7, t11 = torch.cat(t7), torch.cat(t11)
    return t7.reshape(B, K, t7.shape[1], t7.shape[2]), t11.reshape(B, K, t11.shape[1], t11.shape[2])


class BoChinhO(nn.Module):
    '''Adapter duong o (Tuan 26/10: "mo nho truoc de tranh sup va danh gia tiem nang"): x + ra(GELU(vao(LN(x)))), vao = Linear(d, r),
       ra = Linear(r, d) KHOI TAO 0 -> luc dau = x TUNG BIT. `vao` rut so -> ben goi tao trong fork_rng voi hat rieng.'''
    def __init__(s, d=768, r=64):
        super().__init__()
        s.ln, s.vao, s.ra = nn.LayerNorm(d), nn.Linear(d, int(r)), nn.Linear(int(r), d)
        nn.init.zeros_(s.ra.weight)
        nn.init.zeros_(s.ra.bias)

    def forward(s, x):
        return x + s.ra(F.gelu(s.vao(s.ln(x.to(s.ln.weight.dtype))))).to(x.dtype)      # nhanh tinh theo kieu tham so (autocast van ap)


def skip_o(t7, bo, dua, g, w, ckpt=False):
    '''Skip block 7 (cach a cua pha 1b) CO gradient vao adapter + phep chieu: t7 [B, K, N, d] (token o sau block 7, khong grad) -> adapter
       `bo` -> lay mau 2 x 2 diem con / token toan anh (g [B, K, P, 2], w [B, K, P] cua luoi_con) -> DuaVao `dua` -> inj [B, N, d].
       ckpt=True: tinh lai luc backward (torch.utils.checkpoint, use_reentrant=False) -> khong giu kich hoat [B, K, N, d] float32.'''
    def f(t):
        v, phu = lay_mau_con_grad(bo(t), g, w)
        return dua(v, phu)
    return _tuc_luong.checkpoint(f, t7, use_reentrant=False) if ckpt else f(t7)


def cong_skip(h, inj):
    '''h [B, 1 + N, d] (co CLS) + inj [B, N, d] vao token patch; CLS giu nguyen. inj None -> h (cung object).'''
    if inj is None: return h
    return torch.cat([h[:, :1], h[:, 1:] + inj.to(h.dtype)], 1)


@contextlib.contextmanager
def skip_vao_luong(luong, inj, k_skip=8, do=None):
    '''Trong with: luong.buoc(k_skip - 1, h, ...) tra cong_skip(h, inj) -> dau vao block k_skip co skip; dung voi chay_bb_luong (79) /
       chay_bb_luong_dp (B.4) KHONG sua (ca hai goi luong.buoc sau MOI block). do = list | None: them ||inj|| / ||h patch|| TB anh (tensor
       khong grad, khong dong bo GPU). inj None -> khong doi gi. Ra khoi with (ke ca loi): tra luong.buoc goc.'''
    if inj is None:
        yield luong
        return
    assert 'buoc' not in vars(luong), 'skip_vao_luong: luong.buoc da bi boc (long nhau?) -> DUNG'
    goc = luong.buoc

    def buoc(l, h, st, mo_ghi):
        h = goc(l, h, st, mo_ghi)
        if int(l) != int(k_skip) - 1: return h
        if do is not None:
            with torch.no_grad():
                do.append((inj.float().norm(dim=-1).mean(1) / h[:, 1:].float().norm(dim=-1).mean(1).clamp_min(1e-12)).mean())
        return cong_skip(h, inj)
    luong.buoc = buoc
    try:
        yield luong
    finally:
        del luong.buoc


def gop_view_o(gop, h0, h_o):
    '''h0 [B, Q, d] (view 0 = anh toan canh) + h_o [B, K, Q, d] (K o) -> h_bag [B, Q, d] = CUNG phep tinh GopStudy `gop` (kieu 'attn',
       giu None, n_view = 1 + K) tren 1 + K view cua TUNG anh (view = 0..K), viet DANG DAY: amax / tong theo truc view thay scatter_reduce /
       index_add_ (atomic -> GPU lech bit giua hai lan chay, Colab 26/10: cong duong cham C6 dung) -> TAT DINH. Dung tham so cua `gop`
       (e_view, ln, w, b) -> state_dict khong doi. Tham so gop = 0 -> TB 1 + K view.'''
    V = 1 + int(h_o.shape[1])
    assert gop.kieu == 'attn' and gop.n_view == V and tuple(h_o.shape[2:]) == tuple(h0.shape[1:]), (gop.kieu, gop.n_view, tuple(h_o.shape))
    h = torch.cat([h0.unsqueeze(1), h_o.to(h0.dtype)], 1)                                       # [B, V, Q, d]
    hv = h + gop.e_view.unsqueeze(1).to(h.dtype)                                                 # + e_view[v]
    sc = (gop.ln(h) * gop.w).sum(-1) + gop.b                                                     # [B, V, Q]
    w = torch.exp(sc - sc.amax(1, keepdim=True))
    return (w.unsqueeze(-1).to(hv.dtype) * hv).sum(1) / w.sum(1).unsqueeze(-1).to(hv.dtype)


@torch.no_grad()
def thong_ke_o(gop, bo7, bo11, dua, do=None):
    '''Chan doan duong o (so tong hop, khong grad): ||inj|| / ||h patch block 7|| TB (tu `do` cua skip_vao_luong), chuan Linear ra cua hai
       adapter, chuan phep chieu skip, |b| TB va ||e_view|| theo view + ||w|| cua bo gop view.'''
    return dict(ti_le_inj=(float(torch.stack(do).mean()) if do else None), bo7=float(bo7.ra.weight.norm()), bo11=float(bo11.ra.weight.norm()),
                dua=float(dua.chieu.weight.norm()), gop_w=float(gop.w.norm()), gop_b_view=[float(v) for v in gop.b.abs().mean(1)],
                gop_e_view=[float(v) for v in gop.e_view.norm(dim=1)])
