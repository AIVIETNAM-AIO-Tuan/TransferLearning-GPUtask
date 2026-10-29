# ==== nhanh_c buoc p3_C0 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ---- 78x o cat: CHAN RAC OUTPUT TAI GOC (Tuan 27/10: warning khong anh huong tien trinh -> bo cho output de doc; cach lam nhu
#      CHAN_RAC_81 cua experiments/81_jca_hai_luong/dung_81.py). KHONG doi du lieu / ket qua; canh bao / loi KHAC van in ----
# (1) google_auth_httplib2 "per-request timeout": thong bao cua thu vien trong lenh goi Drive API (khoi Drive CHUAN, khong sua) -> ha DUNG
#     logger do xuong ERROR.
# (2) huggingface_hub (RAD-DINO cong khai): khong tim token ngam, khong bao "unauthenticated"; tat thanh tien do tai / nap trong so cua HF
#     (transformers doc co nay luc import o C2) -> env + logger xuong ERROR + bo DUNG canh bao "Error while fetching HF_TOKEN secret".
#     Thanh tien do train / val cua notebook (tqdm.auto) GIU.
# (3) PyTorch "Converting a tensor with requires_grad=True to a scalar": float() tren tham so trong cong kiem (vd C4b) -> bo DUNG cau do.
# ---- nhanh C (Tuan duyet 29/10; SPEC B.2ab): cau hinh theo seed (CAU_HINH_C do nhanh_c.buoc dat); backbone = Phan 1 nhanh C cua CHINH seed
#      nay o stage4_nb84 (TM_BB) ----
assert isinstance(globals().get('CAU_HINH_C'), dict) and CAU_HINH_C.get('nhanh') == 'C', 'chay qua nhanh_c.buoc(...) (dat CAU_HINH_C) -> DUNG'
TM_BB = CAU_HINH_C['tm_x']
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
import os, json, math, time, tarfile, io as _io, re, datetime, copy
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
try:
    from google.colab import drive
    drive.mount('/content/drive'); DATA_DIR = '/content/drive/MyDrive/MIMIC_CXR/planB_out'
except Exception:
    DATA_DIR = os.environ.get('CXRLT_DATA', '')
    assert DATA_DIR, ('khong phai Colab va chua dat CXRLT_DATA -> gan thang duong dan `planB_out`. '
                      'Nhanh du phong "./planB_out" cu bao SAI nguyen nhan.')
OUT_DIR = os.path.join(DATA_DIR, 'cxrlt_2024'); EMB_DIR = os.path.join(OUT_DIR, 'embeddings')
THU_NHANH = bool(CAU_HINH_C['thu'])   # nhanh C: luot thu (Phan 2 / 3 chi khi Tuan bat)
                    #       do VRAM / toc do MOI arm + thap BiomedBERT TRUOC luot dai
TEN_ARM_78T = ('mldec', 'q_hoc', 'txtq', 'txtq_nn', 'txtq_5mt', 'txtq_cau', 'txtq_k4', 'txtq_du', 'txtq_vung', 'vung_kdp',
               'dich_thay', 'dich_tt', 'dich_tb', 'dich_tat_ca', 'txtq_dich', 'cau_bert', 'cau_clip', 'mldec_hr', 'fus_tb_hr', 'fus_attn_hr')   # 78t: 17 arm + 78y mldec_hr + 78z J fus_*_hr (dau J doc [patch ; token vung]), thu tu CO DINH (C4)
LUOT_78Y = 'J_v3'   # 78y (Tuan 13/10; docs/PLAN_JCA_V2.md §7): 'J_v3' = backbone J v3 ep4 (luong vung dong cung) | 'B1_e6' =
                     #      B1 6 epoch ep4 (doi chung, CUNG giao thuc 6 epoch + lo theo study) | 'B1_e6_lam34' = B1 6 epoch
                     #      lambda x3,4 ep4 (doi chung CUNG lambda voi J v3; SPEC_SAU_78Y 1.2). MOT luot MOT phien.
ARMS_GIU = ['mldec', 'mldec_hr', 'fus_tb_hr', 'fus_attn_hr']   # 78z J: mldec (doi chung cac cong 78t), mldec_hr (doi chung QUYET DINH),
                     #      fus_tb_hr / fus_attn_hr = fusion study tren dau ra query cua mldec_hr
LAM_TU = None        # 78v: lambda (vung_kdp) hieu chinh RIENG moi backbone (lan dau), phien sau doc lai
LUOT_78T = ''        # 78v: KHONG dung (giu ten cho C4 / C6 cua 78t)
CHI_LAM = False      # 78v: KHONG dung
RUN_TAG = CAU_HINH_C['run_gts']   # 78z dot 4 dp_ema gop trong so (Tuan duyet 21/10): du doan lai val, KHONG train; thu muc RIENG
assert LUOT_78Y in ('J_v3', 'B1_e6', 'B1_e6_lam34'), f'LUOT_78Y {LUOT_78Y!r}: chi J_v3 / B1_e6 / B1_e6_lam34 (backbone 78x ep4)'
assert list(ARMS_GIU)[:1] == ['mldec'] and len(set(ARMS_GIU)) == len(ARMS_GIU) and set(ARMS_GIU) <= set(TEN_ARM_78T), (
    f'ARMS_GIU {ARMS_GIU}: con cua 17 arm 78t, khong trung, mldec DAU')
assert LAM_TU is None and LUOT_78T == '' and not CHI_LAM, (
    '78v: lambda hieu chinh RIENG moi backbone -> LAM_TU = None, LUOT_78T rong, CHI_LAM = False')
# giu ten bien `RES57` de khong phai sua hang chuc cho; thu muc thi doi.
RES57 = os.path.join(OUT_DIR, CAU_HINH_C['tm'], RUN_TAG)   # CHI de in; GHI qua khoi chuan (cuoi cell)
P_OOF = os.path.join(OUT_DIR, 'stage4_nb52a', 'xgb_oof.npy')
# 26/09: BO ensure_res()/makedirs/_run_info qua FUSE (loi so 2 ngay 23/09). Ghi = khoi chuan o cuoi cell.
assert os.path.isdir(OUT_DIR), (
    f'{OUT_DIR} khong ton tai: tai khoan nay CHUA co shortcut MIMIC_CXR trong My Drive.\n'
    '  drive.google.com -> "Shared with me" -> chuot phai MIMIC_CXR -> "Add shortcut to Drive" -> My Drive.')
LOCAL = '/content/cxrlt_518'
READY_X = os.path.join(OUT_DIR, 'cxrlt_518_ready')   # kho do 41c C3-bis tao

FT_BLOCKS = (8, 9, 10, 11)      # mo bang 4 block tren. `11`: I_location ~ 0, nen chon theo chi phi.
LR_FT_DIV = 10.0                # LR cho block mo bang = LR/10 — chuan LP-FT, tranh quen pretrain
BB_RUN = CAU_HINH_C['run_x']   # nhanh C: Phan 1 cua CHINH seed nay (TM_BB = stage4_nb84)
BIEN_THE_BB = 'dp_ema'                 # luot dot 4 (C2b kiem cfg_d4 cua backbone_ft.pt)
BB_TEP = None              # 78z skip7 (luat Tuan 27/10; SPEC B.2aa): DAT o cuoi C0 = rs_J_bert_dong_ep{EP_BB}.pt (EP_BB tu chon, truoc khi tai backbone)
EP_BB, EP_BB_TAY = None, None   # nhanh C (Tuan 29/10 "epoch cuoi truoc khi bi overfitting"): EP_BB = dinh VAL EMA trong luot, bang nhau -> epoch MUON hon; dat o cuoi C0
SEEDS = (int(CAU_HINH_C['seed']),)   # nhanh C: dau doc CUNG seed voi backbone
print(f'78y nhanh C s{SEEDS[0]}: backbone {TM_BB}/{BB_RUN} (EP_BB = dinh VAL EMA trong luot, cuoi C0) | SEEDS {SEEDS}')
SEED, SIZE, EPOCHS = int(CAU_HINH_C['seed']), 518, 10
BATCH, ACCUM = 256, 1                     # 78t: nhu 78g (EFF 256; MOT micro-batch / buoc -> ung vien cau CHUNG lo)
assert BATCH * ACCUM == 256, (BATCH, ACCUM)
LR, WD, WARMUP, CLIP = 1e-3, 1e-2, 0.05, 5.0
NW = min(8, (os.cpu_count() or 2)); PF_78M = 2   # 78m: RAM DataLoader = BATCH x 3.2 MB x NW x PF ~ 13 GB o batch 256
NW_EVAL = min(4, NW)  # 78m (ra soat 02/10): predict / chan doan doc anh BANG worker (75a: nw=0 -> ~1/3 thoi gian epoch)
CAN_PHIM = False   # 78m: KHONG arm nao doc phim truoc -> DataLoader KHONG tai (tra tensor 1 phan tu), C6 KHONG chay
                   #      backbone tren no
D_MLD, D_ETOK, EHR_CLIP = 512, 256, 10.0
DO_TEST = False     # Tuan 27/10: test chay SAU - C6 KHONG du doan test moi epoch; pte_gopA (mo hinh don) van ghi o Phan 3 C6S

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
assert dev == 'cuda', 'KHONG thay GPU'
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f'thiet bi: {torch.cuda.get_device_name(0)} | '
      f'VRAM {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB | AMP {AMP}')

BB_SRC = BB_CFG_SRC = None   # 78y: rs_ ep4 (+ backbone_ft.pt cua J: cau hinh luong) cua 78x tai QUA API o CUOI C0 -> duong dan local
P_MICRO = os.path.join(OUT_DIR, 'cxrlt_micro_full.parquet')
P_VOCAB = os.path.join(OUT_DIR, 'cxrlt_micro_vocab.json')
for p, hd in ():   # 78z J: C1 KHONG nap EHR / tui vi khuan / XGB -> khong can kiem tep cua 29c / 52a
    assert os.path.exists(p), f'khong thay {p} -> chay {hd} truoc'

INV = json.load(open(os.path.join(OUT_DIR, 'cxrlt_inventory.json')))
LAB40 = INV['labels_task1']; C = len(LAB40)
TIER = {t: np.array([LAB40.index(l) for l in INV['tier'][t]]) for t in ('head','medium','tail')}
ORDER = pd.read_parquet(os.path.join(EMB_DIR, 'emb_index.parquet'))
Y = (pd.read_parquet(os.path.join(OUT_DIR, 'cxrlt_index.parquet'), columns=['dicom_id'] + LAB40)
       .set_index('dicom_id').loc[ORDER.dicom_id].reset_index())
assert (Y.dicom_id.to_numpy() == ORDER.dicom_id.to_numpy()).all()
YM = Y[LAB40].to_numpy('float32')
SPLIT = ORDER.split.to_numpy(); SUB = ORDER.subject_id.to_numpy()
TR, VA = SPLIT == 'train', SPLIT == 'val'; TE = ~(TR | VA)
_rs = np.random.default_rng(SEED); _su = np.unique(SUB[TR])
_sel = _rs.choice(_su, size=int(round(0.15*len(_su))), replace=False)
SL = TR & np.isin(SUB, _sel)          # van tinh, CHI de bao cao co bao nhieu anh duoc thu hoi
TF_ = TR.copy()                        # <<< R2a: GOP lat chon vao train
print(f'R2a: thu hoi {int(SL.sum()):,} anh lat chon vao train '
      f'({int((TR & ~SL).sum()):,} -> {int(TF_.sum()):,}, +{SL.sum()/max((TR&~SL).sum(),1):.1%})')
print('     LAT CHON GIO NAM TRONG TRAIN -> khong duoc chon bat cu thu gi o luot nay.')
# R2a: SL GIO NAM TRONG TF_ nen giao cua chung DUNG BANG SL — kiem THUAN chieu do,
# thay vi kiem roi nhau nhu cac luot truoc.
assert int((TF_ & SL).sum()) == int(SL.sum()), 'lat chon CHUA duoc gop het vao train'
assert int(TF_.sum()) == int(TR.sum()), 'TF_ phai bang DUNG TR o luot R2a'
assert not (set(SUB[TR]) & set(SUB[VA])), 'benh nhan train ro ri sang VAL'
assert not (set(SUB[TR]) & set(SUB[TE])), 'benh nhan train ro ri sang TEST'
I_TF, I_SL, I_VA, I_TE = (np.where(m)[0] for m in (TF_, SL, VA, TE))
print(f'{len(ORDER):,} anh | khop {len(I_TF):,} | chon {len(I_SL):,} | val {len(I_VA):,} | '
      f'test {len(I_TE):,}')
print(f'mo bang block {FT_BLOCKS} (LR/{LR_FT_DIV:.0f}) | E={EPOCHS} | '
      f'BATCH {BATCH}x{ACCUM}={BATCH*ACCUM} | seeds {SEEDS}')

# ---- DOC DRIVE (khoi chuan, CHI doc) — lay xgb_oof_co_test.npy DA GHIM (74a C+0, 25/09) ----
GOC_ID_TAY = os.environ.get('CXRLT_GOC_ID', '')   # nhanh C: ID cxrlt_2024 do o tai goi cua notebook dat (KHONG de trong ma cong khai); rong -> khoi tim theo duong MIMIC_CXR/planB_out/cxrlt_2024
DOC_LOC = '/content/doc78z'
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

# 78z J: KHONG tai xgb_oof_co_test.npy (luat chi-anh 15/10, docs/SPEC_SAU_78Y.md B.0)

# ---- GHI DRIVE (khoi chuan) — ket qua cua 75a vao THU MUC CUA NO ----
DRV_DUONG = (CAU_HINH_C['tm'], RUN_TAG)   # nhanh C: stage4_nb84
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
NGUONG_CHO = 32 << 20   # 78z J: tep dang cho <= 32 MB nam trong tep resume; w_ (~40 MB) tai len MOT lan
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
# ---- 78t: dung luong Drive cua MOT luot du 10 epoch (w_ + rs_ moi epoch giu het, + p_ / pte_ moi arm): decoder uoc tu 78m luot A
# (8 arm ~4.7 GB) x so arm / 8 (= 78g x so arm / 11); thap BiomedBERT hoc cung (cau_clip): tham so MO (~43 trieu, fp32) trong w_ +
# 2 moment AdamW trong rs_ ~0.52 GB / epoch. CHUA do; C4 tinh lai tu so tham so. Dong '~2 GB' cua khoi ghi o tren la so chung.
def uoc_vram_78t(n_arm, n_vung, n_k4, co_thap, B=BATCH, U=1000, L=64, n_mo=6):
    '''Uoc VRAM dinh (GB) cua MOT buoc train 78t - CHUA do; THU_NHANH do that (C6 in VRAM dinh + so ung vien cau U / buoc).
       -> (dinh voi backward RIENG tung arm = cach C6 chay (D-T1, quyet dinh 07/10), chi tiet, dinh neu MOT backward tren TONG
       moi arm nhu 78g). Hang so tu hai lan DO voi MOT backward tren tong (Event: 78m luot A 8 arm 36 GB; 78g 11 arm, 4 arm vung,
       54 GB; B 256, 1369 token, d 512, decoder FP32): chung (dac trung dong cung [B, 1369, 768] fp32 + forward backbone bf16 +
       trong so) ~2.0 GB; do thi MOT arm ~4.25 GB (kv = pin(patch), K / V, attention 8 head + dropout); arm vung +1.3 GB (attention
       tinh lai truoc dropout - GIA DINH, hai lan do khong tach duoc); arm k4 (160 query) +4.0 GB; thap BiomedBERT hoc cung (bf16):
       U cau x L token x n_mo tang mo x ~32 KB / token / tang (~12 GB o U 1000, L 64). Tham so + gradient + 2 moment AdamW ~0.04 GB
       / decoder, thap ~1 GB (110 trieu fp32 + 43 trieu tham so mo x 12 byte). Backward RIENG: CHI do thi cua MOT arm song mot luc
       -> dinh ~ chung + tham so + bo dem khi lui (~2 GB: grad K / V / attention) + do thi LON NHAT. Tuyen tinh theo B.'''
    s = B / 256.0
    thap = U * L * n_mo * 32e3 / 1e9 if co_thap else 0.0
    gb = dict(chung=2.0 * s, tham_so=0.04 * n_arm + (1.0 if co_thap else 0.0), lui=2.0 * s,
              do_thi_max=4.25 * s + max(1.3 * s if n_vung else 0.0, 4.0 * s if n_k4 else 0.0, thap))
    tong = 2.0 * s + 4.25 * s * n_arm + 1.3 * s * n_vung + 4.0 * s * n_k4 + (thap + 1.0 if co_thap else 0.0)
    return float(sum(gb.values())), gb, float(tong)


def uoc_phut_78t(n_arm, n_k4, co_thap, n_anh=None):
    '''Uoc thoi gian (phut) MOT epoch 78t - CHUA do. Hai lan DO (Event): 78m luot A 8 arm 17.3 ph train, 78g 11 arm 21.5 ph train (+ 7
       ph du doan val / test + chan doan) tren 258.871 anh -> ~6.1 ph chung (backbone) + ~1.4 ph / arm; arm k4 ~ 1 arm nua; thap hoc
       cung ~5 ph (PLAN_SAU_78G 8); du doan ~7 ph x so arm / 11 (toi thieu 4). Tuyen tinh theo so anh train.'''
    k = (n_anh or 258871) / 258871.0
    tr = k * (6.1 + 1.4 * (n_arm + n_k4) + (5.0 if co_thap else 0.0)); du = max(4.0, 7.0 * n_arm / 11)
    return tr, du


_A0 = list(TEN_ARM_78T) if ARMS_GIU is None else list(ARMS_GIU)
_GB_78T = 4.7 * len(_A0) / 8 + (0.52 * EPOCHS if 'cau_clip' in _A0 else 0.0)
VRAM_UOC, _VG0, _VT0 = uoc_vram_78t(len(_A0), sum(t in ('txtq_vung', 'vung_kdp') for t in _A0), int('txtq_k4' in _A0),
                                    'cau_clip' in _A0)
_PH0 = uoc_phut_78t(len(_A0), int('txtq_k4' in _A0), 'cau_clip' in _A0, len(I_TF))
print(f'78z J ({RUN_TAG}): {len(_A0)} arm | ghi ~{_GB_78T:.1f} GB len Drive (du {EPOCHS} epoch)'
      + (f' | ⚠ quota chi con {_QUOTA_CON:.1f} GB' if _QUOTA_CON is not None and _QUOTA_CON < _GB_78T + 5 else ''))
print(f'78y: VRAM UOC ~{VRAM_UOC:.0f} GB / buoc (B {BATCH}; backward RIENG tung arm - D-T1; MOT backward tren tong nhu 78g ~'
      f'{_VT0:.0f} GB; theo ten arm, C4 tinh lai): ' + ', '.join(f'{k} {v:.1f}' for k, v in _VG0.items())
      + (' -> > 90 GB: CO THE phai TACH notebook (ARMS_GIU + LUOT_78T + LAM_TU; lambda chung: CHI_LAM); THU_NHANH do that'
         if VRAM_UOC > 90 else ' | THU_NHANH do that'))
print(f'78y: thoi gian UOC ~{_PH0[0]:.0f} ph train + ~{_PH0[1]:.0f} ph du doan / epoch -> ~{EPOCHS * sum(_PH0) / 60:.1f} gio / '
      f'{EPOCHS} epoch (CHUA do; THU_NHANH do img/s that)')
# ---- 78y: checkpoint 78x (CHI DOC, QUA API; ban MOI NHAT neu trung ten) -> BB_SRC (rs_ ep4: trong so) + BB_CFG_SRC (J: backbone_ft.pt
#      CUNG luot, chua cau hinh kien truc luong cfg_luong / cfg_jv2 / cfg_jv3 + hop_tb) — C2 / C2b nap + kiem ----
# ---- nhanh C (Tuan 29/10; luat 27/10 "doc dong cung o DINH VAL trong luot cua CHINH luot"; SPEC B.2ab; kiem dau vao nhu Codex muc 54): EP_BB =
#      epoch co VAL EMA trong luot (dau J, muc anh, toan val; nhat ky Phan 1) CAO NHAT SAU KHI LAM TRON 4 chu so; bang nhau -> epoch MUON hon
#      (epoch cuoi truoc khi sut); moi gia tri huu han trong [0, 1]; EP_BB_TAY: None hoac so nguyen 1-6 -> ghi de. Nguon chon -> CHON_EP_C ->
#      nguon_backbone.json (Phan 3 doi chieu) ----
_kk_ep = kiem_ke((TM_BB, BB_RUN), mau=r'epoch_log_J_bert_dong_ep\d+\.json|rs_J_bert_dong_ep\d+\.pt')[1]
_lg_ep_ds = _kk_ep.get('epoch_log_J_bert_dong_ep6.json') or []
assert _lg_ep_ds, f'thieu epoch_log_J_bert_dong_ep6.json trong cxrlt_2024/{TM_BB}/{BB_RUN} -> Phan 1 chua xong / chua dong_bo -> DUNG'
with open(tai(_lg_ep_ds[0]), encoding='utf-8') as _f_ep: _lg_ep = json.load(_f_ep)
_tho_ep = [(_lg_ep.get(str(e)) or {}).get('mAP_anh') for e in range(6)]
assert all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0.0 <= v <= 1.0 for v in _tho_ep), (
    f'nhat ky Phan 1: VAL EMA ep1-6 thieu epoch / khong huu han / ngoai [0, 1] ({_tho_ep}) -> DUNG')
assert EP_BB_TAY is None or (type(EP_BB_TAY) is int and 1 <= EP_BB_TAY <= 6), f'EP_BB_TAY = {EP_BB_TAY!r}: phai la None hoac so nguyen 1-6 -> DUNG'
VAL_EMA_P1 = [round(float(v), 4) for v in _tho_ep]
EP_BB_DINH = len(VAL_EMA_P1) - VAL_EMA_P1[::-1].index(max(VAL_EMA_P1))   # lan CUOI cua max: bang nhau -> epoch MUON hon (Tuan 29/10)
EP_BB = EP_BB_DINH if EP_BB_TAY is None else EP_BB_TAY
BB_TEP = f'rs_J_bert_dong_ep{EP_BB}.pt'
assert _kk_ep.get(BB_TEP), f'thieu {BB_TEP} trong cxrlt_2024/{TM_BB}/{BB_RUN} (rs_ nhe cua ep{EP_BB}) -> DUNG'
CHON_EP_C = dict(nhat_ky=dict(ten='epoch_log_J_bert_dong_ep6.json', id=_lg_ep_ds[0].get('id'), md5=_lg_ep_ds[0].get('md5')),
                 val_ema_tho=[float(v) for v in _tho_ep], val_ema_lam_tron=VAL_EMA_P1, dinh=EP_BB_DINH, tay=EP_BB_TAY, ep_bb=EP_BB,
                 quy_tac=('TAY qua EP_BB_TAY' if EP_BB_TAY is not None else
                          'dinh VAL EMA trong luot (dau J, muc anh, toan val), lam tron 4 chu so, bang nhau -> epoch MUON hon (Tuan 29/10)'))
print(f"78z nhanh C s{SEEDS[0]}: EP_BB = {EP_BB} ({'TAY (EP_BB_TAY)' if EP_BB_TAY is not None else 'dinh VAL EMA trong luot, bang nhau -> muon hon'}; dinh "
      f"= ep{EP_BB_DINH}) | VAL EMA Phan 1 ep1-6 " + ' '.join(f'{v:.4f}' for v in VAL_EMA_P1) + ' | tho ' + ' '.join(f'{float(v):.6f}' for v in _tho_ep)
      + f" | nhat ky md5 {CHON_EP_C['nhat_ky']['md5']}")
del _kk_ep, _lg_ep_ds, _lg_ep, _tho_ep
_ik_bb, _kk_bb = kiem_ke((TM_BB, BB_RUN), mau=r'rs_[A-Za-z0-9_]+_ep\d+\.pt|backbone_ft\.pt')
_ds_bb = _kk_bb.get(BB_TEP) or []
assert _ds_bb, (f'thieu {BB_TEP} trong cxrlt_2024/{TM_BB}/{BB_RUN} -> 78x luot nay chua co rs_ ep{EP_BB} (GIU MOI rs_, C9 cua 78x); '
                'KHONG doi LUOT_78Y sang luot khac')
if xung_dot(_ds_bb): print(f'  !! {BB_TEP} ({TM_BB}/{BB_RUN}): {len(_ds_bb)} ban KHAC noi dung -> dung ban MOI NHAT ({_ds_bb[0]["mt"]})')
BB_SRC = tai(_ds_bb[0])
_ds_cf = (_kk_bb.get('backbone_ft.pt') or []) if LUOT_78Y == 'J_v3' else []
assert LUOT_78Y != 'J_v3' or _ds_cf, f'thieu backbone_ft.pt (cau hinh luong J) trong cxrlt_2024/{TM_BB}/{BB_RUN} -> DUNG'
BB_CFG_SRC = tai(_ds_cf[0]) if _ds_cf else None
NGUON_BB = dict(duong=[TM_BB, BB_RUN], tep=BB_TEP, ep=EP_BB, luot=LUOT_78Y, id=_ds_bb[0]['id'], md5=_ds_bb[0]['md5'],
                so_ban=len(_ds_bb), cau_hinh=(dict(tep='backbone_ft.pt', id=_ds_cf[0]['id'], md5=_ds_cf[0]['md5']) if _ds_cf else None))
NGUON_BB['chon_ep'] = CHON_EP_C   # nhanh C: cach chon epoch vao nguon_backbone.json (Phan 3 doi chieu)
# CUNG luot 78y phai CUNG backbone: lan dau ghi nguon_backbone.json; phien sau (resume) so md5 -> khac -> DUNG (KHONG tron hai backbone)
_p_nbb = tim_moi_noi('nguon_backbone.json')
if _p_nbb:
    with open(_p_nbb) as _f_nbb: _nbb_cu = json.load(_f_nbb)
    assert _nbb_cu.get('md5') == NGUON_BB['md5'] and _nbb_cu.get('cau_hinh') == NGUON_BB['cau_hinh'], (
        f"luot {RUN_TAG} da chay voi backbone md5 {_nbb_cu.get('md5')} KHAC ban moi nhat {NGUON_BB['md5']} (78x {BB_RUN} chay lai?) "
        '-> doi RUN_TAG (luot moi), KHONG tron hai backbone')
else:
    ghi('nguon_backbone.json', lambda q: ghi_json(q, NGUON_BB))
print(f"78y: backbone {TM_BB}/{BB_RUN}/{BB_TEP} -> {BB_SRC} (md5 {NGUON_BB['md5']}, {len(_ds_bb)} ban)"
      + (f" | cau hinh luong: backbone_ft.pt (md5 {_ds_cf[0]['md5']})" if _ds_cf else '')
      + (' | khop nguon_backbone.json cua luot' if _p_nbb else ' | ghi nguon_backbone.json (lan dau)'))
del _ik_bb, _kk_bb, _ds_bb, _ds_cf, _p_nbb
# ---- nhanh C Phan 3: backbone + cach chon epoch PHAI TRUNG nguon Phan 2 (nguon_backbone.json cua stage4_nb84/r84z_C_s<seed>) - KHONG gop
#      trong so dau doc da train tren backbone / epoch khac ----
_du_n2 = (CAU_HINH_C['tm'], CAU_HINH_C['run_z'])
_kk_n2 = kiem_ke(_du_n2, mau=r'nguon_backbone\.json')[1].get('nguon_backbone.json') or []
assert _kk_n2, f"thieu nguon_backbone.json cua Phan 2 ({'/'.join(_du_n2)}) -> Phan 2 chua chay / chua dong_bo -> DUNG"
with open(tai(_kk_n2[0]), encoding='utf-8') as _f_n2: _n2 = json.load(_f_n2)
_lech_n2 = {k: (_n2.get(k), NGUON_BB.get(k)) for k in ('duong', 'tep', 'ep', 'md5', 'cau_hinh', 'chon_ep') if _n2.get(k) != NGUON_BB.get(k)}
assert not _lech_n2, f'Phan 3: backbone / cach chon epoch KHAC Phan 2 {_lech_n2} -> DUNG (KHONG gop trong so dau doc train tren backbone khac)'
print(f"78z nhanh C s{SEEDS[0]} Phan 3: backbone = nguon Phan 2 (ep{EP_BB}, md5 {NGUON_BB['md5']}, cach chon epoch trung)")
del _du_n2, _kk_n2, _f_n2, _n2, _lech_n2
# ---- 78y view LUC CHAM (Tuan 15/10, buoc 1.0; docs/SPEC_SAU_78Y.md B.1c): co neo cua anh VAL / TEST lay tu tep view luc cham (CSV
#      ViewPosition -> 28e doc anh -> unknown; KHONG BigQuery) — CHI DOC qua API, ghim md5 (C2b dung). Anh train giu tep cu. ----
MD5_VIEW_CHAM = '847724e6bfceda49053eab1bbe303c54'
NEO_DOI_DAT_TRUOC = {'val': 37, 'test': 114}   # so anh doi co neo (buoc 1.0) — C2b kiem
_kk_vc = kiem_ke(('stage4_nb78a', 'nguon_view_cham'), mau=r'cxrlt_view_cham\.parquet')[1].get('cxrlt_view_cham.parquet') or []
assert _kk_vc, 'thieu stage4_nb78a/nguon_view_cham/cxrlt_view_cham.parquet -> chay 78a_nguon_view_cham.ipynb (buoc 1.0) truoc. DUNG'
_ds_vc = [d for d in _kk_vc if d['md5'] == MD5_VIEW_CHAM]
assert _ds_vc, (f"tep view luc cham tren Drive KHAC ban ghim (md5 {[d['md5'] for d in _kk_vc]} != {MD5_VIEW_CHAM}; 1.0 chay lai?) "
                '-> DUNG, bao Claude')
VIEW_CHAM_SRC = tai(_ds_vc[0])
NGUON_VIEW_CHAM = dict(duong=['stage4_nb78a', 'nguon_view_cham'], tep='cxrlt_view_cham.parquet', id=_ds_vc[0]['id'], md5=_ds_vc[0]['md5'])
print(f"78y: tep view luc cham -> {VIEW_CHAM_SRC} (md5 {MD5_VIEW_CHAM}, khop ban ghim; {len(_kk_vc)} ban tren Drive)")
del _kk_vc, _ds_vc

# ---- 78z: cau hinh fusion (NGUYEN VAN experiments/78_chi_anh/cau_hinh/fusion_study.json) ----
CFG_FS = json.loads(r'''{
  "phien_ban": "fs_v1",
  "_ghi_chu": "Giai doan 2.1 fusion study (docs/SPEC_SAU_78Y.md B.2b; Tuan mo Giai doan 2 16/10; luat Event 2026-10-16). Hai dau fusion tren ML-Decoder khoi tao nhu mldec: fus_tb (TB dac trung cac anh cung study + nhung loai view), fus_attn (attention giua cac anh cua study, rieng tung lop). Tham so moi khoi tao 0. Bo view LUC TRAIN voi xac suat p_bo_view (moi study giu >= 1 anh), mat na tu generator rieng (hat SEED * 1000 + hat_gen_fus). View tu tep view luc cham (CSV + 28e) cho CA train va cham. Lo du doan gom tron study <= lo_du_doan anh.",
  "arms": {"fus_tb": "tb", "fus_attn": "attn"},
  "view_ma": {"frontal": 0, "lateral": 1},
  "n_view": 3,
  "p_bo_view": 0.2,
  "lo_du_doan": 32,
  "hat_gen_fus": 31,
  "luat": {"dau_quyet": "fus_attn", "dau_bao_kem": "fus_tb", "doi_chung": "mldec", "nguong": 0.003}
}''')
# ---- khoi 78x (NGUYEN VAN experiments/78_chi_anh/khoi/khoi_lo_study.py) ----


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
# ---- khoi 78z (NGUYEN VAN experiments/78_chi_anh/khoi/khoi_fusion_study.py) ----


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
print(f"78z: fusion {CFG_FS['phien_ban']} | dau {CFG_FS['arms']} | bo view luc train p {CFG_FS['p_bo_view']} | lo du doan <= "
      f"{CFG_FS['lo_du_doan']} anh (gom tron study)")

# ---- nhanh C (Tuan duyet 29/10; SPEC B.2ab): cau hinh o MONG DOI = O_CFG Phan 1 nhanh C (= luot goi o cat); 4 dau chuan (doc 1 view, duong skip) ----
O_CFG_MONG = {'ten': 'o_luoi_phong518_skip7_gop6', 'phien_ban': 'o_j_v1', 'n_o': 5, 'k_skip': 8, 'r_adapter': 64, 'khuc': 240, 'hat': {'bo_o7': 61, 'bo_o11': 62}, 'sang_loc': {'moc_run': ['stage4_nb78x', 'r78x_J_v3b_dp_ema_v2_s42'], 'nguong': 0.003}}
assert ARMS_GIU == ['mldec', 'mldec_hr', 'fus_tb_hr', 'fus_attn_hr'] and O_CFG_MONG['ten'] == 'o_luoi_phong518_skip7_gop6' \
    and int(O_CFG_MONG['n_o']) == 5 and int(O_CFG_MONG['k_skip']) == 8, 'nhanh C 78z: cau hinh lech -> DUNG'
print(f"78z nhanh C s{SEEDS[0]}: backbone {TM_BB}/{BB_RUN} co skip block {int(O_CFG_MONG['k_skip']) - 1} -> {O_CFG_MONG['k_skip']} cua "
      f"{O_CFG_MONG['n_o']} o (dong cung) | 4 dau chuan | backbone ep{EP_BB} ({'TAY' if EP_BB_TAY is not None else 'dinh VAL trong luot'})")
