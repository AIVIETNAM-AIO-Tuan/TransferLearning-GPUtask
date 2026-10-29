# ==== nhanh_c buoc z_C2 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModel, AutoImageProcessor
from torchvision import transforms as T
import shutil

os.makedirs(LOCAL, exist_ok=True)
_n = len([f for f in os.listdir(LOCAL) if f.endswith('.jpg')])
# Uu tien KHO da resize san (41c C3-bis tao): copy + bung, vai phut. Neu chua co thi stage tu shard
# goc 1022px — cham hon ~10 lan vi phai giai ma + resize 377k anh.
if _n < len(ORDER) * 0.999 and os.path.isdir(READY_X):
    for f in sorted(x for x in os.listdir(READY_X) if x.endswith('.tar')):
        src = os.path.join(READY_X, f); tmp = os.path.join('/content', f)
        try:
            shutil.copy(src, tmp)                        # copy TRUOC — khong mo tar tren mount
            assert os.path.getsize(tmp) == os.path.getsize(src), 'copy NGAN tu Drive'
            with tarfile.open(tmp) as tar: tar.extractall(LOCAL, filter='data')
        except Exception as e:
            print(f'  {f} HONG ({type(e).__name__}) -> bo qua')
        if os.path.exists(tmp): os.remove(tmp)
    _n = len([f for f in os.listdir(LOCAL) if f.endswith('.jpg')])
    print(f'hydrate tu kho: {_n:,} / {len(ORDER):,}')
if _n < len(ORDER) * 0.999:
    SHARD_DIR = os.path.join(DATA_DIR, 'cxr_1022_letterbox_cxrlt')
    assert os.path.isdir(SHARD_DIR), f'khong thay CA {READY_X} lan {SHARD_DIR}'
    TARS = sorted(os.path.join(SHARD_DIR, f) for f in os.listdir(SHARD_DIR)
                  if f.endswith('.tar') and f.startswith('lb_'))
    DONE = os.path.join(LOCAL, '_done.json')
    done = set(json.load(open(DONE))) if os.path.exists(DONE) else set()
    def _one(pl):
        nm, blob = pl; dst = os.path.join(LOCAL, os.path.basename(nm))
        if os.path.exists(dst): return
        try:
            im = Image.open(_io.BytesIO(blob)).convert('L').resize((SIZE, SIZE), Image.BICUBIC)
        except Exception:
            return                                       # mot anh hong KHONG duoc giet ca lan chay
        tmp = dst + '.tmp'; im.save(tmp, 'JPEG', quality=95); os.replace(tmp, dst)   # NGUYEN TU
    with ThreadPoolExecutor(8) as pool:
        for tp in tqdm([t for t in TARS if os.path.basename(t) not in done], desc='stage', unit='tar'):
            with open(tp, 'rb') as f: buf = f.read()     # doc BULK; lap tren mount FUSE se TREO
            assert len(buf) == os.path.getsize(tp), f'{os.path.basename(tp)}: doc NGAN tren mount'
            with tarfile.open(fileobj=_io.BytesIO(buf)) as tar:
                jobs = [(m.name, tar.extractfile(m).read())
                        for m in tar.getmembers() if m.name.endswith('.jpg')]
            list(pool.map(_one, jobs)); del buf, jobs
            done.add(os.path.basename(tp)); json.dump(sorted(done), open(DONE, 'w'))
    _n = len([f for f in os.listdir(LOCAL) if f.endswith('.jpg')])
assert _n >= len(ORDER) * 0.999, f'thieu {len(ORDER)-_n:,} anh'
print(f'anh local: {_n:,} / {len(ORDER):,}')

proc = AutoImageProcessor.from_pretrained('microsoft/rad-dino')
MEAN, STD = proc.image_mean[0], proc.image_std[0]
train_tf = T.Compose([T.RandomResizedCrop(SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05),
                                          interpolation=T.InterpolationMode.BICUBIC),
                      T.RandomRotation(7), T.ToTensor(), T.Normalize([MEAN]*3, [STD]*3)])
eval_tf = T.Compose([T.ToTensor(), T.Normalize([MEAN]*3, [STD]*3)])
# 78g: CUNG phep bien doi hinh hoc voi train_tf (RandomResizedCrop + RandomRotation, cung tham so) nhung qua
# torchvision.transforms.v2: MOT lan rut tham so cho CA anh lan hop vung ImaGenome (arm vung, khoi_hop_anh). Anh roi qua
# eval_tf (ToTensor + Normalize) -> CUNG dang tensor voi train_tf. `train_tf` giu lam CO: DS thay s.tf is train_tf -> di
# duong anh + hop. Hop ngoai anh sau xoay -> ClampBoundingBoxes; hop suy bien (< 1 px) -> coi nhu khong co.
from torchvision.transforms import v2 as T2
TF_HOP = T2.Compose([T2.RandomResizedCrop(SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05),
                                          interpolation=T2.InterpolationMode.BICUBIC),
                     T2.RandomRotation(7), T2.ClampBoundingBoxes()])

_ck = torch.load(BB_SRC, map_location='cpu', weights_only=False)
# 78y: cong checkpoint 78x (rs_<luot>_ep4.pt: bb, head, ..., run, luot, ep = epoch - 1; J: + luong) - sai luot / run / epoch -> DUNG
assert isinstance(_ck, dict) and 'bb' in _ck and _ck.get('run') == BB_RUN and _ck.get('luot') == 'B1' and _ck.get('ep') == EP_BB - 1 \
    and (LUOT_78Y != 'J_v3' or 'luong' in _ck), (
    f"{BB_TEP} cua {TM_BB}/{BB_RUN}: run / luot / epoch KHAC ({_ck.get('run')}, {_ck.get('luot')}, ep {_ck.get('ep')}) hoac thieu "
    'luong -> DUNG (KHONG nap backbone sai)')
_v_bb = ((_ck.get('log') or {}).get(str(EP_BB - 1)) or {}).get('mAP_anh')
print(f"78y: checkpoint 78x {BB_RUN}/{BB_TEP}: luot {_ck['luot']} | ep {_ck['ep'] + 1} | val ep{EP_BB} (dau doc trong luot, mAP anh) "
      f"{_v_bb} | transformers luc luu {_ck.get('tf_ver')}")
_sd0 = {k: v.clone() for k, v in AutoModel.from_pretrained('microsoft/rad-dino').state_dict().items()}
backbone = AutoModel.from_pretrained('microsoft/rad-dino')
# 78g (Colab 06/10, Python 3.13): transformers >= 5.18 DOI TEN lop attention cua Dinov2 — attention.attention.query / key /
# value -> attention.q_proj / k_proj / v_proj, attention.output.dense -> attention.o_proj (bang doi ten `Dinov2Model` trong
# conversion_mapping.py cua chinh transformers). from_pretrained tu doi; torch.load + load_state_dict thi KHONG. Checkpoint 38a
# luu theo ten CU -> doi ten CHI khi mo hinh dang chay dung ten MOI; van strict=True. Cong: tep safetensors GOC cua RAD-DINO
# tren hub (ten cu) doi ten bang CUNG ham phai TRUNG TUNG BIT ban from_pretrained tu doi (_sd0) -> bang doi ten khop transformers.
from transformers import __version__ as _TF_VER
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as _doc_st
_DOI_TEN_DINOV2 = (('.attention.attention.query.', '.attention.q_proj.'), ('.attention.attention.key.', '.attention.k_proj.'),
                   ('.attention.attention.value.', '.attention.v_proj.'), ('.attention.output.dense.', '.attention.o_proj.'))


def doi_ten_dinov2(sd, khoa_mo_hinh):
    '''state_dict Dinov2 -> ten khoa cua mo hinh dang chay: da khop -> giu nguyen; khong -> doi ten CU sang MOI (chi doi ten).'''
    if set(sd) <= set(khoa_mo_hinh): return dict(sd)
    out = {}
    for k, v in sd.items():
        for a, b in _DOI_TEN_DINOV2: k = k.replace(a, b)
        out[k] = v
    return out


_hub = doi_ten_dinov2(_doc_st(hf_hub_download('microsoft/rad-dino', 'model.safetensors')), _sd0)
assert set(_hub) == set(_sd0) and all(torch.equal(_hub[k].to(_sd0[k].dtype), _sd0[k]) for k in _sd0), (
    f'transformers {_TF_VER}: doi ten Dinov2 KHAC cach from_pretrained tu doi (tep hub RAD-DINO) -> DUNG')
_doi_bb = not set(_ck['bb']) <= set(_sd0)
backbone.load_state_dict(doi_ten_dinov2(_ck['bb'], _sd0), strict=True)
del _hub
print(f'transformers {_TF_VER}: checkpoint 38a ' + ('theo ten CU -> doi ten attention sang ten moi (cach doi khop tep hub)'
                                                    if _doi_bb else 'khop ten mo hinh, khong doi ten'))
with torch.no_grad():
    _d = max((_sd0[k] - v).abs().max().item() for k, v in backbone.state_dict().items())
del _sd0
assert _d > 1e-6, f'backbone GIONG HET pretrain (lech {_d:.1e}) -> `ft` khong duoc nap'
backbone = backbone.to(dev).eval()
for p in backbone.parameters(): p.requires_grad_(False)
D_CXR = backbone.config.hidden_size
LAYERS = backbone.encoder.layer; NL = len(LAYERS)
GRAD_FROM = min(FT_BLOCKS)
assert max(FT_BLOCKS) < NL and list(FT_BLOCKS) == list(range(GRAD_FROM, NL)), (
    f'FT_BLOCKS {FT_BLOCKS} phai la day LIEN TUC den block cuoi ({NL-1}) — neu khong thi phan '
    'dong cung o TREN chung khong dung chung duoc voi arm ft.')
_cfg = backbone.config
_dp = (getattr(_cfg, 'attention_probs_dropout_prob', 0), getattr(_cfg, 'hidden_dropout_prob', 0),
       getattr(_cfg, 'drop_path_rate', 0))
print(f'BACKBONE 78x/{BB_RUN}/{BB_TEP} (luot {LUOT_78Y}) | {NL} block | d={D_CXR} | lech pretrain {_d:.3f}')
print(f'dropout (attn, hidden, drop_path) = {_dp}'
      + ('  -> bang 0, che do train/eval khong anh huong' if max(_dp) == 0 else
         '  -> KHAC 0: ban sao PHAI chay .eval() de khop ban goc'))

def _out(r):
    '''Tensor TRAN, khong phai tuple: `r[0]` se lay MAU DAU cua batch va broadcasting che lap loi.'''
    return r[0] if isinstance(r, tuple) else r

class DS(Dataset):
    '''Tra ve CA HAI anh: hien tai va PHIM TRUOC (+ `has`). `ehr_perm` cho TRAO NHAN QUA.'''
    def __init__(s, idx, tf, ehr_perm=None): s.idx, s.tf, s.perm = np.asarray(idx), tf, ehr_perm
    def __len__(s): return len(s.idx)
    def _img(s, i):
        im = Image.open(os.path.join(LOCAL, f'{ORDER.dicom_id.iloc[i]}.jpg')).convert('L')
        return s.tf(Image.merge('RGB', (im, im, im)))
    def _img_hop(s, i):
        '''78g: anh train + hop 36 vung (C3c: HOP518, toa do anh 518) qua CUNG phep bien doi TF_HOP. Anh khong co scene
           graph (anh nghieng / ngoai ImaGenome) -> khong hop (co = False).'''
        im = Image.open(os.path.join(LOCAL, f'{ORDER.dicom_id.iloc[i]}.jpg')).convert('L')
        g = int(SG_HANG[i])
        hop = HOP518[g] if g >= 0 else np.full((R_VUNG, 4), np.nan, 'float32')
        im2, hop2, co2 = bien_doi_kem_hop(Image.merge('RGB', (im, im, im)), hop, TF_HOP)
        return eval_tf(im2), torch.from_numpy(np.nan_to_num(hop2)), torch.from_numpy(co2)
    def __getitem__(s, k):
        i = int(s.idx[k]); pr = int(PREV_ROW[i])
        if s.tf is train_tf: x, hop, co_hop = s._img_hop(i)          # 78g: anh + hop vung CUNG phep bien doi
        else: x = s._img(i); hop, co_hop = torch.zeros(1), torch.zeros(1, dtype=torch.bool)
        return dict(image=x, hop=hop, co_hop=co_hop, image_prev=((s._img(pr) if pr >= 0 else torch.zeros_like(x)) if CAN_PHIM
                                         else torch.zeros(1)),   # 78m (ra soat 02/10): khong phim truoc -> 1 phan tu,
                                                                 # khong 3.2 MB / anh trong hang doi DataLoader
                    has=torch.tensor(1.0 if pr >= 0 else 0.0), pos=k,
                    y=torch.from_numpy(YM[i]), row=i,
                    erow=(i if s.perm is None else int(s.perm[k])))

def loader(idx, tf, shuffle, bs, perm=None, nw=NW, drop_last=None):
    return DataLoader(DS(idx, tf, perm), batch_size=bs, shuffle=shuffle, num_workers=nw,
                      pin_memory=True, drop_last=shuffle if drop_last is None else drop_last,
                      persistent_workers=nw > 0, prefetch_factor=PF_78M if nw > 0 else None)   # 78m: PF_78M (C0)

def run_blocks(h, lo, hi, layers=None):
    L = LAYERS if layers is None else layers
    for i in range(lo, hi): h = _out(L[i - (0 if layers is None else lo)](h))
    return h

# --- cong 1: forward CAT DOI phai bang forward NGUYEN VEN ---
_b0 = next(iter(loader(I_VA[:4], eval_tf, False, 4, nw=0))); _x = _b0['image'].to(dev)
with torch.no_grad():
    _full = backbone(pixel_values=_x).last_hidden_state
    _split = backbone.layernorm(run_blocks(run_blocks(backbone.embeddings(_x), 0, GRAD_FROM),
                                           GRAD_FROM, NL))
_e1 = float((_full - _split).abs().max())
assert _e1 < 1e-3, (f'forward CAT DOI lech {_e1:.2e} -> da cat SAI cho trong kien truc HF. '
                    'DUNG chay tiep, moi so sau day vo nghia.')
N_TOK = _full.shape[1] - 1
assert N_TOK == (SIZE // backbone.config.patch_size) ** 2
print(f'cong 1: forward cat doi khop toi {_e1:.1e} | {N_TOK} patch token')
