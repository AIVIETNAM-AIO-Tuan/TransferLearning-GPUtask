# ==== nhanh_c buoc p1_C1 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C1 · ANH LOCAL 518 — kho resize san (41c C3-bis) hoac stage tu shard 1022 (NGUYEN VAN kho anh cua 78g C2 = 75a) ====
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

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
