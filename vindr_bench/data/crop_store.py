# -*- coding: utf-8 -*-
"""Kho cắt giữa của phép thử hình học (PLAN_89 §5.3; Tuấn 13/11 + 16/11): CÙNG chuỗi giải mã của kho 518 module 85, chỉ thay bước hình học.

    85 (kho đệm vuông)   DICOM -> u8 (trừ min / chia max / cắt / đảo MONOCHROME1) [-> cv2.equalizeHist] -> JPEG q95 cv2 -> letterbox 1022 (cạnh DÀI,
                         đệm viền 0) BICUBIC -> JPEG q90 -> 518 BICUBIC -> JPEG q95
    89k (kho cắt giữa)   ... -> JPEG q95 cv2 -> co cạnh NGẮN về 1022 + cắt giữa 1022 (= torchvision Resize(1022, BICUBIC) + CenterCrop(1022)) -> JPEG q90
                         -> 518 BICUBIC -> JPEG q95

Một lượt giải mã ghi CẢ hai bản: 'cg_kcb' (không cân bằng — kho của lượt) và 'cg_cb' (có cân bằng — dự phòng). Khoá / thứ tự / nhãn / tách KHÔNG tự tính:
chép từ kho đệm vuông không cân bằng cùng phần (85b / 85e `_khong_cb`) -> cổng trùng md5 (vindr.load_vindr đọc kho cắt giữa bằng đúng các cổng của kho cũ).

Hàm chép NGUYÊN VĂN từ module 85 (tests/test_crop_store.py so với nguồn trong experiments/85_vindr/dung_85.py, từng bit trên mảng thử):
    letterbox  = _letterbox (28c / 82b / 85, PAD_VALUE 0)            to_u8 = _u8_khong_cb_85 (kho thứ hai của 85, Tuấn 09/11)
    chain(.., 'dem_vuong') o to_u8 = chuyen_85 của kho không cân bằng; chain(cv2.equalizeHist(to_u8(..)), 'dem_vuong') = chuyen_85 của kho cân bằng
Tải DICOM: như 85 (_tai_85 / _lap_85 / _cua_so_85): dịch vụ API riêng MỖI luồng (httplib2 không an toàn đa luồng), thử lại 5 lần, md5 + số byte = Drive,
sha256 = SHA256SUMS.txt của bản PhysioNet trên Drive. Không in mã ảnh (luật PhysioNet): lỗi chỉ đếm theo loại."""
import collections
import hashlib
import io
import itertools
import os
import re
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

SIDE_MID, SIDE_OUT = 1022, 518                       # bước giữa (= shard 1022 của 28c / 85) và ảnh ra (= kho 518 của 85)
JPEG_U8, JPEG_MID, JPEG_OUT = 95, 90, 95             # chất lượng JPEG ba lần nén của 85
GEOMETRIES = ('dem_vuong', 'cat_giua')
VARIANTS = {'cg_kcb': False, 'cg_cb': True}          # kho -> có cv2.equalizeHist
CROP_STORE_KEY = 'cg_kcb'                            # bản của lượt (protocols/geometry.CROP_STORE); 'cg_cb' dự phòng
VD_PATH = ('MIMIC_CXR', 'VinDr')                     # bản PhysioNet v1.0.0 trên Drive (Event 02/11; SHA256SUMS khớp)
SUMS_NAME = 'SHA256SUMS.txt'
FOLDER_MIME = 'application/vnd.google-apps.folder'
DECODE_PACKAGES = ('pydicom', 'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg')   # = GOI_85 (~1/3 ảnh test nén JPEG 2000)
CACH = {'cg_kcb': ('DICOM pixel -> tru min / chia max / cat uint8 -> dao MONOCHROME1 -> KHONG equalizeHist -> JPEG q95 cv2 -> co canh NGAN 1022 + cat '
                   'giua 1022 BICUBIC (= torchvision Resize + CenterCrop) -> JPEG q90 -> 518 BICUBIC -> JPEG q95; KHONG cua so DICOM'),
        'cg_cb': ('DICOM pixel -> tru min / chia max / cat uint8 -> dao MONOCHROME1 -> cv2.equalizeHist -> JPEG q95 cv2 -> co canh NGAN 1022 + cat '
                  'giua 1022 BICUBIC (= torchvision Resize + CenterCrop) -> JPEG q90 -> 518 BICUBIC -> JPEG q95; KHONG cua so DICOM')}


# ---- chuỗi chuyển ảnh ----

def letterbox(im, size, pad=0):                                       # = _letterbox cua 28c NGUYEN VAN (PAD_VALUE 0)
    '''Scale theo canh DAI roi dem cho vuong. Giu nguyen ty le khung, khong cat gi.'''
    w, h = im.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = im.resize((nw, nh), Image.BICUBIC)
    out = Image.new(im.mode, (size, size), pad)
    out.paste(im, ((size - nw) // 2, (size - nh) // 2))     # can giua
    return out


def center_crop(im, size):
    '''Co cạnh NGẮN về `size` rồi cắt giữa size x size = torchvision Resize(size, BICUBIC) + CenterCrop(size) trên ảnh PIL (cùng cách làm tròn:
       cạnh dài int(size * dài / ngắn); lề int(round((cạnh - size) / 2))). Giữ tỉ lệ khung, bỏ phần thừa ở hai đầu cạnh dài.'''
    w, h = im.size
    nw, nh = (size, int(size * h / w)) if w <= h else (int(size * w / h), size)
    im = im.resize((nw, nh), Image.BICUBIC)
    left, top = int(round((nw - size) / 2.0)), int(round((nh - size) / 2.0))
    return im.crop((left, top, left + size, top + size))


def to_u8(a, mono1):
    '''= _u8_khong_cb_85 NGUYÊN VĂN: mảng pixel DICOM 2D -> uint8 như MIMIC-CXR-JPG BỎ cân bằng histogram: trừ min, chia max, cắt, uint8; đảo nếu
       MONOCHROME1. (_mimic_u8_85 của kho cân bằng = cv2.equalizeHist(to_u8(a, mono1)).)'''
    a = np.asarray(a, dtype='float64')
    assert a.ndim == 2, f'mang pixel {a.ndim} chieu'
    a = a - a.min()
    _mx = a.max()
    assert _mx > 0, 'anh mot mau'
    u = (a / _mx * 255.0).astype('uint8')                                  # astype = cat (truncate)
    if mono1: u = 255 - u
    return u


def chain(u8, geometry):
    '''uint8 2D (đã / chưa cân bằng) -> ảnh 'L' 518 x 518 (CHƯA nén JPEG q95 cuối — save_jpeg làm) = phần sau bước u8 của chuyen_85, hình học tuỳ chọn.'''
    import cv2
    assert geometry in GEOMETRIES, geometry
    ok, enc = cv2.imencode('.jpg', np.ascontiguousarray(u8), [cv2.IMWRITE_JPEG_QUALITY, JPEG_U8])   # = tep JPG cua MIMIC-CXR-JPG
    assert ok, 'cv2.imencode loi'
    im = Image.open(io.BytesIO(enc.tobytes())).convert('L')
    im = letterbox(im, SIDE_MID) if geometry == 'dem_vuong' else center_crop(im, SIDE_MID)
    b = io.BytesIO(); im.save(b, 'JPEG', quality=JPEG_MID); b.seek(0)    # = shard 1022 cua 28c (JPEG q90)
    return Image.open(b).convert('L').resize((SIDE_OUT, SIDE_OUT), Image.BICUBIC)   # = C1 cua 78x (518 BICUBIC)


def variants(a, mono1, geometry='cat_giua'):
    '''Mảng pixel -> {kho: ảnh 'L' 518} cho mọi bản của VARIANTS (u8 tính MỘT lần; bản cân bằng = cv2.equalizeHist của cùng u8 — như _mimic_u8_85).'''
    import cv2
    u = to_u8(a, mono1)
    return {k: chain(cv2.equalizeHist(u) if eq else u, geometry) for k, eq in VARIANTS.items()}


def save_jpeg(im, path):
    '''Ghi NGUYÊN TỬ JPEG q95 (= bước cuối của 85: im.save(tmp) rồi os.replace).'''
    tmp = path + '.tmp'
    im.save(tmp, 'JPEG', quality=JPEG_OUT)
    os.replace(tmp, path)


def reference(u8, geometry, size=SIDE_OUT):
    '''Ảnh đối chiếu ĐỘC LẬP với chuỗi (không bước 1022, không JPEG): u8 -> thẳng `size` theo hình học -> mảng uint8. Dùng cho cổng mẫu "= Resize +
       CenterCrop trên ảnh gốc" (cắt giữa: torchvision Resize + CenterCrop THẬT — không dùng center_crop của gói, để lỗi ở hàm đó không tự qua cổng; thiếu
       torchvision mới lùi về center_crop) và đối chứng âm (đệm vuông: letterbox).'''
    im = Image.fromarray(np.ascontiguousarray(u8)).convert('L')
    if geometry == 'dem_vuong':
        return np.asarray(letterbox(im, size))
    try:
        from torchvision import transforms as T
        from torchvision.transforms import InterpolationMode
    except ImportError:
        return np.asarray(center_crop(im, size))
    return np.asarray(T.CenterCrop(size)(T.Resize(size, interpolation=InterpolationMode.BICUBIC)(im)))


def similarity(a, b):
    '''Hai ảnh uint8 cùng hình -> dict(mad (lệch tuyệt đối TB, mức xám), corr (Pearson), max (lệch lớn nhất), trung (đúng từng điểm ảnh)).'''
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    d = np.abs(a - b)
    sa, sb = a.std(), b.std()
    corr = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else float('nan')
    return dict(mad=float(d.mean()), corr=corr, max=float(d.max()), trung=bool(d.max() == 0))


def center_minus_sides(img):
    '''= _giua_ben_85: (giữa: trung thất / tim / cột sống) − (hai bên: phổi), hàng 25–75 % của ảnh 518 — X-quang hiển thị chuẩn thường > 0.'''
    a = np.asarray(img, dtype='float32')[129:389]
    return float(a[:, 233:285].mean() - np.r_[a[:, 104:156].ravel(), a[:, 363:415].ravel()].mean())


# ---- nguồn: bản PhysioNet trên Drive ----

def key_of(rel):
    '''= _khoa_85: đường dẫn tương đối -> (tập, tên bỏ đuôi .dicom / .dcm); tệp ở gốc -> ('goc', tên).'''
    p = rel.split('/')
    if len(p) == 1: return ('goc', p[0])
    if p[0] in ('train', 'test'): return (p[0], re.sub(r'\.(dicom|dcm)$', '', p[-1], flags=re.I))
    return (p[0], p[-1])


def parse_sums(b):
    '''Nội dung SHA256SUMS.txt -> ({khoá: [sha256]}, số dòng không đọc được) (= vòng đọc SUMS của 85).'''
    sums, bad = collections.defaultdict(list), 0
    for line in b.decode('utf-8', 'replace').splitlines():
        m = re.fullmatch(r'\s*([0-9a-fA-F]{64})\s+\*?(.+?)\s*', line)
        if m: sums[key_of(re.sub(r'^\./', '', m.group(2)))].append(m.group(1).lower())
        elif line.strip(): bad += 1
    return dict(sums), bad


_TL = threading.local()


def _api():
    '''MỘT dịch vụ Drive API cho MỖI luồng (= _api_85; xác thực mặc định của Colab sau auth.authenticate_user() ở C0).'''
    if getattr(_TL, 'api', None) is None:
        from googleapiclient.discovery import build
        _TL.api = build('drive', 'v3', cache_discovery=False)
    return _TL.api


def _retry(fn, times=5):
    '''= _lap_85: lỗi -> dịch vụ mới + chờ 2 / 4 / 8 / 16 s; hết lần -> ném lỗi cuối.'''
    for k in range(times):
        try:
            return fn()
        except Exception:
            _TL.api = None
            if k == times - 1: raise
            time.sleep(2 ** (k + 1))


def download_bytes(f):
    '''f = dict(id, md5, size) -> bytes cả tệp QUA API; md5 / số byte lệch Drive -> lỗi (thử lại). (= _tai_85 + kiểm số byte của _bam_85.)'''
    from googleapiclient.http import MediaIoBaseDownload

    def one():
        bio = io.BytesIO()
        dl, done = MediaIoBaseDownload(bio, _api().files().get_media(fileId=f['id'], supportsAllDrives=True), chunksize=32 << 20), False
        while not done: _, done = dl.next_chunk()
        b = bio.getvalue()
        if f.get('md5') and hashlib.md5(b).hexdigest() != f['md5']: raise IOError('md5 ban tai ve lech Drive')
        if f.get('size') is not None and len(b) != int(f['size']): raise IOError('so byte tai ve lech Drive')
        return b
    return _retry(one)


def window(items, fn, n):
    '''= _cua_so_85: (x, kết quả, tên lỗi) ĐÚNG thứ tự items, n luồng, tối đa 2n việc chờ; lỗi MỘT việc -> tên lỗi, không dừng cả vòng.'''
    pool, it, q = ThreadPoolExecutor(n), iter(items), collections.deque()
    try:
        for x in itertools.islice(it, 2 * n): q.append((x, pool.submit(fn, x)))
        while q:
            x, fu = q.popleft()
            for x2 in itertools.islice(it, 1): q.append((x2, pool.submit(fn, x2)))
            try:
                r = fu.result()
            except Exception as e:
                yield x, None, type(e).__name__
            else:
                yield x, r, None
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def list_tree(liet_ke, fid, prefix, depth=0, max_depth=3):
    '''Mọi tệp dưới thư mục fid (đệ quy <= max_depth tầng) qua hàm _liet_ke của khối đọc chuẩn -> [dict(rel, id, md5, size)]; sâu hơn -> lỗi.'''
    out = []
    for x in liet_ke(fid):
        rel = f"{prefix}/{x['name']}"
        if x['mimeType'] == FOLDER_MIME:
            assert depth < max_depth, f'thu muc sau hon {max_depth} tang trong {prefix.split("/")[0]} -> DUNG'
            out += list_tree(liet_ke, x['id'], rel, depth + 1, max_depth)
            continue
        out.append(dict(rel=rel, id=x['id'], md5=x.get('md5Checksum'), size=int(x['size']) if x.get('size') is not None else None))
    return out


def vindr_source(duong, liet_ke, split, keys):
    '''Thư mục VinDr (MIMIC_CXR/VinDr qua _duong của khối đọc chuẩn) -> dict(tep {khoá: dict(id, md5, size)} cho `keys` của tập `split`, sums
       {khoá: [sha256]}, sha256_sums (của chính tệp SUMS), thong_ke). Thiếu / trùng ảnh, SUMS không phủ -> lỗi.'''
    root = duong(VD_PATH)
    assert root, f"khong thay {'/'.join(VD_PATH)} qua API (loi tat MIMIC_CXR / chia se) -> DUNG"
    top = liet_ke(root)
    dirs = collections.Counter(x['name'] for x in top if x['mimeType'] == FOLDER_MIME)
    goc = {x['name']: x for x in top if x['mimeType'] != FOLDER_MIME}
    assert all(dirs[t] == 1 for t in ('annotations', 'train', 'test')) and SUMS_NAME in goc, (
        f"{'/'.join(VD_PATH)} khong dung cau truc (annotations / train / test moi thu muc 1 ban + {SUMS_NAME}) -> DUNG")
    s = goc[SUMS_NAME]
    b = download_bytes(dict(id=s['id'], md5=s.get('md5Checksum'), size=s.get('size')))
    sums, bad = parse_sums(b)
    fid = next(x['id'] for x in top if x['mimeType'] == FOLDER_MIME and x['name'] == split)
    by_key = collections.defaultdict(list)
    for f in list_tree(liet_ke, fid, split):
        if f['rel'].lower().endswith(('.dicom', '.dcm')):
            by_key[key_of(f['rel'])[1]].append(f)
    want = list(keys)
    missing = [k for k in want if k not in by_key]
    dup = [k for k in want if len(by_key.get(k, [])) > 1]
    no_sum = [k for k in want if len(sums.get((split, k), [])) != 1]
    assert not missing and not dup and not no_sum and not bad, (
        f'anh {split}: thieu tren Drive {len(missing)} | trung ten {len(dup)} | khong co dung 1 dong SHA256SUMS {len(no_sum)} | dong SUMS hong {bad} -> DUNG')
    return dict(tep={k: dict(id=by_key[k][0]['id'], md5=by_key[k][0]['md5'], size=by_key[k][0]['size']) for k in want},
                sums={k: sums[(split, k)] for k in want}, sha256_sums=hashlib.sha256(b).hexdigest(),
                thong_ke=dict(dicom_trong_tap=len(by_key), anh_can=len(want), dong_sums=sum(len(v) for v in sums.values())))


# ---- một ảnh ----

def decode(b):
    '''bytes DICOM -> (lỗi | None, mảng pixel, thông tin header) với ĐÚNG các kiểm tra của _mot_anh_85.'''
    import pydicom
    try:
        ds = pydicom.dcmread(io.BytesIO(b), force=True)
        a = ds.pixel_array
    except Exception as e:
        return f'giai ma: {type(e).__name__}', None, None
    pi = str(ds.get('PhotometricInterpretation', ''))
    sl, ic = ds.get('RescaleSlope'), ds.get('RescaleIntercept')
    if pi not in ('MONOCHROME1', 'MONOCHROME2'): return f'Photometric {pi}', None, None
    if int(ds.get('SamplesPerPixel', 1)) != 1 or a.ndim != 2: return 'khong phai anh xam 2D', None, None
    if not ((sl is None or float(sl) == 1) and (ic is None or float(ic) == 0)): return 'rescale khac (1, 0)', None, None
    ts = getattr(ds.file_meta, 'TransferSyntaxUID', None)
    return None, a, dict(pi=pi, ts=str(getattr(ts, 'name', ts)), rows=int(a.shape[0]), cols=int(a.shape[1]), bits=int(ds.get('BitsStored', 0)))


def make_one(src, out_dirs, geometry='cat_giua'):
    '''-> hàm khoá -> (lỗi | None, thông tin header): tải QUA API, sha256 = SHA256SUMS, giải mã + kiểm header, ghi NGUYÊN TỬ mọi bản vào out_dirs[kho].'''
    def one(k):
        try:
            b = download_bytes(src['tep'][k])
        except Exception as e:
            return f'tai: {type(e).__name__}', None
        if hashlib.sha256(b).hexdigest() not in src['sums'][k]: return 'sha256 lech SHA256SUMS', None
        err, a, info = decode(b)
        if err: return err, None
        try:
            ims = variants(a, info['pi'] == 'MONOCHROME1', geometry)
        except Exception as e:
            return f'chuyen: {type(e).__name__}', None
        for kho, im in ims.items():
            save_jpeg(im, os.path.join(out_dirs[kho], f'{k}.jpg'))
        return None, info
    return one


def write_tar(path, image_dir, keys):
    '''tar PHẲNG <khoá>.jpg theo đúng thứ tự keys (= _tar_85 của 85b / 85e).'''
    with tarfile.open(path, 'w') as t:
        for k in keys:
            t.add(os.path.join(image_dir, f'{k}.jpg'), arcname=f'{k}.jpg')


def read_tar_members(tar_path, names):
    '''Đọc MỘT số tệp trong tar (không giải nén cả kho) -> {tên: ảnh 'L' dạng mảng uint8}.'''
    out = {}
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            if m.name in names:
                assert m.isfile() and '/' not in m.name, 'tep la trong tar kho -> DUNG'
                out[m.name] = np.asarray(Image.open(io.BytesIO(t.extractfile(m).read())).convert('L'))
    return out
