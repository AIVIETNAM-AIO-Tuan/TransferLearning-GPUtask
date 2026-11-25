# -*- coding: utf-8 -*-
"""Dựng kho 518 của một bộ (notebook 93k, Colab CPU): nguồn ảnh gốc trên Drive -> cổng toàn vẹn của 88 -> tải nhiều luồng + chuyển ảnh -> cổng -> tar +
bảng + meta.

- Nguồn: external_benchmarks.integrity @ 3c1ada7 (walk_drive + check_shenzhen / check_rsna: md5 + kích thước từng tệp / theo nhóm = mốc bản đã kiểm trên
  máy; 88h đã chạy đạt 12/11); JSRT (0.3.0): datasets.jsrt.check_jsrt = cùng compare_files của 88 với mốc từng tệp datasets/jsrt_manifest (kiểm kê máy
  23/11). Ảnh tải bằng vindr_bench.data.crop_store.download_bytes (dịch vụ API riêng mỗi luồng, thử lại 5 lần, md5 + số byte = Drive)
  qua crop_store.window (6 luồng, đúng thứ tự) — cách của 85 / 89k.
- Chạy lại ô dựng: chỉ làm ảnh còn thiếu (_tt.jsonl local).
- Cổng mẫu (16 ảnh cách đều): tải lại + chuyển lại = ảnh đã lưu TỪNG BIT (tất định); ảnh lưu ≈ letterbox thẳng 518 từ u8 (corr trung vị >= 0,99 — cổng hình học
  thô); chiều mức xám (giữa sáng hơn hai bên) theo nhóm mode / Photometric gốc và theo nhãn — cảnh báo, không dừng (= 85).
Không in khoá ảnh RSNA (mã Kaggle) — chỉ số đếm.
0.5.0: chexpert_sources (liệt kê theo lô hai cây CheXpert + CheXlocalize, ba CSV md5 ghim; không dùng mốc của 88 — CheXpert không có mốc từng tệp: md5 Drive
từng ảnh + 93i đã so khớp zip); EXPECTED_HEADER['chexpert'] = mode L (93j: 902 / 902 ảnh val + test; train chưa đo mode)."""
import collections
import json
import os
import time
import warnings

import numpy as np

from vindr_bench.data import crop_store as CS

from . import chain

N_THREADS = 6
N_SAMPLE, N_SHAPE = 16, 32
STD_MIN = 5.0
CORR_MEDIAN_MIN, CORR_MIN = 0.99, 0.98
ORIENT_WARN = 0.8
# Header ảnh gốc phải gặp (đếm trên bản ở máy đã kiểm — 88 rsna_manifest.KIEM_TREN_MAY; Shenzhen: 88 datasets.py, đếm lại 22/11 trên 662 PNG)
EXPECTED_HEADER = {
    'rsna': {'pi': {'MONOCHROME2': 26684}, 'rows': {1024: 26684}, 'cols': {1024: 26684}, 'bits': {8: 26684}},
    'shenzhen': {'mode': {'P': 635, 'RGB': 27}},
    'jsrt': {'mode': {'P': 247}, 'w': {2048: 247}, 'h': {2048: 247}},          # kiểm kê máy 23/11 (247 PNG palette ĐẢO đồng nhất)
    # CheXpert (0.5.0): JPEG xám 8 bit — 93j đo mode L ở 902 / 902 ảnh val + test; train CHƯA đo mode (93i giải mã 1.023 ảnh mẫu, không in mode) -> kỳ vọng
    # mọi ảnh 'L'; lệch -> C5 DỪNG (chạy lại C5 sau khi Claude xem, KHÔNG phải dựng lại kho: các phần đã trên Drive). Kích thước: không ghim (in ra).
    'chexpert': {'mode': {'L': 191747}},
}
HEADER_COLS = {'png': ('mode', 'w', 'h'), 'dicom': ('pi', 'ts', 'rows', 'cols', 'bits'), 'jpg': ('mode', 'w', 'h')}
KIND = {'rsna': 'dicom', 'shenzhen': 'png', 'jsrt': 'png', 'chexpert': 'jpg'}


def chexpert_sources(lister, duong, loc, log=print):
    '''CheXpert (0.5.0): liệt kê theo lô (store.listing — như 93i, đo 6,9 phút cho cây CheXpert) hai cây: MyDrive/MIMIC_CXR/CheXpert (ảnh train + valid,
       train.csv / valid.csv trong 'CheXpert-v1.0 batch 1 (validate & csv)') và MyDrive/MIMIC_CXR/CheXlocalize (CheXpert/test + CheXpert/test_labels.csv;
       ảnh val/ ở đây = valid của CheXpert từng md5 (93j) -> KHÔNG lấy). -> (tóm tắt, {khoá: dict(id, md5, size)}, {vai CSV: dict(id, ten, md5, size,
       duong)}). Dừng: thư mục sâu chưa duyệt, khoá trùng khác nội dung, ảnh không md5, CSV ghim không thấy đúng chỗ / md5.'''
    from .. import config
    from ..datasets import chexpert as CX
    trees = (('chexpert', CX.SOURCE, ('train', 'valid'), ('train', 'valid')), ('chexlocalize', config.CHEXPERT_TEST_SOURCE, ('test',), ('test',)))
    imgs, csvs, summ = {}, {}, {}
    for tag, source, splits, roles in trees:
        root = duong(source)
        assert root, f"khong thay thu muc MyDrive/{'/'.join(source)} qua API -> DUNG"
        res = lister.walk(root, resume_path=os.path.join(loc, f'_cay_{tag}.pkl'), log=log)
        assert res['chua_duyet'] == 0, f"{tag}: {res['chua_duyet']} thu muc sau chua duyet -> DUNG"
        by_key = collections.defaultdict(list)
        for f in res['files']:
            k = CX.key_of(f['duong'])
            if k and CX.parts(k)[0] in splits:
                by_key[k].append(f)
        bad = sorted(k for k, v in by_key.items() if len({(x['md5'], x['size']) for x in v}) > 1)
        assert not bad, f'{tag}: {len(bad)} khoa trung KHAC noi dung -> DUNG'
        nomd5 = sum(1 for v in by_key.values() if not v[0]['md5'])
        assert nomd5 == 0, f'{tag}: {nomd5} anh khong co md5 tren Drive -> DUNG'
        for k, v in by_key.items():
            assert k not in imgs, f'{k[:6]}…: khoa o ca hai cay -> DUNG'
            imgs[k] = dict(id=v[0]['id'], md5=v[0]['md5'], size=int(v[0]['size']))
        for role in roles:
            ten, md5, _ = CX.CSV_PINS[role]
            c = [f for f in res['files'] if f['ten'] == ten and f['duong'].rpartition('/')[0].rpartition('/')[2] == CX.CSV_DIR[role]]
            ok = [f for f in c if f['md5'] == md5]
            assert ok, (f"{role}: khong thay {CX.CSV_DIR[role]}/{ten} md5 {md5[:8]}… trong MyDrive/{'/'.join(source)} (thay {len(c)} ban cung ten, md5 "
                        f"{[f['md5'][:8] for f in c]}) -> DUNG")
            csvs[role] = dict(id=ok[0]['id'], ten=ten, md5=md5, size=int(ok[0]['size']), duong=ok[0]['duong'])
        per = collections.Counter(CX.parts(k)[0] for k in by_key)
        summ[tag] = dict(tep=len(res['files']), thu_muc=len(res['folders']), anh=dict(per), phut=round(res['thong_ke']['phut'], 1),
                         lan_goi=res['thong_ke']['lan_goi'], trung_nhieu_cho=sum(len(v) > 1 for v in by_key.values()))
    return summ, imgs, csvs


def image_sources(name, duong, liet_ke, source):
    '''-> (kết quả kiểm toàn vẹn (để ghi), {khoá ảnh: dict(id, md5, size)}, {tên tệp phụ: bản Drive}). Kiểm trượt -> dừng.'''
    from external_benchmarks import integrity
    root = duong(source)
    assert root, f"khong thay thu muc MyDrive/{'/'.join(source)} qua API -> DUNG"
    files = integrity.walk_drive(liet_ke, root)
    if name == 'shenzhen':
        res = integrity.check_shenzhen(files)
        assert res['dat'], 'Shenzhen tren Drive KHAC moc (88h) -> DUNG:\n' + integrity.summary('Shenzhen', res)
        imgs = {n[:-4]: dict(id=f['id'], md5=f['md5'], size=f['size']) for n, f in res['png']['chon'].items()}
        extra = {}
    elif name == 'rsna':
        from ..datasets import rsna
        res = integrity.check_rsna(files)
        assert res['dat'], 'RSNA tren Drive KHAC moc (88h) -> DUNG:\n' + integrity.summary('RSNA', res)
        imgs = {}
        for f in files:
            if f['cha'] == rsna.IMAGE_FOLDER and f['ten'].endswith('.dcm'):
                imgs.setdefault(f['ten'][:-4], dict(id=f['id'], md5=f['md5'], size=f['size']))      # trùng tên = cùng md5 (check_rsna đã chặn khác)
        extra = {rsna.LABELS_CSV: res['csv']['chon'][rsna.LABELS_CSV]}
    elif name == 'jsrt':
        from ..datasets import jsrt
        res = jsrt.check_jsrt(files)
        assert res['dat'], 'JSRT tren Drive KHAC moc (kiem ke tren may 23/11) -> DUNG:\n' + integrity.summary('JSRT', res)
        imgs = {n[:-4]: dict(id=f['id'], md5=f['md5'], size=f['size']) for n, f in res['png']['chon'].items()}
        extra = {jsrt.METADATA_CSV: res['metadata']['chon'][jsrt.METADATA_CSV]}
    else:
        raise ValueError(name)
    return integrity.for_log(res), imgs, extra


def _tt_path(loc):
    return os.path.join(loc, '_tt.jsonl')


def done(loc, out_dir):
    '''{khoá: thông tin header} của ảnh đã xong ở lần chạy trước (có tệp trên đĩa).'''
    tt, p = {}, _tt_path(loc)
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    tt[r.pop('k')] = r
    return {k: v for k, v in tt.items() if os.path.exists(os.path.join(out_dir, f'{k}.jpg'))}


def make_one(kind, files, out_dir):
    '''-> hàm khoá -> (lỗi | None, thông tin header): tải QUA API (md5 + số byte = Drive), giải mã, chuyển, ghi NGUYÊN TỬ <khoá>.jpg.'''
    def one(k):
        try:
            b = CS.download_bytes(files[k])
        except Exception as e:
            return f'tai: {type(e).__name__}', None
        err, u, info = chain.to_u8(kind, b)
        if err:
            return err, None
        try:
            im = chain.to_518(u)
        except Exception as e:
            return f'chuyen: {type(e).__name__}', None
        chain.save_jpeg(im, os.path.join(out_dir, f'{k}.jpg'))
        return None, info
    return one


def build(keys, kind, files, loc, out_dir, log=print):
    '''Tải + chuyển mọi ảnh còn thiếu. -> (tt {khoá: header}, lỗi theo loại, cảnh báo theo loại, phút).'''
    tt = done(loc, out_dir)
    todo = [k for k in keys if k not in tt]
    log(f'anh {len(keys):,} | xong tu lan truoc {len(keys) - len(todo):,} | can lam {len(todo):,} ({N_THREADS} luong API)')
    errs, warn_count, t0 = {}, collections.Counter(), time.time()
    one = make_one(kind, files, out_dir)

    def count_warning(message, category, filename, lineno, file=None, line=None):
        warn_count[category.__name__] += 1                     # chỉ đếm theo loại (nguyên văn cảnh báo có thể chứa UID)
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        warnings.showwarning = count_warning
        with open(_tt_path(loc), 'a', encoding='utf-8') as f:
            for i, (k, r, ng) in enumerate(CS.window(todo, one, N_THREADS), 1):
                err = ng or r[0]
                if err is None:
                    tt[k] = r[1]
                    errs.pop(k, None)
                    f.write(json.dumps(dict(k=k, **r[1])) + '\n')
                    f.flush()
                else:
                    errs[k] = err
                if i % 1000 == 0 or (len(todo) < 5000 and i % 200 == 0):
                    log(f'   {i:,} / {len(todo):,} anh | loi {len(errs)} | {(time.time() - t0) / 60:.1f} phut')
    return tt, dict(collections.Counter(errs.values())), dict(warn_count), (time.time() - t0) / 60


def load_gray(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('L'))


def sample_keys(keys, n=N_SAMPLE):
    return list(keys[:: max(1, len(keys) // n)][:n])


def sample_checks(kind, files, out_dir, keys):
    '''Tải lại ảnh gốc của `keys`, chuyển lại: (a) tất định = ảnh đã lưu từng bit; (b) ảnh lưu so với letterbox thẳng 518 từ u8 (không bước 1022, không
       JPEG). -> dict thống kê (không khoá ảnh).'''
    rows = []
    for k in keys:
        b = CS.download_bytes(files[k])
        err, u, _ = chain.to_u8(kind, b)
        assert err is None, f'mau: giai ma loi ({err}) -> DUNG'
        stored = load_gray(os.path.join(out_dir, f'{k}.jpg'))
        again = chain.jpeg_roundtrip(chain.to_518(u))
        rows.append(dict(tat_dinh=bool(np.array_equal(stored, again)), hinh=CS.similarity(stored, CS.reference(u, 'dem_vuong')),
                         ti_le=max(u.shape) / min(u.shape)))
    c = np.array([r['hinh']['corr'] for r in rows])
    out = dict(n=len(rows), tat_dinh=sum(r['tat_dinh'] for r in rows), corr_trung_vi=float(np.median(c)), corr_min=float(c.min()),
               mad_trung_vi=float(np.median([r['hinh']['mad'] for r in rows])), ti_le_max=float(max(r['ti_le'] for r in rows)))
    out['dat'] = dict(tat_dinh=out['tat_dinh'] == len(rows), hinh_hoc=out['corr_trung_vi'] >= CORR_MEDIAN_MIN and out['corr_min'] >= CORR_MIN)
    return out


def header_counts(tt, keys, kind):
    return {c: dict(collections.Counter(tt[k][c] for k in keys)) for c in HEADER_COLS[kind]}


def header_gate(name, counts):
    '''Đếm header ảnh gốc so với EXPECTED_HEADER[name] (chỉ các cột có ghim). -> (đạt, {cột: [có, ghim]} lệch).'''
    exp = EXPECTED_HEADER[name]
    diff = {c: [counts.get(c), want] for c, want in exp.items() if {str(k): v for k, v in (counts.get(c) or {}).items()} != {str(k): v for k, v in want.items()}}
    return not diff, diff


def orientation(out_dir, keys, groups):
    '''Tỉ lệ ảnh "giữa sáng hơn hai bên" (= _giua_ben_85 > 0) theo từng nhóm (vd mode gốc, nhãn). -> {nhóm: tỉ lệ}.'''
    by = collections.defaultdict(list)
    for k, g in zip(keys, groups):
        by[str(g)].append(CS.center_minus_sides(load_gray(os.path.join(out_dir, f'{k}.jpg'))) > 0)
    return {g: round(float(np.mean(v)), 3) for g, v in sorted(by.items())}
