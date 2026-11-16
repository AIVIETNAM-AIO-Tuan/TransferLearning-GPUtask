# -*- coding: utf-8 -*-
"""Ô của 4 notebook dựng kho cắt giữa `89k_kho_cat_giua_<phần>` (PLAN_89 §5.3; Colab CPU). Mỗi hàm = một ô; trạng thái đi theo dict CTX; mọi việc
Drive đi qua hàm của khối Drive chuẩn ở C0 (+ tải DICOM bằng dịch vụ API riêng mỗi luồng như 85 — data/crop_store.py).

    C2 setup    gói giải mã DICOM như 85 (pydicom + pylibjpeg, không ghim, ghi phiên bản) + thư mục local + hàm của khối Drive chuẩn
    C3 sources  kho đệm vuông KHÔNG cân bằng cùng phần (85b / 85e `_khong_cb`): chỉ mục, bảng nhãn, meta, tar + thư mục VinDr (bản PhysioNet):
                SHA256SUMS + DICOM của đúng các khoá phần này (khoá / thứ tự / nhãn / tách KHÔNG tự tính — chép từ kho đệm vuông)
    C4 build    tải QUA API (6 luồng) + sha256 = SUMS + giải mã + chuyển CẢ hai bản (cg_kcb, cg_cb) -> JPEG 518 local; chạy lại ô: bỏ ảnh đã xong
    C5 pack     cổng (đủ ảnh; header = chỉ mục 85; mẫu = Resize + CenterCrop trên ảnh gốc + đối chứng âm đệm vuông; cùng chuỗi 85 ở hình học đệm vuông;
                tất định; SUMS = tệp 85 dùng; chiều mức xám) rồi ghi (local) tar x 2 + chỉ mục + bảng nhãn + meta x 2
    C6 sync     dong_bo() + kiểm tệp trên Drive

Không in mã ảnh (luật PhysioNet): chỉ số đếm, thống kê, tên tệp kho."""
import collections
import copy
import datetime
import hashlib
import importlib.metadata as IM
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np

from .. import __version__
from ..data import crop_store as CS
from ..data import vindr as V

FOLDER = 'stage4_nb89'
PARTS = ('test', 1, 2, 3)
PAD_STORE = 'kcb'                     # nguồn khoá / nhãn / thứ tự + đối chiếu "cùng chuỗi 85"
DRIVE_NAMES = ('kiem_ke', 'tai', 'xung_dot', 'ghi', 'ghi_json', 'dong_bo', 'ten_tren_drive', 'RUN_TAG', 'GHI_LOC', '_duong', '_liet_ke')
N_THREADS = 6                         # = N_LUONG_85B (luồng API tải + chuyển ảnh của 85b / 85e)
N_SAMPLE = 16                         # ảnh mẫu (cách đều theo thứ tự kho) cho cổng hình học / chuỗi 85 / tất định
N_SHAPE = 32                          # ảnh mẫu kiểm 518 x 518 'L' + độ lệch chuẩn mức xám (= 85)
STD_MIN = 5.0                         # = 85: ảnh gần như một màu -> dừng
# Cổng mẫu (hiệu chỉnh 16/11 trên 5 ảnh X-quang mẫu công khai của repo CheXFound, cắt về 4 tỉ lệ khung gồm trung vị VinDr; scratchpad
# hieu_chinh_cong_cat_giua.py): ảnh lưu so với "Resize + CenterCrop" thẳng từ u8 — corr 0,99951–0,99993, lệch TB <= 0,91 mức xám; đối chứng âm (đệm vuông)
# corr <= 0,80, lệch TB >= 32. Chỉ DỪNG khi hỏng thô (ngưỡng cách nền ~20 lần theo 1 − corr); mọi số in ra.
CORR_MEDIAN_MIN, CORR_MIN = 0.99, 0.98
SEPARATION_MIN = 0.05                 # TRUNG VỊ của corr(đúng hình học) − corr(đối chứng âm) trên ảnh mẫu LỆCH VUÔNG (ảnh vuông: cắt giữa = đệm vuông, không tách)
ASPECT_MIN = 1.10                     # ảnh mẫu tính vào phép tách khi cạnh dài / cạnh ngắn >= 1,10 (viền / phần cắt >= ~4,5 % mỗi bên; VinDr trung vị ~1,19)
SAME85_MAD_MEDIAN_MAX, SAME85_CORR_MIN = 1.0, 0.999    # chuỗi của gói ở hình học đệm vuông so với ảnh trong tar 85 (cùng thư viện -> trùng từng bit)
ORIENT_WARN = 0.8                     # = 85: tỉ lệ ảnh "giữa sáng hơn hai bên" < 0,8 -> cảnh báo (không dừng)


def run_tag(part):
    return 'r89k_cat_giua_test' if part == 'test' else f'r89k_cat_giua_train_p{int(part)}'


def notebook_name(part):
    return '89k_kho_cat_giua_test' if part == 'test' else f'89k_kho_cat_giua_train_p{int(part)}'


def _part(part):
    assert part in PARTS, f'phan {part!r} khong thuoc {PARTS} -> DUNG'
    return part


def drive_api(ns):
    missing = [k for k in DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    return {k: ns[k] for k in DRIVE_NAMES}


def _version(p):
    try:
        return IM.version(p)
    except Exception:
        return None


def install_decoders(log=print):
    '''= C0 của 85b: Colab -> pip install pydicom + pylibjpeg (+ libjpeg, openjpeg), KHÔNG ghim (85 không ghim) -> ghi phiên bản. Máy (bộ kiểm): không cài.'''
    try:
        import google.colab  # noqa: F401
        colab = True
    except ImportError:
        colab = False
    if colab:
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *CS.DECODE_PACKAGES], capture_output=True, text=True)
        assert r.returncode == 0, f'pip install {CS.DECODE_PACKAGES} that bai -> DUNG: {r.stderr[-400:]}'
    import cv2
    import pydicom  # noqa: F401
    v = {p: _version(p) for p in CS.DECODE_PACKAGES + ('opencv-python', 'opencv-python-headless', 'Pillow', 'numpy', 'pandas', 'google-api-python-client')}
    v['cv2'] = cv2.__version__
    return colab, v


def setup(part, ns, log=print):
    '''C2. -> CTX.'''
    part = _part(part)
    drive = drive_api(ns)
    assert drive['RUN_TAG'] == run_tag(part), f"RUN_TAG {drive['RUN_TAG']!r} != {run_tag(part)!r} -> DUNG"
    import vindr_bench
    colab, ver = install_decoders(log)
    loc = f'/content/cg_{part}' if colab else os.path.join(tempfile.mkdtemp(), f'cg_{part}')
    out_dirs = {k: os.path.join(loc, k) for k in CS.VARIANTS}
    for p in out_dirs.values():
        os.makedirs(p, exist_ok=True)
    du = os.statvfs(loc) if hasattr(os, 'statvfs') else None
    free = du.f_bavail * du.f_frsize / 1e9 if du else float('nan')
    ctx = dict(phan=part, drive=drive, loc=loc, out_dirs=out_dirs, ban_goi=ver, colab=colab,
               goi=dict(phien_ban=__version__, sha=getattr(vindr_bench, 'GOI_SHA', None)))
    log(f"89k phan {part}: kho cat giua (co canh ngan {CS.SIDE_MID} + cat giua -> {CS.SIDE_OUT}), ghi CA {list(CS.VARIANTS)} | {os.cpu_count()} CPU | "
        f"o dia trong {free:.0f} GB | {N_THREADS} luong API | goi vindr_bench {__version__} @ {(ctx['goi']['sha'] or '?')[:12]} | thu vien giai ma {ver}")
    return ctx


def sources(ctx, log=print):
    '''C3. Kho đệm vuông không cân bằng cùng phần + nguồn DICOM (bản PhysioNet trên Drive). -> dict nguồn.'''
    d, part = ctx['drive'], ctx['phan']
    import pandas as pd
    st = V.STORES[PAD_STORE]
    du = st['test'] if part == 'test' else st['train'][part]
    fn = V.store_files(PAD_STORE, part)
    kk = d['kiem_ke'](du, mau='(' + '|'.join(re.escape(v) for v in fn.values()) + ')')[1]
    used = {}

    def get(name):
        v = kk.get(name)
        assert v, f"thieu {name} trong {'/'.join(du)} -> DUNG"
        if d['xung_dot'](v): log(f"  ! {name}: {len(v)} ban NOI DUNG KHAC -> dung ban MOI NHAT")
        used[name] = dict(md5=v[0]['md5'], so_ban=len(v))
        return d['tai'](v[0])
    cm = pd.read_parquet(get(fn['chi_muc']))
    nh = pd.read_parquet(get(fn['nhan']))
    with open(get(fn['meta']), encoding='utf-8') as f:
        meta = json.load(f)
    keys = cm.khoa.astype(str).tolist()
    assert V.md5_keys(keys) == meta['md5_khoa'] and len(keys) == int(meta['n']) and cm.anh_ok.astype(bool).all(), (
        f'chi muc / meta kho {PAD_STORE} phan {part} KHONG khop -> DUNG')
    assert V.store_style(meta) == PAD_STORE, f'meta kho nguon khong phai {PAD_STORE} ({V.STORE_LABELS[PAD_STORE]}) -> DUNG'
    if part == 'test':
        assert list(nh.khoa.astype(str)) == keys, 'bang nhan test KHAC thu tu chi muc -> DUNG'
    else:
        assert int(meta['phan']) == part and set(keys) <= set(nh.khoa.astype(str)), f'85e phan {part}: meta / bang nhan KHONG khop -> DUNG'
    tar85 = get(fn['tar'])
    split = 'test' if part == 'test' else 'train'
    t0 = time.time()
    src = CS.vindr_source(d['_duong'], d['_liet_ke'], split, keys)
    assert src['sha256_sums'] == meta.get('sha256_sums'), 'SHA256SUMS tren Drive KHAC tep 85 da dung -> DUNG (ban VinDr doi?)'
    ctx.update(khoa=keys, chi_muc_85=cm, nhan_85=nh, meta_85=meta, tar_85=tar85, nguon=src, du_85=du, tep_85=used)
    log(f"kho nguon {'/'.join(du)} ({V.STORE_LABELS[PAD_STORE]}): {len(keys):,} anh | md5 khoa = meta | {len(nh):,} dong nhan | tep {sorted(used)} | "
        f"VinDr ({'/'.join(CS.VD_PATH)}): {src['thong_ke']} | SHA256SUMS = tep 85 da dung | {(time.time() - t0) / 60:.1f} phut")
    return src


def _tt_path(ctx):
    return os.path.join(ctx['loc'], '_tt.jsonl')


def _done(ctx):
    '''Thông tin header các ảnh đã xong từ lần chạy trước (chỉ ảnh có ĐỦ mọi bản trên đĩa).'''
    tt = {}
    p = _tt_path(ctx)
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    tt[r.pop('k')] = r
    return {k: v for k, v in tt.items() if all(os.path.exists(os.path.join(ctx['out_dirs'][s], f'{k}.jpg')) for s in CS.VARIANTS)}


def build(ctx, log=print):
    '''C4. Tải + chuyển mọi ảnh của phần; chạy lại ô chỉ làm ảnh còn thiếu. Lỗi một ảnh -> đếm theo loại (không in mã ảnh); còn thiếu -> dừng.'''
    keys, src = ctx['khoa'], ctx['nguon']
    tt = _done(ctx)
    todo = [k for k in keys if k not in tt]
    log(f"anh {len(keys):,} | xong tu lan truoc {len(keys) - len(todo):,} | can lam {len(todo):,} ({N_THREADS} luong API)")
    errs, warn_count, t0 = {}, collections.Counter(), time.time()
    one = CS.make_one(src, ctx['out_dirs'])

    def count_warning(message, category, filename, lineno, file=None, line=None):
        warn_count[category.__name__] += 1                     # nguyên văn cảnh báo có thể chứa UID -> CHỈ đếm theo loại (= 85)
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        warnings.showwarning = count_warning
        with open(_tt_path(ctx), 'a', encoding='utf-8') as f:
            for i, (k, r, ng) in enumerate(CS.window(todo, one, N_THREADS), 1):
                err = ng or r[0]
                if err is None:
                    tt[k] = r[1]
                    errs.pop(k, None)
                    f.write(json.dumps(dict(k=k, **r[1])) + '\n')
                    f.flush()
                else:
                    errs[k] = err
                if i % 300 == 0:
                    log(f'   {i:,} / {len(todo):,} anh | loi {len(errs)} | {(time.time() - t0) / 60:.1f} phut')
    ok = [k in tt for k in keys]
    ctx.update(tt=tt, canh_bao=dict(warn_count), loi=dict(collections.Counter(errs.values())))
    log(f"anh xong {sum(ok):,} / {len(keys):,} | loi {len(errs)} (theo loai {ctx['loi'] or 0}) | {(time.time() - t0) / 60:.1f} phut | canh bao (theo loai) "
        f"{ctx['canh_bao'] or 0}")
    assert all(ok), 'chua du anh: chay lai C4 (chi lam anh con thieu); van loi -> DUNG, gui Claude dong tren'
    return dict(xong=sum(ok), loi=ctx['loi'], phut=(time.time() - t0) / 60)


def _jpeg_roundtrip(im):
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=CS.JPEG_OUT)
    b.seek(0)
    from PIL import Image
    return np.asarray(Image.open(b).convert('L'))


def _load(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('L'))


def sample_keys(keys, n=N_SAMPLE):
    '''n khoá cách đều theo thứ tự kho (tất định, không ngẫu nhiên).'''
    return list(keys[:: max(1, len(keys) // n)][:n])


def sample_checks(ctx, keys, log=print):
    '''Cổng mẫu: tải lại DICOM của `keys`, giải mã -> u8 -> (a) tất định: chuỗi chạy lại = ảnh đã lưu (cả hai bản); (b) ảnh lưu (cg_kcb) so với Resize +
       CenterCrop thẳng từ u8; (c) đối chứng âm: so với đệm vuông thẳng từ u8; (d) chuỗi của gói ở hình học ĐỆM VUÔNG so với ảnh trong tar 85 (cùng chuỗi
       85). -> dict thống kê (không mã ảnh).'''
    import cv2
    src, od = ctx['nguon'], ctx['out_dirs']
    ref85 = CS.read_tar_members(ctx['tar_85'], {f'{k}.jpg' for k in keys})
    rows = []
    for k in keys:
        b = CS.download_bytes(src['tep'][k])
        assert hashlib.sha256(b).hexdigest() in src['sums'][k], 'mau: sha256 lech SHA256SUMS -> DUNG'
        err, a, info = CS.decode(b)
        assert err is None, f'mau: giai ma loi ({err}) -> DUNG'
        u = CS.to_u8(a, info['pi'] == 'MONOCHROME1')
        stored = {s: _load(os.path.join(od[s], f'{k}.jpg')) for s in CS.VARIANTS}
        again = {s: _jpeg_roundtrip(CS.chain(cv2.equalizeHist(u) if eq else u, 'cat_giua')) for s, eq in CS.VARIANTS.items()}
        pad = _jpeg_roundtrip(CS.chain(u, 'dem_vuong'))
        rows.append(dict(tat_dinh=all(np.array_equal(stored[s], again[s]) for s in CS.VARIANTS), ti_le=max(u.shape) / min(u.shape),
                         dung=CS.similarity(stored[CS.CROP_STORE_KEY], CS.reference(u, 'cat_giua')),
                         am=CS.similarity(stored[CS.CROP_STORE_KEY], CS.reference(u, 'dem_vuong')),
                         chuoi85=CS.similarity(pad, ref85[f'{k}.jpg'])))
    c_ok = np.array([r['dung']['corr'] for r in rows]); c_bad = np.array([r['am']['corr'] for r in rows])
    lech = np.array([r['ti_le'] >= ASPECT_MIN for r in rows])
    m85 = np.array([r['chuoi85']['mad'] for r in rows]); c85 = np.array([r['chuoi85']['corr'] for r in rows])
    out = dict(n=len(rows), tat_dinh=sum(r['tat_dinh'] for r in rows), n_lech_vuong=int(lech.sum()),
               dung=dict(corr_trung_vi=float(np.median(c_ok)), corr_min=float(c_ok.min()), mad_trung_vi=float(np.median([r['dung']['mad'] for r in rows]))),
               am=dict(corr_trung_vi=float(np.median(c_bad)), corr_max=float(c_bad.max()), mad_trung_vi=float(np.median([r['am']['mad'] for r in rows]))),
               tach_trung_vi=float(np.median((c_ok - c_bad)[lech])) if lech.any() else None,
               chuoi85=dict(trung_tung_bit=sum(r['chuoi85']['trung'] for r in rows), mad_trung_vi=float(np.median(m85)), mad_max=float(m85.max()),
                            corr_min=float(c85.min())))
    out['dat'] = dict(tat_dinh=out['tat_dinh'] == len(rows),
                      hinh_hoc=(out['dung']['corr_trung_vi'] >= CORR_MEDIAN_MIN and out['dung']['corr_min'] >= CORR_MIN
                                and (out['tach_trung_vi'] is None or out['tach_trung_vi'] >= SEPARATION_MIN)),
                      chuoi85=out['chuoi85']['mad_trung_vi'] <= SAME85_MAD_MEDIAN_MAX and out['chuoi85']['corr_min'] >= SAME85_CORR_MIN)
    return out


def pack(ctx, log=print):
    '''C5. Cổng rồi ghi (local; lên Drive ở C6) tar x 2 + chỉ mục + bảng nhãn + meta x 2. -> dict cổng.'''
    import pandas as pd
    d, part, keys, tt, od = ctx['drive'], ctx['phan'], ctx['khoa'], ctx['tt'], ctx['out_dirs']
    t0 = time.time()
    # (1) đủ ảnh mọi bản + 518 x 518 'L' + không gần một màu
    assert all(k in tt for k in keys) and all(os.path.exists(os.path.join(od[s], f'{k}.jpg')) for s in CS.VARIANTS for k in keys), 'thieu anh -> chay lai C4'
    shp = [_load(os.path.join(od[s], f'{k}.jpg')) for s in CS.VARIANTS for k in keys[:: max(1, len(keys) // N_SHAPE)]]
    ok_shape = all(a.shape == (CS.SIDE_OUT, CS.SIDE_OUT) for a in shp)
    std = [float(a.std()) for a in shp]
    # (2) header của từng ảnh = chỉ mục kho 85 (cùng DICOM, cùng giải mã)
    cm85 = ctx['chi_muc_85'].set_index(ctx['chi_muc_85'].khoa.astype(str))
    cols = [c for c in ('pi', 'ts', 'rows', 'cols', 'bits') if c in cm85.columns]
    lech_header = sum(1 for k in keys for c in cols if str(tt[k][c]) != str(cm85.at[k, c]))
    # (3) mẫu
    sm = sample_checks(ctx, sample_keys(keys), log)
    # (4) chiều mức xám theo Photometric gốc (= 85; chỉ cảnh báo)
    by_pi = collections.defaultdict(list)
    for k in keys:
        by_pi[tt[k]['pi']].append(CS.center_minus_sides(_load(os.path.join(od[CS.CROP_STORE_KEY], f'{k}.jpg'))) > 0)
    orient = {pi: round(float(np.mean(v)), 3) for pi, v in sorted(by_pi.items())}
    warn_orient = min(orient.values()) < ORIENT_WARN or max(orient.values()) - min(orient.values()) > 0.15
    gate = dict(du_anh=True, hinh_518=ok_shape, std_min=min(std), header_lech=lech_header, mau=sm, chieu_muc_xam=orient)
    log(f"cong kho cat giua phan {part}: {len(keys):,} anh x {len(CS.VARIANTS)} ban | 518 x 518 L ({len(shp)} mau): {'DAT' if ok_shape else 'TRUOT'} | do lech "
        f"chuan muc xam min {min(std):.1f} (> {STD_MIN:g}) | header (Photometric / nen / kich thuoc / bits) khac chi muc 85: {lech_header} o")
    log(f"  mau {sm['n']} anh: tat dinh (chuoi chay lai = anh da luu, ca hai ban) {sm['tat_dinh']} / {sm['n']} | = Resize + CenterCrop tren anh goc: corr "
        f"trung vi {sm['dung']['corr_trung_vi']:.5f}, min {sm['dung']['corr_min']:.5f} (nguong {CORR_MEDIAN_MIN} / {CORR_MIN}), lech TB trung vi "
        f"{sm['dung']['mad_trung_vi']:.2f} muc xam | doi chung am (dem vuong): corr trung vi {sm['am']['corr_trung_vi']:.4f}, max {sm['am']['corr_max']:.4f}, "
        f"lech TB trung vi {sm['am']['mad_trung_vi']:.1f} | tach (dung - am) trung vi tren {sm['n_lech_vuong']} anh lech vuong (canh dai / ngan >= "
        f"{ASPECT_MIN}): " + (f"{sm['tach_trung_vi']:.3f} (>= {SEPARATION_MIN})" if sm['tach_trung_vi'] is not None else 'khong co anh lech vuong -> bo qua'))
    log(f"  cung chuoi 85 (chuoi cua goi o hinh hoc DEM VUONG so voi anh trong tar 85): trung tung bit {sm['chuoi85']['trung_tung_bit']} / {sm['n']} | lech TB "
        f"trung vi {sm['chuoi85']['mad_trung_vi']:.3f}, max {sm['chuoi85']['mad_max']:.3f} muc xam | corr min {sm['chuoi85']['corr_min']:.5f} (nguong "
        f"lech TB trung vi <= {SAME85_MAD_MEDIAN_MAX}, corr >= {SAME85_CORR_MIN})")
    log(f"  chieu muc xam (ti le anh giua sang hon hai ben, theo Photometric goc): {orient}"
        + (' | ⚠ nhom thap / lech nhau -> gui Claude dong nay' if warn_orient else ' | cac nhom cung chieu'))
    assert ok_shape and min(std) > STD_MIN and lech_header == 0, 'anh sai kich thuoc / gan mot mau / header khac chi muc 85 -> DUNG, gui Claude cac dong tren'
    assert all(sm['dat'].values()), f"cong mau TRUOT {[k for k, v in sm['dat'].items() if not v]} -> DUNG, gui Claude cac dong tren"
    # ghi (local)
    names = []
    cm = pd.DataFrame(dict(khoa=keys, **{c: [tt[k][c] for k in keys] for c in ('pi', 'ts', 'rows', 'cols', 'bits')}, anh_ok=[True] * len(keys)))
    for store in CS.VARIANTS:
        fn = V.store_files(store, part)
        if store == CS.CROP_STORE_KEY:                     # chỉ mục + bảng nhãn dùng chung hai bản (cùng tên tệp)
            d['ghi'](fn['chi_muc'], lambda q: cm.to_parquet(q, index=False))
            d['ghi'](fn['nhan'], lambda q: ctx['nhan_85'].to_parquet(q, index=False))
            names += [fn['chi_muc'], fn['nhan']]
        d['ghi'](fn['tar'], lambda q, s=store: CS.write_tar(q, od[s], keys))
        meta = _meta(ctx, store, gate)
        assert V.store_style(meta) == store, f'meta kho {store} ghi sai kieu anh -> DUNG'
        d['ghi'](fn['meta'], lambda q, m=meta: d['ghi_json'](q, m))
        names += [fn['tar'], fn['meta']]
    ctx.update(cong=gate, tep_ghi=names)
    log(f"da ghi {names} (local) -> {FOLDER}/{d['RUN_TAG']} o C6 | {(time.time() - t0) / 60:.1f} phut")
    return gate


def _meta(ctx, store, gate):
    '''Meta kho cắt giữa = meta kho đệm vuông cùng phần (GIỮ mọi trường load_vindr kiểm: n, md5_khoa, nhan, so_duong / phan, md5_nhan, md5_khoa_train,
       lop_chinh_p2 …) + thay phần ảnh / nguồn / thời điểm.'''
    m = copy.deepcopy(ctx['meta_85'])
    tt, keys = ctx['tt'], ctx['khoa']
    anh85 = m.get('anh') or {}
    m.update(t=datetime.datetime.now().isoformat(timespec='seconds'), run=ctx['drive']['RUN_TAG'], ban_goi=ctx['ban_goi'], goi=ctx['goi'],
             nguon_kho_dem_vuong=dict(thu_muc='/'.join(ctx['du_85']), run=ctx['meta_85'].get('run'), tep=ctx['tep_85']),
             cong_kho_cat_giua=json.loads(json.dumps(gate, default=str)),
             ghi_chu=('PLAN_89 5.3 (Tuan 13/11 + 16/11): kho cat giua = chuoi giai ma cua 85, CHI thay hinh hoc (co canh ngan 1022 + cat giua thay dem vuong '
                      '1022). Khoa / thu tu / nhan / tach chep tu kho dem vuong khong can bang cung phan. KHONG id anh trong tep nay.'))
    m['anh'] = dict(photometric=dict(collections.Counter(tt[k]['pi'] for k in keys)), nen=dict(collections.Counter(tt[k]['ts'] for k in keys)),
                    canh_bao=ctx.get('canh_bao') or {}, sang_giua=gate['chieu_muc_xam'], cach=CS.CACH[store], hinh_hoc='cat_giua',
                    can_bang=CS.VARIANTS[store], canh_giua=CS.SIDE_MID, canh_ra=CS.SIDE_OUT, anh_85=dict(cach=anh85.get('cach')))
    return m


def sync(ctx, log=print):
    '''C6. dong_bo() + kiểm các tệp của kho trên Drive.'''
    d = ctx['drive']
    du = d['dong_bo']()
    on = d['ten_tren_drive']()
    need = list(ctx.get('tep_ghi') or [])
    co = bool(need) and set(need) <= on
    log(f"C6: dong_bo {'DU' if du else 'CHUA DU'} | {len(need)} tep kho tren Drive: {'CO du' if co else 'THIEU / C5 CHUA XONG'}"
        + (' -> gui Claude output C2-C6; xong ca 4 notebook 89k thi chay 89v2_*_cg_kcb; roi chay 89c' if du and co else ' | KHONG tat runtime; chay lai C6'))
    return dict(dong_bo=du, du_tep=co)
