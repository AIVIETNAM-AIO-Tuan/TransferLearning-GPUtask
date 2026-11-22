# -*- coding: utf-8 -*-
"""Ô của notebook kho `93k_kho_<bộ>` (PLAN_93 §7; Colab CPU). Mỗi hàm = một ô; trạng thái đi theo dict CTX; Drive qua khối chuẩn ở C0 (+ tải ảnh bằng
dịch vụ API riêng mỗi luồng như 85 / 89k).

    C2 setup    gói giải mã như 85 (pydicom + pylibjpeg; không ghim, ghi phiên bản) + thư mục local + hàm của khối Drive chuẩn
    C3 sources  ảnh gốc trên Drive: cổng toàn vẹn của 88 (md5 + kích thước = mốc) + bảng khoá / tách / cụm / nhãn (RSNA: tệp nối RSNA -> NIH tải theo URL công
                khai + CSV nhãn trên Drive; Shenzhen: tệp chia BenchmarkTransformers theo commit) -> cổng số đếm + vân tay = ghim
    C4 build    tải QUA API (6 luồng) + giải mã + chuỗi kho 518 của 85 (đệm vuông, KHÔNG cân bằng) -> JPEG local; chạy lại ô: chỉ làm ảnh còn thiếu
    C5 pack     cổng (đủ ảnh, 518 x 518, header ảnh gốc = số đếm đã kiểm, mẫu tất định + hình học, chiều mức xám) rồi ghi (local) tar + bảng + meta
    C6 sync     dong_bo() + kiểm tệp trên Drive
Không in khoá ảnh RSNA."""
import datetime
import json
import os
import tempfile
import time

import numpy as np

from .. import __version__, config
from ..datasets import base
from ..store import build as SB
from ..store import chain

DRIVE_NAMES = ('kiem_ke', 'tai', 'xung_dot', 'ghi', 'ghi_json', 'dong_bo', 'ten_tren_drive', 'RUN_TAG', 'GHI_LOC', '_duong', '_liet_ke')
STORES = ('rsna', 'shenzhen')


def notebook_name(name):
    return f'93k_kho_{name}'


def drive_api(ns):
    missing = [k for k in DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    return {k: ns[k] for k in DRIVE_NAMES}


def _packages():
    import external_benchmarks
    import vindr_bench
    import crossdata_bench
    return {m.__name__: dict(phien_ban=m.__version__, sha=getattr(m, 'GOI_SHA', None)) for m in (crossdata_bench, vindr_bench, external_benchmarks)}


def setup(name, ns, log=print):
    '''C2. -> CTX.'''
    assert name in STORES, f'kho {name!r} chua dung ({STORES}) -> DUNG'
    sp = config.dataset(name)
    drive = drive_api(ns)
    assert drive['RUN_TAG'] == sp.store_tag, f"RUN_TAG {drive['RUN_TAG']!r} != {sp.store_tag!r} -> DUNG"
    from vindr_bench.notebook.crop_store import install_decoders
    colab, ver = install_decoders(log)
    loc = f'/content/kho93_{name}' if colab else os.path.join(tempfile.mkdtemp(), f'kho93_{name}')
    out_dir = os.path.join(loc, 'anh')
    os.makedirs(out_dir, exist_ok=True)
    st = os.statvfs(loc) if hasattr(os, 'statvfs') else None
    free = st.f_bavail * st.f_frsize / 1e9 if st else float('nan')
    ctx = dict(bo=name, spec=sp, drive=drive, loc=loc, out_dir=out_dir, ban_goi=ver, colab=colab, goi=_packages(), kind=SB.KIND[name])
    log(f"93k {sp.title}: kho 518 dem vuong KHONG can bang histogram (chuoi kho kcb cua 85) | nguon MyDrive/{'/'.join(sp.source)} | {os.cpu_count()} CPU | "
        f"o dia trong {free:.0f} GB | {SB.N_THREADS} luong API | goi " + ', '.join(f"{k} {v['phien_ban']} @ {(v['sha'] or '?')[:12]}" for k, v in ctx['goi'].items())
        + f" | thu vien giai ma {ver}")
    return ctx


def _table(ctx, extra, log):
    import pandas as pd  # noqa: F401
    name, d = ctx['bo'], ctx['drive']
    if name == 'rsna':
        from external_benchmarks import rsna_manifest as RM
        from ..datasets import rsna
        csv = d['tai'](extra[rsna.LABELS_CSV])
        labels = rsna.read_labels(csv, md5=RM.CSV[rsna.LABELS_CSV][0])
        mapping = rsna.read_mapping(rsna.fetch_mapping(os.path.join(ctx['loc'], 'nguon'), log))
        info = dict(tep_noi=dict(url=rsna.MAPPING_URL, md5=rsna.MAPPING_MD5, byte=rsna.MAPPING_SIZE), nhan_csv=dict(ten=rsna.LABELS_CSV, md5=RM.CSV[rsna.LABELS_CSV][0]),
                    chia=dict(don_vi='nguoi NIH', tang='nguoi co >= 1 anh duong', hat=rsna.SPLIT_SEED, ti_le=rsna.FRACTIONS))
        return rsna.table(mapping, labels), info
    from external_benchmarks import config as EC
    from external_benchmarks import splits as ES
    from ..datasets import shenzhen
    lists = ES.read_shenzhen_lists(ES.download_shenzhen_lists(os.path.join(ctx['loc'], 'nguon'), log))
    info = dict(tep_chia=dict(nguon='jlianglab/BenchmarkTransformers', commit=EC.SHENZHEN_SPLIT_COMMIT, md5={s: v[1] for s, v in EC.SHENZHEN_SPLITS.items()}),
                cum='moi anh mot cum (bo khong co ma nguoi)')
    return shenzhen.table(lists), info


def sources(ctx, log=print):
    '''C3. Nguồn ảnh trên Drive (cổng toàn vẹn 88) + bảng khoá / tách / cụm / nhãn (cổng số đếm + vân tay = ghim). -> bảng.'''
    d, sp, t0 = ctx['drive'], ctx['spec'], time.time()
    kiem, files, extra = SB.image_sources(ctx['bo'], d['_duong'], d['_liet_ke'], sp.source)
    df, info = _table(ctx, extra, log)
    fp = base.check_table(ctx['bo'], df)
    same, diff = base.fingerprint_gate(ctx['bo'], fp)
    assert same, f'van tay bang KHAC ghim {diff} -> DUNG, gui Claude dong nay'
    keys = df.khoa.astype(str).tolist()
    miss = [k for k in keys if k not in files]
    assert not miss, f'{len(miss)} anh trong bang KHONG co tren Drive -> DUNG'
    ctx.update(bang=df, khoa=keys, tep=files, kiem_toan_ven=kiem, nguon_bang=info, van_tay=fp)
    ng = {s: df[df.tach == s].nhom.nunique() for s in config.SPLITS}
    log(f"nguon MyDrive/{'/'.join(sp.source)}: kiem toan ven 88 DAT ({kiem['tong_tep_drive']:,} tep tren Drive) | {len(files):,} anh goc, dung {len(keys):,} | "
        + ' / '.join(f"{s} {fp[s]['so_anh']:,} anh ({fp[s]['so_duong'][0]:,} duong" + (f", {ng[s]:,} nguoi" if sp.unit == 'nguoi' else '') + ')'
                     for s in config.SPLITS)
        + f" | van tay md5 khoa / nhan TRUNG ghim | {(time.time() - t0) / 60:.1f} phut")
    return df


def build(ctx, log=print):
    '''C4. Tải + chuyển mọi ảnh của bộ; chạy lại ô chỉ làm ảnh còn thiếu. Còn thiếu -> dừng (lỗi chỉ đếm theo loại).'''
    keys = ctx['khoa']
    tt, errs, warns, minutes = SB.build(keys, ctx['kind'], ctx['tep'], ctx['loc'], ctx['out_dir'], log)
    ok = sum(k in tt for k in keys)
    ctx.update(tt=tt, loi=errs, canh_bao=warns)
    log(f"anh xong {ok:,} / {len(keys):,} | loi {sum(errs.values()) if errs else 0} (theo loai {errs or 0}) | {minutes:.1f} phut | canh bao (theo loai) {warns or 0}")
    assert ok == len(keys), 'chua du anh: chay lai C4 (chi lam anh con thieu); van loi -> DUNG, gui Claude dong tren'
    return dict(xong=ok, loi=errs, phut=minutes)


def pack(ctx, log=print):
    '''C5. Cổng rồi ghi (local; lên Drive ở C6) tar + bảng + meta. -> dict cổng.'''
    d, name, keys, tt, od, kind = ctx['drive'], ctx['bo'], ctx['khoa'], ctx['tt'], ctx['out_dir'], ctx['kind']
    sp, df, t0 = ctx['spec'], ctx['bang'], time.time()
    assert all(k in tt and os.path.exists(os.path.join(od, f'{k}.jpg')) for k in keys), 'thieu anh -> chay lai C4'
    shp = [SB.load_gray(os.path.join(od, f'{k}.jpg')) for k in keys[:: max(1, len(keys) // SB.N_SHAPE)]]
    ok_shape = all(a.shape == (chain.SIDE_OUT, chain.SIDE_OUT) for a in shp)
    std = [float(a.std()) for a in shp]
    hc = SB.header_counts(tt, keys, kind)
    h_ok, h_diff = SB.header_gate(name, hc)
    sm = SB.sample_checks(kind, ctx['tep'], od, SB.sample_keys(keys))
    lab = df[sp.label_names[0]].to_numpy()
    by_mode = SB.orientation(od, keys, [tt[k][SB.HEADER_COLS[kind][0]] for k in keys])
    by_label = SB.orientation(od, keys, [f'nhan_{int(v)}' for v in lab])
    warn = min(list(by_mode.values()) + list(by_label.values())) < SB.ORIENT_WARN or max(by_mode.values()) - min(by_mode.values()) > 0.15
    gate = dict(du_anh=True, hinh_518=ok_shape, std_min=min(std), header=hc, header_dat=h_ok, mau=sm, chieu_muc_xam=dict(theo_goc=by_mode, theo_nhan=by_label))
    log(f"cong kho {name}: {len(keys):,} anh | 518 x 518 L ({len(shp)} mau): {'DAT' if ok_shape else 'TRUOT'} | do lech chuan muc xam min {min(std):.1f} "
        f"(> {SB.STD_MIN:g}) | header anh goc {hc if kind == 'png' else {c: hc[c] for c in ('pi', 'rows', 'cols', 'bits')}} = so dem da kiem: "
        f"{'DAT' if h_ok else 'TRUOT ' + str(h_diff)}")
    log(f"  mau {sm['n']} anh (tai lai + chuyen lai): tat dinh {sm['tat_dinh']} / {sm['n']} | = letterbox thang 518 tu anh goc: corr trung vi "
        f"{sm['corr_trung_vi']:.5f}, min {sm['corr_min']:.5f} (nguong {SB.CORR_MEDIAN_MIN} / {SB.CORR_MIN}), lech TB trung vi {sm['mad_trung_vi']:.2f} muc xam | "
        f"ti le canh dai / ngan max {sm['ti_le_max']:.3f}")
    log(f"  chieu muc xam (ti le anh giua sang hon hai ben): theo {SB.HEADER_COLS[kind][0]} goc {by_mode} | theo nhan {by_label}"
        + (' | ⚠ nhom thap / lech nhau -> gui Claude dong nay' if warn else ' | cac nhom cung chieu'))
    assert ok_shape and min(std) > SB.STD_MIN and h_ok, 'anh sai kich thuoc / gan mot mau / header anh goc khac so dem da kiem -> DUNG, gui Claude cac dong tren'
    assert all(sm['dat'].values()), f"cong mau TRUOT {[k for k, v in sm['dat'].items() if not v]} -> DUNG, gui Claude cac dong tren"
    fn = base.store_files(name)
    tab = df.copy()
    for c in SB.HEADER_COLS[kind]:
        tab[c] = [tt[k][c] for k in keys]
    d['ghi'](fn['bang'], lambda q: tab.to_parquet(q, index=False))
    from vindr_bench.data import crop_store as CS
    d['ghi'](fn['tar'], lambda q: CS.write_tar(q, od, keys))
    fp = ctx['van_tay']
    meta = dict(bo=name, ten=sp.title, run=d['RUN_TAG'], t=datetime.datetime.now().isoformat(timespec='seconds'), n=len(keys), md5_khoa=base.md5_keys(keys),
                van_tay=fp, nhan=list(sp.label_names), lop_chinh=list(sp.main_classes), don_vi_cum=sp.unit, kho=config.STORE_KEY,
                anh=dict(cach=chain.CACH[kind], canh_ra=chain.SIDE_OUT, canh_giua=1022, can_bang=False, hinh_hoc='dem_vuong', header=hc,
                         canh_bao=ctx.get('canh_bao') or {}, sang_giua=gate['chieu_muc_xam']),
                nguon=dict(thu_muc_drive='MyDrive/' + '/'.join(sp.source), kiem_toan_ven=ctx['kiem_toan_ven'], bang=ctx['nguon_bang']), cong=json.loads(json.dumps(gate, default=str)),
                goi=ctx['goi'], ban_goi=ctx['ban_goi'], ghi_chu='PLAN_93 4.2: mot kho dem vuong KHONG can bang moi bo, chuoi kho kcb cua 85. KHONG id Drive trong tep nay.')
    assert base.NO_EQ_MARK in meta['anh']['cach']
    d['ghi'](fn['meta'], lambda q: d['ghi_json'](q, meta))
    ctx.update(cong=gate, tep_ghi=list(fn.values()))
    log(f"da ghi {list(fn.values())} (local) -> {config.FOLDER}/{d['RUN_TAG']} o C6 | {(time.time() - t0) / 60:.1f} phut")
    return gate


def sync(ctx, log=print):
    '''C6. dong_bo() + kiểm các tệp của kho trên Drive.'''
    d = ctx['drive']
    du = d['dong_bo']()
    on = d['ten_tren_drive']()
    need = list(ctx.get('tep_ghi') or [])
    co = bool(need) and set(need) <= on
    log(f"C6: dong_bo {'DU' if du else 'CHUA DU'} | {len(need)} tep kho tren Drive: {'CO du' if co else 'THIEU / C5 CHUA XONG'}"
        + (f" -> gui Claude output C2-C6; roi chay 93c (chep sang corntoun1505) va cac notebook 93v_{ctx['bo']}_*" if du and co
           else ' | KHONG tat runtime; chay lai C6'))
    return dict(dong_bo=du, du_tep=co)
