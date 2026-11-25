# -*- coding: utf-8 -*-
"""Ô của notebook kho `93k_kho_chexpert` (0.5.0; PLAN_93 §6 phiên 7 — Tuấn 25/11: CheXpert chia ĐÚNG CheXFound "(A)" + 5 nhãn; Colab CPU, ƯỚC 7–10 giờ).

    C2 setup    hàm của khối Drive chuẩn (+ hàm resume) + thư mục local + phiên (chặn hai phiên cùng chạy) + đọc tiến độ từ tệp resume
    C3 sources  liệt kê theo lô hai cây (MIMIC_CXR/CheXpert: ảnh train + valid + train.csv / valid.csv; MIMIC_CXR/CheXlocalize: CheXpert/test + test_labels.csv)
                -> ba CSV md5 GHIM -> bảng chia CheXFound (datasets.chexpert.table_chexfound) -> cổng: 191.027 / 202 / 518 ảnh, ca dương test = 93j, md5 khoá test
                = tiền tố 93j, người không vắt tách, mọi khoá có ảnh trên Drive
    C4 build    KHO CHIA PHẦN (datasets.base.SHARD ảnh mỗi phần, theo thứ tự bảng): tải QUA API (6 luồng) + chuỗi kho 518 của 85 (đệm vuông, KHÔNG cân bằng) ->
                tar phần -> ghi() + luu_resume() của khối chuẩn (chế độ ghi cuối: tệp lớn đang chờ tải lên NGAY, md5) -> tiến độ nằm trong tệp resume; runtime
                chết -> mở lại, Run all: phần đã lên Drive (đúng md5) bỏ qua, chỉ làm lại phần dở. In tốc độ ĐO + giờ còn lại.
    C5 pack     TẢI LẠI mọi phần từ Drive (md5 = lúc ghi), giải nén (tên mục = khoá) -> cổng trên đúng bản notebook lượt sẽ đọc: đủ ảnh, 518 x 518, header ảnh gốc
                (mode L), mẫu tất định + hình học, chiều mức xám -> bảng + meta (danh sách phần, md5 CSV ghim, vân tay)
    C6 sync     dong_bo() + kiểm bảng / meta / mọi phần trên Drive
Không in khoá ảnh."""
import collections
import datetime
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

from .. import __version__, config
from ..datasets import base
from ..datasets import chexpert as CX
from ..store import build as SB
from ..store import chain
from ..store import listing as L
from . import sessions as SS

NAME = 'chexpert'
NOTEBOOK = '93k_kho_chexpert'
DRIVE_NAMES = ('kiem_ke', 'tai', 'xung_dot', 'ghi', 'ghi_json', 'dong_bo', 'ten_tren_drive', 'tim_moi_noi', 'khoi_phuc_cho', 'luu_resume', 'cho_nho', 'TEN_RESUME',
               'RUN_TAG', 'GHI_LOC', '_duong', '_liet_ke')
SHARD_RE = r'x518_chexpert_p\d{2}\.tar'
VERIFY_THREADS = 3                        # C5: tải lại phần trên runtime CPU (RAM ~12 GB): 3 phần ~0,45 GB cùng lúc


def _md5_file(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_tar(path, image_dir, keys):
    '''tar PHẲNG <khoá>.jpg theo thứ tự keys (= crop_store.write_tar) nhưng TẤT ĐỊNH: giờ sửa = 0, quyền 0644, chủ 0 / '' -> cùng ảnh thì cùng md5 (phần
       dựng lại ở phiên sau trùng từng byte phần cũ). base._extract đọc như tar thường.'''
    import tarfile
    with tarfile.open(path, 'w') as t:
        for k in keys:
            p = os.path.join(image_dir, f'{k}.jpg')
            ti = tarfile.TarInfo(f'{k}.jpg')
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = os.path.getsize(p), 0, 0o644, 0, 0, '', ''
            with open(p, 'rb') as f:
                t.addfile(ti, f)


def _packages():
    from .store import _packages as P
    return P()


def setup(ns, api_factory=None, log=print):
    '''C2. -> CTX.'''
    import torch
    missing = [k for k in DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    d = {k: ns[k] for k in DRIVE_NAMES}
    sp = config.dataset(NAME)
    assert d['RUN_TAG'] == sp.store_tag, f"RUN_TAG {d['RUN_TAG']!r} != {sp.store_tag!r} -> DUNG"
    colab = 'google.colab' in sys.modules
    loc = '/content/kho93_chexpert' if colab else os.path.join(tempfile.mkdtemp(), 'kho93_chexpert')
    os.makedirs(os.path.join(loc, 'anh'), exist_ok=True)
    tok = SS.token(ns, d['GHI_LOC'])
    st = dict(run=d['RUN_TAG'], t=time.strftime('%Y%m%d_%H%M%S'), phan={}, tt={}, phien=[])
    path = d['tim_moi_noi'](d['TEN_RESUME'])
    lost = []
    if path:
        got = torch.load(path, map_location='cpu', weights_only=False)
        assert got.get('run') == d['RUN_TAG'], f"tep resume cua luot KHAC ({got.get('run')}) -> DUNG"
        age = SS.resume_age_min(d['kiem_ke'], (config.FOLDER, d['RUN_TAG']), d['TEN_RESUME'])
        ok, msg = SS.lock_check(got, tok, age, force=ns.get('MO_KHOA_PHIEN') or '')
        log(f'phien: {msg}')
        assert ok, msg
        # phần ghi trong tệp resume nhưng KHÔNG có trên Drive đúng md5 (tải lên hỏng rồi runtime chết) -> bỏ khỏi tiến độ VÀ khỏi danh sách "đã lên" TRƯỚC khi
        # khôi phục (khối chuẩn nhớ tên đã lên -> ghi() cùng tên ở C4 sẽ bị chặn "ghi hai lần")
        kk = d['kiem_ke']((config.FOLDER, d['RUN_TAG']), mau=SHARD_RE)[1]
        on = {t: {b['md5'] for b in v} for t, v in kk.items()}
        phan = {t: v for t, v in (got.get('phan') or {}).items() if v['md5'] in on.get(t, set())}
        lost = sorted(set(got.get('phan') or {}) - set(phan))
        cho = dict(got.get('cho') or {})
        cho['da_len'] = {t: h for t, h in (cho.get('da_len') or {}).items() if t not in lost}
        d['khoi_phuc_cho'](cho)
        st.update(t=got['t'], phan=phan, tt=dict(got.get('tt') or {}), phien=list(got.get('phien') or []))
    st_ = os.statvfs(loc) if hasattr(os, 'statvfs') else None
    free = st_.f_bavail * st_.f_frsize / 1e9 if st_ else float('nan')
    ctx = dict(drive=d, spec=sp, loc=loc, lister=L.Lister(api_factory or L.default_api), goi=_packages(), colab=colab, tt=st, token=tok, ns=ns, t0=time.time(),
               threads=SB.threads_for())
    log(f"93k {sp.title}: kho 518 dem vuong KHONG can bang histogram (chuoi kho kcb cua 85), CHIA PHAN {base.SHARD:,} anh | {os.cpu_count()} CPU | o dia trong "
        f"{free:.0f} GB | {ctx['threads']} luong tai + chuyen anh (2 x CPU, 6..16) | goi "
        + ', '.join(f"{k} {v['phien_ban']} @ {(v['sha'] or '?')[:12]}" for k, v in ctx['goi'].items()))
    log('  ' + (f"RESUME: luot {st['t']} — {len(st['phan'])} phan da tren Drive (dung md5), {len(st['phien'])} phien truoc"
                + (f" | {len(lost)} phan ghi trong tep resume nhung KHONG thay tren Drive -> C4 lam lai" if lost else '') if path
                else 'phien dau (chua co tep resume)'))
    return ctx


def _save(ctx):
    '''Tệp resume = tiến độ (phần đã ghi + header ảnh gốc) + tệp đang chờ (cho_nho: tệp lớn — phần tar — tải lên NGAY). -> True khi lên Drive.'''
    import torch
    d, st = ctx['drive'], ctx['tt']
    ph = st['phien']
    if not ph or ph[-1].get('token') != ctx['token']:
        ph.append(dict(token=ctx['token'], bat_dau=time.strftime('%Y-%m-%d %H:%M:%S')))
    ph[-1]['ghi_cuoi'] = time.strftime('%Y-%m-%d %H:%M:%S')
    ok = d['luu_resume'](lambda q: torch.save(dict(run=st['run'], t=st['t'], phan=st['phan'], tt=st['tt'], phien=ph, cho=d['cho_nho']()), q))
    if ok and (ctx.get('ns') or {}).get('_RESUME_ID'):
        SS.prune_revisions(_api(ctx), ctx['ns']['_RESUME_ID'], print)
    return ok


def _api(ctx):
    if 'api' not in ctx:
        try:
            ctx['api'] = L.default_api()
        except Exception:
            ctx['api'] = None
    return ctx['api']


def sources(ctx, log=print):
    '''C3. Nguồn + ba CSV ghim + bảng chia CheXFound + cổng. -> bảng.'''
    d, t0 = ctx['drive'], time.time()
    summ, imgs, csvs = SB.chexpert_sources(ctx['lister'], d['_duong'], ctx['loc'], log=log)
    frames = {}
    for role, f in csvs.items():
        frames[role] = CX.read_csv(d['tai'](dict(id=f['id'], ten=f['ten'], md5=f['md5'], size=f['size'])))
        assert len(frames[role]) == CX.CSV_PINS[role][2], f"{role}: {f['ten']} {len(frames[role]):,} dong != {CX.CSV_PINS[role][2]:,} -> DUNG"
    df, info = CX.table_chexfound(frames)
    fp = base.check_table(NAME, df)
    same, diff = base.fingerprint_gate(NAME, fp)
    assert same, f'van tay bang KHAC ghim {diff} -> DUNG, gui Claude dong nay'
    keys = df.khoa.astype(str).tolist()
    miss = collections.Counter(CX.parts(k)[0] for k in keys if k not in imgs)
    assert not miss, f'anh cua bang KHONG co tren Drive theo tach {dict(miss)} -> DUNG'
    ctx.update(bang=df, khoa=keys, tep=imgs, nguon=summ, csv={r: dict(ten=f['ten'], md5=f['md5'], dong=CX.CSV_PINS[r][2], duong=f['duong']) for r, f in csvs.items()},
               bang_info=info, van_tay=fp)
    for tag, s in summ.items():
        log(f"  cay {tag}: {s['thu_muc']:,} thu muc + {s['tep']:,} tep | anh dung {s['anh']} | trung nhieu cho {s['trung_nhieu_cho']:,} | {s['lan_goi']:,} lan goi API, "
            f"{s['phut']} phut (DO)")
    log('  CSV (md5 = ban ghim): ' + ' | '.join(f"{r} {c['ten']} {c['dong']:,} dong" for r, c in ctx['csv'].items()))
    log('  bang chia CheXFound: ' + ' | '.join(f"{r} {v['thang']:,} anh thang = so dong CheXFound giu {v['chexfound_giu']:,}, {v['nguoi']:,} nguoi" for r, v in info.items()))
    sp = config.dataset(NAME)
    log(f"  so anh {[fp[s]['so_anh'] for s in config.SPLITS]} (ghim {list(sp.sizes)}) | ca duong theo nhan {list(sp.label_names)}: "
        + ' / '.join(f"{s} {fp[s]['so_duong']}" for s in config.SPLITS) + f" (test ghim {list(sp.positives[2])}) | md5 khoa / nhan: "
        + ' | '.join(f"{s} {fp[s]['md5_khoa'][:12]} / {fp[s]['md5_nhan'][:12]}" for s in config.SPLITS)
        + f" (khoa = tien to ghim {config.FINGERPRINT_PREFIX[NAME]}: DAT) | {(time.time() - t0) / 60:.1f} phut")
    return df


def _shard_on_drive(ctx):
    '''{tên phần: [md5 các bản]} trong thư mục kho trên Drive.'''
    kk = ctx['drive']['kiem_ke']((config.FOLDER, ctx['spec'].store_tag), mau=SHARD_RE)[1]
    return {t: {b['md5'] for b in v} for t, v in kk.items()}


def build(ctx, log=print):
    '''C4. Dựng kho theo phần; phần đã lên Drive (đúng md5 đã ghi) bỏ qua. -> dict tiến độ.'''
    d, st, keys = ctx['drive'], ctx['tt'], ctx['khoa']
    sl = base.shard_slices(len(keys), base.SHARD)
    names = [base.shard_name(NAME, k) for k in range(len(sl))]
    on = _shard_on_drive(ctx)
    lost = [t for t, v in st['phan'].items() if v['md5'] not in on.get(t, set())]
    for t in lost:
        st['phan'].pop(t)
    stale = [t for t, v in st['phan'].items() if t not in names or v['md5_khoa'] != base.md5_keys(keys[sl[names.index(t)][0]:sl[names.index(t)][1]])]
    assert not stale, f'tep resume ghi {len(stale)} phan KHAC bang hien tai -> DUNG (bang doi?), gui Claude output C2-C4'
    todo = [k for k, t in enumerate(names) if t not in st['phan']]
    log(f"kho chia phan: {len(names)} phan x <= {base.SHARD:,} anh | da tren Drive (dung md5) {len(names) - len(todo)} | can lam {len(todo)}"
        + (f" | {len(lost)} phan ghi trong tep resume nhung KHONG thay tren Drive -> lam lai" if lost else ''))
    t0, done_n, quiet = time.time(), 0, (lambda s='': None)
    for i, k in enumerate(todo, 1):
        a, b = sl[k]
        ks, ten = keys[a:b], names[k]
        od = os.path.join(ctx['loc'], 'anh', f'p{k:02d}')
        os.makedirs(od, exist_ok=True)
        t1, errs = time.time(), {}
        da_ghi = (ctx.get('ns') or {}).get('DA_GHI') or {}                    # tệp đã ghi() trong PHIÊN này (khối chuẩn)
        lp = os.path.join(d['GHI_LOC'], ten)
        if ten in da_ghi and os.path.exists(lp):
            # 0.5.1: tar của phần này đã ghi ở phiên này rồi bị ngắt giữa lúc tải lên -> KHÔNG ghi lại (khối chuẩn chặn "ghi hai lần"); tar còn trong hàng chờ,
            # _save() -> cho_nho() tải lên tiếp. Header lấy từ _tt.jsonl (ảnh còn trên đĩa tới khi phần lên Drive) hoặc tiến độ đã có.
            tt = SB.done(ctx['loc'], od)
            md5 = da_ghi[ten]
            log(f'  {ten}: tar da ghi o phien nay (ngat giua luc tai len) -> chi tai len tiep, khong dung lai')
        else:
            for _ in range(3):                                   # lỗi tải / giải mã từng ảnh: thử lại tối đa 3 lượt cho ảnh còn thiếu
                tt, errs, warns, _m = SB.build(ks, SB.KIND[NAME], ctx['tep'], ctx['loc'], od, log=quiet, threads=ctx.get('threads'))
                if all(x in tt for x in ks):
                    break
            assert all(x in tt for x in ks), f'{ten}: {sum(x not in tt for x in ks)} anh loi sau 3 luot (theo loai {errs}) -> chay lai C4; van loi -> gui Claude dong nay'
            lp = d['ghi'](ten, lambda q: write_tar(q, od, ks))
            md5 = _md5_file(lp)
        miss = [x for x in ks if x not in tt and x not in st['tt']]
        assert not miss, f'{ten}: thieu header anh goc cua {len(miss)} anh -> chay lai C4 sau khi Restart session; van loi -> gui Claude dong nay'
        st['phan'][ten] = dict(n=len(ks), md5_khoa=base.md5_keys(ks), md5=md5)
        st['tt'].update({x: (tt[x]['mode'], int(tt[x]['w']), int(tt[x]['h'])) for x in ks if x in tt})
        ok = _save(ctx)
        shutil.rmtree(od, ignore_errors=True)
        done_n += len(ks)
        el = time.time() - t0
        rate = done_n / el if el > 0 else float('nan')
        left = sum(sl[j][1] - sl[j][0] for j in todo[i:])
        log(f"  {ten}: {len(ks):,} anh | {(time.time() - t1) / 60:.1f} phut | tep resume + phan len Drive: {'DAT' if ok else 'CHUA (lan luu sau thu lai)'} | "
            f"{i} / {len(todo)} phan phien nay | toc do DO {rate:.1f} anh / giay | con ~{left / rate / 3600:.1f} gio (UOC theo toc do do)")
    miss = [t for t in names if t not in st['phan']]
    log(f"C4 xong: {len(names) - len(miss)} / {len(names)} phan | phien nay {(time.time() - t0) / 60:.1f} phut")
    assert not miss, f'chua du phan {len(miss)} -> chay lai C4'
    return dict(phan=len(names), lam_phien_nay=len(todo), phut=(time.time() - t0) / 60)


def _header_counts(st, keys):
    tt = st['tt']
    return {c: dict(collections.Counter(tt[k][i] for k in keys)) for i, c in enumerate(SB.HEADER_COLS[SB.KIND[NAME]])}


def pack(ctx, log=print):
    '''C5. Tải LẠI mọi phần từ Drive -> giải nén -> cổng trên đúng bản notebook lượt đọc -> bảng + meta (local; lên Drive ở C6). -> dict cổng.'''
    d, st, keys, df, sp = ctx['drive'], ctx['tt'], ctx['khoa'], ctx['bang'], ctx['spec']
    t0 = time.time()
    sl = base.shard_slices(len(keys), base.SHARD)
    names = [base.shard_name(NAME, k) for k in range(len(sl))]
    assert all(t in st['phan'] for t in names), 'chua du phan -> chay lai C4'
    d['cho_nho']()                                                # phần còn trong hàng chờ (tải lên hỏng ở C4) -> thử tải lên lần nữa trước khi kiểm
    phan = [dict(ten=t, **st['phan'][t]) for t in names]
    kk = d['kiem_ke']((config.FOLDER, sp.store_tag), mau=SHARD_RE)[1]
    vd = os.path.join(ctx['loc'], 'kiem')
    shutil.rmtree(vd, ignore_errors=True)
    os.makedirs(vd)
    shutil.rmtree(os.path.join(ctx['loc'], 'anh'), ignore_errors=True)          # ảnh dựng đã nằm trong tar (GHI_LOC + Drive)
    n_p, gb, minutes = base._fetch_shards(kk, dict(bo=NAME, co_phan=base.SHARD, phan=phan), keys, vd, (config.FOLDER, sp.store_tag), {}, None, VERIFY_THREADS,
                                          None, log)
    log(f"tai lai tu Drive: {n_p} phan, {gb:.2f} GB, {minutes:.1f} phut (DO) | md5 tung phan = luc ghi = Drive | muc trong tar = khoa cua phan")
    shp = [SB.load_gray(os.path.join(vd, f'{k}.jpg')) for k in keys[:: max(1, len(keys) // SB.N_SHAPE)]]
    ok_shape = all(a.shape == (chain.SIDE_OUT, chain.SIDE_OUT) for a in shp)
    std = [float(a.std()) for a in shp]
    hc = _header_counts(st, keys)
    h_ok, h_diff = SB.header_gate(NAME, hc)
    sm = SB.sample_checks(SB.KIND[NAME], ctx['tep'], vd, SB.sample_keys(keys))
    by_split = SB.orientation(vd, keys, df.tach.astype(str).tolist())
    anypos = (df[list(sp.label_names)].to_numpy().sum(1) > 0).astype(int)
    by_pos = SB.orientation(vd, keys, [f'co_duong_{v}' for v in anypos])
    warn = min(list(by_split.values()) + list(by_pos.values())) < SB.ORIENT_WARN
    wh = {c: sorted(hc[c].items(), key=lambda x: -x[1])[:5] for c in ('w', 'h')}
    gate = dict(du_anh=True, hinh_518=ok_shape, std_min=min(std), header=dict(mode=hc['mode']), header_dat=h_ok, kich_thuoc_goc_pho_bien=wh, mau=sm,
                chieu_muc_xam=dict(theo_tach=by_split, theo_co_duong=by_pos))
    log(f"cong kho {NAME}: {len(keys):,} anh | 518 x 518 L ({len(shp)} mau): {'DAT' if ok_shape else 'TRUOT'} | do lech chuan muc xam min {min(std):.1f} (> {SB.STD_MIN:g}) "
        f"| mode anh goc {hc['mode']} = {SB.EXPECTED_HEADER[NAME]['mode']}: {'DAT' if h_ok else 'TRUOT ' + str(h_diff)} | kich thuoc goc pho bien rong {wh['w'][:3]} cao "
        f"{wh['h'][:3]}")
    log(f"  mau {sm['n']} anh (tai lai + chuyen lai): tat dinh {sm['tat_dinh']} / {sm['n']} | = letterbox thang 518 tu anh goc: corr trung vi {sm['corr_trung_vi']:.5f}, "
        f"min {sm['corr_min']:.5f} (nguong {SB.CORR_MEDIAN_MIN} / {SB.CORR_MIN}) | ti le canh dai / ngan max {sm['ti_le_max']:.3f}")
    log(f"  chieu muc xam (ti le anh giua sang hon hai ben): theo tach {by_split} | theo co duong {by_pos}" + (' | ⚠ nhom thap -> gui Claude dong nay' if warn else ''))
    assert ok_shape and min(std) > SB.STD_MIN and h_ok, 'anh sai kich thuoc / gan mot mau / mode anh goc KHAC L -> DUNG, gui Claude cac dong tren (chi chay lai C5)'
    assert all(sm['dat'].values()), f"cong mau TRUOT {[k for k, v in sm['dat'].items() if not v]} -> DUNG, gui Claude cac dong tren"
    fn = base.store_files(NAME)
    tab = df.copy()
    for i, c in enumerate(SB.HEADER_COLS[SB.KIND[NAME]]):
        tab[c] = [st['tt'][k][i] for k in keys]
    d['ghi'](fn['bang'], lambda q: tab.to_parquet(q, index=False))
    meta = dict(bo=NAME, ten=sp.title, run=d['RUN_TAG'], t=datetime.datetime.now().isoformat(timespec='seconds'), n=len(keys), md5_khoa=base.md5_keys(keys),
                van_tay=ctx['van_tay'], nhan=list(sp.label_names), lop_chinh=list(sp.main_classes), don_vi_cum=sp.unit, kho=config.STORE_KEY,
                co_phan=base.SHARD, phan=phan, csv=ctx['csv'], bang=ctx['bang_info'],
                chia='CheXFound (Tuan 25/11, (A)): train = anh thang train.csv, val = anh thang valid.csv, test = anh thang test_labels.csv (CheXlocalize); '
                     'nhan 1 -> 1, 0 / -1 / trong -> 0; 5 nhan cuoc thi',
                anh=dict(cach=chain.CACH[SB.KIND[NAME]], canh_ra=chain.SIDE_OUT, canh_giua=1022, can_bang=False, hinh_hoc='dem_vuong', header=dict(mode=hc['mode']),
                         sang_giua=gate['chieu_muc_xam']),
                nguon=dict(thu_muc_drive=['MyDrive/' + '/'.join(CX.SOURCE), 'MyDrive/' + '/'.join(config.CHEXPERT_TEST_SOURCE)], cay=ctx['nguon']),
                cong=json.loads(json.dumps(gate, default=str)), phien=st['phien'], goi=ctx['goi'], phien_ban_goi=__version__,
                ghi_chu='PLAN_93 4.2 + 6 phien 7: mot kho dem vuong KHONG can bang, chuoi kho kcb cua 85, chia phan. KHONG id Drive trong tep nay.')
    assert base.NO_EQ_MARK in meta['anh']['cach']
    d['ghi'](fn['meta'], lambda q: d['ghi_json'](q, meta))
    ctx.update(cong=gate, tep_ghi=[fn['bang'], fn['meta']] + names)
    log(f"da ghi {fn['bang']} + {fn['meta']} (local) -> Drive o C6 | C5 {(time.time() - t0) / 60:.1f} phut")
    return gate


def sync(ctx, log=print):
    '''C6. dong_bo() + kiểm bảng / meta / mọi phần trên Drive.'''
    d = ctx['drive']
    du = d['dong_bo']()
    on = d['ten_tren_drive']()
    need = list(ctx.get('tep_ghi') or [])
    co = bool(need) and set(need) <= on
    log(f"C6: dong_bo {'DU' if du else 'CHUA DU'} | {len(need)} tep kho tren Drive (bang + meta + {len(need) - 2} phan): {'CO du' if co else 'THIEU / C5 CHUA XONG'} | "
        f"ca luot phien nay {(time.time() - ctx['t0']) / 60:.1f} phut"
        + (' -> gui Claude output C2-C6; roi chay 93c (chep sang corntoun1505) va 93v_chexpert_chexzero' if du and co else ' | KHONG tat runtime; chay lai C6'))
    return dict(dong_bo=du, du_tep=co)
