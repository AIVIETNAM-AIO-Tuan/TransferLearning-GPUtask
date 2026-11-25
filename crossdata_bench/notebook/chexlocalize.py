# -*- coding: utf-8 -*-
"""Ô của notebook kiểm kê `93j_kiem_ke_chexlocalize` (Colab CPU, CHỈ ĐỌC; PLAN_93 §6 phiên 5; Tuấn 25/11: chia CheXpert theo CheXFound -> cần bộ test chính
thức, nằm trong CheXlocalize ở MyDrive/MIMIC_CXR/CheXlocalize).

    C2  setup      hàm của khối Drive chuẩn + thư mục CheXlocalize + thư mục batch 1 của CheXpert (valid.csv / train.csv / ảnh valid) qua API
    C3  tree       = inventory.tree (cả cây theo lô, khung cây, loại tệp, ảnh theo khoá) + danh mục zip trong cây (nếu còn zip): số ảnh theo tách, tên CSV
    C4  csvs       = inventory.csvs (mọi CSV, vai theo cột Path) — cần vai test (test_labels.csv) + valid (val_labels.csv)
    C5  match      = inventory.match (CSV <-> ảnh trên Drive) + số test / val so với bài (668 / 500 / 500; 234 / 200) và CheXFound (518 / 202 ảnh thẳng)
    C6  reference  nhãn test <-> groundtruth.csv của GitHub (commit + md5 ghim) theo lần chụp, 14 nhãn; val CheXlocalize <-> valid CheXpert (nhãn + md5 từng
                   ảnh); người test không có ở train / valid của CheXpert
    C7  images     MỌI ảnh test + val: tải, md5 = Drive, giải mã được (mode, kích thước); 6 luồng; đo thời gian
    C8  write      bảng test + bảng val (khoá, người, lần chụp, thẳng / nghiêng, 14 nhãn, md5) + số ca dương theo ảnh thẳng + JSON + danh sách tệp (không ID
                   Drive) -> dong_bo() -> phán quyết từng mục
Không ghi / xoá gì trong thư mục dữ liệu. Không in khoá ảnh (chỉ số đếm + khoảng mã người)."""
import collections
import datetime
import hashlib
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import __version__
from ..datasets import base
from ..datasets import chexlocalize as CL
from ..datasets import chexpert as CX
from ..store import listing as L
from ..store import zip_remote as ZR
from . import inventory as INV

NOTEBOOK = '93j_kiem_ke_chexlocalize'
RUN_TAG = 'r93j_kiem_ke_chexlocalize'
OUT = dict(json='kiem_ke_chexlocalize.json', test='chexlocalize_test_bang.parquet', val='chexlocalize_val_bang.parquet', ds='chexlocalize_danh_sach_drive.parquet')
ROLES = ('valid', 'test')


def setup(ns, api_factory=None, log=print):
    '''C2. -> CTX.'''
    missing = [k for k in INV.DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    d = {k: ns[k] for k in INV.DRIVE_NAMES}
    assert d['RUN_TAG'] == RUN_TAG, f"RUN_TAG {d['RUN_TAG']!r} != {RUN_TAG!r} -> DUNG"
    root = d['_duong'](CL.SOURCE)
    assert root, f"khong thay thu muc MyDrive/{'/'.join(CL.SOURCE)} qua API -> soat lai ten (hoa / thuong, dau cach, gach duoi) roi bao Claude -> DUNG"
    b1 = d['_duong'](CX.SOURCE + (CL.CHEXPERT_B1,))
    loc = '/content/kk93_chexlocalize' if 'google.colab' in sys.modules else tempfile.mkdtemp(prefix='kk93j_')
    os.makedirs(loc, exist_ok=True)
    from .store import _packages
    ctx = dict(drive=d, root=root, b1=b1, lister=L.Lister(api_factory or L.default_api), loc=loc, goi=_packages(), t0=time.time(), kq={})
    log(f"93j kiem ke CheXlocalize: thu muc MyDrive/{'/'.join(CL.SOURCE)} THAY qua API | goi " + ', '.join(
        f"{k} {v['phien_ban']} @ {(v['sha'] or '?')[:12]}" for k, v in ctx['goi'].items()) + f' | {os.cpu_count()} CPU')
    log(f"  thu muc MyDrive/{'/'.join(CX.SOURCE)}/{CL.CHEXPERT_B1}: " + ('THAY (C6 so val + nguoi voi CheXpert)' if b1 else
                                                                         'KHONG thay -> C6 khong so duoc voi CheXpert (canh bao)'))
    return ctx


def tree(ctx, log=print):
    '''C3. inventory.tree + danh mục zip trong cây (chỉ đọc danh mục từ xa, không giải nén).'''
    summ = INV.tree(ctx, log=log)
    rows = []
    for a in sorted((f for f in ctx['cay']['files'] if INV.ARCHIVE.search(f['ten'])), key=lambda f: f['duong']):
        row = dict(duong=a['duong'], gb=round(a['size'] / 1e9, 2))
        if not a['ten'].lower().endswith('.zip'):
            row['loi'] = 'khong phai .zip (khong doc danh muc tu xa duoc) — chi liet ke'
        else:
            try:
                zf, rf = ZR.open_remote(lambda x, y, i=a['id']: ctx['lister'].fetch_range(i, x, y), a['size'])
                ents = ZR.listing(zf)
                ks = [CX.key_of(n) for n, _, _ in ents]
                row.update(muc=len(ents), anh_theo_tach=dict(collections.Counter(CX.parts(k)[0] for k in ks if k)),
                           csv=sorted({n.rpartition('/')[2] for n, _, _ in ents if n.lower().endswith('.csv')})[:20])
            except Exception as e:
                row['loi'] = f'{type(e).__name__}: {str(e)[:160]}'
        rows.append(row)
    if rows:
        log(f'  tep nen trong cay: {len(rows)}')
        for r in rows:
            log(f"    {r['duong']} ({r['gb']} GB) | " + (r.get('loi') or f"{r['muc']:,} muc | anh theo tach {r['anh_theo_tach']} | CSV {r['csv']}"))
        if not any(CX.parts(k)[0] == 'test' for k in ctx['anh']):
            log('  ⚠ CHUA co anh test giai nen tren Drive — anh test (neu co) chi nam trong zip o tren -> gui Claude output C3')
    ctx['kq'] = dict(ctx['kq'], zip=rows)
    return summ


def csvs(ctx, log=print):
    '''C4. inventory.csvs, chỉ đòi vai valid + test.'''
    return INV.csvs(ctx, log=log, expect=ROLES)


def match(ctx, log=print):
    '''C5. inventory.match + số đếm test / val (CSV và Drive) so với bài và CheXFound.'''
    out = INV.match(ctx, log=log)
    ch, fr, anh, cnt = ctx.get('csv_chon') or {}, ctx.get('csv_bang') or {}, ctx['anh'], {}
    for role in ROLES:
        ks = {k for k in anh if CX.parts(k)[0] == role}
        cnt[role] = dict(csv=CL.counts(fr[ch[role]['duong']], role) if role in ch else None,
                         drive=dict(anh=len(ks), nguoi=len({CX.parts(k)[1] for k in ks}), lan_chup=len({CL.study_of(k) for k in ks}),
                                    thang=int(sum(CX.is_frontal(k) for k in ks))))
        c, p = cnt[role]['csv'], CL.PAPER[role]
        cf = CL.CHEXFOUND['test' if role == 'test' else 'val']
        log(f"  {role}: CSV " + (f"{c['anh']:,} anh / {c['nguoi']:,} nguoi / {c['lan_chup']:,} lan chup / {c['thang']:,} anh thang" if c else 'KHONG co')
            + f" | Drive {cnt[role]['drive']['anh']:,} anh / {cnt[role]['drive']['thang']:,} thang | theo bai {p['anh']} anh / {p['nguoi']} nguoi"
            + (f" / {p['lan_chup']} lan chup" if 'lan_chup' in p else '') + f" | CheXFound {cf} anh thang")
    ctx['kq'] = dict(ctx['kq'], so_dem=cnt)
    return dict(doi_chieu=out, so_dem=cnt)


def reference(ctx, fetch=None, md5=None, log=print):
    '''C6. (a) nhãn test <-> groundtruth.csv (GitHub, commit + md5 ghim; None -> hằng của datasets.chexlocalize LÚC GỌI); (b) CheXpert batch 1: người test
       không có ở train.csv / valid.csv; val_labels.csv <-> valid.csv (nhãn) và ảnh val <-> ảnh valid (md5 từng khoá).'''
    fetch = fetch or CL.fetch
    md5 = CL.GT_MD5 if md5 is None else md5
    d, ch, fr, r = ctx['drive'], ctx.get('csv_chon') or {}, ctx.get('csv_bang') or {}, {}
    t0 = time.time()
    try:
        data = fetch()
        got = hashlib.md5(data).hexdigest()
        r['gt'] = dict(md5_dung=got == md5, byte=len(data), commit=CL.GT_SHA[:12])
        if got == md5:
            gt = CL.read_groundtruth(data)
            r['gt'].update(dong=gt['dong'], lan_chup=len(gt['lan_chup']), sai_dang=gt['sai_dang'], trung=gt['trung'], thieu_cot=gt['thieu_cot'],
                           duong_theo_lan_chup={c: sum(v[c] == '1' for v in gt['lan_chup'].values()) for c in CX.LABELS})
            if 'test' in ch:
                r['so_test'] = CL.compare_test(fr[ch['test']['duong']], gt)
    except Exception as e:
        r['gt'] = dict(loi=f'{type(e).__name__}: {str(e)[:160]}')
    g = r['gt']
    log(f"groundtruth.csv (GitHub {CL.GT_REPO} @ {CL.GT_SHA[:12]}): " + (g.get('loi') and f"KHONG tai duoc ({g['loi']}) -> chay lai C6" or (
        f"md5 {'= ban ghim' if g['md5_dung'] else 'KHAC ban ghim -> KHONG so'} ({g['byte']:,} byte)"
        + (f" | {g['dong']} dong, {g['lan_chup']} lan chup, sai dang {g['sai_dang']}, trung {g['trung']}" if g['md5_dung'] else ''))))
    s = r.get('so_test')
    if s:
        log(f"  nhan test <-> groundtruth theo lan chup: {s['anh']:,} anh CSV ({s['anh_co_tham_chieu']:,} co lan chup trong groundtruth) | lan chup CSV "
            f"{s['lan_chup_csv']} / groundtruth {s['lan_chup_gt']} | chi o CSV {s['chi_o_csv']}, chi o groundtruth {s['chi_o_gt']} | anh LECH nhan {s['anh_lech']}"
            + (f" {s['lech_theo_nhan']}" if s['anh_lech'] else '') + f" | gia tri khac 0 / 1 {s['gia_tri_khac_0_1'] or 0}"
            + (f" | thieu cot {s['thieu_cot']}" if s['thieu_cot'] else ''))
    if ctx['b1']:
        w = ctx['lister'].walk(ctx['b1'], log=lambda m: None)
        csvf = {f['ten']: f for f in w['files'] if f['tang'] == 1 and f['ten'] in ('train.csv', 'valid.csv')}
        cx = {n: CX.read_csv(d['tai'](dict(id=f['id'], ten=f['ten'], md5=f['md5'], size=f['size']))) for n, f in sorted(csvf.items())}
        cx_img = {}
        for f in w['files']:
            k = CX.key_of(f['duong'])
            if k and CX.parts(k)[0] == 'valid':
                cx_img.setdefault(k, f)
        r['chexpert'] = dict(csv=sorted(cx), anh_valid=len(cx_img))
        if 'test' in ch and cx:
            tp = CL.patients(fr[ch['test']['duong']])
            r['nguoi_chung_test'] = {n: len(tp & CL.patients(df)) for n, df in cx.items()}
        if 'valid' in ch and 'valid.csv' in cx:
            r['so_val'] = CL.compare_val(fr[ch['valid']['duong']], cx['valid.csv'])
        lx = {k: f for k, f in ctx['anh'].items() if CX.parts(k)[0] == 'valid'}
        com = set(lx) & set(cx_img)
        r['val_anh'] = dict(chexlocalize=len(lx), chexpert=len(cx_img), chung=len(com), md5_trung=int(sum(lx[k]['md5'] == cx_img[k]['md5'] for k in com)),
                            chi_o_chexlocalize=len(set(lx) - set(cx_img)), chi_o_chexpert=len(set(cx_img) - set(lx)))
        log(f"  CheXpert ({CL.CHEXPERT_B1}): CSV {r['chexpert']['csv']} | anh valid {len(cx_img)}")
        if 'nguoi_chung_test' in r:
            log(f"  nguoi test co trong CSV CheXpert: {r['nguoi_chung_test']}")
        if 'so_val' in r:
            v = r['so_val']
            log(f"  nhan val <-> valid.csv CheXpert ({v['nhan_so']} nhan): chung {v['chung']} anh | chi o CheXlocalize {v['chi_o_chexlocalize']}, chi o CheXpert "
                f"{v['chi_o_chexpert']} | anh LECH {v['anh_lech']}" + (f" {v['lech_theo_nhan']}" if v['anh_lech'] else ''))
        a = r['val_anh']
        log(f"  anh val CheXlocalize {a['chexlocalize']} <-> anh valid CheXpert {a['chexpert']}: chung {a['chung']}, md5 trung {a['md5_trung']} / {a['chung']} | chi o "
            f"CheXlocalize {a['chi_o_chexlocalize']}, chi o CheXpert {a['chi_o_chexpert']}")
    else:
        r['chexpert'] = None
        log(f"  KHONG thay MyDrive/{'/'.join(CX.SOURCE)}/{CL.CHEXPERT_B1} -> khong so nguoi / val voi CheXpert")
    r['giay'] = round(time.time() - t0)
    ctx['kq'] = dict(ctx['kq'], tham_chieu=r)
    return r


def images(ctx, log=print):
    '''C7. MỌI ảnh test + val trên Drive: tải, md5 = Drive, giải mã được (mode, kích thước). 6 luồng; đo thời gian.'''
    lister, anh, t0 = ctx['lister'], ctx['anh'], time.time()
    ks = sorted(k for k in anh if CX.parts(k)[0] in ROLES)
    with ThreadPoolExecutor(lister.threads) as pool:
        res = list(pool.map(lambda k: INV._check_one(lister, anh[k], None), ks))
    out = {}
    for role in ROLES:
        rr = [(k, x) for k, x in zip(ks, res) if CX.parts(k)[0] == role]
        w = [x['wh'][0] for _, x in rr if x.get('wh')]
        h = [x['wh'][1] for _, x in rr if x.get('wh')]
        bad = [k for k, x in rr if not x.get('md5_dung') or 'mode' not in x]
        out[role] = dict(n=len(rr), md5_dung=int(sum(bool(x.get('md5_dung')) for _, x in rr)), giai_ma=int(sum('mode' in x for _, x in rr)),
                         khong_0_byte=int(sum(anh[k]['size'] > 0 for k, _ in rr)), mode=dict(collections.Counter(x['mode'] for _, x in rr if 'mode' in x)),
                         loi=dict(collections.Counter(x['loi'] for _, x in rr if 'loi' in x)), rong=[min(w), int(np.median(w)), max(w)] if w else None,
                         cao=[min(h), int(np.median(h)), max(h)] if h else None, gb=round(sum(anh[k]['size'] for k, _ in rr) / 1e9, 3),
                         hong_nguoi=INV._people_range(bad) if bad else '')
        o = out[role]
        log(f"{role}: {o['n']} anh ({o['gb']} GB) | md5 = Drive {o['md5_dung']} / {o['n']} | giai ma {o['giai_ma']} / {o['n']} | khong 0 byte {o['khong_0_byte']} / "
            f"{o['n']} | mode {o['mode']} | rong min / trung vi / max {o['rong']} | cao {o['cao']} | loi {o['loi'] or 0}" + (f" ({o['hong_nguoi']})" if bad else ''))
    out['giay'] = round(time.time() - t0)
    log(f"  thoi gian C7 (DO): {out['giay']} giay, {lister.threads} luong, thu lai API {lister.retried}")
    ctx['kq'] = dict(ctx['kq'], anh=out)
    return out


def verdict(ctx):
    '''-> [(mục, đạt, ghi chú)]; đạt None = chỉ cảnh báo.'''
    kq, ch, out = ctx['kq'], ctx.get('csv_chon') or {}, []
    cay = kq.get('cay') or {}
    out.append(('cay duyet het', cay.get('chua_duyet', 0) == 0, f"{cay.get('chua_duyet', 0)} thu muc chua duyet"))
    out.append(('CSV test (bac si gan)', 'test' in ch, ch['test']['duong'] if 'test' in ch else 'khong co'))
    out.append(('CSV val', True if 'valid' in ch else None, ch['valid']['duong'] if 'valid' in ch else 'khong co'))
    dc = (kq.get('doi_chieu') or {}).get('theo_vai') or {}
    for role in ROLES:
        if role in dc:
            v = dc[role]
            out.append((f'anh cua CSV {role} co du tren Drive', v['thieu'] == 0 if role == 'test' else (True if v['thieu'] == 0 else None),
                        f"thieu {v['thieu']:,}, tren Drive ngoai CSV {v['thua_tren_drive']:,}"))
    sd = kq.get('so_dem') or {}
    c = (sd.get('test') or {}).get('csv')
    p = CL.PAPER['test']
    out.append(('test = so cua bai (668 anh / 500 nguoi / 500 lan chup)', bool(c) and (c['anh'], c['nguoi'], c['lan_chup']) == (p['anh'], p['nguoi'], p['lan_chup']),
                f"{c['anh']} / {c['nguoi']} / {c['lan_chup']}" if c else 'khong co CSV test'))
    out.append(('anh thang test = 518 (CheXFound)', None if not c or c['thang'] != CL.CHEXFOUND['test'] else True, f"{c['thang']} anh thang" if c else '-'))
    tc = kq.get('tham_chieu') or {}
    g, s = tc.get('gt') or {}, tc.get('so_test')
    if 'loi' in g or not g.get('md5_dung') or s is None:
        out.append(('nhan test = groundtruth.csv (GitHub, theo lan chup, 14 nhan)', False, g.get('loi') or ('md5 khac ban ghim' if g and not g.get('md5_dung') else
                                                                                                    'khong co CSV test')))
    else:
        out.append(('nhan test chi 0 / 1, du 14 cot', not s['gia_tri_khac_0_1'] and not s['thieu_cot'], f"khac 0 / 1 {s['gia_tri_khac_0_1'] or 0}, thieu cot {s['thieu_cot']}"))
        okg = (s['anh_lech'] == 0 and s['chi_o_csv'] == 0 and s['chi_o_gt'] == 0 and s['anh_co_tham_chieu'] == s['anh'] and g['lan_chup'] == p['lan_chup']
               and g['sai_dang'] == 0 and g['trung'] == 0)
        out.append(('nhan test = groundtruth.csv (GitHub, theo lan chup, 14 nhan)', okg, f"anh lech {s['anh_lech']}, lan chup chi o CSV {s['chi_o_csv']} / chi o "
                                                                                       f"groundtruth {s['chi_o_gt']}, groundtruth {g['lan_chup']} lan chup"))
    if tc.get('chexpert') is None:
        out.append(('so voi CheXpert (nguoi, val)', None, 'khong thay thu muc batch 1 cua CheXpert'))
    else:
        nc = tc.get('nguoi_chung_test')
        out.append(('nguoi test khong co o train / valid CheXpert', None if nc is None else not any(nc.values()), str(nc) if nc is not None else 'khong doc duoc CSV'))
        sv, va = tc.get('so_val'), tc.get('val_anh') or {}
        same = (sv is not None and sv['anh_lech'] == 0 and sv['chi_o_chexlocalize'] == 0 and sv['chi_o_chexpert'] == 0 and va.get('chung') == va.get('md5_trung')
                and va.get('chi_o_chexlocalize') == 0 and va.get('chi_o_chexpert') == 0 and va.get('chung', 0) > 0)
        out.append(('val CheXlocalize = valid CheXpert (cung anh, cung md5, cung nhan)', True if same else None,
                    f"nhan: {sv and dict(chung=sv['chung'], lech=sv['anh_lech'])} | anh: chung {va.get('chung')}, md5 trung {va.get('md5_trung')}, chi o CheXlocalize "
                    f"{va.get('chi_o_chexlocalize')}, chi o CheXpert {va.get('chi_o_chexpert')}"))
    a = kq.get('anh') or {}
    if a:
        ok = all(a[r]['md5_dung'] == a[r]['giai_ma'] == a[r]['khong_0_byte'] == a[r]['n'] for r in ROLES) and a['test']['n'] > 0
        out.append(('moi anh test + val: md5 = Drive, giai ma duoc, khong 0 byte', ok, ' | '.join(f"{r} {a[r]['md5_dung']} / {a[r]['giai_ma']} / {a[r]['n']}" for r in ROLES)))
    else:
        out.append(('anh test + val giai ma', False, 'chua chay C7'))
    out.append(('khoa trung khac noi dung', cay.get('khoa_trung_khac_noi_dung', 0) == 0, str(cay.get('khoa_trung_khac_noi_dung', 0))))
    out.append(('anh khong dung dang CheXpert (anh khac trong cay)', True if cay.get('anh_sai_dang', 0) == 0 else None, f"{cay.get('anh_sai_dang', 0):,} anh"))
    return out


def write(ctx, log=print):
    '''C8. Bảng test + bảng val + số ca dương theo ảnh thẳng + JSON + danh sách tệp -> dong_bo() -> phán quyết.'''
    import pandas as pd
    d, ch, fr, anh = ctx['drive'], ctx.get('csv_chon') or {}, ctx.get('csv_bang') or {}, ctx['anh']
    tabs = {role: CL.image_table(fr[ch[role]['duong']], role, anh) for role in ROLES if role in ch}
    if 'test' in tabs:
        t = tabs['test']
        pos, pos_all = CL.positives(t), CL.positives(t, frontal_only=False)
        cs = [c for c in CX.LABELS if pos[c] >= CX.MIN_TEST_POS]
        gpos = ((ctx['kq'].get('tham_chieu') or {}).get('gt') or {}).get('duong_theo_lan_chup') or {}
        fp = base.fingerprint_of(t.khoa[t.thang].tolist(), t.tach[t.thang].to_numpy(), t.loc[t.thang, list(CX.LABELS)].to_numpy())['test']
        ctx['kq'] = dict(ctx['kq'], bang_test=dict(anh=int(len(t)), thang=int(t.thang.sum()), nguoi=int(t.nhom.nunique()), duong_anh_thang=pos, duong_moi_anh=pos_all,
                                                   tap_lop_10=cs, nhan_cuoc_thi=list(CX.COMPETITION), van_tay_anh_thang=dict(md5_khoa=fp['md5_khoa'], md5_nhan=fp['md5_nhan'])))
        log(f"BANG TEST: {len(t):,} anh ({int(t.thang.sum()):,} thang, {t.nhom.nunique():,} nguoi) | so ca DUONG: anh thang / moi anh / lan chup (groundtruth) — "
            f"[tap lop] = >= {CX.MIN_TEST_POS} anh thang duong; [cuoc thi] = 5 nhan cua CheXFound:")
        for c in CX.LABELS:
            log(f"    {c:<27}{pos[c]:>5,} / {pos_all[c]:>5,} / {gpos.get(c, '-'):>4}" + ('  [tap lop]' if c in cs else '') + ('  [cuoc thi]' if c in CX.COMPETITION else ''))
        log(f"  tap lop (>= {CX.MIN_TEST_POS} anh thang duong): {len(cs)} nhan | van tay anh thang (md5 khoa / md5 nhan, 14 cot) {fp['md5_khoa'][:12]} / {fp['md5_nhan'][:12]}")
    v = verdict(ctx)
    ok = all(x[1] is not False for x in v)
    js = dict(notebook=NOTEBOOK, run=d['RUN_TAG'], t=datetime.datetime.now().isoformat(timespec='seconds'), phien_ban=__version__, goi=ctx['goi'],
              nguon='MyDrive/' + '/'.join(CL.SOURCE), tham_chieu_nhan=dict(repo=CL.GT_REPO, commit=CL.GT_SHA, md5=CL.GT_MD5), ket_qua=ctx['kq'],
              phan_quyet=[dict(muc=a, dat=b, ghi_chu=c) for a, b, c in v], dat=ok, phut=round((time.time() - ctx['t0']) / 60, 1),
              ghi_chu='PLAN_93 6 phien 5. KHONG ID Drive trong tep nay; bang test / val = du lieu kiem ke, cach chia (A / B) + tap nhan chua khoa.')
    names = [OUT['json'], OUT['ds']]
    d['ghi'](OUT['json'], lambda q: d['ghi_json'](q, js))
    ds = pd.DataFrame([dict(duong=f['duong'], byte=f['size'], md5=f['md5']) for f in ctx['cay']['files']])
    d['ghi'](OUT['ds'], lambda q: ds.to_parquet(q, index=False))
    for role, key in (('test', 'test'), ('valid', 'val')):
        if role in tabs:
            d['ghi'](OUT[key], lambda q, t=tabs[role]: t.to_parquet(q, index=False))
            names.append(OUT[key])
    du = d['dong_bo']()
    co = set(names) <= d['ten_tren_drive']()
    log(f"C8: dong_bo {'DU' if du else 'CHUA DU'} | {len(names)} tep tren Drive: {'CO du' if co else 'THIEU -> chay lai C8, KHONG tat runtime'} | tong {js['phut']} phut")
    log('PHAN QUYET KIEM KE CheXlocalize:')
    for a, b, c in v:
        log(f"  {'DAT  ' if b else ('CANH BAO' if b is None else 'TRUOT')} {a} | {c}")
    log(f"=> {'KIEM KE DAT' if ok else 'KIEM KE CHUA DAT'}" + ('' if ok else ' (muc TRUOT o tren) -> gui Claude output C2-C8'))
    ctx['bang_test'] = tabs.get('test')
    res = dict(dat=ok, dong_bo=du, du_tep=co, phan_quyet=v)
    ctx['ket_qua'] = res
    return res
