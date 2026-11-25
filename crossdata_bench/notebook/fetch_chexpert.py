# -*- coding: utf-8 -*-
"""Ô của notebook `93h_tai_chexpert_chexlocalize` (Colab CPU, GHI; PLAN_93 §6 phiên 6; Tuấn 25/11 "làm cách 1 đi"): tải thư mục CheXpert/ (ảnh val + test
chính thức + 2 CSV nhãn) của bộ CheXlocalize từ trang AIMI CŨ trên Azure (link SAS của nút Download) THẲNG vào MyDrive/MIMIC_CXR/CheXlocalize/CheXpert/,
giữ nguyên cấu trúc -> notebook 93j (0.4.2) chạy lại y nguyên.

    C2  setup + link   khối Drive chuẩn + thư mục CheXlocalize + tài khoản API có phải CHỦ không; nhập link qua ô ẩn (getpass: không hiện, không lưu trong
                       notebook, không in) -> kiểm https / *.blob.core.windows.net / chữ ký / quyền đọc + liệt kê / còn hạn -> thử liệt kê
    C3  plan           liệt kê cả container (đo thời gian) -> phân loại tệp dưới CheXpert/ (ảnh theo khoá CheXpert, 2 CSV, rác .DS_Store, mục thư mục giả, tệp
                       khác) -> so với Drive (đã có đúng -> bỏ qua: chạy lại sau đứt; có sẵn mà khác -> không ghi đè) -> CỔNG trước khi tải: 234 ảnh val +
                       668 ảnh test + 2 CSV, 0 ảnh sai dạng / trùng khoá / xung đột -> chưa đạt thì C4 KHÔNG tải
    C4  copy           tạo thư mục còn thiếu theo tầng -> mỗi tệp: tải từ Azure (số byte + Content-MD5 nếu có) -> ảnh phải giải mã được -> tải lên Drive
                       (Lister.upload: md5 máy chủ = dữ liệu, thử lại không đẻ bản trùng); 6 luồng; tiến độ + thời gian ĐO; chạy lại ô: chỉ làm tệp còn thiếu
    C5  verify         liệt kê lại CheXpert/ trên Drive -> từng tệp: đúng 1 bản, số byte = Azure, md5 = dữ liệu tải (Azure không có md5 và tệp không tải trong
                       runtime này -> tải lại từ Azure để so) -> JSON + bảng tệp (không link, không ID Drive) -> dong_bo() -> phán quyết
Chỉ GHI vào MyDrive/MIMIC_CXR/CheXlocalize/CheXpert/ (tạo thư mục + tải tệp lên); không xoá / ghi đè gì. Không in khoá ảnh (số đếm, dạng tên, khoảng mã
người). Link SAS chỉ nằm trong bộ nhớ (CTX['sas'], repr không có chữ ký); mọi thông báo lỗi đã che."""
import collections
import datetime
import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .. import __version__
from ..datasets import chexlocalize as CL
from ..datasets import chexpert as CX
from ..store import azure_blob as AB
from ..store import listing as L
from . import inventory as INV

NOTEBOOK = '93h_tai_chexpert_chexlocalize'
RUN_TAG = 'r93h_tai_chexpert_chexlocalize'
OUT = dict(json='tai_chexpert_chexlocalize.json', ds='chexpert_chexlocalize_tep.parquet')
ROLES = ('valid', 'test')
JUNK = ('.DS_Store', 'Thumbs.db', 'desktop.ini')
BIG = 50 << 20                                              # tệp "khác" lớn hơn: không tải (chỉ liệt kê)
MIME = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.csv': 'text/csv', '.json': 'application/json'}
PROMPT = 'Dan link tai (SAS) cua trang AIMI cu roi Enter — link KHONG hien, KHONG luu: '


def setup(ns, api_factory=None, log=print):
    '''C2 (phần 1). -> CTX.'''
    missing = [k for k in INV.DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    d = {k: ns[k] for k in INV.DRIVE_NAMES}
    assert d['RUN_TAG'] == RUN_TAG, f"RUN_TAG {d['RUN_TAG']!r} != {RUN_TAG!r} -> DUNG"
    root = d['_duong'](CL.SOURCE)
    assert root, f"khong thay thu muc MyDrive/{'/'.join(CL.SOURCE)} qua API -> soat lai ten roi bao Claude -> DUNG"
    from .store import _packages
    lst = L.Lister(api_factory or L.default_api)
    try:
        info = lst.get(root)
        chu, them, qloi = bool(info.get('ownedByMe')), bool((info.get('capabilities') or {}).get('canAddChildren')), None
    except Exception as e:
        chu, them, qloi = False, False, type(e).__name__
    ctx = dict(drive=d, root=root, lister=lst, goi=_packages(), t0=time.time(), kq={}, chu=chu, them_con=them, sas=None, xong={}, fetch=AB.http_get,
               sleep=time.sleep, allow_host=None)
    log(f"93h tai CheXpert/ cua CheXlocalize: thu muc MyDrive/{'/'.join(CL.SOURCE)} THAY qua API | goi " + ', '.join(
        f"{k} {v['phien_ban']} @ {(v['sha'] or '?')[:12]}" for k, v in ctx['goi'].items()))
    log(f"  tai khoan API {'LA' if chu else 'KHONG phai'} CHU thu muc CheXlocalize (them tep: {'duoc' if them else 'khong'}" + (f'; loi doi quyen {qloi}' if qloi else '')
        + ') -> ' + ('C4 ghi duoc' if chu and them else 'C4 se KHONG ghi (tep tai len se thuoc tai khoan nay) -> ngat runtime, chay lai, chon tai khoan CHU o C0'))
    return ctx


def link(ctx, url=None, ask=None, now=None, log=print):
    '''C2 (phần 2). Nhập link SAS qua ô ẩn -> kiểm -> thử liệt kê. Sai -> dừng ô (thông báo KHÔNG chứa link).'''
    ctx['sas'] = None
    if url is None:
        import getpass
        url = (ask or getpass.getpass)(PROMPT)
    try:
        sas = AB.parse(url, allow_host=ctx.get('allow_host'))
    except ValueError as e:
        raise AssertionError(f'link KHONG dung: {e} -> chay lai C2') from None
    del url
    i = AB.info(sas, now)
    log(f"link: may chu {i['may_chu']} | container {i['container']} | loai tai nguyen {i['sr']} | quyen {i['sp']} (doc: {'co' if i['doc'] else 'KHONG'}, liet ke: "
        f"{'co' if i['liet_ke'] else 'KHONG'}) | hieu luc {i['bat_dau'] or '-'} -> {i['het_han'] or '-'}" + (f" (con {i['con_gio']:,.0f} gio)" if i['con_gio'] is not None else ''))
    bad = []
    if i['sr'] != 'c':
        bad.append(f"loai tai nguyen {i['sr']!r} != 'c' (can link cho ca container)")
    if not i['doc']:
        bad.append('thieu quyen doc (r)')
    if not i['liet_ke']:
        bad.append('thieu quyen liet ke (l)')
    if i['het_han_roi']:
        bad.append(f"link HET HAN luc {i['het_han']} -> lay link moi o trang AIMI cu")
    if i['chua_toi_gio']:
        bad.append(f"link chua toi gio dung ({i['bat_dau']})")
    if i['het_han'] and i['con_gio'] is None:
        bad.append('khong doc duoc han dung (se)')
    assert not bad, 'link KHONG dung: ' + '; '.join(bad) + ' -> chay lai C2'
    t = time.time()
    try:
        first = AB.list_blobs(sas, CL.AZ_PREFIX, fetch=ctx['fetch'], sleep=ctx['sleep'], page=5, max_pages=1)
    except AB.AzureError as e:
        raise AssertionError(f'thu liet ke Azure THAT BAI: {e}') from None
    ctx['sas'] = sas
    ctx['kq'] = dict(ctx['kq'], link=i, thu_liet_ke=len(first))
    log(f"  thu liet ke '{CL.AZ_PREFIX}': " + (f'THAY tep ({time.time() - t:.1f} giay) -> link dung duoc' if first else
                                                 f"KHONG co tep nao bat dau bang '{CL.AZ_PREFIX}' -> C3 liet ke ca container de xem co gi"))
    return i


def _classify(b):
    name = b['name']
    base = name.rpartition('/')[2]
    if b.get('thu_muc') or name.endswith('/') or not base:
        return 'thu_muc', None
    if base in JUNK or base.startswith('._'):
        return 'rac', None
    k = CX.key_of(name)
    if k:
        return 'anh', k
    if base.lower().endswith(('.jpg', '.jpeg', '.png', '.dcm')):
        return 'anh_sai_dang', None
    if name in tuple(CL.AZ_PREFIX + c for c in CL.AZ_CSV):
        return 'csv', None
    return 'khac', None


def _drive_state(ctx):
    '''Thư mục CheXpert/ trên Drive -> (số thư mục CheXpert, {đường: [tệp]}, {đường thư mục: id}, số đường thư mục trùng).'''
    lst = ctx['lister']
    dest = lst.find_child(ctx['root'], CL.AZ_PREFIX.rstrip('/'), folder=True)
    files, folders, dup = collections.defaultdict(list), {}, 0
    if len(dest) == 1:
        w = lst.walk(dest[0]['id'], log=lambda m: None)
        for f in w['files']:
            files[CL.AZ_PREFIX + f['duong']].append(f)
        fc = collections.Counter(f['duong'] for f in w['folders'])
        dup = sum(1 for v in fc.values() if v > 1)
        folders = {CL.AZ_PREFIX + f['duong']: f['id'] for f in w['folders']}
        folders[CL.AZ_PREFIX] = dest[0]['id']
    return len(dest), files, folders, dup


def plan(ctx, log=print):
    '''C3. Liệt kê container -> kế hoạch + cổng trước khi tải.'''
    assert ctx.get('sas'), 'chua co link hop le (C2) -> chay lai C2'
    t0 = time.time()
    blobs = AB.list_blobs(ctx['sas'], '', fetch=ctx['fetch'], sleep=ctx['sleep'])
    sec = time.time() - t0
    real = [b for b in blobs if not b['thu_muc']]
    top = collections.Counter(b['name'].split('/', 1)[0] + ('/' if '/' in b['name'] else '') for b in real)
    top_b = collections.Counter()
    for b in real:
        top_b[b['name'].split('/', 1)[0] + ('/' if '/' in b['name'] else '')] += b['size']
    log(f"Azure container {ctx['sas'].container}: {len(real):,} tep, {sum(b['size'] for b in real) / 1e9:.2f} GB (liet ke {sec:.1f} giay) | trang AIMI cu ghi "
        f"{CL.AZ_FILES:,} tep | theo thu muc goc: " + ', '.join(f'{k} {n:,} tep / {top_b[k] / 1e9:.2f} GB' for k, n in sorted(top.items())))
    kinds, items, by_key = collections.defaultdict(list), [], collections.defaultdict(list)
    for b in blobs:
        if not b['name'].startswith(CL.AZ_PREFIX):
            continue
        kind, k = _classify(b)
        kinds[kind].append(b)
        if kind == 'anh':
            by_key[k].append(b)
        if kind in ('anh', 'csv') or (kind == 'khac' and b['size'] <= BIG):
            items.append(dict(name=b['name'], size=b['size'], md5_az=b['md5'], loai=kind, khoa=k, vai=CX.parts(k)[0] if k else None,
                              thang=bool(k and CX.is_frontal(k))))
    lon = [b for b in kinds['khac'] if b['size'] > BIG]
    trung = sorted(k for k, v in by_key.items() if len(v) > 1)
    per = {}
    for role in ROLES:
        ks = [it for it in items if it['vai'] == role]
        per[role] = dict(anh=len(ks), thang=sum(it['thang'] for it in ks), nguoi=len({CX.parts(it['khoa'])[1] for it in ks}),
                         gb=round(sum(it['size'] for it in ks) / 1e9, 3), nguoi_khoang=INV._people_range([it['khoa'] for it in ks]) if ks else '')
    csv_have = sorted(it['name'].rpartition('/')[2] for it in items if it['loai'] == 'csv')
    n_md5 = sum(1 for it in items if it['md5_az'])
    md5_bad = sum(1 for b in blobs if b['name'].startswith(CL.AZ_PREFIX) and b['md5_sai_dang'])
    other_roles = collections.Counter(it['vai'] for it in items if it['vai'] and it['vai'] not in ROLES)
    log(f"duoi '{CL.AZ_PREFIX}': " + ' | '.join(f"{role} {per[role]['anh']:,} anh ({per[role]['thang']:,} thang, {per[role]['gb']} GB, "
                                                f"{per[role]['nguoi_khoang'] or '0 nguoi'})" for role in ROLES)
        + (f' | anh tach khac {dict(other_roles)}' if other_roles else '') + f' | CSV {csv_have} | rac (.DS_Store ...) {len(kinds["rac"])} (khong tai) | muc thu muc gia '
        f"{len(kinds['thu_muc'])} | tep khac {len(kinds['khac'])}" + (f" [{INV._pattern([b['name'] for b in kinds['khac']])}]" if kinds['khac'] else '')
        + (f' (> 50 MB, KHONG tai: {len(lon)})' if lon else '') + f" | anh sai dang {len(kinds['anh_sai_dang'])}"
        + (f" [{INV._pattern([b['name'] for b in kinds['anh_sai_dang']])}]" if kinds['anh_sai_dang'] else '') + f' | trung khoa {len(trung)}')
    log(f"  md5 cua Azure (Content-MD5): co {n_md5:,} / {len(items):,} tep" + (f' | md5 sai dang {md5_bad}' if md5_bad else '')
        + (' -> khong co md5 thi chi so so byte + giai ma luc tai; C5 so md5 Drive voi du lieu tai' if n_md5 < len(items) else ''))
    log(f"  so voi bai: val {CL.PAPER['valid']['anh']} anh / {CL.PAPER['valid']['nguoi']} nguoi, test {CL.PAPER['test']['anh']} anh / {CL.PAPER['test']['nguoi']} nguoi | "
        f"CheXFound: {CL.CHEXFOUND['val']} / {CL.CHEXFOUND['test']} anh thang")
    n_dest, on, folders, fdup = _drive_state(ctx)
    names = {it['name'] for it in items}
    for it in items:
        c = on.get(it['name'], [])
        if not c:
            it['trang_thai'] = 'can_tai'
        elif len(c) == 1 and c[0]['size'] == it['size'] and (it['md5_az'] is None or c[0]['md5'] == it['md5_az']):
            it['trang_thai'] = 'da_co'
        else:
            it['trang_thai'] = 'xung_dot'
    st = collections.Counter(it['trang_thai'] for it in items)
    extra = sorted(p for p in on if p not in names)
    log(f"  Drive MyDrive/{'/'.join(CL.SOURCE)}/{CL.AZ_PREFIX}: " + ('CHUA co' if n_dest == 0 else (
        f"{n_dest} thu muc cung ten" if n_dest > 1 else f"co {sum(len(v) for v in on.values()):,} tep") + f" | da co dung (bo qua) {st['da_co']:,}, can tai {st['can_tai']:,}, "
        f"CO SAN MA KHAC (khong ghi de) {st['xung_dot']:,}, tep ngoai danh muc Azure {len(extra):,}" + (f', duong thu muc trung {fdup}' if fdup else '')))
    p = CL.PAPER
    gate = [(f"Azure: anh val = {p['valid']['anh']}", per['valid']['anh'] == p['valid']['anh'], per['valid']['anh']),
            (f"Azure: anh test = {p['test']['anh']}", per['test']['anh'] == p['test']['anh'], per['test']['anh']),
            ('Azure: du 2 CSV (val_labels.csv, test_labels.csv)', set(csv_have) == set(CL.AZ_CSV), csv_have),
            ('Azure: 0 anh sai dang / 0 trung khoa / 0 anh tach khac', not kinds['anh_sai_dang'] and not trung and not other_roles,
             f"{len(kinds['anh_sai_dang'])} / {len(trung)} / {sum(other_roles.values())}"),
            ('Azure: md5 dung dang, anh + CSV > 0 byte', md5_bad == 0 and all(it['size'] > 0 for it in items if it['loai'] in ('anh', 'csv')),
             f"md5 sai dang {md5_bad}, 0 byte {sum(1 for it in items if it['loai'] in ('anh', 'csv') and it['size'] == 0)}"),
            ('Drive: <= 1 thu muc CheXpert, 0 duong thu muc trung, 0 tep co san khac noi dung', n_dest <= 1 and fdup == 0 and st['xung_dot'] == 0,
             f"{n_dest} / {fdup} / {st['xung_dot']}")]
    ok = all(g[1] for g in gate)
    ctx.update(ke_hoach=items, cho_tai=ok, thu_muc=folders)
    ctx['kq'] = dict(ctx['kq'], azure=dict(tep=len(real), gb=round(sum(b['size'] for b in real) / 1e9, 3), theo_thu_muc_goc=dict(top), giay_liet_ke=round(sec, 1),
                                           chexpert=dict(theo_tach=per, csv=csv_have, rac=len(kinds['rac']), thu_muc_gia=len(kinds['thu_muc']), khac=len(kinds['khac']),
                                                         khac_lon_khong_tai=len(lon), anh_sai_dang=len(kinds['anh_sai_dang']), trung_khoa=len(trung),
                                                         md5_azure=n_md5, tep_ke_hoach=len(items))),
                     drive_truoc=dict(thu_muc_chexpert=n_dest, trang_thai=dict(st), ngoai_danh_muc=len(extra), duong_thu_muc_trung=fdup),
                     cong_c3=[dict(muc=a, dat=b, gia_tri=str(c)) for a, b, c in gate])
    for a, b, c in gate:
        log(f"  {'DAT  ' if b else 'TRUOT'} {a} | {c}")
    todo = [it for it in items if it['trang_thai'] == 'can_tai']
    log(f"CONG C3: {'DAT' if ok else 'CHUA DAT'} -> " + (f"C4 tai {len(todo):,} tep ({sum(it['size'] for it in todo) / 1e9:.2f} GB)" if ok else
                                                         'C4 KHONG tai; gui Claude output C2-C3'))
    return ok


def _parent(p):
    return p[:-1].rpartition('/')[0] + '/'


def _ensure_folders(ctx, paths, log):
    '''Tạo thư mục còn thiếu theo TẦNG (cùng tầng song song: tên khác nhau -> không tranh nhau; create_folder dùng lại thư mục có sẵn). -> số thư mục tạo.'''
    lst, fmap, made = ctx['lister'], ctx['thu_muc'], 0
    if CL.AZ_PREFIX not in fmap:
        fmap[CL.AZ_PREFIX] = lst.create_folder(ctx['root'], CL.AZ_PREFIX.rstrip('/'))
        made += 1
    need = set()
    for p in paths:
        segs = p.split('/')[:-1]
        need.update('/'.join(segs[:j]) + '/' for j in range(1, len(segs) + 1))
    by_depth = collections.defaultdict(list)
    for p in sorted(x for x in need if x not in fmap):
        by_depth[p.count('/')].append(p)
    for depth in sorted(by_depth):
        ps, t = by_depth[depth], time.time()
        with ThreadPoolExecutor(lst.threads) as pool:
            ids = list(pool.map(lambda q: lst.create_folder(fmap[_parent(q)], q[:-1].rpartition('/')[2]), ps))
        fmap.update(zip(ps, ids))
        made += len(ps)
        log(f'  thu muc tang {depth}: tao / dung lai {len(ps):,} ({(time.time() - t) / 60:.1f} phut, DO)')
    return made


def copy(ctx, log=print, every=60.0):
    '''C4. Azure -> kiểm -> Drive. Chạy lại ô: chỉ làm tệp còn thiếu.'''
    assert ctx.get('ke_hoach') is not None, 'chua chay C3 -> chay C3'
    kq = dict(can=0, xong=0, loi=0, loi_vd={}, thu_muc_tao=0, ly_do_khong_ghi=None)
    why = ('cong C3 chua dat' if not ctx.get('cho_tai') else 'tai khoan API KHONG phai CHU MyDrive/MIMIC_CXR/CheXlocalize (tep tai len se thuoc tai khoan nay) -> chay '
           'lai bang tai khoan chu (C0 chon tai khoan)' if not ctx['chu'] else 'tai khoan khong duoc them tep vao thu muc' if not ctx['them_con'] else None)
    if why:
        kq['ly_do_khong_ghi'] = why
        log(f'C4: KHONG GHI GI: {why}')
        ctx['kq'] = dict(ctx['kq'], tai=kq)
        return kq
    todo = [it for it in ctx['ke_hoach'] if it['trang_thai'] == 'can_tai' and it['name'] not in ctx['xong']]
    kq['can'] = len(todo)
    total = sum(it['size'] for it in todo)
    log(f"C4: tai {len(todo):,} tep ({total / 1e9:.2f} GB) tu Azure -> MyDrive/{'/'.join(CL.SOURCE)}/{CL.AZ_PREFIX} | {ctx['lister'].threads} luong | da xong truoc do "
        f"{len(ctx['xong']):,}")
    t0 = time.time()
    kq['thu_muc_tao'] = _ensure_folders(ctx, [it['name'] for it in todo], log) if todo else 0
    lst, sas, fmap = ctx['lister'], ctx['sas'], ctx['thu_muc']

    def one(it):
        data = AB.get_blob(sas, it['name'], it['size'], it['md5_az'], fetch=ctx['fetch'], sleep=ctx['sleep'])
        r = dict(md5=hashlib.md5(data).hexdigest())
        r['md5_az_dung'] = None if not it['md5_az'] else r['md5'] == it['md5_az']
        if it['loai'] == 'anh':
            r['mode'], r['wh'] = INV._decode(data)              # không giải mã được -> lỗi, KHÔNG tải lên
        folder, _, base = it['name'].rpartition('/')
        up = lst.upload(fmap[folder + '/'], base, data, mime=MIME.get(os.path.splitext(base)[1].lower(), 'application/octet-stream'))
        if up.get('md5Checksum') != r['md5'] or int(up.get('size') or -1) != len(data):
            raise IOError('Drive tra md5 / so byte khac du lieu')
        r['id'] = up['id']                                      # chỉ trong bộ nhớ (không ghi JSON)
        return r
    errs, vd, done_b, last, t1 = collections.Counter(), {}, 0, time.time(), time.time()
    pool = ThreadPoolExecutor(lst.threads)
    try:                                                        # ngắt ô -> huỷ tệp chưa chạy; chạy lại ô làm tiếp
        futs = {pool.submit(one, it): it for it in todo}
        for j, f in enumerate(as_completed(futs), 1):
            it = futs[f]
            try:
                ctx['xong'][it['name']] = f.result()
                done_b += it['size']
            except Exception as e:
                name = type(e).__name__ + (f"[{e.ma}]" if isinstance(e, AB.AzureError) else '')
                errs[name] += 1
                vd.setdefault(name, sas.redact(str(e))[:200])
            if time.time() - last >= every and j < len(todo):
                el = time.time() - t1
                rate = done_b / el / 1e6 if el else 0
                log(f'  {j:,} / {len(todo):,} tep | {done_b / 1e9:.2f} / {total / 1e9:.2f} GB | {rate:.1f} MB/s | {el / 60:.1f} phut, con ~'
                    f'{el / j * (len(todo) - j) / 60:.1f} phut (DO) | loi {sum(errs.values())}')
                last = time.time()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    xs = [ctx['xong'][it['name']] for it in todo if it['name'] in ctx['xong']]
    w = [x['wh'][0] for x in xs if x.get('wh')]
    el = time.time() - t1
    kq.update(xong=len(xs), loi=sum(errs.values()), loi_vd=vd, phut=round((time.time() - t0) / 60, 2), mb_giay=round(done_b / el / 1e6, 1) if el else None,
              md5_azure_dung=sum(1 for x in xs if x['md5_az_dung']), md5_azure_khong_co=sum(1 for x in xs if x['md5_az_dung'] is None),
              anh_giai_ma=sum(1 for x in xs if 'mode' in x), mode=dict(collections.Counter(x['mode'] for x in xs if 'mode' in x)),
              rong=[min(w), int(np.median(w)), max(w)] if w else None)
    ctx['kq'] = dict(ctx['kq'], tai=kq)
    log(f"  xong {kq['xong']:,} / {kq['can']:,} tep | md5 = Azure {kq['md5_azure_dung']:,} (Azure khong co md5: {kq['md5_azure_khong_co']:,}) | anh giai ma "
        f"{kq['anh_giai_ma']:,} | mode {kq['mode']} | rong min / trung vi / max {kq['rong']} | thu muc tao / dung lai {kq['thu_muc_tao']:,} | loi {kq['loi']}"
        + (f' {dict(errs)} vd {vd}' if errs else '') + f" | {kq['phut']} phut, {kq['mb_giay']} MB/s (DO)"
        + (' -> CHAY LAI C4 (chi lam tep con thieu)' if errs else ''))
    return kq


def verdict(ctx, rows, extra, fdup, n_dest, re_err):
    '''-> [(mục, đạt, ghi chú)]; đạt None = chỉ cảnh báo.'''
    kq, p, out = ctx['kq'], CL.PAPER, []
    out.append(('link hop le (https, Azure, doc + liet ke, con han)', bool(kq.get('link')), str(kq.get('link', {}).get('het_han'))))
    out.append(('cong C3: Azure du 234 val + 668 test + 2 CSV, 0 anh sai dang / trung khoa, khong xung dot tren Drive', ctx.get('cho_tai') is True,
                '; '.join(f"{g['muc']} = {g['gia_tri']}" for g in kq.get('cong_c3', []) if not g['dat']) or 'dat'))
    t = kq.get('tai') or {}
    out.append(('C4: moi tep can tai da tai + kiem (so byte, md5 Azure neu co, anh giai ma) + len Drive', not t.get('ly_do_khong_ghi') and t.get('loi', 1) == 0
                and t.get('xong') == t.get('can'), t.get('ly_do_khong_ghi') or f"{t.get('xong')} / {t.get('can')}, loi {t.get('loi')}"))
    st = collections.Counter(r['trang_thai'] for r in rows)
    out.append(('Drive: MOI tep dung 1 ban, so byte = Azure, md5 = du lieu Azure', bool(rows) and st['dung'] == len(rows),
                ', '.join(f'{k} {v:,}' for k, v in sorted(st.items())) + (f' | loi tai lai de so {re_err}' if re_err else '')))
    got = {role: sum(1 for r in rows if r['vai'] == role and r['trang_thai'] == 'dung') for role in ROLES}
    out.append((f"anh dung tren Drive: val {p['valid']['anh']} / test {p['test']['anh']}", got['valid'] == p['valid']['anh'] and got['test'] == p['test']['anh'],
                f"{got['valid']} / {got['test']}"))
    out.append(('Drive: 1 thu muc CheXpert, 0 duong thu muc trung', n_dest == 1 and fdup == 0, f'{n_dest} / {fdup}'))
    out.append(('tep tren Drive ngoai danh muc Azure', True if not extra else None, f'{len(extra):,}'))
    n_md5 = sum(1 for r in rows if r['md5_azure'])
    out.append(('Azure co md5 cho moi tep (so doc lap voi Azure)', True if rows and n_md5 == len(rows) else None, f'{n_md5:,} / {len(rows):,}'))
    ft = sum(1 for r in rows if r['vai'] == 'test' and r['thang'] and r['trang_thai'] == 'dung')
    out.append((f"anh thang test = {CL.CHEXFOUND['test']} (CheXFound)", True if ft == CL.CHEXFOUND['test'] else None, f'{ft}'))
    return out


def verify(ctx, log=print):
    '''C5. Liệt kê lại Drive -> so từng tệp -> JSON + bảng -> dong_bo() -> phán quyết.'''
    import pandas as pd
    d, items, t0 = ctx['drive'], ctx.get('ke_hoach') or [], time.time()
    n_dest, on, _, fdup = _drive_state(ctx)
    names = {it['name'] for it in items}
    extra = sorted(p for p in on if p not in names)
    ref, need = {}, []
    for it in items:                                            # md5 tham chiếu: của Azure nếu có (độc lập với C4); không có -> dữ liệu C4 tải; không nữa -> tải lại
        x = ctx['xong'].get(it['name'])
        if it['md5_az']:
            ref[it['name']] = it['md5_az']
        elif x:
            ref[it['name']] = x['md5']
        elif len(on.get(it['name'], [])) == 1:
            need.append(it)
    re_err = collections.Counter()
    if need:                                                    # Azure không có md5 + tệp không tải trong runtime này -> tải lại để so
        log(f'C5: tai lai {len(need):,} tep tu Azure de so md5 (Azure khong co md5, tep khong tai trong runtime nay)')

        def md5_of(it):
            return hashlib.md5(AB.get_blob(ctx['sas'], it['name'], it['size'], None, fetch=ctx['fetch'], sleep=ctx['sleep'])).hexdigest()
        with ThreadPoolExecutor(ctx['lister'].threads) as pool:
            futs = {pool.submit(md5_of, it): it for it in need}
            for f in as_completed(futs):
                try:
                    ref[futs[f]['name']] = f.result()
                except Exception as e:
                    re_err[type(e).__name__] += 1
    rows = []
    for it in items:
        c, x = on.get(it['name'], []), ctx['xong'].get(it['name']) or {}
        if not c:
            st = 'thieu'
        elif len(c) > 1:
            st = 'nhieu_ban'
        elif c[0]['size'] != it['size']:
            st = 'lech_byte'
        elif ref.get(it['name']) is None:
            st = 'khong_so_duoc'
        elif c[0]['md5'] != ref[it['name']]:
            st = 'lech_md5'
        else:
            st = 'dung'
        rows.append(dict(duong=it['name'], loai=it['loai'], vai=it['vai'], thang=it['thang'], byte=it['size'], md5=ref.get(it['name']), md5_azure=bool(it['md5_az']),
                         trang_thai=st, mode=x.get('mode'), rong=(x.get('wh') or (None, None))[0], cao=(x.get('wh') or (None, None))[1]))
    v = verdict(ctx, rows, extra, fdup, n_dest, dict(re_err))
    ok = all(x[1] is not False for x in v)
    st = collections.Counter(r['trang_thai'] for r in rows)
    ctx['kq'] = dict(ctx['kq'], doi_chieu=dict(trang_thai=dict(st), ngoai_danh_muc=len(extra), duong_thu_muc_trung=fdup, thu_muc_chexpert=n_dest,
                                               tai_lai_de_so=len(need), loi_tai_lai=dict(re_err), giay=round(time.time() - t0)))
    js = dict(notebook=NOTEBOOK, run=d['RUN_TAG'], t=datetime.datetime.now().isoformat(timespec='seconds'), phien_ban=__version__, goi=ctx['goi'],
              nguon=dict(trang=CL.AIMI_OLD, tien_to=CL.AZ_PREFIX), dich='MyDrive/' + '/'.join(CL.SOURCE) + '/' + CL.AZ_PREFIX, ket_qua=ctx['kq'],
              phan_quyet=[dict(muc=a, dat=b, ghi_chu=c) for a, b, c in v], dat=ok, phut=round((time.time() - ctx['t0']) / 60, 1),
              ghi_chu='PLAN_93 6 phien 6. KHONG link SAS, KHONG ID Drive trong tep nay. Sau khi dat: chay lai 93j_kiem_ke_chexlocalize.')
    d['ghi'](OUT['json'], lambda q: d['ghi_json'](q, js))
    d['ghi'](OUT['ds'], lambda q: pd.DataFrame(rows).to_parquet(q, index=False))
    du = d['dong_bo']()
    co = {OUT['json'], OUT['ds']} <= d['ten_tren_drive']()
    log(f"C5: Drive {CL.AZ_PREFIX}: " + ', '.join(f'{k} {n:,}' for k, n in sorted(st.items())) + f" | ngoai danh muc {len(extra):,} | dong_bo {'DU' if du else 'CHUA DU'} | "
        f"2 tep ket qua tren Drive: {'CO du' if co else 'THIEU -> chay lai C5, KHONG tat runtime'} | tong {js['phut']} phut")
    log('PHAN QUYET TAI CheXpert/ cua CheXlocalize:')
    for a, b, c in v:
        log(f"  {'DAT  ' if b else ('CANH BAO' if b is None else 'TRUOT')} {a} | {c}")
    log(f"=> {'TAI DAT -> chay lai 93j_kiem_ke_chexlocalize (y nguyen)' if ok else 'TAI CHUA DAT (muc TRUOT o tren) -> gui Claude output C2-C5'}")
    res = dict(dat=ok, dong_bo=du, du_tep=co, phan_quyet=v)
    ctx['ket_qua'] = res
    return res
