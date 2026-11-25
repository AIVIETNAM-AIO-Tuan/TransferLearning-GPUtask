# -*- coding: utf-8 -*-
"""Ô của notebook kiểm kê `93i_kiem_ke_chexpert` (Colab CPU; PLAN_93 §3 / §10 điểm 6; Tuấn 24/11: CheXpert bản đầy đủ đã giải nén ở MyDrive/MIMIC_CXR/CheXpert).

0.4.0 chỉ đọc. 0.4.1 (Tuấn 24/11 sau lượt đầu: "có thiếu ảnh nào thì kiểm trong zip có không thì lấy ra được không, sau đó xoá zip luôn" -> "dựng đi"):
  - khoá nhận bố cục batch của Stanford AIMI (datasets.chexpert) — lượt đầu 223.414 ảnh train thành "không đúng dạng" (Event 24/11);
  - C6 so zip <-> Drive theo KHOÁ (ảnh) / theo đường dẫn (tệp khác), không theo tên trần (lượt đầu: dòng "zip … DAT" không kiểm gì cho train);
    ~1.000 ảnh mẫu rải theo zip (md5 = Drive, CRC = zip, giải mã); đo thời gian từng bước;
  - C7 GHI (chỉ khi tài khoản API là CHỦ thư mục CheXpert): ảnh thiếu / lệch kích thước mà zip có -> lấy từ zip -> tải lên đúng thư mục -> md5;
  - C10 chuyển zip (trong cây, đã so khớp hết) vào THÙNG RÁC khi phán quyết C9 đạt hết + đủ quyền; không đủ quyền -> in tên để Tuấn xoá tay.
Không dừng giữa chừng vì dữ liệu lạ (trừ không thấy thư mục / lỗi API): ghi nhận mọi điều rồi C9 in phán quyết từng mục.

    C2  setup       hàm của khối Drive chuẩn + thư mục gốc qua API + quyền của tài khoản API trên thư mục (chủ? thêm con được?)
    C3  tree        liệt kê CẢ cây theo lô (store.listing) -> thư mục / tệp / GB theo tầng, khung cây, loại tệp, ảnh theo khoá + theo thư mục batch
    C4  csvs        tải + đọc mọi CSV trong cây -> vai (train / valid / test theo cột Path), số dòng / người / thẳng-nghiêng / AP-PA, giá trị nhãn
    C5  match       CSV <-> ảnh trên Drive: thiếu / thừa theo tách
    C6  zips        tìm zip -> đọc danh mục từ xa (giữ mở cho C7) -> so theo khoá / đường dẫn + kích thước; ảnh mẫu rải theo zip; thời gian từng bước
    C7  restore     lấy ảnh thiếu / lệch từ zip (chỉ khi là chủ; <= MAX_RESTORE) -> tải lên -> md5 -> liệt kê lại -> đối chiếu lại C5 / C6
    C8  propose     bảng ĐỀ XUẤT (mặc định PLAN_93 §10 điểm 6) + vân tay so lượt 0.4.0
    C9  write       ghi JSON + bảng đề xuất + danh sách tệp (không ID Drive) -> dong_bo() -> phán quyết từng mục
    C10 trash_zips  (XOA_ZIP) zip đã so khớp -> thùng rác khi C9 đạt hết + đủ quyền -> ghi JSON nhỏ
Không in ảnh; khoá ảnh chỉ in dạng khoảng mã người khi thiếu tệp."""
import collections
import datetime
import hashlib
import io
import os
import re
import sys
import tempfile
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import __version__
from ..datasets import base
from ..datasets import chexpert as CX
from ..store import listing as L
from ..store import zip_remote as ZR

NOTEBOOK = '93i_kiem_ke_chexpert'
RUN_TAG = 'r93i_kiem_ke_chexpert_v2'                     # lượt đầu (0.4.0): r93i_kiem_ke_chexpert
DRIVE_NAMES = ('tai', 'ghi', 'ghi_json', 'dong_bo', 'ten_tren_drive', 'RUN_TAG', 'GHI_LOC', '_duong', '_liet_ke')
OUT = dict(json='kiem_ke_chexpert.json', bang='chexpert_de_xuat_bang.parquet', ds='chexpert_danh_sach_drive.parquet', xoa='kiem_ke_chexpert_xoa_zip')
CSV_MAX = 300 << 20
N_SAMPLE, N_SAMPLE_MIN = 1000, 24                         # ảnh mẫu: ~1.000 rải theo số ảnh mỗi zip, mỗi zip >= 24
MAX_RESTORE = 1000                                         # quá số này ảnh cần lấy từ zip -> không lấy, báo Claude (bố cục lạ / thiếu hàng loạt)
ARCHIVE = re.compile(r'\.(zip|7z|rar|tar|tgz|gz|z\d\d|\d{3})$', re.I)
BIG_FOLDER = 30
# vân tay bảng đề xuất của lượt 0.4.0 (24/11; khoá từ cột Path của CSV — 0.4.1 không được đổi; md5 khoá / md5 nhãn, 12 ký tự đầu)
FP_040 = {'train': ('da5b68a3ec66', '4cca644252fd'), 'val': ('734229f222dd', '4be4cf5638d6'), 'test': ('0fda940ec3cd', 'ddf480313937')}


def setup(ns, api_factory=None, log=print):
    '''C2. -> CTX.'''
    missing = [k for k in DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    d = {k: ns[k] for k in DRIVE_NAMES}
    assert d['RUN_TAG'] == RUN_TAG, f"RUN_TAG {d['RUN_TAG']!r} != {RUN_TAG!r} -> DUNG"
    root = d['_duong'](CX.SOURCE)
    assert root, (f"khong thay thu muc MyDrive/{'/'.join(CX.SOURCE)} qua API -> soat lai ten (hoa / thuong, dau cach, gach duoi) roi bao Claude -> DUNG")
    parent = d['_duong'](CX.SOURCE[:1])
    loc = '/content/kk93_chexpert' if 'google.colab' in sys.modules else tempfile.mkdtemp(prefix='kk93_')
    os.makedirs(loc, exist_ok=True)
    from .store import _packages
    lst = L.Lister(api_factory or L.default_api)
    try:
        info = lst.get(root)
        chu, them, qloi = bool(info.get('ownedByMe')), bool((info.get('capabilities') or {}).get('canAddChildren')), None
    except Exception as e:
        chu, them, qloi = False, False, type(e).__name__
    ctx = dict(drive=d, root=root, parent=parent, lister=lst, loc=loc, goi=_packages(), t0=time.time(), kq={}, chu=chu, them_con=them, zip_ent={}, zip_obj={},
               zip_khong_doc=[])
    log(f"93i kiem ke CheXpert: thu muc MyDrive/{'/'.join(CX.SOURCE)} THAY qua API | goi " + ', '.join(
        f"{k} {v['phien_ban']} @ {(v['sha'] or '?')[:12]}" for k, v in ctx['goi'].items()) + f' | {os.cpu_count()} CPU')
    log(f"  tai khoan API {'LA' if chu else 'KHONG phai'} CHU thu muc CheXpert (them tep vao thu muc: {'duoc' if them else 'khong'}"
        + (f'; loi doi quyen {qloi}' if qloi else '') + ') -> ' + (
            'C7 ghi duoc anh lay tu zip (neu thieu); C10 chuyen zip vao thung rac neu moi phep so dat' if chu else
            'C7 se KHONG ghi (tep tai len se thuoc tai khoan nay), C10 co the khong du quyen -> muon tu dong: ngat runtime, chay lai, chon tai khoan CHU o C0'))
    return ctx


# ---------------- C3 ----------------
def _pattern(names, top=3):
    c = collections.Counter(re.sub(r'\d+', '#', n) for n in names)
    return ', '.join(f'{p} x {n:,}' for p, n in c.most_common(top))


def _skeleton(folders, files, max_lines=60):
    '''Khung cây: thư mục tầng 1–3 kèm số con; thư mục có > BIG_FOLDER thư mục con chỉ in số + mẫu tên con (không liệt kê).'''
    sub, nf = collections.defaultdict(list), collections.Counter()
    for f in folders:
        sub[f['duong'][:-1].rpartition('/')[0]].append(f['duong'][:-1].rpartition('/')[2])
    for f in files:
        nf[f['duong'].rpartition('/')[0]] += 1
    lines = [f"(goc) {len(sub[''])} thu muc, {nf['']} tep"]

    def go(path, depth):
        kids = sorted(sub[path])
        if len(kids) > BIG_FOLDER:
            lines.append('  ' * depth + f'[{len(kids):,} thu muc con: {_pattern(kids)}]')
            return
        for k in kids:
            p = f'{path}/{k}' if path else k
            if len(lines) >= max_lines:
                return
            lines.append('  ' * depth + f'{k}/  ({len(sub[p]):,} thu muc, {nf[p]:,} tep)')
            if depth < 3:
                go(p, depth + 1)
    go('', 1)
    return lines


def tree(ctx, log=print):
    '''C3. Liệt kê cả cây theo lô -> CTX['cay'], CTX['anh'] ({khoá: bản trên Drive}).'''
    log('liet ke cay qua API theo lo (moi lan goi hoi con cua toi da 100 thu muc, 6 luong); in tien do moi phut — thoi gian con lai DO o tang dang chay')
    res = ctx['lister'].walk(ctx['root'], resume_path=os.path.join(ctx['loc'], '_cay.pkl'), log=log)
    files, folders = res['files'], res['folders']
    by_key, odd = collections.defaultdict(list), []
    for f in files:
        k = CX.key_of(f['duong'])
        if k:
            by_key[k].append(f)
        elif f['ten'].lower().endswith(('.jpg', '.jpeg', '.png', '.dcm')):
            odd.append(f['duong'])
    conflict = sorted(k for k, v in by_key.items() if len({(x['md5'], x['size']) for x in v}) > 1)
    dup = sum(1 for v in by_key.values() if len(v) > 1)
    anh = {k: v[0] for k, v in by_key.items()}
    per = collections.defaultdict(lambda: dict(anh=0, thang=0, nguoi=set(), byte=0))
    for k, f in anh.items():
        s, p, _, _ = CX.parts(k)
        per[s]['anh'] += 1
        per[s]['thang'] += CX.is_frontal(k)
        per[s]['nguoi'].add(p)
        per[s]['byte'] += f['size']
    per = {s: dict(anh=v['anh'], thang=v['thang'], nguoi=len(v['nguoi']), gb=round(v['byte'] / 1e9, 2)) for s, v in sorted(per.items())}
    heads = collections.Counter(CX.head_of(f['duong']) for f in anh.values())
    ext = collections.Counter(os.path.splitext(f['ten'])[1].lower() or '(khong duoi)' for f in files)
    other = sorted((f['duong'], f['size']) for f in files if not CX.key_of(f['duong']))
    st = res['thong_ke']
    summ = dict(thu_muc=len(folders), tep=len(files), gb=round(sum(f['size'] for f in files) / 1e9, 2), chua_duyet=res['chua_duyet'], loai_tep=dict(ext),
                anh_theo_tach=per, anh_theo_thu_muc=dict(sorted(heads.items())), khoa_trung_nhieu_cho=dup, khoa_trung_khac_noi_dung=len(conflict),
                anh_sai_dang=len(odd), anh_sai_dang_vd=odd[:3], anh_khong_md5=sum(1 for f in anh.values() if not f['md5']),
                anh_0_byte=sum(1 for f in anh.values() if f['size'] == 0), tep_khac=[dict(duong=p, mb=round(s / 1e6, 2)) for p, s in other[:200]],
                so_tep_khac=len(other), shortcut=sum(f['mime'] == L.SHORTCUT_MIME for f in files), api=st)
    ctx.update(cay=res, anh=anh, kq=dict(ctx['kq'], cay=summ), xung_dot_anh=conflict)
    log(f"cay: {len(folders):,} thu muc + {len(files):,} tep, {summ['gb']:,} GB | {st['lan_goi']:,} lan goi API, thu lai {st['thu_lai']}, chia lo {st['chia_lo']} "
        f"(K cuoi {st['k_cuoi']}) | {st['phut']:.1f} phut" + (f" | ⚠ {res['chua_duyet']:,} thu muc SAU HON {L.MAX_DEPTH} tang chua duyet" if res['chua_duyet'] else ''))
    log('  theo tang: ' + ' | '.join(f"t{x['tang']} {x['thu_muc_con']:,} tm + {x['tep']:,} tep ({x['lan_goi']:,} goi, {x['phut']} ph)" for x in st['tang']))
    log('  khung cay:')
    for line in _skeleton(folders, files):
        log('    ' + line)
    log('  loai tep: ' + ', '.join(f'{e} {n:,}' for e, n in ext.most_common(12)))
    log('  anh CheXpert theo khoa (duoi .../train|valid|test/patient/study/view.jpg HOAC .../<thu muc "(train k)">/patient/study/view.jpg): ' + ' | '.join(
        f"{s} {v['anh']:,} anh ({v['thang']:,} thang, {v['nguoi']:,} nguoi, {v['gb']} GB)" for s, v in per.items())
        + f" | theo bai: train {CX.PAPER['train'][0]:,} / valid {CX.PAPER['valid'][0]} / test {CX.PAPER['test'][0]} anh")
    log('  anh theo thu muc (phan truoc patient): ' + ' | '.join(f"{h or '(goc)'} {n:,}" for h, n in sorted(heads.items())))
    log(f"  trung khoa o nhieu cho: {dup:,} (khac noi dung: {len(conflict):,}) | anh khong dung dang: {len(odd):,}" + (f' vd {odd[:3]}' if odd else '')
        + f" | anh khong md5: {summ['anh_khong_md5']} | anh 0 byte: {summ['anh_0_byte']} | shortcut trong cay: {summ['shortcut']}")
    log(f'  tep KHONG phai anh CheXpert ({len(other):,}):')
    for p, s in other[:60]:
        log(f'    {p}  ({s / 1e6:,.2f} MB)')
    if len(other) > 60:
        log(f'    ... con {len(other) - 60:,} tep (du danh sach trong JSON ket qua)')
    return summ


# ---------------- C4 ----------------
def csvs(ctx, log=print, expect=CX.SPLITS_CX):
    '''C4. Tải + đọc mọi CSV của cây. -> {vai: CSV chọn}. `expect`: tách phải có CSV (thiếu -> dòng ⚠); 93j (CheXlocalize, 0.4.2) chỉ valid + test.'''
    d, frames, cands = ctx['drive'], {}, []
    for f in sorted(ctx['cay']['files'], key=lambda x: x['duong']):
        if not f['ten'].lower().endswith('.csv'):
            continue
        row = dict(ten=f['ten'], duong=f['duong'], mb=round(f['size'] / 1e6, 2), md5=f['md5'])
        if f['size'] > CSV_MAX:
            row['bo_qua'] = f'lon hon {CSV_MAX >> 20} MB'
        else:
            try:
                df = CX.read_csv(d['tai'](dict(id=f['id'], ten=f['ten'], md5=f['md5'], size=f['size'])))
                desc = CX.describe(df)
                row.update(desc=desc, vai=CX.role_of(desc), dong=desc['dong'])
                frames[f['duong']] = df
            except Exception as e:
                row['loi'] = f'{type(e).__name__}: {str(e)[:120]}'
        cands.append(row)
    chosen = CX.pick(cands)
    ctx.update(csv_bang=frames, csv_chon=chosen, kq=dict(ctx['kq'], csv=dict(ung_vien=cands, chon={r: c['duong'] for r, c in chosen.items()})))
    log(f'CSV trong cay: {len(cands)}')
    for c in cands:
        x = c.get('desc') or {}
        log(f"  {c['duong']} | {c['mb']} MB | md5 {c['md5']} | " + (c.get('bo_qua') or c.get('loi') or (
            f"{x['dong']:,} dong | vai {c['vai'] or '-'} | tach trong Path {x.get('tach_trong_path')} | {x.get('nguoi', 0):,} nguoi, {x.get('lan_chup', 0):,} lan chup"
            f" | thang {x.get('thang', 0):,} / nghieng {x.get('nghieng', 0):,} | AP/PA (anh thang) {x.get('ap_pa_anh_thang')} | Path sai dang {x.get('path_sai_dang')}"
            f" | khoa trung {x.get('khoa_trung')} | thieu cot nhan {x['nhan_thieu'] or 0}")))
    log('  CHON: ' + (' | '.join(f'{r} = {c["duong"]}' for r, c in chosen.items()) or 'KHONG co CSV nao dung dang CheXpert'))
    if chosen:
        roles = list(chosen)
        log('  gia tri nhan (so DONG) — ' + ' | '.join(f'{r}: 1 / 0 / -1 (khong chac) / trong' for r in roles))
        for lab in CX.LABELS:
            log(f'    {lab:<27}' + ' | '.join(
                ' / '.join(f"{chosen[r]['desc']['gia_tri_nhan'][lab].get(v, 0):,}" for v in ('1', '0', '-1', 'trong'))
                + (f" (khac {chosen[r]['desc']['gia_tri_nhan'][lab]['khac']})" if chosen[r]['desc']['gia_tri_nhan'][lab].get('khac') else '') for r in roles))
    for r in expect:
        if r not in chosen:
            log(f'  ⚠ khong co CSV vai {r}')
    return chosen


# ---------------- C5 ----------------
def _people_range(keys):
    p = sorted({CX.parts(k)[1] for k in keys})
    return f"{len(p):,} nguoi, {p[0]} … {p[-1]}" if p else '0 nguoi'


def _csv_keys(ctx):
    return {role: {k for k in (CX.key_of(p) for p in ctx['csv_bang'][c['duong']].Path) if k} for role, c in ctx['csv_chon'].items()}


def match(ctx, log=print, nhan=''):
    '''C5 (và lại sau C7). CSV chọn <-> ảnh trên Drive theo khoá.'''
    anh, out = ctx['anh'], {}
    for role, ks in _csv_keys(ctx).items():
        miss = sorted(k for k in ks if k not in anh)
        extra = sorted(k for k in anh if CX.parts(k)[0] == role and k not in ks)
        out[role] = dict(khoa_csv=len(ks), co_tren_drive=len(ks) - len(miss), thieu=len(miss), thieu_thang=sum(CX.is_frontal(k) for k in miss),
                         thua_tren_drive=len(extra), thieu_nguoi=_people_range(miss) if miss else '', thua_nguoi=_people_range(extra) if extra else '')
        log(f"{nhan}{role}: CSV {len(ks):,} khoa | co tren Drive {len(ks) - len(miss):,} | THIEU {len(miss):,} ({out[role]['thieu_thang']:,} anh thang"
            + (f"; {out[role]['thieu_nguoi']}" if miss else '') + f") | tren Drive ngoai CSV {len(extra):,}" + (f" ({out[role]['thua_nguoi']})" if extra else ''))
    loose = collections.Counter(CX.parts(k)[0] for k in anh if CX.parts(k)[0] not in ctx['csv_chon'])
    if loose:
        log(f'{nhan}  anh tren Drive thuoc tach KHONG co CSV: {dict(loose)}')
    ctx['kq'] = dict(ctx['kq'], doi_chieu=dict(theo_vai=out, anh_tach_khong_csv=dict(loose)))
    return out


# ---------------- C6 ----------------
def _find_archives(ctx, log):
    found = {}
    for f in ctx['cay']['files']:
        if ARCHIVE.search(f['ten']):
            found[f['id']] = dict(ten=f['ten'], duong=f['duong'], id=f['id'], size=f['size'], noi='trong cay')
    if ctx['parent']:
        for x in ctx['drive']['_liet_ke'](ctx['parent']):
            if x.get('mimeType') != L.FOLDER_MIME and ARCHIVE.search(x['name']) and 'chexpert' in x['name'].lower():
                found.setdefault(x['id'], dict(ten=x['name'], duong='MIMIC_CXR/' + x['name'], id=x['id'], size=int(x.get('size') or 0), noi='trong MIMIC_CXR'))
    try:
        for x in ctx['lister'].search_name('CheXpert'):
            if x.get('mimeType') != L.FOLDER_MIME and ARCHIVE.search(x['name']):
                found.setdefault(x['id'], dict(ten=x['name'], duong='(tim theo ten) ' + x['name'], id=x['id'], size=int(x.get('size') or 0), noi='tim theo ten'))
    except Exception as e:
        log(f'  tim theo ten khap Drive: loi {type(e).__name__} (bo qua)')
    return sorted(found.values(), key=lambda z: z['duong'])


def _decode(b):
    from PIL import Image
    im = Image.open(io.BytesIO(b))
    im.load()
    return im.mode, im.size


def _other_index(files):
    '''Hậu tố đường dẫn (theo đoạn) -> [tệp], CHỈ cho tệp trên Drive KHÔNG phải ảnh có khoá (CSV, tệp lạ): mục zip không phải ảnh so theo đường dẫn — ảnh
       có khoá không vào chỉ mục nên một mục zip "view1_frontal.jpg" không khớp nhầm hàng nghìn ảnh cùng tên (lỗi của 0.4.0).'''
    idx = collections.defaultdict(list)
    for f in files:
        if not CX.key_of(f['duong']):
            segs = f['duong'].split('/')
            for i in range(len(segs)):
                idx['/'.join(segs[i:])].append(f)
    return idx


def _find_other(idx, name):
    segs = name.strip('/').split('/')
    for i in range(len(segs)):
        c = idx.get('/'.join(segs[i:]))
        if c:
            return c
    return []


def _compare(ctx):
    '''So mọi zip đã đọc với cây hiện tại -> (hàng mỗi zip, số ảnh trên Drive ngoài mọi zip, số khoá xung đột giữa các zip). Gán z['ok'].'''
    anh, idx = ctx['anh'], _other_index(ctx['cay']['files'])
    rows, zkeys, conflict = [], {}, set()
    for zid, z in ctx['zip_ent'].items():
        a, img = z['a'], z['img']
        for k, (n, s, c) in img.items():
            if k in zkeys and zkeys[k][:2] != (s, c):
                conflict.add(k)
            zkeys.setdefault(k, (s, c, zid, n))
        miss = sorted(k for k in img if k not in anh)
        bad = sorted(k for k in img if k in anh and anh[k]['size'] != img[k][1])
        om, ob = [], []
        for n, s, _ in z['oth']:
            c = _find_other(idx, n)
            if not c:
                om.append(n)
            elif all(f['size'] != s for f in c):
                ob.append(n)
        z['ok'] = bool(img) and not miss and not bad and not om and not ob and not z['trung']
        rows.append(dict(ten=a['ten'], duong=a['duong'], gb=round(a['size'] / 1e9, 2), noi=a['noi'], muc=z['muc'], anh=len(img), co_tren_drive=len(img) - len(miss),
                         thieu_tren_drive=len(miss), khac_kich_thuoc=len(bad), thieu_nguoi=_people_range(miss) if miss else '', tep_khac=len(z['oth']),
                         tep_khac_thieu=len(om), tep_khac_lech_kich_thuoc=len(ob), tep_khac_dang=_pattern([n for n, _, _ in z['oth']]),
                         dang_ten_anh=z['dang'], trung_trong_zip=len(z['trung']), lan_doc=z['io']['lan_doc'], byte_doc=z['io']['byte_doc'], giay=z['giay'],
                         dat=z['ok']))
    rows += ctx['zip_khong_doc']
    rows.sort(key=lambda r: r['duong'])
    outside = sum(1 for k in anh if k not in zkeys) if zkeys else 0
    ctx['zkeys'] = zkeys
    return rows, outside, len(conflict)


def _log_zips(rows, log, nhan=''):
    for r in rows:
        log(f"{nhan}  {r['duong']} ({r['gb']} GB, {r['noi']}) | " + (r.get('loi') or (
            f"{r['muc']:,} muc, {r['anh']:,} anh | co tren Drive {r['co_tren_drive']:,} | THIEU {r['thieu_tren_drive']:,}"
            + (f" ({r['thieu_nguoi']})" if r['thieu_tren_drive'] else '') + f" | khac kich thuoc {r['khac_kich_thuoc']:,} | trung khoa khac noi dung trong zip "
            f"{r['trung_trong_zip']} | tep khac {r['tep_khac']} (khong thay tren Drive {r['tep_khac_thieu']}, lech kich thuoc {r['tep_khac_lech_kich_thuoc']})"
            + (f" [{r['tep_khac_dang']}]" if r['tep_khac'] else '') + f" | dang ten anh [{r['dang_ten_anh']}] | doc danh muc {r['lan_doc']} lan, "
            f"{r['byte_doc'] / 1e6:.1f} MB, {r['giay']:.0f} giay | {'KHOP HET' if r['dat'] else 'CHUA KHOP'}")))


def _check_one(lister, f, crc):
    r = {}
    try:
        b = lister.fetch_range(f['id'], 0, f['size'] - 1) if f['size'] else b''
        r['md5_dung'] = (f['md5'] is None) or hashlib.md5(b).hexdigest() == f['md5']
        if crc is not None:
            r['crc_dung'] = (zlib.crc32(b) & 0xffffffff) == crc
        r['mode'], r['wh'] = _decode(b)
    except Exception as e:
        r['loi'] = type(e).__name__
    return r


def _sample(ctx):
    '''Khoá mẫu: rải đều theo khoá trong từng zip (ảnh có trên Drive, đúng kích thước), số mẫu theo tỉ lệ số ảnh, mỗi zip >= N_SAMPLE_MIN; không zip -> rải
       đều trên mọi ảnh. -> [(khoá, tên zip | None)].'''
    anh, per, seen = ctx['anh'], [], set()
    for zid, z in sorted(ctx['zip_ent'].items(), key=lambda x: x[1]['a']['duong']):
        ks = sorted(k for k, (_, s, _) in z['img'].items() if k in anh and anh[k]['size'] == s and k not in seen)
        seen.update(ks)
        if ks:
            per.append((z['a']['ten'], ks))
    if not per:
        ks = sorted(anh)
        return [(ks[i], None) for i in dict.fromkeys(np.linspace(0, len(ks) - 1, min(N_SAMPLE, len(ks))).astype(int))] if ks else []
    total = sum(len(ks) for _, ks in per)
    out = []
    for name, ks in per:
        n = min(len(ks), max(N_SAMPLE_MIN, int(round(N_SAMPLE * len(ks) / total))))
        out += [(ks[i], name) for i in dict.fromkeys(np.linspace(0, len(ks) - 1, n).astype(int))]
    return out


def zips(ctx, log=print):
    '''C6. Zip: danh mục từ xa (giữ mở cho C7) -> so theo khoá / đường dẫn + kích thước với bản giải nén; ảnh mẫu rải theo zip: md5 = Drive, CRC = zip,
       giải mã được. Thời gian từng bước + số lần thử lại.'''
    lister, anh, t0, r0 = ctx['lister'], ctx['anh'], time.time(), ctx['lister'].retried
    arcs = _find_archives(ctx, log)
    t_find = time.time() - t0
    log(f'tep nen tim thay: {len(arcs)} ({t_find:.0f} giay)' + ('' if arcs else ' -> KHONG co zip de so: chi kiem theo CSV (C5) + anh mau giai ma duoc'))
    ctx['zip_ent'], ctx['zip_obj'], ctx['zip_khong_doc'] = {}, {}, []
    for a in arcs:
        if not a['ten'].lower().endswith('.zip'):
            ctx['zip_khong_doc'].append(dict(ten=a['ten'], duong=a['duong'], gb=round(a['size'] / 1e9, 2), noi=a['noi'],
                                             loi='khong phai .zip (khong doc danh muc tu xa duoc) — chi liet ke'))
            continue
        t1, rr = time.time(), lister.retried
        try:
            zf, rf = ZR.open_remote(lambda x, y, i=a['id']: lister.fetch_range(i, x, y), a['size'])
            hint = CX.split_hint(a['ten'])
            img, oth, trung = {}, [], []
            ents = ZR.listing(zf)
            for n, s, c in ents:
                k = CX.key_of(n, hint)
                if not k:
                    oth.append((n, s, c))
                elif k in img:
                    if img[k][1:] != (s, c):
                        trung.append(k)
                else:
                    img[k] = (n, s, c)
            ctx['zip_ent'][a['id']] = dict(a=a, img=img, oth=oth, trung=trung, muc=len(ents), dang=_pattern([n for n, _, _ in img.values()], top=2),
                                           io=dict(lan_doc=rf.requests, byte_doc=rf.bytes), giay=time.time() - t1, thu_lai=lister.retried - rr)
            ctx['zip_obj'][a['id']] = zf
        except Exception as e:
            ctx['zip_khong_doc'].append(dict(ten=a['ten'], duong=a['duong'], gb=round(a['size'] / 1e9, 2), noi=a['noi'], loi=f'{type(e).__name__}: {str(e)[:160]}',
                                             giay=time.time() - t1))
    t_read = time.time() - t0 - t_find
    rows, outside, conflict = _compare(ctx)
    _log_zips(rows, log)
    if ctx['zip_ent']:
        log(f'  anh tren Drive KHONG co trong zip nao: {outside:,} | khoa xung dot giua cac zip (cung khoa, khac kich thuoc / CRC): {conflict}')
    t2 = time.time()
    picks = _sample(ctx)
    zk = ctx['zkeys']
    with ThreadPoolExecutor(lister.threads) as pool:
        res = list(pool.map(lambda kz: _check_one(lister, anh[kz[0]], zk[kz[0]][1] if kz[0] in zk else None), picks))
    sm = [dict(r, tach=CX.parts(k)[0], zip=z) for (k, z), r in zip(picks, res)]
    t_sample = time.time() - t2
    mode = collections.Counter(r.get('mode') for r in sm if r.get('mode'))
    w = [r['wh'][0] for r in sm if r.get('wh')]
    h = [r['wh'][1] for r in sm if r.get('wh')]
    theo_zip = {}
    for r in sm:
        if r['zip']:
            t = theo_zip.setdefault(r['zip'], dict(n=0, crc_dung=0))
            t['n'] += 1
            t['crc_dung'] += bool(r.get('crc_dung'))
    s = dict(n=len(sm), md5_dung=sum(bool(r.get('md5_dung')) for r in sm), crc_so=sum('crc_dung' in r for r in sm), crc_dung=sum(bool(r.get('crc_dung')) for r in sm),
             giai_ma=sum('mode' in r for r in sm), loi=dict(collections.Counter(r['loi'] for r in sm if 'loi' in r)), mode=dict(mode),
             rong=[min(w), int(np.median(w)), max(w)] if w else None, cao=[min(h), int(np.median(h)), max(h)] if h else None,
             theo_tach=dict(collections.Counter(r['tach'] for r in sm)), theo_zip=theo_zip, phut=round(t_sample / 60, 2))
    log(f"  {s['n']} anh mau (rai theo zip {({k: v['n'] for k, v in theo_zip.items()})}; theo tach {s['theo_tach']}; {lister.threads} luong): md5 = Drive "
        f"{s['md5_dung']} / {s['n']} | CRC = zip {s['crc_dung']} / {s['crc_so']} | giai ma {s['giai_ma']} / {s['n']} | mode {s['mode']} | rong min / trung vi / max "
        f"{s['rong']} | cao {s['cao']} | loi {s['loi'] or 0}")
    tm = dict(tim_zip_giay=round(t_find), doc_danh_muc_giay=round(t_read), mau_giay=round(t_sample), thu_lai=lister.retried - r0,
              theo_zip={z['a']['ten']: dict(giay=round(z['giay']), thu_lai=z['thu_lai']) for z in ctx['zip_ent'].values()})
    zt = ', '.join(f"{k} {v['giay']} s / thu lai {v['thu_lai']}" for k, v in tm['theo_zip'].items())
    log(f"  thoi gian C6 (DO): tim zip {tm['tim_zip_giay']} s | doc danh muc {tm['doc_danh_muc_giay']} s ({zt}) | anh mau {tm['mau_giay']} s | thu lai API "
        f"trong C6 {tm['thu_lai']} | tong {(time.time() - t0) / 60:.1f} phut")
    ctx['kq'] = dict(ctx['kq'], zip=dict(tep=rows, anh_ngoai_zip=outside, xung_dot_giua_zip=conflict, co_zip=bool(ctx['zip_ent']), thoi_gian=tm), mau=s)
    return dict(tep=rows, mau=s)


# ---------------- C7 ----------------
def _ensure_dir(lister, fmap, root, path, created):
    '''path (kết thúc '/', tương đối từ gốc) -> id; tạo các tầng còn thiếu (create_folder dùng lại thư mục có sẵn).'''
    if path in fmap:
        return fmap[path]
    up, _, name = path[:-1].rpartition('/')
    pid = root if not up else _ensure_dir(lister, fmap, root, up + '/', created)
    fid = lister.create_folder(pid, name)
    fmap[path] = fid
    created.append(path)
    return fid


def _zip_head(ctx, z, n, split):
    '''Thư mục (tương đối từ gốc, kết thúc '/') để đặt ảnh tách `split` lấy từ zip z: phần trước patient… mà ảnh CÙNG zip + CÙNG tách CÓ trên Drive đang
       dùng (đa số; một zip có thể chứa cả train lẫn valid ở hai thư mục); không còn ảnh nào như vậy -> nơi giải nén mặc định (thư mục chứa zip + phần trước
       patient trong tên mục, hoặc thư mục tên zip). None nếu zip ngoài cây.'''
    hz = z.setdefault('head', {})
    if split not in hz:                                  # quyết một lần, TRƯỚC khi C7 thêm ảnh của zip + tách này
        heads = collections.Counter(CX.head_of(ctx['anh'][k]['duong']) for k in z['img'] if k in ctx['anh'] and CX.parts(k)[0] == split)
        hz[split] = heads.most_common(1)[0][0] if heads else None
    if hz[split] is not None:
        return hz[split]
    a = z['a']
    if a['noi'] != 'trong cay':
        return None
    base_ = a['duong'].rpartition('/')[0]
    base_ = base_ + '/' if base_ else ''
    h = CX.head_of(n)
    return base_ + (h if h else a['ten'][:-4] + '/')


def restore(ctx, log=print):
    '''C7. Ảnh thiếu / lệch kích thước trên Drive mà zip có -> lấy từ zip (đọc từ xa đúng tệp đó; CRC) -> tải lên đúng thư mục -> md5 máy chủ = dữ liệu ->
       bản lệch cũ vào thùng rác -> liệt kê lại thư mục để xác nhận -> đối chiếu lại C5 / C6. Chỉ ghi khi tài khoản API là CHỦ thư mục CheXpert.'''
    lister, anh, t0 = ctx['lister'], ctx['anh'], time.time()
    ctx['kq'] = dict(ctx['kq'], doi_chieu_truoc_c7=ctx['kq'].get('doi_chieu'), zip_truoc_c7=ctx['kq'].get('zip'))
    todo, seen = [], set(ctx.get('xung_dot_anh') or ())   # khoá có nhiều bản khác nội dung trên Drive (C3, TRƯỢT riêng): không đoán bản nào để thay
    for zid, z in sorted(ctx['zip_ent'].items(), key=lambda x: x[1]['a']['duong']):
        for k, (n, s, c) in sorted(z['img'].items()):
            if k in seen:
                continue
            if k not in anh:
                todo.append(dict(k=k, zid=zid, n=n, s=s, c=c, loai='thieu'))
            elif anh[k]['size'] != s:
                todo.append(dict(k=k, zid=zid, n=n, s=s, c=c, loai='lech'))
            else:
                continue
            seen.add(k)
    csv_all = set().union(*_csv_keys(ctx).values()) if ctx.get('csv_chon') else set()
    not_in_zip = sorted(k for k in csv_all if k not in anh and k not in ctx.get('zkeys', {}))
    kq = dict(can=len(todo), thieu=sum(t['loai'] == 'thieu' for t in todo), lech=sum(t['loai'] == 'lech' for t in todo), xong=0, xac_nhan=0, loi=0, loi_vd={},
              thu_muc_tao=0, ban_lech_vao_thung_rac=0, csv_thieu_khong_trong_zip=len(not_in_zip), ly_do_khong_ghi=None)
    msg_nz = f" | anh cua CSV thieu tren Drive MA zip cung khong co: {len(not_in_zip):,}" + (f' ({_people_range(not_in_zip)})' if not_in_zip else '')
    if not todo:
        log('C7: khong co anh thieu / lech kich thuoc ma zip co -> KHONG ghi gi len Drive' + msg_nz)
        ctx['kq'] = dict(ctx['kq'], lay_lai=kq)
        return kq
    why = None
    if (ctx['kq'].get('cay') or {}).get('anh_sai_dang'):
        why = 'C3 co anh khong dung dang (bo doc chua hieu bo cuc) — khong ghi khi chua ro, bao Claude'
    elif len(todo) > MAX_RESTORE:
        why = f'{len(todo):,} anh > {MAX_RESTORE:,} (thieu hang loat / bo cuc la) — khong lay, bao Claude'
    elif not ctx['chu']:
        why = 'tai khoan API KHONG phai chu thu muc CheXpert (tep tai len se thuoc tai khoan nay) — chay lai bang tai khoan chu'
    if why:
        kq['ly_do_khong_ghi'] = why
        log(f"C7: can lay {len(todo):,} anh tu zip (thieu {kq['thieu']:,}, lech kich thuoc {kq['lech']:,}; {_people_range([t['k'] for t in todo])}) -> KHONG GHI: {why}"
            + msg_nz)
        ctx['kq'] = dict(ctx['kq'], lay_lai=kq)
        return kq
    log(f"C7: lay {len(todo):,} anh tu zip (thieu {kq['thieu']:,}, lech kich thuoc {kq['lech']:,}; {_people_range([t['k'] for t in todo])})")
    fmap = {f['duong']: f['id'] for f in ctx['cay']['folders']}
    created, done, errs = [], [], collections.Counter()
    for t in todo:
        z = ctx['zip_ent'][t['zid']]
        try:
            if t['loai'] == 'lech':
                old = anh[t['k']]
                folder, name = old['duong'].rpartition('/')[0] + '/', old['ten']
            else:
                sp, p, st, v = CX.parts(t['k'])
                head = _zip_head(ctx, z, t['n'], sp)
                if head is None:
                    raise LookupError('zip ngoai cay, khong biet dat anh vao dau')
                folder, name = f'{head}{p}/{st}/', v + '.jpg'
            if CX.key_of((folder + name).lstrip('/')) != t['k']:          # đọc lại khoá từ đường đích: đặt sai chỗ (vd ảnh valid vào thư mục train) -> không ghi
                raise LookupError('duong dich doc lai ra khoa khac')
            fid = ctx['root'] if folder == '/' else _ensure_dir(lister, fmap, ctx['root'], folder, created)
            zf = ctx['zip_obj'].get(t['zid'])
            if zf is None:
                zf, _ = ZR.open_remote(lambda x, y, i=t['zid']: lister.fetch_range(i, x, y), z['a']['size'])
                ctx['zip_obj'][t['zid']] = zf
            data = ZR.read_entry(zf, t['n'])
            if len(data) != t['s'] or (zlib.crc32(data) & 0xffffffff) != t['c']:
                raise IOError('du lieu lay tu zip lech danh muc')
            up = lister.upload(fid, name, data, mime='image/jpeg')
            nf = dict(duong=folder + name, ten=name, id=up['id'], size=len(data), md5=up.get('md5Checksum'), mime='image/jpeg', tang=folder.count('/') + 1)
            if t['loai'] == 'lech' and anh[t['k']]['id'] != up['id']:
                if lister.trash(anh[t['k']]['id']):
                    kq['ban_lech_vao_thung_rac'] += 1
                ctx['cay']['files'] = [f for f in ctx['cay']['files'] if f['id'] != anh[t['k']]['id']]
            ctx['cay']['files'].append(nf)
            anh[t['k']] = nf
            done.append(dict(t, fid=fid, name=name, md5=hashlib.md5(data).hexdigest()))
        except Exception as e:
            errs[type(e).__name__] += 1
    for p in created:
        ctx['cay']['folders'].append(dict(duong=p, tang=p.count('/'), id=fmap[p]))
    kids = collections.defaultdict(list)
    for f in lister.children(sorted({d['fid'] for d in done})) if done else []:
        for par in f.get('parents') or []:
            kids[(par, f['name'])].append(f)
    ok_n = sum(1 for d in done if len(kids[(d['fid'], d['name'])]) == 1 and kids[(d['fid'], d['name'])][0].get('md5Checksum') == d['md5']
               and int(kids[(d['fid'], d['name'])][0].get('size') or -1) == d['s'])
    kq.update(xong=len(done), xac_nhan=ok_n, loi=sum(errs.values()), loi_vd=dict(errs), thu_muc_tao=len(created), phut=round((time.time() - t0) / 60, 2))
    log(f"  lay xong {len(done):,} / {len(todo):,} | liet ke lai thu muc: dung 1 ban, md5 + kich thuoc = du lieu zip {ok_n:,} / {len(done):,} | loi {kq['loi']}"
        + (f' {dict(errs)}' if errs else '') + f" | thu muc tao moi {len(created)} | ban lech cu vao thung rac {kq['ban_lech_vao_thung_rac']} | {kq['phut']} phut" + msg_nz)
    ctx['kq'] = dict(ctx['kq'], lay_lai=kq)
    match(ctx, log, nhan='  sau C7 — ')
    rows, outside, conflict = _compare(ctx)
    _log_zips(rows, log, nhan='  sau C7 — ')
    z_old = dict(ctx['kq'].get('zip') or {})
    ctx['kq'] = dict(ctx['kq'], zip=dict(z_old, tep=rows, anh_ngoai_zip=outside, xung_dot_giua_zip=conflict))
    return kq


# ---------------- C8 ----------------
def propose(ctx, log=print):
    '''C8. Bảng ĐỀ XUẤT theo mặc định PLAN_93 §10 điểm 6. -> DataFrame | None.'''
    ch, fr, anh = ctx['csv_chon'], ctx['csv_bang'], ctx['anh']
    if 'train' not in ch or not ({'test', 'valid'} & set(ch)):
        log('KHONG de xuat duoc: thieu CSV train hoac ca test lan valid -> gui Claude output C3-C4')
        ctx['kq'] = dict(ctx['kq'], de_xuat=None)
        return None
    test_role = 'test' if 'test' in ch else 'valid'
    t = CX.table(fr[ch['train']['duong']], fr[ch[test_role]['duong']], test_role)
    t['md5'] = [anh[k]['md5'] if k in anh else None for k in t.khoa]
    t['byte'] = [anh[k]['size'] if k in anh else -1 for k in t.khoa]
    ppl = {s: set(t.nhom[t.tach == s]) for s in CX.ORDER}
    overlap = {f'{a}-{b}': len(ppl[a] & ppl[b]) for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    miss = sorted(k for k in t.khoa if k not in anh)
    fp = base.fingerprint_of(t.khoa.tolist(), t.tach.to_numpy(), t[list(CX.LABELS)].to_numpy())
    pos, cs = CX.positives(t), CX.class_set(t)
    n = {s: int((t.tach == s).sum()) for s in CX.ORDER}
    appa = {s: dict(collections.Counter(t.ap_pa[t.tach == s])) for s in CX.ORDER}
    same040 = all((fp[s]['md5_khoa'][:12], fp[s]['md5_nhan'][:12]) == FP_040[s] for s in CX.ORDER)
    pd_ = dict(test_tu=test_role, test_csv=ch[test_role]['duong'], train_csv=ch['train']['duong'], hat=CX.SUBSET_SEED, n_tong=CX.N_TOTAL, n_val=CX.N_VAL,
               so_anh=n, so_nguoi={s: len(v) for s, v in ppl.items()}, nguoi_chung=overlap, thieu_tren_drive=len(miss), duong=pos, tap_lop=cs,
               nhan_cuoc_thi=list(CX.COMPETITION), nguong_lop=CX.MIN_TEST_POS, ap_pa=appa, van_tay_trung_040=same040,
               van_tay={s: dict(md5_khoa=v['md5_khoa'], md5_nhan=v['md5_nhan'], so_anh=v['so_anh']) for s, v in fp.items()})
    ctx.update(bang=t, kq=dict(ctx['kq'], de_xuat=pd_))
    log(f"DE XUAT (mac dinh PLAN_93 10.6 — Tuan sua duoc truoc khi dung kho): chi anh THANG; tap hoc tu {ch['train']['duong']} — nguoi hoan vi hat "
        f"{CX.SUBSET_SEED}, lay du moi anh thang cua tung nguoi: val toi khi >= {CX.N_VAL:,} anh, roi train toi khi tong >= {CX.N_TOTAL:,}; test = "
        f"{ch[test_role]['duong']} ({'bac si gan' if test_role == 'test' else 'VALID thay test — khong co CSV test'}); nhan: 1 -> 1, 0 / -1 / trong -> 0")
    log('  ' + ' | '.join(f"{s} {n[s]:,} anh, {len(ppl[s]):,} nguoi" for s in CX.ORDER) + f' | nguoi chung giua tach {overlap} | anh thieu tren Drive {len(miss):,}'
        + (f' ({_people_range(miss)})' if miss else ''))
    log('  AP/PA (anh thang): ' + ' | '.join(f'{s} {appa[s]}' for s in CX.ORDER))
    log(f'  so anh DUONG (train / val / test) — tap lop cua luat = nhan co >= {CX.MIN_TEST_POS} anh duong o test:')
    for c in CX.LABELS:
        log(f"    {c:<27}{pos['train'][c]:>7,} / {pos['val'][c]:>5,} / {pos['test'][c]:>4,}" + ('  [tap lop]' if c in cs else '')
            + ('  [cuoc thi]' if c in CX.COMPETITION else ''))
    log(f'  tap lop: {len(cs)} nhan | van tay (md5 khoa / md5 nhan, 14 cot): ' + ' | '.join(f"{s} {fp[s]['md5_khoa'][:12]} / {fp[s]['md5_nhan'][:12]}" for s in CX.ORDER)
        + f" | = luot 0.4.0 (24/11): {'TRUNG' if same040 else 'KHAC — bao Claude'}")
    if test_role == 'test' and 'valid' in ch:
        log(f"  valid ({ch['valid']['duong']}): KHONG dung trong de xuat (RAD-DINO co the da thay anh valid — PLAN_93 2, suy tu so dem)")
    return t


# ---------------- C9 ----------------
def verdict(ctx):
    '''-> [(mục, đạt, ghi chú)]; đạt None = chỉ cảnh báo.'''
    kq, ch, out = ctx['kq'], ctx.get('csv_chon') or {}, []
    out.append(('CSV train', 'train' in ch, ch.get('train', {}).get('duong', 'khong co')))
    out.append(('CSV test (bac si gan)', True if 'test' in ch else None, ch['test']['duong'] if 'test' in ch else 'khong co -> test = valid'))
    dc = (kq.get('doi_chieu') or {}).get('theo_vai') or {}
    for r, v in dc.items():
        out.append((f'anh cua CSV {r} co du tren Drive', v['thieu'] == 0, f"thieu {v['thieu']:,}"))
    cay = kq.get('cay') or {}
    out.append(('khoa trung khac noi dung', cay.get('khoa_trung_khac_noi_dung', 0) == 0, str(cay.get('khoa_trung_khac_noi_dung', 0))))
    out.append(('cay duyet het', cay.get('chua_duyet', 0) == 0, f"{cay.get('chua_duyet', 0)} thu muc chua duyet"))
    out.append(('anh dung dang (bo doc hieu bo cuc)', cay.get('anh_sai_dang', 0) == 0, f"{cay.get('anh_sai_dang', 0):,} anh khong dung dang"))
    z = kq.get('zip') or {}
    read = [r for r in z.get('tep', []) if 'muc' in r]
    if read:
        out.append(('zip: moi anh (theo khoa) + moi tep khac (theo duong dan) co tren Drive, dung kich thuoc', all(r['dat'] for r in read),
                    '; '.join(f"{r['ten']}: anh {r['anh']:,}, thieu {r['thieu_tren_drive']}, lech {r['khac_kich_thuoc']}, tep khac thieu {r['tep_khac_thieu']} / "
                              f"lech {r['tep_khac_lech_kich_thuoc']}, trung trong zip {r['trung_trong_zip']}" for r in read)))
        out.append(('zip: khong khoa xung dot giua cac zip', z.get('xung_dot_giua_zip', 0) == 0, str(z.get('xung_dot_giua_zip', 0))))
    else:
        out.append(('zip', None, 'khong doc duoc zip nao -> khong so duoc voi ban goc'))
    bad_zip = [r['ten'] for r in z.get('tep', []) if r['ten'].lower().endswith('.zip') and 'loi' in r]
    if bad_zip:
        out.append(('zip doc duoc danh muc', None, f'khong doc duoc: {bad_zip} (vd zip chia nhieu phan / hong) -> so sanh chi tren zip doc duoc; zip nay khong xoa'))
    m = kq.get('mau') or {}
    if m:
        out.append(('anh mau: md5 = Drive, giai ma duoc', m['md5_dung'] == m['n'] and m['giai_ma'] == m['n'] and m['n'] > 0,
                    f"{m['md5_dung']} / {m['giai_ma']} / {m['n']}"))
        if read:
            cover = all(m['theo_zip'].get(r['ten'], {}).get('n', 0) > 0 for r in read if r['anh'])
            out.append(('anh mau: CRC = zip, moi zip co mau', m['crc_dung'] == m['crc_so'] == m['n'] and cover,
                        f"{m['crc_dung']} / {m['crc_so']} / {m['n']}; theo zip " + ', '.join(f"{k} {v['crc_dung']}/{v['n']}" for k, v in m['theo_zip'].items())))
    ll = kq.get('lay_lai')
    if ll is not None:
        ok = ll['can'] == 0 or (ll['ly_do_khong_ghi'] is None and ll['loi'] == 0 and ll['xac_nhan'] == ll['can'])
        out.append(('C7 lay anh thieu / lech tu zip', ok, f"can {ll['can']}, xong {ll['xong']}, xac nhan {ll['xac_nhan']}, loi {ll['loi']}"
                    + (f" — {ll['ly_do_khong_ghi']}" if ll['ly_do_khong_ghi'] else '')))
    de = kq.get('de_xuat')
    if de:
        out.append(('de xuat: du anh tren Drive, khong nguoi chung giua tach', de['thieu_tren_drive'] == 0 and not any(de['nguoi_chung'].values()),
                    f"thieu {de['thieu_tren_drive']}, nguoi chung {de['nguoi_chung']}"))
        out.append(('de xuat: van tay = luot 0.4.0', True if de['van_tay_trung_040'] else None, 'trung' if de['van_tay_trung_040'] else 'KHAC — bao Claude'))
    else:
        out.append(('de xuat', False, 'chua co'))
    return out


def write(ctx, log=print):
    '''C9. Ghi (local) JSON + bảng đề xuất + danh sách tệp -> dong_bo() -> phán quyết. -> CTX['ket_qua'] (C10 đọc).'''
    import pandas as pd
    d = ctx['drive']
    v = verdict(ctx)
    ok = all(x[1] is not False for x in v)
    js = dict(notebook=NOTEBOOK, run=d['RUN_TAG'], t=datetime.datetime.now().isoformat(timespec='seconds'), phien_ban=__version__, goi=ctx['goi'],
              nguon='MyDrive/' + '/'.join(CX.SOURCE), tai_khoan_la_chu=ctx['chu'], ket_qua=ctx['kq'], phan_quyet=[dict(muc=a, dat=b, ghi_chu=c) for a, b, c in v],
              dat=ok, phut=round((time.time() - ctx['t0']) / 60, 1),
              ghi_chu='PLAN_93 3 / 10.6. KHONG ID Drive trong tep nay; bang de xuat = mac dinh, chua khoa.')
    names = [OUT['json'], OUT['ds']]
    d['ghi'](OUT['json'], lambda q: d['ghi_json'](q, js))
    ds = pd.DataFrame([dict(duong=f['duong'], byte=f['size'], md5=f['md5']) for f in ctx['cay']['files']])
    d['ghi'](OUT['ds'], lambda q: ds.to_parquet(q, index=False))
    if ctx.get('bang') is not None:
        t = ctx['bang']
        d['ghi'](OUT['bang'], lambda q: t.to_parquet(q, index=False))
        names.append(OUT['bang'])
    du = d['dong_bo']()
    on = d['ten_tren_drive']()
    co = set(names) <= on
    log(f"C9: dong_bo {'DU' if du else 'CHUA DU'} | {len(names)} tep tren Drive: {'CO du' if co else 'THIEU -> chay lai C9, KHONG tat runtime'} | "
        f"tong {js['phut']} phut")
    log('PHAN QUYET KIEM KE:')
    for a, b, c in v:
        log(f"  {'DAT  ' if b else ('CANH BAO' if b is None else 'TRUOT')} {a} | {c}")
    log(f"=> {'KIEM KE DAT' if ok else 'KIEM KE CHUA DAT'}" + ('' if ok else ' (muc TRUOT o tren) -> C10 se KHONG dung toi zip'))
    res = dict(dat=ok, dong_bo=du, du_tep=co, phan_quyet=v)
    ctx['ket_qua'] = res
    return res


# ---------------- C10 ----------------
def trash_zips(ctx, xoa=True, log=print):
    '''C10. Zip TRONG CÂY đã so khớp hết (mọi ảnh theo khoá + mọi tệp khác theo đường dẫn, có mặt + đúng kích thước; ảnh mẫu của zip đó khớp CRC hết) ->
       THÙNG RÁC (không xoá vĩnh viễn), chỉ khi phán quyết C9 đạt hết và kết quả C9 đã lên Drive. Zip ngoài cây / đọc lỗi / chưa khớp: không đụng tới.
       Không đủ quyền (canTrash) -> in tên để Tuấn xoá tay. Ghi JSON nhỏ (tên, GB, kết quả; không ID) rồi dong_bo().'''
    d, lister, r9 = ctx['drive'], ctx['lister'], ctx.get('ket_qua')
    if not xoa:
        log('C10: XOA_ZIP = False -> khong dung toi zip')
        return dict(xoa=False)
    stop = []
    if not r9:
        stop.append('chua chay C9')
    else:
        if not r9['dat']:
            stop.append('phan quyet C9 chua DAT')
        if not (r9['dong_bo'] and r9['du_tep']):
            stop.append('ket qua C9 chua len du Drive')
    m = (ctx['kq'].get('mau') or {}).get('theo_zip') or {}
    cand, skip = [], []
    for zid, z in sorted(ctx['zip_ent'].items(), key=lambda x: x[1]['a']['duong']):
        a, why = z['a'], None
        if a['noi'] != 'trong cay':
            why = f"{a['noi']} (ngoai thu muc CheXpert)"
        elif not z.get('ok'):
            why = 'chua khop het voi Drive'
        elif not (m.get(a['ten'], {}).get('n', 0) > 0 and m[a['ten']]['crc_dung'] == m[a['ten']]['n']):
            why = 'anh mau chua khop CRC het / khong co mau'
        (skip if why else cand).append((zid, a, why))
    for r in ctx['zip_khong_doc']:
        skip.append((None, r, r.get('loi', 'khong doc')))
    for _, a, why in skip:
        log(f"  khong dung toi: {a['duong']} — {why}")
    out = dict(xoa=True, dung=stop, da_vao_thung_rac=[], da_o_thung_rac_tu_truoc=[], khong_du_quyen=[], loi=[],
               bo_qua=[dict(duong=a['duong'], ly_do=w) for _, a, w in skip])
    if stop or not cand:
        log('C10: KHONG chuyen zip nao vao thung rac — ' + ('; '.join(stop) if stop else 'khong co zip nao du dieu kien'))
    else:
        for zid, a, _ in cand:
            try:
                info = lister.get(zid)
                if info.get('trashed'):
                    out['da_o_thung_rac_tu_truoc'].append(a['ten'])
                    continue
                if info.get('name') != a['ten'] or int(info.get('size') or -1) != a['size']:
                    out['loi'].append(f"{a['ten']}: ten / kich thuoc doi tu luc doc")
                    continue
                if not (info.get('capabilities') or {}).get('canTrash'):
                    out['khong_du_quyen'].append(a['ten'])
                    continue
                lister.trash(zid)
                if lister.get(zid).get('trashed'):
                    out['da_vao_thung_rac'].append(a['ten'])
                else:
                    out['loi'].append(f"{a['ten']}: may chu chua bao trashed")
            except Exception as e:
                out['loi'].append(f"{a['ten']}: {type(e).__name__}")
        gb = {a['ten']: round(a['size'] / 1e9, 2) for _, a, _ in cand}
        out['gb_vao_thung_rac'] = round(sum(gb[n] for n in out['da_vao_thung_rac']), 2)
        log(f"C10: vao THUNG RAC {len(out['da_vao_thung_rac'])} zip ({out['gb_vao_thung_rac']:,} GB): {out['da_vao_thung_rac']}"
            + (f" | da o thung rac tu truoc: {out['da_o_thung_rac_tu_truoc']}" if out['da_o_thung_rac_tu_truoc'] else '')
            + (f" | KHONG DU QUYEN (Tuan xoa tay tren Drive): {out['khong_du_quyen']}" if out['khong_du_quyen'] else '')
            + (f" | LOI: {out['loi']}" if out['loi'] else ''))
        log('  thung rac van tinh dung luong toi khi don; khoi phuc duoc trong 30 ngay. DON THUNG RAC SAU KHI Claude doc output C2-C10.')
    n = ctx.get('_xoa_lan', 0) + 1
    ctx['_xoa_lan'] = n
    name = OUT['xoa'] + ('' if n == 1 else f'_{n}') + '.json'
    js = dict(notebook=NOTEBOOK, run=d['RUN_TAG'], t=datetime.datetime.now().isoformat(timespec='seconds'), phien_ban=__version__, ket_qua=out)
    d['ghi'](name, lambda q: d['ghi_json'](q, js))
    du = d['dong_bo']()
    log(f"  ghi {name}: dong_bo {'DU' if du else 'CHUA DU -> chay lai o nay'} | gui Claude output C2-C10")
    return out
