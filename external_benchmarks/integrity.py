# -*- coding: utf-8 -*-
"""Toàn vẹn dữ liệu trên Drive (PLAN_88 §14.20; Tuấn 12/11: "hãy kiểm tra tính toàn vẹn của 2 data"): bảng (tên, md5, kích thước) của từng tệp trên
Drive — CHỈ đọc siêu dữ liệu qua Drive API (md5Checksum + size), không tải — so với MỐC tính trên máy Claude từ bản cục bộ đã kiểm:
  - Shenzhen (MyDrive/MIMIC_CXR/Shenzhen): mốc TỪNG tệp (shenzhen_manifest: 662 PNG + shenzhen_metadata.csv) -> thiếu / thừa / khác md5 / khác kích thước
    chỉ đúng tên tệp;
  - RSNA (MyDrive/MIMIC_CXR/RSNA_pneumonia): ~30 nghìn tệp, quá lớn để ghim từng tệp -> mốc TỔNG + 256 NHÓM (nhóm = 2 ký tự hex đầu của md5(tên tệp)) cho
    DICOM train / test (rsna_manifest) + từng tệp cho 3 CSV -> lệch thì chỉ ra nhóm (số tệp / dung lượng / nội dung).
Khoá = TÊN tệp (không đường dẫn: cây thư mục trên Drive có thể lồng khác máy); tập con RSNA chọn theo tên thư mục cha (stage_2_train_images /
stage_2_test_images). Tệp trùng tên: cùng md5 -> dùng một bản; khác md5 -> DỪNG. Không giữ trường nào khác của Drive (chủ sở hữu / email)."""
import hashlib
import re

FOLDER_MIME = 'application/vnd.google-apps.folder'
N_BUCKETS_HEX = 2                 # 2 ký tự hex -> 256 nhóm


def walk_drive(list_children, root_id, max_depth=6):
    '''Duyệt cây thư mục Drive từ root_id. list_children(id) -> [dict(id, name, mimeType, md5Checksum?, size?, ...)] (= _liet_ke của khối Drive chuẩn).
       -> [dict(id, ten, md5, size, duong, cha)] cho mọi TỆP (bỏ thư mục), duong = đường dẫn tương đối từ root, cha = tên thư mục chứa trực tiếp.'''
    out, stack = [], [(root_id, '', '', 0)]
    while stack:
        fid, path, parent, depth = stack.pop()
        for f in list_children(fid):
            if f.get('mimeType') == FOLDER_MIME:
                if depth < max_depth:
                    stack.append((f['id'], f"{path}{f['name']}/", f['name'], depth + 1))
                continue
            out.append(dict(id=f['id'], ten=f['name'], md5=f.get('md5Checksum'), size=int(f.get('size') or 0), duong=path + f['name'], cha=parent))
    return sorted(out, key=lambda x: x['duong'])


def _bucket(name):
    return hashlib.md5(name.encode('utf-8')).hexdigest()[:N_BUCKETS_HEX]


def table_digest(rows):
    '''rows: [(tên, md5, kích thước)] -> dict(n, kich_thuoc, md5 (của bảng "tên\\tmd5\\tkích thước\\n" sắp theo tên), nhom {2 hex: [n, kích thước, md5]}).'''
    rows = sorted(rows)
    lines = [f'{n}\t{m}\t{s}\n' for n, m, s in rows]
    groups = {}
    for (n, m, s), line in zip(rows, lines):
        g = groups.setdefault(_bucket(n), [0, 0, hashlib.md5()])
        g[0] += 1
        g[1] += s
        g[2].update(line.encode('utf-8'))
    return dict(n=len(rows), kich_thuoc=sum(s for _, _, s in rows), md5=hashlib.md5(''.join(lines).encode('utf-8')).hexdigest(),
                nhom={b: [g[0], g[1], g[2].hexdigest()] for b, g in sorted(groups.items())})


def _dedupe(files):
    '''{tên: bản} — bản trùng tên cùng md5 + kích thước -> một bản; khác -> ghi vào xung_dot.'''
    by, conflict = {}, {}
    for f in files:
        prev = by.get(f['ten'])
        if prev is None:
            by[f['ten']] = f
        elif (prev['md5'], prev['size']) != (f['md5'], f['size']):
            conflict.setdefault(f['ten'], [prev['duong']]).append(f['duong'])
    return by, conflict


def compare_files(files, expected):
    '''Từng tệp: files [dict(ten, md5, size, duong)] (đã lọc) vs expected {tên: (md5, kích thước)}. -> dict(dat, so_tep, thieu, thua, khac_md5,
       khac_kich_thuoc, khong_md5, xung_dot, chon {tên: bản dùng}). Thừa (tên ngoài mốc) không làm trượt.'''
    by, conflict = _dedupe(files)
    no_md5 = sorted(n for n, f in by.items() if not f['md5'])
    missing = sorted(set(expected) - set(by))
    extra = sorted(set(by) - set(expected))
    bad_md5 = sorted(n for n in set(expected) & set(by) if by[n]['md5'] and by[n]['md5'] != expected[n][0])
    bad_size = sorted(n for n in set(expected) & set(by) if by[n]['size'] != expected[n][1])
    ok = not (missing or bad_md5 or bad_size or no_md5 or conflict)
    return dict(dat=ok, so_tep=len(set(expected) & set(by)), thieu=missing, thua=extra, khac_md5=bad_md5, khac_kich_thuoc=bad_size, khong_md5=no_md5,
                xung_dot=conflict, chon={n: by[n] for n in sorted(set(expected) & set(by))})


def compare_digest(files, expected):
    '''Theo nhóm: files (đã lọc, một tập con) vs expected = table_digest của mốc. -> dict(dat, n, kich_thuoc, md5_bang, khong_md5, xung_dot, nhom_lech
       [dict(nhom, drive [n, kích thước], moc [n, kích thước], cung_so_tep_va_dung_luong)]). Chỉ nhóm lệch mới in -> biết thiếu / thừa tệp hay khác nội dung.'''
    by, conflict = _dedupe(files)
    no_md5 = sorted(n for n, f in by.items() if not f['md5'])
    got = table_digest([(n, f['md5'] or '', f['size']) for n, f in by.items()])
    diff = []
    for b in sorted(set(got['nhom']) | set(expected['nhom'])):
        g, e = got['nhom'].get(b, [0, 0, '']), expected['nhom'].get(b, [0, 0, ''])
        if g != e:
            diff.append(dict(nhom=b, drive=g[:2], moc=e[:2], cung_so_tep_va_dung_luong=g[:2] == e[:2]))
    ok = got['md5'] == expected['md5'] and not diff and not no_md5 and not conflict
    return dict(dat=ok, n=got['n'], n_moc=expected['n'], kich_thuoc=got['kich_thuoc'], kich_thuoc_moc=expected['kich_thuoc'], md5_bang=got['md5'],
                khong_md5=no_md5, xung_dot=conflict, nhom_lech=diff)


_SZ_PNG = re.compile(r'CHNCXR_\d{4}_[01]\.png')


def check_shenzhen(files):
    '''Shenzhen trên Drive vs mốc từng tệp (662 PNG + shenzhen_metadata.csv). -> dict(dat, png, metadata, tong_tep_drive, thua_ngoai_mau).'''
    from . import shenzhen_manifest as man
    png = compare_files([f for f in files if _SZ_PNG.fullmatch(f['ten'])], man.PNG)
    meta = compare_files([f for f in files if f['ten'] == man.META[0]], {man.META[0]: man.META[1:]})
    other = sorted(f['duong'] for f in files if not _SZ_PNG.fullmatch(f['ten']) and f['ten'] != man.META[0])
    return dict(dat=png['dat'] and meta['dat'], png=png, metadata=meta, tong_tep_drive=len(files), thua_ngoai_mau=other[:20], so_thua_ngoai_mau=len(other))


def check_rsna(files):
    '''RSNA trên Drive vs mốc: DICOM train (thư mục cha stage_2_train_images) / test (stage_2_test_images) theo tổng + 256 nhóm; 3 CSV từng tệp.
       -> dict(dat, train, test, csv, tong_tep_drive, thua_ngoai_mau).'''
    from . import rsna_manifest as man
    sub = {k: [f for f in files if f['cha'] == folder and f['ten'].endswith('.dcm')] for k, folder in man.FOLDERS.items()}
    res = {k: compare_digest(sub[k], man.DIGESTS[k]) for k in man.FOLDERS}
    csv = compare_files([f for f in files if f['ten'] in man.CSV], man.CSV)
    used = {id(f) for v in sub.values() for f in v} | {id(f) for f in files if f['ten'] in man.CSV}
    other = sorted(f['duong'] for f in files if id(f) not in used)
    return dict(dat=all(r['dat'] for r in res.values()) and csv['dat'], csv=csv, tong_tep_drive=len(files), thua_ngoai_mau=other[:20],
                so_thua_ngoai_mau=len(other), **res)


def for_log(res):
    '''Kết quả check_* để GHI (JSON): bỏ 'chon' (bản Drive đã chọn từng tệp — mang ID tệp Drive, dài), giữ mọi số đếm / tên lệch.'''
    return {k: ({kk: vv for kk, vv in v.items() if kk != 'chon'} if isinstance(v, dict) else v) for k, v in res.items()}


def summary(name, res):
    '''Vài dòng tóm tắt (không in mã tệp RSNA từng ảnh — chỉ số đếm / nhóm; tên PNG Shenzhen công khai).'''
    lines = [f"{name}: {'DAT — trung moc tung byte (md5 + kich thuoc)' if res['dat'] else 'KHONG DAT'} | {res['tong_tep_drive']} tep tren Drive, "
             f"{res['so_thua_ngoai_mau']} tep ngoai mau (khong tinh){': ' + ', '.join(res['thua_ngoai_mau'][:5]) if res['thua_ngoai_mau'] else ''}"]
    for k, r in res.items():
        if not isinstance(r, dict) or 'dat' not in r:
            continue
        if 'nhom_lech' in r:
            lines.append(f"  {k}: {r['n']} / {r['n_moc']} tep, {r['kich_thuoc']:,} / {r['kich_thuoc_moc']:,} B -> {'trung' if r['dat'] else 'LECH'}"
                         + (f" | nhom lech {len(r['nhom_lech'])}: " + '; '.join(f"{d['nhom']} drive {d['drive']} moc {d['moc']}" for d in r['nhom_lech'][:8])
                            if r['nhom_lech'] else '') + (f" | khong md5 {len(r['khong_md5'])}" if r['khong_md5'] else '')
                         + (f" | trung ten khac noi dung {len(r['xung_dot'])}" if r['xung_dot'] else ''))
        else:
            lines.append(f"  {k}: {r['so_tep']} tep khop ten -> {'trung' if r['dat'] else 'LECH'}"
                         + ''.join(f" | {lab} {len(r[key])}: {', '.join(r[key][:5])}" for key, lab in
                                   (('thieu', 'thieu'), ('khac_md5', 'khac md5'), ('khac_kich_thuoc', 'khac kich thuoc'), ('khong_md5', 'khong md5'))
                                   if r[key])
                         + (f" | trung ten khac noi dung {len(r['xung_dot'])}" if r['xung_dot'] else ''))
    return '\n'.join(lines)
