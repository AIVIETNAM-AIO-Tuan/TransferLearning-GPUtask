# -*- coding: utf-8 -*-
"""Day goi nhanh_c (CHI cac tep .py trong experiments/84_nhanh_c/nhanh_c) len GitHub, nhanh mo coi `nhanh-c` (Tuan 29/10: "chi push moi cac
   file python do"; docs/SPEC_SAU_78Y.md B.2ab).

   Commit dung bang chi muc TAM (GIT_INDEX_FILE): cay chi gom nhanh_c/**/*.py, cha = dinh origin/nhanh-c neu co -> KHONG dung cay lam viec,
   chi muc that hay nhanh main (main di truoc origin 46 commit: KHONG day). Quet bi mat (email / token / ID Drive) truoc khi day. Ghi
   da_day.json (commit + blob tung tep) -> dung_84.py --notebook ghim commit nay. Chay: python day_nhanh_c.py [--thu] (--thu: dung commit, KHONG day)."""
import datetime, hashlib, json, os, re, subprocess, sys, tempfile

D = os.path.dirname(os.path.abspath(__file__))
GOI = os.path.join(D, 'nhanh_c')
REPO = os.path.dirname(os.path.dirname(D))
NHANH = 'nhanh-c'
TEP_DA_DAY = os.path.join(D, 'da_day.json')
MAU_BI_MAT = {'email': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z.]{2,}',
              'token': r'ghp_|github_pat_|AIza[0-9A-Za-z_-]{20}|ya29\.|\bhf_[A-Za-z]{20}|BEGIN [A-Z ]*PRIVATE KEY',
              'id_drive': r"['\"/=]1[A-Za-z0-9_-]{27,43}['\"\s)/&]"}


def git(*a, env=None, nhap=None):
    r = subprocess.run(['git', '-C', REPO, *a], capture_output=True, text=True, env=env, input=nhap)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(a)}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def tep_goi():
    out = {}
    for goc, _, fs in os.walk(GOI):
        if '__pycache__' in goc: continue
        for f in sorted(fs):
            p = os.path.join(goc, f)
            assert f.endswith('.py'), f'goi co tep KHONG phai .py: {p} -> DUNG'
            out['nhanh_c/' + os.path.relpath(p, GOI).replace(os.sep, '/')] = open(p, 'rb').read()
    return dict(sorted(out.items()))


def blob_git(b):
    return hashlib.sha1(b'blob %d\0' % len(b) + b).hexdigest()


def main(thu=False):
    tep = tep_goi()
    bm = {k: sorted({m for b in tep.values() for m in re.findall(v, b.decode('utf-8'))}) for k, v in MAU_BI_MAT.items()}
    assert not any(bm.values()), f'goi chua chuoi bi mat / ID -> KHONG day: {bm}'
    try:
        git('fetch', '--quiet', 'origin', f'{NHANH}:refs/remotes/origin/{NHANH}')
    except RuntimeError as e:
        if "couldn't find remote ref" not in str(e) and 'not found' not in str(e).lower():
            raise
    try:
        cha = git('rev-parse', '--verify', '-q', f'refs/remotes/origin/{NHANH}')
    except RuntimeError:
        cha = ''
    chi_muc = os.path.join(tempfile.mkdtemp(), 'index_nhanh_c')
    env = dict(os.environ, GIT_INDEX_FILE=chi_muc)
    git('read-tree', '--empty', env=env)
    blob = {}
    for ten, b in tep.items():
        p = os.path.join(GOI, *ten.split('/')[1:])
        sha_b = git('hash-object', '-w', '--no-filters', p)
        assert sha_b == blob_git(b), f'{ten}: blob git {sha_b} KHAC tinh tay -> DUNG'
        git('update-index', '--add', '--cacheinfo', f'100644,{sha_b},{ten}', env=env)
        blob[ten.split('/', 1)[1]] = sha_b
    cay = git('write-tree', env=env)
    if cha and git('rev-parse', f'{cha}^{{tree}}') == cay:
        sha = cha
        print(f'goi KHONG doi so voi origin/{NHANH} ({cha[:12]}) -> khong tao commit moi')
    else:
        with open(os.path.join(REPO, '.git', 'NHANH_C_MSG'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(f"nhanh-c: goi nhanh_c ({len(tep)} tep .py)\n\nNhanh C chay theo seed - moi buoc = mot o notebook da kiem cua 78 (goi o cat s42 /"
                    " skip7), nguyen van + va theo seed; sinh boi experiments/84_nhanh_c/dung_84.py (docs/SPEC_SAU_78Y.md B.2ab).\n"
                    "Nhanh mo coi: CHI goi nhanh_c, khong lich su main.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
        sha = git('commit-tree', cay, *(['-p', cha] if cha else []), '-F', os.path.join(REPO, '.git', 'NHANH_C_MSG'))
        os.remove(os.path.join(REPO, '.git', 'NHANH_C_MSG'))
    ten_cay = git('ls-tree', '-r', '--name-only', sha).split('\n')
    assert ten_cay == list(tep) and all(t.startswith('nhanh_c/') and t.endswith('.py') for t in ten_cay), 'cay commit KHAC danh sach goi -> DUNG'
    if thu:
        print(f'THU: commit {sha} ({len(ten_cay)} tep, cha {cha[:12] or "-"}) - KHONG day')
        return sha
    if sha != cha:
        git('push', 'origin', f'{sha}:refs/heads/{NHANH}')
    tren = git('ls-remote', 'origin', f'refs/heads/{NHANH}').split('\t')[0]
    assert tren == sha, f'origin/{NHANH} = {tren} KHAC commit vua day {sha} -> DUNG'
    git('update-ref', f'refs/heads/{NHANH}', sha)
    json.dump(dict(sha=sha, cay=cay, cha=cha or None, nhanh=NHANH, url=git('remote', 'get-url', 'origin'), so_tep=len(tep), blob=blob,
                   t=datetime.datetime.now().isoformat(timespec='seconds')), open(TEP_DA_DAY, 'w', encoding='utf-8'), indent=1, sort_keys=True)
    print(f'da day {NHANH} = {sha} ({len(tep)} tep .py; cha {cha[:12] or "- (nhanh moi)"}) | origin xac nhan | da_day.json')
    return sha


if __name__ == '__main__':
    main(thu='--thu' in sys.argv)
