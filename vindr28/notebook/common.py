# -*- coding: utf-8 -*-
"""Phần dùng chung của hai notebook 94 (CPU): khối Drive, in số, ghi JSON / CSV, bộ nhớ đệm + tệp resume."""
import csv
import io
import json
import re

import numpy as np

from .. import config as C
from .. import stats28 as S

DRIVE_NAMES = ('kiem_ke', 'tai', 'xung_dot', 'ghi', 'ghi_json', 'dong_bo', 'ten_tren_drive', 'RUN_TAG', 'GHI_LOC', 'luu_resume', 'TEN_RESUME')
CACHE_RE = r'bo_dem_28_\d{8}_\d{6}\.npz'


def quiet(*a, **k):
    '''Tắt log của hàm vindr_bench / frozen_probe gọi lại (chữ của chúng nói "mAP21"); notebook 94 tự in bảng 28 nhãn.'''
    return None


def drive_api(ns, run_tag):
    missing = [k for k in DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    d = {k: ns[k] for k in DRIVE_NAMES}
    assert d['RUN_TAG'] == run_tag, f"RUN_TAG {d['RUN_TAG']!r} != {run_tag!r} -> DUNG"
    return d


def f3(x, nd=3):
    return '  -  ' if x is None or x != x else f'{x:.{nd}f}'


def s3(x, nd=3):
    return '   -  ' if x is None or x != x else f'{x:+.{nd}f}'


def ci(v, nd=3):
    return '[   -   ]' if not v or v[0] is None or v[0] != v[0] else f'[{s3(v[0], nd)}, {s3(v[1], nd)}]'


def above(v):
    '''Khoảng (cận dưới, cận trên) nằm hẳn trên 0 / dưới 0 / chứa 0 -> '>0' | '<0' | 'chua 0'.'''
    if not v or v[0] is None or v[0] != v[0]:
        return '-'
    return '>0' if v[0] > 0 else '<0' if v[1] < 0 else 'chua 0'


def load_caches(d, folder_tag, log=print):
    '''Bộ nhớ đệm 28 nhãn của lượt trước (tệp bo_dem_28_* MỚI NHẤT) + tệp resume (lượt đang dở) trong thư mục ghi của notebook -> {khoá: set_stats28}.'''
    kk = d['kiem_ke'](folder_tag, mau=rf"{re.escape(d['TEN_RESUME'])}|{CACHE_RE}")[1]
    cache, src = {}, []
    full = sorted(n for n in kk if re.fullmatch(CACHE_RE, n))
    if full:
        c = S.load_cache(d['tai'](kk[full[-1]][0]))
        cache.update(c)
        src.append(f'{full[-1]} ({len(c)} bo)')
    if d['TEN_RESUME'] in kk:
        c = S.load_cache(d['tai'](kk[d['TEN_RESUME']][0]))
        cache.update(c)
        src.append(f"{d['TEN_RESUME']} ({len(c)} bo, luot dang do)")
    log(f"bo nho dem 28 nhan: {' + '.join(src) if src else 'chua co (tinh tu dau)'}")
    return cache


def save_resume(d, cache):
    '''Ghi đè MỘT tệp resume trên Drive (luu_resume của khối chuẩn). Không ném lỗi (khối chuẩn tự in cảnh báo).'''
    def w(q):
        with open(q, 'wb') as fh:
            S.save_cache(fh, cache)
    return d['luu_resume'](w)


def write_cache(d, name, cache):
    def w(q):
        with open(q, 'wb') as fh:
            S.save_cache(fh, cache)
    d['ghi'](name, w)


def jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda v: v.item() if hasattr(v, 'item') else (v.tolist() if hasattr(v, 'tolist') else str(v))))


def write_csv(d, name, rows):
    if not rows:
        return False
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
    text = buf.getvalue()

    def _w(q):
        with open(q, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(text)
    d['ghi'](name, _w)
    return True


def pct_ci(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float('nan')] * 2


def run_jobs(ctx, jobs, keys, cache, log=print, workers=None, on_new=None):
    '''Tính set_stats28 cho các bộ CHƯA có trong bộ nhớ đệm (theo khoá), lưu resume mỗi ~10 %. on_new(i, st) -> dict cổng (ghi vào st['cong']) hoặc None.
       -> [set_stats28 của mọi bộ, đúng thứ tự jobs] (bản đã đưa về float32 như bộ nhớ đệm).'''
    import os
    import time
    d = ctx['drive']
    todo = [i for i, k in enumerate(keys) if k not in cache]
    workers = workers or min(2, os.cpu_count() or 1)
    log(f"{len(jobs)} bo du doan | da co trong bo nho dem 28 nhan: {len(jobs) - len(todo)} | can tinh: {len(todo)} ({workers} tien trinh; do tren may: "
        f"~2,5 giay / bo diem lien tuc, ~0,8 giay / bo diem roi rac, mot nhan)")
    every = max(1, int(len(todo) * C.RESUME_EVERY))
    done = [0]
    t0 = time.time()

    def prog(k, st):
        i = todo[k]
        g = on_new(i, st) if on_new is not None else None
        st = S.stored(st)
        if g is not None:
            st['cong'] = g
        cache[keys[i]] = st
        done[0] += 1
        if done[0] % every == 0 and done[0] < len(todo):
            save_resume(d, cache)

    if todo:
        S.many([jobs[i] for i in todo], workers, log=log, on_progress=prog)
        save_resume(d, cache)
    ctx['phut_tinh'] = (time.time() - t0) / 60
    return [cache[k] for k in keys]
