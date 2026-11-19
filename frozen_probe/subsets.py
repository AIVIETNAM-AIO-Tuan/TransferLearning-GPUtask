# -*- coding: utf-8 -*-
"""Lần rút chỉ số theo tỉ lệ nhãn (PLAN_92 §11.2): rút ngẫu nhiên KHÔNG lặp từ bể học, hạt cố định -> CÙNG chỉ số cho mọi hàng (ghép cặp)."""
import hashlib

import numpy as np

from . import config as C


def size(n_pool, fraction):
    '''Số ảnh của một lần rút: round(tỉ lệ x bể) (bể 15.000 -> 150 / 1.500 / 15.000).'''
    return int(round(float(fraction) * int(n_pool)))


def draw(pool, fraction, d, seed=C.DRAW_SEED):
    '''Lần rút thứ d ở tỉ lệ `fraction` -> chỉ số (sắp tăng) trong `pool`. Tỉ lệ 1,0 = cả bể (không ngẫu nhiên).'''
    pool = np.sort(np.asarray(pool, dtype=np.int64))
    n = size(len(pool), fraction)
    assert 0 < n <= len(pool), f'lan rut {fraction}: {n} anh / be {len(pool)} -> DUNG'
    if n == len(pool):
        return pool.copy()
    rng = np.random.default_rng([int(seed), int(round(float(fraction) * 1000)), int(d)])
    return np.sort(rng.choice(pool, size=n, replace=False))


def all_draws(pool, seed=C.DRAW_SEED):
    '''-> {(tỉ lệ, d): chỉ số} cho mọi tỉ lệ / lần rút của config.'''
    return {(f, d): draw(pool, f, d, seed) for f in C.FRACTIONS for d in range(C.N_DRAWS[f])}


def md5(draws):
    '''md5 của mọi lần rút (thứ tự khoá cố định) — in ở notebook để đối chiếu giữa các lần chạy.'''
    h = hashlib.md5()
    for k in sorted(draws):
        h.update(repr(k).encode())
        h.update(np.ascontiguousarray(draws[k], dtype=np.int64).tobytes())
    return h.hexdigest()
