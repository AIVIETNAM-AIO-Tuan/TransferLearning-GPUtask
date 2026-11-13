# -*- coding: utf-8 -*-
"""Cổng số học trên đặc trưng của lớp bọc (PLAN_89 §8 mục 4): so hai tệp đặc trưng 4 tầng cuối (patch + vector toàn ảnh, cùng ảnh đầu vào) của hai
lượt chạy khác cách tính (fp16 / fp32, GPU / CPU, môi trường cũ / mới).

Luật (chép relative_gate của vindr_protocol.upstream.gpu_probe @ 0.8.2 — đã chạy ở 86j — đổi ĐÚNG phần chặn hỏng nặng):
  - tương đối theo NỀN (cùng phép so ở môi trường mới, cùng lớp bọc / ảnh / GPU):
      lệch ≤ 2 × lệch_nền + 1e-3   VÀ   1 − cos_min ≤ 4 × (1 − cos_min_nền) + 1e-4
    (1 − cos ≈ góc² / 2 -> hệ số 4 trên 1 − cos = sai số gấp 2, khớp hệ số 2 của lệch);
  - chặn hỏng nặng BỀN (thay chặn tuyệt đối lệch ≤ 2e-2 / cos_min ≥ 0,99 đã làm 86j_raddino_chuan dừng dù môi trường cũ = nền):
      trung vị cos ≥ 0,999   VÀ   phân vị 1e-2 của cos ≥ 0,99   (theo token, phần tệ nhất trong patch / vector).
    Nền đã đo (trung vị / phân vị 1e-2): RAD-DINO thô 512 ~1,000000 / 0,999975; RAD-DINO qua layernorm 518 0,999999 / 0,999786; BiomedCLIP CLS 1,0 / 1,0.
    Hỏng thật (kernel sai, tràn số) làm tụt phần lớn token -> trung vị / phân vị bắt được; vài token lệch do fp16 thì không.
  - "lệch" = max |Δ| / max |tham chiếu| trên cả mảng; cos theo từng vector token (patch và vector toàn ảnh, 4 tầng, mọi ảnh).

Kết quả cổng ghi luôn chặn bền của NỀN (nen_qua_chan_ben): nền trượt chặn bền = chặn chưa hiệu chỉnh cho encoder này HOẶC fp16 hỏng ở cả hai môi
trường — notebook dừng và in rõ trường hợp nào (memory: thử mọi ngưỡng trên nền của đúng loại đặc trưng)."""
import numpy as np

RULE = dict(he_so_lech=2.0, cong_lech=1e-3, he_so_thieu_cos=4.0, cong_thieu_cos=1e-4, cos_trung_vi_san=0.999, cos_q01_san=0.99)
PARTS = ('patch', 'cls')


def _cos(R, A):
    Rf, Af = R.reshape(-1, R.shape[-1]), A.reshape(-1, A.shape[-1])
    return (Rf * Af).sum(1) / np.maximum(np.linalg.norm(Rf, axis=1) * np.linalg.norm(Af, axis=1), 1e-12)


def compare_features(ref_npz, test_npz):
    '''Hai tệp .npz (khoá 'patch' [L, B, T, D] và 'cls' [L, B, D]) -> mỗi phần: lệch tương đối, cos_min / trung vị / phân vị 1e-2 / TB; tổng = phần tệ
       nhất. Chỉ phần có ở cả hai tệp; 'cls' bắt buộc.'''
    r, a = np.load(ref_npz), np.load(test_npz)
    parts = [p for p in PARTS if p in r.files and p in a.files]
    assert 'cls' in parts, 'thieu cls -> DUNG'
    out = {}
    for part in parts:
        R, A = r[part].astype(np.float64), a[part].astype(np.float64)
        assert R.shape == A.shape, (part, R.shape, A.shape)
        cos = _cos(R, A)
        out[part] = dict(lech_tuong_doi=float(np.abs(A - R).max() / max(np.abs(R).max(), 1e-12)), cos_min=float(cos.min()),
                         cos_trung_vi=float(np.median(cos)), cos_q01=float(np.quantile(cos, 1e-2)), cos_tb=float(cos.mean()), so_vector=int(cos.size))
    for k, f in (('lech_tuong_doi', max), ('cos_min', min), ('cos_trung_vi', min), ('cos_q01', min)):
        out[k] = f(out[p][k] for p in parts)
    out['phan'] = parts
    return out


def robust_guard(cmp, rule=RULE):
    '''Chặn hỏng nặng bền: trung vị cos ≥ cos_trung_vi_san và phân vị 1e-2 ≥ cos_q01_san (phần tệ nhất).'''
    return bool(cmp['cos_trung_vi'] >= rule['cos_trung_vi_san'] and cmp['cos_q01'] >= rule['cos_q01_san'])


def relative_gate(cmp_test, cmp_floor, rule=RULE):
    '''cmp_* = compare_features(tham chiếu, thử) của phép so cần kiểm và của NỀN. -> dict(dat, ngưỡng, nền, chặn bền của cả hai, luật).'''
    assert cmp_test['phan'] == cmp_floor['phan'], (cmp_test['phan'], cmp_floor['phan'])
    lim_l = rule['he_so_lech'] * cmp_floor['lech_tuong_doi'] + rule['cong_lech']
    lim_t = rule['he_so_thieu_cos'] * (1.0 - cmp_floor['cos_min']) + rule['cong_thieu_cos']
    lech, cos = cmp_test['lech_tuong_doi'], cmp_test['cos_min']
    tuong_doi = bool(lech <= lim_l and (1.0 - cos) <= lim_t)
    ben, ben_nen = robust_guard(cmp_test, rule), robust_guard(cmp_floor, rule)
    return dict(dat=bool(tuong_doi and ben), dat_tuong_doi=tuong_doi, dat_chan_ben=ben, nen_qua_chan_ben=ben_nen, lech=lech, cos_min=cos,
                cos_trung_vi=cmp_test['cos_trung_vi'], cos_q01=cmp_test['cos_q01'], nguong_lech=lim_l, nguong_cos_min=1.0 - lim_t,
                lech_nen=cmp_floor['lech_tuong_doi'], cos_min_nen=cmp_floor['cos_min'], cos_trung_vi_nen=cmp_floor['cos_trung_vi'],
                cos_q01_nen=cmp_floor['cos_q01'], luat=dict(rule), phan=cmp_test['phan'])


def gate_text(name, g):
    '''Một dòng tóm tắt + chẩn đoán khi trượt.'''
    s = (f"{name}: lech {g['lech']:.2e} <= {g['nguong_lech']:.2e} (nen {g['lech_nen']:.2e}) | cos_min {g['cos_min']:.6f} >= {g['nguong_cos_min']:.6f} "
         f"(nen {g['cos_min_nen']:.6f}) | chan ben: cos trung vi {g['cos_trung_vi']:.6f} (nen {g['cos_trung_vi_nen']:.6f}) >= "
         f"{g['luat']['cos_trung_vi_san']}, cos q1e-2 {g['cos_q01']:.6f} (nen {g['cos_q01_nen']:.6f}) >= {g['luat']['cos_q01_san']} -> "
         f"{'DAT' if g['dat'] else 'TRUOT'}")
    if not g['dat']:
        if not g['dat_tuong_doi']:
            s += ' | TRUOT phan tuong doi: moi truong cu lech HON nen'
        if not g['dat_chan_ben']:
            s += (' | TRUOT chan ben; nen ' + ('DAT chan ben -> moi truong cu hong' if g['nen_qua_chan_ben'] else
                                               'CUNG TRUOT chan ben -> chan chua hieu chinh cho encoder nay HOAC fp16 hong o ca hai moi truong'))
    return s


def feature_stats(ref_npz, test_npz, n_worst=5):
    '''Thống kê để đọc / lưu (KHÔNG phải cổng): mỗi phần -> lệch, cos_min, phân vị 1e-4 / 1e-3 / 1e-2 / 0,5, số vector dưới 0,999, cos_min theo tầng,
       n_worst vector tệ nhất (tầng −L..−1, cos, độ lớn / trung vị độ lớn cùng tầng ở tham chiếu).'''
    r, a = np.load(ref_npz), np.load(test_npz)
    out = {}
    for part in [p for p in PARTS if p in r.files and p in a.files]:
        R, A = r[part].astype(np.float64), a[part].astype(np.float64)
        assert R.shape == A.shape, (part, R.shape, A.shape)
        R, A = R.reshape(R.shape[0], -1, R.shape[-1]), A.reshape(A.shape[0], -1, A.shape[-1])
        nr = np.linalg.norm(R, axis=-1)
        cos = (R * A).sum(-1) / np.maximum(nr * np.linalg.norm(A, axis=-1), 1e-12)
        ratio = nr / np.maximum(np.median(nr, axis=1, keepdims=True), 1e-12)
        worst = np.argsort(cos, axis=None)[:n_worst]
        out[part] = dict(lech=float(np.abs(A - R).max() / max(np.abs(R).max(), 1e-12)), cos_min=float(cos.min()),
                         phan_vi=[float(v) for v in np.quantile(cos, [1e-4, 1e-3, 1e-2, 0.5])], duoi_0999=int((cos < 0.999).sum()), tong=int(cos.size),
                         theo_tang=[float(v) for v in cos.min(axis=1)],
                         te_nhat=[[int(i // cos.shape[1]) - cos.shape[0], float(cos.flat[i]), float(ratio.flat[i])] for i in worst])
    return out


def stats_text(name, st):
    return '\n'.join(f"{name} [{p}] lech {v['lech']:.2e} | cos_min {v['cos_min']:.6f} | cos q 1e-4 / 1e-3 / 1e-2 / 0,5: "
                     + ' / '.join(f'{x:.6f}' for x in v['phan_vi']) + f" | duoi 0,999: {v['duoi_0999']}/{v['tong']} | theo tang: "
                     + ' '.join(f'{x:.6f}' for x in v['theo_tang']) for p, v in st.items())
