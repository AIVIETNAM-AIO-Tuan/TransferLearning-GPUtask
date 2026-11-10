"""Nạp dự đoán test VinDr của các hàng (85f mốc B s42 + SOTA; 87c backbone seed mới) QUA API, kèm cổng thứ tự ảnh / nhãn / lớp số chính /
cấu hình đầu / tập val. Một hàng = dict(folder=(thư mục, lượt), bb=tên trong tệp, heads=..., p1_seed=seed tệp P1 hoặc None, notebook=...)."""
import re

import numpy as np

from . import config


def rows_anchor_and_sota():
    '''Hàng của bảng 85g: mốc B s42 + RAD-DINO / CheXFound / CheXWorld (85f, lưới khoá).'''
    rows = {config.ANCHOR: dict(folder=(config.FOLDER_85F, config.RUN_85F[config.ANCHOR]), bb=config.ANCHOR, heads=config.B_HEADS, p1_seed=None,
                                notebook=config.NOTEBOOK_85F[config.ANCHOR])}
    rows.update({x: dict(folder=(config.FOLDER_85F, config.RUN_85F[x]), bb=x, heads=config.SOTA_HEADS, p1_seed=None, notebook=config.NOTEBOOK_85F[x])
                 for x in config.SOTA})
    return rows


def rows_k3():
    '''Hàng của bảng K3: B s42 (85f) + B s44 _lai / s45 / s46 (87c) — có P1; SOTA (85f) — chỉ P2.'''
    b = {config.ANCHOR_ROW: dict(folder=(config.FOLDER_85F, config.RUN_85F[config.ANCHOR]), bb=config.ANCHOR, heads=config.B_HEADS, p1_seed=42,
                                 new=False, notebook=config.NOTEBOOK_85F[config.ANCHOR])}
    b.update({config.ROW[s]: dict(folder=(config.FOLDER_K3, config.RUN_K3[s]), bb=config.ROW[s], heads=config.B_HEADS, p1_seed=s, new=True,
                                  notebook=config.NOTEBOOK_K3[s][:-len('.ipynb')]) for s in config.K3_SEEDS})
    sota = {x: dict(folder=(config.FOLDER_85F, config.RUN_85F[x]), bb=x, heads=config.SOTA_HEADS, p1_seed=None, new=False,
                    notebook=config.NOTEBOOK_85F[x]) for x in config.SOTA}
    return b, sota


def load_p2(reader, test, rows, check_run=False):
    '''Dự đoán P2 test [5 lần học đầu, ảnh, 28] float64 của mọi hàng x đầu + JSON của 85f / 87c.
       Cổng: tên hàng (+ RUN_TAG nếu check_run), md5 thứ tự ảnh test = 85b, nhãn, 21 lớp số chính, đủ đầu, hình / hữu hạn;
       cấu hình học đầu + tập val GIỐNG NHAU ở mọi hàng (cùng điều kiện). -> (pred {(hàng, đầu): mảng}, js {hàng: json})'''
    pred, js = {}, {}
    for h, c in rows.items():
        du, bb = tuple(c['folder']), c['bb']
        kc = reader.inventory(du, r'(p2_.+_test\.npy|p2_.+_\d+_\d+\.json)')
        cands = sorted(t for t in kc if re.fullmatch(rf'p2_{re.escape(bb)}_\d+_\d+\.json', t))
        assert cands, f"thieu p2_{bb}_*.json trong {'/'.join(du)} -> chay {c['notebook']} truoc. DUNG"
        j = reader.load_json(du, kc, cands[-1])
        assert (j['bb'] == bb and (not check_run or j['run'] == du[1]) and j['md5_khoa_test'] == test.meta['md5_khoa'] and j['nhan'] == test.label_names
                and j['lop_chinh'] == test.main_classes), f'{h}: ten / RUN_TAG / thu tu anh / nhan / lop so chinh KHAC -> DUNG'
        assert set(c['heads']) <= set(j['dau']), f"{h}: thieu dau {c['heads']} -> DUNG"
        js[h] = j
        for d in c['heads']:
            v = np.load(reader.fetch(du, kc, f'p2_{bb}_{d}_test.npy')).astype('float64')
            assert v.shape == (5, len(test.keys), len(test.label_names)) and np.isfinite(v).all(), f'{h} {d}: du doan P2 sai hinh / NaN -> DUNG'
            pred[(h, d)] = v
    names = list(rows)
    assert all(js[b]['cau_hinh'] == js[names[0]]['cau_hinh'] for b in names) and len({js[b]['md5_khoa_val'] for b in names}) == 1, (
        'cau hinh hoc dau / tap val KHAC giua cac hang -> DUNG')
    return pred, js


def load_p1(reader, test, rows, heads=config.P1_HEADS):
    '''Dự đoán P1 (đầu MIMIC 40 lớp, KHÔNG học lại) [ảnh, 40] float64 của các hàng có p1_seed + JSON du_doan_vd_*.
       Cổng: tên hàng, md5 thứ tự ảnh = 85b, LAB40 giống nhau ở mọi hàng, hình / hữu hạn. -> (pred {(hàng, đầu): mảng}, js, lab40)'''
    pred, js = {}, {}
    for h, c in rows.items():
        if c.get('p1_seed') is None: continue
        du = tuple(c['folder'])
        kc = reader.inventory(du, r'(p_vd_[a-z_]+_s\d+\.npy|du_doan_vd_\d+_\d+\.json)')
        cands = sorted(t for t in kc if re.fullmatch(r'du_doan_vd_\d+_\d+\.json', t))
        assert cands, f"thieu du_doan_vd_*.json (P1) trong {'/'.join(du)} -> DUNG"
        j = reader.load_json(du, kc, cands[-1])
        assert j['bb'] == c['bb'] and j['md5_khoa_vd'] == test.meta['md5_khoa'], f'{h}: P1 ten / thu tu anh KHAC -> DUNG'
        js[h] = j
        for d in heads:
            v = np.load(reader.fetch(du, kc, f"p_vd_{d}_s{c['p1_seed']}.npy")).astype('float64')
            assert v.shape == (len(test.keys), len(j['lab40'])) and np.isfinite(v).all(), f'{h} {d}: du doan P1 sai hinh / NaN -> DUNG'
            pred[(h, d)] = v
    names = list(js)
    lab40 = list(js[names[0]]['lab40'])
    assert all(js[b]['lab40'] == lab40 for b in names), 'LAB40 KHAC giua cac backbone seed -> DUNG'
    return pred, js, lab40


def p1_scores(P, lab40, mapping, classes):
    '''Điểm từng nhãn VinDr = MAX xác suất các cột MIMIC ánh xạ (ánh xạ chốt ở 85b; = _diem_d của 85d).'''
    return np.stack([P[:, [lab40.index(c) for c in mapping[t]]].max(1) for t in classes], 1)


def p1_classes(test, lab40):
    '''Ánh xạ VinDr -> cột MIMIC + 12 lớp số chính P1 (nhóm "chat", >= ng_chinh ca dương; = luật 85b).'''
    mapping = {a['vindr']: list(a['mimic']) for a in test.meta['anh_xa']}
    classes = list(test.meta['lop_chinh'])
    assert set(c for t in classes for c in mapping[t]) <= set(lab40) and classes == [
        a['vindr'] for a in test.meta['anh_xa'] if a['nhom'] == 'chat' and test.meta['so_duong'][a['vindr']] >= test.meta['ng_chinh']], (
        'lop so chinh P1 KHAC luat 85b -> DUNG')
    return mapping, classes
