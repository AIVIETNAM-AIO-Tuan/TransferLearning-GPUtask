# -*- coding: utf-8 -*-
"""Gom kết quả các lượt của module 89 từ Drive cho notebook tổng hợp (qua khối đọc chuẩn kiem_ke / tai — gói không gọi Drive, không chứa ID Drive).

    Ver1   stage4_nb89/r89v1_<encoder>[_k64]   mỗi loại đầu: du_doan_<nhãn>_<đầu>_<t>.npz (preds, target) + luot_<nhãn>_<đầu>_<t>.json
           CheXFound Linear-4 + GLoRI: dùng lại `86f` (stage4_nb86f/r86f_chexfound_moi_truong_ho, đầu lin / glori) — cổng PLAN_89 §4.3
    Ver2   stage4_nb89/r89v2_<encoder>_<kho>   luot_<nhãn>_<t>.json + du_doan_<nhãn>_<t>.npz (test_<nhánh>_s<k>, val_…, y_test)

Lượt chưa có trên Drive -> ghi vào danh sách thiếu (không lỗi): notebook tổng hợp chạy được khi mới có một phần kết quả. Mỗi thư mục lấy bộ tệp có dấu
thời gian MỚI NHẤT mà đủ cặp JSON + dự đoán. Không in mã ảnh; chỉ số tổng hợp."""
import json
import re
from dataclasses import dataclass

import numpy as np

from ..encoders.spec import SPECS
from ..protocols import geometry as GEO

FOLDER = 'stage4_nb89'
HEADS = ('lin1', 'lin4', 'mldec', 'glori')
TITLES = {'lin1': 'Linear-1', 'lin4': 'Linear-4', 'mldec': 'ML-Decoder', 'glori': 'GLoRI'}
V1_TAGS = {'biomedclip': 'r89v1_biomedclip'}                 # lượt gói 0.2.0 (không chia khúc) — Tuấn 13/11; hàng khác: r89v1_<encoder>_k64[hậu tố]
V1_GRID = (1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3)        # lưới 9 lr của bài (Ver1)
REUSE_86F = dict(encoder='chexfound', folder=('stage4_nb86f', 'r86f_chexfound_moi_truong_ho'), heads={'lin4': 'lin', 'glori': 'glori'})
V2_STORES = ('kcb', 'cb', GEO.CROP_STORE)
STORE_TEXT = {'kcb': 'dem vuong, KHONG can bang histogram', 'cb': 'dem vuong, CO can bang histogram',
              GEO.CROP_STORE: 'co canh ngan + CAT GIUA, KHONG can bang histogram'}
STORE_ROWS = {GEO.CROP_STORE: GEO.ROWS}       # kho chỉ chạy cho một số hàng (phép thử hình học); kho khác: mọi hàng
T = r'(\d{8}_\d{6})'


@dataclass
class PredSet:
    '''Một bộ dự đoán test [N, 28] của một đầu đã học xong.'''
    ver: str                 # 'v1' | 'v2'
    store: str               # 'kcb' | 'cb'
    encoder: str
    head: str                # lin1 | lin4 | mldec | glori
    arm: str                 # 'luat' (lr theo AUROC val) | 'ap' (lr theo AP val, chỉ khi khác luật — Ver2)
    seed: int
    lr: float
    edge: bool               # lr ở biên lưới của phiên bản đó
    preds: np.ndarray
    target: np.ndarray
    source: str              # RUN_TAG nguồn (không ID)
    val_auroc: float = None  # AUROC val lúc chọn lr (Ver1: số mã họ tính, dropout bật; Ver2: eval())

    @property
    def cell(self):
        return (self.ver, self.store, self.encoder, self.head, self.arm)


def v1_tag(encoder):
    '''= notebook.ver1.run_tag (kể cả spec.run_suffix: PubMedCLIP chuẩn hoá ImageNet -> r89v1_pubmedclip_k64_imnet; lượt cũ không đọc).'''
    return V1_TAGS.get(encoder, f"r89v1_{encoder}_k64{_suffix(encoder)}")


def _suffix(encoder):
    return getattr(SPECS.get(encoder), 'run_suffix', '')


def v2_tag(encoder, store):
    '''= notebook.ver2.run_tag (kể cả spec.run_suffix).'''
    return f'r89v2_{encoder}_{store}{_suffix(encoder)}'


def _read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _pairs(kk, re_json, re_npz):
    '''kk = {tên: [bản mới nhất trước]} -> {khoá (các nhóm trước dấu thời gian): (t, bản json, bản npz)} với t mới nhất có ĐỦ cặp.'''
    found = {}
    for name, versions in kk.items():
        for kind, rx in (('json', re_json), ('npz', re_npz)):
            m = re.fullmatch(rx, name)
            if m:
                found.setdefault((m.groups()[:-1], m.groups()[-1]), {})[kind] = versions[0]
    out = {}
    for (key, t), v in found.items():
        if len(v) == 2 and (key not in out or t > out[key][0]):
            out[key] = (t, v['json'], v['npz'])
    return out


def load_v1(drive, encoder):
    '''-> ([PredSet], thông tin nhãn (nhan, lop_chinh) | None). Thư mục / tệp chưa có -> [].'''
    tag = v1_tag(encoder)
    lab = re.escape(tag[1:])
    heads = '|'.join(HEADS)
    kk = drive['kiem_ke']((FOLDER, tag), mau=rf'(du_doan|luot)_{lab}_({heads})_\d{{8}}_\d{{6}}\.(npz|json)|bang_89v1_{re.escape(encoder)}_\d{{8}}_\d{{6}}\.json')[1]
    sets = []
    for (kind,), (t, bj, bn) in sorted(_pairs(kk, rf'luot_{lab}_({heads})_{T}\.json', rf'du_doan_{lab}_({heads})_{T}\.npz').items()):
        js = _read_json(drive['tai'](bj))
        with np.load(drive['tai'](bn)) as z:
            preds, target = z['preds'].astype(np.float32), z['target'].astype(np.int8)
        lr = float(js['chon']['lr'])
        sets.append(PredSet('v1', 'kcb', encoder, kind, 'luat', 0, lr, lr in (V1_GRID[0], V1_GRID[-1]), preds, target, tag, float(js['chon']['auroc_val'])))
    names = None
    bang = sorted(n for n in kk if n.startswith('bang_'))
    if bang:
        vt = _read_json(drive['tai'](kk[bang[-1]][0])).get('van_tay') or {}
        if vt.get('nhan') and vt.get('lop_chinh'):
            names = (list(vt['nhan']), list(vt['lop_chinh']))
    return sets, names


def load_86f(drive):
    '''CheXFound Linear-4 + GLoRI của `86f` (mã + môi trường của họ, cùng điều kiện Ver1). -> ([PredSet], {đầu: lưới lr từng lượt}).'''
    kk = drive['kiem_ke'](REUSE_86F['folder'], mau=r'(du_doan|luot)_86[a-z0-9_]*_(lin|glori)_\d{8}_\d{6}\.(npz|json)')[1]
    back = {v: k for k, v in REUSE_86F['heads'].items()}
    sets, grids = [], {}
    pairs = _pairs(kk, rf'luot_(86[a-z0-9_]*)_(lin|glori)_{T}\.json', rf'du_doan_(86[a-z0-9_]*)_(lin|glori)_{T}\.npz')
    for (_, kind), (t, bj, bn) in sorted(pairs.items()):
        js = _read_json(drive['tai'](bj))
        with np.load(drive['tai'](bn)) as z:
            preds, target = z['preds'].astype(np.float32), z['target'].astype(np.int8)
        lr = float(js['chon']['lr'])
        head = back[kind]
        sets.append(PredSet('v1', 'kcb', REUSE_86F['encoder'], head, 'luat', 0, lr, lr in (V1_GRID[0], V1_GRID[-1]), preds, target,
                            REUSE_86F['folder'][1], float(js['chon']['auroc_val'])))
        grids[head] = [list(x.get('lrs') or []) for x in js.get('luot', [])]
    return sets, grids


def load_v2(drive, encoder, store):
    '''-> ([PredSet], (nhan, lop_chinh) | None, đường dò lr {đầu: [(lr, AUROC val cuối, AP val cuối)]}, bản chọn lr). Chưa có -> ([], None, {}, {}).'''
    tag = v2_tag(encoder, store)
    lab = re.escape(tag[1:])
    kk = drive['kiem_ke']((FOLDER, tag), mau=rf'(luot|du_doan)_{lab}_\d{{8}}_\d{{6}}\.(json|npz)')[1]
    pairs = _pairs(kk, rf'luot_({lab})_{T}\.json', rf'du_doan_({lab})_{T}\.npz')
    if not pairs:
        return [], None, {}, {}
    t, bj, bn = next(iter(pairs.values()))
    js = _read_json(drive['tai'](bj))
    with np.load(drive['tai'](bn)) as z:
        arr = {k: z[k] for k in z.files}
    target = arr['y_test'].astype(np.int8)
    sets = []
    for k in js['loai_dau']:
        ch = js['chon'][k]
        for crit, arm, name in (('auroc', 'luat', k), ('ap', 'ap', f'{k}_ap')):
            if crit == 'ap' and ch['ap_trung_luat']:
                continue
            seeds = sorted(int(n.rsplit('_s', 1)[1]) for n in arr if re.fullmatch(rf'test_{name}_s\d+', n))
            for s in seeds:
                sets.append(PredSet('v2', store, encoder, k, arm, s, float(ch[crit]['lr']), bool(ch[crit]['bien_luoi']), arr[f'test_{name}_s{s}'].astype(np.float32),
                                    target, tag, float(ch[crit]['auc'])))
    curves = {}
    for name, c in js['do_lr'].items():
        _, k, lr = name.split('|')
        curves.setdefault(k, []).append((float(lr), float(c['auc'][-1]), float(c['ap'][-1])))
    return sets, (list(js['nhan']), list(js['lop_chinh'])), {k: sorted(v) for k, v in curves.items()}, js['chon']


def collect(drive, encoders, stores=V2_STORES, log=print):
    '''Gom mọi lượt đang có. -> dict(sets, thieu [(mô tả)], nhan, lop_chinh, duong_lr {(encoder, kho): …}, chon_v2, kiem_86f, co {mô tả lượt: số bộ}).'''
    sets, missing, names, have, curves, chon = [], [], None, {}, {}, {}
    for e in encoders:
        s, nm = load_v1(drive, e)
        names = names or nm
        need = [h for h in HEADS if not (e == REUSE_86F['encoder'] and h in REUSE_86F['heads'])]
        got = {x.head for x in s}
        have[f'Ver1 {e} ({v1_tag(e)})'] = len(s)
        if set(need) - got:
            missing.append(f"Ver1 {e}: thieu dau {sorted(set(need) - got)} trong {FOLDER}/{v1_tag(e)}")
        sets += [x for x in s if x.head in need]
    kiem = None
    if REUSE_86F['encoder'] in encoders:
        s, grids = load_86f(drive)
        have[f"Ver1 chexfound Linear-4 + GLoRI ({'/'.join(REUSE_86F['folder'])})"] = len(s)
        ref = next((x for x in sets if x.ver == 'v1' and x.encoder == REUSE_86F['encoder']), None)
        same = bool(s) and ref is not None and all(np.array_equal(x.target, ref.target) for x in s)
        flat = {h: sorted(v for g in grids.get(h, []) for v in g) for h in grids}
        kiem = dict(co=sorted(x.head for x in s), nhan_test_trung_89=same, luoi_du_9lr={h: flat[h] == sorted(V1_GRID) for h in flat},
                    glori_hai_nua=[len(g) for g in grids.get('glori', [])])
        if len(s) < len(REUSE_86F['heads']):
            missing.append(f"Ver1 chexfound: thieu Linear-4 / GLoRI cua 86f trong {'/'.join(REUSE_86F['folder'])}")
        if s and not same:
            missing.append('Ver1 chexfound: nhan test cua 86f KHAC luot 89 (cong PLAN_89 4.3 muc 2 TRUOT) -> KHONG dung lai, can chay Linear-4 + GLoRI o 89')
        else:
            sets += s
    for e in encoders:
        for st in stores:
            if e not in STORE_ROWS.get(st, (e,)):
                continue                                                 # kho cắt giữa: chỉ 8 hàng của phép thử hình học (protocols/geometry.py)
            s, nm, cv, ch = load_v2(drive, e, st)
            names = names or nm
            have[f'Ver2 {e} kho {st} ({v2_tag(e, st)})'] = len(s)
            if not s:
                missing.append(f'Ver2 {e} kho {st}: chua co ket qua trong {FOLDER}/{v2_tag(e, st)}')
            sets += s
            if cv:
                curves[(e, st)], chon[(e, st)] = cv, ch
    assert sets, 'khong co luot nao tren Drive -> chay notebook 89v1_* / 89v2_* truoc. DUNG'
    assert names, 'khong doc duoc ten nhan / lop so chinh tu ket qua -> DUNG'
    for x in sets:
        assert x.preds.shape == x.target.shape and x.preds.shape[1] == len(names[0]) and np.isfinite(x.preds).all(), f'{x.cell}: du doan hong -> DUNG'
    return dict(sets=sets, thieu=missing, nhan=names[0], lop_chinh=names[1], duong_lr=curves, chon_v2=chon, kiem_86f=kiem, co=have)
