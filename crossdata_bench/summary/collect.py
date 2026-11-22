# -*- coding: utf-8 -*-
"""Gom các lượt `93v_<bộ>_<encoder>` từ Drive cho notebook tổng hợp `93t` (qua khối đọc chuẩn kiem_ke / tai — gói không gọi Drive, không chứa ID).

    stage4_nb93/r93v_<bộ>_<encoder>   luot_93v_<bộ>_<encoder>_<t>.json + du_doan_93v_<bộ>_<encoder>_<t>.npz (+ bang_…_<t>.json của ô C7)
    mảng .npz (= vindr_bench.protocols.ours.pipeline.assemble + ô C6 của 93v): test_<nhánh>_s<k>, val_<nhánh>_s<k>, y_test, y_val, idx_test, idx_val,
    nhom_test, nhom_val; <nhánh> = loại đầu (lr của LUẬT) hoặc <loại>_ap (lr theo AP val, chỉ khi khác).

Mỗi thư mục lấy bộ tệp có dấu thời gian MỚI NHẤT có đủ cặp JSON + dự đoán. Lượt chưa có -> danh sách thiếu (không lỗi). Lượt có mà trượt cổng -> DỪNG.
Cổng từng lượt: bộ / encoder / Ver2 / kho / lưới 13 lr / 5 hạt / epoch / số ảnh / md5 khoá test = ghim / md5 nhãn test + val = ghim / số dương test /
đủ mảng 5 hạt mỗi nhánh, đúng hình, hữu hạn / số bước + dự đoán hữu hạn (cổng của lượt) / vindr_bench = commit ghim / điểm từng hạt = bảng C7 của lượt.
Cổng giữa các lượt của một bộ: nhãn, thứ tự ảnh, mã cụm của test (và val) TRÙNG từng phần tử -> ghép cặp theo cụm được. Không in mã ảnh / mã người."""
import hashlib
import json
import re
from dataclasses import dataclass, field

import numpy as np

from .. import config
from ..datasets import base
from ..evaluation import single as S

T = r'(\d{8}_\d{6})'
N_SEEDS = 5


@dataclass
class PredSet:
    '''Dự đoán của MỘT hạt của một nhánh (encoder, đầu, nhánh) trên một bộ.'''
    dataset: str
    encoder: str
    head: str
    arm: str                  # 'luat' (lr theo AUROC val) | 'ap' (lr theo AP val, chỉ khi khác luật)
    seed: int
    lr: float
    edge: bool                # lr ở biên lưới 13 lr
    test: np.ndarray          # [N_test, C] xác suất
    val: np.ndarray           # [N_val, C] xác suất cuối pha 1
    source: str               # RUN_TAG nguồn (không ID)
    val_auroc: float = None   # AUROC val lần chấm cuối pha 1 của hạt 0 (số chọn lr)


@dataclass
class Bundle:
    '''Mọi lượt đã gom của một bộ.'''
    name: str
    runs: dict = field(default_factory=dict)        # encoder -> dict(t, tag, js, bang (| None), cu [dấu thời gian cũ hơn], xung_dot)
    sets: list = field(default_factory=list)
    y_test: np.ndarray = None
    y_val: np.ndarray = None
    g_test: np.ndarray = None                       # mã cụm từng ảnh test (RSNA: người NIH; Shenzhen: ảnh)
    g_val: np.ndarray = None
    missing: list = field(default_factory=list)
    gates: dict = field(default_factory=dict)       # encoder -> {cổng: bool}


def md5_strings(v):
    return hashlib.md5('\n'.join(np.asarray(v).astype(str).tolist()).encode('utf-8')).hexdigest()


def find_run(drive, name, encoder):
    '''-> dict(t, tag, luot, du_doan, bang, cu, xung_dot) (bản Drive, chưa tải) | None.'''
    tag = config.dataset(name).run_tag(encoder)
    lab = re.escape(tag[1:])
    rx = rf'(luot|du_doan|bang)_{lab}_{T}\.(json|npz)'
    kk = drive['kiem_ke']((config.FOLDER, tag), mau=rx)[1]
    found = {}
    for n, versions in kk.items():
        m = re.fullmatch(rx, n)
        if m and versions:
            found.setdefault(m.group(2), {})[m.group(1)] = versions
    full = sorted(t for t, v in found.items() if 'luot' in v and 'du_doan' in v)
    if not full:
        return None
    t = full[-1]
    v = found[t]
    return dict(t=t, tag=tag, luot=v['luot'][0], du_doan=v['du_doan'][0], bang=(v['bang'][0] if 'bang' in v else None), cu=full[:-1],
                xung_dot=sorted(k for k, vers in v.items() if drive['xung_dot'](vers)))


def _read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _arms(js):
    '''-> [(loại, nhánh 'luat' | 'ap', nhãn mảng, tiêu chí trong js['chon'])] theo thứ tự loai_dau; nhánh theo AP chỉ khi khác lr của luật.'''
    out = []
    for k in js['loai_dau']:
        out.append((k, 'luat', k, 'auroc'))
        if not js['chon'][k]['ap_trung_luat']:
            out.append((k, 'ap', f'{k}_ap', 'ap'))
    return out


def check_run(name, encoder, js, arr, bang):
    '''Cổng một lượt. -> {tên cổng: bool} (+ khoá '_info' mô tả ngắn).'''
    from vindr_bench.protocols.ours import config as C
    sp = config.dataset(name)
    fp = config.FINGERPRINT[name]
    g = {}
    g['bo + encoder + Ver2 + kho kcb'] = (js.get('bo') == name and (js.get('encoder') or {}).get('name') == encoder and js.get('phien_ban_ver') == 'Ver2'
                                         and js.get('kho_anh') == config.STORE_KEY)
    g['luoi 13 lr + 5 hat + epoch'] = (list(map(float, js.get('luoi', []))) == list(C.LR_GRID) and js.get('so_seed') == N_SEEDS
                                      and js.get('epoch') == dict(pha1=sp.epochs, pha2=sp.epochs))
    g['so anh train / val / test'] = (js.get('n_train'), js.get('n_val'), js.get('n_test')) == tuple(sp.sizes)
    g['md5 khoa test = ghim'] = js.get('md5_khoa_test') == fp['test'][0]
    g['md5 nhan test + val = ghim'] = base.md5_labels(arr['y_test']) == fp['test'][1] and base.md5_labels(arr['y_val']) == fp['val'][1]
    g['so duong test'] = int(np.asarray(arr['y_test']).sum()) == sp.positives[2]
    g['cong cua luot (so buoc, du doan huu han)'] = bool((js.get('kiem') or {}).get('so_buoc_dung')) and bool((js.get('kiem') or {}).get('du_doan_huu_han'))
    g['vindr_bench = commit ghim'] = ((js.get('goi') or {}).get('vindr_bench') or {}).get('sha') == config.PACKAGES['vindr_bench']['sha']
    n_t, n_v, c = sp.sizes[2], sp.sizes[1], len(sp.label_names)
    want = {f'{part}_{lab}_s{s}': n for _, _, lab, _ in _arms(js) for s in range(N_SEEDS) for part, n in (('test', n_t), ('val', n_v))}
    have = {n for n in arr if re.fullmatch(r'(test|val)_.+_s\d+', n)}
    shapes = all(arr[n].shape == (m, c) and bool(np.isfinite(arr[n]).all()) for n, m in want.items() if n in arr)
    g['du mang 5 hat moi nhanh, dung hinh, huu han'] = set(want) == have and shapes
    g['nhom test / val co du'] = arr.get('nhom_test') is not None and arr['nhom_test'].shape == (n_t,) and arr.get('nhom_val') is not None
    if bang is not None and g['du mang 5 hat moi nhanh, dung hinh, huu han']:   # điểm từng hạt tính lại = số ô C7 của chính lượt (cùng S.point) -> nhánh / hạt đúng
        y = arr['y_test']
        same = True
        for k, arm, lab, crit in _arms(js):
            per = [S.point(y, arr[f'test_{lab}_s{s}']) for s in range(N_SEEDS)]
            ref = ((bang.get('diem') or {}).get(lab) or {}).get('tung_hat') or {}
            same = same and all(len(ref.get(m, [])) == N_SEEDS and max(abs(a[m] - b) for a, b in zip(per, ref[m])) <= 1e-9 for m in ('ap', 'auprc_thang', 'auroc'))
        g['diem tung hat = bang C7 cua luot'] = same
    else:
        g['diem tung hat = bang C7 cua luot'] = False
    return g


def collect(drive, names, encoders=config.TABLE_ORDER, log=print):
    '''-> {bộ: Bundle}. Lượt thiếu -> Bundle.missing; lượt trượt cổng -> AssertionError (in đủ cổng trước khi dừng).'''
    out, bad = {}, []
    for name in names:
        sp = config.dataset(name)
        bd = Bundle(name)
        ref = None
        for e in encoders:
            f = find_run(drive, name, e)
            if f is None:
                bd.missing.append(f'{name} {e}: chua co ket qua trong {config.FOLDER}/{sp.run_tag(e)}')
                continue
            js = _read_json(drive['tai'](f['luot']))
            with np.load(drive['tai'](f['du_doan']), allow_pickle=False) as z:
                arr = {k: z[k] for k in z.files}
            bang = _read_json(drive['tai'](f['bang'])) if f['bang'] else None
            gates = check_run(name, e, js, arr, bang)
            if f['xung_dot']:
                gates['khong xung dot ban trung ten'] = False
            # giữa các lượt của bộ: nhãn / thứ tự ảnh / mã cụm test + val trùng lượt đầu tiên
            key = dict(y_test=base.md5_labels(arr['y_test']), y_val=base.md5_labels(arr['y_val']), idx_test=hashlib.md5(np.asarray(arr['idx_test'], dtype=np.int64)
                       .tobytes()).hexdigest(), g_test=md5_strings(arr['nhom_test']), g_val=md5_strings(arr['nhom_val']))
            if ref is None:
                ref = key
                bd.y_test, bd.y_val = arr['y_test'].astype(np.int8), arr['y_val'].astype(np.int8)
                bd.g_test, bd.g_val = arr['nhom_test'].astype(str), arr['nhom_val'].astype(str)
            gates['nhan / thu tu anh / ma cum = luot dau cua bo'] = key == ref
            bd.gates[e] = gates
            bd.runs[e] = dict(t=f['t'], tag=f['tag'], js=js, bang=bang, cu=f['cu'], xung_dot=f['xung_dot'], goi_93=((js.get('goi') or {}).get('crossdata_bench') or {}).get('sha'))
            failed = [k for k, v in gates.items() if not v]
            if failed:
                bad.append(f'{name} {e}: {failed}')
                continue
            ch = js['chon']
            for k, arm, lab, crit in _arms(js):
                for s in range(N_SEEDS):
                    bd.sets.append(PredSet(name, e, k, arm, s, float(ch[k][crit]['lr']), bool(ch[k][crit]['bien_luoi']), arr[f'test_{lab}_s{s}'].astype(np.float32),
                                           arr[f'val_{lab}_s{s}'].astype(np.float32), f['tag'], float(ch[k][crit]['auc'])))
        if bd.g_test is not None:
            n_cl = len(set(bd.g_test.tolist()))
            ok_cl = n_cl == config.TEST_CLUSTERS[name]
            log(f"  {name}: so cum test {n_cl:,} (ghim {config.TEST_CLUSTERS[name]:,}) -> {'DAT' if ok_cl else 'TRUOT'}")
            if not ok_cl:
                bad.append(f'{name}: so cum test {n_cl} != {config.TEST_CLUSTERS[name]}')
        out[name] = bd
    for name, bd in out.items():
        for e, gates in bd.gates.items():
            r = bd.runs[e]
            log(f"  {name:<9}{config.ROW_TITLES[e]:<12} {r['tag']} @ {r['t']} | cong {sum(gates.values())}/{len(gates)}"
                + ('' if all(gates.values()) else f" TRUOT {[k for k, v in gates.items() if not v]}") + (f" | ban cu hon bo qua: {len(r['cu'])}" if r['cu'] else ''))
    assert not bad, 'luot TRUOT cong -> DUNG (gui Claude output C3): ' + ' | '.join(bad)
    return out


def curves(js):
    '''Đường dò lr của một lượt: {đầu: [(lr, AUROC val cuối pha 1, AP val cuối pha 1)]} (hạt 0; = cách đọc của vindr_bench.evaluation.collect.load_v2).'''
    out = {}
    for name, c in js['do_lr'].items():
        _, k, lr = name.split('|')
        out.setdefault(k, []).append((float(lr), float(c['auc'][-1]), float(c['ap'][-1])))
    return {k: sorted(v) for k, v in out.items()}
