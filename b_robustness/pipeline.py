"""Điều phối từng notebook 87 (PLAN_KIEM_THU §15–§16). Hàm trả kết quả + {tên tệp: nội dung}; notebook ghi bằng khối Drive chuẩn qua save().

87a  run_source_audit      K0 — kiểm nguồn backbone 5 seed
87b  run_p2_statistics     K5 — thống kê lại P2 đã có (đối chiếu JSON 85g)
87c  SeedTransfer          K3 — P1 + trích token + học lại đầu cho một backbone seed (sau backbone_b.run_part3)
87d  run_seed_table        K3 — bảng P1 / P2 qua backbone seed (đối chiếu JSON 85d / 85g)"""
import datetime

import numpy as np

from . import audit, backbone_b, config, heads, predictions, seed_table, sensitivity, transfer, vindr
from .drive import DriveReader


def _ts():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def _json_keys(d):
    '''Khoá tuple -> chuỗi "a | b | c" (JSON).'''
    return {(' | '.join(map(str, k)) if isinstance(k, tuple) else k): v for k, v in d.items()}


def save(ghi, ghi_json, files, log=print):
    '''Ghi {tên: nội dung} bằng hàm của khối Drive chuẩn (local; lên Drive ở C9): .npy = mảng, .json = dict.'''
    for name, obj in files.items():
        if name.endswith('.npy'): ghi(name, lambda q, v=obj: np.save(q, v))
        elif name.endswith('.json'): ghi(name, lambda q, o=obj: ghi_json(q, o))
        else: raise ValueError(f'kieu tep khong ho tro: {name}')
    log(f"da ghi {len(files)} tep (local; len Drive o C9): {', '.join(sorted(files))[:300]}")


def check_synced(dong_bo, ten_tren_drive, expected, label, log=print):
    '''= C9 của 85f: dong_bo() rồi kiểm từng tệp mong đợi đã lên Drive -> (đủ?, thiếu).'''
    du = dong_bo()
    tren = ten_tren_drive()
    thieu = [t for t in expected if t not in tren]
    log(f"{label}: dong_bo {'DU' if du else 'CHUA DU'} | {len(expected) - len(thieu)} / {len(expected)} tep tren Drive"
        + (f' | THIEU {thieu} -> KHONG tat runtime; chay lai o nay, roi bao Claude' if thieu or not du else ' -> gui Claude output cac o'))
    return bool(du) and not thieu, thieu


def _load_table_json(reader, folder, pattern, hint):
    inv = reader.inventory(folder, pattern)
    assert inv, f"thieu {hint} trong {'/'.join(folder)} -> chay {'85g' if 'p2' in hint else '85d_bang_p1_B'} truoc. DUNG"
    name = sorted(inv)[-1]
    return reader.load_json(folder, inv, name), name


def package_commit():
    import b_robustness
    return getattr(b_robustness, 'GOI_SHA', None)


# ===================================================================================== 87a
def run_source_audit(kiem_ke, tai, xung_dot, log=print):
    reader = DriveReader(kiem_ke, tai, xung_dot, log)
    inv = audit.inventory(reader, log=log)
    res = audit.inspect(reader, inv, log=log)
    concl = audit.conclude(res, log=log)
    ts = _ts()
    files = {f'kiem_nguon_{ts}.json': dict(
        t=ts, nguon=config.K0_SOURCES, ket_qua=res, ket_luan=concl, thieu={h: v['thieu'] for h, v in inv.items()},
        nhieu_ban={h: v['nhieu_ban'] for h, v in inv.items()}, so_tep={h: v['so_tep'] for h, v in inv.items()},
        goi=dict(b_robustness=package_commit(), nhanh_c=config.NHANH_C_COMMIT, nhanh_c_s45_s46=config.NHANH_C_COMMIT_S45_S46, buoc=list(config.PART3_STEPS)),
        so_quyet_dinh_nb84={config.ROW[s]: dict(gia_tri=config.DECISION_SCORE[s], tep=config.RUN_NOTEBOOK_84[s], commit=config.RUN_COMMIT_84[s]) for s in config.K3_SEEDS},
        ghi_chu='87a (PLAN_KIEM_THU §15.3): K0 kiem nguon; du P2 = rs_ ep5 + module dung; du P1 = + w_ / p_ du + nguon_backbone.json khop.')}
    return dict(result=res, conclusion=concl, files=files)


# ===================================================================================== 87b
def run_p2_statistics(kiem_ke, tai, xung_dot, n_boot=config.N_BOOT, log=print):
    reader = DriveReader(kiem_ke, tai, xung_dot, log)
    test = vindr.load_test_labels(reader)
    rows = predictions.rows_anchor_and_sota()
    pred, js = predictions.load_p2(reader, test, rows)
    log(f'VinDr test: {len(test.keys):,} anh | {len(test.main_classes)} lop so chinh | 7 lop bai RAD-DINO | {len(test.any_classes)} lop co duong | hang {list(rows)}')
    js85g, name85g = _load_table_json(reader, config.TABLE_85G, r'bang_p2_\d+_\d+\.json', 'bang_p2_*.json')
    r = sensitivity.run_k5(test, pred, js85g, name85g, n_boot=n_boot, log=log)
    ts = _ts()
    files = {f'k5_thong_ke_{ts}.json': dict(
        t=ts, hang=r['rows'], moc=r['anchor'], sota=r['sota'], lop_chinh=r['main'], n_anh=len(test.keys), hat_nua=config.HALF_SEED, hat_boot=config.BOOT_SEED,
        n_boot=n_boot, boot_bi_loai=r['boot_skipped'], don=_json_keys(r['single']), ensemble=_json_keys(r['ensemble']), ci_ensemble=_json_keys(r['ci_ensemble']),
        ci_cap=_json_keys(r['ci_pair']), duong=r['positives'], duong_nua=r['positives_half'], hon_kem=r['win_loss'], bo_tung_lop=_json_keys(r['leave_one_out']),
        bo_rib=_json_keys(r['drop_rib']), json_85g=name85g, nguon=reader.sources, goi=package_commit(),
        ghi_chu='87b (PLAN_KIEM_THU §15.4): K5 mo ta / do nhay sau khi da thay ket qua; luat P2 (85g) KHONG doi.')}
    return dict(result=r, files=files)


# ===================================================================================== 87d
def run_seed_table(kiem_ke, tai, xung_dot, n_boot=config.N_BOOT, log=print):
    reader = DriveReader(kiem_ke, tai, xung_dot, log)
    test = vindr.load_test_labels(reader)
    b_rows, sota_rows = predictions.rows_k3()
    pred2, js2 = predictions.load_p2(reader, test, {**b_rows, **sota_rows}, check_run=True)
    pred1, js1, lab40 = predictions.load_p1(reader, test, b_rows)
    _, classes1 = predictions.p1_classes(test, lab40)
    log(f"VinDr test: {len(test.keys):,} anh | P2 {len(test.main_classes)} lop | P1 {len(classes1)} lop | backbone seed B {list(b_rows)} | SOTA {list(sota_rows)}")
    log('backbone seed MOI (K3): ' + ', '.join(h for h, c in b_rows.items() if c['new']) + ' | B_s42 = seed phat trien (test da doc o 85d / 85g) -> bao RIENG')
    js85d, name85d = _load_table_json(reader, config.TABLE_85D, r'bang_p1_\d+_\d+\.json', 'bang_p1_*.json')
    p1 = seed_table.p1_table(test, pred1, lab40, b_rows, js85d, name85d, n_boot=n_boot, log=log)
    js85g, name85g = _load_table_json(reader, config.TABLE_85G, r'bang_p2_\d+_\d+\.json', 'bang_p2_*.json')
    p2 = seed_table.p2_table(test, pred2, b_rows, sota_rows, js85g, name85g, n_boot=n_boot, log=log)
    ts = _ts()
    files = {f'k3_bang_{ts}.json': dict(
        t=ts, hang_b=list(b_rows), seed_moi=p2['new'], sota=list(sota_rows), lop_p1=p1['classes'], lop_p2=test.main_classes, n_anh=len(test.keys),
        hat_nua=config.HALF_SEED, hat_boot=config.BOOT_SEED, n_boot=n_boot, boot_bi_loai=dict(p1=p1['boot_skipped'], p2=p2['boot_skipped']),
        p1={f'{h} | {d}': dict(r, ci=p1['ci'][(h, d)]) for (h, d), r in p1['result'].items()},
        p2={f'{b} | {d}': dict(r, ci=p2['ci_ens'][(b, d)], don=p2['single'][(b, d)]) for (b, d, _), r in p2['result'].items()},
        ci_cap={f'{a[0]}|{a[1]} - {c[0]}|{c[1]} | {m}': v for (a, c, m), v in p2['ci_pair'].items()},
        phan_xu={b: {x: dict(chenh=dl, ci=ci, ket_luan=v) for x, (dl, ci, v) in px.items()} for b, px in p2['verdict'].items()},
        qua_seed_p1=p1['across_seeds'], qua_seed_p2=_json_keys(p2['across_seeds']), json_85g=name85g, json_85d=name85d, nguon=reader.sources, goi=package_commit(),
        ghi_chu='87d (PLAN_KIEM_THU §15.6): bang mo ta qua backbone seed; s42 = seed phat trien (bao rieng); luat P2 KHONG doi.')}
    return dict(p1=p1, p2=p2, files=files)


# ===================================================================================== 87c
class SeedTransfer:
    '''Một backbone seed B mới sau backbone_b.run_part3(globals(), NC, seed): P1 (C6V) -> dữ liệu P2 (D) -> token (T) -> học đầu (H) -> ghi (G + M).
       Hàm Drive = hàm của khối ghi / đọc chuẩn mà bước bd3_C0 đã định nghĩa trong notebook (truyền vào tường minh).'''

    def __init__(self, ns, nhanh_c, seed, kiem_ke, tai, xung_dot, ghi, ghi_json, log=print):
        assert seed in config.K3_SEEDS, seed
        self.view = backbone_b.Part3View(ns)
        self.nhanh_c, self.seed, self.row, self.run_tag = nhanh_c, seed, config.ROW[seed], config.RUN_K3[seed]
        self.decision = config.DECISION_SCORE[seed]
        self.kiem_ke, self.tai, self.xung_dot, self.ghi, self.ghi_json, self.log = kiem_ke, tai, xung_dot, ghi, ghi_json, log
        assert ns.get('RUN_TAG') == self.run_tag and tuple(ns.get('DRV_DUONG', ())) == (config.FOLDER_K3, self.run_tag), (
            f"RUN_TAG / DRV_DUONG cua Phan 3 KHAC {config.FOLDER_K3}/{self.run_tag} -> chay o Phan 3 bang backbone_b.run_part3. DUNG")
        assert int(ns['SEED']) == seed and ns['CAU_HINH_C']['run_x'] == f'r84x_{self.row}', 'namespace mang seed KHAC -> DUNG'
        self.p1 = self.data = self.tokens = self.result = None
        self.written = []

    def _reader(self):
        return DriveReader(self.kiem_ke, self.tai, self.xung_dot, self.log)

    def run_p1(self):
        self.p1 = transfer.predict_p1(self.view, self._reader(), self.row, self.decision, log=self.log)
        files = transfer.p1_files(self.view, self.p1, self.row, self.decision, self.seed, self.run_tag)
        save(self.ghi, self.ghi_json, files, self.log); self.written += list(files)
        self.log(f'KHONG doc nhan VinDr (bang 87d cham).')

    def load_p2_data(self):
        self.data = vindr.load_p2_data(self._reader(), image_dir=self.view.ns['LOCAL'], log=self.log)

    def extract_tokens(self):
        assert self.p1 is not None and self.data is not None, 'chay o P1 va o du lieu P2 truoc -> DUNG'
        self.tokens = transfer.extract_tokens(self.view, self.data, self.p1, self.row, self.decision, log=self.log)

    def train_heads(self, **kw):
        '''kw -> heads.train_p2_heads (CHỈ để kiểm offline với cấu hình nhỏ; notebook gọi không tham số = lưới khoá).'''
        assert self.tokens is not None, 'chay o trich token truoc -> DUNG'
        t = self.tokens
        self.result = heads.train_p2_heads(t.tok, t.vung, t.glob, self.data.labels, self.data.split, self.data.label_names, self.data.main_classes,
                                           self.view.head_ops(), self.row, log=self.log, **kw)

    def write_p2(self):
        assert self.result is not None, 'chay o hoc dau truoc -> DUNG'
        files = transfer.p2_files(self.data, self.tokens, self.result, self.row, self.run_tag)
        m = transfer.manifest(self.view, getattr(self.nhanh_c, '_NHANH_C_SHA_DA_NAP', None), package_commit(), self.seed, self.row, self.run_tag,
                              self.tokens, self.result, transfer.sha256_steps(self.nhanh_c), config.RUN_NOTEBOOK_84[self.seed])
        files[f'manifest_{self.row}.json'] = m
        save(self.ghi, self.ghi_json, files, self.log); self.written += list(files)
        self.log(f"87c manifest {self.row}: nhanh_c {str(m['goi']['nhanh_c']['commit'])[:8]} (luot goc {m['goi']['nhanh_c']['commit_luot_goc'][:8]}) | "
                 f"b_robustness {str(m['goi']['b_robustness'])[:8]} | so quyet dinh tinh lai {m['so_quyet_dinh_tinh_lai']:.4f} (nb 84: {self.decision}) | EP_BB {m['ep_bb']}")

    def expected_files(self):
        heads_p2 = list(self.result['heads']) if self.result else list(config.B_HEADS)
        return ([f'p2_{self.row}_{d}_{t}.npy' for d in heads_p2 for t in ('val', 'test')] + [f'p_vd_{d}_s{self.seed}.npy' for d in self.view.ns['DAU_S']]
                + [f'manifest_{self.row}.json'])

    def sync(self, dong_bo, ten_tren_drive):
        return check_synced(dong_bo, ten_tren_drive, self.expected_files(), f'87c C9 ({self.row})', self.log)
