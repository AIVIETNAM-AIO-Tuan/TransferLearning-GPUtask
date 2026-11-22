# -*- coding: utf-8 -*-
"""Ba hàng SEED 43 cho phép kiểm ít nhãn VinDr (docs/PLAN_90_SO_SANH_GIAI_PHAU.md §13.8; Tuấn 22/11 "dựa vào plan đó dựng luôn"): (3) B.4 v2,
(1b) một lớp luồng vùng sau encoder chỉ đọc, (0) không luồng vùng — lượt 78x seed 43, 6 epoch, trọng số EMA epoch 4 (`rs_*_ep4.pt`), đúng các tệp mà
`78r_dac_trung_vt_*_s43` đã đọc và qua cổng seed 43 (22/11).

Lớp bọc = vindr_bench.encoders.vt ở commit ghim (c81af62) NGUYÊN VẸN — cùng mã đã trích ba hàng s42 của 92a; chỉ khác TỆP. Hàng s43 được đăng ký TẠM vào
vindr_bench (spec.SPECS + vt.ROWS; mọi trường chép từ hàng s42 cùng vị trí, đổi tên hàng / lượt / md5) trong lúc dựng rồi gỡ ra (registered()), nên sổ
đăng ký của vindr_bench sau khi dựng như cũ. Cổng thêm (= cổng seed 43 ở C2b của 78r s43): run của tệp = lượt s43 VÀ ghép cặp thứ tự lô = J v3b s43
(cfg_d4['ghep_cap'] của backbone_ft.pt ở hàng J; cfg_kl['ghep_cap'] của rs_ ở hàng (0)).

md5 = output C0 của ba notebook 78r_dac_trung_vt_*_s43 đã chạy 22/11 (tests/test_s43.py đối chiếu). Đây là md5 checkpoint của ta (không phải mã ảnh) —
tools/publish.py chỉ cho chuỗi 32 hex ở tệp này khi nó thuộc PINS. Không chứa ID Drive."""
import contextlib
import dataclasses

GHEP_CAP = ('stage4_nb78x', 'r78x_J_v3b_s43')          # dấu vân thứ tự lô của cả ba lượt s43 (Event 22/11 "Seed 43 huấn luyện xong …")
S43 = {
    'vt_b4v2_s43': dict(base='vt_b4v2', run='r78x_J_v3b_dp_ema_v2_s43', file_md5='eb874061aadd78fcad895d4847871c1c',
                        cfg_md5='32e81b05f235d2e74c590e8ad04edb2d'),
    'vt_sau1_s43': dict(base='vt_sau1', run='r78x_J_v3b_dp_ema_vt_sau1_s43', file_md5='ab1d7cf6e9bea5a8ab36d0f09e335c45',
                        cfg_md5='f75cfa4fe85114fccb02e337e64d10b6'),
    'vt_khong_luong_s43': dict(base='vt_khong_luong', run='r78x_B1_dp_ema_s43', file_md5='40f6acb1d58863fe15dc83752bde8898', cfg_md5=None),
}
ROWS = tuple(S43)
PINS = frozenset(m for r in S43.values() for m in (r['file_md5'], r['cfg_md5']) if m)
assert len(ROWS) == 3 and len(PINS) == 5 and all(r['run'].endswith('_s43') for r in S43.values())


def spec(name, file_md5=None, cfg_md5=None):
    '''EncoderSpec của hàng s43 = spec hàng s42 cùng vị trí (vindr_bench) đổi tên hàng / lượt / md5. file_md5 / cfg_md5 khác None: CHỈ bộ kiểm (tệp giả).'''
    from vindr_bench.encoders import spec as SP
    r = S43[name]
    s = SP.SPECS[r['base']]
    old = s.drive_dir.rsplit('/', 1)[1]
    assert old.endswith('_s42') and old[:-4] == r['run'][:-4], f'{name}: luot s42 {old} khong cung goc voi {r["run"]} -> DUNG'
    assert bool(s.extra_files) == bool(r['cfg_md5']), f'{name}: tep phu cua hang s42 KHAC hang s43 -> DUNG'
    return dataclasses.replace(
        s, name=name, title=f'{s.title} — seed 43', weights=s.weights.replace(old, r['run']), drive_dir=s.drive_dir[:-len(old)] + r['run'],
        file_md5=file_md5 or r['file_md5'], extra_files=tuple((n, cfg_md5 or r['cfg_md5'], z) for n, _, z in s.extra_files),
        source_checked=s.source_checked + '; seed 43 (22/11): cung lop boc, tep s43 = tep 78r s43 da doc (md5 = output C0), cong seed 43 cua frozen_probe',
        notes=s.notes + ' | hang SEED 43 (PLAN_90 13.8)')


@contextlib.contextmanager
def registered(name, file_md5=None, cfg_md5=None):
    '''Đăng ký TẠM hàng s43 vào vindr_bench (spec.SPECS + vt.ROWS) trong khối with; ra khỏi khối (kể cả khi lỗi) -> gỡ, sổ đăng ký như cũ. -> spec.'''
    from vindr_bench.encoders import spec as SP
    from vindr_bench.encoders import vt as VT
    assert name in S43, f'{name}: khong phai hang s43 {ROWS} -> DUNG'
    assert name not in SP.SPECS and name not in VT.ROWS, f'{name}: da co trong so dang ky vindr_bench -> DUNG'
    assert VT.SPECS is SP.SPECS, 'vindr_bench.encoders.vt dung bang spec khac spec.SPECS -> DUNG'
    SP.SPECS[name] = spec(name, file_md5, cfg_md5)
    VT.ROWS[name] = dict(VT.ROWS[S43[name]['base']], run=S43[name]['run'])
    try:
        yield SP.SPECS[name]
    finally:
        SP.SPECS.pop(name, None)
        VT.ROWS.pop(name, None)


def seed_gate(name, sp, path=None, cfg_path=None):
    '''Cổng seed 43 (= C2b của 78r s43) trên tệp thật: hàng J đọc backbone_ft.pt (run + cfg_d4['ghep_cap']); hàng (0) đọc rs_ (run + cfg_kl['ghep_cap']).
       md5 = ghim (safe_load). -> dict(dat, run, ghep_cap, nguon) — số đo trong tiến trình, không gán hằng.'''
    from vindr_bench.checks import safe_load
    from vindr_bench.encoders import files
    from vindr_bench.encoders import vt as VT
    r = S43[name]
    if r['cfg_md5'] is None:
        path = path or files.weights_path(sp)
        ck, _ = safe_load.load_checkpoint(path, expect_md5=sp.file_md5, expect_size=sp.file_size)
        run, gc, src = ck.get('run'), (ck.get('cfg_kl') or {}).get('ghep_cap'), f"{sp.drive_file}: run + cfg_kl['ghep_cap']"
        del ck
    else:
        cfg_path = cfg_path or files.weights_path(sp, VT.CFG_FILE)
        extra = {n: (m, z) for n, m, z in sp.extra_files}
        cfj, _ = safe_load.load_checkpoint(cfg_path, expect_md5=extra[VT.CFG_FILE][0], expect_size=extra[VT.CFG_FILE][1], allow=('torch_version',))
        run, gc, src = cfj.get('run'), (cfj.get('cfg_d4') or {}).get('ghep_cap'), f"{VT.CFG_FILE}: run + cfg_d4['ghep_cap']"
        del cfj
    gc = list(gc) if isinstance(gc, (list, tuple)) else gc
    return dict(dat=bool(run == r['run'] and gc == list(GHEP_CAP)), run=run, ghep_cap=gc, nguon=src, can=dict(run=r['run'], ghep_cap=list(GHEP_CAP)))


def build(name, path=None, cfg_path=None, file_md5=None, cfg_md5=None, **kw):
    '''-> lớp bọc vindr_bench (B4v2 / RadDinoFT) ở eval(), CPU, của hàng s43; nap['dat'] = cổng của vindr_bench VÀ cổng seed 43 (nap['cong_seed43']).
       kw -> vindr_bench.encoders.vt.build (input_mean / input_std / attn). Cổng seed 43 trượt -> dừng TRƯỚC khi dựng mô hình.'''
    from vindr_bench.encoders import vt as VT
    with registered(name, file_md5, cfg_md5) as sp:
        gate = seed_gate(name, sp, path, cfg_path)
        assert gate['dat'], (f"{name}: cong seed 43 TRUOT ({gate['nguon']}: run {gate['run']} / ghep_cap {gate['ghep_cap']} KHAC {gate['can']['run']} / "
                             f"{gate['can']['ghep_cap']}) -> DUNG")
        enc = VT.build(name, path=path, cfg_path=cfg_path, **kw)
    assert enc.nap is not None and enc.spec.name == name
    enc.nap.update(encoder=name, cong_seed43=gate, hang_s42=S43[name]['base'])
    enc.nap['dat'] = bool(enc.nap['dat'] and gate['dat'])
    return enc
