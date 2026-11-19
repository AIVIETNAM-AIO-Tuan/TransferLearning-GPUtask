# -*- coding: utf-8 -*-
"""Ô của notebook `92a_trich_dac_trung_vindr` (G4; PLAN_92 §11.4). Mỗi hàm = một ô; trạng thái đi theo dict CTX; mọi việc Drive qua hàm của khối
Drive chuẩn ở C0. Ghi Drive MỘT lần lúc cuối (C6); mất runtime giữa chừng -> Run all lại (~30–45 phút, chưa đo).

    C2 setup     thư viện notebook + CỔNG bộ thư viện / GPU của 89 Ver2 (cùng môi trường -> cùng đặc trưng)
    C3 data      kho VinDr `kcb` + vân tay khoá / nhãn = các lượt 89
    C4 extract   8 hàng: tệp trọng số (md5 ghim) -> lớp bọc + cổng nạp + số học bf16 vs fp32 -> trích 18.000 ảnh (cổng bộ nhớ ở lô đầu) -> tệp đặc trưng;
                 chạy lại ô: chỉ làm hàng chưa xong trong phiên
    C5 manifest  tệp nhãn / tách (không mã ảnh) + manifest (md5 từng tệp, cổng từng hàng, môi trường)
    C6 sync      dong_bo() + kiểm tệp trên Drive"""
import gc
import json
import time

from .. import __version__
from .. import config as C
from .. import features as FT

DRIVE_NAMES = ('kiem_ke', 'tai', 'xung_dot', 'ghi', 'ghi_json', 'dong_bo', 'ten_tren_drive', 'GHI_LOC', 'RUN_TAG', '_duong', '_liet_ke')
IMAGE_DIR = '/content/vd518_kcb'
MIN_FREE_GB = 15
GATE_IMAGES = 16


def _write_bytes(path, b):
    with open(path, 'wb') as f:
        f.write(b)


def drive_api(ns):
    missing = [k for k in DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    return {k: ns[k] for k in DRIVE_NAMES}


def _packages():
    import vindr_bench
    import frozen_probe
    return dict(frozen_probe=dict(phien_ban=__version__, sha=getattr(frozen_probe, 'FP_SHA', None)),
                vindr_bench=dict(phien_ban=vindr_bench.__version__, sha=getattr(vindr_bench, 'GOI_SHA', None)))


def setup(ns, log=print):
    '''C2. -> CTX. Bộ thư viện / GPU khác bộ đã chốt của 89 Ver2 -> dừng.'''
    from vindr_bench.protocols.chexfound_env import environment as E
    from vindr_bench.protocols.ours import gates as G
    from vindr_bench.protocols.ours import runtime as RT
    drive = drive_api(ns)
    assert drive['RUN_TAG'] == C.RUN_TAGS['trich'], f"RUN_TAG {drive['RUN_TAG']!r} != {C.RUN_TAGS['trich']!r} -> DUNG"
    mt = E.install_notebook_packages(min_free_gb=MIN_FREE_GB)
    v = G.versions()
    lib_ok, lib_diff = G.library_gate(v)
    gpu = G.gpu_info()
    device, amp = RT.device_and_amp()
    pk = _packages()
    log(f"92a moi truong: Python {v['python']} | torch {v['torch']} (CUDA {v['cuda']}) | transformers {v['transformers']} | timm {v['timm']} | GPU {gpu['ten']} "
        f"{gpu['tong_gib'] or 0:.0f} GiB | backbone autocast {str(amp).replace('torch.', '')} | frozen_probe {pk['frozen_probe']['phien_ban']} @ "
        f"{(pk['frozen_probe']['sha'] or '?')[:12]} + vindr_bench {pk['vindr_bench']['phien_ban']} @ {(pk['vindr_bench']['sha'] or '?')[:12]}")
    assert lib_ok, f'bo thu vien KHAC bo da chot cua 89 Ver2 {G.VER2_LIBS}: {lib_diff} (co / ghim) -> DUNG, gui Claude dong tren'
    assert device == 'cuda' and G.gpu_gate(gpu), f"GPU {gpu['ten']!r} khong phai G4 ({G.GPU_MARK}) -> Runtime > Change runtime type > G4"
    log(f"hang ({len(C.ROWS)}): " + ' | '.join(C.TEXT[r] for r in C.ROWS))
    log(f"dac trung: {' | '.join(f'{k} = {C.FEATURE_TEXT[k]}' for k in C.FEATURES)} | float16 | lo {C.EXTRACT_BATCH}")
    return dict(drive=drive, device=device, amp=amp, goi=pk, moi_truong=dict(thu_vien=v, gpu=gpu, pip=mt), xong={}, t=time.strftime('%Y%m%d_%H%M%S'))


def data(ctx, log=print):
    '''C3. Kho VinDr `kcb` + vân tay (phải trùng các lượt 89). -> VinDrData.'''
    from vindr_bench.data import vindr as V
    from vindr_bench.protocols.ours import gates as G
    d = ctx['drive']
    vd = V.load_vindr(d['kiem_ke'], d['tai'], d['xung_dot'], IMAGE_DIR, store=C.STORE, log=log)
    fp = vd.fingerprint()
    same, diff = G.data_gate(fp)
    log(vd.summary())
    log('van tay md5 khoa / nhan: ' + ' | '.join(f"{s} {fp[s]['md5_khoa'][:8]} / {fp[s]['md5_nhan'][:8]}" for s in V.SPLITS)
        + (' | TRUNG cac luot 89' if same else f' | KHAC cac luot 89 (co / ghim): {diff}'))
    assert same, f'van tay du lieu KHAC cac luot 89 {diff} -> DUNG, gui Claude dong tren'
    ctx.update(vindr=vd, van_tay=fp)
    return vd


def _one(ctx, row, log):
    import torch
    from vindr_bench.checks import weights as WT
    from vindr_bench.data import loaders as L
    from vindr_bench.encoders import ABLATION_ROWS, SPECS
    from vindr_bench.encoders import files as W
    from vindr_bench.notebook import ver2 as V2
    from vindr_bench.protocols.ours import backbone as BB
    from vindr_bench.protocols.ours import gates as G
    from vindr_bench.protocols.ours import runtime as RT
    d, vd, spec = ctx['drive'], ctx['vindr'], SPECS[row]
    t0 = time.time()
    tep = W.fetch(spec, d, log=log) if spec.drive_dir else None
    for t in ([tep] + list(tep.get('phu') or [])) if tep else []:
        log(f"  tep tu Drive: {spec.drive_dir}/{t['tep']} | md5 {t['md5']} = ghim | {t['byte']:,} B | {t['so_ban']} ban tren Drive")
    with V2.quiet_hub():
        bb = BB.load(row, ctx['device'], ctx['amp'])
    nap = bb.enc.nap
    log(f"  nap trong so {'DAT' if nap['dat'] else 'TRUOT'} | {WT.loading_text(nap)}")
    if row in ABLATION_ROWS:
        log('  ' + V2.ablation_text(nap))
    assert nap['dat'], f'{row}: nap trong so KHONG dat -> DUNG'
    _, tf_eval = L.build_transforms(bb.spec)
    ds = L.VinDrImages(vd.keys, vd.image_dir, tf_eval)
    x = torch.stack([ds[int(i)][0] for i in vd.indices('val')[:GATE_IMAGES]])
    g = G.numeric_check(bb, x)
    log('  ' + G.numeric_text(g))
    assert g['dat'], f'{row}: dac trung bf16 HONG NANG so voi fp32 -> DUNG'
    total = torch.cuda.get_device_properties(0).total_memory
    feats = FT.extract(bb, vd, batch_size=C.EXTRACT_BATCH, num_workers=RT.default_num_workers(), log=log,
                       first_batch_gate=lambda peak: G.memory_ok(peak, total))
    b = FT.feature_bytes(feats, row, spec.name)
    name = FT.feature_file(row)
    d['ghi'](name, lambda q, b=b: _write_bytes(q, b))
    rec = dict(hang=row, ma=C.CODE[row], dieu_kien=C.TEXT[row], tep=name, md5=FT.md5_bytes(b), byte=len(b), so_anh=feats['so_anh'], chieu=int(bb.dim),
               phut_trich=round(feats['phut'], 2), phut_tong=round((time.time() - t0) / 60, 2), gpu_dinh_gib=round(feats['gpu_dinh_gib'], 2),
               gpu_tong_gib=round(total / 2 ** 30, 1), nap=WT.loading_text(nap), so_hoc=dict(dat=g['dat'], patch=g['patch'], vector=g['vector']),
               tep_trong_so=tep, anh=int(spec.image_size), token_patch=int(spec.n_patches))
    log(f"  {C.CODE[row]}: {feats['so_anh']:,} anh x {bb.dim} chieu x 2 dac trung (float16) | trich {feats['phut']:.1f} phut, ca hang {rec['phut_tong']:.1f} phut | "
        f"GPU dinh {feats['gpu_dinh_gib']:.1f} / {total / 2 ** 30:.0f} GiB | tep {name} {len(b) / 2 ** 20:.1f} MB, md5 {rec['md5'][:12]} (ghi local; len Drive o C6)")
    del bb, feats, b, x, ds
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def extract(ctx, log=print):
    '''C4. 8 hàng, lần lượt; hàng đã xong trong phiên bỏ qua (chạy lại ô sau khi sửa lỗi một hàng).'''
    t0 = time.time()
    for k, row in enumerate(C.ROWS, 1):
        if row in ctx['xong']:
            log(f'[{k}/{len(C.ROWS)}] {C.TEXT[row]}: da xong trong phien -> bo qua')
            continue
        log(f'[{k}/{len(C.ROWS)}] {C.TEXT[row]}')
        ctx['xong'][row] = _one(ctx, row, log)
    log(f"xong {len(ctx['xong'])} / {len(C.ROWS)} hang | {(time.time() - t0) / 60:.1f} phut (o nay)")
    return ctx['xong']


def manifest(ctx, log=print):
    '''C5. Tệp nhãn / tách + manifest. -> dict manifest.'''
    assert set(ctx['xong']) == set(C.ROWS), f"chua du hang: thieu {[r for r in C.ROWS if r not in ctx['xong']]} -> chay lai C4"
    d, vd = ctx['drive'], ctx['vindr']
    b = FT.label_bytes(vd, ctx['van_tay'])
    d['ghi'](FT.LABEL_FILE, lambda q: _write_bytes(q, b))
    man = dict(run=C.RUN_TAGS['trich'], t=ctx['t'], goi=ctx['goi'], kho=C.STORE, van_tay=ctx['van_tay'], so_anh=len(vd.keys), thu_tu_hang=list(C.ROWS),
               dac_trung=list(C.FEATURES), tep_nhan=dict(tep=FT.LABEL_FILE, md5=FT.md5_bytes(b), byte=len(b)), hang=ctx['xong'],
               moi_truong=dict(thu_vien=ctx['moi_truong']['thu_vien'], gpu=ctx['moi_truong']['gpu']),
               ghi_chu='dac trung dong cung 92a (PLAN_92 11): mean_patch = TB token patch tang cuoi sau LayerNorm (= G cua 78r), glob = vector toan anh '
                       'cua lop boc; float16; thu tu hang = thu tu anh cua VinDrData (van tay); KHONG co ma anh')
    man = json.loads(json.dumps(man, default=str))
    d['ghi'](FT.MANIFEST_FILE, lambda q: d['ghi_json'](q, man))
    log(f"manifest: {len(ctx['xong'])} tep dac trung + tep nhan ({len(b) / 2 ** 10:.0f} KB, md5 {man['tep_nhan']['md5'][:12]}) | tong "
        f"{sum(r['byte'] for r in ctx['xong'].values()) / 2 ** 20:.0f} MB")
    ctx['manifest'] = man
    return man


def sync(ctx, log=print):
    '''C6. dong_bo() + kiểm tệp trên Drive.'''
    d = ctx['drive']
    du = d['dong_bo']()
    on = d['ten_tren_drive']()
    need = [FT.feature_file(r) for r in C.ROWS] + [FT.LABEL_FILE, FT.MANIFEST_FILE]
    co = set(need) <= set(on)
    log(f"C6: dong_bo {'DU' if du else 'CHUA DU'} | {len(need)} tep cua luot tren Drive: {'CO du' if co else 'THIEU ' + str(sorted(set(need) - set(on)))}"
        + ('' if du and co else ' | KHONG tat runtime; chay lai C6') + (' -> chay 92b (CPU)' if du and co else ''))
    return dict(dong_bo=du, du_tep=co)
