# -*- coding: utf-8 -*-
"""Ô của notebook `92f_trich_dac_trung_vindr_s43` (G4; PLAN_90 §13.8) — cùng khuôn 92a (extract.py: Drive, kho, vân tay, cổng thư viện / GPU / bộ nhớ /
số học bf16 vs fp32, vòng trích, tệp đặc trưng KHÔNG đổi), chỉ khác: ba hàng seed 43 qua lớp bọc encoders/vt_s43 (= lớp bọc s42 của vindr_bench, tệp s43)
+ cổng seed 43. Ghi Drive MỘT lần (C6).

    C2 setup     thư viện notebook + CỔNG bộ thư viện / GPU của 89 Ver2 (cùng môi trường với 92a -> cùng đường đặc trưng)
    C3 data      kho VinDr `kcb` + vân tay = các lượt 89 (= extract.data)
    C4 extract   3 hàng: tệp 78x s43 (md5 ghim) -> cổng seed 43 -> lớp bọc vindr_bench + cổng nạp / cổng chỉ đọc -> số học bf16 vs fp32 -> trích 18.000 ảnh
                 (cổng bộ nhớ ở lô đầu) -> tệp đặc trưng; chạy lại ô: chỉ làm hàng chưa xong
    C5 manifest  tệp nhãn / tách + manifest (md5 từng tệp, cổng từng hàng, môi trường)
    C6 sync      dong_bo() + kiểm tệp trên Drive"""
import gc
import json
import time

from .. import config as C
from .. import config_s43 as S
from .. import features as FT
from . import extract as X


def setup(ns, log=print):
    '''C2. -> CTX. Bộ thư viện / GPU khác bộ đã chốt của 89 Ver2 -> dừng.'''
    from vindr_bench.protocols.chexfound_env import environment as E
    from vindr_bench.protocols.ours import gates as G
    from vindr_bench.protocols.ours import runtime as RT
    drive = X.drive_api(ns)
    assert drive['RUN_TAG'] == S.RUN_TAGS['trich'], f"RUN_TAG {drive['RUN_TAG']!r} != {S.RUN_TAGS['trich']!r} -> DUNG"
    mt = E.install_notebook_packages(min_free_gb=X.MIN_FREE_GB)
    v = G.versions()
    lib_ok, lib_diff = G.library_gate(v)
    gpu = G.gpu_info()
    device, amp = RT.device_and_amp()
    pk = X._packages()
    log(f"92f moi truong: Python {v['python']} | torch {v['torch']} (CUDA {v['cuda']}) | transformers {v['transformers']} | timm {v['timm']} | GPU {gpu['ten']} "
        f"{gpu['tong_gib'] or 0:.0f} GiB | backbone autocast {str(amp).replace('torch.', '')} | frozen_probe {pk['frozen_probe']['phien_ban']} @ "
        f"{(pk['frozen_probe']['sha'] or '?')[:12]} + vindr_bench {pk['vindr_bench']['phien_ban']} @ {(pk['vindr_bench']['sha'] or '?')[:12]}")
    assert lib_ok, f'bo thu vien KHAC bo da chot cua 89 Ver2 {G.VER2_LIBS}: {lib_diff} (co / ghim) -> DUNG, gui Claude dong tren'
    assert device == 'cuda' and G.gpu_gate(gpu), f"GPU {gpu['ten']!r} khong phai G4 ({G.GPU_MARK}) -> Runtime > Change runtime type > G4"
    log(f"hang ({len(S.ROWS)}): " + ' | '.join(S.TEXT[r] for r in S.ROWS))
    log(f"dac trung: {' | '.join(f'{k} = {C.FEATURE_TEXT[k]}' for k in S.FEATURES)} | float16 | lo {C.EXTRACT_BATCH}")
    return dict(drive=drive, device=device, amp=amp, goi=pk, moi_truong=dict(thu_vien=v, gpu=gpu, pip=mt), xong={}, t=time.strftime('%Y%m%d_%H%M%S'))


def data(ctx, log=print):
    '''C3. = 92a C3 (kho `kcb`, vân tay = các lượt 89).'''
    return X.data(ctx, log)


def _one(ctx, row, log):
    import torch
    from vindr_bench.checks import weights as WT
    from vindr_bench.data import loaders as L
    from vindr_bench.encoders import files as W
    from vindr_bench.notebook import ver2 as V2
    from vindr_bench.protocols.ours import backbone as BBK
    from vindr_bench.protocols.ours import gates as G
    from vindr_bench.protocols.ours import runtime as RT
    from ..encoders import vt_s43 as VS
    d, vd, spec = ctx['drive'], ctx['vindr'], VS.spec(row)
    t0 = time.time()
    tep = W.fetch(spec, d, log=log)
    for t in [tep] + list(tep.get('phu') or []):
        log(f"  tep tu Drive: {spec.drive_dir}/{t['tep']} | md5 {t['md5']} = ghim | {t['byte']:,} B | {t['so_ban']} ban tren Drive")
    with V2.quiet_hub():
        enc = VS.build(row)
    nap = enc.nap
    sg = nap['cong_seed43']
    log(f"  cong seed 43 ({sg['nguon']}): run {sg['run']} | ghep cap thu tu lo {'/'.join(sg['ghep_cap'] or [])} -> {'DAT' if sg['dat'] else 'TRUOT'}")
    log(f"  nap trong so {'DAT' if nap['dat'] else 'TRUOT'} | {WT.loading_text(nap)}")
    log('  ' + V2.ablation_text(nap) + f" | lop boc vindr_bench cua hang s42 {nap['hang_s42']} (cung ma), tep s43")
    assert nap['dat'], f'{row}: nap trong so / cong seed 43 KHONG dat -> DUNG'
    enc = enc.to(ctx['device']).eval()
    bb = BBK.Backbone(enc, ctx['device'], ctx['amp'])
    _, tf_eval = L.build_transforms(bb.spec)
    ds = L.VinDrImages(vd.keys, vd.image_dir, tf_eval)
    x = torch.stack([ds[int(i)][0] for i in vd.indices('val')[:X.GATE_IMAGES]])
    g = G.numeric_check(bb, x)
    log('  ' + G.numeric_text(g))
    assert g['dat'], f'{row}: dac trung bf16 HONG NANG so voi fp32 -> DUNG'
    total = torch.cuda.get_device_properties(0).total_memory
    feats = FT.extract(bb, vd, batch_size=C.EXTRACT_BATCH, num_workers=RT.default_num_workers(), log=log,
                       first_batch_gate=lambda peak: G.memory_ok(peak, total))
    b = FT.feature_bytes(feats, row, spec.name)
    name = FT.feature_file(row)
    d['ghi'](name, lambda q, b=b: X._write_bytes(q, b))
    rec = dict(hang=row, ma=S.CODE[row], dieu_kien=S.TEXT[row], hang_s42=nap['hang_s42'], tep=name, md5=FT.md5_bytes(b), byte=len(b), so_anh=feats['so_anh'],
               chieu=int(bb.dim), phut_trich=round(feats['phut'], 2), phut_tong=round((time.time() - t0) / 60, 2), gpu_dinh_gib=round(feats['gpu_dinh_gib'], 2),
               gpu_tong_gib=round(total / 2 ** 30, 1), nap=WT.loading_text(nap), cong_seed43=sg, lech_rad_dino_goc=nap.get('lech_rad_dino_goc'),
               ema=nap.get('ema'), luong=nap.get('luong'), cong_chi_doc=nap.get('cong_chi_doc'),
               so_hoc=dict(dat=g['dat'], patch=g['patch'], vector=g['vector']), tep_trong_so=tep, anh=int(spec.image_size), token_patch=int(spec.n_patches))
    log(f"  {S.CODE[row]}: {feats['so_anh']:,} anh x {bb.dim} chieu x 2 dac trung (float16) | trich {feats['phut']:.1f} phut, ca hang {rec['phut_tong']:.1f} phut | "
        f"GPU dinh {feats['gpu_dinh_gib']:.1f} / {total / 2 ** 30:.0f} GiB | tep {name} {len(b) / 2 ** 20:.1f} MB, md5 {rec['md5'][:12]} (ghi local; len Drive o C6)")
    del bb, enc, feats, b, x, ds
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def extract(ctx, log=print):
    '''C4. 3 hàng, lần lượt; hàng đã xong trong phiên bỏ qua (chạy lại ô sau khi sửa lỗi một hàng).'''
    t0 = time.time()
    for k, row in enumerate(S.ROWS, 1):
        if row in ctx['xong']:
            log(f'[{k}/{len(S.ROWS)}] {S.TEXT[row]}: da xong trong phien -> bo qua')
            continue
        log(f'[{k}/{len(S.ROWS)}] {S.TEXT[row]}')
        ctx['xong'][row] = _one(ctx, row, log)
        log(f"  -> xong {len(ctx['xong'])} / {len(S.ROWS)} hang | {(time.time() - t0) / 60:.1f} phut tu dau o")
    log(f"xong {len(ctx['xong'])} / {len(S.ROWS)} hang | {(time.time() - t0) / 60:.1f} phut (o nay)")
    return ctx['xong']


def manifest(ctx, log=print):
    '''C5. Tệp nhãn / tách + manifest. -> dict manifest.'''
    assert set(ctx['xong']) == set(S.ROWS), f"chua du hang: thieu {[r for r in S.ROWS if r not in ctx['xong']]} -> chay lai C4"
    d, vd = ctx['drive'], ctx['vindr']
    b = FT.label_bytes(vd, ctx['van_tay'])
    d['ghi'](FT.LABEL_FILE, lambda q: X._write_bytes(q, b))
    man = dict(run=S.RUN_TAGS['trich'], t=ctx['t'], goi=ctx['goi'], kho=C.STORE, van_tay=ctx['van_tay'], so_anh=len(vd.keys), thu_tu_hang=list(S.ROWS),
               hang_s42=dict(S.BASE), dac_trung=list(S.FEATURES), tep_nhan=dict(tep=FT.LABEL_FILE, md5=FT.md5_bytes(b), byte=len(b)), hang=ctx['xong'],
               moi_truong=dict(thu_vien=ctx['moi_truong']['thu_vien'], gpu=ctx['moi_truong']['gpu']),
               ghi_chu='dac trung dong cung 92f (PLAN_90 13.8): 3 hang seed 43 (3) / (1b) / (0), lop boc vindr_bench cua hang s42 cung vi tri (tep s43, cong '
                       'seed 43); mean_patch = TB token patch tang cuoi sau LayerNorm (= G cua 78r), glob = CLS; float16; thu tu hang = thu tu anh cua '
                       'VinDrData (van tay); KHONG co ma anh')
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
    need = [FT.feature_file(r) for r in S.ROWS] + [FT.LABEL_FILE, FT.MANIFEST_FILE]
    co = set(need) <= set(on)
    log(f"C6: dong_bo {'DU' if du else 'CHUA DU'} | {len(need)} tep cua luot tren Drive: {'CO du' if co else 'THIEU ' + str(sorted(set(need) - set(on)))}"
        + ('' if du and co else ' | KHONG tat runtime; chay lai C6') + (' -> chay 92g (CPU)' if du and co else ''))
    return dict(dong_bo=du, du_tep=co)
