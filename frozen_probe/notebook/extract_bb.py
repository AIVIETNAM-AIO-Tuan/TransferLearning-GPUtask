# -*- coding: utf-8 -*-
"""Ô của notebook `92d_trich_dac_trung_vindr_bb` (G4; PLAN_92 §13) — cùng khuôn 92a (extract.py: Drive, kho, vân tay, cổng thư viện / GPU / bộ nhớ /
số học, tệp đặc trưng KHÔNG đổi), chỉ khác: sáu hàng của ba cặp mới, lớp bọc frozen_probe.encoders (không phải vindr_bench). Ghi Drive MỘT lần (C6).

    C2 setup     thư viện notebook + CỔNG bộ thư viện / GPU của 89 Ver2 + HF_TOKEN (DINOv3 gated) — trước mọi tải
    C3 data      kho VinDr `kcb` + vân tay = các lượt 89 (= extract.data)
    C4 extract   6 hàng: tệp 78x (md5 ghim) [+ checkpoint CheXFound gốc] -> lớp bọc + cổng (lượt, mặt nạ, nạp, lệch pretrain, forward) -> số học bf16
                 vs fp32 + tỉ lệ ghi ngược trên ảnh VinDr (in cạnh nhật ký lượt học) -> trích 18.000 ảnh (cổng bộ nhớ ở lô đầu) -> tệp đặc trưng;
                 chạy lại ô: chỉ làm hàng chưa xong
    C5 manifest  tệp nhãn / tách + manifest (md5 từng tệp, cổng từng hàng, môi trường)
    C6 sync      dong_bo() + kiểm tệp trên Drive"""
import gc
import json
import time

from .. import config_bb as CB
from .. import features as FT
from . import extract as X

MIN_FREE_GB = 25                       # 6 hàng x (rs_ + backbone_ft.pt) + checkpoint CheXFound gốc + kho ảnh
WRITE_BACK_TOL = 1e-3                  # tỉ lệ ghi ngược trên ảnh VinDr (bf16) <= rho + WRITE_BACK_TOL


def setup(ns, log=print):
    '''C2. -> CTX. Bộ thư viện / GPU khác bộ đã chốt của 89 Ver2, hoặc thiếu HF_TOKEN -> dừng TRƯỚC mọi tải.'''
    from vindr_bench.protocols.chexfound_env import environment as E
    from vindr_bench.protocols.ours import gates as G
    from vindr_bench.protocols.ours import runtime as RT
    from ..encoders import bb as BBW
    drive = X.drive_api(ns)
    assert drive['RUN_TAG'] == CB.RUN_TAGS['trich'], f"RUN_TAG {drive['RUN_TAG']!r} != {CB.RUN_TAGS['trich']!r} -> DUNG"
    mt = E.install_notebook_packages(min_free_gb=MIN_FREE_GB)
    v = G.versions()
    lib_ok, lib_diff = G.library_gate(v)
    gpu = G.gpu_info()
    device, amp = RT.device_and_amp()
    pk = X._packages()
    tok = BBW.hf_token()
    log(f"92d moi truong: Python {v['python']} | torch {v['torch']} (CUDA {v['cuda']}) | transformers {v['transformers']} | timm {v['timm']} | GPU {gpu['ten']} "
        f"{gpu['tong_gib'] or 0:.0f} GiB | backbone autocast {str(amp).replace('torch.', '')} | frozen_probe {pk['frozen_probe']['phien_ban']} @ "
        f"{(pk['frozen_probe']['sha'] or '?')[:12]} + vindr_bench {pk['vindr_bench']['phien_ban']} @ {(pk['vindr_bench']['sha'] or '?')[:12]} | HF_TOKEN "
        f"{'co' if tok else 'THIEU'}")
    assert lib_ok, f'bo thu vien KHAC bo da chot cua 89 Ver2 {G.VER2_LIBS}: {lib_diff} (co / ghim) -> DUNG, gui Claude dong tren'
    assert device == 'cuda' and G.gpu_gate(gpu), f"GPU {gpu['ten']!r} khong phai G4 ({G.GPU_MARK}) -> Runtime > Change runtime type > G4"
    assert tok, 'thieu HF_TOKEN trong Colab Secrets (DINOv3 la mo hinh gated, cung token cua luot 78x thang 10) -> them roi chay lai tu C0'
    log(f"hang ({len(CB.ROWS)}): " + ' | '.join(CB.TEXT[r] for r in CB.ROWS))
    log(f"dac trung: {' | '.join(f'{k} = {X.C.FEATURE_TEXT[k]}' for k in CB.FEATURES)} | float16 | lo {X.C.EXTRACT_BATCH}")
    return dict(drive=drive, device=device, amp=amp, goi=pk, moi_truong=dict(thu_vien=v, gpu=gpu, pip=mt), xong={}, t=time.strftime('%Y%m%d_%H%M%S'), token=tok)


def data(ctx, log=print):
    '''C3. = 92a C3 (kho `kcb`, vân tay = các lượt 89).'''
    return X.data(ctx, log)


def _nap_text(nap):
    from vindr_bench.checks import weights as WT
    cl, cf, lu = nap['cong_luot'], nap['cong_forward'], nap.get('luong')
    s = (f"lượt {cl['run']} ep{cl['ep']} | EMA {cl['ema_n']} lan cap nhat (he so {cl['ema_he_so']:.4f}) | transformers luc luu {cl.get('tf_luu')} | "
         + (f"cfg_kl {cl['cfg_kl']['ten']} / {cl['cfg_kl']['encoder']}" if 'cfg_kl' in cl else
            f"cfg_d4 {cl['cfg_d4']} | ghi nguoc x{cl['g']:g}, tran {cl['rho']:g} | cfg_p92 {cl['cfg_p92']}")
         + f"\n  mat na = forward goc (lech {nap['cong_mat_na']['lech']}) | lech pretrain {nap['lech_pre']['do']} = 78r ({nap['lech_pre']['ghim']}) | "
         + WT.loading_text(nap))
    if lu:
        s += (f"\n  luong vung: R {lu['R']} | tang {lu['tang']} | {lu['n_head']} dau, ff {lu['ff']} | P {lu['n_dau']} ({lu['lop']}) | neo view {lu['view_neo']} | "
              f"cong forward (CPU fp32, {cf['so_anh']} anh tong hop): _blocks = upper_luong_hr tung bit {cf['trung_bit_upper_luong_hr']}, {cf['so_lop_jca_chay']} lop "
              f"JCA, ti le ghi {[round(t, 4) for t in cf['ti_le_ghi']]}")
    else:
        s += f"\n  cong forward (CPU fp32, {cf['so_anh']} anh tong hop): tang cuoi = forward day du cua mat na tung bit {cf['trung_bit_forward_day_du']}"
    return s


def _one(ctx, row, log):
    import torch
    from vindr_bench.checks import weights as WT
    from vindr_bench.data import loaders as L
    from vindr_bench.encoders import SPECS as VSPECS
    from vindr_bench.encoders import files as W
    from vindr_bench.notebook import ver2 as V2
    from vindr_bench.protocols.ours import backbone as BBK
    from vindr_bench.protocols.ours import gates as G
    from vindr_bench.protocols.ours import runtime as RT
    from ..encoders import bb as BBW
    from ..encoders.rows import ROWS
    d, vd, spec = ctx['drive'], ctx['vindr'], ROWS[row]
    t0 = time.time()
    tep = W.fetch(spec, d, log=log)
    for t in [tep] + list(tep.get('phu') or []):
        log(f"  tep tu Drive: {spec.drive_dir}/{t['tep']} | md5 {t['md5']} = ghim | {t['byte']:,} B | {t['so_ban']} ban tren Drive")
    kw = dict(token=ctx['token'])
    if spec.encoder == 'chexfound':
        if 'chexfound_goc' not in ctx:
            ctx['chexfound_goc'] = W.fetch(VSPECS['chexfound'], d, log=log)
            g0 = ctx['chexfound_goc']
            log(f"  CheXFound tien huan luyen: {VSPECS['chexfound'].drive_dir}/{g0['tep']} | md5 {g0['md5']} = ghim (vindr_bench) | {g0['byte']:,} B")
        kw['chexfound_path'] = W.weights_path(VSPECS['chexfound'])
    with V2.quiet_hub():
        enc = BBW.build(row, **kw)
    nap = enc.nap
    log('  ' + _nap_text(nap))
    assert nap['dat'], f'{row}: cong nap / forward KHONG dat -> DUNG'
    enc = enc.to(ctx['device']).eval()
    bb = BBK.Backbone(enc, ctx['device'], ctx['amp'])
    _, tf_eval = L.build_transforms(bb.spec)
    ds = L.VinDrImages(vd.keys, vd.image_dir, tf_eval)
    x = torch.stack([ds[int(i)][0] for i in vd.indices('val')[:X.GATE_IMAGES]])
    g = G.numeric_check(bb, x)
    log('  ' + G.numeric_text(g))
    assert g['dat'], f'{row}: dac trung bf16 HONG NANG so voi fp32 -> DUNG'
    wb = []
    if spec.hang == 'co':
        with torch.autocast('cuda', dtype=ctx['amp']):
            wb = enc.write_back(x.to(ctx['device']))
        ok_wb = len(wb) == len(spec.tang) and all(t == t and 0.0 < t <= float(spec.rho) + WRITE_BACK_TOL for t in wb)
        log(f"  ghi nguoc ||dh|| / ||h|| tren {len(x)} anh val VinDr (bf16): {[round(t, 4) for t in wb]} | nhat ky luot hoc ep{spec.ep_bb} (anh CXR-LT): "
            f"{[round(float(t), 4) for t in (enc.ghi_log or [])]} | <= tran {spec.rho:g}: {'DAT' if ok_wb else 'TRUOT'}")
        assert ok_wb, f'{row}: ti le ghi nguoc tren anh VinDr ngoai (0, tran] hoac khong du {len(spec.tang)} lop -> DUNG'
    total = torch.cuda.get_device_properties(0).total_memory
    feats = FT.extract(bb, vd, batch_size=X.C.EXTRACT_BATCH, num_workers=RT.default_num_workers(), log=log,
                       first_batch_gate=lambda peak: G.memory_ok(peak, total))
    b = FT.feature_bytes(feats, row, spec.name)
    name = FT.feature_file(row)
    d['ghi'](name, lambda q, b=b: X._write_bytes(q, b))
    rec = dict(hang=row, ma=CB.CODE[row], dieu_kien=CB.TEXT[row], tep=name, md5=FT.md5_bytes(b), byte=len(b), so_anh=feats['so_anh'], chieu=int(bb.dim),
               phut_trich=round(feats['phut'], 2), phut_tong=round((time.time() - t0) / 60, 2), gpu_dinh_gib=round(feats['gpu_dinh_gib'], 2),
               gpu_tong_gib=round(total / 2 ** 30, 1), nap=WT.loading_text(nap), cong_luot=nap['cong_luot'], lech_pre=nap['lech_pre'],
               cong_mat_na=nap['cong_mat_na'], cong_forward=nap['cong_forward'], luong=nap.get('luong'), ghi_nguoc_vindr=wb, ghi_nguoc_log=enc.ghi_log,
               so_hoc=dict(dat=g['dat'], patch=g['patch'], vector=g['vector']), tep_trong_so=tep, anh=int(spec.image_size), token_patch=int(spec.n_patches),
               tien_huan_luyen=nap['tien_huan_luyen'])
    log(f"  {CB.CODE[row]}: {feats['so_anh']:,} anh x {bb.dim} chieu x 2 dac trung (float16) | trich {feats['phut']:.1f} phut, ca hang {rec['phut_tong']:.1f} phut | "
        f"GPU dinh {feats['gpu_dinh_gib']:.1f} / {total / 2 ** 30:.0f} GiB | tep {name} {len(b) / 2 ** 20:.1f} MB, md5 {rec['md5'][:12]} (ghi local; len Drive o C6)")
    del bb, enc, feats, b, x, ds
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def extract(ctx, log=print):
    '''C4. 6 hàng, lần lượt; hàng đã xong trong phiên bỏ qua (chạy lại ô sau khi sửa lỗi một hàng).'''
    t0 = time.time()
    for k, row in enumerate(CB.ROWS, 1):
        if row in ctx['xong']:
            log(f'[{k}/{len(CB.ROWS)}] {CB.TEXT[row]}: da xong trong phien -> bo qua')
            continue
        log(f'[{k}/{len(CB.ROWS)}] {CB.TEXT[row]}')
        ctx['xong'][row] = _one(ctx, row, log)
        log(f"  -> xong {len(ctx['xong'])} / {len(CB.ROWS)} hang | {(time.time() - t0) / 60:.1f} phut tu dau o")
    log(f"xong {len(ctx['xong'])} / {len(CB.ROWS)} hang | {(time.time() - t0) / 60:.1f} phut (o nay)")
    return ctx['xong']


def manifest(ctx, log=print):
    '''C5. Tệp nhãn / tách + manifest. -> dict manifest.'''
    assert set(ctx['xong']) == set(CB.ROWS), f"chua du hang: thieu {[r for r in CB.ROWS if r not in ctx['xong']]} -> chay lai C4"
    d, vd = ctx['drive'], ctx['vindr']
    b = FT.label_bytes(vd, ctx['van_tay'])
    d['ghi'](FT.LABEL_FILE, lambda q: X._write_bytes(q, b))
    man = dict(run=CB.RUN_TAGS['trich'], t=ctx['t'], goi=ctx['goi'], kho=X.C.STORE, van_tay=ctx['van_tay'], so_anh=len(vd.keys), thu_tu_hang=list(CB.ROWS),
               cap={e: list(p) for e, p in CB.PAIRS.items()}, dac_trung=list(CB.FEATURES), tep_nhan=dict(tep=FT.LABEL_FILE, md5=FT.md5_bytes(b), byte=len(b)),
               hang=ctx['xong'], moi_truong=dict(thu_vien=ctx['moi_truong']['thu_vien'], gpu=ctx['moi_truong']['gpu']),
               ghi_chu='dac trung dong cung 92d (PLAN_92 13): 6 hang = 3 cap CO / KHONG luong vung (DINOv3, CheXFound x1, BiomedCLIP), rs_ trong so EMA = tep '
                       'cua 78r; mean_patch = TB token patch tang cuoi sau layernorm cua mat na (= G cua 78r), glob = CLS; float16; thu tu hang = thu tu anh cua '
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
    need = [FT.feature_file(r) for r in CB.ROWS] + [FT.LABEL_FILE, FT.MANIFEST_FILE]
    co = set(need) <= set(on)
    log(f"C6: dong_bo {'DU' if du else 'CHUA DU'} | {len(need)} tep cua luot tren Drive: {'CO du' if co else 'THIEU ' + str(sorted(set(need) - set(on)))}"
        + ('' if du and co else ' | KHONG tat runtime; chay lai C6') + (' -> chay 92e (CPU)' if du and co else ''))
    return dict(dong_bo=du, du_tep=co)
