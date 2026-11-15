# -*- coding: utf-8 -*-
"""Ô của notebook Ver2 `89v2_<encoder>_<kho>` (PLAN_89 §5; G4 RTX PRO 6000). Mỗi hàm = một ô; trạng thái đi theo dict CTX; mọi việc Drive đi qua hàm
của khối Drive chuẩn ở C0 (drive_api lấy đúng các tên cần, thiếu -> dừng).

    C2 setup   thư viện notebook + CỔNG bộ thư viện đã chốt (torch / CUDA / transformers / timm) + GPU G4 + đọc tệp resume
    C3 data    kho ảnh của lượt (kcb = đệm vuông không cân bằng | cb = đệm vuông có cân bằng) + vân tay khoá / nhãn so với các lượt Ver1
    C4 gate    tệp trọng số (Drive / Hugging Face) -> lớp bọc + cổng nạp trọng số (từng tensor = tệp) + số học bf16 so với fp32 trên ảnh val thật
    C5 smoke   chạy thử CẢ ba giai đoạn trên tập con ở lô 256 với đủ 4 loại đầu x 13 lr (số đo vô nghĩa) + cổng bộ nhớ GPU + phục hồi từ trạng thái
               resume; bỏ qua khi lượt đang resume
    C6 train   lượt thật: dò lr (13 lr, seed 0) -> thêm 4 hạt ở lr chọn -> pha 2 trên train + val -> dự đoán test MỘT lần; resume theo giai đoạn
    C7 report  val cuối pha 1 + test: TB 5 hạt ± SD, ensemble; panel 28 (torchmetrics) / 27 / 21 / 7 — CHỈ MÔ TẢ (khoảng tin cậy + luật: bảng gộp);
               lỗi ở đây không làm mất C6
    C8 sync    dong_bo() + kiểm tệp trên Drive"""
import contextlib
import dataclasses
import json
import os
import re
import time
import traceback

from .. import __version__
from ..checks import weights as WT
from ..data import loaders as L
from ..data import vindr as V
from ..encoders import BUILT, SPECS
from ..encoders import files as W
from ..evaluation import metrics as M
from ..evaluation import panels as P
from ..protocols.chexfound_env import environment as E
from ..protocols.ours import backbone as BB
from ..protocols.ours import config as C
from ..protocols.ours import gates as G
from ..protocols.ours import pipeline as PL
from ..protocols.ours import runtime as RT

FOLDER = 'stage4_nb89'
DRIVE_NAMES = ('kiem_ke', 'tai', 'xung_dot', 'ghi', 'ghi_json', 'dong_bo', 'ten_tren_drive', 'tim_moi_noi', 'khoi_phuc_cho', 'luu_resume', 'cho_nho',
               'GHI_LOC', 'TEN_RESUME', 'RUN_TAG', '_duong', '_liet_ke')   # = notebook/ver1.DRIVE_NAMES
STORES = ('kcb', 'cb')                 # kho đệm vuông của 85 (PLAN_89 §5.2); kho cắt giữa (§5.3): thêm khi dựng kho
STRICT_FINGERPRINT = ('kcb',)          # kho của các lượt Ver1: vân tay PHẢI trùng; kho khác: in trùng / khác (bảng gộp ghép theo khoá ảnh)
MIN_FREE_GB = 15
GATE_IMAGES = 16                       # số ảnh val cho cổng số học bf16 vs fp32
SMOKE_SUBSET = dict(train=512, val=256, test=256)
SMOKE_RUN = C.MainRun(epochs_phase1=1, epochs_phase2=1, n_seeds=2)
ROUNDTRIP_TOL = 1e-3                   # dự đoán val (xác suất) của đầu phục hồi từ trạng thái resume so với đầu đang sống: cùng trọng số, backbone chạy lại
                                       # trên GPU -> không so từng bit; nạp sai trọng số cho lệch cỡ 1e-2 trở lên. In độ lệch thật ở C5.


def run_tag(encoder, store):
    return f'r89v2_{encoder}_{store}'


def notebook_name(encoder, store):
    return f'89v2_{encoder}_{store}'


def drive_api(ns):
    '''Các hàm / hằng của khối Drive chuẩn (C0) mà notebook dùng -> dict. Thiếu tên nào -> dừng (C0 chưa chạy / khối đổi).'''
    missing = [k for k in DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    return {k: ns[k] for k in DRIVE_NAMES}


def _label(tag):
    m = re.fullmatch(r'r(89[a-z0-9_]+)', tag)
    assert m, f'RUN_TAG {tag!r} khong co dang r89<...> -> DUNG'
    return m.group(1)


def _resume_state(drive, log):
    import torch
    tag = drive['RUN_TAG']
    path = drive['tim_moi_noi'](drive['TEN_RESUME'])
    if not path:
        return dict(run=tag, nhan=_label(tag), t=time.strftime('%Y%m%d_%H%M%S'), giai_doan={}, xong=[])
    st = torch.load(path, map_location='cpu', weights_only=False)
    assert st.get('run') == tag, f"tep resume cua luot KHAC ({st.get('run')} != {tag}) -> DOI RUN_TAG o C0, KHONG xoa gi"
    drive['khoi_phuc_cho'](st.get('cho'))
    gd = st.get('giai_doan') or {}
    log(f"RESUME: luot {st['t']} — " + (f"da XONG ({st['xong']})" if st.get('xong') else f"da xong giai doan {sorted(k for k in gd if k in ('s1', 's2')) or '-'}")
        + ' -> khoi phuc, KHONG chay lai phan da xong')
    return dict(run=tag, nhan=_label(tag), t=st['t'], giai_doan=gd, xong=list(st.get('xong') or []))


@contextlib.contextmanager
def quiet_hub():
    '''Tắt thanh tiến độ + cảnh báo "HF_TOKEN" của Hugging Face trong lúc nạp trọng số (output notebook sạch; không đổi số).'''
    import warnings
    tl, old = None, None
    with contextlib.suppress(Exception):
        import huggingface_hub.utils as hu
        hu.disable_progress_bars()
    with contextlib.suppress(Exception):
        import transformers.utils.logging as tl
        old = tl.get_verbosity()
        tl.disable_progress_bar()
        tl.set_verbosity_error()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r'(?s).*HF_TOKEN')
            yield
    finally:
        if tl is not None and old is not None: tl.set_verbosity(old)


def setup(encoder, store, package_dir, ns, log=print):
    '''C2. -> CTX. Bộ thư viện / GPU khác bộ đã chốt -> dừng (PLAN_89 §8 mục 7, §1 Runtime).'''
    assert encoder in BUILT, f'{encoder}: lop boc chua dung ({BUILT}) -> DUNG'
    assert store in STORES, f'kho {store!r} khong thuoc {STORES} -> DUNG'
    drive = drive_api(ns)
    assert drive['RUN_TAG'] == run_tag(encoder, store), f"RUN_TAG {drive['RUN_TAG']!r} != {run_tag(encoder, store)!r} -> DUNG"
    import vindr_bench
    mt = E.install_notebook_packages(min_free_gb=MIN_FREE_GB)
    v = G.versions()
    lib_ok, lib_diff = G.library_gate(v)
    gpu = G.gpu_info()
    device, amp = RT.device_and_amp()
    spec = SPECS[encoder]
    log(f"moi truong Ver2: Python {v['python']} | torch {v['torch']} (CUDA {v['cuda']}, cuDNN {v['cudnn']}) | transformers {v['transformers']} | timm "
        f"{v['timm']} | torchvision {v['torchvision']} | huggingface_hub {v['huggingface_hub']} | safetensors {v['safetensors']} | torchmetrics "
        f"{v['torchmetrics']} | GPU {gpu['ten']} {gpu['tong_gib'] or 0:.0f} GiB | backbone autocast {str(amp).replace('torch.', '')}, dau doc fp32 (TF32) | "
        f"o dia trong {mt['dia_trong_gb'] or 0:.0f} GB" + (f" | xung dot pip (ghi lai): {mt['xung_dot_pip']}" if mt['xung_dot_pip'] else ''))
    assert lib_ok, (f'bo thu vien KHAC bo da chot cua Ver2 {G.VER2_LIBS}: {lib_diff} (co / ghim) -> DUNG (PLAN_89 8 muc 7): KHONG chay tiep, gui Claude '
                    'dong "moi truong Ver2" o tren')
    assert device == 'cuda' and G.gpu_gate(gpu), (f"GPU {gpu['ten']!r} khong phai G4 ({G.GPU_MARK}) -> Runtime > Change runtime type > G4, roi chay lai "
                                                  '(PLAN_89 1: moi luot Ver2 cung GPU)')
    st = _resume_state(drive, log)
    ctx = dict(encoder=encoder, kho=store, spec=spec, drive=drive, device=device, amp=amp, anh_dir=f'/content/vd518_{store}', tt=st,
               moi_truong=dict(thu_vien=v, gpu=gpu, pip=mt), goi=dict(phien_ban=__version__, sha=getattr(vindr_bench, 'GOI_SHA', None)))
    log(f"encoder {spec.title}: {spec.arch}, anh {spec.image_size}, chuan hoa goc {spec.mean} / {spec.std}, vector toan anh {spec.pool} | kho {store} = "
        f"{V.STORE_LABELS[store]} | dau: {[C.TITLES[k] for k in C.HEAD_KINDS]} | luoi {len(C.LR_GRID)} lr {C.LR_GRID[0]:g} .. {C.LR_GRID[-1]:g}, "
        f"{C.EPOCHS_LOCKED} + {C.EPOCHS_LOCKED} epoch, {C.MainRun().n_seeds} hat | goi vindr_bench {__version__} @ {(ctx['goi']['sha'] or '?')[:12]}")
    return ctx


def data(ctx, log=print):
    '''C3. Kho ảnh của lượt + vân tay. -> VinDrData.'''
    d = ctx['drive']
    vd = V.load_vindr(d['kiem_ke'], d['tai'], d['xung_dot'], ctx['anh_dir'], store=ctx['kho'], log=log)
    fp = vd.fingerprint()
    same, diff = G.data_gate(fp)
    ctx.update(vindr=vd, van_tay=fp, van_tay_trung_ver1=same)
    log(vd.summary())
    log('van tay md5 khoa / nhan: ' + ' | '.join(f"{s} {fp[s]['md5_khoa'][:8]} / {fp[s]['md5_nhan'][:8]}" for s in V.SPLITS)
        + (' | TRUNG cac luot Ver1' if same else f' | KHAC cac luot Ver1 (co / ghim): {diff}'))
    assert same or ctx['kho'] not in STRICT_FINGERPRINT, (f"kho {ctx['kho']}: van tay du lieu KHAC cac luot Ver1 {diff} -> DUNG (PLAN_89 8 muc 6), gui "
                                                           'Claude dong tren')
    return vd


def gate(ctx, log=print):
    '''C4. Lớp bọc + cổng nạp trọng số + số học bf16 so với fp32 (ghi số; chỉ dừng khi hỏng nặng — protocols/ours/gates.py).'''
    import torch
    d, spec, vd = ctx['drive'], ctx['spec'], ctx['vindr']
    ctx['tep_trong_so'] = W.fetch(spec, d, log=log) if spec.drive_dir else None
    if ctx['tep_trong_so']:
        for t in [ctx['tep_trong_so']] + list(ctx['tep_trong_so'].get('phu') or []):
            log(f"tep tu Drive: {spec.drive_dir}/{t['tep']} | md5 {t['md5']} = ghim | {t['byte']:,} B | {t['so_ban']} ban tren Drive")
    with quiet_hub():
        bb = BB.load(ctx['encoder'], ctx['device'], ctx['amp'])
    nap = bb.enc.nap
    log(f"nap trong so {'DAT' if nap['dat'] else 'TRUOT'} | {WT.loading_text(nap)}")
    assert nap['dat'], 'nap trong so KHONG dat -> DUNG'
    _, tf_eval = L.build_transforms(bb.spec)
    ds = L.VinDrImages(vd.keys, vd.image_dir, tf_eval)
    x = torch.stack([ds[int(i)][0] for i in vd.indices('val')[:GATE_IMAGES]])
    g = G.numeric_check(bb, x)
    log(G.numeric_text(g))
    assert g['dat'], 'dac trung bf16 HONG NANG so voi fp32 (hoac khong huu han / 4 tang trung nhau) -> DUNG, gui Claude dong tren'
    n_par = sum(p.numel() for p in bb.enc.parameters())
    log(f"lop boc {spec.title}: {n_par / 1e6:.1f} M tham so, DONG CUNG (dac trung duoi no_grad), {spec.n_patches} token patch x {spec.embed_dim} chieu, "
        f"backbone theo khuc {bb.enc.forward_chunk} anh")
    ctx.update(backbone=bb, cong=dict(nap=json.loads(json.dumps(nap, default=str)), so_hoc=g, tep=ctx['tep_trong_so'], so_tham_so=n_par))
    return ctx['cong']


def _smoke_log(log):
    '''Lượt chạy thử: bỏ dòng từng epoch và bảng lr (số đo vô nghĩa), giữ dòng mở đầu + dòng kết thúc.'''
    return lambda s='': log('  ' + s.strip()) if s.strip().startswith(('Ver2 ', 'xong:')) else None


def smoke(ctx, log=print):
    '''C5. Chạy thử trên tập con ở lô 256 với đủ đầu của lượt thật; trượt -> dừng trước lượt đầy đủ.'''
    import torch
    st = ctx['tt']
    if st['giai_doan'] or st['xong']:
        log('RESUME: bo qua chay thu (da dat o phien truoc)')
        ctx['thu'] = dict(bo_qua_vi_resume=True)
        return ctx['thu']
    t0 = time.time()
    pl = PL.prepare(ctx['vindr'], ctx['backbone'], main=SMOKE_RUN, subset=SMOKE_SUBSET)
    state = {}
    result, arrays = PL.run(pl, state=state, save=None, run_tag='chay_thu', log=_smoke_log(log))
    peak = torch.cuda.max_memory_allocated() if ctx['device'] == 'cuda' else 0
    total = torch.cuda.get_device_properties(0).total_memory if ctx['device'] == 'cuda' else 0
    lech = PL.roundtrip_check(pl, state, arrays)
    n_test = sum(1 for k in arrays if k.startswith('test_'))
    ck = dict(so_buoc=result['kiem']['so_buoc_dung'], huu_han=result['kiem']['du_doan_huu_han'],
              bo_nho=(ctx['device'] != 'cuda' or G.memory_ok(peak, total)), phuc_hoi=lech <= ROUNDTRIP_TOL,
              du_dau=n_test >= len(C.HEAD_KINDS) * SMOKE_RUN.n_seeds and all(arrays[k].shape == (SMOKE_SUBSET['test'], len(ctx['vindr'].label_names))
                                                                             for k in arrays if k.startswith('test_')))
    thu = dict(dat=all(ck.values()), kiem=ck, gpu_dinh_gib=peak / 2 ** 30, gpu_tong_gib=total / 2 ** 30, lech_phuc_hoi=lech, so_dau_test=n_test,
               chon={k: {c: v[c]['lr'] for c in C.CRITERIA} for k, v in result['chon'].items()}, phut=(time.time() - t0) / 60, tap_con=SMOKE_SUBSET)
    log(f"chay thu {'DAT' if thu['dat'] else 'TRUOT'} ({thu['phut']:.1f} phut; {len(C.HEAD_KINDS)} loai dau x {len(C.LR_GRID)} lr o lo {C.Protocol().batch_size}, "
        f"tap con {SMOKE_SUBSET}, so do VO NGHIA): so buoc {'DAT' if ck['so_buoc'] else 'TRUOT'} | du doan huu han {'DAT' if ck['huu_han'] else 'TRUOT'} | "
        f"{n_test} bo du doan test dung hinh {'DAT' if ck['du_dau'] else 'TRUOT'} | GPU dinh {thu['gpu_dinh_gib']:.1f} / {thu['gpu_tong_gib']:.1f} GiB "
        f"(nguong {G.MEM_LIMIT:.0%}: {'DAT' if ck['bo_nho'] else 'TRUOT'}) | phuc hoi tu trang thai resume: lech du doan val {lech:.1e} "
        f"(<= {ROUNDTRIP_TOL:g}: {'DAT' if ck['phuc_hoi'] else 'TRUOT'})")
    del state, arrays, result, pl
    if ctx['device'] == 'cuda': torch.cuda.empty_cache()
    ctx['thu'] = thu
    assert thu['dat'], f"chay thu TRUOT: {[k for k, v in ck.items() if not v]} -> DUNG truoc luot day du, gui Claude output C5"
    return thu


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


def _files(st):
    return f"luot_{st['nhan']}_{st['t']}.json", f"du_doan_{st['nhan']}_{st['t']}.npz"


def train(ctx, log=print):
    '''C6. Lượt thật có resume theo giai đoạn (tệp resume ghi đè sau giai đoạn (1), (2) và khi xong). -> kết quả (dict).'''
    import numpy as np
    import torch
    d, st = ctx['drive'], ctx['tt']
    if st['xong']:
        with open(os.path.join(d['GHI_LOC'], st['xong'][0]), encoding='utf-8') as f:
            ctx['kq'] = json.load(f)
        log(f"RESUME: luot da xong -> doc lai {st['xong'][0]} (khong hoc lai)")
        return ctx['kq']
    save = lambda gd, xong=(): d['luu_resume'](lambda q: torch.save(dict(run=st['run'], t=st['t'], giai_doan=gd, xong=list(xong), cho=d['cho_nho']()), q))
    pl = PL.prepare(ctx['vindr'], ctx['backbone'])
    result, arrays = PL.run(pl, state=st['giai_doan'], save=save, run_tag=st['run'], log=log)
    result.update(t=st['t'], encoder=dataclasses.asdict(ctx['spec']), moi_truong=ctx['moi_truong'], goi=ctx['goi'], cong=ctx['cong'], van_tay=ctx['van_tay'],
                  van_tay_trung_ver1=ctx['van_tay_trung_ver1'], chay_thu=ctx.get('thu'), thu_vien_ghim=G.VER2_LIBS)
    names = _files(st)
    d['ghi'](names[1], lambda q: np.savez_compressed(q, **arrays))
    js = _jsonable(result)
    d['ghi'](names[0], lambda q: d['ghi_json'](q, js))
    st['xong'] = list(names)
    save({}, names)
    ctx['kq'] = js
    c = result['chon']
    log(f"da ghi {names[0]} + {names[1]} (local; len Drive o C8) | lr cua luat: "
        + ', '.join(f"{C.TITLES[k]} {c[k][C.RULE]['lr']:g}" + (' (BIEN LUOI)' if c[k][C.RULE]['bien_luoi'] else '') for k in c)
        + ' | lr theo AP khac luat: ' + (', '.join(f"{C.TITLES[k]} {c[k]['ap']['lr']:g}" for k in c if not c[k]['ap_trung_luat']) or 'khong'))
    return js


def _pm(v):
    import numpy as np
    v = np.asarray(v, dtype=np.float64)
    return f'{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.4f}'


def score(ctx):
    '''Chấm các mảng dự đoán của lượt (điểm, không bootstrap). -> {nhánh: dict}. Số chính của Ver2 = TB metric của các hạt (PLAN_89 §7.2); ensemble
       (TB xác suất) báo kèm.'''
    import numpy as np
    d, st, vd = ctx['drive'], ctx['tt'], ctx['vindr']
    with np.load(os.path.join(d['GHI_LOC'], st['xong'][1])) as z:
        arr = {k: z[k] for k in z.files}
    y_test, y_val = arr['y_test'], arr['y_val']
    assert np.array_equal(y_test, vd.labels[vd.indices('test')]) and np.array_equal(y_val, vd.labels[vd.indices('val')]), 'nhan trong tep du doan KHAC kho -> DUNG'
    cols = P.panel_columns(vd.label_names, vd.main_classes, y_test)
    half = M.halves(len(y_test), P.HALF_SEED)
    out = {}
    for k in ctx['kq']['loai_dau']:
        for crit in C.CRITERIA:
            tag = PL.arm_tag(k, crit)
            ch = ctx['kq']['chon'][k]
            if crit != C.RULE and ch['ap_trung_luat']:
                continue
            seeds = sorted(int(n.rsplit('_s', 1)[1]) for n in arr if n.startswith(f'test_{tag}_s'))
            pt = [arr[f'test_{tag}_s{s}'] for s in seeds]
            pv = [arr[f'val_{tag}_s{s}'] for s in seeds]
            per = [{p: P.point(q, y_test, cols[p]) for p in P.PANELS} for q in pt]
            ens = {p: P.point(np.mean(pt, 0), y_test, cols[p]) for p in P.PANELS}
            val = [M.ap_auroc_tm0103(y_val, q) for q in pv]
            out[tag] = dict(loai=k, tieu_chi=crit, lr=ch[crit]['lr'], bien_luoi=ch[crit]['bien_luoi'], so_hat=len(seeds),
                            tung_hat={p: {m: [float(x[p][m]) for x in per] for m in ('ap', 'auprc_thang', 'auroc')} for p in P.PANELS},
                            ensemble={p: {m: float(ens[p][m]) for m in ('ap', 'auprc_thang', 'auroc')} for p in P.PANELS},
                            ap21_hai_nua=[[float(P.point(q[half == h], y_test[half == h], cols['p21'])['ap']) for q in pt] for h in (0, 1)],
                            val=dict(ap28=[float(x[1]) for x in val], auroc28=[float(x[3]) for x in val]),
                            ap21_tung_lop_tb=[float(x) for x in np.mean([x['p21']['ap_lop'] for x in per], 0)])
    tm = M.tm_scores(arr[f"test_{PL.arm_tag(ctx['kq']['loai_dau'][0], C.RULE)}_s0"], y_test.astype(np.int64))
    pk = P.point(arr[f"test_{PL.arm_tag(ctx['kq']['loai_dau'][0], C.RULE)}_s0"], y_test, cols['p28'])
    return out, dict(lech_ap=abs(tm[0] - pk['ap']), lech_auroc=abs(tm[1] - pk['auroc'])), dict(cot={p: cols[p] for p in P.PANELS},
                                                                                                so_duong_test=[int(v) for v in y_test.sum(0)])


def report(ctx, log=print):
    '''C7 (CHỈ MÔ TẢ). Lỗi ở đây KHÔNG làm mất C6: in lỗi, trả None — C8 vẫn đồng bộ.'''
    try:
        import numpy as np
        d, st, sp, kq = ctx['drive'], ctx['tt'], ctx['spec'], ctx['kq']
        sc, tm, info = score(ctx)
        log(f"\n== Ver2 {sp.title} — ma + moi truong cua ta (G4), kho {ctx['kho']} = {V.STORE_LABELS[ctx['kho']]}, anh {sp.image_size}, chuan hoa goc, "
            f"backbone {kq['amp_backbone'].replace('torch.', '')}, dau doc eval() khi cham; {kq['so_seed']} hat: so = TB ± SD mau giua cac hat (so chinh), "
            f"'ens' = ensemble TB xac suat (bao kem); KHONG co khoang tin cay o day (bang gop) ==")
        log('--- VAL cuoi pha 1 (1.500 anh, 28 cot ngu nghia torchmetrics 0.10.3; chi de theo doi) ---')
        for tag, s in sc.items():
            name = C.TITLES[s['loai']] if s['tieu_chi'] == C.RULE else ''
            log(f"{name:<10} {PL.CRIT_TEXT[s['tieu_chi']]} lr {s['lr']:<7g} AUROC28 {_pm(s['val']['auroc28'])} | AP28 {_pm(s['val']['ap28'])}"
                + (' | lr O BIEN LUOI' if s['bien_luoi'] else ''))
        log('--- TEST (3.000 anh; AP = AP bac thang, AUPRC = hinh thang; p21 = 21 lop so chinh — so cua luat) ---')
        for tag, s in sc.items():
            name = C.TITLES[s['loai']] if s['tieu_chi'] == C.RULE else ''
            th, en = s['tung_hat'], s['ensemble']
            log(f"{name:<10} {PL.CRIT_TEXT[s['tieu_chi']]} lr {s['lr']:<7g} p21: AP {_pm(th['p21']['ap'])} (ens {en['p21']['ap']:.4f}) | AUPRC thang "
                f"{_pm(th['p21']['auprc_thang'])} (ens {en['p21']['auprc_thang']:.4f}) | AUROC {_pm(th['p21']['auroc'])} (ens {en['p21']['auroc']:.4f})")
            log(f"{'':<10} {'':<7} hai nua AP21 TB {np.mean(s['ap21_hai_nua'][0]):.4f} / {np.mean(s['ap21_hai_nua'][1]):.4f} | p28 torchmetrics: AP "
                f"{_pm(th['p28']['ap'])} AUROC {_pm(th['p28']['auroc'])} | p27: AP {_pm(th['p27']['ap'])} AUPRC thang {_pm(th['p27']['auprc_thang'])} | "
                f"p7: AP {_pm(th['p7']['ap'])} AUPRC thang {_pm(th['p7']['auprc_thang'])}")
        khac = [C.TITLES[k] for k, c in kq['chon'].items() if not c['ap_trung_luat']]
        log(f"nhanh 'theo AP' (mo ta, khong vao luat): " + (f"lr KHAC luat o {khac}" if khac else 'trung lr cua luat o moi loai dau')
            + f" | doi chieu torchmetrics 0.10.3 that (p28, mot bo du doan): lech AP {tm['lech_ap']:.1e}, AUROC {tm['lech_auroc']:.1e}")
        name = f"bang_{st['nhan']}_{st['t']}.json"
        obj = _jsonable(dict(run=d['RUN_TAG'], t=st['t'], goi=ctx['goi'], phien_ban_ver='Ver2', encoder=dataclasses.asdict(sp), kho=ctx['kho'],
                             diem=sc, doi_chieu_torchmetrics=tm, panel=info, chon=kq['chon'], tep_luot=st['xong'],
                             ghi_chu='diem tren toan test, khong bootstrap; so chinh = TB cac hat (PLAN_89 7.2); luat + khoang dong thoi: bang gop'))
        d['ghi'](name, lambda q: d['ghi_json'](q, obj))
        ctx['tep_bang'] = name
        log(f'\nda ghi {name} (local; len Drive o C8)')
        return sc
    except Exception:
        log('⚠ C7 LOI — ket qua C6 KHONG mat: chay tiep C8, roi gui Claude loi duoi day')
        traceback.print_exc()
        return None


def sync(ctx, log=print):
    '''C8. dong_bo() + kiểm 2 tệp của lượt + bảng C7 trên Drive.'''
    d = ctx['drive']
    du = d['dong_bo']()
    need, bg = list(ctx['tt']['xong']), ctx.get('tep_bang')
    on = d['ten_tren_drive']()
    co = len(need) == 2 and set(need) <= on
    log(f"C8: dong_bo {'DU' if du else 'CHUA DU'} | 2 tep cua luot (JSON + du doan) tren Drive: {'CO' if co else 'THIEU / C6 CHUA XONG'} | bang C7: "
        f"{'CO' if bg and bg in on else 'THIEU (C7 loi / chua chay)'}"
        + (' -> gui Claude output C2-C8; roi chay 89c (chep sang corntoun1505)' if du and co else ' | KHONG tat runtime; chay lai C8, roi bao Claude'))
    return dict(dong_bo=du, du_tep=co, bang=bool(bg and bg in on))
