# -*- coding: utf-8 -*-
"""Ô của notebook lượt `93v_<bộ>_<encoder>` (PLAN_93 §7; G4 RTX PRO 6000) = các ô Ver2 của `89` (vindr_bench.notebook.ver2 @ c81af62) trên kho của bộ ngoài.
Mỗi hàm = một ô; trạng thái đi theo dict CTX; Drive qua khối chuẩn ở C0.

    C2 setup   = ver2.setup: thư viện notebook + CỔNG bộ thư viện đã chốt của Ver2 (torch / CUDA / transformers / timm) + GPU G4 + đọc tệp resume
    C3 data    kho 518 của bộ (93k) + cổng: md5 khoá = meta, số ảnh / số dương / cụm không vắt tách, vân tay = ghim
    C4 gate    GỌI NGUYÊN ver2.gate: trọng số -> lớp bọc + cổng nạp (từng tensor = tệp) + số học bf16 so với fp32 trên 16 ảnh val của bộ
    C5 smoke   chạy thử CẢ ba giai đoạn trên tập con (<= 512 / 256 / 256 ảnh) ở lô 256, đủ 4 loại đầu x 13 lr + cổng bộ nhớ GPU <= 70 % + phục hồi từ trạng
               thái resume; 0.5.0: + lượt 2 + 2 epoch chạy liền và chạy có BA lần ngắt (giữa mỗi giai đoạn; nạp lại qua torch.save / load như phiên mới)
               -> dự đoán val + test trùng (<= ROUNDTRIP_TOL); bỏ qua khi lượt đang resume. 0.5.2: HAI lần chạy liền (nền nhiễu GPU) — luật
               resume_verdict: dự đoán + đường học của các ĐẦU ĐƯỢC CHỌN <= max(1e-3, 3 x nền); đường học mọi đầu dò lr chỉ in kèm nền.
               0.6.0 (= 0.5.3, Tuấn 27/11 "Gộp 0.5.3"): các lần chạy so sánh dưới heads_math (attention MATH ở đầu đọc, backbone giữ kernel) -> A, B
               trùng TỪNG BIT thì chạy tiếp phải trùng TỪNG BIT; A, B vẫn lệch -> luật dự phòng của ô C5c (thêm C, D; gần một lần chạy liền hơn cặp xa nhất)
    C6 train   lượt thật (pipeline 89 nguyên văn, E = DatasetSpec.epochs); resume theo giai đoạn; 0.5.0: + lưu GIỮA giai đoạn ~30 phút (protocol/resume) + xoá
               phiên bản cũ của tệp resume + in tốc độ đo / giờ còn lại; dự đoán test MỘT lần
    C7 report  val cuối pha 1 + test: TB 5 hạt ± SD, ensemble, hai nửa theo cụm — CHỈ MÔ TẢ (khoảng tin cậy + luật: 93t); lỗi ở đây không làm mất C6
    C8 sync    dong_bo() + kiểm tệp trên Drive
0.5.0 (CheXpert (A), nhiều phiên): C2 chặn hai phiên cùng chạy một lượt (notebook/sessions); C3 nạp kho chia phần (datasets/base) — quota tải của một phần
hết -> tải bản sao phía máy chủ.
0.6.0 (Tuấn 27/11, PLAN_93 §13): C2 nhận hàng thêm (config.EXTRA_ROWS = ASG) ngoài 9 hàng."""
import contextlib
import dataclasses
import json
import os
import re
import time
import traceback

import numpy as np

from .. import config
from ..datasets import base
from ..evaluation import single as S
from ..protocol import resume as RS
from ..protocol import run as R
from . import sessions as SS

MIN_FREE_GB = 15
SMOKE_SUBSET = dict(train=512, val=256, test=256)
ROUNDTRIP_TOL = 1e-3                      # = vindr_bench.notebook.ver2.ROUNDTRIP_TOL
RESUME_STOPS = (('s1', 1), ('s2', 1), ('s3', 1))   # chạy thử chạy tiếp: ngắt sau bước 1 của MỖI giai đoạn (giữa epoch)
# Lượt chạy tiếp của C5: 2 epoch mỗi giai đoạn -> sau mỗi lần ngắt còn >= 3 bước (với 1 epoch chỉ còn 1 bước: bước đó dùng lr đã nạp qua optimizer, lịch lr
# nạp sai chỉ lộ từ bước SAU — đột biến R3 25/11 không bị bắt khi 1 epoch)
RESUME_RUN = dict(epochs_phase1=2, epochs_phase2=2, n_seeds=2)
# 0.6.0 (= 0.5.3 đề xuất 26/11; Tuấn 27/11 "Gộp 0.5.3"): luật "chạy tiếp = chạy liền" của C5. Đo 26/11 (Event "ô `C5d` (CheXWorld)"): với attention SDPA
# kiểu MATH hai lần chạy liền trùng TỪNG BIT (24 / 24 mảng, mọi đầu); lệch của 0.5.2 đến từ kernel attention hợp nhất (nền đo bằng MỘT cặp báo sai — nhiễu
# có hai nhánh, ô C5c). Các lần chạy so sánh chạy dưới heads_math (MATH chỉ ở đầu đọc; backbone giữ đúng kernel của lượt thật) -> so TỪNG BIT; hai lần chạy
# liền vẫn lệch (nguồn không tất định khác) -> thêm EXTRA_REFS rồi áp luật dự phòng của ô C5c. Lượt thật (C6) KHÔNG đổi kernel.
EXTRA_REFS = ('C', 'D')
SDP_FLAGS = ('flash', 'mem_efficient', 'math', 'cudnn')     # cờ toàn cục torch.backends.cuda.<cờ>_sdp_enabled / enable_<cờ>_sdp (có thì dùng)


def run_tag(name, encoder):
    return config.dataset(name).run_tag(encoder)


def notebook_name(name, encoder):
    return f'93v_{name}_{encoder}'


def _label(tag):
    m = re.fullmatch(r'r(93v_[a-z0-9_]+)', tag)
    assert m, f'RUN_TAG {tag!r} khong co dang r93v_<...> -> DUNG'
    return m.group(1)


def _packages():
    import external_benchmarks
    import vindr_bench
    import crossdata_bench
    return {m.__name__: dict(phien_ban=m.__version__, sha=getattr(m, 'GOI_SHA', None)) for m in (crossdata_bench, vindr_bench, external_benchmarks)}


def _resume_state(drive, log, my_token=None, force=''):
    '''= ver2._resume_state, nhãn tệp theo r93v_<...>. 0.5.0: chặn hai phiên (sessions.lock_check) TRƯỚC khi khôi phục; in điểm lưu giữa giai đoạn. 0.5.1: force
       = MO_KHOA_PHIEN (mã mở khoá một lần).'''
    import torch
    tag = drive['RUN_TAG']
    path = drive['tim_moi_noi'](drive['TEN_RESUME'])
    if not path:
        return dict(run=tag, nhan=_label(tag), t=time.strftime('%Y%m%d_%H%M%S'), giai_doan={}, xong=[])
    st = torch.load(path, map_location='cpu', weights_only=False)
    assert st.get('run') == tag, f"tep resume cua luot KHAC ({st.get('run')} != {tag}) -> DOI RUN_TAG o C0, KHONG xoa gi"
    if my_token is not None:
        age = SS.resume_age_min(drive['kiem_ke'], (config.FOLDER, tag), drive['TEN_RESUME'])
        ok, msg = SS.lock_check(st, my_token, age, force=force)
        log(f'phien: {msg}')
        assert ok, msg
    drive['khoi_phuc_cho'](st.get('cho'))
    gd = st.get('giai_doan') or {}
    dang = gd.get('dang')
    log(f"RESUME: luot {st['t']} — " + (f"da XONG ({st['xong']})" if st.get('xong') else f"da xong giai doan {sorted(k for k in gd if k in ('s1', 's2')) or '-'}"
        + (f" + diem luu GIUA giai doan {dang['giai_doan']} o buoc {dang['buoc']:,} ({dang['t']})" if dang else ''))
        + f" | da chay {len(gd.get('phien') or [])} phien -> khoi phuc, KHONG chay lai phan da xong")
    return dict(run=tag, nhan=_label(tag), t=st['t'], giai_doan=gd, xong=list(st.get('xong') or []))


def setup(name, encoder, ns, log=print):
    '''C2. -> CTX. Bộ thư viện / GPU khác bộ đã chốt của Ver2 -> dừng (như 89).'''
    from vindr_bench.encoders import BUILT, SPECS
    from vindr_bench.notebook import ver2
    from vindr_bench.protocols.chexfound_env import environment as E
    from vindr_bench.protocols.ours import config as C
    from vindr_bench.protocols.ours import gates as G
    from vindr_bench.protocols.ours import runtime as RT
    sp = config.dataset(name)
    rows = config.RUN_ORDER + config.EXTRA_ROWS                     # 0.6.0: + hàng thêm (ASG, PLAN_93 §13) — họ phép so riêng ở 93t
    assert encoder in rows and encoder in BUILT, f'{encoder}: khong thuoc 9 hang cua 93 {config.RUN_ORDER} / hang them {config.EXTRA_ROWS} -> DUNG'
    drive = ver2.drive_api(ns)
    assert drive['RUN_TAG'] == run_tag(name, encoder), f"RUN_TAG {drive['RUN_TAG']!r} != {run_tag(name, encoder)!r} -> DUNG"
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
    assert lib_ok, (f'bo thu vien KHAC bo da chot cua Ver2 {G.VER2_LIBS}: {lib_diff} (co / ghim) -> DUNG: KHONG chay tiep, gui Claude dong "moi truong Ver2" o tren')
    assert device == 'cuda' and G.gpu_gate(gpu), (f"GPU {gpu['ten']!r} khong phai G4 ({G.GPU_MARK}) -> Runtime > Change runtime type > G4, roi chay lai")
    tok = SS.token(ns, drive['GHI_LOC'])
    st = _resume_state(drive, log, my_token=tok, force=ns.get('MO_KHOA_PHIEN') or '')
    ctx = dict(encoder=encoder, bo=name, kho=config.STORE_KEY, spec=spec, ds=sp, drive=drive, device=device, amp=amp, anh_dir=f'/content/x518_{name}', tt=st,
               moi_truong=dict(thu_vien=v, gpu=gpu, pip=mt), goi=_packages(), ns=ns, token=tok)
    main = R.main_run(name)
    log(f"encoder {spec.title}: {spec.arch}, anh {spec.image_size}, chuan hoa goc {spec.mean} / {spec.std}, vector toan anh {spec.pool} | bo {sp.title}, kho 518 "
        f"dem vuong KHONG can bang | dau: {[C.TITLES[k] for k in C.HEAD_KINDS]} | luoi {len(C.LR_GRID)} lr {C.LR_GRID[0]:g} .. {C.LR_GRID[-1]:g}, "
        f"{main.epochs_phase1} + {main.epochs_phase2} epoch, {main.n_seeds} hat | goi " + ', '.join(f"{k} {g['phien_ban']} @ {(g['sha'] or '?')[:12]}" for k, g in ctx['goi'].items()))
    return ctx


def _quota_copier(ns, log):
    '''Bản sao phía máy chủ của một phần kho khi quota tải của tệp hết (memory: "bản sao có quota mới") — vào thư mục kết quả của lượt, bỏ vào thùng rác sau khi tải.'''
    parent = (ns or {}).get('DRV_ID')
    if not parent:
        return None
    from ..store import listing as L

    def copy(f):
        api = L.default_api()
        r = api.files().copy(fileId=f['id'], body={'name': f"_ban_sao_tam_{time.strftime('%H%M%S')}.tar", 'parents': [parent]}, fields='id,md5Checksum,size',
                             supportsAllDrives=True).execute()

        def cleanup():
            try:
                L.default_api().files().update(fileId=r['id'], body={'trashed': True}, supportsAllDrives=True).execute()
            except Exception as e:
                log(f'  ! bo ban sao tam vao thung rac khong duoc ({type(e).__name__}) -> xoa tay tep _ban_sao_tam_* trong thu muc ket qua')
        return dict(id=r['id'], md5=r.get('md5Checksum') or f['md5'], size=int(r.get('size') or f['size'])), cleanup
    return copy


def data(ctx, log=print):
    '''C3. Kho của bộ + cổng (base.load_store; kho chia phần: tải song song + md5 từng phần). -> StoreData.'''
    d = ctx['drive']
    sd = base.load_store(d['kiem_ke'], d['tai'], d['xung_dot'], ctx['anh_dir'], ctx['bo'], log=log, on_quota=_quota_copier(ctx.get('ns'), log))
    fp = sd.fingerprint()
    ctx.update(vindr=sd, van_tay=fp)
    log(sd.summary())
    log('van tay md5 khoa / nhan: ' + ' | '.join(f"{s} {fp[s]['md5_khoa'][:8]} / {fp[s]['md5_nhan'][:8]}" for s in config.SPLITS) + ' | TRUNG ghim (93k)')
    return sd


def gate(ctx, log=print):
    '''C4 = vindr_bench.notebook.ver2.gate NGUYÊN VĂN (đọc ctx: drive, spec, vindr, encoder, device, amp).'''
    from vindr_bench.notebook import ver2
    return ver2.gate(ctx, log=log)


def _smoke_log(log):
    return lambda s='': log('  ' + s.strip()) if s.strip().startswith(('Ver2 ', 'xong:')) else None


def curves_of(result):
    '''Đường học KHÔNG phụ thuộc lr được chọn: mọi đầu dò lr + thêm hạt (auc / ap / auprc / auc21 / loss theo epoch) + loss pha 2.'''
    out = {f'do_lr|{k}': v for k, v in result['do_lr'].items()}
    out.update({f'them_seed|{k}': v for k, v in result['them_seed'].items()})
    out.update({f'pha2|{k}': dict(loss=v['loss']) for k, v in result['pha2'].items()})
    return out


def curve_gap(a, b, keys=None):
    '''Lệch TƯƠNG ĐỐI tối đa |x - y| / max(1, |x|) giữa hai bộ đường học trên các đầu `keys` (mặc định: mọi đầu chung); NaN phải cùng chỗ; khác số epoch / thiếu
       đầu -> inf.'''
    keys = sorted(set(a) & set(b)) if keys is None else list(keys)
    m = 0.0
    for k in keys:
        if k not in a or k not in b or set(a[k]) != set(b[k]):
            return float('inf')
        for f in a[k]:
            x, y = np.asarray(a[k][f], np.float64), np.asarray(b[k][f], np.float64)
            if x.shape != y.shape or not np.array_equal(np.isnan(x), np.isnan(y)):
                return float('inf')
            ok = ~np.isnan(x)
            if ok.any():
                m = max(m, float((np.abs(x[ok] - y[ok]) / np.maximum(1.0, np.abs(x[ok]))).max()))
    return m


def lr_choice(result):
    '''lr được chọn {loại đầu: {tiêu chí: lr}} của một lần chạy (result['chon'] của pipeline 89).'''
    from vindr_bench.protocols.ours import config as C
    return {k: {c: float(v[c]['lr']) for c in C.CRITERIA} for k, v in result['chon'].items()}


def chosen_keys(result):
    '''(0.5.2) Khoá trong curves_of(result) của CÁC ĐẦU ĐƯỢC CHỌN: đầu dò lr hạt 0 ở lr chọn (mỗi loại, mỗi tiêu chí) + mọi đầu thêm hạt + mọi đầu pha 2.
       Tên đầu dò lr = '<hạt>|<loại>|<lr:g>' (training.HeadRun.name); không tìm được đúng MỘT đầu -> dừng.'''
    keys = {f'them_seed|{k}' for k in result['them_seed']} | {f'pha2|{k}' for k in result['pha2']}
    for kind, c in lr_choice(result).items():
        for lr in c.values():
            hit = [k for k in result['do_lr'] if k.split('|')[-2] == kind and float(k.split('|')[-1]) == lr]
            assert len(hit) == 1, f'dau do lr cua {kind} lr {lr:g}: {len(hit)} dau (can dung 1) -> DUNG'
            keys.add(f'do_lr|{hit[0]}')
    return sorted(keys)


def compare_runs(res_x, arr_x, res_y, arr_y):
    '''(0.5.2) Lệch giữa hai lần chạy cùng cấu hình -> dict(chon_khac: lr chọn / bộ mảng khác, du_doan: |p| tối đa trên mọi mảng val_* / test_*,
       dau_chon: đường học tương đối tối đa của các đầu được chọn (chosen_keys), do_lr: đường học tương đối tối đa của MỌI đầu dò lr, so_mang).
       Khác lựa chọn -> du_doan = dau_chon = inf (không so được).'''
    kx = sorted(k for k in arr_x if k.startswith(('val_', 'test_')))
    ky = sorted(k for k in arr_y if k.startswith(('val_', 'test_')))
    cx, cy = curves_of(res_x), curves_of(res_y)
    probe = [k for k in cx if k.startswith('do_lr|')]
    out = dict(chon_khac=kx != ky or lr_choice(res_x) != lr_choice(res_y), do_lr=curve_gap(cx, cy, probe) if set(probe) <= set(cy) else float('inf'))
    if out['chon_khac']:
        out.update(du_doan=float('inf'), dau_chon=float('inf'), so_mang=0)
    else:
        keys = chosen_keys(res_x)
        out.update(du_doan=max(float(np.abs(np.asarray(arr_x[k], np.float64) - np.asarray(arr_y[k], np.float64)).max()) for k in kx),
                   dau_chon=curve_gap(cx, cy, keys) if set(keys) <= set(cy) else float('inf'), so_mang=len(kx))
    return out


@contextlib.contextmanager
def heads_math(backbone):
    '''(0.6.0) Trong khối: attention SDPA chỉ dùng kernel MATH (tất định) — trừ lúc gọi backbone.features: khi đó bật lại ĐÚNG các kernel đang bật lúc vào
       khối (backbone chạy như lượt thật). Ra khối (kể cả khi lỗi): trả cờ, trả backbone.features. Chỉ dùng cho các lần chạy so sánh của C5; C6 không đổi.
       -> cờ lúc vào khối.'''
    import torch
    B = torch.backends.cuda
    flags = [f for f in SDP_FLAGS if hasattr(B, f'{f}_sdp_enabled') and hasattr(B, f'enable_{f}_sdp')]
    assert 'math' in flags, 'torch khong co co SDPA math -> DUNG'
    saved = {f: bool(getattr(B, f'{f}_sdp_enabled')()) for f in flags}
    only_math = {f: f == 'math' for f in flags}

    def apply(st):
        for f, v in st.items():
            getattr(B, f'enable_{f}_sdp')(v)
    had = 'features' in vars(backbone)
    orig = backbone.features

    def features(*a, **kw):
        apply(saved)
        try:
            return orig(*a, **kw)
        finally:
            apply(only_math)
    backbone.features = features
    apply(only_math)
    try:
        yield dict(saved)
    finally:
        apply(saved)
        if had:
            backbone.features = orig
        else:
            del backbone.features


def resume_verdict(hits, n_stops, refs, got):
    '''(0.6.0) Luật đạt "chạy tiếp = chạy liền" của C5 — mọi lần chạy dưới heads_math. refs = {tên: (result, arrays)}: các lần chạy liền giống hệt nhau
       (A, B; dự phòng thêm EXTRA_REFS); got = (result, arrays) của lần chạy tiếp (có ngắt). Mọi chế độ: đủ số lần ngắt.
       - 'tung_bit': A, B trùng TỪNG BIT (cùng lr chọn; dự đoán, đường học đầu được chọn, đường học MỌI đầu dò lr) -> chạy tiếp phải trùng A TỪNG BIT.
       - 'nhieu': A, B lệch -> cần đủ 2 + len(EXTRA_REFS) lần chạy liền (thiếu -> can_them = True, chưa phán) -> luật của ô C5c (26/11): chạy tiếp CÙNG lr
         chọn với ít nhất một lần chạy liền, và lệch tới lần chạy liền gần nhất (cùng lr chọn) <= lệch lớn nhất giữa các cặp chạy liền (cùng lr chọn) — cho
         cả dự đoán lẫn đường học đầu được chọn.'''
    names = sorted(refs)
    assert len(names) >= 2, 'can it nhat HAI lan chay lien -> DUNG'
    pairs = {(x, y): compare_runs(*refs[x], *refs[y]) for i, x in enumerate(names) for y in names[i + 1:]}
    so = {n: compare_runs(*refs[n], *got) for n in names}
    a, b = names[0], names[1]
    nen = pairs[(a, b)]
    out = dict(so_lan_ngat=hits, nen=nen, so=so, cap={f'{x}-{y}': v for (x, y), v in pairs.items()}, so_lan_lien=len(names))
    if not nen['chon_khac'] and nen['du_doan'] == 0 and nen['dau_chon'] == 0 and nen['do_lr'] == 0:
        s = so[a]
        dat = hits == n_stops and not s['chon_khac'] and s['du_doan'] == 0 and s['dau_chon'] == 0 and s['do_lr'] == 0
        return dict(out, che_do='tung_bit', dat=dat, can_them=False, khop=[a] if dat else [], nguong=dict(du_doan=0.0, dau_chon=0.0, do_lr=0.0))
    if len(names) < 2 + len(EXTRA_REFS):
        return dict(out, che_do='nhieu', dat=False, can_them=True, khop=[], nguong=None)
    lim = {}
    for f in ('du_doan', 'dau_chon'):
        v = [p[f] for p in pairs.values() if not p['chon_khac'] and np.isfinite(p[f])]
        lim[f] = max(v) if v else None
    near = {f: min((so[n][f] for n in names if not so[n]['chon_khac'] and np.isfinite(so[n][f])), default=float('inf')) for f in ('du_doan', 'dau_chon')}
    khop = [n for n in names if not so[n]['chon_khac']]
    dat = hits == n_stops and bool(khop) and all(lim[f] is not None and near[f] <= lim[f] for f in lim)
    return dict(out, che_do='nhieu', dat=dat, can_them=False, khop=khop, nguong=lim, gan_nhat=near)


def resume_equal(sd, backbone, name, run_cfg, subset, ref_arrays, stops=RESUME_STOPS, log=None, protocol=None, out=None):
    '''(0.5.0) Chạy lại trên tập con với các lần NGẮT `stops` (Checkpointer.stop_after); giữa hai lần chạy, trạng thái đi qua torch.save -> torch.load (như
       phiên mới: đầu, optimizer, kế hoạch dựng lại). -> (lệch tối đa |p| so với ref_arrays trên mọi mảng val_* / test_* — inf nếu bộ mảng KHÁC, số lần
       ngắt đã xảy ra, số mảng so). out (dict): + 'duong' (đường học của lần chạy tiếp), 'chon_khac' (bộ mảng khác), do_lr / them_seed / pha2, (0.5.2)
       'ket_qua' + 'mang' (result + arrays của lần chạy tiếp, cho resume_verdict). protocol: như R.prepare (None = mặc định 89; bộ kiểm máy dùng lưới nhỏ).'''
    import io
    import torch
    blobs = []

    def save(state):
        b = io.BytesIO()
        torch.save(state, b)
        blobs.append(b.getvalue())
        return True
    quiet = (lambda s='': None) if log is None else log
    state, hits, arrays, result = {}, 0, None, None
    for stop in tuple(stops) + (None,):
        pl = R.prepare(sd, backbone, name, main=run_cfg, subset=subset, protocol=protocol)
        ck = RS.Checkpointer(state, save, every_s=1e12, log=quiet, stop_after=stop)
        try:
            with ck.active():
                result, arrays = R.run(pl, state=state, save=ck.save_stage, run_tag='chay_thu_chay_tiep', log=quiet)
        except RS.Interrupted:
            hits += 1
            state = torch.load(io.BytesIO(blobs[-1]), map_location='cpu', weights_only=False)
            continue
    assert arrays is not None and result is not None, 'chay tiep: lan chay cuoi khong xong -> DUNG'
    names = sorted(k for k in ref_arrays if k.startswith(('val_', 'test_')))
    got = sorted(k for k in arrays if k.startswith(('val_', 'test_')))
    diff_arms = got != names                                    # lr được chọn khác (hoà gần + nhiễu GPU, hoặc chạy tiếp sai) -> so bằng đường học
    lech = float('inf') if diff_arms else max(float(np.abs(np.asarray(arrays[k], np.float64) - np.asarray(ref_arrays[k], np.float64)).max()) for k in names)
    if out is not None:
        out.update(duong=curves_of(result), chon_khac=diff_arms, do_lr=result['do_lr'], them_seed=result['them_seed'],
                   pha2={k: v['loss'] for k, v in result['pha2'].items()}, ket_qua=result, mang=arrays)
    return lech, hits, len(names)


def _gaps(s):
    return ('lr chon KHAC' if s['chon_khac'] else f"du doan {s['du_doan']:.1e} ({s['so_mang']} mang), dau duoc chon {s['dau_chon']:.1e}") + f", moi dau do lr {s['do_lr']:.1e}"


def _resume_line(ver, minutes):
    '''Dòng in của "chạy tiếp = chạy liền" (0.6.0): chế độ, so với lần chạy liền, kết luận.'''
    head = (f"  chay tiep = chay lien ({RESUME_RUN['epochs_phase1']} + {RESUME_RUN['epochs_phase2']} epoch; attention MATH o dau doc, backbone giu kernel cua "
            f"luot that): {ver['so_lan_ngat']} lan ngat ({', '.join(f'{g} buoc {b}' for g, b in RESUME_STOPS)}; nap lai qua torch.save / load nhu phien moi) | ")
    if ver['che_do'] == 'tung_bit':
        body = f"hai lan chay lien A, B TRUNG TUNG BIT -> chay tiep vs A (phai = 0): {_gaps(ver['so'][sorted(ver['so'])[0]])}"
    else:
        lim, near = ver.get('nguong') or {}, ver.get('gan_nhat') or {}
        body = (f"{ver['so_lan_lien']} lan chay lien, A vs B KHONG trung tung bit ({_gaps(ver['nen'])}) -> luat du phong (o C5c): chay tiep vs lan chay lien "
                f"gan nhat cung lr chon: du doan {near.get('du_doan', float('nan')):.1e}, dau duoc chon {near.get('dau_chon', float('nan')):.1e} | lech lon "
                f"nhat giua cac cap chay lien: " + (f"du doan {lim['du_doan']:.1e}, dau duoc chon {lim['dau_chon']:.1e}"
                                                     if lim and all(v is not None for v in lim.values()) else 'KHONG co cap so duoc'))
    return head + body + f" | {'DAT' if ver['dat'] else 'TRUOT'} | {minutes:.1f} phut"


def smoke(ctx, log=print):
    '''C5. Chạy thử trên tập con ở lô 256 với đủ đầu của lượt thật; trượt -> dừng trước lượt đầy đủ.'''
    import torch
    from vindr_bench.protocols.ours import config as C
    from vindr_bench.protocols.ours import gates as G
    st, sd = ctx['tt'], ctx['vindr']
    if st['giai_doan'] or st['xong']:
        log('RESUME: bo qua chay thu (da dat o phien truoc)')
        ctx['thu'] = dict(bo_qua_vi_resume=True)
        return ctx['thu']
    t0 = time.time()
    sub = {s: min(n, len(sd.indices(s))) for s, n in SMOKE_SUBSET.items()}
    run = C.MainRun(epochs_phase1=1, epochs_phase2=1, n_seeds=2)              # = vindr_bench.notebook.ver2.SMOKE_RUN
    pl = R.prepare(sd, ctx['backbone'], ctx['bo'], main=run, subset=sub)
    state = {}
    result, arrays = R.run(pl, state=state, save=None, run_tag='chay_thu', log=_smoke_log(log))
    peak = torch.cuda.max_memory_allocated() if ctx['device'] == 'cuda' else 0
    total = torch.cuda.get_device_properties(0).total_memory if ctx['device'] == 'cuda' else 0
    lech = R.roundtrip_check(pl, state, arrays)
    tests = [k for k in arrays if k.startswith('test_')]
    t1 = time.time()                       # 0.5.0: chạy tiếp (ngắt giữa mỗi giai đoạn) = chạy liền; 0.6.0: mọi lần chạy so sánh dưới heads_math -> từng bit
    run_ct = C.MainRun(**RESUME_RUN)

    def lien(n):
        return R.run(R.prepare(sd, ctx['backbone'], ctx['bo'], main=run_ct, subset=sub), state={}, save=None, run_tag=f'chay_thu_lien_{n}', log=lambda s='': None)
    with heads_math(ctx['backbone']):
        refs = {n: lien(n) for n in ('A', 'B')}
        info_ct = {}
        _, hits, _ = resume_equal(sd, ctx['backbone'], ctx['bo'], run_ct, sub, refs['A'][1], out=info_ct)
        got = (info_ct['ket_qua'], info_ct['mang'])
        ver = resume_verdict(hits, len(RESUME_STOPS), refs, got)
        if ver['can_them']:                 # A, B lệch dù đầu đọc MATH -> luật dự phòng (thêm hai lần chạy liền)
            refs.update({n: lien(n) for n in EXTRA_REFS})
            ver = resume_verdict(hits, len(RESUME_STOPS), refs, got)
    del refs, info_ct, got
    ck = dict(so_buoc=result['kiem']['so_buoc_dung'], huu_han=result['kiem']['du_doan_huu_han'],
              bo_nho=(ctx['device'] != 'cuda' or G.memory_ok(peak, total)), phuc_hoi=lech <= ROUNDTRIP_TOL,
              du_dau=len(tests) >= len(C.HEAD_KINDS) * run.n_seeds and all(arrays[k].shape == (sub['test'], len(sd.label_names)) for k in tests),
              chay_tiep=ver['dat'])
    thu = dict(dat=all(ck.values()), kiem=ck, gpu_dinh_gib=peak / 2 ** 30, gpu_tong_gib=total / 2 ** 30, lech_phuc_hoi=lech, so_dau_test=len(tests),
               chon=lr_choice(result), phut=(time.time() - t0) / 60, tap_con=sub,
               chay_tiep=dict(ver, diem_ngat=[list(x) for x in RESUME_STOPS], phut=(time.time() - t1) / 60))
    log(f"chay thu {'DAT' if thu['dat'] else 'TRUOT'} ({thu['phut']:.1f} phut; {len(C.HEAD_KINDS)} loai dau x {len(C.LR_GRID)} lr o lo {C.Protocol().batch_size}, "
        f"tap con {sub}, so do VO NGHIA): so buoc {'DAT' if ck['so_buoc'] else 'TRUOT'} | du doan huu han {'DAT' if ck['huu_han'] else 'TRUOT'} | "
        f"{len(tests)} bo du doan test dung hinh [{sub['test']}, {len(sd.label_names)}] {'DAT' if ck['du_dau'] else 'TRUOT'} | GPU dinh {thu['gpu_dinh_gib']:.1f} / "
        f"{thu['gpu_tong_gib']:.1f} GiB (nguong {G.MEM_LIMIT:.0%}: {'DAT' if ck['bo_nho'] else 'TRUOT'}) | phuc hoi tu trang thai resume: lech du doan val "
        f"{lech:.1e} (<= {ROUNDTRIP_TOL:g}: {'DAT' if ck['phuc_hoi'] else 'TRUOT'})")
    log(_resume_line(ver, thu['chay_tiep']['phut']))
    del state, arrays, result, pl
    if ctx['device'] == 'cuda': torch.cuda.empty_cache()
    ctx['thu'] = thu
    assert thu['dat'], f"chay thu TRUOT: {[k for k, v in ck.items() if not v]} -> DUNG truoc luot day du, gui Claude output C5"
    return thu


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


def _files(st):
    return f"luot_{st['nhan']}_{st['t']}.json", f"du_doan_{st['nhan']}_{st['t']}.npz"


def _saver(ctx, log):
    '''save(giai_doan, xong=()) -> True / False: ghi tệp resume (luu_resume của khối chuẩn, kèm tệp đang chờ) rồi xoá phiên bản cũ của tệp trên Drive (0.5.0).'''
    import torch
    d, st, ns = ctx['drive'], ctx['tt'], ctx.get('ns') or {}
    api = []

    def save(gd, xong=()):
        ok = d['luu_resume'](lambda q: torch.save(dict(run=st['run'], t=st['t'], giai_doan=gd, xong=list(xong), cho=d['cho_nho']()), q))
        if ok and ns.get('_RESUME_ID'):
            if not api:
                from ..store import listing as L
                try:
                    api.append(L.default_api())
                except Exception:
                    api.append(None)
            SS.prune_revisions(api[0], ns.get('_RESUME_ID'), log)
        return ok
    return save


def train(ctx, log=print, every_min=RS.EVERY_MIN):
    '''C6. Lượt thật có resume theo giai đoạn (tệp resume ghi đè sau giai đoạn (1), (2) và khi xong) + 0.5.0: lưu GIỮA giai đoạn mỗi `every_min` phút
       (protocol/resume.Checkpointer) — phiên đứt thì mở lại notebook (tài khoản nào cũng được, Editor của thư mục) -> Run all -> học tiếp từ điểm lưu. -> kết quả.'''
    from vindr_bench.protocols.ours import config as C
    from vindr_bench.protocols.ours import gates as G
    d, st, sd = ctx['drive'], ctx['tt'], ctx['vindr']
    if st['xong']:
        with open(os.path.join(d['GHI_LOC'], st['xong'][0]), encoding='utf-8') as f:
            ctx['kq'] = json.load(f)
        log(f"RESUME: luot da xong -> doc lai {st['xong'][0]} (khong hoc lai)")
        return ctx['kq']
    save = _saver(ctx, log)
    pl = R.prepare(sd, ctx['backbone'], ctx['bo'])
    ck = RS.Checkpointer(st['giai_doan'], lambda gd: save(gd), every_s=float(every_min) * 60, log=log, plan_steps=dict(s1=pl.n1, s2=pl.n1, s3=pl.n2),
                         token=ctx.get('token'))
    log(f"luu giua giai doan moi ~{every_min:g} phut (diem luu + tep dang cho -> tep resume tren Drive; phien ban cu xoa) | buoc: do lr {pl.n1:,}, them hat {pl.n1:,}, "
        f"pha 2 {pl.n2:,} | phien dut -> mo lai notebook (Editor nao cung duoc) -> Run all -> hoc tiep tu diem luu gan nhat")
    with ck.active():
        result, arrays = R.run(pl, state=st['giai_doan'], save=ck.save_stage, run_tag=st['run'], log=log)
    arrays.update(nhom_val=sd.groups[pl.va].astype(str), nhom_test=sd.groups[pl.te].astype(str))
    result.update(t=st['t'], bo=ctx['bo'], bo_ten=ctx['ds'].title, don_vi_cum=ctx['ds'].unit, tiep_xuc=ctx['ds'].exposure, encoder=dataclasses.asdict(ctx['spec']),
                  moi_truong=ctx['moi_truong'], goi=ctx['goi'], cong=ctx['cong'], van_tay=ctx['van_tay'], chay_thu=ctx.get('thu'), thu_vien_ghim=G.VER2_LIBS,
                  phien_chay=list(st['giai_doan'].get('phien') or []), luu_giua_giai_doan_phien_cuoi=ck.saves,
                  mot_nhan='run_head boc: logit [B] -> [B, 1] (Linear / GLoRI squeeze(-1) khi C = 1); C >= 2 khong doi',
                  ghi_chu='PLAN_93 (= 89 Ver2 tren bo ngoai): du doan test MOT lan sau pha 2; so do test cham o C7 va 93t.')
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
    v = np.asarray(v, dtype=np.float64)
    return f'{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.4f}'


def score(ctx):
    '''Điểm các mảng dự đoán của lượt (không bootstrap). -> {nhánh: dict}. Số chính = TB metric của các hạt; ensemble (TB xác suất) báo kèm.'''
    from vindr_bench.protocols.ours import config as C
    from vindr_bench.protocols.ours import pipeline as PL
    d, st, sd = ctx['drive'], ctx['tt'], ctx['vindr']
    with np.load(os.path.join(d['GHI_LOC'], st['xong'][1])) as z:
        arr = {k: z[k] for k in z.files}
    y_test, y_val = arr['y_test'], arr['y_val']
    assert np.array_equal(y_test, sd.labels[sd.indices('test')]) and np.array_equal(y_val, sd.labels[sd.indices('val')]), 'nhan trong tep du doan KHAC kho -> DUNG'
    half = S.halves(arr['nhom_test'])
    out = {}
    for k in ctx['kq']['loai_dau']:
        ch = ctx['kq']['chon'][k]
        for crit in C.CRITERIA:
            if crit != C.RULE and ch['ap_trung_luat']:
                continue
            tag = PL.arm_tag(k, crit)
            seeds = sorted(int(n.rsplit('_s', 1)[1]) for n in arr if n.startswith(f'test_{tag}_s'))
            pt = [arr[f'test_{tag}_s{s}'] for s in seeds]
            pv = [arr[f'val_{tag}_s{s}'] for s in seeds]
            per = [S.point(y_test, q) for q in pt]
            out[tag] = dict(loai=k, tieu_chi=crit, lr=ch[crit]['lr'], bien_luoi=ch[crit]['bien_luoi'], so_hat=len(seeds),
                            tung_hat={m: [x[m] for x in per] for m in ('ap', 'auprc_thang', 'auroc')},
                            ensemble=S.point(y_test, np.mean(pt, 0)),
                            hai_nua_ap=[[S.point(y_test[half == h], q[half == h])['ap'] for q in pt] for h in (0, 1)],
                            val={m: [x[m] for x in (S.point(y_val, q) for q in pv)] for m in ('ap', 'auprc_thang', 'auroc')})
    return out, dict(n_test=int(len(y_test)), so_duong_test=[int(v) for v in y_test.sum(0)], n_val=int(len(y_val)),
                     nua=[int((half == h).sum()) for h in (0, 1)], cum_nua=[len(set(arr['nhom_test'][half == h].tolist())) for h in (0, 1)])


def report(ctx, log=print):
    '''C7 (CHỈ MÔ TẢ). Lỗi ở đây KHÔNG làm mất C6: in lỗi, trả None — C8 vẫn đồng bộ.'''
    try:
        from vindr_bench.protocols.ours import config as C
        from vindr_bench.protocols.ours import pipeline as PL
        d, st, sp, kq, ds = ctx['drive'], ctx['tt'], ctx['spec'], ctx['kq'], ctx['ds']
        sc, info = score(ctx)
        log(f"\n== {sp.title} tren {ds.title} — giao thuc 89 Ver2 (G4), kho 518 dem vuong KHONG can bang, anh {sp.image_size}, chuan hoa goc, backbone "
            f"{kq['amp_backbone'].replace('torch.', '')}, dau doc eval() khi cham; {kq['so_seed']} hat: so = TB ± SD mau giua cac hat (so chinh), 'ens' = "
            f"ensemble TB xac suat (bao kem); AP = AP bac thang, AUPRC = hinh thang; KHONG co khoang tin cay o day (93t) ==")
        log(f"--- VAL cuoi pha 1 ({info['n_val']:,} anh; chi de theo doi) ---")
        for tag, s in sc.items():
            name = C.TITLES[s['loai']] if s['tieu_chi'] == C.RULE else ''
            log(f"{name:<10} {PL.CRIT_TEXT[s['tieu_chi']]} lr {s['lr']:<7g} AUROC {_pm(s['val']['auroc'])} | AP {_pm(s['val']['ap'])} | AUPRC thang "
                f"{_pm(s['val']['auprc_thang'])}" + (' | lr O BIEN LUOI' if s['bien_luoi'] else ''))
        log(f"--- TEST ({info['n_test']:,} anh, {info['so_duong_test']} duong; hai nua theo {'nguoi' if ds.unit == 'nguoi' else 'anh'}: {info['nua']} anh / "
            f"{info['cum_nua']} cum; AP = so cua luat) ---")
        for tag, s in sc.items():
            name = C.TITLES[s['loai']] if s['tieu_chi'] == C.RULE else ''
            th, en = s['tung_hat'], s['ensemble']
            log(f"{name:<10} {PL.CRIT_TEXT[s['tieu_chi']]} lr {s['lr']:<7g} AP {_pm(th['ap'])} (ens {en['ap']:.4f}) | AUPRC thang {_pm(th['auprc_thang'])} "
                f"(ens {en['auprc_thang']:.4f}) | AUROC {_pm(th['auroc'])} (ens {en['auroc']:.4f}) | AP hai nua TB {np.mean(s['hai_nua_ap'][0]):.4f} / "
                f"{np.mean(s['hai_nua_ap'][1]):.4f}")
        khac = [C.TITLES[k] for k, c in kq['chon'].items() if not c['ap_trung_luat']]
        log("nhanh 'theo AP' (mo ta, khong vao luat): " + (f"lr KHAC luat o {khac}" if khac else 'trung lr cua luat o moi loai dau')
            + f" | tiep xuc tien huan luyen cua bo: {ds.exposure}")
        name = f"bang_{st['nhan']}_{st['t']}.json"
        obj = _jsonable(dict(run=d['RUN_TAG'], t=st['t'], goi=ctx['goi'], phien_ban_ver='Ver2', bo=ctx['bo'], encoder=dataclasses.asdict(sp), kho=ctx['kho'],
                             diem=sc, thong_tin=info, chon=kq['chon'], tep_luot=st['xong'],
                             ghi_chu='diem tren toan test, khong bootstrap; so chinh = TB cac hat; luat + khoang dong thoi: 93t'))
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
        + (' -> gui Claude output C2-C8; roi chay 93c (chep sang corntoun1505)' if du and co else ' | KHONG tat runtime; chay lai C8, roi bao Claude'))
    return dict(dong_bo=du, du_tep=co, bang=bool(bg and bg in on))
