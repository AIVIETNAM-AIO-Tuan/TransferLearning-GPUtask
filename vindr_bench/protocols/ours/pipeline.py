# -*- coding: utf-8 -*-
"""Điều phối Ver2 (PLAN_89 §5.1 = lượt chính của `86b`, PLAN_86 §4.4) cho MỘT encoder trên MỘT kho ảnh:

    (1) dò lr     4 loại đầu x 13 lr, chỉ số seed 0, E1 epoch trên train, chấm val mỗi epoch (đầu đọc eval());
                  lr của LUẬT = AUROC val lần chấm cuối lớn nhất (hoà -> lr nhỏ); lr "theo AP" = AP val lớn nhất (nhánh MÔ TẢ, config.CRITERIA)
    (2) thêm hạt  seed 1 .. n-1 ở lr đã chọn (lr theo AP khác lr của luật -> thêm cả các hạt của lr đó)
    (3) pha 2     mỗi đầu học TIẾP từ trọng số pha 1 của chính nó trên train + val, E2 epoch, optimizer / cosine / bước đếm mới, checkpoint cuối;
                  dự đoán test MỘT lần — không tính số đo test ở đây (chấm ở notebook/ver2.report và bảng gộp)

Backbone đông cứng chạy lại mỗi lô trên ảnh tăng cường; mọi đầu bước trên cùng đặc trưng; mỗi đầu có optimizer, lịch lr và dòng ngẫu nhiên riêng
(thêm / bớt đầu khác không đổi đầu này). train_parallel CHÉP NGUYÊN VĂN từ vindr_protocol/pipeline.py của 86 (tests/test_ver2.py so cây cú pháp).

Khôi phục giữa chừng: sau (1) và (2) trạng thái (đường val + trọng số các đầu cần cho giai đoạn sau) xuất thành dict lưu được bằng torch.save;
run(state=...) bỏ qua giai đoạn đã xong. Không ghi Drive: hàm trả kết quả, notebook ghi bằng khối Drive chuẩn."""
import datetime
import time
from dataclasses import dataclass

import numpy as np
import torch

from ...data import loaders as data
from ...data import vindr as V
from . import config, runtime, selection, training
from .config import CRITERIA, HEAD_KINDS, RULE, TITLES

CURVE_FIELDS = ('val_auroc', 'val_ap', 'val_auroc_main', 'val_auprc', 'train_loss')
CRIT_TEXT = {'auroc': 'LUAT   ', 'ap': 'theo AP'}


def _sync(device):
    if device == 'cuda': torch.cuda.synchronize()


def train_parallel(runs, backbone, dataset, indices, labels, protocol, total_steps, steps_per_epoch, on_epoch_end=None,
                   snapshot_at=None, snapshot_runs=None, num_workers=0):
    '''Mỗi bước MỘT lô của dòng mẫu vô hạn qua backbone, rồi mọi đầu còn học bước một lần trên cùng đặc trưng. Cuối mỗi epoch (steps_per_epoch
       bước): ghi loss TB epoch cho các đầu đã học trọn epoch, gọi on_epoch_end(epoch, các đầu đó).
       -> (thời gian {cho, dac, dau, anh}, ảnh chụp trọng số của snapshot_runs ({khoá: đầu}) ngay sau bước snapshot_at).'''
    dev = backbone.device
    y_all = torch.as_tensor(np.asarray(labels), dtype=torch.float32, device=dev)
    timing, snapshot = dict(cho=0.0, dac=0.0, dau=0.0, anh=0), {}
    loader = data.train_loader(dataset, indices, protocol.data_seed, 0, total_steps, protocol.batch_size, num_workers, pin_memory=(dev == 'cuda'))
    t_wait = time.time()
    for s, (x, j) in enumerate(loader):
        timing['cho'] += time.time() - t_wait
        t1 = time.time()
        x = x.to(dev, non_blocking=True)
        y = y_all[j.to(dev, non_blocking=True)]
        blocks, last = backbone.features(x)
        _sync(dev)
        t2 = time.time()
        timing['dac'] += t2 - t1
        for run in runs:
            if run.active: training.train_step(run, blocks, last, y, protocol.loss_divisor)
        del blocks, last
        _sync(dev)
        timing['dau'] += time.time() - t2
        timing['anh'] += len(j)
        if snapshot_at is not None and s + 1 == snapshot_at:
            snapshot = {k: {n: v.detach().clone() for n, v in r.model.state_dict().items()} for k, r in (snapshot_runs or {}).items()}
        if (s + 1) % steps_per_epoch == 0:
            epoch = (s + 1) // steps_per_epoch
            done = [r for r in runs if r.max_steps >= epoch * steps_per_epoch]
            for r in done: training.close_epoch(r, steps_per_epoch)
            if on_epoch_end is not None: on_epoch_end(epoch, done)
        t_wait = time.time()
    return timing, snapshot


# ---- kế hoạch của một lượt ----
@dataclass
class Plan:
    '''Mọi thứ dùng chung giữa các giai đoạn của một lượt (một encoder x một kho ảnh).'''
    vindr: object
    backbone: object
    protocol: config.Protocol
    main: config.MainRun
    curve: config.CurveDiagnostic
    kinds: tuple
    nw: int
    ds_tr: object
    ds_ev: object
    tr: np.ndarray
    va: np.ndarray
    te: np.ndarray
    trva: np.ndarray
    spe1: int
    spe2: int
    n1: int
    n2: int
    n_cls: int
    dim: int
    y_val: np.ndarray
    main_idx: list
    subset: object = None


def prepare(vindr, backbone, protocol=None, main=None, curve=None, kinds=HEAD_KINDS, num_workers=None, subset=None):
    '''-> Plan. subset = None (lượt thật) | dict(train=, val=, test=): chỉ lấy bấy nhiêu ảnh ĐẦU của mỗi tách (lượt chạy thử, số đo vô nghĩa).'''
    protocol, main, curve = protocol or config.Protocol(), main or config.MainRun(), curve or config.CurveDiagnostic()
    assert kinds and set(kinds) <= set(HEAD_KINDS) and len(set(kinds)) == len(kinds), kinds
    nw = runtime.default_num_workers() if num_workers is None else num_workers
    runtime.allow_tf32_heads()
    tr, va, te = vindr.indices('train'), vindr.indices('val'), vindr.indices('test')
    if subset:
        tr, va, te = tr[:int(subset['train'])], va[:int(subset['val'])], te[:int(subset['test'])]
    assert len(tr) and len(va) and len(te) and not (set(tr.tolist()) & set(va.tolist()) or set(te.tolist()) & set(np.r_[tr, va].tolist())), (
        'tach train / val / test rong hoac chong nhau -> DUNG')
    assert set(np.unique(vindr.labels[np.r_[tr, va]]).tolist()) <= {0.0, 1.0}, 'nhan train / val khong nhi phan -> DUNG'
    tf_train, tf_eval = data.build_transforms(backbone.spec)
    ds_tr = data.VinDrImages(vindr.keys, vindr.image_dir, tf_train, seed=protocol.data_seed)
    ds_ev = data.VinDrImages(vindr.keys, vindr.image_dir, tf_eval)
    trva = np.sort(np.r_[tr, va])
    bs = protocol.batch_size
    spe1, spe2 = config.steps_per_epoch(len(tr), bs), config.steps_per_epoch(len(trva), bs)
    return Plan(vindr=vindr, backbone=backbone, protocol=protocol, main=main, curve=curve, kinds=tuple(kinds), nw=nw, ds_tr=ds_tr, ds_ev=ds_ev,
                tr=tr, va=va, te=te, trva=trva, spe1=spe1, spe2=spe2, n1=main.epochs_phase1 * spe1, n2=main.epochs_phase2 * spe2,
                n_cls=len(vindr.label_names), dim=backbone.dim, y_val=vindr.labels[va],
                main_idx=[vindr.label_names.index(c) for c in vindr.main_classes], subset=subset)


def _val_loader(P):
    return data.eval_loader(P.ds_ev, P.va, P.protocol.batch_size, P.nw, P.backbone.device == 'cuda')


def _curves(run):
    return dict(auc=run.val_auroc, ap=run.val_ap, auprc=run.val_auprc, auc21=run.val_auroc_main, loss=run.train_loss)


def _timing(t, val_s):
    busy = max(1e-9, t['cho'] + t['dac'] + t['dau'])
    return dict(phut_hoc=busy / 60, phut_val=val_s / 60, anh_moi_giay=t['anh'] / busy,
                ti_le=dict(cho_du_lieu=t['cho'] / busy, encoder=t['dac'] / busy, dau_doc=t['dau'] / busy))


# ---- xuất / nạp lại một đầu (tệp resume giữa các giai đoạn) ----
def export_run(run, weights=True):
    '''-> dict lưu được bằng torch.save: cấu hình + đường val / loss (+ trọng số trên CPU). Không lưu optimizer: pha 2 tạo optimizer mới (như mã gốc).'''
    d = dict(head_type=run.head_type, lr=float(run.lr), seed=int(run.seed), max_steps=int(run.max_steps), step=int(run.step), tags=dict(run.tags))
    d.update({f: [float(v) for v in getattr(run, f)] for f in CURVE_FIELDS})
    if weights:
        d['state'] = {k: v.detach().to('cpu', copy=True) for k, v in run.model.state_dict().items()}
    return d


def import_run(d, P):
    '''Đầu ĐÃ HỌC XONG pha 1 từ export_run (nạp chặt trọng số) — chỉ để dự đoán và làm gốc cho pha 2.'''
    run = training.new_run(d['head_type'], d['lr'], d['seed'], d['max_steps'], P.dim, P.n_cls, P.backbone.device, P.protocol, **d['tags'])
    run.model.load_state_dict(d['state'])
    run.step = int(d['step'])
    for f in CURVE_FIELDS:
        setattr(run, f, list(d[f]))
    assert not run.active, f'{run.name}: dau trong tep resume chua hoc xong ({run.step} / {run.max_steps}) -> DUNG'
    return run


# ---- (1) dò lr ----
def _phase1_logger(P, tag, t0, t_val, log):
    def f(epoch, done):
        t = time.time()
        training.evaluate(done, P.backbone, _val_loader(P), P.y_val, P.main_idx)
        t_val[0] += time.time() - t
        log(f'  [{tag}] epoch {epoch:>2}/{P.main.epochs_phase1}: {len(done)} dau | AUROC val tot nhat theo loai '
            + ', '.join(f'{TITLES[k]} {selection.max_finite([r.val_auroc[-1] for r in done if r.head_type == k]):.4f}' for k in P.kinds)
            + f' | {(time.time() - t0) / 60:.1f} phut')
    return f


def choose(P, runs, log=print):
    '''lr của mỗi loại đầu theo từng tiêu chí (lần chấm val cuối pha 1). -> {loại: {tiêu chí: đầu}}; tiêu chí cùng chọn một lr -> CÙNG đối tượng.'''
    grid = P.protocol.lr_grid
    log("\n=== lr chon (lan cham val CUOI pha 1, 28 nhan, ngu nghia torchmetrics 0.10.3; LUAT = AUROC val; 'theo AP' = nhanh MO TA; AP = bac thang, "
        "AUPRC = hinh thang) ===")
    out = {}
    for k in P.kinds:
        rs = [r for r in runs if r.head_type == k]
        c = {crit: selection.best_by(rs, crit) for crit in CRITERIA}
        assert c[RULE] is not None, f'{k}: khong chon duoc lr (moi lr phan ky) -> loi ky thuat. DUNG'
        for crit in CRITERIA:
            if c[crit] is None: c[crit] = c[RULE]
        out[k] = c
        for crit in CRITERIA:
            r = c[crit]
            note = '' if crit == RULE else (' | = lr cua luat (khong hoc them)' if r is c[RULE] else ' | KHAC lr cua luat -> hoc them hat + pha 2 (mo ta)')
            log(f"{TITLES[k] if crit == RULE else '':<10} {CRIT_TEXT[crit]} lr {r.lr:<7g} AUROC 28 {r.val_auroc[-1]:.4f} | AP 28 {r.val_ap[-1]:.4f} | AUPRC "
                f"{r.val_auprc[-1]:.4f} | AUROC 21 {r.val_auroc_main[-1]:.4f}" + (' | lr O BIEN LUOI' if r.lr in (grid[0], grid[-1]) else '') + note)
    return out


def stage_search(P, log=print):
    dev = P.backbone.device
    runs = [training.new_run(k, lr, config.head_seed(P.protocol, P.main.epochs_phase1, k, i, 0), P.n1, P.dim, P.n_cls, dev, P.protocol, seed_index=0)
            for k in P.kinds for i, lr in enumerate(P.protocol.lr_grid)]
    assert len({r.seed for r in runs}) == len(runs), 'hat dau trung -> DUNG'
    t0, t_val = time.time(), [0.0]
    timing, _ = train_parallel(runs, P.backbone, P.ds_tr, P.tr, P.vindr.labels, P.protocol, P.n1, P.spe1, _phase1_logger(P, 'do lr', t0, t_val, log),
                               num_workers=P.nw)
    assert all(r.step == r.max_steps and len(r.val_auroc) == P.main.epochs_phase1 for r in runs), 'do lr: so buoc / so lan cham val SAI -> DUNG'
    return dict(runs=runs, chosen=choose(P, runs, log), toc_do=_timing(timing, t_val[0]))


def distinct_choices(P, chosen):
    '''-> [(loại, đầu seed 0 đã chọn)] không trùng (lr của luật trước; lr theo AP chỉ khi KHÁC).'''
    out = []
    for k in P.kinds:
        for crit in CRITERIA:
            if not any(r is chosen[k][crit] for _, r in out): out.append((k, chosen[k][crit]))
    return out


def pack_search(P, s1):
    '''Trạng thái giai đoạn (1) cho tệp resume: đường val MỌI đầu + trọng số CHỈ các đầu đã chọn.'''
    return dict(do_lr={r.name: _curves(r) for r in s1['runs']},
                phan_ky=[r.name for r in s1['runs'] if not np.isfinite(r.val_auroc).all()],
                chon={k: {crit: float(r.lr) for crit, r in c.items()} for k, c in s1['chosen'].items()},
                dau=[export_run(r) for _, r in distinct_choices(P, s1['chosen'])], toc_do=s1['toc_do'])


def unpack_chosen(P, packed):
    heads = {(d['head_type'], d['lr']): import_run(d, P) for d in packed['dau']}
    return {k: {crit: heads[(k, float(lr))] for crit, lr in c.items()} for k, c in packed['chon'].items()}


# ---- (2) thêm hạt ----
def stage_seeds(P, chosen, log=print):
    dev, grid = P.backbone.device, list(P.protocol.lr_grid)
    seeds = [training.new_run(k, r.lr, config.head_seed(P.protocol, P.main.epochs_phase1, k, grid.index(r.lr), s), P.n1, P.dim, P.n_cls, dev,
                              P.protocol, seed_index=s) for s in range(1, P.main.n_seeds) for k, r in distinct_choices(P, chosen)]
    t0, t_val, timing = time.time(), [0.0], dict(cho=0.0, dac=0.0, dau=0.0, anh=0)
    if seeds:
        timing, _ = train_parallel(seeds, P.backbone, P.ds_tr, P.tr, P.vindr.labels, P.protocol, P.n1, P.spe1,
                                   _phase1_logger(P, 'them hat', t0, t_val, log), num_workers=P.nw)
    assert all(r.step == r.max_steps and len(r.val_auroc) == P.main.epochs_phase1 for r in seeds), 'them hat: so buoc / so lan cham val SAI -> DUNG'
    return dict(runs=seeds, toc_do=_timing(timing, t_val[0]))


def arm_tag(kind, criterion):
    '''Nhãn của một nhánh trong tên mảng / JSON: luật -> 'lin1'; theo AP -> 'lin1_ap'.'''
    return kind if criterion == RULE else f'{kind}_{criterion}'


def arms(P, chosen, seeds):
    '''-> {(loại, tiêu chí): [đầu pha 1 của seed 0 .. n-1]}; nhánh theo AP TRÙNG lr của luật dùng CHUNG danh sách của luật.'''
    by = {(r.head_type, r.lr, r.tags['seed_index']): r for r in seeds}
    out = {}
    for k in P.kinds:
        for crit in CRITERIA:
            r0 = chosen[k][crit]
            if crit != RULE and r0 is chosen[k][RULE]:
                out[(k, crit)] = out[(k, RULE)]
            else:
                out[(k, crit)] = [r0] + [by[(k, r0.lr, s)] for s in range(1, P.main.n_seeds)]
    return out


def unique_arms(P, arm):
    '''-> [(nhãn nhánh, chỉ số seed, đầu pha 1)], mỗi đầu một lần (bỏ nhánh theo AP trùng luật).'''
    out = []
    for k in P.kinds:
        for crit in CRITERIA:
            if crit != RULE and arm[(k, crit)] is arm[(k, RULE)]: continue
            out += [(arm_tag(k, crit), s, r) for s, r in enumerate(arm[(k, crit)])]
    return out


# ---- (3) pha 2 + dự đoán test ----
def stage_phase2(P, arm, log=print):
    bs, pin = P.protocol.batch_size, P.backbone.device == 'cuda'
    ua = unique_arms(P, arm)
    order, phase1 = [(tag, s) for tag, s, _ in ua], [r for _, _, r in ua]
    p_val = training.predict(phase1, P.backbone, _val_loader(P))                       # val cuối pha 1 (đầu đọc eval())
    phase2 = [training.continue_run(r, P.n2, P.protocol, P.main.phase2_seed_offset, phase=2) for r in phase1]
    t0 = time.time()

    def on_epoch(epoch, done):
        if epoch % 5 == 0 or epoch == P.main.epochs_phase2:
            log(f'  [pha 2] epoch {epoch:>2}/{P.main.epochs_phase2}: {len(done)} dau | loss train TB theo loai '
                + ', '.join(f'{TITLES[k]} {np.mean([r.train_loss[-1] for r in done if r.head_type == k]):.3f}' for k in P.kinds)
                + f' | {(time.time() - t0) / 60:.1f} phut')

    timing, _ = train_parallel(phase2, P.backbone, P.ds_tr, P.trva, P.vindr.labels, P.protocol, P.n2, P.spe2, on_epoch, num_workers=P.nw)
    p_test = training.predict(phase2, P.backbone, data.eval_loader(P.ds_ev, P.te, bs, P.nw, pin))
    return dict(order=order, phase1=phase1, phase2=phase2, p_val=p_val, p_test=p_test, toc_do=_timing(timing, 0.0))


def assemble(P, state, arm, s3, minutes, run_tag=''):
    '''-> (kết quả ghi JSON được, mảng dự đoán {tên: ndarray} để lưu .npz). Tên mảng: val_<nhánh>_s<k> (cuối pha 1), test_<nhánh>_s<k> (sau pha 2);
       <nhánh> = loại đầu (lr của LUẬT) hoặc <loại>_ap (lr theo AP, chỉ khi khác lr của luật).'''
    import vindr_bench
    pr, mn, bb, vd, grid = P.protocol, P.main, P.backbone, P.vindr, P.protocol.lr_grid
    order, dev = s3['order'], P.backbone.device
    arrays = {f'val_{tag}_s{s}': p.astype(np.float32) for (tag, s), p in zip(order, s3['p_val'])}
    arrays.update({f'test_{tag}_s{s}': p.astype(np.float32) for (tag, s), p in zip(order, s3['p_test'])})
    arrays.update(y_val=vd.labels[P.va].astype(np.float32), y_test=vd.labels[P.te].astype(np.float32), idx_val=P.va.astype(np.int64),
                  idx_test=P.te.astype(np.int64))
    finite = bool(all(np.isfinite(p).all() for p in s3['p_val'] + s3['p_test']))
    steps_ok = all(r.step == r.max_steps for r in s3['phase1'] + s3['phase2'])
    diag = {f'{tag}|s{s}': selection.learning_curve_summary(r.val_auroc, r.train_loss, P.curve.drop_threshold, P.curve.gain_window)
            for (tag, s), r in zip(order, s3['phase1'])}
    pick = lambda r: dict(lr=r.lr, auc=r.val_auroc[-1], ap=r.val_ap[-1], auprc=r.val_auprc[-1], auc21=r.val_auroc_main[-1],
                          bien_luoi=r.lr in (grid[0], grid[-1]))
    chon = {k: dict({crit: pick(arm[(k, crit)][0]) for crit in CRITERIA}, ap_trung_luat=arm[(k, 'ap')] is arm[(k, RULE)]) for k in P.kinds}
    es = bb.enc.spec
    result = dict(
        t=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), run=run_tag, phien_ban_ver='Ver2', commit_chexfound=config.CHEXFOUND_COMMIT,
        goi_sha=getattr(vindr_bench, 'GOI_SHA', None), goi_phien_ban=vindr_bench.__version__, bb=bb.name, kho_anh=vd.store,
        kieu_anh=V.STORE_LABELS[vd.store], kich=es.image_size, mean=[float(v) for v in es.mean], std=[float(v) for v in es.std],
        vector_toan_anh=es.pool, amp_backbone=str(bb.amp) if dev == 'cuda' else 'khong (CPU)', tf32_dau=True, loai_dau=list(P.kinds),
        tieu_chi=dict(luat=RULE, tat_ca=list(CRITERIA)), luoi=list(grid), lo=pr.batch_size, chia=pr.loss_divisor, wd=pr.weight_decay,
        hat_du_lieu=pr.data_seed, hat_dau=pr.head_seed, so_seed=mn.n_seeds, lech_hat_pha2=mn.phase2_seed_offset,
        epoch=dict(pha1=mn.epochs_phase1, pha2=mn.epochs_phase2), buoc_moi_epoch=dict(pha1=P.spe1, pha2=P.spe2),
        tong_buoc=dict(pha1=P.n1, pha2=P.n2, moi_dau=P.n1 + P.n2), n_train=len(P.tr), n_val=len(P.va), n_test=len(P.te),
        tap_con_chay_thu=P.subset, md5_khoa=V.md5_keys(vd.keys), md5_khoa_test=V.md5_keys([vd.keys[i] for i in P.te]), nhan=list(vd.label_names),
        lop_chinh=list(vd.main_classes), do_lr=state['s1']['do_lr'], chon=chon, them_seed=state['s2']['them_seed'], duong_hoc=diag,
        chan_doan=dict(nguong_giam=P.curve.drop_threshold, cua_so_tang=P.curve.gain_window),
        pha2={f'{tag}|s{s}': dict(hat=r.seed, lr=r.lr, loss=r.train_loss, buoc=r.step) for (tag, s), r in zip(order, s3['phase2'])},
        nhanh=[dict(nhan=tag, seed=s, lr=r.lr, hat_pha1=r.seed) for (tag, s), r in zip(order, s3['phase1'])],
        kiem=dict(dau_phan_ky_do_lr=state['s1']['phan_ky'], so_buoc_dung=steps_ok, du_doan_huu_han=finite),
        toc_do=dict(do_lr=state['s1']['toc_do'], seed=state['s2']['toc_do'], pha2=s3['toc_do'], phut_phien_nay=minutes,
                    gpu_gb_dinh=(torch.cuda.max_memory_allocated() / 1e9 if dev == 'cuda' else None), thiet_bi=runtime.device_name(dev), so_worker=P.nw,
                    ram_gb=list(runtime.ram_gb()), giai_doan_khoi_phuc=state.get('khoi_phuc', [])),
        mang=sorted(arrays), nguon=vd.sources,
        ghi_chu='PLAN_89 5.1 (Ver2 = giao thuc 86b): du doan test MOT lan sau pha 2; so do test cham o notebook (C7) va bang gop.')
    return result, arrays


def run(P, state=None, save=None, run_tag='', log=print):
    '''Một lượt đầy đủ. state: dict các giai đoạn ĐÃ XONG (đọc từ tệp resume; None / {} = từ đầu) — được cập nhật tại chỗ; save(state) gọi sau giai
       đoạn (1) và (2). -> (kết quả, mảng dự đoán). Không chọn được lr cho một loại đầu (mọi lr phân kỳ) = lỗi kỹ thuật -> ném lỗi.'''
    state = {} if state is None else state
    pr, mn, dev = P.protocol, P.main, P.backbone.device
    t0, ram = time.time(), runtime.ram_gb()
    if dev == 'cuda': torch.cuda.reset_peak_memory_stats()
    log(f'Ver2 {P.backbone.name} | kho {P.vindr.store} = {V.STORE_LABELS[P.vindr.store]} | (1) do lr {len(P.kinds)} x {len(pr.lr_grid)} dau, seed 0, '
        f'{mn.epochs_phase1} epoch x {P.spe1} buoc tren train {len(P.tr):,} anh | (2) them {mn.n_seeds - 1} seed o lr chon | (3) pha 2: hoc tiep '
        f'{mn.epochs_phase2} epoch x {P.spe2} buoc tren train + val {len(P.trva):,} anh, du doan test {len(P.te):,} anh MOT lan | lo {pr.batch_size} | '
        f'{P.nw} worker | RAM trong / tong {ram[0]} / {ram[1]} GB' + (' | TAP CON CHAY THU' if P.subset else ''))
    if 's1' in state:
        chosen = unpack_chosen(P, state['s1'])
        state.setdefault('khoi_phuc', []).append('s1')
        log('RESUME: giai doan (1) do lr da xong -> khoi phuc cac dau da chon: '
            + ', '.join(f"{TITLES[k]} {' / '.join(f'{lr:g}' for lr in dict.fromkeys(c.values()))}" for k, c in state['s1']['chon'].items()))
    else:
        s1 = stage_search(P, log)
        chosen = s1['chosen']
        state['s1'] = pack_search(P, s1)
        del s1
        if save is not None: save(state)
    if 's2' in state:
        seeds = [import_run(d, P) for d in state['s2']['dau']]
        state.setdefault('khoi_phuc', []).append('s2')
        log(f"RESUME: giai doan (2) them hat da xong -> khoi phuc {len(seeds)} dau")
    else:
        s2 = stage_seeds(P, chosen, log)
        seeds = s2['runs']
        state['s2'] = dict(dau=[export_run(r) for r in seeds], them_seed={r.name: _curves(r) for r in seeds}, toc_do=s2['toc_do'])
        if save is not None: save(state)
    arm = arms(P, chosen, seeds)
    s3 = stage_phase2(P, arm, log)
    result, arrays = assemble(P, state, arm, s3, (time.time() - t0) / 60, run_tag)
    log(f"\nxong: {'DAT' if result['kiem']['so_buoc_dung'] else 'TRUOT'} so buoc | {'DAT' if result['kiem']['du_doan_huu_han'] else 'TRUOT'} du doan "
        f"val + test huu han | du doan test {len(s3['p_test'])} dau x {s3['p_test'][0].shape} | phien nay {result['toc_do']['phut_phien_nay']:.1f} phut | "
        "GPU dinh " + (f"{result['toc_do']['gpu_gb_dinh']:.1f} GB" if result['toc_do']['gpu_gb_dinh'] is not None else '-'))
    return result, arrays


def roundtrip_check(P, state, arrays):
    '''Phục hồi các đầu pha 1 từ `state` (đúng đường của resume) rồi dự đoán val: phải trùng mảng val_* của lượt vừa chạy. -> lệch tối đa.'''
    chosen = unpack_chosen(P, state['s1'])
    seeds = [import_run(d, P) for d in state['s2']['dau']]
    ua = unique_arms(P, arms(P, chosen, seeds))
    preds = training.predict([r for _, _, r in ua], P.backbone, _val_loader(P))
    return max(float(np.abs(p.astype(np.float32) - arrays[f'val_{tag}_s{s}']).max()) for (tag, s, _), p in zip(ua, preds))
