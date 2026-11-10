"""Điều phối: vòng học song song nhiều đầu trên cùng đặc trưng (backbone đông cứng chạy lại mỗi lô trên ảnh tăng cường — PLAN_86 F1),
chạy thử pha 1 (86a — PLAN_86 §5.1) và các phép kiểm (dự đoán lặp lại ở eval(), dropout bật khi chấm như mã gốc, một đầu riêng = trong nhóm).
Không ghi Drive: hàm trả kết quả, notebook ghi bằng khối Drive chuẩn."""
import datetime
import os
import time

import numpy as np
import torch
from PIL import Image

from . import config, data, features, heads, runtime, selection, training
from .config import HEAD_TYPES
from .metrics import ap_auroc_tm0103


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


# ---- các phép kiểm ----
def check_eval_repeatable(runs, backbone, loader_fn):
    '''Dự đoán val hai lần ở eval() -> {loại: lệch tối đa} (phải 0) + dự đoán lần đầu.'''
    a, b = training.predict(runs, backbone, loader_fn()), training.predict(runs, backbone, loader_fn())
    return {r.head_type: float(np.abs(x - y).max()) for r, x, y in zip(runs, a, b)}, a


def eval_minus_train_mode(runs, backbone, loader_fn, labels, probs_eval):
    '''AUROC val khi chấm ở eval() (sửa của 86) - khi để dropout bật như mã gốc (F7) -> {loại: chênh}.'''
    probs_train = training.predict(runs, backbone, loader_fn(), mode='train')
    return {r.head_type: ap_auroc_tm0103(labels, pe)[3] - ap_auroc_tm0103(labels, pt)[3]
            for r, pe, pt in zip(runs, probs_eval, probs_train) if np.isfinite(pt).all()}


def check_solo_vs_group(group_snapshot, check_runs, backbone, dataset, indices, labels, protocol, steps, probe_images, num_workers=0):
    '''Mỗi đầu kiểm học MỘT MÌNH `steps` bước đầu của cùng dòng mẫu (cùng hạt, cùng lịch lr) -> so với ảnh chụp của nó trong nhóm ở cùng bước:
       lệch logit trên một lô val ở eval() + lệch trọng số. -> {loại: {logit, trong_so, buoc}}.'''
    dev, n_cls = backbone.device, int(np.asarray(labels).shape[1])
    y_all = torch.as_tensor(np.asarray(labels), dtype=torch.float32, device=dev)
    blocks_v, last_v = backbone.features(probe_images.to(dev))
    out = {}
    for t, r0 in check_runs.items():
        r1 = training.new_run(t, r0.lr, r0.seed, r0.max_steps, backbone.dim, n_cls, dev, protocol, **r0.tags)
        for x, j in data.train_loader(dataset, indices, protocol.data_seed, 0, steps, protocol.batch_size, num_workers, pin_memory=(dev == 'cuda')):
            blocks, last = backbone.features(x.to(dev, non_blocking=True))
            training.train_step(r1, blocks, last, y_all[j.to(dev, non_blocking=True)], protocol.loss_divisor)
            del blocks, last
        m0 = heads.build_head(t, r0.seed, backbone.dim, n_cls, dev, protocol)
        m0.load_state_dict(group_snapshot[t])
        m0.eval(); r1.model.eval()
        with torch.no_grad():
            l0 = heads.run_head(t, m0, blocks_v, last_v).float()
            l1 = heads.run_head(t, r1.model, blocks_v, last_v).float()
        out[t] = dict(logit=float((l0 - l1).abs().max()), trong_so=training.max_weight_diff(group_snapshot[t], r1.model.state_dict()), buoc=r1.step)
        r1.model.train()
    return out


# ---- chạy thử pha 1 (86a) ----
def run_phase1_pilot(vindr, backbone, protocol=config.Protocol(), pilot=config.Pilot(), num_workers=None, run_tag='', log=print):
    '''CHẠY THỬ pha 1: 3 loại đầu x 13 lr x các ngân sách epoch (cosine theo từng ngân sách) học song song trên train, chấm val mỗi epoch
       (đầu đọc eval()); chọn lr theo AUROC val; luật khoá epoch; tốc độ / bộ nhớ; các phép kiểm. KHÔNG dự đoán test. Cổng trước vòng học ném
       lỗi; SAU vòng học không ném lỗi: mục trượt nằm ở kiem_dat (để notebook vẫn ghi + đồng bộ kết quả). -> dict kết quả (ghi JSON được).'''
    nw = runtime.default_num_workers() if num_workers is None else num_workers
    dev, pin = backbone.device, backbone.device == 'cuda'
    runtime.allow_tf32_heads()
    tr, va = vindr.indices('train'), vindr.indices('val')
    assert len(tr) and len(va) and not set(tr.tolist()) & set(va.tolist()), 'tach train / val rong hoac chong nhau -> DUNG'
    assert set(np.unique(vindr.labels[np.r_[tr, va]]).tolist()) <= {0.0, 1.0}, 'nhan train / val khong nhi phan -> DUNG'   # mã gốc: nhãn < 0 -> trọng số 0
    tf_train, tf_eval = data.build_transforms(backbone.spec)
    ds_tr = data.VinDrImages(vindr.keys, vindr.image_dir, tf_train, seed=protocol.data_seed)
    ds_ev = data.VinDrImages(vindr.keys, vindr.image_dir, tf_eval)
    gate = features.check_features(backbone, Image.open(os.path.join(vindr.image_dir, f'{vindr.keys[int(tr[0])]}.jpg')).convert('RGB'), tf_eval)
    spe = config.steps_per_epoch(len(tr), protocol.batch_size)
    total = max(pilot.budgets) * spe
    n_cls, dim = len(vindr.label_names), backbone.dim
    runs = [training.new_run(t, lr, config.head_seed(protocol, E, t, i), E * spe, dim, n_cls, dev, protocol, budget=E)
            for E in pilot.budgets for t in HEAD_TYPES for i, lr in enumerate(protocol.lr_grid)]
    assert len({r.seed for r in runs}) == len(runs), 'hat dau trung -> DUNG'
    check_runs = {t: next(r for r in runs if r.tags['budget'] == min(pilot.budgets) and r.head_type == t and r.lr == pilot.check_lr) for t in HEAD_TYPES}
    y_val, main_idx = vindr.labels[va], [vindr.label_names.index(c) for c in vindr.main_classes]
    val_loader = lambda: data.eval_loader(ds_ev, va, protocol.batch_size, nw, pin)
    ram0, t0, t_val = runtime.ram_gb(), time.time(), [0.0]
    if dev == 'cuda': torch.cuda.reset_peak_memory_stats()
    log(f'86a: {len(runs)} dau ({len(HEAD_TYPES)} loai x {len(protocol.lr_grid)} lr x ngan sach {pilot.budgets}) | train {len(tr):,} anh, {spe} buoc / '
        f'epoch (lo {protocol.batch_size}, dong mau vo han), {total} buoc | val {len(va):,} anh | {nw} worker | backbone autocast '
        f'{str(backbone.amp).split(".")[-1]}, dau doc fp32 (TF32) | RAM trong / tong {ram0[0]} / {ram0[1]} GB')

    def on_epoch_end(epoch, done):
        t = time.time()
        training.evaluate(done, backbone, val_loader(), y_val, main_idx)
        t_val[0] += time.time() - t
        log(f'  epoch {epoch:>2}: {len(done)} dau | AUROC val tot nhat theo loai (28 nhan, torchmetrics) '
            + ', '.join(f'{ty} {selection.max_finite([r.val_auroc[-1] for r in done if r.head_type == ty]):.4f}' for ty in HEAD_TYPES)
            + f' | {(time.time() - t0) / 60:.1f} phut')

    timing, snapshot = train_parallel(runs, backbone, ds_tr, tr, vindr.labels, protocol, total, spe, on_epoch_end, pilot.solo_check_steps, check_runs, nw)
    # ---- chọn lr + luật khoá epoch ----
    kiem = dict(so_buoc_va_lan_cham=all(r.step == r.max_steps and len(r.val_auroc) == r.tags['budget'] for r in runs),
                dau_phan_ky=[r.name for r in runs if not np.isfinite(r.val_auroc).all()], dac_trung_tang_cuoi=gate['lech'])
    best = {(E, t): selection.best_run([r for r in runs if r.tags['budget'] == E and r.head_type == t]) for E in pilot.budgets for t in HEAD_TYPES}
    e_lock, diff = selection.epoch_rule(best, pilot.budgets, pilot.epoch_rule_threshold) if all(best.values()) else (None, {})
    log(f"\n=== chon lr (AUROC val 28 nhan lan cham cuoi, ngu nghia torchmetrics 0.10.3) | dau phan ky (val NaN): {len(kiem['dau_phan_ky'])} ===")
    log(f"{'loai':<7}{'E':>4}{'lr':>9}{'AUROC 28':>10}{'AP 28':>8}{'AUPRC':>8}{'AUROC 21':>10}{'hoi tu':>8}  duong AUROC val (lr chon, tung epoch)"
        "   [AP = AP bac thang = 'AUPRC' cua bai CheXFound; AUPRC = hinh thang, bao kem nhu 85]")
    for (E, t), r in best.items():
        if r is None:
            log(f'{t:<7}{E:>4}  ⚠ KHONG dau nao co AUROC val > 0'); continue
        log(f"{t:<7}{E:>4}{r.lr:>9g}{r.val_auroc[-1]:>10.4f}{r.val_ap[-1]:>8.4f}{r.val_auprc[-1]:>8.4f}{r.val_auroc_main[-1]:>10.4f}"
            f"{('co' if selection.converged(r, pilot.convergence_threshold) else 'KHONG'):>8}  " + ' '.join(f'{v:.3f}' for v in r.val_auroc)
            + (' | ⚠ lr o MEP luoi' if r.lr in (protocol.lr_grid[0], protocol.lr_grid[-1]) else ''))
    if e_lock is not None:
        log(f'LUAT khoa epoch (dat truoc, PLAN_86 5.1): AUROC val tot nhat E={max(pilot.budgets)} - E={min(pilot.budgets)} theo loai '
            + ', '.join(f'{k} {v:+.4f}' for k, v in diff.items()) + f' | nguong {pilot.epoch_rule_threshold} -> E1 = E2 = {e_lock}')
    # ---- kiểm ----
    chosen = [r for r in (best[(min(pilot.budgets), t)] for t in HEAD_TYPES) if r is not None]
    kiem['lap_lai_eval'], p_eval = check_eval_repeatable(chosen, backbone, val_loader)
    kiem['auc_eval_tru_train'] = eval_minus_train_mode(chosen, backbone, val_loader, y_val, p_eval)
    probe, _ = next(iter(data.eval_loader(ds_ev, va[:protocol.batch_size], protocol.batch_size)))
    solo = check_solo_vs_group(snapshot, check_runs, backbone, ds_tr, tr, vindr.labels, protocol, pilot.solo_check_steps, probe, nw)
    kiem['mot_dau_rieng_tru_trong_nhom'] = solo
    busy = max(1e-9, timing['cho'] + timing['dac'] + timing['dau'])
    toc_do = dict(anh_moi_giay=timing['anh'] / busy, phut_hoc=busy / 60, phut_val=t_val[0] / 60, phut_tong=(time.time() - t0) / 60,
                  ti_le=dict(cho_du_lieu=timing['cho'] / busy, encoder=timing['dac'] / busy, dau_doc=timing['dau'] / busy),
                  gpu_gb_dinh=(torch.cuda.max_memory_allocated() / 1e9 if dev == 'cuda' else None), so_dau=len(runs), so_buoc=total,
                  thiet_bi=(torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'), so_worker=nw, ram_gb_trong_dau_cuoi=[ram0[0], runtime.ram_gb()[0]],
                  ram_gb_tong=ram0[1])
    kiem_dat = dict(so_buoc_va_lan_cham=kiem['so_buoc_va_lan_cham'], chon_du=all(v is not None for v in best.values()),
                    lap_lai_eval=len(kiem['lap_lai_eval']) == len(HEAD_TYPES) and all(v == 0.0 for v in kiem['lap_lai_eval'].values()),
                    mot_dau_rieng=len(solo) == len(HEAD_TYPES) and all(v['logit'] <= pilot.solo_check_logit_tol and v['buoc'] == pilot.solo_check_steps
                                                                       for v in solo.values()))
    mark = lambda ok: '✓' if ok else '⚠'
    log(f"\nkiem: {mark(kiem_dat['so_buoc_va_lan_cham'])} so buoc + so lan cham val dung | {mark(kiem_dat['chon_du'])} chon duoc lr cho moi loai x "
        f"ngan sach | {mark(kiem_dat['lap_lai_eval'])} du doan val lap lai o eval() (lech max {kiem['lap_lai_eval']}) | {mark(kiem_dat['mot_dau_rieng'])} "
        f"mot dau rieng so voi trong nhom sau {pilot.solo_check_steps} buoc (lech logit max <= {pilot.solo_check_logit_tol:g}): "
        + ', '.join(f"{k} logit {v['logit']:.1e} / trong so {v['trong_so']:.1e}" for k, v in solo.items()))
    log(f"do them: AUROC val eval() - train() (dropout bat khi cham nhu ma goc) " + ', '.join(f'{k} {v:+.4f}' for k, v in kiem['auc_eval_tru_train'].items())
        + f" | toc do {toc_do['anh_moi_giay']:.0f} anh / s luot hoc (cho du lieu {toc_do['ti_le']['cho_du_lieu']:.0%}, encoder {toc_do['ti_le']['encoder']:.0%}, "
        f"dau doc {toc_do['ti_le']['dau_doc']:.0%}) | val {toc_do['phut_val']:.1f} phut | tong {toc_do['phut_tong']:.1f} phut | GPU dinh "
        + (f"{toc_do['gpu_gb_dinh']:.1f} GB" if toc_do['gpu_gb_dinh'] is not None else '-'))
    sp = backbone.spec
    return dict(
        t=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), run=run_tag, commit_chexfound=config.CHEXFOUND_COMMIT, bb=backbone.name,
        nguon_bb=backbone.source, kich=sp.image_size, mean=[float(v) for v in sp.mean], std=[float(v) for v in sp.std], amp_backbone=str(backbone.amp),
        tf32_dau=True, luoi=list(protocol.lr_grid), ngan_sach=list(pilot.budgets), lo=protocol.batch_size, chia=protocol.loss_divisor,
        wd=protocol.weight_decay, hat_du_lieu=protocol.data_seed, hat_dau=protocol.head_seed, n_train=len(tr), n_val=len(va), buoc_moi_epoch=spe,
        tong_buoc={str(E): E * spe for E in pilot.budgets}, md5_khoa=data.md5_keys(vindr.keys), nhan=list(vindr.label_names),
        lop_chinh=list(vindr.main_classes), kho_anh=vindr.store,
        duong={r.name: dict(auc=r.val_auroc, ap=r.val_ap, auprc=r.val_auprc, auc21=r.val_auroc_main, loss=r.train_loss) for r in runs},
        chon={f'{E}|{t}': (None if r is None else dict(lr=r.lr, auc=r.val_auroc[-1], ap=r.val_ap[-1], auprc=r.val_auprc[-1], auc21=r.val_auroc_main[-1],
                                                       hoi_tu=selection.converged(r, pilot.convergence_threshold)))
              for (E, t), r in best.items()},
        luat_epoch=dict(chenh=diff, nguong=pilot.epoch_rule_threshold, e_chot=e_lock), kiem=kiem, kiem_dat=kiem_dat, toc_do=toc_do,
        nguon=vindr.sources, ghi_chu='PLAN_86 7 buoc 2: chay thu pha 1, CHI val, KHONG du doan test (goi vindr_protocol).')
