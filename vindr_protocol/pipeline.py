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
def run_phase1_pilot(vindr, backbone, protocol=config.Protocol(), pilot=config.Pilot(), num_workers=None, run_tag='', log=print,
                     curve=config.CurveDiagnostic()):
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
    log(f"{'loai':<7}{'E':>4}{'lr':>9}{'AUROC 28':>10}{'AP 28':>8}{'AUPRC':>8}{'AUROC 21':>10}{'dinh (ep)':>15}{'giam':>8}"
        f"{f'tang {curve.gain_window} cuoi':>13}  duong AUROC val (lr chon, tung epoch)"
        "   [AP = AP bac thang = 'AUPRC' cua bai CheXFound; AUPRC = hinh thang, bao kem nhu 85; chan doan duong hoc §5.2: CHI mo ta]")
    curves = {}
    for (E, t), r in best.items():
        if r is None:
            log(f'{t:<7}{E:>4}  ⚠ KHONG dau nao co AUROC val > 0'); continue
        c = curves[(E, t)] = selection.learning_curve_summary(r.val_auroc, r.train_loss, curve.drop_threshold, curve.gain_window)
        gain = '-' if c['gain_last'] is None else f"{c['gain_last']:+.4f}"
        log(f"{t:<7}{E:>4}{r.lr:>9g}{r.val_auroc[-1]:>10.4f}{r.val_ap[-1]:>8.4f}{r.val_auprc[-1]:>8.4f}{r.val_auroc_main[-1]:>10.4f}"
            f"{c['peak_auroc']:>10.4f} ({c['peak_epoch']:>2}){c['drop_after_peak']:>8.4f}{gain:>13}  " + ' '.join(f'{v:.3f}' for v in r.val_auroc)
            + (' | val_drop_after_peak' if c['val_drop_after_peak'] else '') + (' | ⚠ lr o MEP luoi' if r.lr in (protocol.lr_grid[0], protocol.lr_grid[-1]) else ''))
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
                                                       duong_hoc=curves[(E, t)]))
              for (E, t), r in best.items()},
        luat_epoch=dict(chenh=diff, nguong=pilot.epoch_rule_threshold, e_chot=e_lock), kiem=kiem, kiem_dat=kiem_dat, toc_do=toc_do,
        nguon=vindr.sources, ghi_chu='PLAN_86 7 buoc 2: chay thu pha 1, CHI val, KHONG du doan test (goi vindr_protocol).')


# ---- lượt chính (86b) ----
def _curves(run):
    return dict(auc=run.val_auroc, ap=run.val_ap, auprc=run.val_auprc, auc21=run.val_auroc_main, loss=run.train_loss)


def _timing(t, val_s):
    busy = max(1e-9, t['cho'] + t['dac'] + t['dau'])
    return dict(phut_hoc=busy / 60, phut_val=val_s / 60, anh_moi_giay=t['anh'] / busy,
                ti_le=dict(cho_du_lieu=t['cho'] / busy, encoder=t['dac'] / busy, dau_doc=t['dau'] / busy))


def _paper_key(run):
    '''Tên ngắn của một đầu nhánh sát bài (mảng .npz / JSON; chỉ [A-Za-z0-9_]): E<ngân sách>_<loại>_lr<lr: '.' -> 'p', '-' -> 'm'> (vd lr0p002, lr1em05).'''
    return f"E{run.tags['budget']}_{run.head_type}_lr" + f'{run.lr:g}'.replace('.', 'p').replace('-', 'm')


def _unique_runs(runs):
    '''Bỏ đầu trùng (cùng đối tượng), giữ thứ tự xuất hiện.'''
    seen, out = set(), []
    for r in runs:
        if id(r) not in seen:
            seen.add(id(r)); out.append(r)
    return out


def run_main(vindr, backbone, protocol=config.Protocol(), main=config.MainRun(), curve=config.CurveDiagnostic(), num_workers=None, run_tag='',
             extra=None, log=print, paper=None):
    '''LƯỢT CHÍNH cho MỘT backbone (PLAN_86 §4.4, §5.2):
         (1) dò lr: 3 loại đầu x 13 lr, seed 0, E1 epoch trên train, chấm val mỗi epoch; lr = AUROC val lần chấm cuối lớn nhất (hoà -> lr nhỏ);
         (2) thêm seed 1 .. n-1 ở lr đã chọn (E1 epoch, chấm val mỗi epoch);
         (3) chẩn đoán đường học cho mọi seed (CHỈ mô tả) + dự đoán val lần cuối pha 1;
         (4) pha 2: mỗi đầu học TIẾP từ trọng số pha 1 của chính nó trên train + val, E2 epoch, optimizer / cosine / bước đếm mới, checkpoint cuối;
         (5) dự đoán test MỘT lần sau pha 2 — KHÔNG tính số đo test (chấm ở notebook bảng).
       paper (config.PaperArms, PLAN_86 §11.3) khác None -> thêm NHÁNH SÁT BÀI vào cùng các lượt (đầu thêm, RNG đầu tách -> đầu của lượt chính
       không đổi): (1) thêm đầu loại paper.head_types x 13 lr x ngân sách paper.budgets (seed 0, cosine theo ngân sách của nó), chọn lr lưới 13
       và lưới 9; (4) đầu đã chọn học tiếp cùng ngân sách trên train + val; (5) dự đoán val cuối pha 1 + test ở mọi paper.predict_modes. Mặc
       định None = lượt chính y như 86b_raddino.
       Cổng trước vòng học ném lỗi; không chọn được lr cho một loại đầu (mọi lr phân kỳ) = lỗi kỹ thuật -> ném lỗi.
       -> (kết quả ghi JSON được, mảng dự đoán {tên: ndarray float32} để lưu .npz).'''
    import dataclasses

    import vindr_protocol
    nw = runtime.default_num_workers() if num_workers is None else num_workers
    dev, pin, bs = backbone.device, backbone.device == 'cuda', protocol.batch_size
    runtime.allow_tf32_heads()
    tr, va, te = vindr.indices('train'), vindr.indices('val'), vindr.indices('test')
    assert len(tr) and len(va) and len(te) and not (set(tr.tolist()) & set(va.tolist()) or set(te.tolist()) & set(np.r_[tr, va].tolist())), (
        'tach train / val / test rong hoac chong nhau -> DUNG')
    assert set(np.unique(vindr.labels[np.r_[tr, va]]).tolist()) <= {0.0, 1.0}, 'nhan train / val khong nhi phan -> DUNG'
    tf_train, tf_eval = data.build_transforms(backbone.spec)
    ds_tr = data.VinDrImages(vindr.keys, vindr.image_dir, tf_train, seed=protocol.data_seed)
    ds_ev = data.VinDrImages(vindr.keys, vindr.image_dir, tf_eval)
    gate = features.check_features(backbone, Image.open(os.path.join(vindr.image_dir, f'{vindr.keys[int(tr[0])]}.jpg')).convert('RGB'), tf_eval)
    trva = np.sort(np.r_[tr, va])
    spe1, spe2 = config.steps_per_epoch(len(tr), bs), config.steps_per_epoch(len(trva), bs)
    n1, n2 = main.epochs_phase1 * spe1, main.epochs_phase2 * spe2
    n_cls, dim = len(vindr.label_names), backbone.dim
    y_val, main_idx = vindr.labels[va], [vindr.label_names.index(c) for c in vindr.main_classes]
    val_loader = lambda: data.eval_loader(ds_ev, va, bs, nw, pin)
    ram0, t0, t_val = runtime.ram_gb(), time.time(), dict(do_lr=0.0, seed=0.0)
    if dev == 'cuda': torch.cuda.reset_peak_memory_stats()

    def on_epoch_val(tag):
        def f(epoch, done):
            t = time.time()
            training.evaluate(done, backbone, val_loader(), y_val, main_idx)
            t_val[tag] += time.time() - t
            md = [r for r in done if 'arm' not in r.tags]                     # log: CHỈ đầu lượt chính (đầu sát bài in riêng)
            log(f'  [{tag}] epoch {epoch:>2}/{main.epochs_phase1}: {len(md)} dau | AUROC val tot nhat theo loai '
                + ', '.join(f'{ty} {selection.max_finite([r.val_auroc[-1] for r in md if r.head_type == ty]):.4f}' for ty in HEAD_TYPES)
                + (f' | + {len(done) - len(md)} dau sat bai' if len(done) > len(md) else '') + f' | {(time.time() - t0) / 60:.1f} phut')
        return f

    def on_epoch_phase2(epoch, done):
        if epoch % 5 == 0 or epoch == main.epochs_phase2:
            md = [r for r in done if 'arm' not in r.tags]
            log(f'  [pha 2] epoch {epoch:>2}/{main.epochs_phase2}: {len(md)} dau | loss train TB theo loai '
                + ', '.join(f'{ty} {np.mean([r.train_loss[-1] for r in md if r.head_type == ty]):.3f}' for ty in HEAD_TYPES)
                + (f' | + {len(done) - len(md)} dau sat bai' if len(done) > len(md) else '') + f' | {(time.time() - t0) / 60:.1f} phut')

    log(f'86b {backbone.name}: (1) do lr {len(HEAD_TYPES)} x {len(protocol.lr_grid)} dau, seed 0, {main.epochs_phase1} epoch x {spe1} buoc tren train '
        f'{len(tr):,} anh | (2) them {main.n_seeds - 1} seed x {len(HEAD_TYPES)} dau o lr chon | (3) pha 2: {main.n_seeds * len(HEAD_TYPES)} dau hoc tiep '
        f'{main.epochs_phase2} epoch x {spe2} buoc tren train + val {len(trva):,} anh | (4) du doan test {len(te):,} anh MOT lan | '
        f'kho anh {vindr.store} = {data.STORE_LABELS[vindr.store]} | {nw} worker | RAM trong / tong {ram0[0]} / {ram0[1]} GB')
    # ---- (1) dò lr (+ đầu nhánh sát bài cùng lượt) ----
    search = [training.new_run(t, lr, config.head_seed(protocol, main.epochs_phase1, t, i, 0), n1, dim, n_cls, dev, protocol, seed_index=0)
              for t in HEAD_TYPES for i, lr in enumerate(protocol.lr_grid)]
    arms = []
    if paper is not None:
        assert (all(0 < E <= min(main.epochs_phase1, main.epochs_phase2) for E in paper.budgets) and set(paper.head_types) <= set(HEAD_TYPES)
                and tuple(paper.paper_grid) == tuple(protocol.lr_grid[:len(paper.paper_grid)])
                and set(paper.predict_modes) <= {'eval', 'train', 'train_fp16'}), f'cau hinh nhanh sat bai {paper} sai -> DUNG'
        arms = [training.new_run(t, lr, config.head_seed(protocol, E, t, i, 0), E * spe1, dim, n_cls, dev, protocol, arm='paper', budget=E, seed_index=0)
                for E in paper.budgets for t in paper.head_types for i, lr in enumerate(protocol.lr_grid)]
        log(f'nhanh SAT BAI (PLAN_86 11.3, CHI mo ta): {len(arms)} dau = {list(paper.head_types)} x {len(protocol.lr_grid)} lr x E1 = E2 '
            f'{list(paper.budgets)}, seed 0, hoc CUNG luot (dau them; RNG dau tach -> dau luot chinh khong doi)')
    assert len({r.seed for r in search + arms}) == len(search) + len(arms), 'hat dau trung -> DUNG'
    tim_search, _ = train_parallel(search + arms, backbone, ds_tr, tr, vindr.labels, protocol, n1, spe1, on_epoch_val('do_lr'), num_workers=nw)
    best = {t: selection.best_run([r for r in search if r.head_type == t]) for t in HEAD_TYPES}
    assert all(best.values()), f'khong chon duoc lr cho {[t for t, r in best.items() if r is None]} (moi lr phan ky) -> loi ky thuat. DUNG'
    log(f"\n=== lr chon (AUROC val 28 nhan lan cham cuoi, ngu nghia torchmetrics 0.10.3; AP = bac thang, AUPRC = hinh thang) ===")
    for t, r in best.items():
        log(f"{t:<6} lr {r.lr:<7g} AUROC 28 {r.val_auroc[-1]:.4f} | AP 28 {r.val_ap[-1]:.4f} | AUPRC {r.val_auprc[-1]:.4f} | AUROC 21 "
            f"{r.val_auroc_main[-1]:.4f}" + (' | ⚠ lr o MEP luoi' if r.lr in (protocol.lr_grid[0], protocol.lr_grid[-1]) else ''))
    paper_choice = {}
    if paper is not None:
        log(f"\n=== nhanh SAT BAI: lr chon theo AUROC val 28 nhan lan cham cuoi (luoi 13 = mac dinh ma; luoi 9 = bai {paper.paper_grid[0]:g} .. "
            f"{paper.paper_grid[-1]:g}) — CHI mo ta ===")
        for E in paper.budgets:
            for t in paper.head_types:
                rs = [r for r in arms if r.tags['budget'] == E and r.head_type == t]
                c = dict(g13=selection.best_run(rs), g9=selection.best_run([r for r in rs if r.lr in paper.paper_grid]))
                assert all(c.values()), f'nhanh sat bai E={E} {t}: khong chon duoc lr (moi lr phan ky) -> loi ky thuat. DUNG'
                paper_choice[(E, t)] = c
                for g, r in c.items():
                    grid = protocol.lr_grid if g == 'g13' else paper.paper_grid
                    log(f"E={E:<3}{t:<6}{g:<4} lr {r.lr:<7g} AUROC 28 {r.val_auroc[-1]:.4f} | AP 28 {r.val_ap[-1]:.4f} | AUPRC {r.val_auprc[-1]:.4f} | "
                        f"AUROC 21 {r.val_auroc_main[-1]:.4f}" + (' | ⚠ lr o MEP luoi' if r.lr in (grid[0], grid[-1]) else '')
                        + (' | = luoi 13' if g == 'g9' and r is c['g13'] else ''))
    # ---- (2) thêm seed ở lr đã chọn ----
    seeds = [training.new_run(t, best[t].lr, config.head_seed(protocol, main.epochs_phase1, t, protocol.lr_grid.index(best[t].lr), k), n1, dim, n_cls,
                              dev, protocol, seed_index=k) for k in range(1, main.n_seeds) for t in HEAD_TYPES]
    tim_seed, _ = train_parallel(seeds, backbone, ds_tr, tr, vindr.labels, protocol, n1, spe1, on_epoch_val('seed'), num_workers=nw)
    order = [(t, k) for t in HEAD_TYPES for k in range(main.n_seeds)]
    phase1 = {**{(t, 0): best[t] for t in HEAD_TYPES}, **{(r.head_type, r.tags['seed_index']): r for r in seeds}}
    assert sorted(phase1) == sorted(order), 'thieu dau pha 1 cho mot (loai, seed) -> DUNG'
    # ---- (3) chẩn đoán đường học (§5.2, chỉ mô tả) + dự đoán val cuối pha 1 ----
    diag = {f'{t}|s{k}': selection.learning_curve_summary(phase1[(t, k)].val_auroc, phase1[(t, k)].train_loss, curve.drop_threshold, curve.gain_window)
            for t, k in order}
    log(f"\n=== chan doan duong hoc pha 1 (§5.2, CHI mo ta; co khi giam sau dinh > {curve.drop_threshold}) ===")
    for name, c in diag.items():
        if not c['finite']:
            log(f'{name:<9} ⚠ duong val co NaN / Inf (phan ky) -> xu ly ky thuat, khong phai chan doan duong hoc'); continue
        fm = lambda v, spec: '-' if v is None else format(v, spec)
        log(f"{name:<9} dinh {c['peak_auroc']:.4f} (epoch {c['peak_epoch']:>2}) | cuoi {c['last_auroc']:.4f} | giam {c['drop_after_peak']:.4f} | tang "
            f"{curve.gain_window} epoch cuoi {fm(c['gain_last'], '+.4f')} | loss train dinh / cuoi {fm(c['train_loss_at_peak'], '.2f')} / "
            f"{fm(c['train_loss_last'], '.2f')}" + (' | val_drop_after_peak' if c['val_drop_after_peak'] else ''))
    p_val = training.predict([phase1[o] for o in order], backbone, val_loader())
    paper_p1 = _unique_runs([r for c in paper_choice.values() for r in c.values()])          # đầu sát bài đã chọn (lưới 13 / lưới 9, bỏ trùng)
    paper_diag = {_paper_key(r): selection.learning_curve_summary(r.val_auroc, r.train_loss, curve.drop_threshold, curve.gain_window) for r in paper_p1}
    p_val_paper = training.predict(paper_p1, backbone, val_loader()) if paper_p1 else []
    # ---- (4) pha 2: học tiếp trên train + val (đầu sát bài: cùng lượt, ngân sách của nó) ----
    phase2 = [training.continue_run(phase1[o], n2, protocol, main.phase2_seed_offset, phase=2) for o in order]
    paper_p2 = [training.continue_run(r, r.tags['budget'] * spe2, protocol, main.phase2_seed_offset, phase=2) for r in paper_p1]
    tim_p2, _ = train_parallel(phase2 + paper_p2, backbone, ds_tr, trva, vindr.labels, protocol, n2, spe2, on_epoch_phase2, num_workers=nw)
    # ---- (5) dự đoán test MỘT lần (đầu sát bài: thêm các chế độ chấm của mã gốc) ----
    p_test = training.predict(phase2, backbone, data.eval_loader(ds_ev, te, bs, nw, pin))
    p_test_paper = {}
    for mode in (paper.predict_modes if paper_p2 else ()):
        bb_mode = dataclasses.replace(backbone, amp=torch.float16) if mode == 'train_fp16' else backbone
        p_test_paper[mode] = training.predict(paper_p2, bb_mode, data.eval_loader(ds_ev, te, bs, nw, pin), mode=('eval' if mode == 'eval' else 'train'))
    finite = bool(all(np.isfinite(p).all() for p in p_val + p_test + p_val_paper + sum(p_test_paper.values(), [])))
    steps_ok = all(r.step == r.max_steps for r in search + seeds + phase2 + arms + paper_p2)
    toc_do = dict(do_lr=_timing(tim_search, t_val['do_lr']), seed=_timing(tim_seed, t_val['seed']), pha2=_timing(tim_p2, 0.0),
                  phut_tong=(time.time() - t0) / 60, gpu_gb_dinh=(torch.cuda.max_memory_allocated() / 1e9 if dev == 'cuda' else None),
                  thiet_bi=runtime.device_name(dev), so_worker=nw, ram_gb_trong_dau_cuoi=[ram0[0], runtime.ram_gb()[0]], ram_gb_tong=ram0[1])
    log(f"\nxong: {'✓' if steps_ok else '⚠'} so buoc dung | {'✓' if finite else '⚠'} du doan val + test huu han | du doan test {len(p_test)} dau x "
        f"{p_test[0].shape} (CHUA cham — o notebook bang) | tong {toc_do['phut_tong']:.1f} phut | GPU dinh "
        + (f"{toc_do['gpu_gb_dinh']:.1f} GB" if toc_do['gpu_gb_dinh'] is not None else '-')
        + (f" | nhanh sat bai: {len(paper_p2)} dau pha 2, du doan test {list(p_test_paper)}" if paper_p2 else ''))
    arrays = {f'val_{t}_s{k}': p.astype(np.float32) for (t, k), p in zip(order, p_val)}
    arrays.update({f'test_{t}_s{k}': p.astype(np.float32) for (t, k), p in zip(order, p_test)})
    arrays.update({f'val_paper_{_paper_key(r)}': p.astype(np.float32) for r, p in zip(paper_p1, p_val_paper)})
    arrays.update({f'test_paper_{_paper_key(r)}_{mode}': p.astype(np.float32) for mode, ps in p_test_paper.items() for r, p in zip(paper_p2, ps)})
    arrays.update(y_val=vindr.labels[va].astype(np.float32), y_test=vindr.labels[te].astype(np.float32), idx_val=va.astype(np.int64),
                  idx_test=te.astype(np.int64))
    sp = backbone.spec
    result = dict(
        t=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'), run=run_tag, commit_chexfound=config.CHEXFOUND_COMMIT,
        goi_sha=getattr(vindr_protocol, 'GOI_SHA', None), goi_phien_ban=vindr_protocol.__version__, bb=backbone.name, nguon_bb=backbone.source,
        kho_anh=vindr.store, kieu_anh=data.STORE_LABELS[vindr.store], kich=sp.image_size, mean=[float(v) for v in sp.mean],
        std=[float(v) for v in sp.std], amp_backbone=str(backbone.amp), tf32_dau=True, luoi=list(protocol.lr_grid), lo=bs,
        chia=protocol.loss_divisor, wd=protocol.weight_decay, hat_du_lieu=protocol.data_seed, hat_dau=protocol.head_seed, so_seed=main.n_seeds, lech_hat_pha2=main.phase2_seed_offset,
        epoch=dict(pha1=main.epochs_phase1, pha2=main.epochs_phase2), buoc_moi_epoch=dict(pha1=spe1, pha2=spe2),
        tong_buoc=dict(pha1=n1, pha2=n2, moi_dau=n1 + n2), n_train=len(tr), n_val=len(va), n_test=len(te), md5_khoa=data.md5_keys(vindr.keys),
        md5_khoa_test=data.md5_keys([vindr.keys[i] for i in te]), nhan=list(vindr.label_names), lop_chinh=list(vindr.main_classes),
        do_lr={r.name: _curves(r) for r in search},
        chon={t: dict(lr=r.lr, auc=r.val_auroc[-1], ap=r.val_ap[-1], auprc=r.val_auprc[-1], auc21=r.val_auroc_main[-1],
                      mep_luoi=r.lr in (protocol.lr_grid[0], protocol.lr_grid[-1])) for t, r in best.items()},
        them_seed={r.name: _curves(r) for r in seeds}, duong_hoc=diag, chan_doan=dict(nguong_giam=curve.drop_threshold, cua_so_tang=curve.gain_window),
        pha2={f'{t}|s{k}': dict(hat=r.seed, loss=r.train_loss, buoc=r.step) for (t, k), r in zip(order, phase2)},
        kiem=dict(dac_trung_tang_cuoi=gate['lech'], dau_phan_ky_do_lr=[r.name for r in search if not np.isfinite(r.val_auroc).all()],
                  so_buoc_dung=steps_ok, du_doan_huu_han=finite),
        toc_do=toc_do, mang=sorted(arrays), nguon=vindr.sources,
        ghi_chu='PLAN_86 4.4: luot chinh; du doan test MOT lan sau pha 2, CHUA cham (notebook bang 86d).')
    if paper is not None:
        pick = lambda r, grid: dict(lr=r.lr, khoa=_paper_key(r), auc=r.val_auroc[-1], ap=r.val_ap[-1], auprc=r.val_auprc[-1], auc21=r.val_auroc_main[-1],
                                    mep_luoi=r.lr in (grid[0], grid[-1]))
        result['sat_bai'] = dict(
            cau_hinh=dict(ngan_sach=list(paper.budgets), loai_dau=list(paper.head_types), luoi_bai=list(paper.paper_grid), che_do=list(paper.predict_modes),
                          seed=0, buoc_moi_dau={str(E): dict(pha1=E * spe1, pha2=E * spe2) for E in paper.budgets}),
            do_lr={r.name: _curves(r) for r in arms},
            chon={f'E{E}|{t}': dict(g13=pick(c['g13'], protocol.lr_grid), g9=pick(c['g9'], paper.paper_grid), trung=c['g9'] is c['g13'])
                  for (E, t), c in paper_choice.items()},
            duong_hoc=paper_diag, pha2={_paper_key(r): dict(hat=r.seed, loss=r.train_loss, buoc=r.step) for r in paper_p2},
            ghi_chu='PLAN_86 11.3: nhanh SAT BAI CheXFound (hoc cung luot, CHI mo ta, khong vao luat 4.5); mang val_paper_<khoa> (eval) + '
                    'test_paper_<khoa>_<che do>.')
    if extra: result.update(extra)
    return result, arrays
