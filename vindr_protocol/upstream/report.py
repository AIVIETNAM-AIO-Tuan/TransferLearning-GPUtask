"""Chấm 86e (PLAN_86 §12.5) — CHỈ MÔ TẢ, không chọn gì theo test.

Dự đoán test của mã họ (--infer; xác suất sigmoid của họ) -> macro AP / AUROC 28 cột bằng torchmetrics 0.10.3 (đúng lớp + average='macro' như build_metric
của họ; cổng: trùng số họ in ở --infer), TB + CI95 của 1.000 lần bootstrap theo ảnh (họ không công bố mã bootstrap: rút ảnh có hoàn lại, hạt 8603 — cùng
dãy rút với ô C5 của 86b), kèm AP / AUROC 27 lớp, AUPRC hình thang, mAP 21 (hàm của gói). Đặt cạnh Bảng IV và nhánh sát bài E = 10 của 86b (cùng dữ liệu,
cùng lần rút -> hiệu ghép cặp mã họ − mã ta). Cổng chéo: số của hàng 86b tính bằng torchmetrics ở đây phải trùng số ô C5 của 86b (tính bằng hàm của gói)."""
import json
import warnings

import numpy as np

from .. import metrics
from ..paper_calibration import PAPER_TABLE_IV

N_BOOT, BOOT_SEED = 1000, 8603
RESULT_86B = ('stage4_nb86b', 'r86b_chexfound')
HEADS = ('lin', 'glori')
HEAD_TEXT = {'lin': 'Linear', 'glori': 'GLoRI'}


def tm_scores(preds, target):
    '''Macro AP / AUROC bằng torchmetrics 0.10.3: MultilabelAveragePrecision / MultilabelAUROC(num_labels, average='macro') như build_metric của họ
       (AP lớp không có ca dương -> NaN, bỏ; AUROC lớp đó -> 0, giữ). -> (ap, auroc).'''
    import torch
    p = torch.as_tensor(np.asarray(preds), dtype=torch.float32)
    t = torch.as_tensor(np.asarray(target), dtype=torch.long)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')                 # pkg_resources / lớp không có ca dương: cảnh báo của torchmetrics 0.10.3, không đổi số
        import torchmetrics
        from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
        assert torchmetrics.__version__ == '0.10.3', f'torchmetrics {torchmetrics.__version__} != 0.10.3 (ban CheXFound ghim) -> DUNG'
        ap = MultilabelAveragePrecision(num_labels=p.shape[1], average='macro')(p, t)
        auroc = MultilabelAUROC(num_labels=p.shape[1], average='macro')(p, t)
    return float(ap), float(auroc)


def bootstrap_indices(n_images, n_boot=N_BOOT, seed=BOOT_SEED):
    '''Dãy rút giống metrics.bootstrap_weights (cùng hạt -> cùng ảnh từng lần rút): mỗi lần n ảnh có hoàn lại.'''
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_images, n_images) for _ in range(n_boot)]


def score_row(preds, target, main_idx, boot_idx):
    '''Một hàng: 28 cột (torchmetrics) toàn test + TB / CI95 bootstrap; 27 lớp, AUPRC hình thang 27, mAP 21 (hàm của gói). + mảng bootstrap (ghép cặp).'''
    ap28, au28 = tm_scores(preds, target)
    boot = np.array([tm_scores(preds[i], target[i]) for i in boot_idx])
    ap_c, _, au_c, _ = metrics.ap_auroc_tm0103(target, preds)
    au_valid = np.array([v if 0 < target[:, j].sum() < len(target) else np.nan for j, v in enumerate(au_c)])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    row = dict(ap28=ap28, auroc28=au28, ap28_tb=float(boot[:, 0].mean()), ap28_ci=ci(boot[:, 0]), auroc28_tb=float(boot[:, 1].mean()),
               auroc28_ci=ci(boot[:, 1]), ap27=float(np.nanmean(ap_c)), auroc27=float(np.nanmean(au_valid)),
               auprc27=float(metrics.auprc_trapezoid(target, preds)[1]), ap21=float(np.nanmean(ap_c[main_idx])))
    return row, boot


def load_86b_rows(kiem_ke, tai, xung_dot, vindr, budget=10, mode='train_fp16', log=print):
    '''Dự đoán test của nhánh sát bài 86b (E = budget, lưới 13, chế độ chấm `mode`) từ Drive; cổng: chỉ số + nhãn test trùng dữ liệu 86e.'''
    kk = kiem_ke(RESULT_86B, mau=r'((luot|hieu_chinh)_86b_chexfound_\d{8}_\d{6}\.json|du_doan_86b_chexfound_\d{8}_\d{6}\.npz)')[1]
    js = sorted(t for t in kk if t.startswith('luot_') and t.endswith('.json'))
    assert js, f"chua co ket qua 86b_chexfound trong {'/'.join(RESULT_86B)} -> DUNG"
    if xung_dot(kk[js[-1]]): log(f'  ! {js[-1]}: nhieu ban NOI DUNG KHAC -> dung ban MOI NHAT')
    with open(tai(kk[js[-1]][0]), encoding='utf-8') as f: meta = json.load(f)
    npz, hc = meta['tep_du_doan'], f"hieu_chinh_86b_chexfound_{meta['t']}.json"
    assert npz in kk and hc in kk, f'thieu {npz} / {hc} -> DUNG'
    with open(tai(kk[hc][0]), encoding='utf-8') as f: c5 = {r['khoa']: r for r in json.load(f)['hang']}
    te = vindr.indices('test')
    with np.load(tai(kk[npz][0])) as z:
        assert np.array_equal(z['idx_test'], te) and np.array_equal(z['y_test'], vindr.labels[te].astype(np.float32)), (
            '86b: thu tu / nhan test KHAC du lieu 86e -> DUNG')
        rows = {}
        for t in HEADS:
            c = meta['sat_bai']['chon'][f'E{budget}|{t}']['g13']
            rows[t] = dict(probs=z[f"test_paper_{c['khoa']}_{mode}"].astype(np.float64), lr=c['lr'], khoa=c['khoa'],
                           c5=c5[f'sat_E{budget}_g13_{mode}_{t}'])
    return rows, dict(json=js[-1], npz=npz, hieu_chinh=hc, ngan_sach=budget, che_do=mode)


def run(results, vindr, kiem_ke, tai, xung_dot, n_boot=N_BOOT, seed=BOOT_SEED, log=print):
    '''results = {'lin': run_head(...), 'glori': run_head(...)} của runner. Chấm, in bảng, trả dict ghi JSON được.'''
    te = vindr.indices('test')
    target = vindr.labels[te].astype(np.int64)
    main_idx = [vindr.label_names.index(c) for c in vindr.main_classes]
    boot_idx = bootstrap_indices(len(te), n_boot, seed)
    ours_86b, src_86b = load_86b_rows(kiem_ke, tai, xung_dot, vindr, log=log)
    out = dict(n_boot=n_boot, hat=seed, bang_iv=PAPER_TABLE_IV, nguon_86b=src_86b, hang={}, hieu={}, ket_luan={}, cong={})
    for t in HEADS:
        r = results[t]
        with np.load(r['infer']['du_doan']) as z:
            preds, tgt = z['preds'].astype(np.float64), z['target'].astype(np.int64)
        assert preds.shape == target.shape and np.array_equal(tgt, target), f'{t}: target --infer KHAC nhan test (thu tu CSV) -> DUNG'
        row_up, boot_up = score_row(preds, target, main_idx, boot_idx)
        row_86b, boot_86b = score_row(ours_86b[t]['probs'], target, main_idx, boot_idx)
        printed = r['infer']['chi_so_ho_in']
        c5 = ours_86b[t]['c5']
        lech_c5 = max(abs(row_86b[k] - c5[k]) for k in ('ap28', 'auroc28', 'ap28_tb', 'auroc28_tb'))
        out['cong'][t] = dict(ap_ho_in=printed['ap'], auroc_ho_in=printed['auroc'], ap_tinh_lai=row_up['ap28'], auroc_tinh_lai=row_up['auroc28'],
                              trung=bool(abs(printed['ap'] - row_up['ap28']) < 5e-5 and abs(printed['auroc'] - row_up['auroc28']) < 5e-5),
                              lech_86b_c5=float(lech_c5), trung_86b_c5=bool(lech_c5 < 1e-4))
        d = boot_up - boot_86b
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out['hang'][t] = dict(ma_goc=dict(**row_up, lr=r['chon']['lr'], auroc_val=r['chon']['auroc_val'], hai_nua=r['hai_nua']),
                              ma_ta_86b=dict(**row_86b, lr=ours_86b[t]['lr']))
        out['hieu'][t] = dict(ap28=row_up['ap28'] - row_86b['ap28'], ap28_ci=ci(d[:, 0]), auroc28=row_up['auroc28'] - row_86b['auroc28'],
                              auroc28_ci=ci(d[:, 1]))
        p = PAPER_TABLE_IV[t]
        out['ket_luan'][t] = dict(ap_bai_trong_ci=bool(row_up['ap28_ci'][0] <= p['ap'] <= row_up['ap28_ci'][1]),
                                  auroc_bai_trong_ci=bool(row_up['auroc28_ci'][0] <= p['auroc'] <= row_up['auroc28_ci'][1]))
        out['ket_luan'][t]['tai_tao_duoc'] = out['ket_luan'][t]['ap_bai_trong_ci'] and out['ket_luan'][t]['auroc_bai_trong_ci']
    _print(out, log)
    return out


def _print(out, log):
    fm = lambda v: f'{v:.3f}'
    fci = lambda c: f'[{c[0]:.3f}; {c[1]:.3f}]'
    log('=== C8 · MA GOC CheXFound (86e) vs Bang IV vs ma ta (86b nhanh sat bai E = 10, che do dropout bat + fp16) — test 3.000 anh, 28 cot torchmetrics '
        f"0.10.3; TB + CI95 = {out['n_boot']} lan rut anh (hat {out['hat']}); CHI MO TA ===")
    for t in HEADS:
        h, p, c = out['hang'][t], PAPER_TABLE_IV[t], out['cong'][t]
        log(f"\n--- {HEAD_TEXT[t]} --- cong: so ho in o --infer AP {c['ap_ho_in']:.4f} / AUROC {c['auroc_ho_in']:.4f} | tinh lai {c['ap_tinh_lai']:.4f} / "
            f"{c['auroc_tinh_lai']:.4f} -> {'TRUNG' if c['trung'] else '⚠ LECH'} | hang 86b tinh bang torchmetrics - o C5 86b (ham cua goi): lech max "
            f"{c['lech_86b_c5']:.1e} -> {'TRUNG' if c['trung_86b_c5'] else '⚠ LECH'}")
        log(f"{'hang (dieu kien)':<78}{'AP28':>7}{'AP28 TB [CI95]':>24}{'AUROC28':>9}{'AUROC28 TB [CI95]':>24}{'AP27':>7}{'AUROC27':>9}{'AUPRC27':>9}{'mAP21':>7}")
        log(f"{'BAI Bang IV':<78}{'':>7}{fm(p['ap']) + ' ' + fci(p['ap_ci']):>24}{'':>9}{fm(p['auroc']) + ' ' + fci(p['auroc_ci']):>24}")
        for key, text in (('ma_goc', f"MA GOC linear_glori.py, lr {h['ma_goc']['lr']:g} (val AUROC {h['ma_goc']['auroc_val']:.4f}, dropout bat)"
                                     + (', 2 nua luoi' if h['ma_goc']['hai_nua'] else '')),
                          ('ma_ta_86b', f"MA TA 86b nhanh sat bai E=10, lr {h['ma_ta_86b']['lr']:g}")):
            x = h[key]
            log(f"{text[:77]:<78}{fm(x['ap28']):>7}{fm(x['ap28_tb']) + ' ' + fci(x['ap28_ci']):>24}{fm(x['auroc28']):>9}"
                f"{fm(x['auroc28_tb']) + ' ' + fci(x['auroc28_ci']):>24}{fm(x['ap27']):>7}{fm(x['auroc27']):>9}{fm(x['auprc27']):>9}{fm(x['ap21']):>7}")
        d, k = out['hieu'][t], out['ket_luan'][t]
        log(f"hieu ma goc - ma ta (ghep cap, cung lan rut): AP28 {d['ap28']:+.3f} {fci(d['ap28_ci'])} | AUROC28 {d['auroc28']:+.3f} {fci(d['auroc28_ci'])}")
        log(f"tieu chi (dat truoc, PLAN_86 12.5): AP bai trong CI ma goc {'CO' if k['ap_bai_trong_ci'] else 'KHONG'} | AUROC bai trong CI "
            f"{'CO' if k['auroc_bai_trong_ci'] else 'KHONG'} -> {'TAI TAO DUOC' if k['tai_tao_duoc'] else 'CHUA tai tao'}")
