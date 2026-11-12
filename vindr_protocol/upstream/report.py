"""Chấm 86e / 86f / 86g (PLAN_86 §12.5, §13.2, §14.1) — CHỈ MÔ TẢ, không chọn gì theo test. 86f: + hàng 86e; 86g (ảnh 224): + hàng 86f (ghép cặp
86g − 86f = ảnh hưởng của kích thước ảnh ở Linear) + hàng GLoRI "86f nửa 1" đọc từ nhật ký GLoRI của 86f.

Dự đoán test của mã họ = lần chấm test CUỐI của pha 2 trong lượt học (xác suất sigmoid của họ, chép đúng như vừa vào torchmetrics; Tuấn 12/11: không
chạy --infer riêng) -> macro AP / AUROC 28 cột bằng torchmetrics 0.10.3 (đúng lớp + average='macro' như build_metric của họ; cổng: AUROC tính lại = số họ
tính ở chính lần chấm đó — mã họ chỉ tính AUROC lúc chấm (multilabel_auroc mặc định), nên AP không có số đối chiếu), TB + CI95 của 1.000 lần bootstrap
theo ảnh (họ không công bố mã bootstrap: rút ảnh có hoàn lại, hạt 8603 — cùng dãy rút với ô C5 của 86b), kèm AP / AUROC 27 lớp, AUPRC hình thang, mAP 21
(hàm của gói). Đặt cạnh Bảng IV và nhánh sát bài E = 10 của 86b (cùng dữ liệu, cùng lần rút -> hiệu ghép cặp mã họ − mã ta). Cổng chéo: số của hàng 86b
tính bằng torchmetrics ở đây phải trùng số ô C5 của 86b (tính bằng hàm của gói)."""
import json
import re
import warnings

import numpy as np

from .. import metrics
from ..paper_calibration import PAPER_TABLE_IV

N_BOOT, BOOT_SEED = 1000, 8603
RESULT_86B = ('stage4_nb86b', 'r86b_chexfound')
RESULT_86E = ('stage4_nb86e', 'r86e_chexfound_ma_goc')      # mã họ, thư viện mới, G4 — hàng so sánh của 86f (PLAN_86 §13.2)
RESULT_86F = ('stage4_nb86f', 'r86f_chexfound_moi_truong_ho')   # mã họ, môi trường thư viện của họ, A100, ảnh 512 — hàng so sánh của 86g (§14.1)
_FILE_REF = r'(luot|du_doan)_{label}_(lin|glori)_(\d{{8}}_\d{{6}})\.(json|npz)'
_FILE_86E = re.compile(_FILE_REF.format(label='86e'))
_NK_86F = re.compile(r'nhat_ky_86f_glori_(\d{8}_\d{6})\.tar\.gz')                 # nhật ký GLoRI của 86f (thư mục lượt chạy, không .pth)
_NK_MEMBER = re.compile(r'glori_hoc_nua[12]/cham/[A-Za-z0-9_]+\.(json|npz)')      # CHỈ tệp lần chấm của hai nửa lưới
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


def load_ref_rows(kiem_ke, tai, xung_dot, vindr, label, folder, log=print):
    '''Dự đoán test của một lượt MÃ GỐC khác (`label` = '86e' / '86f') từ Drive: lượt MỚI NHẤT có đủ JSON + npz của cả hai loại đầu. Chưa có ->
       (None, None). Cổng: nhãn test trong npz = nhãn test của dữ liệu này (cùng thứ tự).'''
    pat = re.compile(_FILE_REF.format(label=label))
    kk = kiem_ke(folder, mau=pat.pattern)[1]
    have = {}
    for name in kk:
        m = pat.fullmatch(name)
        if m: have.setdefault(m.group(3), set()).add((m.group(1), m.group(2)))
    need = {(k, h) for k in ('luot', 'du_doan') for h in HEADS}
    full = sorted(t for t, s in have.items() if s >= need)
    if not full:
        log(f"  {label}: chua co ket qua du hai loai dau trong {'/'.join(folder)} -> KHONG co hang {label} (so sau o bang CPU)")
        return None, None
    t = full[-1]
    te = vindr.indices('test')
    rows = {}
    for h in HEADS:
        js, npz = f'luot_{label}_{h}_{t}.json', f'du_doan_{label}_{h}_{t}.npz'
        for n in (js, npz):
            if xung_dot(kk[n]): log(f'  ! {n}: nhieu ban NOI DUNG KHAC -> dung ban MOI NHAT')
        with open(tai(kk[js][0]), encoding='utf-8') as f: meta = json.load(f)
        with np.load(tai(kk[npz][0])) as z:
            probs, tgt = z['preds'].astype(np.float64), z['target'].astype(np.int64)
        assert np.array_equal(tgt, vindr.labels[te].astype(np.int64)), f'{label}: thu tu / nhan test KHAC du lieu nay -> DUNG'
        rows[h] = dict(probs=probs, lr=meta['chon']['lr'], anh=meta.get('anh', 512), hai_nua=bool(meta.get('hai_nua')))
    return rows, dict(t=t, luot_khac=[x for x in full if x != t])


def load_86e_rows(kiem_ke, tai, xung_dot, vindr, log=print):
    '''= load_ref_rows của 86e (mã họ, thư viện mới, G4) — hàng so sánh của 86f.'''
    return load_ref_rows(kiem_ke, tai, xung_dot, vindr, '86e', RESULT_86E, log=log)


def load_86f_halves(kiem_ke, tai, xung_dot, vindr, log=print):
    '''86f GLoRI chạy hai nửa lưới (9 lr cùng lúc tràn A100): từ tệp nhật ký GLoRI MỚI NHẤT trên Drive (thư mục lượt chạy, không .pth) -> (a) lần
       chấm test CUỐI của pha 2 NỬA 1 = lr val chọn TRONG nửa 1 (KHÔNG phải lr cuối của 86f; chỉ mô tả, không chọn gì theo test) + (b) AUROC val (số họ
       tính) từng lr ở lần chấm val cuối pha 1 của mỗi nửa (lựa chọn lr sát nhau đến đâu). Chỉ giải nén tệp lần chấm của hai nửa (tên theo khuôn).
       Cổng: lần chấm test cuối MỘT đầu; target = nhãn test (cùng thứ tự); AUROC tính lại = số họ tính. Chưa có -> (None, None).'''
    import os
    import tarfile
    import tempfile
    from .runner import load_evaluation, lr_from_name, read_evaluations
    kk = kiem_ke(RESULT_86F, mau=_NK_86F.pattern)[1]
    names = sorted(n for n in kk if _NK_86F.fullmatch(n))
    if not names:
        log(f"  86f: chua co nhat ky GLoRI trong {'/'.join(RESULT_86F)} -> KHONG co hang 86f nua 1")
        return None, None
    name = names[-1]
    if xung_dot(kk[name]): log(f'  ! {name}: nhieu ban NOI DUNG KHAC -> dung ban MOI NHAT')
    d = tempfile.mkdtemp(prefix='nhat_ky_86f_')
    with tarfile.open(tai(kk[name][0]), 'r:gz') as tf:
        members = [m for m in tf.getmembers() if m.isfile() and _NK_MEMBER.fullmatch(m.name)]
        assert members, f'{name}: khong co tep lan cham cua hai nua luoi (86f khong chay hai nua?) -> DUNG'
        for m in members:
            try:
                tf.extract(m, d, filter='data')
            except TypeError:                                   # Python cũ: extract chưa có filter (tên đã lọc theo khuôn, không '..' / tuyệt đối)
                tf.extract(m, d)
    val = {}
    for h in ('nua1', 'nua2'):
        recs = [r for r in read_evaluations(os.path.join(d, f'glori_hoc_{h}')) if r['tach'] == 'val']
        assert recs, f'{name}: {h} khong co lan cham val -> DUNG'
        val[h] = {f'{lr_from_name(n):g}': float(recs[-1]['so_ho_tinh'][n]['auroc']) for n in recs[-1]['ten']}
    test = [r for r in read_evaluations(os.path.join(d, 'glori_hoc_nua1')) if r['tach'] == 'test']
    assert test and len(test[-1]['ten']) == 1, f'{name}: nua 1 thieu lan cham test cuoi MOT dau (pha 2) -> DUNG'
    target, preds = load_evaluation(test[-1])
    te = vindr.indices('test')
    assert np.array_equal(target, vindr.labels[te].astype(np.int64)), '86f nua 1: target lan cham test KHAC nhan test (thu tu) -> DUNG'
    n1 = test[-1]['ten'][0]
    auc_ho = float(test[-1]['so_ho_tinh'][n1]['auroc'])
    auc_lai = tm_scores(preds[n1], target)[1]
    assert abs(auc_ho - auc_lai) < 5e-5, f'86f nua 1: AUROC tinh lai {auc_lai:.6f} != so ho tinh {auc_ho:.6f} -> DUNG'
    row = dict(probs=preds[n1].astype(np.float64), lr=lr_from_name(n1), auroc_ho_tinh=auc_ho, iteration=test[-1]['iteration'])
    return row, dict(tep=name, val_auroc=val, so_tep_cham=len(members))


def run(results, vindr, kiem_ke, tai, xung_dot, n_boot=N_BOOT, seed=BOOT_SEED, compare_86e=False, label='86e', compare_with=None, half_86f=False,
        log=print):
    '''results = {'lin': run_head(...), 'glori': run_head(...)} của runner. Chấm, in bảng, trả dict ghi JSON được. compare_86e (86f): thêm hàng 86e (G4,
       thư viện mới) từ Drive + hiệu ghép cặp {label} − 86e; 86e chưa chạy -> ghi chú, không lỗi. compare_with = (nhãn, thư mục Drive, mô tả) (86g:
       ('86f', RESULT_86F, ...)): hàng của lượt mã gốc đó + hiệu ghép cặp {label} − nhãn (khoá 'ref' / 'hieu_ref'). half_86f (86g): thêm hàng GLoRI
       "86f nửa 1" (lr val chọn trong nửa 1) + hiệu ghép cặp với 86b + AUROC val từng lr của hai nửa (load_86f_halves) — CHỈ mô tả.'''
    te = vindr.indices('test')
    target = vindr.labels[te].astype(np.int64)
    main_idx = [vindr.label_names.index(c) for c in vindr.main_classes]
    boot_idx = bootstrap_indices(len(te), n_boot, seed)
    ours_86b, src_86b = load_86b_rows(kiem_ke, tai, xung_dot, vindr, log=log)
    rows_86e, src_86e = load_86e_rows(kiem_ke, tai, xung_dot, vindr, log=log) if compare_86e else (None, None)
    rows_ref, src_ref = (load_ref_rows(kiem_ke, tai, xung_dot, vindr, compare_with[0], compare_with[1], log=log) if compare_with else (None, None))
    half, src_half = load_86f_halves(kiem_ke, tai, xung_dot, vindr, log=log) if half_86f else (None, None)
    out = dict(n_boot=n_boot, hat=seed, nhan=label, bang_iv=PAPER_TABLE_IV, nguon_86b=src_86b, nguon_86e=src_86e, so_86e=bool(compare_86e), hang={},
               hieu={}, hieu_86e={}, ket_luan={}, cong={}, ref=(list(compare_with) if compare_with else None), nguon_ref=src_ref, hieu_ref={},
               so_nua1_86f=bool(half_86f), nguon_nua1_86f=src_half, hieu_nua1_86f={})
    for t in HEADS:
        r = results[t]
        with np.load(r['du_doan']) as z:
            preds, tgt = z['preds'].astype(np.float64), z['target'].astype(np.int64)
        assert preds.shape == target.shape and np.array_equal(tgt, target), f'{t}: target lan cham test KHAC nhan test (thu tu CSV) -> DUNG'
        row_up, boot_up = score_row(preds, target, main_idx, boot_idx)
        row_86b, boot_86b = score_row(ours_86b[t]['probs'], target, main_idx, boot_idx)
        printed = r['test_ho_ghi_cuoi_pha2']['ket_qua']['auroc']             # = số họ tính ở lần chấm test cuối (cổng check_run của runner)
        c5 = ours_86b[t]['c5']
        lech_c5 = max(abs(row_86b[k] - c5[k]) for k in ('ap28', 'auroc28', 'ap28_tb', 'auroc28_tb'))
        out['cong'][t] = dict(auroc_ho_tinh=float(printed), auroc_tinh_lai=row_up['auroc28'], ap_tinh_lai=row_up['ap28'],
                              trung=bool(abs(printed - row_up['auroc28']) < 5e-5 and r['cong']['test']['dat'] and r['cong']['val']['dat']),
                              cong_val=r['cong']['val'], lech_86b_c5=float(lech_c5), trung_86b_c5=bool(lech_c5 < 1e-4))
        d = boot_up - boot_86b
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out['hang'][t] = dict(ma_goc=dict(**row_up, lr=r['chon']['lr'], auroc_val=r['chon']['auroc_val'], hai_nua=r['hai_nua'], anh=r.get('anh', 512)),
                              ma_ta_86b=dict(**row_86b, lr=ours_86b[t]['lr']))
        out['hieu'][t] = dict(ap28=row_up['ap28'] - row_86b['ap28'], ap28_ci=ci(d[:, 0]), auroc28=row_up['auroc28'] - row_86b['auroc28'],
                              auroc28_ci=ci(d[:, 1]))
        if rows_86e is not None:
            row_e, boot_e = score_row(rows_86e[t]['probs'], target, main_idx, boot_idx)
            out['hang'][t]['g4_86e'] = dict(**row_e, lr=rows_86e[t]['lr'])
            de = boot_up - boot_e
            out['hieu_86e'][t] = dict(ap28=row_up['ap28'] - row_e['ap28'], ap28_ci=ci(de[:, 0]), auroc28=row_up['auroc28'] - row_e['auroc28'],
                                      auroc28_ci=ci(de[:, 1]))
        if rows_ref is not None:
            row_r, boot_r = score_row(rows_ref[t]['probs'], target, main_idx, boot_idx)
            out['hang'][t]['ref'] = dict(**row_r, lr=rows_ref[t]['lr'], anh=rows_ref[t]['anh'], hai_nua=rows_ref[t]['hai_nua'])
            dr = boot_up - boot_r
            out['hieu_ref'][t] = dict(ap28=row_up['ap28'] - row_r['ap28'], ap28_ci=ci(dr[:, 0]), auroc28=row_up['auroc28'] - row_r['auroc28'],
                                      auroc28_ci=ci(dr[:, 1]))
        if half is not None and t == 'glori':
            row_h, boot_h = score_row(half['probs'], target, main_idx, boot_idx)
            out['hang'][t]['nua1_86f'] = dict(**row_h, lr=half['lr'], auroc_ho_tinh=half['auroc_ho_tinh'])
            dh = boot_h - boot_86b
            out['hieu_nua1_86f'][t] = dict(ap28=row_h['ap28'] - row_86b['ap28'], ap28_ci=ci(dh[:, 0]), auroc28=row_h['auroc28'] - row_86b['auroc28'],
                                           auroc28_ci=ci(dh[:, 1]), lr_86b=ours_86b[t]['lr'], cung_lr=bool(abs(half['lr'] - ours_86b[t]['lr']) < 1e-12))
        p = PAPER_TABLE_IV[t]
        out['ket_luan'][t] = dict(ap_bai_trong_ci=bool(row_up['ap28_ci'][0] <= p['ap'] <= row_up['ap28_ci'][1]),
                                  auroc_bai_trong_ci=bool(row_up['auroc28_ci'][0] <= p['auroc'] <= row_up['auroc28_ci'][1]))
        out['ket_luan'][t]['tai_tao_duoc'] = out['ket_luan'][t]['ap_bai_trong_ci'] and out['ket_luan'][t]['auroc_bai_trong_ci']
    _print(out, log)
    return out


def _print(out, log):
    fm = lambda v: f'{v:.3f}'
    fci = lambda c: f'[{c[0]:.3f}; {c[1]:.3f}]'
    ref = out.get('ref')
    log(f"=== C8 · MA GOC CheXFound ({out['nhan']}) vs Bang IV vs ma ta (86b nhanh sat bai E = 10, che do dropout bat + fp16)"
        + (' vs 86e (ma goc, thu vien moi, G4)' if out['so_86e'] else '') + (f' vs {ref[0]} ({ref[2]})' if ref else '')
        + ' — test 3.000 anh, 28 cot torchmetrics '
        f"0.10.3; TB + CI95 = {out['n_boot']} lan rut anh (hat {out['hat']}); CHI MO TA ===")
    if out['so_86e'] and out['nguon_86e'] is None:
        log('86e (G4) chua co ket qua tren Drive -> chua so duoc moi truong thu vien (so sau o bang CPU)')
    if ref and out.get('nguon_ref') is None:
        log(f'{ref[0]} chua co ket qua tren Drive -> chua co hieu ghep cap {out["nhan"]} - {ref[0]}')
    if out.get('so_nua1_86f') and out.get('nguon_nua1_86f') is None:
        log('86f: chua co nhat ky GLoRI hai nua tren Drive -> chua co hang 86f nua 1')
    for t in HEADS:
        h, p, c = out['hang'][t], PAPER_TABLE_IV[t], out['cong'][t]
        cv = c['cong_val']
        log(f"\n--- {HEAD_TEXT[t]} --- cong: AUROC test ho tinh o lan cham cuoi pha 2 {c['auroc_ho_tinh']:.4f} | tinh lai tu du doan da chep "
            f"{c['auroc_tinh_lai']:.4f} -> {'TRUNG' if c['trung'] else '⚠ LECH'} (AP: ma ho khong tinh luc cham -> chi so tinh lai) | val cuoi pha 1: "
            f"{cv['so_dau']} lr, AUROC tinh lai - ho tinh lech max {cv['lech_max']:.1e}, lr ho chon {'DUNG' if cv['chon_dung_luat'] else '⚠ SAI'} luat "
            f"AUROC lon nhat{'' if cv['chon_trung_tinh_lai'] else ' (⚠ khac lr lon nhat theo so tinh lai — gan hoa)'} | hang 86b tinh bang torchmetrics - "
            f"o C5 86b (ham cua goi): lech max {c['lech_86b_c5']:.1e} -> {'TRUNG' if c['trung_86b_c5'] else '⚠ LECH'}")
        log(f"{'hang (dieu kien)':<78}{'AP28':>7}{'AP28 TB [CI95]':>24}{'AUROC28':>9}{'AUROC28 TB [CI95]':>24}{'AP27':>7}{'AUROC27':>9}{'AUPRC27':>9}{'mAP21':>7}")
        log(f"{'BAI Bang IV':<78}{'':>7}{fm(p['ap']) + ' ' + fci(p['ap_ci']):>24}{'':>9}{fm(p['auroc']) + ' ' + fci(p['auroc_ci']):>24}")
        mg = h['ma_goc']
        rows_txt = [('ma_goc', f"MA GOC linear_glori.py ({out['nhan']})" + (f", anh {mg['anh']}" if mg.get('anh', 512) != 512 else '')
                                + f", lr {mg['lr']:g} (val AUROC {mg['auroc_val']:.4f}, dropout bat)" + (', 2 nua luoi' if mg['hai_nua'] else ''))]
        if 'g4_86e' in h:
            rows_txt.append(('g4_86e', f"MA GOC 86e (thu vien moi, G4), lr {h['g4_86e']['lr']:g}"))
        if 'ref' in h:
            rows_txt.append(('ref', f"MA GOC {ref[0]} (anh {h['ref']['anh']}), lr {h['ref']['lr']:g}" + (', 2 nua luoi' if h['ref']['hai_nua'] else '')))
        rows_txt.append(('ma_ta_86b', f"MA TA 86b nhanh sat bai E=10, lr {h['ma_ta_86b']['lr']:g}"))
        if 'nua1_86f' in h:
            rows_txt.append(('nua1_86f', f"MA GOC 86f NUA 1 luoi, lr {h['nua1_86f']['lr']:g} (lr val chon TRONG nua 1, KHONG phai lr cuoi; mo ta)"))
        for key, text in rows_txt:
            x = h[key]
            log(f"{text[:77]:<78}{fm(x['ap28']):>7}{fm(x['ap28_tb']) + ' ' + fci(x['ap28_ci']):>24}{fm(x['auroc28']):>9}"
                f"{fm(x['auroc28_tb']) + ' ' + fci(x['auroc28_ci']):>24}{fm(x['ap27']):>7}{fm(x['auroc27']):>9}{fm(x['auprc27']):>9}{fm(x['ap21']):>7}")
        d, k = out['hieu'][t], out['ket_luan'][t]
        log(f"hieu ma goc - ma ta (ghep cap, cung lan rut): AP28 {d['ap28']:+.3f} {fci(d['ap28_ci'])} | AUROC28 {d['auroc28']:+.3f} {fci(d['auroc28_ci'])}")
        if t in out['hieu_86e']:
            e = out['hieu_86e'][t]
            log(f"hieu {out['nhan']} - 86e (ghep cap; = anh huong GOP moi truong thu vien + GPU, cung ma, cung du lieu): AP28 {e['ap28']:+.3f} "
                f"{fci(e['ap28_ci'])} | AUROC28 {e['auroc28']:+.3f} {fci(e['auroc28_ci'])}")
        if t in out.get('hieu_ref', {}):
            e = out['hieu_ref'][t]
            log(f"hieu {out['nhan']} - {ref[0]} (ghep cap, cung lan rut; {ref[2]}): AP28 {e['ap28']:+.3f} {fci(e['ap28_ci'])} | AUROC28 "
                f"{e['auroc28']:+.3f} {fci(e['auroc28_ci'])}")
        if t in out.get('hieu_nua1_86f', {}):
            e = out['hieu_nua1_86f'][t]
            log(f"hieu 86f nua 1 - ma ta 86b (ghep cap; {'CUNG' if e['cung_lr'] else 'KHAC'} lr: {h['nua1_86f']['lr']:g} vs {e['lr_86b']:g}; tach phan "
                f"lech GLoRI ma ho - ma ta do chon lr): AP28 {e['ap28']:+.3f} {fci(e['ap28_ci'])} | AUROC28 {e['auroc28']:+.3f} {fci(e['auroc28_ci'])}")
            va = out['nguon_nua1_86f']['val_auroc']
            log('AUROC val lan cham cuoi pha 1 cua 86f (so ho tinh, dropout bat) theo lr: '
                + ' | '.join(f"{n}: " + ', '.join(f'{lr} {v:.4f}' for lr, v in va[n].items()) for n in ('nua1', 'nua2')))
        log(f"tieu chi (dat truoc, PLAN_86 12.5): AP bai trong CI ma goc {'CO' if k['ap_bai_trong_ci'] else 'KHONG'} | AUROC bai trong CI "
            f"{'CO' if k['auroc_bai_trong_ci'] else 'KHONG'} -> {'TAI TAO DUOC' if k['tai_tao_duoc'] else 'CHUA tai tao'}")


# ===================================================================================== 86j (PLAN_86 §16): encoder KHÁC qua mã + môi trường của họ
PAPER_TABLE_IV_ENC = {          # bài CheXFound Bảng IV, VinDr-CXR (AUPRC = AP bậc thang của torchmetrics; TB bootstrap — suy luận §11.6)
    'raddino': {'lin': dict(ap=0.102, ap_ci=(0.096, 0.107), auroc=0.477, auroc_ci=(0.441, 0.507)),
                'glori': dict(ap=0.155, ap_ci=(0.147, 0.163), auroc=0.680, auroc_ci=(0.630, 0.718))},
    'biomedclip': {'lin': dict(ap=0.100, ap_ci=(0.094, 0.106), auroc=0.587, auroc_ci=(0.544, 0.618)),
                   'glori': dict(ap=0.094, ap_ci=(0.089, 0.102), auroc=0.537, auroc_ci=(0.497, 0.566))}}
ENCODER_PAPER_KEY = {'raddino_kieu_ho': 'raddino', 'raddino_chuan': 'raddino', 'biomedclip_ho': 'biomedclip'}
ENCODER_TEXT = {'raddino_kieu_ho': 'RAD-DINO (a) kieu lop boc HF cua ho: hidden_states tho, chuan hoa ImageNet, anh 512',
                'raddino_chuan': 'RAD-DINO (b) chuan: qua layernorm cuoi, chuan hoa RAD-DINO, anh 518',
                'biomedclip_ho': 'BiomedCLIP lop boc BiomedCLIPBase cua ho (vector 512), anh 224'}


def run_encoder(results, vindr, encoder, kiem_ke, tai, xung_dot, label='86j', n_boot=N_BOOT, seed=BOOT_SEED, log=print):
    """86j (§16.5) — CHỈ MÔ TẢ. results = {đầu: run_head(...)} của encoder (BiomedCLIP: chỉ 'lin'). Chấm như run (28 cột torchmetrics 0.10.3, TB +
       CI95 bootstrap), cạnh Bảng IV của CHÍNH encoder đó (+ hàng CheXFound của bài, bối cảnh) và CheXFound 86f (hiệu ghép cặp encoder − CheXFound, cùng
       mã / môi trường / GPU / dữ liệu / lần rút). "Tái tạo số bài" = AUPRC (= AP bậc thang) VÀ AUROC của Bảng IV cùng trong CI95 của hàng."""
    te = vindr.indices('test')
    target = vindr.labels[te].astype(np.int64)
    main_idx = [vindr.label_names.index(c) for c in vindr.main_classes]
    boot_idx = bootstrap_indices(len(te), n_boot, seed)
    rows_86f, src_86f = load_ref_rows(kiem_ke, tai, xung_dot, vindr, '86f', RESULT_86F, log=log)
    paper = PAPER_TABLE_IV_ENC[ENCODER_PAPER_KEY[encoder]]
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = dict(n_boot=n_boot, hat=seed, nhan=label, encoder=encoder, bang_iv=paper, bang_iv_chexfound=PAPER_TABLE_IV, nguon_86f=src_86f, hang={},
               hieu_86f={}, ket_luan={}, cong={})
    for t, r in results.items():
        with np.load(r['du_doan']) as z:
            preds, tgt = z['preds'].astype(np.float64), z['target'].astype(np.int64)
        assert preds.shape == target.shape and np.array_equal(tgt, target), f'{t}: target lan cham test KHAC nhan test (thu tu CSV) -> DUNG'
        assert r.get('encoder') == encoder, f"{t}: luot hoc encoder {r.get('encoder')!r} != {encoder!r} -> DUNG"
        row, boot = score_row(preds, target, main_idx, boot_idx)
        printed = r['test_ho_ghi_cuoi_pha2']['ket_qua']['auroc']
        out['cong'][t] = dict(auroc_ho_tinh=float(printed), auroc_tinh_lai=row['auroc28'], cong_val=r['cong']['val'],
                              trung=bool(abs(printed - row['auroc28']) < 5e-5 and r['cong']['test']['dat'] and r['cong']['val']['dat']))
        out['hang'][t] = dict(encoder=dict(**row, lr=r['chon']['lr'], auroc_val=r['chon']['auroc_val'], hai_nua=r['hai_nua'], anh=r.get('anh')))
        if rows_86f is not None:
            row_f, boot_f = score_row(rows_86f[t]['probs'], target, main_idx, boot_idx)
            out['hang'][t]['chexfound_86f'] = dict(**row_f, lr=rows_86f[t]['lr'], anh=rows_86f[t]['anh'], hai_nua=rows_86f[t]['hai_nua'])
            d = boot - boot_f
            out['hieu_86f'][t] = dict(ap28=row['ap28'] - row_f['ap28'], ap28_ci=ci(d[:, 0]), auroc28=row['auroc28'] - row_f['auroc28'],
                                      auroc28_ci=ci(d[:, 1]))
        p = paper[t]
        k = dict(ap_bai_trong_ci=bool(row['ap28_ci'][0] <= p['ap'] <= row['ap28_ci'][1]),
                 auroc_bai_trong_ci=bool(row['auroc28_ci'][0] <= p['auroc'] <= row['auroc28_ci'][1]))
        k['tai_tao_duoc'] = k['ap_bai_trong_ci'] and k['auroc_bai_trong_ci']
        out['ket_luan'][t] = k
    _print_encoder(out, log)
    return out


def _print_encoder(out, log):
    fm = lambda v: f'{v:.3f}'
    fci = lambda c: f'[{c[0]:.3f}; {c[1]:.3f}]'
    enc = out['encoder']
    log(f"=== C8 · {ENCODER_TEXT[enc]} — MA + MOI TRUONG CUA CheXFound ({out['nhan']}) vs Bang IV vs CheXFound 86f (cung ma / moi truong / GPU / du lieu)"
        f" — test 3.000 anh, 28 cot torchmetrics 0.10.3; TB + CI95 = {out['n_boot']} lan rut anh (hat {out['hat']}); CHI MO TA ===")
    if out['nguon_86f'] is None:
        log('86f chua co ket qua tren Drive -> chua co hang / hieu ghep cap voi CheXFound')
    for t in out['hang']:
        h, p, c, pc = out['hang'][t], out['bang_iv'][t], out['cong'][t], out['bang_iv_chexfound'][t]
        cv = c['cong_val']
        log(f"\n--- {HEAD_TEXT[t]} --- cong: AUROC test ho tinh {c['auroc_ho_tinh']:.4f} | tinh lai {c['auroc_tinh_lai']:.4f} -> "
            f"{'TRUNG' if c['trung'] else '⚠ LECH'} | val cuoi pha 1: {cv['so_dau']} lr, lech max {cv['lech_max']:.1e}, lr ho chon "
            f"{'DUNG' if cv['chon_dung_luat'] else '⚠ SAI'} luat")
        log(f"{'hang (dieu kien)':<84}{'AP28':>7}{'AP28 TB [CI95]':>24}{'AUROC28':>9}{'AUROC28 TB [CI95]':>24}{'AP27':>7}{'AUROC27':>9}{'AUPRC27':>9}"
            f"{'mAP21':>7}")
        log(f"{'BAI Bang IV: ' + ENCODER_PAPER_KEY[enc]:<84}{'':>7}{fm(p['ap']) + ' ' + fci(p['ap_ci']):>24}{'':>9}"
            f"{fm(p['auroc']) + ' ' + fci(p['auroc_ci']):>24}")
        log(f"{'BAI Bang IV: CheXFound (boi canh)':<84}{'':>7}{fm(pc['ap']) + ' ' + fci(pc['ap_ci']):>24}{'':>9}"
            f"{fm(pc['auroc']) + ' ' + fci(pc['auroc_ci']):>24}")
        e0 = h['encoder']
        rows_txt = [('encoder', f"{enc} qua ma + moi truong cua ho, anh {e0['anh']}, lr {e0['lr']:g} (val AUROC {e0['auroc_val']:.4f})"
                                + (', 2 nua luoi' if e0['hai_nua'] else ''))]
        if 'chexfound_86f' in h:
            f0 = h['chexfound_86f']
            rows_txt.append(('chexfound_86f', f"CheXFound 86f (ma + moi truong cua ho, A100, anh {f0['anh']}), lr {f0['lr']:g}"
                                              + (', 2 nua luoi' if f0['hai_nua'] else '')))
        for key, text in rows_txt:
            x = h[key]
            log(f"{text[:83]:<84}{fm(x['ap28']):>7}{fm(x['ap28_tb']) + ' ' + fci(x['ap28_ci']):>24}{fm(x['auroc28']):>9}"
                f"{fm(x['auroc28_tb']) + ' ' + fci(x['auroc28_ci']):>24}{fm(x['ap27']):>7}{fm(x['auroc27']):>9}{fm(x['auprc27']):>9}{fm(x['ap21']):>7}")
        if t in out['hieu_86f']:
            e = out['hieu_86f'][t]
            log(f"hieu {enc} - CheXFound 86f (ghep cap, cung lan rut): AP28 {e['ap28']:+.3f} {fci(e['ap28_ci'])} | AUROC28 {e['auroc28']:+.3f} "
                f"{fci(e['auroc28_ci'])}")
        k = out['ket_luan'][t]
        log(f"tieu chi (dat truoc, PLAN_86 16.5): AP bai trong CI {'CO' if k['ap_bai_trong_ci'] else 'KHONG'} | AUROC bai trong CI "
            f"{'CO' if k['auroc_bai_trong_ci'] else 'KHONG'} -> {'TAI TAO DUOC so bai' if k['tai_tao_duoc'] else 'KHONG tai tao duoc so bai'}")
