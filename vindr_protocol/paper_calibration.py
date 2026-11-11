"""Ô hiệu chỉnh với bài CheXFound (PLAN_86 §11.3; notebook 86b_chexfound ô C5): chấm các hàng CheXFound của 86b theo QUY ƯỚC BÁO CỦA BÀI —
28 cột, ngữ nghĩa torchmetrics 0.10.3 (AP bỏ lớp không có ca dương; AUROC lớp đó = 0 và vẫn tính), số trên toàn test và TB + CI95 của 1.000 lần
bootstrap theo ảnh (Bảng X của bài ghi số là TB bootstrap) — đặt cạnh Bảng IV. Kèm 27 lớp có ca dương, AUPRC hình thang, mAP 21 lớp.

CHỈ MÔ TẢ: không chọn / đổi gì theo test; luật §4.5 ở 86d không đổi. Tiêu chí "tái tạo được" ĐẶT TRƯỚC (§11.3): ở một đầu, số của bài nằm trong
CI95 bootstrap của hàng sát bài E = 10, lưới 13 lr, chế độ 'train_fp16' cho CẢ AP và AUROC 28 cột."""
import numpy as np

from . import metrics

# Bảng IV (arXiv 2502.05142 v2), VinDr-CXR, CheXFound: "AUPRC" của bài = AP bậc thang (torchmetrics MultilabelAveragePrecision, PLAN_86 §4.5 (e))
PAPER_TABLE_IV = {'lin': dict(ap=0.298, ap_ci=(0.280, 0.317), auroc=0.838, auroc_ci=(0.776, 0.866)),
                  'glori': dict(ap=0.341, ap_ci=(0.319, 0.367), auroc=0.872, auroc_ci=(0.813, 0.898))}
N_BOOT, BOOT_SEED = 1000, 8603                  # bài: 1.000 lần bootstrap; hạt của ta
CRITERION = dict(budget=10, grid='g13', mode='train_fp16')
MODE_TEXT = {'eval': 'dau doc eval(), encoder bf16 (sua cua 86)', 'train': 'dropout BAT khi cham (nhu ma goc), encoder bf16',
             'train_fp16': 'dropout BAT + encoder fp16 (gan cach cham cua bai nhat)'}
HEAD_TEXT = {'lin': 'Linear', 'glori': 'GLoRI', 'ml': 'ML-Decoder'}


def build_rows(result, arrays):
    '''Hàng CheXFound cần chấm -> [dict(khoa, dieu_kien (chữ), dau, probs [N, 28])]: lượt chính (ensemble 5 seed; seed 0 = một đầu như bài) và
       nhánh sát bài (mỗi ngân sách x đầu: lưới 13 ở mọi chế độ; lưới 9 của bài ở chế độ gần họ nhất khi lr khác lưới 13).'''
    rows, E, amp = [], result['epoch']['pha1'], str(result['amp_backbone']).split('.')[-1]
    for t in ('lin', 'glori', 'ml'):
        ks = [f'test_{t}_s{k}' for k in range(result['so_seed'])]
        if not all(k in arrays for k in ks): continue
        rows.append(dict(khoa=f'chinh_E{E}_ens_{t}', dau=t, probs=np.mean([arrays[k].astype(np.float64) for k in ks], 0),
                         dieu_kien=f'luot chinh E1=E2={E}, TB {len(ks)} seed (ensemble), dau doc eval(), encoder {amp}'))
        rows.append(dict(khoa=f'chinh_E{E}_s0_{t}', dau=t, probs=arrays[ks[0]].astype(np.float64),
                         dieu_kien=f'luot chinh E1=E2={E}, seed 0 (mot dau nhu bai), dau doc eval(), encoder {amp}'))
    sb = result.get('sat_bai')
    for Eb in (sb['cau_hinh']['ngan_sach'] if sb else []):
        for t in sb['cau_hinh']['loai_dau']:
            c = sb['chon'][f'E{Eb}|{t}']
            for mode in sb['cau_hinh']['che_do']:
                rows.append(dict(khoa=f'sat_E{Eb}_g13_{mode}_{t}', dau=t, probs=arrays[f"test_paper_{c['g13']['khoa']}_{mode}"].astype(np.float64),
                                 dieu_kien=f"sat bai E1=E2={Eb}, seed 0, luoi 13 lr (lr {c['g13']['lr']:g}), {MODE_TEXT[mode]}"))
            if not c['trung'] and CRITERION['mode'] in sb['cau_hinh']['che_do']:
                m = CRITERION['mode']
                rows.append(dict(khoa=f'sat_E{Eb}_g9_{m}_{t}', dau=t, probs=arrays[f"test_paper_{c['g9']['khoa']}_{m}"].astype(np.float64),
                                 dieu_kien=f"sat bai E1=E2={Eb}, seed 0, luoi 9 lr cua bai (lr {c['g9']['lr']:g}), {MODE_TEXT[m]}"))
    return rows


def score(labels, probs, main_idx, weights):
    '''Một hàng: số trên toàn test (28 cột kiểu bài; 27 lớp có ca dương; AUPRC hình thang 27; mAP 21) + TB / CI95 bootstrap (28 cột kiểu bài).'''
    pre = metrics.sort_once(labels, probs)
    one = np.ones(len(labels))
    ap_c, au_c, tr_c = metrics.ap_weighted(pre, one), metrics.auroc_weighted(pre, one), metrics.auprc_trapezoid_weighted(pre, one)
    ap28, au28 = metrics.tm0103_macro(ap_c, au_c)
    boot = np.array([metrics.tm0103_macro(metrics.ap_weighted(pre, w), metrics.auroc_weighted(pre, w)) for w in weights])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(ap28=ap28, auroc28=au28, ap28_tb=float(boot[:, 0].mean()), ap28_ci=ci(boot[:, 0]), auroc28_tb=float(boot[:, 1].mean()),
                auroc28_ci=ci(boot[:, 1]), ap27=float(np.nanmean(ap_c)), auroc27=float(np.nanmean(au_c)), so_lop27=int(np.isfinite(au_c).sum()),
                auprc27=float(np.nanmean(tr_c)), ap21=float(np.nanmean(ap_c[main_idx])))


def run(result, arrays, label_names, main_classes, n_boot=N_BOOT, seed=BOOT_SEED, log=print):
    '''Chấm mọi hàng (build_rows) trên y_test của arrays, in bảng cạnh Bảng IV + tiêu chí đặt trước. -> dict ghi JSON được.'''
    labels = np.asarray(arrays['y_test'], dtype=np.float64)
    assert labels.shape[1] == len(label_names) == 28, 'nhan test khong phai 28 cot -> DUNG'
    main_idx = [label_names.index(c) for c in main_classes]
    weights = metrics.bootstrap_weights(len(labels), n_boot, seed)
    rows = build_rows(result, arrays)
    assert rows, 'khong co hang CheXFound nao de cham -> DUNG'
    out = []
    for r in rows:
        assert r['probs'].shape == labels.shape and np.isfinite(r['probs']).all(), f"{r['khoa']}: du doan sai hinh / khong huu han -> DUNG"
        out.append(dict(khoa=r['khoa'], dau=r['dau'], dieu_kien=r['dieu_kien'], **score(labels, r['probs'], main_idx, weights)))
    no_pos = [label_names[j] for j in range(labels.shape[1]) if labels[:, j].sum() == 0]
    log(f"=== C5 · HIEU CHINH voi Bang IV bai CheXFound (PLAN_86 11.3) — CHI MO TA, khong chon / doi gi theo test ===\n"
        f"test {len(labels):,} anh | 28 cot kieu bai (torchmetrics 0.10.3): AP bo lop khong co ca duong ({', '.join(no_pos) or '-'}) -> AP 28 = TB "
        f"{labels.shape[1] - len(no_pos)} lop; AUROC tinh lop do = 0 | 'TB boot' = TB {n_boot} lan rut anh co hoan lai (hat {seed}), CI95 = phan vi "
        f"2,5 / 97,5 (moi lan rut: lop mat het ca duong -> AP bo, AUROC 0) | AP = AP bac thang (= 'AUPRC' cua bai) | AUPRC = hinh thang (bao kem)")
    fm = lambda v: f'{v:.3f}'
    fci = lambda c: f'[{c[0]:.3f}; {c[1]:.3f}]'
    for t in ('lin', 'glori', 'ml'):
        rr = [x for x in out if x['dau'] == t]
        if not rr: continue
        log(f"\n--- {HEAD_TEXT[t]} ---\n{'hang (dieu kien)':<100}{'AP28':>7}{'AP28 TB boot [CI95]':>26}{'AUROC28':>9}{'AUROC28 TB boot [CI95]':>26}"
            f"{'AP27':>7}{'AUROC27':>9}{'AUPRC27':>9}{'mAP21':>7}")
        if t in PAPER_TABLE_IV:
            p = PAPER_TABLE_IV[t]
            log(f"{'BAI CheXFound Bang IV (so bao = TB bootstrap theo Bang X?; 28 cot)':<100}{'':>7}{fm(p['ap']) + ' ' + fci(p['ap_ci']):>26}{'':>9}"
                f"{fm(p['auroc']) + ' ' + fci(p['auroc_ci']):>26}")
        for x in rr:
            log(f"{x['dieu_kien'][:99]:<100}{fm(x['ap28']):>7}{fm(x['ap28_tb']) + ' ' + fci(x['ap28_ci']):>26}{fm(x['auroc28']):>9}"
                f"{fm(x['auroc28_tb']) + ' ' + fci(x['auroc28_ci']):>26}{fm(x['ap27']):>7}{fm(x['auroc27']):>9}{fm(x['auprc27']):>9}{fm(x['ap21']):>7}")
    verdict = {}
    for t, p in PAPER_TABLE_IV.items():
        k = f"sat_E{CRITERION['budget']}_{CRITERION['grid']}_{CRITERION['mode']}_{t}"
        x = next((r for r in out if r['khoa'] == k), None)
        if x is None: continue
        in_ap = x['ap28_ci'][0] <= p['ap'] <= x['ap28_ci'][1]
        in_au = x['auroc28_ci'][0] <= p['auroc'] <= x['auroc28_ci'][1]
        verdict[t] = dict(hang=k, ap_bai_trong_ci=in_ap, auroc_bai_trong_ci=in_au, tai_tao_duoc=bool(in_ap and in_au),
                          ap_ta_tru_bai=x['ap28_tb'] - p['ap'], auroc_ta_tru_bai=x['auroc28_tb'] - p['auroc'])
    log(f"\nTIEU CHI DAT TRUOC (PLAN_86 11.3): so bai nam trong CI95 boot cua hang sat bai E={CRITERION['budget']}, luoi 13, {CRITERION['mode']} "
        "cho CA AP va AUROC 28 cot")
    for t, v in verdict.items():
        log(f"  {HEAD_TEXT[t]:<7}: AP bai trong CI {'CO' if v['ap_bai_trong_ci'] else 'KHONG'} (TB boot ta - bai {v['ap_ta_tru_bai']:+.3f}) | AUROC bai "
            f"trong CI {'CO' if v['auroc_bai_trong_ci'] else 'KHONG'} ({v['auroc_ta_tru_bai']:+.3f}) -> {'TAI TAO DUOC' if v['tai_tao_duoc'] else 'CHUA tai tao'}")
    return dict(n_boot=n_boot, hat=seed, bang_iv=PAPER_TABLE_IV, tieu_chi=CRITERION, lop_khong_ca_duong=no_pos, hang=out, ket_luan=verdict,
                ghi_chu='PLAN_86 11.3: CHI mo ta; khong chon / doi gi theo test.')
