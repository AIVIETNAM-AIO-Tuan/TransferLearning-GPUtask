"""K3 (PLAN_KIEM_THU §7.2, §15.6): bảng VinDr P1 / P2 qua backbone seed B — MÔ TẢ, không luật tổng hợp mới.

P1 (đầu MIMIC 40 lớp, KHÔNG học lại; 12 lớp, ánh xạ chốt ở 85b): mAP / mAUROC / mAUPRC, hai nửa, CI95 theo seed; B s42 phải trùng JSON 85d.
P2 (21 lớp; ML-Decoder / chỉ patch / linear): đầu đơn TB ± SD mẫu, ensemble + CI95, B_seed − X + phán xử DẠNG luật P2 từng seed;
qua seed: các seed MỚI (s43 / s44 _lai / s45 / s46) TB ± SD mẫu, B s42 báo RIÊNG (seed phát triển, test đã đọc); B s42 + SOTA phải trùng JSON 85g.
Không ensemble nhiều backbone, không gọi n backbone x 5 đầu là 5n seed."""
import time

import numpy as np

from . import config
from .metrics import ap_cols, bootstrap_class_ap, ci95, mean_sd, summary
from .predictions import p1_classes, p1_scores
from .rules import p2_verdict


def p1_table(test, pred_p1, lab40, b_rows, js_85d, json_85d_name, n_boot=config.N_BOOT, log=print):
    '''P1 theo backbone seed -> dict(classes, result {(hàng, đầu): tóm tắt}, ci, across_seeds, boot_skipped).'''
    mapping, classes = p1_classes(test, lab40)
    Y1 = test.Y[:, test.columns(classes)]
    scores = {k: p1_scores(P, lab40, mapping, classes) for k, P in pred_p1.items()}
    res = {k: summary(Y1, S, test.half) for k, S in scores.items()}
    draws, skipped, _ = bootstrap_class_ap(Y1, scores, classes, config.BOOT_SEED, n_boot)
    ci = {k: ci95(v.mean(1)) for k, v in draws.items()}
    # ---- đối chiếu 85d: B s42 phải trùng (cùng dự đoán, ánh xạ, hạt, cách rút) ----
    b42 = b_rows[config.ANCHOR_ROW]['bb']
    lech = [abs(js_85d['ket_qua'][f'{b42} | {d} | chinh'][m] - res[(config.ANCHOR_ROW, d)][m]) for d in config.P1_HEADS for m in ('ap', 'auc', 'auprc')]
    lech += [abs(js_85d['ci'][f'{b42} | {d} | ap']['ci'][i] - ci[(config.ANCHOR_ROW, d)]['ci'][i]) for d in config.P1_HEADS for i in (0, 1)]
    assert max(lech) < 1e-9, f'P1 B_s42 KHAC json 85d (lech max {max(lech):.1e}) -> DUNG'
    log(f'doi chieu 85d ({json_85d_name}): P1 B_s42 TRUNG (lech max {max(lech):.1e}) | bootstrap {n_boot} lan, bi loai {skipped}')
    new = [h for h, c in b_rows.items() if c['new']]
    across = {d: dict(mean_sd([res[(h, d)]['ap'] for h in new]), gia_tri={h: res[(h, d)]['ap'] for h in new}) for d in config.P1_HEADS}
    log(f"\n=== P1 (VinDr test, {len(test.keys):,} anh, {len(classes)} lop, dau MIMIC KHONG hoc lai) theo backbone seed ===")
    log(f"{'hang':<12}{'dau':<13}{'mAP':>8}{'CI95':>20}{'nua A':>8}{'nua B':>8}{'mAUROC':>8}{'mAUPRC':>8}")
    for (h, d), r in res.items():
        log(f"{h + (' (s42)' if not b_rows[h]['new'] else ''):<12}{d:<13}{r['ap']:>8.4f}  [{ci[(h, d)]['ci'][0]:.4f}; {ci[(h, d)]['ci'][1]:.4f}]{r['ap_A']:>8.4f}{r['ap_B']:>8.4f}"
            f"{r['auc']:>8.4f}{r['auprc']:>8.4f}")
    for d, v in across.items():
        log(f"  {len(new)} seed MOI ({', '.join(new)}), {d}: mAP TB {v['tb']:.4f} +- {v['sd']:.4f} (SD mau) | B_s42 (rieng) {res[(config.ANCHOR_ROW, d)]['ap']:.4f}")
    return dict(classes=classes, result=res, ci=ci, across_seeds=across, boot_skipped=skipped, new=new, cross_check_85d=max(lech))


def p2_table(test, pred, b_rows, sota_rows, js_85g, json_85g_name, n_boot=config.N_BOOT, log=print):
    '''P2 theo backbone seed -> dict(result, single, ci_ens, ci_pair, verdict, across_seeds, boot_skipped).'''
    ix = test.columns(test.main_classes)
    Y = test.Y
    ens = {k: v.mean(0) for k, v in pred.items()}
    res = {(b, d, 'chinh'): summary(Y[:, ix], P[:, ix], test.half) for (b, d), P in ens.items()}
    single = {k: mean_sd([float(np.nanmean(ap_cols(Y[:, ix], s[:, ix]))) for s in v]) for k, v in pred.items()}
    t0 = time.time()
    draws, skipped, _ = bootstrap_class_ap(Y[:, ix], {k: P[:, ix] for k, P in ens.items()}, list(test.main_classes), config.BOOT_SEED, n_boot)
    ci_ens = {k: ci95(v.mean(1)) for k, v in draws.items()}
    ci_pair = {((b, d), (x, d), 'ap'): ci95((draws[(b, d)] - draws[(x, d)]).mean(1)) for b in b_rows for x in sota_rows for d in config.SOTA_HEADS}
    verdict = {b: {x: p2_verdict(res[(b, 'ml', 'chinh')], res[(x, 'ml', 'chinh')], ci_pair[((b, 'ml'), (x, 'ml'), 'ap')]['ci']) for x in sota_rows} for b in b_rows}
    # ---- đối chiếu 85g: B s42 + SOTA phải trùng (ensemble, CI ghép cặp, phán xử) ----
    ten = {b: c['bb'] for b, c in list(b_rows.items()) + list(sota_rows.items())}
    lech = [abs(js_85g['ket_qua'][f"{ten[b]} | {d} | chinh"][m] - res[(b, d, 'chinh')][m]) for b in [config.ANCHOR_ROW] + list(sota_rows)
            for d in (config.B_HEADS if b in b_rows else config.SOTA_HEADS) for m in ('ap', 'auc', 'auprc')]
    lech += [abs(js_85g['ci'][f"{ten[config.ANCHOR_ROW]}|{d} - {x}|{d} | ap"]['ci'][i] - ci_pair[((config.ANCHOR_ROW, d), (x, d), 'ap')]['ci'][i])
             for x in sota_rows for d in config.SOTA_HEADS for i in (0, 1)]
    px85 = {x: js_85g['luat']['phan_xu'][x]['ket_luan'] for x in sota_rows}
    assert max(lech) < 1e-9 and px85 == {x: verdict[config.ANCHOR_ROW][x][2] for x in sota_rows}, (
        f'B_s42 / SOTA KHAC json 85g (lech max {max(lech):.1e}; phan xu {px85}) -> DUNG')
    log(f'doi chieu 85g ({json_85g_name}): B_s42 + SOTA TRUNG (lech max {max(lech):.1e}), phan xu TRUNG | bootstrap {n_boot} lan, bi loai {skipped}, '
        f'{(time.time() - t0) / 60:.1f} phut')
    log(f"\n=== P2 (VinDr test, {len(test.keys):,} anh, {len(test.main_classes)} lop; backbone DONG CUNG, dau hoc lai) theo hang ===")
    log(f"{'hang':<12}{'dau':<10}{'mAP ens':>9}{'CI95':>20}{'nua A':>8}{'nua B':>8}{'mAUROC':>8}{'mAUPRC':>8}{'5 dau don TB +- SD':>22}")
    for (b, d, _t), r in res.items():
        c, t = ci_ens[(b, d)]['ci'], single[(b, d)]
        log(f"{b:<12}{d:<10}{r['ap']:>9.4f}  [{c[0]:.4f}; {c[1]:.4f}]{r['ap_A']:>8.4f}{r['ap_B']:>8.4f}{r['auc']:>8.4f}{r['auprc']:>8.4f}{t['tb']:>14.4f} +- {t['sd']:.4f}")
    log(f"\n=== B_seed - X (dau ML-Decoder, mAP {len(test.main_classes)} lop): chenh [CI95] (nua A / nua B) -> phan xu DANG LUAT P2 (mo ta tung seed; luat P2 = B_s42 o 85g) ===")
    for b in b_rows:
        for x, (dl, ci, v) in verdict[b].items():
            log(f"  {b:<10} - {x:<10}: {dl['ap']:+.4f} [{ci[0]:+.4f}; {ci[1]:+.4f}] ({dl['ap_A']:+.4f} / {dl['ap_B']:+.4f}) -> {v} | mAUROC {dl['auc']:+.4f}"
                + (' | (s42: seed phat trien, test da doc)' if not b_rows[b]['new'] else ''))
    new = [h for h, c in b_rows.items() if c['new']]
    across = {}
    for d in config.B_HEADS:
        across[('ens', d)] = dict(mean_sd([res[(h, d, 'chinh')]['ap'] for h in new]), gia_tri={h: res[(h, d, 'chinh')]['ap'] for h in new})
    for d in config.SOTA_HEADS:
        for x in sota_rows:
            v = {h: res[(h, d, 'chinh')]['ap'] - res[(x, d, 'chinh')]['ap'] for h in new}
            across[('chenh', d, x)] = dict(mean_sd(list(v.values())), gia_tri=v,
                                           so_seed_b_hon=sum(verdict[h][x][2] == 'B HON' for h in new) if d == 'ml' else None)
    log(f"\n=== Qua backbone seed: {len(new)} seed MOI ({', '.join(new)}) — TB +- SD MAU; B_s42 RIENG; KHONG gop thanh {5 * len(new)} seed, KHONG ensemble nhieu backbone ===")
    for d in config.B_HEADS:
        v = across[('ens', d)]
        log(f"  ensemble {d:<9}: mAP {v['tb']:.4f} +- {v['sd']:.4f} | B_s42 {res[(config.ANCHOR_ROW, d, 'chinh')]['ap']:.4f}")
    for d in config.SOTA_HEADS:
        for x in sota_rows:
            v = across[('chenh', d, x)]
            log(f"  B - {x:<10} {d:<4}: {v['tb']:+.4f} +- {v['sd']:.4f}" + (f" | so seed MOI 'B HON' (dang luat) {v['so_seed_b_hon']} / {len(new)}" if d == 'ml' else '')
                + f" | B_s42 {res[(config.ANCHOR_ROW, d, 'chinh')]['ap'] - res[(x, d, 'chinh')]['ap']:+.4f}")
    log('Bang bo sung mo ta muc on dinh qua backbone seed (PLAN_KIEM_THU §7.2); luat P2 (85g, B_s42) KHONG doi, KHONG co luat tong hop moi.')
    return dict(result=res, single=single, ci_ens=ci_ens, ci_pair=ci_pair, verdict=verdict, across_seeds=across, boot_skipped=skipped, new=new,
                cross_check_85g=max(lech))
