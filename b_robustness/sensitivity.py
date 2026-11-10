"""K5 (PLAN_KIEM_THU §9, §15.4): thống kê lại P2 VinDr đã có — MÔ TẢ / ĐỘ NHẠY sau khi đã thấy kết quả; luật P2 (85g) KHÔNG đổi.

(1) TB ± SD mẫu (ddof = 1) của 5 đầu đơn TÁCH khỏi điểm ensemble (TB xác suất 5 đầu) + CI95 ensemble; (2) tập 21 / 7 / 27 nhãn;
(3) từng lớp, số lớp hơn / kém, số ca dương từng nửa; (4) bỏ từng lớp; (5) bỏ Rib fracture: chênh 20 lớp + CI95 ghép cặp (cùng lần rút) + hai nửa.
Số ensemble + CI ghép cặp 21 lớp phải trùng JSON 85g (cùng dự đoán, hạt, cách rút)."""
import time

import numpy as np

from . import config
from .metrics import ap_cols, bootstrap_class_ap, ci95, mean_sd, summary

MEASURES = ('ap', 'auc', 'auprc')


def label_sets(test):
    return {'chinh': list(test.main_classes), 'raddino7': list(config.RADDINO7), 'tat_ca': list(test.any_classes)}


def cross_check_85g(js, ens_summary, ci_pair, anchor, sota_rows, tol=1e-9):
    '''Ensemble (mAP / mAUROC / mAUPRC 21 lớp) mọi hàng x đầu + CI95 ghép cặp mAP B − X phải TRÙNG JSON 85g -> độ lệch lớn nhất.'''
    lech = [abs(js['ket_qua'][f'{b} | {d} | chinh'][m] - ens_summary[(b, d, 'chinh')][m]) for (b, d, t) in ens_summary if t == 'chinh' for m in MEASURES]
    lech += [abs(js['ci'][f'{anchor}|{d} - {x}|{d} | ap']['ci'][i] - ci_pair[(d, x, 'chinh')]['ci'][i]) for d in ('ml', 'lin') for x in sota_rows for i in (0, 1)]
    assert max(lech) < tol, f'so / CI KHAC json 85g (lech max {max(lech):.1e}) -> du lieu / ma doi. DUNG'
    return max(lech)


def run_k5(test, pred, js_85g, json_85g_name, anchor=config.ANCHOR, sota=config.SOTA, n_boot=config.N_BOOT, log=print):
    '''-> dict kết quả K5 (khoá tuple; pipeline đổi sang JSON). In các bảng.'''
    Y, half, main = test.Y, test.half, list(test.main_classes)
    sets = label_sets(test)
    ix = {t: test.columns(l) for t, l in sets.items()}
    ens = {k: v.mean(0) for k, v in pred.items()}                                  # ensemble = TB xác suất 5 đầu (= 85g)
    kq = {(b, d, t): summary(Y[:, i], P[:, i], half) for (b, d), P in ens.items() for t, i in ix.items()}
    single = {(b, d, t): [summary(Y[:, i], pred[(b, d)][k][:, i], half) for k in range(len(pred[(b, d)]))] for (b, d) in pred for t, i in ix.items()}
    tom_don = {k: {m: mean_sd([x[m] for x in v]) for m in MEASURES} for k, v in single.items()}
    t0 = time.time()
    draws, skipped, by_class = bootstrap_class_ap(Y[:, ix['chinh']], {k: P[:, ix['chinh']] for k, P in ens.items()}, main, config.BOOT_SEED, n_boot)
    log(f'bootstrap {n_boot} lan theo {len(test.keys):,} anh: {(time.time() - t0) / 60:.1f} phut | lan rut bi loai {skipped}'
        + (f' theo lop {dict(by_class)}' if skipped else ''))
    ci_ens = {k: ci95(v.mean(1)) for k, v in draws.items()}
    rows = list(dict.fromkeys(b for b, _ in pred))
    sota_rows = [b for b in rows if b != anchor and b in sota]
    i_rib = main.index(config.RIB)
    keep20 = [j for j in range(len(main)) if j != i_rib]
    ci_pair = {}
    for d in ('ml', 'lin'):
        for x in sota_rows:
            dl = draws[(anchor, d)] - draws[(x, d)]
            ci_pair[(d, x, 'chinh')], ci_pair[(d, x, 'bo_rib')] = ci95(dl.mean(1)), ci95(dl[:, keep20].mean(1))
    lech = cross_check_85g(js_85g, kq, ci_pair, anchor, sota_rows)
    log(f'doi chieu 85g ({json_85g_name}): ensemble + CI ghep cap 21 lop TRUNG (lech max {lech:.1e})')
    # ---- (1) đầu đơn vs ensemble ----
    log(f"\n=== (1) 5 dau DON (TB +- SD MAU, ddof = 1) vs ENSEMBLE (TB xac suat 5 dau) — 21 lop, VinDr test {len(test.keys):,} anh ===")
    log(f"{'hang':<28}{'dau':<10}{'mAP don':>16}{'mAP ens':>9}{'CI95 ens':>20}{'mAUROC don':>17}{'ens':>8}{'mAUPRC don':>17}{'ens':>8}")
    for (b, d) in ens:
        t, e, c = tom_don[(b, d, 'chinh')], kq[(b, d, 'chinh')], ci_ens[(b, d)]['ci']
        log(f"{b:<28}{d:<10}{t['ap']['tb']:>9.4f}+-{t['ap']['sd']:.4f}{e['ap']:>9.4f}  [{c[0]:.4f}; {c[1]:.4f}]{t['auc']['tb']:>10.4f}+-{t['auc']['sd']:.4f}"
            f"{e['auc']:>8.4f}{t['auprc']['tb']:>10.4f}+-{t['auprc']['sd']:.4f}{e['auprc']:>8.4f}")
    log('ghi chu: 85g in "mAP 21" = ENSEMBLE va "+-seed" = SD (ddof = 0) cua 5 dau don -> hai loai so KHAC nhau; bao tach nhu bang tren')
    # ---- (2) tập nhãn ----
    log(f"\n=== (2) ensemble theo tap nhan: 21 lop (>= 10 duong, luat) / 7 lop bai RAD-DINO / {len(sets['tat_ca'])} lop co duong (KHONG CI) ===")
    for (b, d) in ens:
        log(f"{b:<28}{d:<10}" + ''.join(f"{t}: mAP {kq[(b, d, t)]['ap']:.4f} / mAUROC {kq[(b, d, t)]['auc']:.4f}   " for t in sets))
    # ---- (3) từng lớp ----
    apl = {b: np.array(kq[(b, 'ml', 'chinh')]['ap_lop'], 'float64') for b in [anchor] + sota_rows}
    names = test.label_names
    duong = {c: int(Y[:, names.index(c)].sum()) for c in main}
    duong_nua = {c: [int(Y[half == h, names.index(c)].sum()) for h in (0, 1)] for c in main}
    log(f"\n=== (3) tung lop (21, dau ML-Decoder, ensemble): AP cua B va chenh B - X; so duong (toan / nua A / nua B) ===")
    log(f"{'lop':<20}{'duong':>7}{'nua A':>7}{'nua B':>7}{'AP B':>8}" + ''.join(f'{"B - " + x[:9]:>15}' for x in sota_rows))
    for j, cl in enumerate(main):
        log(f"{cl:<20}{duong[cl]:>7}{duong_nua[cl][0]:>7}{duong_nua[cl][1]:>7}{apl[anchor][j]:>8.4f}" + ''.join(f'{apl[anchor][j] - apl[x][j]:>+15.4f}' for x in sota_rows))
    hon_kem = {x: dict(hon=int((apl[anchor] > apl[x]).sum()), kem=int((apl[anchor] < apl[x]).sum()), bang=int((apl[anchor] == apl[x]).sum())) for x in sota_rows}
    log('so lop B hon / kem / bang: ' + ' | '.join(f"{x}: {v['hon']} / {v['kem']} / {v['bang']}" for x, v in hon_kem.items()))
    min_half = min(min(v) for v in duong_nua.values())
    log(f"so duong it nhat trong mot nua: {min_half} ({[c for c, v in duong_nua.items() if min(v) == min_half]}) -> hai nua deu giu du 21 lop: {min_half > 0}")
    # ---- (4) bỏ từng lớp ----
    loco = {}
    for d in ('ml', 'lin'):
        for x in sota_rows:
            dl = np.array(kq[(anchor, d, 'chinh')]['ap_lop'], 'float64') - np.array(kq[(x, d, 'chinh')]['ap_lop'], 'float64')
            lo = {c: float(np.delete(dl, j).mean()) for j, c in enumerate(main)}
            loco[(d, x)] = dict(day_du=float(dl.mean()), moi_lop=lo, nho=min(lo, key=lo.get), lon=max(lo, key=lo.get),
                                dong_gop={c: float(dl[j] / len(dl)) for j, c in enumerate(main)})
    log(f"\n=== (4) bo tung lop (21 -> 20): chenh mAP B - X nho nhat / lon nhat + lop bi bo; lop dong gop nhieu nhat vao chenh 21 lop ===")
    for (d, x), v in loco.items():
        top = sorted(v['dong_gop'], key=lambda c: -v['dong_gop'][c])[:3]
        log(f"  {d:<4} B - {x:<10}: day du {v['day_du']:+.4f} | bo lop -> nho nhat {v['moi_lop'][v['nho']]:+.4f} (bo {v['nho']}) / lon nhat "
            f"{v['moi_lop'][v['lon']]:+.4f} (bo {v['lon']}) | dong gop lon nhat: " + ', '.join(f"{c} {v['dong_gop'][c]:+.4f} ({v['dong_gop'][c] / v['day_du']:.0%})" for c in top))
    # ---- (5) bỏ Rib fracture ----
    drop_rib = {}
    Yc = Y[:, ix['chinh']]
    for d in ('ml', 'lin'):
        for x in sota_rows:
            Pb, Px = ens[(anchor, d)][:, ix['chinh']], ens[(x, d)][:, ix['chinh']]
            nua = [float(np.nanmean(ap_cols(Yc[half == h][:, keep20], Pb[half == h][:, keep20])) - np.nanmean(ap_cols(Yc[half == h][:, keep20], Px[half == h][:, keep20])))
                   for h in (0, 1)]
            d20 = loco[(d, x)]['moi_lop'][config.RIB]
            ci = ci_pair[(d, x, 'bo_rib')]['ci']
            drop_rib[(d, x)] = dict(chenh_21=loco[(d, x)]['day_du'], chenh_20=d20, ci_20=ci, nua_20=nua,
                                    phan_rib=loco[(d, x)]['dong_gop'][config.RIB] / loco[(d, x)]['day_du'],
                                    dang_luat=bool(d20 >= config.RULE_THRESHOLD and ci[0] > 0 and min(nua) > 0))
    log(f"\n=== (5) bo Rib fracture ({duong[config.RIB]} duong o test): chenh mAP B - X tren 20 lop + CI95 ghep cap (cung lan rut) ===")
    for (d, x), v in drop_rib.items():
        log(f"  {d:<4} B - {x:<10}: 21 lop {v['chenh_21']:+.4f} (Rib {v['phan_rib']:.0%}) -> 20 lop {v['chenh_20']:+.4f} [{v['ci_20'][0]:+.4f}; {v['ci_20'][1]:+.4f}] "
            f"(nua A {v['nua_20'][0]:+.4f} / nua B {v['nua_20'][1]:+.4f}) | dang luat (doi chieu, KHONG phai luat): {'giu' if v['dang_luat'] else 'KHONG giu'}")
    log('Phan tich do nhay SAU khi da thay ket qua (PLAN_KIEM_THU §9.2): KHONG thay tap chinh, KHONG doi luat P2 (85g).')
    return dict(rows=rows, anchor=anchor, sota=sota_rows, main=main, single=tom_don, ensemble=kq, ci_ensemble=ci_ens, ci_pair=ci_pair,
                positives=duong, positives_half=duong_nua, win_loss=hon_kem, leave_one_out=loco, drop_rib=drop_rib, boot_skipped=skipped,
                boot_skipped_classes=dict(by_class), cross_check_85g=lech)
