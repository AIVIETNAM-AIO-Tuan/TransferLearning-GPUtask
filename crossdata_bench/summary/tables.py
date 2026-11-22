# -*- coding: utf-8 -*-
"""Bảng chữ của notebook `93t` — MỖI BỘ MỘT PHẦN (Tuấn 22/11: tách từng bộ). Hàm nhận `bo` = kết quả ô C4 của một bộ (notebook.summary.score) và in;
trả hàng để ghi CSV. Không đọc Drive.

Mọi bảng: AP bậc thang (sklearn average_precision = "AUPRC" của bài CheXFound) + AUPRC hình thang (sklearn auc trên đường PR) + AUROC; điều kiện ghi
bằng chữ ở đầu mỗi bảng và ở từng hàng của bảng dài; mức tiếp xúc tiền huấn luyện ở từng hàng; hiệu luôn ghi chiều "B − X"."""
from .. import config
from ..evaluation.bootstrap import METRICS, VERDICT_TEXT

HEAD_PAIRS = (('lin4', 'lin1'), ('mldec', 'lin4'), ('glori', 'lin4'), ('glori', 'mldec'))      # = vindr_bench.notebook.summary.HEAD_PAIRS (mô tả)
M_SHORT = {'ap': 'AP', 'auprc_thang': 'AUPRC', 'auroc': 'AUROC'}


def f3(x, nd=3):
    return '  -  ' if x is None or x != x else f'{x:.{nd}f}'


def d3(x):
    return '   -  ' if x is None or x != x else f'{x:+.3f}'


def ci(c):
    return f'[{f3(c[0])}, {f3(c[1])}]' if c else '[ -  ]'


def dci(c):
    return f'[{d3(c[0])}, {d3(c[1])}]' if c else '[ -  ]'


def heads():
    from vindr_bench.protocols.ours import config as C
    return C.HEAD_KINDS, C.TITLES


def encoders(bo):
    return [e for e in config.TABLE_ORDER if any(k[0] == e for k in bo['cells'])]


def row_text(e):
    from vindr_bench.encoders import SPECS
    s = SPECS[e]
    return f'{config.ROW_TITLES[e]} ({s.image_size}, {s.pool})'


def condition_text(bo):
    '''Điều kiện CHUNG của mọi hàng trong phần của một bộ (in ở đầu mỗi bảng).'''
    sp = config.dataset(bo['name'])
    n, pos, n_cl = bo['n_test'], bo['so_duong_test'], bo['so_cum_test']
    unit = 'nguoi NIH' if sp.unit == 'nguoi' else 'anh'
    return (f"{sp.title} | test {n} anh, {pos} duong ({100 * pos / n:.1f} %), {n_cl} cum ({unit}) | giao thuc 89 Ver2 (ma + moi truong cua ta, G4): backbone "
            f"DONG CUNG o anh + chuan hoa goc tung encoder, kho 518 dem vuong KHONG can bang histogram, 4 dau x 13 lr, lr chon theo AUROC val (lan cham cuoi "
            f"pha 1), 5 hat dau doc, pha 2 tren train + val, du doan test MOT lan, dau doc eval() khi cham | so chinh = TB 5 hat | AP = AP bac thang, AUPRC = "
            f"hinh thang | CI95 DANH NGHIA = bootstrap {bo['n_boot']} lan theo cum ({unit}, hat {config.BOOT_SEED}), chua hieu chinh nhieu phep so; hai nua "
            f"test theo cum (hat {config.HALF_SEED}): {bo['nua_anh'][0]} / {bo['nua_anh'][1]} anh")


def val_table(bo, log=print):
    '''Val cuối pha 1 (5.476 / 65 ảnh) — CHỈ THEO DÕI (lr đã chọn ở C6 của từng lượt; không chọn gì ở đây).'''
    hk, ht = heads()
    log(f"\n--- VAL cuoi pha 1 ({bo['n_val']} anh; CHI THEO DOI — khong chon gi o day): AP / AUROC, TB 5 hat ± SD; nhanh cua LUAT ---")
    log(f"{'hang (anh, vector toan anh)':<36}" + ''.join(f'{ht[h]:>27}' for h in hk))
    for e in encoders(bo):
        line = f'{row_text(e):<36}'
        for h in hk:
            c = bo['cells'].get((e, h, 'luat'))
            v = c['val'] if c else None
            line += (f"{f3(v['ap']['tb'])}±{f3(v['ap']['sd'], 4)} / {f3(v['auroc']['tb'])}" if v else '-').rjust(27)
        log(line)


def matrices(bo, log=print):
    '''Ma trận encoder × đầu cho từng số đo (test, TB 5 hạt; * = lr của luật ở biên lưới).'''
    hk, ht = heads()
    for m in METRICS:
        log(f"\n--- TEST: {M_SHORT[m]} ({'AP bac thang = so cua luat' if m == 'ap' else ('hinh thang' if m == 'auprc_thang' else 'AUROC')}), TB 5 hat; "
            f"* = lr o bien luoi; [tiep xuc] = muc tiep xuc tien huan luyen voi anh cua bo ---")
        log(f"{'hang (anh, vector toan anh) [tiep xuc]':<62}" + ''.join(f'{ht[h]:>12}' for h in hk))
        for e in encoders(bo):
            line = f"{row_text(e) + ' [' + config.EXPOSURE[bo['name']][e] + ']':<62}"
            for h in hk:
                c = bo['cells'].get((e, h, 'luat'))
                line += (f"{f3(c['diem'][m])}{'*' if c['bien'] else ' '}" if c else '  -   ').rjust(12)
            log(line)


def long_table(bo, log=print):
    '''Bảng dài: mỗi hàng một (encoder, đầu, nhánh) với ĐỦ điều kiện. -> list dict (CSV).'''
    hk, ht = heads()
    sp = config.dataset(bo['name'])
    rows = []
    log(f"\n--- BANG DAI ({sp.title}; moi hang: encoder, dau, nhanh; nhanh 'ap' = lr theo AP val, MO TA, chi khi khac lr cua luat) ---")
    log(f"{'hang':<26}{'dau':<11}{'nhanh':<6}{'lr':>8}  {'AP [CI95 danh nghia]':<22}{'SD hat':>7}{'ens':>7}  {'AUPRC [CI95]':<22}{'AUROC [CI95]':<22}"
        f"{'AP hai nua':<15}tiep xuc")
    for e in encoders(bo):
        for h in hk:
            for arm in ('luat', 'ap'):
                c = bo['cells'].get((e, h, arm))
                if c is None:
                    continue
                d, sd, en = c['diem'], c['sd'], c['ensemble'] or {}
                log(f"{config.ROW_TITLES[e]:<26}{ht[h]:<11}{arm:<6}{c['lr']:>8g}{'*' if c['bien'] else ' '} {f3(d['ap'])} {ci(c['ci']['ap']):<16}"
                    f"{f3(sd['ap'], 4):>7}{f3(en.get('ap')):>7}  {f3(d['auprc_thang'])} {ci(c['ci']['auprc_thang']):<16}{f3(d['auroc'])} {ci(c['ci']['auroc']):<16}"
                    f"{f3(c['nua'][0])} / {f3(c['nua'][1])}  {config.EXPOSURE[bo['name']][e]}")
                rows.append(dict(bo=bo['name'], dieu_kien=condition_text(bo), encoder=e, hang=row_text(e), tiep_xuc_tien_huan_luyen=config.EXPOSURE[bo['name']][e],
                                 dau=h, nhanh=arm, lr=c['lr'], lr_o_bien=c['bien'], so_hat=c['so_hat'],
                                 AP_bac_thang=d['ap'], AP_ci_thap=c['ci']['ap'][0], AP_ci_cao=c['ci']['ap'][1], AP_sd_giua_hat=sd['ap'], AP_ensemble=en.get('ap'),
                                 AUPRC_hinh_thang=d['auprc_thang'], AUPRC_ci_thap=c['ci']['auprc_thang'][0], AUPRC_ci_cao=c['ci']['auprc_thang'][1],
                                 AUPRC_sd_giua_hat=sd['auprc_thang'], AUPRC_ensemble=en.get('auprc_thang'),
                                 AUROC=d['auroc'], AUROC_ci_thap=c['ci']['auroc'][0], AUROC_ci_cao=c['ci']['auroc'][1], AUROC_sd_giua_hat=sd['auroc'],
                                 AUROC_ensemble=en.get('auroc'), AP_nua_0=c['nua'][0], AP_nua_1=c['nua'][1],
                                 val_AP_tb=c['val']['ap']['tb'], val_AUROC_tb=c['val']['auroc']['tb'], lan_rut_bi_loai=c['loai']['ap'], nguon=c['nguon']))
    return rows


def ranks(bo, log=print):
    '''Thứ hạng theo AP (số của luật) ở từng đầu. -> {đầu: [encoder theo thứ tự]}.'''
    hk, ht = heads()
    out = {}
    log('\n--- THU HANG theo AP test (TB 5 hat; MO TA — thu hang khong phai phep so) ---')
    for h in hk:
        v = {e: bo['cells'][(e, h, 'luat')]['diem']['ap'] for e in encoders(bo) if (e, h, 'luat') in bo['cells']}
        order = sorted(v, key=lambda e: -v[e])
        out[h] = order
        log(f"{ht[h]:<11}" + ' > '.join(f'{config.ROW_TITLES[e]} {f3(v[e])}' for e in order))
    return out


def ceiling_text(tr, log=print):
    if tr['tot_nhat'] is None:
        log('dieu khoan tran: chua co o doi chung nao')
        return
    b = tr['tot_nhat']
    _, ht = heads()
    log(f"dieu khoan tran (dat truoc, PLAN_93 5): o doi chung cao nhat = {config.ROW_TITLES[b['encoder']]} {ht[b['dau']]} AP {f3(b['gia_tri'], 4)} "
        f"{'>=' if tr['ap_dung'] else '<'} {tr['nguong']} -> " + ('LUAT THANH MO TA o bo nay (khong con cho cho +0,01)' if tr['ap_dung'] else 'luat AP DUNG'))
    log('   cach doc theo tung dau (in kem): ' + ' | '.join(f"{ht[h]}: {config.ROW_TITLES[p['encoder']]} {f3(p['gia_tri'], 4)}"
                                                           f"{' >= nguong' if p['dat_nguong'] else ''}" for h, p in tr['theo_dau'].items())
        + (' -> hai cach doc CUNG ket qua' if tr['hai_cach_doc_trung'] else ' -> HAI CACH DOC KHAC NHAU: theo PLAN_93 5 dung cach doc ca bo; bao Tuan'))


def family_table(bo, fam, log=print):
    '''In họ B − X. -> list dict (CSV).'''
    hk, ht = heads()
    sp = config.dataset(bo['name'])
    rows = []
    ref = config.REFERENCE
    if fam['du']:
        log(f"khoang DONG THOI max-T 95 % cua ho {fam['so_phep']} phep: q = {fam['q']:.2f} ({fam['lan_rut_dung']} / {bo['n_boot']} lan rut dung)"
            + (' | MO TA: dieu khoan tran -> khong phan quyet' if fam['mo_ta'] else ''))
    else:
        log(f"HO CHUA DU ({fam['co']} / {fam['so_phep']} phep co du lieu) -> chi hieu DANH NGHIA, CHUA ap luat")
    log(f"cach doc: hieu = {config.ROW_TITLES[ref]} - X (duong = B cao hon), ghep cap theo cum tren cung lan rut; 'dong thoi' = khoang da hieu chinh cho ca "
        f"{fam['so_phep']} phep so; 'danh nghia' = khoang rieng tung phep; CI chua 0 = CHUA phat hien khac biet, khong phai 'bang nhau'")
    for h in hk:
        log(f'--- {ht[h]} ---')
        for r in [r for r in fam['phep_so'] if r['dau'] == h]:
            k = r['kem']
            log(f"    B - {config.ROW_TITLES[r['doi_chung']]:<12} AP {d3(r['d'])} dong thoi {dci(r['ci_dong_thoi'])} | danh nghia {dci(r['ci_danh_nghia'])} | hai nua "
                f"{d3(r['nua'][0])} / {d3(r['nua'][1])} -> {VERDICT_TEXT.get(r['ket_luan'], r['ket_luan'])} | AUPRC {d3(k['auprc_thang']['d'])} "
                f"{dci(k['auprc_thang']['ci_danh_nghia'])} | AUROC {d3(k['auroc']['d'])} {dci(k['auroc']['ci_danh_nghia'])}")
            rows.append(dict(bo=bo['name'], dieu_kien=condition_text(bo), dau=h, phep_so=f"{ref} - {r['doi_chung']}", doi_chung=r['doi_chung'],
                             tiep_xuc_B=config.EXPOSURE[bo['name']][ref], tiep_xuc_X=config.EXPOSURE[bo['name']][r['doi_chung']],
                             hieu_AP=r['d'], dong_thoi_thap=(r['ci_dong_thoi'] or [None])[0], dong_thoi_cao=(r['ci_dong_thoi'] or [None, None])[1],
                             danh_nghia_thap=r['ci_danh_nghia'][0], danh_nghia_cao=r['ci_danh_nghia'][1], nua_0=r['nua'][0], nua_1=r['nua'][1], q=fam['q'],
                             ket_luan=r['ket_luan'], mo_ta_dieu_khoan_tran=fam['mo_ta'], hieu_AUPRC_hinh_thang=k['auprc_thang']['d'],
                             AUPRC_danh_nghia_thap=k['auprc_thang']['ci_danh_nghia'][0], AUPRC_danh_nghia_cao=k['auprc_thang']['ci_danh_nghia'][1],
                             hieu_AUROC=k['auroc']['d'], AUROC_danh_nghia_thap=k['auroc']['ci_danh_nghia'][0], AUROC_danh_nghia_cao=k['auroc']['ci_danh_nghia'][1],
                             don_vi_bootstrap=sp.unit))
    if fam['du'] and not fam['mo_ta']:
        cnt = {h: {v: sum(1 for r in fam['phep_so'] if r['dau'] == h and r['ket_luan'] == v) for v in ('hon', 'kem', 'khong phan biet')} for h in hk}
        tot = {v: sum(c[v] for c in cnt.values()) for v in ('hon', 'kem', 'khong phan biet')}
        log(f"TONG ({fam['so_phep']} phep): B hon {tot['hon']} | B kem {tot['kem']} | khong phan biet {tot['khong phan biet']} — theo dau: "
            + ' | '.join(f"{ht[h]} {c['hon']}/{c['kem']}/{c['khong phan biet']}" for h, c in cnt.items()) + ' (hon / kem / khong phan biet)')
    return rows


def descriptive_tables(bo, out, log=print):
    '''Hiệu MÔ TẢ (CI danh nghĩa, không vào luật): giữa đầu đọc của cùng encoder; nhánh theo AP − luật.'''
    hk, ht = heads()
    log(f"\n--- MO TA: hieu giua DAU DOC (cung encoder; AP, CI95 danh nghia) --- {'hang':<14}" + ''.join(f"{ht[a] + ' - ' + ht[b]:>34}" for a, b in HEAD_PAIRS))
    for e in encoders(bo):
        line = f"{'':<4}{config.ROW_TITLES[e]:<14}"
        for a, b in HEAD_PAIRS:
            r = out['giua_dau'].get(f'{e}|{a}-{b}')
            line += (f"{d3(r['ap']['d'])} {dci(r['ap']['ci'])}" if r else '-').rjust(34)
        log(line)
    log("\n--- MO TA: nhanh 'theo AP' (lr co AP val cao nhat) - LUAT (lr co AUROC val cao nhat), chi o hai lr KHAC nhau ---")
    if not out['nhanh_ap']:
        log('    khong co o nao (hai tieu chi chon cung lr o moi o)')
    for r in out['nhanh_ap']:
        log(f"    {config.ROW_TITLES[r['encoder']]:<12}{ht[r['dau']]:<11} lr theo AP {r['lr_ap']:g} - lr cua luat {r['lr_luat']:g}: AP {d3(r['hieu']['ap']['d'])} "
            f"{dci(r['hieu']['ap']['ci'])} | AUROC {d3(r['hieu']['auroc']['d'])} {dci(r['hieu']['auroc']['ci'])}")


def edge_flags(bo, log=print):
    hk, ht = heads()
    fl = [f"{config.ROW_TITLES[e]} {ht[h]} lr {bo['cells'][(e, h, 'luat')]['lr']:g}" for e in encoders(bo) for h in hk
          if (e, h, 'luat') in bo['cells'] and bo['cells'][(e, h, 'luat')]['bien']]
    log('co "lr cua luat o bien luoi 13 lr" (luoi giu nhu 89, khong noi): ' + ('; '.join(fl) if fl else 'khong o nao'))
    return fl
