# -*- coding: utf-8 -*-
"""Ô của notebook Ver1 `89v1_<encoder>` (PLAN_89 §4; A100 80 GB). Mỗi hàm = một ô; trạng thái đi theo dict CTX; mọi việc Drive đi qua hàm của khối
Drive chuẩn ở C0 (drive_api lấy đúng các tên cần, thiếu -> dừng).

    C2 setup   thư viện notebook + môi trường thư viện của họ + repo CheXFound @ 7e288d0 + kiểm import; đọc tệp resume (nếu có)
    C3 data    kho đệm vuông KHÔNG cân bằng (= 86f) + CSV theo khuôn của họ + vân tay khoá / nhãn từng tách
    C4 gate    config.yaml của CheXFound (setup + autocast fp16 của họ) + cổng số học 6 tiến trình + nạp trọng số trong mỗi tiến trình
    C5 smoke   chạy thử 4 loại đầu (số đo vô nghĩa); bỏ qua khi resume đã có đầu xong
    C6 heads   Linear-1, Linear-4, ML-Decoder, GLoRI bằng mã của họ (resume từng loại đầu)
    C7 report  chấm panel 28 / 27 / 21 / 7 + bootstrap, cạnh Bảng IV CheXFound + Bảng 1a RAD-DINO (CHỈ MÔ TẢ); lỗi không làm mất C6
    C8 sync    dong_bo() + kiểm tệp trên Drive"""
import dataclasses
import json
import os
import traceback

from .. import __version__
from ..data import vindr as V
from ..encoders import BUILT, SPECS
from ..evaluation import panels as P
from ..protocols.chexfound_env import dataset as D
from ..protocols.chexfound_env import environment as E
from ..protocols.chexfound_env import probes
from ..protocols.chexfound_env import runner as R
from ..encoders import files as W

FOLDER = 'stage4_nb89'
DRIVE_NAMES = ('kiem_ke', 'tai', 'xung_dot', 'ghi', 'ghi_json', 'dong_bo', 'ten_tren_drive', 'tim_moi_noi', 'khoi_phuc_cho', 'luu_resume', 'cho_nho',
               'GHI_LOC', 'TEN_RESUME', 'RUN_TAG', '_duong', '_liet_ke')   # _duong / _liet_ke: tệp trọng số ngoài cxrlt_2024 (MIMIC_CXR/MRM, /CheXZero)
THEIR_MODEL = ('chexfound',)          # Ver1: mô hình + bộ nạp của chính mã họ (không lớp bọc), như 86f
HEADS_V1 = {'chexfound': ('lin1', 'mldec')}   # CheXFound: Linear-4 + GLoRI dùng lại 86f sau cổng PLAN_89 §4.3 (bảng gộp, bước 5)
WEIGHTS_PLACEHOLDER = 'khong_dung__lop_boc_thay_mo_hinh'      # --pretrained-weights: lớp bọc thay setup_and_build_model -> mã họ không đọc
PAPER_HEAD = {'lin4': 'Linear', 'glori': 'GLoRI'}               # cột Bảng IV CheXFound tương ứng (linear probing của họ = 4 tầng cuối)


def run_tag(encoder):
    '''_k64 (0.2.1, Tuấn 13/11 "chạy theo khúc cho đồng bộ"): mọi lượt Ver1 tính backbone theo khúc 64 ảnh. Lượt cũ r89v1_raddino (OOM ở Linear-4) và
       r89v1_biomedclip (không chia khúc) giữ trên Drive làm hồ sơ.'''
    return f'r89v1_{encoder}_k64'


def drive_api(ns):
    '''Các hàm / hằng của khối Drive chuẩn (C0) mà notebook dùng -> dict. Thiếu tên nào -> dừng (C0 chưa chạy / khối đổi).'''
    missing = [k for k in DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    return {k: ns[k] for k in DRIVE_NAMES}


def setup(encoder, package_dir, ns, log=print):
    '''C2. -> CTX.'''
    assert encoder in BUILT or encoder in THEIR_MODEL, f'{encoder}: lop boc chua dung ({BUILT}) -> DUNG'
    drive = drive_api(ns)
    assert drive['RUN_TAG'] == run_tag(encoder), f"RUN_TAG {drive['RUN_TAG']!r} != {run_tag(encoder)!r} -> DUNG"
    import vindr_bench
    mt = E.install_notebook_packages()
    ho = E.build_their_env(log=log)
    cf = E.clone_chexfound(os.environ.get('CHEXFOUND_DIR', '/content/CheXFound'))
    env = E.TheirEnv(python=ho['python'], repo=cf['repo'], package_dir=package_dir)
    pb = E.check_imports(env, expect_xformers=True)
    st = R.resume_state(drive['tim_moi_noi'](drive['TEN_RESUME']), drive['RUN_TAG'], drive['khoi_phuc_cho'], log=log)
    spec = SPECS[encoder]
    ctx = dict(encoder=encoder, spec=spec, drive=drive, env=env, out=f'/content/out_89v1_{encoder}', csv='/content/vindr_csv_89v1',
               kinds=HEADS_V1.get(encoder, R.HEAD_ORDER), mo_hinh_cua_ho=encoder in THEIR_MODEL,
               anh_dir='/content/vd518_kcb', moi_truong=mt, moi_truong_ho=ho, chexfound=cf, phien_ban=pb, tt=st,
               goi=dict(phien_ban=__version__, sha=getattr(vindr_bench, 'GOI_SHA', None)))
    i = ho['thong_tin']
    log(f"moi truong cua ho ({ho['phut']:.1f} phut): Python {i['python']} | torch {i['phien_ban']['torch']} (CUDA {i['cuda']}, cuDNN {i['cudnn']}) | "
        f"xFormers {i['phien_ban']['xformers']} (lech so attention tuong minh {i['xformers_lech']:.1e}) | transformers {i['phien_ban']['transformers']} | "
        f"timm {i['phien_ban']['timm']} | GPU {i['gpu']} {i['bo_nho_gb']:.0f} GB\n"
        f"ma cua ho dung xFormers o {pb['xformers_cua_ho']} | CheXFound @ {cf['head'][:12]} | notebook: Python {mt['python']}, torchmetrics "
        f"{mt['moi']['torchmetrics']}, transformers {mt['moi']['transformers']}, timm {mt['moi']['timm']} | o dia trong {mt['dia_trong_gb'] or 0:.0f} GB"
        + (f" | xung dot pip (ghi lai): {mt['xung_dot_pip']}" if mt['xung_dot_pip'] else ''))
    log(f"encoder {spec.title}: {spec.arch}, anh {spec.image_size}, chuan hoa goc {spec.mean} / {spec.std}, vector toan anh {spec.pool}"
        + (' | MO HINH + BO NAP CUA HO (khong lop boc, nhu 86f)' if ctx['mo_hinh_cua_ho'] else '') + f" | dau: {list(ctx['kinds'])} | goi vindr_bench "
        f"{__version__} @ {(ctx['goi']['sha'] or '?')[:12]}" + (f" | RESUME: da xong {sorted(st['xong'])}" if st['xong'] else ''))
    return ctx


def data(ctx, log=print):
    '''C3. Kho đệm vuông không cân bằng (Ver1) + CSV của họ. -> VinDrData.'''
    d = ctx['drive']
    vd = V.load_vindr(d['kiem_ke'], d['tai'], d['xung_dot'], ctx['anh_dir'], store=V.VER1_STORE, log=log)
    tach = D.write_split_csvs(vd, ctx['csv'])
    ctx.update(vindr=vd, tach=tach, van_tay=vd.fingerprint())
    log(vd.summary())
    log(f"CSV: train {tach['train']:,} / val {tach['val']:,} / test {tach['test']:,} anh x {len(tach['class_names'])} nhan | md5 khoa / nhan: "
        + ' | '.join(f"{s} {ctx['van_tay'][s]['md5_khoa'][:8]} / {ctx['van_tay'][s]['md5_nhan'][:8]}" for s in V.SPLITS))
    return vd


def gate(ctx, log=print):
    '''C4. config.yaml của CheXFound + Ver1Run + cổng số học (probes.numeric_gate). Trượt -> dừng.'''
    d, spec = ctx['drive'], ctx['spec']
    cfg = E.fetch_config(d['kiem_ke'], d['tai'], d['xung_dot'], log=log)
    ctx['tep_trong_so'] = W.fetch(spec, d, log=log) if spec.drive_dir else None
    if ctx['tep_trong_so']:
        t = ctx['tep_trong_so']
        log(f"trong so tu Drive: {spec.drive_dir}/{t['tep']} | md5 {t['md5']} = ghim | {t['byte']:,} B | {t['so_ban']} ban tren Drive")
    theirs = ctx['mo_hinh_cua_ho']
    ctx['run'] = R.Ver1Run(env=ctx['env'], encoder=None if theirs else ctx['encoder'], image_size=spec.image_size, csv_root=ctx['csv'],
                           config_path=cfg, weights_path=W.weights_path(spec) if theirs else WEIGHTS_PLACEHOLDER, out_root=ctx['out'])
    if theirs:
        ctx['cong'] = dict(mo_hinh_cua_ho=True, tep=ctx['tep_trong_so'])
        log('mo hinh + bo nap cua chinh ma CheXFound (nhu 86f): KHONG co lop boc -> khong co cong so hoc lop boc; nap trong so kiem trong MOI tien '
            'trinh hoc (so tung tensor voi tep, C5 / C6)')
        return ctx['cong']
    g = probes.numeric_gate(ctx['env'], ctx['encoder'], ctx['csv'], os.path.join(ctx['out'], 'cong_dac_trung'), spec.image_size, log=log)
    ctx['cong'] = g
    log(f"config.yaml cua CheXFound: {os.path.basename(cfg)} (chi setup + autocast fp16; trong so CheXFound KHONG dung) | cong so hoc: DAT")
    return g


def smoke(ctx, log=print):
    '''C5. Chạy thử 4 loại đầu; resume đã có đầu xong -> bỏ qua (đã đạt ở phiên trước).'''
    if ctx['tt']['xong']:
        log(f"RESUME (da xong {sorted(ctx['tt']['xong'])}): bo qua chay thu")
        ctx['thu'] = dict(bo_qua_vi_resume=True)
        return ctx['thu']
    thu = R.smoke_test(ctx['run'], kinds=ctx['kinds'], log=log)
    for k, v in thu.items():
        m = v['bo_nho'] or {}
        log(f"chay thu {R.HEADS[k]['title']:10s}: {'DAT' if v['dat'] else 'TRUOT'} | {v['so_dau']} dau | GPU dinh "
            f"{m.get('dinh_cap_phat_gib', float('nan')):.1f} / {m.get('tong_gib', float('nan')):.1f} GiB (nguong {R.MEM_LIMIT:.0%}: "
            f"{'DAT' if v['bo_nho_dat'] else 'TRUOT'}) | du doan {v['hinh_du_doan']} | {v['phut']:.1f} phut | lech tinh lai val "
            f"{v['cong']['val']['lech_max']:.1e} / test {v['cong']['test']['lech_max']:.1e} | tien trinh hoc: "
            + ('ML-Decoder ' + str([x['so_dau'] for x in v['kiem_tien_trinh']['mldecoder']]) + ' dau' if k == 'mldec' else 'nap trong so dat'))
    assert all(v['dat'] for v in thu.values()), (f'chay thu TRUOT: {[k for k, v in thu.items() if not v["dat"]]} (bo nho GPU / cong) -> DUNG truoc luot '
                                                 'day du, gui Claude output C5')
    ctx['thu'] = thu
    return thu


def heads(ctx, kinds=None, log=print):
    '''C6. Từng loại đầu có resume (tệp resume ghi đè sau mỗi loại đầu). -> {loại: kết quả}.'''
    import torch
    d = ctx['drive']
    st = ctx['tt']
    save = lambda tt: d['luu_resume'](lambda q: torch.save(dict(tt, cho=d['cho_nho']()), q))
    kinds = kinds or ctx['kinds']
    out = {}
    for k in kinds:
        out[k] = R.head_step(ctx['run'], k, st, d['GHI_LOC'], d['ghi'], save, log=log)
        log(R.head_summary(out[k]))
    ctx['kq'] = out
    return out


def _fmt(x):
    return 'nan' if x is None or x != x else f'{x:.3f}'


def _fmt_ci(b):
    return f"{_fmt(b['tb'])} [{_fmt(b['ci'][0])}, {_fmt(b['ci'][1])}]"


def score_heads(ctx, log=print):
    '''Chấm dự đoán test đã lưu của từng loại đầu (target = test.csv theo thứ tự). -> {loại: điểm}.'''
    import numpy as np
    vd = ctx['vindr']
    y_test = vd.labels[vd.indices('test')].astype(np.int64)
    out = {}
    for k, r in ctx['kq'].items():
        with np.load(r['du_doan']) as z:
            preds, target = z['preds'], z['target']
        assert np.array_equal(target, y_test), f'{k}: target du doan KHAC nhan test (thu tu) -> DUNG'
        out[k] = P.score_predictions(preds, target, vd.label_names, vd.main_classes)
    return out


def report(ctx, log=print):
    '''C7 (CHỈ MÔ TẢ). Lỗi ở đây KHÔNG làm mất C6: in lỗi, trả None — C8 vẫn đồng bộ.'''
    try:
        enc, d, st = ctx['encoder'], ctx['drive'], ctx['tt']
        sc = score_heads(ctx, log=log)
        paper = P.PAPER_TABLE_IV.get(enc, {})
        log(f"\n== Ver1 {ctx['spec'].title} — ma + moi truong CheXFound (A100), kho dem vuong KHONG can bang, anh {ctx['spec'].image_size}, chuan hoa goc, "
            f"1 luot hat 0; dau doc train() khi cham (dropout bat, nhu ma ho); CI = bootstrap {P.N_BOOT} lan theo anh (danh nghia) ==")
        for k, s in sc.items():
            r, b = ctx['kq'][k], s['bootstrap']
            log(f"{R.HEADS[k]['title']:10s} lr {r['chon']['lr']:g} (AUROC val {r['chon']['auroc_val']:.4f}) | p28 torchmetrics: AP {s['p28_tm']['ap']:.3f} "
                f"(bootstrap {_fmt_ci(b['p28']['ap'])}) AUROC {s['p28_tm']['auroc']:.3f} | p27: AP {_fmt(s['diem']['p27']['ap'])} AUPRC thang "
                f"{_fmt(s['diem']['p27']['auprc_thang'])} AUROC {_fmt(s['diem']['p27']['auroc'])}")
            log(f"{'':10s} p21: AP {_fmt(s['diem']['p21']['ap'])} {_fmt_ci(b['p21']['ap'])} AUPRC thang {_fmt(s['diem']['p21']['auprc_thang'])} AUROC "
                f"{_fmt(s['diem']['p21']['auroc'])} (loai {b['p21']['so_lan_loai']}/{b['p21']['so_lan']}) | hai nua AP21 {_fmt(s['ap21_hai_nua'][0])} / "
                f"{_fmt(s['ap21_hai_nua'][1])} | p7: AP {_fmt(s['diem']['p7']['ap'])} {_fmt_ci(b['p7']['ap'])} AUPRC thang {_fmt(s['diem']['p7']['auprc_thang'])} "
                f"(loai {b['p7']['so_lan_loai']})")
            if k in PAPER_HEAD and k in paper:
                p = paper[k]
                log(f"{'':10s} bai CheXFound Bang IV ({PAPER_HEAD[k]}, {ctx['spec'].title}; baseline cua bai, cach chay khong cong bo): AP {p['ap']:.3f} "
                    f"[{p['ap_ci'][0]:.3f}, {p['ap_ci'][1]:.3f}] AUROC {p['auroc']:.3f} [{p['auroc_ci'][0]:.3f}, {p['auroc_ci'][1]:.3f}]")
        log(f"bai RAD-DINO Bang 1a (7 dau hieu, checkpoint cua bai, cach tinh AUPRC khong noi): {P.RADDINO_PAPER_7:.3f} — chi la moc tham khao")
        name = f"bang_89v1_{enc}_{st['t']}.json"
        spec = dataclasses.asdict(ctx['spec'])
        obj = dict(run=d['RUN_TAG'], t=st['t'], goi=ctx['goi'], encoder=spec, phien_ban_ver='Ver1', kho=V.VER1_STORE, moi_truong=ctx['moi_truong'],
                   moi_truong_ho=ctx['moi_truong_ho'], chexfound=ctx['chexfound'], phien_ban=ctx['phien_ban'], env=ctx['env'].to_dict(),
                   tach=ctx['tach'], van_tay=ctx['van_tay'], nguon_du_lieu=ctx['vindr'].sources, cong=ctx['cong'], chay_thu=ctx.get('thu'),
                   ket_qua_dau=ctx['kq'], diem=sc, luat=dict(n_boot=P.N_BOOT, hat_boot=P.BOOT_SEED, hat_nua=P.HALF_SEED))
        obj = json.loads(json.dumps(obj, default=str))
        d['ghi'](name, lambda q: d['ghi_json'](q, obj))
        ctx['tep_bang'] = name
        log(f'\nda ghi {name} (local; len Drive o C8)')
        return sc
    except Exception:
        log('⚠ C7 LOI — ket qua C6 KHONG mat: chay tiep C8, roi gui Claude loi duoi day')
        traceback.print_exc()
        return None


def sync(ctx, kinds=None, log=print):
    '''C8. dong_bo() + kiểm 3 tệp mỗi loại đầu + bảng C7 trên Drive.'''
    d = ctx['drive']
    kinds = kinds or ctx['kinds']
    du = d['dong_bo']()
    need = [t for k in kinds for t in ctx['tt']['xong'].get(k, [])]
    bg = ctx.get('tep_bang')
    on = d['ten_tren_drive']()
    co = len(need) == 3 * len(kinds) and set(need) <= on
    log(f"C8: dong_bo {'DU' if du else 'CHUA DU'} | {len(kinds)} loai dau x 3 tep tren Drive: {'CO' if co else 'THIEU / C6 CHUA XONG'} | bang C7: "
        f"{'CO' if bg and bg in on else 'THIEU (C7 loi / chua chay)'}"
        + (' -> gui Claude output C2-C8; roi chay 89c (chep sang corntoun1505)' if du and co else ' | KHONG tat runtime; chay lai C8, roi bao Claude'))
    return dict(dong_bo=du, du_tep=co, bang=bool(bg and bg in on))
