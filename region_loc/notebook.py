# -*- coding: utf-8 -*-
"""Ô của notebook `90c_dinh_vi_*` (PLAN_90 §11). Mỗi hàm = một ô; trạng thái đi theo dict CTX; mọi việc Drive qua hàm của khối Drive chuẩn ở C0.

    C2 setup     khối Drive + thiết bị + phiên bản hai gói + chế độ (định cỡ trên phần train | val + test khi ngưỡng đã khoá)
    C3 data      tệp hộp (sha256 ghim) + split CXR-LT + CỔNG số đếm ghim -> chỉ giữ phần được phép -> khung toạ độ + cổng hình học -> ảnh từ kho 518
    C4 model     tệp trọng số (md5 ghim) -> mô hình + ba đầu (nạp chặt) + cổng thứ tự vùng
    C5 run       forward mọi ảnh cần (một lượt) -> hộp, điểm theo vùng, attention của 8 dấu hiệu
    C6 calibrate phần TRAIN (ảnh mô hình đã thấy): sàn / trần / độ rộng khoảng tin cậy của từng số đo -> số để khoá ngưỡng. KHÔNG phải kết quả.
    C7 evaluate  val + test: chỉ chạy khi config.THRESHOLDS đã khoá (bản gói sau); chưa khoá -> in lý do, không đọc gì
    C8 sync      dong_bo()

Chỉ in SỐ TỔNG HỢP (luật PhysioNet: không ảnh, không hộp / nhãn từng ảnh trong output); mảng từng đơn vị ghi vào tệp trên Drive."""
import hashlib
import json
import os
import time

import numpy as np

from . import __version__
from . import config as C
from . import geometry as G
from . import images as IM
from . import maps as MP
from . import metrics as MT
from . import model as MD
from . import tables as T

DRIVE_NAMES = ('kiem_ke', 'tai', 'ghi', 'ghi_json', 'dong_bo', 'GHI_LOC', 'RUN_TAG', '_duong', '_liet_ke')
RUN_TAG = 'r90c_dinh_vi_arre_s42'
IMAGE_DIR = '/content/anh_518_dinh_vi'
SOURCES = ('attention', 'nhan_vung', 'bat_thuong')       # ba bản đồ của C2


def drive_api(ns):
    missing = [k for k in DRIVE_NAMES if k not in ns]
    assert not missing, f'thieu {missing} cua khoi Drive chuan (chay C0 truoc) -> DUNG'
    return {k: ns[k] for k in DRIVE_NAMES}


def _sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def _versions(drive, where, path, name):
    '''where = 'cxrlt' (đường dẫn tính từ cxrlt_2024, qua kiem_ke) | 'mydrive' (tính từ My Drive, qua _duong + _liet_ke). -> danh sách bản, mới trước.'''
    if where == 'cxrlt':
        return list(drive['kiem_ke'](tuple(path))[1].get(name) or [])
    folder = drive['_duong'](tuple(path))
    if not folder:
        return []
    out = [dict(id=f['id'], ten=f['name'], md5=f.get('md5Checksum'), size=int(f.get('size') or 0), mt=f.get('modifiedTime', ''))
           for f in drive['_liet_ke'](folder) if f['name'] == name]
    return sorted(out, key=lambda b: b['mt'], reverse=True)


def fetch_file(drive, places, name, sha256=None):
    '''Lấy tệp `name` từ nơi đầu tiên có bản khớp sha256 (None = bản mới nhất). places = [(where, path), ...]. -> (đường dẫn local, mô tả nơi lấy).'''
    seen = []
    for where, path in places:
        for v in _versions(drive, where, path, name):
            p = drive['tai'](v)
            got = _sha256(p) if sha256 else None
            seen.append((where, '/'.join(path), got))
            if sha256 is None or got == sha256:
                return p, f"{'cxrlt_2024' if where == 'cxrlt' else 'My Drive'}/{'/'.join(path)}/{name}"
    raise AssertionError(f'{name}: khong co ban nao khop sha256 ghim {sha256} o {[(w, "/".join(p)) for w, p in places]} (da thu {seen}) -> DUNG')


def _imagenome_places(sub):
    return [('cxrlt', C.IMAGENOME_COPY + (sub,)), ('cxrlt', C.IMAGENOME_COPY + ('1.0.0', sub)), ('mydrive', C.IMAGENOME_ORIGINAL + (sub,))]


def setup(ns, log=print):
    import torch
    import vindr_bench
    drive = drive_api(ns)
    assert drive['RUN_TAG'] == RUN_TAG, f"RUN_TAG cua C0 {drive['RUN_TAG']!r} khac {RUN_TAG!r} -> notebook khong phai ban dung cho goi nay -> DUNG"
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert dev == 'cuda', 'can GPU (Runtime > Change runtime type) -> DUNG'
    mode = 'dinh_co' if C.THRESHOLDS is None else 'val_test'
    ctx = dict(drive=drive, log=log, device=dev, mode=mode, splits=('train',) if mode == 'dinh_co' else C.SPLITS, t0=time.time())
    log(f"region_loc {__version__} @ {str(ns.get('LOC_SHA', '?'))[:12]} | vindr_bench {vindr_bench.__version__} @ {str(getattr(vindr_bench, 'GOI_SHA', '?'))[:12]} | "
        f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    log(f"che do: {'DINH CO tren phan TRAIN (nguong chua khoa -> val / test KHONG duoc doc)' if mode == 'dinh_co' else 'val + test (nguong da khoa)'} | "
        f'RUN_TAG {RUN_TAG} | ghi {C.FOLDER}')
    return ctx


def data(ctx):
    drive, log = ctx['drive'], ctx['log']
    p_names, w0 = fetch_file(drive, _imagenome_places(C.REGION_LIST[0]), C.REGION_LIST[1], C.REGION_LIST[2])
    p_gold, w1 = fetch_file(drive, _imagenome_places(C.GOLD_BOX[0]), C.GOLD_BOX[1], C.GOLD_BOX[2])
    p_auto, w2 = fetch_file(drive, _imagenome_places(C.AUTO_BOX[0]), C.AUTO_BOX[1], C.AUTO_BOX[2])
    p_ms, w3 = fetch_file(drive, [('mydrive', C.MSCXR_DIR)], C.MSCXR_FILE[0], C.MSCXR_FILE[1])
    p_order, w4 = fetch_file(drive, [('cxrlt', C.ORDER_PATH[:-1])], C.ORDER_PATH[-1])
    p_meta, w5 = fetch_file(drive, [('mydrive', C.META_DIR)], C.META_FILE)
    for w in (w0, w1, w2, w3, w4, w5):
        log(f'  tep: {w}')
    names, order = T.region_names(p_names), T.read_order(p_order)
    # ---- đếm theo split TRƯỚC, rồi chỉ giữ phần được phép; hộp của phần còn lại không được dùng ----
    gold = T.attach_split(T.read_gold(p_gold), order, 'gold')
    ms = T.attach_split(T.read_mscxr(p_ms), order, 'MS-CXR')
    n_gold, n_ms = T.check_counts(gold, C.GOLD_COUNTS, 'gold ImaGenome'), T.check_counts(ms, C.MSCXR_COUNTS, 'MS-CXR')
    assert gold.bbox_name.nunique() == C.N_GOLD_REGIONS, gold.bbox_name.nunique()
    gold, ms = T.keep(gold, ctx['splits']), T.keep(ms, ctx['splits'])
    auto = T.read_auto(p_auto)
    auto = auto[auto.dicom_id.isin(set(gold.dicom_id))].reset_index(drop=True)
    log(f"so dem theo split (anh, benh nhan, hop) = so ghim | gold {n_gold} | MS-CXR {n_ms} -> GIU {ctx['splits']}: gold {gold.dicom_id.nunique()} anh / "
        f'{len(gold)} hop, MS-CXR {ms.dicom_id.nunique()} anh / {len(ms)} hop')
    sizes = T.read_sizes(p_meta, gold.dicom_id.unique())
    gold, g224 = T.gold_frame(gold, sizes, names)
    auto = T.auto_frame(auto, gold, names)
    ms = T.mscxr_frame(ms)
    log(f"cong hinh hoc (toa do 224 cua tep gold so voi letterbox 224 tu hop goc + kich thuoc metadata): {g224['share_224']:.2%} hop lech <= {C.TOL_224_PX} px "
        f"(trung vi {g224['err_224_median']:.3f}, lon nhat {g224['err_224_max']:.2f} px) -> DAT")
    ids = sorted(set(gold.dicom_id) | set(ms.dicom_id))
    st = IM.fetch_images(drive, ids, IMAGE_DIR, log)
    log(f"anh: {st['n_can']} can | {st['n_co_san']} co san | {st['n_lay']} lay tu {st['n_tar_doc']} / {st['n_tar']} tep .tar cua kho 518")
    ctx.update(names=names, gold=gold, auto=auto, ms=ms, ids=ids, cong_224=g224, so_dem=dict(gold=n_gold, mscxr=n_ms))
    return dict(n_anh=len(ids), gold=len(gold), auto=len(auto), mscxr=len(ms))


def model(ctx):
    import torch
    from vindr_bench.encoders import files
    from vindr_bench.encoders.spec import SPECS
    drive, log = ctx['drive'], ctx['log']
    got = files.fetch(SPECS[C.ENCODER], drive, log=log)
    loc = MD.build().to(ctx['device'])
    info = loc.info
    log(f"mo hinh: {info['run']} ep {info['ep']} | tep {got['tep']} md5 {str(got['md5'])[:8]} | dau doc {info['n_head']} tensor, docv {info['n_docv']}, "
        f"dauv {info['n_dauv']} (nap chat) | kenh nhan-vung {len(info['mil_idx'])} nhan, du 8 dau hieu MS-CXR")
    log(f"nhat ky huan luyen (nhom train giu rieng, hop silver): IoU luong {info['nhat_ky'].get('luong_iou')} | hop trung binh "
        f"{info['nhat_ky'].get('luong_iou_tb')} | AUC bat thuong {info['nhat_ky'].get('luong_auc_bt')}")
    tr = ctx['gold'][ctx['gold'].split == 'train']
    if len(tr):
        og = MD.order_gate(loc.enc.luong.hop_tb.float().cpu().numpy(), tr)
        log(f"cong thu tu vung (hop_tb cua checkpoint so voi hop gold trung binh theo ten, phan train): {og['n_vung']} vung, IoU TB {og['iou_tb']:.3f}, nho nhat "
            f"{og['iou_min']:.3f} (>= {C.MIN_ORDER_IOU}); moi vung khop nhat voi chinh no -> DAT")
        ctx['cong_thu_tu'] = og
    else:
        log('cong thu tu vung: khong co hang train trong che do nay -> da kiem o luot dinh co')
    ctx.update(loc=loc, hop_tb=G.cxcywh_to_xyxy(loc.enc.luong.hop_tb.float().cpu().numpy()), mean=float(SPECS[C.ENCODER].mean[0]),
               std=float(SPECS[C.ENCODER].std[0]), mo_hinh=info)
    torch.cuda.reset_peak_memory_stats()
    return dict(run=info['run'], ep=info['ep'])


def run(ctx):
    import torch
    loc, ids, log = ctx['loc'], ctx['ids'], ctx['log']
    jf = [C.LAB40.index(f) for f in C.FINDINGS]
    box, bt, z, attn = [], [], [], []
    t0 = time.time()
    for i in range(0, len(ids), C.BATCH):
        x = IM.load_batch(IMAGE_DIR, ids[i:i + C.BATCH], ctx['mean'], ctx['std']).to(ctx['device'])
        o = loc(x)
        box.append(o['box'].numpy()); bt.append(o['bt'].numpy()); z.append(o['z'][:, :, jf].numpy()); attn.append(o['attn'][:, jf].numpy())
        del x, o
    out = dict(box=np.concatenate(box), bt=np.concatenate(bt), z=np.concatenate(z), attn=np.concatenate(attn))
    n = len(ids)
    assert out['box'].shape == (n, C.N_REGIONS, 4) and out['z'].shape == (n, C.N_REGIONS, len(jf)) and out['attn'].shape == (
        n, len(jf), C.GRID * C.GRID + C.N_REGIONS) and all(np.isfinite(v).all() for v in out.values()), {k: v.shape for k, v in out.items()}
    tong = out['attn'].sum(-1)
    assert float(np.abs(tong - 1).max()) < 1e-3, f'attention khong co tong 1 (lech {np.abs(tong - 1).max():.2e}) -> DUNG'
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    log(f'forward {n} anh, lo {C.BATCH}: {time.time() - t0:.0f} giay | VRAM dinh {peak:.1f} / {tot:.0f} GB ({peak / tot:.0%})')
    ctx.update(out=out, row={d: i for i, d in enumerate(ids)})
    return dict(n=n, vram_dinh_gb=round(peak, 2))


def _c1(ctx, split):
    '''C1 trên một split: đơn vị = (ảnh, vùng) có hộp gold. -> (bảng từng đơn vị, tổng hợp).'''
    g = ctx['gold'][ctx['gold'].split == split]
    rows, k = np.array([ctx['row'][d] for d in g.dicom_id]), g.k.to_numpy()
    gb = g[['g1', 'g2', 'g3', 'g4']].to_numpy()
    i_m = G.iou(ctx['out']['box'][rows, k], gb)
    i_0 = G.iou(ctx['hop_tb'][k], gb)
    a = g[['dicom_id', 'k']].merge(ctx['auto'], on=['dicom_id', 'k'], how='left')
    i_a = np.full(len(g), np.nan)
    co = a.a1.notna().to_numpy()
    i_a[co] = G.iou(a.loc[co, ['a1', 'a2', 'a3', 'a4']].to_numpy(), gb[co])
    pat, kw = g.subject_id.to_numpy(), dict(n_boot=C.N_BOOT, seed=C.BOOT_SEED, key_names=ctx['names'])
    s = dict(luong=MT.summarize(i_m, k, pat, **kw), hop_tb=MT.summarize(i_0, k, pat, **kw), tu_dong=MT.summarize(i_a, k, pat, **kw),
             luong_tru_hop_tb=MT.summarize(i_m - i_0, k, pat, **kw), luong_tru_tu_dong=MT.summarize(i_m - i_a, k, pat, **kw),
             ti_le_05=dict(luong=MT.share_at(i_m, 0.5), hop_tb=MT.share_at(i_0, 0.5), tu_dong=MT.share_at(i_a, 0.5)),
             n_anh=int(g.dicom_id.nunique()))
    return dict(dicom_id=g.dicom_id.to_numpy(), k=k, iou_luong=i_m, iou_hop_tb=i_0, iou_tu_dong=i_a), s


def _c2(ctx, split):
    '''C2 trên một split: đơn vị = (ảnh, dấu hiệu); mặt nạ = hợp các hộp tổn thương của dấu hiệu đó trong ảnh. -> (bảng từng đơn vị, tổng hợp).'''
    m = ctx['ms'][ctx['ms'].split == split]
    o, sig = ctx['out'], lambda v: 1.0 / (1.0 + np.exp(-v))
    rec = []
    for (d, j), gr in m.groupby(['dicom_id', 'j'], sort=True):
        r, f = ctx['row'][d], C.FINDINGS.index(C.LAB40[int(j)])
        mask = MP.box_mask(gr[['b1', 'b2', 'b3', 'b4']].to_numpy(), C.GRID)
        am, share = MP.attention_map(o['attn'][r, f], C.GRID)
        ms_ = dict(attention=am, nhan_vung=MP.paint(o['box'][r], sig(o['z'][r, :, f]), C.GRID), bat_thuong=MP.paint(o['box'][r], sig(o['bt'][r]), C.GRID))
        rec.append(dict(dicom_id=d, f=f, benh_nhan=gr.subject_id.iloc[0], dien_tich=MP.area(mask), attention_o_token_vung=share,
                        **{f'trung_{s}': MP.hit(ms_[s], mask) for s in SOURCES}, **{f'kl_{s}': MP.energy(ms_[s], mask) for s in SOURCES}))
    u = {k: np.array([x[k] for x in rec]) for k in rec[0]}
    kw = dict(n_boot=C.N_BOOT, seed=C.BOOT_SEED, key_names=list(C.FINDINGS))
    s = dict(dien_tich=MT.summarize(u['dien_tich'], u['f'], u['benh_nhan'], **kw),
             attention_o_token_vung=MT.summarize(u['attention_o_token_vung'], u['f'], u['benh_nhan'], **kw), n_don_vi=len(rec))
    for src in SOURCES:
        for kieu in ('trung', 'kl'):
            v = u[f'{kieu}_{src}']
            s[f'{kieu}_{src}'] = MT.summarize(v, u['f'], u['benh_nhan'], **kw)
            s[f'{kieu}_{src}_tru_may_rui'] = MT.summarize(v - u['dien_tich'], u['f'], u['benh_nhan'], **kw)
    return u, s


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def _report(ctx, split):
    log = ctx['log']
    u1, s1 = _c1(ctx, split)
    u2, s2 = _c2(ctx, split)
    log(f"\n==== C1 giai phau | {split} | {s1['n_anh']} anh, {s1['luong']['n']} hop gold, {len(s1['luong']['per_key'])} vung | IoU so voi hop gold ====")
    log(f"  luong vung (ArrE)                : {MT.fmt(s1['luong'])} | IoU >= 0,5: {s1['ti_le_05']['luong']:.3f}")
    log(f"  san: hop trung binh co dinh      : {MT.fmt(s1['hop_tb'])} | IoU >= 0,5: {s1['ti_le_05']['hop_tb']:.3f}")
    log(f"  tran tham chieu: hop tu dong     : {MT.fmt(s1['tu_dong'])} | IoU >= 0,5: {s1['ti_le_05']['tu_dong']:.3f}  (gold lam bang cach SUA cac hop nay)")
    log(f"  luong - san (ghep cap)           : {MT.fmt(s1['luong_tru_hop_tb'])}")
    log(f"  luong - tran tham chieu (ghep cap): {MT.fmt(s1['luong_tru_tu_dong'])}")
    log('  theo vung (IoU luong / san / n hop):')
    for name in ctx['names']:
        if name in s1['luong']['per_key']:
            a, n = s1['luong']['per_key'][name]
            log(f"    {name:<28} {a:.3f} / {s1['hop_tb']['per_key'][name][0]:.3f} / {n}")
    log(f"\n==== C2 ton thuong | {split} | {s2['n_don_vi']} cap (anh, dau hieu) | muc may rui = ti le dien tich hop ====")
    log(f"  dien tich hop (may rui)          : {MT.fmt(s2['dien_tich'])}")
    log(f"  phan attention o 36 token vung   : {MT.fmt(s2['attention_o_token_vung'])}")
    ten = dict(attention='attention cua dau doc', nhan_vung='diem nhan theo vung', bat_thuong='co bat thuong tung vung')
    for src in SOURCES:
        log(f"  [{ten[src]}]")
        log(f"    trung diem                     : {MT.fmt(s2['trung_' + src])}")
        log(f"    trung diem - may rui (ghep cap): {MT.fmt(s2['trung_' + src + '_tru_may_rui'])}")
        log(f"    khoi luong trong hop           : {MT.fmt(s2['kl_' + src])}")
        log(f"    khoi luong - may rui (ghep cap): {MT.fmt(s2['kl_' + src + '_tru_may_rui'])}")
    log('  theo dau hieu (n | dien tich | trung diem: attention / nhan-vung / bat thuong):')
    for f in C.FINDINGS:
        if f in s2['dien_tich']['per_key']:
            a, n = s2['dien_tich']['per_key'][f]
            log(f"    {f:<17} {n:>4} | {a:.3f} | " + ' / '.join(f"{s2['trung_' + s]['per_key'][f][0]:.3f}" for s in SOURCES))
    return dict(c1=s1, c2=s2), dict(c1=u1, c2=u2)


def _write(ctx, name, summary, units):
    drive = ctx['drive']
    drive['ghi'](f'{name}.json', lambda q: drive['ghi_json'](q, _jsonable(summary)))
    flat = {f'{a}__{k}': np.asarray(v) for a, d in units.items() for k, v in d.items()}
    drive['ghi'](f'{name}_don_vi.npz', lambda q: np.savez_compressed(q, **flat))


def calibrate(ctx):
    '''Phần TRAIN: ảnh mô hình ĐÃ THẤY lúc học (có hộp silver gần trùng gold) -> chỉ để định cỡ sàn / trần / độ rộng khoảng tin cậy và bắt lỗi đường chấm.'''
    log = ctx['log']
    log('PHAN TRAIN — anh mo hinh da thay luc hoc: so duoi day de DINH CO va khoa nguong, KHONG phai ket qua cua phep thu.')
    s, u = _report(ctx, 'train')
    hon = s['c1']['luong']['pooled'] - s['c1']['hop_tb']['pooled']
    assert hon >= C.MIN_GAIN_OVER_MEAN_BOX, (f'cong duong cham: IoU luong - san = {hon:.3f} < {C.MIN_GAIN_OVER_MEAN_BOX} tren gold train -> sai hinh hoc / thu tu '
                                             f'vung / trong so -> DUNG')
    log(f"\ncong duong cham: IoU luong - san = {hon:.3f} >= {C.MIN_GAIN_OVER_MEAN_BOX} -> DAT | nhat ky huan luyen (giu rieng, silver): "
        f"{ctx['mo_hinh']['nhat_ky'].get('luong_iou')}")
    summary = dict(che_do='dinh_co', split='train', goi=__version__, run=RUN_TAG, mo_hinh=ctx['mo_hinh'], so_dem=ctx['so_dem'], cong_224=ctx['cong_224'],
                   cong_thu_tu=ctx.get('cong_thu_tu'), n_boot=C.N_BOOT, hat_boot=C.BOOT_SEED, **s)
    _write(ctx, 'dinh_co_train', summary, u)
    ctx['dinh_co'] = summary
    return dict(iou_luong=s['c1']['luong']['pooled'], iou_san=s['c1']['hop_tb']['pooled'], iou_tran=s['c1']['tu_dong']['pooled'])


def evaluate(ctx):
    '''val + test — CHỈ khi ngưỡng đã khoá (config.THRESHOLDS). Chưa khoá: không đọc gì, in lý do (Run all vẫn chạy tới ô sync).'''
    log = ctx['log']
    if C.THRESHOLDS is None:
        log('val / test: CHUA DOC — nguong cua luat doc chua khoa (PLAN_90 §11.5). Gui output C6 cho Claude; nguong khoa o ban goi sau, roi moi mo val / test.')
        return None
    assert ctx['mode'] == 'val_test', ctx['mode']
    res = {}
    for split in ('val', 'test'):
        s, u = _report(ctx, split)
        _write(ctx, f'dinh_vi_{split}', dict(che_do='val_test', split=split, goi=__version__, run=RUN_TAG, nguong=C.THRESHOLDS, mo_hinh=ctx['mo_hinh'],
                                             so_dem=ctx['so_dem'], n_boot=C.N_BOOT, hat_boot=C.BOOT_SEED, **s), u)
        res[split] = s
    return res


def sync(ctx):
    ctx['drive']['dong_bo']()
    ctx['log'](f"xong sau {(time.time() - ctx['t0']) / 60:.1f} phut")
