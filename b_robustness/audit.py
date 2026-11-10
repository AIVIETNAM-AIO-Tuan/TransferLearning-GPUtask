"""K0 (PLAN_KIEM_THU §3, §15.3): kiểm nguồn từng backbone seed B qua Drive API — chỉ tên / md5 / số tệp / trường checkpoint, không in id.

Thư mục Phần 1 (x): rs_J_bert_dong_ep5.pt (run = thư mục, ep = 4 đếm từ 0, có bb / luong / dua_o / bo_o7 / ema = EMA, không bo_o11 / gop_o),
backbone_ft.pt (run, luot_79a = J_bert_dong, cfg_d4, hop_tb trùng rs_), epoch_log_J_bert_dong_ep6.json.
Thư mục Phần 2 (z): nguon_backbone.json (epoch + md5 khớp rs_ ep5), w_s<seed>_ep1–10.pt, p_<đầu>_s<seed>_ep1–10.npy.
Thư mục Phần 3 (gts): JSON gộp trọng số -> số quyết định, đối chiếu số in ở notebook đã chạy.
"Đủ cho P2" = rs_ ep5 + module đúng; "đủ cho P1" = + w_ / p_ đủ + nguon_backbone.json khớp (cổng mạnh của ô T)."""
import gc
import hashlib
import re

import torch

from . import config

HEADS_PART2 = config.P1_ALL_HEADS


def _md5_tensor(t):
    return hashlib.md5(t.float().cpu().numpy().tobytes()).hexdigest()


def required_files(seed):
    ep = config.K0_BACKBONE_EPOCH
    return dict(x=[f'rs_J_bert_dong_ep{ep}.pt', 'backbone_ft.pt', 'epoch_log_J_bert_dong_ep6.json'],
                z=['nguon_backbone.json'] + [f'w_s{seed}_ep{e}.pt' for e in config.K0_PART2_EPOCHS]
                  + [f'p_{d}_s{seed}_ep{e}.npy' for d in HEADS_PART2 for e in config.K0_PART2_EPOCHS])


def inventory(reader, sources=config.K0_SOURCES, log=print):
    '''Tệp có / thiếu / nhiều bản ở thư mục Phần 1 / 2 / 3 của từng seed.'''
    out = {}
    for h, c in sources.items():
        need = required_files(c['seed'])
        kk = {p: reader.inventory(c[p], r'.+') for p in ('x', 'z', 'gts')}
        out[h] = dict(kk=kk, can=need, thieu={p: [t for t in need[p] if not kk[p].get(t)] for p in ('x', 'z')},
                      nhieu_ban={p: [t for t in need[p] if len(kk[p].get(t) or []) > 1] for p in ('x', 'z')},
                      gts_json=sorted(t for t in kk['gts'] if re.fullmatch(r'gop_trong_so_\d+_\d+\.json', t)), so_tep={p: len(kk[p]) for p in kk})
        k = out[h]
        log(f"{h:<10} {c['variant']:<36} | Phan 1 thieu {len(k['thieu']['x'])} / {len(need['x'])} | Phan 2 thieu {len(k['thieu']['z'])} / {len(need['z'])} | "
            f"Phan 3 json {len(k['gts_json'])} | tep nhieu ban {sum(len(v) for v in k['nhieu_ban'].values())}"
            + (f" | THIEU {(k['thieu']['x'] + k['thieu']['z'])[:4]}" if k['thieu']['x'] or k['thieu']['z'] else ''))
    return out


def inspect(reader, inv, sources=config.K0_SOURCES, log=print):
    '''Nội dung checkpoint + số quyết định -> {hàng: dict(rs, ft, nguon_p2, so_qd_drive, p2_san_sang, p1_san_sang, so_qd_khop)}.'''
    ep = config.K0_BACKBONE_EPOCH
    res = {}
    for h, c in sources.items():
        k, r = inv[h], dict(seed=c['seed'], bien_the=c['variant'], k3=c['k3'])
        name = f'rs_J_bert_dong_ep{ep}.pt'
        if k['kk']['x'].get(name):
            v = k['kk']['x'][name]
            ck = torch.load(reader.fetch(c['x'], k['kk']['x'], name), map_location='cpu', weights_only=False)
            r['rs'] = dict(md5=v[0]['md5'], run=ck.get('run'), luot=ck.get('luot'), ep=ck.get('ep'),
                           co=[x for x in ('bb', 'luong', 'dua_o', 'bo_o7', 'ema', 'log') if x in ck], cam=[x for x in ('bo_o11', 'gop_o') if x in ck],
                           ema=(ck.get('ema') or {}).get('trong_so'), val_ep=((ck.get('log') or {}).get(str(ep - 1)) or {}).get('mAP_anh'),
                           hop=_md5_tensor(ck['luong']['hop_tb']) if isinstance(ck.get('luong'), dict) and 'hop_tb' in ck['luong'] else None)
            del ck; gc.collect()
        if k['kk']['x'].get('backbone_ft.pt'):
            v = k['kk']['x']['backbone_ft.pt']
            cf = torch.load(reader.fetch(c['x'], k['kk']['x'], 'backbone_ft.pt'), map_location='cpu', weights_only=False)
            cd4 = cf.get('cfg_d4') or {}
            r['ft'] = dict(md5=v[0]['md5'], run=cf.get('run'), luot_79a=cf.get('luot_79a'),
                           cfg_d4=dict(ten=cd4.get('ten'), epochs=cd4.get('epochs'), trong_so=cd4.get('trong_so'), khe=cd4.get('khe')),
                           co_cfg=[x for x in ('cfg_luong', 'cfg_jv2', 'cfg_jv3', 'cfg_jv3b', 'cfg_o') if isinstance(cf.get(x), dict)],
                           cfg_o=(cf.get('cfg_o') or {}).get('ten'), hop=_md5_tensor(cf['hop_tb']) if 'hop_tb' in cf else None)
            del cf; gc.collect()
        if k['kk']['z'].get('nguon_backbone.json'):
            n2 = reader.load_json(c['z'], k['kk']['z'], 'nguon_backbone.json')
            r['nguon_p2'] = dict(tep=n2.get('tep'), ep=n2.get('ep'), md5=n2.get('md5'), duong=n2.get('duong'), quy_tac=(n2.get('chon_ep') or {}).get('quy_tac'))
        if k['gts_json']:
            j3 = reader.load_json(c['gts'], k['kk']['gts'], k['gts_json'][-1])
            q3 = (j3.get('ket_qua') or {}).get('fus_attn_hr | gop trong so ep1-10') or {}
            r['so_qd_drive'] = None if q3.get('toan') is None else float(q3['toan'])
            r['tai_lap_p3'] = j3.get('cong_tai_lap')
        rs, ft, n2 = r.get('rs') or {}, r.get('ft') or {}, r.get('nguon_p2') or {}
        r['p2_san_sang'] = bool(rs and rs['run'] == c['x'][1] and rs['ep'] == ep - 1 and {'bb', 'luong', 'dua_o', 'bo_o7'} <= set(rs['co'])
                                and not rs['cam'] and rs['ema'] == 'EMA' and ft and ft['run'] == c['x'][1] and ft['luot_79a'] == 'J_bert_dong'
                                and rs['hop'] is not None and ft['hop'] == rs['hop'])
        r['p1_san_sang'] = bool(r['p2_san_sang'] and not k['thieu']['x'] and not k['thieu']['z'] and n2.get('md5') == rs.get('md5')
                                and n2.get('ep') == ep and n2.get('tep') == f'rs_J_bert_dong_ep{ep}.pt' and list(n2.get('duong') or []) == list(c['x']))
        r['so_qd_khop'] = (None if c['decision_nb'] is None or r.get('so_qd_drive') is None else abs(round(r['so_qd_drive'], 4) - c['decision_nb']) < 5e-5)
        res[h] = r
    log(f"{'hang':<11}{'bien the':<38}{'K3':>4}{'du P2':>7}{'du P1':>7}{'ep rs_':>8}{'so QD Drive':>13}{'so QD nb':>10}{'khop':>6}  quy tac epoch (Phan 2)")
    for h, r in res.items():
        c = sources[h]
        e = f"ep{r['rs']['ep'] + 1}" if r.get('rs') and r['rs'].get('ep') is not None else 'thieu'
        qd = f"{r['so_qd_drive']:.4f}" if r.get('so_qd_drive') is not None else '—'
        qn = str(c['decision_nb']) if c['decision_nb'] is not None else '—'
        kh = str(r['so_qd_khop']) if r['so_qd_khop'] is not None else '—'
        qt = ((r.get('nguon_p2') or {}).get('quy_tac') or '—')[:70]
        log(f"{h:<11}{c['variant'][:37]:<38}{'co' if c['k3'] else '-':>4}{str(r['p2_san_sang']):>7}{str(r['p1_san_sang']):>7}{e:>8}{qd:>13}{qn:>10}{kh:>6}  {qt}")
    return res


def conclude(res, sources=config.K0_SOURCES, log=print):
    '''Kết luận cho 87c (+ sự thật về recipe s43: Tuấn quyết theo recipe / artifact, không theo điểm VinDr).'''
    k3 = [h for h, c in sources.items() if c['k3']]
    k3_p1 = [h for h in k3 if res[h]['p1_san_sang']]
    k3_p2 = [h for h in k3 if res[h]['p2_san_sang'] and not res[h]['p1_san_sang']]
    log(f"K3 (87c): {len(k3_p1)} / {len(k3)} seed du cho P1 + P2 {k3_p1}" + (f" | CHI du cho P2 (thieu nguon P1 — bao Claude truoc khi chay 87c): {k3_p2}" if k3_p2 else '')
        + (f" | THIEU encoder: {[h for h in k3 if not res[h]['p2_san_sang']]}" if any(not res[h]['p2_san_sang'] for h in k3) else ''))
    s43 = res.get('B_s43') or {}
    log(f"s43 (bien the B, KHONG thuoc K3): du P2 {s43.get('p2_san_sang')} / du P1 {s43.get('p1_san_sang')} | backbone ep "
        f"{((s43.get('rs') or {}).get('ep', -2) + 1) if s43.get('rs') else 'thieu'} | quy tac epoch {(s43.get('nguon_p2') or {}).get('quy_tac')} -> Tuan quyet")
    log(f"so quyet dinh Drive khop so in o notebook: {[h for h, r in res.items() if r['so_qd_khop']]} | KHAC: {[h for h, r in res.items() if r['so_qd_khop'] is False]}")
    return dict(k3_p1=k3_p1, k3_chi_p2=k3_p2, k3_thieu=[h for h in k3 if not res[h]['p2_san_sang']])
