# ==== nhanh_c buoc z_C3C — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C3c · TXTQ + DICH MOI + CAU (CHI luc train): khoi dung chung + bang muc anh / study tu 78e / 78f / 78q / 78s (78t) ====
# docs/PLAN_SAU_78G.md 4-6 (Tuan duyet 06/10). Khoi chep NGUYEN VAN tu experiments/78_chi_anh/khoi/ (test_khoi_*.py kiem).
#   txtq: query lop tu van ban - 78e (40 mo ta lop, = 78g), N(0, 1) (doi chung nghia), 78s (5 cach dien dat / cau report khang
#         dinh / 4 nguyen mau k-means); z-score QUA cac hang (nhu 78g).
#   vung: hop 36 vung (78f) -> toa do anh 518 + mat na vung goi y KHONG du phong cua nhan duong (78q) - nhu vung_kdp cua 78g.
#   dich: 7 nhom dich moi muc anh (78s dich_moi_train.npz; -1 = mask): T1 thay, T2 chac, T3 muc do, T4 ben, T5 ong / thiet bi,
#         T6 ky thuat chup, T7 ket cau.
#   cau : cau report TRAIN (78s) THEO MA cau - KHONG doc van ban: CSR study -> cau tren study POOL, vector BiomedBERT dong cung
#         (z-score) / token (thap hoc cung), trong so 1 / log(1 + tan so).
#   CHI anh / study cua benh nhan TRAIN, TRU 2% benh nhan GIU RIENG (chan doan), vao loss phu. Luc cham KHONG goi gi o day (cong C6:
#   go moi bang -> du doan trung tung bit). Cell nay CHI doc (qua API) va KHONG ghi gi len Drive.
assert 'MLDecoder' in globals() and 'TF_HOP' in globals() and 'N_TOK' in globals(), 'chay C2 / C3 truoc'
assert 'kiem_ke' in globals() and 'tai' in globals(), 'chay C0 truoc (khoi doc Drive)'

# ---- khoi_ehr_tuong_phan + khoi_vung + khoi_loss + khoi_hop_anh + khoi_cau + khoi_txtq + khoi_dich (NGUYEN VAN) ----
def bo_ngau_nhien(x, p, gen):
    '''Dropout voi mat na rut tu `gen` (Generator CUNG thiet bi voi x) — KHONG cham dong ngau nhien chung.'''
    if p <= 0: return x
    giu = torch.rand(x.shape, generator=gen, device=x.device) >= p
    return x * giu.to(x.dtype) / (1.0 - p)


class ThapEHR(nn.Module):
    '''EHR muc study -> z [N, d_chieu]. Vao: x [N, d_in] (da chuan hoa nhu C1) + tui vi sinh (id, so lan) neu co V_mic.
       (Linear -> LayerNorm -> GELU -> Dropout) x 2 lop 512, roi Linear -> 256; phep chieu Linear 256 -> 128.
       Dropout rut mat na tu `gen` — BAT BUOC co `gen` khi .train() (khong bao gio roi ve dong chung).'''
    def __init__(s, d_in, V_mic=None, d_tui=64, d_an=512, d_ra=256, d_chieu=128, p_bo=0.1):
        super().__init__()
        s.p_bo = p_bo
        s.tui = nn.Parameter(torch.randn(V_mic, d_tui) * d_tui ** -0.5) if V_mic else None
        s.ln_tui = nn.LayerNorm(d_tui) if V_mic else None
        s.l1 = nn.Linear(d_in + (d_tui if V_mic else 0), d_an); s.n1 = nn.LayerNorm(d_an)
        s.l2 = nn.Linear(d_an, d_an); s.n2 = nn.LayerNorm(d_an)
        s.l3 = nn.Linear(d_an, d_ra); s.chieu = nn.Linear(d_ra, d_chieu)

    def forward(s, x, mids=None, mcnt=None, gen=None):
        assert gen is not None or not s.training or s.p_bo <= 0, 'ThapEHR.train() phai co generator RIENG (gen)'
        if s.tui is not None:
            x = torch.cat([x, s.ln_tui(F.embedding_bag(mids, s.tui, per_sample_weights=mcnt, mode='sum'))], 1)
        h = F.gelu(s.n1(s.l1(x)))
        if s.training: h = bo_ngau_nhien(h, s.p_bo, gen)
        h = F.gelu(s.n2(s.l2(h)))
        if s.training: h = bo_ngau_nhien(h, s.p_bo, gen)
        return s.chieu(s.l3(h))


class ChieuAnh(nn.Module):
    '''Dac trung anh [B, d_vao] -> (z [B, d_chieu], du doan hoi quy [B, d_hq] hoac None). Linear -> LayerNorm -> GELU
       (lop an 512) -> Linear; dau hoi quy doc lop an. KHONG dropout -> khong rut so ngau nhien nao.'''
    def __init__(s, d_vao, d_an=512, d_chieu=128, d_hq=0):
        super().__init__()
        s.l1 = nn.Linear(d_vao, d_an); s.n1 = nn.LayerNorm(d_an); s.l2 = nn.Linear(d_an, d_chieu)
        s.hq = nn.Linear(d_an, d_hq) if d_hq else None

    def forward(s, f):
        a = F.gelu(s.n1(s.l1(f)))
        return s.l2(a), (s.hq(a) if s.hq is not None else None)


class PhuEHR(nn.Module):
    '''Module loss phu cua MOT arm: chieu anh (+ dau hoi quy neu d_hq), thap EHR + nhiet do hoc duoc neu d_ehr.
       Nhiet do tau = 0.01 + 0.99 * sigmoid(tho): luon trong (0.01, 1), khong ket o bien; khoi tao tau0.'''
    def __init__(s, d_anh, d_ehr=0, V_mic=None, d_hq=0, tau0=0.1):
        super().__init__()
        s.anh = ChieuAnh(d_anh, d_hq=d_hq)
        s.ehr = ThapEHR(d_ehr, V_mic) if d_ehr else None
        s.tho = nn.Parameter(torch.tensor(math.log((tau0 - 0.01) / (1.0 - tau0)))) if d_ehr else None

    def tau(s):
        return 0.01 + 0.99 * torch.sigmoid(s.tho)


def scarf(x, pool, ti_le, gen):
    '''Tang cuong kieu SCARF: moi o (hang, cot) voi xac suat `ti_le` thay bang gia tri CUNG COT cua mot hang NGAU NHIEN
       cua `pool` (phan phoi bien thuc nghiem cua cot do). x [N, D], pool [M, D], cung thiet bi voi `gen`.'''
    if ti_le <= 0: return x
    N, D = x.shape
    m = torch.rand((N, D), generator=gen, device=x.device) < ti_le
    r = torch.randint(pool.shape[0], (N, D), generator=gen, device=x.device)
    return torch.where(m, pool[r, torch.arange(D, device=x.device).expand(N, D)], x)


def lay_am(n_st, K, sub_st, sub_cam, gen):
    '''K chi so study NGAU NHIEN (co hoan lai) trong [0, n_st), BO study cua benh nhan trong `sub_cam` (cua batch).'''
    i = torch.randint(n_st, (K,), generator=gen, device=sub_st.device)
    return i[~torch.isin(sub_st[i], sub_cam)]


@torch.no_grad()
def muc_tieu(xr_neo, xr_ung, duong, loai, alpha, tau_s):
    '''Muc tieu mem moi hang = (1 - alpha) * [duong] / so duong + alpha * softmax(cos(EHR goc neo, EHR goc ung vien) /
       tau_s); o `loai` = 0 (khong duong, khong am). xr_* = EHR GOC (khong tang cuong). Moi hang PHAI co >= 1 duong.'''
    cung = duong.float(); cung = cung / cung.sum(1, keepdim=True)
    if alpha <= 0: return cung
    s = (F.normalize(xr_neo.float(), dim=1) @ F.normalize(xr_ung.float(), dim=1).T) / tau_s
    return (1 - alpha) * cung + alpha * torch.softmax(s.masked_fill(loai, float('-inf')), 1)


def ce_mem(logit, muc, loai):
    '''Cross-entropy voi muc tieu mem, TB theo hang; o `loai` bi bo khoi mau so (logit -inf) va co muc tieu 0.'''
    lp = torch.log_softmax(logit.masked_fill(loai, float('-inf')), 1).masked_fill(loai, 0.0)
    return -(muc * lp).sum(1).mean()


def loss_tuong_phan(za, ze, xr, cs, sub_a, sub_e, n_b, alpha, tau_s, tau):
    '''InfoNCE DOI XUNG voi muc tieu mem (docs/PLAN_EHR_TUONG_PHAN.md §4.3); 0.5 moi chieu.
       za [Ba, k] anh; ze [Ne, k] ung vien EHR = n_b study CUA batch (dung dau, khong trung) roi mau am; xr [Ne, D] EHR
       GOC cua ung vien; cs [Ba] study cua tung anh trong [0, n_b); sub_a [Ba], sub_e [Ne] benh nhan.
       Anh -> EHR: ung vien = ca Ne. EHR -> anh: neo = n_b study cua batch, ung vien = Ba anh (nhieu anh cung study ->
       phan duong chia deu). Ung vien CUNG benh nhan KHAC study bi loai khoi mau so. Tra ve (loss, ti le anh chon dung
       study cua no trong batch + mau am).'''
    za = F.normalize(za.float(), dim=1); ze = F.normalize(ze.float(), dim=1)
    j = torch.arange(ze.shape[0], device=za.device)
    duong = cs[:, None] == j[None, :]
    loai = (sub_a[:, None] == sub_e[None, :]) & ~duong
    s_ae = za @ ze.T
    l_ae = ce_mem(s_ae / tau, muc_tieu(xr[cs], xr, duong, loai, alpha, tau_s), loai)
    duong2 = duong[:, :n_b].T
    loai2 = (sub_e[:n_b, None] == sub_a[None, :]) & ~duong2
    l_ea = ce_mem(ze[:n_b] @ za.T / tau, muc_tieu(xr[:n_b], xr[cs], duong2, loai2, alpha, tau_s), loai2)
    with torch.no_grad():
        dung = float((s_ae.masked_fill(loai, float('-inf')).argmax(1) == cs).float().mean())
    return 0.5 * l_ae + 0.5 * l_ea, dung


def loss_hoi_quy(pred, y, co, nhi_phan):
    '''Hoi quy EHR tu anh (doi chung kieu PaCX): MSE o cot lien tuc, BCE (logit) o cot nhi phan, TB tren moi o CO do.
       pred, y [N, D]; co [N, D] (o co gia tri o EHR goc); nhi_phan [D]. O / cot thieu KHONG tinh.'''
    pred = pred.float(); y = y.float(); m = co.float()
    l = torch.where(nhi_phan[None, :], F.binary_cross_entropy_with_logits(pred, y.clamp(0, 1), reduction='none'),
                    (pred - y) ** 2)
    return (l * m).sum() / m.sum().clamp_min(1.0)


def phu_batch(cfg, mod, f_anh, st, T, gen, K=4096, ti_le_scarf=0.3, p_tui=0.2):
    '''Loss phu cua MOT arm tren MOT micro-batch. Tra ve None neu khong anh nao duoc vao loss phu, nguoc lai dict:
       n_anh, va `tp` (+ `dung`) neu cfg['tp'], `hq` neu cfg['hq'].
       cfg: dict(tp, hq, alpha, tau_s, bien, xao); mod: PhuEHR cua arm; f_anh [B, d] dac trung anh (CO gradient);
       st [B] chi so study (= hang2st[hang], -1 = khong vao loss phu); T: bang muc study tren thiet bi (C3c):
       n_st, sub_st, X[bien], co_tui[bien], mids, mcnt, pi_xao, Y_hq, co_hq, nhi_phan.
       Arm `xao`: ung vien (ke ca study cua chinh anh) mang NOI DUNG EHR cua study pi_xao[.] — chi cap anh <-> EHR doi;
       luat loai cung benh nhan va lay mau am van theo DANH TINH study, nhu arm khong xao.'''
    v = st >= 0
    if not bool(v.any()): return None
    stv = st[v]
    z, hq = mod.anh(f_anh[v].float())
    out = dict(n_anh=int(v.sum()))
    if cfg.get('hq'):
        out['hq'] = loss_hoi_quy(hq, T['Y_hq'][stv], T['co_hq'][stv], T['nhi_phan'])
    if cfg.get('tp'):
        us, cs = torch.unique(stv, return_inverse=True)
        ung = torch.cat([us, lay_am(T['n_st'], K, T['sub_st'], T['sub_st'][us], gen)])
        nd = T['pi_xao'][ung] if cfg.get('xao') else ung
        X = T['X'][cfg['bien']]
        xr = X[nd]
        xin = scarf(xr, X, ti_le_scarf, gen)
        if T['co_tui'][cfg['bien']]:
            giu = (torch.rand(len(nd), generator=gen, device=xr.device) >= p_tui).to(T['mcnt'].dtype)
            ze = mod.ehr(xin, T['mids'][nd], T['mcnt'][nd] * giu[:, None], gen=gen)
        else:
            ze = mod.ehr(xin, gen=gen)
        out['tp'], out['dung'] = loss_tuong_phan(z, ze, xr, cs, T['sub_st'][stv], T['sub_st'][ung], len(us),
                                                 cfg['alpha'], cfg['tau_s'], mod.tau())
    return out


def bang_study(hang_vao, study_hang, sub_hang, n_hang):
    '''Bang muc study cho loss phu. hang_vao: hang (ORDER) DUOC vao loss phu. Tra ve (hang_dd [n_st] = hang DAU cua
       moi study, sub_st [n_st], hang2st [n_hang] = chi so study, -1 neu hang khong vao). Study xep theo study_id.'''
    hang_vao = np.sort(np.asarray(hang_vao, 'int64'))
    _, dau, inv = np.unique(np.asarray(study_hang)[hang_vao], return_index=True, return_inverse=True)
    hang_dd = hang_vao[dau]; sub_st = np.asarray(sub_hang)[hang_dd]
    assert (np.asarray(sub_hang)[hang_vao] == sub_st[inv]).all(), 'mot study thuoc NHIEU benh nhan'
    hang2st = np.full(n_hang, -1, 'int64'); hang2st[hang_vao] = inv
    return hang_dd, sub_st, hang2st


def chon_giu_rieng(sub_train, ti_le, hat):
    '''Nhom benh nhan train GIU RIENG (hat co dinh): vao loss nhan nhu thuong, KHONG vao loss phu — de chan doan.'''
    u = np.unique(np.asarray(sub_train))
    return np.sort(np.random.default_rng(hat).choice(u, size=max(1, int(round(ti_le * len(u)))), replace=False))


def hoan_vi_xao(sub_st, hat):
    '''Hoan vi pi tren n study cho doi chung xao cap: pi[k] != k va sub[pi[k]] != sub[k] voi MOI k. Xep study theo benh
       nhan (thu tu benh nhan ngau nhien, hat co dinh) roi dich vong n // 2 vi tri: study cua moi benh nhan la mot doan
       lien tiep ngan hon n // 2 nen luon roi sang benh nhan khac.'''
    sub_st = np.asarray(sub_st); n = len(sub_st)
    u = np.unique(sub_st); hang = np.random.default_rng(hat).permutation(len(u))[np.searchsorted(u, sub_st)]
    tt = np.lexsort((np.arange(n), hang))
    pi = np.empty(n, 'int64'); pi[tt] = tt[(np.arange(n) + n // 2) % n]
    assert (pi != np.arange(n)).all() and (sub_st[pi] != sub_st).all(), 'hoan vi xao giu study / benh nhan cu'
    assert np.array_equal(np.sort(pi), np.arange(n)), 'pi khong phai hoan vi'
    return pi


def he_so_tang(buoc, tong, ti_le=0.1):
    '''h(t): tang tuyen tinh 0 -> 1 trong `ti_le` dau cua tong so buoc toi uu (buoc 0 -> 0).'''
    return float(min(1.0, max(0.0, buoc / max(1.0, ti_le * tong))))


def chuan_grad(loss, params):
    '''||d loss / d params|| (chuan L2 gop), KHONG cong vao .grad (autograd.grad, giu do thi cho backward sau).'''
    g = [x for x in torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True) if x is not None]
    return float(torch.sqrt(sum((x.float() ** 2).sum() for x in g))) if g else 0.0


@torch.no_grad()
def truy_hoi(za, ze, cs, sub_a, sub_e, n_ung=1000, ks=(1, 5, 10), hat=0):
    '''Chan doan (nhom GIU RIENG): anh -> EHR. Moi anh xep hang study cua no giua n_ung ung vien = no + (n_ung - 1)
       study NGAU NHIEN cua benh nhan KHAC. Tra ve {R@k}: ngau nhien = k / n_ung.'''
    za = F.normalize(za.float(), dim=1); ze = F.normalize(ze.float(), dim=1)
    S = za @ ze.T
    R = torch.rand(S.shape, generator=torch.Generator().manual_seed(hat)).to(S.device)
    R = R.masked_fill(sub_a[:, None] == sub_e[None, :], 2.0)              # cung benh nhan (ke ca chinh no): khong lam am
    k_am = int(min(n_ung - 1, int((R < 2).sum(1).min())))
    chon = R.topk(k_am, dim=1, largest=False).indices
    hang = (S.gather(1, chon) > S.gather(1, cs[:, None])).sum(1)
    return dict({f'R@{k}': float((hang < k).float().mean()) for k in ks}, n_am=k_am)


def probe_r2(X, y, nhom, lam=0.1, k=5, hat=0):
    '''Chan doan: R^2 NGOAI MAU cua hoi quy ridge y ~ X, k fold THEO NHOM (benh nhan); X chuan hoa theo fold train.'''
    X = np.asarray(X, 'float64'); y = np.asarray(y, 'float64'); nhom = np.asarray(nhom)
    u = np.unique(nhom); fold = (np.random.default_rng(hat).permutation(len(u)) % k)[np.searchsorted(u, nhom)]
    du = np.full(len(y), np.nan)
    for i in range(k):
        tr, te = fold != i, fold == i
        if not te.any() or tr.sum() < 2: continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A = (X[tr] - mu) / sd; my = y[tr].mean()
        w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (y[tr] - my))
        du[te] = ((X[te] - mu) / sd) @ w + my
    ok = np.isfinite(du)
    return float(1 - ((y[ok] - du[ok]) ** 2).sum() / max(((y[ok] - y[ok].mean()) ** 2).sum(), 1e-12))

import re
import numpy as np


def doc_ds_vung(txt):
    '''semantics/objects_detectable_by_bbox_pipeline_v1.txt -> danh sach ten vung (giu thu tu tep; moi dong "ten,").'''
    return [l.strip().rstrip(',').strip() for l in txt.splitlines() if l.strip()]


def doc_ds_thuoc_tinh(txt):
    '''semantics/attribute_relations_v1.txt -> danh sach khoa "loai|gia tri" (moi dong "loai|yes|gia tri,").'''
    out = []
    for l in txt.splitlines():
        l = l.strip().rstrip(',')
        if not l: continue
        loai, yn, gt = l.split('|', 2)
        assert yn == 'yes', l
        out.append(f'{loai}|{gt}')
    return out


def dicom_tu_entry(ten):
    '''"scene_graph/<dicom>_SceneGraph.json" -> dicom; ten khac (thu muc, tep khac) -> None.'''
    m = re.fullmatch(r'(?:.*/)?([0-9a-f]{8}(?:-[0-9a-f]{8}){4})_SceneGraph\.json', ten)
    return m.group(1) if m else None


def chon_entry_train(ten_entries, dicom_train):
    '''Chon entry THEO TEN, TRUOC khi mo: chi entry ma dicom thuoc tap train (val / test / ngoai CXR-LT bi bo).'''
    s = set(dicom_train)
    return [n for n in ten_entries if dicom_tu_entry(n) in s]


def doc_scene_graph(j, vung_idx, thay_the=None):
    '''Mot scene graph (dict) -> (hop, co, cau):
       hop: [(vung, ox1, oy1, ox2, oy2, x1, y1, x2, y2)] toa do ANH GOC (original_*) + toa do 224 cua ImaGenome (de kiem
            kich thuoc anh), chi vung co trong `vung_idx`;
       co : [(vung, khoa, thay)] thuoc tinh 'yes' ("loai|gia tri") gan vung (gom nlp|normal / nlp|abnormal), khong lap;
            thay = 1 neu den tu vung CHI co trong report (khong hop) qua `thay_the` {ten vung report: [chi so vung co hop]};
       cau: [(vung, phrase_id, section, text)] cau report CUA CHINH study gan vung CO HOP, khong lap theo (vung, phrase_id).
       Vung ngoai `vung_idx` va ngoai `thay_the` bi bo; cau cua vung thay the KHONG gan vao vung co hop.'''
    hop, co, cau = [], set(), {}
    thay_the = thay_the or {}
    for o in j.get('objects', []):
        v = vung_idx.get(o.get('bbox_name'))
        if v is None: continue
        hop.append((v, float(o['original_x1']), float(o['original_y1']), float(o['original_x2']), float(o['original_y2']),
                    float(o['x1']), float(o['y1']), float(o['x2']), float(o['y2'])))
    for a in j.get('attributes', []):
        ten = a.get('bbox_name')
        v = vung_idx.get(ten)
        dich = [v] if v is not None else list(thay_the.get(ten, []))
        if not dich: continue
        thay = 0 if v is not None else 1
        ph, pid, sec = a.get('phrases', []), a.get('phrase_IDs', []), a.get('sections', [])
        for k, ds in enumerate(a.get('attributes', [])):
            for s in ds:
                loai, yn, gt = s.split('|', 2)
                if yn == 'yes':
                    for d in dich: co.add((d, f'{loai}|{gt}', thay))
            if v is not None and k < len(ph) and k < len(pid):
                cau.setdefault((v, pid[k]), (v, pid[k], sec[k] if k < len(sec) else '', ph[k]))
    # cung (vung, khoa) co ca nguon that va thay the -> giu ban THAT (thay = 0)
    tot = {}
    for d, kk, t in co: tot[(d, kk)] = min(t, tot.get((d, kk), 1))
    return hop, sorted((d, kk, t) for (d, kk), t in tot.items()), list(cau.values())


def mat_na(vung_ds):
    '''Danh sach chi so vung -> mat na bit int64 (vung k -> bit k; can < 63 vung).'''
    m = 0
    for v in vung_ds: m |= 1 << int(v)
    return m


def vung_cua(m, R):
    '''Mat na bit -> danh sach chi so vung (tang dan).'''
    return [k for k in range(R) if (int(m) >> k) & 1]


def hop_sang_letterbox(x1, y1, x2, y2, W, H, SIZE):
    '''Hop toa do ANH GOC (W x H) -> toa do anh letterbox SIZE x SIZE, CUNG HINH HOC voi `_letterbox` cua 28c: scale theo
       canh DAI s = SIZE / max(W, H), nw = max(1, round(W s)), nh = max(1, round(H s)), dan CAN GIUA tai
       ((SIZE - nw) // 2, (SIZE - nh) // 2). Nhan mang numpy (vecto hoa). test_78f so voi chinh ham 28c tren anh that gia.'''
    W = np.asarray(W, 'float64'); H = np.asarray(H, 'float64')
    s = SIZE / np.maximum(W, H)
    nw = np.maximum(1, np.round(W * s)); nh = np.maximum(1, np.round(H * s))
    ox = (SIZE - nw) // 2; oy = (SIZE - nh) // 2
    fx, fy = nw / W, nh / H
    return (ox + np.asarray(x1) * fx, oy + np.asarray(y1) * fy, ox + np.asarray(x2) * fx, oy + np.asarray(y2) * fy)


def iou(a, b):
    '''IoU cua cac hop [n, 4] (x1, y1, x2, y2) tung cap.'''
    a = np.asarray(a, 'float64'); b = np.asarray(b, 'float64')
    iw = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    gi = iw * ih
    dt = lambda h: np.clip(h[:, 2] - h[:, 0], 0, None) * np.clip(h[:, 3] - h[:, 1], 0, None)
    return gi / np.maximum(dt(a) + dt(b) - gi, 1e-9)

import math
import torch, torch.nn as nn, torch.nn.functional as F


def bit_sang_bool(L, R):
    '''Mat na bit int64 [..., C] -> bool [..., C, R] (vung k <-> bit k).'''
    k = torch.arange(R, device=L.device)
    return ((L.unsqueeze(-1) >> k) & 1).bool()


def cung_nhan(Ya, Yc=None):
    '''[Ba, C] (va [Bc, C]) {0, 1} -> [Ba, Bc] bool: hai anh CUNG vector nhan (ca C nhan). Yc = None -> Ya voi chinh no.'''
    Ya = Ya.bool(); Yc = Ya if Yc is None else Yc.bool()
    return (Ya.unsqueeze(1) == Yc.unsqueeze(0)).all(-1)


def trong_so_rep3(Ya, La, R, beta=2.0, Yc=None, Lc=None):
    '''Phuong an 3. Ya [Ba, C] {0, 1}, La [Ba, C] int64 mat na vi tri DA BIET cua neo (0 = chua biet; bit o nhan am bi bo
       qua); Yc / Lc cua ung vien (None = chinh neo). -> W [Ba, Bc] >= 0: 0 neu khac vector nhan; trong nhom
       exp(beta (J - 0.5)) neu co it nhat mot nhan ma CA HAI biet vi tri, nguoc lai 1. Ung vien = neo -> W doi xung,
       W_ii = e^(beta / 2) neu neo biet vi tri cua it nhat mot nhan, nguoc lai 1.'''
    Yc = Ya if Yc is None else Yc; Lc = La if Lc is None else Lc
    Fa = (bit_sang_bool(La, R) & Ya.bool().unsqueeze(-1)).float()                # [Ba, C, R]
    Fc = (bit_sang_bool(Lc, R) & Yc.bool().unsqueeze(-1)).float()                # [Bc, C, R]
    na, nc = Fa.sum(-1), Fc.sum(-1)                                              # so vung moi (anh, nhan)
    giao = torch.einsum('ijr,kjr->ikj', Fa, Fc)                                  # [Ba, Bc, C]
    hop = na.unsqueeze(1) + nc.unsqueeze(0) - giao
    hai = (na > 0).float().unsqueeze(1) * (nc > 0).float().unsqueeze(0)          # ca hai biet vi tri cua nhan j
    so = hai.sum(-1)
    J = (hai * giao / hop.clamp_min(1)).sum(-1) / so.clamp_min(1)
    w = torch.where(so > 0, torch.exp(beta * (J - 0.5)), torch.ones_like(J))
    return w * cung_nhan(Ya, Yc).float()


def trong_so_rep1(Ya, Ea, tau_t=0.05, Yc=None, Ec=None):
    '''Phuong an 1 (doi chung). Ea [Ba, d] vector report goc (78e) cua neo, Ec cua ung vien (None = chinh neo).
       -> W = [cung vector nhan] * exp((cos(E_i, E_k) - 1) / tau_t).'''
    Yc = Ya if Yc is None else Yc; Ec = Ea if Ec is None else Ec
    a = F.normalize(Ea.float(), dim=-1); c = F.normalize(Ec.float(), dim=-1)
    return torch.exp((a @ c.T - 1.0) / tau_t) * cung_nhan(Ya, Yc).float()


def infonce_mem(za, zt, W, co, tau=0.07):
    '''InfoNCE DOI XUNG voi muc tieu mem, ung vien = CHINH lo. za / zt [B, k] (chua chuan hoa); W [B, B] >= 0; co [B] bool:
       anh co muc tieu (hang / cot ngoai `co` bi bo). Anh i -> text: muc tieu W_ik / sum_k W_ik; text k -> anh: W_ik / sum_i W_ik.
       -> loss trung binh tren cac hang hop le (0, van noi do thi, neu khong co hang).'''
    co = co.bool()
    if int(co.sum()) == 0: return (za.sum() + zt.sum()) * 0.0
    a = F.normalize(za[co].float(), dim=-1)
    t = F.normalize(zt[co].float(), dim=-1)
    S = a @ t.T / tau
    Wc = W[co][:, co].float()
    assert bool((Wc.sum(1) > 0).all()), 'hang muc tieu rong (W_ii phai > 0)'
    la = -((Wc / Wc.sum(1, keepdim=True)) * torch.log_softmax(S, 1)).sum(1).mean()
    lt = -((Wc / Wc.sum(0, keepdim=True)) * torch.log_softmax(S, 0)).sum(0).mean()
    return 0.5 * (la + lt)


def infonce_ung_vien(za, zt, W, loai, tau):
    '''InfoNCE DOI XUNG, muc tieu mem, ung vien NGOAI lo. za [B, k] anh (neo); zt [B + K, k] text ung vien — B dau = text cua
       CHINH cac neo (cung thu tu); W [B, B + K] >= 0 (W_ii > 0); loai [B, B + K] bool (cung benh nhan, khac study): bo khoi
       mau so va muc tieu. Anh -> text: B + K ung vien; text -> anh: B text dau, ung vien = B anh. Dung ce_mem (khoi_ehr_tuong_phan).
       -> (loss, nhom@1 = ti le neo co top-1 text voi W > 0).'''
    B = za.shape[0]
    a = F.normalize(za.float(), dim=-1); t = F.normalize(zt.float(), dim=-1)
    W = W.float().masked_fill(loai, 0.0)
    assert bool((W[:, :B].diagonal() > 0).all()), 'text cua chinh neo phai co trong so > 0'
    S = (a @ t.T) / tau
    l_at = ce_mem(S, W / W.sum(1, keepdim=True), loai)
    Wt, lo2 = W[:, :B].T, loai[:, :B].T
    l_ta = ce_mem(S[:, :B].T, Wt / Wt.sum(1, keepdim=True), lo2)
    with torch.no_grad():
        dung = float((W.gather(1, S.masked_fill(loai, float('-inf')).argmax(1, keepdim=True)) > 0).float().mean())
    return 0.5 * (l_at + l_ta), dung


class PhuRep(nn.Module):
    '''Module loss phu `rep` cua MOT arm: chieu anh (ChieuAnh cua khoi_ehr_tuong_phan: TB 40 dau ra query -> 128) + thap text
       (Linear -> LayerNorm -> GELU -> Linear -> 128) + nhiet do hoc duoc tau = 0.01 + 0.99 sigmoid(tho), khoi tao tau0 (nhu
       PhuEHR). Vector text vao DA chuan hoa z-score (78q C4: vi tri nam trong huong phuong sai nho). Khong dropout.'''
    def __init__(s, d_anh, d_text=768, d_an=512, d_chieu=128, tau0=0.1):
        super().__init__()
        s.anh = ChieuAnh(d_anh, d_an, d_chieu)
        s.text = nn.Sequential(nn.Linear(d_text, d_an), nn.LayerNorm(d_an), nn.GELU(), nn.Linear(d_an, d_chieu))
        s.tho = nn.Parameter(torch.tensor(math.log((tau0 - 0.01) / (1.0 - tau0))))

    def tau(s):
        return 0.01 + 0.99 * torch.sigmoid(s.tho)


def attn_truoc_dropout(mha, q, kv):
    '''Attention TRUNG BINH cac head cua nn.MultiheadAttention (batch_first, khong bias_k / add_zero_attn) TINH LAI tu in_proj,
       KHONG dropout. q [B, Lq, d], kv [B, Lk, d] -> [B, Lq, Lk], tong moi hang = 1. O che do eval = trong so `mha` tra ve.'''
    d, h = mha.embed_dim, mha.num_heads; dh = d // h
    W, b = mha.in_proj_weight, mha.in_proj_bias
    Q = F.linear(q, W[:d], None if b is None else b[:d])
    K = F.linear(kv, W[d:2 * d], None if b is None else b[d:2 * d])
    Bq, Lq, Lk = q.shape[0], q.shape[1], kv.shape[1]
    Q = Q.view(Bq, Lq, h, dh).transpose(1, 2); K = K.view(Bq, Lk, h, dh).transpose(1, 2)
    return torch.softmax((Q @ K.transpose(-1, -2)) / math.sqrt(dh), -1).mean(1)


def o_luoi_tu_hop(hop, co, G, o):
    '''Hop [B, R, 4] (x1, y1, x2, y2; pixel cua anh vuong G * o) + co [B, R] -> [B, R, G * G] bool: o luoi (cua so o x o, thu tu
       hang-cot nhu token patch) co TAM nam trong hop, CONG o chua TAM hop (hop nho hon mot o van co it nhat mot o). Vung
       khong co (co = False; toa do co the NaN) -> toan False.'''
    co = co.bool(); hop = torch.nan_to_num(hop.float()) * co.unsqueeze(-1)
    c = (torch.arange(G, device=hop.device, dtype=torch.float32) + 0.5) * o
    cy, cx = torch.meshgrid(c, c, indexing='ij')
    cx, cy = cx.reshape(-1), cy.reshape(-1)                                       # [P]
    x1, y1, x2, y2 = (hop[..., k].unsqueeze(-1) for k in range(4))
    trong = (cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)                       # [B, R, P]
    tx = ((hop[..., 0] + hop[..., 2]) * 0.5 / o).floor().clamp(0, G - 1).long()
    ty = ((hop[..., 1] + hop[..., 3]) * 0.5 / o).floor().clamp(0, G - 1).long()
    tam = torch.zeros_like(trong).scatter_(-1, (ty * G + tx).unsqueeze(-1), True)
    return (trong | tam) & co.unsqueeze(-1)


def mat_na_nhan(bits, trong, R):
    '''bits [B, C] int64 (vung r <-> bit r) + trong [B, R, P] bool -> [B, C, P] bool: o thuoc it nhat mot vung cua nhan.'''
    return torch.einsum('bcr,brp->bcp', bit_sang_bool(bits, R).float(), trong.float()) > 0


def loss_vung(A, M, y, eps=1e-6):
    '''A [B, C, P] attention (tong hang 1); M [B, C, P] bool vung goi y; y [B, C] nhan. Cap (anh, nhan) hop le = nhan DUONG va
       M khong rong. -> (TB -log(khoi attention trong vung + eps), so cap, khoi TB trong vung) hoac (None, 0, None).'''
    khoi = (A * M.to(A.dtype)).sum(-1)                                           # [B, C]
    hl = (y > 0.5) & M.any(-1)
    n = int(hl.sum())
    if n == 0: return None, 0, None
    return -torch.log(khoi[hl] + eps).mean(), n, float(khoi[hl].detach().mean())

import numpy as np, torch


def bang_hop_518(anh, vung, x1, y1, x2, y2, W, H, n_anh, R, S0=1022, S=518):
    '''Hop 78f (anh, vung, toa do CHUAN HOA [0, 1] theo anh goc W[anh] x H[anh]) -> [n_anh, R, 4] float32 toa do pixel cua anh
       vuong S (letterbox S0 roi resize S). Vung khong co hop -> NaN.'''
    a = np.asarray(anh, 'int64'); v = np.asarray(vung, 'int64')
    Wa = np.asarray(W, 'float64')[a]; Ha = np.asarray(H, 'float64')[a]
    bx1, by1, bx2, by2 = hop_sang_letterbox(np.asarray(x1, 'float64') * Wa, np.asarray(y1, 'float64') * Ha,
                                            np.asarray(x2, 'float64') * Wa, np.asarray(y2, 'float64') * Ha, Wa, Ha, S0)
    out = np.full((n_anh, R, 4), np.nan, 'float32')
    out[a, v] = np.stack([bx1, by1, bx2, by2], 1) * (S / S0)
    return out


def bien_doi_kem_hop(img, hop, tf):
    '''Anh PIL + hop [R, 4] (pixel cua anh, NaN = khong co) -> (anh PIL da bien doi, hop [R, 4] float32 sau bien doi (NaN =
       khong co / suy bien < 1 px), co [R] bool). `tf` = torchvision.transforms.v2 nhan (anh, BoundingBoxes) va tra cung kieu.'''
    from torchvision import tv_tensors
    hop = np.asarray(hop, 'float32'); co = np.isfinite(hop).all(1)
    bb = tv_tensors.BoundingBoxes(torch.from_numpy(np.ascontiguousarray(hop[co])).reshape(-1, 4), format='XYXY',
                                  canvas_size=(img.size[1], img.size[0]))
    img2, bb2 = tf(img, bb)
    b = bb2.as_subclass(torch.Tensor).numpy().astype('float32')
    out = np.full_like(hop, np.nan); out[co] = b
    hl = co.copy(); hl[co] = (b[:, 2] - b[:, 0] >= 1.0) & (b[:, 3] - b[:, 1] >= 1.0)
    out[~hl] = np.nan
    return img2, out, hl

def bang_cau(st_dong, ma_dong, n_st):
    '''Cap (study, cau) -> CSR (ptr long [n_st + 1], idx long [nnz]): cau cua study s = idx[ptr[s]:ptr[s + 1]], TANG dan.
       st_dong = chi so study 0..n_st-1, ma_dong = ma_cau (0 <= ma < 2^31): mang numpy song song, moi cap MOT lan (trung ->
       DUNG). Chay mot lan (numpy, CPU, O(nnz log nnz); 3M cap ~0.5 s); ben goi chuyen .to(dev).'''
    st = np.asarray(st_dong); ma = np.asarray(ma_dong)
    assert st.ndim == 1 and st.shape == ma.shape, (st.shape, ma.shape)
    assert np.issubdtype(st.dtype, np.integer) and np.issubdtype(ma.dtype, np.integer), (st.dtype, ma.dtype)
    st = st.astype('int64'); ma = ma.astype('int64'); n_st = int(n_st)
    assert len(st) == 0 or (st.min() >= 0 and st.max() < n_st and ma.min() >= 0 and ma.max() < 2 ** 31), \
        'chi so study ngoai [0, n_st) / ma cau ngoai [0, 2^31) -> DUNG'
    k = np.sort(st * (1 << 31) + ma)                   # khoa ghep: xep theo study roi theo ma cau (1 lan sort, nhanh)
    assert not (k[1:] == k[:-1]).any(), 'cap (study, cau) TRUNG -> DUNG'
    ptr = np.zeros(n_st + 1, 'int64'); ptr[1:] = np.cumsum(np.bincount(k >> 31, minlength=n_st))
    return torch.from_numpy(ptr), torch.from_numpy(k & ((1 << 31) - 1))


def _cau_cua_st(us, ptr, idx):
    '''Cau cua cac study `us` (long [k], TANG dan; -1 = khong co study -> doan rong) -> (seg [T] = vi tri study trong us,
       dau [T] = vi tri dau doan cua phan tu, cau [T]): noi tiep theo thu tu us, trong moi study tang dan. Study >= 0 ma doan
       CSR RONG (khong co cau) -> DUNG: hang cua no phai co st = -1. O(k + T).'''
    if us.numel() == 0:
        z = torch.zeros(0, dtype=torch.long, device=us.device); return z, z, z
    assert int(us[-1]) < ptr.shape[0] - 1, 'chi so study >= n_st cua bang_cau -> DUNG'
    co = us >= 0; uc = us.clamp_min(0)
    d0 = ptr[uc]; n = torch.where(co, ptr[uc + 1] - d0, torch.zeros_like(d0))
    T, n_rong = torch.stack([n.sum(), (co & (n == 0)).sum()]).tolist()            # MOT lan dong bo cho ca hai so
    assert n_rong == 0, f'{n_rong} study (st >= 0) KHONG co cau -> hang cua no phai co st = -1 (hop_le False) -> DUNG'
    seg = torch.repeat_interleave(torch.arange(us.numel(), device=us.device), n, output_size=T)
    dau = (torch.cumsum(n, 0) - n)[seg]
    return seg, dau, idx[d0[seg] + torch.arange(T, device=us.device) - dau].long()


def lay_cau_lo(st, ptr, idx, toi_da, gen):
    '''Ung vien cau cua lo -> ma_u long [U] TANG dan, khong trung. Voi moi study KHAC NHAU s >= 0 trong `st` (xet theo s TANG
       dan): rut min(toi_da, n_s) cau KHAC NHAU cua s, deu, khong hoan lai (khoa ngau nhien tu `gen` + xep hang trong tung
       doan); hop lai. Rut theo STUDY (hai anh cung study: mot lan rut; thu tu hang khong doi ket qua). st = -1 -> khong gop
       gi; khong hang nao st >= 0 -> tensor rong; study st >= 0 KHONG co cau -> DUNG (phai st = -1). toi_da None -> lay het,
       khong rut so ngau nhien. Khong cham dong ngau nhien chung. O(T log T), T = tong so cau cua cac study trong lo.'''
    st = st.long(); vo = torch.zeros(0, dtype=torch.long, device=st.device)
    us = torch.unique(st); us = us[us >= 0]
    seg, dau, cau = _cau_cua_st(us, ptr, idx)
    T = cau.numel()
    if T == 0: return vo
    if toi_da is None: return torch.unique(cau)
    assert int(toi_da) >= 1 and gen is not None, 'lay_cau_lo: toi_da >= 1 va generator RIENG (gen)'
    khoa = seg * (1 << 31) + torch.randint(1 << 31, (T,), generator=gen, device=st.device)   # theo study, trong study ngau nhien
    o = torch.argsort(khoa, stable=True)          # seg da tang dan -> doan k van o [dau_k, dau_k + n_k) sau khi xep
    hang = torch.arange(T, device=st.device) - dau
    return torch.unique(cau[o[hang < int(toi_da)]])


def ma_tran_duong(st, ptr, idx, ma_u):
    '''P bool [B, U]: P[b, u] = ma_u[u] thuoc TAP CAU DAY DU cua study st[b] (st[b] = -1 -> ca hang False; study st >= 0 KHONG
       co cau -> DUNG, ca khi U = 0). ma_u bat ky thu tu (0 <= ma < 2^31). Khoa ghep (vi tri study trong lo, ma cau) tang ngat
       -> searchsorted. O(T + B U log T).'''
    st = st.long(); B, U = st.shape[0], ma_u.shape[0]
    P0 = torch.zeros(B, U, dtype=torch.bool, device=st.device)
    if B == 0 or not bool((st >= 0).any()): return P0
    us, inv = torch.unique(st, return_inverse=True)
    seg, _, cau = _cau_cua_st(us, ptr, idx)                              # >= 1 cau (study st >= 0 rong -> DUNG o day)
    if U == 0: return P0
    khoa = seg * (1 << 32) + cau
    assert bool((khoa[1:] > khoa[:-1]).all()), 'idx khong tang ngat trong study (dung bang_cau) -> DUNG'
    q = inv[:, None] * (1 << 32) + ma_u.long()[None, :]
    j = torch.searchsorted(khoa, q).clamp_max(khoa.numel() - 1)
    return (khoa[j] == q) & (st >= 0)[:, None]


def mat_na_loai(P, sub, st):
    '''loai bool [B, U] = ~P[b, u] VA co b' cung benh nhan (sub[b'] == sub[b]), KHAC study (st[b'] != st[b]), st[b'] >= 0,
       P[b', u]: cau cua study khac cung benh nhan ma khong thuoc study nay -> KHONG lam am. O(B^2 U) (mot matmul 0/1 float32).'''
    P = P.bool()
    A = (sub[:, None] == sub[None, :]) & (st[:, None] != st[None, :]) & (st[None, :] >= 0)
    with torch.autocast(device_type=P.device.type, enabled=False):
        E = (A.float() @ P.float()) > 0
    return E & ~P


def trong_so_cau(tan_so):
    '''w = 1 / log(1 + tan_so) float32 (cau_hinh dich_78s cau.trong_so): cau rat pho bien (vd cau "binh thuong") bot chi phoi
       gradient. tan_so = so study train chua cau, >= 1 (khong -> DUNG). Vao tensor -> giu thiet bi; numpy -> CPU.'''
    t = tan_so if torch.is_tensor(tan_so) else torch.as_tensor(np.asarray(tan_so))
    assert bool((t >= 1).all()), 'tan_so < 1 -> DUNG'
    return (1.0 / torch.log1p(t.double())).float()


def infonce_cau(z_anh, z_cau, P, loai, w, tau, hop_le=None):
    '''InfoNCE DOI XUNG, DA DUONG, muc cau; ung vien CHI trong lo. z_anh [B, k], z_cau [U, k] (chua chuan hoa); P, loai bool
       [B, U] (ma_tran_duong, mat_na_loai); w [U] (trong_so_cau[ma_u]); tau so / tensor 0-d (NhietDo); hop_le [B] bool (None =
       moi hang): hang False bi bo HAN (khong lam neo, khong lam ung vien; z_anh cua no dat 0 TRUOC khi chuan hoa -> NaN / inf
       o hang do khong lot vao loss, gradient 0). O loai KHONG BAO GIO la duong (duong = P & ~loai).
       S = normalize(z_anh) @ normalize(z_cau).T / tau, float32 (tat autocast).
       - anh -> cau: neo = hang hop le co >= 1 duong; muc tieu ti le P[b, u] w[u]; o loai bi bo (ce_mem); TB deu tren neo.
       - cau -> anh: neo = cau co >= 1 hang hop le duong; muc tieu deu tren cac hang do; ung vien = hang hop le tru loai.T;
         TB co trong so w[u] / TB(w tren cac neo).
       -> (loss = 0.5 (anh->cau + cau->anh), dung1 = tensor 0-d khong grad: ti le neo anh->cau co top-1 (ngoai o loai) la
       duong). Khong neo nao -> loss = +0.0 (van noi do thi, gradient 0), dung1 = nan: ben goi CHI cong dung1 khi huu han, dem
       rieng. Hang / cot KHONG lam neo duoc go mat na truoc log_softmax -> khong NaN. Khong dong bo thiet bi (khong .item()).
       O(B U k).'''
    B, U, dev = z_anh.shape[0], z_cau.shape[0], z_anh.device
    hl = torch.ones(B, dtype=torch.bool, device=dev) if hop_le is None else hop_le.bool()
    za = z_anh.float().masked_fill(~hl[:, None], 0.0)                    # hang KHONG hop le = 0 (NaN / inf o do khong lot)
    if B == 0 or U == 0:
        return (za.sum() + z_cau.float().sum()) * 0.0 + 0.0, torch.full((), float('nan'), device=dev)
    loai = loai.bool()
    Pd = P.bool() & ~loai & hl[:, None]                                  # duong HOP LE
    with torch.autocast(device_type=dev.type, enabled=False):
        S = (F.normalize(za, dim=1) @ F.normalize(z_cau.float(), dim=1).T) / tau
        w = w.float()
        neo1 = Pd.any(1); n1 = neo1.sum()
        Pw = Pd.float() * w[None, :]
        lo1 = loai & neo1[:, None]
        l1 = ce_mem(S, Pw / Pw.sum(1, keepdim=True).clamp_min(1e-12), lo1) * (B / n1.clamp_min(1))
        neo2 = Pd.any(0); a = w * neo2.float()
        P2 = Pd.T.float()
        lo2 = (loai.T | ~hl[None, :]) & neo2[:, None]
        l2 = ce_mem(S.T, P2 / P2.sum(1, keepdim=True).clamp_min(1.0) * a[:, None], lo2) * (U / a.sum().clamp_min(1e-12))
        loss = 0.5 * (l1 + l2) + 0.0                                     # + 0.0: -0.0 -> 0.0 khi khong co neo
    with torch.no_grad():
        top = S.masked_fill(lo1, float('-inf')).argmax(1, keepdim=True)
        dung1 = (Pd.gather(1, top)[:, 0] & neo1).sum().float() / n1.float()
    return loss, dung1


class ThapVector(nn.Module):
    '''Thap cho vector cau BiomedBERT DONG CUNG (78t cau_bert, 78u B1): x [U, d_vao] float32 DA z-score boi ben goi
       (cau_emb_thong_ke) -> Linear -> LayerNorm -> GELU -> Linear -> [U, d_ra] (s.d_ra). Khong dropout: forward khong rut so
       ngau nhien. Khoi tao (nn.Linear) TIEU dong ngau nhien chung -> ben goi tao trong torch.random.fork_rng voi hat rieng.'''
    def __init__(s, d_vao=768, d_an=512, d_ra=256):
        super().__init__()
        s.d_ra = int(d_ra)
        s.mlp = nn.Sequential(nn.Linear(d_vao, d_an), nn.LayerNorm(d_an), nn.GELU(), nn.Linear(d_an, d_ra))

    def forward(s, x):
        return s.mlp(x)


class ThapBert(nn.Module):
    '''Thap van ban HOC CUNG (CLIP; 78t cau_clip, 78u B2): tok [U, L] (cau_token.npy: [CLS] ... [SEP], pad 0 ben phai) -> mask =
       tok != 0 -> mdl(input_ids=tok, attention_mask=mask.long()).last_hidden_state -> TB theo mask (float32) -> Linear ->
       LayerNorm -> GELU -> Linear -> [U, d_ra] (s.d_ra). Cot chi co pad bi cat truoc khi vao mdl (pad KHONG doi ket qua). mdl la
       module con (nhom LR rieng: s.mdl / s.dau; hoc mot phan: mo_bert_tren). mdl.training (dropout) -> BAT BUOC `gen`: forward
       cua mdl chay voi dong toan cuc (CPU / CUDA cua tok) TAM gieo tu gen trong fork_rng roi tra lai nguyen trang -> dong chung
       khong doi. eval: khong can gen; gen truyen vao KHONG bi tieu. Khoi tao s.dau (nn.Linear) TIEU dong ngau nhien chung ->
       ben goi tao trong torch.random.fork_rng voi hat rieng.
       chuan = (TB, DO LECH) [2, d] co dinh (78s cau_emb_thong_ke.npy) -> vector TB theo mask duoc z-score TRUOC s.dau (buffer,
       khong hoc): cung dau vao voi ThapVector tren vector dong cung DA z-score -> cau_clip - cau_bert (78t) va B2 - B1 (78u) CHI
       khac o tinh hoc cung (quyet dinh D-T2, review 07/10: vector BiomedBERT TB tho cos ~0.98 -> khong chuan hoa = them mot yeu
       to thu hai). s.dau LUON chay float32 (tat autocast) nhu ThapVector chay ngoai autocast.'''
    def __init__(s, mdl, d_an=512, d_ra=256, d_mdl=None, chuan=None):
        super().__init__()
        s.mdl = mdl; s.d_ra = int(d_ra)
        d = int(mdl.config.hidden_size if d_mdl is None else d_mdl)
        s.dau = nn.Sequential(nn.Linear(d, d_an), nn.LayerNorm(d_an), nn.GELU(), nn.Linear(d_an, d_ra))
        if chuan is None:
            s.register_buffer('chuan_tb', None); s.register_buffer('chuan_sd', None)
        else:
            tb = torch.as_tensor(np.asarray(chuan[0], dtype='float32')).clone()
            sd = torch.as_tensor(np.asarray(chuan[1], dtype='float32')).clone()
            assert tuple(tb.shape) == (d,) and tuple(sd.shape) == (d,), f'ThapBert: chuan phai la [2, {d}] -> DUNG'
            assert bool(torch.isfinite(tb).all()) and bool((sd > 0).all()), 'ThapBert: chuan khong huu han / do lech <= 0 -> DUNG'
            s.register_buffer('chuan_tb', tb); s.register_buffer('chuan_sd', sd)

    def _dau32(s, x):
        '''z-score co dinh (neu co chuan) roi s.dau, LUON float32 (tat autocast cua thiet bi).'''
        x = x.float()
        if s.chuan_tb is not None: x = (x - s.chuan_tb) / s.chuan_sd
        with torch.autocast(device_type=x.device.type, enabled=False):
            return s.dau(x)

    def forward(s, tok, gen=None):
        tok = tok.long(); dv = tok.device
        if tok.shape[0] == 0 or tok.shape[1] == 0:
            return s._dau32(torch.zeros(tok.shape[0], s.dau[0].in_features, device=dv))
        mask = tok != 0
        L = max(1, int((torch.arange(1, tok.shape[1] + 1, device=dv) * mask.any(0)).max()))
        tok, mask = tok[:, :L], mask[:, :L]
        if s.mdl.training:
            assert gen is not None, 'ThapBert: mdl.train() (dropout) phai co generator RIENG (gen)'
            assert dv.type in ('cpu', 'cuda'), f'ThapBert: thiet bi {dv} chua ho tro fork_rng'
            hat = int(torch.randint(1 << 62, (1,), generator=gen, device=gen.device))
            with torch.random.fork_rng(devices=[dv] if dv.type == 'cuda' else [], device_type=dv.type):
                if dv.type == 'cuda':
                    with torch.cuda.device(dv): torch.cuda.manual_seed(hat)
                else:
                    torch.default_generator.manual_seed(hat)
                h = s.mdl(input_ids=tok, attention_mask=mask.long()).last_hidden_state
        else:
            h = s.mdl(input_ids=tok, attention_mask=mask.long()).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return s._dau32((h.float() * m).sum(1) / m.sum(1).clamp_min(1.0))


def mo_bert_tren(mdl, n_mo):
    '''Hoc cung MOT PHAN bo ma hoa BERT (78t cau_clip, 78u B2; kieu ConVIRT): dong cung MOI tham so cua mdl (embeddings, cac
       tang duoi, pooler neu co) roi MO n_mo tang encoder TREN CUNG (mdl.encoder.layer[so tang - n_mo:]; 0 <= n_mo <= so tang;
       BiomedBERT 12 tang, THAP_MO 6). Dau chieu ThapBert.dau khong thuoc mdl -> van hoc. Chi doi requires_grad (dropout moi
       tang van chay khi .train(), rut tu gen trong ThapBert); goi TRUOC khi tao optimizer; goi lai -> dong lai tu dau. Khong
       rut so ngau nhien. Tra ve mdl.'''
    tang = mdl.encoder.layer; n = len(tang); n_mo = int(n_mo)
    assert 0 <= n_mo <= n, f'mo_bert_tren: n_mo {n_mo} ngoai [0, {n}] -> DUNG'
    mdl.requires_grad_(False); tang[n - n_mo:].requires_grad_(True)
    return mdl


class NhietDo(nn.Module):
    '''Nhiet do hoc duoc tau = 0.01 + 0.99 * sigmoid(tho) (nhu PhuEHR): luon trong (0.01, 1), khoi tao tau0. nd() = nd.tau().
       Khoi tao khong rut so ngau nhien.'''
    def __init__(s, tau0=0.07):
        super().__init__()
        assert 0.01 < tau0 < 1.0, tau0
        s.tho = nn.Parameter(torch.tensor(math.log((tau0 - 0.01) / (1.0 - tau0))))

    def tau(s):
        return 0.01 + 0.99 * torch.sigmoid(s.tho)

    def forward(s):
        return s.tau()


class PhuCau(nn.Module):
    '''Module loss phu `cau` cua MOT arm / luot (78t cau_bert / cau_clip, 78u B1 / B2): .anh = ChieuAnh(d_anh -> d_an ->
       d_chieu) (khoi_ehr_tuong_phan; khong dropout, khong dau hoi quy); .text = thap truyen vao (ThapVector / ThapBert, phai co
       thap.d_ra == d_chieu); .nhiet = NhietDo(tau0). z_anh(f) [B, d_anh] -> [B, d_chieu] (vao .float()); tau() = .nhiet.tau().
       Thap do ben goi tao TRUOC; trong PhuCau chi .anh tieu so ngau nhien (khoi tao) -> tao PhuCau trong torch.random.fork_rng
       voi hat rieng (cung hat -> .anh trung voi ChieuAnh(d_anh, d_an, d_chieu) tao rieng).'''
    def __init__(s, d_anh, thap, d_an=512, d_chieu=256, tau0=0.07):
        super().__init__()
        assert getattr(thap, 'd_ra', None) == d_chieu, f'PhuCau: thap.d_ra {getattr(thap, "d_ra", None)} != d_chieu {d_chieu}'
        s.anh = ChieuAnh(d_anh, d_an=d_an, d_chieu=d_chieu); s.text = thap; s.nhiet = NhietDo(tau0)

    def z_anh(s, f):
        return s.anh(f.float())[0]

    def tau(s):
        return s.nhiet.tau()

def z_lop(E):
    '''E [n, d] (numpy / mang) -> torch float32 [n, d] (CPU): z-score tung chieu QUA n hang, ddof 0, + 1e-6 - DUNG bieu thuc
       E_MOTA_T cua 78g C3c ((E - E.mean(0)) / (E.std(0) + 1e-6), E float32) -> trung tung bit.'''
    E = np.asarray(E).astype('float32')
    assert E.ndim == 2 and E.shape[0] >= 2 and np.isfinite(E).all(), ('z_lop: E phai [n >= 2, d] huu han', E.shape)
    return torch.from_numpy((E - E.mean(0)) / (E.std(0) + 1e-6)).float()


def e_ngau_nhien(C, d, hat):
    '''txtq_nn: vector lop NGAU NHIEN co dinh N(0, 1) [C, d] float32 tu np.random.default_rng(hat) (KHONG cham dong torch) -
       doi chung "nghia van ban": cung hinh, cung z-score, cung Linear / LayerNorm voi txtq.'''
    return np.random.default_rng(int(hat)).standard_normal((int(C), int(d))).astype('float32')


def e_nguyen_mau(P):
    '''78s nguyen_mau_lop.npy [C, K, d] -> [C*K, d], hang j*K + k = nguyen mau k cua lop j (thu tu GroupFCK).'''
    P = np.asarray(P)
    assert P.ndim == 3, P.shape
    return P.reshape(P.shape[0] * P.shape[1], P.shape[2])


class QueryTuTextDu(nn.Module):
    '''txtq_du: query = LN(Linear(E)) + Delta, Delta [n, d] HOC DUOC khoi tao 0. Nhan module QueryTuText DA TAO `goc` (78g) va
       dung CHINH E (buffer) / lin / ln cua no: khong tao tham so moi ngoai Delta, khong rut so ngau nhien -> luc dau trung goc
       TUNG BIT. state_dict = cua goc (E, lin.*, ln.*) + delta.'''
    def __init__(s, goc):
        super().__init__()
        s.register_buffer('E', goc.E); s.lin = goc.lin; s.ln = goc.ln
        s.delta = nn.Parameter(torch.zeros(goc.E.shape[0], goc.lin.out_features, dtype=goc.lin.weight.dtype,
                                           device=goc.lin.weight.device))

    def co_so(s):
        return s.ln(s.lin(s.E))

    def forward(s):
        return s.co_so() + s.delta

    @torch.no_grad()
    def ti_le_delta(s):
        '''||Delta|| / ||LN(Linear(E))|| (Frobenius) - chan doan: query troi khoi van ban bao nhieu.'''
        return float(s.delta.norm() / s.co_so().norm().clamp_min(1e-12))


class GroupFCK(nn.Module):
    '''Dau ra K nguyen mau / lop (txtq_k4): logit_jk = <h[:, j*K + k], w_jk> + b_jk -> logit_j = logsumexp_k logit_jk - log K.
       Khoi tao TU GroupFC `goc` (C nhom: w [C, d], b [C]) bang LAP K lan moi lop (hang j*K + k = (w_j, b_j)) - khong rut so ngau
       nhien. forward(h [B, C*K, d]) -> [B, C]; logit_k(h) -> [B, C, K].'''
    def __init__(s, goc, K):
        super().__init__()
        s.K, s.C = int(K), int(goc.w.shape[0])
        assert s.K >= 1, K
        s.w = nn.Parameter(goc.w.detach().repeat_interleave(s.K, 0).clone())
        s.b = nn.Parameter(goc.b.detach().repeat_interleave(s.K, 0).clone())

    def logit_k(s, h):
        return ((h * s.w).sum(-1) + s.b).unflatten(-1, (s.C, s.K))

    def forward(s, h):
        return torch.logsumexp(s.logit_k(h), -1) - math.log(s.K)


class MLDecoderTxtQK(MLDecoder):
    '''txtq_k4 = MLDecoderTxtQ (78g) voi C*K hang query (E_mo_ta [C*K, d_text]: K nguyen mau / lop, hang j*K + k) va dau ra
       GroupFCK. Phan chung tao TRUOC (CUNG dong ngau nhien -> CUNG khoi tao voi mldec, ke ca GroupFC goc C nhom), roi qtxt =
       QueryTuText(E_mo_ta, d) trong fork_rng voi hat rieng (= hat cua txtq -> CUNG Linear / LayerNorm), roi out = GroupFCK(out
       goc, K) (= GroupFC cua mldec lap K lan). forward / mldec_h / q_cua nhu MLDecoderTxtQ: h [B, C*K, d], logit [B, C].'''
    def __init__(s, *a, E_mo_ta=None, hat=0, K=4, **k):
        super().__init__(*a, **k)
        assert E_mo_ta is not None and E_mo_ta.shape[0] == s.out.w.shape[0] * int(K), (
            'MLDecoderTxtQK: E_mo_ta phai co C*K hang', None if E_mo_ta is None else tuple(E_mo_ta.shape), int(K))
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            torch.manual_seed(hat); s.qtxt = QueryTuText(E_mo_ta, s.q.weight.shape[1])
        s.out = GroupFCK(s.out, K)

    def forward(s, patch, etok=None, need_w=False):
        q = s.qtxt().unsqueeze(0).expand(patch.shape[0], -1, -1)
        kv = s.pin(patch)
        if s.t1 is not None: kv = s.t1(kv, s.ein(etok))
        h, w = s.img(q, kv, need_w); return s.out(h), w


def gop_query_k(A, K):
    '''Gia tri theo hang query [B, C*K, ...] (vd attention lop -> patch) -> TB K hang cua moi lop [B, C, ...]; K = 1: giu nguyen.'''
    K = int(K)
    return A if K == 1 else A.unflatten(1, (A.shape[1] // K, K)).mean(2)


@torch.no_grad()
def cos_query(q_nay, q_dau):
    '''cos(q hien tai, q luc dau) TUNG hang [n] (float32) - chan doan: query lop troi xa diem khoi dau (van ban) bao nhieu.'''
    return F.cosine_similarity(q_nay.float(), q_dau.float(), dim=-1)


@torch.no_grad()
def dem_thang(lk, y=None):
    '''Nguyen mau "thang" (logit lon nhat) cua tung (anh, lop): lk [B, C, K] (GroupFCK.logit_k) -> dem long [C, K]; y [B, C]
       {0, 1} (tuy chon): CHI dem cap (anh, lop) DUONG.'''
    oh = F.one_hot(lk.argmax(-1), lk.shape[-1]).float()
    if y is not None: oh = oh * (y > 0.5).float().unsqueeze(-1)
    return oh.sum(0).long()

class DauDich(nn.Module):
    '''Dau dich moi cua MOT arm (chi cac nhom `nhom`, thu tu co dinh T1..T7). d = chieu h, C = so lop, idx3 / idx4 = chi so lop
       cua T3_nhan / T4_nhan, n5 / n6 / n7 = so bit T5 / T6 / T7, hat = hat khoi tao (cung hat + cung nhom -> cung trong so).
       forward(h [B, C, d]) -> {nhom: logit}: T1 / T2 [B, C]; T3 / T4 [B, n, 3]; T5 / T6 / T7 [B, n]. chon_dich(tgt) chon cot dich
       khop dau ra (T3 / T4: cot idx3 / idx4).'''
    NHOM = ('T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7')

    def __init__(s, d, C, idx3, idx4, n5, n6, n7, nhom, hat=0):
        super().__init__()
        nhom = tuple(nhom)
        assert nhom and len(set(nhom)) == len(nhom) and set(nhom) <= set(DauDich.NHOM), ('DauDich: nhom sai', nhom)
        s.nhom = tuple(g for g in DauDich.NHOM if g in nhom)
        s.register_buffer('idx3', torch.as_tensor(np.asarray(idx3, 'int64')).reshape(-1))
        s.register_buffer('idx4', torch.as_tensor(np.asarray(idx4, 'int64')).reshape(-1))
        n = dict(T1=int(C), T2=int(C), T3=int(s.idx3.numel()), T4=int(s.idx4.numel()), T5=int(n5), T6=int(n6), T7=int(n7))
        s.w, s.b = nn.ParameterDict(), nn.ParameterDict()
        for g in s.nhom:
            assert n[g] > 0, f'DauDich: nhom {g} khong co dau ra'
            gen = torch.Generator().manual_seed(int(hat) * 100 + DauDich.NHOM.index(g))
            hinh = (n[g], 3, int(d)) if g in ('T3', 'T4') else (n[g], int(d))
            s.w[g] = nn.Parameter(torch.randn(hinh, generator=gen) * float(d) ** -0.5)
            s.b[g] = nn.Parameter(torch.zeros(hinh[:-1]))

    def forward(s, h):
        out = {}
        for g in s.nhom:
            w, b = s.w[g], s.b[g]
            if g in ('T1', 'T2'): out[g] = (h * w).sum(-1) + b
            elif g in ('T3', 'T4'): out[g] = torch.einsum('bnd,nkd->bnk', h[:, s.idx3 if g == 'T3' else s.idx4], w) + b
            else: out[g] = h.mean(1) @ w.T + b
        return out

    def chon_dich(s, tgt):
        '''{nhom: dich [B, cot]} (cot T1..T4 = 40 lop) -> {nhom cua dau: dich khop dau ra} (T3 / T4: cot idx3 / idx4).'''
        return {g: (tgt[g][:, s.idx3] if g == 'T3' else tgt[g][:, s.idx4] if g == 'T4' else tgt[g]) for g in s.nhom}


def loss_dich(out, tgt):
    '''out = DauDich(h), tgt = DauDich.chon_dich(...) (int, -1 = mask) -> TB deu tren cac nhom co >= 1 dich hop le cua loss TB
       co mat na (nhom nhi phan: BCE logit; T3 / T4: CE 3 lop); khong nhom nao co dich -> None. O -1: gradient DUNG 0.'''
    ls, ns = [], []
    for g, z in out.items():
        t = tgt[g].long(); m = t >= 0; z = z.float()
        if z.dim() == t.dim() + 1:
            z = torch.where(m.unsqueeze(-1), z, torch.zeros_like(z))
            l = F.cross_entropy(z.reshape(-1, z.shape[-1]), t.reshape(-1), ignore_index=-1, reduction='sum')
        else:
            l = (F.binary_cross_entropy_with_logits(torch.where(m, z, torch.zeros_like(z)), t.clamp_min(0).float(),
                                                    reduction='none') * m).sum()
        n = m.sum(); ls.append(l / n.clamp_min(1)); ns.append(n)
    if not ls: return None
    co = torch.stack(ns) > 0
    if not bool(co.any()): return None
    return (torch.stack(ls) * co).sum() / co.sum()


def auc_hang(s, y):
    '''AUC hang (Mann-Whitney, hang TB khi diem trung nhau) cua diem s [N] voi nhan y [N] {-1, 0, 1} (-1 bo qua); thieu mot lop
       -> nan. Numpy, khong scipy.'''
    s = np.asarray(s, 'float64').reshape(-1); y = np.asarray(y).reshape(-1)
    m = y >= 0; s, y = s[m], y[m]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0: return float('nan')
    o = np.argsort(s, kind='mergesort'); ss = s[o]
    nhom = np.cumsum(np.r_[True, ss[1:] != ss[:-1]]) - 1
    tb = np.bincount(nhom, np.arange(1, len(ss) + 1, dtype='float64')) / np.bincount(nhom)
    r = np.empty(len(ss)); r[o] = tb[nhom]
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def chinh_xac(z, t):
    '''Do chinh xac argmax cua z [N, k] voi nhan t [N] (-1 bo qua); khong mau nao -> nan.'''
    z = np.asarray(z); t = np.asarray(t).reshape(-1); m = t >= 0
    return float((z.reshape(len(t), -1)[m].argmax(-1) == t[m]).mean()) if m.any() else float('nan')


def chan_doan_dich(out, tgt):
    '''Chan doan dau dich (nhom giu rieng): out {nhom: logit numpy noi cac lo}, tgt {nhom: dich numpy KHOP dau ra (chon_dich)} ->
       dict: T1 / T2: `<g>_auc` = TB AUC tren cac lop CO du hai lop (None neu khong lop nao) + `<g>_n_lop`; T3 / T4: `<g>_acc`
       (None neu khong mau) + `<g>_n`; T5 / T6 / T7: `<g>_auc` = list AUC TUNG bit (None khi thieu lop).'''
    r = {}
    for g, z in out.items():
        z = np.asarray(z); t = np.asarray(tgt[g])
        if g in ('T1', 'T2'):
            a = [v for v in (auc_hang(z[:, j], t[:, j]) for j in range(z.shape[1])) if np.isfinite(v)]
            r[f'{g}_auc'] = float(np.mean(a)) if a else None; r[f'{g}_n_lop'] = len(a)
        elif g in ('T3', 'T4'):
            v = chinh_xac(z.reshape(-1, z.shape[-1]), t.reshape(-1))
            r[f'{g}_acc'] = v if np.isfinite(v) else None; r[f'{g}_n'] = int((t >= 0).sum())
        else:
            r[f'{g}_auc'] = [v if np.isfinite(v) else None for v in (auc_hang(z[:, j], t[:, j]) for j in range(z.shape[1]))]
    return r

# ---- cau hinh - CHOT TRUOC (Tuan duyet 06/10; 1 seed = sang loc; KHONG chinh theo so val) ----
MUC_TIEU = json.loads(r'''{
 "_ghi_chu": "Tuan quyet 05/10 (Event 2026-10-05): dich cua arm rep (78g) = PHUONG AN 3 - co / khong lay tu nhan CXR-LT (nhan dung de cham), vi tri lay tu ImaGenome CHI KHI hai parser dong y (nhan CXR-LT duong + ImaGenome 'yes' tren vung CO HOP THAT, khong qua vung thay the); thieu vi tri = CHUA BIET (trong so trung tinh), khong phai khong co. Phuong an 1 (vector report 78e, cap duong theo vector nhan) chay cung luot lam doi chung. Vung du phong 'hai phoi' cho loss vung khi ImaGenome thieu vi tri cua nhan nam trong phoi. Dung o 78q (du lieu) va khoi_loss.py (78g).",
 "phien_ban": "muc_tieu_v1",
 "rep3": {
  "dinh_dang_cum": "<nhan>: <vung>, <vung>. khi biet vi tri; <nhan>. khi chua biet. Vung theo thu tu 36 vung ImaGenome (semantics/objects_detectable_by_bbox_pipeline_v1.txt).",
  "vector_anh": "trung binh vector BiomedBERT cua cac cum (moi nhan duong mot cum); anh khong co nhan duong -> khong co muc tieu",
  "beta": 2.0,
  "max_tok_cum": 256,
  "_ghi_chu_max_tok": "cum dai nhat co the ~15 vung (78f C5: toi da 15 vung / (anh, nhan)) x ~5 token / ten vung -> ~90 token; 256 du rong. C3 DUNG neu cum nao vuot (khong cat)."
 },
 "vung_du_phong": {
  "nhan": ["Pneumonia", "Infiltration", "Consolidation", "Pneumothorax", "Tuberculosis"],
  "vung": ["right lung", "left lung"]
 },
 "kiem_vi_tri": {
  "nguong_ap_ben": 0.95,
  "_ghi_chu": "Dat truoc khi co ket qua: probe tuyen tinh tu vector cum -> co vung ben trai / ben phai; AP trung binh hai ben >= nguong thi vector BiomedBERT giu vi tri, dung duoc cho phuong an 3; duoi nguong -> 78g can them nhung vung hoc duoc (cho Tuan quyet)."
 }
}''')
R_CAO = 0.3                        # lambda = r x ||grad L_nhan|| / ||grad L_phan|| MOI phan (vung / dich / cau), hieu chinh TRUOC train
# TAT, KHONG xoa (Tuan 07/10; 78s that: T3 muc do 0 o hop le, T7 ket cau 11 duong, bit mo xuong uc LUON 0 - ImaGenome silver gan
# nhu khong co): nhom DICH_TAT bo khoi MOI arm (C4: dich_tt = T4); bit DICH_BIT_TAT -> o -1 (khong vao loss / chan doan).
# Bat lai: DICH_TAT = () / DICH_BIT_TAT = {} (doi -> lambda + checkpoint cua luot cu KHONG dung lai duoc: RUN_TAG moi).
DICH_TAT = ('T3', 'T7')
DICH_BIT_TAT = {'T5': ('device|sternotomy wires',)}
_THU = bool(globals().get('THU_NHANH', False))
TI_LE_TANG = 0.0 if _THU else 0.1  # h(t): 0 -> 1 trong 10% so buoc toi uu dau (nhu 78m)
HC_BUOC, HC_DO = (5, 3) if _THU else (100, 50)   # hieu chinh lambda: HC_BUOC buoc chi nhan, roi do tren HC_DO micro-batch
N_DO_TL = 3                        # moi epoch: do ||grad phu|| / ||grad nhan|| tren 3 micro-batch dau (CHI chan doan)
TI_LE_GIU, HAT_GIU = 0.02, 7802    # 2% benh nhan train GIU RIENG (CUNG hat 78m)
O_LUOI = 14                        # patch RAD-DINO 14 px -> luoi G_LUOI x G_LUOI token
G_LUOI = SIZE // O_LUOI
assert G_LUOI * G_LUOI == N_TOK, (G_LUOI, N_TOK)
HAT_NN = 7803                      # txtq_nn: E = N(0, 1) [40, 768] tu np.random.default_rng(HAT_NN), z-score qua 40 lop
K_NM = 4                           # txtq_k4: K nguyen mau / lop (78s; dich_78s.json nguyen_mau.k) -> C*K query, LSE - log K
TOI_DA_CAU = 6                     # loss cau: <= TOI_DA_CAU cau / study moi buoc tu GEN_PHU (dich_78s.json cau)
THAP_MO, LR_THAP = 6, 1e-5         # cau_clip: BiomedBERT mo 6 tang tren (ConVIRT), LR 1e-5 (WD 2-D); dau chieu: LR dau doc
TAU0_CAU = 0.07                    # nhiet do khoi tao loss cau (NhietDo, hoc duoc)
# ---- BiomedBERT GHIM (NGUYEN VAN C0 cua 77a, = 78e / 78s): thap hoc cung cau_clip; token cua 78s phai tu CUNG ban ----
TXT_MODEL   = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
TXT_REV     = 'e1354b7a3a09615f6aba48dfad4b7a613eef7062'   # sha HuggingFace (2023-11-06), ghim 26/09
NGUON_78 = {'e': ('stage4_nb78e', 'r78e_v1'), 'f': ('stage4_nb78f', 'r78f_v1'), 'q': ('stage4_nb78q', 'r78q_v1'),
            's': ('stage4_nb78s', 'r78s_v1')}   # CHI DOC

# ---- doc dau ra 78e / 78f / 78q / 78s (CHI DOC, qua API; ban MOI NHAT neu trung ten) ----
_KK78 = {}


def _lay78(nb, ten):
    if nb not in _KK78: _KK78[nb] = kiem_ke(NGUON_78[nb])[1]          # MOT lan kiem ke moi notebook nguon
    ds = _KK78[nb].get(ten) or []
    assert ds, f'thieu {ten} trong cxrlt_2024/{"/".join(NGUON_78[nb])} -> chay 78{nb} truoc'
    if xung_dot(ds): print(f'  ⚠ {ten}: {len(ds)} ban KHAC noi dung -> dung ban moi nhat ({ds[0]["mt"]})')
    return tai(ds[0])


_t0 = time.time()
META_78Q = json.load(open(_lay78('q', 'meta_78q.json'), encoding='utf-8'))
assert META_78Q['muc_tieu']['phien_ban'] == MUC_TIEU['phien_ban'], 'muc tieu luc chay 78q KHAC cau hinh kho -> DUNG'
VUNG = list(META_78Q['vung']); R_VUNG = len(VUNG); VUNG_IDX = {v: i for i, v in enumerate(VUNG)}
assert R_VUNG == 36, R_VUNG
VG = pd.read_parquet(_lay78('q', 'vung_goi_y_78q.parquet'))                    # anh (78f), dicom_id, nhan, mat_na, nguon
ANH_F = pd.read_parquet(_lay78('f', 'anh_train.parquet'), columns=['anh', 'dicom_id', 'W', 'H'])
VT = pd.read_parquet(_lay78('f', 'vung_train.parquet'), columns=['anh', 'vung', 'x1', 'y1', 'x2', 'y2'])
E_MOTA = np.load(_lay78('e', 'mo_ta_40_lop_emb.npy')).astype('float32')         # [40, 768], thu tu LAB40
assert ANH_F.anh.tolist() == list(range(len(ANH_F))), 'anh_train 78f: cot anh phai la 0..n-1'
assert VG.anh.between(0, len(ANH_F) - 1).all() and (ANH_F.dicom_id.to_numpy()[VG.anh.to_numpy()] == VG.dicom_id.to_numpy()).all(), \
    'vung_goi_y (78q) va anh_train (78f) KHAC phien ban (anh <-> dicom lech) -> DUNG'
assert VT.anh.between(0, len(ANH_F) - 1).all() and VT.vung.between(0, R_VUNG - 1).all(), 'vung_train 78f: chi so anh / vung ngoai bang'
# ---- 78s (stage4_nb78s/r78s_v1, spec 2.2): cau train THEO MA (KHONG doc cot van ban), dich moi, vector van ban ----
META_78S = json.load(open(_lay78('s', 'meta_78s.json'), encoding='utf-8'))
assert META_78S.get('model') == TXT_MODEL and META_78S.get('revision') == TXT_REV and META_78S.get('max_tok') == 64, (
    f"78s ma hoa bang {META_78S.get('model')}@{META_78S.get('revision')} max_tok {META_78S.get('max_tok')} KHAC ghim 77a / 78e "
    f'({TXT_MODEL}@{TXT_REV}, 64) -> token / vector cau lech thap BiomedBERT, DUNG')
DICH_TEN = json.load(open(_lay78('s', 'dich_ten.json'), encoding='utf-8'))
assert DICH_TEN.get('phien_ban') == 'dich_v1' and DICH_TEN.get('nhan') == LAB40 and len(DICH_TEN.get('T5', ())) == 17 \
    and len(DICH_TEN.get('T6', ())) == 5 and len(DICH_TEN.get('T7', ())) == 5 and len(DICH_TEN.get('T3_nhan', ())) > 0 \
    and len(DICH_TEN.get('T4_nhan', ())) > 0 and set(DICH_TEN['T3_nhan']) | set(DICH_TEN['T4_nhan']) <= set(LAB40), \
    'dich_ten.json (78s) KHAC hop dong dich_v1 / thu tu LAB40 -> DUNG'
CAU_T = pd.read_parquet(_lay78('s', 'cau_train.parquet'), columns=['ma_cau', 'tan_so'])          # KHONG doc cot van ban
CAU_ST = pd.read_parquet(_lay78('s', 'cau_study_train.parquet'), columns=['study_id', 'ma_cau'])
E_CAU = np.load(_lay78('s', 'cau_emb.npy'), mmap_mode='r')                                      # float16 [n_cau, 768]
TK_CAU = np.load(_lay78('s', 'cau_emb_thong_ke.npy')).astype('float32')                         # [2, 768] TB, do lech chuan
TOK_CAU = np.load(_lay78('s', 'cau_token.npy'))                                                 # int32 [n_cau, 64], pad 0 ben phai
DICH_IDX = pd.read_parquet(_lay78('s', 'dich_moi_index.parquet'), columns=['dicom_id'])
with np.load(_lay78('s', 'dich_moi_train.npz')) as _z: DICH_NP = {k: _z[k] for k in _z.files}
E_5MT = np.load(_lay78('s', 'mo_ta_5x40_emb.npy')).astype('float32')                            # [40, 5, 768]
E_CAU_LOP = np.load(_lay78('s', 'txt_cau_lop.npy')).astype('float32')                           # [40, 768]
E_NM = np.load(_lay78('s', 'nguyen_mau_lop.npy')).astype('float32')                             # [40, 4, 768]
# nguyen_mau_meta.json (spec 2.2.8; khoa `fallback` - quyet dinh D-S5): theo ten nhan {nhan: muc} | list C muc (thu tu LAB40) |
# {'nhan': LAB40, 'meta': [...]}; moi muc CHI so dem + `fallback` (lop dung mo ta lop thay cau report: khong tu khoa / < 20 cau)
_NMM = json.load(open(_lay78('s', 'nguyen_mau_meta.json'), encoding='utf-8'))
if isinstance(_NMM, dict) and isinstance(_NMM.get('meta'), list):
    assert _NMM.get('nhan', LAB40) == LAB40, 'nguyen_mau_meta.json (78s): thu tu nhan KHAC LAB40 -> DUNG'
    _NMM = _NMM['meta']
NM_META = [_NMM.get(l) for l in LAB40] if isinstance(_NMM, dict) else list(_NMM)
assert len(NM_META) == C and all(isinstance(m, dict) and isinstance(m.get('fallback'), bool) for m in NM_META), \
    'nguyen_mau_meta.json (78s): can 40 muc (theo ten nhan / thu tu LAB40), moi muc co `fallback` bool (D-S5) -> DUNG'
del _NMM
N_CAU, D_TEXT = len(CAU_T), E_MOTA.shape[1]
assert E_MOTA.shape == (C, D_TEXT), E_MOTA.shape
assert CAU_T.ma_cau.tolist() == list(range(N_CAU)) and (CAU_T.tan_so >= 1).all(), 'cau_train (78s): ma_cau 0..n-1, tan_so >= 1'
assert E_CAU.dtype == np.float16 and E_CAU.shape == (N_CAU, D_TEXT) and TOK_CAU.dtype == np.int32 and TOK_CAU.shape == (N_CAU, 64) \
    and TK_CAU.shape == (2, D_TEXT) and np.isfinite(TK_CAU).all() and (TK_CAU[1] > 0).all(), \
    'vector / token / thong ke cau (78s) lech hop dong (kieu / hinh) -> DUNG'
assert CAU_ST.ma_cau.between(0, N_CAU - 1).all(), 'cau_study_train (78s): ma_cau ngoai cau_train -> 78s lech phien ban, DUNG'
_nz = TOK_CAU != 0
assert (TOK_CAU[:, 0] == TOK_CAU[0, 0]).all() and TOK_CAU[0, 0] != 0 and (_nz[:, 1:] <= _nz[:, :-1]).all() \
    and (_nz.sum(1) >= 2).all(), 'cau_token (78s): phai [CLS] ... [SEP], dem 0 ben PHAI -> DUNG'
del _nz
assert E_5MT.shape == (C, 5, D_TEXT) and E_CAU_LOP.shape == (C, D_TEXT) and E_NM.shape == (C, K_NM, D_TEXT) \
    and all(np.isfinite(x).all() for x in (E_5MT, E_CAU_LOP, E_NM)), 'vector lop 78s sai hinh / khong huu han -> DUNG'
print(f'78e / 78f / 78q / 78s: {len(ANH_F):,} anh thang ({len(VT):,} hop), {len(VG):,} (anh, nhan) vung goi y | {N_CAU:,} cau, '
      f'{len(CAU_ST):,} cap (study, cau) | dich {len(DICH_IDX):,} anh train | {time.time() - _t0:.0f} s')

# ---- kiem: MOI bang CHI anh / study TRAIN; vung goi y 78q = nhan duong train ----
N_O = len(ORDER)
_ri = pd.Series(np.arange(N_O), index=ORDER.dicom_id.to_numpy())
for _ten, _d in (('vung_goi_y_78q', VG.dicom_id), ('anh_train (78f)', ANH_F.dicom_id), ('dich_moi_index (78s)', DICH_IDX.dicom_id)):
    assert _d.isin(_ri.index).all() and TF_[_ri.loc[_d].to_numpy()].all(), f'{_ten}: co anh NGOAI train -> DUNG'
assert CAU_ST.study_id.isin(ORDER.study_id.to_numpy()[TF_]).all(), 'cau_study_train (78s) co study NGOAI train -> DUNG'
assert (YM[_ri.loc[VG.dicom_id].to_numpy(), VG.nhan.to_numpy('int64')] > 0.5).all(), 'vung goi y 78q tren nhan AM -> 78q lech, DUNG'

# ---- 2% benh nhan GIU RIENG (chan doan; KHONG vao loss phu) ----
GIU_RIENG = chon_giu_rieng(SUB[TF_], TI_LE_GIU, HAT_GIU)
_vao = TF_ & ~np.isin(SUB, GIU_RIENG)
I_GIU = np.where(TF_ & np.isin(SUB, GIU_RIENG))[0]

# ---- vung: hop 36 vung -> toa do anh 518; mat na vung goi y (du phong / khong) theo anh 78f ----
SG_HANG = np.full(N_O, -1, 'int64'); SG_HANG[_ri.loc[ANH_F.dicom_id].to_numpy()] = ANH_F.anh.to_numpy()
HOP518 = bang_hop_518(VT.anh, VT.vung, VT.x1, VT.y1, VT.x2, VT.y2, ANH_F.W.to_numpy(), ANH_F.H.to_numpy(), len(ANH_F),
                      R_VUNG, 1022, SIZE)
M_KDP = np.zeros((len(ANH_F), C), 'int64')                                       # 78t: CHI vung KHONG du phong (vung_kdp)
_kdp = VG.nguon.astype(str).to_numpy() != 'du_phong'
M_KDP[VG.anh.to_numpy()[_kdp], VG.nhan.to_numpy()[_kdp]] = VG.mat_na.to_numpy()[_kdp]
# cong HINH HOC: kho anh 518 = letterbox 1022 -> 518 (C2). Vung DEM ngoai khung anh goc phai TOI, trong khung SANG.
_lech_vuong = ANH_F[(ANH_F.W - ANH_F.H).abs() > 0.05 * ANH_F[['W', 'H']].max(1)]    # anh KHONG vuong -> co vung dem
assert len(_lech_vuong), 'khong anh thang nao khong vuong -> khong kiem duoc hinh hoc kho anh'
_mau = _lech_vuong.sample(n=min(200, len(_lech_vuong)), random_state=0)
_toi, _sang = [], []
for _r in _mau.itertuples(index=False):
    _x0, _y0, _x1, _y1 = (float(v) * SIZE / 1022 for v in hop_sang_letterbox(0.0, 0.0, _r.W, _r.H, _r.W, _r.H, 1022))
    _im = np.asarray(Image.open(os.path.join(LOCAL, f'{_r.dicom_id}.jpg')).convert('L'), 'float32')
    _ngoai = np.ones(_im.shape, bool)
    _ngoai[max(int(_y0) - 3, 0):int(np.ceil(_y1)) + 3, max(int(_x0) - 3, 0):int(np.ceil(_x1)) + 3] = False
    _trong = np.zeros(_im.shape, bool)
    _trong[int(np.ceil(_y0)) + 3:int(_y1) - 3, int(np.ceil(_x0)) + 3:int(_x1) - 3] = True
    if _ngoai.sum() >= 100 and _trong.sum() >= 100: _toi.append(_im[_ngoai].mean()); _sang.append(_im[_trong].mean())
assert len(_toi) >= min(50, len(_mau)) and np.median(_toi) < 8 and np.median(_sang) > 40, (
    f'hinh hoc kho 518 KHAC letterbox 1022 -> 518: vung dem trung vi {np.median(_toi):.1f}, khung anh {np.median(_sang):.1f} '
    f'({len(_toi)} anh) -> hop vung LECH anh, DUNG')
print(f'cong hinh hoc: {len(_toi)} anh — vung dem ngoai letterbox trung vi {np.median(_toi):.1f} / 255, trong khung '
      f'{np.median(_sang):.1f} / 255 -> hop vung khop anh 518')

# ---- POOL = anh train KHONG giu rieng (vao loss phu) ----
POOL = np.where(_vao)[0]
HANG2POOL = np.full(N_O, -1, 'int64'); HANG2POOL[POOL] = np.arange(len(POOL))
assert not (set(SUB[POOL].tolist()) & set(GIU_RIENG.tolist())) and not (set(SUB[POOL].tolist()) & set(SUB[~TF_].tolist())), \
    'benh nhan giu rieng / val / test trong POOL loss phu'

# ---- dich moi (78s): mang int8 theo dich_moi_index (= anh train theo thu tu ORDER) -> bang theo hang ORDER, CHI hang POOL ----
_KHOA_DICH = {'T1': 'T1_thay', 'T2': 'T2_chac', 'T3': 'T3_muc_do', 'T4': 'T4_ben', 'T5': 'T5_ong_tb', 'T6': 'T6_ky_thuat',
              'T7': 'T7_ket_cau'}
NHOM_DICH = tuple(_KHOA_DICH)
IDX3 = [LAB40.index(l) for l in DICH_TEN['T3_nhan']]; IDX4 = [LAB40.index(l) for l in DICH_TEN['T4_nhan']]
N5, N6, N7 = len(DICH_TEN['T5']), len(DICH_TEN['T6']), len(DICH_TEN['T7'])
_hd = _ri.loc[DICH_IDX.dicom_id].to_numpy(); N_DICH = len(_hd)
assert np.array_equal(_hd, np.where(TF_)[0]), 'dich_moi_index (78s) KHAC anh train theo thu tu ORDER -> 78s lech phien ban, DUNG'
_gt = {'T1': (C, (-1, 0, 1)), 'T2': (C, (-1, 0, 1)), 'T3': (C, (-1, 0, 1, 2)), 'T4': (C, (-1, 0, 1, 2)), 'T5': (N5, (-1, 0, 1)),
       'T6': (N6, (-1, 0, 1)), 'T7': (N7, (-1, 0, 1))}
for _g, (_k, _v) in _gt.items():
    _a = DICH_NP.get(_KHOA_DICH[_g])
    assert _a is not None and _a.dtype == np.int8 and _a.shape == (N_DICH, _k) and np.isin(_a, _v).all(), \
        f'dich {_g} (78s {_KHOA_DICH[_g]}): thieu / sai kieu / hinh / gia tri -> DUNG'
_yd = YM[_hd] > 0.5
assert not ((DICH_NP['T1_thay'] == 1) & ~_yd).any(), 'dich T1 = 1 tren nhan AM -> 78s lech nhan, DUNG'
for _g, _kk in (('T2', None), ('T3', IDX3), ('T4', IDX4)):
    _a = DICH_NP[_KHOA_DICH[_g]] >= 0
    assert not (_a & ~_yd).any(), f'dich {_g}: co dich tren nhan AM -> 78s lech nhan, DUNG'
    assert _kk is None or not np.delete(_a, _kk, axis=1).any(), f'dich {_g}: co dich NGOAI {_g}_nhan -> DUNG'
# TAT (C3c, KHONG xoa): nhom DICH_TAT van nap + kiem (bat lai duoc), C4 bo khoi arm; bit DICH_BIT_TAT -> o -1 tren ban sao
assert set(DICH_TAT) <= set(NHOM_DICH) and set(DICH_BIT_TAT) <= {'T5', 'T6', 'T7'} and all(
    _b and set(_b) <= set(DICH_TEN[_g]) for _g, _b in DICH_BIT_TAT.items()), \
    f'DICH_TAT {DICH_TAT} / DICH_BIT_TAT {DICH_BIT_TAT} (C3c): nhom / bit khong co trong dich_ten.json (78s) -> DUNG'
_DN = {_g: np.array(DICH_NP[_KHOA_DICH[_g]]) for _g in NHOM_DICH}
for _g, _b in DICH_BIT_TAT.items():
    _DN[_g][:, [DICH_TEN[_g].index(_x) for _x in _b]] = -1
_vd = np.full(N_O, -1, 'int64'); _vd[_hd] = np.arange(N_DICH)
HANG2DICH = np.full(N_O, N_DICH, 'int64'); HANG2DICH[POOL] = _vd[POOL]          # ngoai POOL -> hang CUOI (moi o -1)
DICH_T = {_g: torch.from_numpy(np.concatenate([_DN[_g], np.full((1, _k), -1, 'int8')])).to(dev)
          for _g, (_k, _) in _gt.items()}
HANG2DICH_T = torch.from_numpy(HANG2DICH).to(dev)
DICH_G = {_g: torch.from_numpy(_DN[_g][_vd[I_GIU]]).to(dev) for _g in NHOM_DICH}   # GIU RIENG: CHI chan doan
_n_dich = {_g: int((_DN[_g][_vd[POOL]] >= 0).sum()) for _g in NHOM_DICH}
del DICH_NP, _DN, _yd

# ---- cau (78s, THEO MA cau): CSR study -> cau tren study POOL (+ study GIU RIENG cho chan doan truy hoi). ST: hang ORDER -> chi so
#      study POOL CO >= 1 cau, -1 voi moi hang khac (giu rieng / ngoai train / study khong cau) - khoi_cau BAT BUOC ----
_sd_o = ORDER.study_id.to_numpy()
_cs, _cm = CAU_ST.study_id.to_numpy('int64'), CAU_ST.ma_cau.to_numpy('int64')


def _csr_cau(hang):
    '''CSR (ptr, idx) cua cac study cua tap hang ORDER `hang` (study = np.unique(study_id), tang dan) + st [len(hang)] = chi so study
       cua tung hang neu study co >= 1 cau, -1 neu khong.'''
    us = np.unique(_sd_o[hang])
    if len(us) == 0: return torch.zeros(1, dtype=torch.long), torch.zeros(0, dtype=torch.long), np.zeros(0, 'int64')
    i = np.searchsorted(us, _cs); co = (i < len(us)) & (us[np.minimum(i, len(us) - 1)] == _cs)
    ptr, idx = bang_cau(i[co], _cm[co], len(us))
    j = np.searchsorted(us, _sd_o[hang])
    return ptr, idx, np.where(np.diff(ptr.numpy())[j] > 0, j, -1).astype('int64')


PTR, IDX, _st = _csr_cau(POOL)
ST = np.full(N_O, -1, 'int64'); ST[POOL] = _st
PTR_G, IDX_G, ST_G = _csr_cau(I_GIU)                                            # ST_G theo thu tu I_GIU
assert (ST[~_vao] == -1).all() and (ST[POOL] < len(PTR) - 1).all(), 'ST sai (hang ngoai POOL co study)'
PTR_T, IDX_T, ST_T = PTR.to(dev), IDX.to(dev), torch.from_numpy(ST).to(dev)
PTR_G_T, IDX_G_T, ST_G_T = PTR_G.to(dev), IDX_G.to(dev), torch.from_numpy(ST_G).to(dev)
MA_G_T = torch.unique(IDX_G_T)                                                  # kho cau cua nhom giu rieng (ung vien truy hoi)
SUB_T = torch.from_numpy(SUB.astype('int64')).to(dev)
W_CAU_T = trong_so_cau(CAU_T.tan_so.to_numpy()).to(dev)                          # 1 / log(1 + tan_so) theo ma cau
Z_CAU = np.empty(E_CAU.shape, 'float16')
for _i in range(0, N_CAU, 1 << 16):                                             # z-score (cau_emb_thong_ke) tung khoi, giu float16
    Z_CAU[_i:_i + (1 << 16)] = ((np.asarray(E_CAU[_i:_i + (1 << 16)], 'float32') - TK_CAU[0]) / TK_CAU[1]).astype('float16')
assert np.isfinite(Z_CAU).all(), 'vector cau z-score khong huu han -> DUNG'
Z_CAU_T, TOK_T = torch.from_numpy(Z_CAU).to(dev), torch.from_numpy(TOK_CAU).to(dev)
_n_st_p, _n_cau_tb = len(PTR) - 1, float(np.diff(PTR.numpy()).mean())
del Z_CAU, E_CAU, TOK_CAU, CAU_ST, _cs, _cm

# ---- txtq: vector query THO moi bien the -> z-score QUA cac hang (khoi_txtq.z_lop = bieu thuc E_MOTA_T cua 78g) ----
E_MOTA_T = z_lop(E_MOTA)                                                        # = 78g (txtq, txtq_du, txtq_vung, txtq_dich)
E_ARM = {'mota': E_MOTA_T, 'nn': z_lop(e_ngau_nhien(C, D_TEXT, HAT_NN)), '5mt': z_lop(E_5MT.mean(1)), 'cau': z_lop(E_CAU_LOP),
         'k4': z_lop(e_nguyen_mau(E_NM))}
assert all(tuple(v.shape) == ((C * K_NM) if k == 'k4' else C, D_TEXT) for k, v in E_ARM.items()), {k: v.shape for k, v in E_ARM.items()}
# lop du phong (fallback, D-S5): txt_cau_lop = mo ta lop, K nguyen mau = K ban sao cua no (txtq_k4 = txtq_cau o lop do)
_DP_NM = [LAB40[j] for j in range(C) if NM_META[j]['fallback']]
assert all((E_NM[j] == E_CAU_LOP[j][None, :]).all() for j in range(C) if NM_META[j]['fallback']), \
    'nguyen_mau_meta (78s): lop fallback ma nguyen mau KHAC K ban sao txt_cau_lop -> 78s lech, DUNG'
_NM_TRUNG = [LAB40[j] for j in range(C) if not NM_META[j]['fallback'] and (E_NM[j] == E_NM[j, :1]).all()]
if _NM_TRUNG: print(f'  ⚠ {len(_NM_TRUNG)} lop KHONG fallback ma {K_NM} nguyen mau TRUNG nhau (k-means suy bien?): {_NM_TRUNG}')
NHAN_PHOI = [LAB40.index(l) for l in MUC_TIEU['vung_du_phong']['nhan']]         # 5 nhan trong phoi (chan doan tach rieng)
PHOI_T = torch.zeros(C, dtype=torch.bool, device=dev); PHOI_T[NHAN_PHOI] = True
HANG2POOL_T = torch.from_numpy(HANG2POOL).to(dev); SG_HANG_T = torch.from_numpy(SG_HANG).to(dev)
M_KDP_T = torch.from_numpy(M_KDP).to(dev)
print(f'C3c 78t: POOL {len(POOL):,} anh train ({len(np.unique(SUB[POOL])):,} benh nhan) | GIU RIENG {len(GIU_RIENG):,} benh nhan, '
      f'{len(I_GIU):,} anh | vung: {int((SG_HANG >= 0).sum()):,} anh co hop, (anh, nhan) co vung kdp {int((M_KDP > 0).sum()):,}')
print('  dich (o hop le tren POOL): ' + ' '.join(f'{g} {n:,}' for g, n in _n_dich.items()) + f' | T3 {len(IDX3)} nhan, '
      f'T4 {len(IDX4)} nhan, T5 / T6 / T7 {N5} / {N6} / {N7} bit | TAT (C3c): nhom {DICH_TAT or "-"}, bit {DICH_BIT_TAT or "-"}')
print(f'  cau: {N_CAU:,} cau khac nhau | POOL {_n_st_p:,} study (TB {_n_cau_tb:.1f} cau), {int((ST[POOL] >= 0).sum()):,} / '
      f'{len(POOL):,} anh co cau | GIU RIENG kho {len(MA_G_T):,} cau, {int((ST_G >= 0).sum()):,} / {len(I_GIU):,} anh co cau')
print('  txtq: E ' + ', '.join(f'{k} {tuple(v.shape)}' for k, v in E_ARM.items()) + f' | {len(_DP_NM)} lop fallback (txtq_cau / '
      f'txtq_k4 dung mo ta lop 78s): {_DP_NM}')

# ---- 78g `txtq`: query lop tu 40 mo ta lop (78e), z-score qua 40 lop roi Linear -> LayerNorm ----
class QueryTuText(nn.Module):
    def __init__(s, E, d):
        super().__init__(); s.register_buffer('E', E.float()); s.lin = nn.Linear(E.shape[1], d); s.ln = nn.LayerNorm(d)

    def forward(s):
        return s.ln(s.lin(s.E))


class MLDecoderTxtQ(MLDecoder):
    '''78g txtq: MLDecoder (phan chung tao TRUOC, CUNG dong ngau nhien -> CUNG khoi tao voi mldec) + query lop tu text. qtxt
       khoi tao trong fork_rng voi hat rieng -> KHONG tieu dong ngau nhien chung. forward = mldec_h tung bit (cong C4).'''
    def __init__(s, *a, E_mo_ta=None, hat=0, **k):
        super().__init__(*a, **k)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            torch.manual_seed(hat); s.qtxt = QueryTuText(E_mo_ta, s.q.weight.shape[1])

    def forward(s, patch, etok=None, need_w=False):
        q = s.qtxt().unsqueeze(0).expand(patch.shape[0], -1, -1)
        kv = s.pin(patch)
        if s.t1 is not None: kv = s.t1(kv, s.ein(etok))
        h, w = s.img(q, kv, need_w); return s.out(h), w


def q_cua(m):
    '''Query lop [40, d] cua decoder `m` (txtq: tu text; q_hoc: hoc duoc; con lai: co dinh).'''
    return m.qtxt() if hasattr(m, 'qtxt') else m.q.weight


def mldec_h(m, patch):
    '''MOT forward decoder -> (logit, h = 40 dau ra query TRUOC GroupFC, kv = pin(patch)). CUNG phep tinh MLDecoder /
       MLDecoderTxtQ.forward (khong Tier1) -> logit TRUNG tung bit (cong C4, eval + train cung dong ngau nhien).'''
    q = q_cua(m).unsqueeze(0).expand(patch.shape[0], -1, -1)
    kv = m.pin(patch)
    h, _ = m.img(q, kv, False)
    return m.out(h), h, kv


_KHOA_C3C = (THU_NHANH, RUN_TAG)   # C6 kiem: doi C0 thi PHAI chay lai C3c
