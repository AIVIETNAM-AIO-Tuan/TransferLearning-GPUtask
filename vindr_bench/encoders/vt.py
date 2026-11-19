# -*- coding: utf-8 -*-
"""Năm hàng của đối chứng "vị trí mô-đun vùng" (docs/PLAN_90_SO_SANH_GIAI_PHAU.md Việc B, §10.1–§10.2; Tuấn duyệt 17/11, dựng VinDr 19/11) — lớp bọc
ĐÔNG CỨNG theo giao diện chung (base.EncoderWrapper), như B. Không học, không đổi trọng số / phép tính; bốn đầu đọc dùng y hệt mọi hàng.

Nguồn: lượt 78x seed 42, 6 epoch, trọng số EMA epoch 4 (`rs_*_ep4.pt`, khoá 'ep' = 3) — đúng backbone mà 78r / 78z của Việc B đọc. Drive
`cxrlt_2024/stage4_nb78x/<lượt>/`:

  khoá            hàng  lượt 78x                             mô-đun vùng lúc học                               lớp bọc
  vt_b4v2         (3)   r78x_J_v3b_dp_ema_v2_s42             trong encoder (sau block 8–11) + GHI NGƯỢC        B4v2  (= B / ArrE KHÔNG đường ô cắt)
  vt_mot_chieu    (2)   r78x_J_v3b_dp_ema_vt_mot_chieu_s42   trong encoder, CHỈ ĐỌC                            RadDinoFT
  vt_sau4         (1a)  r78x_J_v3b_dp_ema_vt_sau4_s42        4 lớp JCA SAU block cuối, chỉ đọc                 RadDinoFT
  vt_sau1         (1b)  r78x_J_v3b_dp_ema_vt_sau1_s42        1 lớp JCA sau block cuối, chỉ đọc                 RadDinoFT
  vt_khong_luong  (0)   r78x_B1_dp_ema_s42                   KHÔNG có (B1 + drop-path + EMA)                   RadDinoFT

- B4v2: RAD-DINO tinh chỉnh ('bb') + luồng vùng J v3b ('luong'; hệ số ghi ngược g, trần rho theo ảnh): block 0..7 thường, block 8..11 mỗi block một bước
  luồng vùng (ghi ngược vào token ảnh) -> 4 tầng cuối SAU bước luồng vùng, qua layernorm cuối. = B._blocks bỏ token_o7 / skip_o / cong_skip (= vòng
  upper_luong_hr của 78y / 78z); CLS = token 0, patch = token 1.. (1369).
- RadDinoFT: luồng vùng KHÔNG ghi vào token ảnh (chỉ đọc) hoặc không có -> token ảnh = forward RAD-DINO với trọng số tinh chỉnh: đặc trưng = đúng phép
  tính của raddino.RadDino (hidden_states 4 tầng cuối qua layernorm). CỔNG lúc dựng cho ba hàng chỉ đọc (PLAN_90 §10.2 "trùng forward đầy đủ từng bit"):
  dựng LẠI luồng vùng đã học từ backbone_ft.pt + 'luong' (nạp chặt), chạy forward ĐẦY ĐỦ (block + mọi bước luồng vùng như lúc học) trên GATE_IMAGES ảnh
  tổng hợp, CPU fp32 -> 4 tầng PHẢI trùng _blocks TỪNG BIT, luồng vùng phải thật sự chạy (đủ số lớp JCA) và không có tỉ lệ ghi ngược. Hàng (0): không
  luồng vùng -> chỉ cổng cấu hình (cfg_kl).
- Cổng "đúng lượt" = các cổng đã ĐẠT trên chính các tệp này ở C2b của 78r VT / 78z đợt 4 / 78z VT (18/11): run / luot / ep / trọng số EMA; cfg_d4 (dp_ema,
  ep_bb 4, 6 epoch, không khe); cfg_vt (biến thể, hai_chieu False); cfg_kl (hàng (0)); không module ô cắt.

Giả định (ghi rõ): mọi ảnh VinDr là ảnh thẳng -> neo = (cfg_jv2['view_neo'] == 'frontal') = 1, như B. Chỉ dùng cú pháp / API có ở Python 3.9 + torch 2.0."""
import torch

from ..checks import safe_load, weights
from . import b_src as S
from . import files
from . import vt_src as VS
from .base import EncoderWrapper
from .raddino import RENAME, REPO, REVISION
from .spec import IMAGENET_MEAN, IMAGENET_STD, SPECS

EPOCH = 4                                   # rs_*_ep4.pt = trọng số EMA epoch 4 (khoá 'ep' = 3): số quyết định của B.4 v2 / 78r / PLAN_90 §10.1
CFG_FILE = 'backbone_ft.pt'
LUOT = 'B1'                                 # khoá 'luot' của mọi rs_ 78x (nhánh 78u) — như B
LUOT_J = 'J_bert_dong'                      # luot_79a / luot_chon của mọi lượt J
CROP_KEYS = ('bo_o7', 'dua_o', 'bo_o11', 'gop_o')
EXPECT = dict(R=36, tang=(8, 9, 10, 11), view_neo='frontal', d4='dp_ema', ep_bb=4, epochs=6)
EXPECT_GHI = dict(g=5.0, rho=0.1)           # hàng (3): hệ số ghi ngược + trần (= B / ArrE)
EXPECT_KL = dict(ten='b1_dp_ema', ep_bb=4, epochs=6)
ROWS = {
    'vt_b4v2': dict(hang='(3)', run='r78x_J_v3b_dp_ema_v2_s42', luong='ghi_nguoc', vt=None),
    'vt_mot_chieu': dict(hang='(2)', run='r78x_J_v3b_dp_ema_vt_mot_chieu_s42', luong='chi_doc', vt=dict(ten='mot_chieu', vi_tri='trong_encoder', n_lop=4)),
    'vt_sau4': dict(hang='(1a)', run='r78x_J_v3b_dp_ema_vt_sau4_s42', luong='chi_doc', vt=dict(ten='sau4', vi_tri='sau_encoder', n_lop=4)),
    'vt_sau1': dict(hang='(1b)', run='r78x_J_v3b_dp_ema_vt_sau1_s42', luong='chi_doc', vt=dict(ten='sau1', vi_tri='sau_encoder', n_lop=1)),
    'vt_khong_luong': dict(hang='(0)', run='r78x_B1_dp_ema_s42', luong=None, vt=None),
}
GATE_IMAGES, GATE_SEED = 2, 19             # cổng chỉ đọc: 2 ảnh tổng hợp N(0, 1) ở chuẩn hoá gốc, generator riêng (không tiêu dòng ngẫu nhiên chung)


def _with(prefix, sd):
    return {f'{prefix}.{k}': v for k, v in sd.items()}


def load_files(spec, path=None, cfg_path=None):
    '''-> (ck, nguồn ck, cfj | None, nguồn cfj | None) sau cổng "đúng lượt" của hàng spec.name (xem đầu tệp). Hàng (0): không backbone_ft.pt.'''
    row = ROWS[spec.name]
    path = path or files.weights_path(spec)
    # GLOBAL của pickle: rs_ chỉ lõi -> weights_only=True; backbone_ft.pt có torch.torch_version.TorchVersion (như tệp thật của B, 15/11)
    ck, src = safe_load.load_checkpoint(path, expect_md5=spec.file_md5, expect_size=spec.file_size)
    assert isinstance(ck, dict) and isinstance(ck.get('bb'), dict) and ck.get('run') == row['run'] and ck.get('luot') == LUOT and ck.get('ep') == EPOCH - 1, (
        f"{spec.name}: checkpoint KHONG phai {row['run']} / luot {LUOT} / ep {EPOCH} (co {ck.get('run')}, {ck.get('luot')}, ep {ck.get('ep')}) -> DUNG")
    assert not any(k in ck for k in CROP_KEYS), f'{spec.name}: checkpoint co module o cat {[k for k in CROP_KEYS if k in ck]} -> DUNG (hang Viec B khong o cat)'
    ema = ck.get('ema')
    assert isinstance(ema, dict) and ema.get('trong_so') == 'EMA' and int(ema.get('n', 0)) > 0, f'{spec.name}: rs_ KHONG phai trong so EMA ({ema}) -> DUNG'
    if row['luong'] is None:
        kl = ck.get('cfg_kl')
        assert 'luong' not in ck and isinstance(kl, dict) and kl.get('ten') == EXPECT_KL['ten'] and int(kl.get('ep_bb', -1)) == EXPECT_KL['ep_bb'] \
            and int(kl.get('epochs', -1)) == EXPECT_KL['epochs'] and kl.get('luong') is None and kl.get('trong_so') == 'EMA', (
            f"{spec.name}: cfg_kl {kl} / co luong {'luong' in ck} KHAC hang khong luong vung (b1_dp_ema, ep_bb 4, 6 epoch, trong so EMA) -> DUNG")
        return ck, src, None, None
    assert isinstance(ck.get('luong'), dict) and 'cfg_kl' not in ck, f'{spec.name}: checkpoint thieu luong vung / co cfg_kl -> DUNG'
    extra = {n: (m, z) for n, m, z in spec.extra_files}
    cfg_path = cfg_path or files.weights_path(spec, CFG_FILE)
    cfj, src_cfg = safe_load.load_checkpoint(cfg_path, expect_md5=extra[CFG_FILE][0], expect_size=extra[CFG_FILE][1], allow=('torch_version',))
    assert isinstance(cfj, dict) and cfj.get('run') == row['run'] and cfj.get('luot_79a') == LUOT_J and 'cfg_o' not in cfj and all(
        isinstance(cfj.get(k), dict) for k in ('cfg_luong', 'cfg_jv2', 'cfg_jv3', 'cfg_jv3b', 'cfg_d4')), (
        f"{spec.name}: {CFG_FILE} khong phai cau hinh cua {row['run']} (run / luot_79a / cfg_* / co cfg_o) -> DUNG")
    c79, cd4 = cfj['cfg_luong'], cfj['cfg_d4']
    assert c79.get('luot_chon') == LUOT_J and torch.equal(cfj['hop_tb'].float(), ck['luong']['hop_tb'].float().cpu()), (
        f'{spec.name}: hop_tb / luot_chon cua {CFG_FILE} KHAC checkpoint -> DUNG')
    assert cd4.get('ten') == EXPECT['d4'] and int(cd4.get('ep_bb', -1)) == EXPECT['ep_bb'] and int(cd4.get('epochs', -1)) == EXPECT['epochs'] \
        and cd4.get('khe') is None and not any(k.startswith('khe.') for k in ck['luong']) and cd4.get('trong_so') == 'EMA', (
        f"{spec.name}: cfg_d4 {cd4} KHAC luot {EXPECT['d4']} / ep_bb {EXPECT['ep_bb']} / {EXPECT['epochs']} epoch / khong khe / trong so EMA -> DUNG")
    assert str(cfj['cfg_jv2'].get('view_neo')) == EXPECT['view_neo'], f"{spec.name}: view_neo {cfj['cfg_jv2'].get('view_neo')} KHAC 'frontal' -> DUNG"
    hai = bool(c79['luot'][LUOT_J]['hai_chieu'])
    vt_f, vt_c = cfj.get('cfg_vt'), ck.get('cfg_vt')
    if row['vt'] is None:
        assert hai and vt_f is None and vt_c is None, f'{spec.name}: hang ghi nguoc phai hai chieu, khong cfg_vt (hai_chieu {hai}, cfg_vt {vt_f}) -> DUNG'
    else:
        assert not hai and isinstance(vt_f, dict) and vt_f.get('hai_chieu') is False and all(vt_f.get(k) == v for k, v in row['vt'].items()) \
            and (vt_c is None or vt_c == vt_f), f"{spec.name}: cfg_vt {vt_f} (rs_: {vt_c}) / hai_chieu {hai} KHAC bien the chi doc {row['vt']} -> DUNG"
    return ck, src, cfj, src_cfg


def raddino_ft(spec, ck, revision=REVISION, attn=None):
    '''RAD-DINO (HF, revision ghim của lớp bọc RAD-DINO) + trọng số tinh chỉnh 'bb' (đổi tên khoá theo bản transformers, như C2 của 86b_B / lớp bọc B).
       -> (mô hình, attention dùng, lệch lớn nhất so với RAD-DINO gốc, phiên bản transformers).'''
    import transformers
    kw = dict(revision=revision)
    if attn:
        kw['attn_implementation'] = attn
    model = transformers.AutoModel.from_pretrained(REPO, **kw)
    used = getattr(model.config, '_attn_implementation', None)
    assert not attn or used == attn, f'{spec.name}: attention {used!r} != yeu cau {attn!r} -> DUNG'
    sd0 = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(S.doi_ten_dinov2(ck['bb'], sd0), strict=True)
    lech = max(float((sd0[k].float() - v.float()).abs().max()) for k, v in model.state_dict().items())
    assert lech > 1e-6, f'{spec.name}: backbone GIONG HET RAD-DINO goc -> trong so tinh chinh KHONG duoc nap -> DUNG'
    cfg = model.config
    assert (cfg.patch_size, cfg.hidden_size) == (spec.patch_size, spec.embed_dim), (cfg.patch_size, cfg.hidden_size)
    return model, used, lech, transformers.__version__


def build_luong(spec, cfj, luong_sd, vt=None):
    '''Luồng vùng đã học (= C2b của 78z / 78z VT / lớp bọc B): LuongVungNeo theo cfg_luong -> hệ số ghi ngược g -> trần rho -> (biến thể sau encoder:
       đổi lớp, cùng khoá state_dict) -> NẠP CHẶT. -> (luồng ở eval, không grad; dict cấu hình đã dựng).'''
    c79, cv3, cv3b = cfj['cfg_luong'], cfj['cfg_jv3'], cfj['cfg_jv3b']
    luot = c79['luot'][c79['luot_chon']]
    hop = cfj['hop_tb'].float()
    with torch.random.fork_rng(devices=[]):                    # khởi tạo (ghi đè ngay khi nạp) không tiêu dòng ngẫu nhiên chung
        torch.manual_seed(0)
        lu = S.LuongVungNeo(hop.numpy(), d=spec.embed_dim, R=int(hop.shape[0]), tang=tuple(c79['tang']), n_head=int(c79['n_head']), ff=int(c79['ff']),
                            beta=float(c79['beta']), t_mem=float(c79['t_mem']), hai_chieu=bool(luot['hai_chieu']), neo=bool(luot['neo']),
                            luoi=spec.image_size // spec.patch_size, ckpt=False)
        n_g = S.tang_ghi_nguoc(lu, float(cv3['ghi_nguoc_g']))
        n_t = S.ap_tran_ghi(lu, float(cv3b['tran_ghi_rho']))
    n_chay = len(lu.lop)
    if vt is not None and vt['vi_tri'] == 'sau_encoder':
        n_chay, _ = VS.ap_sau_enc(lu, int(vt['n_lop']))
    lu.load_state_dict(luong_sd, strict=True)
    assert lu.R == EXPECT['R'] and tuple(lu.tang) == EXPECT['tang'] and n_g == n_t == len(lu.tang), (
        f'{spec.name}: luong vung R {lu.R} / tang {lu.tang} / so lop doi {n_g}, {n_t} KHAC luot Viec B -> DUNG')
    return lu.eval().requires_grad_(False), dict(R=lu.R, tang=list(lu.tang), g=float(cv3['ghi_nguoc_g']), rho=float(cv3b['tran_ghi_rho']),
                                                 hai_chieu=bool(lu.hai_chieu), lop=type(lu).__name__, so_lop_jca_chay=int(n_chay))


def forward_luong(model, luong, x, neo, n):
    '''Forward ĐẦY ĐỦ như lúc học / như 78y-78z: embeddings -> block 0..tang[0]-1 -> block tang[0]..cuối, sau MỖI block một bước luồng vùng (mo_ghi True:
       ghi ngược khi luồng hai chiều) -> (n tầng cuối qua layernorm cuối, trạng thái luồng). x = ảnh ở chuẩn hoá GỐC.'''
    B, layers = int(x.shape[0]), model.encoder.layer
    h = model.embeddings(x)
    for i in range(0, luong.tang[0]):
        h = S._out_luong(layers[i](h))
    luong.dat_neo(torch.full((B,), float(neo), device=x.device))
    st = luong.bat_dau(B)
    outs = []
    for l in range(luong.tang[0], len(layers)):
        h = S._out_luong(layers[l](h))
        h = luong.buoc(l, h, st, True)
        outs.append(h)
    assert len(outs) >= n, f'luong vung tu block {luong.tang[0]}: chi {len(outs)} tang < {n} -> DUNG'
    return [model.layernorm(o) for o in outs[-n:]], st


def _source(name):
    return f"cxrlt_2024/stage4_nb78x/{ROWS[name]['run']} (luot 78x, ep {EPOCH}, trong so EMA)"


class B4v2(EncoderWrapper):
    '''Hàng (3): luồng vùng trong encoder + ghi ngược (xem đầu tệp).'''
    n_prefix = 1

    def __init__(self, name='vt_b4v2', input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None, cfg_path=None, attn=None, view='frontal',
                 revision=REVISION):
        assert ROWS[name]['luong'] == 'ghi_nguoc', f'{name}: khong phai hang ghi nguoc -> dung RadDinoFT'
        spec = SPECS[name]
        super().__init__(spec, input_mean, input_std)
        ck, src, cfj, src_cfg = load_files(spec, path, cfg_path)
        self.model, used, lech_pre, tv = raddino_ft(spec, ck, revision, attn)
        self.luong, info = build_luong(spec, cfj, ck['luong'])
        L = len(self.model.encoder.layer)
        assert info['hai_chieu'] is True and (info['g'], info['rho']) == (EXPECT_GHI['g'], EXPECT_GHI['rho']) and list(self.luong.tang) == list(
            range(self.luong.tang[0], L)), f'{name}: luong {info} KHAC hang ghi nguoc (g 5, rho 0,1, tang lien tuc toi block cuoi {L - 1}) -> DUNG'
        self.neo = float(str(cfj['cfg_jv2']['view_neo']) == str(view))
        for m in (self.model, self.luong):                                 # như B: đông cứng, requires_grad False
            m.eval().requires_grad_(False)
        state = {**_with('bb', self.model.state_dict()), **_with('luong', self.luong.state_dict())}
        filet = {**_with('bb', ck['bb']), **_with('luong', ck['luong'])}
        self.nap = weights.record(source=f'{_source(name)} + {CFG_FILE}', file_name=spec.drive_file,
                                  library=f'transformers {tv} ({REPO}@{revision[:12]}) + ma khoi 78 / 79 (encoders/b_src.py)',
                                  model_state=state, file_tensors=filet, lib_info=None, rename=RENAME, nguon_tep=src, nguon_cau_hinh=src_cfg,
                                  attention=used, lech_rad_dino_goc=lech_pre, trong_so='EMA', ema={k: ck['ema'].get(k) for k in ('n', 'he_so', 'trong_so')},
                                  hang=ROWS[name]['hang'], luong=dict(info, neo=self.neo, view_neo=str(cfj['cfg_jv2']['view_neo'])),
                                  khoa_khong_dung=sorted(k for k in ck if k not in ('bb', 'luong')))
        self.train_log = ck.get('log')                                     # nhật ký huấn luyện (tỉ lệ ghi ngược từng lớp: bộ kiểm)
        del ck, cfj

    def _blocks(self, x, n):
        '''x = ảnh 518 chuẩn hoá gốc -> n tầng cuối (đầu ra block 8..11 SAU bước luồng vùng) qua layernorm, [B, 1 + 1369, 768].'''
        return forward_luong(self.model, self.luong, x, self.neo, n)[0]


class RadDinoFT(EncoderWrapper):
    '''Hàng (0), (1a), (1b), (2): RAD-DINO + trọng số tinh chỉnh của lượt (xem đầu tệp); hàng chỉ đọc có cổng "trùng forward đầy đủ" lúc dựng.'''
    n_prefix = 1

    def __init__(self, name, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None, cfg_path=None, attn=None, view='frontal', revision=REVISION):
        row = ROWS[name]
        assert row['luong'] in (None, 'chi_doc'), f'{name}: hang ghi nguoc -> dung B4v2'
        spec = SPECS[name]
        super().__init__(spec, input_mean, input_std)
        ck, src, cfj, src_cfg = load_files(spec, path, cfg_path)
        self.model, used, lech_pre, tv = raddino_ft(spec, ck, revision, attn)
        self.model.eval().requires_grad_(False)
        cong = info = luong_tep = None
        if row['luong'] == 'chi_doc':
            lu, info = build_luong(spec, cfj, ck['luong'], row['vt'])
            assert info['hai_chieu'] is False, f'{name}: luong vung hai chieu o hang chi doc -> DUNG'
            luong_tep = weights.compare_to_file(lu.state_dict(), ck['luong'])
            cong = self.full_forward_check(lu, float(str(cfj['cfg_jv2']['view_neo']) == str(view)), row['vt'])
            info = dict(info, view_neo=str(cfj['cfg_jv2']['view_neo']))
            del lu
        self.nap = weights.record(source=_source(name) + (f' + {CFG_FILE}' if cfj is not None else ''), file_name=spec.drive_file,
                                  library=f'transformers {tv} ({REPO}@{revision[:12]})' + (' + ma khoi 78 / 79 (cong chi doc)' if cong else ''),
                                  model_state=_with('bb', self.model.state_dict()), file_tensors=_with('bb', ck['bb']), lib_info=None, rename=RENAME,
                                  nguon_tep=src, nguon_cau_hinh=src_cfg, attention=used, lech_rad_dino_goc=lech_pre, trong_so='EMA',
                                  ema={k: ck['ema'].get(k) for k in ('n', 'he_so', 'trong_so')}, hang=row['hang'], luong=info,
                                  luong_so_voi_tep=luong_tep, cong_chi_doc=cong, khoa_khong_dung=sorted(k for k in ck if k != 'bb'))
        if cong is not None:                                               # cổng lúc dựng: luồng đã nạp = tệp + forward đầy đủ = _blocks từng bit
            self.nap['dat'] = bool(self.nap['dat'] and cong['dat'] and weights.loading_ok(dict(self.nap, so_voi_tep=luong_tep)))
        del ck, cfj

    @torch.no_grad()
    def full_forward_check(self, luong, neo, vt):
        '''Cổng chỉ đọc: forward đầy đủ (block + mọi bước luồng vùng đã học) = _blocks TỪNG BIT ở 4 tầng; luồng chạy đủ lớp JCA, không ghi ngược.
           GATE_IMAGES ảnh tổng hợp ở chuẩn hoá gốc, CPU fp32 (bên gọi: mô hình còn ở CPU lúc dựng). -> dict (số đo trong tiến trình).'''
        dev = next(self.model.parameters()).device
        s = self.spec.image_size
        x = torch.randn(GATE_IMAGES, 3, s, s, generator=torch.Generator().manual_seed(GATE_SEED)).to(dev)
        a = self._blocks(x, 4)
        b, st = forward_luong(self.model, luong, x, neo, 4)
        bits = len(a) == len(b) == 4 and all(torch.equal(p, q) for p, q in zip(a, b))
        lech = max(float((p.float() - q.float()).abs().max()) for p, q in zip(a, b))
        n_can = len(luong.tang) if vt['vi_tri'] == 'trong_encoder' else int(vt['n_lop'])
        n_hop = len(st['hop_cac_tang'])
        return dict(dat=bool(bits and n_hop == n_can and not st['ghi']), trung_bit=bool(bits), lech_max=lech, so_lop_jca_chay=n_hop, so_lop_can=n_can,
                    so_ti_le_ghi=len(st['ghi']), so_anh=GATE_IMAGES, thiet_bi=str(dev), kieu=str(x.dtype).replace('torch.', ''))

    def _blocks(self, x, n):
        '''= raddino.RadDino._blocks: hidden_states n tầng cuối, mỗi tầng qua layernorm cuối.'''
        hidden = self.model(x, output_hidden_states=True).hidden_states[-n:]
        return [self.model.layernorm(h) for h in hidden]


def build(name, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None, cfg_path=None, attn=None):
    assert name in ROWS, f'{name}: khong thuoc 5 hang Viec B {tuple(ROWS)} -> DUNG'
    cls = B4v2 if ROWS[name]['luong'] == 'ghi_nguoc' else RadDinoFT
    return cls(name, input_mean, input_std, path=path, cfg_path=cfg_path, attn=attn).eval()
