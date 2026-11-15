# -*- coding: utf-8 -*-
"""B (s42) — lớp bọc ĐÔNG CỨNG theo giao diện chung (base.EncoderWrapper), như 8 encoder còn lại (Tuấn 15/11: "freeze kiến trúc backbone của B lại, dùng
nó như một backbone như các kiến trúc khác"). Không học, không đổi trọng số / phép tính; bốn đầu đọc dùng y hệt mọi hàng (ML-Decoder KHÔNG đọc thêm token
vùng — bỏ điểm đó của PLAN_89 §2, như `86b_B`).

Nguồn: lượt 78x `r78x_J_v3b_dp_ema_v2_skip7_s42` (= mốc B của 85f / 86b_B), epoch 5 (Tuấn khoá 28/10), trên Drive `cxrlt_2024/stage4_nb78x/<lượt>/`:
  rs_J_bert_dong_ep5.pt   'bb' (RAD-DINO tinh chỉnh, trọng số EMA) · 'luong' (luồng vùng) · 'bo_o7' + 'dua_o' (skip block 7 từ 5 ô) · 'run', 'luot', 'ep', 'ema'…
  backbone_ft.pt          cấu hình: cfg_luong (79) · cfg_jv2 (neo theo view) · cfg_jv3 (hệ số ghi ngược g) · cfg_jv3b (trần ghi ngược rho) · cfg_d4 · cfg_o · hop_tb
Mã forward = encoders/b_src.py (chép nguyên văn các tệp khối của 78 / 79 mà 85f / 86b_B nhúng vào ô). Dựng y như ô C2 / C2b / C6S / C7B của 86b_B:

  ảnh 518 (chuẩn hoá gốc của RAD-DINO) -> token_o7: 5 ô phóng từ chính ảnh 518 qua block 0..7 -> skip_o (BoChinhO + DuaVao) = inj
                                        -> block 0..7 của ảnh toàn cảnh, + inj vào token patch (cong_skip)                  = h8 (shared_lo)
  block 8..11, sau MỖI block một bước luồng vùng (LopJCANeoTran: JCA có neo + ghi ngược nhân g + trần rho theo ảnh); neo = 1 (ảnh thẳng)
  -> n tầng cuối = đầu ra block 8..11 SAU bước luồng vùng, qua layernorm cuối; CLS = token 0 (vector toàn ảnh), patch = token 1.. (1369)

Giả định (ghi rõ): mọi ảnh VinDr là ảnh thẳng -> neo = (cfg_jv2['view_neo'] == 'frontal'), như NEO_B86 của 86b_B. Chỉ dùng cú pháp / API có ở Python 3.9 +
torch 2.0."""
import torch

from ..checks import safe_load, weights
from . import b_src as S
from . import files
from .base import EncoderWrapper
from .raddino import RENAME, REPO, REVISION
from .spec import IMAGENET_MEAN, IMAGENET_STD, SPECS

RUN, LUOT, EPOCH = 'r78x_J_v3b_dp_ema_v2_skip7_s42', 'B1', 5
CFG_FILE = 'backbone_ft.pt'
# cfg_o mong đợi = O_CFG_MONG của 85f / 86b_B (skip7, 5 ô, skip vào block 8, adapter r 64, khúc 240 ô)
O_CFG = {'ten': 'o_luoi_phong518_skip7', 'phien_ban': 'o_j_v1', 'n_o': 5, 'k_skip': 8, 'r_adapter': 64, 'khuc': 240, 'hat': {'bo_o7': 61},
         'sang_loc': {'moc_run': ['stage4_nb78x', 'r78x_J_v3b_dp_ema_v2_s42'], 'nguong': 0.003}}
# số đã in ở C2 / C2b của 86b_B (12/11) — cổng "đúng lượt": R 36 vùng, tầng 8–11, g 5, trần rho 0,1, ảnh thẳng = neo
EXPECT = dict(R=36, tang=(8, 9, 10, 11), g=5.0, rho=0.1, view_neo='frontal', luot_79a='J_bert_dong', luot_chon='J_bert_dong', d4='dp_ema')
PARTS = ('bb', 'luong', 'bo_o7', 'dua_o')


def _with(prefix, sd):
    return {f'{prefix}.{k}': v for k, v in sd.items()}


class B(EncoderWrapper):
    n_prefix = 1

    def __init__(self, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None, cfg_path=None, attn=None, view='frontal', revision=REVISION):
        spec = SPECS['b']
        super().__init__(spec, input_mean, input_std)
        import transformers
        path = path or files.weights_path(spec)
        cfg_path = cfg_path or files.weights_path(spec, CFG_FILE)
        extra = {n: (m, z) for n, m, z in spec.extra_files}
        # GLOBAL của pickle (tệp thật 15/11): rs_ chỉ lõi -> weights_only=True; backbone_ft.pt thêm torch.torch_version.TorchVersion (chuỗi phiên bản torch)
        ck, src = safe_load.load_checkpoint(path, expect_md5=spec.file_md5, expect_size=spec.file_size)
        cfj, src_cfg = safe_load.load_checkpoint(cfg_path, expect_md5=extra[CFG_FILE][0], expect_size=extra[CFG_FILE][1], allow=('torch_version',))
        # ---- cổng "đúng lượt" (= cổng của C2 / C2b của 86b_B) ----
        assert isinstance(ck, dict) and all(k in ck for k in PARTS) and ck.get('run') == RUN and ck.get('luot') == LUOT and ck.get('ep') == EPOCH - 1, (
            f"B: checkpoint KHONG phai {RUN} / luot {LUOT} / ep {EPOCH} (co {ck.get('run')}, {ck.get('luot')}, ep {ck.get('ep')}) hoac thieu {PARTS} -> DUNG")
        assert not any(k in ck for k in ('bo_o11', 'gop_o')), 'B: module o KHONG dung cau truc skip7 (co bo_o11 / gop_o) -> DUNG'
        assert isinstance(cfj, dict) and cfj.get('run') == RUN and cfj.get('luot_79a') == EXPECT['luot_79a'] and all(
            isinstance(cfj.get(k), dict) for k in ('cfg_luong', 'cfg_jv2', 'cfg_jv3', 'cfg_jv3b', 'cfg_d4', 'cfg_o')), (
            f'B: {CFG_FILE} khong phai cau hinh cua {RUN} (run / luot_79a / cfg_*) -> DUNG')
        c79, cv2, cv3, cv3b, cd4, co = (cfj[k] for k in ('cfg_luong', 'cfg_jv2', 'cfg_jv3', 'cfg_jv3b', 'cfg_d4', 'cfg_o'))
        hop = cfj['hop_tb'].float()
        ema = ck.get('ema')
        assert c79.get('luot_chon') == EXPECT['luot_chon'] and torch.equal(hop, ck['luong']['hop_tb'].float().cpu()), (
            f'B: hop_tb / luot_chon cua {CFG_FILE} KHAC checkpoint -> DUNG')
        assert cd4.get('ten') == EXPECT['d4'] and cd4.get('khe') is None and (cd4.get('trong_so') == 'EMA') == isinstance(ema, dict), (
            f'B: cfg_d4 {cd4} KHAC luot {EXPECT["d4"]} / trong so -> DUNG')
        assert co == O_CFG, f'B: cfg_o {co} KHAC cau hinh skip7 mong doi -> DUNG'
        # ---- backbone RAD-DINO (revision ghim của lớp bọc RAD-DINO) + trọng số tinh chỉnh 'bb' (đổi tên khoá theo bản transformers, như C2 của 86b_B) ----
        kw = dict(revision=revision)
        if attn:
            kw['attn_implementation'] = attn
        self.model = transformers.AutoModel.from_pretrained(REPO, **kw)
        used = getattr(self.model.config, '_attn_implementation', None)
        assert not attn or used == attn, f'B: attention {used!r} != yeu cau {attn!r} -> DUNG'
        sd0 = {k: v.clone() for k, v in self.model.state_dict().items()}
        bb = S.doi_ten_dinov2(ck['bb'], sd0)
        self.model.load_state_dict(bb, strict=True)
        lech_pre = max(float((sd0[k].float() - v.float()).abs().max()) for k, v in self.model.state_dict().items())
        assert lech_pre > 1e-6, 'B: backbone GIONG HET RAD-DINO goc -> trong so tinh chinh KHONG duoc nap -> DUNG'
        del sd0
        cfg = self.model.config
        assert (cfg.patch_size, cfg.hidden_size) == (spec.patch_size, spec.embed_dim), (cfg.patch_size, cfg.hidden_size)
        # ---- luồng vùng J v3b (= C2b của 86b_B): LuongVungNeo -> hệ số ghi ngược g -> trần rho -> nạp chặt ----
        luot = c79['luot'][c79['luot_chon']]
        grid = spec.image_size // spec.patch_size
        with torch.random.fork_rng(devices=[]):                    # khởi tạo (ghi đè ngay khi nạp) không tiêu dòng ngẫu nhiên chung
            torch.manual_seed(0)
            self.luong = S.LuongVungNeo(hop.numpy(), d=spec.embed_dim, R=int(hop.shape[0]), tang=tuple(c79['tang']), n_head=int(c79['n_head']),
                                        ff=int(c79['ff']), beta=float(c79['beta']), t_mem=float(c79['t_mem']), hai_chieu=bool(luot['hai_chieu']),
                                        neo=bool(luot['neo']), luoi=grid, ckpt=False)
            n_g = S.tang_ghi_nguoc(self.luong, float(cv3['ghi_nguoc_g']))
            n_t = S.ap_tran_ghi(self.luong, float(cv3b['tran_ghi_rho']))
            self.bo_o7, self.dua_o = S.BoChinhO(spec.embed_dim, int(co['r_adapter'])), S.DuaVao(spec.embed_dim)
        self.luong.load_state_dict(ck['luong'], strict=True)
        self.bo_o7.load_state_dict(ck['bo_o7'], strict=True)
        self.dua_o.load_state_dict(ck['dua_o'], strict=True)
        self.k_skip, self.khuc, self.n_o = int(co['k_skip']), int(co['khuc']), int(co['n_o'])
        L = len(self.model.encoder.layer)
        assert (self.luong.R, tuple(self.luong.tang), float(cv3['ghi_nguoc_g']), float(cv3b['tran_ghi_rho'])) == (
            EXPECT['R'], EXPECT['tang'], EXPECT['g'], EXPECT['rho']) and n_g == n_t == len(self.luong.tang), (
            f'B: luong vung R {self.luong.R} / tang {self.luong.tang} / g / rho KHAC lan chay 86b_B -> DUNG')
        assert self.k_skip == self.luong.tang[0] and list(self.luong.tang) == list(range(self.k_skip, L)), (
            f'B: skip vao block {self.k_skip} / luong tang {self.luong.tang} KHONG lien tuc toi block cuoi {L - 1} -> DUNG')
        assert str(cv2.get('view_neo')) == EXPECT['view_neo'], f"B: view_neo {cv2.get('view_neo')} KHAC 'frontal' -> DUNG"
        self.neo = float(str(cv2['view_neo']) == str(view))
        self.mean0, self.std0 = float(spec.mean[0]), float(spec.std[0])
        self.register_buffer('luoi_o', S.luoi_518('cpu'), persistent=False)
        g, w = S.luoi_con(S.hinh_luoi(1, 'cpu'))
        self.register_buffer('g_o', g, persistent=False)
        self.register_buffer('w_o', w, persistent=False)
        assert self.luoi_o.shape[0] == self.n_o == g.shape[1], (tuple(self.luoi_o.shape), self.n_o, tuple(g.shape))
        for m in (self.model, self.luong, self.bo_o7, self.dua_o):            # như 85f / 86b_B: đông cứng, requires_grad False
            m.eval().requires_grad_(False)
        state = {**_with('bb', self.model.state_dict()), **_with('luong', self.luong.state_dict()), **_with('bo_o7', self.bo_o7.state_dict()),
                 **_with('dua_o', self.dua_o.state_dict())}
        filet = {**_with('bb', ck['bb']), **_with('luong', ck['luong']), **_with('bo_o7', ck['bo_o7']), **_with('dua_o', ck['dua_o'])}
        self.nap = weights.record(source=f'cxrlt_2024/stage4_nb78x/{RUN} (luot 78x, ep {EPOCH}) + {CFG_FILE}', file_name=spec.drive_file,
                                  library=f'transformers {transformers.__version__} ({REPO}@{revision[:12]}) + ma khoi 78 / 79 (encoders/b_src.py)',
                                  model_state=state, file_tensors=filet, lib_info=None, rename=RENAME, nguon_tep=src, nguon_cau_hinh=src_cfg,
                                  attention=used, lech_rad_dino_goc=lech_pre, trong_so=cd4.get('trong_so'),
                                  ema=({k: ema.get(k) for k in ('n', 'he_so', 'trong_so')} if isinstance(ema, dict) else None),
                                  luong=dict(R=self.luong.R, tang=list(self.luong.tang), g=float(cv3['ghi_nguoc_g']), rho=float(cv3b['tran_ghi_rho']),
                                             neo=self.neo, view_neo=str(cv2['view_neo'])), skip=dict(n_o=self.n_o, k_skip=self.k_skip, khuc=self.khuc),
                                  khoa_khong_dung=sorted(k for k in ck if k not in PARTS))
        self.train_log = ck.get('log')                                 # nhật ký huấn luyện (tỉ lệ ghi ngược từng lớp ở ep cuối: cổng của bộ kiểm)
        del ck, cfj

    def _blocks(self, x, n):
        '''x = ảnh 518 chuẩn hoá gốc -> n tầng cuối (đầu ra block 8..11 sau bước luồng vùng) qua layernorm, [B, 1 + 1369, 768].'''
        m, B = self.model, int(x.shape[0])
        layers = m.encoder.layer
        t7 = S.token_o7(m, x, self.luoi_o, self.mean0, self.std0, self.khuc, self.k_skip - 1)
        inj = S.skip_o(t7, self.bo_o7, self.dua_o, self.g_o.expand(B, -1, -1, -1).contiguous(), self.w_o.expand(B, -1, -1).contiguous())
        del t7
        h = m.embeddings(x)
        for i in range(0, self.k_skip):                                # = run_blocks(embeddings(img), 0, GRAD_FROM) của C2
            h = S._out_luong(layers[i](h))
        h = S.cong_skip(h, inj)                                        # = shared_lo của C6S
        self.luong.dat_neo(torch.full((B,), self.neo, device=x.device))
        st = self.luong.bat_dau(B)
        outs = []
        for l in range(self.luong.tang[0], len(layers)):               # = vòng của upper_luong_hr / vindr_protocol.features.b_luong
            h = S._out_luong(layers[l](h))
            h = self.luong.buoc(l, h, st, True)
            outs.append(h)
        assert len(outs) >= n, f'B: luong vung tu block {self.luong.tang[0]}: chi {len(outs)} tang < {n} -> DUNG'
        return [m.layernorm(o) for o in outs[-n:]]


def build(input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None, cfg_path=None, attn=None):
    return B(input_mean, input_std, path=path, cfg_path=cfg_path, attn=attn).eval()
