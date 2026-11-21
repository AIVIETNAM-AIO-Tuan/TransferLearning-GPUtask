# -*- coding: utf-8 -*-
"""Lớp bọc ĐÔNG CỨNG cho sáu hàng của PLAN_92 §13 (backbone 78x CÓ / KHÔNG luồng vùng trên DINOv3, CheXFound, BiomedCLIP) theo giao diện chung
vindr_bench.encoders.base.EncoderWrapper (như lớp bọc RAD-DINO vt.py của 89). Không học, không đổi trọng số / phép tính.

Dựng = ĐÚNG đường của 78r_dac_trung_bb_* (đã chạy 20–21/11):
  1. tệp: rs_<luot>_ep<E>.pt (+ backbone_ft.pt ở hàng CÓ) md5 ghim (rows.py); cổng "đúng lượt" = cổng C2 / C2b của 78r BB (run, luot, ep, trọng số EMA,
     số lần cập nhật EMA, cfg_d4 / cfg_p92 hàng CÓ, cfg_kl hàng KHÔNG, không module ô cắt);
  2. encoder TIỀN HUẤN LUYỆN như bộ nạp của lượt học (dung_78x_bb.TAI_BB_KHUON: DINOv3 HF gated + HF_TOKEN; CheXFound teacher_checkpoint.pth của
     cxrlt_2024/chexfound (md5 ghim ở vindr_bench SPECS['chexfound']); BiomedCLIP HF, tháp ảnh timm, pos_embed nội suy 14 -> 32) -> MatNaBB (bb_src, chép
     nguyên văn khoi_bb_khac); cổng MẶT NẠ = forward gốc TỪNG BIT (như C2 của 78r BB);
  3. nạp rs_['bb'] CHẶT (khoá mặt nạ, không đổi tên); lệch so với tiền huấn luyện (3 chữ số) = số 78r đã in ("lech pretrain");
  4. hàng CÓ: luồng vùng = C2b của 78r BB: LuongVungNeo(cfg_luong) -> tang_ghi_nguoc(g) -> ap_tran_ghi(rho) -> ap_n_dau(P) khi P > 1 -> nạp chặt;
  5. forward = upper_frozen của 78r: embeddings -> block 0 .. tang[0] - 1 -> (CÓ) mỗi block tang[0] .. cuối một bước luồng vùng CÓ ghi ngược (neo = 1:
     ảnh thẳng) / (KHÔNG) block thường -> layernorm của mặt nạ (bỏ register) — token patch = [:, 1:]; vector toàn ảnh = CLS (spec.pool 'cls').
Cổng lúc dựng (CPU fp32, 2 ảnh tổng hợp): CÓ — 4 tầng của _blocks = upper_luong_hr (bản chép nguyên văn khoi_doc_dong_cung của vindr_bench.b_src, hàm
78r gọi) TỪNG BIT, luồng chạy đủ 4 lớp, tỉ lệ ghi ngược hữu hạn và <= rho; KHÔNG — tầng cuối của _blocks = forward đầy đủ của mặt nạ TỪNG BIT.

Ảnh vào lớp bọc: 518, chuẩn hoá của bên gọi (ImageNet, biến đổi của 89) -> to_native -> chuẩn hoá RAD-DINO (đường 518 của lượt học) -> mặt nạ tự đổi
sang chuẩn hoá của encoder và thu về 512 (bilinear, antialias) như lúc học. Giả định (ghi rõ): mọi ảnh VinDr là ảnh thẳng -> neo = 1, như lớp bọc RAD-DINO.
CheXFound: config.yaml của họ KHÔNG đọc (ở lượt học chỉ dùng để đối chiếu với hình tensor; kiến trúc suy từ checkpoint — cau_hinh_dinov2_goc)."""
import dataclasses
import os

import torch

from vindr_bench.checks import safe_load, weights
from vindr_bench.encoders import b_src as S
from vindr_bench.encoders import files as W
from vindr_bench.encoders.base import EncoderWrapper
from vindr_bench.encoders.spec import IMAGENET_MEAN, IMAGENET_STD

from . import bb_src as K
from .rows import CFG_FILE, CROP_KEYS, LUOT, LUOT_J, PRE, RAD_DINO, ROWS, SIZE

ALLOW = ('torch_version', 'numpy_scalar')     # GLOBAL dữ liệu cho phép trong pickle của rs_ / backbone_ft.pt (lớp dữ liệu, không lớp mô hình)
GATE_IMAGES, GATE_SEED = 2, 19                # cổng lúc dựng: ảnh tổng hợp N(0, 1) ở chuẩn hoá gốc, generator riêng
MASK_SEED = 1                                 # cổng mặt nạ = forward gốc: ảnh như C2 của 78r BB (randn 1 x 3 x 518 x 518, hạt 1)
RATIO_TOL = 1e-4                              # tỉ lệ ghi ngược <= rho + RATIO_TOL (trần theo ảnh -> tỉ lệ cả lô <= rho)


def _with(prefix, sd):
    return {f'{prefix}.{k}': v for k, v in sd.items()}


def hf_token():
    try:
        from google.colab import userdata
        return userdata.get('HF_TOKEN')
    except Exception:
        return os.environ.get('HF_TOKEN')


# ------------------------------------------------------------------------------------------------ 1. tệp + cổng "đúng lượt"
def load_files(spec, path=None, cfg_path=None):
    '''-> (ck, nguồn ck, cfj | None, nguồn cfj | None, cổng {tên: giá trị đã kiểm}). Trượt -> AssertionError (dừng).'''
    path = path or W.weights_path(spec)
    ck, src = safe_load.load_checkpoint(path, expect_md5=spec.file_md5, allow=ALLOW)
    co = spec.hang == 'co'
    assert isinstance(ck, dict) and isinstance(ck.get('bb'), dict) and ck.get('run') == spec.run and ck.get('luot') == LUOT \
        and ck.get('ep') == spec.ep_bb - 1 and (('luong' in ck) == co), (
        f"{spec.name}: {spec.drive_file} KHONG phai {spec.run} / luot {LUOT} / ep {spec.ep_bb} / {'co' if co else 'khong'} luong (co {ck.get('run')}, "
        f"{ck.get('luot')}, ep {ck.get('ep')}, luong {'luong' in ck}) -> DUNG")
    assert not any(k in ck for k in CROP_KEYS), f'{spec.name}: checkpoint co module o cat {[k for k in CROP_KEYS if k in ck]} -> DUNG'
    ema = ck.get('ema')
    assert isinstance(ema, dict) and ema.get('trong_so') == 'EMA' and int(ema.get('n', 0)) == spec.ema_n, (
        f"{spec.name}: rs_ KHONG phai trong so EMA {spec.ema_n} lan cap nhat ({ema}) -> DUNG")
    cong = dict(run=ck['run'], luot=ck['luot'], ep=int(ck['ep']) + 1, ema_n=int(ema['n']), ema_he_so=float(ema.get('he_so', float('nan'))),
                tf_luu=ck.get('tf_ver'))
    if not co:                                                         # = CONG_KL_R của dung_78r_bb
        kl = ck.get('cfg_kl')
        assert 'cfg_kl' in ck and isinstance(kl, dict) and kl.get('ten') == 'b1_dp_ema_bb' and kl.get('encoder') == spec.encoder \
            and int(kl.get('ep_bb', -1)) == spec.ep_bb and int(kl.get('epochs', -1)) == spec.epochs and kl.get('luong') is None \
            and kl.get('trong_so') == 'EMA', (
            f"{spec.name}: cfg_kl {kl} KHAC hang KHONG encoder {spec.encoder} (b1_dp_ema_bb, ep_bb {spec.ep_bb}, {spec.epochs} epoch, trong so EMA) -> DUNG")
        cong.update(cfg_kl=dict(ten=kl['ten'], encoder=kl['encoder'], ep_bb=int(kl['ep_bb']), epochs=int(kl['epochs'])))
        return ck, src, None, None, cong
    assert isinstance(ck.get('luong'), dict) and 'cfg_kl' not in ck, f'{spec.name}: checkpoint thieu luong vung / co cfg_kl -> DUNG'
    cfg_path = cfg_path or W.weights_path(spec, CFG_FILE)
    cfj, src_cfg = safe_load.load_checkpoint(cfg_path, expect_md5=spec.cfg_md5, allow=ALLOW)
    assert isinstance(cfj, dict) and cfj.get('run') == spec.run and cfj.get('luot_79a') == LUOT_J and all(
        isinstance(cfj.get(k), dict) for k in ('cfg_luong', 'cfg_jv2', 'cfg_jv3', 'cfg_jv3b', 'cfg_d4')), (
        f"{spec.name}: {CFG_FILE} khong phai cau hinh J v3b cua {spec.run} (run / luot_79a / cfg_luong, cfg_jv2, cfg_jv3, cfg_jv3b, cfg_d4) -> DUNG")
    c79, cd4 = cfj['cfg_luong'], cfj['cfg_d4']
    assert c79.get('luot_chon') == LUOT_J and torch.equal(cfj['hop_tb'].float(), ck['luong']['hop_tb'].float().cpu()), (
        f'{spec.name}: hop_tb / luot_chon cua {CFG_FILE} KHAC rs_ -> DUNG')
    assert cd4.get('ten') == 'dp_ema' and int(cd4.get('ep_bb', -1)) == spec.ep_bb and int(cd4.get('epochs', -1)) == spec.epochs and cd4.get('khe') is None \
        and not any(k.startswith('khe.') for k in ck['luong']) and cd4.get('trong_so') == 'EMA', (
        f"{spec.name}: cfg_d4 {cd4} KHAC dp_ema / ep_bb {spec.ep_bb} / {spec.epochs} epoch / khong khe / trong so EMA -> DUNG")
    g, rho = float(cfj['cfg_jv3']['ghi_nguoc_g']), float(cfj['cfg_jv3b']['tran_ghi_rho'])
    assert (g, rho) == (spec.g, spec.rho), f'{spec.name}: ghi nguoc x{g:g} / tran {rho:g} KHAC hang (x{spec.g:g} / {spec.rho:g}) -> DUNG'
    p92 = cfj.get('cfg_p92')
    if spec.khoa_p92 is None:                                          # lượt tháng 10 (DINOv3 CÓ): không có cfg_p92
        assert p92 is None, f'{spec.name}: backbone_ft.pt co cfg_p92 {p92} nhung hang la luot thang 10 -> DUNG'
    else:                                                              # = CONG_P92_R của dung_78r_bb
        assert isinstance(p92, dict) and p92.get('khoa') == spec.khoa_p92 and p92.get('encoder') == spec.encoder and float(p92.get('g', -1)) == g \
            and float(p92.get('tran_rho', -1)) == rho, f'{spec.name}: cfg_p92 {p92} KHAC hang {spec.khoa_p92} / {spec.encoder} / x{g:g} / {rho:g} -> DUNG'
    cong.update(cfg_d4=dict(ten=cd4['ten'], ep_bb=int(cd4['ep_bb']), epochs=int(cd4['epochs'])), g=g, rho=rho,
                cfg_p92=None if p92 is None else dict(khoa=p92['khoa'], encoder=p92['encoder'], g=float(p92['g']), tran_rho=float(p92['tran_rho'])))
    return ck, src, cfj, src_cfg, cong


# ------------------------------------------------------------------------------------------------ 2. encoder tiền huấn luyện (= TAI_BB_KHUON)
def pretrained(encoder, chexfound_path=None, token=None):
    '''-> (MatNaBB với trọng số TIỀN HUẤN LUYỆN (CPU, float32, eval, còn tham chiếu mô hình gốc cho cổng mặt nạ), dict nguồn). = tao_bb_khac(MEAN, STD)
       của lượt học (dung_78x_bb.TAI_BB_KHUON) với MEAN / STD = chuẩn hoá RAD-DINO. Tạo trong fork_rng (CPU, hạt 0) như lượt học.'''
    cfg = PRE[encoder]
    m_in, s_in = RAD_DINO
    S_ = int(cfg['S'])
    src = dict(encoder=encoder)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        if encoder == 'dinov3':
            from transformers import AutoImageProcessor, DINOv3ViTModel
            tok = token or hf_token()
            assert tok, 'DINOv3 la mo hinh gated: chap nhan giay phep tren Hugging Face, tao token, them HF_TOKEN vao Colab Secrets -> DUNG'
            pr = AutoImageProcessor.from_pretrained(cfg['hf'], token=tok)
            goc = DINOv3ViTModel.from_pretrained(cfg['hf'], token=tok, drop_path_rate=0.0, attention_dropout=0.0).float().eval()
            mb = K.tao_dinov3(goc, m_in, s_in, list(pr.image_mean), list(pr.image_std), S_)
            src.update(nguon=f"HF {cfg['hf']}", chuan_hoa_encoder=[list(pr.image_mean), list(pr.image_std)])
        elif encoder == 'chexfound':
            from transformers import Dinov2Config, Dinov2Model, Dinov2WithRegistersConfig, Dinov2WithRegistersModel
            from vindr_bench.encoders import SPECS
            sp = SPECS['chexfound']
            sd, s0 = safe_load.load_checkpoint(chexfound_path or W.weights_path(sp), expect_md5=sp.file_md5, expect_size=sp.file_size)
            cf = K.cau_hinh_dinov2_goc(sd)
            kw = dict(hidden_size=cf['d'], num_hidden_layers=cf['L'], num_attention_heads=cf['n_head'], mlp_ratio=cf['mlp_ratio'],
                      patch_size=cf['patch'], image_size=cf['luoi'] * cf['patch'], qkv_bias=cf['qkv_bias'], use_swiglu_ffn=cf['swiglu'],
                      drop_path_rate=0.0, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, layer_norm_eps=1e-6)
            goc = (Dinov2WithRegistersModel(Dinov2WithRegistersConfig(num_register_tokens=cf['n_reg'], **kw)) if cf['n_reg']
                   else Dinov2Model(Dinov2Config(**kw)))
            goc.load_state_dict(K.doi_khoa_dinov2_goc(sd, goc.state_dict().keys()), strict=True)
            del sd
            mb = K.tao_dinov2_hf(goc.float().eval(), m_in, s_in, cfg['mean'], cfg['std'], S_, ten='chexfound')
            src.update(nguon=f"{sp.drive_dir}/{sp.drive_file}", md5=s0.get('md5'), cau_hinh=cf, chuan_hoa_encoder=[list(cfg['mean']), list(cfg['std'])])
        elif encoder == 'biomedclip':
            import timm
            from huggingface_hub import hf_hub_download, list_repo_files
            fs = list_repo_files(cfg['hf'])
            f = next((x for x in ('open_clip_model.safetensors', 'open_clip_pytorch_model.bin') if x in fs), None)
            assert f, f"{cfg['hf']}: khong thay tep trong so open_clip ({fs[:8]}) -> DUNG"
            p = hf_hub_download(cfg['hf'], f)
            if f.endswith('.safetensors'):
                from safetensors.torch import load_file
                sd = load_file(p)
            else:
                sd = torch.load(p, map_location='cpu', weights_only=True)
            tr = {k[len('visual.trunk.'):]: v for k, v in sd.items() if k.startswith('visual.trunk.')}
            assert tr, f'{f}: khong co khoa visual.trunk.* -> DUNG'
            tr['pos_embed'] = K.noi_suy_pos(tr['pos_embed'], S_ // 16, 1)
            trunk = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0, img_size=S_)
            trunk.load_state_dict(tr, strict=True)
            del sd, tr
            mb = K.tao_timm_vit(trunk.float().eval(), m_in, s_in, cfg['mean'], cfg['std'], S_, ten='biomedclip')
            src.update(nguon=f"HF {cfg['hf']} / {f}", timm=timm.__version__, chuan_hoa_encoder=[list(cfg['mean']), list(cfg['std'])])
        else:
            raise AssertionError(f'encoder {encoder} la -> DUNG')
    assert mb.luoi == SIZE // mb.config.patch_size, f'luoi mat na {mb.luoi} != {SIZE} // patch ({mb.config.patch_size}) -> DUNG'
    import transformers
    src['transformers'] = transformers.__version__
    return mb.eval(), src


@torch.no_grad()
def mask_gate(mb):
    '''Cổng C2 của 78r BB: forward mặt nạ = forward GỐC của encoder trên ảnh randn(1, 3, 518, 518) hạt 1 — lệch phải 0,0. -> lệch.'''
    x = torch.randn(1, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(MASK_SEED))
    return float((mb(pixel_values=x).last_hidden_state - K.so_goc(mb, x)).abs().max().item())


# ------------------------------------------------------------------------------------------------ 4. luồng vùng (= C2b của 78r BB)
def build_luong(spec, cfj, luong_sd, mb):
    '''-> (luồng ở eval, không grad; dict cấu hình đã dựng).'''
    c79, cv3, cv3b = cfj['cfg_luong'], cfj['cfg_jv3'], cfj['cfg_jv3b']
    luot = c79['luot'][c79['luot_chon']]
    hop = cfj['hop_tb'].float()
    with torch.random.fork_rng(devices=[]):                           # khởi tạo (ghi đè ngay khi nạp) không tiêu dòng ngẫu nhiên chung
        torch.manual_seed(0)
        lu = S.LuongVungNeo(hop.numpy(), d=int(mb.config.hidden_size), R=int(hop.shape[0]), tang=tuple(c79['tang']), n_head=int(c79['n_head']), ff=int(c79['ff']),
                            beta=float(c79['beta']), t_mem=float(c79['t_mem']), hai_chieu=bool(luot['hai_chieu']), neo=bool(luot['neo']), luoi=mb.luoi,
                            ckpt=False)
    n_g = S.tang_ghi_nguoc(lu, float(cv3['ghi_nguoc_g']))
    n_t = S.ap_tran_ghi(lu, float(cv3b['tran_ghi_rho']))
    assert n_t == n_g and all(type(l) is S.LopJCANeoTran and l.rho == float(cv3b['tran_ghi_rho']) for l in lu.lop), 'J v3b: tran chua gan du lop JCA -> DUNG'
    n_p = K.ap_n_dau(lu, mb.n_dau) if mb.n_dau > 1 else 0
    lu.load_state_dict(luong_sd, strict=True)
    L = len(mb.encoder.layer)
    assert lu.R == 36 and tuple(lu.tang) == spec.tang == tuple(range(L - 4, L)) and n_g == len(lu.tang) and bool(lu.hai_chieu) \
        and (n_p == (len(lu.lop) if mb.n_dau > 1 else 0)), (
        f'{spec.name}: luong R {lu.R} / tang {lu.tang} (can {spec.tang}, block cuoi {L - 1}) / so lop doi {n_g}, {n_t}, {n_p} / hai_chieu {lu.hai_chieu} -> DUNG')
    return lu.eval().requires_grad_(False), dict(R=int(lu.R), tang=list(lu.tang), n_head=int(c79['n_head']), ff=int(c79['ff']), g=float(cv3['ghi_nguoc_g']),
                                                 rho=float(cv3b['tran_ghi_rho']), hai_chieu=bool(lu.hai_chieu), n_dau=int(mb.n_dau), so_lop_doi_P=int(n_p),
                                                 lop=type(lu.lop[0]).__name__)


# ------------------------------------------------------------------------------------------------ lớp bọc
class BBRow(EncoderWrapper):
    '''Một trong sáu hàng (rows.ROWS). spec= / pre= chỉ cho bộ kiểm (spec thay thế; (MatNaBB tiền huấn luyện, nguồn) dựng sẵn).'''
    n_prefix = 1

    def __init__(self, name, input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD, path=None, cfg_path=None, chexfound_path=None, token=None, view='frontal',
                 spec=None, pre=None):
        spec = spec or ROWS[name]
        assert spec.name == name, (spec.name, name)
        super().__init__(spec, input_mean, input_std)
        ck, src, cfj, src_cfg, cong = load_files(spec, path, cfg_path)
        mb, src_pre = pre if pre is not None else pretrained(spec.encoder, chexfound_path, token)
        lech_mask = mask_gate(mb)
        assert lech_mask == 0.0, f'{name}: mat na {mb.ten} KHAC forward goc cua encoder (lech {lech_mask:.2e}) -> DUNG'
        K.bo_goc(mb)
        sd0 = {k: v.clone() for k, v in mb.state_dict().items()}
        assert set(ck['bb']) == set(sd0), f"{name}: khoa backbone lech mat na: {sorted(set(ck['bb']) ^ set(sd0))[:4]} -> DUNG"
        mb.load_state_dict(dict(ck['bb']), strict=True)
        with torch.no_grad():
            lech = max((sd0[k] - v).abs().max().item() for k, v in mb.state_dict().items())
        del sd0
        assert f'{lech:.3f}' == spec.lech_pre, f'{name}: lech so voi tien huan luyen {lech:.3f} KHAC so 78r da in ({spec.lech_pre}) -> DUNG'
        self.model = mb.eval().requires_grad_(False)
        assert int(mb.config.hidden_size) == int(spec.embed_dim), f'{name}: chieu token {mb.config.hidden_size} != {spec.embed_dim} -> DUNG'
        L = len(self.model.encoder.layer)
        assert spec.tang == tuple(range(L - 4, L)), f'{name}: tang {spec.tang} khong phai 4 block cuoi cua {L} block -> DUNG'
        self.grad_from = spec.tang[0]
        self.luong, info, self.neo = None, None, None
        if spec.hang == 'co':
            self.luong, info = build_luong(spec, cfj, ck['luong'], self.model)
            self.neo = float(str(cfj['cfg_jv2']['view_neo']) == str(view))
            info['view_neo'] = str(cfj['cfg_jv2']['view_neo'])
        log_ep = ((ck.get('log') or {}).get(str(spec.ep_bb - 1)) or {})
        self.ghi_log = (log_ep.get('luong') or {}).get('ghi_tang')
        self.val_log = log_ep.get('mAP_anh')
        state = _with('bb', self.model.state_dict())
        filet = _with('bb', ck['bb'])
        if self.luong is not None:
            state.update(_with('luong', self.luong.state_dict()))
            filet.update(_with('luong', ck['luong']))
        self.nap = weights.record(source=f"{spec.drive_dir} (luot 78x, ep {spec.ep_bb}, trong so EMA)" + (f' + {CFG_FILE}' if cfj is not None else ''),
                                  file_name=spec.drive_file, library=f"transformers {src_pre.get('transformers')} + mat na khoi_bb_khac (frozen_probe.encoders.bb_src)"
                                  + (' + luong vung (vindr_bench.encoders.b_src)' if self.luong is not None else ''),
                                  model_state=state, file_tensors=filet, lib_info=None, nguon_tep=src, nguon_cau_hinh=src_cfg, trong_so='EMA', cong_luot=cong,
                                  tien_huan_luyen=src_pre, cong_mat_na=dict(lech=lech_mask, dat=True), lech_pre=dict(do=f'{lech:.3f}', ghim=spec.lech_pre),
                                  hang=spec.hang, luong=info, val_log=self.val_log, ghi_log=self.ghi_log,
                                  khoa_khong_dung=sorted(k for k in ck if k not in ('bb', 'luong')))
        chk = self.forward_check()
        self.nap['cong_forward'] = chk
        self.nap['dat'] = bool(self.nap['dat'] and chk['dat'])
        del ck, cfj

    def _prefix(self, x):
        '''embeddings -> block 0 .. grad_from - 1 (= shared_lo / run_blocks của 78r).'''
        L = self.model.encoder.layer
        h = self.model.embeddings(x)
        for i in range(0, self.grad_from):
            h = S._out_luong(L[i](h))
        return h

    def _run(self, x, n):
        '''-> (n tầng cuối qua layernorm của mặt nạ [B, 1 + T, D], trạng thái luồng | None). x = ảnh 518 ở chuẩn hoá gốc (RAD-DINO).'''
        L = self.model.encoder.layer
        h = self._prefix(x)
        outs, st = [], None
        if self.luong is None:
            for l in range(self.grad_from, len(L)):
                h = S._out_luong(L[l](h))
                outs.append(h)
        else:
            self.luong.dat_neo(torch.full((int(x.shape[0]),), float(self.neo), device=x.device))
            st = self.luong.bat_dau(int(x.shape[0]))
            for l in range(self.luong.tang[0], len(L)):
                h = S._out_luong(L[l](h))
                h = self.luong.buoc(l, h, st, True)
                outs.append(h)
        assert len(outs) >= n, f'{self.spec.name}: chi {len(outs)} tang < {n} -> DUNG'
        return [self.model.layernorm(o) for o in outs[-n:]], st

    def _blocks(self, x, n):
        return self._run(x, n)[0]

    @torch.no_grad()
    def write_back(self, x):
        '''Tỉ lệ ghi ngược ‖dh‖ / ‖h‖ từng lớp trên lô x (chuẩn hoá của bên gọi) -> list float (hàng KHÔNG: []).'''
        if self.luong is None:
            return []
        _, st = self._run(self.to_native(x), 1)
        return [float(t) for t in st['ghi']]

    @torch.no_grad()
    def forward_check(self):
        '''Cổng lúc dựng (CPU fp32, GATE_IMAGES ảnh tổng hợp ở chuẩn hoá gốc) -> dict (số đo trong tiến trình).'''
        dev = next(self.model.parameters()).device
        x = torch.randn(GATE_IMAGES, 3, SIZE, SIZE, generator=torch.Generator().manual_seed(GATE_SEED)).to(dev)
        a, st = self._run(x, 4)
        out = dict(so_anh=GATE_IMAGES, thiet_bi=str(dev), kieu=str(x.dtype).replace('torch.', ''))
        if self.luong is None:
            full = self.model(pixel_values=x).last_hidden_state
            bits = torch.equal(a[-1], full)
            out.update(dat=bool(bits), trung_bit_forward_day_du=bool(bits), lech_max=float((a[-1] - full).abs().max()))
            return out
        self.luong.dat_neo(torch.full((GATE_IMAGES,), float(self.neo), device=dev))
        p, r, ghi = S.upper_luong_hr(self.model, self.luong, self._prefix(x), mo_ghi=True)
        bits = torch.equal(a[-1][:, 1:].float(), p)
        ti = [float(t) for t in st['ghi']]
        ok_ti = len(ti) == len(self.luong.tang) and all(t == t and 0.0 < t <= float(self.spec.rho) + RATIO_TOL for t in ti)
        out.update(dat=bool(bits and len(st['hop_cac_tang']) == len(self.luong.tang) and ok_ti), trung_bit_upper_luong_hr=bool(bits),
                   lech_max=float((a[-1][:, 1:].float() - p).abs().max()), so_lop_jca_chay=len(st['hop_cac_tang']), ti_le_ghi=ti,
                   ti_le_ghi_upper_luong_hr=[float(t) for t in ghi], token_vung=list(r.shape))
        return out


def build(name, check=True, **kw):
    '''-> lớp bọc ở eval() (CPU) + bản ghi cổng (.nap). check=True: cổng trượt -> AssertionError.'''
    assert name in ROWS, f'{name}: khong thuoc 6 hang PLAN_92 13 {tuple(ROWS)} -> DUNG'
    enc = BBRow(name, **kw).eval()
    enc.nap['encoder'] = name
    if check:
        assert enc.nap['dat'], f"{name}: cong nap / forward KHONG dat: {weights.loading_text(enc.nap)} | forward {enc.nap['cong_forward']} -> DUNG"
    return enc


def replace_spec(name, **kw):
    '''Spec thay thế (bộ kiểm): dataclasses.replace của rows.ROWS[name].'''
    return dataclasses.replace(ROWS[name], **kw)
