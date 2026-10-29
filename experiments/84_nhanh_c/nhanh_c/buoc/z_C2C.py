# ==== nhanh_c buoc z_C2C — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C2c · 78z: STUDY + VIEW (luc cham) cua tung hang ORDER cho fusion + bang dem (CHI so) + generator bo view ====
# Luat chi-anh (docs/PLAN_SAU_78Y.md A.1b): fusion CHI dung study_id (KHONG subject_id) + view tu tep view LUC CHAM (CSV ViewPosition ->
# 28e; KHONG BigQuery) cho CA train va cham (cung mot nghia cua nhung view). Bat dong nhan giua cac anh cung study (train): CHI dem.
assert 'VIEW_CHAM_SRC' in globals() and 'CFG_FS' in globals() and 'GopStudy' in globals(), 'chay C0 (78z) truoc'
ST_ID = ORDER.study_id.to_numpy().astype('int64')
ST_FUS_T = torch.from_numpy(ST_ID).to(dev)
_vc = pd.read_parquet(VIEW_CHAM_SRC, columns=['dicom_id', 'view_group']).set_index('dicom_id').view_group
assert ORDER.dicom_id.isin(_vc.index).all(), 'tep view luc cham thieu anh cua ORDER -> DUNG'
_vg = _vc.reindex(ORDER.dicom_id.to_numpy()).astype(str).to_numpy()
VIEW_FUS = np.full(len(ORDER), int(CFG_FS['n_view']) - 1, 'int64')            # khac / unknown = ma cuoi
for _ten_v, _ma_v in CFG_FS['view_ma'].items(): VIEW_FUS[_vg == _ten_v] = int(_ma_v)
assert int(VIEW_FUS.max()) < int(CFG_FS['n_view']) and int(VIEW_FUS.min()) >= 0, 'ma view ngoai 0..n_view-1 -> DUNG'
VIEW_FUS_T = torch.from_numpy(VIEW_FUS).to(dev)
P_BO_VIEW, LO_DU_DOAN = float(CFG_FS['p_bo_view']), int(CFG_FS['lo_du_doan'])
GEN_FUS = torch.Generator(device=dev); GEN_FUS.manual_seed(SEEDS[0] * 1000 + int(CFG_FS['hat_gen_fus']))   # dong RIENG: CHI mat na bo view
for _ten_t, _ix_t in (('train', I_TF), ('val', I_VA), ('test', I_TE)):
    _u_t, _c_t = np.unique(ST_ID[_ix_t], return_counts=True)
    print(f"{_ten_t}: {len(_ix_t):,} anh / {len(_u_t):,} study | study nhieu anh {(_c_t > 1).mean():.1%} (anh thuoc study nhieu anh "
          f"{_c_t[_c_t > 1].sum() / max(len(_ix_t), 1):.1%}) | anh / study TB {_c_t.mean():.2f}, max {_c_t.max()} | view frontal / lateral / "
          'khac: ' + ' / '.join(f'{(VIEW_FUS[_ix_t] == k).mean():.1%}' for k in range(int(CFG_FS['n_view']))))
_us_t, _inv_t, _cnt_t = np.unique(ST_ID[I_TF], return_inverse=True, return_counts=True)
_yt = YM[I_TF]
_mx_t = np.full((len(_us_t), C), -np.inf, 'float32'); np.maximum.at(_mx_t, _inv_t, _yt)
_mn_t = np.full((len(_us_t), C), np.inf, 'float32'); np.minimum.at(_mn_t, _inv_t, _yt)
_bd_t = (_mx_t != _mn_t)[_cnt_t > 1]
print(f'bat dong nhan giua cac anh CUNG study (train, {int((_cnt_t > 1).sum()):,} study nhieu anh): study co >= 1 nhan khac '
      f'{_bd_t.any(1).mean():.2%} | nhieu nhat: ' + ', '.join(f'{LAB40[j]} {_bd_t[:, j].mean():.2%}' for j in np.argsort(-_bd_t.mean(0))[:5])
      + ' (loss tren TUNG anh voi logit chung cua study)')
print(f"78z C2c: fusion {CFG_FS['phien_ban']} | dau {CFG_FS['arms']} | bo view luc train p {P_BO_VIEW} (GEN_FUS rieng) | lo du doan <= "
      f'{LO_DU_DOAN} anh, gom tron study | view tu tep luc cham (md5 {MD5_VIEW_CHAM[:8]})')
del _vc, _vg, _us_t, _inv_t, _cnt_t, _yt, _mx_t, _mn_t, _bd_t
_KHOA_C2C = (THU_NHANH, RUN_TAG)   # C4 / C6 kiem: doi C0 thi PHAI chay lai C2c
