# ==== nhanh_c buoc p1_C9 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C9 · dong_bo() (ghi cuoi) — CHAY TRUOC KHI TAT RUNTIME + kiem ke: rs_ NHE ep1..EPOCHS (78z dung ep4) + backbone_ft.pt phai co, tep
#      resume da xoa ====
_ok9 = dong_bo()
_k9 = kiem_ke(DRV_DUONG)[1]
_can_rs9 = [f'rs_{LUOT_79A}_ep{e}.pt' for e in range(1, EPOCHS + 1)]
_can_rs9 += ([f'rs_tho_{LUOT_79A}_ep{EP_BB_D4}.pt'] if BIEN_THE_D4 == 'dp_ema' and EP_BB_D4 <= _EP_CHAY else [])   # 78x dot 4 B.4: trong so THO ep4
_thieu_rs9 = [t for t in _can_rs9 if t not in _k9]
_bb9 = _k9.get('backbone_ft.pt') or []
_kiem9 = False
if _bb9:
    try:
        _c9 = torch.load(tai(_bb9[0]), map_location='cpu', weights_only=False)
        _kiem9 = (isinstance(_c9, dict) and {'bb', 'head', 'run', 'luot', 'E', 'ep', 'tf_ver', 'meta', 'luong', 'cfg_luong', 'cfg_d4'} <= set(_c9)
                  and _c9['luot'] == LUOT_78U and _c9['run'] == RUN_TAG and _c9['cfg_d4'].get('ten') == BIEN_THE_D4 and int(_c9['E']) == EPOCHS
                  and any(k.startswith('khe.') for k in _c9['luong']) == (CFG_D4_CHON['khe'] is not None)
                  and (globals().get('KHOA_38A') is None or set(_c9['bb']) == set(KHOA_38A))
                  and all(bool(torch.isfinite(v).all()) for v in _c9['bb'].values() if v.is_floating_point()))
        del _c9
    except Exception as _e9:
        print(f'  !! khong doc duoc backbone_ft.pt ({type(_e9).__name__}: {_e9})')
_res9 = TEN_RESUME in _k9
print(f"78x dot 4 C9: dong_bo {'DU' if _ok9 else 'CHUA DU'} | backbone_ft.pt "
      + ('co, qua kiem (khoa, cfg_d4, khe dung luot)' if _kiem9 else 'THIEU / KHONG qua kiem')
      + f" | rs_ NHE: {len(_can_rs9) - len(_thieu_rs9)} / {len(_can_rs9)} (thieu {_thieu_rs9 or 'khong'}) | tep resume "
      + ('CON (dong_bo chua du?)' if _res9 else 'da xoa'))
if not THU_NHANH and not (_ok9 and _kiem9 and not _thieu_rs9 and not _res9):
    print('  !! THIEU tep can cho 78z hoac dong_bo chua du — KHONG tat runtime; chay lai C9, roi bao Claude')
# ---- B.4 v2 (mot notebook): phan 1 XONG (dong_bo du + backbone_ft.pt qua kiem + rs_ du + tep resume da xoa) -> moi duoc sang phan 2 ----
_OK_PHAN_1 = bool(_ok9 and _kiem9 and not _thieu_rs9 and not _res9 and not THU_NHANH)
print(f"B.4 v2: phan 1 {'XONG -> sang phan 2 (78z)' if _OK_PHAN_1 else 'CHUA XONG -> o chuyen se DUNG (khong sang phan 2)'}")
# ---- nhanh C (Tuan duyet 29/10; SPEC B.2ab): CONG SUC KHOE truoc phan 2 (thay sang loc so B.4 v2) - VAL trong luot (EMA, muc anh, nhat ky
#      C6) TB ep1-6 cua luot nay so C s42 (nhat ky stage4_nb78x/r78x_J_v3b_dp_ema_v2_o_s42, CHI doc qua API; TB tinh lai = moc +- 5e-4): thap
#      hon moc QUA nguong -> YEU (o chuyen phan dung; Tuan dat TIEP_TUC_C = True trong notebook roi chay lai o chuyen neu quyet chay tiep) ----
SUC_KHOE_C = None
if not THU_NHANH:
    _kk_sk = kiem_ke(tuple(CAU_HINH_C['moc_x']), mau=r'epoch_log_J_bert_dong_ep\d+\.json')[1].get(f'epoch_log_J_bert_dong_ep{EPOCHS}.json') or []
    assert _kk_sk, f"{'/'.join(CAU_HINH_C['moc_x'])}: thieu epoch_log_J_bert_dong_ep{EPOCHS}.json -> KHONG kiem suc khoe duoc, DUNG"
    with open(tai(_kk_sk[0]), encoding='utf-8') as _f_sk: _lg_sk = json.load(_f_sk)
    assert all(str(e) in _lg_sk and str(e) in LOG for e in range(EPOCHS)), 'cong suc khoe: nhat ky thieu epoch -> DUNG'
    _moc_sk = [float(_lg_sk[str(e)]['mAP_anh']) for e in range(EPOCHS)]
    _nay_sk = [float(LOG[str(e)]['mAP_anh']) for e in range(EPOCHS)]
    _mt_sk, _ng_sk = float(CAU_HINH_C['suc_khoe']['moc_tb']), float(CAU_HINH_C['suc_khoe']['nguong'])
    assert abs(float(np.mean(_moc_sk)) - _mt_sk) < 5e-4, f'cong: TB VAL C s42 tinh lai {np.mean(_moc_sk):.4f} KHAC moc {_mt_sk} -> DUNG'
    _ch_sk = float(np.mean(_nay_sk) - np.mean(_moc_sk))
    SUC_KHOE_C = dict(nay=_nay_sk, moc=_moc_sk, tb_nay=float(np.mean(_nay_sk)), tb_moc=float(np.mean(_moc_sk)), chenh=_ch_sk, nguong=_ng_sk,
                      ket_luan=('DAT' if _ch_sk >= -_ng_sk else 'YEU'), md5_moc=_kk_sk[0]['md5'], seed=SEED)
    ghi('suc_khoe_c.json', lambda q: ghi_json(q, SUC_KHOE_C))
    _ok_sk = dong_bo()
    print(f"nhanh C s{SEED} CONG SUC KHOE: VAL trong luot (EMA) TB ep1-{EPOCHS} {SUC_KHOE_C['tb_nay']:.4f} - C s42 {SUC_KHOE_C['tb_moc']:.4f} = "
          f"{_ch_sk:+.4f} -> {SUC_KHOE_C['ket_luan']} (>= -{_ng_sk} DAT; thap hon YEU) | tung epoch nay "
          + ' '.join(f'{v:.4f}' for v in _nay_sk) + ' / C s42 ' + ' '.join(f'{v:.4f}' for v in _moc_sk)
          + ('' if _ok_sk else ' | !! suc_khoe_c.json chua len Drive (chay lai C9)'))
    del _kk_sk, _f_sk, _lg_sk, _moc_sk, _nay_sk, _mt_sk, _ng_sk, _ch_sk, _ok_sk
