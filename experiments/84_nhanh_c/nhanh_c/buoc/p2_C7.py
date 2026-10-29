# ==== nhanh_c buoc p2_C7 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C7 · BANG 78z J v3b - VAL tung epoch tu nhat ky tren Drive (1 seed = SANG LOC). Chay duoc ngay sau C0 ====
# Doc epoch_log MOI NHAT cua seed QUA API -> du moi epoch ke ca khi C6 chay qua nhieu phien. Khong ghi gi, khong can GPU.
#   arm - mldec = tac dung cua phan khac (cung forward backbone, cung thu tu batch). Chenh < 0.003: chua phan biet voi nhieu seed.
#   Luat qua sang loc (docs/PLAN_SAU_78G.md 1.2: TB moi epoch, gop view, > 0 o CA HAI nua val va >= 0.003 toan val) tinh o 78a
#   bang 78a cua 78z J v3b - so o day CHI de theo doi (fusion DA gop study; dau khac CHUA gop view).
assert 'tim_moi_noi' in globals(), 'chay C0 truoc'
SK7 = f's{SEEDS[0]}'
_lp7 = tim_moi_noi(f'epoch_log_{SK7}_cuoi.json')
for _e7 in range(EPOCHS, 0, -1):
    if _lp7: break
    _lp7 = tim_moi_noi(f'epoch_log_{SK7}_ep{_e7}.json')
assert _lp7, f'chua co epoch_log nao cua {SK7} tren Drive cxrlt_2024/{"/".join(DRV_DUONG)} -> chay C6 truoc'
with open(_lp7) as _f7: L7 = json.load(_f7).get(SK7, {})
EP7 = sorted(int(k) for k in L7 if k.isdigit())
assert EP7, f'{os.path.basename(_lp7)}: khong co epoch nao cua {SK7}'
A7 = list(L7[str(EP7[0])]); DC7 = A7[0]
R7 = L7[str(EP7[-1])]
P7 = [t for t in A7 if 'lam' in R7[t]]
_f7f = lambda v, w=4: '-' if v is None else f'{v:.{w}f}'
print(f'78z J {RUN_TAG} {SK7}: {len(EP7)}/{EPOCHS} epoch trong {os.path.basename(_lp7)} | done={bool(L7.get("done"))}'
      + (f' | RESUME tai epoch {[e + 1 for e in L7["resume"]]}' if L7.get('resume') else ''))
_mat7 = [e + 1 for e in range(EP7[-1] + 1) if e not in EP7]
if _mat7: print(f'  !! nhat ky THIEU epoch {_mat7} -> mAP cac epoch do tinh lai duoc tu p_ tren Drive')
for _ten7, _k7 in (('VAL mAP_anh', 'mAP_anh'), ('VAL tail', 'tail')):
    print(f'\n{_ten7:<12}' + ''.join(f'{t:>12}' for t in A7))
    for e in EP7:
        r7 = L7[str(e)]
        print(f'  ep{e + 1:<9}' + ''.join(f'{r7[t][_k7]:>12.4f}' for t in A7))
    print(f'  {"- " + DC7 + " (cuoi)":<11}' + ''.join(f'{R7[t][_k7] - R7[DC7][_k7]:>+12.4f}' for t in A7))
    print(f'  {"- " + DC7 + " (TB ep)":<11}' + ''.join(
        f'{np.mean([L7[str(e)][t][_k7] - L7[str(e)][DC7][_k7] for e in EP7]):>+12.4f}' for t in A7))
print(f'\nDINH va CUOI (VAL mAP_anh; doi chung {DC7}):')
for t in A7:
    v7 = [L7[str(e)][t]['mAP_anh'] for e in EP7]; i7 = int(np.argmax(v7))
    print(f'  {t:<12} dinh ep{EP7[i7] + 1} {v7[i7]:.4f} | cuoi {v7[-1]:.4f} | cuoi - doi chung '
          f'{v7[-1] - R7[DC7]["mAP_anh"]:+.4f} | TB ep - doi chung {np.mean(v7) - np.mean([L7[str(e)][DC7]["mAP_anh"] for e in EP7]):+.4f}')
_SS7 = (('txtq', 'txtq_nn', 'phan cua NGHIA van ban'), ('txtq', 'q_hoc', 'van ban vs query hoc'),
        ('txtq_5mt', 'txtq', '5 cach dien dat'), ('txtq_cau', 'txtq', 'cau report khang dinh'), ('txtq_k4', 'txtq', '4 nguyen mau'),
        ('txtq_du', 'txtq', 'cho phep lech (Delta)'), ('txtq_vung', 'txtq', 'them vung'), ('txtq_vung', 'vung_kdp', 'them txtq'),
        ('txtq_dich', 'txtq', 'them dich'), ('txtq_dich', 'dich_tat_ca', 'them txtq'), ('cau_clip', 'cau_bert', 'thap HOC CUNG vs vector'))
print('\nSO SANH RIENG (VAL mAP_anh; cuoi | TB moi epoch; 1 seed):')
for a7, b7, y7 in _SS7:
    if a7 in A7 and b7 in A7:
        print(f'  {a7 + " - " + b7:<24} {R7[a7]["mAP_anh"] - R7[b7]["mAP_anh"]:+.4f} | '
              f'{np.mean([L7[str(e)][a7]["mAP_anh"] - L7[str(e)][b7]["mAP_anh"] for e in EP7]):+.4f}   ({y7})')
print('\nLOSS PHU (TB micro-batch co phan do) | tau | h(t) | g = ||grad lambda*L_phu|| / ||grad L_nhan|| (3 micro-batch dau):')
for e in EP7:
    r7 = L7[str(e)]
    print(f'  ep{e + 1:<3}' + ' | '.join(f" {t} " + ' '.join(f"{k} {_f7f(r7[t].get('loss_' + k), 3)}" for k in ('vung', 'dich', 'cau')
                                                            if r7[t].get('loss_' + k) is not None)
                                        + f" tau {_f7f(r7[t].get('tau'), 3)} h {_f7f(r7[t].get('he_so'), 2)} "
                                          f"g {_f7f(r7[t].get('ti_le_grad'), 2)}" for t in P7))
print(f'\nGIU RIENG (2% benh nhan train KHONG vao loss phu), ep{EP7[-1] + 1}:')
print('  khoi attention lop -> patch trong vung goi y KHONG du phong | 5 nhan trong phoi:')
for t in A7:
    print(f"    {t:<12} {_f7f(R7[t].get('khoi_kdp'), 3):>7} | {_f7f(R7[t].get('khoi_kdp_phoi'), 3):>7}")
_tb7 = lambda v: None if not v or all(x is None for x in v) else float(np.mean([x for x in v if x is not None]))
_D7 = [t for t in A7 if any(k in R7[t] for k in ('T1_auc', 'T2_auc', 'T3_acc', 'T4_acc', 'T5_auc', 'T6_auc', 'T7_auc'))]
if _D7:
    print('  dau dich: AUC T1 / T2 (TB lop du hai lop) / T5 / T6 / T7 (TB bit) | do chinh xac T3 / T4:')
    for t in _D7:
        print(f'    {t:<12} ' + ' / '.join(_f7f(R7[t].get(k + '_auc') if k in ('T1', 'T2') else _tb7(R7[t].get(k + '_auc')), 3)
                                     for k in ('T1', 'T2', 'T5', 'T6', 'T7'))
              + f" | {_f7f(R7[t].get('T3_acc'), 3)} / {_f7f(R7[t].get('T4_acc'), 3)}")
_C7 = [t for t in A7 if 'cau_top1' in R7[t]]
if _C7:
    print('  truy hoi anh -> cau (kho cau cua study giu rieng): top1 / hit10 (ngau nhien):')
    for t in _C7:
        print(f"    {t:<12} {_f7f(R7[t]['cau_top1'], 3)} / {_f7f(R7[t]['cau_hit10'], 3)} ({_f7f(R7[t]['cau_top1_nn'], 3)} / "
              f"{_f7f(R7[t]['cau_hit10_nn'], 3)}) | {R7[t].get('cau_n_anh')} anh, {R7[t].get('cau_n_ung')} cau")
_Q7 = [t for t in A7 if 'cos_q' in R7[t]]
if _Q7:
    print('  query: cos(q cuoi, q luc dau) TB / nho nhat' + ' | ||Delta|| / ||co so|| (txtq_du) | nguyen mau thang (txtq_k4):')
    for t in _Q7:
        _th = np.array(R7[t]['thang']) if 'thang' in R7[t] else None
        _ty = (_th.max(1) / np.maximum(_th.sum(1), 1))[_th.sum(1) > 0] if _th is not None else None
        print(f"    {t:<12} {_f7f(R7[t]['cos_q'], 3)} / {_f7f(R7[t].get('cos_q_min'), 3)}"
              + (f" | {_f7f(R7[t]['ti_le_delta'], 3)}" if 'ti_le_delta' in R7[t] else '')
              + (f' | ti le nguyen mau thang nhieu nhat TB {_ty.mean():.2f} ({int((_ty > 0.9).sum())} / {len(_ty)} lop > 0.9)'
                 if _ty is not None and len(_ty) else ''))
print('\n1 seed = SANG LOC. Gop view + hai nua val + bootstrap theo benh nhan: bang 78a cua 78z J v3b.')
