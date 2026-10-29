# ==== nhanh_c buoc z_C1 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C1 · 78z: KHONG EHR / XGB / tui vi khuan / phim truoc (luat chi-anh 15/10; docs/SPEC_SAU_78Y.md B.0) ====
# Moi dau cua 78z CHI ANH -> C1 cua 53 / 56 / 75a (217 cot EHR, 40 cot XGB, tui vi khuan, phim truoc) KHONG nap, KHONG tai tep XGB test.
# Bien GIU CHO (cung ten, cung so hang ORDER) de may 78t / 78v chay nguyen van: FEAT 0 cot, tui vi khuan rong, PREV_ROW = -1 (khong phim
# truoc; CAN_PHIM = False). C6 go FEAT / MIDS / MCNT o cong duong cham (du doan phai trung tung bit).
assert not CAN_PHIM, '78z: khong arm nao doc phim truoc -> DUNG'
FEAT = np.zeros((len(ORDER), 0), 'float32')
V_MIC, PAD_ID, MIC_L = 1, 0, 1
MIDS = np.zeros((len(ORDER), MIC_L), 'int64'); MCNT = np.zeros((len(ORDER), MIC_L), 'float32')
PREV_ROW = np.full(len(ORDER), -1, 'int64')
print(f'78z C1: KHONG EHR / XGB / tui vi khuan / phim truoc | FEAT {FEAT.shape} | PREV_ROW = -1 ({len(ORDER):,} hang)')
