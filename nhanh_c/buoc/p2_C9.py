# ==== nhanh_c buoc p2_C9 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C9 · dong_bo() — CHAY TRUOC KHI TAT RUNTIME ====
# Doi chieu MOI file da ghi voi Drive QUA API, theo ID thu muc, bang md5 (docs/QUY_TAC_GHI_DRIVE.md §0).
dong_bo()

# ---- 78t: rs_ (optimizer + lich + dong ngau nhien; voi thap BiomedBERT ~0.35 GB / epoch) CHI can de CHAY TIEP. Khi luot XONG
# (epoch_log_<seed>_cuoi.json co done) va DA KIEM (C7 + bang 78a cua 78z J v3b), Tuan CO THE xoa rs_ tren Drive BANG TAY. Cell nay CHI in.
_rs9 = sorted((x for x in ten_tren_drive() if re.fullmatch(r'rs_s\d+_ep\d+\.pt', x)),
              key=lambda x: (x.split('_ep')[0], int(re.search(r'_ep(\d+)\.pt$', x).group(1))))
for _s9 in SEEDS:
    _lp9 = tim_moi_noi(f'epoch_log_s{_s9}_cuoi.json')
    _xong9 = False
    if _lp9:
        with open(_lp9) as _f9: _xong9 = bool(json.load(_f9).get(f's{_s9}', {}).get('done'))
    _ds9 = [x for x in _rs9 if x.startswith(f'rs_s{_s9}_')]
    print(f's{_s9}: luot ' + ('XONG' if _xong9 else 'CHUA xong -> GIU moi rs_') + f' | {len(_ds9)} tep rs_ tren Drive'
          + (': CO THE xoa sau khi kiem (Tuan quyet, KHONG tu xoa): ' + ', '.join(_ds9) if _xong9 and _ds9 else ''))
# ---- B.4 v2 (mot notebook): phan 2 XONG (luot 10 / 10 epoch + dong_bo du) -> moi duoc sang phan 3 ----
_OK_PHAN_2 = bool(_xong9) and bool(dong_bo()) and not THU_NHANH
print(f"B.4 v2: phan 2 {'XONG -> sang phan 3 (gop trong so)' if _OK_PHAN_2 else 'CHUA XONG -> o chuyen se DUNG (khong sang phan 3)'}")
