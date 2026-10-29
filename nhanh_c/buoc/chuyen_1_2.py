# ==== nhanh_c buoc chuyen_1_2 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== nhanh C: CONG SUC KHOE truoc khi sang phan 2 - DAT -> sang phan 2; YEU -> DUNG (bao Claude); Tuan quyet chay tiep: chay mot o
#      `TIEP_TUC_C = True` trong notebook roi chay lai o nay ====
_sk_chuyen = globals().get('SUC_KHOE_C') or {}
assert _sk_chuyen.get('ket_luan') == 'DAT' or (globals().get('TIEP_TUC_C') is True and _sk_chuyen.get('ket_luan') == 'YEU'), (
    f"CONG SUC KHOE {_sk_chuyen.get('ket_luan')} (chenh {_sk_chuyen.get('chenh')}) -> KHONG sang phan 2. YEU: bao Claude; Tuan quyet chay tiep: "
    'dat TIEP_TUC_C = True roi chay lai o nay')
del _sk_chuyen
# ==== CHUYEN PHAN 1 -> PHAN 2 (B.4 v2, mot notebook) - nhu Restart session: tra hook goc, xoa MOI bien cua phan 1 (tep da len Drive o C9),
#      giai phong GPU; phan 2 chay tu C0 cua no. Phan 1 chua xong -> DUNG o day (Run all dung lai, KHONG sang phan 2). ====
assert globals().get('_OK_PHAN_1') is True, 'PHAN 1 chua xong / C9 chua du -> DUNG, KHONG sang phan 2 (chay lai phan 1 hoac bao Claude)'
import sys as _sys_chuyen, threading as _thr_chuyen
for _hk_chuyen in ('_HOOK_GOC_79A', '_HOOK_GOC_78V'):        # C4a / C5b gan hook doc bien toan cuc -> tra hook GOC truoc khi xoa bien
    if _hk_chuyen in globals(): _thr_chuyen.excepthook, _sys_chuyen.unraisablehook = globals()[_hk_chuyen]
get_ipython().run_line_magic('reset', '-f')
import gc as _gc_chuyen, torch as _torch_chuyen
_gc_chuyen.collect()
if _torch_chuyen.cuda.is_available(): _torch_chuyen.cuda.empty_cache()
assert not any(k in globals() for k in ('RUN_TAG', 'ghi', 'dong_bo', 'DRV_DUONG', 'TEN_RESUME', '_OK_PHAN_1', '_HOOK_GOC_79A', '_HOOK_GOC_78V')), (
    'xoa bien chua het -> DUNG')
print(f"phan 2: bat dau tu namespace sach | GPU dang giu {(_torch_chuyen.cuda.memory_allocated() / 2 ** 30) if _torch_chuyen.cuda.is_available() else 0.0:.1f} GiB")
del _gc_chuyen, _torch_chuyen
