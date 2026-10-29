# ==== nhanh_c buoc chuyen_2_3 — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== CHUYEN PHAN 2 -> PHAN 3 (B.4 v2, mot notebook) - nhu Restart session: tra hook goc, xoa MOI bien cua phan 2 (tep da len Drive o C9),
#      giai phong GPU; phan 3 chay tu C0 cua no. Phan 2 chua xong -> DUNG o day (Run all dung lai, KHONG sang phan 3). ====
assert globals().get('_OK_PHAN_2') is True, 'PHAN 2 chua xong / C9 chua du -> DUNG, KHONG sang phan 3 (chay lai phan 2 hoac bao Claude)'
import sys as _sys_chuyen, threading as _thr_chuyen
for _hk_chuyen in ('_HOOK_GOC_79A', '_HOOK_GOC_78V'):        # C4a / C5b gan hook doc bien toan cuc -> tra hook GOC truoc khi xoa bien
    if _hk_chuyen in globals(): _thr_chuyen.excepthook, _sys_chuyen.unraisablehook = globals()[_hk_chuyen]
get_ipython().run_line_magic('reset', '-f')
import gc as _gc_chuyen, torch as _torch_chuyen
_gc_chuyen.collect()
if _torch_chuyen.cuda.is_available(): _torch_chuyen.cuda.empty_cache()
assert not any(k in globals() for k in ('RUN_TAG', 'ghi', 'dong_bo', 'DRV_DUONG', 'TEN_RESUME', '_OK_PHAN_2', '_HOOK_GOC_79A', '_HOOK_GOC_78V')), (
    'xoa bien chua het -> DUNG')
print(f"phan 3: bat dau tu namespace sach | GPU dang giu {(_torch_chuyen.cuda.memory_allocated() / 2 ** 30) if _torch_chuyen.cuda.is_available() else 0.0:.1f} GiB")
del _gc_chuyen, _torch_chuyen
