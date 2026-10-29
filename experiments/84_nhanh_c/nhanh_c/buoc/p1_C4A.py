# ==== nhanh_c buoc p1_C4A — SINH TU DONG boi experiments/84_nhanh_c/dung_84.py tu bo dung 78 (KHONG sua tay; sha256 kiem o nhanh_c.buoc) ====
# ==== C4a · LOC NHIEU TAT WORKER DataLoader (Python 3.13 + fork) — chay TRUOC C4b, moi phien; KHONG doi du lieu / ket qua ====
# Chi AN dung 3 thong bao da biet, sinh ra KHI TAT worker (sau khi moi lo da giao xong):
#   (1) luong 'QueueFeederThread': OSError [Errno 9] Bad file descriptor -> ValueError 'semaphore or lock released too many times'
#   (2) _ConnectionBase.__del__: OSError [Errno 9] Bad file descriptor
#   (3) _MultiProcessingDataLoaderIter.__del__: AssertionError 'can only test / join a child process'
# Moi loi KHAC (worker chet, CUDA OOM, assert cua ta, ...) van in nhu cu. Worker fork SAU cell nay thua ke bo loc.
import errno, sys, threading

if '_HOOK_GOC_79A' not in globals():                       # chay lai cell -> KHONG boc chong (khong de quy)
    _HOOK_GOC_79A = (threading.excepthook, sys.unraisablehook)
_NHIEU_79A = {'feeder': 0, 'ket_noi': 0, 'iter_con': 0}    # dem trong tien trinh CHINH (worker loc nhung khong dem duoc)


def _loc_luong_78v(a, _goc=_HOOK_GOC_79A[0]):
    e = a.exc_value
    if (getattr(a.thread, 'name', '') or '').startswith('QueueFeederThread') and (
            (isinstance(e, OSError) and e.errno == errno.EBADF)
            or (isinstance(e, ValueError) and 'released too many times' in str(e))):
        _NHIEU_79A['feeder'] += 1
        return
    _goc(a)


def _loc_del_78v(u, _goc=_HOOK_GOC_79A[1]):
    ten, e = getattr(u.object, '__qualname__', '') or '', u.exc_value
    if ten == '_ConnectionBase.__del__' and isinstance(e, OSError) and e.errno == errno.EBADF:
        _NHIEU_79A['ket_noi'] += 1
        return
    if ten == '_MultiProcessingDataLoaderIter.__del__' and isinstance(e, AssertionError) and 'child process' in str(e):
        _NHIEU_79A['iter_con'] += 1
        return
    _goc(u)


threading.excepthook, sys.unraisablehook = _loc_luong_78v, _loc_del_78v
print('C4a: loc 3 thong bao nhieu khi tat worker DataLoader (loi khac van in) | dem: print(_NHIEU_79A)')
