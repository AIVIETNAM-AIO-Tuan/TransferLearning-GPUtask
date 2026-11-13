# -*- coding: utf-8 -*-
"""Cổng nạp trọng số: mô hình ĐÃ nạp phải bằng tệp trọng số trên đĩa từng tensor, từng bit — không thiếu, không thừa, không khác hình / giá trị.
Kết quả là SỐ ĐO trong chính tiến trình dựng mô hình (không gán hằng). Chép từ vindr_protocol.upstream.encoders_ho (module 86, gói 0.8.2 — đã kiểm
bằng tệp thật + đột biến 13/11), tách khỏi phần mô hình; thêm tham số `rename` để mỗi encoder tự khai luật đổi tên của thư viện.

Thư viện có trả thông tin nạp (transformers: output_loading_info) -> đưa vào `library_info`; thư viện không trả (open_clip, timm, torch thuần) ->
None, KHÔNG ghi danh sách rỗng thay cho nó."""
import collections
import hashlib

import torch

LIST_MAX = 20


def _digest(t):
    '''Dấu nội dung (hình + sha1 giá trị) — số thực đổi sang float64 trước (chính xác với fp16 / bf16 / fp32) -> bằng nhau <=> cùng giá trị.'''
    t = t.detach().cpu()
    t = t.double() if t.is_floating_point() else t
    return tuple(t.shape), hashlib.sha1(t.contiguous().numpy().tobytes()).hexdigest()


def _apply(rename, k):
    for a, b in rename:
        k = k.replace(a, b)
    return k


def compare_to_file(state, file_tensors, prefix='', rename=()):
    '''So TỪNG tensor của state_dict mô hình với tệp trọng số (chỉ khoá tệp bắt đầu bằng `prefix`, bỏ tiền tố trước khi so). `rename`: các cặp
       (tên trong mô hình, tên trong tệp) áp lên cả hai phía trước khi ghép tên. Khoá còn lại hai phía ghép theo NỘI DUNG (đổi tên chưa biết vẫn so
       được giá trị). Ghép theo tên: hình trùng và giá trị BẰNG từng bit (ép tensor mô hình về kiểu của tệp). -> dict (danh sách cắt LIST_MAX).'''
    model = {_apply(rename, k): v for k, v in state.items()}
    file = {_apply(rename, k[len(prefix):]): v for k, v in file_tensors.items() if k.startswith(prefix)}
    both = sorted(set(model) & set(file))
    shape = [k for k in both if tuple(model[k].shape) != tuple(file[k].shape)]
    value = [k for k in both if k not in shape and not torch.equal(model[k].detach().cpu().to(file[k].dtype), file[k])]
    rest_m, rest_f = sorted(set(model) - set(file)), sorted(set(file) - set(model))
    pool = collections.Counter(_digest(file[k]) for k in rest_f)
    by_value, missing = [], []
    for k in rest_m:
        d = _digest(model[k])
        if pool[d] > 0:
            pool[d] -= 1
            by_value.append(k)
        else:
            missing.append(k)
    used = collections.Counter(_digest(model[k]) for k in by_value)
    extra = []
    for k in rest_f:
        d = _digest(file[k])
        if used[d] > 0:
            used[d] -= 1
        else:
            extra.append(k)
    return dict(so_khoa=len(model), so_khoa_tep=len(file), trung_ten=len(both) - len(shape) - len(value), trung_gia_tri_khac_ten=len(by_value),
                so_thieu=len(missing), thieu=missing[:LIST_MAX], so_thua=len(extra), thua=extra[:LIST_MAX], so_khac_hinh=len(shape),
                khac_hinh=shape[:LIST_MAX], so_khac_gia_tri=len(value), khac_gia_tri=value[:LIST_MAX],
                kieu_tep=sorted({str(v.dtype) for v in file.values()}))


def library_info(info):
    '''output_loading_info của transformers -> dict(thieu, thua, sai_hinh, loi) (chuỗi, đã sắp).'''
    s = lambda x: sorted(str(v) for v in (x or []))
    return dict(thieu=s(info.get('missing_keys')), thua=s(info.get('unexpected_keys')), sai_hinh=s(info.get('mismatched_keys')),
                loi=s(info.get('error_msgs')))


def record(source, file_name, library, model_state, file_tensors, lib_info=None, prefix='', rename=(), **extra):
    '''Bản ghi nạp trọng số của một encoder: nguồn, tệp, thư viện, thông tin nạp của thư viện (None nếu thư viện không trả), so với tệp, `dat`.'''
    nap = dict(nguon=source, tep=file_name, thu_vien=library, thong_tin_nap=lib_info,
               so_voi_tep=compare_to_file(model_state, file_tensors, prefix=prefix, rename=rename), **extra)
    nap['dat'] = loading_ok(nap)
    return nap


def loading_ok(nap):
    '''Luật cổng: thông tin nạp của thư viện (nếu có) không khoá thiếu / thừa / sai hình / lỗi, VÀ mọi tensor của mô hình = tệp, có ít nhất một tensor.'''
    t, s = nap['thong_tin_nap'], nap['so_voi_tep']
    lib_ok = t is None or not (t['thieu'] or t['thua'] or t['sai_hinh'] or t['loi'])
    return bool(lib_ok and s['so_khoa'] > 0 and s['so_thieu'] == 0 and s['so_thua'] == 0 and s['so_khac_hinh'] == 0 and s['so_khac_gia_tri'] == 0
                and s['trung_ten'] + s['trung_gia_tri_khac_ten'] == s['so_khoa'])


def loading_text(nap):
    '''Một dòng tóm tắt cho output notebook / log.'''
    t, s = nap['thong_tin_nap'], nap['so_voi_tep']
    lib = ('thu vien khong tra thong tin nap' if t is None else
           f"thong tin nap cua thu vien: thieu {len(t['thieu'])} / thua {len(t['thua'])} / sai hinh {len(t['sai_hinh'])} / loi {len(t['loi'])}")
    return (f"nap trong so {'DAT' if nap['dat'] else 'TRUOT'} | {lib} | so voi tep {nap['tep']}: {s['trung_ten'] + s['trung_gia_tri_khac_ten']}/"
            f"{s['so_khoa']} tensor bang tung bit (khac ten cung gia tri {s['trung_gia_tri_khac_ten']}), thieu {s['so_thieu']}, thua {s['so_thua']}, "
            f"khac hinh {s['so_khac_hinh']}, khac gia tri {s['so_khac_gia_tri']} | {nap['thu_vien']}")
