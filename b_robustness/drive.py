"""Đọc Drive qua khối đọc chuẩn của notebook (không gọi Drive trực tiếp, không chứa ID).

kiem_ke(du, mau) -> (id thư mục, {tên tệp: [bản, mới nhất trước]}); tai(bản) -> đường dẫn local; xung_dot(các bản) -> True khi các bản khác nội dung.
= cách _lay_g / _lay_d / _lay_p2 của 85g / 85d / 85f lấy tệp: thiếu -> dừng, nhiều bản khác nội dung -> báo và dùng bản mới nhất."""
import json
import re


class DriveReader:
    '''Một bộ đọc cho một nhóm tệp (vd nguồn P1, nguồn P2): .sources ghi id / md5 / số bản từng tệp đã đọc (vào JSON kết quả).'''

    def __init__(self, kiem_ke, tai, xung_dot, log=print):
        self.kiem_ke, self.tai, self.xung_dot, self.log = kiem_ke, tai, xung_dot, log
        self.sources = {}

    def inventory(self, folder, pattern):
        '''{tên: [bản]} của các tệp khớp `pattern` (regex toàn tên) trong thư mục `folder` (tuple tính từ cxrlt_2024).'''
        return self.kiem_ke(tuple(folder), mau=pattern)[1]

    def fetch(self, folder, inv, name):
        '''Tải bản mới nhất của `name` -> đường dẫn local; thiếu -> AssertionError.'''
        v = inv.get(name)
        assert v, f"thieu {name} trong {'/'.join(folder)} -> chay notebook nguon truoc. DUNG"
        if self.xung_dot(v):
            self.log(f"  ! {name}: {len(v)} ban NOI DUNG KHAC -> dung ban MOI NHAT ({v[0]['mt']})")
        self.sources['/'.join(folder) + '/' + name] = dict(id=v[0]['id'], md5=v[0]['md5'], so_ban=len(v))
        return self.tai(v[0])

    def load_json(self, folder, inv, name):
        with open(self.fetch(folder, inv, name), encoding='utf-8') as f:
            return json.load(f)


def latest(names, pattern):
    '''Tên mới nhất (sắp theo tên = theo dấu thời gian _YYYYmmdd_HHMMSS) khớp `pattern`, hoặc None.'''
    m = sorted(t for t in names if re.fullmatch(pattern, t))
    return m[-1] if m else None
