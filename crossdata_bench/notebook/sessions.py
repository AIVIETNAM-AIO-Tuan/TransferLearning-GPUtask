# -*- coding: utf-8 -*-
"""Phiên chạy của một lượt dài (0.5.0; PLAN_93 §6 phiên 7 — CheXpert (A) chạy nhiều phiên / nhiều tài khoản Colab):

    token           mã phiên (giữ trong namespace của notebook: chạy lại ô trong CÙNG runtime = cùng phiên)
    lock_check      chặn hai phiên cùng chạy một lượt: tệp resume do phiên KHÁC ghi trong LOCK_MIN phút gần nhất -> dừng (chờ / ngắt phiên kia)
    prune_revisions sau mỗi lần ghi tệp resume: xoá các PHIÊN BẢN CŨ của tệp trên Drive (Drive giữ phiên bản cũ của tệp tải lên tới 30 ngày / 100 bản; có tính
                    dung lượng hay không: chưa kiểm — xoá để không phụ thuộc). Không đủ quyền / lỗi -> ghi chú MỘT lần, không dừng lượt.
Không in ID Drive, không in email."""
import datetime
import re
import uuid

LOCK_MIN = 45.0                          # > chu kỳ lưu 30 phút + thời gian tải tệp resume


def token(ns):
    return ns.setdefault('_PHIEN_93', uuid.uuid4().hex[:12])


def drive_time(s):
    '''modifiedTime của Drive ('2026-11-25T11:00:41.123Z') -> datetime UTC (bỏ phần lẻ giây).'''
    m = re.match(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)', str(s or ''))
    return datetime.datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%S').replace(tzinfo=datetime.timezone.utc) if m else None


def resume_age_min(kiem_ke, duong, ten, now=None):
    '''Tuổi (phút) của bản MỚI NHẤT tệp `ten` trong thư mục `duong` (khối đọc chuẩn kiem_ke) — None nếu chưa có.'''
    v = kiem_ke(duong, mau=re.escape(ten))[1].get(ten)
    if not v:
        return None
    t = drive_time(v[0].get('mt'))
    if t is None:
        return None
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return (now - t).total_seconds() / 60


def lock_check(state, my_token, age_min, limit=LOCK_MIN):
    '''-> (được chạy, thông báo). state = nội dung tệp resume (dict, có thể có 'phien' = [dict(token, ...)] trong giai_doan); age_min = tuổi tệp resume.'''
    if age_min is None or not state:
        return True, 'chua co tep resume -> phien dau'
    ph = (state.get('giai_doan') or {}).get('phien') or state.get('phien') or []
    last = ph[-1]['token'] if ph else None
    if last == my_token:
        return True, 'tep resume do CHINH phien nay ghi (chay lai o trong cung runtime)'
    if age_min >= limit:
        return True, f'tep resume ghi {age_min:.0f} phut truoc boi phien khac (>= {limit:.0f} phut) -> phien kia coi nhu da dung, chay tiep'
    return False, (f'tep resume vua duoc MOT PHIEN KHAC ghi {age_min:.0f} phut truoc (< {limit:.0f} phut) -> co the phien kia VAN DANG CHAY. Kiem tra va '
                   f'ngat phien kia (Runtime > Disconnect and delete runtime), doi ~{limit - age_min:.0f} phut roi chay lai tu C2. KHONG chay hai phien cung luc.')


_NOTED = set()                            # tệp đã báo "không xoá được phiên bản cũ" (báo một lần mỗi tệp)


def prune_revisions(api, file_id, log=print):
    '''Xoá mọi phiên bản (revision) CŨ của tệp `file_id` (giữ bản đầu = mới nhất). -> số bản đã xoá (None nếu lỗi).'''
    if not file_id or api is None:
        return None
    try:
        head = api.files().get(fileId=file_id, fields='headRevisionId', supportsAllDrives=True).execute().get('headRevisionId')
        revs, tok = [], None
        while True:
            r = api.revisions().list(fileId=file_id, fields='nextPageToken,revisions(id)', pageSize=200, pageToken=tok).execute()
            revs += [x['id'] for x in r.get('revisions', [])]
            tok = r.get('nextPageToken')
            if not tok:
                break
        old = [x for x in revs if x != head]
        assert head and head in revs, 'khong xac dinh duoc phien ban moi nhat'
        for x in old:
            api.revisions().delete(fileId=file_id, revisionId=x).execute()
        return len(old)
    except Exception as e:
        if file_id not in _NOTED:
            _NOTED.add(file_id)
            log(f'  (ghi chu) xoa phien ban cu cua tep resume KHONG duoc ({type(e).__name__}) -> bo qua; Drive tu xoa ban cu sau 30 ngay / 100 ban')
        return None
