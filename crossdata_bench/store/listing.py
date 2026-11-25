# -*- coding: utf-8 -*-
"""Liệt kê một cây thư mục LỚN trên Drive qua API (CheXpert bản đầy đủ: ~65.000 thư mục người + ~190.000 thư mục lần chụp + ~224.000 ảnh) — CHỈ đọc
siêu dữ liệu (tên, kích thước, md5Checksum), không tải tệp.

Vì sao không dùng external_benchmarks.integrity.walk_drive: nó gọi files.list MỘT lần cho MỖI thư mục (~255.000 lần gọi với CheXpert). Ở đây mỗi lần gọi hỏi con
của K thư mục cùng lúc: q = "('a' in parents or 'b' in parents …) and trashed = false" (trả về kèm `parents` để gán con cho đúng cha), duyệt theo TẦNG,
nhiều luồng (dịch vụ API riêng mỗi luồng = cách của 85 / 89k). Máy chủ chê truy vấn (HTTP 400, vd quá nhiều vế OR) -> chia đôi lô, và K của các lô sau giảm
theo; 403 / 429 / 5xx / lỗi mạng -> dịch vụ mới + chờ 2, 4, … 64 s, tối đa RETRIES lần. Mỗi tầng xong ghi trạng thái vào tệp local -> chạy lại ô thì tiếp
từ tầng chưa xong (mất runtime: làm lại từ đầu). Không đi theo shortcut bên trong cây (đếm riêng). Không giữ trường chủ sở hữu / email.

0.4.1 (Tuấn 24/11: "thiếu ảnh nào thì lấy từ zip ra, sau đó xoá zip"): thêm các lệnh GHI dùng ở ô C7 / C10 của 93i — get (quyền: ownedByMe,
capabilities), find_child, create_folder, upload (md5 máy chủ = md5 dữ liệu, sai thì chuyển bản vừa lên vào thùng rác rồi thử lại), trash (chuyển vào thùng rác,
KHÔNG xoá vĩnh viễn). create_folder / upload tra trước theo tên trong thư mục cha -> thử lại sau lỗi mạng không đẻ bản trùng."""
import hashlib
import io
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor

FOLDER_MIME = 'application/vnd.google-apps.folder'
SHORTCUT_MIME = 'application/vnd.google-apps.shortcut'
FIELDS = 'nextPageToken, files(id,name,mimeType,size,md5Checksum,parents)'
GET_FIELDS = 'id,name,mimeType,size,md5Checksum,trashed,ownedByMe,capabilities(canTrash,canAddChildren)'
UP_FIELDS = 'id,name,size,md5Checksum,parents'
K_START, THREADS, MAX_DEPTH, RETRIES = 100, 6, 8, 8
PAGE = 1000
RESUMABLE_OVER = 4 << 20                                             # tệp > 4 MB tải lên kiểu resumable (multipart chỉ cho tệp nhỏ)


def _esc(s):
    return str(s).replace('\\', '\\\\').replace("'", "\\'")


def default_media(data, mime):
    from googleapiclient.http import MediaIoBaseUpload
    return MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=len(data) > RESUMABLE_OVER, chunksize=32 << 20)


def status_of(e):
    '''Mã HTTP của lỗi googleapiclient (HttpError.resp.status / status_code), không có -> None.'''
    r = getattr(e, 'resp', None)
    s = getattr(r, 'status', None) if r is not None else None
    if s is None:
        s = getattr(e, 'status_code', None)
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def default_api():
    import logging
    from googleapiclient.discovery import build
    logging.getLogger('googleapiclient.http').setLevel(logging.ERROR)   # 0.4.3: cảnh báo "403 rateLimitExceeded" (Lister tự thử lại + đếm) không ra output
    return build('drive', 'v3', cache_discovery=False)


class Lister:
    def __init__(self, api_factory=default_api, threads=THREADS, k=K_START, retries=RETRIES, sleep=time.sleep, page=PAGE, media=default_media):
        self._factory, self._tl, self._lock = api_factory, threading.local(), threading.Lock()
        self.threads, self.k, self.retries, self._sleep, self.page, self._media = threads, k, retries, sleep, page, media
        self.calls = self.retried = self.split = self.writes = 0

    # ---- một lần gọi ----
    def _api(self):
        if getattr(self._tl, 'api', None) is None:
            self._tl.api = self._factory()
        return self._tl.api

    def _count(self, name):
        with self._lock:
            setattr(self, name, getattr(self, name) + 1)

    def _retry(self, fn):
        for a in range(self.retries):
            try:
                return fn()
            except Exception as e:
                if status_of(e) == 400:
                    raise                                        # truy vấn bị chê -> người gọi chia lô
                self._tl.api = None
                self._count('retried')
                if a == self.retries - 1:
                    raise
                self._sleep(min(64, 2 ** (a + 1)))

    def query(self, q):
        '''Mọi trang của files.list(q) -> [dict].'''
        out, tok = [], None
        while True:
            r = self._retry(lambda: self._api().files().list(q=q, pageSize=self.page, pageToken=tok, fields=FIELDS, includeItemsFromAllDrives=True,
                                                            supportsAllDrives=True).execute())
            self._count('calls')
            out += r.get('files', [])
            tok = r.get('nextPageToken')
            if not tok:
                return out

    def _children(self, ids):
        q = '(' + ' or '.join(f"'{i}' in parents" for i in ids) + ') and trashed = false'
        try:
            return self.query(q)
        except Exception as e:
            if status_of(e) == 400 and len(ids) > 1:
                h = len(ids) // 2
                with self._lock:
                    self.split += 1
                    self.k = min(self.k, h)
                return self._children(ids[:h]) + self._children(ids[h:])
            raise

    def children(self, ids):
        '''Con của mọi thư mục trong ids (lô theo K hiện tại).'''
        out = []
        for i in range(0, len(ids), max(1, self.k)):
            out += self._children(ids[i:i + max(1, self.k)])
        return out

    def search_name(self, prefix):
        '''Tìm khắp Drive theo tên bắt đầu bằng `prefix` (toán tử contains của Drive là so tiền tố từ) — dùng tìm tệp zip ngoài cây.'''
        return self.query(f"name contains '{prefix}' and trashed = false")

    def fetch_range(self, file_id, a, b):
        '''Byte [a, b] (gồm b) của một tệp QUA API (header Range — cách MediaIoBaseDownload tải từng khúc). Số byte khác khoảng -> lỗi (thử lại).'''
        def one():
            req = self._api().files().get_media(fileId=file_id, supportsAllDrives=True)
            req.headers['range'] = f'bytes={a}-{b}'
            data = req.execute()
            if len(data) != b - a + 1:
                raise IOError(f'Drive tra {len(data)} byte cho khoang {b - a + 1} byte')
            return data
        return self._retry(one)

    # ---- một tệp: quyền / ghi (93i C7 / C10) ----
    def get(self, file_id, fields=GET_FIELDS):
        r = self._retry(lambda: self._api().files().get(fileId=file_id, fields=fields, supportsAllDrives=True).execute())
        self._count('calls')
        return r

    def find_child(self, parent, name, folder=False):
        '''Con (chưa vào thùng rác) tên đúng `name` của `parent`; folder=True -> chỉ thư mục.'''
        q = f"'{parent}' in parents and name = '{_esc(name)}' and trashed = false" + (f" and mimeType = '{FOLDER_MIME}'" if folder else '')
        return self.query(q)

    def create_folder(self, parent, name):
        '''-> id thư mục con `name` của `parent` (có sẵn thì dùng lại, không tạo bản trùng).'''
        def one():
            ex = self.find_child(parent, name, folder=True)
            if ex:
                return ex[0]['id']
            r = self._api().files().create(body={'name': name, 'mimeType': FOLDER_MIME, 'parents': [parent]}, fields='id', supportsAllDrives=True).execute()
            self._count('writes')
            return r['id']
        return self._retry(one)

    def upload(self, parent, name, data, mime='application/octet-stream'):
        '''Tải `data` lên thành tệp `name` trong `parent` -> dict(id, name, size, md5Checksum). Đã có bản cùng tên + cùng md5 + cùng kích thước (lần thử
           trước đã lên) -> trả bản đó. md5 / kích thước máy chủ khác dữ liệu -> chuyển bản vừa lên vào thùng rác rồi lỗi (thử lại).'''
        md5 = hashlib.md5(data).hexdigest()

        def one():
            for f in self.find_child(parent, name):
                if f.get('md5Checksum') == md5 and int(f.get('size') or -1) == len(data):
                    return f
            r = self._api().files().create(body={'name': name, 'parents': [parent]}, media_body=self._media(data, mime), fields=UP_FIELDS,
                                           supportsAllDrives=True).execute()
            self._count('writes')
            if r.get('md5Checksum') != md5 or int(r.get('size') or -1) != len(data):
                self.trash(r['id'])
                raise IOError(f"{name}: may chu md5 {r.get('md5Checksum')} / {r.get('size')} byte != du lieu {md5} / {len(data)} byte")
            return r
        return self._retry(one)

    def trash(self, file_id):
        '''Chuyển vào THÙNG RÁC (khôi phục được trong 30 ngày; vẫn tính dung lượng tới khi dọn) -> trashed của máy chủ.'''
        r = self._retry(lambda: self._api().files().update(fileId=file_id, body={'trashed': True}, fields='id,trashed', supportsAllDrives=True).execute())
        self._count('writes')
        return bool(r.get('trashed'))

    # ---- cả cây ----
    def walk(self, root_id, max_depth=MAX_DEPTH, resume_path=None, log=print, every=60.0):
        '''-> dict(files [dict(duong, ten, id, size, md5, mime, tang)], folders [dict(duong, tang, id)], chua_duyet (số thư mục sâu hơn max_depth), thong_ke).
           duong = đường dẫn tương đối từ gốc (thư mục kết thúc '/').'''
        st = None
        if resume_path and os.path.exists(resume_path):
            with open(resume_path, 'rb') as f:
                st = pickle.load(f)
            if st.get('root') != root_id:
                st = None
        if st is None:
            st = dict(root=root_id, depth=0, frontier=[(root_id, '')], files=[], folders=[], levels=[], phut=0.0)
        else:
            log(f"   tiep tu tang {st['depth'] + 1} (lan chay truoc da xong {st['depth']} tang, {len(st['files']):,} tep)")
        t_all = time.time()
        while st['frontier'] and st['depth'] < max_depth:
            ids = [i for i, _ in st['frontier']]
            path_of = dict(st['frontier'])
            k = max(1, self.k)
            chunks = [ids[i:i + k] for i in range(0, len(ids), k)]
            nxt, files, n_items, t0, last, calls0 = [], [], 0, time.time(), time.time(), self.calls
            depth = st['depth'] + 1
            pool = ThreadPoolExecutor(self.threads)
            try:                                                 # ngắt ô (Interrupt) -> huỷ các lô chưa chạy, không chờ hết tầng
                for j, res in enumerate(pool.map(self.children, chunks), 1):
                    for f in res:
                        for p in f.get('parents') or []:
                            if p not in path_of:
                                continue
                            rel = path_of[p] + f['name']
                            if f.get('mimeType') == FOLDER_MIME:
                                nxt.append((f['id'], rel + '/'))
                            else:
                                files.append(dict(duong=rel, ten=f['name'], id=f['id'], size=int(f.get('size') or 0), md5=f.get('md5Checksum'),
                                                  mime=f.get('mimeType'), tang=depth))
                    n_items += len(res)
                    if time.time() - last >= every and j < len(chunks):
                        el = time.time() - t0
                        log(f'   tang {depth}: {j:,} / {len(chunks):,} lo ({len(ids):,} thu muc cha) | {n_items:,} muc | {self.calls - calls0:,} lan goi | '
                            f'{el / 60:.1f} phut, con ~{el / j * (len(chunks) - j) / 60:.1f} phut (DO o tang nay) | thu lai {self.retried} | K {self.k}')
                        last = time.time()
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
            st['folders'] += [dict(duong=r, tang=depth, id=i) for i, r in nxt]
            st['files'] += files
            st['levels'].append(dict(tang=depth, thu_muc_cha=len(ids), thu_muc_con=len(nxt), tep=len(files), lan_goi=self.calls - calls0,
                                     phut=round((time.time() - t0) / 60, 2), k=k))
            st['frontier'], st['depth'] = nxt, depth
            st['phut'] += (time.time() - t0) / 60
            if resume_path:
                tmp = resume_path + '.tmp'
                with open(tmp, 'wb') as f:
                    pickle.dump(st, f)
                os.replace(tmp, resume_path)
            log(f'   tang {depth} xong: {len(ids):,} thu muc cha -> {len(nxt):,} thu muc + {len(files):,} tep | {self.calls - calls0:,} lan goi (K {k}) | '
                f'{(time.time() - t0) / 60:.1f} phut')
        return dict(files=st['files'], folders=st['folders'], chua_duyet=len(st['frontier']),
                    thong_ke=dict(tang=st['levels'], lan_goi=self.calls, thu_lai=self.retried, chia_lo=self.split, k_cuoi=self.k,
                                  phut=round(st['phut'], 2), phut_lan_nay=round((time.time() - t_all) / 60, 2)))
