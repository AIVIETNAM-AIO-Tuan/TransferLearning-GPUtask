# -*- coding: utf-8 -*-
"""Đọc DANH MỤC của một tệp zip nằm trên Drive mà không tải cả tệp (zip CheXpert hàng chục / hàng trăm GB): zipfile của Python chỉ cần đọc phần cuối tệp
(bản ghi kết thúc, zip64) và khối danh mục trung tâm — RangeFile là tệp ảo "seek + read" lấy từng khúc byte qua `fetch(a, b)` (Drive API, header Range).
Danh mục cho mỗi tệp: tên, kích thước gốc, CRC-32 -> so với bản giải nén trên Drive (kích thước) và với tệp mẫu tải về (CRC). 0.4.1: lấy MỘT tệp
ra khỏi zip từ xa (read_entry — đọc header cục bộ + dữ liệu của đúng tệp đó) để bù ảnh thiếu / hỏng trên Drive."""
import collections
import zipfile
import zlib

BLOCK = 1 << 20
BIG = 4 * BLOCK


class RangeFile:
    '''Tệp ảo chỉ đọc trên fetch(a, b) -> bytes [a, b] (gồm b). Đọc nhỏ đi qua khối 1 MB có nhớ đệm (LRU 16 khối); đọc lớn (> 4 MB, vd khối danh mục) một lần.'''

    def __init__(self, fetch, size, block=BLOCK, keep=16):
        self._fetch, self._size, self._block, self._keep = fetch, int(size), block, keep
        self._pos, self._cache = 0, collections.OrderedDict()
        self.requests = self.bytes = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, off, whence=0):
        base = {0: 0, 1: self._pos, 2: self._size}[whence]
        pos = base + off
        if pos < 0:
            raise ValueError('seek truoc dau tep')
        self._pos = pos
        return pos

    def _get(self, a, b):
        '''Byte [a, b) qua fetch.'''
        data = self._fetch(a, b - 1)
        assert len(data) == b - a, f'fetch tra {len(data)} byte, can {b - a}'
        self.requests += 1
        self.bytes += len(data)
        return data

    def _blk(self, i):
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        a = i * self._block
        data = self._get(a, min(self._size, a + self._block))
        self._cache[i] = data
        while len(self._cache) > self._keep:
            self._cache.popitem(last=False)
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._size - self._pos
        n = max(0, min(n, self._size - self._pos))
        if n == 0:
            return b''
        a, b = self._pos, self._pos + n
        if n > BIG:
            data = self._get(a, b)
        else:
            out = []
            for i in range(a // self._block, (b - 1) // self._block + 1):
                blk, s = self._blk(i), i * self._block
                out.append(blk[max(a, s) - s:min(b, s + len(blk)) - s])
            data = b''.join(out)
        self._pos = b
        return data

    def close(self):
        self._cache.clear()


def open_remote(fetch, size):
    '''-> (zipfile.ZipFile trên RangeFile, RangeFile). Mở = đọc danh mục một lần; giữ đối tượng để lấy tệp sau (read_entry) không phải đọc lại danh mục.
       Không an toàn đa luồng (nhớ đệm khối của RangeFile) — lấy tệp tuần tự.'''
    rf = RangeFile(fetch, size)
    return zipfile.ZipFile(rf), rf


def listing(z):
    '''[(tên, kích thước gốc, CRC-32)] — chỉ tệp, bỏ thư mục.'''
    return [(i.filename, int(i.file_size), int(i.CRC)) for i in z.infolist() if not i.is_dir()]


def entries(fetch, size):
    '''Danh mục zip -> ([(tên, kích thước gốc, CRC-32)] — chỉ tệp, bỏ thư mục), dict(lan_doc, byte_doc)).'''
    z, rf = open_remote(fetch, size)
    with z:
        out = listing(z)
    return out, dict(lan_doc=rf.requests, byte_doc=rf.bytes)


def read_entry(z, name):
    '''Một tệp trong zip (đọc từ xa đúng phần của nó: header cục bộ + dữ liệu) -> bytes. zipfile tự so CRC-32 khi đọc hết (lệch -> BadZipFile);
       kiểm thêm kích thước + CRC với danh mục để không dựa vào một chỗ.'''
    info = z.getinfo(name)
    data = z.read(info)
    if len(data) != info.file_size or (zlib.crc32(data) & 0xffffffff) != info.CRC:
        raise zipfile.BadZipFile(f'{name}: {len(data)} byte / CRC lech danh muc')
    return data
