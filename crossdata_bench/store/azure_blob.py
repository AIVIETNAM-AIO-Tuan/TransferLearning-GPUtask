# -*- coding: utf-8 -*-
"""Đọc một container Azure Blob qua link SAS (0.4.3; notebook 93h — trang AIMI cũ cấp link SAS cho cả bộ CheXlocalize: thư mục CheXpert/ + CheXlocalize/;
Event 2026-11-25 "Module 93 phiên 6 …"). CHỈ thư viện chuẩn (urllib + xml), không cần azure-storage-blob / AzCopy.

- parse(url) -> Sas: https, máy chủ <tài khoản>.blob.core.windows.net, đường dẫn = đúng một container, có chữ ký `sig`. Chuỗi truy vấn giữ NGUYÊN (sig đã mã
  hoá URL — không mã hoá lại). repr / str của Sas KHÔNG chứa chữ ký; info() đọc quyền (sp), loại tài nguyên (sr), hiệu lực (st / se) để in.
- list_blobs(sas, prefix) -> [dict(name, size, md5 (hex | None), md5_sai_dang, kieu, thu_muc)]: REST "List Blobs" (restype=container&comp=list,
  include=metadata), phân trang theo NextMarker. thu_muc = mục giả "thư mục" của tài khoản có không gian tên phân cấp (metadata hdi_isfolder=true hoặc
  ResourceType=directory).
- get_blob(sas, name, size, md5) -> bytes: số byte = danh mục; md5 = Content-MD5 của Azure nếu Azure có; lệch -> tải lại (RETRIES_DATA lần) rồi lỗi.
- Lỗi -> AzureError(ma, http): ma = mã lỗi Azure (AuthenticationFailed: hết hạn / sai chữ ký; AuthorizationPermissionMismatch: thiếu quyền; …) hoặc
  LoiMang; 5xx / 408 / 429 / lỗi mạng: thử lại, chờ 2, 4, … 32 s. Mọi thông báo đi qua Sas.redact -> link / chữ ký không bao giờ vào thông báo lỗi.
Hàm `fetch(url) -> (mã HTTP, header, thân)` thay được (bộ kiểm: Azure giả trong bộ nhớ + máy chủ HTTP local)."""
import base64
import datetime
import hashlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HOST_RE = re.compile(r'^[a-z0-9]{3,24}\.blob\.core\.windows\.net$')
UA = 'crossdata-bench'
RETRIES, RETRIES_DATA, TIMEOUT, PAGE = 6, 3, 120, 5000
RETRY_HTTP = (408, 429, 500, 502, 503, 504)
HINT = {
    'AuthenticationFailed': 'link het han / chua toi gio dung / chep thieu ky tu -> lay link MOI o trang AIMI cu (Download) roi chay lai C2',
    'AuthorizationPermissionMismatch': 'link khong co quyen nay (can doc + liet ke)',
    'AuthorizationResourceTypeMismatch': 'link khong phai link cho ca container (sr=c)',
    'ContainerNotFound': 'khong thay container -> chep lai nguyen link',
    'BlobNotFound': 'khong thay tep tren Azure',
    'LoiMang': 'loi mang sau nhieu lan thu -> chay lai o',
}


class AzureError(Exception):
    def __init__(self, ma, http=None, chi_tiet=''):
        self.ma, self.http = ma, http
        super().__init__(f"Azure {http or '-'} {ma}: {HINT.get(ma, 'xem chi tiet')}" + (f' | {chi_tiet}' if chi_tiet else ''))


class Sas:
    '''Link SAS của một container. KHÔNG in / lưu chuỗi truy vấn: repr chỉ có máy chủ, container, sp, se.'''

    def __init__(self, base, container, query, params):
        self.base, self.container, self._query, self.params = base, container, query, params
        sig = urllib.parse.parse_qs(query, keep_blank_values=True).get('sig', [''])[0]
        self._secrets = sorted({x for x in (query, sig, urllib.parse.quote(sig, safe=''), urllib.parse.quote(sig)) if x}, key=len, reverse=True)

    def __repr__(self):
        return f"Sas({self.base}/{self.container}, sp={self.params.get('sp')}, se={self.params.get('se')})"

    __str__ = __repr__

    def __reduce__(self):                               # không pickle / deepcopy ra chữ ký
        raise TypeError('Sas khong luu duoc (chua chu ky)')

    def url(self, name='', extra=''):
        p = f'{self.base}/{self.container}' + ('/' + urllib.parse.quote(name, safe='/') if name else '')
        return p + '?' + (extra + '&' if extra else '') + self._query

    def redact(self, s):
        s = str(s)
        for x in self._secrets:
            s = s.replace(x, '<link>')
        return re.sub(r'(?i)sig=[^&\s\'"]+', 'sig=<an>', s)


def parse(url, allow_host=None):
    '''-> Sas. allow_host: CHỈ cho bộ kiểm (máy chủ local, cho phép http). Lỗi -> ValueError (thông báo không chứa link).'''
    u = (url or '').strip().strip('"').strip("'").strip()
    if not u:
        raise ValueError('link rong -> dan lai link')
    pr = urllib.parse.urlsplit(u)
    host = (pr.hostname or '').lower()
    test = allow_host is not None and host == allow_host
    if pr.scheme != 'https' and not test:
        raise ValueError(f'link phai bat dau bang https:// (dang la {pr.scheme or "-"}://)')
    if not (HOST_RE.match(host) or test):
        raise ValueError(f'may chu {host or "-"} khong phai <tai khoan>.blob.core.windows.net -> can link tai (SAS) cua nut Download, khong phai link trang web')
    parts = [p for p in pr.path.split('/') if p]
    if len(parts) != 1:
        raise ValueError(f'duong dan co {len(parts)} doan (can dung 1 = ten container) -> chep lai NGUYEN link')
    params = urllib.parse.parse_qs(pr.query, keep_blank_values=True)
    if not params.get('sig', [''])[0]:
        raise ValueError('link thieu chu ky (sig) -> chep lai NGUYEN link (ca phan sau dau ?)')
    return Sas(f'{pr.scheme}://{pr.netloc}', urllib.parse.unquote(parts[0]), pr.query, {k: v[0] for k, v in params.items() if k != 'sig'})


def _when(s):
    if not s:
        return None
    for f in ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%MZ', '%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%d'):
        try:
            return datetime.datetime.strptime(s, f).replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            pass
    return 'khong doc duoc'


def info(sas, now=None):
    '''-> dict quyền / hiệu lực (không chữ ký). now: datetime UTC (bộ kiểm).'''
    now = now or datetime.datetime.now(datetime.timezone.utc)
    p = sas.params
    st, se = _when(p.get('st')), _when(p.get('se'))
    ok_t = isinstance(se, datetime.datetime)
    return dict(may_chu=urllib.parse.urlsplit(sas.base).hostname, container=sas.container, sr=p.get('sr'), sp=p.get('sp'), sv=p.get('sv'),
                doc='r' in (p.get('sp') or ''), liet_ke='l' in (p.get('sp') or ''), bat_dau=p.get('st'), het_han=p.get('se'),
                con_gio=round((se - now).total_seconds() / 3600, 1) if ok_t else None, het_han_roi=bool(ok_t and se <= now),
                chua_toi_gio=bool(isinstance(st, datetime.datetime) and st > now))


def http_get(url, timeout=TIMEOUT):
    '''-> (mã HTTP, header, thân). Lỗi HTTP trả về như kết quả (không ném); lỗi mạng ném (người gọi che link + thử lại).'''
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        try:
            body = e.read()
        except Exception:
            body = b''
        return e.code, dict(e.headers or {}), body


def _local(tag):
    return tag.rsplit('}', 1)[-1]


def _xml(body):
    return ET.fromstring(body[3:] if body[:3] == b'\xef\xbb\xbf' else body)


def _error(body):
    try:
        root = _xml(body)
    except Exception:
        return None, ''
    code = msg = det = ''
    for el in root.iter():
        t = (el.text or '').strip()
        if _local(el.tag) == 'Code':
            code = t
        elif _local(el.tag) == 'Message' and t:
            msg = t.splitlines()[0][:200]
        elif _local(el.tag) == 'AuthenticationErrorDetail' and 'time frame' in t:   # chỉ giữ chi tiết hết hạn (ngày giờ); "string to sign" bỏ
            det = t.splitlines()[0][:200]
    return code or None, msg + (f' | {det}' if det else '')


def _call(sas, url, fetch, sleep, retries=RETRIES):
    last = None
    for a in range(retries):
        try:
            st, hd, body = fetch(url)
        except Exception as e:                          # lỗi mạng / đọc dở: thử lại; thông báo đã che link, KHÔNG nối lỗi gốc
            last = AzureError('LoiMang', None, sas.redact(f'{type(e).__name__}: {e}')[:200])
        else:
            if 200 <= st < 300:
                return hd, body
            code, msg = _error(body)
            err = AzureError(code or f'HTTP{st}', st, sas.redact(msg))
            if st not in RETRY_HTTP:
                raise err
            last = err
        if a < retries - 1:
            sleep(min(32, 2 ** (a + 1)))
    raise last


def _b64_hex(s):
    try:
        b = base64.b64decode(s, validate=True)
        return b.hex() if len(b) == 16 else None
    except Exception:
        return None


def list_blobs(sas, prefix='', fetch=http_get, sleep=time.sleep, page=PAGE, max_pages=None):
    '''Mọi blob có tên bắt đầu bằng prefix (mọi trang; max_pages: dừng sớm — thử link) -> [dict].'''
    out, marker, n = [], '', 0
    while True:
        n += 1
        extra = f'restype=container&comp=list&include=metadata&maxresults={int(page)}'
        if prefix:
            extra += '&prefix=' + urllib.parse.quote(prefix, safe='')
        if marker:
            extra += '&marker=' + urllib.parse.quote(marker, safe='')
        _, body = _call(sas, sas.url('', extra), fetch, sleep)
        root = _xml(body)
        for b in root.iter():
            if _local(b.tag) != 'Blob':
                continue
            f = {_local(c.tag): c for c in b}
            props = {_local(c.tag): (c.text or '').strip() for c in f['Properties']} if 'Properties' in f else {}
            meta = {_local(c.tag).lower(): (c.text or '').strip().lower() for c in f['Metadata']} if 'Metadata' in f else {}
            m64 = props.get('Content-MD5') or ''
            out.append(dict(name=(f['Name'].text or '') if 'Name' in f else '', size=int(props.get('Content-Length') or 0), md5=_b64_hex(m64) if m64 else None,
                            md5_sai_dang=bool(m64) and _b64_hex(m64) is None, kieu=props.get('BlobType'),
                            thu_muc=meta.get('hdi_isfolder') == 'true' or props.get('ResourceType') == 'directory'))
        marker = ''
        for el in root:
            if _local(el.tag) == 'NextMarker':
                marker = (el.text or '').strip()
        if not marker or (max_pages and n >= max_pages):
            return out


def get_blob(sas, name, size=None, md5=None, fetch=http_get, sleep=time.sleep):
    '''bytes của một blob; số byte / md5 lệch danh mục -> tải lại, quá RETRIES_DATA lần -> AzureError('DuLieuLech').'''
    why = ''
    for a in range(RETRIES_DATA):
        _, body = _call(sas, sas.url(name), fetch, sleep)
        if size is not None and len(body) != size:
            why = f'so byte {len(body)} != danh muc {size}'
        elif md5 and hashlib.md5(body).hexdigest() != md5:
            why = 'md5 != Content-MD5 cua Azure'
        else:
            return body
        if a < RETRIES_DATA - 1:
            sleep(2 ** (a + 1))
    raise AzureError('DuLieuLech', None, f'{why} sau {RETRIES_DATA} lan tai')
