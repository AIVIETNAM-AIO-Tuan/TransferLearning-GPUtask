# -*- coding: utf-8 -*-
"""Biểu đồ của notebook tổng hợp (matplotlib, ảnh tĩnh). Mỗi hàm nhận số đã tính (ô của evaluation/compare.py) và trả Figure — không đọc Drive.

SỐ VẼ = điểm trên CẢ tập test ở panel 28 nhãn (Tuấn 15/11: chỉ cần 28 nhãn): AP bậc thang, AUROC, AUPRC hình thang theo ngữ nghĩa torchmetrics 0.10.3
(như Bảng IV CheXFound) — lớp không có ca dương ở test bị bỏ khỏi AP và AUPRC, AUROC của lớp đó = 0 vẫn tính. KHÔNG vẽ khoảng tin cậy: bootstrap trên 28
nhãn không ổn (lớp hiếm làm nhiều lần rút thiếu ca dương); CI + luật §7.4 vẫn ở bảng mAP21 (C5–C6). Câu ghi chú về lớp không có ca dương do notebook
dựng từ dữ liệu (summary._note28) rồi truyền vào `note`.

Chọn dạng theo việc của dữ liệu; màu ĐÃ CHẠY bộ kiểm bảng màu (dataviz validate_palette, nền sáng #fcfcfb, 15/11):
  - dots       độ lớn giữa encoder: biểu đồ chấm, MỖI ĐẦU ĐỌC MỘT Ô, tên encoder ở trục y (không ghi chữ cạnh chấm); tối đa 3 điều kiện = 3 màu đầu của
               bảng màu (mọi cặp: CVD ΔE 9,2; thường 24,0 — đạt) + khác hình chấm
  - heat       lưới encoder x đầu đọc: bản đồ nhiệt MỘT sắc (xanh, nhạt -> đậm), có số trong ô (đây cũng là bảng)
  - versus     Ver1 so với Ver2 từng encoder: tán xạ một màu, đường y = x, Spearman ở tiêu đề ô; nhãn tên ĐẶT KHÔNG CHỒNG (_labels)
  - ref_diff   hiệu điểm hàng tham chiếu − X (mô tả, không CI): bản đồ nhiệt PHÂN KỲ đỏ <-> xám <-> xanh, số trong mọi ô
  - cond_diff  hiệu điểm điều kiện a − điều kiện b của cùng encoder (0.9.0: kho cắt giữa − kho đệm vuông): cùng khuôn với ref_diff
  - lr_curves  số đo val theo lr: đường, 4 đầu = 4 màu theo thứ tự cố định (cặp kề nhau: đạt)
  Điều kiện thứ tư (0.9.0, kho cắt giữa): KHÔNG thêm màu thứ tư vào `dots` — chấm của các điều kiện chồng lên nhau nên mọi cặp màu phải phân biệt, và ô màu thứ
  tư (vàng cạnh cam) trượt sàn mọi cặp của bảng màu (dataviz palette.md) -> dots giữ 3 điều kiện; kho cắt giữa ở `heat` (mỗi điều kiện một ô, thang một sắc) và
  ở `cond_diff`.
Bố cục chung (_layout): tiêu đề trái-trên; DƯỚI các ô lần lượt chú giải / thang màu rồi ghi chú gói dòng — không chữ nào đè lên ô. Chữ màu mực; lưới mảnh;
không viền trên / phải. Xanh ngọc và vàng dưới 3:1 trên nền sáng -> luôn kèm số (bảng C5, hình lưới)."""
import textwrap

import numpy as np

INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
COND = {('v1', 'kcb'): ('#2a78d6', 'o'), ('v2', 'kcb'): ('#eb6834', 's'), ('v2', 'cb'): ('#1baf7a', 'D')}      # 3 màu đầu: đạt mọi cặp
HEAD_COLOR = {'lin1': '#2a78d6', 'lin4': '#eb6834', 'mldec': '#1baf7a', 'glori': '#eda100'}                   # thứ tự cố định: đạt cặp kề
SEQ = ('#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b')
DIV = ('#e34948', '#f0efec', '#2a78d6')          # âm (hàng tham chiếu kém) — giữa xám — dương (hàng tham chiếu hơn)
PANEL = 'p28'
METRICS = ('ap', 'auroc', 'auprc_thang')
SLUG = {'ap': 'ap', 'auroc': 'auroc', 'auprc_thang': 'auprc'}
PANEL_TEXT = {'p28': '28 nhãn', 'p7': '7 dấu hiệu RAD-DINO'}       # 0.9.1: hình theo panel (28 nhãn; 7 dấu hiệu Bảng 1a bài RAD-DINO)
PANEL_SHORT = {'p28': '28', 'p7': '7'}
_MT = {'ap': 'AP {} (bậc thang)', 'auroc': 'AUROC {}', 'auprc_thang': 'AUPRC {} (hình thang)'}
_MS = {'ap': 'AP', 'auroc': 'AUROC', 'auprc_thang': 'AUPRC'}


def metric_text(metric, panel=PANEL):
    return _MT[metric].format(PANEL_TEXT[panel])


def metric_short(metric, panel=PANEL):
    return f'{_MS[metric]} {PANEL_SHORT[panel]}'


METRIC_TEXT = {m: metric_text(m) for m in METRICS}       # = chữ của hình 28 nhãn (không đổi so với 0.7.0)
METRIC_SHORT = {m: metric_short(m) for m in METRICS}
FS = dict(title=9.0, base=8.0, tick=7.5, cell=7.0, label=6.8, note=7.0)        # cỡ chữ (pt)


def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': FS['base'], 'axes.titlesize': FS['base'] + 0.5, 'axes.labelsize': FS['base'], 'xtick.labelsize': FS['tick'],
                                'ytick.labelsize': FS['tick'], 'legend.fontsize': FS['tick'], 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                                'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'axes.titlecolor': INK, 'figure.facecolor': SURFACE,
                                'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'legend.frameon': False})
    return plt


def _clean(ax, grid='x'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelcolor=INK2)


def _bare(ax):
    '''Ô bản đồ nhiệt: không viền, vạch trắng (màu nền) giữa các ô.'''
    ax.tick_params(length=0, labelcolor=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(which='minor', color=SURFACE, linewidth=2)
    ax.tick_params(which='minor', length=0)


def _cmap(colors, name):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, list(colors))


def _fmt(v, sign=False):
    return (f'{v:+.3f}' if sign else f'{v:.3f}').replace('.', ',')


def _val(cells, cond, enc, head, metric, panel=PANEL):
    '''Số của ô (điều kiện, encoder, đầu) hoặc None (chưa có lượt / NaN).'''
    c = cells.get((cond[0], cond[1], enc, head))
    if c is None:
        return None
    v = c['diem'][panel][metric]
    return float(v) if v == v else None


def _wrap(note, width_in):
    '''Ghi chú -> (các dòng gói theo bề ngang hình, chiều cao dải ghi chú (inch)).'''
    lines = []
    for para in (note or '').split('\n'):
        lines += textwrap.wrap(para, width=max(40, int(width_in * 72 / (FS['note'] * 0.6)))) if para.strip() else []
    return lines, (len(lines) * FS['note'] * 1.3 / 72 + 0.08 if lines else 0.06)


def _layout(fig, title=None, note=None, legend=None, cbar=None, tight=True):
    '''Bố cục chung, gọi SAU khi vẽ xong các ô: tiêu đề trái-trên; dải DƯỚI các ô (từ trên xuống): chú giải một hàng, thang màu ngang, ghi chú gói dòng.
       legend = (handles, labels); cbar = (mappable, nhãn). tight=False: vị trí ô đã đặt tay (versus) -> chỉ đặt tiêu đề / dải dưới. Không gọi tight_layout
       lần nữa sau hàm này.'''
    W, H = fig.get_size_inches()
    lines, note_h = _wrap(note, W)
    cb_h = 0.48 if cbar else 0.0
    leg_h = 0.30 if legend else 0.0
    top_h = 0.36 if title else 0.08
    if tight:
        fig.tight_layout(rect=(0, (note_h + cb_h + leg_h) / H, 1, 1 - top_h / H))
    if title:
        fig.text(0.006, 1 - 0.07 / H, title, ha='left', va='top', fontsize=FS['title'], color=INK)
    if lines:
        fig.text(0.006, 0.04 / H, '\n'.join(lines), ha='left', va='bottom', fontsize=FS['note'], color=INK2, linespacing=1.3)
    if cbar:
        im, label = cbar
        cax = fig.add_axes([0.36, (note_h + 0.26) / H, 0.32, 0.09 / H], label='thang_mau')
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelcolor=INK2, labelsize=FS['tick'] - 0.5, pad=1.5)
        fig.text(0.35, (note_h + 0.305) / H, label, ha='right', va='center', fontsize=FS['tick'], color=INK2)
    if legend:
        hd, lb = legend
        fig.legend(hd, lb, loc='center', ncol=len(lb), bbox_to_anchor=(0.5, (note_h + cb_h + leg_h / 2) / H), handletextpad=0.4, columnspacing=1.8,
                   borderaxespad=0.0)


def _renderer(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer() if hasattr(fig.canvas, 'get_renderer') else fig._get_renderer()


def _labels(ax, xs, ys, texts, rnd):
    '''Nhãn cạnh chấm KHÔNG chồng nhau (gọi SAU _layout + vẽ nháp): đo hộp chữ thật bằng renderer; đặt lần lượt từ chấm cao nhất xuống. Mỗi nhãn thử bên
       phải và bên trái chấm (bên nào vừa ô; không bên nào vừa -> dời ngang vào trong ô), mỗi bên lấy vị trí dọc GẦN chỗ muốn nhất mà không đè nhãn đã đặt
       hay chấm khác và nằm trong ô, rồi chọn bên phải dời ít hơn (ưu tiên nhẹ bên phải). Ứng viên dọc = chỗ muốn, hai mép ô, sát trên / sát dưới từng
       vật cản cùng dải ngang — điểm gần nhất của tập khả thi luôn là một trong số đó. Hết chỗ (ô quá chật) -> giữ chỗ muốn. Nhãn cách xa chấm có đường
       dẫn mảnh. -> list Annotation (cùng thứ tự `texts`).'''
    fig = ax.figure
    bb = ax.get_window_extent(rnd)
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    P = ax.transData.transform(np.column_stack([xs, ys]))
    k = fig.dpi / 72.0
    gap, mk, pad, eps = 3.6 * k, 3.0 * k, 1.5 * k, 1e-3          # chấm -> chữ, nửa cạnh ô chấm, khe tối thiểu giữa hai hộp (điểm ảnh)
    sizes = []
    for t in texts:
        tmp = ax.text(0, 0, t, fontsize=FS['label'])
        e = tmp.get_window_extent(rnd)
        tmp.remove()
        sizes.append((e.width, e.height))
    marks = [(px - mk, py - mk, 2 * mk, 2 * mk) for px, py in P]
    lo, hi = bb.y0 + pad, bb.y1 - pad

    def hit(a, b):
        return a[0] < b[0] + b[2] + pad and b[0] < a[0] + a[2] + pad and a[1] < b[1] + b[3] + pad and b[1] < a[1] + a[3] + pad

    placed, out = [], [None] * len(texts)
    for i in sorted(range(len(texts)), key=lambda j: (-P[j, 1], j)):
        (px, py), (w, h) = P[i], sizes[i]
        obst = placed + [m for j, m in enumerate(marks) if j != i]
        want = py - h / 2
        xs0 = [px + gap, px - gap - w] if px + gap + w <= bb.x1 - pad else [px - gap - w, px + gap]
        xs0 = [x for x in xs0 if bb.x0 + pad <= x and x + w <= bb.x1 - pad]
        xs0.append(min(max(px + gap, bb.x0 + pad), bb.x1 - pad - w))       # không bên nào vừa ô -> dời ngang vào trong ô (có đường dẫn)
        best, cost = None, None
        for r, x0 in enumerate(xs0):
            cands = {want, lo, hi - h}
            for b in obst:
                if x0 < b[0] + b[2] + pad and b[0] < x0 + w + pad:
                    cands |= {b[1] + b[3] + pad + eps, b[1] - pad - h - eps}
            ok = [y for y in cands if lo - eps <= y <= hi - h + eps and not any(hit((x0, y, w, h), b) for b in obst)]
            if ok:
                y = min(ok, key=lambda y: (abs(y - want), y))
                c = (abs(y - want) + (0.0 if r == 0 else 0.25 * h) + (0.0 if r < len(xs0) - 1 or len(xs0) == 1 else 2 * h), r)   # bên ưu tiên nhẹ; dời ngang phạt
                if cost is None or c < cost:
                    best, cost = (x0, y), c
        if best is None:
            best = (xs0[0], want)
        placed.append((best[0], best[1], w, h))
        out[i] = best
    inv = ax.transData.inverted()
    anns = []
    for i, (x0, y) in enumerate(out):
        (px, py), (w, h) = P[i], sizes[i]
        right = x0 + w / 2 >= px                                             # nhãn nằm bên phải chấm -> neo mép trái của chữ
        tx, ty = inv.transform((x0 if right else x0 + w, y + h / 2))
        far = np.hypot(max(x0 - px, 0.0, px - x0 - w), max(y - py, 0.0, py - y - h)) > gap + 0.45 * h
        anns.append(ax.annotate(texts[i], xy=(xs[i], ys[i]), xytext=(tx, ty), textcoords='data', ha='left' if right else 'right', va='center',
                                fontsize=FS['label'], color=INK2, annotation_clip=False, zorder=4,
                                arrowprops=dict(arrowstyle='-', color=MUTED, linewidth=0.5, shrinkA=0.5, shrinkB=3.0) if far else None))
    return anns


def _cond_handles(conds, cond_title):
    from matplotlib.lines import Line2D
    return ([Line2D([], [], linestyle='', marker=COND[c][1], color=COND[c][0], markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=0.6) for c in conds],
            [cond_title[c] for c in conds])


def dots(cells, encoders, heads, conds, metric, enc_title, head_title, cond_title, note=None, panel=PANEL):
    '''cells[(ver, kho, encoder, đầu)] = ô. Một ô hình mỗi đầu đọc; trục y = encoder (cùng thứ tự mọi ô và mọi hình); mỗi điều kiện một chấm (màu + hình);
       chấm = điểm trên cả test (Ver2: TB các hạt). Không khoảng tin cậy (xem docstring module).'''
    assert all(c in COND for c in conds), f'dots: dieu kien {[c for c in conds if c not in COND]} khong co mau (toi da 3 dieu kien — docstring module)'
    plt = _plt()
    n = len(encoders)
    fig, axes = plt.subplots(1, len(heads), figsize=(2.55 * len(heads) + 1.4, 0.32 * n + 1.9), sharey=True, sharex=True, squeeze=False)
    axes = axes[0]
    off = np.linspace(-0.2, 0.2, len(conds)) if len(conds) > 1 else np.zeros(1)
    for ax, h in zip(axes, heads):
        for i in range(n - 1):
            ax.axhline(i + 0.5, color=GRID, linewidth=0.5, zorder=0)
        for c, dy in zip(conds, off):
            col, mk = COND[c]
            pts = [(i, _val(cells, c, e, h, metric, panel)) for i, e in enumerate(encoders)]
            pts = [(i, v) for i, v in pts if v is not None]
            if pts:
                ax.plot([v for _, v in pts], [i + dy for i, _ in pts], linestyle='', marker=mk, color=col, markersize=4.8, markeredgecolor=SURFACE,
                        markeredgewidth=0.6, zorder=3)
        ax.set_title(head_title[h], loc='left')
        _clean(ax, 'x')
        ax.set_yticks(range(n))
        ax.set_yticklabels([enc_title[e] for e in encoders])
        ax.set_xlabel(metric_short(metric, panel))
    axes[0].set_ylim(n - 0.5, -0.5)
    _layout(fig, f'{metric_text(metric, panel)} trên test VinDr-CXR — mỗi ô một đầu đọc, mỗi hàng một encoder', note, legend=_cond_handles(conds, cond_title))
    return fig


def heat(cells, encoders, heads, conds, metric, enc_title, head_title, cond_title, note=None, panel=PANEL):
    '''Lưới encoder x đầu đọc của một số đo, mỗi điều kiện một ô, CÙNG thang màu; số trong mọi ô ('–' = chưa có lượt).'''
    plt = _plt()
    vals = [v for c in conds for e in encoders for h in heads for v in [_val(cells, c, e, h, metric, panel)] if v is not None]
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    fig, axes = plt.subplots(1, len(conds), figsize=(2.85 * len(conds) + 1.5, 0.33 * len(encoders) + 2.15), sharey=True, squeeze=False)
    axes = axes[0]
    cm = _cmap(SEQ, 'mot_sac')
    im = None
    for ax, c in zip(axes, conds):
        A = np.array([[np.nan if _val(cells, c, e, h, metric, panel) is None else _val(cells, c, e, h, metric, panel) for h in heads] for e in encoders])
        im = ax.imshow(np.ma.masked_invalid(A), cmap=cm, vmin=vmin, vmax=vmax, aspect='auto')
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    dark = (A[i, j] - vmin) / max(vmax - vmin, 1e-9) > 0.42      # chữ trắng từ nấc xanh giữa trở lên (đủ tương phản)
                    ax.text(j, i, _fmt(A[i, j]), ha='center', va='center', fontsize=FS['cell'], color='#ffffff' if dark else INK)
                else:
                    ax.text(j, i, '–', ha='center', va='center', fontsize=FS['cell'], color=MUTED)
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels([head_title[h] for h in heads])
        ax.set_yticks(range(len(encoders)))
        ax.set_yticklabels([enc_title[e] for e in encoders])
        ax.set_xticks(np.arange(0.5, len(heads) - 1, 1), minor=True)          # vạch ngăn chỉ GIỮA các ô (vạch ở mép ô để lộ 1 điểm ảnh của ảnh)
        ax.set_yticks(np.arange(0.5, len(encoders) - 1, 1), minor=True)
        ax.set_xlim(-0.5, len(heads) - 0.5)
        ax.set_ylim(len(encoders) - 0.5, -0.5)
        ax.set_title(cond_title[c], loc='left')
        _bare(ax)
    _layout(fig, f'{metric_text(metric, panel)} trên test VinDr-CXR — lưới encoder x đầu đọc', note,
            cbar=(im, metric_short(metric, panel)) if im is not None else None)
    return fig


def findings(cells, encoders, heads, cond, names, metric, enc_title, head_title, cond_title, note=None, panel='p7', per_key='p7_lop'):
    '''TỪNG dấu hiệu của một panel (mặc định 7 dấu hiệu Bảng 1a bài RAD-DINO) ở MỘT điều kiện: lưới các ô (mỗi ô một đầu đọc, 2 cột), hàng = encoder có số ở
       điều kiện này, cột = từng dấu hiệu theo `names` + "TB" (= số macro của panel ở ô đó, = trung bình các cột trước); số trong mọi ô ('–' = chưa có lượt);
       một thang màu một sắc chung cả hình. cells[(ver, kho, encoder, đầu)][per_key][metric] = list giá trị từng dấu hiệu (Ver2: TB các hạt). -> Figure | None.'''
    plt = _plt()
    rows = [e for e in encoders if any(per_key in (cells.get((cond[0], cond[1], e, h)) or {}) for h in heads)]
    if not rows:
        return None
    labels = list(names) + [f'TB {len(names)}']

    def vals(e, h):
        c = cells.get((cond[0], cond[1], e, h))
        if c is None or per_key not in c:
            return [np.nan] * len(labels)
        v = list(c[per_key][metric]) + [c['diem'][panel][metric]]
        assert len(v) == len(labels), (len(v), len(labels))
        return [float(x) if x == x else np.nan for x in v]
    A = {h: np.array([vals(e, h) for e in rows]) for h in heads}
    fin = np.concatenate([a[np.isfinite(a)] for a in A.values()])
    vmin, vmax = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
    ncol = 2 if len(heads) > 1 else 1
    nrow = int(np.ceil(len(heads) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * (0.52 * len(labels) + 0.3) + 1.7, nrow * (0.3 * len(rows) + 0.75) + 2.3), sharex=True, sharey=True,
                             squeeze=False)
    cm = _cmap(SEQ, 'mot_sac')
    im = None
    for ax, h in zip(axes.ravel(), heads):
        a = A[h]
        im = ax.imshow(np.ma.masked_invalid(a), cmap=cm, vmin=vmin, vmax=vmax, aspect='auto')
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if np.isfinite(a[i, j]):
                    dark = (a[i, j] - vmin) / max(vmax - vmin, 1e-9) > 0.42
                    ax.text(j, i, _fmt(a[i, j]), ha='center', va='center', fontsize=FS['cell'] - 0.5, color='#ffffff' if dark else INK)
                else:
                    ax.text(j, i, '–', ha='center', va='center', fontsize=FS['cell'], color=MUTED)
        ax.axvline(len(names) - 0.5, color=SURFACE, linewidth=4)           # tách cột TB khỏi các dấu hiệu
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha='right', rotation_mode='anchor')
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([enc_title[e] for e in rows])
        ax.set_xticks(np.arange(0.5, len(labels) - 1, 1), minor=True)
        ax.set_yticks(np.arange(0.5, len(rows) - 1, 1), minor=True)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_title(head_title[h], loc='left')
        _bare(ax)
    for ax in axes.ravel()[len(heads):]:
        ax.set_visible(False)
    _layout(fig, f'{metric_text(metric, panel)} từng dấu hiệu — {cond_title[cond]} (hàng: encoder, cột: dấu hiệu; mỗi ô một đầu đọc)', note,
            cbar=(im, metric_short(metric, panel)))
    return fig


def versus(cells, encoders, heads, x_cond, y_cond, metric, enc_title, head_title, cond_title, note=None, panel=PANEL):
    '''Mỗi đầu đọc một ô: trục x = số ở điều kiện x_cond, trục y = ở y_cond; mỗi chấm một encoder có cả hai; đường y = x; Spearman (mô tả) ở tiêu đề ô.
       Nhãn tên đặt không chồng (_labels). Không tính hiệu y - x (PLAN_89 §7.4: Ver1 và Ver2 chỉ đặt cạnh).'''
    from .compare import spearman
    plt = _plt()
    both = {h: [e for e in encoders if _val(cells, x_cond, e, h, metric, panel) is not None and _val(cells, y_cond, e, h, metric, panel) is not None]
            for h in heads}
    v = [_val(cells, c, e, h, metric, panel) for h in heads for e in both[h] for c in (x_cond, y_cond)]
    if not v:
        return None
    span = max(v) - min(v)
    lo, hi = min(v) - 0.08 * span - 0.005, max(v) + 0.08 * span + 0.005
    # Ô VUÔNG ĐẶT TAY (không set_aspect + tight_layout: tight_layout đo sai lề của ô có tỉ lệ cố định -> ghi chú đè nhãn trục, đo 15/11)
    n, s, left, gap, right, x_h, ax_t, top_h = len(heads), 2.45, 0.80, 0.25, 0.15, 0.50, 0.26, 0.36
    W = left + n * s + (n - 1) * gap + right
    note_h = _wrap(note, W)[1]
    H = note_h + x_h + s + ax_t + top_h
    fig = plt.figure(figsize=(W, H))
    axes = []
    for k, h in enumerate(heads):
        ax = fig.add_axes([(left + k * (s + gap)) / W, (note_h + x_h) / H, s / W, s / H], sharex=axes[0] if axes else None, sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=0.9, zorder=1)
        xs = [_val(cells, x_cond, e, h, metric, panel) for e in both[h]]
        ys = [_val(cells, y_cond, e, h, metric, panel) for e in both[h]]
        ax.plot(xs, ys, linestyle='', marker='o', color=HEAD_COLOR['lin1'], markersize=4.6, markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3)
        rho = spearman(xs, ys)
        ax.set_title(head_title[h] + (f'  ·  Spearman {rho:.2f} ({len(xs)} encoder)'.replace('.', ',') if rho == rho else ''), loc='left')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        _clean(ax, 'both')
        if k == 0:
            ax.set_ylabel(f'{metric_short(metric, panel)} · {cond_title[y_cond]}', fontsize=FS['tick'])
        else:
            ax.tick_params(labelleft=False)
    fig.text((left + (n * s + (n - 1) * gap) / 2) / W, (note_h + 0.06) / H, f'{metric_short(metric, panel)} · {cond_title[x_cond]}', ha='center', va='bottom',
             fontsize=FS['tick'], color=INK2)                                       # nhãn trục x chung (một lần cho cả hàng ô)
    _layout(fig, f'{metric_text(metric, panel)}: {cond_title[x_cond]} (trục x) so với {cond_title[y_cond]} (trục y) — mỗi chấm một encoder', note, tight=False)
    rnd = _renderer(fig)
    for ax, h in zip(axes, heads):
        _labels(ax, [_val(cells, x_cond, e, h, metric, panel) for e in both[h]], [_val(cells, y_cond, e, h, metric, panel) for e in both[h]],
                [enc_title[e] for e in both[h]], rnd)
    return fig


def _diff_heat(rows, d, heads, metrics, row_title, head_title, title, cbar_label, note, panel=PANEL):
    '''Bản đồ nhiệt PHÂN KỲ của hiệu d(hàng, đầu, số đo): mỗi số đo một ô, hàng = rows, cột = đầu đọc; số có dấu trong mọi ô ('·' = không tính được);
       thang đối xứng quanh 0 chung cho mọi ô. Dùng chung cho ref_diff (B − X) và cond_diff (điều kiện a − điều kiện b).'''
    plt = _plt()
    D = {m: np.array([[d(x, h, m) for h in heads] for x in rows]) for m in metrics}
    fin = np.concatenate([np.abs(A[np.isfinite(A)]) for A in D.values()])
    vm = float(fin.max()) if fin.size and fin.max() > 0 else 0.05
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.85 * len(metrics) + 1.5, 0.33 * len(rows) + 2.15), sharey=True, squeeze=False)
    axes = axes[0]
    cm = _cmap(DIV, 'phan_ky')
    im = None
    for ax, m in zip(axes, metrics):
        A = D[m]
        im = ax.imshow(np.ma.masked_invalid(A), cmap=cm, vmin=-vm, vmax=vm, aspect='auto')
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, _fmt(A[i, j], sign=True), ha='center', va='center', fontsize=FS['cell'], color='#ffffff' if abs(A[i, j]) / vm > 0.6 else INK)
                else:
                    ax.text(j, i, '·', ha='center', va='center', fontsize=FS['cell'], color=MUTED)
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels([head_title[h] for h in heads])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([row_title[x] for x in rows])
        ax.set_xticks(np.arange(0.5, len(heads) - 1, 1), minor=True)
        ax.set_yticks(np.arange(0.5, len(rows) - 1, 1), minor=True)
        ax.set_xlim(-0.5, len(heads) - 0.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_title(metric_short(m, panel), loc='left')
        _bare(ax)
    _layout(fig, title, note, cbar=(im, cbar_label))
    return fig


def ref_diff(cells, ref, encoders, heads, cond, metrics, enc_title, head_title, cond_title, note=None, panel=PANEL):
    '''Hiệu điểm ref − X (cùng điều kiện, cùng đầu) cho từng số đo: mỗi số đo một ô, hàng = X, cột = đầu đọc; chỉ tính khi hai ô cùng bộ nhãn test theo thứ
       tự ('·' = không tính được). Mô tả: không CI, không phải luật §7.4. -> Figure | None (chưa có ref hoặc không có X nào).'''
    def d(x, h, m):
        a, b = cells.get((cond[0], cond[1], ref, h)), cells.get((cond[0], cond[1], x, h))
        if a is None or b is None or a.get('khoa_nhan') != b.get('khoa_nhan'):
            return np.nan
        return a['diem'][panel][m] - b['diem'][panel][m]
    others = [x for x in encoders if x != ref and any(np.isfinite(d(x, h, METRICS[0])) for h in heads)]
    if not others:
        return None
    return _diff_heat(others, d, heads, metrics, enc_title, head_title,
                      f'Hiệu điểm {enc_title[ref]} − X trên {PANEL_TEXT[panel]} — {cond_title[cond]} (mỗi hàng một X, cùng đầu đọc)',
                      f'{enc_title[ref]} − X (xanh: {enc_title[ref]} cao hơn)', note, panel)


def cond_diff(cells, encoders, heads, cond_a, cond_b, metrics, enc_title, head_title, short_a, short_b, note=None, panel=PANEL):
    '''Hiệu điểm cond_a − cond_b của CÙNG encoder, cùng đầu (vd kho cắt giữa − kho đệm vuông, PLAN_89 §5.3): mỗi số đo một ô, hàng = encoder có cả hai
       điều kiện; chỉ tính khi hai ô cùng bộ nhãn test theo thứ tự ('·' = không tính được). Hai lượt học riêng, mỗi bên TB các hạt -> MÔ TẢ (không CI;
       luật của họ hình học ở bảng mAP21). short_a / short_b: tên ngắn của hai điều kiện cho tiêu đề + thang màu. -> Figure | None.'''
    def d(x, h, m):
        a, b = cells.get((cond_a[0], cond_a[1], x, h)), cells.get((cond_b[0], cond_b[1], x, h))
        if a is None or b is None or a.get('khoa_nhan') != b.get('khoa_nhan'):
            return np.nan
        return a['diem'][panel][m] - b['diem'][panel][m]
    rows = [x for x in encoders if any(np.isfinite(d(x, h, METRICS[0])) for h in heads)]
    if not rows:
        return None
    return _diff_heat(rows, d, heads, metrics, enc_title, head_title,
                      f'Hiệu điểm {short_a} − {short_b} trên {PANEL_TEXT[panel]} — cùng encoder, cùng đầu đọc (mỗi hàng một encoder)',
                      f'{short_a} − {short_b} (xanh: {short_a} cao hơn)', note, panel)


def lr_curves(curves, chosen, encoders, heads, store, metric, enc_title, head_title, title, note=None):
    '''curves[(encoder, kho)][đầu] = [(lr, AUROC val cuối, AP val cuối)]; chosen[(encoder, kho)][đầu] = bản chọn lr của lượt. Mỗi encoder một ô; trục x =
       lr (log); đường = số đo val lần chấm cuối pha 1 theo lr; chấm tròn = lr của luật (AUROC val), thoi rỗng = lr theo AP val.'''
    from matplotlib.lines import Line2D
    plt = _plt()
    col = 1 if metric == 'auc' else 2
    encs = [e for e in encoders if (e, store) in curves]
    if not encs:
        return None
    n = len(encs)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.75 * ncol + 0.6, 2.25 * nrow + 1.2), sharex=True, squeeze=False)
    for ax, e in zip(axes.ravel(), encs):
        for h in heads:
            pts = [p for p in curves[(e, store)].get(h, []) if np.isfinite(p[col])]
            if not pts:
                continue
            x, y = [p[0] for p in pts], [p[col] for p in pts]
            ax.plot(x, y, '-', color=HEAD_COLOR[h], linewidth=1.5)
            ch = chosen.get((e, store), {}).get(h)
            if ch:
                by = dict(zip(x, y))
                lr_rule, lr_ap = ch['auroc']['lr'], ch['ap']['lr']
                if lr_rule in by:
                    ax.plot(lr_rule, by[lr_rule], 'o', color=HEAD_COLOR[h], markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=0.9)
                if lr_ap in by and lr_ap != lr_rule:
                    ax.plot(lr_ap, by[lr_ap], 'D', markerfacecolor=SURFACE, markeredgecolor=HEAD_COLOR[h], markersize=5.5, markeredgewidth=1.3)
        ax.set_xscale('log')
        ax.set_title(enc_title[e], loc='left')
        _clean(ax, 'y')
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for j in range(ncol):                                     # nhãn trục x ở ô thấp nhất CÒN HIỆN của mỗi cột
        rows = [r for r in range(nrow) if r * ncol + j < n]
        if rows:
            axes[rows[-1], j].set_xlabel('lr')
            axes[rows[-1], j].tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel('AUROC val 28 nhãn' if metric == 'auc' else 'AP val 28 nhãn')
    hd = [Line2D([], [], color=HEAD_COLOR[h], linewidth=1.5) for h in heads]
    lb = [head_title[h] for h in heads]
    hd += [Line2D([], [], marker='o', color=MUTED, linestyle='', markersize=5.5),
           Line2D([], [], marker='D', markerfacecolor=SURFACE, markeredgecolor=MUTED, linestyle='', markersize=5.5, markeredgewidth=1.3)]
    lb += ['lr của luật (AUROC val)', 'lr theo AP val (khi khác)']
    _layout(fig, title, note, legend=(hd, lb))
    return fig
