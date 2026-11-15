# -*- coding: utf-8 -*-
"""Biểu đồ của notebook tổng hợp (matplotlib, ảnh tĩnh). Mỗi hàm nhận số đã tính (ô của evaluation/compare.py) và trả Figure — không đọc Drive.

Chọn dạng theo việc của dữ liệu, màu chọn sau cùng và ĐÃ CHẠY bộ kiểm bảng màu (dataviz validate_palette, nền sáng #fcfcfb, 15/11):
  - dots      so độ lớn + khoảng tin cậy giữa các encoder: biểu đồ chấm, MỖI ĐẦU ĐỌC MỘT Ô (4 đầu cùng một ô không qua được ngưỡng phân biệt màu cho mọi
              cặp); trong ô, tối đa 3 điều kiện = 3 màu đầu của bảng màu (mọi cặp: CVD ΔE 9,2; thường 24,0 — đạt) + khác hình chấm + chú giải
  - heat      lưới encoder x đầu đọc: bản đồ nhiệt MỘT sắc (xanh, nhạt -> đậm), có số trong ô (đây cũng là bảng)
  - versus    Ver1 so với Ver2 từng encoder: tán xạ một màu, nhãn tên ngay cạnh chấm, đường y = x (chỉ mô tả; không tính Ver2 - Ver1)
  - pairs     hiệu ghép cặp hàng - cột: bản đồ nhiệt PHÂN KỲ xanh <-> đỏ, giữa là xám trung tính; chỉ ghi số ở ô đạt dạng luật (danh nghĩa)
  - lr_curves số đo val theo lr: đường, 4 đầu = 4 màu theo thứ tự cố định của bảng màu (cặp kề nhau: đạt) + chú giải; mỗi số đo một hình (không hai trục)
Chữ dùng màu mực (không dùng màu của chuỗi); lưới mảnh, liền; không viền trên / phải. Màu xanh ngọc và vàng dưới 3:1 trên nền sáng -> luôn kèm bảng số."""
import numpy as np

INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
COND = {('v1', 'kcb'): ('#2a78d6', 'o'), ('v2', 'kcb'): ('#eb6834', 's'), ('v2', 'cb'): ('#1baf7a', 'D')}      # 3 màu đầu: đạt mọi cặp
HEAD_COLOR = {'lin1': '#2a78d6', 'lin4': '#eb6834', 'mldec': '#1baf7a', 'glori': '#eda100'}                   # thứ tự cố định: đạt cặp kề
SEQ = ('#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b')
DIV = ('#e34948', '#f0efec', '#2a78d6')          # âm (hàng kém cột) — giữa xám — dương (hàng hơn cột)


def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.labelsize': 8.5, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                                'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'axes.titlecolor': INK, 'figure.facecolor': SURFACE,
                                'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE, 'legend.frameon': False})
    return plt


def _clean(ax, grid='x'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelcolor=INK2)


def _cmap(colors, name):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, list(colors))


def dots(cells, encoders, heads, conds, enc_title, head_title, cond_title):
    '''cells[(ver, kho, encoder, đầu)] = ô. Một ô hình mỗi đầu đọc; trục y = encoder (cùng thứ tự mọi ô); chấm = mAP21, vạch = CI95 danh nghĩa.'''
    plt = _plt()
    fig, axes = plt.subplots(1, len(heads), figsize=(3.1 * len(heads) + 1.2, 0.42 * len(encoders) + 1.9), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    off = np.linspace(-0.24, 0.24, len(conds)) if len(conds) > 1 else [0.0]
    for ax, h in zip(axes, heads):
        for (c, dy) in zip(conds, off):
            col, mk = COND[c]
            pts = [(i, cells[(c[0], c[1], e, h)]) for i, e in enumerate(encoders) if (c[0], c[1], e, h) in cells]
            if not pts:
                continue
            y = np.array([i for i, _ in pts]) + dy
            x = np.array([v['diem']['p21']['ap'] for _, v in pts])
            lo, hi = np.array([v['ci21'][0] for _, v in pts]), np.array([v['ci21'][1] for _, v in pts])
            ax.hlines(y, lo, hi, color=col, linewidth=1.4, alpha=0.9)
            ax.plot(x, y, mk, color=col, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=0.8, label=cond_title[c])
        ax.set_title(head_title[h], loc='left')
        _clean(ax)
        ax.set_yticks(range(len(encoders)))
        ax.set_yticklabels([enc_title[e] for e in encoders])
        ax.set_xlabel('mAP 21 lớp (AP bậc thang)')
    axes[0].invert_yaxis()
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc='upper center', ncol=len(conds), bbox_to_anchor=(0.5, 0.995), handletextpad=0.3, columnspacing=1.6)
    fig.suptitle('mAP 21 lớp trên test VinDr — chấm: điểm (Ver2: trung bình 5 hạt); vạch: CI95 bootstrap theo ảnh (danh nghĩa)', y=0.02, va='bottom',
                 fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    return fig


def heat(cells, encoders, heads, conds, enc_title, head_title, cond_title):
    '''Lưới encoder x đầu đọc của mAP21, mỗi điều kiện một ô, CÙNG thang màu.'''
    plt = _plt()
    vals = [cells[k]['diem']['p21']['ap'] for k in cells if (k[0], k[1]) in conds and k[2] in encoders and k[3] in heads]
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    fig, axes = plt.subplots(1, len(conds), figsize=(3.5 * len(conds) + 1.6, 0.4 * len(encoders) + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    cm = _cmap(SEQ, 'mot_sac')
    im = None
    for ax, c in zip(axes, conds):
        A = np.array([[cells[(c[0], c[1], e, h)]['diem']['p21']['ap'] if (c[0], c[1], e, h) in cells else np.nan for h in heads] for e in encoders])
        im = ax.imshow(np.ma.masked_invalid(A), cmap=cm, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xlim(-0.5, len(heads) - 0.5)
        ax.set_ylim(len(encoders) - 0.5, -0.5)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    dark = (A[i, j] - vmin) / max(vmax - vmin, 1e-9) > 0.42      # chữ trắng từ nấc xanh giữa trở lên (đủ tương phản)
                    ax.text(j, i, f'{A[i, j]:.3f}'.replace('.', ','), ha='center', va='center', fontsize=8, color='#ffffff' if dark else INK)
                else:
                    ax.text(j, i, '–', ha='center', va='center', fontsize=8, color=MUTED)
        ax.set_xticks(range(len(heads)))
        ax.set_xticklabels([head_title[h] for h in heads], rotation=0)
        ax.set_yticks(range(len(encoders)))
        ax.set_yticklabels([enc_title[e] for e in encoders])
        ax.set_title(cond_title[c], loc='left')
        ax.tick_params(length=0, labelcolor=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(heads), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(encoders), 1), minor=True)
        ax.grid(which='minor', color=SURFACE, linewidth=2)
        ax.tick_params(which='minor', length=0)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    if im is not None:
        cb = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.015)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelcolor=INK2)
        cb.set_label('mAP 21 lớp', color=INK2)
    return fig


def versus(cells, encoders, heads, x_cond, y_cond, enc_title, head_title, cond_title):
    '''Mỗi đầu đọc một ô: trục x = mAP21 ở điều kiện x_cond, trục y = ở y_cond; mỗi chấm một encoder (ghi tên cạnh chấm); đường y = x.'''
    plt = _plt()
    fig, axes = plt.subplots(1, len(heads), figsize=(3.0 * len(heads) + 0.6, 3.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    both = [(e, h) for e in encoders for h in heads if (x_cond[0], x_cond[1], e, h) in cells and (y_cond[0], y_cond[1], e, h) in cells]
    v = [cells[(c[0], c[1], e, h)]['diem']['p21']['ap'] for e, h in both for c in (x_cond, y_cond)]
    lo, hi = (min(v) - 0.02, max(v) + 0.02) if v else (0.0, 1.0)
    for ax, h in zip(axes, heads):
        ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1.0)
        for e in encoders:
            if (e, h) not in both:
                continue
            x, y = cells[(x_cond[0], x_cond[1], e, h)]['diem']['p21']['ap'], cells[(y_cond[0], y_cond[1], e, h)]['diem']['p21']['ap']
            ax.plot(x, y, 'o', color=HEAD_COLOR['lin1'], markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=0.8)
            ax.annotate(enc_title[e], (x, y), xytext=(4, 3), textcoords='offset points', fontsize=7, color=INK2)
        ax.set_title(head_title[h], loc='left')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        _clean(ax, 'both')
    fig.supxlabel(f'mAP 21 lớp — {cond_title[x_cond]}', fontsize=8.5, color=INK2)          # một nhãn chung: nhãn từng ô dài, chồng lên nhau
    fig.supylabel(f'mAP 21 lớp — {cond_title[y_cond]}', fontsize=8.5, color=INK2)
    fig.tight_layout()
    return fig


def pairs(table, encoders, heads, enc_title, head_title, title):
    '''table[(đầu, hàng, cột)] = kết quả compare.diff(hàng, cột). Ô màu = hiệu hàng - cột (xanh: hàng hơn); số chỉ ghi ở ô đạt dạng luật (danh nghĩa).'''
    plt = _plt()
    vals = [abs(v['d']) for v in table.values()]
    vm = max(vals) if vals else 0.1
    fig, axes = plt.subplots(1, len(heads), figsize=(3.05 * len(heads) + 1.6, 0.36 * len(encoders) + 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    cm = _cmap(DIV, 'phan_ky')
    im = None
    for ax, h in zip(axes, heads):
        A = np.array([[table[(h, a, b)]['d'] if (h, a, b) in table else np.nan for b in encoders] for a in encoders])
        im = ax.imshow(np.ma.masked_invalid(A), cmap=cm, vmin=-vm, vmax=vm, aspect='auto')
        for i, a in enumerate(encoders):
            for j, b in enumerate(encoders):
                r = table.get((h, a, b))
                if r is not None and r['ket_luan'] != 'khong phan biet':
                    ax.text(j, i, f"{r['d']:+.2f}".replace('.', ','), ha='center', va='center', fontsize=6.5,
                            color='#ffffff' if abs(r['d']) / vm > 0.6 else INK)
        ax.set_xticks(range(len(encoders)))
        ax.set_xticklabels([enc_title[e] for e in encoders], rotation=90)
        ax.set_yticks(range(len(encoders)))
        ax.set_yticklabels([enc_title[e] for e in encoders])
        ax.set_title(head_title[h], loc='left')
        ax.tick_params(length=0, labelcolor=INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(encoders), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(encoders), 1), minor=True)
        ax.grid(which='minor', color=SURFACE, linewidth=1.5)
        ax.tick_params(which='minor', length=0)
    fig.suptitle(title, x=0.01, ha='left', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 0.93, 0.95))
    if im is not None:
        cb = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.015)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelcolor=INK2)
        cb.set_label('hiệu mAP 21 lớp (hàng − cột)', color=INK2)
    return fig


def lr_curves(curves, chosen, encoders, heads, store, metric, enc_title, head_title, title):
    '''curves[(encoder, kho)][đầu] = [(lr, AUROC val cuối, AP val cuối)]; chosen[(encoder, kho)][đầu] = bản chọn lr của lượt. Mỗi encoder một ô; trục x =
       lr (log); đường = số đo val lần chấm cuối pha 1 theo lr; chấm tròn = lr của luật (AUROC val), thoi rỗng = lr theo AP val.'''
    plt = _plt()
    col = 1 if metric == 'auc' else 2
    encs = [e for e in encoders if (e, store) in curves]
    if not encs:
        return None
    n = len(encs)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol + 0.5, 2.5 * nrow + 0.8), sharex=True, squeeze=False)
    for ax, e in zip(axes.ravel(), encs):
        for h in heads:
            pts = [p for p in curves[(e, store)].get(h, []) if np.isfinite(p[col])]
            if not pts:
                continue
            x, y = [p[0] for p in pts], [p[col] for p in pts]
            ax.plot(x, y, '-', color=HEAD_COLOR[h], linewidth=1.6, label=head_title[h])
            ch = chosen.get((e, store), {}).get(h)
            if ch:
                by = dict(zip(x, y))
                lr_rule, lr_ap = ch['auroc']['lr'], ch['ap']['lr']
                if lr_rule in by:
                    ax.plot(lr_rule, by[lr_rule], 'o', color=HEAD_COLOR[h], markersize=6, markeredgecolor=SURFACE, markeredgewidth=0.9)
                if lr_ap in by and lr_ap != lr_rule:
                    ax.plot(lr_ap, by[lr_ap], 'D', markerfacecolor=SURFACE, markeredgecolor=HEAD_COLOR[h], markersize=6, markeredgewidth=1.4)
        ax.set_xscale('log')
        ax.set_title(enc_title[e], loc='left')
        _clean(ax, 'y')
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('lr')
    for ax in axes[:, 0]:
        ax.set_ylabel('AUROC val (28 cột)' if metric == 'auc' else 'AP val (28 cột)')
    hd, lb = axes[0, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    hd += [Line2D([], [], marker='o', color=MUTED, linestyle='', markersize=6), Line2D([], [], marker='D', markerfacecolor=SURFACE, markeredgecolor=MUTED,
                                                                                         linestyle='', markersize=6, markeredgewidth=1.4)]
    lb += ['lr của luật (AUROC val)', 'lr theo AP val (khi khác)']
    top = 1 - 0.62 / fig.get_figheight()                     # dải trên: tiêu đề (trên) rồi chú giải (dưới) — không đè nhau ở mọi chiều cao hình
    fig.suptitle(title, x=0.01, y=0.995, ha='left', va='top', fontsize=9.5)
    fig.legend(hd, lb, loc='upper center', ncol=len(lb), bbox_to_anchor=(0.5, 1 - 0.27 / fig.get_figheight()), handletextpad=0.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig
