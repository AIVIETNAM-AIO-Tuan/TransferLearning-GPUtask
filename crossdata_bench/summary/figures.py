# -*- coding: utf-8 -*-
"""Hình của notebook `93t` — MỖI BỘ MỘT BỘ HÌNH (tiêu đề mang tên bộ). Hàm nhận `bo` (kết quả ô C4 + C6 của một bộ) và trả Figure; không đọc Drive.

Bảng màu + bố cục dùng lại của vindr_bench.evaluation.figures @ c81af62 (đã chạy bộ kiểm bảng màu dataviz, nền sáng, 15/11): cặp màu đầu xanh #2a78d6 /
cam #eb6834 (đạt mọi cặp) cho "đối chứng" / "B"; một sắc SEQ cho lưới; phân kỳ DIV (đỏ — xám — xanh) cho hiệu; 4 màu đầu đọc HEAD_COLOR (đạt cặp kề).
Hàm vẽ của 89 ghi cứng "VinDr-CXR" / "28 nhãn" trong tiêu đề nên phần này có hàm riêng, chỉ mượn _plt / _layout / _clean / _bare / _cmap / _fmt.

    dots_ci      một số đo: mỗi đầu đọc một ô, hàng = encoder (cùng thứ tự mọi hình của bộ), chấm = TB 5 hạt, vạch = CI95 danh nghĩa (bootstrap theo cụm)
    heat3        lưới encoder × đầu của ba số đo (mỗi số đo một ô, thang màu RIÊNG từng ô — số trong ô là số đo; * = lr ở biên lưới)
    family       họ B − X: vạch = khoảng ĐỒNG THỜI max-T 95 %, chấm đặc = đạt luật (B hơn / B kém), chấm rỗng = không phân biệt; đường 0 + ±0,01.
                 Điều khoản trần / họ chưa đủ: mọi chấm rỗng, chú giải ghi rõ (không phán quyết)
    diff_heat    hiệu điểm B − X của ba số đo (mô tả, không CI): phân kỳ, thang đối xứng chung
    lr_curves    đường dò lr (val cuối pha 1, hạt 0) theo lr, 4 đầu = 4 màu; chấm tròn = lr của luật, thoi rỗng = lr theo AP val (khi khác)"""
import numpy as np

from .. import config
from ..evaluation.bootstrap import METRICS

M_TEXT = {'ap': 'AP bậc thang', 'auprc_thang': 'AUPRC hình thang', 'auroc': 'AUROC'}
SLUG = {'ap': 'ap', 'auprc_thang': 'auprc', 'auroc': 'auroc'}
REF_COLOR, X_COLOR = '#eb6834', '#2a78d6'


def _fg():
    from vindr_bench.evaluation import figures as FG
    return FG


def _heads():
    from vindr_bench.protocols.ours import config as C
    return C.HEAD_KINDS, C.TITLES


def order(bo):
    '''Thứ tự hàng dùng chung cho mọi hình của một bộ: AP (số của luật) cao nhất qua 4 đầu, giảm dần.'''
    encs = [e for e in config.ALL_ROWS if any(k[0] == e and k[2] == 'luat' for k in bo['cells'])]       # 0.6.0: + hàng thêm (ASG)
    best = {e: max(c['diem']['ap'] for k, c in bo['cells'].items() if k[0] == e and k[2] == 'luat') for e in encs}
    return sorted(encs, key=lambda e: -best[e])


def _title(bo):
    return config.dataset(bo['name']).title


def dots_ci(bo, metric, note=None):
    from matplotlib.lines import Line2D
    FG = _fg()
    plt = FG._plt()
    hk, ht = _heads()
    encs = order(bo)
    n = len(encs)
    fig, axes = plt.subplots(1, len(hk), figsize=(2.55 * len(hk) + 1.4, 0.32 * n + 1.9), sharey=True, sharex=True, squeeze=False)
    axes = axes[0]
    for ax, h in zip(axes, hk):
        for i in range(n - 1):
            ax.axhline(i + 0.5, color=FG.GRID, linewidth=0.5, zorder=0)
        for i, e in enumerate(encs):
            c = bo['cells'].get((e, h, 'luat'))
            if c is None:
                continue
            col = REF_COLOR if e == config.REFERENCE else X_COLOR
            lo, hi = c['ci'][metric]
            if lo == lo and hi == hi:
                ax.plot([lo, hi], [i, i], '-', color=col, linewidth=1.6, solid_capstyle='round', zorder=2, gid='khoang')
            ax.plot([c['diem'][metric]], [i], linestyle='', marker='o', color=col, markersize=4.8, markeredgecolor=FG.SURFACE, markeredgewidth=0.6, zorder=3,
                    gid='diem')
        ax.set_title(ht[h], loc='left')
        FG._clean(ax, 'x')
        ax.set_yticks(range(n))
        ax.set_yticklabels([config.ROW_TITLES[e] for e in encs])
        ax.set_xlabel(M_TEXT[metric])
    axes[0].set_ylim(n - 0.5, -0.5)
    hd = [Line2D([], [], marker='o', color=c, linestyle='-', linewidth=1.6, markersize=5.0, markeredgecolor=FG.SURFACE, markeredgewidth=0.6)
          for c in (REF_COLOR, X_COLOR)]
    FG._layout(fig, f'{M_TEXT[metric]} trên test {_title(bo)} — chấm = TB 5 hạt, vạch = CI95 danh nghĩa (bootstrap theo cụm)', note,
               legend=(hd, [config.ROW_TITLES[config.REFERENCE], 'đối chứng']))
    return fig


def heat3(bo, note=None):
    FG = _fg()
    plt = FG._plt()
    hk, ht = _heads()
    encs = order(bo)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(2.85 * len(METRICS) + 1.5, 0.33 * len(encs) + 2.15), sharey=True, squeeze=False)
    cm = FG._cmap(FG.SEQ, 'mot_sac')
    for ax, m in zip(axes[0], METRICS):
        A = np.array([[bo['cells'][(e, h, 'luat')]['diem'][m] if (e, h, 'luat') in bo['cells'] else np.nan for h in hk] for e in encs], dtype=float)
        fin = A[np.isfinite(A)]
        vmin, vmax = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
        ax.imshow(np.ma.masked_invalid(A), cmap=cm, vmin=vmin, vmax=vmax, aspect='auto')
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    dark = (A[i, j] - vmin) / max(vmax - vmin, 1e-9) > 0.42
                    edge = bo['cells'][(encs[i], hk[j], 'luat')]['bien']
                    ax.text(j, i, FG._fmt(A[i, j]) + ('*' if edge else ''), ha='center', va='center', fontsize=FG.FS['cell'], color='#ffffff' if dark else FG.INK)
                else:
                    ax.text(j, i, '–', ha='center', va='center', fontsize=FG.FS['cell'], color=FG.MUTED)
        ax.set_xticks(range(len(hk)))
        ax.set_xticklabels([ht[h] for h in hk])
        ax.set_yticks(range(len(encs)))
        ax.set_yticklabels([config.ROW_TITLES[e] for e in encs])
        ax.set_xticks(np.arange(0.5, len(hk) - 1, 1), minor=True)
        ax.set_yticks(np.arange(0.5, len(encs) - 1, 1), minor=True)
        ax.set_xlim(-0.5, len(hk) - 0.5)
        ax.set_ylim(len(encs) - 0.5, -0.5)
        ax.set_title(f'{M_TEXT[m]} (thang màu riêng)', loc='left')
        FG._bare(ax)
    FG._layout(fig, f'Test {_title(bo)} — lưới encoder × đầu đọc, TB 5 hạt (* = lr của luật ở biên lưới 13 lr)', note)
    return fig


def family(bo, fam, note=None, rule='PLAN_93 §5', name='B − X', desc_text='MÔ TẢ — điều khoản trần (đối chứng tốt nhất ≥ 0,98): không phán quyết'):
    '''0.6.0: hàng = các đối chứng CÓ trong họ (họ 32 phép: 8 hàng như cũ; họ riêng: ASG); rule / name / desc_text cho tiêu đề + chú giải.'''
    from matplotlib.lines import Line2D
    FG = _fg()
    plt = FG._plt()
    hk, ht = _heads()
    rows = [x for x in order(bo) if x != config.REFERENCE and any(r['doi_chung'] == x for r in fam['phep_so'])]
    items = {(r['doi_chung'], r['dau']): r for r in fam['phep_so']}
    if not items:
        return None
    decide = fam['du'] and not fam['mo_ta']
    n = len(rows)
    fig, axes = plt.subplots(1, len(hk), figsize=(2.55 * len(hk) + 1.4, 0.42 * n + 1.9), sharey=True, sharex=True, squeeze=False)
    axes = axes[0]
    for ax, h in zip(axes, hk):
        for i in range(n - 1):
            ax.axhline(i + 0.5, color=FG.GRID, linewidth=0.5, zorder=0)
        ax.axvline(0.0, color=FG.MUTED, linewidth=0.9, zorder=1)
        for s in (-config.RULE_DELTA, config.RULE_DELTA):
            ax.axvline(s, color=FG.AXIS, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
        for i, x in enumerate(rows):
            r = items.get((x, h))
            if r is None:
                continue
            lo, hi = r['ci_dong_thoi'] or r['ci_danh_nghia']
            ax.plot([lo, hi], [i, i], '-', color=FG.ONE_HUE, linewidth=2.0, solid_capstyle='round', zorder=2, gid='khoang')
            filled = decide and r['ket_luan'] in ('hon', 'kem')
            if filled:
                ax.plot([r['d']], [i], linestyle='', marker='o', color=FG.ONE_HUE, markersize=6.0, markeredgecolor=FG.SURFACE, markeredgewidth=0.9, zorder=4, gid='dat')
            else:
                ax.plot([r['d']], [i], linestyle='', marker='o', markerfacecolor=FG.SURFACE, markeredgecolor=FG.ONE_HUE, markersize=5.4, markeredgewidth=1.4,
                        zorder=4, gid='chua')
        ax.set_title(ht[h], loc='left')
        FG._clean(ax, 'x')
        ax.set_yticks(range(n))
        ax.set_yticklabels([config.ROW_TITLES[x] for x in rows])
        ax.set_xlabel(f'{config.ROW_TITLES[config.REFERENCE]} − X (AP bậc thang)')
    axes[0].set_ylim(n - 0.5, -0.5)
    full = Line2D([], [], linestyle='', marker='o', color=FG.ONE_HUE, markersize=6.0, markeredgecolor=FG.SURFACE, markeredgewidth=0.9)
    hollow = Line2D([], [], linestyle='', marker='o', markerfacecolor=FG.SURFACE, markeredgecolor=FG.ONE_HUE, markersize=5.4, markeredgewidth=1.4)
    bar = Line2D([], [], color=FG.ONE_HUE, linewidth=2.0)
    if decide:
        legend = ([full, hollow, bar], ['đạt luật (B hơn / B kém)', 'không phân biệt', f'khoảng đồng thời 95 % (max-T, q = {fam["q"]:.2f})'])
        head = f'Họ {fam["so_phep"]} phép {name} trên test {_title(bo)} — luật {rule}'
    elif fam['mo_ta'] and fam['du']:
        legend = ([hollow, bar], [desc_text, f'khoảng đồng thời 95 % (max-T, q = {fam["q"]:.2f})'])
        head = f'Họ {fam["so_phep"]} phép {name} trên test {_title(bo)} — CHỈ MÔ TẢ'
    else:
        legend = ([hollow, bar], ['họ chưa đủ: chưa áp luật', 'CI95 danh nghĩa'])
        head = f'{name} trên test {_title(bo)} — họ chưa đủ ({fam["co"]} / {fam["so_phep"]} phép)'
    FG._layout(fig, head, note, legend=legend)
    return fig


def diff_heat(bo, note=None):
    FG = _fg()
    plt = FG._plt()
    hk, ht = _heads()
    ref = config.REFERENCE
    rows = [x for x in order(bo) if x != ref]

    def d(x, h, m):
        a, b = bo['cells'].get((ref, h, 'luat')), bo['cells'].get((x, h, 'luat'))
        return np.nan if a is None or b is None else a['diem'][m] - b['diem'][m]
    D = {m: np.array([[d(x, h, m) for h in hk] for x in rows], dtype=float) for m in METRICS}
    fin = np.concatenate([np.abs(A[np.isfinite(A)]) for A in D.values()])
    if not fin.size:
        return None
    vm = float(fin.max()) if fin.max() > 0 else 0.05
    fig, axes = plt.subplots(1, len(METRICS), figsize=(2.85 * len(METRICS) + 1.5, 0.33 * len(rows) + 2.15), sharey=True, squeeze=False)
    cm = FG._cmap(FG.DIV, 'phan_ky')
    im = None
    for ax, m in zip(axes[0], METRICS):
        A = D[m]
        im = ax.imshow(np.ma.masked_invalid(A), cmap=cm, vmin=-vm, vmax=vm, aspect='auto')
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isfinite(A[i, j]):
                    ax.text(j, i, FG._fmt(A[i, j], sign=True), ha='center', va='center', fontsize=FG.FS['cell'], color='#ffffff' if abs(A[i, j]) / vm > 0.6 else FG.INK)
                else:
                    ax.text(j, i, '·', ha='center', va='center', fontsize=FG.FS['cell'], color=FG.MUTED)
        ax.set_xticks(range(len(hk)))
        ax.set_xticklabels([ht[h] for h in hk])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([config.ROW_TITLES[x] for x in rows])
        ax.set_xticks(np.arange(0.5, len(hk) - 1, 1), minor=True)
        ax.set_yticks(np.arange(0.5, len(rows) - 1, 1), minor=True)
        ax.set_xlim(-0.5, len(hk) - 0.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_title(M_TEXT[m], loc='left')
        FG._bare(ax)
    FG._layout(fig, f'Hiệu điểm {config.ROW_TITLES[ref]} − X trên test {_title(bo)} — MÔ TẢ (không CI; luật ở hình họ B − X)', note,
               cbar=(im, f'{config.ROW_TITLES[ref]} − X (xanh: B cao hơn)'))
    return fig


def lr_curves(bo, metric, note=None):
    '''metric: 'auc' | 'ap' (số đo val lần chấm cuối pha 1 của hạt 0, như đường dò lr của 89).'''
    from matplotlib.lines import Line2D
    FG = _fg()
    plt = FG._plt()
    hk, ht = _heads()
    col = 1 if metric == 'auc' else 2
    encs = [e for e in order(bo) if e in bo['duong_lr']]
    if not encs:
        return None
    n = len(encs)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol + 0.6, 2.15 * nrow + 1.2), sharex=True, squeeze=False)
    for ax, e in zip(axes.ravel(), encs):
        for h in hk:
            pts = [p for p in bo['duong_lr'][e].get(h, []) if np.isfinite(p[col])]
            if not pts:
                continue
            x, y = [p[0] for p in pts], [p[col] for p in pts]
            ax.plot(x, y, '-', color=FG.HEAD_COLOR[h], linewidth=1.5)
            ch = bo['chon'].get(e, {}).get(h)
            if ch:
                by = dict(zip(x, y))
                lr_rule, lr_ap = ch['auroc']['lr'], ch['ap']['lr']
                if lr_rule in by:
                    ax.plot(lr_rule, by[lr_rule], 'o', color=FG.HEAD_COLOR[h], markersize=5.5, markeredgecolor=FG.SURFACE, markeredgewidth=0.9)
                if lr_ap in by and lr_ap != lr_rule:
                    ax.plot(lr_ap, by[lr_ap], 'D', markerfacecolor=FG.SURFACE, markeredgecolor=FG.HEAD_COLOR[h], markersize=5.5, markeredgewidth=1.3)
        ax.set_xscale('log')
        ax.set_title(config.ROW_TITLES[e], loc='left')
        FG._clean(ax, 'y')
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for j in range(ncol):
        rr = [r for r in range(nrow) if r * ncol + j < n]
        if rr:
            axes[rr[-1], j].set_xlabel('lr')
            axes[rr[-1], j].tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel('AUROC val' if metric == 'auc' else 'AP val (bậc thang)')
    hd = [Line2D([], [], color=FG.HEAD_COLOR[h], linewidth=1.5) for h in hk]
    lb = [ht[h] for h in hk]
    hd += [Line2D([], [], marker='o', color=FG.MUTED, linestyle='', markersize=5.5),
           Line2D([], [], marker='D', markerfacecolor=FG.SURFACE, markeredgecolor=FG.MUTED, linestyle='', markersize=5.5, markeredgewidth=1.3)]
    lb += ['lr của luật (AUROC val)', 'lr theo AP val (khi khác)']
    FG._layout(fig, f'{_title(bo)}: {"AUROC" if metric == "auc" else "AP"} val (lần chấm cuối pha 1, hạt 0) theo lr — chỗ chọn lr', note, legend=(hd, lb))
    return fig
