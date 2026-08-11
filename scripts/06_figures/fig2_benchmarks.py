#!/usr/bin/env python3
"""Figure 2 — Quality stratification reshapes published benchmarks.

Panels:
  a   Method rank slope, all vs specific perturbations (Wei et al.)
  b   Top-ranked method on all vs specific perturbations, across 8 benchmark settings
  c   Per-method dataset-weighted mean performance across the three quality classes, on 4 genetic benchmarks
  d   Ceiling efficiency on specific perturbations (Wei and Ahlmann-Eltze, shared y-axis)
  e   Linear-model performance vs the amount of reliable training data

Outputs to figures/ and figures/panels/.
"""
import sys, os
THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, '..', '03_preprocess'))
from config import *  # noqa: F401,F403
from _paths import FIG_DIR, FIG_PANEL_DIR, save_fig, save_fig_panel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import wilcoxon

setup_style()
import matplotlib as _mpl
_mpl.rcParams['axes.labelsize']  = 9
_mpl.rcParams['axes.titlesize']  = 9
_mpl.rcParams['xtick.labelsize'] = 8
_mpl.rcParams['ytick.labelsize'] = 8
_mpl.rcParams['legend.fontsize'] = 8


# ── DATA ─────────────────────────────────────────────────────────────
wei_gen   = rename_triage(pd.read_csv(intermediate_path('wei_genetic_merged_2d.csv')))
wei_combo = rename_triage(pd.read_csv(intermediate_path('wei_genetic_combo_merged_2d.csv')))
SKIP_WEI = {'baseControl', 'trainMean', 'scFoundation'}


def _prep_ae(fname, val_col):
    d = rename_triage(pd.read_csv(intermediate_path(fname)))
    d = d[d.train == 'test']
    return d.groupby(['method', 'dataset', 'perturbation', 'triage_2d'],
                     as_index=False).agg(**{val_col: (val_col, 'mean')})


ae_a = _prep_ae('ae_panel_a_merged_2d.csv', 'r2_delta')
ae_d = _prep_ae('ae_doubles_merged_2d.csv', 'r2_delta')

ae_a_full = rename_triage(pd.read_csv(intermediate_path('ae_panel_a_merged_2d.csv')))
ae_a_full = ae_a_full[ae_a_full.train == 'test']
ae_a_full = ae_a_full.groupby(
    ['method', 'dataset', 'perturbation', 'triage_2d'], as_index=False).agg(
    r2_delta=('r2_delta', 'mean'),
    reliability=('reliability', 'first'),
    ceiling=('ceiling', 'first'))


# ════════════ Panel a — Rank slope (Wei) ═════════════════════════════
def _rank_slope(ax, df, metric, bl, pcol, skip, title):
    exc_all = ds_weighted_excess(df, metric, bl, pcol, skip, category=None)
    exc_gen = ds_weighted_excess(df, metric, bl, pcol, skip, category='Specific')
    rank_all = pd.Series(exc_all).rank(ascending=False).astype(int)
    rank_gen = pd.Series(exc_gen).rank(ascending=False).astype(int)
    methods = sorted(exc_gen.keys(), key=lambda m: rank_gen[m])
    n_m = len(methods)

    X_LEFT, X_RIGHT = 0.0, 0.45
    LABEL_PAD = 0.02

    for m in methods:
        ra = rank_all[m]; rg = rank_gen[m]; delta = ra - rg
        if abs(delta) >= 3:
            color = C_POS if delta > 0 else C_NEG
            lw, alpha = 1.8, 0.92
        else:
            color, lw, alpha = '#444444', 1.0, 0.85
        ax.plot([X_LEFT, X_RIGHT], [ra, rg], color=color, lw=lw, alpha=alpha,
                solid_capstyle='round', zorder=3)
        ax.scatter([X_LEFT, X_RIGHT], [ra, rg], color=color, s=24, zorder=5,
                   edgecolors='white', linewidth=0.4)
        ax.text(X_LEFT - LABEL_PAD, ra, f"{m} ({ra})",
                ha='right', va='center', fontsize=NM_LABEL,
                color=color, fontweight='bold')
        ax.text(X_RIGHT + LABEL_PAD, rg, f"({rg}) {m}",
                ha='left', va='center', fontsize=NM_LABEL,
                color=color, fontweight='bold')

    ax.set_xlim(X_LEFT - 0.40, X_RIGHT + 0.40)
    ax.set_ylim(n_m + 0.7, 0.3)
    ax.set_xticks([X_LEFT, X_RIGHT])
    ax.set_xticklabels(['All', 'Specific'], fontsize=NM_LABEL, fontweight='bold')
    ax.tick_params(left=False, labelleft=False)
    for s in ('left', 'bottom', 'top', 'right'):
        ax.spines[s].set_visible(False)
    ax.axvline(X_LEFT,  color='#e0e0e0', lw=0.5, zorder=0)
    ax.axvline(X_RIGHT, color='#e0e0e0', lw=0.5, zorder=0)
    ax.set_title(title, fontsize=NM_TITLE, pad=4)

    from matplotlib.lines import Line2D as _L
    handles = [
        _L([0], [0], color=C_POS, lw=1.8, label='Rise ≥3 ranks'),
        _L([0], [0], color=C_NEG, lw=1.8, label='Drop ≥3 ranks'),
        _L([0], [0], color='#444', lw=1.0, label='|Δ|<3'),
    ]
    ax.legend(handles=handles, fontsize=NM_LABEL, frameon=False,
              loc='lower center', bbox_to_anchor=(0.5, -0.20),
              ncol=3, handlelength=1.5, columnspacing=1.4, borderpad=0.0)


def panel_a_rank_wei(ax):
    _rank_slope(ax, wei_gen, 'pcc', 'trainMean', 'perturbation',
                SKIP_WEI, 'Wei et al. (PCC)')


# ════════════ Panel b — Top-method change table ══════════════════════
# Settings and skip sets live in config.BENCHMARK_SETTINGS so that Supplementary Figure 9, which repeats
# this comparison across metrics, ranks methods identically.
EVALS = BENCHMARK_SETTINGS
_top_records = []
for _name, _fname, _metric, _bl, _pcol, _skip in EVALS:
    _df = load_benchmark(_fname)
    _top_all, _exc_all = top_method(_df, _metric, True, _bl, _pcol, _skip, None)
    _top_gen, _exc_gen = top_method(_df, _metric, True, _bl, _pcol, _skip, 'Specific')
    _rank_gen = pd.Series(_exc_gen).rank(ascending=False).astype(int)
    _top_records.append(dict(
        benchmark_label=_name, n_methods=len(_exc_all),
        metric=_metric,
        top_all=_top_all, top_gen=_top_gen,
        changed=_top_all != _top_gen,
        original_top_new_rank=int(_rank_gen[_top_all]),
    ))
tops = pd.DataFrame(_top_records)


def panel_b_top_change(ax):
    df = tops.copy().reset_index(drop=True)
    n = len(df)
    edges  = [0.00, 0.33, 0.57, 0.785, 1.00]
    headers = ["Benchmark", "Metric", "Top on All", "Top on Specific"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(4)]

    ROW_H = 1.0
    HEADER_Y = 0.5
    BODY_Y0 = 1.5

    y_top = 0.0
    y_mid = ROW_H
    y_bot = ROW_H + n * ROW_H

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(y_bot, y_top)
    ax.axis('off')

    ax.plot([edges[0], edges[-1]], [y_top, y_top], '-', color='black',
            lw=1.0, clip_on=False)
    ax.plot([edges[0], edges[-1]], [y_mid, y_mid], '-', color='black',
            lw=0.6, clip_on=False)
    ax.plot([edges[0], edges[-1]], [y_bot, y_bot], '-', color='black',
            lw=1.0, clip_on=False)
    for x in edges[1:-1]:
        ax.plot([x, x], [y_top, y_bot], '-', color='#bbb', lw=0.4,
                clip_on=False)

    for cx, h in zip(centers, headers):
        ax.text(cx, HEADER_Y, h, ha='center', va='center',
                fontsize=NM_LABEL, fontweight='bold')

    for i, row in df.iterrows():
        y = BODY_Y0 + i
        ax.text(centers[0], y, row['benchmark_label'].replace('\n', ' '),
                ha='center', va='center', fontsize=NM_LABEL)
        ax.text(centers[1], y,
                METRIC_DISPLAY.get(row['metric'], row['metric']),
                ha='center', va='center', fontsize=NM_LABEL - 1,
                style='italic', color='#444')
        ax.text(centers[2], y, row['top_all'],
                ha='center', va='center', fontsize=NM_LABEL)
        gen_weight = 'bold' if row['changed'] else 'normal'
        ax.text(centers[3], y, row['top_gen'],
                ha='center', va='center', fontsize=NM_LABEL,
                fontweight=gen_weight)


# ════════════ Panel c — 1×4 slope plots ═══════════════════════════════
CLASSES = ['Specific', 'Shared', 'Unreliable']
CLASS_LABELS = ['Specific', 'Shared', 'Unreliable']
COL_HERO = '#1e8449'
# Readable names for the per-benchmark baselines, so the legend always names the curve drawn.
BASELINE_DISPLAY = {'trainMean': 'training-set mean', 'mean': 'training-set mean',
                    'additive_model': 'additive model'}
COL_BL = '#444444'
COL_OTHER = '#cccccc'

PANEL_B_DATA = [
    ('Wei: Genetic single', wei_gen,   'pcc',      'PCC-Δ (100 DEG)'),
    ('Wei: Genetic combo',  wei_combo, 'pcc',      'PCC-Δ (100 DEG)'),
    ('AE: Genetic single',  ae_a,      'r2_delta', 'PCC-Δ (1000 expr)'),
    ('AE: Genetic combo',   ae_d,      'r2_delta', 'PCC-Δ (1000 expr)'),
]

# Baseline and excess definition per benchmark, matching panel b so that the method highlighted here as
# "top on Specific" is the same one panel b names.
_PANEL_C_SPEC = {
    'Wei: Genetic single': ('trainMean',      {'baseControl', 'trainMean', 'scFoundation'}),
    'Wei: Genetic combo':  ('trainMean',      {'baseControl', 'trainMean', 'scFoundation'}),
    'AE: Genetic single':  ('mean',           set()),
    'AE: Genetic combo':   ('additive_model', {'no_change'}),
}


def _panel_c_highlights(label, df, val_col):
    """Return (baseline, top-on-specific, methods to skip) for one panel-c benchmark."""
    baseline, skip = _PANEL_C_SPEC[label]
    exc = ds_weighted_excess(df, val_col, baseline, 'perturbation', skip, 'Specific')
    hero = max(exc, key=exc.get) if exc else None
    return baseline, hero, skip


HIGHLIGHTS = {label: _panel_c_highlights(label, df, val_col)
              for label, df, val_col, _ in PANEL_B_DATA}


def panel_c_slope(ax, df, val_col, title, ylabel, baseline, hero, skip,
                  show_ylabel=True, show_legend=True):
    # Dataset-weighted mean, matching panels a and b: average within a dataset, then across datasets.
    # Wei et al. likewise report means. A pooled statistic would let the two large Replogle screens set
    # the value for every panel, and pooling is also what made the highlighted method here disagree with
    # the one panel b names.
    piv = (df.groupby(['method', 'triage_2d', 'dataset'])[val_col].mean()
             .groupby(['method', 'triage_2d']).mean().unstack('triage_2d'))
    for c in CLASSES:
        if c not in piv.columns:
            piv[c] = np.nan
    piv = piv[CLASSES]

    x = np.arange(3)
    for method, row in piv.iterrows():
        if method in skip or method in (baseline, hero):
            continue
        v = row.values
        if np.any(np.isnan(v)):
            continue
        ax.plot(x, v, '-', color=COL_OTHER, lw=0.8, alpha=0.6,
                marker='o', ms=2.5, zorder=2)

    if baseline in piv.index:
        v = piv.loc[baseline].values
        ax.plot(x, v, '--', color=COL_BL, lw=1.6, marker='s', ms=5,
                markerfacecolor='white', markeredgewidth=1.2,
                markeredgecolor=COL_BL, zorder=4,
                label=f'{BASELINE_DISPLAY.get(baseline, baseline)} (baseline)')

    if hero is not None and hero in piv.index:
        v = piv.loc[hero].values
        if not np.any(np.isnan(v)):
            ax.plot(x, v, '-', color=COL_HERO, lw=2.2, marker='o', ms=6,
                    markerfacecolor='white', markeredgewidth=1.5,
                    markeredgecolor=COL_HERO, zorder=5,
                    label='top method on Specific')

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS, fontsize=NM_LABEL)
    if show_ylabel:
        ax.set_ylabel(f'Mean {ylabel}', fontsize=NM_LABEL)
    ax.set_title(title, fontsize=NM_TITLE, pad=4, fontweight='bold')
    ax.tick_params(axis='y', labelsize=NM_LABEL)
    ax.set_xlim(-0.30, 2.30)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.5, 1.0])
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', linestyle=':', linewidth=0.4, alpha=0.4)
    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1.0),
                  fontsize=NM_LABEL, frameon=False,
                  handlelength=1.6, labelspacing=0.3, borderpad=0.2)


# ════════════ Panel d — Ceiling efficiency (Wei + AE) ════════════════
def _ce_one(ax, df, val_col, ceiling_col, suite_label, metric_label,
            baselines_to_skip, show_ylabel=True):
    info = df[df.triage_2d == 'Specific']
    methods = sorted([m for m in info.method.unique()
                      if m not in baselines_to_skip])
    res = {}
    for m in methods:
        sub = info[info.method == m]
        per_ds = {}
        for _, r in sub.iterrows():
            denom = r[ceiling_col]
            if denom and denom > 0.05:
                per_ds.setdefault(r.dataset, []).append(r[val_col] / denom)
        res[m] = (float(np.mean([np.mean(v) for v in per_ds.values()]))
                  if per_ds else 0)
    ser = pd.Series(res).sort_values(ascending=False)
    x = np.arange(len(ser))
    cols = [C_POS for _ in ser.values]
    cols[0] = '#1B7837'
    ax.bar(x, ser.values, width=0.78, color=cols, alpha=0.88,
           edgecolor='white', linewidth=0.5)
    ax.axhline(1.0, color='#666', ls=':', lw=1.0)
    for i, (m, v) in enumerate(ser.items()):
        ax.text(i, v + 0.025, f'{v:.2f}', ha='center', va='bottom',
                fontsize=NM_LABEL, fontweight='bold', color=cols[i])
    ax.set_xticks(x)
    ax.set_xticklabels(ser.index, fontsize=NM_LABEL, rotation=35, ha='right')
    ax.tick_params(axis='y', labelsize=NM_LABEL)
    if show_ylabel:
        ax.set_ylabel("Ceiling efficiency\non Specific perts",
                      fontsize=NM_LABEL)
    ax.set_title(suite_label, fontsize=NM_TITLE, pad=2, fontweight='bold')
    ax.set_ylim(0, 1.18)
    ax.set_xlim(-0.6, len(ser) - 0.4)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def panel_d_ce_wei(ax):
    _ce_one(ax, wei_gen, 'pcc', 'ceiling',
            'Wei: Genetic single (PCC-Δ, 100 DEG)', r'PCC-Δ / $\sqrt{\rho}$',
            baselines_to_skip=SKIP_WEI)


def panel_d_ce_ae(ax):
    SKIP_AE_BL = {'mean'}
    # Ahlmann-Eltze's 'r2_delta' column holds cor(pred - baseline, obs - baseline), a Pearson
    # correlation rather than a squared quantity, so its attenuation bound is sqrt(rho) = 'ceiling'.
    _ce_one(ax, ae_a_full, 'r2_delta', 'ceiling',
            'AE: Genetic single (PCC-Δ, 1000 expr)', r'PCC-Δ / $\sqrt{\rho}$',
            show_ylabel=False,
            baselines_to_skip=SKIP_AE_BL)


# ════════════ Panel e — Retraining ═══════════════════════════════════
def panel_e_reliable_quantity(ax):
    """Current panel e: performance vs amount of reliable training data, on the specific test pool
    (compact line plot, embedded). Standalone version: fig2e_reliable_quantity.py."""
    d = pd.read_csv(intermediate_path('retraining/reliable_quantity_sweep.csv'))
    d["quality_label"] = canonical_quality(d["quality_label"])
    d = d[d.quality_label == "Specific"]          # specific test pool
    # Genetic-single datasets only; see config.RETRAIN_SINGLE_DATASETS.
    d = d[d.dataset.isin(RETRAIN_SINGLE_DATASETS)]
    DS = sorted(d.dataset.unique())
    # Fractions read from the arms present in the sweep, so a changed sweep grid cannot silently drop points.
    FR = sorted(float(a.rsplit("_", 1)[-1]) for a in d.arm.unique() if a.startswith("reliable_sub_")) + [1.00]
    arm_for = lambda f: "reliable" if f == 1.0 else f"reliable_sub_{f:.2f}"
    # Both benchmarks score PCC-Δ; they differ in the gene set.
    METR = [("Wei PCC-Δ", "wei_pcc_delta_top100", True, "#1f6fb2"),
            ("Wei MSE", "wei_mse_top100", False, "#d1495b"),
            ("Wei common-DEGs", "wei_common_degs", True, "#2e8b57"),
            ("AE PCC-Δ", "ae_pearson_dlt_top1000", True, "#e6822e"),
            ("Systema centroid", "ca_specific", True, "#7a5195")]
    sizes = lambda arm: d[d.arm == arm].groupby("dataset")["n_train"].first(); allsz = sizes("all")
    perf = lambda arm, col: d[d.arm == arm].groupby("dataset")[col].mean()
    def rel(col, hib, arm):
        # Dataset-weighted mean of raw scores, then normalised to train-all (matches the standalone fig2e).
        a = perf("all", col).reindex(DS); v = perf(arm, col).reindex(DS)
        keep = a.notna() & v.notna()
        if not keep.any():
            return np.nan
        am, vm = a[keep].mean(), v[keep].mean()
        return (vm / am) if hib else (am / vm)
    xfrac = {f: float((sizes(arm_for(f)) / allsz).mean() * 100) for f in FR}
    xspec = float((sizes("specific") / allsz).mean() * 100); Xpct = xfrac[1.0]
    ax.axhline(1.0, color="#555", lw=0.8, ls="--", zorder=2)
    ax.axvspan(Xpct, 100, color="#bdbdbd", alpha=0.12, lw=0, zorder=0)
    for label, col, hib, c in METR:
        xs = [xfrac[f] for f in FR]; ys = [rel(col, hib, arm_for(f)) for f in FR]
        ax.plot(xs, ys, color=c, lw=1.2, marker="o", ms=2.6, zorder=4, label=label)
        ax.plot([Xpct, 100], [ys[-1], 1.0], color=c, lw=0.8, ls="--", alpha=0.5, zorder=3)
        ax.scatter([100], [1.0], color=c, marker="*", s=32, zorder=5, edgecolor="white", linewidth=0.4)
        ax.scatter([xspec], [rel(col, hib, "specific")], color=c, marker="s", s=12, zorder=5, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("% of all training perturbations used", fontsize=NM_LABEL)
    ax.set_ylabel("performance on specific test\n(relative to train-all)", fontsize=NM_LABEL)
    ax.set_xlim(0, 103); ax.set_xticks(list(range(0, 101, 20))); ax.tick_params(labelsize=NM_LABEL)
    ax.legend(loc="lower right", fontsize=NM_LABEL - 2, handlelength=1.2, labelspacing=0.25)
    # Matches the standalone fig2e title: the panel is the linear model, not the deep models.
    ax.set_title("Linear model: performance vs amount of reliable training data",
                 fontsize=NM_TITLE, pad=4, fontweight='bold')


# ══════════════════════════════════════════════════════════════════════
# COMBINED FIGURE (3 rows × 2 cols) — same as current main fig2
# ══════════════════════════════════════════════════════════════════════
FIGSIZE = (NM_FULL_W * 1.85, NM_FULL_W * 1.65)
fig = plt.figure(figsize=FIGSIZE)
outer = gridspec.GridSpec(3, 2, figure=fig,
                          height_ratios=[1.05, 0.95, 0.95],
                          width_ratios=[1.0, 1.0],
                          left=0.06, right=0.98, top=0.96, bottom=0.06,
                          hspace=0.42, wspace=0.28)

ax_a = fig.add_subplot(outer[0, 0]); panel_a_rank_wei(ax_a)
ax_b = fig.add_subplot(outer[0, 1]); panel_b_top_change(ax_b)

c_grid = outer[1, :].subgridspec(1, 4, wspace=0.25)
ax_c = [fig.add_subplot(c_grid[0, 0])]
for j in range(1, 4):
    ax_c.append(fig.add_subplot(c_grid[0, j], sharey=ax_c[0]))
for k, (ax, (label, df, val, metric)) in enumerate(zip(ax_c, PANEL_B_DATA)):
    bl, hero, skip = HIGHLIGHTS[label]
    show_legend = (k == 3)
    show_ylabel = (k == 0)
    panel_c_slope(ax, df, val, label, metric, bl, hero, skip,
                  show_ylabel=show_ylabel, show_legend=show_legend)
    if k > 0:
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', left=False)

d_grid = outer[2, 0].subgridspec(1, 2, width_ratios=[12, 6], wspace=0.10)
ax_d_wei = fig.add_subplot(d_grid[0, 0])
ax_d_ae  = fig.add_subplot(d_grid[0, 1], sharey=ax_d_wei)
panel_d_ce_wei(ax_d_wei)
panel_d_ce_ae(ax_d_ae)
ax_d_ae.tick_params(axis='y', labelleft=False)

ax_e = fig.add_subplot(outer[2, 1]); panel_e_reliable_quantity(ax_e)


def _label_at(ax, letter, x_offset=-0.040, y_offset=0.018):
    bb = ax.get_position()
    fig.text(bb.x0 + x_offset, bb.y1 + y_offset, letter,
             fontsize=14, fontweight='bold', va='top', ha='left',
             family='Arial')


_label_at(ax_a, 'a')
_label_at(ax_b, 'b')
_label_at(ax_c[0], 'c')
_label_at(ax_d_wei, 'd')
_label_at(ax_e, 'e')

save_fig(fig, 'fig2_combined')
plt.close(fig)


# ── Individual panels ────────────────────────────────────────────────
save_fig_panel(panel_a_rank_wei, 'fig2a_rank_wei',
              figsize=(NM_COL_W * 1.30, NM_COL_W * 1.20))
save_fig_panel(panel_b_top_change, 'fig2b_top_change',
              figsize=(NM_COL_W * 1.40, NM_COL_W * 0.85))
save_fig_panel(panel_d_ce_wei, 'fig2d_ce_wei',
              figsize=(NM_COL_W * 1.10, NM_COL_W * 0.85))
save_fig_panel(panel_d_ce_ae, 'fig2d_ce_ae',
              figsize=(NM_COL_W * 0.90, NM_COL_W * 0.85))
# Panel e standalone is exported by fig2e_reliable_quantity.py (-> figures/panels/).

for label, df, val, metric in PANEL_B_DATA:
    bl, hero, skip = HIGHLIGHTS[label]
    short = label.replace(': ', '_').replace(' ', '_').lower()

    def _make_c(_df=df, _val=val, _label=label, _metric=metric,
                _bl=bl, _hero=hero, _skip=skip):
        return lambda ax: panel_c_slope(ax, _df, _val, _label, _metric,
                                        _bl, _hero, _skip)

    save_fig_panel(_make_c(), f'fig2c_{short}',
                  figsize=(NM_COL_W * 1.10, NM_COL_W * 0.85))

print("✓ Figure 2 complete.")
