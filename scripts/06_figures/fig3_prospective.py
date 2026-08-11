#!/usr/bin/env python3
"""Figure 3 — Prospective experimental design from a one-parameter
reliability model.

Six panels (a-f):

  Row 1: a (Replogle K562 representative) | b (all 29 datasets overlay)
         | c (R² histogram, fit quality)
  Row 2: d (pilot accuracy curve, kneedle elbow at 28 cells)
         | e (28-cell pilot vs full-data scatter)
         | f (pilot vs cross-experiment transfer)

Outputs to figures/.
"""
import sys, os, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

warnings.filterwarnings("ignore")

THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, os.path.join(THIS_DIR, '..', '03_preprocess'))
from config import *  # noqa: F401,F403
from _paths import FIG_DIR, FIG_PANEL_DIR, save_fig, save_fig_panel  # noqa: E402

setup_style()


# ── DATA ─────────────────────────────────────────────────────────────
def load_curves(rdir, prefix, n_col, ctx, pert_id_col):
    dfs = []
    for f in sorted(Path(rdir).glob(f"{prefix}*.csv")):
        if "summary" in f.name or "metadata" in f.name:
            continue
        ds = f.stem.replace(prefix, "")
        df = pd.read_csv(f)
        df["dataset"] = ds
        df["context_type"] = ctx
        df["n_cells"] = df[n_col]
        # One curve per (perturbation, condition), matching the unit used in the fits table. A dataset
        # can profile the same perturbation in several conditions (Frangieh has three), and keying on
        # the perturbation alone would trace a single line through them.
        # Cellular files already key on `condition` (which encodes context|perturbation); genetic files
        # key on `perturbation` and need the condition appended to stay one curve per unit.
        if "condition" in df.columns and pert_id_col != "condition":
            df["pert_id"] = df[pert_id_col].astype(str) + "|" + df["condition"].astype(str)
        else:
            df["pert_id"] = df[pert_id_col]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


fits = pd.read_csv(str(FITS_DIR / 'fits_all_perturbations.csv'))
pilot_ext = pd.read_csv(str(FITS_DIR / 'pilot_validation_extended.csv'))

gen_curves = load_curves(str(RELIABILITY_GEN_DIR), "reliability_",
                         "n_ko_cells", "genetic", "perturbation")
cel_curves = load_curves(str(RELIABILITY_CEL_DIR), "reliability_cc_",
                         "n_stim_cells", "cellular", "condition")

RHO_COL = ("sb_from_median" if "sb_from_median" in gen_curves.columns
           else "sb_from_mean")
print(f"Using reliability column: {RHO_COL}")

# ── Project palette ─────────────────────────────────────────────────
PERT_COLOR     = "#4878CF"
CONTRAST_COLOR = "#E07A5F"
# Observed-median dots: deliberately NOT the dataset color, which is also used for the faint traces
# and the IQR band behind them; a dark navy separates the data points from their own background.
MEDIAN_LINE    = "#1B4F9C"   # dark blue; deliberately NOT the dataset color, which is
                             # also the faint traces and the IQR band sitting behind it
GREY_REF       = "#666666"


# ══════════════════════════════════════════════════════════════════════
# Panels a–c: representative datasets
# ══════════════════════════════════════════════════════════════════════
def panel_abc_dataset(ax, curves, fits_df, dataset, ctx, color,
                      show_ylabel=True):
    X_LO, X_HI = 8, 100          # axis range; the model curve is drawn over exactly this span
    fixed = curves[
        (curves["dataset"] == dataset) &
        (curves["n_mode"] == "fixed") &
        (curves["gene_mode"] == "all_genes")
    ]
    # Restrict to units that also have a parametric fit, so the traces, the median band, the model
    # line and the reported n all describe the same set of perturbations.
    _fitted = set(fits_df[(fits_df["dataset"] == dataset) &
                          (fits_df["context_type"] == ctx)]["pert_id"])
    fixed = fixed[fixed["pert_id"].isin(_fitted)]
    if len(fixed) == 0:
        return

    rng = np.random.default_rng(42)
    perts = fixed["pert_id"].unique()
    sample = rng.choice(perts, min(40, len(perts)), replace=False)
    for p in sample:
        pdata = fixed[fixed["pert_id"] == p].sort_values("n_half")
        ax.plot(2 * pdata["n_half"], pdata[RHO_COL],
                "-", color=color, alpha=0.18, lw=0.5, rasterized=True)

    agg = fixed.groupby("n_half").agg(
        med=(RHO_COL, "median"),
        q25=(RHO_COL, lambda x: np.nanpercentile(x, 25)),
        q75=(RHO_COL, lambda x: np.nanpercentile(x, 75)),
    ).reset_index()
    N = 2 * agg["n_half"].values

    ax.fill_between(N, agg["q25"], agg["q75"], alpha=0.20, color=color,
                    linewidth=0)
    # Observed median, drawn as a solid line under the dashed model curve.
    ax.plot(N, agg["med"].values, "-", color=MEDIAN_LINE, lw=1.4, zorder=4)

    ds_fits = fits_df[(fits_df["dataset"] == dataset) &
                      (fits_df["context_type"] == ctx)]
    n_perts = ds_fits["pert_id"].nunique() if len(ds_fits) > 0 else 0
    if n_perts > 0:
        tau2 = ds_fits["tau_sq"].median()
        # Start at the axis minimum: a path drawn outside the axes stays in the PDF/SVG behind a
        # clip mask, and reappears when the figure is converted to shapes.
        N_fine = np.linspace(X_LO, X_HI, 200)
        # rho is strictly monotonic in tau^2, so rho(N, median tau^2) is the median of the
        # per-perturbation model curves, not a fit to the median points.
        # Dashed, so the observed median stays visible where the two coincide.
        ax.plot(N_fine, rho_model(N_fine, tau2), color=CONTRAST_COLOR, lw=1.7, zorder=6,
                dashes=(5.5, 3.5), solid_capstyle="butt")

    ax.axhline(0.5, color="grey", ls=":", alpha=0.3, lw=0.5)
    ax.set_xlim(X_LO, X_HI)
    ax.set_xticks([8, 20, 40, 60, 80, 100])
    ax.set_title(f"{dataset.replace('_', ' ')} — median fit (n = {n_perts:,})",
                 pad=6, fontsize=NM_TITLE, fontweight="bold")
    ax.set_xlabel("Cells per perturbation (N)", fontsize=NM_LABEL)
    if show_ylabel:
        ax.set_ylabel("Reliability (ρ)", fontsize=NM_LABEL)
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(labelsize=NM_LABEL)
    ax.set_box_aspect(1)


def panel_d_overlay(ax):
    n_ds_drawn = 0
    for ds in fits["dataset"].unique():
        for src, ctx in ((gen_curves, "genetic"), (cel_curves, "cellular")):
            fixed = src[(src["dataset"] == ds) &
                        (src["n_mode"] == "fixed") &
                        (src["gene_mode"] == "all_genes")]
            if len(fixed) == 0:
                continue
            # Restrict to units that also have a parametric fit, matching panel a and Supplementary
            # Figure 7, so the drawn curves and the reported n describe the same perturbations.
            _fitted = set(fits[(fits["dataset"] == ds) &
                               (fits["context_type"] == ctx)]["pert_id"])
            fixed = fixed[fixed["pert_id"].isin(_fitted)]
            if len(fixed) == 0:
                continue
            agg = fixed.groupby("n_half")[RHO_COL].median().reset_index()
            ax.plot(2 * agg["n_half"], agg[RHO_COL], "-",
                    color=PERT_COLOR, alpha=0.55, lw=0.7)
            n_ds_drawn += 1
            break

    X_MIN, X_MAX = 8, 100
    N_fine = np.linspace(X_MIN, X_MAX, 200)
    ref_specs = [(0.001, ":"), (0.005, "--"), (0.02, "-."), (0.1, "-")]
    label_positions = []
    for tau2, ls in ref_specs:
        rho = rho_model(N_fine, tau2)
        ax.plot(N_fine, rho, ls=ls, color="black", lw=0.9, alpha=0.85,
                clip_on=True, zorder=4)
        label_positions.append((X_MAX, rho[-1], tau2))
    label_positions.sort(key=lambda t: t[1])
    min_sep = 0.08
    for i in range(1, len(label_positions)):
        if label_positions[i][1] - label_positions[i - 1][1] < min_sep:
            label_positions[i] = (label_positions[i][0],
                                  label_positions[i - 1][1] + min_sep,
                                  label_positions[i][2])
    from matplotlib.transforms import blended_transform_factory
    label_tf = blended_transform_factory(ax.transAxes, ax.transData)
    for lx, ly, tau2 in label_positions:
        ax.text(1.015, min(ly, 0.99), rf"$\tau^2$={tau2:g}",
                fontsize=NM_LABEL, color="black",
                ha="left", va="center", clip_on=False,
                transform=label_tf)

    ax.axhline(0.5, color=GREY_REF, ls=":", alpha=0.35, lw=0.5)
    ax.set_xlabel("Cells per perturbation (N)", fontsize=NM_LABEL)
    ax.set_ylabel("Reliability (ρ)", fontsize=NM_LABEL)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_xticks([8, 20, 40, 60, 80, 100])
    ax.tick_params(labelsize=NM_LABEL)
    # One fit per (dataset, perturbation); perturbation names recur across datasets, so count fits.
    n_fits = len(fits)
    ax.set_title(f"All {n_ds_drawn} datasets (n = {n_fits:,})",
                 pad=6, fontsize=NM_TITLE, fontweight="bold")
    ax.set_box_aspect(1)


def panel_e_fit_quality(ax):
    r2 = fits["fit_r2"].dropna().values
    r2 = r2[np.isfinite(r2)]
    # Show the full distribution so the annotated median and n describe every bar drawn.
    LO = float(np.floor(r2.min()))
    bins = np.linspace(LO, 1.001, 60)
    ax.hist(r2, bins=bins, color="#4878CF",
            alpha=0.6, edgecolor="white", linewidth=0.3)
    med = np.median(r2)
    ax.axvline(med, color="black", ls="--", lw=0.8, alpha=0.7)
    ax.text(0.03, 0.96,
            f"median R² = {med:.2f}\nn = {len(r2):,}",
            transform=ax.transAxes, fontsize=NM_LABEL,
            ha="left", va="top", color="black")
    ax.set_xlabel(r"Model R²", fontsize=NM_LABEL)
    ax.set_ylabel("Perturbations", fontsize=NM_LABEL)
    ax.set_title("Model fit quality", pad=6,
                 fontsize=NM_TITLE, fontweight="bold")
    ax.tick_params(labelsize=NM_LABEL)
    ax.set_xlim(LO, 1.003)
    ax.set_box_aspect(1)


# ══════════════════════════════════════════════════════════════════════
# Panels d, e, f: pilot validation
# ══════════════════════════════════════════════════════════════════════
def panel_f_accuracy(ax):
    stats = []
    for pN in sorted(pilot_ext["pilot_N"].unique()):
        sub = pilot_ext[
            (pilot_ext["pilot_N"] == pN) &
            (pilot_ext["tau_full"] > 0) &
            (pilot_ext["tau_pilot"] > 0)
        ]
        if len(sub) < 20:
            continue
        r = np.corrcoef(np.log10(sub["tau_full"]),
                        np.log10(sub["tau_pilot"]))[0, 1]
        fold = sub["nstar_pilot"] / sub["nstar_full"]
        w2x = (100 * ((fold >= 0.5) & (fold <= 2)).mean()
               if len(fold) > 0 else 0)
        stats.append({"pilot_N": pN, "tau_r": r, "within_2x": w2x,
                      "n": len(sub)})
    df = pd.DataFrame(stats)

    c1 = "#4878CF"
    ln1, = ax.plot(df["pilot_N"], df["tau_r"], "o-", color=c1, ms=2.5,
                   lw=1.0, zorder=3, markeredgecolor="white",
                   markeredgewidth=0.25)
    ax.set_ylabel(r"$\tau^2$ correlation (r)", color=c1, fontsize=NM_LABEL)
    ax.tick_params(axis="y", labelcolor=c1, labelsize=NM_LABEL)
    ax.tick_params(axis="x", labelsize=NM_LABEL)
    ax.set_ylim(0.55, 1.02)

    ax2 = ax.twinx()
    c2 = "#e8a838"
    ln2, = ax2.plot(df["pilot_N"], df["within_2x"], "D--", color=c2, ms=2.0,
                    lw=0.9, zorder=3, markeredgecolor="white",
                    markeredgewidth=0.25)
    ax2.set_ylabel("% perts within 2-fold", color=c2, fontsize=NM_LABEL)
    ax2.tick_params(axis="y", labelcolor=c2, labelsize=NM_LABEL)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.5)
    ax2.set_ylim(45, 102)

    def _kneedle_idx(x, y):
        x_n = (x - x.min()) / (x.max() - x.min())
        y_n = (y - y.min()) / (y.max() - y.min())
        return int(np.argmax(y_n - x_n))

    if len(df) >= 3:
        x_arr = df["pilot_N"].values
        idx_r = _kneedle_idx(x_arr, df["tau_r"].values)
        idx_w = _kneedle_idx(x_arr, df["within_2x"].values)
        ex_r, ey_r = int(x_arr[idx_r]), float(df["tau_r"].values[idx_r])
        ex_w, ey_w = int(x_arr[idx_w]), float(df["within_2x"].values[idx_w])
        ax.scatter([ex_r], [ey_r], s=42, color=c1, zorder=5,
                   edgecolor="black", linewidth=0.8)
        ax2.scatter([ex_w], [ey_w], s=42, color=c2, zorder=5,
                    edgecolor="black", linewidth=0.8, marker="D")
        ax.text(min(ex_r, ex_w) - 2, 0.97, "elbow",
                fontsize=NM_LABEL, color="black", style="italic",
                ha="right", va="top",
                transform=ax.get_xaxis_transform())

    ax.set_xlabel("Pilot size (cells per perturbation)", fontsize=NM_LABEL)
    ax.set_title("Prediction accuracy vs. pilot size",
                 pad=6, fontsize=NM_TITLE, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])

    ax.legend([ln1, ln2], [r"$\tau^2$ correlation", "% within 2-fold"],
              fontsize=NM_LABEL, loc="lower right",
              frameon=False, borderpad=0.3,
              handlelength=1.4, labelspacing=0.3)
    ax.set_box_aspect(1)


def panel_g_scatter(ax):
    pilot_N = 28
    sub = pilot_ext[
        (pilot_ext["pilot_N"] == pilot_N) &
        (pilot_ext["tau_full"] > 0) &
        (pilot_ext["tau_pilot"] > 0)
    ].copy()

    log_f = np.log10(sub["nstar_full"].values)
    log_p = np.log10(sub["nstar_pilot"].values)

    sc = None
    try:
        xy = np.vstack([log_f, log_p])
        density = gaussian_kde(xy)(xy)
        idx = density.argsort()
        sc = ax.scatter(log_f[idx], log_p[idx], c=density[idx], s=3,
                        cmap="viridis", alpha=0.3, rasterized=True,
                        edgecolors="none")
    except Exception:
        ax.scatter(log_f, log_p, s=2, alpha=0.08,
                   color="#4878CF", rasterized=True, edgecolors="none")

    lims = [min(log_f.min(), log_p.min()) - 0.15,
            max(log_f.max(), log_p.max()) + 0.15]
    ax.plot(lims, lims, "k-", lw=0.6, alpha=0.3)
    ax.fill_between(lims,
                    [l - np.log10(2) for l in lims],
                    [l + np.log10(2) for l in lims],
                    alpha=0.06, color="#4878CF", zorder=0)

    fold = sub["nstar_pilot"] / sub["nstar_full"]
    w2x = 100 * ((fold >= 0.5) & (fold <= 2)).mean()
    r = np.corrcoef(log_f, log_p)[0, 1]
    ax.text(0.04, 0.96,
            f"r = {r:.2f}\n{w2x:.0f}% within 2-fold",
            transform=ax.transAxes, fontsize=NM_LABEL, va="top",
            ha="left", color="black")

    ax.set_xlabel(r"$\log_{10}$(N*) — full data", fontsize=NM_LABEL)
    ax.set_ylabel(rf"$\log_{{10}}$(N*) — {pilot_N}-cell pilot",
                  fontsize=NM_LABEL)
    ax.set_title(f"{pilot_N}-cell pilot vs. full data (n = {len(sub):,})",
                 pad=6, fontsize=NM_TITLE, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.tick_params(labelsize=NM_LABEL)

    if sc is not None:
        cax = ax.inset_axes([1.05, 0.15, 0.035, 0.5])
        cbar = ax.figure.colorbar(sc, cax=cax)
        cbar.set_label("density", fontsize=NM_LABEL,
                       rotation=270, labelpad=8)
        cbar.outline.set_linewidth(0.4)
        cbar.set_ticks([sc.get_array().min(), sc.get_array().max()])
        cbar.set_ticklabels(["low", "high"])
        cbar.ax.tick_params(labelsize=NM_LABEL, length=2, pad=2)


def panel_h_transfer(ax):
    gen = fits[(fits["context_type"] == "genetic") &
               (fits["tau_sq"] > 0)].copy()
    gene_ds = gen.groupby("pert_id")["dataset"].nunique()
    shared = gene_ds[gene_ds >= 2].index
    gen_shared = gen[gen["pert_id"].isin(shared)]

    gene_log_taus = defaultdict(list)
    for _, row in gen_shared.iterrows():
        gene_log_taus[row["pert_id"]].append(
            np.log10(max(row["tau_sq"], 1e-8)))

    cross_diffs = []
    means_list, w_vars_list = [], []
    for gene, vals in gene_log_taus.items():
        if len(vals) >= 2:
            means_list.append(np.mean(vals))
            w_vars_list.append(np.var(vals, ddof=0))
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    cross_diffs.append(vals[i] - vals[j])

    cross_diffs = np.array(cross_diffs)

    between = np.var(means_list)
    within = np.mean(w_vars_list)
    icc = between / (between + within) if (between + within) > 0 else 0

    pilot_diffs = np.array([])
    if len(pilot_ext) > 0:
        sub = pilot_ext[
            (pilot_ext["pilot_N"] == 28) &
            (pilot_ext["tau_full"] > 0) &
            (pilot_ext["tau_pilot"] > 0)
        ]
        if len(sub) > 0:
            pilot_diffs = (
                np.log10(sub["tau_pilot"].clip(lower=1e-8).values) -
                np.log10(sub["tau_full"].clip(lower=1e-8).values)
            )

    bins = np.linspace(-4, 4, 65)

    pilot_w2x = (100 * (np.abs(pilot_diffs) <= np.log10(2)).mean()
                 if len(pilot_diffs) > 0 else 0)
    cross_w2x = 100 * (np.abs(cross_diffs) <= np.log10(2)).mean()

    if len(pilot_diffs) > 0:
        ax.hist(pilot_diffs.clip(-4, 4), bins=bins, density=True, alpha=0.55,
                color="#4878CF", edgecolor="white", linewidth=0.3,
                label=f"28-cell pilot ({pilot_w2x:.0f}%)", zorder=3)

    ax.hist(cross_diffs.clip(-4, 4), bins=bins, density=True, alpha=0.45,
            color="#C0392B", edgecolor="white", linewidth=0.3,
            label=f"cross-experiment ({cross_w2x:.0f}%)", zorder=2)

    ax.axvline(0, color="black", ls="-", lw=0.6, alpha=0.3)
    for boundary in [-np.log10(2), np.log10(2)]:
        ax.axvline(boundary, color="grey", ls=":", lw=0.6, alpha=0.4)

    ax.set_xlabel(r"$\Delta\,\log_{10}(\tau^2)$", fontsize=NM_LABEL)
    ax.set_ylabel("Density", fontsize=NM_LABEL)
    ax.set_title("Pilot vs. cross-experiment error",
                 pad=6, fontsize=NM_TITLE, fontweight="bold")
    ax.legend(fontsize=NM_LABEL, loc="upper left",
              bbox_to_anchor=(1.02, 1.0),
              frameon=False, borderpad=0.3, handlelength=1.2,
              title="within 2-fold", title_fontsize=NM_LABEL)
    ax.set_xlim(-4, 4)
    ax.tick_params(labelsize=NM_LABEL)
    ax.set_box_aspect(1)


# ══════════════════════════════════════════════════════════════════════
# COMBINED FIGURE — 2 rows × 3 cols
#   Row 1: a (Replogle K562) | b (overlay) | c (R² hist)
#   Row 2: d (pilot accuracy) | e (scatter) | f (cross-exp)
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(NM_FULL_W * 1.7, NM_FULL_W * 0.95))
gs = gridspec.GridSpec(
    2, 18, figure=fig,
    hspace=0.65, wspace=0.45,
    left=0.05, right=0.98, top=0.93, bottom=0.10
)

ax_a = fig.add_subplot(gs[0, 0:6])
panel_abc_dataset(ax_a, gen_curves, fits, "Replogle_K562essential",
                  "genetic", "#4878CF", show_ylabel=True)

ax_b = fig.add_subplot(gs[0, 6:12])
panel_d_overlay(ax_b)

ax_c = fig.add_subplot(gs[0, 12:18])
panel_e_fit_quality(ax_c)

ax_d = fig.add_subplot(gs[1, 0:6])
panel_f_accuracy(ax_d)

ax_e = fig.add_subplot(gs[1, 6:12])
panel_g_scatter(ax_e)

ax_f = fig.add_subplot(gs[1, 12:18])
panel_h_transfer(ax_f)

# Panel labels
fig.text(0.01, 0.96, 'a', fontsize=NM_PANEL, fontweight='bold',
         va='top', fontfamily='sans-serif')
fig.text(0.34, 0.96, 'b', fontsize=NM_PANEL, fontweight='bold', va='top')
fig.text(0.67, 0.96, 'c', fontsize=NM_PANEL, fontweight='bold', va='top')
fig.text(0.01, 0.49, 'd', fontsize=NM_PANEL, fontweight='bold', va='top')
fig.text(0.34, 0.49, 'e', fontsize=NM_PANEL, fontweight='bold', va='top')
fig.text(0.67, 0.49, 'f', fontsize=NM_PANEL, fontweight='bold', va='top')

save_fig(fig, 'fig3_combined')
plt.close(fig)


# ── Individual panels ────────────────────────────────────────────────
def _panel_replogle(ax):
    panel_abc_dataset(ax, gen_curves, fits, "Replogle_K562essential",
                      "genetic", "#4878CF")


save_fig_panel(_panel_replogle, 'fig3a_replogle_k562',
              figsize=(NM_COL_W, NM_COL_W * 0.78))
save_fig_panel(panel_d_overlay, 'fig3b_all_datasets',
              figsize=(NM_COL_W * 1.2, NM_COL_W * 0.78))
save_fig_panel(panel_e_fit_quality, 'fig3c_model_fit',
              figsize=(NM_COL_W, NM_COL_W * 0.78))
save_fig_panel(panel_f_accuracy, 'fig3d_pilot_accuracy',
              figsize=(NM_FULL_W * 0.55, NM_COL_W * 0.7))
save_fig_panel(panel_g_scatter, 'fig3e_pilot_scatter',
              figsize=(NM_COL_W * 1.1, NM_COL_W * 1.1))
save_fig_panel(panel_h_transfer, 'fig3f_pilot_vs_crossexp',
              figsize=(NM_FULL_W * 0.55, NM_COL_W * 0.7))

print("✓ Figure 3 complete.")

