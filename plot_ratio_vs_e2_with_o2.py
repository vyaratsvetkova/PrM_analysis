"""
Produces two plots:
  ratio_vs_E2_with_O2_curves.png     — one point per E2 with uncertainties  + .list
  ratio_vs_E2_with_O2_allpoints.png  — all individual points, no error bars  + .list
Reads:  $PRM_REDUCED_DIR/results_e2_scan.csv
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import core as f

O2_PPB_LIST = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 1.0, 1.5, 2.0)

RC = {"font.size": 13, "axes.labelsize": 15,
      "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11}


def _write_list(png_path: str, filenames):
    list_path = png_path.replace(".png", ".list")
    with open(list_path, "w") as fh:
        for fn in filenames:
            fh.write(f"{fn}\n")


def _theory_curves(ax):
    E2_grid = np.linspace(0.0, 500.0, 600)
    for i, o2 in enumerate(O2_PPB_LIST):
        theory = f.predicted_survival_QA_over_QC(
            E2_grid, O2_ppb=o2, T_K=config.T_K,
            csv_name=config.RESULTS_E2_SCAN_CSV)
        lbl = f"$n$ = {o2} ppb" if i == 0 else f"{o2}"
        ax.plot(E2_grid, theory, label=lbl)


def main():
    red_dir = f.environment("PRM_REDUCED_DIR")
    df = f._read_results_df(config.RESULTS_E2_SCAN_CSV)
    filenames = df["filename"].dropna().tolist()

    # ----------------------------------------------------------------
    # Plot 1: grouped mean ± uncertainty
    # ----------------------------------------------------------------
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    _theory_curves(ax)

    eb_kw = dict(fmt="o", markersize=4, capsize=2, elinewidth=1,
                 color="black", ecolor="black", markeredgecolor="black", zorder=10)

    if "group" in df.columns:
        df["group"] = df["group"].astype(str)
        for _, dfg in df.groupby("group", sort=True):
            xs, ys, ysig = f._grouped_mean_and_err(dfg, "E2", "ratio", "ratio_err")
            ax.errorbar(xs, ys, yerr=ysig, **eb_kw)
    else:
        xs, ys, ysig = f._grouped_mean_and_err(df, "E2", "ratio", "ratio_err")
        ax.errorbar(xs, ys, yerr=ysig, **eb_kw)

    ax.set_xlabel(r"Drift Field $E_2$ [V/cm]")
    ax.set_ylabel(r"$Q_A/Q_C$")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(left=0)
    ax.legend(loc="best")
    plt.tight_layout()
    out1 = os.path.join(red_dir, "ratio_vs_E2_with_O2_curves.png")
    plt.savefig(out1, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")
    _write_list(out1, filenames)

    # ----------------------------------------------------------------
    # Plot 2: all individual points (no grouping, no error bars)
    # ----------------------------------------------------------------
    d = df.copy()
    d["E2"]    = pd.to_numeric(d["E2"],    errors="coerce")
    d["ratio"] = pd.to_numeric(d["ratio"], errors="coerce")
    d = d.dropna(subset=["E2", "ratio"]).query("E2 >= 100")

    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    _theory_curves(ax)

    ax.plot(d["E2"], d["ratio"], "o", markersize=3,
            color="black", alpha=0.6, zorder=10, label="Data (all)")

    ax.set_xlabel(r"Drift Field $E_2$ [V/cm]")
    ax.set_ylabel(r"$Q_A/Q_C$")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(left=0)
    ax.legend(loc="best")
    plt.tight_layout()
    out2 = os.path.join(red_dir, "ratio_vs_E2_with_O2_allpoints.png")
    plt.savefig(out2, dpi=400, bbox_inches="tight")

    # Save grouped data that was plotted
    all_xs, all_ys, all_ysig = [], [], []
    if "group" in df.columns:
        for _, dfg in df.groupby("group", sort=True):
            x, y, ye = f._grouped_mean_and_err(dfg, "E2", "ratio", "ratio_err")
            all_xs.extend(x);
            all_ys.extend(y);
            all_ysig.extend(ye)
    else:
        all_xs, all_ys, all_ysig = f._grouped_mean_and_err(df, "E2", "ratio", "ratio_err")
    pd.DataFrame({"E2": all_xs, "ratio_mean": all_ys, "ratio_err": all_ysig}).to_csv(
        os.path.join(red_dir, "plot_data_ratio_vs_e2_grouped.csv"), index=False)


    plt.close()
    print(f"Saved: {out2}")
    _write_list(out2, d["filename"].dropna().tolist() if "filename" in d.columns else filenames)


if __name__ == "__main__":
    main()
