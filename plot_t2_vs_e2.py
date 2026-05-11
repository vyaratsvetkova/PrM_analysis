"""
Produces two plots:
  t2_vs_E2_with_theory.png   — one point per E2 ± uncertainty, theory band  + .list
  t2_vs_E2_allpoints.png     — all individual points, no error bars          + .list
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

RC = {"font.size": 13, "axes.labelsize": 15,
      "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 11}


def _write_list(png_path: str, filenames):
    list_path = png_path.replace(".png", ".list")
    with open(list_path, "w") as fh:
        for fn in filenames:
            fh.write(f"{fn}\n")


def _theory_band(ax):
    E2_grid  = np.linspace(95.0, 210.0, 400)
    v_theory = f.predicted_vdrift_cm_per_us(E2_grid, T_K=config.T_K)
    t2_th    = config.D2_CM / v_theory
    t2_lo    = (config.D2_CM - config.DD2_CM) / v_theory
    t2_hi    = (config.D2_CM + config.DD2_CM) / v_theory
    ax.plot(E2_grid, t2_th, lw=2.0, label="Prediction")
    ax.fill_between(E2_grid, t2_lo, t2_hi, color="lightgray", alpha=0.6,
                    label=rf"$d = {config.D2_CM:.1f} \pm {config.DD2_CM:.1f}$ cm")


def main():
    red_dir = f.environment("PRM_REDUCED_DIR")
    df = f._read_results_df(config.RESULTS_E2_SCAN_CSV)

    d = df.copy()
    for c in ("E2", "t2", "t2_err"):
        d[c] = pd.to_numeric(d.get(c, np.nan), errors="coerce")
    d = d.dropna(subset=["E2", "t2"]).query("t2 > 0 and E2 >= 100")
    if d.empty:
        print("No valid data."); return

    filenames_all = df["filename"].dropna().tolist()
    filenames_d   = d["filename"].dropna().tolist() if "filename" in d.columns else filenames_all

    # ----------------------------------------------------------------
    # Plot 1: grouped mean ± uncertainty + theory band
    # ----------------------------------------------------------------
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    _theory_band(ax)

    eb_kw = dict(fmt="o", markersize=4, capsize=2, elinewidth=1,
                 color="black", ecolor="black", markeredgecolor="black", zorder=10)

    def _summ(dfg):
        xs, ys, ysig = [], [], []
        for e2, g in dfg.groupby("E2", sort=True):
            xs.append(float(e2)); ys.append(float(g["t2"].mean()))
            te = g["t2_err"].to_numpy(); te = te[np.isfinite(te)]
            ysig.append(float(np.sqrt(np.mean(te**2))) if len(te) else float("nan"))
        return xs, ys, ysig

    if "group" in d.columns:
        d["group"] = d["group"].astype(str)
        for _, dfg in d.groupby("group", sort=True):
            ax.errorbar(*_summ(dfg), **eb_kw)
    else:
        ax.errorbar(*_summ(d), **eb_kw)

    ax.set_xlabel(r"Drift Field $E_2$ [V/cm]")
    ax.set_ylabel(r"Transit Time $T_2$ [µs]")
    ax.legend()
    plt.tight_layout()
    out1 = os.path.join(red_dir, "t2_vs_E2_with_theory.png")
    plt.savefig(out1, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")
    _write_list(out1, filenames_d)

    # ----------------------------------------------------------------
    # Plot 2: all individual points
    # ----------------------------------------------------------------
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    _theory_band(ax)

    ax.plot(d["E2"], d["t2"], "o", markersize=3,
            color="black", alpha=0.6, zorder=10, label="Data (all)")

    ax.set_xlabel(r"Drift Field $E_2$ [V/cm]")
    ax.set_ylabel(r"Transit Time $T_2$ [µs]")
    ax.legend()
    plt.tight_layout()
    out2 = os.path.join(red_dir, "t2_vs_E2_allpoints.png")
    plt.savefig(out2, dpi=400, bbox_inches="tight")

    # Save what was plotted
    all_xs, all_ys, all_ysig = [], [], []
    if "group" in d.columns:
        for _, dfg in d.groupby("group", sort=True):
            x, y, ye = _summ(dfg)
            all_xs.extend(x);
            all_ys.extend(y);
            all_ysig.extend(ye)
    else:
        all_xs, all_ys, all_ysig = _summ(d)
    plot_df = pd.DataFrame({"E2": all_xs, "t2_mean": all_ys, "t2_err": all_ysig})
    plot_csv = os.path.join(red_dir, "plot_data_t2_vs_e2.csv")
    plot_df.to_csv(plot_csv, index=False)
    print(f"Saved plot data to {plot_csv}")

    plt.close()
    print(f"Saved: {out2}")
    _write_list(out2, filenames_d)


if __name__ == "__main__":
    main()
