"""
Reads:  $PRM_REDUCED_DIR/results_e2_scan.csv
Saves:  ratio_vs_E2.png  + .list
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import core as f


def _write_list(png_path: str, filenames):
    list_path = png_path.replace(".png", ".list")
    with open(list_path, "w") as fh:
        for fn in filenames:
            fh.write(f"{fn}\n")


def main():
    red_dir = f.environment("PRM_REDUCED_DIR")
    df = f._read_results_df(config.RESULTS_E2_SCAN_CSV)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    if "group" in df.columns:
        df["group"] = df["group"].astype(str)
        for grp, dfg in df.groupby("group", sort=True):
            xs, ys, ysig = f._grouped_mean_and_err(dfg, "E2", "ratio", "ratio_err")
            ax.errorbar(xs, ys, yerr=ysig, fmt="o", markersize=4,
                        capsize=3, elinewidth=1, label=str(grp))
    else:
        xs, ys, ysig = f._grouped_mean_and_err(df, "E2", "ratio", "ratio_err")
        ax.errorbar(xs, ys, yerr=ysig, fmt="o", markersize=4,
                    capsize=3, elinewidth=1)

    ax.set_xlabel("E2 [V/cm]")
    ax.set_ylabel("QA/QC")
    ax.set_xlim(80, 210)
    if "group" in df.columns:
        ax.legend()

    plt.tight_layout()
    out = os.path.join(red_dir, "ratio_vs_E2.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")

    # Save what was plotted
    plot_df = pd.DataFrame({"E2": xs, "ratio_mean": ys, "ratio_err": ysig})
    plot_csv = os.path.join(red_dir, "plot_data_ratio_vs_e2.csv")
    plot_df.to_csv(plot_csv, index=False)
    print(f"Saved plot data to {plot_csv}")

    plt.close()
    print(f"Saved: {out}")
    _write_list(out, df["filename"].dropna().tolist())


if __name__ == "__main__":
    main()
