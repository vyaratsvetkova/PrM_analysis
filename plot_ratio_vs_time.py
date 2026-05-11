"""
Reads:  $PRM_REDUCED_DIR/results_e2_scan.csv
Saves:  ratio_vs_time.png  + .list
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

    df["ts"] = df["filename"].apply(f._ts_from_name)
    df = df.dropna(subset=["ts", "ratio"]).sort_values("ts").reset_index(drop=True)
    if df.empty:
        print("No valid data."); return

    t0 = df.loc[0, "ts"]
    x_hours = (df["ts"] - t0).dt.total_seconds() / 3600.0

    fig, ax = plt.subplots(figsize=(12, 4.5))

    if "group" in df.columns:
        df["group"] = df["group"].astype(str)
        for grp, dfg in df.groupby("group", sort=True):
            idx = dfg.index
            ax.errorbar(
                x_hours[idx], dfg["ratio"],
                yerr=dfg["ratio_err"] if "ratio_err" in dfg.columns else None,
                fmt="o", markersize=4, capsize=3,
                elinewidth=1, label=str(grp),
            )
    else:
        ax.errorbar(
            x_hours, df["ratio"],
            yerr=df["ratio_err"] if "ratio_err" in df.columns else None,
            fmt="o", markersize=4, capsize=3, elinewidth=1, color="steelblue",
        )

    ax.axhline(y=1, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Time since first run [hours]")
    ax.set_ylabel("QA/QC")
    if "group" in df.columns:
        ax.legend()

    plt.tight_layout()
    out = os.path.join(red_dir, "ratio_vs_time.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")

    # Save what was plotted
    plot_df = df[["filename", "ratio"]].copy()
    plot_df["hours"] = x_hours.values
    if "ratio_err" in df.columns:
        plot_df["ratio_err"] = df["ratio_err"].values
    plot_csv = os.path.join(red_dir, "plot_data_ratio_vs_time.csv")
    plot_df.to_csv(plot_csv, index=False)
    print(f"Saved plot data to {plot_csv}")

    plt.close()
    print(f"Saved: {out}")
    _write_list(out, df["filename"].tolist())


if __name__ == "__main__":
    main()
