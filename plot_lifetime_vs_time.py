"""
Reads:  $PRM_REDUCED_DIR/results_lifetime.csv
Saves:  results_lifetime_tau_vs_time.png          + .list + .csv
        results_lifetime_qa_qc_vs_time.png        + .list + .csv
        results_lifetime_q_vs_time.png            + .list + .csv
        results_lifetime_tau_vs_time_3min.png     + .list + .csv
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import core as f


def _write_list(png_path: str, filenames):
    list_path = png_path.replace(".png", ".list")
    with open(list_path, "w") as fh:
        for fn in filenames:
            fh.write(f"{fn}\n")


def _write_csv(png_path: str, df_plot: pd.DataFrame):
    csv_path = png_path.replace(".png", ".csv")
    df_plot.to_csv(csv_path, index=False)


def main():
    red_dir  = f.environment("PRM_REDUCED_DIR")
    csv_name = config.RESULTS_LIFETIME_CSV

    in_csv = os.path.join(red_dir, csv_name)
    if not os.path.exists(in_csv):
        print(f"Not found: {in_csv}"); return

    df = pd.read_csv(in_csv)
    df["ts"]      = df["filename"].apply(f._ts_from_name)
    df            = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    t0_ts         = df.loc[0, "ts"]
    df["time_h"]  = (df["ts"] - t0_ts).dt.total_seconds() / 3600.0
    tau_col       = "tau_e (lifetime)" if "tau_e (lifetime)" in df.columns else "tau_e"
    stem          = csv_name.replace(".csv", "")
    filenames     = df["filename"].dropna().tolist()

    # Generate all plots via core
    f.lifetime_plot(csv_name=csv_name)

    # ── 1. tau_e vs time ────────────────────────────────────────────────────
    png1 = os.path.join(red_dir, f"{stem}_tau_vs_time.png")
    cols1 = ["filename", "time_h", tau_col]
    if "tau_e_err" in df.columns: cols1.append("tau_e_err")
    _write_list(png1, filenames)
    _write_csv(png1, df[[c for c in cols1 if c in df.columns]])

    # ── 2. QA/QC vs time ────────────────────────────────────────────────────
    png2 = os.path.join(red_dir, f"{stem}_qa_qc_vs_time.png")
    cols2 = ["filename", "time_h", "ratio"]
    if "ratio_err" in df.columns: cols2.append("ratio_err")
    _write_list(png2, filenames)
    _write_csv(png2, df[[c for c in cols2 if c in df.columns]])

    # ── 3. QA and QC vs time ────────────────────────────────────────────────
    png3 = os.path.join(red_dir, f"{stem}_q_vs_time.png")
    cols3 = ["filename", "time_h", "QA", "QC"]
    if "QA_err" in df.columns: cols3.append("QA_err")
    if "QC_err" in df.columns: cols3.append("QC_err")
    _write_list(png3, filenames)
    _write_csv(png3, df[[c for c in cols3 if c in df.columns]])

    # ── 4. 3-min group averages ──────────────────────────────────────────────
    png4 = os.path.join(red_dir, f"{stem}_tau_vs_time_3min.png")
    d = df[["ts", "time_h", "filename", tau_col]].copy()
    if "tau_e_err" in df.columns:
        d["tau_e_err"] = df["tau_e_err"]
    d["gid"] = f.group_ids_by_gap(d["ts"])

    rows = []
    for _, gdf in d.groupby("gid"):
        if len(gdf) == 0: continue
        N     = len(gdf)
        errs  = gdf["tau_e_err"].to_numpy() if "tau_e_err" in gdf.columns else np.full(N, np.nan)
        mean_tau = float(gdf[tau_col].mean())
        mean_t   = float(gdf["time_h"].mean())
        err_mean = float(np.sqrt(np.nansum(errs**2)) / N) if np.any(np.isfinite(errs)) else np.nan
        rows.append({
            "time_h":        mean_t,
            "tau_e_mean":    mean_tau,
            "tau_e_err_mean": err_mean,
            "N":             N,
            "filenames":     ";".join(gdf["filename"].tolist()),
        })

    df_3min = pd.DataFrame(rows)
    _write_list(png4, filenames)
    _write_csv(png4, df_3min)

    print(f"Saved plots + .list + .csv to {red_dir}")


if __name__ == "__main__":
    main()
