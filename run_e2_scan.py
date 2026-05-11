"""
run_e2_scan.py - PRM analysis for E2 field scan data.
Saves results to $PRM_REDUCED_DIR/results_e2_scan.csv.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import core as f

META_CSV = f.environment("PRM_METADATA_CSV")
DATA_DIR = f.environment("PRM_DATA_DIR")
OUT_DIR  = f.environment("PRM_OUTPUT_DIR")
RED_DIR  = f.environment("PRM_REDUCED_DIR")



def run_analysis(
    *, typ_unc: Optional[dict] = None,
    out_csv_name: str = config.RESULTS_E2_SCAN_CSV,
    make_plots: bool = True,) -> str:
    rows: List[dict] = []
    group_paths: Dict[str, List[str]] = {}

    for pref in config.RUN_PREFIXES:
        """Selects files whose filename starts with the date prefix (e.g. "20250917") and whose metadata Comments column contains "E2 =". 
        Applies the anode tail threshold and crossing rejection, same as the lifetime selection."""
        good, skipped, _ = f.choose_waveforms_by_prefix_and_e2(data_dir=DATA_DIR, meta_csv=META_CSV, run_prefix=pref)
        group_paths[pref] = good
        print(f"[{pref}] Selected {len(good)} files, skipped {len(skipped)}.")

    all_selected = [os.path.basename(p) for pref in config.RUN_PREFIXES for p in group_paths.get(pref, [])]
    efield_map = f.efield_map_from_metadata(META_CSV, all_selected)           #Reads the E1, E2, E3 columns from the metadata CSV for every selected file

    for pref in config.RUN_PREFIXES:
        for path in group_paths[pref]:
            filename = os.path.basename(path)
            ef = efield_map.get(filename, {"E1": float("nan"), "E2": float("nan"), "E3": float("nan")})
            E1 = float(ef.get("E1", float("nan")))
            E2 = float(ef.get("E2", float("nan")))
            E3 = float(ef.get("E3", float("nan")))

            fit = f.fit_one_file_q(
                path,
                fit_t_max_us=None,
                make_plot=make_plots,
                plot_tag=f"_E2_{E2:g}" if np.isfinite(E2) else "",
                out_dir=OUT_DIR,
                bump_left_offset=config.BUMP_LEFT_OFFSET_E2_US,
                bump_right_offset=config.BUMP_RIGHT_OFFSET_E2_US,)
            """Fit each file. fit_t_max_us=None means upper limit = anode peak + POST_ANODE_PEAK_US. Also uses BUMP_LEFT_OFFSET_E2_US = 40 us 
            (wider than the 10 µs used for lifetime) because the bump can appear further from the cathode peak at lower fields."""
            if fit is None:
                continue

            row = dict(fit)
            row["group"] = pref
            row["E1"] = E1;  row["E2"] = E2;  row["E3"] = E3
            row["E3_over_E1"] = E3/E1
            row["E3_over_E2"] = E3/E2
            row["E1_over_E2"] = E1/E2

            if typ_unc is not None:
                """Pass 1 passes typ_unc=None so this is skipped. Pass 2 passes the estimated uncertainties."""
                row.update(f._propagate_uncertainties(fit, typ_unc))

            rows.append(row)

    out_path = os.path.join(RED_DIR, out_csv_name)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

    # Save list of files used in this pass
    used_path = out_path.replace(".csv", "_used.list")
    with open(used_path, "w") as fh:
        for r in rows:
            fh.write(f"{r['filename']}\n")
    print(f"Saved used-file list to {used_path}")

    return out_path


def main():
    # Pass 1
    initial_path = run_analysis(typ_unc=None,
                                out_csv_name=config.INITIAL_RESULTS_E2_CSV,
                                make_plots=False)
    """Fits everything, saves results_q_initial_e2.csv"""

    # Estimate uncertainties
    typ_unc = f.estimate_typical_uncertainties_3min_from_results(
        results_csv_path=initial_path,
        save_to=config.TYP_UNC_E2_CSV,
        group_by_col="E2",   # group by field value, not by time
    )

    # Pass 2
    run_analysis(typ_unc=typ_unc,
                 out_csv_name=config.RESULTS_E2_SCAN_CSV,
                 make_plots=True)


if __name__ == "__main__":
    main()
