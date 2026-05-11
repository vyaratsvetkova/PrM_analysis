"""
Saves results to $PRM_REDUCED_DIR/results_lifetime.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import core as f

META_CSV = f.environment("PRM_METADATA_CSV")
DATA_DIR = f.environment("PRM_DATA_DIR")
OUT_DIR  = f.environment("PRM_OUTPUT_DIR")
RED_DIR  = f.environment("PRM_REDUCED_DIR")




def main():
    # Pass 1: initial fit -> typical uncertainties
    """Fits every selected waveform once without any uncertainty propagation, saves the results to results_q_initial.csv,
    then calls estimate_typical_uncertainties_3min_from_results to compute the within-3-minute scatter for each parameter.
    Returns a dict like {"t2": 2.3, "ratio": 0.029, ...}"""
    typ_unc = f.estimate_typical_uncertainties_3min_q(
        data_dir=DATA_DIR,
        meta_csv=META_CSV,
        save_to=config.TYP_UNC_CSV,
        in_results=config.INITIAL_RESULTS_CSV,)

    # Pass 2: full fit with uncertainty propagation
    """Full fit with uncertainty propagation"""
    rows = []
    good, skipped = f.choose_waveforms(DATA_DIR, META_CSV)
    print(f"Selected {len(good)} files, skipped {len(skipped)}.")

    for path in good:

        fit = f.fit_one_file_q(
            path,
            fit_t_max_us=config.FIT_T_MAX_US,
            make_plot=True,
            out_dir=OUT_DIR,
            bump_left_offset=config.BUMP_LEFT_OFFSET_US,
            bump_right_offset=config.BUMP_RIGHT_OFFSET_US,)        #Fit each file
        if fit is None:
            continue

        row = dict(fit)
        row.update(f._propagate_uncertainties(fit, typ_unc))       #Propagate uncertainties
        rows.append(row)

    out_csv = os.path.join(RED_DIR, config.RESULTS_LIFETIME_CSV)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    used = os.path.join(RED_DIR, "lifetime_analysis_used.list")
    with open(used, "w") as fh:
        for r in rows:
            fh.write(f"{r['filename']}\n")
    print(f"Saved used-file list to {used}")


if __name__ == "__main__":
    main()
