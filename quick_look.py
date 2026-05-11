"""
quick_look.py - Fits a single waveform and displays results immediately.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import core as f


def fit_single_file(path: str, show_plot: bool = True):
    fit = f.fit_one_file_q(
        path,
        fit_t_max_us=config.FIT_T_MAX_US,
        make_plot=False,
        bump_left_offset=config.BUMP_LEFT_OFFSET_US,
        bump_right_offset=config.BUMP_RIGHT_OFFSET_US,
    )
    if fit is None:
        print(f"Fit failed: {os.path.basename(path)}")
        return None

    if show_plot:
        result = f.read_waveform(path, smooth=True)
        if result is None:
            return fit
        t_us, anode_mV, cathode_mV = result
        anode0, cathode0, _, _ = f.zero_baseline(anode_mV, cathode_mV)
        t_full = t_us[t_us <= config.FIT_T_MAX_US]

        t0 = fit["t0"]; t_c_peak = fit["t_c_peak"]
        t_a_start = fit["t_a_start"]; t_a_peak = fit["t_a_peak"]
        t_b_start = fit["t_b_start"]; t_b_peak = fit["t_b_peak"]
        QA = fit["QA"]; QC = fit["QC"]; QB = fit["QB"]

        p_c = f.make_cathode_q_model().make_params(
            t0=t0, t_c_peak=t_c_peak, QC=QC, tau=config.TAU_CATHODE_US)
        for p in p_c.values():
            p.set(vary=False)
        y_c   = f.make_cathode_q_model().eval(p_c, t=t_full)
        y_a_m = f.anode_main_basis_Q(t_full, t_a_start, t_a_peak, QA, config.TAU_ANODE_US)
        y_a_b = f.anode_bump_basis_Q(t_full, t_b_start, t_b_peak, QB, config.TAU_ANODE_US)

        ex_an = ((t_us >= config.EXCLUDE_ANODE_START_US)
                 & (t_us <= config.EXCLUDE_ANODE_END_US))
        ex_ca = ((t_us >= config.EXCLUDE_CATHODE_START_US)
                 & (t_us <= config.EXCLUDE_CATHODE_END_US))
        an_p  = anode0.copy();   an_p[ex_an]  = np.nan
        ca_p  = cathode0.copy(); ca_p[ex_ca]  = np.nan

        tau_e = fit["tau_e (lifetime)"]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(t_us, an_p,  lw=3, color="red",  alpha=0.3, label="Anode data")
        ax.plot(t_us, ca_p,  lw=3, color="blue", alpha=0.3, label="Cathode data")
        ax.plot(t_full, y_c,       "--", lw=2, label="Cathode fit")
        ax.plot(t_full, y_a_m,     "--", lw=2, label="Anode main")
        ax.plot(t_full, y_a_b,     "--", lw=2, label="Anode bump")
        ax.plot(t_full, y_a_m+y_a_b, "--", lw=2, label="Anode full")
        ax.set_xlabel("Time [µs]", fontsize=18)
        ax.set_ylabel("Voltage [mV]", fontsize=18)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=12)
        tau_str = f"{tau_e:.0f}" if np.isfinite(tau_e) else "nan"
        ax.set_title(f"{fit['filename']}   τ_e = {tau_str} µs   Q_A/Q_C = {fit['ratio']:.3f}",
                     fontsize=14)
        plt.tight_layout()
        plt.show()

    return fit


def main():
    parser = argparse.ArgumentParser(description="Quick-look PRM waveform fit")
    parser.add_argument("files", nargs="+", help="Waveform CSV file(s)")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--quiet",   action="store_true",
                        help="Print only tau_e (pipe-friendly)")
    args = parser.parse_args()

    data_dir = os.environ.get("PRM_DATA_DIR", ".")

    for filename in args.files:
        path = filename if os.path.exists(filename) \
               else os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"File not found: {filename}"); continue

        r = fit_single_file(path, show_plot=not args.no_plot and not args.quiet)
        if r is None:
            print(f"FAILED: {filename}"); continue

        if args.quiet:
            tau_e   = r["tau_e (lifetime)"]
            tau_str = f"{tau_e:.1f}" if np.isfinite(tau_e) else "nan"
            print(f"{r['filename']} tau_e = {tau_str} µs")
        else:
            tau_e = r["tau_e (lifetime)"]
            t1, t2, t3 = r["t1"], r["t2"], r["t3"]
            amp_c = r["QC"] * f.peak_factor_c(t1, config.TAU_CATHODE_US)
            amp_a = r["QA"] * f.peak_factor_a(t3, config.TAU_ANODE_US)
            print(f"\n{'='*45}")
            print(f" File: {r['filename']}")
            print(f" τ_e: {tau_e:.1f} µs" if np.isfinite(tau_e) else " τ_e: nan")
            print(f" Q_A/Q_C: {r['ratio']:.4f}")
            print(f" Q_C: {r['QC']:.2f} fC")
            print(f" Q_A: {r['QA']:.2f} fC")
            print(f" Amp_c: {amp_c:.2f} mV")
            print(f" Amp_a: {amp_a:.2f} mV")
            print(f" t_drift: {r['t_drift']:.2f} µs")
            print(f" T1/T2/T3: {t1:.1f} / {t2:.1f} / {t3:.1f} µs")
            print(f"{'='*45}")


if __name__ == "__main__":
    main()
