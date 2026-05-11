import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import core as f

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEFAULT_FILE = ("20250916T155238.csv")
XLIM         = (-5.0, 270.0)
YLIM         = (-70.0,  80.0)
Y_SPAN       = YLIM[1] - YLIM[0]
Y_ARROW      = 68.0     # height of T1/T2/T3 arrows
Y_LABEL      = YLIM[0] + 10.0  # height of numbered markers


def make_plot(filename: str, out_path: str | None = None) -> None:

    # ── Locate file ─────────────────────────────────────────────────────────
    path = filename
    if not os.path.exists(path):
        data_dir = os.environ.get("PRM_DATA_DIR", ".")
        path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        sys.exit(f"File not found: {filename}")

    # ── Read waveform ────────────────────────────────────────────────────────
    result = f.read_waveform(path, smooth=True)
    if result is None:
        sys.exit(f"Could not read waveform: {path}")
    t_us, anode_mV, cathode_mV = result
    anode0, cathode0, _, _ = f.zero_baseline(anode_mV, cathode_mV)

    # ── Fit ──────────────────────────────────────────────────────────────────
    fit = f.fit_one_file_q(path, fit_t_max_us=config.FIT_T_MAX_US, make_plot=False)
    if fit is None:
        sys.exit(f"Fit failed for {path}")

    t0        = fit["t0"];        t_c_peak  = fit["t_c_peak"]
    t_a_start = fit["t_a_start"]; t_a_peak  = fit["t_a_peak"]
    t_b_start = fit["t_b_start"]; t_b_peak  = fit["t_b_peak"]
    QA = fit["QA"]; QC = fit["QC"]; QB = fit["QB"]

    # ── Build fitted curves ──────────────────────────────────────────────────
    t_full = t_us[t_us <= config.FIT_T_MAX_US]

    p_c = f.make_cathode_q_model().make_params(
        t0=t0, t_c_peak=t_c_peak, QC=QC, tau=config.TAU_CATHODE_US)
    for p in p_c.values():
        p.set(vary=False)
    y_c_full = f.make_cathode_q_model().eval(p_c, t=t_full)

    y_a_main = f.anode_main_basis_Q(t_full, t_a_start, t_a_peak, QA, config.TAU_ANODE_US)
    y_a_bump = f.anode_bump_basis_Q(t_full, t_b_start, t_b_peak, QB, config.TAU_ANODE_US)
    y_a_full = y_a_main + y_a_bump

    # ── Plot ─────────────────────────────────────────────────────────────────
    fs = 17
    plt.rcParams.update({
        "font.size":        fs,
        "axes.labelsize":   fs,
        "xtick.labelsize":  fs,
        "ytick.labelsize":  fs,
        "legend.fontsize":  fs - 4,
    })

    fig, ax = plt.subplots(figsize=(14, 10))

    # Data (plotted twice so legend entries appear after fits)
    ax.plot(t_us, anode0,   color="magenta",   lw=1.1)
    ax.plot(t_us, cathode0, color="royalblue", lw=1.1)

    # Fitted components
    ax.plot(t_full, y_c_full, color="navy",       lw=1.8, ls="--", label="Cathode fit")
    ax.plot(t_full, y_a_main, color="darkorange", lw=1.8, ls="--", label="Anode main fit")
    ax.plot(t_full, y_a_bump, color="green",      lw=1.1, ls="--", label="Anode bump fit")
    ax.plot(t_full, y_a_full, color="red",        lw=1.8, ls="--", label="Anode total fit")

    # Data again on top with legend labels
    ax.plot(t_us, anode0,   color="magenta",   lw=1.7, label="Anode data")
    ax.plot(t_us, cathode0, color="royalblue", lw=1.7, label="Cathode data")

    # ── T1 / T2 / T3 arrows ─────────────────────────────────────────────────
    def draw_T(x1, x2, label):
        ax.annotate("", xy=(x1, Y_ARROW), xytext=(x2, Y_ARROW),
                    arrowprops=dict(arrowstyle="<->", lw=1.6, color="black",
                                    mutation_scale=10, shrinkA=0, shrinkB=0))
        ax.text(0.5*(x1+x2), Y_ARROW + 0.020*Y_SPAN, label,
                ha="center", va="bottom", fontsize=fs)

    draw_T(t0,       t_c_peak,  r"$T_1$")
    draw_T(t_c_peak, t_a_start, r"$T_2$")
    draw_T(t_a_start, t_a_peak, r"$T_3$")

    # ── Vertical timing markers ──────────────────────────────────────────────
    markers = [
        ("1", t0,        2.0, 0.60),
        ("2", t_c_peak,  2.0, 0.60),
        ("3", t_b_start, 1.5, 0.35),
        ("4", t_b_peak,  1.5, 0.35),
        ("5", t_a_start, 2.0, 0.60),
        ("6", t_a_peak,  2.0, 0.60),
    ]

    for label, tx, lw, alpha in markers:
        ax.axvline(tx, color="black", linestyle="--", lw=lw, alpha=alpha)
        offset = {"2": -1.0, "3": 1.0}.get(label, 0.0)
        ax.text(tx + offset, Y_LABEL, label, ha="center", va="top", fontsize=fs,
                bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="gray", alpha=0.75))

    # ── Timing key box ───────────────────────────────────────────────────────
    timing_text = (
        r"1: $t_{c,start}$"  "\n"
        r"2: $t_{c,peak}$"   "\n"
        r"3: $t_{b,start}$"  "\n"
        r"4: $t_{b,peak}$"   "\n"
        r"5: $t_{a,start}$"  "\n"
        r"6: $t_{a,peak}$"
    )
    ax.text(0.99, 0.05, timing_text, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=fs,
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="gray", alpha=0.75))

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("Time [µs]", fontsize=fs)
    ax.set_ylabel("Voltage [mV]", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(loc="upper right", fontsize=fs - 4)

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=400, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRM timing diagram plot")
    parser.add_argument("--file", default=DEFAULT_FILE,
                        help="Waveform CSV filename or full path")
    parser.add_argument("--out",  default=None,
                        help="Output PNG path (omit to show interactively)")
    args = parser.parse_args()

    make_plot(args.file, out_path=args.out)
