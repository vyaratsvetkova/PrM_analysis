from __future__ import annotations
import glob
import os
import re
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Minimizer, Model
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import savgol_filter

import config


# ===========================================================================
# SECTION 1: Environment
# ===========================================================================

def environment(name: str) -> str:
    """Return a required environment variable or exit."""
    val = os.environ.get(name)
    if not val:
        import sys
        sys.exit(f"[ERROR] Missing environment variable: {name}")
    return val


def find_waveforms_csvs(data_dir: str) -> List[str]:
    """Sorted list of all .csv waveform files in data_dir."""
    return sorted(glob.glob(os.path.join(data_dir, "*.csv")))


def read_metadata(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def read_waveform(
    csv_path: str,
    smooth: bool = True,) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Read one waveform CSV → (t_us, anode_mV, cathode_mV).
    Returns None if filename does not start with "2".
    Applies Savitzky–Golay smoothing when smooth=True.
    """
    fname = os.path.basename(csv_path)
    if not fname.startswith("2"):
        return None

    df = pd.read_csv(csv_path)
    t_us       = pd.to_numeric(df["Time(s)"]).astype(float).values * 1e6
    anode_mV   = pd.to_numeric(df["CH3V"]).astype(float).values * 1000.0
    cathode_mV = pd.to_numeric(df["CH4V"]).astype(float).values * 1000.0

    if smooth:
        for arr in (anode_mV, cathode_mV):
            arr[:] = savgol_filter(arr, window_length=config.SMOOTH_WINDOW, polyorder=config.SMOOTH_POLY)
    return t_us, anode_mV, cathode_mV


def zero_baseline(
    anode_mV: np.ndarray,
    cathode_mV: np.ndarray,
    n0: int = config.BASELINE_SAMPLES,) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Subtract median of the first n0 samples from each channel."""
    b_a = float(np.median(anode_mV[:n0]))
    b_c = float(np.median(cathode_mV[:n0]))
    return anode_mV - b_a, cathode_mV - b_c, b_a, b_c


# ===========================================================================
# SECTION 2: Sigmoid step function
# ===========================================================================

def sigmoid(t: np.ndarray, t0: float) -> np.ndarray:
    x = -config.K_SIGMOID * (t - t0)
    with np.errstate(over="ignore", under="ignore"):
        return 1.0 / (1.0 + np.exp(x))


# ===========================================================================
# SECTION 3: Basis Functions & Models
# ===========================================================================

def cathode_basis_Q(t, t0, t_c_peak, QC, tau):
    T1 = float(t_c_peak - t0)
    tau = float(tau)

    K_C = config.h0_c * tau / T1
    R_C = ((1.0 - np.exp(-(t - float(t0)) / tau))
        * sigmoid(t, t0)
        * (1.0 - sigmoid(t, t_c_peak)))
    D_C = ((1.0 - np.exp(-T1 / tau))
        * np.exp(-(t - float(t_c_peak)) / tau)
        * sigmoid(t, t_c_peak))

    return -float(QC) * K_C * (R_C + D_C)

def anode_main_basis_Q(t, t_a_start, t_a_peak, QA, tau):
    T3 = float(t_a_peak - t_a_start)
    tau = float(tau)

    K_A = config.h0_a * tau / T3
    R_A = (
        (1.0 - np.exp(-(t - float(t_a_start)) / tau))
        * sigmoid(t, t_a_start)
        * (1.0 - sigmoid(t, t_a_peak)))
    D_A = (
        (1.0 - np.exp(-T3 / tau))
        * np.exp(-(t - float(t_a_peak)) / tau)
        * sigmoid(t, t_a_peak))

    return float(QA) * K_A * (R_A + D_A)

def anode_bump_basis_Q(t, t_b_start, t_b_peak, QB, tau):
    Tb = float(t_b_peak - t_b_start)
    tau = float(tau)
    K_B = config.h0_a * tau / Tb
    R_B = (
        (1.0 - np.exp(-(t - float(t_b_start)) / tau))
        * sigmoid(t, t_b_start)
        * (1.0 - sigmoid(t, t_b_peak)))
    D_B = (
        (1.0 - np.exp(-Tb / tau))
        * np.exp(-(t - float(t_b_peak)) / tau)
        * sigmoid(t, t_b_peak))

    return float(QB) * K_B * (R_B + D_B)


def anode_full_basis_Q(t, t_a_start, t_a_peak, QA, t_b_start, t_b_peak, QB, tau):
    """Full anode = main + bump"""
    return (anode_main_basis_Q(t, t_a_start, t_a_peak, QA, tau)
            + anode_bump_basis_Q(t, t_b_start, t_b_peak, QB, tau))


def make_cathode_q_model() -> Model:
    return Model(cathode_basis_Q, independent_vars=["t"])

def make_anode_full_q_model() -> Model:
    return Model(anode_full_basis_Q, independent_vars=["t"])


# ---------------------------------------------------------------------------
# Peak conversion factors for initial Q guesses
# ---------------------------------------------------------------------------

def peak_factor_c(T1: float, tau: float) -> float:
    """V_peak / QC = h0_c * (tau/T1) * (1 - exp(-T1/tau))"""
    return config.h0_c * (tau / T1) * (1.0 - np.exp(-T1 / tau))


def peak_factor_a(T: float, tau: float) -> float:
    """V_peak / QA = h0_a * (tau/T) * (1 - exp(-T/tau))"""
    return config.h0_a * (tau / T) * (1.0 - np.exp(-T / tau))

def peak_constrained_residual(
    model: Model,
    params,
    t: np.ndarray,
    data: np.ndarray,
    t_peak_data: float,
    peak_window_us: float = 5.0,
    penalty_sigma_mV: float = config.PEAK_PENALTY_SIGMA_MV,
    n_penalty: int = config.PEAK_PENALTY_N,) -> np.ndarray:
    """Residual with a soft penalty forcing the model peak to match the data peak."""
    model_y = model.eval(params, t=t)
    if not np.all(np.isfinite(model_y)):
        return np.full(len(data) + n_penalty, 1e6)

    resid = data - model_y

    win   = (t >= t_peak_data - peak_window_us) & (t <= t_peak_data + peak_window_us)
    t_win = t[win]
    if len(t_win) >= 3:
        y_win = model.eval(params, t=t_win)
        if not np.all(np.isfinite(y_win)):
            return np.full(len(data) + n_penalty, 1e6)
        model_peak = float(np.max(y_win))
    else:
        y1 = model.eval(params, t=np.array([t_peak_data]))
        if not np.all(np.isfinite(y1)):
            return np.full(len(data) + n_penalty, 1e6)
        model_peak = float(y1[0])

    # The target peak is the observed maximum in the data window
    data_win = data[win]
    a_max = float(np.max(data_win)) if len(data_win) > 0 else float(data[np.argmin(np.abs(t - t_peak_data))])

    diff = model_peak - a_max
    return np.append(resid, np.full(n_penalty, diff / penalty_sigma_mV))


# ===========================================================================
# SECTION 4: Initial Guess Estimation
# ===========================================================================

_SAVGOL_W = config.SMOOTH_WINDOW_GUESS
_SAVGOL_P = config.SMOOTH_POLY_GUESS


def detect_cathode_start(time_us, signal):
    mask1 = (time_us >= -180) & (time_us <= 10)
    mask2 = (time_us >= 30) & (time_us <= 50)

    if np.sum(mask1) < 2 or np.sum(mask2) < 2:
        slope = np.gradient(signal, time_us)
        base = 0.2 * np.min(slope)
        cands = np.where(slope < base)[0]
        idx = int(cands[0]) if len(cands) > 0 else np.argmin(signal)
        return time_us[idx], signal[idx], 0.0, 0.0, None, None

    coef1, cov1 = np.polyfit(time_us[mask1], signal[mask1], 1, cov=True)
    coef2, cov2 = np.polyfit(time_us[mask2], signal[mask2], 1, cov=True)
    a1, b1 = coef1;  a2, b2 = coef2
    sa1, sb1 = np.sqrt(np.diag(cov1))
    sa2, sb2 = np.sqrt(np.diag(cov2))

    t_int = (b2 - b1) / (a1 - a2) if not np.isclose(a1, a2) \
            else (time_us[mask1].mean() + time_us[mask2].mean()) / 2
    y_int = a1 * t_int + b1
    denom = (a1 - a2) ** 2
    sig_t = np.sqrt(
        (-(b2-b1)/denom*sa1)**2 + (-1/(a1-a2)*sb1)**2
        + ((b2-b1)/denom*sa2)**2 + (1/(a1-a2)*sb2)**2
    )
    sig_y = np.sqrt((t_int*sa1)**2 + sb1**2 + (a1*sig_t)**2)
    return t_int, y_int, sig_t, sig_y, coef1, coef2


def analyze_file(filepath: str) -> Optional[Dict[str, float]]:
    """
    Extract initial parameter guesses from raw waveform data.
    Returns dict with t_a_start, t_a_peak, amp_a, t0, t_c_peak, amp_c,
    or None if a valid anode rise time cannot be found.
    """
    data    = pd.read_csv(filepath)
    time_us = data["Time(s)"].values * 1e6
    raw_ch3 = data["CH3V"].values
    raw_ch4 = data["CH4V"].values

    t_start, *_ = detect_cathode_start(time_us, raw_ch4)
    pre_idx     = max(10, np.searchsorted(time_us, t_start))

    voltage_ch3 = (raw_ch3 - np.mean(raw_ch3[:pre_idx])) * 1e3
    voltage_ch4 = (raw_ch4 - np.mean(raw_ch4[:pre_idx])) * 1e3

    def _smooth(x):
        if len(x) < _SAVGOL_W:
            return x.copy()
        win = min(_SAVGOL_W, len(x) - (1 - len(x) % 2))
        return savgol_filter(x, window_length=win, polyorder=_SAVGOL_P)

    fch3 = _smooth(voltage_ch3)
    fch4 = _smooth(voltage_ch4)

    x1, y1, *_ = detect_cathode_start(time_us, fch4)

    slope        = np.gradient(fch3, time_us)
    anode_peak_i = np.argmax(fch3)

    valid = False
    anode_start_i = 0
    for factor in np.linspace(0.1, 0.8, 100):
        cands = np.where(slope > factor * np.max(slope))[0]
        if len(cands) == 0:
            continue
        anode_start_i = cands[0]
        rt = time_us[anode_peak_i] - time_us[anode_start_i]
        if config.ANODE_RISE_TIME_MIN_US < rt < config.ANODE_RISE_TIME_MAX_US:
            valid = True
            break

    if not valid:
        print(f"{os.path.basename(filepath)}: could not find valid rise time, skipping.")
        return None

    t_a_start_us = time_us[anode_start_i]
    cath_win = (time_us > x1) & (time_us < t_a_start_us)
    if np.sum(cath_win) >= 3:
        c_peak_i_local = int(np.argmin(fch4[cath_win]))
        c_peak_i = int(np.where(cath_win)[0][c_peak_i_local])
    else:
        c_peak_i = int(np.argmin(fch4))   # fallback

    return {
        "t_a_start": time_us[anode_start_i],
        "t_a_peak":  time_us[anode_peak_i],
        "amp_a":     fch3[anode_peak_i] - fch3[anode_start_i],
        "t0":        x1,
        "t_c_peak":  time_us[c_peak_i],
        "amp_c":     fch4[c_peak_i] - y1,
    }


# ===========================================================================
# SECTION 5: Per-file fit
# ===========================================================================

_CATH_Q_MODEL = make_cathode_q_model()
_AN_Q_MODEL   = make_anode_full_q_model()


def fit_one_file_q(
    path: str, *,
    fit_t_max_us: Optional[float] = None,
    make_plot: bool = True,
    plot_tag: str = "",
    out_dir: Optional[str] = None,
    bump_left_offset: float = config.BUMP_LEFT_OFFSET_US,
    bump_right_offset: float = config.BUMP_RIGHT_OFFSET_US,) -> Optional[Dict]:
    """
    Fit one waveform.
    fit_t_max_us: upper time limit.  None → anode peak + POST_ANODE_PEAK_US (for E2 scan).
    """
    result = read_waveform(path)
    if result is None:
        return None

    filename = os.path.basename(path)
    t_us, anode_mV, cathode_mV = result
    anode0, cathode0, _, _     = zero_baseline(anode_mV, cathode_mV)

    if fit_t_max_us is None:
        i_a0         = int(np.argmax(anode0))
        fit_t_max_us = float(t_us[i_a0]) + config.POST_ANODE_PEAK_US

    ex_an  = (t_us >= config.EXCLUDE_ANODE_START_US)   & (t_us <= config.EXCLUDE_ANODE_END_US)
    ex_ca  = (t_us >= config.EXCLUDE_CATHODE_START_US) & (t_us <= config.EXCLUDE_CATHODE_END_US)
    mask_an = (t_us <= fit_t_max_us) & (~ex_an)
    mask_ca = (t_us <= fit_t_max_us) & (~ex_ca)

    t_c_fit, c_fit = t_us[mask_ca], cathode0[mask_ca]
    t_a_fit, a_fit = t_us[mask_an], anode0[mask_an]

    m_ca = np.isfinite(t_c_fit) & np.isfinite(c_fit)
    t_c_fit, c_fit = t_c_fit[m_ca], c_fit[m_ca]
    m_an = np.isfinite(t_a_fit) & np.isfinite(a_fit)
    t_a_fit, a_fit = t_a_fit[m_an], a_fit[m_an]

    if len(t_c_fit) < 10 or len(t_a_fit) < 10:
        return None

    t_full = t_us[t_us <= fit_t_max_us]

    guess = analyze_file(path)
    if guess is None:
        return None

    # --- Cathode fit ---
    i_c         = int(np.argmin(c_fit))
    c_min       = abs(float(c_fit[i_c]))
    t_c_peak_d  = float(t_c_fit[i_c])

    mask_t0      = (t_c_fit < t_c_peak_d) & (t_c_fit > 0)
    t_cath, v_cath = t_c_fit[mask_t0], c_fit[mask_t0]
    if len(t_cath) < 3:
        return None
    t0_guess_max = float(t_cath[int(np.argmin(np.abs(v_cath - 0.2 * c_min)))])

    t1_g = float(guess["t_c_peak"] - guess["t0"])
    Fc_g = peak_factor_c(t1_g, config.TAU_CATHODE_US)
    QC_g = max(1e-12, (c_min / Fc_g) if (np.isfinite(Fc_g) and Fc_g != 0) else 1.0)

    p_c = _CATH_Q_MODEL.make_params(
        t0=float(guess["t0"]), t_c_peak=float(guess["t_c_peak"]),
        QC=QC_g, tau=config.TAU_CATHODE_US)
    p_c["t0"].set(min=0.0, max=t0_guess_max)
    p_c["t_c_peak"].set(min=t_c_peak_d - 10.0, max=t_c_peak_d + 10.0)
    p_c["QC"].set(min=0.0, max=max(1e-12, QC_g))
    p_c["tau"].set(vary=False)

    res_c    = _CATH_Q_MODEL.fit(c_fit, params=p_c, t=t_c_fit)
    t0       = float(res_c.params["t0"].value)
    t_c_peak = float(res_c.params["t_c_peak"].value)
    QC       = float(res_c.params["QC"].value)

    # --- Bump window ---
    bump_mask    = ((t_us >= t_c_peak - bump_left_offset)
                    & (t_us <= float(guess["t_a_start"]) - bump_right_offset))
    t_b_win, a_b_win = t_us[bump_mask], anode0[bump_mask]
    if len(t_b_win) < 5:
        return None

    i_b        = int(np.argmax(a_b_win))
    t_b_peak_g = float(t_b_win[i_b])
    b_peak_mV  = float(a_b_win[i_b])
    b_max_mV   = float(np.max(a_b_win))

    left_m      = t_b_win < t_b_peak_g
    t_left, a_left = t_b_win[left_m], a_b_win[left_m]
    t_b_start_g = float(t_left[int(np.argmin(np.abs(a_left - 0.15 * b_peak_mV)))]) \
        if len(t_left) >= 3 else t_b_peak_g - 10.0
    if t_b_start_g >= t_b_peak_g:
        t_b_start_g = t_b_peak_g - 10.0
    tb_g = max(5.0, t_b_peak_g - t_b_start_g)

    # --- Anode fit ---
    i_a        = int(np.argmax(a_fit))
    a_max      = float(a_fit[i_a])
    t_a_peak_d = float(t_a_fit[i_a])

    t3_g = float(guess["t_a_peak"] - guess["t_a_start"])
    if t3_g <= 0:
        return None
    Fa_g = peak_factor_a(t3_g, config.TAU_ANODE_US)
    QA_g = max(1e-12, (float(guess["amp_a"]) / Fa_g) if (np.isfinite(Fa_g) and Fa_g != 0) else 1.0)
    Fb_g = peak_factor_a(tb_g, config.TAU_ANODE_US)
    QB_g = max(1e-12, (b_peak_mV / Fb_g) if (np.isfinite(Fb_g) and Fb_g != 0) else 1.0)

    p_a = _AN_Q_MODEL.make_params(
        t_a_start=float(guess["t_a_start"]), t_a_peak=float(guess["t_a_peak"]),
        QA=QA_g, t_b_start=t_b_start_g, t_b_peak=t_b_peak_g,
        QB=QB_g, tau=config.TAU_ANODE_US)
    p_a["t_a_peak"].set(min=t_a_peak_d - 5.0, max=t_a_peak_d)
    p_a["t_a_start"].set(min=t_b_peak_g, max=t_a_peak_d)
    p_a["t_b_peak"].set(min=t_b_peak_g - 25.0, max=t_b_peak_g + 10.0)
    p_a["t_b_start"].set(min=t_c_peak_d - 15.0, max=t_b_peak_g)
    p_a["QA"].set(min=0.0, max=max(1e-12, a_max / max(Fa_g, 1e-12)))
    p_a["QB"].set(min=0.0, max=max(1e-12, b_max_mV / max(Fb_g, 1e-12)) * 1.05)
    p_a["tau"].set(vary=False)

    def _resid(params, t, data):
        return peak_constrained_residual(_AN_Q_MODEL, params, t, data, t_a_peak_d)

    res_a     = Minimizer(_resid, p_a, fcn_args=(t_a_fit, a_fit)).minimize()
    t_a_start = float(res_a.params["t_a_start"].value)
    t_a_peak  = float(res_a.params["t_a_peak"].value)
    t_b_start = float(res_a.params["t_b_start"].value)
    t_b_peak  = float(res_a.params["t_b_peak"].value)
    QA = float(res_a.params["QA"].value)
    QB = float(res_a.params["QB"].value)

    t1 = t_c_peak - t0
    t2 = t_a_start - t_c_peak
    t3 = t_a_peak - t_a_start
    tb = t_b_peak - t_b_start
    t_drift = t2 + 0.5 * (t3 + t1)
    ratio = QA / QC if QC != 0 else float("nan")
    tau_e = (-t_drift / np.log(ratio)
               if (np.isfinite(ratio) and ratio > 0 and ratio != 1.0)
               else float("nan"))

    Fc = peak_factor_c(t1, config.TAU_CATHODE_US)
    Fa = peak_factor_a(t3, config.TAU_ANODE_US)
    Fb = peak_factor_a(tb, config.TAU_ANODE_US)

    # --- Optional per-event plot ---
    if make_plot:
        if out_dir is None:
            out_dir = os.environ.get("PRM_OUTPUT_DIR", ".")
        y_c  = _CATH_Q_MODEL.eval(res_c.params, t=t_full)
        y_am = anode_main_basis_Q(t_full, t_a_start, t_a_peak, QA, config.TAU_ANODE_US)
        y_ab = anode_bump_basis_Q(t_full, t_b_start, t_b_peak, QB, config.TAU_ANODE_US)

        an_p = anode0.copy(); an_p[ex_an] = np.nan
        ca_p = cathode0.copy(); ca_p[ex_ca] = np.nan

        img = os.path.join(out_dir, filename.replace(".csv", f"{plot_tag}_fit.png"))
        plot_fit(t_us, an_p, ca_p,
                 t_full, y_c, t_full, y_am, t_full, y_ab, t_full, y_am + y_ab,
                 img, title=f"{filename}{plot_tag}")

    return {
        "filename": filename,
        "fit_t_max_us": fit_t_max_us,
        "t0": t0,
        "t_c_peak": t_c_peak,
        "t_a_start": t_a_start,
        "t_a_peak": t_a_peak,
        "t_b_start": t_b_start,
        "t_b_peak": t_b_peak,
        "t1": t1, "t2": t2, "t3": t3, "tb": tb,
        "t_drift": t_drift,
        "FC": Fc, "FA": Fa, "FB": Fb,
        "QC": QC, "QA": QA, "QB": QB,
        "ratio": ratio,
        "tau_e (lifetime)": tau_e,
    }


# ===========================================================================
# SECTION 6: Per-event fit plot
# ===========================================================================

def plot_fit(
    t_us, anode0, cathode0,
    t_c_fit, y_c_fit,
    t_a_main, y_a_main,
    t_a_bump, y_a_bump,
    t_a_full, y_a_full,
    out, title: str = "",
) -> None:
    """Save a per-event fit PNG"""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(t_us, anode0, lw=1.5, color="red",  alpha=0.6, label="Anode data")
    ax.plot(t_us, cathode0, lw=1.5, color="blue", alpha=0.6, label="Cathode data")
    ax.plot(t_c_fit, y_c_fit, "--", lw=1.5, label="Cathode fit")
    ax.plot(t_a_main, y_a_main, "--", lw=1.5, label="Anode main")
    ax.plot(t_a_bump, y_a_bump, "--", lw=1.5, label="Anode bump")
    ax.plot(t_a_full, y_a_full, "--", lw=1.5, label="Anode full")
    ax.set_xlabel("Time [µs]", fontsize=13)
    ax.set_ylabel("Voltage [mV]", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="upper right")
    if title:
        ax.set_title(title, fontsize=10)
    if isinstance(out, str):
        fig.savefig(out, bbox_inches="tight", dpi=config.FIT_PLOT_DPI)
    else:
        out.savefig(fig)
    plt.close(fig)


# ===========================================================================
# SECTION 7: Timestamp and grouping
# ===========================================================================

def _ts_from_name(name: str) -> Optional[datetime]:
    m = re.match(r"(\d{8}T\d{6})", os.path.basename(name))
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S") if m else None

ts_from_name = _ts_from_name


def group_ids_by_gap(
    ts_series: Iterable[datetime],
    max_gap_seconds: float = 180.0,
) -> List[int]:
    ts   = list(ts_series)
    gids: List[int] = []
    gid  = 0
    for i, t in enumerate(ts):
        if i == 0:
            gids.append(gid); continue
        if (t - ts[i - 1]).total_seconds() <= max_gap_seconds:
            gids.append(gid)
        else:
            gid += 1; gids.append(gid)
    return gids


# ===========================================================================
# SECTION 8: Uncertainty estimation
# ===========================================================================

def estimate_typical_uncertainties_3min_q(
    data_dir: Optional[str] = None,
    meta_csv: Optional[str] = None,
    save_to: Optional[str] = None,
    in_results: str = config.INITIAL_RESULTS_CSV,
) -> Dict[str, float]:
    """
    Initial pass (no uncertainty propagation) --> estimate typical parameter uncertainties from 3-min scatter.
    """
    if data_dir is None:
        data_dir = environment("PRM_DATA_DIR")
    if meta_csv is None:
        meta_csv = environment("PRM_METADATA_CSV")
    red_dir = environment("PRM_REDUCED_DIR")

    rows = []
    good, _ = choose_waveforms(data_dir, meta_csv)
    print(f"Initial pass: {len(good)} files selected.")

    for path in good:
        if not os.path.exists(path):
            continue
        fit = fit_one_file_q(
            path,
            fit_t_max_us=config.FIT_T_MAX_US,
            make_plot=False,
            bump_left_offset=config.BUMP_LEFT_OFFSET_US,
            bump_right_offset=config.BUMP_RIGHT_OFFSET_US,)
        if fit:
            rows.append(fit)

    out_csv = os.path.join(red_dir, in_results)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    return estimate_typical_uncertainties_3min_from_results(results_csv_path=out_csv, save_to=save_to)       #generates the initial results CSV and uses estimate_typical_uncertainties_3min_from_results()


def estimate_typical_uncertainties_3min_from_results(
    *, results_csv_path: str,
    save_to: Optional[str] = None,
    max_gap_seconds: float = 180.0,
    group_by_col: Optional[str] = None,
    params: Optional[List[str]] = None) -> Dict[str, float]:  #manually choose files to analyze
    """
    Mean within-group scatter for each fit parameter.

    Grouping:
      group_by_col=None  (default) - group consecutive files within max_gap_seconds of each other. Used for the lifetime run.
      group_by_col="E2"            - group by E2 field value. Used for the E2 scan.
    """
    df = pd.read_csv(results_csv_path)                        #Loads results CSV
    if "filename" not in df.columns:
        raise ValueError("results CSV must have a 'filename' column")

    df["ts"] = df["filename"].apply(_ts_from_name)            # extracts the time
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)   #Deletes rows where timestamp extraction failed + orders rows by time
    if df.empty:
        return {}

    if params is None:    #If you didn't manually specify parameters
        candidate = ["t0", "t_c_peak", "t_a_start", "t_a_peak",
                     "t_b_start", "t_b_peak", "QC", "QA", "QB",
                     "t1", "t2", "t3", "t_drift", "ratio"]          # <---added the ratio and drift times
        params = [p for p in candidate if p in df.columns]

    # Assign group IDs
    if group_by_col is not None:
        # E2 scan: group by field value so scatter is estimated within each field
        if group_by_col not in df.columns:
            raise ValueError(f"group_by_col '{group_by_col}' not in results CSV")
        df[group_by_col] = pd.to_numeric(df[group_by_col], errors="coerce")
        df["gid"] = pd.factorize(df[group_by_col])[0]
    else:
        # Lifetime run: group consecutive files within max_gap_seconds
        df["gid"] = group_ids_by_gap(df["ts"], max_gap_seconds=max_gap_seconds)    #Creates a group ID

    scatters: Dict[str, List[float]] = {p: [] for p in params}                     #Creates a dictionary, each list will store the scatter from each group

    for _, gdf in df.groupby("gid", sort=True):      #Loops over each group separately
        if len(gdf) < 2:                             # need at least 2 points to compute a std
            continue
        for p in params:                              #Loops over the parameters
            vals = pd.to_numeric(gdf[p], errors="coerce").to_numpy()       #Convert safely to numeric
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2:
                scatters[p].append(float(np.nanstd(vals, ddof=1)))         #Computes the standard deviation for the param in that group, for example: scatters["QA"] = [0.003, 0.002, 0.005, ...] after all groups have looped through

    typical = {p: float(np.nanmean(v)) if v else float("nan") for p, v in scatters.items()}     #Averages the scatter for each param across groups

    if save_to:
        red_dir  = os.environ.get("PRM_REDUCED_DIR") or os.path.dirname(results_csv_path)
        out_path = os.path.join(red_dir, save_to)
        pd.DataFrame([{"param": p, "typical_uncertainty": typical[p], "num_groups_used": len(scatters[p])} for p in params]).to_csv(out_path, index=False)
        print(f"Saved typical uncertainties to {out_path}")

    return typical



def _propagate_uncertainties(row: dict, typ_unc: dict) -> dict:
    def u(name):
        try:
            return float(typ_unc.get(name, float("nan")))
        except:
            return float("nan")

    t0_e = u("t0")
    tcp_e = u("t_c_peak")
    tas_e = u("t_a_start")
    tap_e = u("t_a_peak")
    tbs_e = u("t_b_start")
    tbp_e = u("t_b_peak")
    QC_e = u("QC")
    QA_e = u("QA")
    QB_e = u("QB")

    t1 = row["t1"]
    t3 = row["t3"]
    tb = row["tb"]
    t_drift = row["t_drift"]
    ratio = row["ratio"]
    QC = row["QC"]
    QA = row["QA"]
    QB = row["QB"]

    t1_var = u("t1") ** 2
    t2_var = u("t2") ** 2
    t3_var = u("t3") ** 2
    tb_var = tbp_e ** 2 + tbs_e ** 2
    td_var = u("t_drift") ** 2

    ratio_var = u("ratio") ** 2


    ln_r = np.log(ratio)
    tau_e_var = (td_var / ln_r ** 2 + (t_drift ** 2 * ratio_var) / (ratio ** 2 * ln_r ** 4))




    v_drift = config.L_CM / t_drift
    v_drift_var = ((config.DL_CM / t_drift) ** 2 + (config.L_CM * np.sqrt(td_var) / t_drift ** 2) ** 2)


    def _s(v):
        return float(np.sqrt(v)) if (np.isfinite(v) and v >= 0) else float("nan")

    return {
        "t0_err": t0_e,
        "t_c_peak_err": tcp_e,
        "t_a_start_err": tas_e,
        "t_a_peak_err": tap_e,
        "t_b_start_err": tbs_e,
        "t_b_peak_err": tbp_e,
        "t1_err": _s(t1_var),
        "t2_err": _s(t2_var),
        "t3_err": _s(t3_var),
        "tb_err": _s(tb_var),
        "t_drift_err": _s(td_var),
        "QC_err": QC_e,
        "QA_err": QA_e,
        "QB_err": QB_e,
        "ratio_err": _s(ratio_var),
        "tau_e_err": _s(tau_e_var),
        "v_drift_cm_per_us": v_drift,
        "v_drift_err": _s(v_drift_var),
    }

# ===========================================================================
# SECTION 9: Waveform selection (standard lifetime)
# ===========================================================================

def _tails_cross(
    t_tail: np.ndarray,
    tail_a: np.ndarray,
    tail_c: np.ndarray,
    t_cross_limit_us: float = config.T_CROSS_LIMIT_US,
) -> bool:
    """Reject if da = tail_a - tail_c changes sign before t_cross_limit_us."""
    lim = t_tail <= t_cross_limit_us
    if not np.any(lim):
        return False
    da = tail_a[lim] - tail_c[lim]
    return bool(np.any(da[:-1] * da[1:] <= 0.0))


def choose_waveforms(
    data_dir: str,
    meta_csv: str,
    vc_exp: int = config.VC_EXP,
    vag_exp: int = config.VAG_EXP,
    va_exp: int = config.VA_EXP,
    anode_threshold_mV: float = config.ANODE_TAIL_THRESHOLD_MV,
    skip_after_peak_pts: int = config.SKIP_AFTER_PEAK_PTS,
    t_cross_limit_us: float = config.T_CROSS_LIMIT_US,
    debug: bool = False,
) -> Tuple[List[str], List[str]]:
    """Select waveforms matching voltage settings and basic quality cuts."""
    meta    = read_metadata(meta_csv)
    mask    = ((meta["Vc [V]"]  == vc_exp)
               & (meta["Vag [V]"] == vag_exp)
               & (meta["Va [V]"]  == va_exp))
    allowed = set(meta.loc[mask, "Filename"].astype(str))

    matching: List[str] = []
    skipped:  List[str] = []

    for path in find_waveforms_csvs(data_dir):
        fname = os.path.basename(path)
        if fname not in allowed:
            skipped.append(path); continue
        try:
            res = read_waveform(path)
            if res is None:
                skipped.append(path); continue
            t_us, anode_mV, cathode_mV = res
            anode0, cathode0, _, _ = zero_baseline(anode_mV, cathode_mV)

            a_peak_idx = int(np.argmax(anode0))
            start = a_peak_idx + skip_after_peak_pts
            if start >= len(t_us) - 1:
                if debug: print(f"{fname}: no tail -> skip")
                skipped.append(path); continue

            t_tail = t_us[start:]
            tail_a = anode0[start:]
            tail_c = cathode0[start:]

            if np.min(np.abs(tail_a)) > anode_threshold_mV:
                if debug: print(f"{fname}: anode tail never within {anode_threshold_mV} mV -> skip")
                skipped.append(path); continue

            if _tails_cross(t_tail, tail_a, tail_c, t_cross_limit_us):
                if debug: print(f"{fname}: tails cross/approach before {t_cross_limit_us} us -> skip")
                skipped.append(path); continue

            matching.append(path)
        except Exception as exc:
            if debug: print(f"{fname}: {exc} -> skip")
            skipped.append(path)

    return matching, skipped


# ===========================================================================
# SECTION 10: E2-scan file selection
# ===========================================================================

def parse_e2_from_comment(comment: str) -> Optional[float]:
    if comment is None:
        return None
    m = re.match(
        r"^E2\s*=\s*([+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?)",
        str(comment).strip(),
    )
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def efield_map_from_metadata(
    meta_csv: str,
    filenames: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    """Return filename → {E1, E2, E3} from metadata columns."""
    meta = read_metadata(meta_csv)

    def _norm(s):
        s = os.path.basename(str(s).strip())
        return s if s.lower().endswith(".csv") else s + ".csv"

    meta["_fn"] = meta["Filename"].astype(str).apply(_norm)
    col_map: Dict[str, str] = {}
    for target in ("E1", "E2", "E3"):
        for c in meta.columns:
            if c.strip().upper() == target:
                col_map[target] = c; break

    wanted = {os.path.basename(str(f).strip()) for f in filenames}
    result: Dict[str, Dict[str, float]] = {}
    for _, row in meta.iterrows():
        fn = row["_fn"]
        if fn not in wanted:
            continue
        entry: Dict[str, float] = {}
        for target in ("E1", "E2", "E3"):
            if target in col_map:
                try:    entry[target] = float(row[col_map[target]])
                except: entry[target] = float("nan")
            else:
                entry[target] = float("nan")
        result[fn] = entry
    return result


def choose_waveforms_by_prefix_and_e2(
    data_dir: str,
    meta_csv: str,
    run_prefix: str,
    anode_threshold_mV: float = config.ANODE_TAIL_THRESHOLD_E2_MV,
    skip_after_peak_pts: int = config.SKIP_AFTER_PEAK_PTS,
    t_cross_limit_us: float = config.T_CROSS_LIMIT_US,
    debug: bool = False,
) -> Tuple[List[str], List[str], Dict[str, float]]:
    """Select E2-scan waveforms: matching run_prefix AND 'E2 =' comment.

    Quality cuts (identical logic to choose_waveforms):
      1. Anode tail must return within anode_threshold_mV of baseline.
      2. Tails must not cross or nearly cross before t_cross_limit_us.
    """
    meta = read_metadata(meta_csv)
    fn   = meta["Filename"].astype(str).str.strip()

    if "Comments" in meta.columns:
        cm = meta["Comments"]
    elif "Comment" in meta.columns:
        cm = meta["Comment"]
    else:
        cm = pd.Series([""] * len(meta))
    cm = cm.astype(str).fillna("").str.strip()

    mask_run   = fn.str.startswith(str(run_prefix))
    mask_cmt   = cm.str.match(r'^\s*"?\s*E2\s*=')
    valid_meta = meta.loc[mask_run & mask_cmt].copy()

    def _norm(s):
        s = os.path.basename(str(s).strip())
        return s if s.lower().endswith(".csv") else s + ".csv"

    valid_meta["Filename_norm"] = valid_meta["Filename"].astype(str).apply(_norm)

    e2_map: Dict[str, float] = {}
    for _, r in valid_meta.iterrows():
        fn_n = str(r["Filename_norm"])
        cval = str(r.get("Comments", r.get("Comment", "")))
        e2   = parse_e2_from_comment(cval)
        e2_map[fn_n] = float(e2) if e2 is not None else float("nan")

    allowed   = set(valid_meta["Filename_norm"])
    matching: List[str] = []
    skipped:  List[str] = []

    for path in find_waveforms_csvs(data_dir):
        fname = os.path.basename(path)
        if fname not in allowed:
            skipped.append(path); continue
        try:
            res = read_waveform(path)
            if res is None:
                skipped.append(path); continue
            t_us, anode_mV, cathode_mV = res
            anode0, cathode0, _, _ = zero_baseline(anode_mV, cathode_mV)

            a_peak_idx = int(np.argmax(anode0))
            start = a_peak_idx + skip_after_peak_pts
            if start >= len(t_us) - 1:
                if debug: print(f"{fname}: no tail → skip")
                skipped.append(path); continue

            t_tail = t_us[start:]
            tail_a = anode0[start:]
            tail_c = cathode0[start:]

            # Cut 1: anode tail must return near baseline
            if np.min(np.abs(tail_a)) > anode_threshold_mV:
                if debug:
                    print(f"{fname}: anode tail min={np.min(np.abs(tail_a)):.2f} mV → skip")
                skipped.append(path); continue

            # Cut 2: tails must not cross or nearly cross before t_cross_limit_us
            if _tails_cross(t_tail, tail_a, tail_c, t_cross_limit_us):
                if debug: print(f"{fname}: tails cross/approach before {t_cross_limit_us} us -> skip")
                skipped.append(path); continue

            matching.append(path)
        except Exception as exc:
            if debug: print(f"{fname}: {exc} → skip")
            skipped.append(path)

    return matching, skipped, e2_map


# ===========================================================================
# SECTION 11: Results helpers
# ===========================================================================

def _read_results_df(csv_name: Optional[str] = None) -> pd.DataFrame:
    if csv_name is None:
        csv_name = config.RESULTS_LIFETIME_CSV
    red_dir = environment("PRM_REDUCED_DIR")
    path    = os.path.join(red_dir, csv_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    return pd.read_csv(path)


def _grouped_mean_and_err(df, x_col, y_col, yerr_col, min_e2=100.0):
    d = df.copy()
    d[x_col]  = pd.to_numeric(d[x_col],  errors="coerce")
    d[y_col]  = pd.to_numeric(d[y_col],  errors="coerce")
    if yerr_col in d.columns:
        d[yerr_col] = pd.to_numeric(d[yerr_col], errors="coerce")
    else:
        d[yerr_col] = np.nan
    d = d.dropna(subset=[x_col, y_col])
    if x_col == "E2":
        d = d[d["E2"] >= min_e2]
    xs, ys, yerrs = [], [], []
    for x, g in d.groupby(x_col, sort=True):
        g = g.dropna(subset=[y_col])
        if len(g) == 0: continue
        N = len(g)
        xs.append(float(x)); ys.append(float(g[y_col].mean()))
        sig = g[yerr_col].to_numpy() if yerr_col in g.columns else np.array([])
        sig = sig[np.isfinite(sig)]
        yerrs.append(float(np.sqrt(np.sum(sig**2)) / N) if len(sig) else float("nan"))
    return xs, ys, yerrs


# ===========================================================================
# SECTION 12: Theory predictions
# ===========================================================================

def electron_mobility_mu_cm2_per_V_us(
    E_kV_per_cm: np.ndarray,
    T_K: float = config.T_K,
    T0_K: float = config.T0_K,
) -> np.ndarray:
    """Walkowiak LAr electron mobility [cm²/(V·µs)]."""
    E  = np.asarray(E_kV_per_cm, dtype=float)
    a0, a1, a2, a3, a4, a5 = 551.6, 7953.7, 4440.43, 4.29, 43.63, 0.2053
    num = a0 + a1*E + a2*E**1.5 + a3*E**2.5
    den = 1.0 + (a1/a0)*E + a4*E**2.0 + a5*E**3.0
    return (num / den) * (T_K / T0_K)**(-1.5) / 1e6


def attachment_rate_kA_O2_per_s(E_kV_per_cm: np.ndarray) -> np.ndarray:
    """O2 attachment rate kA [s⁻¹]."""
    E  = np.asarray(E_kV_per_cm, dtype=float)
    a1, a2 = 76.2749, 4.24596
    b1, b2, b3, b4 = 2.62643, 0.0632332, 0.0632332, -0.000211009
    num = (a1/b1) + a1*E + a2*E**2
    den = 1.0 + b1*E + b2*E**2 + b3*E**3 + b4*E**4
    return (10.0**11) * (num / den)


def predicted_tau_us_from_O2_ppb(E_V_per_cm: np.ndarray, O2_ppb: float) -> np.ndarray:
    """Predicted lifetime [µs] for given E and O2 concentration."""
    E_kV = np.asarray(E_V_per_cm, dtype=float) / 1000.0
    kA   = attachment_rate_kA_O2_per_s(E_kV)
    return 1.0 / (kA * float(O2_ppb) * 1e-9) * 1e6


def predicted_survival_QA_over_QC(
    E2_V_per_cm: np.ndarray,
    O2_ppb: float,
    T_K: float = config.T_K,
    E1_over_E2=None,
    E3_over_E2=None,
    csv_name: Optional[str] = None,) -> np.ndarray:
    """
    Predict QA/QC survival ratio.
    """
    E2 = np.asarray(E2_V_per_cm, dtype=float)

    if E1_over_E2 is None or E3_over_E2 is None:
        try:
            df = _read_results_df(csv_name)
            if "E1_over_E2" in df.columns and "E3_over_E2" in df.columns:
                d  = df.dropna(subset=["E2", "E1_over_E2", "E3_over_E2"])
                if len(d) > 0:
                    grp = d.groupby("E2", sort=True)
                    e2p = grp["E2"].first().values
                    r1p = grp["E1_over_E2"].mean().values
                    r3p = grp["E3_over_E2"].mean().values
                    if len(e2p) >= 2:
                        if E1_over_E2 is None:
                            E1_over_E2 = np.interp(E2, e2p, r1p, left=r1p[0], right=r1p[-1])
                        if E3_over_E2 is None:
                            E3_over_E2 = np.interp(E2, e2p, r3p, left=r3p[0], right=r3p[-1])
                    else:
                        if E1_over_E2 is None: E1_over_E2 = float(r1p[0])
                        if E3_over_E2 is None: E3_over_E2 = float(r3p[0])
        except Exception:
            pass

    if E1_over_E2 is None: E1_over_E2 = 1.0 / 2.42
    if E3_over_E2 is None: E3_over_E2 = 2.42

    E1 = np.asarray(E1_over_E2, dtype=float) * E2
    E3 = np.asarray(E3_over_E2, dtype=float) * E2

    safe = (E1 > 0) & (E2 > 0) & (E3 > 0)
    result = np.full_like(E2, float("nan"))

    if np.any(safe):
        E1s = E1[safe]; E2s = E2[safe]; E3s = E3[safe]
        mu1 = electron_mobility_mu_cm2_per_V_us(E1s/1000.0, T_K=T_K, T0_K=config.T0_K)
        mu2 = electron_mobility_mu_cm2_per_V_us(E2s/1000.0, T_K=T_K, T0_K=config.T0_K)
        mu3 = electron_mobility_mu_cm2_per_V_us(E3s/1000.0, T_K=T_K, T0_K=config.T0_K)

        t1 = config.D1_CM / (mu1 * E1s)
        t2 = config.D2_CM / (mu2 * E2s)
        t3 = config.D3_CM / (mu3 * E3s)

        tau1 = predicted_tau_us_from_O2_ppb(E1s, O2_ppb)
        tau2 = predicted_tau_us_from_O2_ppb(E2s, O2_ppb)
        tau3 = predicted_tau_us_from_O2_ppb(E3s, O2_ppb)

        result[safe] = np.exp(-(t1/tau1 + t2/tau2 + t3/tau3))

    return result


def predicted_vdrift_cm_per_us(
    E2_V_per_cm: np.ndarray,
    T_K: float = config.T_K,
) -> np.ndarray:
    """Predicted drift velocity [cm/µs]."""
    E2  = np.asarray(E2_V_per_cm, dtype=float)
    mu2 = electron_mobility_mu_cm2_per_V_us(E2/1000.0, T_K=T_K, T0_K=config.T0_K)
    return mu2 * E2


# ===========================================================================
# SECTION 13: Lifetime vs time plots
# ===========================================================================

def lifetime_plot(csv_name: Optional[str] = None) -> None:
    """
    tau_e vs time, QA/QC vs time, Q vs time, and 3-min averages.
    Reads from csv_name (default: RESULTS_LIFETIME_CSV).
    Called from plot_lifetime_vs_time.py.
    """
    if csv_name is None:
        csv_name = config.RESULTS_LIFETIME_CSV
    red_dir = os.environ.get("PRM_REDUCED_DIR")
    if not red_dir:
        print("PRM_REDUCED_DIR not set, skipping.");
        return

    in_csv = os.path.join(red_dir, csv_name)
    if not os.path.exists(in_csv):
        print(f"{in_csv} not found, skipping.");
        return

    df = pd.read_csv(in_csv)
    df["ts"] = df["filename"].apply(_ts_from_name)
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if df.empty: return

    t0 = df.loc[0, "ts"]
    x_hours = (df["ts"] - t0).dt.total_seconds() / 3600.0
    tau_col = "tau_e (lifetime)" if "tau_e (lifetime)" in df.columns else "tau_e"
    stem = csv_name.replace(".csv", "")

    # tau_e vs time
    plt.figure(figsize=(15, 6))
    plt.rcParams.update({"font.size": 15})
    plt.errorbar(x_hours, df[tau_col], yerr=df.get("tau_e_err"),
                 fmt="o", color="black", ecolor="black", capsize=4)
    plt.xlabel("Time since first run [hours]")
    plt.ylabel("Electron lifetime τ [µs]")
    plt.savefig(os.path.join(red_dir, f"{stem}_tau_vs_time.png"), bbox_inches="tight")
    plt.close()

    # QA/QC vs time
    plt.figure(figsize=(12, 4.5))
    plt.errorbar(x_hours, df["ratio"], yerr=df.get("ratio_err"),
                 fmt="o", color="black", ecolor="black", elinewidth=1, capsize=3, markersize=4)
    plt.xlabel("Time since first run [hours]")
    plt.ylabel("QA/QC")
    plt.axhline(y=1, color="black", linestyle="--", linewidth=1)
    plt.savefig(os.path.join(red_dir, f"{stem}_qa_qc_vs_time.png"), bbox_inches="tight")
    plt.close()

    # QA and QC vs time
    plt.figure(figsize=(12, 4.5))
    plt.errorbar(x_hours, df["QA"], yerr=df.get("QA_err"),
                 fmt="o", color="black", ecolor="black", elinewidth=1, capsize=3, markersize=4, label="QA")
    plt.errorbar(x_hours, df["QC"], yerr=df.get("QC_err"),
                 fmt="o", color="black", ecolor="black", elinewidth=1, capsize=3, markersize=4, label="QC")
    plt.xlabel("Time since first run [hours]")
    plt.ylabel("Charge [fC]")
    plt.legend()
    plt.savefig(os.path.join(red_dir, f"{stem}_q_vs_time.png"), bbox_inches="tight")
    plt.close()

    # 3-min group averages
    d = pd.DataFrame({"ts": df["ts"], "x_hours": x_hours,
                      "tau": df[tau_col], "err": df.get("tau_e_err", np.nan)})
    d["gid"] = group_ids_by_gap(d["ts"])

    xs, ys, ysig = [], [], []
    for _, gdf in d.groupby("gid"):
        if len(gdf) == 0: continue
        N = len(gdf)
        errs = gdf["err"].to_numpy()
        xs.append(float((gdf["ts"].mean() - t0).total_seconds() / 3600.0))
        ys.append(float(gdf["tau"].mean()))
        ysig.append(float(np.sqrt(np.sum(errs ** 2)) / N)
                    if np.any(np.isfinite(errs)) else float("nan"))

    if ys:
        gmean = float(np.mean(ys))
        gerr = float(np.std(ys, ddof=1) / np.sqrt(len(ys)))
        print(f"Global mean lifetime: {gmean:.1f} ± {gerr:.1f} µs")

        plt.figure(figsize=(15, 6))
        plt.rcParams.update({"font.size": 15})
        plt.errorbar(xs, ys, yerr=ysig, fmt="o", color="black", ecolor="black", capsize=4)
        plt.axhline(gmean, color="black", linestyle="--", linewidth=1.8,
                    label=f"Mean: {round(gmean, -1):.0f} ± {round(gerr, -1):.0f} µs")
        plt.xlabel("Time since first run [hours]")
        plt.ylabel("Electron lifetime [µs]")
        plt.legend()
        plt.savefig(os.path.join(red_dir, f"{stem}_tau_vs_time_3min.png"), bbox_inches="tight")
        plt.close()

    print(f"Lifetime plots saved to {red_dir}")


# ===========================================================================
# SECTION 14: Timing-diagram plots
# ===========================================================================

def plot_timing_diagram_for_file(
    filename_or_path: str,
    out_path: str,
    title: str = "",) -> None:

    path = filename_or_path
    if not os.path.exists(path):
        path = os.path.join(environment("PRM_DATA_DIR"), filename_or_path)

    result = read_waveform(path)
    if result is None:
        raise ValueError(f"Invalid waveform file: {path}")
    t_us, anode_mV, cathode_mV = result
    anode0, cathode0, _, _ = zero_baseline(anode_mV, cathode_mV)

    fit = fit_one_file_q(path, fit_t_max_us=config.FIT_T_MAX_US, make_plot=False)
    if fit is None:
        raise ValueError(f"Fit failed for {path}")

    t0 = fit["t0"]; t_c_peak = fit["t_c_peak"]
    t_a_start = fit["t_a_start"]; t_a_peak = fit["t_a_peak"]
    t_b_start = fit["t_b_start"]; t_b_peak = fit["t_b_peak"]
    QA = fit["QA"]; QC = fit["QC"]; QB = fit["QB"]

    t_full = t_us[t_us <= config.FIT_T_MAX_US]

    p_c = _CATH_Q_MODEL.make_params(
        t0=t0, t_c_peak=t_c_peak, QC=QC, tau=config.TAU_CATHODE_US)
    for p in p_c.values():
        p.set(vary=False)
    y_c_full = _CATH_Q_MODEL.eval(p_c, t=t_full)
    y_a_main = anode_main_basis_Q(t_full, t_a_start, t_a_peak, QA, config.TAU_ANODE_US)
    y_a_bump = anode_bump_basis_Q(t_full, t_b_start, t_b_peak, QB, config.TAU_ANODE_US)
    y_a_full = y_a_main + y_a_bump

    v_c_peak        = float(np.interp(t_c_peak, t_full, y_c_full))
    v_a_main_at_peak = float(y_a_main[int(np.argmin(np.abs(t_full - t_a_peak)))])

    plt.figure(figsize=(23.0, 15.0))
    plt.rcParams.update({"font.size": 25, "axes.labelsize": 25,
                         "xtick.labelsize": 28, "ytick.labelsize": 28,
                         "legend.fontsize": 25})

    C_C = "navy"; C_AM = "darkorange"; C_AB = "green"; C_AF = "red"

    plt.plot(t_us, anode0,   color="magenta",   lw=3.0, label="Anode data")
    plt.plot(t_us, cathode0, color="royalblue", lw=3.0, label="Cathode data")
    plt.plot(t_full, y_c_full, "--", color=C_C,  lw=1.8, alpha=0.8, label="Cathode fit")
    plt.plot(t_full, y_a_main, "--", color=C_AM, lw=1.8, alpha=0.8, label="Anode main fit")
    plt.plot(t_full, y_a_bump, "--", color=C_AB, lw=1.8, alpha=0.8, label="Anode bump fit")
    plt.plot(t_full, y_a_full, "--", color=C_AF, lw=1.8, alpha=0.8, label="Anode full fit")

    plt.axvline(t0,        color="dodgerblue", ls="--", lw=1.6)
    plt.axvline(t_c_peak,  color="red",        ls="--", lw=1.6)
    plt.axvline(t_a_start, color="green",      ls="--", lw=1.6)
    plt.axvline(t_a_peak,  color="orange",     ls="--", lw=1.6)

    y_max  = float(np.nanmax([np.nanmax(anode0), np.nanmax(y_a_full)]))
    y_time = 0.18 * y_max
    for xa, xb, lbl in [(t0, t_c_peak, "T1"),
                         (t_c_peak, t_a_start, "T2"),
                         (t_a_start, t_a_peak, "T3")]:
        xm = 0.5 * (xa + xb)
        plt.annotate("", xy=(xa, y_time), xytext=(xb, y_time),
                     arrowprops=dict(arrowstyle="<->", lw=2.0, color="black"))
        plt.text(xm, y_time + 0.03*y_max, lbl, ha="center", va="bottom", fontsize=25)

    x_va = t_a_peak + 20
    plt.annotate("", xy=(x_va, 0.0), xytext=(x_va, v_a_main_at_peak),
                 arrowprops=dict(arrowstyle="<->", lw=2.4, color=C_AM))
    plt.text(x_va+7, 0.5*v_a_main_at_peak, r"$V_a$", color=C_AM, va="center", fontsize=28)

    x_vc = t_c_peak - 32
    plt.annotate("", xy=(x_vc, 0.0), xytext=(x_vc, v_c_peak),
                 arrowprops=dict(arrowstyle="<->", lw=2.4, color=C_C))
    plt.text(x_vc-7, 0.5*v_c_peak, r"$V_c$", color=C_C, va="center", ha="right", fontsize=28)

    plt.xlabel("Time [µs]"); plt.ylabel("Voltage [mV]")
    plt.xlim(-50.0, 475.0)

    ax    = plt.gca()
    axins = inset_axes(ax, width="34%", height="30%", loc="lower right",
                       bbox_to_anchor=(0.0, 0.009, 1.0, 1.0),
                       bbox_transform=ax.transAxes, borderpad=1.2)
    axins.plot(t_full, y_a_full, lw=2.2, color=C_AF)
    for tx, lbl in [(0.0, "t0"), (t_b_start, "t1"), (t_a_start, "t2")]:
        axins.axvline(tx, color="black", ls="--", lw=1.4)
        axins.text(tx - 6.0, 0.97*float(np.nanmax(y_a_full)), lbl,
                   ha="right", va="top", fontsize=17)
    axins.set_xlim(-20.0, min(float(config.FIT_T_MAX_US), 300.0))
    ymax_ins = float(np.nanmax(y_a_full))
    axins.set_ylim(-0.05*ymax_ins, 1.05*ymax_ins)
    axins.set_xlabel("t [µs]", fontsize=12); axins.set_ylabel("V [mV]", fontsize=12)
    axins.tick_params(labelsize=11)
    if axins.get_legend(): axins.get_legend().remove()

    ax.legend(loc="upper right", framealpha=1.0).set_zorder(100)
    if title: plt.title(title, fontsize=14)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved timing diagram: {out_path}")
