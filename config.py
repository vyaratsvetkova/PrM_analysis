import os
import sys

def env(name: str) -> str:
    """Return required environment variable or exit with a clear message."""
    val = os.environ.get(name)
    if not val:
        sys.exit(f"[ERROR] Missing environment variable: {name}")
    return val


# E2-scan run selection

RUN_PREFIXES = ["20250917"]

# Expected voltage settings for the standard lifetime run
VC_EXP = -109   # Vc  [V]
VAG_EXP = 957   # Vag [V]
VA_EXP = 1356   # Va  [V]



# ===========================================================================
# Constants
# ===========================================================================

Cf_a = 1.4
Cf_c = 1.4
VOUT_GAIN = 2.0

h0_a = VOUT_GAIN / Cf_a   #[mV/fC]
h0_c = VOUT_GAIN / Cf_c



# ===========================================================================
# Decay constants [us]
# ===========================================================================

TAU_CATHODE_US = 133.8
TAU_ANODE_US   = 134.5


# ===========================================================================
# Temperature [K]
# ===========================================================================

T_K = 90.28   #LAr temperature
T0_K = 87.3


# ===========================================================================
# Detector geometry [cm]
# ===========================================================================

D1_CM  = 1.9 # cathode → cathode grid
D2_CM  = 5.8 # cathode grid → anode grid  (main drift region)
D3_CM  = 1.2 # anode grid → anode
DD2_CM = 0.2 # uncertainty on D2
L_CM   = 17.6 # total drift length
DL_CM  = 0.2 # uncertainty on L


# ===========================================================================
# Sigmoid function
# ===========================================================================

DT = 0.001
K_SIGMOID = 5.0 / DT


# ===========================================================================
# Waveform smoothing (Savitzky–Golay)
# ===========================================================================

SMOOTH_WINDOW = 11   #window for analysis smoothing
SMOOTH_POLY = 2
SMOOTH_WINDOW_GUESS = 21   #window for initial-guess smoothing
SMOOTH_POLY_GUESS = 3


# ===========================================================================
# Fit window
# ===========================================================================

FIT_T_MAX_US = 380.0   # standard analysis upper limit
POST_ANODE_PEAK_US = 100.0   # extra time after anode peak for E2-scan fits


# ===========================================================================
# Fit exclusion windows
# ===========================================================================

EXCLUDE_ANODE_START_US = -10.0
EXCLUDE_ANODE_END_US = 25.0
EXCLUDE_CATHODE_START_US = 0.0
EXCLUDE_CATHODE_END_US = 15.0


# ===========================================================================
# Initial-guess parameters
# ===========================================================================

ANODE_RISE_TIME_MIN_US = 8.0
ANODE_RISE_TIME_MAX_US = 37.0


# ===========================================================================
# Bump-window offsets
# ===========================================================================

BUMP_LEFT_OFFSET_US = 10.0   # lifetime: bump window starts at t_c_peak - this
BUMP_RIGHT_OFFSET_US = 10.0   # lifetime: bump window ends at t_a_start_guess - this
BUMP_LEFT_OFFSET_E2_US = 40.0   # E2-scan: wider left edge
BUMP_RIGHT_OFFSET_E2_US = 10.0


# ===========================================================================
# Waveform selection quality cuts
# ===========================================================================


# Anode tail must return within this many mV of baseline after the peak.
ANODE_TAIL_THRESHOLD_MV = 5.0      # lifetime run
ANODE_TAIL_THRESHOLD_E2_MV = 5.0   # E2 scan

# Crossing rejection: after the anode peak, da = tail_anode - tail_cathode
#A waveform is rejected if within T_CROSS_LIMIT_US:
#   da changes sign  ->  channels actually crossed
T_CROSS_LIMIT_US = 680.0  # µs

# How many samples after the anode peak to skip before starting tail checks.
SKIP_AFTER_PEAK_PTS = 5


# ===========================================================================
# Baseline
# ===========================================================================

BASELINE_SAMPLES = 50


# ===========================================================================
# Anode peak penalty
# ===========================================================================

PEAK_PENALTY_SIGMA_MV = 0.05
PEAK_PENALTY_N = 10


# ===========================================================================
# Per-event fit plot quality
# ===========================================================================

FIT_PLOT_DPI = 80


# ===========================================================================
# Output file names
# ===========================================================================

RESULTS_LIFETIME_CSV = "results_lifetime.csv"
RESULTS_E2_SCAN_CSV = "results_e2_scan.csv"
INITIAL_RESULTS_CSV = "results_q_initial.csv"
INITIAL_RESULTS_E2_CSV = "results_q_initial_e2.csv"
TYP_UNC_CSV = "typical_uncertainties_q.csv"
TYP_UNC_E2_CSV = "typical_uncertainties_q_e2.csv"


