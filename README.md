#PrM_analysis
Analysis code for Purity Monitor (PrM) waveforms in liquid argon - waveform fitting, electron lifetime extraction, and E2 field scan analysis.
---

1. Environment variables

Set these before running any script. 

PRM_METADATA_CSV - The metadata CSV containing run settings (voltages, E fields, comments) 
PRM_DATA_DIR - The folder with the raw waveform CSV files; It should only contain files for the run you are interested in
PRM_OUTPUT_DIR - Where the fit PNGs are saved 
PRM_REDUCED_DIR - Where results CSVs, uncertainty files, and summary plots are saved 


2. What each file does

config.py - Where all constants live
core.py - All shared functions
quick_look.py - Fit a single waveform immediately and show the fit plot. Useful during data-taking to check signal quality. For example: python quick_look.py 20250917T011011.csv 
run_lifetime.py - Analysis for the lifetime run
run_e2_scan.py - Analysis for the E2 field scan.

plot_lifetime_vs_time.py - Reads results_lifetime.csv; Generates tau vs time (for all individual files and for the 3-min groups), QA/QC vs time
plot_ratio_vs_time.py - Reads results_e2_scan.csv; Generates QA/QC vs time since first run 
plot_ratio_vs_e2.py - Reads results_e2_scan.csv; Generates QA/QC vs E2
plot_ratio_vs_e2_with_o2.py - Reads results_e2_scan.csv; Generates QA/QC vs E2 overlaid with theory curves
plot_t2_vs_e2.py - Reads results_e2_scan.csv; Generates Transit time T2 vs E2 with theory band

Each plot script also writes a .list file alongside every plot.



3. What to change between runs

#For a new standard lifetime run

In config.py, update the expected voltage settings to match the run:

VC_EXP   = -109   # Vc  [V]
VAG_EXP =  957   # Vag [V]
VA_EXP   = 1356   # Va  [V]

These are used by `run_lifetime.py` to select only waveforms matching those voltages from the metadata CSV.



#For a new E2 scan

In config.py, update the run prefix to match the date of your new scan:

RUN_PREFIXES = ["20250917"]   # <- change to your run date, e.g. ["20260115"]

This selects only files whose filenames start with that date string. Multiple prefixes can be listed if the scan spanned more than one day.
Files are selected for the E2 scan by looking for rows where the Comments column contains E2 = — this identifies which runs belong to the scan. The actual E1/E2/E3 field values are then read from dedicated E1, E2, E3 columns in the metadata CSV.



#Check the detector geometry

In config.py, update the relevant sections:

#Preamplifier
Cf_a = 1.4        # anode feedback capacitance [pF]
Cf_c = 1.4        # cathode feedback capacitance [pF]
VOUT_GAIN = 2.0   # CSP output gain

#Time constants
TAU_CATHODE_US = 133.8
TAU_ANODE_US    = 134.5

#Detector distances
D1_CM  = 1.9   # cathode → cathode grid [cm]
D2_CM  = 5.8   # cathode grid → anode grid [cm]
D3_CM  = 1.2   # anode grid → anode [cm]


4. Typical order of running

Define the environments

run_lifetime.py 
run_e2_scan.py
---> These generate the .csv files

plot_lifetime_vs_time.py
plot_ratio_vs_time.py
plot_ratio_vs_e2.py
plot_ratio_vs_e2_with_o2.py
plot_t2_vs_e2.py
---> These make the plots
