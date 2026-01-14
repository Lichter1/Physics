#!/usr/bin/env python3
"""
Script to analyze noise amplitude vs resistance relationship (CORRECTED VERSION).
Converts dB values to V², applies preamplifier and window corrections,
fits a linear function, and displays results.

CORRECTIONS MADE:
1. Fixed dBm to V² conversion (now properly accounts for 50Ω load and mW→W conversion)
2. Added option for impedance matching correction
3. Added verification output for all parameters
4. Added diagnostic information to help identify remaining issues
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# =============================================================================
# MEASUREMENT PARAMETERS - UPDATE THESE ACCORDING TO YOUR SETUP
# =============================================================================

# === Instrument Settings ===
PREAMPLIFIER_GAIN = 100  # G (e.g., SR552 ×100 voltage gain)
WINDOW_CORRECTION_FACTOR = 1.30  # C_win for Hamming window (verify with FFT analyzer manual!)
ANALYZER_INPUT_IMPEDANCE = 50  # Ω (standard for spectrum analyzers)
SIGMA_MULTIPLIER = 2  # Set to 1 for 1σ (68% CI), 2 for 2σ (95% CI)

# === Impedance Correction Settings ===
# If your preamplifier has HIGH input impedance (>100kΩ), set this to True
# This corrects for voltage divider effect at the analyzer's 50Ω input
APPLY_IMPEDANCE_CORRECTION = True  # CHANGE THIS BASED ON YOUR SETUP!
PREAMP_INPUT_IMPEDANCE = 1e6  # Ω - estimate if preamp has high input impedance

# === Physical Parameters ===
TEMPERATURE = 287.75  # K (14.6°C from folder name)
DELTA_F = 250  # Hz (bandwidth - verify if this is RBW or ENBW!)
BOLTZMANN_EXPECTED = 1.380649e-23  # J/K (known value)

# =============================================================================
# CORRECTION FACTORS EXPLANATION
# =============================================================================
"""
1. dBm → V² conversion:
   - dBm is power in dB relative to 1 milliwatt
   - P_watts = 10^(dBm/10) / 1000
   - V² = P_watts × R_load (where R_load = 50Ω for spectrum analyzers)

2. Impedance matching correction:
   - If preamp has high input impedance, the 50Ω analyzer input creates
     a voltage divider with the source resistance R
   - Measured: V_meas = V_oc × R_analyzer / (R_source + R_analyzer)
   - Need to correct: V_oc² = V_meas² × [(R_source + R_analyzer) / R_analyzer]²

3. Preamplifier gain correction:
   - Measured signal is amplified by gain G
   - True signal: V_true = V_measured / G
   - V²_true = V²_measured / G²

4. Window correction:
   - FFT windowing (e.g., Hamming) affects noise power measurement
   - Correction factor C_win compensates for this
   - Typical values: Hamming ≈ 1.30-1.37 (depends on definition!)
"""

# =============================================================================


def dbm_to_v_squared(dbm_value, r_load=50):
    """
    Convert dBm to V² across a load resistance.

    **CORRECTED VERSION**

    Parameters:
    -----------
    dbm_value : float or array
        Power in dBm (dB relative to 1 milliwatt)
    r_load : float
        Load resistance in ohms (default: 50Ω for spectrum analyzers)

    Returns:
    --------
    v_squared : float or array
        Voltage squared (V²) across the load

    Notes:
    ------
    dBm definition: P_mW = 10^(dBm/10) milliwatts
    Power-voltage relation: P = V²/R
    Therefore: V² = P × R = [10^(dBm/10) / 1000] × R_load
    """
    # Step 1: Convert dBm to power in watts
    p_watts = 10 ** (dbm_value / 10) / 1000  # mW → W

    # Step 2: Convert power to V² using P = V²/R
    # P = V²/R  =>  V² = P × R
    v_squared = p_watts * r_load

    return v_squared


def correct_for_impedance_matching(v_measured_squared, r_source, r_analyzer=50):
    """
    Correct measured V² for impedance matching effects.

    When measuring with a 50Ω analyzer, the source resistance R forms a
    voltage divider with the analyzer input impedance.

    Voltage division: V_measured = V_oc × [R_analyzer / (R_source + R_analyzer)]

    Therefore: V_oc² = V_measured² × [(R_source + R_analyzer) / R_analyzer]²

    Parameters:
    -----------
    v_measured_squared : float or array
        Measured V² values
    r_source : float or array
        Source resistance(s) in ohms
    r_analyzer : float
        Analyzer input impedance in ohms (typically 50Ω)

    Returns:
    --------
    v_oc_squared : float or array
        Open-circuit V² (corrected for loading effect)
    """
    correction_factor = ((r_source + r_analyzer) / r_analyzer) ** 2
    return v_measured_squared * correction_factor


def apply_corrections(v2_meas, gain, c_win, r_source=None, apply_impedance_corr=False, r_analyzer=50):
    """
    Apply preamplifier gain, window correction, and optionally impedance correction.

    Full correction chain:
    1. Remove preamplifier gain: V²_ungained = V²_meas / G²
    2. Remove window effect: V²_unwindowed = V²_ungained / C_win
    3. (Optional) Correct for impedance matching: V²_oc = V²_unwindowed × correction

    Simplified: V²_corr = V²_meas / (G² × C_win) × [impedance correction]

    Parameters:
    -----------
    v2_meas : array-like
        Measured V² values (after converting from dBm)
    gain : float
        Preamplifier voltage gain G
    c_win : float
        Window correction factor
    r_source : array-like, optional
        Source resistances (needed if apply_impedance_corr=True)
    apply_impedance_corr : bool
        Whether to apply impedance matching correction
    r_analyzer : float
        Analyzer input impedance

    Returns:
    --------
    v2_corr : array-like
        Corrected V² values at the resistor
    """
    # Remove gain and window effects
    v2_corr = v2_meas / (gain**2 * c_win)

    # Optionally correct for impedance matching
    if apply_impedance_corr:
        if r_source is None:
            raise ValueError("r_source must be provided for impedance correction")
        v2_corr = correct_for_impedance_matching(v2_corr, r_source, r_analyzer)

    return v2_corr


def parse_results_file(filepath):
    """Parse the noise analysis results file"""
    resistances = []
    v2_db_mean = []
    v2_db_sigma = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Skip header lines and parse data
    for line in lines:
        if 'Ω' in line and '|' in line:
            parts = line.split('|')

            # Extract resistance (remove Ω and k for kilo-ohms)
            resistance_str = parts[0].strip().replace('Ω', '').strip()
            if 'k' in resistance_str:
                resistance = float(resistance_str.replace('k', '')) * 1000
            else:
                resistance = float(resistance_str)

            # Extract mean V² (dB)
            mean_db = float(parts[1].strip())

            # Extract σ (dB)
            sigma_db = float(parts[2].strip())

            resistances.append(resistance)
            v2_db_mean.append(mean_db)
            v2_db_sigma.append(sigma_db)

    return np.array(resistances), np.array(v2_db_mean), np.array(v2_db_sigma)


def propagate_db_uncertainty_to_linear(v2_db, sigma_db, gain, c_win,
                                       r_source=None, apply_impedance_corr=False,
                                       r_analyzer=50):
    """
    Propagate uncertainty from dB scale to linear scale (V²) with corrections.

    **CORRECTED VERSION**

    Steps:
    1. Convert dBm to linear: V²_meas = 10^(dBm/10) × 0.05  [corrected!]
    2. Propagate uncertainty: σ_V²_meas = V²_meas × ln(10)/10 × σ_dB
    3. Apply corrections: V²_corr = V²_meas / (G² · C_win) × [impedance corr]
    4. Propagate uncertainty through corrections
    """
    # Step 1: Convert dBm to linear (CORRECTED)
    v2_meas = dbm_to_v_squared(v2_db, r_load=ANALYZER_INPUT_IMPEDANCE)

    # Step 2: Error propagation from dB to linear
    # d(10^x)/dx = 10^x * ln(10), where x = dB/10
    sigma_v2_meas = v2_meas * np.log(10) / 10 * sigma_db

    # Step 3 & 4: Apply corrections and propagate uncertainty
    v2_corr = apply_corrections(v2_meas, gain, c_win, r_source,
                                apply_impedance_corr, r_analyzer)

    # Uncertainty scales with the correction factor
    correction_factor = gain**2 * c_win
    sigma_v2_corr = sigma_v2_meas / correction_factor

    # If impedance correction applied, uncertainty scales further
    if apply_impedance_corr and r_source is not None:
        impedance_factor = ((r_source + r_analyzer) / r_analyzer) ** 2
        sigma_v2_corr = sigma_v2_corr * impedance_factor
        # Note: This ignores uncertainty in R (assumed exact)

    return v2_corr, sigma_v2_corr


def perform_weighted_linear_fit(x, y, sigma_y):
    """
    Perform weighted linear regression with chi-squared calculation.
    Fits: y = a*x + b
    Returns: a, b, a_err, b_err, chi2, reduced_chi2
    """
    # Weights are inverse variance
    weights = 1.0 / (sigma_y ** 2)

    # Weighted sums
    W = np.sum(weights)
    W_x = np.sum(weights * x)
    W_y = np.sum(weights * y)
    W_xx = np.sum(weights * x * x)
    W_xy = np.sum(weights * x * y)

    # Calculate slope (a) and intercept (b) for y = a*x + b
    delta = W * W_xx - W_x ** 2
    a = (W * W_xy - W_x * W_y) / delta  # slope
    b = (W_xx * W_y - W_x * W_xy) / delta  # intercept

    # Uncertainties
    a_err = np.sqrt(W / delta)
    b_err = np.sqrt(W_xx / delta)

    # Chi-squared
    y_fit = a * x + b
    chi2 = np.sum(weights * (y - y_fit) ** 2)

    # Degrees of freedom
    dof = len(x) - 2  # 2 parameters (slope, intercept)
    reduced_chi2 = chi2 / dof

    return a, b, a_err, b_err, chi2, reduced_chi2


def calculate_boltzmann_constant(a, a_err, T, delta_f):
    """
    Calculate Boltzmann constant from slope of V² vs R fit.

    From Johnson-Nyquist relation: V² = 4kTRΔf
    Slope a = 4kTΔf, so k = a / (4TΔf)

    Parameters:
    -----------
    a : float
        Slope from linear fit (V²/Ω)
    a_err : float
        Uncertainty in slope (1σ)
    T : float
        Temperature (K)
    delta_f : float
        Bandwidth (Hz)

    Returns:
    --------
    k : float
        Boltzmann constant (J/K)
    k_err : float
        Uncertainty in k (1σ)
    """
    k = a / (4 * T * delta_f)
    # Error propagation: k_err/k = a_err/a (T and Δf are exact)
    k_err = a_err / (4 * T * delta_f)
    return k, k_err


def calculate_internal_resistance(a, b, a_err, b_err):
    """
    Calculate effective internal resistance R₀ = b/a

    From the model: V² = a·R + b
    where b represents the noise from internal resistance R₀
    and a = 4kTΔf, so b = 4kTΔf·R₀ = a·R₀
    Therefore: R₀ = b/a

    Parameters:
    -----------
    a, b : float
        Slope and intercept from fit
    a_err, b_err : float
        Uncertainties (1σ)

    Returns:
    --------
    R0 : float
        Internal resistance (Ω)
    R0_err : float
        Uncertainty in R0 (1σ)
    """
    R0 = b / a
    # Error propagation for quotient: (ΔR0/R0)² = (Δa/a)² + (Δb/b)²
    R0_err = R0 * np.sqrt((a_err/a)**2 + (b_err/b)**2)
    return R0, R0_err


def perform_t_test(measured, measured_err, expected):
    """
    Perform t-test to compare measured value with expected value.

    t = |measured - expected| / measured_err

    Parameters:
    -----------
    measured : float
        Measured value
    measured_err : float
        Uncertainty in measured value (1σ)
    expected : float
        Expected/reference value

    Returns:
    --------
    t_statistic : float
        t-test statistic
    agreement : str
        Interpretation of agreement
    """
    t_statistic = abs(measured - expected) / measured_err

    # Interpret t-statistic
    if t_statistic < 1:
        agreement = "Excellent agreement (< 1σ)"
    elif t_statistic < 2:
        agreement = "Good agreement (< 2σ)"
    elif t_statistic < 3:
        agreement = "Acceptable agreement (< 3σ)"
    else:
        agreement = f"Significant discrepancy (> 3σ)"

    return t_statistic, agreement


def verify_parameters():
    """Print verification information for all measurement parameters"""
    print("\n" + "="*70)
    print("PARAMETER VERIFICATION AND DIAGNOSTICS")
    print("="*70)

    print("\n1. UNIT CONVERSION:")
    print(f"   Analyzer input impedance: {ANALYZER_INPUT_IMPEDANCE} Ω")
    print(f"   Conversion: dBm → watts → V² across {ANALYZER_INPUT_IMPEDANCE}Ω load")
    print(f"   Formula: V² = [10^(dBm/10) / 1000] × {ANALYZER_INPUT_IMPEDANCE}")

    print("\n2. PREAMPLIFIER:")
    print(f"   Gain: G = {PREAMPLIFIER_GAIN}")
    print(f"   As voltage gain: {PREAMPLIFIER_GAIN}× (linear)")
    print(f"   As dB: {20*np.log10(PREAMPLIFIER_GAIN):.1f} dB (voltage)")
    print(f"   As power gain: {PREAMPLIFIER_GAIN**2}× or {10*np.log10(PREAMPLIFIER_GAIN**2):.1f} dB")
    print(f"   ⚠️  VERIFY: Is this the voltage gain (not power gain)?")

    print("\n3. WINDOW CORRECTION:")
    print(f"   C_win = {WINDOW_CORRECTION_FACTOR}")
    print(f"   Typical values for Hamming window:")
    print(f"     - Power spectrum: 1.36-1.37")
    print(f"     - Amplitude spectrum: different")
    print(f"   ⚠️  VERIFY: Check your FFT analyzer manual!")

    print("\n4. IMPEDANCE MATCHING:")
    print(f"   Apply correction: {APPLY_IMPEDANCE_CORRECTION}")
    if APPLY_IMPEDANCE_CORRECTION:
        print(f"   Preamp input impedance: {PREAMP_INPUT_IMPEDANCE/1e6:.1f} MΩ")
        print(f"   Analyzer input: {ANALYZER_INPUT_IMPEDANCE} Ω")
        print(f"   Correction accounts for voltage divider at analyzer input")
        print(f"   ⚠️  VERIFY: Is preamp input impedance >> source resistance?")
    else:
        print(f"   ⚠️  WARNING: Not correcting for impedance matching!")
        print(f"   This may cause systematic error if preamp has high input Z")

    print("\n5. PHYSICAL CONSTANTS:")
    print(f"   Temperature: T = {TEMPERATURE} K = {TEMPERATURE-273.15:.2f} °C")
    print(f"   Bandwidth: Δf = {DELTA_F} Hz")
    print(f"   ⚠️  VERIFY: Is this Resolution BW (RBW) or Equivalent Noise BW (ENBW)?")
    print(f"   For Hamming window: ENBW ≈ 1.36 × RBW")

    print("\n6. COMBINED CORRECTION FACTOR:")
    corr_factor = PREAMPLIFIER_GAIN**2 * WINDOW_CORRECTION_FACTOR
    print(f"   G² × C_win = {corr_factor:.2f}")
    print(f"   Measured V² is divided by this to get true V²")

    print("\n7. EXPECTED THERMAL NOISE:")
    print(f"   Expected slope: a = 4kTΔf")
    expected_slope = 4 * BOLTZMANN_EXPECTED * TEMPERATURE * DELTA_F
    print(f"                     = 4 × {BOLTZMANN_EXPECTED:.3e} × {TEMPERATURE} × {DELTA_F}")
    print(f"                     = {expected_slope:.4e} V²/Ω")

    print("\n" + "="*70)


def main():
    # File path
    script_dir = Path(__file__).parent
    results_file = script_dir / 'output/results/noise_analysis_results.txt'

    # Verify parameters
    verify_parameters()

    # Display correction parameters
    print("\n" + "="*60)
    print("CORRECTION PARAMETERS (Applied to Analysis)")
    print("="*60)
    print(f"Preamplifier Gain (G):           {PREAMPLIFIER_GAIN}")
    print(f"Window Correction Factor (C_win): {WINDOW_CORRECTION_FACTOR}")
    print(f"Analyzer Input Impedance:         {ANALYZER_INPUT_IMPEDANCE} Ω")
    print(f"Apply Impedance Correction:       {APPLY_IMPEDANCE_CORRECTION}")
    print(f"Combined correction (G² · C_win): {PREAMPLIFIER_GAIN**2 * WINDOW_CORRECTION_FACTOR:.2f}")
    confidence_level = 68.3 if SIGMA_MULTIPLIER == 1 else 95.4 if SIGMA_MULTIPLIER == 2 else SIGMA_MULTIPLIER
    print(f"Uncertainty multiplier:          {SIGMA_MULTIPLIER}σ ({confidence_level}% CI)")
    print("="*60 + "\n")

    # Parse data
    print("Parsing data from results file...")
    resistances, v2_db_mean, v2_db_sigma = parse_results_file(results_file)

    # Convert to linear scale (V²) and apply corrections
    print("\n" + "="*60)
    print("UNIT CONVERSION AND CORRECTIONS")
    print("="*60)
    print("Step 1: Convert dBm → V² (CORRECTED)")
    print(f"  Formula: V² = [10^(dBm/10) / 1000] × {ANALYZER_INPUT_IMPEDANCE}Ω")
    print("Step 2: Apply preamplifier and window corrections")
    print(f"  V²_corr = V²_meas / (G² · C_win)")
    if APPLY_IMPEDANCE_CORRECTION:
        print("Step 3: Apply impedance matching correction")
        print(f"  V²_oc = V²_corr × [(R + {ANALYZER_INPUT_IMPEDANCE}) / {ANALYZER_INPUT_IMPEDANCE}]²")
    print("="*60 + "\n")

    v2_corr, v2_sigma_corr = propagate_db_uncertainty_to_linear(
        v2_db_mean, v2_db_sigma, PREAMPLIFIER_GAIN, WINDOW_CORRECTION_FACTOR,
        r_source=resistances if APPLY_IMPEDANCE_CORRECTION else None,
        apply_impedance_corr=APPLY_IMPEDANCE_CORRECTION,
        r_analyzer=ANALYZER_INPUT_IMPEDANCE
    )

    print(f"Data Summary (Corrected Values):")
    print(f"Resistances (Ω): {resistances}")
    print(f"V²_corr values (V²): {v2_corr}")
    print(f"Uncertainties (V²): {v2_sigma_corr}")

    # Perform weighted linear fit: V² = a*R + b
    print("\nPerforming weighted linear regression...")
    a, b, a_err, b_err, chi2, reduced_chi2 = \
        perform_weighted_linear_fit(resistances, v2_corr, v2_sigma_corr)

    # Apply sigma multiplier to uncertainties
    a_err_reported = a_err * SIGMA_MULTIPLIER
    b_err_reported = b_err * SIGMA_MULTIPLIER

    # Display fit results
    print("\n" + "="*60)
    print(f"LINEAR FIT RESULTS: V² = a·R + b ({SIGMA_MULTIPLIER}σ uncertainties)")
    print("="*60)
    print(f"Slope (a):      {a:.4e} ± {a_err_reported:.4e} V²/Ω")
    print(f"Intercept (b):  {b:.4e} ± {b_err_reported:.4e} V²")
    print(f"χ²:             {chi2:.4f}")
    print(f"χ²/dof:         {reduced_chi2:.4f}")
    print(f"Degrees of freedom: {len(resistances) - 2}")

    # Interpret chi-squared
    if reduced_chi2 < 0.5:
        chi_interpretation = "Overestimated uncertainties"
    elif reduced_chi2 < 2:
        chi_interpretation = "Good fit, uncertainties reasonable"
    else:
        chi_interpretation = "Underestimated uncertainties or model inadequate"
    print(f"χ²/dof interpretation: {chi_interpretation}")
    print("="*60)

    # Calculate Boltzmann constant
    print("\n" + "="*60)
    print("DERIVED PHYSICAL QUANTITIES")
    print("="*60)

    k_measured, k_err_1sigma = calculate_boltzmann_constant(a, a_err, TEMPERATURE, DELTA_F)
    k_err_reported = k_err_1sigma * SIGMA_MULTIPLIER

    print(f"\nBoltzmann constant (k):")
    print(f"  Measured:  k = ({k_measured:.4e} ± {k_err_reported:.4e}) J/K  ({SIGMA_MULTIPLIER}σ)")
    print(f"  Expected:  k = {BOLTZMANN_EXPECTED:.4e} J/K")

    # Calculate relative error
    relative_error = abs(k_measured - BOLTZMANN_EXPECTED) / BOLTZMANN_EXPECTED * 100
    print(f"  Relative error: {relative_error:.2f}%")

    # Perform t-test for Boltzmann constant (always use 2σ for t-test)
    t_k, agreement_k = perform_t_test(k_measured, 2*k_err_1sigma, BOLTZMANN_EXPECTED)
    print(f"  t-statistic: {t_k:.2f}")
    print(f"  Agreement: {agreement_k}")

    # Calculate effective internal resistance
    R0, R0_err_1sigma = calculate_internal_resistance(a, b, a_err, b_err)
    R0_err_reported = R0_err_1sigma * SIGMA_MULTIPLIER

    print(f"\nEffective internal resistance (R₀):")
    print(f"  R₀ = ({R0:.2f} ± {R0_err_reported:.2f}) Ω  ({SIGMA_MULTIPLIER}σ)")

    # Diagnostic: compare with expected slope
    expected_slope = 4 * BOLTZMANN_EXPECTED * TEMPERATURE * DELTA_F
    print(f"\n--- DIAGNOSTIC ---")
    print(f"Expected slope (using known k_B): {expected_slope:.4e} V²/Ω")
    print(f"Measured slope:                   {a:.4e} V²/Ω")
    print(f"Ratio (measured/expected):        {a/expected_slope:.3f}")

    print("="*60)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot data with error bars (using multiplied uncertainties)
    ax.errorbar(resistances, v2_corr, yerr=v2_sigma_corr * SIGMA_MULTIPLIER,
                fmt='o', markersize=8, capsize=5, capthick=2,
                label=f'Measured data (±{SIGMA_MULTIPLIER}σ)', color='blue', ecolor='blue', alpha=0.7)

    # Plot fit line
    x_fit = np.linspace(0, max(resistances) * 1.1, 100)
    y_fit = a * x_fit + b
    ax.plot(x_fit, y_fit, 'r-', linewidth=2,
            label=f'Linear fit: V² = a·R + b')

    # Add text box with fit parameters
    impedance_corr_text = "Yes" if APPLY_IMPEDANCE_CORRECTION else "No"
    textstr = '\n'.join([
        f'Fit Results (±{SIGMA_MULTIPLIER}σ):',
        f'a = ({a:.4e} ± {a_err_reported:.4e}) V²/Ω',
        f'b = ({b:.4e} ± {b_err_reported:.4e}) V²',
        f'χ²/dof = {reduced_chi2:.4f}',
        '',
        'Derived Quantities:',
        f'k = ({k_measured:.3e} ± {k_err_reported:.3e}) J/K',
        f't-stat = {t_k:.2f} ({agreement_k.split("(")[0].strip()})',
        f'R₀ = ({R0:.1f} ± {R0_err_reported:.1f}) Ω',
        '',
        f'Corrections: G={PREAMPLIFIER_GAIN}, C_win={WINDOW_CORRECTION_FACTOR}',
        f'Impedance corr: {impedance_corr_text}'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')

    # Labels and formatting
    ax.set_xlabel('Resistance (Ω)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Noise Amplitude V²_corr (V²)', fontsize=12, fontweight='bold')
    title_suffix = " [CORRECTED]" if not APPLY_IMPEDANCE_CORRECTION else " [CORRECTED + Impedance]"
    ax.set_title('White Noise Amplitude vs Resistance' + title_suffix,
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Save figure
    output_dir = script_dir / 'output/results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'noise_vs_resistance_fit_CORRECTED.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Show plot
    plt.show()

    print("\nAnalysis complete!")
    print("\n⚠️  IMPORTANT: Review the parameter verification above!")
    print("    Make sure all instrument settings are correct.")


if __name__ == '__main__':
    main()
