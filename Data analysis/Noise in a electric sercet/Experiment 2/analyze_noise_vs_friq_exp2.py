#!/usr/bin/env python3
"""
Script to analyze noise amplitude vs bandwidth (FFT frequency resolution) relationship.
Experiment 2: Fixed R = 68.3 kΩ, T = 14.6°C, varying bandwidth (Δf)

From Johnson-Nyquist: V² = 4kTRΔf
With fixed R and T, we expect: V² = (4kTR) · Δf = slope · Δf
So noise power should scale linearly with bandwidth.

This script:
- Converts dB values to V²
- Applies preamplifier and window corrections
- Fits a linear function V² = a·Δf + b
- Derives physical quantities from the fit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import re

# =============================================================================
# MEASUREMENT PARAMETERS - UPDATE THESE ACCORDING TO YOUR SETUP
# =============================================================================
PREAMPLIFIER_GAIN = 100  # G (e.g., SR552 ×100)
WINDOW_CORRECTION_FACTOR = 1.30  # C_win for Hamming window
SIGMA_MULTIPLIER = 2  # Set to 1 for 1σ (68% CI), 2 for 2σ (95% CI)

# Physical parameters for Boltzmann constant calculation
TEMPERATURE = 287.75  # K (14.6°C converted to Kelvin)
RESISTANCE = 68300  # Ω (68.3 kΩ)
BOLTZMANN_EXPECTED = 1.380649e-23  # J/K (known value)
# =============================================================================


def db_to_v_squared(db_value):
    """Convert dB value to V²"""
    # dB = 10 * log10(V²)
    # V² = 10^(dB/10)
    return 10 ** (db_value / 10)


def apply_corrections(v2_meas, gain, c_win):
    """
    Apply preamplifier gain and window correction.

    V²_corr = V²_meas / (G² · C_win)

    Parameters:
    -----------
    v2_meas : array-like
        Measured V² values (after converting from dB)
    gain : float
        Preamplifier gain G
    c_win : float
        Window correction factor (1.30 for Hamming)

    Returns:
    --------
    v2_corr : array-like
        Corrected V² values at the resistor input
    """
    return v2_meas / (gain**2 * c_win)


def parse_results_file(filepath):
    """
    Parse the noise analysis results file for Experiment 2.

    Format:
    Bandwidth | μ (Linear) | σ (Linear) | Mean (dB)

    Returns linear power values directly (no dB conversion needed).
    """
    bandwidths = []
    mu_linear = []
    sigma_linear = []
    mean_db = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Skip header lines and parse data
    for line in lines:
        if 'Hz' in line and '|' in line and 'Bandwidth' not in line:
            parts = line.split('|')

            # Extract bandwidth (remove Hz)
            bw_str = parts[0].strip().replace('Hz', '').strip()
            bandwidth = float(bw_str)

            # Extract μ (linear power) - Gaussian fit mean
            mu = float(parts[1].strip())

            # Extract σ (linear power) - Gaussian fit sigma
            sigma = float(parts[2].strip())

            # Extract mean dB (for reference)
            db_val = float(parts[3].strip())

            bandwidths.append(bandwidth)
            mu_linear.append(mu)
            sigma_linear.append(sigma)
            mean_db.append(db_val)

    return np.array(bandwidths), np.array(mu_linear), np.array(sigma_linear), np.array(mean_db)


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
    reduced_chi2 = chi2 / dof if dof > 0 else float('inf')

    return a, b, a_err, b_err, chi2, reduced_chi2


def calculate_boltzmann_constant(a, a_err, T, R):
    """
    Calculate Boltzmann constant from slope of V² vs Δf fit.

    From Johnson-Nyquist relation: V² = 4kTRΔf
    Slope a = 4kTR, so k = a / (4TR)

    Parameters:
    -----------
    a : float
        Slope from linear fit (V²/Hz)
    a_err : float
        Uncertainty in slope (1σ)
    T : float
        Temperature (K)
    R : float
        Resistance (Ω)

    Returns:
    --------
    k : float
        Boltzmann constant (J/K)
    k_err : float
        Uncertainty in k (1σ)
    """
    k = a / (4 * T * R)
    # Error propagation: k_err/k = a_err/a (T and R are exact)
    k_err = a_err / (4 * T * R)
    return k, k_err


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


def main():
    # File path
    script_dir = Path(__file__).parent
    results_file = script_dir / 'output/Experiment_2/results/noise_analysis_results.txt'

    # Create output directory
    output_dir = script_dir / 'output' / 'Experiment_2' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Display correction parameters
    print("="*60)
    print("CORRECTION PARAMETERS")
    print("="*60)
    print(f"Preamplifier Gain (G):           {PREAMPLIFIER_GAIN}")
    print(f"Window Correction Factor (C_win): {WINDOW_CORRECTION_FACTOR}")
    print(f"Combined correction factor (G² · C_win): {PREAMPLIFIER_GAIN**2 * WINDOW_CORRECTION_FACTOR:.2f}")
    confidence_level = 68.3 if SIGMA_MULTIPLIER == 1 else 95.4 if SIGMA_MULTIPLIER == 2 else SIGMA_MULTIPLIER
    print(f"Uncertainty multiplier:          {SIGMA_MULTIPLIER}σ ({confidence_level}% CI)")
    print("="*60 + "\n")

    # Display fixed experimental parameters
    print("="*60)
    print("EXPERIMENT 2 - FIXED PARAMETERS")
    print("="*60)
    print(f"Temperature (T):  {TEMPERATURE} K ({TEMPERATURE - 273.15:.1f}°C)")
    print(f"Resistance (R):   {RESISTANCE} Ω ({RESISTANCE/1000:.1f} kΩ)")
    print("Variable:         FFT Bandwidth (Δf)")
    print("="*60 + "\n")

    # Check if results file exists
    if not results_file.exists():
        print(f"Error: Results file not found at {results_file}")
        print("Please run noise_filter_analysis_exp2.py first to generate the results.")
        return

    # Parse data (now in linear power scale from Gaussian fit)
    print("Parsing data from results file...")
    bandwidths, v2_linear, v2_sigma_linear, v2_db_ref = parse_results_file(results_file)

    print(f"\nData from Gaussian fit (Linear Power Scale):")
    print(f"  Bandwidths (Hz): {bandwidths}")
    print(f"  μ (linear):      {v2_linear}")
    print(f"  σ (linear):      {v2_sigma_linear}")
    print(f"  Mean (dB ref):   {v2_db_ref}")

    # Apply preamplifier gain and window corrections to linear values
    print("\nApplying corrections to linear power values...")
    print(f"  V²_corr = V²_meas / (G² · C_win)")
    print(f"  Correction factor: {PREAMPLIFIER_GAIN**2 * WINDOW_CORRECTION_FACTOR:.2f}")

    v2_corr = apply_corrections(v2_linear, PREAMPLIFIER_GAIN, WINDOW_CORRECTION_FACTOR)
    v2_sigma_corr = v2_sigma_linear / (PREAMPLIFIER_GAIN**2 * WINDOW_CORRECTION_FACTOR)

    print(f"\nData Summary (Corrected Values):")
    print(f"Bandwidths (Hz): {bandwidths}")
    print(f"V²_corr values (V²): {v2_corr}")
    print(f"Uncertainties (V²): {v2_sigma_corr}")

    # Perform weighted linear fit: V² = a*Δf + b
    print("\nPerforming weighted linear regression...")
    a, b, a_err, b_err, chi2, reduced_chi2 = \
        perform_weighted_linear_fit(bandwidths, v2_corr, v2_sigma_corr)

    # Apply sigma multiplier to uncertainties
    a_err_reported = a_err * SIGMA_MULTIPLIER
    b_err_reported = b_err * SIGMA_MULTIPLIER

    # Display fit results
    print("\n" + "="*60)
    print(f"LINEAR FIT RESULTS: V² = a·Δf + b ({SIGMA_MULTIPLIER}σ uncertainties)")
    print("="*60)
    print(f"Slope (a):      {a:.4e} ± {a_err_reported:.4e} V²/Hz")
    print(f"Intercept (b):  {b:.4e} ± {b_err_reported:.4e} V²")
    print(f"χ²:             {chi2:.4f}")
    print(f"χ²/dof:         {reduced_chi2:.4f}")
    print(f"Degrees of freedom: {len(bandwidths) - 2}")
    print("="*60)

    # Calculate Boltzmann constant
    print("\n" + "="*60)
    print("DERIVED PHYSICAL QUANTITIES")
    print("="*60)

    k_measured, k_err_1sigma = calculate_boltzmann_constant(a, a_err, TEMPERATURE, RESISTANCE)
    k_err_reported = k_err_1sigma * SIGMA_MULTIPLIER

    print(f"\nBoltzmann constant (k):")
    print(f"  From: V² = 4kTRΔf, slope a = 4kTR")
    print(f"  k = a / (4TR)")
    print(f"  Measured:  k = ({k_measured:.4e} ± {k_err_reported:.4e}) J/K  ({SIGMA_MULTIPLIER}σ)")
    print(f"  Expected:  k = {BOLTZMANN_EXPECTED:.4e} J/K")

    # Calculate relative error
    relative_error = abs(k_measured - BOLTZMANN_EXPECTED) / BOLTZMANN_EXPECTED * 100
    print(f"  Relative error: {relative_error:.2f}%")

    # Perform t-test for Boltzmann constant
    t_k, agreement_k = perform_t_test(k_measured, k_err_1sigma, BOLTZMANN_EXPECTED)
    print(f"  t-statistic: {t_k:.2f}")
    print(f"  Agreement: {agreement_k}")

    # Calculate expected slope for comparison
    expected_slope = 4 * BOLTZMANN_EXPECTED * TEMPERATURE * RESISTANCE
    print(f"\nExpected slope (from known k):")
    print(f"  a_expected = 4kTR = {expected_slope:.4e} V²/Hz")
    print(f"  a_measured = {a:.4e} V²/Hz")
    print(f"  Ratio (measured/expected): {a/expected_slope:.4f}")

    print("="*60)

    # Create figure
    fig, ax = plt.subplots(figsize=(11, 6))

    # Plot data with error bars (using multiplied uncertainties)
    ax.errorbar(bandwidths, v2_corr, yerr=v2_sigma_corr * SIGMA_MULTIPLIER,
                fmt='o', markersize=8, capsize=5, capthick=2,
                label=f'Measured data ({SIGMA_MULTIPLIER}σ error bars)', color='blue', ecolor='blue', alpha=0.7)

    # Plot fit line
    x_fit = np.linspace(0, max(bandwidths) * 1.1, 100)
    y_fit = a * x_fit + b
    ax.plot(x_fit, y_fit, 'r-', linewidth=2,
            label=f'Linear fit: V² = a·Δf + b')

    # Add text box with fit parameters
    textstr = '\n'.join([
        f'Fit Results ({SIGMA_MULTIPLIER}σ):',
        f'a = ({a:.2e} ± {a_err_reported:.2e}) V²/Hz',
        f'b = ({b:.2e} ± {b_err_reported:.2e}) V²',
        f'χ²/dof = {reduced_chi2:.4f}',
        '',
        'Derived Quantities:',
        f'k = ({k_measured:.2e} ± {k_err_reported:.2e}) J/K',
        f'k_expected = {BOLTZMANN_EXPECTED:.2e} J/K',
        f'Relative error: {relative_error:.1f}%',
        '',
        f'Fixed: R={RESISTANCE/1000:.1f}kΩ, T={TEMPERATURE:.1f}K',
        f'Corrections: G={PREAMPLIFIER_GAIN}, C_win={WINDOW_CORRECTION_FACTOR}'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, family='monospace')

    # Labels and formatting
    ax.set_xlabel('Bandwidth Δf (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Noise Amplitude V²_corr (V²)', fontsize=12, fontweight='bold')
    ax.set_title('White Noise Amplitude vs Bandwidth - Experiment 2\n(R=68.3kΩ, T=14.6°C)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Set axis to start from 0
    ax.set_xlim(0, max(bandwidths) * 1.15)
    ax.set_ylim(bottom=0)

    # Save figure
    output_file = output_dir / 'noise_vs_bandwidth_fit.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")

    # Show plot
    plt.show()

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
