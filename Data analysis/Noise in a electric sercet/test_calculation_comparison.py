#!/usr/bin/env python3
"""
Test Calculation: Compare OLD vs NEW conversion methods

This script calculates what Boltzmann constant you would get with:
1. OLD method (incorrect dBm to V² conversion)
2. NEW method (corrected dBm to V² conversion)
3. NEW + impedance correction

This helps you see the impact of the corrections.
"""

import numpy as np
from pathlib import Path

# Physical constants
K_B_EXPECTED = 1.380649e-23  # J/K
TEMPERATURE = 287.75  # K
DELTA_F = 250  # Hz
GAIN = 100
C_WIN = 1.30
R_ANALYZER = 50  # Ω


def parse_results_file(filepath):
    """Parse the noise analysis results file"""
    resistances = []
    v2_db_mean = []
    v2_db_sigma = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if 'Ω' in line and '|' in line:
            parts = line.split('|')
            resistance_str = parts[0].strip().replace('Ω', '').strip()
            if 'k' in resistance_str:
                resistance = float(resistance_str.replace('k', '')) * 1000
            else:
                resistance = float(resistance_str)

            mean_db = float(parts[1].strip())
            sigma_db = float(parts[2].strip())

            resistances.append(resistance)
            v2_db_mean.append(mean_db)
            v2_db_sigma.append(sigma_db)

    return np.array(resistances), np.array(v2_db_mean), np.array(v2_db_sigma)


def weighted_linear_fit(x, y, sigma_y):
    """Weighted linear regression"""
    weights = 1.0 / (sigma_y ** 2)
    W = np.sum(weights)
    W_x = np.sum(weights * x)
    W_y = np.sum(weights * y)
    W_xx = np.sum(weights * x * x)
    W_xy = np.sum(weights * x * y)

    delta = W * W_xx - W_x ** 2
    a = (W * W_xy - W_x * W_y) / delta
    b = (W_xx * W_y - W_x * W_xy) / delta
    a_err = np.sqrt(W / delta)
    b_err = np.sqrt(W_xx / delta)

    y_fit = a * x + b
    chi2 = np.sum(weights * (y - y_fit) ** 2)
    dof = len(x) - 2
    reduced_chi2 = chi2 / dof

    return a, b, a_err, b_err, chi2, reduced_chi2


def calculate_boltzmann(a, a_err, T, delta_f):
    """Calculate Boltzmann constant from slope"""
    k = a / (4 * T * delta_f)
    k_err = a_err / (4 * T * delta_f)
    return k, k_err


def t_statistic(measured, measured_err, expected):
    """Calculate t-statistic"""
    return abs(measured - expected) / measured_err


def print_section(title):
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70)


def main():
    # Load data
    script_dir = Path(__file__).parent
    results_file = script_dir / 'output/results/noise_analysis_results.txt'

    print_section("TEST CALCULATION: OLD vs NEW METHODS")

    print("\nLoading data from:", results_file)
    resistances, v2_db, v2_db_sigma = parse_results_file(results_file)

    print(f"Number of data points: {len(resistances)}")
    print(f"Resistance range: {resistances.min():.0f} - {resistances.max():.0f} Ω")
    print(f"V² (dBm) range: {v2_db.min():.2f} - {v2_db.max():.2f} dBm")

    # =========================================================================
    # METHOD 1: OLD (INCORRECT) - as in original script
    # =========================================================================
    print_section("METHOD 1: OLD (INCORRECT) CONVERSION")
    print("Formula: V² = 10^(dBm/10)  [treats dBm as dimensionless]")

    # Old conversion
    v2_linear_old = 10 ** (v2_db / 10)
    sigma_v2_old = v2_linear_old * np.log(10) / 10 * v2_db_sigma

    # Apply corrections (gain and window)
    v2_corr_old = v2_linear_old / (GAIN**2 * C_WIN)
    sigma_v2_corr_old = sigma_v2_old / (GAIN**2 * C_WIN)

    # Fit
    a_old, b_old, a_err_old, b_err_old, chi2_old, rchi2_old = \
        weighted_linear_fit(resistances, v2_corr_old, sigma_v2_corr_old)

    # Calculate k_B
    k_old, k_err_old = calculate_boltzmann(a_old, a_err_old, TEMPERATURE, DELTA_F)
    t_old = t_statistic(k_old, 2*k_err_old, K_B_EXPECTED)
    rel_err_old = abs(k_old - K_B_EXPECTED) / K_B_EXPECTED * 100

    print(f"\nResults:")
    print(f"  Slope a: {a_old:.4e} ± {2*a_err_old:.4e} V²/Ω")
    print(f"  k_B: {k_old:.4e} ± {2*k_err_old:.4e} J/K")
    print(f"  Expected k_B: {K_B_EXPECTED:.4e} J/K")
    print(f"  Relative error: {rel_err_old:.1f}%")
    print(f"  t-statistic: {t_old:.2f}")
    print(f"  χ²/dof: {rchi2_old:.4f}")

    # =========================================================================
    # METHOD 2: NEW (CORRECTED) - with proper dBm to V² conversion
    # =========================================================================
    print_section("METHOD 2: NEW (CORRECTED) CONVERSION")
    print(f"Formula: V² = [10^(dBm/10) / 1000] × {R_ANALYZER}Ω")
    print(f"         = 10^(dBm/10) × 0.05")

    # New conversion
    v2_linear_new = 10 ** (v2_db / 10) * 0.05  # Corrected!
    sigma_v2_new = v2_linear_new * np.log(10) / 10 * v2_db_sigma

    # Apply corrections
    v2_corr_new = v2_linear_new / (GAIN**2 * C_WIN)
    sigma_v2_corr_new = sigma_v2_new / (GAIN**2 * C_WIN)

    # Fit
    a_new, b_new, a_err_new, b_err_new, chi2_new, rchi2_new = \
        weighted_linear_fit(resistances, v2_corr_new, sigma_v2_corr_new)

    # Calculate k_B
    k_new, k_err_new = calculate_boltzmann(a_new, a_err_new, TEMPERATURE, DELTA_F)
    t_new = t_statistic(k_new, 2*k_err_new, K_B_EXPECTED)
    rel_err_new = abs(k_new - K_B_EXPECTED) / K_B_EXPECTED * 100

    print(f"\nResults:")
    print(f"  Slope a: {a_new:.4e} ± {2*a_err_new:.4e} V²/Ω")
    print(f"  k_B: {k_new:.4e} ± {2*k_err_new:.4e} J/K")
    print(f"  Expected k_B: {K_B_EXPECTED:.4e} J/K")
    print(f"  Relative error: {rel_err_new:.1f}%")
    print(f"  t-statistic: {t_new:.2f}")
    print(f"  χ²/dof: {rchi2_new:.4f}")

    # =========================================================================
    # METHOD 3: NEW + IMPEDANCE CORRECTION
    # =========================================================================
    print_section("METHOD 3: NEW + IMPEDANCE MATCHING CORRECTION")
    print(f"Formula: V²_oc = V²_meas × [(R + {R_ANALYZER}) / {R_ANALYZER}]²")

    # Apply impedance correction
    impedance_factor = ((resistances + R_ANALYZER) / R_ANALYZER) ** 2
    v2_corr_imp = v2_corr_new * impedance_factor
    sigma_v2_corr_imp = sigma_v2_corr_new * impedance_factor

    # Fit
    a_imp, b_imp, a_err_imp, b_err_imp, chi2_imp, rchi2_imp = \
        weighted_linear_fit(resistances, v2_corr_imp, sigma_v2_corr_imp)

    # Calculate k_B
    k_imp, k_err_imp = calculate_boltzmann(a_imp, a_err_imp, TEMPERATURE, DELTA_F)
    t_imp = t_statistic(k_imp, 2*k_err_imp, K_B_EXPECTED)
    rel_err_imp = abs(k_imp - K_B_EXPECTED) / K_B_EXPECTED * 100

    print(f"\nResults:")
    print(f"  Slope a: {a_imp:.4e} ± {2*a_err_imp:.4e} V²/Ω")
    print(f"  k_B: {k_imp:.4e} ± {2*k_err_imp:.4e} J/K")
    print(f"  Expected k_B: {K_B_EXPECTED:.4e} J/K")
    print(f"  Relative error: {rel_err_imp:.1f}%")
    print(f"  t-statistic: {t_imp:.2f}")
    print(f"  χ²/dof: {rchi2_imp:.4f}")

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print_section("COMPARISON SUMMARY")

    print(f"\n{'Method':<30} {'k_B (J/K)':<20} {'Rel Error':<12} {'t-stat':<8} {'χ²/dof':<8}")
    print("-" * 80)
    print(f"{'1. OLD (incorrect)':<30} {k_old:.4e}    {rel_err_old:>6.1f}%    {t_old:>6.2f}  {rchi2_old:>6.2f}")
    print(f"{'2. NEW (corrected)':<30} {k_new:.4e}    {rel_err_new:>6.1f}%    {t_new:>6.2f}  {rchi2_new:>6.2f}")
    print(f"{'3. NEW + impedance':<30} {k_imp:.4e}    {rel_err_imp:>6.1f}%    {t_imp:>6.2f}  {rchi2_imp:>6.2f}")
    print(f"{'Expected':<30} {K_B_EXPECTED:.4e}    {0:>6.1f}%    {0:>6.2f}")
    print("-" * 80)

    print(f"\nImprovement factors:")
    print(f"  OLD → NEW: t-statistic reduced by {t_old/t_new:.2f}×")
    print(f"  OLD → NEW+imp: t-statistic reduced by {t_old/t_imp:.2f}×")

    print(f"\nSlope comparison:")
    expected_slope = 4 * K_B_EXPECTED * TEMPERATURE * DELTA_F
    print(f"  Expected slope (theory): {expected_slope:.4e} V²/Ω")
    print(f"  Method 1 (OLD): {a_old:.4e} V²/Ω (ratio: {a_old/expected_slope:.3f})")
    print(f"  Method 2 (NEW): {a_new:.4e} V²/Ω (ratio: {a_new/expected_slope:.3f})")
    print(f"  Method 3 (NEW+imp): {a_imp:.4e} V²/Ω (ratio: {a_imp/expected_slope:.3f})")

    # =========================================================================
    # INTERPRETATION
    # =========================================================================
    print_section("INTERPRETATION & RECOMMENDATIONS")

    print("\n1. UNIT CONVERSION FIX:")
    print(f"   The corrected dBm→V² conversion improves t-statistic by ~{t_old/t_new:.1f}×")
    if t_new < 3:
        print("   ✓ Method 2 brings t-statistic to acceptable range!")
    else:
        print("   ⚠️  Method 2 still has high t-statistic - check other parameters")

    print("\n2. IMPEDANCE CORRECTION:")
    if abs(t_imp - t_new) / t_new > 0.1:
        print(f"   Impedance correction changes t-statistic by {abs(t_imp-t_new)/t_new*100:.1f}%")
        if t_imp < t_new:
            print("   ✓ Impedance correction further improves results!")
        else:
            print("   ✗ Impedance correction makes it worse - may not be needed")
    else:
        print("   Impedance correction has minimal effect (<10%)")

    print("\n3. CHI-SQUARED:")
    if rchi2_new < 2:
        print(f"   χ²/dof = {rchi2_new:.2f} - Good fit!")
    else:
        print(f"   χ²/dof = {rchi2_new:.2f} - High, suggests:")
        print("     - Underestimated uncertainties")
        print("     - Systematic errors not accounted for")
        print("     - Model inadequacy")

    print("\n4. RECOMMENDED METHOD:")
    best_t = min(t_new, t_imp)
    if best_t == t_new:
        print("   → Use METHOD 2 (corrected conversion, no impedance correction)")
        print("   → This suggests your preamp has low output impedance")
    else:
        print("   → Use METHOD 3 (corrected conversion + impedance correction)")
        print("   → This suggests impedance matching effects are present")

    print("\n5. REMAINING DISCREPANCY:")
    if min(t_new, t_imp) > 2:
        print(f"   t-statistic is still {min(t_new, t_imp):.1f} (> 2σ)")
        print("   Possible causes:")
        print("     - Gain not exactly 100 (check calibration!)")
        print("     - Window correction factor incorrect")
        print("     - Bandwidth should be ENBW not RBW (factor ~1.36)")
        print("     - Temperature measurement error")
        print("     - Systematic errors in measurement setup")
    else:
        print(f"   ✓ t-statistic = {min(t_new, t_imp):.1f} - Excellent agreement!")

    print("\n" + "="*70)
    print("Run the corrected script to get full results with plots:")
    print("  python3 analyze_noise_vs_resistance_CORRECTED.py")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
