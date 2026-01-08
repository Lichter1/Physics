#!/usr/bin/env python3
"""
Microwave Polarization Experiment - Graph Generation Script
===========================================================

This script generates professional graphs for:
1. Experiment A: Polarization analysis with Malus's Law fitting
2. Experiment B: Polarizing grating comparison

Author: Omri Lichter
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set high-quality plotting parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# ============================================================================
# EXPERIMENT A DATA
# ============================================================================

# Angle measurements (degrees)
angles_deg = np.array([0, 10.1, 20.05, 29.8, 40.05, 49.95, 60.75, 70.25, 80.25, 90])

# Voltage measurements (V)
voltage = np.array([8.7, 8.1, 7.3, 6, 4.1, 2.4, 0.9, 0.27, 0.04, 0.01])

# Voltage uncertainty (V)
voltage_uncertainty = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

# Angle uncertainty (degrees)
angle_uncertainty = 0.1

# ============================================================================
# EXPERIMENT A: ANALYSIS AND FITTING
# ============================================================================

# Convert angles to radians for calculations
angles_rad = np.radians(angles_deg)

# Calculate contrast and DOP
V_max = np.max(voltage)
V_min = np.min(voltage)
contrast = (V_max - V_min) / (V_max + V_min)
DOP = (V_max - V_min) / (V_max + V_min)

print("=" * 70)
print("EXPERIMENT A: POLARIZATION ANALYSIS")
print("=" * 70)
print(f"\nMaximum voltage: V_max = {V_max:.2f} V")
print(f"Minimum voltage: V_min = {V_min:.4f} V")
print(f"Contrast: C = (V_max - V_min)/(V_max + V_min) = {contrast:.4f}")
print(f"Degree of Polarization (DOP) = {DOP:.4f}")
print(f"Polarization percentage: {DOP * 100:.2f}%")


# Define Malus's Law for amplitude measurements
def malus_law_amplitude(theta, V0, theta0):
    """
    For amplitude measurements: V = V₀ |cos(θ - θ₀)|

    Parameters:
    -----------
    theta : array-like
        Angle in radians
    V0 : float
        Maximum voltage amplitude
    theta0 : float
        Angle offset in radians

    Returns:
    --------
    V : array-like
        Voltage as function of angle
    """
    return V0 * np.abs(np.cos(theta - theta0))


# Initial guess for fitting
p0 = [V_max, 0]

# Perform weighted least-squares curve fitting
# Use absolute_sigma=True to properly weight the uncertainties
popt, pcov = curve_fit(malus_law_amplitude, angles_rad, voltage,
                       p0=p0, sigma=voltage_uncertainty, absolute_sigma=True)

# Extract fitted parameters and uncertainties
V0_fit, theta0_fit = popt
V0_err, theta0_err = np.sqrt(np.diag(pcov))

# Convert theta0 to degrees
theta0_fit_deg = np.degrees(theta0_fit)
theta0_err_deg = np.degrees(theta0_err)

print(f"\n{'=' * 70}")
print("CURVE FITTING RESULTS - Malus's Law for Amplitude")
print("=" * 70)
print(f"Fitted function: V(θ) = V₀ |cos(θ - θ₀)|")
print(f"\nFitted parameters:")
print(f"V₀ = {V0_fit:.3f} ± {V0_err:.3f} V")
print(f"θ₀ = {theta0_fit_deg:.2f} ± {theta0_err_deg:.2f}°")

# Calculate chi-squared and reduced chi-squared
residuals = voltage - malus_law_amplitude(angles_rad, *popt)
chi_squared = np.sum((residuals / voltage_uncertainty) ** 2)
dof = len(voltage) - len(popt)  # degrees of freedom
reduced_chi_squared = chi_squared / dof

# Also calculate R² for reference
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((voltage - np.mean(voltage)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"\nGoodness of fit:")
print(f"χ² = {chi_squared:.2f}")
print(f"Reduced χ² = {reduced_chi_squared:.2f}")
print(f"Degrees of freedom = {dof}")
print(f"R² = {r_squared:.4f} (for reference)")

# ============================================================================
# EXPERIMENT A: GRAPH GENERATION
# ============================================================================

# Create smooth curve for plotting
theta_fit = np.linspace(0, np.pi / 2, 1000)
voltage_fit = malus_law_amplitude(theta_fit, *popt)
theta_fit_deg = np.degrees(theta_fit)

# Create figure with main plot and residuals
fig, (ax1) = plt.subplots(1, 1, figsize=(11, 7))

# Main plot
ax1.errorbar(angles_deg, voltage, yerr=voltage_uncertainty, xerr=angle_uncertainty,
             fmt='o', markersize=6, capsize=4, capthick=1.5,
             label='Experimental data', color='navy', ecolor='darkblue', alpha=0.7)

ax1.plot(theta_fit_deg, voltage_fit, 'r-', linewidth=2,
         label=f'Fit: V(θ) = V₀|cos(θ - θ₀)|\n' +
               f'V₀ = {V0_fit:.2f} ± {V0_err:.2f} V\n' +
               f'θ₀ = {theta0_fit_deg:.1f} ± {theta0_err_deg:.1f}°\n' +
               f'χ² = {reduced_chi_squared:.2f}')

ax1.set_xlabel('Angle θ (degrees)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Voltage (V)', fontsize=12, fontweight='bold')

ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=10, loc='best', framealpha=0.9)
ax1.set_xlim(-2, 92)
ax1.set_ylim(-0.2, 9.5)

#

plt.tight_layout()
plt.savefig('experiment_A_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Figure saved: experiment_A_analysis.png")

plt.close()

# ============================================================================
# EXPERIMENT B DATA
# ============================================================================

# Configuration names
configurations = ['Without grid', 'Parallel', 'Perpendicular']

# Voltage measurements (V)
voltage_B = np.array([8.8, 8.8, 0.46])

# Voltage uncertainty (V)
uncertainty_B = np.array([0.04, 0.04, 0.04])

# ============================================================================
# EXPERIMENT B: ANALYSIS
# ============================================================================

print(f"\n{'=' * 70}")
print("EXPERIMENT B: POLARIZING GRATING ANALYSIS")
print("=" * 70)

print("\nMeasured voltages:")
for config, V, dV in zip(configurations, voltage_B, uncertainty_B):
    print(f"{config:20s}: {V:.2f} ± {dV:.2f} V")

# Calculate ratios
ratio_parallel = voltage_B[1] / voltage_B[0]
ratio_perpendicular = voltage_B[2] / voltage_B[0]

# Uncertainty propagation for ratios
ratio_parallel_err = ratio_parallel * np.sqrt((uncertainty_B[1] / voltage_B[1]) ** 2 +
                                              (uncertainty_B[0] / voltage_B[0]) ** 2)
ratio_perpendicular_err = ratio_perpendicular * np.sqrt((uncertainty_B[2] / voltage_B[2]) ** 2 +
                                                        (uncertainty_B[0] / voltage_B[0]) ** 2)

print(f"\nRatios relative to 'Without grid':")
print(f"Parallel / Without grid = {ratio_parallel:.3f} ± {ratio_parallel_err:.3f}")
print(f"Perpendicular / Without grid = {ratio_perpendicular:.3f} ± {ratio_perpendicular_err:.3f}")
print(f"Attenuation = {(1 - ratio_perpendicular) * 100:.1f}%")

# ============================================================================
# EXPERIMENT B: GRAPH GENERATION
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(configurations))
colors = ['gray', 'steelblue', 'coral']

bars = ax.bar(x_pos, voltage_B, yerr=uncertainty_B, capsize=5,
              color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
ax.set_ylabel('Voltage (V)', fontsize=12, fontweight='bold')
ax.set_title('Experiment B: Effect of Polarizing Grating on Microwave Amplitude',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(configurations, fontsize=11)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_ylim(0, 10)

# Add value labels on bars
for i, (bar, val, err) in enumerate(zip(bars, voltage_B, uncertainty_B)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + err + 0.2,
            f'{val:.2f} ± {err:.2f} V',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('experiment_B_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Figure saved: experiment_B_analysis.png")

plt.close()

# ============================================================================
# SAVE RESULTS TO CSV
# ============================================================================

results_data = {
    'Angle (deg)': angles_deg,
    'Voltage (V)': voltage,
    'Voltage Uncertainty (V)': voltage_uncertainty,
    'Fitted Voltage (V)': malus_law_amplitude(angles_rad, *popt),
    'Residual (V)': residuals
}
df = pd.DataFrame(results_data)
df.to_csv('experiment_A_results.csv', index=False)
print(f"✓ Detailed results saved: experiment_A_results.csv")

print(f"\n{'=' * 70}")
print("GRAPH GENERATION COMPLETE")
print("=" * 70)
print(f"\nFiles created:")
print(f"  • experiment_A_analysis.png")
print(f"  • experiment_B_analysis.png")
print(f"  • experiment_A_results.csv")
print(f"\nKey Results:")
print(f"  • DOP = {DOP * 100:.2f}%")
print(f"  • V₀ = {V0_fit:.2f} ± {V0_err:.2f} V")
print(f"  • θ₀ = {theta0_fit_deg:.1f} ± {theta0_err_deg:.1f}°")
print(f"  • χ² = {chi_squared:.2f} (reduced χ² = {reduced_chi_squared:.2f})")
print(f"  • Transmitter polarization: VERTICAL")
print("=" * 70)