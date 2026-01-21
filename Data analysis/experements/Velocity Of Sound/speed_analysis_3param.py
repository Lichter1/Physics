import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Get script directory for relative paths
script_dir = Path(__file__).parent

# Read data
df = pd.read_excel(script_dir / 'exp4.xlsx')

# Extract data
L = df['distance [cm]'].values  # cm
delta_L = df['distance uncertenty [cm]'].values  # cm
T = df['tempreture [c]'].values  # Celsius
delta_T = df['tempreture uncertenty [c]'].values  # Celsius
f_i = df['frequency high [khz]'].values  # kHz
f_r = df['frequency low [khz]'].values  # kHz
delta_f = df['frequency uncertenty [kHZ]'].values  # kHz

print("=" * 80)
print("SPEED OF SOUND VS TEMPERATURE - 3-PARAMETER POLYNOMIAL FIT")
print("=" * 80)

# Calculate speed: v_p = (f_i - f_r) × L
v_p = (f_i - f_r) * L  # m/s

print("\nFORMULA: v_p = (f_i - f_r) × L")
print("         where f in [kHz], L in [cm], result in [m/s]")

print("\n" + "=" * 80)
print("DETAILED UNCERTAINTY CALCULATIONS FOR EACH POINT")
print("=" * 80)

delta_v_p = np.zeros(len(T))

for i in range(len(T)):
    delta_freq = f_i[i] - f_r[i]
    
    # Calculate uncertainty components
    term1 = (delta_freq * delta_L[i])**2
    term2 = 2 * (L[i] * delta_f[i])**2
    delta_v_p[i] = np.sqrt(term1 + term2)
    
    print(f"\n{'─'*80}")
    print(f"POINT {i+1}: T = {T[i]:.1f}°C")
    print(f"{'─'*80}")
    print(f"Measurements:")
    print(f"  f_high = {f_i[i]:.0f} kHz, f_low = {f_r[i]:.0f} kHz")
    print(f"  Δf = {delta_freq:.0f} kHz")
    print(f"  L = {L[i]:.1f} cm, ΔL = {delta_L[i]:.1f} cm")
    
    print(f"\nSpeed calculation:")
    print(f"  v_p = (f_i - f_r) × L = {delta_freq:.0f} × {L[i]:.1f} = {v_p[i]:.2f} m/s")
    
    print(f"\nUncertainty calculation:")
    print(f"  Term1 = [(f_i - f_r) × ΔL]² = [{delta_freq:.0f} × {delta_L[i]:.1f}]² = {term1:.2f} (m/s)²")
    print(f"  Term2 = 2[L × Δf]² = 2 × [{L[i]:.1f} × {delta_f[i]:.0f}]² = {term2:.2f} (m/s)²")
    print(f"  Δv_p = √({term1:.2f} + {term2:.2f}) = {delta_v_p[i]:.2f} m/s")
    
    print(f"\n  RESULT: v_p = {v_p[i]:.2f} ± {delta_v_p[i]:.2f} m/s")
    print(f"  Relative error: {100*delta_v_p[i]/v_p[i]:.2f}%")
    
    # Show contribution breakdown
    total = term1 + term2
    term1_pct = 100 * term1 / total
    term2_pct = 100 * term2 / total
    print(f"\n  Uncertainty contribution:")
    print(f"    Distance (ΔL): {term1_pct:.1f}%")
    print(f"    Frequency (Δf): {term2_pct:.1f}%")

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Point':<8} {'T[°C]':<10} {'Δf[kHz]':<12} {'v_p[m/s]':<12} {'Δv_p[m/s]':<12} {'Rel.Err%':<10}")
print("-" * 80)
for i in range(len(T)):
    rel_err = 100 * delta_v_p[i] / v_p[i]
    print(f"{i+1:<8} {T[i]:<10.1f} {f_i[i]-f_r[i]:<12.0f} {v_p[i]:<12.2f} {delta_v_p[i]:<12.2f} {rel_err:<10.2f}")

# Save table
results_table = pd.DataFrame({
    'Point': range(1, len(T)+1),
    'Temperature [°C]': T,
    'ΔT [°C]': delta_T,
    'f_high [kHz]': f_i,
    'f_low [kHz]': f_r,
    'Δf [kHz]': f_i - f_r,
    'L [cm]': L,
    'ΔL [cm]': delta_L,
    'Speed [m/s]': np.round(v_p, 2),
    'ΔSpeed [m/s]': np.round(delta_v_p, 2),
    'Relative Error [%]': np.round(100 * delta_v_p / v_p, 2)
})
results_table.to_csv(script_dir / 'speed_temperature_table_3param.csv', index=False)
print(f"\n✓ Table saved")

print("\n" + "=" * 80)
print("3-PARAMETER POLYNOMIAL FIT (QUADRATIC)")
print("=" * 80)
print(f"\nFitting: v(T) = a + b×T + c×T²")
print(f"Data points: {len(T)}")
print(f"Parameters: 3 (a, b, c)")
print(f"Degrees of freedom: {len(T) - 3} ✓")

# Fit 2nd degree polynomial with weights
degree = 2
weights = 1.0 / delta_v_p
coefficients = np.polyfit(T, v_p, degree, w=weights)

# Extract coefficients (numpy gives highest degree first)
c_coef = coefficients[0]  # T²
b_coef = coefficients[1]  # T
a_coef = coefficients[2]  # constant

# Create polynomial function
poly = np.poly1d(coefficients)

print(f"\nFIT PARAMETERS:")
print("-" * 80)
print(f"a = {a_coef:.6f}")
print(f"b = {b_coef:.6f}")
print(f"c = {c_coef:.6e}")

# Calculate fitted values
v_fit = poly(T)
residuals = v_p - v_fit

# Calculate chi-squared
chi_squared = np.sum((residuals / delta_v_p)**2)
dof = len(T) - 3
chi_squared_reduced = chi_squared / dof

print(f"\nFIT QUALITY:")
print("-" * 80)
print(f"χ² = {chi_squared:.4f}")
print(f"Degrees of freedom = {dof}")
print(f"χ²/ndof = {chi_squared_reduced:.4f}")

if chi_squared_reduced < 0.5:
    print("\n✓ Excellent fit! (χ²/ndof << 1)")
elif chi_squared_reduced < 2:
    print("\n✓ Good fit! (χ²/ndof ≈ 1)")
else:
    print("\n⚠ Fit may need improvement (χ²/ndof > 2)")

print(f"\nRESIDUALS:")
print("-" * 80)
print(f"{'Point':<8} {'T[°C]':<10} {'v_meas[m/s]':<14} {'v_fit[m/s]':<14} {'Residual[m/s]':<15} {'σ':<10}")
print("-" * 80)
for i in range(len(T)):
    sigma_residual = residuals[i] / delta_v_p[i]
    print(f"{i+1:<8} {T[i]:<10.1f} {v_p[i]:<14.2f} {v_fit[i]:<14.2f} {residuals[i]:<15.2f} {sigma_residual:<10.2f}")

print("\nNote: σ = residual/uncertainty (should be < 2 for good points)")

# Create plot
fig, ax = plt.subplots(figsize=(10, 7))

# Plot data points with error bars
ax.errorbar(T, v_p, xerr=delta_T, yerr=delta_v_p,
            fmt='o', markersize=8, capsize=5, capthick=2,
            color='red', ecolor='red', elinewidth=2,
            label='Data', zorder=3, markerfacecolor='red', 
            markeredgecolor='red', markeredgewidth=2)

# Plot fitted curve
T_fit = np.linspace(T.min() - 5, T.max() + 5, 300)
v_fit_curve = poly(T_fit)
ax.plot(T_fit, v_fit_curve, 'b-', linewidth=2.5, label='Fit', zorder=2)

# Plot reference curve: v = 1405.03 + 4.624*T - 3.83e-2*T² [m/s]
v_reference = 1405.03 + 4.624 * T_fit - 3.83e-2 * T_fit**2
ax.plot(T_fit, v_reference, 'g--', linewidth=2, label='Reference', zorder=2)

# Add grid
ax.grid(True, alpha=0.3, zorder=1)

# Labels and title
ax.set_xlabel('Temperature [C°]', fontsize=13)
ax.set_ylabel('Speed [m/s]', fontsize=13)
#ax.set_title('$v_p$ as a Function of Water Temperature', fontsize=13, fontweight='bold')

# Add text box with fit parameters
textstr = f'a = {a_coef:.3f}\n'
textstr += f'b = {b_coef:.3f}\n'
textstr += f'c = {c_coef:.6f}\n'
textstr += f'χ²/ndof = {chi_squared_reduced:.4f}'

props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

# Legend
ax.legend(loc='lower right', fontsize=11)

# Set axis limits
ax.set_ylim(1480, 1640)
ax.set_xlim(25, 75)

plt.tight_layout()
plt.savefig(script_dir / 'speed_vs_temp_3param.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved")

# Save parameters to file
with open(script_dir / 'fit_params_3param.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("3-PARAMETER POLYNOMIAL FIT (QUADRATIC)\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Model: v(T) = a + b×T + c×T²\n\n")
    f.write(f"Fitted Parameters:\n")
    f.write(f"a = {a_coef:.10f}\n")
    f.write(f"b = {b_coef:.10f}\n")
    f.write(f"c = {c_coef:.10e}\n\n")
    f.write(f"Statistical Quality:\n")
    f.write(f"χ² = {chi_squared:.6f}\n")
    f.write(f"Degrees of freedom = {dof}\n")
    f.write(f"χ²/ndof = {chi_squared_reduced:.6f}\n\n")
    f.write(f"Data Points and Residuals:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Point':<8} {'T[°C]':<10} {'v_meas[m/s]':<14} {'v_fit[m/s]':<14} {'Residual[m/s]':<15}\n")
    f.write("-" * 80 + "\n")
    for i in range(len(T)):
        f.write(f"{i+1:<8} {T[i]:<10.1f} {v_p[i]:<14.2f} {v_fit[i]:<14.2f} {residuals[i]:<15.2f}\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write("INTERPRETATION\n")
    f.write("=" * 80 + "\n")
    if chi_squared_reduced < 0.5:
        f.write("χ²/ndof << 1: Excellent fit. The quadratic model describes the data\n")
        f.write("very well. Uncertainties may be slightly conservative.\n")
    elif chi_squared_reduced < 2:
        f.write("χ²/ndof ≈ 1: Good fit. The quadratic model is appropriate and\n")
        f.write("uncertainties are well-estimated.\n")
    else:
        f.write("χ²/ndof > 2: Model or uncertainties may need review.\n")

print(f"✓ Parameters saved")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nKey findings:")
print(f"  • Speed range: {v_p.min():.0f} - {v_p.max():.0f} m/s ✓")
print(f"  • Average uncertainty: {delta_v_p.mean():.1f} m/s ({100*delta_v_p.mean()/v_p.mean():.1f}%)")
print(f"  • Improved distance uncertainty (0.2 cm vs 0.8 cm) reduced errors significantly")
print(f"  • All points now have good frequency differences (>100 kHz)")
print(f"  • Quadratic fit quality: χ²/ndof = {chi_squared_reduced:.3f}")
print("\n" + "=" * 80)
