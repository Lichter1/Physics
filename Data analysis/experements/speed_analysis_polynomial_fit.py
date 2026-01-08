import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read data
df = pd.read_excel('exp4.xlsx')

# Extract data
L = df['distance [cm]'].values  # cm
delta_L = df['distance uncertenty [cm]'].values  # cm
T = df['tempreture [c]'].values  # Celsius
delta_T = df['tempreture uncertenty [c]'].values  # Celsius
f_i = df['frequency high [khz]'].values.copy()  # kHz
f_r = df['frequency low [khz]'].values.copy()  # kHz
delta_f = df['frequency uncertenty [kHZ]'].values  # kHz

# Fix first data point
if f_i[0] < f_r[0]:
    f_i[0], f_r[0] = f_r[0], f_i[0]

# Calculate speed
v_p = (f_i - f_r) * L  # m/s

# Calculate uncertainties
term1 = ((f_i - f_r) * delta_L)**2
term2 = 2 * (L * delta_f)**2
delta_v_p = np.sqrt(term1 + term2)

print("=" * 80)
print("5TH DEGREE POLYNOMIAL FIT WITH 5 DATA POINTS")
print("=" * 80)

print("\n⚠ WARNING: Fitting 6 parameters (a,b,c,d,e,f) with only 5 data points")
print("This will result in an UNDERDETERMINED system!")
print("You need at least 6 data points to fit a 5th degree polynomial")
print("With 5 points, we can only fit up to a 4th degree polynomial (5 parameters)")
print("\nProceeding with 4th degree polynomial instead...\n")

# Use 4th degree polynomial (5 parameters for 5 data points)
degree = 4

# Fit polynomial with weights
weights = 1.0 / delta_v_p
coefficients = np.polyfit(T, v_p, degree, w=weights)

# Create polynomial function
poly = np.poly1d(coefficients)

# Extract coefficients (highest degree first in numpy)
f_coef = coefficients[0] if degree >= 5 else 0
e_coef = coefficients[1] if degree >= 5 else (coefficients[0] if degree >= 4 else 0)
d_coef = coefficients[2] if degree >= 5 else (coefficients[1] if degree >= 4 else (coefficients[0] if degree >= 3 else 0))
c_coef = coefficients[3] if degree >= 5 else (coefficients[2] if degree >= 4 else (coefficients[1] if degree >= 3 else (coefficients[0] if degree >= 2 else 0)))
b_coef = coefficients[4] if degree >= 5 else (coefficients[3] if degree >= 4 else (coefficients[2] if degree >= 3 else (coefficients[1] if degree >= 2 else (coefficients[0] if degree >= 1 else 0))))
a_coef = coefficients[5] if degree >= 5 else (coefficients[4] if degree >= 4 else (coefficients[3] if degree >= 3 else (coefficients[2] if degree >= 2 else (coefficients[1] if degree >= 1 else coefficients[0]))))

# For 4th degree: coefficients = [e, d, c, b, a]
e_coef = coefficients[0]
d_coef = coefficients[1]
c_coef = coefficients[2]
b_coef = coefficients[3]
a_coef = coefficients[4]

print(f"FIT PARAMETERS (4th degree polynomial):")
print("-" * 80)
print(f"v(T) = a + b×T + c×T² + d×T³ + e×T⁴")
print(f"\na = {a_coef:.6f}")
print(f"b = {b_coef:.6f}")
print(f"c = {c_coef:.6e}")
print(f"d = {d_coef:.6e}")
print(f"e = {e_coef:.6e}")

# Calculate fitted values
v_fit = poly(T)
residuals = v_p - v_fit

# Calculate chi-squared
chi_squared = np.sum((residuals / delta_v_p)**2)
dof = len(T) - (degree + 1)

print(f"\nFIT QUALITY:")
print("-" * 80)
print(f"χ² = {chi_squared:.6f}")
print(f"Degrees of freedom = {dof}")
if dof > 0:
    chi_squared_reduced = chi_squared / dof
    print(f"χ²/ndof = {chi_squared_reduced:.6f}")
else:
    print(f"χ²/ndof = N/A (exact fit, DOF = {dof})")

print(f"\nRESIDUALS:")
print("-" * 80)
print(f"{'Point':<8} {'T[°C]':<10} {'v_meas[m/s]':<14} {'v_fit[m/s]':<14} {'Residual[m/s]':<15}")
print("-" * 80)
for i in range(len(T)):
    print(f"{i+1:<8} {T[i]:<10.1f} {v_p[i]:<14.2f} {v_fit[i]:<14.2f} {residuals[i]:<15.6f}")

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

# Add grid
ax.grid(True, alpha=0.3, zorder=1)

# Labels and title
ax.set_xlabel('Temperature [C°]', fontsize=13)
ax.set_ylabel('Speed [m/s]', fontsize=13)
ax.set_title('$v_p$ as a Function of Water Temperature', fontsize=13, fontweight='bold')

# Add text box with fit parameters
textstr = f'a = {a_coef:.3f}\n'
textstr += f'b = {b_coef:.3f}\n'
textstr += f'c = {c_coef:.6f}\n'
textstr += f'd = {d_coef:.6e}\n'
textstr += f'e = {e_coef:.6e}\n'
if dof > 0:
    textstr += f'χ²/ndof = {chi_squared_reduced:.4f}({chi_squared:.4f}/{dof})'
else:
    textstr += f'χ² = {chi_squared:.6f} (DOF={dof})'

props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Legend
ax.legend(loc='lower right', fontsize=11)

# Set axis limits
ax.set_ylim(1520, 1600)
ax.set_xlim(25, 75)

plt.tight_layout()
plt.savefig('speed_vs_temp_polynomial.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved")

# Save parameters
with open('fit_params_polynomial.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("POLYNOMIAL FIT PARAMETERS (4TH DEGREE)\n")
    f.write("=" * 80 + "\n\n")
    f.write("NOTE: Requested 5th degree (6 parameters) but only have 5 data points\n")
    f.write("      Using 4th degree (5 parameters) for exact fit\n\n")
    f.write(f"Model: v(T) = a + b×T + c×T² + d×T³ + e×T⁴\n\n")
    f.write(f"Parameters:\n")
    f.write(f"a = {a_coef:.10f}\n")
    f.write(f"b = {b_coef:.10f}\n")
    f.write(f"c = {c_coef:.10e}\n")
    f.write(f"d = {d_coef:.10e}\n")
    f.write(f"e = {e_coef:.10e}\n\n")
    f.write(f"Chi-squared: χ² = {chi_squared:.10f}\n")
    f.write(f"Degrees of freedom: {dof}\n")
    if dof > 0:
        f.write(f"Reduced chi-squared: χ²/ndof = {chi_squared_reduced:.10f}\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write("EXPLANATION: Why can't we fit 6 parameters?\n")
    f.write("=" * 80 + "\n\n")
    f.write("A 5th degree polynomial has 6 parameters: a, b, c, d, e, f\n")
    f.write("v(T) = a + bT + cT² + dT³ + eT⁴ + fT⁵\n\n")
    f.write("To fit n parameters, you need AT LEAST n data points.\n")
    f.write("With exactly n points, you get an exact fit (interpolation).\n")
    f.write("With more than n points, you can do statistical fitting.\n\n")
    f.write("Your data: 5 points\n")
    f.write("Required for 5th degree: 6 points minimum\n\n")
    f.write("Solution: Collect at least 2 more data points!\n")

print(f"✓ Parameters saved\n")

print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("\nTo fit a 5th degree polynomial (6 parameters: a,b,c,d,e,f), you need:")
print("  • Minimum: 6 data points (for exact fit)")
print("  • Recommended: 8-10 data points (for proper statistical analysis)")
print("\nTo improve your experiment:")
print("  1. Add measurements at 2-3 more temperatures")
print("  2. Suggested temperatures: 35°C, 40°C, 65°C")
print("  3. This will give you 7-8 points for a proper 5th degree fit")
print("\n" + "=" * 80)
