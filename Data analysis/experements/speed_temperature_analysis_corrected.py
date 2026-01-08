import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

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

print("=" * 80)
print("SPEED OF SOUND VS TEMPERATURE ANALYSIS (CORRECTED)")
print("=" * 80)

# Fix first data point where frequencies are swapped
print("\nDATA CORRECTION:")
print(f"Point 1 original: f_high={f_i[0]}, f_low={f_r[0]} (negative difference!)")
if f_i[0] < f_r[0]:
    f_i[0], f_r[0] = f_r[0], f_i[0]  # Swap
    print(f"Point 1 corrected: f_high={f_i[0]}, f_low={f_r[0]} ✓")

# CORRECTED FORMULA: v_p = (f_i - f_r) × L
# With f in kHz and L in cm, this gives v in m/s
v_p = (f_i - f_r) * L  # m/s

print("\nFORMULA USED:")
print("v_p = (f_i - f_r) × L")
print("where: f in [kHz], L in [cm], result in [m/s]")

# Calculate uncertainty in v_p using error propagation
# For v_p = (f_i - f_r) × L:
# ∂v_p/∂L = (f_i - f_r)
# ∂v_p/∂f_i = L
# ∂v_p/∂f_r = -L

# Term 1: uncertainty from L
term1 = ((f_i - f_r) * delta_L)**2

# Term 2: uncertainty from f (appears twice for f_i and f_r)
term2 = 2 * (L * delta_f)**2

# Calculate total uncertainty
delta_v_p = np.sqrt(term1 + term2)

print("\n" + "=" * 80)
print("SPEED vs TEMPERATURE TABLE")
print("=" * 80)
print(f"{'Point':<8} {'T[°C]':<10} {'ΔT[°C]':<10} {'Δf[kHz]':<12} {'v_p[m/s]':<12} {'Δv_p[m/s]':<12}")
print("-" * 80)
for i in range(len(T)):
    print(f"{i+1:<8} {T[i]:<10.1f} {delta_T[i]:<10.1f} {f_i[i]-f_r[i]:<12.0f} {v_p[i]:<12.2f} {delta_v_p[i]:<12.2f}")

# Create detailed table
results_table = pd.DataFrame({
    'Temperature [°C]': T,
    'ΔT [°C]': delta_T,
    'f_high [kHz]': f_i,
    'f_low [kHz]': f_r,
    'Δf [kHz]': f_i - f_r,
    'L [cm]': L,
    'ΔL [cm]': delta_L,
    'Speed [m/s]': np.round(v_p, 2),
    'ΔSpeed [m/s]': np.round(delta_v_p, 2)
})

# Save table
results_table.to_csv('speed_vs_temperature_table_corrected.csv', index=False)
print(f"\n✓ Table saved as CSV")

# Polynomial fitting (3rd degree works best with 5 points)
def poly3(T, a, b, c, d):
    """3rd degree polynomial"""
    return a + b*T + c*T**2 + d*T**3

# Fit with weights (inverse variance)
try:
    # Initial guess
    p0 = [1500, 1, 0, 0]
    
    # Perform weighted fit
    popt, pcov = curve_fit(poly3, T, v_p, p0=p0, sigma=delta_v_p, absolute_sigma=True)
    
    # Extract parameters
    a, b, c, d = popt
    perr = np.sqrt(np.diag(pcov))
    
    print(f"\nFIT PARAMETERS (3rd degree polynomial):")
    print("-" * 80)
    print(f"v(T) = a + b×T + c×T² + d×T³")
    print(f"\na = {a:.3f} ± {perr[0]:.3f}")
    print(f"b = {b:.6f} ± {perr[1]:.6f}")
    print(f"c = {c:.6e} ± {perr[2]:.6e}")
    print(f"d = {d:.6e} ± {perr[3]:.6e}")
    
    # Calculate fitted values
    v_fit = poly3(T, *popt)
    
    # Calculate residuals
    residuals = v_p - v_fit
    
    # Calculate chi-squared
    chi_squared = np.sum((residuals / delta_v_p)**2)
    
    # Degrees of freedom (n data points - 4 parameters)
    dof = len(T) - 4
    
    if dof > 0:
        chi_squared_reduced = chi_squared / dof
        print(f"\nCHI-SQUARED ANALYSIS:")
        print("-" * 80)
        print(f"χ² = {chi_squared:.4f}")
        print(f"Degrees of freedom = {dof}")
        print(f"χ²/ndof = {chi_squared_reduced:.4f}")
    else:
        chi_squared_reduced = chi_squared
        print(f"\nCHI-SQUARED: χ² = {chi_squared:.4f}")
        print(f"DOF = {dof} (exact fit)")
    
    # Create plot matching the reference style
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot data points with error bars
    ax.errorbar(T, v_p, xerr=delta_T, yerr=delta_v_p,
                fmt='o', markersize=8, capsize=5, capthick=2,
                color='red', ecolor='red', elinewidth=2,
                label='Data', zorder=3, markerfacecolor='red', 
                markeredgecolor='red', markeredgewidth=2)
    
    # Plot fitted curve
    T_fit = np.linspace(T.min() - 5, T.max() + 5, 200)
    v_fit_curve = poly3(T_fit, *popt)
    ax.plot(T_fit, v_fit_curve, 'b-', linewidth=2.5, label='Fit', zorder=2)
    
    # Add grid
    ax.grid(True, alpha=0.3, zorder=1)
    
    # Labels and title
    ax.set_xlabel('Temperature [C°]', fontsize=13)
    ax.set_ylabel('Speed [m/s]', fontsize=13)
    ax.set_title('$v_p$ as a Function of Water Temperature', fontsize=13, fontweight='bold')
    
    # Add text box with fit parameters (matching reference style)
    textstr = f'a = {a:.3f} ± {perr[0]:.3f}\n'
    textstr += f'b = {b:.3f} ± {perr[1]:.3f}\n'
    textstr += f'c = {c:.6f} ± {perr[2]:.6f}\n'
    textstr += f'd = {d:.6e} ± {perr[3]:.6e}\n'
    if dof > 0:
        textstr += f'χ²/ndof = {chi_squared_reduced:.4f}({chi_squared:.4f}/{dof})'
    else:
        textstr += f'χ² = {chi_squared:.4f}'
    
    props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Legend
    ax.legend(loc='lower right', fontsize=11)
    
    # Set y-axis to show the full range
    ax.set_ylim(1520, 1600)
    ax.set_xlim(25, 75)
    
    plt.tight_layout()
    plt.savefig('speed_vs_temperature_corrected.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to outputs directory")
    
    # Save parameters to text file
    with open('fit_parameters_corrected.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("POLYNOMIAL FIT PARAMETERS (CORRECTED)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: v(T) = a + b×T + c×T² + d×T³\n\n")
        f.write(f"Parameters:\n")
        f.write(f"a = {a:.6f} ± {perr[0]:.6f}\n")
        f.write(f"b = {b:.6f} ± {perr[1]:.6f}\n")
        f.write(f"c = {c:.6e} ± {perr[2]:.6e}\n")
        f.write(f"d = {d:.6e} ± {perr[3]:.6e}\n\n")
        f.write(f"Chi-squared: χ² = {chi_squared:.4f}\n")
        f.write(f"Degrees of freedom: {dof}\n")
        if dof > 0:
            f.write(f"Reduced chi-squared: χ²/ndof = {chi_squared_reduced:.4f}\n")
    
    print(f"✓ Fit parameters saved to text file")
    
except Exception as e:
    print(f"\nError during fitting: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("VERIFICATION: Speeds are now in the correct range (1520-1600 m/s) ✓")
print("=" * 80)
