import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Read data
df = pd.read_excel('exp3.xlsx')
time = df['Time [µs]'].values
distance = df['Distance [cm]'].values

# Uncertainties
sigma_time = 0.4  # µs
sigma_distance = 0.8  # cm

print("=" * 70)
print("LINEAR FIT THROUGH ORIGIN WITH UNCERTAINTY ANALYSIS")
print("=" * 70)

# For linear fit through origin: y = a*x
# Using weighted least squares with uncertainties

# Weights (inverse variance)
weights = 1 / sigma_distance**2

# Calculate slope (a) for fit through origin
# a = Σ(w_i * x_i * y_i) / Σ(w_i * x_i²)
numerator = np.sum(weights * time * distance)
denominator = np.sum(weights * time**2)
a = numerator / denominator

# Uncertainty in slope
# σ_a² = 1 / Σ(w_i * x_i²) = 1 / Σ(x_i² / σ_y²)
sigma_a = np.sqrt(1 / denominator)

print(f"\nFIT PARAMETERS:")
print(f"Slope (a) = {a:.6f} ± {sigma_a:.6f} cm/µs")
print(f"Intercept (b) = 0 (forced)")

# Calculate fitted values
y_fit = a * time

# Calculate residuals
residuals = distance - y_fit

# Calculate chi-squared
chi_squared = np.sum((residuals / sigma_distance)**2)

# Degrees of freedom (n data points - 1 parameter)
dof = len(time) - 1
chi_squared_reduced = chi_squared / dof

print(f"\nCHI-SQUARED ANALYSIS:")
print(f"χ² = {chi_squared:.3f}")
print(f"Degrees of freedom = {dof}")
print(f"χ²/ndof = {chi_squared_reduced:.3f}")

print(f"\nDATA POINTS AND RESIDUALS:")
print(f"{'i':<5} {'Time (µs)':<12} {'Distance (cm)':<15} {'Fitted (cm)':<15} {'Residual (cm)':<15}")
print("-" * 70)
for i in range(len(time)):
    print(f"{i+1:<5} {time[i]:<12.1f} {distance[i]:<15.1f} {y_fit[i]:<15.2f} {residuals[i]:<15.2f}")

# Create the plot
fig, ax = plt.subplots(figsize=(10, 7))

# Plot data points with error bars
ax.errorbar(time, distance, xerr=sigma_time, yerr=sigma_distance, 
            fmt='o', markersize=8, capsize=5, capthick=2,
            color='#2E4057', ecolor='#2E4057', elinewidth=2,
            label='Data points', zorder=3)

# Plot fitted line
x_fit = np.linspace(0, max(time) * 1.05, 100)
y_fit_line = a * x_fit
ax.plot(x_fit, y_fit_line, 'r-', linewidth=2.5, label='Linear fit', zorder=2)

# Add grid
ax.grid(True, alpha=0.3, zorder=1)

# Labels and title
ax.set_xlabel('time [µs]', fontsize=14, fontweight='bold')
ax.set_ylabel('Distance [cm]', fontsize=14, fontweight='bold')
#ax.set_title('Distance between transmitter and detector as a function of reception time',
        #     fontsize=13, fontweight='bold', pad=15)

# Add text box with fit parameters
textstr = f'a = {a:.4f}±{sigma_a:.4f}\n'
#textstr += f'b = 0 (forced)\n'
textstr += f'χ²/ndof = {chi_squared_reduced:.3f}({chi_squared:.3f}/{dof})'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

# Set axis limits to start from origin
ax.set_xlim(0, max(time) * 1.05)
ax.set_ylim(0, max(distance) * 1.05)

# Legend
ax.legend(loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig('distance_vs_time_fit.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Plot saved to outputs directory")

# Also create a summary text file
with open('fit_analysis_summary.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("LINEAR FIT THROUGH ORIGIN - UNCERTAINTY ANALYSIS\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("EXPERIMENTAL DATA:\n")
    f.write(f"Uncertainty in time: ±{sigma_time} µs\n")
    f.write(f"Uncertainty in distance: ±{sigma_distance} cm\n\n")
    
    f.write("FIT MODEL: Distance = a × time (forced through origin)\n\n")
    
    f.write("FIT PARAMETERS:\n")
    f.write(f"Slope (a) = {a:.6f} ± {sigma_a:.6f} cm/µs\n")
    f.write(f"Intercept (b) = 0 (forced)\n\n")
    
    f.write("CHI-SQUARED ANALYSIS:\n")
    f.write(f"χ² = {chi_squared:.3f}\n")
    f.write(f"Degrees of freedom = {dof}\n")
    f.write(f"χ²/ndof = {chi_squared_reduced:.3f}\n\n")
    
    f.write("INTERPRETATION:\n")
    if chi_squared_reduced < 2:
        f.write("The reduced chi-squared is close to 1, indicating a good fit.\n")
    elif chi_squared_reduced < 5:
        f.write("The reduced chi-squared suggests a reasonable fit.\n")
    else:
        f.write("The reduced chi-squared is high, suggesting either underestimated\n")
        f.write("uncertainties or systematic effects not captured by the linear model.\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("DATA TABLE\n")
    f.write("=" * 70 + "\n")
    f.write(f"{'Point':<8} {'Time (µs)':<12} {'Distance (cm)':<15} {'Fitted (cm)':<15} {'Residual (cm)':<15}\n")
    f.write("-" * 70 + "\n")
    for i in range(len(time)):
        f.write(f"{i+1:<8} {time[i]:<12.1f} {distance[i]:<15.1f} {y_fit[i]:<15.2f} {residuals[i]:<15.2f}\n")

print(f"✓ Summary saved to outputs directory")
print("\n" + "=" * 70)
