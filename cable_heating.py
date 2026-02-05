import numpy as np
import matplotlib.pyplot as plt

# ==================== MATERIAL PROPERTIES ====================
MATERIALS = {
    'copper': {
        'resistivity': 1.68e-8,      # Ω·m at 20°C
        'temp_coeff': 0.00393,       # /°C
        'density': 8960,             # kg/m³
        'specific_heat': 385,        # J/kg·K
    },
    'aluminum': {
        'resistivity': 2.82e-8,      # Ω·m at 20°C
        'temp_coeff': 0.00403,       # /°C
        'density': 2700,             # kg/m³
        'specific_heat': 900,        # J/kg·K
    }
}

# ==================== FUNCTIONS ====================

def awg_to_diameter(awg):
    """Convert AWG to diameter in meters"""
    d_mm = 0.127 * (92 ** ((36 - awg) / 39))
    return d_mm * 1e-3  # convert to meters

def calculate_resistance(length, awg, material, temp_avg=20):
    """Calculate wire resistance at given temperature"""
    d = awg_to_diameter(awg)
    A = np.pi * d**2 / 4
    
    rho_20 = MATERIALS[material]['resistivity']
    alpha = MATERIALS[material]['temp_coeff']
    
    # Resistance at average temperature
    rho = rho_20 * (1 + alpha * (temp_avg - 20))
    R = rho * length / A
    
    return R, A, d

def calculate_mass(length, area, material):
    """Calculate wire mass"""
    density = MATERIALS[material]['density']
    volume = area * length
    mass = density * volume
    return mass

def calculate_heating_time(length, awg, material, current, 
                          T_target, T_ambient=25, 
                          include_heat_loss=False, h=15):
    """
    Calculate time to heat cable to target temperature
    
    Parameters:
    - length: cable length (m)
    - awg: wire gauge
    - material: 'copper' or 'aluminum'
    - current: current in Amperes
    - T_target: target temperature (°C)
    - T_ambient: ambient temperature (°C)
    - include_heat_loss: whether to include convection cooling
    - h: convection coefficient (W/m²·K), typical: 10-25 for still air
    """
    
    # Temperature rise
    delta_T = T_target - T_ambient
    T_avg = (T_target + T_ambient) / 2
    
    # Wire properties
    R, A, d = calculate_resistance(length, awg, material, T_avg)
    mass = calculate_mass(length, A, material)
    c = MATERIALS[material]['specific_heat']
    
    # Power dissipation
    P = current**2 * R
    
    # Heat required
    Q = mass * c * delta_T
    
    if not include_heat_loss:
        # Simple case: no heat loss
        time = Q / P
    else:
        # With heat loss - solve exponential equation
        A_surface = np.pi * d * length
        P_loss_max = h * A_surface * delta_T
        
        # Check if heating is possible
        if P <= P_loss_max:
            return np.inf  # Cannot reach target temp
        
        # Time constant
        tau = mass * c / (h * A_surface)
        
        # Steady state temperature rise
        delta_T_ss = P / (h * A_surface)
        
        if delta_T >= delta_T_ss:
            return np.inf  # Cannot reach target temp
        
        # Time to reach target
        time = -tau * np.log(1 - delta_T / delta_T_ss)
    
    return time

# ==================== USER INPUT ====================

print("=" * 60)
print("CABLE HEATING TIME CALCULATOR")
print("=" * 60)

# Get user inputs
material_input = input("\nMaterial (copper/aluminum) [default copper]: ").lower().strip()
material = material_input if material_input in MATERIALS else 'copper'

length = float(input("Cable length (m) [default 1]: ") or 1)
current = float(input("Current (A) [default 40]: ") or 40)
T_target = float(input("Target temperature (°C): "))
T_ambient = float(input("Ambient temperature (°C) [default 25]: ") or 25)

heat_loss = input("Include heat loss? (y/n) [default n]: ").lower().strip()
include_heat_loss = heat_loss == 'y'

if include_heat_loss:
    h = float(input("Convection coefficient (W/m²·K) [default 15]: ") or 15)
    print("  (Still air: ~10, Moving air: ~25-50)")
else:
    h = 15

# AWG range for plotting
awg_min = int(input("\nMinimum AWG [default 10]: ") or 10)
awg_max = int(input("Maximum AWG [default 30]: ") or 30)

# ==================== CALCULATIONS ====================

awg_values = np.arange(awg_min, awg_max + 1)
times = []

print("\nCalculating...")

for awg in awg_values:
    t = calculate_heating_time(length, awg, material, current,
                               T_target, T_ambient,
                               include_heat_loss, h)
    times.append(t)

times = np.array(times)

# ==================== DISPLAY RESULTS ====================

print("\n" + "=" * 60)
print("SAMPLE CALCULATIONS")
print("=" * 60)

# Show a few example AWG points
example_awgs = [awg_min, (awg_min + awg_max)//2, awg_max]
for awg in example_awgs:
    t = calculate_heating_time(length, awg, material, current,
                               T_target, T_ambient,
                               include_heat_loss, h)
    R, A, d = calculate_resistance(length, awg, material)
    P = current**2 * R

    print(f"\nAWG: {awg}")
    print(f"  Wire diameter: {d*1000:.3f} mm")
    print(f"  Resistance: {R:.4f} Ω")
    print(f"  Power: {P:.2f} W")
    if np.isinf(t):
        print(f"  Time: INFINITE (cannot reach target temp)")
    else:
        print(f"  Time: {t:.2f} seconds ({t/60:.2f} minutes)")

# ==================== PLOTTING ====================

plt.figure(figsize=(12, 7))

# Filter out infinite values for plotting
valid_mask = ~np.isinf(times)
awg_valid = awg_values[valid_mask]
times_valid = times[valid_mask]

plt.subplot(2, 1, 1)
plt.plot(awg_valid, times_valid, 'b-o', linewidth=2, markersize=4)
plt.xlabel('AWG', fontsize=12)
plt.ylabel('Heating Time (seconds)', fontsize=12)
plt.title(f'Heating Time vs AWG\n'
          f'{material.capitalize()}, L={length}m, I={current}A, '
          f'ΔT={T_target-T_ambient}°C', fontsize=14)
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()  # Higher AWG = thinner wire, so invert for intuition

plt.subplot(2, 1, 2)
plt.plot(awg_valid, times_valid/60, 'r-o', linewidth=2, markersize=4)
plt.xlabel('AWG', fontsize=12)
plt.ylabel('Heating Time (minutes)', fontsize=12)
plt.title('Heating Time (in minutes)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()


plt.tight_layout()
plt.savefig('cable_heating_graph.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Graph saved to cable_heating_graph.png")
print("=" * 60)