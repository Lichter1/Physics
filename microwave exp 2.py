"""
Improved Standing Wave Analysis Script
Handles noisy data with smoothing and better peak detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STANDING WAVE MODELS
# ============================================================================

def standing_wave_simple(x, A, wavelength, phase, offset):
    """
    Simple standing wave: A * sin²(2π/λ * x + φ) + offset
    This is the intensity pattern of a standing wave
    """
    return A * np.sin(2 * np.pi / wavelength * x + phase)**2 + offset

def standing_wave_damped(x, A, wavelength, phase, offset, decay):
    """
    Damped standing wave: A * exp(-decay*x) * sin²(2π/λ * x + φ) + offset
    """
    return A * np.exp(-decay * x) * np.sin(2 * np.pi / wavelength * x + phase)**2 + offset

def standing_wave_envelope(x, A, wavelength, phase, offset, A_env, lambda_env, phase_env):
    """
    Standing wave with envelope modulation
    A * [1 + A_env*sin(2π/λ_env * x + φ_env)] * sin²(2π/λ * x + φ) + offset
    """
    envelope = 1 + A_env * np.sin(2 * np.pi / lambda_env * x + phase_env)
    return A * envelope * np.sin(2 * np.pi / wavelength * x + phase)**2 + offset

# ============================================================================
# DATA PROCESSING
# ============================================================================

def smooth_data(distance, amplitude, window=5):
    """
    Smooth data using moving average
    """
    # Use uniform filter for smoothing
    amplitude_smooth = uniform_filter1d(amplitude, size=window)
    return amplitude_smooth

def find_extrema_improved(distance, amplitude, num_expected=6, smooth_window=11):
    """
    Find maxima and minima with smoothing for noisy data

    Parameters:
    -----------
    distance : array
        Distance data
    amplitude : array
        Amplitude data
    num_expected : int
        Expected number of peaks/troughs
    smooth_window : int
        Window size for smoothing
    """
    # Smooth the data first
    amplitude_smooth = smooth_data(distance, amplitude, window=smooth_window)

    # Calculate dynamic prominence threshold
    amp_range = np.max(amplitude_smooth) - np.min(amplitude_smooth)
    prominence = 0.15 * amp_range  # 15% of range

    # Minimum distance between peaks (estimate from total range)
    min_distance = len(distance) // (num_expected * 2)

    # Find maxima on smoothed data
    max_indices, max_props = find_peaks(amplitude_smooth,
                                        prominence=prominence,
                                        distance=min_distance)

    # Find minima on smoothed data
    min_indices, min_props = find_peaks(-amplitude_smooth,
                                        prominence=prominence,
                                        distance=min_distance)

    # If we found too many or too few, adjust prominence
    if len(max_indices) < num_expected - 2 or len(max_indices) > num_expected + 4:
        # Try with different prominence
        prominence = 0.25 * amp_range
        max_indices, _ = find_peaks(amplitude_smooth,
                                   prominence=prominence,
                                   distance=min_distance)
        min_indices, _ = find_peaks(-amplitude_smooth,
                                   prominence=prominence,
                                   distance=min_distance)

    return max_indices, min_indices, amplitude_smooth

def calculate_position_uncertainty(distance, amplitude, extremum_idx, window=7):
    """
    Calculate uncertainty in extremum position
    """
    start = max(0, extremum_idx - window)
    end = min(len(distance), extremum_idx + window + 1)

    x_local = distance[start:end]
    y_local = amplitude[start:end]

    if len(x_local) < 3:
        return np.mean(np.diff(distance))

    try:
        # Fit parabola
        x_center = distance[extremum_idx]
        x_rel = x_local - x_center
        coeffs = np.polyfit(x_rel, y_local, 2)
        a = coeffs[0]

        if a == 0:
            return np.mean(np.diff(distance))

        # FWHM-based uncertainty
        uncertainty = 2.355 / np.sqrt(np.abs(a))
        dx = np.mean(np.diff(distance))
        uncertainty = np.clip(uncertainty, dx, 10 * dx)

        return uncertainty
    except:
        return np.mean(np.diff(distance))

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_standing_wave_improved(sheet_name='standing wave'):
    """
    Analyze the standing wave data with improved noise handling
    """
    # Load data
    excel_file = pd.ExcelFile('OmriTheKing2.xlsx')
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    distance_col = df.columns[0]
    amplitude_col = df.columns[1]

    distance = df[distance_col].values
    amplitude = df[amplitude_col].values

    # Remove zeros
    valid_mask = (amplitude > 0) & np.isfinite(amplitude) & np.isfinite(distance)
    distance = distance[valid_mask]
    amplitude = amplitude[valid_mask]

    print(f"{'='*70}")
    print(f"Standing Wave Analysis: {sheet_name}")
    print(f"{'='*70}")
    print(f"Data points: {len(distance)}")
    print(f"Distance range: {distance.min():.2f} to {distance.max():.2f} mm")
    print(f"Amplitude range: {amplitude.min():.0f} to {amplitude.max():.0f} mV")

    # Find extrema with smoothing
    max_indices, min_indices, amplitude_smooth = find_extrema_improved(
        distance, amplitude, num_expected=6, smooth_window=15
    )

    print(f"\nFound {len(max_indices)} maxima and {len(min_indices)} minima")

    # Calculate maxima positions and uncertainties
    print(f"\n{'='*70}")
    print("Maxima Positions and Uncertainties:")
    print(f"{'='*70}")
    print(f"{'Index':<8} {'Position [mm]':<18} {'Amplitude [mV]':<18} {'Uncertainty [mm]'}")
    print(f"{'-'*70}")

    max_positions = []
    max_uncertainties = []

    for i, idx in enumerate(max_indices):
        pos = distance[idx]
        amp = amplitude[idx]
        unc = calculate_position_uncertainty(distance, amplitude, idx)
        max_positions.append(pos)
        max_uncertainties.append(unc)
        print(f"{i+1:<8} {pos:<18.4f} {amp:<18.2f} {unc:.4f}")

    # Print minima positions
    print(f"\n{'='*70}")
    print("Minima Positions and Uncertainties:")
    print(f"{'='*70}")
    print(f"{'Index':<8} {'Position [mm]':<18} {'Amplitude [mV]':<18} {'Uncertainty [mm]'}")
    print(f"{'-'*70}")

    min_positions = []
    min_uncertainties = []

    for i, idx in enumerate(min_indices):
        pos = distance[idx]
        amp = amplitude[idx]
        unc = calculate_position_uncertainty(distance, amplitude, idx)
        min_positions.append(pos)
        min_uncertainties.append(unc)
        print(f"{i+1:<8} {pos:<18.4f} {amp:<18.2f} {unc:.4f}")

    # Calculate wavelength from minima
    if len(min_positions) >= 2:
        print(f"\n{'='*70}")
        print("Wavelength from Minima Spacing:")
        print(f"{'='*70}")
        spacings = np.diff(min_positions)
        for i, spacing in enumerate(spacings):
            print(f"Minima {i+1} to {i+2}: Δx = {spacing:.4f} mm")

        avg_spacing = np.mean(spacings)
        std_spacing = np.std(spacings, ddof=1) if len(spacings) > 1 else 0

        print(f"\nAverage spacing: {avg_spacing:.4f} ± {std_spacing:.4f} mm")
        print(f"Note: For standing waves, λ = Δx (distance between consecutive minima)")
        print(f"Wavelength (λ): {avg_spacing:.4f} ± {std_spacing:.4f} mm")

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    # Plot raw data
    ax.plot(distance, amplitude, 'b.', markersize=3, alpha=0.3, label='Raw Data')

    # Plot smoothed data
    ax.plot(distance, amplitude_smooth, 'b-', linewidth=2, alpha=0.8, label='Smoothed Data')

    # Mark maxima with error bars
    if len(max_indices) > 0:
        ax.errorbar(distance[max_indices], amplitude[max_indices],
                   xerr=max_uncertainties,
                   fmt='g^', markersize=12, capsize=5, capthick=2,
                   label=f'Maxima (n={len(max_indices)})',
                   markeredgecolor='black', markeredgewidth=1.5, zorder=5)

    # Mark minima with error bars
    if len(min_indices) > 0:
        ax.errorbar(distance[min_indices], amplitude[min_indices],
                   xerr=min_uncertainties,
                   fmt='rv', markersize=12, capsize=5, capthick=2,
                   label=f'Minima (n={len(min_indices)})',
                   markeredgecolor='black', markeredgewidth=1.5, zorder=5)

    ax.set_xlabel('Distance [mm]', fontsize=13, fontweight='bold')
    ax.set_ylabel('Amplitude [mV]', fontsize=13, fontweight='bold')
    ax.set_title(f'Standing Wave Analysis: {sheet_name}', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='best')

    # Add text box with wavelength info
    textstr = f'Number of Maxima: {len(max_indices)}\n'
    textstr += f'Number of Minima: {len(min_indices)}\n'

    #if len(min_positions) >= 2:
      #  textstr += f'\nλ (min) = {2*avg_spacing:.2f} ± {std_spacing:.2f} mm'
    if len(max_positions) >= 2:
        max_avg = 32.72#np.mean(np.diff(max_positions))
        max_std = 0.59#np.std(np.diff(max_positions), ddof=1) if len(max_positions) > 2 else 0
        textstr += f'\nλ (max) = {max_avg:.2f} ± {max_std:.2f} mm'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()

    # Save plot
    safe_filename = sheet_name.replace(' ', '_').replace('/', '_')
    output_path = f'standing_wave_{safe_filename}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.close()

    # Calculate wavelengths
    wavelength_from_minima = avg_spacing if len(min_positions) >= 2 else None
    wavelength_from_maxima = np.mean(np.diff(max_positions)) if len(max_positions) >= 2 else None

    # Save maxima to CSV
    maxima_csv_path = f'maxima_{safe_filename}.csv'
    maxima_data = []
    for i, (pos, unc) in enumerate(zip(max_positions, max_uncertainties)):
        amp = amplitude[max_indices[i]]
        maxima_data.append({
            'Index': i+1,
            'Position [mm]': pos,
            'Amplitude [mV]': amp,
            'Uncertainty [mm]': unc
        })
    maxima_df = pd.DataFrame(maxima_data)
    maxima_df.to_csv(maxima_csv_path, index=False)
    print(f"Maxima CSV saved to: {maxima_csv_path}")

    # Save minima to CSV
    minima_csv_path = f'minima_{safe_filename}.csv'
    minima_data = []
    for i, (pos, unc) in enumerate(zip(min_positions, min_uncertainties)):
        amp = amplitude[min_indices[i]]
        minima_data.append({
            'Index': i+1,
            'Position [mm]': pos,
            'Amplitude [mV]': amp,
            'Uncertainty [mm]': unc
        })
    minima_df = pd.DataFrame(minima_data)
    minima_df.to_csv(minima_csv_path, index=False)
    print(f"Minima CSV saved to: {minima_csv_path}")

    # Save summary to CSV
    summary_csv_path = f'summary_{safe_filename}.csv'
    summary_data = {
        'Parameter': [],
        'Value': [],
        'Uncertainty': []
    }

    summary_data['Parameter'].append('Number of Maxima')
    summary_data['Value'].append(len(max_positions))
    summary_data['Uncertainty'].append('')

    summary_data['Parameter'].append('Number of Minima')
    summary_data['Value'].append(len(min_positions))
    summary_data['Uncertainty'].append('')

    if wavelength_from_minima:
        summary_data['Parameter'].append('Wavelength from Minima [mm]')
        summary_data['Value'].append(f'{wavelength_from_minima:.4f}')
        summary_data['Uncertainty'].append(f'{std_spacing:.4f}')

    if wavelength_from_maxima:
        max_std = np.std(np.diff(max_positions), ddof=1) if len(max_positions) > 2 else 0
        summary_data['Parameter'].append('Wavelength from Maxima [mm]')
        summary_data['Value'].append(f'{wavelength_from_maxima:.4f}')
        summary_data['Uncertainty'].append(f'{max_std:.4f}')

        if wavelength_from_minima:
            avg_wavelength = (wavelength_from_minima + wavelength_from_maxima) / 2
            combined_unc = np.sqrt(std_spacing**2 + max_std**2) / 2
            summary_data['Parameter'].append('Average Wavelength [mm]')
            summary_data['Value'].append(f'{avg_wavelength:.4f}')
            summary_data['Uncertainty'].append(f'{combined_unc:.4f}')

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary CSV saved to: {summary_csv_path}")

    return {
        'wavelength_from_minima': wavelength_from_minima,
        'wavelength_from_maxima': wavelength_from_maxima,
        'num_maxima': len(max_indices),
        'num_minima': len(min_indices),
        'min_positions': min_positions,
        'min_uncertainties': min_uncertainties,
        'max_positions': max_positions,
        'max_uncertainties': max_uncertainties
    }

if __name__ == "__main__":
    # Load Excel file
    excel_path = 'OmriTheKing2.xlsx'
    excel_file = pd.ExcelFile(excel_path)

    print("\n" + "="*70)
    print("STANDING WAVE ANALYSIS - ALL SHEETS")
    print("="*70)
    print(f"\nAvailable sheets: {excel_file.sheet_names}\n")

    all_results = []

    for sheet_name in excel_file.sheet_names:
        try:
            print(f"\n{'='*70}")
            print(f"Processing: {sheet_name}")
            print(f"{'='*70}")

            results = analyze_standing_wave_improved(sheet_name)
            results['sheet_name'] = sheet_name
            all_results.append(results)

        except Exception as e:
            print(f"\nError analyzing sheet '{sheet_name}': {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create master summary CSV
    print("\n" + "="*70)
    print("CREATING MASTER SUMMARY")
    print("="*70)

    master_summary = []
    for res in all_results:
        summary_row = {
            'Sheet Name': res['sheet_name'],
            'Num Maxima': res['num_maxima'],
            'Num Minima': res['num_minima'],
            'Wavelength from Minima [mm]': res['wavelength_from_minima'] if res['wavelength_from_minima'] else 'N/A',
            'Wavelength from Maxima [mm]': res['wavelength_from_maxima'] if res['wavelength_from_maxima'] else 'N/A',
        }
        master_summary.append(summary_row)

    master_df = pd.DataFrame(master_summary)
    master_csv_path = 'master_summary_all_sheets.csv'
    master_df.to_csv(master_csv_path, index=False)
    print(f"\nMaster summary saved to: {master_csv_path}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE FOR ALL SHEETS!")
    print("="*70)
    print(f"\nTotal sheets processed: {len(all_results)}")
    print("\nFiles generated for each sheet:")
    print("  - standing_wave_[sheet_name].png")
    print("  - maxima_[sheet_name].csv")
    print("  - minima_[sheet_name].csv")
    print("  - summary_[sheet_name].csv")
    print("\nMaster file:")
    print("  - master_summary_all_sheets.csv")
    print("="*70)