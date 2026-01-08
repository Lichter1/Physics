"""
Standing Wave Comparison Analysis
Compares "with glass" and "without glass" datasets on the same plot
Calculates distances between corresponding maxima and minima
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

def smooth_data(distance, amplitude, window=5):
    """Smooth data using moving average"""
    amplitude_smooth = uniform_filter1d(amplitude, size=window)
    return amplitude_smooth

def find_extrema_improved(distance, amplitude, num_expected=6, smooth_window=11):
    """Find maxima and minima with smoothing for noisy data"""
    # Smooth the data first
    amplitude_smooth = smooth_data(distance, amplitude, window=smooth_window)

    # Calculate dynamic prominence threshold
    amp_range = np.max(amplitude_smooth) - np.min(amplitude_smooth)
    prominence = 0.05 * amp_range  # Very low to catch all peaks

    # Minimum distance between peaks - more flexible to catch early peaks
    min_distance = max(3, len(distance) // (num_expected * 4))  # Very flexible

    # Find maxima on smoothed data
    max_indices, max_props = find_peaks(amplitude_smooth,
                                        prominence=prominence,
                                        distance=min_distance)

    # Find minima on smoothed data
    min_indices, min_props = find_peaks(-amplitude_smooth,
                                        prominence=prominence,
                                        distance=min_distance)

    return max_indices, min_indices, amplitude_smooth

def calculate_position_uncertainty(distance, amplitude, extremum_idx, window=7):
    """
    Calculate uncertainty in extremum position using local curvature
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

def load_and_clean_data(df):
    """Load and clean the data, removing zeros and invalid points"""
    distance_col = df.columns[0]
    amplitude_col = df.columns[1]

    distance = df[distance_col].values
    amplitude = df[amplitude_col].values

    # Remove initial zeros and invalid data
    valid_mask = (amplitude > 0) & np.isfinite(amplitude) & np.isfinite(distance)
    distance = distance[valid_mask]
    amplitude = amplitude[valid_mask]

    return distance, amplitude

def find_nearest_extremum(pos, other_positions):
    """
    Find the nearest extremum and its left and right neighbors

    Returns: (nearest_idx, left_idx, right_idx, nearest_distance, left_distance, right_distance)
    """
    if len(other_positions) == 0:
        return None, None, None, None, None, None

    # Find nearest
    distances = np.abs(np.array(other_positions) - pos)
    nearest_idx = np.argmin(distances)
    nearest_distance = distances[nearest_idx]

    # Find left neighbor (one position to the left)
    left_idx = nearest_idx - 1 if nearest_idx > 0 else None
    left_distance = abs(other_positions[left_idx] - pos) if left_idx is not None else None

    # Find right neighbor (one position to the right)
    right_idx = nearest_idx + 1 if nearest_idx < len(other_positions) - 1 else None
    right_distance = abs(other_positions[right_idx] - pos) if right_idx is not None else None

    return nearest_idx, left_idx, right_idx, nearest_distance, left_distance, right_distance

# ============================================================================
# COMPARISON ANALYSIS
# ============================================================================

def compare_standing_waves(sheet1='without Perspex', sheet2='with Perspex'):
    """
    Compare two standing wave datasets
    """
    # Load Excel file
    excel_path = 'OmriTheKing2.xlsx'
    excel_file = pd.ExcelFile(excel_path)

    # Load both datasets
    df1 = pd.read_excel(excel_file, sheet_name=sheet1)
    df2 = pd.read_excel(excel_file, sheet_name=sheet2)

    distance1, amplitude1 = load_and_clean_data(df1)
    distance2, amplitude2 = load_and_clean_data(df2)

    # Find extrema for both
    max_indices1, min_indices1, amplitude_smooth1 = find_extrema_improved(
        distance1, amplitude1, num_expected=6, smooth_window=15
    )
    max_indices2, min_indices2, amplitude_smooth2 = find_extrema_improved(
        distance2, amplitude2, num_expected=6, smooth_window=15
    )

    # Extract positions
    max_pos1 = distance1[max_indices1]
    min_pos1 = distance1[min_indices1]
    max_pos2 = distance2[max_indices2]
    min_pos2 = distance2[min_indices2]

    # Calculate uncertainties for all extrema
    max_unc1 = [calculate_position_uncertainty(distance1, amplitude1, idx) for idx in max_indices1]
    min_unc1 = [calculate_position_uncertainty(distance1, amplitude1, idx) for idx in min_indices1]
    max_unc2 = [calculate_position_uncertainty(distance2, amplitude2, idx) for idx in max_indices2]
    min_unc2 = [calculate_position_uncertainty(distance2, amplitude2, idx) for idx in min_indices2]

    print(f"\n{'='*70}")
    print(f"COMPARISON ANALYSIS")
    print(f"{'='*70}")
    print(f"Dataset 1: {sheet1}")
    print(f"  Maxima: {len(max_pos1)}, Minima: {len(min_pos1)}")
    print(f"Dataset 2: {sheet2}")
    print(f"  Maxima: {len(max_pos2)}, Minima: {len(min_pos2)}")

    # Create tables for export
    print(f"\n{'='*70}")
    print(f"MAXIMA POSITIONS - {sheet1}")
    print(f"{'='*70}")
    print(f"{'Index':<8} {'Position [mm]':<18} {'Uncertainty [mm]'}")
    print(f"{'-'*70}")

    maxima1_data = []
    for i, (pos, unc) in enumerate(zip(max_pos1, max_unc1)):
        print(f"{i+1:<8} {pos:<18.4f} {unc:.4f}")
        maxima1_data.append({
            'Index': i+1,
            'Position [mm]': pos,
            'Uncertainty [mm]': unc
        })

    print(f"\n{'='*70}")
    print(f"MAXIMA POSITIONS - {sheet2}")
    print(f"{'='*70}")
    print(f"{'Index':<8} {'Position [mm]':<18} {'Uncertainty [mm]'}")
    print(f"{'-'*70}")

    maxima2_data = []
    for i, (pos, unc) in enumerate(zip(max_pos2, max_unc2)):
        print(f"{i+1:<8} {pos:<18.4f} {unc:.4f}")
        maxima2_data.append({
            'Index': i+1,
            'Position [mm]': pos,
            'Uncertainty [mm]': unc
        })

    print(f"\n{'='*70}")
    print(f"MINIMA POSITIONS - {sheet1}")
    print(f"{'='*70}")
    print(f"{'Index':<8} {'Position [mm]':<18} {'Uncertainty [mm]'}")
    print(f"{'-'*70}")

    minima1_data = []
    for i, (pos, unc) in enumerate(zip(min_pos1, min_unc1)):
        print(f"{i+1:<8} {pos:<18.4f} {unc:.4f}")
        minima1_data.append({
            'Index': i+1,
            'Position [mm]': pos,
            'Uncertainty [mm]': unc
        })

    print(f"\n{'='*70}")
    print(f"MINIMA POSITIONS - {sheet2}")
    print(f"{'='*70}")
    print(f"{'Index':<8} {'Position [mm]':<18} {'Uncertainty [mm]'}")
    print(f"{'-'*70}")

    minima2_data = []
    for i, (pos, unc) in enumerate(zip(min_pos2, min_unc2)):
        print(f"{i+1:<8} {pos:<18.4f} {unc:.4f}")
        minima2_data.append({
            'Index': i+1,
            'Position [mm]': pos,
            'Uncertainty [mm]': unc
        })

    # Create comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Plot dataset 1 (without glass) in blue - increased marker size
    ax.plot(distance1, amplitude1, '.', color='lightblue', markersize=4, alpha=0.6,
            label=f'{sheet1} - Raw Data')
    ax.plot(distance1, amplitude_smooth1, '-', color='blue', linewidth=2, alpha=0.8,
            label=f'{sheet1} - Smoothed')

    # Plot dataset 2 (with glass) in red - increased marker size
    ax.plot(distance2, amplitude2, '.', color='lightcoral', markersize=4, alpha=0.6,
            label=f'{sheet2} - Raw Data')
    ax.plot(distance2, amplitude_smooth2, '-', color='red', linewidth=2, alpha=0.8,
            label=f'{sheet2} - Smoothed')

    # Mark maxima for dataset 1 with error bars - smaller markers
    ax.errorbar(max_pos1, amplitude1[max_indices1], xerr=max_unc1,
                fmt='^', color='darkblue', markersize=7, capsize=4, capthick=1.5,
                label=f'{sheet1} - Maxima',
                markeredgecolor='black', markeredgewidth=1.2, zorder=5, elinewidth=1.5)

    # Mark maxima for dataset 2 with error bars - smaller markers
    ax.errorbar(max_pos2, amplitude2[max_indices2], xerr=max_unc2,
                fmt='^', color='darkred', markersize=7, capsize=4, capthick=1.5,
                label=f'{sheet2} - Maxima',
                markeredgecolor='black', markeredgewidth=1.2, zorder=5, elinewidth=1.5)

    # Mark minima for dataset 1 with error bars - smaller markers
    ax.errorbar(min_pos1, amplitude1[min_indices1], xerr=min_unc1,
                fmt='v', color='cyan', markersize=7, capsize=4, capthick=1.5,
                label=f'{sheet1} - Minima',
                markeredgecolor='black', markeredgewidth=1.2, zorder=5, elinewidth=1.5)

    # Mark minima for dataset 2 with error bars - smaller markers
    ax.errorbar(min_pos2, amplitude2[min_indices2], xerr=min_unc2,
                fmt='v', color='orange', markersize=7, capsize=4, capthick=1.5,
                label=f'{sheet2} - Minima',
                markeredgecolor='black', markeredgewidth=1.2, zorder=5, elinewidth=1.5)

    ax.set_xlabel('Distance [mm]', fontsize=13, fontweight='bold')
    ax.set_ylabel('Amplitude [mV]', fontsize=13, fontweight='bold')
    #ax.set_title(f'Standing Wave Comparison: {sheet1} vs {sheet2}',
       #         fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=9, loc='best', ncol=2,facecolor='wheat')

    plt.tight_layout()

    # Save plot
    output_path = 'standing_wave_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Comparison plot saved to: {output_path}")
    plt.close()

    # Save maxima positions to CSV
    maxima1_df = pd.DataFrame(maxima1_data)
    maxima1_csv = f'maxima_{sheet1.replace(" ", "_")}_positions.csv'
    maxima1_df.to_csv(maxima1_csv, index=False)
    print(f"Maxima positions ({sheet1}) saved to: {maxima1_csv}")

    maxima2_df = pd.DataFrame(maxima2_data)
    maxima2_csv = f'maxima_{sheet2.replace(" ", "_")}_positions.csv'
    maxima2_df.to_csv(maxima2_csv, index=False)
    print(f"Maxima positions ({sheet2}) saved to: {maxima2_csv}")

    # Save minima positions to CSV
    minima1_df = pd.DataFrame(minima1_data)
    minima1_csv = f'minima_{sheet1.replace(" ", "_")}_positions.csv'
    minima1_df.to_csv(minima1_csv, index=False)
    print(f"Minima positions ({sheet1}) saved to: {minima1_csv}")

    minima2_df = pd.DataFrame(minima2_data)
    minima2_csv = f'minima_{sheet2.replace(" ", "_")}_positions.csv'
    minima2_df.to_csv(minima2_csv, index=False)
    print(f"Minima positions ({sheet2}) saved to: {minima2_csv}")

    print(f"\n{'='*70}")
    print("COMPARISON ANALYSIS COMPLETE!")
    print(f"{'='*70}")

    return {
        'maxima1': maxima1_df,
        'maxima2': maxima2_df,
        'minima1': minima1_df,
        'minima2': minima2_df
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = compare_standing_waves(sheet1='without Perspex', sheet2='with Perspex')