#!/usr/bin/env python3
"""
White Noise Filtering and Statistical Analysis - Experiment 3
Filters out interference peaks and 1/f noise to isolate white noise baseline.
Fixed parameters: Bandwidth = 250 Hz, Resistance = 1 kΩ
Variable: Temperature
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import re


def load_data(filepath):
    """Load frequency and amplitude data from file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    freq = float(parts[0])
                    amp = float(parts[1])
                    data.append([freq, amp])
                except (ValueError, IndexError):
                    continue

    return np.array(data) if data else np.array([]).reshape(0, 2)


def filter_white_noise(frequencies, amplitudes, method='iqr', threshold=3.0, neighbor_points=3):
    """
    Filter data to extract only white noise baseline.

    Parameters:
    -----------
    frequencies : array
        Frequency values
    amplitudes : array
        Amplitude values in dB
    method : str
        Filtering method: 'iqr' (Interquartile Range) or 'mad' (Median Absolute Deviation)
    threshold : float
        Number of IQRs or MADs above median to consider as outlier
    neighbor_points : int
        Number of neighboring points to remove on each side of detected outliers (default: 3)

    Returns:
    --------
    mask : boolean array
        True for white noise baseline points, False for peaks/outliers
    """

    # Convert from dB to linear scale (power)
    linear = 10 ** (amplitudes / 10)

    if method == 'iqr':
        # Interquartile Range method (robust to outliers)
        q1 = np.percentile(linear, 25)
        q3 = np.percentile(linear, 75)
        iqr = q3 - q1
        median = np.median(linear)

        # Keep only values below median + threshold * IQR
        # This removes the high peaks while keeping the baseline
        upper_bound = median + threshold * iqr
        mask = linear < upper_bound

    elif method == 'mad':
        # Median Absolute Deviation method (very robust)
        median = np.median(linear)
        mad = np.median(np.abs(linear - median))

        # Keep only values within threshold * MAD from median
        upper_bound = median + threshold * mad
        mask = linear < upper_bound

    else:
        raise ValueError(f"Unknown method: {method}")

    # Expand rejection to neighboring points
    if neighbor_points > 0:
        # Find indices of rejected points
        rejected_indices = np.where(~mask)[0]

        # Create a new mask to mark all points to be rejected
        expanded_mask = mask.copy()

        for idx in rejected_indices:
            # Mark neighbors to the left
            start = max(0, idx - neighbor_points)
            # Mark neighbors to the right
            end = min(len(mask), idx + neighbor_points + 1)
            # Reject all points in this range
            expanded_mask[start:end] = False

        mask = expanded_mask

    return mask


def parse_temperature(filename):
    """Extract temperature value in °C from filename."""
    name = filename.replace('.txt', '').strip()
    try:
        value = float(name)
        return value
    except ValueError:
        return None


def analyze_single_temperature(filepath, output_dir=".", show_plots=True):
    """
    Analyze a single temperature measurement file.

    Parameters:
    -----------
    filepath : str or Path
        Path to data file
    output_dir : str
        Directory to save output plots
    show_plots : bool
        Whether to display plots interactively
    """

    filepath = Path(filepath)
    output_dir = Path(output_dir)

    # Create organized output directories
    filtered_dir = output_dir / "output" / "Experiment_3" / "filtered_plots"
    distribution_dir = output_dir / "output" / "Experiment_3" / "distribution_plots"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    distribution_dir.mkdir(parents=True, exist_ok=True)

    temperature = parse_temperature(filepath.name)
    temp_label = f'{temperature:.1f} °C'

    print(f"\n{'='*70}")
    print(f"Analyzing: {filepath.name} (Temperature = {temp_label})")
    print(f"{'='*70}")

    # Load data
    data = load_data(filepath)
    frequencies_full = data[:, 0]
    amplitudes_full = data[:, 1]

    print(f"Total data points: {len(frequencies_full)}")
    print(f"Frequency range: {frequencies_full[0]:.1f} Hz to {frequencies_full[-1]:.1f} Hz")
    print(f"Amplitude range: {amplitudes_full.min():.2f} dB to {amplitudes_full.max():.2f} dB")

    # Define allowed frequency ranges (clean regions without external interference)
    # Range 1: 2500 - 14000 Hz
    # Range 2: 95000 - 100000 Hz
    FREQ_RANGE_1 = (2000, 14500)
    FREQ_RANGE_2 = (91000, 100000)

    # Create mask for allowed frequency ranges
    freq_range_mask = ((frequencies_full >= FREQ_RANGE_1[0]) & (frequencies_full <= FREQ_RANGE_1[1])) | \
                      ((frequencies_full >= FREQ_RANGE_2[0]) & (frequencies_full <= FREQ_RANGE_2[1]))

    # Restrict data to allowed frequency ranges
    frequencies = frequencies_full[freq_range_mask]
    amplitudes = amplitudes_full[freq_range_mask]

    print(f"\nRestricting analysis to clean frequency regions:")
    print(f"  Range 1: {FREQ_RANGE_1[0]} - {FREQ_RANGE_1[1]} Hz")
    print(f"  Range 2: {FREQ_RANGE_2[0]} - {FREQ_RANGE_2[1]} Hz")
    print(f"  Data points in allowed ranges: {len(frequencies)} ({100*len(frequencies)/len(frequencies_full):.1f}%)")

    # Apply filtering on the restricted data
    print("\nApplying IQR-based filtering to remove peaks...")
    print("Removing 3 neighboring points on each side of detected outliers...")
    mask = filter_white_noise(frequencies, amplitudes, method='iqr', threshold=2.5, neighbor_points=3)

    # Separate filtered and rejected data (within allowed ranges)
    freq_accepted = frequencies[mask]
    amp_accepted = amplitudes[mask]
    freq_rejected = frequencies[~mask]
    amp_rejected = amplitudes[~mask]

    # For plotting, we also need to show the excluded frequency regions
    freq_excluded = frequencies_full[~freq_range_mask]
    amp_excluded = amplitudes_full[~freq_range_mask]

    print(f"Accepted points (white noise): {len(freq_accepted)} ({100*len(freq_accepted)/len(frequencies):.1f}%)")
    print(f"Rejected points (peaks): {len(freq_rejected)} ({100*len(freq_rejected)/len(frequencies):.1f}%)")

    # Convert from dB to linear scale (power)
    # dB values represent 10*log10(P), so P = 10^(dB/10)
    linear_accepted = 10 ** (amp_accepted / 10)

    # Calculate statistics in LINEAR scale (physically meaningful for thermal noise)
    # Thermal noise voltage has Gaussian distribution, so power (V²) analysis should be in linear
    mean_linear = np.mean(linear_accepted)
    std_linear = np.std(linear_accepted)
    median_linear = np.median(linear_accepted)

    # Also report dB equivalents for reference
    mean_db = 10 * np.log10(mean_linear)
    median_db = 10 * np.log10(median_linear)

    print(f"\nWhite Noise Statistics (Linear Scale - Power):")
    print(f"  Mean (μ): {mean_linear:.6e}")
    print(f"  Std Dev (σ): {std_linear:.6e}")
    print(f"  Median: {median_linear:.6e}")
    print(f"  Coefficient of Variation (σ/μ): {std_linear/mean_linear:.4f}")
    print(f"\n  dB equivalents (for reference):")
    print(f"    Mean: {mean_db:.4f} dB")
    print(f"    Median: {median_db:.4f} dB")

    # Fit Gaussian to the LINEAR power values (correct approach)
    mu_linear, sigma_linear = stats.norm.fit(linear_accepted)
    print(f"\nGaussian Fit (Linear Power Scale):")
    print(f"  μ (mean): {mu_linear:.6e}")
    print(f"  σ (sigma): {sigma_linear:.6e}")
    print(f"  Uncertainty range (±1σ): [{mu_linear-sigma_linear:.6e}, {mu_linear+sigma_linear:.6e}]")
    print(f"  Uncertainty range (±2σ): [{mu_linear-2*sigma_linear:.6e}, {mu_linear+2*sigma_linear:.6e}]")
    print(f"  In dB: μ = {10*np.log10(mu_linear):.4f} dB")

    # Create main plot with inset zoom
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot 1: Filtered data visualization (full spectrum)
    # First plot excluded frequency regions (gray)
    ax.scatter(freq_excluded, amp_excluded, c='gray', s=3, alpha=0.3,
                label=f'Excluded regions ({len(freq_excluded)} pts)', marker='.')
    # Then plot rejected points within allowed regions (red)
    ax.scatter(freq_rejected, amp_rejected, c='red', s=10, alpha=0.6,
                label=f'Rejected ({len(freq_rejected)} pts)', marker='x')
    # Finally plot accepted white noise points (green)
    ax.scatter(freq_accepted, amp_accepted, c='green', s=5, alpha=0.4,
                label=f'Accepted - White Noise ({len(freq_accepted)} pts)', marker='.')

    # Add shaded regions to indicate allowed frequency ranges
    ax.axvspan(FREQ_RANGE_1[0], FREQ_RANGE_1[1], alpha=0.1, color='green', label='Analysis regions')
    ax.axvspan(FREQ_RANGE_2[0], FREQ_RANGE_2[1], alpha=0.1, color='green')

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Amplitude (dB)', fontsize=12)
    ax.set_title(f'White Noise Filtering - Temperature = {temp_label} (R=1kΩ, Δf=250Hz)\nAnalysis regions: {FREQ_RANGE_1[0]}-{FREQ_RANGE_1[1]} Hz, {FREQ_RANGE_2[0]}-{FREQ_RANGE_2[1]} Hz',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, frequencies_full[-1])

    # Add zoomed-in inset showing detail of the first analysis region
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="75%", height="55%", loc='lower left',
                       bbox_to_anchor=(0.15, 0.30, 1, 1), bbox_transform=ax.transAxes)

    # Focus zoom on the first allowed region (2500-14000 Hz)
    zoom_freq_min = FREQ_RANGE_1[0]
    zoom_freq_max = FREQ_RANGE_1[1]
    zoom_mask_freq = (frequencies >= zoom_freq_min) & (frequencies <= zoom_freq_max)

    zoom_accepted = mask & zoom_mask_freq
    zoom_rejected = (~mask) & zoom_mask_freq

    # Plot the zoomed data in the inset
    axins.scatter(frequencies[zoom_rejected], amplitudes[zoom_rejected],
                c='red', s=20, alpha=0.6, label='Rejected', marker='x')
    axins.scatter(frequencies[zoom_accepted], amplitudes[zoom_accepted],
                c='green', s=15, alpha=0.5, label='White Noise', marker='.')

    # Set zoom limits for inset
    axins.set_xlim(zoom_freq_min, zoom_freq_max)
    # Calculate appropriate y-limits for zoom region
    zoom_data = amplitudes[zoom_mask_freq]
    if len(zoom_data) > 0:
        y_margin = (zoom_data.max() - zoom_data.min()) * 0.1
        axins.set_ylim(zoom_data.min() - y_margin, zoom_data.max() + y_margin)
    axins.grid(True, alpha=0.3)
    axins.set_title(f'Zoomed View: {zoom_freq_min:.0f} - {zoom_freq_max:.0f} Hz (Region 1)',
                    fontsize=10, fontweight='bold', pad=5)
    axins.tick_params(labelsize=8)

    # Add a rectangle in the main plot to show zoomed region
    from matplotlib.patches import Rectangle
    if len(zoom_data) > 0:
        y_margin = (zoom_data.max() - zoom_data.min()) * 0.1
        rect = Rectangle((zoom_freq_min, zoom_data.min() - y_margin),
                        zoom_freq_max - zoom_freq_min, zoom_data.max() - zoom_data.min() + 2*y_margin,
                        linewidth=1.5, edgecolor='black',
                        facecolor='none', linestyle='--', alpha=0.5)
        ax.add_patch(rect)

    # Adjust layout
    fig.set_constrained_layout(False)
    plt.subplots_adjust(right=0.95)

    # Save main figure
    temp_filename = f'{temperature:.1f}'.replace('.', '_')
    output_path = filtered_dir / f"filtered_{temp_filename}C.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nMain plot saved to: {output_path}")

    # Create separate distribution plot - using LINEAR power values (correct for thermal noise)
    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))

    n_bins = 40
    ax_dist.hist(linear_accepted, bins=n_bins, density=True,
                 alpha=0.7, color='green', edgecolor='black',
                 label='White Noise Data (Linear Power)')

    # Plot fitted Gaussian on linear values
    x_range = np.linspace(linear_accepted.min(), linear_accepted.max(), 200)
    gaussian_fit = stats.norm.pdf(x_range, mu_linear, sigma_linear)
    ax_dist.plot(x_range, gaussian_fit, 'r-', linewidth=2.5,
                 label=f'Gaussian Fit\nμ={mu_linear:.3e}\nσ={sigma_linear:.3e}')

    # Mark mean and ±1σ, ±2σ
    ax_dist.axvline(mu_linear, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Mean (μ)')
    ax_dist.axvline(mu_linear - sigma_linear, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='±1σ')
    ax_dist.axvline(mu_linear + sigma_linear, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax_dist.axvline(mu_linear - 2*sigma_linear, color='purple', linestyle=':', linewidth=1, alpha=0.7, label='±2σ')
    ax_dist.axvline(mu_linear + 2*sigma_linear, color='purple', linestyle=':', linewidth=1, alpha=0.7)

    ax_dist.set_xlabel('V² [V]', fontsize=12)
    ax_dist.set_ylabel('Probability Density', fontsize=12)
    ax_dist.set_title(f'Distribution of White Noise Power - Temperature = {temp_label}',
                     fontsize=12, fontweight='bold')
    ax_dist.legend(loc='upper right', fontsize=9)
    ax_dist.grid(True, alpha=0.3)

    # Use scientific notation for x-axis
    ax_dist.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))

    # Save distribution figure
    dist_output_path = distribution_dir / f"distribution_{temp_filename}C.png"
    plt.savefig(dist_output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {dist_output_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_dist)

    # Return filtered data for further analysis
    return {
        'temperature': temperature,
        'frequencies': freq_accepted,
        'amplitudes_db': amp_accepted,
        'amplitudes_linear': linear_accepted,
        'mask': mask,
        'rejection_rate': 100 * len(freq_rejected) / len(frequencies),
        'mean_db': mean_db,
        'median_db': median_db,
        'mu_linear': mu_linear,         # Gaussian fit mean (linear power)
        'sigma_linear': sigma_linear,   # Gaussian fit std dev (linear power)
        'mean_linear': mean_linear,
        'std_linear': std_linear
    }


def analyze_all_temperatures(data_folder="experiment 3_ Bandwidth = 250Hz, R=1kohm, T_uncertainty = 0.3c",
                             output_dir=".", show_plots=False):
    """Analyze all temperature measurements in a folder."""

    data_folder = Path(data_folder)
    results = {}

    print("\n" + "="*70)
    print("ANALYZING ALL TEMPERATURE MEASUREMENTS")
    print("="*70)

    # Find all data files
    data_files = sorted(data_folder.glob("*.txt"))

    for data_file in data_files:
        temperature = parse_temperature(data_file.name)
        if temperature is not None:
            result = analyze_single_temperature(data_file, output_dir=output_dir,
                                               show_plots=show_plots)
            results[temperature] = result

    # Create summary table
    print("\n" + "="*85)
    print("SUMMARY TABLE - ALL TEMPERATURES (Linear Power Scale)")
    print("="*85)
    print(f"{'Temperature':>12} | {'μ (Linear)':>14} | {'σ (Linear)':>14} | {'Mean (dB)':>12} | {'Points':>7}")
    print("-" * 85)

    for temperature in sorted(results.keys()):
        r = results[temperature]
        temp_label = f"{temperature:.1f} °C"
        print(f"{temp_label:>12} | {r['mu_linear']:>14.6e} | {r['sigma_linear']:>14.6e} | {r['mean_db']:>12.4f} | {len(r['frequencies']):>7}")

    print("="*70)

    return results


def main():
    import sys

    # Get the directory where the script is located
    script_dir = Path(__file__).parent
    data_folder = script_dir / "experiment 3_ Bandwidth = 250Hz, R=1kohm, T_uncertainty = 0.3c"

    if True:  # Analyze all temperatures
        # Analyze all temperatures
        results = analyze_all_temperatures(
            data_folder=data_folder,
            output_dir=script_dir,
            show_plots=False
        )

        # Save results to file
        results_dir = script_dir / "output" / "Experiment_3" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / "noise_analysis_results.txt"
        with open(results_file, "w") as f:
            f.write("White Noise Analysis Results - Experiment 3 (Linear Power Scale)\n")
            f.write("Fixed: R = 1 kΩ, Δf = 250 Hz\n")
            f.write("Variable: Temperature\n")
            f.write("="*90 + "\n\n")
            f.write("Analysis frequency ranges (clean regions):\n")
            f.write("  Range 1: 2500 - 14000 Hz\n")
            f.write("  Range 2: 95000 - 100000 Hz\n\n")
            f.write("Gaussian fit performed on LINEAR power values (correct for thermal noise analysis)\n")
            f.write("Power = 10^(dB/10)\n\n")
            f.write(f"{'Temperature':>12} | {'μ (Linear)':>14} | {'σ (Linear)':>14} | {'Mean (dB)':>12}\n")
            f.write("-" * 90 + "\n")
            for temperature in sorted(results.keys()):
                r = results[temperature]
                temp_label = f"{temperature:.1f} °C"
                f.write(f"{temp_label:>12} | {r['mu_linear']:>14.6e} | {r['sigma_linear']:>14.6e} | {r['mean_db']:>12.4f}\n")

        print(f"\nResults saved to: {results_file}")

    else:
        # Test with single temperature file only
        data_file = data_folder / "25.txt"

        if data_file.exists():
            result = analyze_single_temperature(data_file, output_dir=script_dir, show_plots=True)

            print(f"\n{'='*70}")
            print("ANALYSIS COMPLETE - FINAL RESULTS (Linear Power Scale)")
            print(f"{'='*70}")
            print(f"Rejection rate: {result['rejection_rate']:.1f}%")
            print(f"White noise points retained: {len(result['frequencies'])}")
            print(f"\nGaussian Fit (Linear Power):")
            print(f"  μ = {result['mu_linear']:.6e}")
            print(f"  σ = {result['sigma_linear']:.6e}")
            print(f"  Uncertainty (±1σ): [{result['mu_linear']-result['sigma_linear']:.6e}, {result['mu_linear']+result['sigma_linear']:.6e}]")
            print(f"  Uncertainty (±2σ): [{result['mu_linear']-2*result['sigma_linear']:.6e}, {result['mu_linear']+2*result['sigma_linear']:.6e}]")
            print(f"\nIn dB (for reference):")
            print(f"  Mean: {result['mean_db']:.4f} dB")
            print(f"{'='*70}")

        else:
            print("Error: Could not find the data file!")


if __name__ == "__main__":
    main()
