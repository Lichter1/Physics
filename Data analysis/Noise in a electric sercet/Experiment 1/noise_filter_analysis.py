#!/usr/bin/env python3
"""
White Noise Filtering and Statistical Analysis
Filters out interference peaks and 1/f noise to isolate white noise baseline
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


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


def analyze_single_resistor(filepath, output_dir=".", show_plots=True):
    """
    Analyze a single resistor measurement file.

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
    filtered_dir = output_dir / "output" / "filtered_plots"
    distribution_dir = output_dir / "output" / "distribution_plots"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    distribution_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Analyzing: {filepath.name}")
    print(f"{'='*70}")

    # Load data
    data = load_data(filepath)
    frequencies = data[:, 0]
    amplitudes = data[:, 1]

    print(f"Total data points: {len(frequencies)}")
    print(f"Frequency range: {frequencies[0]:.1f} Hz to {frequencies[-1]:.1f} Hz")
    print(f"Amplitude range: {amplitudes.min():.2f} dB to {amplitudes.max():.2f} dB")

    # Apply filtering
    print("\nApplying IQR-based filtering to remove peaks...")
    print("Removing 3 neighboring points on each side of detected outliers...")
    mask = filter_white_noise(frequencies, amplitudes, method='iqr', threshold=2.5, neighbor_points=3)

    # Separate filtered and rejected data
    freq_accepted = frequencies[mask]
    amp_accepted = amplitudes[mask]
    freq_rejected = frequencies[~mask]
    amp_rejected = amplitudes[~mask]

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
    ax.scatter(freq_rejected, amp_rejected, c='red', s=10, alpha=0.6,
                label=f'Rejected ({len(freq_rejected)} pts)', marker='x')
    ax.scatter(freq_accepted, amp_accepted, c='green', s=5, alpha=0.4,
                label=f'Accepted - White Noise ({len(freq_accepted)} pts)', marker='.')

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Amplitude (dB)', fontsize=12)
    ax.set_title(f'White Noise Filtering - {filepath.stem}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, frequencies[-1])

    # Add zoomed-in inset showing detail
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="75%", height="55%", loc='lower left',
                       bbox_to_anchor=(0.15, 0.30, 1, 1), bbox_transform=ax.transAxes)

    # Calculate zoom region (middle portion of the frequency range)
    mid_freq = frequencies[len(frequencies)//2]
    freq_window = (frequencies[-1] - frequencies[0]) / 6
    zoom_mask_freq = (frequencies > mid_freq - freq_window/2) & (frequencies < mid_freq + freq_window/2)

    zoom_accepted = mask & zoom_mask_freq
    zoom_rejected = (~mask) & zoom_mask_freq

    # Plot the zoomed data in the inset
    axins.scatter(frequencies[zoom_rejected], amplitudes[zoom_rejected],
                c='red', s=20, alpha=0.6, label='Rejected', marker='x')
    axins.scatter(frequencies[zoom_accepted], amplitudes[zoom_accepted],
                c='green', s=15, alpha=0.5, label='White Noise', marker='.')

    # Set zoom limits for inset
    axins.set_xlim(mid_freq - freq_window/2, mid_freq + freq_window/2)
    # Calculate appropriate y-limits for zoom region
    zoom_data = amplitudes[zoom_mask_freq]
    if len(zoom_data) > 0:
        y_margin = (zoom_data.max() - zoom_data.min()) * 0.1
        axins.set_ylim(zoom_data.min() - y_margin, zoom_data.max() + y_margin)
    axins.grid(True, alpha=0.3)
    axins.set_title(f'Zoomed View: {mid_freq-freq_window/2:.0f} - {mid_freq+freq_window/2:.0f} Hz',
                    fontsize=10, fontweight='bold', pad=5)
    axins.tick_params(labelsize=8)

    # Add a rectangle in the main plot to show zoomed region
    from matplotlib.patches import Rectangle
    if len(zoom_data) > 0:
        y_margin = (zoom_data.max() - zoom_data.min()) * 0.1
        rect = Rectangle((mid_freq - freq_window/2, zoom_data.min() - y_margin),
                        freq_window, zoom_data.max() - zoom_data.min() + 2*y_margin,
                        linewidth=1.5, edgecolor='black',
                        facecolor='none', linestyle='--', alpha=0.5)
        ax.add_patch(rect)

    # Adjust layout
    fig.set_constrained_layout(False)
    plt.subplots_adjust(right=0.95)

    # Save main figure
    output_path = filtered_dir / f"filtered_{filepath.stem}.png"
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
    ax_dist.set_title(f'Distribution of White Noise Power  - {filepath.stem}', fontsize=12, fontweight='bold')
    ax_dist.legend(loc='upper right', fontsize=9)
    ax_dist.grid(True, alpha=0.3)

    # Use scientific notation for x-axis
    ax_dist.ticklabel_format(axis='x', style='scientific', scilimits=(0,0))

    # Save distribution figure
    dist_output_path = distribution_dir / f"distribution_{filepath.stem}.png"
    plt.savefig(dist_output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {dist_output_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)
        plt.close(fig_dist)

    # Return filtered data for further analysis
    return {
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


def parse_resistance(filename):
    """Extract resistance value in ohms from filename."""
    import re
    name = filename.replace('.txt', '').replace('ido', '').strip()
    pattern = r'([\d.]+)\s*(k|kohm|ohm)?'
    match = re.search(pattern, name, re.IGNORECASE)

    if match:
        value = float(match.group(1))
        unit = match.group(2).lower() if match.group(2) else ''
        if 'k' in unit:
            return value * 1000
        else:
            return value
    return None


def analyze_all_resistors(data_folder="experiment 1 bandwidth 250Hz 14.6[C]",
                          output_dir=".", show_plots=False):
    """Analyze all resistor measurements in a folder."""

    data_folder = Path(data_folder)
    results = {}

    print("\n" + "="*70)
    print("ANALYZING ALL RESISTOR MEASUREMENTS")
    print("="*70)

    # Find all data files
    data_files = sorted(data_folder.glob("*.txt"))

    for data_file in data_files:
        resistance = parse_resistance(data_file.name)
        if resistance is not None:
            result = analyze_single_resistor(data_file, output_dir=output_dir,
                                            show_plots=show_plots)
            results[resistance] = result

    # Create summary table
    print("\n" + "="*85)
    print("SUMMARY TABLE - ALL RESISTORS (Linear Power Scale)")
    print("="*85)
    print(f"{'Resistance':>12} | {'μ (Linear)':>14} | {'σ (Linear)':>14} | {'Mean (dB)':>12} | {'Points':>7}")
    print("-" * 85)

    for resistance in sorted(results.keys()):
        r = results[resistance]
        r_label = f"{resistance:.0f} Ω" if resistance < 1000 else f"{resistance/1000:.1f} kΩ"
        print(f"{r_label:>12} | {r['mu_linear']:>14.6e} | {r['sigma_linear']:>14.6e} | {r['mean_db']:>12.4f} | {len(r['frequencies']):>7}")

    print("="*70)

    return results


def main():
    import sys

    # Get the directory where the script is located
    script_dir = Path(__file__).parent
    data_folder = script_dir / "experiment 1 bandwidth 250Hz 14.6[C]"

    if True: #len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Analyze all resistors
        results = analyze_all_resistors(
            data_folder=data_folder,
            output_dir=script_dir,
            show_plots=False
        )

        # Save results to file
        results_dir = script_dir / "output" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / "noise_analysis_results.txt"
        with open(results_file, "w") as f:
            f.write("White Noise Analysis Results (Linear Power Scale)\n")
            f.write("="*90 + "\n\n")
            f.write("Gaussian fit performed on LINEAR power values (correct for thermal noise analysis)\n")
            f.write("Power = 10^(dB/10)\n\n")
            f.write(f"{'Resistance':>12} | {'μ (Linear)':>14} | {'σ (Linear)':>14} | {'Mean (dB)':>12}\n")
            f.write("-" * 90 + "\n")
            for resistance in sorted(results.keys()):
                r = results[resistance]
                r_label = f"{resistance:.0f} Ω" if resistance < 1000 else f"{resistance/1000:.1f} kΩ"
                f.write(f"{r_label:>12} | {r['mu_linear']:>14.6e} | {r['sigma_linear']:>14.6e} | {r['mean_db']:>12.4f}\n")

        print(f"\nResults saved to: {results_file}")

    else:
        # Test with 913 ohm resistor only
        data_file = data_folder / "913ohm ido.txt"

        if not data_file.exists():
            data_file = data_folder / "913 ohm.txt"

        if not data_file.exists():
            print(f"Error: Could not find data file at {data_file}")
            print("\nSearching for 913 ohm files...")
            for f in data_folder.glob("*913*"):
                print(f"  Found: {f}")
                data_file = f
                break

        if data_file.exists():
            result = analyze_single_resistor(data_file, output_dir=script_dir, show_plots=True)

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
            print(f"\nTo analyze all resistors, run: python3 {sys.argv[0]} --all")

        else:
            print("Error: Could not find the data file!")


if __name__ == "__main__":
    main()
