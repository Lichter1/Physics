#!/usr/bin/env python3
"""
White Noise Analysis Script
Analyzes FFT data from white noise measurements in electronic circuits
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm


def parse_resistance(filename):
    """
    Extract resistance value in ohms from filename.
    Examples: '1k ohm ido.txt' -> 1000, '2.4 kohm ido.txt' -> 2400, '913ohm ido.txt' -> 913
    """
    # Remove file extension and extra text
    name = filename.replace('.txt', '').replace('ido', '').strip()

    # Match patterns like "1k", "2.4 kohm", "220k", "913"
    pattern = r'([\d.]+)\s*(k|kohm|ohm)?'
    match = re.search(pattern, name, re.IGNORECASE)

    if match:
        value = float(match.group(1))
        unit = match.group(2).lower() if match.group(2) else ''

        # Convert to ohms
        if 'k' in unit:
            return value * 1000
        else:
            return value

    return None


def load_data(filepath):
    """Load frequency and amplitude data from file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
            # Parse the data (format: \t frequency \t amplitude)
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    freq = float(parts[0])
                    amp = float(parts[1])
                    data.append([freq, amp])
                except (ValueError, IndexError):
                    continue

    return np.array(data) if data else np.array([]).reshape(0, 2)


def normalize_spectrum(frequencies, amplitudes):
    """
    Normalize spectrum by converting from dB to linear scale,
    then normalize to mean value.
    Uses median of stable region for robust normalization.
    """
    # Convert from dB to linear scale (voltage: 10^(dB/20), power: 10^(dB/10))
    # For noise measurements, we typically use power: 10^(dB/10)
    linear = 10 ** (amplitudes / 10)

    # Find stable baseline region (exclude obvious spikes)
    # Use percentile-based approach to avoid outliers
    sorted_linear = np.sort(linear)
    # Use values between 10th and 60th percentile as baseline
    n = len(sorted_linear)
    baseline_start = int(0.1 * n)
    baseline_end = int(0.6 * n)
    baseline = sorted_linear[baseline_start:baseline_end]

    # Use median of baseline for normalization (more robust than mean)
    median_val = np.median(baseline)

    # Normalize by dividing by the median baseline
    normalized = linear / median_val

    return normalized


def calculate_sampling_range(normalized_spectra):
    """Calculate the average of V² over all spectra."""
    # Average all normalized spectra
    avg_spectrum = np.mean(normalized_spectra, axis=0)
    mean_v2 = np.mean(avg_spectrum)
    return mean_v2


def main():
    # Path to experiment data
    data_folder = Path("experiment 1 bandwidth 250Hz 14.6[C]")

    # Dictionary to store data: {resistance: (frequencies, amplitudes)}
    spectra_data = {}

    # Load all data files
    print("Loading data files...")
    for filepath in sorted(data_folder.glob("*.txt")):
        resistance = parse_resistance(filepath.name)
        if resistance is not None:
            print(f"  Loading {filepath.name} -> R = {resistance:.1f} Ω")
            data = load_data(filepath)
            if len(data) > 0:
                frequencies = data[:, 0]
                amplitudes = data[:, 1]
                spectra_data[resistance] = (frequencies, amplitudes)

    # Sort by resistance value
    sorted_resistances = sorted(spectra_data.keys())
    print(f"\nLoaded {len(sorted_resistances)} spectra")
    print(f"Resistance range: {sorted_resistances[0]:.1f} Ω to {sorted_resistances[-1]:.1f} Ω")

    # Normalize all spectra
    print("\nNormalizing spectra...")
    normalized_data = {}
    for resistance in sorted_resistances:
        frequencies, amplitudes = spectra_data[resistance]
        normalized = normalize_spectrum(frequencies, amplitudes)
        normalized_data[resistance] = (frequencies, normalized)

    # Calculate average V² for sampling range indicator
    all_normalized = [normalized_data[r][1] for r in sorted_resistances]
    avg_v2 = calculate_sampling_range(all_normalized)

    # Create color map (blue to red)
    colors = plt.get_cmap('coolwarm')
    n_spectra = len(sorted_resistances)

    # Create the plot
    print("\nCreating plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each spectrum with color gradient
    for idx, resistance in enumerate(sorted_resistances):
        frequencies, normalized = normalized_data[resistance]
        color = colors(idx / (n_spectra - 1))

        # Plot the spectrum
        ax.plot(frequencies, normalized, color=color, alpha=0.7, linewidth=0.8,
                label=f'{resistance:.0f} Ω' if resistance < 1000 else f'{resistance/1000:.1f} kΩ')

    # Add horizontal line for average V² (sampling range)
    ax.axhline(y=avg_v2, color='black', linestyle='--', linewidth=1.5,
               label=f'<V²> sampling range = {avg_v2:.2f}')

    # Formatting
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Normalized Amplitude', fontsize=12)
    ax.set_title('Normalized view of the noise spectra (all R)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100000)  # 0 to 100 kHz

    # Set y-axis limits based on data, but cap extreme values
    all_normalized_values = np.concatenate([normalized_data[r][1] for r in sorted_resistances])
    y_max = min(np.percentile(all_normalized_values, 99), 2.0)  # Use 99th percentile or 2.0, whichever is smaller
    ax.set_ylim(0, max(y_max, 2.0))

    # Add legend (optional - can be removed if too cluttered)
    # ax.legend(loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()

    # Save the figure
    output_file = "noise_spectra_normalized.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")

    # Show the plot
    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for resistance in sorted_resistances:
        frequencies, normalized = normalized_data[resistance]
        r_label = f"{resistance:.0f} Ω" if resistance < 1000 else f"{resistance/1000:.1f} kΩ"
        print(f"\n{r_label}:")
        print(f"  Mean normalized amplitude: {np.mean(normalized):.3f}")
        print(f"  Std deviation: {np.std(normalized):.3f}")
        print(f"  Min: {np.min(normalized):.3f}, Max: {np.max(normalized):.3f}")

    print(f"\nOverall average <V²>: {avg_v2:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()
