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
    Examples: '1k ohm.txt' -> 1000, '2.4k ohm.txt' -> 2400, '913 ohm.txt' -> 913
    """
    # Remove file extension
    name = filename.replace('.txt', '').strip()

    # Match patterns like "1k ohm", "2.4k ohm", "220k ohm", "913 ohm"
    # This regex captures the number and the optional 'k' multiplier
    pattern = r'([\d.]+)\s*k?\s*ohm'
    match = re.search(pattern, name, re.IGNORECASE)

    if match:
        value = float(match.group(1))

        # Check if 'k' is present in the matched string
        if 'k' in match.group(0).lower():
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


##def normalize_spectrum(frequencies, amplitudes):
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
    # Path to experiment data (relative to this script's location)
    script_dir = Path(__file__).parent
    data_folder = script_dir / "experiment 1 bandwidth 250Hz 14.6[C]"

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

    if len(sorted_resistances) == 0:
        print("Error: No data files were loaded. Check that:")
        print(f"  1. The folder '{data_folder}' exists")
        print("  2. It contains .txt files with resistance values in the filename")
        print("  3. Filenames match the expected pattern (e.g., '1k ohm.txt', '913 ohm.txt')")
        return

    print(f"Resistance range: {sorted_resistances[0]:.1f} Ω to {sorted_resistances[-1]:.1f} Ω")

    # Use raw data without normalization
    print("\nPreparing raw spectra data...")

    # Calculate the global min and max for y-axis scaling
    all_amplitudes = []
    for resistance in sorted_resistances:
        frequencies, amplitudes = spectra_data[resistance]
        all_amplitudes.extend(amplitudes)

    y_min = np.min(all_amplitudes)
    y_max = np.max(all_amplitudes)
    # Add some padding to the y-axis
    y_padding = (y_max - y_min) * 0.05
    y_min -= y_padding
    y_max += y_padding

    # Create custom color map (blue to red through purple, avoiding white)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['blue', 'purple', 'red']
    colors = LinearSegmentedColormap.from_list('blue_purple_red', colors_list)
    n_spectra = len(sorted_resistances)

    # Calculate logarithmic normalization for color mapping
    log_resistances = np.log10(sorted_resistances)
    log_min = log_resistances[0]
    log_max = log_resistances[-1]

    # Create the plot
    print("\nCreating plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each spectrum with color gradient (logarithmic)
    for idx, resistance in enumerate(sorted_resistances):
        frequencies, amplitudes = spectra_data[resistance]
        # Map resistance to color using logarithmic scale
        log_r = np.log10(resistance)
        color_position = (log_r - log_min) / (log_max - log_min)
        color = colors(color_position)

        # Plot the spectrum
        ax.plot(frequencies, amplitudes, color=color, alpha=0.7, linewidth=0.8,
                label=f'{resistance:.0f} Ω' if resistance < 1000 else f'{resistance/1000:.1f} kΩ')

    # Formatting
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Amplitude (dBm)', fontsize=12)
    ax.set_title('Noise spectra (all R)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100000)  # 0 to 100 kHz

    # Set y-axis limits based on data
    ax.set_ylim(y_min, y_max)

    # Add zoomed-in inset at the bottom showing detail
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="75%", height="55%", loc='lower left',
                       bbox_to_anchor=(0.15, 0.35, 1, 1), bbox_transform=ax.transAxes)

    # Plot the same data in the inset with zoom on the noise floor
    for idx, resistance in enumerate(sorted_resistances):
        frequencies, amplitudes = spectra_data[resistance]
        log_r = np.log10(resistance)
        color_position = (log_r - log_min) / (log_max - log_min)
        color = colors(color_position)
        axins.plot(frequencies, amplitudes, color=color, alpha=0.7, linewidth=0.8)

    # Set zoom limits for inset (focus on noise floor region)
    axins.set_xlim(0, 100000)
    axins.set_ylim(-105, -70)
    axins.grid(True, alpha=0.3)
    axins.set_title('Zoomed View: Noise Floor Detail', fontsize=10, fontweight='bold', pad=5)
    #axins.set_xlabel('Frequency (Hz)', fontsize=9)
    #axins.set_ylabel('Amplitude (dBm)', fontsize=9)
    axins.tick_params(labelsize=8)

    # Add a rectangle in the main plot to show zoomed region
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, -105), 100000, 35, linewidth=1.5, edgecolor='black',
                     facecolor='none', linestyle='--', alpha=0.5)
    ax.add_patch(rect)

    # Add colorbar to show resistance gradient (logarithmic scale)
    from matplotlib.colors import LogNorm
    sm = plt.cm.ScalarMappable(cmap=colors, norm=LogNorm(vmin=sorted_resistances[0], vmax=sorted_resistances[-1]))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Resistance (Ω)')

    # Format colorbar ticks to show in kΩ for large values
    cbar_ticks = sorted_resistances
    cbar.set_ticks(cbar_ticks)
    cbar_labels = [f'{r:.0f}' if r < 1000 else f'{r/1000:.1f}k' for r in cbar_ticks]
    cbar.set_ticklabels(cbar_labels)

    # Adjust layout (use constrained_layout instead of tight_layout to avoid warning with inset)
    fig.set_constrained_layout(False)
    plt.subplots_adjust(right=0.9)

    # Save the figure
    output_file = script_dir / "noise_spectra_raw.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")

    # Show the plot
    plt.show()

    # Create separate plots for each resistance
    print("\nCreating individual plots for each resistance...")
    for idx, resistance in enumerate(sorted_resistances):
        frequencies, amplitudes = spectra_data[resistance]

        # Calculate color for this resistance (same as in combined plot)
        log_r = np.log10(resistance)
        color_position = (log_r - log_min) / (log_max - log_min)
        line_color = colors(color_position)

        # Create individual plot
        fig_individual, ax_individual = plt.subplots(figsize=(12, 6))

        # Plot the spectrum
        ax_individual.plot(frequencies, amplitudes, color=line_color, alpha=0.7, linewidth=1.2)

        # Formatting (same as combined plot)
        resistance_label = f'{resistance:.0f} Ω' if resistance < 1000 else f'{resistance/1000:.1f} kΩ'
        ax_individual.set_xlabel('Frequency (Hz)', fontsize=12)
        ax_individual.set_ylabel('Amplitude (dBm)', fontsize=12)
        ax_individual.set_title(f'Noise spectrum - R = {resistance_label}', fontsize=14, fontweight='bold')
        ax_individual.grid(True, alpha=0.3)
        ax_individual.set_xlim(0, 100000)  # 0 to 100 kHz
        ax_individual.set_ylim(-105, -70)  # Fixed y-axis for noise floor detail

        plt.tight_layout()

        # Save individual plot
        resistance_filename = f'{resistance:.0f}' if resistance < 1000 else f'{resistance/1000:.1f}k'
        individual_output = script_dir / f"noise_spectrum_{resistance_filename}_ohm.png"
        plt.savefig(individual_output, dpi=300, bbox_inches='tight')
        print(f"  Saved: {individual_output.name}")

        plt.close(fig_individual)

    print("\nAll individual plots saved!")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for resistance in sorted_resistances:
        frequencies, amplitudes = spectra_data[resistance]
        r_label = f"{resistance:.0f} Ω" if resistance < 1000 else f"{resistance/1000:.1f} kΩ"
        print(f"\n{r_label}:")
        print(f"  Mean amplitude: {np.mean(amplitudes):.3f} dBm")
        print(f"  Std deviation: {np.std(amplitudes):.3f} dBm")
        print(f"  Min: {np.min(amplitudes):.3f} dBm, Max: {np.max(amplitudes):.3f} dBm")

    print("="*60)


if __name__ == "__main__":
    main()
