#!/usr/bin/env python3
"""
White Noise Analysis Script - Experiment 2
Analyzes FFT data from white noise measurements with varying bandwidth.
Fixed parameters: Temperature = 14.6°C, Resistance = 68.3 kΩ
Variable: FFT Bandwidth
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm


def parse_bandwidth(filename):
    """
    Extract bandwidth value in Hz from filename.
    Examples: '125hz.txt' -> 125, '31.25hz.txt' -> 31.25, '7.813Hz.txt' -> 7.813
    """
    # Remove file extension
    name = filename.replace('.txt', '').strip()

    # Match patterns like "125hz", "31.25hz", "7.813Hz"
    pattern = r'([\d.]+)\s*hz'
    match = re.search(pattern, name, re.IGNORECASE)

    if match:
        value = float(match.group(1))
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


def main():
    # Path to experiment data (relative to this script's location)
    script_dir = Path(__file__).parent
    data_folder = script_dir / "experiment 2 - tempreture 14.6 R=68.3kohm"

    # Create output directory structure
    output_dir = script_dir / "output" / "Experiment_2" / "noise_spectra"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store data: {bandwidth: (frequencies, amplitudes)}
    spectra_data = {}

    # Load all data files
    print("Loading data files...")
    for filepath in sorted(data_folder.glob("*.txt")):
        bandwidth = parse_bandwidth(filepath.name)
        if bandwidth is not None:
            print(f"  Loading {filepath.name} -> Bandwidth = {bandwidth:.3f} Hz")
            data = load_data(filepath)
            if len(data) > 0:
                frequencies = data[:, 0]
                amplitudes = data[:, 1]
                spectra_data[bandwidth] = (frequencies, amplitudes)

    # Sort by bandwidth value
    sorted_bandwidths = sorted(spectra_data.keys())
    print(f"\nLoaded {len(sorted_bandwidths)} spectra")

    if len(sorted_bandwidths) == 0:
        print("Error: No data files were loaded. Check that:")
        print(f"  1. The folder '{data_folder}' exists")
        print("  2. It contains .txt files with bandwidth values in the filename")
        print("  3. Filenames match the expected pattern (e.g., '125hz.txt', '31.25hz.txt')")
        return

    print(f"Bandwidth range: {sorted_bandwidths[0]:.3f} Hz to {sorted_bandwidths[-1]:.3f} Hz")

    # Use raw data without normalization
    print("\nPreparing raw spectra data...")

    # Calculate the global min and max for y-axis scaling
    all_amplitudes = []
    for bandwidth in sorted_bandwidths:
        frequencies, amplitudes = spectra_data[bandwidth]
        all_amplitudes.extend(amplitudes)

    y_min = np.min(all_amplitudes)
    y_max = np.max(all_amplitudes)
    # Add some padding to the y-axis
    y_padding = (y_max - y_min) * 0.05
    y_min -= y_padding
    y_max += y_padding

    # Create custom color map (blue to red through purple, avoiding white)
    colors_list = ['blue', 'purple', 'red']
    colors = LinearSegmentedColormap.from_list('blue_purple_red', colors_list)
    n_spectra = len(sorted_bandwidths)

    # Calculate logarithmic normalization for color mapping
    log_bandwidths = np.log10(sorted_bandwidths)
    log_min = log_bandwidths[0]
    log_max = log_bandwidths[-1]

    # Create the plot
    print("\nCreating plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each spectrum with color gradient (logarithmic)
    for idx, bandwidth in enumerate(sorted_bandwidths):
        frequencies, amplitudes = spectra_data[bandwidth]
        # Map bandwidth to color using logarithmic scale
        log_bw = np.log10(bandwidth)
        color_position = (log_bw - log_min) / (log_max - log_min)
        color = colors(color_position)

        # Plot the spectrum
        label = f'{bandwidth:.2f} Hz' if bandwidth < 10 else f'{bandwidth:.1f} Hz'
        ax.plot(frequencies, amplitudes, color=color, alpha=0.7, linewidth=0.8,
                label=label)

    # Formatting
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Amplitude (dBm)', fontsize=12)
    ax.set_title('Noise spectra - Experiment 2 (R=68.3kΩ, T=14.6°C, varying bandwidth)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100000)  # 0 to 100 kHz

    # Set y-axis limits based on data
    ax.set_ylim(y_min, y_max)

    # Add zoomed-in inset at the bottom showing detail
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="75%", height="55%", loc='lower left',
                       bbox_to_anchor=(0.15, 0.35, 1, 1), bbox_transform=ax.transAxes)

    # Plot the same data in the inset with zoom on the noise floor
    for idx, bandwidth in enumerate(sorted_bandwidths):
        frequencies, amplitudes = spectra_data[bandwidth]
        log_bw = np.log10(bandwidth)
        color_position = (log_bw - log_min) / (log_max - log_min)
        color = colors(color_position)
        axins.plot(frequencies, amplitudes, color=color, alpha=0.7, linewidth=0.8)

    # Set zoom limits for inset (focus on noise floor region)
    axins.set_xlim(0, 50000)
    axins.set_ylim(-105, -70)
    axins.grid(True, alpha=0.3)
    axins.set_title('Zoomed View: Noise Floor Detail', fontsize=10, fontweight='bold', pad=5)
    axins.tick_params(labelsize=8)

    # Add a rectangle in the main plot to show zoomed region
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, -105), 50000, 35, linewidth=1.5, edgecolor='black',
                     facecolor='none', linestyle='--', alpha=0.5)
    ax.add_patch(rect)

    # Add colorbar to show bandwidth gradient (logarithmic scale)
    from matplotlib.colors import LogNorm
    sm = plt.cm.ScalarMappable(cmap=colors, norm=LogNorm(vmin=sorted_bandwidths[0], vmax=sorted_bandwidths[-1]))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Bandwidth (Hz)')

    # Format colorbar ticks to show bandwidth values
    cbar_ticks = sorted_bandwidths
    cbar.set_ticks(cbar_ticks)
    cbar_labels = [f'{bw:.2f}' if bw < 10 else f'{bw:.1f}' for bw in cbar_ticks]
    cbar.set_ticklabels(cbar_labels)

    # Adjust layout (use constrained_layout instead of tight_layout to avoid warning with inset)
    fig.set_constrained_layout(False)
    plt.subplots_adjust(right=0.9)

    # Save the figure
    output_file = output_dir / "noise_spectra_raw.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")

    # Show the plot
    plt.show()

    # Create separate plots for each bandwidth
    print("\nCreating individual plots for each bandwidth...")
    for idx, bandwidth in enumerate(sorted_bandwidths):
        frequencies, amplitudes = spectra_data[bandwidth]

        # Calculate color for this bandwidth (same as in combined plot)
        log_bw = np.log10(bandwidth)
        color_position = (log_bw - log_min) / (log_max - log_min)
        line_color = colors(color_position)

        # Create individual plot
        fig_individual, ax_individual = plt.subplots(figsize=(12, 6))

        # Plot the spectrum
        ax_individual.plot(frequencies, amplitudes, color=line_color, alpha=0.7, linewidth=1.2)

        # Formatting (same as combined plot)
        bw_label = f'{bandwidth:.2f} Hz' if bandwidth < 10 else f'{bandwidth:.1f} Hz'
        ax_individual.set_xlabel('Frequency (Hz)', fontsize=12)
        ax_individual.set_ylabel('Amplitude (dBm)', fontsize=12)
        ax_individual.set_title(f'Noise spectrum - Bandwidth = {bw_label} (R=68.3kΩ, T=14.6°C)',
                               fontsize=14, fontweight='bold')
        ax_individual.grid(True, alpha=0.3)
        ax_individual.set_xlim(0, 100000)  # 0 to 100 kHz
        ax_individual.set_ylim(-105, -70)  # Fixed y-axis for noise floor detail

        plt.tight_layout()

        # Save individual plot
        bw_filename = f'{bandwidth:.3f}'.replace('.', '_')
        individual_output = output_dir / f"noise_spectrum_{bw_filename}_hz.png"
        plt.savefig(individual_output, dpi=300, bbox_inches='tight')
        print(f"  Saved: {individual_output.name}")

        plt.close(fig_individual)

    print("\nAll individual plots saved!")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for bandwidth in sorted_bandwidths:
        frequencies, amplitudes = spectra_data[bandwidth]
        bw_label = f'{bandwidth:.2f} Hz' if bandwidth < 10 else f'{bandwidth:.1f} Hz'
        print(f"\nBandwidth = {bw_label}:")
        print(f"  Mean amplitude: {np.mean(amplitudes):.3f} dBm")
        print(f"  Std deviation: {np.std(amplitudes):.3f} dBm")
        print(f"  Min: {np.min(amplitudes):.3f} dBm, Max: {np.max(amplitudes):.3f} dBm")

    print("="*60)


if __name__ == "__main__":
    main()
