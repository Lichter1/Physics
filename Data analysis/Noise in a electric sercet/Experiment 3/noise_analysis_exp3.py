#!/usr/bin/env python3
"""
White Noise Analysis Script - Experiment 3
Analyzes FFT data from white noise measurements with varying temperature.
Fixed parameters: Bandwidth = 250 Hz, Resistance = 1 kΩ
Variable: Temperature
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm


def parse_temperature(filename):
    """
    Extract temperature value in °C from filename.
    Examples: '25.txt' -> 25, '18.2.txt' -> 18.2, '100.3.txt' -> 100.3
    """
    # Remove file extension
    name = filename.replace('.txt', '').strip()

    # Match temperature value (the filename is just the temperature)
    try:
        value = float(name)
        return value
    except ValueError:
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
    data_folder = script_dir / "experiment 3_ Bandwidth = 250Hz, R=1kohm, T_uncertainty = 0.3c"

    # Create output directory structure
    output_dir = script_dir / "output" / "Experiment_3" / "noise_spectra"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store data: {temperature: (frequencies, amplitudes)}
    spectra_data = {}

    # Load all data files
    print("Loading data files...")
    for filepath in sorted(data_folder.glob("*.txt")):
        temperature = parse_temperature(filepath.name)
        if temperature is not None:
            print(f"  Loading {filepath.name} -> Temperature = {temperature:.1f} °C")
            data = load_data(filepath)
            if len(data) > 0:
                frequencies = data[:, 0]
                amplitudes = data[:, 1]
                spectra_data[temperature] = (frequencies, amplitudes)

    # Sort by temperature value
    sorted_temperatures = sorted(spectra_data.keys())
    print(f"\nLoaded {len(sorted_temperatures)} spectra")

    if len(sorted_temperatures) == 0:
        print("Error: No data files were loaded. Check that:")
        print(f"  1. The folder '{data_folder}' exists")
        print("  2. It contains .txt files with temperature values in the filename")
        print("  3. Filenames match the expected pattern (e.g., '25.txt', '18.2.txt')")
        return

    print(f"Temperature range: {sorted_temperatures[0]:.1f} °C to {sorted_temperatures[-1]:.1f} °C")

    # Use raw data without normalization
    print("\nPreparing raw spectra data...")

    # Calculate the global min and max for y-axis scaling
    all_amplitudes = []
    for temperature in sorted_temperatures:
        frequencies, amplitudes = spectra_data[temperature]
        all_amplitudes.extend(amplitudes)

    y_min = np.min(all_amplitudes)
    y_max = np.max(all_amplitudes)
    # Add some padding to the y-axis
    y_padding = (y_max - y_min) * 0.05
    y_min -= y_padding
    y_max += y_padding

    # Create custom color map (blue to red - cold to hot)
    colors_list = ['blue', 'purple', 'red']
    colors = LinearSegmentedColormap.from_list('blue_purple_red', colors_list)
    n_spectra = len(sorted_temperatures)

    # Calculate linear normalization for color mapping (temperature is linear)
    temp_min = sorted_temperatures[0]
    temp_max = sorted_temperatures[-1]

    # Create the plot
    print("\nCreating plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each spectrum with color gradient (linear for temperature)
    for idx, temperature in enumerate(sorted_temperatures):
        frequencies, amplitudes = spectra_data[temperature]
        # Map temperature to color using linear scale
        color_position = (temperature - temp_min) / (temp_max - temp_min)
        color = colors(color_position)

        # Plot the spectrum
        label = f'{temperature:.1f} °C'
        ax.plot(frequencies, amplitudes, color=color, alpha=0.7, linewidth=0.8,
                label=label)

    # Formatting
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Amplitude (dBm)', fontsize=12)
    ax.set_title('Noise spectra - Experiment 3 (R=1kΩ, Δf=250Hz, varying temperature)',
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
    for idx, temperature in enumerate(sorted_temperatures):
        frequencies, amplitudes = spectra_data[temperature]
        color_position = (temperature - temp_min) / (temp_max - temp_min)
        color = colors(color_position)
        axins.plot(frequencies, amplitudes, color=color, alpha=0.7, linewidth=0.8)

    # Set zoom limits for inset (focus on noise floor region)
    axins.set_xlim(80000, 100000)  # 80 kHz to 100 kHz
    axins.set_ylim(-105, -70)
    axins.grid(True, alpha=0.3)
    axins.set_title('Zoomed View: Noise Floor Detail', fontsize=10, fontweight='bold', pad=5)
    axins.tick_params(labelsize=8)

    # Add a rectangle in the main plot to show zoomed region (matches inset zoom limits)
    from matplotlib.patches import Rectangle
    zoom_x_min, zoom_x_max = 80000, 100000  # Same as axins.set_xlim
    zoom_y_min, zoom_y_max = -105, -70      # Same as axins.set_ylim
    rect = Rectangle((zoom_x_min, zoom_y_min), zoom_x_max - zoom_x_min, zoom_y_max - zoom_y_min,
                     linewidth=1.5, edgecolor='black', facecolor='none', linestyle='--', alpha=0.5)
    ax.add_patch(rect)

    # Add colorbar to show temperature gradient (linear scale)
    from matplotlib.colors import Normalize
    sm = plt.cm.ScalarMappable(cmap=colors, norm=Normalize(vmin=sorted_temperatures[0], vmax=sorted_temperatures[-1]))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Temperature (°C)')

    # Format colorbar ticks to show temperature values
    cbar_ticks = sorted_temperatures
    cbar.set_ticks(cbar_ticks)
    cbar_labels = [f'{t:.1f}' for t in cbar_ticks]
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

    # Create separate plots for each temperature
    print("\nCreating individual plots for each temperature...")
    for idx, temperature in enumerate(sorted_temperatures):
        frequencies, amplitudes = spectra_data[temperature]

        # Calculate color for this temperature (same as in combined plot)
        color_position = (temperature - temp_min) / (temp_max - temp_min)
        line_color = colors(color_position)

        # Create individual plot
        fig_individual, ax_individual = plt.subplots(figsize=(12, 6))

        # Plot the spectrum
        ax_individual.plot(frequencies, amplitudes, color=line_color, alpha=0.7, linewidth=1.2)

        # Formatting (same as combined plot)
        temp_label = f'{temperature:.1f} °C'
        ax_individual.set_xlabel('Frequency (Hz)', fontsize=12)
        ax_individual.set_ylabel('Amplitude (dBm)', fontsize=12)
        ax_individual.set_title(f'Noise spectrum - Temperature = {temp_label} (R=1kΩ, Δf=250Hz)',
                               fontsize=14, fontweight='bold')
        ax_individual.grid(True, alpha=0.3)
        ax_individual.set_xlim(0, 100000)  # 0 to 100 kHz
        ax_individual.set_ylim(-105, -70)  # Fixed y-axis for noise floor detail

        plt.tight_layout()

        # Save individual plot
        temp_filename = f'{temperature:.1f}'.replace('.', '_')
        individual_output = output_dir / f"noise_spectrum_{temp_filename}_C.png"
        plt.savefig(individual_output, dpi=300, bbox_inches='tight')
        print(f"  Saved: {individual_output.name}")

        plt.close(fig_individual)

    print("\nAll individual plots saved!")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for temperature in sorted_temperatures:
        frequencies, amplitudes = spectra_data[temperature]
        temp_label = f'{temperature:.1f} °C'
        print(f"\nTemperature = {temp_label}:")
        print(f"  Mean amplitude: {np.mean(amplitudes):.3f} dBm")
        print(f"  Std deviation: {np.std(amplitudes):.3f} dBm")
        print(f"  Min: {np.min(amplitudes):.3f} dBm, Max: {np.max(amplitudes):.3f} dBm")

    print("="*60)


if __name__ == "__main__":
    main()
