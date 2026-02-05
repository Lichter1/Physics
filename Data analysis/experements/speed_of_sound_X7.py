#!/usr/bin/env python3
"""
Speed of Sound Analysis from Excel X7
Infers speed of sound from resonance frequency vs number of nodes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

def linear_fit(x, m, b):
    """Linear fit: y = m*x + b"""
    return m * x + b

def analyze_tube_data(sheet_name, excel_path='X7.xlsx', tube_length_cm=89.8, tube_length_error_cm=0.3):
    """
    Analyze data for one tube configuration
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Assume columns: frequency_Hz, num_nodes
    if 'frequency_Hz' not in df.columns or 'num_nodes' not in df.columns:
        print(f"Warning: Expected columns 'frequency_Hz' and 'num_nodes' not found in {sheet_name}")
        return None

    freq = df['frequency_Hz'].values
    num_nodes = df['num_nodes'].values

    # Fit f = m * n + b
    popt, pcov = curve_fit(linear_fit, num_nodes, freq)
    m, b = popt
    m_err = np.sqrt(pcov[0,0])
    b_err = np.sqrt(pcov[1,1])

    # Calculate speed of sound
    if 'closed' in sheet_name.lower():
        # For closed tube: f = (v * n) / (2 * L)
        # So m = v / (2*L), v = m * 2 * L
        v = m * 2 * (tube_length_cm / 100)  # Convert cm to m
        v_err = m_err * 2 * (tube_length_cm / 100)
    elif 'open' in sheet_name.lower():
        # For open tube: f = (v * n) / (4 * L)
        # So m = v / (4*L), v = m * 4 * L
        v = m * 4 * (tube_length_cm / 100)
        v_err = m_err * 4 * (tube_length_cm / 100)
    else:
        print(f"Warning: Cannot determine tube type from sheet name '{sheet_name}'")
        v = np.nan
        v_err = np.nan

    return {
        'sheet': sheet_name,
        'frequencies': freq,
        'num_nodes': num_nodes,
        'slope': m,
        'slope_error': m_err,
        'intercept': b,
        'intercept_error': b_err,
        'speed_sound': v,
        'speed_sound_error': v_err
    }

def main():
    excel_path = 'X7.xlsx'
    tube_length_cm = 89.8
    tube_length_error_cm = 0.3

    try:
        excel_file = pd.ExcelFile(excel_path)
        print(f"Available sheets: {excel_file.sheet_names}")
    except FileNotFoundError:
        print(f"Error: {excel_path} not found. Please ensure the Excel file exists.")
        print("Using sample data for demonstration...")
        # Create sample data
        create_sample_data()
        excel_path = 'sample_X7.xlsx'
        excel_file = pd.ExcelFile(excel_path)

    results = []
    for sheet in excel_file.sheet_names:
        if 'open' in sheet.lower() or 'closed' in sheet.lower():
            print(f"Analyzing sheet: {sheet}")
            result = analyze_tube_data(sheet, excel_path, tube_length_cm, tube_length_error_cm)
            if result:
                results.append(result)

    # Plot results
    plt.figure(figsize=(12, 8))

    for res in results:
        plt.subplot(2, 2, 1)
        plt.errorbar(res['num_nodes'], res['frequencies'],
                    yerr=0, xerr=0, label=res['sheet'], marker='o')
        # Fit line
        n_fit = np.linspace(min(res['num_nodes']), max(res['num_nodes']), 100)
        f_fit = linear_fit(n_fit, res['slope'], res['intercept'])
        plt.plot(n_fit, f_fit, '--', label=f'{res["sheet"]} fit')

    plt.xlabel('Number of Nodes')
    plt.ylabel('Resonance Frequency (Hz)')
    plt.title('Resonance Frequency vs Number of Nodes')
    plt.legend()
    plt.grid(True)

    # Speed of sound bar chart
    plt.subplot(2, 2, 2)
    sheets = [r['sheet'] for r in results]
    speeds = [r['speed_sound'] for r in results]
    speed_errors = [r['speed_sound_error'] for r in results]

    plt.bar(sheets, speeds, yerr=speed_errors, capsize=5)
    plt.ylabel('Speed of Sound (m/s)')
    plt.title('Calculated Speed of Sound')
    plt.grid(True, axis='y')

    # Slope values
    plt.subplot(2, 2, 3)
    slopes = [r['slope'] for r in results]
    slope_errors = [r['slope_error'] for r in results]

    plt.bar(sheets, slopes, yerr=slope_errors, capsize=5)
    plt.ylabel('Slope (Hz/node)')
    plt.title('Fit Slope')
    plt.grid(True, axis='y')

    # Summary text
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = f"Tube Length: {tube_length_cm} ± {tube_length_error_cm} cm\n\n"
    for res in results:
        summary_text += f"{res['sheet']}:\n"
        summary_text += f"  Speed: {res['speed_sound']:.1f} ± {res['speed_sound_error']:.1f} m/s\n"
        summary_text += f"  Slope: {res['slope']:.1f} ± {res['slope_error']:.1f} Hz/node\n\n"

    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('X7_speed_of_sound_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save results to CSV
    summary_data = []
    for res in results:
        summary_data.append({
            'Sheet': res['sheet'],
            'Speed_of_Sound_m_s': res['speed_sound'],
            'Speed_Error_m_s': res['speed_sound_error'],
            'Slope_Hz_per_node': res['slope'],
            'Slope_Error': res['slope_error'],
            'Intercept_Hz': res['intercept'],
            'Intercept_Error': res['intercept_error']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('X7_speed_analysis_summary.csv', index=False)
    print("\nResults saved to: X7_speed_analysis_summary.csv")
    print("Plot saved to: X7_speed_of_sound_analysis.png")

def create_sample_data():
    """Create sample data for demonstration"""
    # Sample data for closed tube
    closed_data = {
        'frequency_Hz': [500, 1000, 1500, 2000],
        'num_nodes': [1, 2, 3, 4]
    }
    df_closed = pd.DataFrame(closed_data)

    # Sample data for open tube
    open_data = {
        'frequency_Hz': [250, 500, 750, 1000],
        'num_nodes': [1, 2, 3, 4]
    }
    df_open = pd.DataFrame(open_data)

    with pd.ExcelWriter('sample_X7.xlsx') as writer:
        df_closed.to_excel(writer, sheet_name='Closed Tube', index=False)
        df_open.to_excel(writer, sheet_name='Open Tube', index=False)

if __name__ == "__main__":
    main()