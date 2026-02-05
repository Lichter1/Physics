#!/usr/bin/env python3
"""
Organize Excel data for standing wave experiment S7
Extract number of nodes vs frequency for open and closed tube tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def count_maxima(row):
    """Count the number of non-NaN maxima positions"""
    count = 0
    i = 1
    while f'max{i}_cm' in row.index:
        if not pd.isna(row[f'max{i}_cm']):
            count += 1
        i += 1
    return count

def analyze_sheet(sheet_name, excel_path='S7.xlsx'):
    """
    Analyze a sheet and extract frequency vs number of nodes
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Assume columns: frequency_Hz, error_freq_Hz, max1_cm, max2_cm, ...
    results = []

    for idx, row in df.iterrows():
        freq = row['frequency_Hz']
        error_freq = row['error_freq_Hz']
        num_maxima = count_maxima(row)

        # For standing waves, number of nodes = number of minima
        # But if we have maxima positions, number of minima = number of maxima - 1 (approximately)
        # For closed tube: nodes = antinodes
        # For open tube: nodes = antinodes - 1
        # But let's assume num_nodes = num_maxima for simplicity, or adjust based on tube type

        if 'open' in sheet_name.lower():
            num_nodes = num_maxima - 1  # Open tube: one less node
        else:
            num_nodes = num_maxima  # Closed tube: equal nodes and antinodes

        results.append({
            'frequency_Hz': freq,
            'error_freq_Hz': error_freq,
            'num_maxima': num_maxima,
            'num_nodes': num_nodes
        })

    return pd.DataFrame(results)

def main():
    excel_path = 'S7.xlsx'

    try:
        excel_file = pd.ExcelFile(excel_path)
        print(f"Available sheets: {excel_file.sheet_names}")
    except FileNotFoundError:
        print(f"Error: {excel_path} not found. Please ensure the Excel file exists.")
        return

    # Analyze each sheet
    results = {}
    for sheet in excel_file.sheet_names:
        if 'open' in sheet.lower() or 'closed' in sheet.lower():
            print(f"Analyzing sheet: {sheet}")
            results[sheet] = analyze_sheet(sheet, excel_path)

    # Create summary table
    summary_data = []
    for sheet, df in results.items():
        for _, row in df.iterrows():
            summary_data.append({
                'Test': sheet,
                'Frequency_Hz': row['frequency_Hz'],
                'Error_Freq_Hz': row['error_freq_Hz'],
                'Number_of_Nodes': row['num_nodes']
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('S7_organized_summary.csv', index=False)

    print("\nOrganized data saved to: S7_organized_summary.csv")
    print("\nSummary:")
    print(summary_df)

    # Plot
    plt.figure(figsize=(10, 6))
    for sheet, df in results.items():
        plt.errorbar(df['frequency_Hz'], df['num_nodes'],
                    xerr=df['error_freq_Hz'], label=sheet, marker='o')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Number of Nodes')
    plt.title('Number of Nodes vs Frequency for Standing Wave Experiments')
    plt.legend()
    plt.grid(True)
    plt.savefig('S7_nodes_vs_frequency.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()