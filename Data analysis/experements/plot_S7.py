import pandas as pd
import matplotlib.pyplot as plt

# Read the organized data
df = pd.read_csv('S7_organized_summary.csv')

# Plot
plt.figure(figsize=(10, 6))

for test in df['Test'].unique():
    subset = df[df['Test'] == test]
    plt.errorbar(subset['Frequency_Hz'], subset['Number_of_Nodes'],
                xerr=subset['Error_Freq_Hz'], label=test, marker='o')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Number of Nodes')
plt.title('Number of Nodes vs Frequency for Standing Wave Experiments')
plt.legend()
plt.grid(True)
plt.savefig('S7_nodes_vs_frequency.png', dpi=300, bbox_inches='tight')
plt.show()