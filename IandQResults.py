import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File path for the results Excel file
results_file_path = r'C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ResultsSpreadsheets\Results_311024.xlsx'

# Load the results data from the Excel file
results = pd.read_excel(results_file_path)

plt.figure(figsize=(10, 6))

# Plot scaled_inphase and scaled_quadrature vs latitude
plt.plot(results['latitude'], results['scaled_inphase'], label='Inphase', color="#8EA8FC")
plt.plot(results['latitude'], results['scaled_quadrature'], label='Quadrature', color="#32DFAB")


plt.xlabel('Latitude', fontsize=14)
plt.ylabel('Signal Response (ppm)', fontsize=14)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ticklabel_format(style='plain', axis='x')  # <-- Add this line
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))  # <-- And this for no decimals
plt.tight_layout()
plt.grid(True)
plt.show()