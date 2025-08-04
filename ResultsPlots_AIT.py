## plotting apparent ice thickness results for August, with contents of most south and most north points. 

import pandas as pd
import matplotlib.pyplot as plt

# File path to the Excel files
excel_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\Repeat_HCP_AITResults\results_cut\AIT_Results_HCP_cut_ArrayB.xlsx"
aug_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\August_Earlier.xlsx"

# Load the main data
df = pd.read_excel(excel_path)

# Find the indices of the southernmost and northernmost points
south_idx = df['latitude'].idxmin()
north_idx = df['latitude'].idxmax()
start_idx = min(south_idx, north_idx)
end_idx = max(south_idx, north_idx)

# # Slice the DataFrame to include only data between (and including) these points
# df_cut = df.loc[start_idx:end_idx].reset_index(drop=True)

# Load the August_Earlier data from the specified sheet
df_aug = pd.read_excel(aug_path, sheet_name="ArrayB")

# # Remove all values of apparent ice thickness between 0 and -1.5 m of depth
# df_cut = df_cut[( -df_cut['Apparent Ice Thickness (IDW)'] < -1.5 ) | ( -df_cut['Apparent Ice Thickness (IDW)'] > 0 )]

plt.figure(figsize=(10, 6))
plt.plot(df['latitude'], -df['Apparent Ice Thickness (IDW)'], marker='o', linestyle='-', color="#B088FF", label='Apparent Ice Thickness')

# ...rest of your code...
# Plot CI thickness vs latitude with diamonds (negative for depth)
if 'latitude' in df_aug.columns and 'CI thickness' in df_aug.columns:
    plt.scatter(df_aug['latitude'], -df_aug['CI thickness'], color="#0974CC", marker='D', s=60, label='CI Point Measurements', zorder=5)

# Plot SIPL (CI + SIPL) vs latitude with diamonds (negative for depth)
if all(col in df_aug.columns for col in ['latitude', 'CI thickness', 'SIPL thickness']):
    sipl_sum = df_aug['CI thickness'] + df_aug['SIPL thickness']
    plt.scatter(df_aug['latitude'], -sipl_sum, color="#C90C02", marker='D', s=60, label='SIPL Point Measurements', zorder=6)

plt.xlabel("Latitude", fontsize=14)
plt.ylabel("Depth (m)", fontsize=14)
plt.ylim(-4, 0)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()



