import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# File path to the Excel files
# HCP_noSIPL_curves = r"C:\Users\esk22\Downloads\REPEAT_NOSIPLCurves.xlsx"
HCP_noSIPL_curves_cut = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\REPEAT_NOSIPLCurves_cut.xlsx"
VCP_noSIPL_curves = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\ApparentIce_noSIPL_curves.xlsx"
excel_path_aug = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\August_Earlier.xlsx"
csv_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\AugustIce_FilteredDatasets\041124finalT_A_filtered.csv"

# Plotting the model curves and fitting August data 
plt.figure(figsize=(10, 6))

# --- Plot all HCP model curves ---
hcp_xls = pd.ExcelFile(HCP_noSIPL_curves_cut)
hcp_sheet_names = hcp_xls.sheet_names
for sheet in hcp_sheet_names:
    df = pd.read_excel(HCP_noSIPL_curves_cut, sheet_name=sheet)
    if 'Q' in df.columns and 'Apparent Ice Thickness' in df.columns:
        plt.plot(df['Q'], df['Apparent Ice Thickness'], color='blue', alpha=0.7)
# Add a single proxy line for the legend
plt.plot([], [], color='blue', label='HCP forward model no SIPL')

# --- Plot all VCP model curves ---
vcp_xls = pd.ExcelFile(VCP_noSIPL_curves)
vcp_sheet_names = vcp_xls.sheet_names
for sheet in vcp_sheet_names:
    df = pd.read_excel(VCP_noSIPL_curves, sheet_name=sheet)
    if 'Q' in df.columns and 'Apparent Ice Thickness' in df.columns:
        plt.plot(df['Q'], df['Apparent Ice Thickness'], color='green', alpha=0.7)
# Add a single proxy line for the legend
plt.plot([], [], color='green', label='VCP forward model no SIPL')

# Load and plot the August measured data
df_aug = pd.read_excel(excel_path_aug, sheet_name="Aug precise")
if 'Q_scaled_HCP' in df_aug.columns and 'Ice + Snow thickness' in df_aug.columns:
    plt.scatter(df_aug['Q_scaled_HCP'], df_aug['Ice + Snow thickness'], color='red', marker='o', s=50, label='August ice point measurements')

plt.xlabel("Quadrature (x10^4 ppm)")
plt.ylabel("Apparent Ice Thickness (m)")
plt.legend()
plt.xlim(0, 25)
plt.grid(True)
plt.tight_layout()
plt.show()


# Load the model curve 
df_curve = pd.read_excel(HCP_noSIPL_curves_cut, sheet_name=0) 

# Load the CSV data
df_csv = pd.read_csv(csv_path)

# Prepare output list
apparent_thickness_idw = []

# For each Quadrature value in the CSV, find the two closest Qs in the model and apply inverse distance weighting
for q_val in df_csv['Quad_scaled']:
    # Compute absolute differences
    diffs = np.abs(df_curve['Q'] - q_val)
    # Get indices of the two closest points
    idxs = diffs.nsmallest(2).index
    q1, q2 = df_curve.loc[idxs[0], 'Q'], df_curve.loc[idxs[1], 'Q']
    t1, t2 = df_curve.loc[idxs[0], 'Apparent Ice Thickness'], df_curve.loc[idxs[1], 'Apparent Ice Thickness']
    d1, d2 = abs(q_val - q1), abs(q_val - q2)
    # Avoid division by zero (if exact match)
    if d1 == 0 and d2 == 0:
        thickness = (t1 + t2) / 2
    elif d1 == 0:
        thickness = t1
    elif d2 == 0:
        thickness = t2
    else:
        w1, w2 = 1/d1, 1/d2
        thickness = (w1 * t1 + w2 * t2) / (w1 + w2)
    apparent_thickness_idw.append(thickness)

# Add the interpolated thickness to the dataframe
df_csv['Apparent Ice Thickness (IDW)'] = apparent_thickness_idw

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df_curve['Q'], df_curve['Apparent Ice Thickness'], color='blue', alpha=0.7, label='forward model no SIPL')
plt.scatter(df_csv['Quad_scaled'], df_csv['Apparent Ice Thickness (IDW)'], color='green', marker='o', s=50, label='IDW thickness results 04/11/24')
plt.xlabel("Quadrature (ppm)")
plt.ylabel("Apparent Ice Thickness (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Prepare DataFrame with Quadrature and corresponding Apparent Ice Thickness (IDW)
output_df = pd.DataFrame({
    'Quad_scaled': df_csv['Quad_scaled'],
    'longitude': df_csv['longitude'],
    'latitude': df_csv['latitude'],
    'Apparent Ice Thickness (IDW)': df_csv['Apparent Ice Thickness (IDW)']
})

# Output path
#output_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\Repeat_HCP_AITResults\SCALED_results_HCP\AIT_HCP_scaled.xlsx"
output_df.to_excel(output_path, index=False)
print(f"Saved to {output_path}")
