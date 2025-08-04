import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# File path to the Excel file
excel_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\HaasMethod\HaasMethodModelCurves_HCP.xlsx"
# csv_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ResultsSpreadsheets\Results_311024.csv"

# Load the sheet names
sheet_names = pd.ExcelFile(excel_path).sheet_names

plt.figure(figsize=(10, 6))

# Use a wider spectrum colormap
cmap = plt.cm.get_cmap('tab20', len(sheet_names))

for idx, sheet in enumerate(sheet_names):
    df = pd.read_excel(excel_path, sheet_name=sheet)
    if all(col in df.columns for col in ['SIPL thickness', 'I', 'Q']):
        color = cmap(idx)
        plt.plot(df['SIPL thickness'], df['I'], color=color, label=f'{sheet} mS/m: I (solid), Q (dashed)')
        plt.plot(df['SIPL thickness'], df['Q'], color=color, linestyle='--')  # No label for Q

plt.xlabel('SIPL Thickness (m)')
plt.ylabel('I & Q')
plt.title('Inphase and Quadrature vs SIPL Thickness for varying SIPL Conductivity')
plt.legend(fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# Define your custom color mapping for each conductivity
custom_colors = {
    0:    "#FF0000",   # red
    300:  "#FF6666",   # light red
    600:  "#FF9900",   # orange
    800:  "#FFC000",   # amber
    900:  "#FAFA38",   # yellow
    1200: "#99FF99",   # light green
    1500: "#00FFFF",   # aqua
    1800: "#66CCFF",   # light blue
    2100: "#0051FF",   # blue
    2400: "#070797",   # dark blue
    2700: "#800080",   # purple
}

# Get conductivities as integers from sheet names
conductivities = [int(s) for s in sheet_names]
sorted_indices = np.argsort(conductivities)
sorted_sheets = [sheet_names[i] for i in sorted_indices]
sorted_conductivities = [conductivities[i] for i in sorted_indices]

# --- Plot for I only ---
plt.figure(figsize=(10, 6))
for sheet, cond in zip(sorted_sheets, sorted_conductivities):
    df = pd.read_excel(excel_path, sheet_name=sheet)
    if all(col in df.columns for col in ['SIPL thickness', 'I']):
        color = custom_colors.get(cond, "#888888")  # fallback to gray if not found
        plt.plot(df['SIPL thickness'], df['I'], color=color, label=f'{cond}')
plt.xlabel('SIPL Thickness (m)')
plt.ylabel('Inphase (ppm)')
# plt.title('Inphase vs SIPL Thickness for varying SIPL Conductivity')
plt.legend(title="SIPL Conductivity (mS/m)", fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot for Q only ---
plt.figure(figsize=(10, 6))
for sheet, cond in zip(sorted_sheets, sorted_conductivities):
    df = pd.read_excel(excel_path, sheet_name=sheet)
    if all(col in df.columns for col in ['SIPL thickness', 'Q']):
        color = custom_colors.get(cond, "#888888")
        plt.plot(df['SIPL thickness'], df['Q'], color=color, label=f'{cond}')
plt.xlabel('SIPL Thickness (m)')
plt.ylabel('Quadrature (ppm)')
# plt.title('Quadrature vs SIPL Thickness for varying SIPL Conductivity')
plt.legend(title="SIPL Conductivity (mS/m)", fontsize=8, ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()
