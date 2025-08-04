import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
earlier_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\HaasMethod\August_Earlier_Haas.xlsx"
curves_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\ApparentIce_noSIPL_curves.xlsx"
output_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\HaasMethod\EarlierHaasMethod_ApparentIceThickness.xlsx"

# Load data
df_haas = pd.read_excel(earlier_path, sheet_name="EarlierHaasMethodPoints")
df_curves = pd.read_excel(curves_path, sheet_name='Sheet1')  # Use the first sheet or specify if needed

# Function for inverse distance weighting
def idw(x, x1, x2, y1, y2):
    d1 = abs(x - x1)
    d2 = abs(x - x2)
    if d1 == 0 and d2 == 0:
        return (y1 + y2) / 2
    elif d1 == 0:
        return y1
    elif d2 == 0:
        return y2
    else:
        w1, w2 = 1/d1, 1/d2
        return (w1 * y1 + w2 * y2) / (w1 + w2)

# For I measured
apparent_I = []
for i_val in df_haas['I scaled']:
    diffs = np.abs(df_curves['I'] - i_val)
    idxs = diffs.nsmallest(2).index
    i1, i2 = df_curves.loc[idxs[0], 'I'], df_curves.loc[idxs[1], 'I']
    t1, t2 = df_curves.loc[idxs[0], 'Apparent Ice Thickness'], df_curves.loc[idxs[1], 'Apparent Ice Thickness']
    apparent_I.append(idw(i_val, i1, i2, t1, t2))

# For Q measured
apparent_Q = []
for q_val in df_haas['Q scaled']:
    diffs = np.abs(df_curves['Q'] - q_val)
    idxs = diffs.nsmallest(2).index
    q1, q2 = df_curves.loc[idxs[0], 'Q'], df_curves.loc[idxs[1], 'Q']
    t1, t2 = df_curves.loc[idxs[0], 'Apparent Ice Thickness'], df_curves.loc[idxs[1], 'Apparent Ice Thickness']
    apparent_Q.append(idw(q_val, q1, q2, t1, t2))

# # Save to Excel
# output_df = pd.DataFrame({
#     'I scaled': df_haas['I scaled'],
#     'Apparent Ice Thickness (I)': apparent_I,
#     'Q scaled': df_haas['Q scaled'],
#     'Apparent Ice Thickness (Q)': apparent_Q
# })

# # Calculate alpha for each point
# # Avoid division by zero by using np.where
# sipl_thickness = df_haas['SIPL thickness (m)']
# numerator = output_df['Apparent Ice Thickness (I)'] - output_df['Apparent Ice Thickness (Q)']
# alpha = np.where(sipl_thickness != 0, numerator / sipl_thickness, np.nan)

# output_df['alpha'] = alpha
# output_df.to_excel(output_path, index=False)
# print(f"Saved to {output_path}")

# Plot Apparent Ice Thickness vs Q
plt.figure(figsize=(7, 5))
plt.plot(df_curves['Q'], df_curves['Apparent Ice Thickness'], 'o-', label='Apparent Ice Thickness vs Q', zorder=1)
plt.scatter(df_haas['Q scaled'], df_haas['Apparent Ice Thickness (Q)'], color='red', label='scaled point measurements', zorder=2)
plt.xlabel("Quadrature (x10^4 ppm)")
plt.ylabel("Apparent Ice Thickness (m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot Apparent Ice Thickness vs I
plt.figure(figsize=(7, 5))
plt.plot(df_curves['I'], df_curves['Apparent Ice Thickness'], 'o-', color='green', label='Apparent Ice Thickness vs I', zorder =1)
plt.scatter(df_haas['I scaled'], df_haas['Apparent Ice Thickness (I)'], color='red', label='scaled point measurements', zorder=2)
plt.xlabel("Inphase (x10^4 ppm)")
plt.ylabel("Apparent Ice Thickness (m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
