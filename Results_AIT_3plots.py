import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File path to the Excel files
# excel_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\ArrayB_AIT_NoOutliers.xlsx"
excel_path = r'C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\Repeat_HCP_AITResults\SCALED_results_HCP\NoOutliers_Scaled_HCP_AITResults\NoOutliers_Results_ArrayB.xlsx'
aug_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\August_Earlier.xlsx"

# Load the main data and the August_Earlier data from the specified sheet
df = pd.read_excel(excel_path)
df_aug = pd.read_excel(aug_path, sheet_name="ArrayB")

# Sort by latitude just in case
df = df.sort_values('latitude').reset_index(drop=True)

# Setting a threshold for a "large gap" in latitude
lat_diff = df['latitude'].diff().abs()
threshold = 0.01  

# Find indices where the gap is too large
breaks = lat_diff > threshold
segment_indices = np.where(breaks)[0]

plt.figure(figsize=(10, 6))

# Split into segments and plot each
start = 0
for idx in segment_indices:
    plt.plot(
        df['latitude'].iloc[start:idx],
        df['Apparent Ice Thickness (IDW)'].iloc[start:idx],
        color="#B088FF", marker='o', linestyle='-', label='AIT Array B' if start == 0 else "", zorder=2
    )
    start = idx
# Plot the last segment
plt.plot(
    df['latitude'].iloc[start:],
    df['Apparent Ice Thickness (IDW)'].iloc[start:],
    color="#B088FF", marker='o', linestyle='-', label='' if start > 0 else 'AIT Array B', zorder=2
)

# Plot CI thickness vs latitude with diamonds
if 'latitude' in df_aug.columns and 'CI thickness' in df_aug.columns:
    plt.scatter(df_aug['latitude'], df_aug['CI thickness'], color="#0974CC", marker='D', s=60, label='CI Point Measurements', zorder=5)

# Plot SIPL (CI + SIPL) vs latitude with diamonds
if all(col in df_aug.columns for col in ['latitude', 'CI thickness', 'SIPL thickness']):
    sipl_sum = df_aug['CI thickness'] + df_aug['SIPL thickness']
    plt.scatter(df_aug['latitude'], sipl_sum, color="#C90C02", marker='D', s=60, label='SIPL Point Measurements', zorder=6)

# --- Add lines of best fit and calculate slopes ---

# 1. Apparent Ice Thickness
x1 = df['latitude']
y1 = df['Apparent Ice Thickness (IDW)']
coeffs1 = np.polyfit(x1, y1, 1)
fit1 = np.polyval(coeffs1, x1)
slope1 = coeffs1[0]
plt.plot(
    x1, fit1,
    color="#4B016D", linestyle='--',
    label=f'Best Fit: AIT',
    zorder=3
)

# 2. CI Point Measurements
slope2 = None
if 'latitude' in df_aug.columns and 'CI thickness' in df_aug.columns:
    x2 = df_aug['latitude']
    y2 = df_aug['CI thickness']
    if len(x2) > 1:
        coeffs2 = np.polyfit(x2, y2, 1)
        fit2 = np.polyval(coeffs2, x2)
        slope2 = coeffs2[0]
        plt.plot(
            x2, fit2,
            color="#156BB3", linestyle='--',
            label=f'Best Fit: CI',
            zorder=7
        )

# 3. SIPL Point Measurements
slope3 = None
if all(col in df_aug.columns for col in ['latitude', 'CI thickness', 'SIPL thickness']):
    x3 = df_aug['latitude']
    y3 = df_aug['CI thickness'] + df_aug['SIPL thickness']
    if len(x3) > 1:
        coeffs3 = np.polyfit(x3, y3, 1)
        fit3 = np.polyval(coeffs3, x3)
        slope3 = coeffs3[0]
        plt.plot(
            x3, fit3,
            color="#DA0E03", linestyle='--',
            label=f'Best Fit: SIPL',
            zorder=8
        )

# # --- Add slope labels at the top of the plot ---
# ymax = plt.ylim()[0]  # Top of the plot (since y-axis is reversed)
# xmid = np.mean(plt.xlim())

# label_y = ymax + 0.2  # Slightly above the top

# labels = [
#     f"Slope (Apparent): {slope1:.3f}",
#     f"Slope (CI): {slope2:.3f}" if slope2 is not None else "",
#     f"Slope (SIPL): {slope3:.3f}" if slope3 is not None else ""
# ]
# label_text = "   ".join([l for l in labels if l])

# plt.text(xmid, label_y, label_text, ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.xlabel("Latitude", fontsize=14)
plt.ylabel("Depth (m)", fontsize=14)
plt.ylim(4, 0)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.tight_layout()

# Get handles and labels from the current axes
handles, labels = plt.gca().get_legend_handles_labels()

# Specify your desired order (example below)
desired_order = [
    'AIT Array B',
    'Best Fit: AIT',
    'CI Point Measurements',
    'Best Fit: CI',
    'SIPL Point Measurements',
    'Best Fit: SIPL'
]

# Reorder handles and labels
ordered_handles = [handles[labels.index(l)] for l in desired_order if l in labels]
ordered_labels = [l for l in desired_order if l in labels]

plt.legend(ordered_handles, ordered_labels, fontsize=14)


plt.show()





# ## Script for plotting thickness against distance from a reference point

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # File path to the Excel files
# excel_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\281024_AIT_NoOutliers.xlsx"
# aug_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\August_Earlier.xlsx"

# # Load the main data and the August_Earlier data from the specified sheet
# df = pd.read_excel(excel_path)
# df_aug = pd.read_excel(aug_path, sheet_name="281024_A")

# # Sort by latitude just in case
# df = df.sort_values('latitude').reset_index(drop=True)

# # Convert latitude to distance (km) from the minimum latitude
# reference_lat = df['latitude'].min()
# df['distance_km'] = (df['latitude'] - reference_lat) * 111.32

# # Setting a threshold for a "large gap" in distance (km)
# dist_diff = df['distance_km'].diff().abs()
# threshold = 0.01 * 111.32  # equivalent to your latitude threshold, in km

# # Find indices where the gap is too large
# breaks = dist_diff > threshold
# segment_indices = np.where(breaks)[0]

# # Split into segments and plot each
# start = 0
# plt.figure(figsize=(10, 6))
# for idx in segment_indices:
#     plt.plot(
#         df['distance_km'].iloc[start:idx],
#         df['Apparent Ice Thickness (IDW)'].iloc[start:idx],
#         color="#B088FF", marker='o', linestyle='-', label='Apparent Ice Thickness' if start == 0 else "", zorder=2
#     )
#     start = idx
# # Plot the last segment
# plt.plot(
#     df['distance_km'].iloc[start:],
#     df['Apparent Ice Thickness (IDW)'].iloc[start:],
#     color="#B088FF", marker='o', linestyle='-', label='' if start > 0 else 'Apparent Ice Thickness', zorder=2
# )

# # Convert point measurements to distance as well
# if 'latitude' in df_aug.columns:
#     df_aug['distance_km'] = (df_aug['latitude'] - reference_lat) * 111.32

# # Plot CI thickness vs distance with diamonds (negative for depth)
# if 'distance_km' in df_aug.columns and 'CI thickness' in df_aug.columns:
#     plt.scatter(df_aug['distance_km'], df_aug['CI thickness'], color="#0974CC", marker='D', s=60, label='CI Point Measurements', zorder=5)

# # Plot SIPL (CI + SIPL) vs distance with diamonds (negative for depth)
# if all(col in df_aug.columns for col in ['distance_km', 'CI thickness', 'SIPL thickness']):
#     sipl_sum = df_aug['CI thickness'] + df_aug['SIPL thickness']
#     plt.scatter(df_aug['distance_km'], sipl_sum, color="#C90C02", marker='D', s=60, label='SIPL Point Measurements', zorder=6)


# # Load the HaasMethod results
# haas_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\HaasMethod\PlottingCIandSIPLfromAIT\281024_EarlierHaasMethod_ApparentIceThickness.xlsx"
# df_haas = pd.read_excel(haas_path)

# # Convert latitude to distance (km) using the same reference
# if 'latitude' in df_haas.columns:
#     df_haas['distance_km'] = (df_haas['latitude'] - reference_lat) * 111.32
# else:
#     # If latitude is not present, you need to add it to the HaasMethod file
#     raise ValueError("The HaasMethod file must contain a 'latitude' column for distance conversion.")

# plt.xlabel("Distance (km)", fontsize=14)
# plt.ylabel("Depth (m)", fontsize=14)
# plt.ylim(4, 0)
# plt.legend(fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.grid(True)
# plt.tight_layout()
# plt.show()




## Plotting all three AIT datasets with point measurements

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths
aug_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\August_Earlier.xlsx"
excel_path1 = r'C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\Repeat_HCP_AITResults\SCALED_results_HCP\NoOutliers_Scaled_HCP_AITResults\NoOutliers_Results_251024.xlsx'
excel_path2 = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\Repeat_HCP_AITResults\SCALED_results_HCP\NoOutliers_Scaled_HCP_AITResults\NoOutliers_Results_281024.xlsx"
excel_path3 = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\Repeat_HCP_AITResults\SCALED_results_HCP\NoOutliers_Scaled_HCP_AITResults\NoOutliers_Results_041124.xlsx"

# Load data
df1 = pd.read_excel(excel_path1).sort_values('latitude').reset_index(drop=True)
df2 = pd.read_excel(excel_path2).sort_values('latitude').reset_index(drop=True)
df3 = pd.read_excel(excel_path3).sort_values('latitude').reset_index(drop=True)
# df3 = df3[df3['latitude'] >= -77.869].reset_index(drop=True)  # Adjusting the values included in the plot
df_aug1 = pd.read_excel(aug_path, sheet_name="251024_A")
df_aug2 = pd.read_excel(aug_path, sheet_name="281024_A")
df_aug3 = pd.read_excel(aug_path, sheet_name="041124_A")


# Define colors for AIT best fit lines all black
ait_colors = ["#000000", "#000000", "#000000"]  

# Plot setup
fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True, sharey=True)

# Common settings
xlim = (-77.90, -77.82)
xticks = np.linspace(-77.90, -77.82, 9)
xtick_labels = [f"{x:.2f}" if i % 2 == 0 else "" for i, x in enumerate(xticks)]
plt.xticks(xticks, xtick_labels, fontsize=14)
ylim = (4, 0)
yticks = np.linspace(4, 0, 5)
ytick_labels = [f"{y:.1f}" if i % 2 == 0 else "" for i, y in enumerate(yticks)]
threshold = 0.01

# For collecting handles for the main legend
main_legend_handles = []
main_legend_labels = []

# --- First subplot ---
ax0 = axes[0]
ax0.set_yticks(yticks)
ax0.set_yticklabels(ytick_labels, fontsize=14)
lat_diff1 = df1['latitude'].diff().abs()
breaks1 = lat_diff1 > threshold
segment_indices1 = np.where(breaks1)[0]
start = 0
for idx in segment_indices1:
    h_ait, = ax0.plot(
        df1['latitude'].iloc[start:idx],
        df1['Apparent Ice Thickness (IDW)'].iloc[start:idx],
        color="#B088FF", marker='o', linestyle='-', label='AIT 25/10/24', zorder=2
    )
    start = idx
h_ait, = ax0.plot(
    df1['latitude'].iloc[start:],
    df1['Apparent Ice Thickness (IDW)'].iloc[start:],
    color="#B088FF", marker='o', linestyle='-', label='AIT 25/10/24', zorder=2
)

# Line of best fit for AIT (dark purple)
if len(df1['latitude']) > 1:
    coeffs = np.polyfit(df1['latitude'], df1['Apparent Ice Thickness (IDW)'], 1)
    fit = np.polyval(coeffs, df1['latitude'])
    h_ait_fit, = ax0.plot(df1['latitude'], fit, color=ait_colors[0], linestyle='--', label='Best Fit AIT', zorder=3)

# CI and SIPL points and best fit lines
h_ci = h_sipl = h_fit_ci = h_fit_sipl = None
if 'latitude' in df_aug1.columns and 'CI thickness' in df_aug1.columns:
    h_ci = ax0.scatter(df_aug1['latitude'], df_aug1['CI thickness'], color="#0974CC", marker='D', s=60, label='CI Points', zorder=5)
    # Best fit for CI
    if len(df_aug1['latitude']) > 1:
        coeffs_ci = np.polyfit(df_aug1['latitude'], df_aug1['CI thickness'], 1)
        fit_ci = np.polyval(coeffs_ci, df_aug1['latitude'])
        h_fit_ci, = ax0.plot(df_aug1['latitude'], fit_ci, color="#0974CC", linestyle='--', label='Best Fit CI', zorder=6)
if all(col in df_aug1.columns for col in ['latitude', 'CI thickness', 'SIPL thickness']):
    sipl_sum = df_aug1['CI thickness'] + df_aug1['SIPL thickness']
    h_sipl = ax0.scatter(df_aug1['latitude'], sipl_sum, color="#C90C02", marker='D', s=60, label='SIPL Points', zorder=7)
    # Best fit for SIPL
    if len(df_aug1['latitude']) > 1:
        coeffs_sipl = np.polyfit(df_aug1['latitude'], sipl_sum, 1)
        fit_sipl = np.polyval(coeffs_sipl, df_aug1['latitude'])
        h_fit_sipl, = ax0.plot(df_aug1['latitude'], fit_sipl, color="#C90C02", linestyle='--', label='Best Fit SIPL', zorder=8)

# Subplot legend (AIT only)
ax0.legend([h_ait, h_ait_fit], ['AIT 25/10/24', 'Best Fit AIT'], loc='upper left', fontsize=11, frameon=True, facecolor='white', edgecolor='black')

# Collect CI/SIPL handles for main legend
if h_ci is not None and h_fit_ci is not None:
    main_legend_handles.append(h_ci)
    main_legend_labels.append('CI Points')
    main_legend_handles.append(h_fit_ci)
    main_legend_labels.append('Best Fit CI')
if h_sipl is not None and h_fit_sipl is not None:
    main_legend_handles.append(h_sipl)
    main_legend_labels.append('SIPL Points')
    main_legend_handles.append(h_fit_sipl)
    main_legend_labels.append('Best Fit SIPL')

ax0.set_ylabel("Depth (m)", fontsize=14)
ax0.set_xlim(xlim)
ax0.set_ylim(ylim)
ax0.grid(True)

# --- Second subplot ---
ax1 = axes[1]
ax1.set_yticks(yticks)
ax1.set_yticklabels(ytick_labels, fontsize=14)
lat_diff2 = df2['latitude'].diff().abs()
breaks2 = lat_diff2 > threshold
segment_indices2 = np.where(breaks2)[0]
start = 0
for idx in segment_indices2:
    h_ait, = ax1.plot(
        df2['latitude'].iloc[start:idx],
        df2['Apparent Ice Thickness (IDW)'].iloc[start:idx],
        color="#B088FF", marker='o', linestyle='-', label='AIT 28/10/24', zorder=2
    )
    start = idx
h_ait, = ax1.plot(
    df2['latitude'].iloc[start:],
    df2['Apparent Ice Thickness (IDW)'].iloc[start:],
    color="#B088FF", marker='o', linestyle='-', label='AIT 28/10/24', zorder=2
)

# Line of best fit for AIT (medium purple)
if len(df2['latitude']) > 1:
    coeffs = np.polyfit(df2['latitude'], df2['Apparent Ice Thickness (IDW)'], 1)
    fit = np.polyval(coeffs, df2['latitude'])
    h_ait_fit, = ax1.plot(df2['latitude'], fit, color=ait_colors[1], linestyle='--', label='Best Fit AIT', zorder=3)

# CI and SIPL points and best fit lines
h_ci = h_sipl = h_fit_ci = h_fit_sipl = None
if 'latitude' in df_aug2.columns and 'CI thickness' in df_aug2.columns:
    h_ci = ax1.scatter(df_aug2['latitude'], df_aug2['CI thickness'], color="#0974CC", marker='D', s=60, label='CI Points', zorder=5)
    # Best fit for CI
    if len(df_aug2['latitude']) > 1:
        coeffs_ci = np.polyfit(df_aug2['latitude'], df_aug2['CI thickness'], 1)
        fit_ci = np.polyval(coeffs_ci, df_aug2['latitude'])
        h_fit_ci, = ax1.plot(df_aug2['latitude'], fit_ci, color="#0974CC", linestyle='--', label='Best Fit CI', zorder=6)
if all(col in df_aug2.columns for col in ['latitude', 'CI thickness', 'SIPL thickness']):
    sipl_sum = df_aug2['CI thickness'] + df_aug2['SIPL thickness']
    h_sipl = ax1.scatter(df_aug2['latitude'], sipl_sum, color="#C90C02", marker='D', s=60, label='SIPL Points', zorder=7)
    # Best fit for SIPL
    if len(df_aug2['latitude']) > 1:
        coeffs_sipl = np.polyfit(df_aug2['latitude'], sipl_sum, 1)
        fit_sipl = np.polyval(coeffs_sipl, df_aug2['latitude'])
        h_fit_sipl, = ax1.plot(df_aug2['latitude'], fit_sipl, color="#C90C02", linestyle='--', label='Best Fit SIPL', zorder=8)

# Subplot legend (AIT only)
#ax1.legend([h_ait, h_ait_fit], ['AIT 28/10/24', 'Best Fit AIT'], loc='upper left', fontsize=11, frameon=True, facecolor='white', edgecolor='black')
ait_legend = ax1.legend([h_ait, h_ait_fit], ['AIT 28/10/24', 'Best Fit AIT'], loc='upper left', fontsize=11, frameon=True, facecolor='white', edgecolor='black')
ax1.add_artist(ait_legend)

# Collect CI/SIPL handles for main legend
if h_ci is not None and h_fit_ci is not None:
    main_legend_handles.append(h_ci)
    main_legend_labels.append('CI Points')
    main_legend_handles.append(h_fit_ci)
    main_legend_labels.append('Best Fit CI')
if h_sipl is not None and h_fit_sipl is not None:
    main_legend_handles.append(h_sipl)
    main_legend_labels.append('SIPL Points')
    main_legend_handles.append(h_fit_sipl)
    main_legend_labels.append('Best Fit SIPL')

ax1.set_ylabel("Depth (m)", fontsize=14)
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.grid(True)

# --- Third subplot ---
ax2 = axes[2]
ax2.set_yticks(yticks)
ax2.set_yticklabels(ytick_labels, fontsize=14)
lat_diff3 = df3['latitude'].diff().abs()
breaks3 = lat_diff3 > threshold
segment_indices3 = np.where(breaks3)[0]
start = 0
for idx in segment_indices3:
    h_ait, = ax2.plot(
        df3['latitude'].iloc[start:idx],
        df3['Apparent Ice Thickness (IDW)'].iloc[start:idx],
        color="#B088FF", marker='o', linestyle='-', label='AIT 04/11/24', zorder=2
    )
    start = idx
h_ait, = ax2.plot(
    df3['latitude'].iloc[start:],
    df3['Apparent Ice Thickness (IDW)'].iloc[start:],
    color="#B088FF", marker='o', linestyle='-', label='AIT 04/11/24', zorder=2
)

# Only one line of best fit: fit using restricted latitude range, plot across all data
fit_mask = (df3['latitude'] >= -77.869) & (df3['latitude'] <= -77.835)
x_fit = df3.loc[fit_mask, 'latitude']
y_fit = df3.loc[fit_mask, 'Apparent Ice Thickness (IDW)']
if len(x_fit) > 1:
    coeffs = np.polyfit(x_fit, y_fit, 1)
    # Evaluate the fit across the full latitude range
    fit_full = np.polyval(coeffs, df3['latitude'])
    h_ait_fit, = ax2.plot(df3['latitude'], fit_full, color=ait_colors[2], linestyle='--', label='Best Fit AIT (restricted)', zorder=3)

# CI and SIPL points and best fit lines
h_ci = h_sipl = h_fit_ci = h_fit_sipl = None
if 'latitude' in df_aug3.columns and 'CI thickness' in df_aug3.columns:
    h_ci = ax2.scatter(df_aug3['latitude'], df_aug3['CI thickness'], color="#006CC4", marker='D', s=60, label='CI Points', zorder=5)
    # Best fit for CI
    if len(df_aug3['latitude']) > 1:
        coeffs_ci = np.polyfit(df_aug3['latitude'], df_aug3['CI thickness'], 1)
        fit_ci = np.polyval(coeffs_ci, df_aug3['latitude'])
        h_fit_ci, = ax2.plot(df_aug3['latitude'], fit_ci, color="#006CC4", linestyle='--', label='Best Fit CI', zorder=6)
if all(col in df_aug3.columns for col in ['latitude', 'CI thickness', 'SIPL thickness']):
    sipl_sum = df_aug3['CI thickness'] + df_aug3['SIPL thickness']
    h_sipl = ax2.scatter(df_aug3['latitude'], sipl_sum, color="#C90C02", marker='D', s=60, label='SIPL Points', zorder=7)
    # Best fit for SIPL
    if len(df_aug3['latitude']) > 1:
        coeffs_sipl = np.polyfit(df_aug3['latitude'], sipl_sum, 1)
        fit_sipl = np.polyval(coeffs_sipl, df_aug3['latitude'])
        h_fit_sipl, = ax2.plot(df_aug3['latitude'], fit_sipl, color="#C90C02", linestyle='--', label='Best Fit SIPL', zorder=8)

# Subplot legend (AIT only)
ax2.legend([h_ait, h_ait_fit], ['AIT 04/11/24', 'Best Fit AIT'], loc='upper left', fontsize=11, frameon=True, facecolor='white', edgecolor='black')

# Collect CI/SIPL handles for main legend
if h_ci is not None and h_fit_ci is not None:
    main_legend_handles.append(h_ci)
    main_legend_labels.append('CI Points')
    main_legend_handles.append(h_fit_ci)
    main_legend_labels.append('Best Fit CI')
if h_sipl is not None and h_fit_sipl is not None:
    main_legend_handles.append(h_sipl)
    main_legend_labels.append('SIPL Points')
    main_legend_handles.append(h_fit_sipl)
    main_legend_labels.append('Best Fit SIPL')

ax2.set_ylabel("Depth (m)", fontsize=14)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.grid(True)

# Remove duplicates from main legend
unique_labels = []
unique_handles = []
for h, l in zip(main_legend_handles, main_legend_labels):
    if l not in unique_labels:
        unique_labels.append(l)
        unique_handles.append(h)

# X-axis label and ticks for all
axes[-1].set_xlabel("Latitude", fontsize=14)
plt.xticks(xticks, xtick_labels, fontsize=14)
plt.tight_layout()

# Main legend (center right on middle plot)
# axes[1].legend(unique_handles, unique_labels, loc='center right', fontsize=11, frameon=True, facecolor='white', edgecolor='black')
ax1.legend(unique_handles, unique_labels, loc='center right', fontsize=11, frameon=True, facecolor='white', edgecolor='black')
plt.show()

