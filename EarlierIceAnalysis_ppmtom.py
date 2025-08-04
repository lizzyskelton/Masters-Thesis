## Earlier Ice analysis script 

# # Earlier Ice I and Q measurement data to CI and SIPL thickness measurements using the forward model and inversion techniques. 
# Note: I would run this script in parts and have sections commented out. the parts not commented are the parts I would typically run (but not at once). 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

print("Script is running...")

# Load the Excel file
file_path = '/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/HCPEarlierIce800_Trendlines.xlsx'
sheet_names = pd.ExcelFile(file_path).sheet_names

# Define the columns to be used for the x and y axes
x_column = 'I'
y_column = 'Q'
color_column = 'SIPL'

# Define the custom color map
colors = ["#40E0D0", "#00008B"]  # Bright turquoise to dark blue
cmap = LinearSegmentedColormap.from_list("turquoise_to_blue", colors)
norm = plt.Normalize(vmin=0, vmax=10)

# Plot the data
plt.figure(figsize=(10, 6))

# Loop through each sheet and plot the data
for sheet_name in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # Divide the values by 10,000
    df[x_column] = df[x_column] / 10000
    df[y_column] = df[y_column] / 10000
    # Connect the dots with a line (thin, black, and behind the dots)
    plt.plot(df[x_column], df[y_column], linestyle='-', color='black', linewidth=0.5, zorder=1)
    # Scatter plot with smaller dots
    plt.scatter(df[x_column], df[y_column], c=df[color_column], cmap=cmap, norm=norm, marker='o', s=20, zorder=2)
    # Add a thick red line for values with SIPL = 0
    df_sipl_0 = df[df[color_column] == 0]
    plt.plot(df_sipl_0[x_column], df_sipl_0[y_column], linestyle='-', color='red', linewidth=2, zorder=3)

# Add color bar
plt.colorbar(label="SIPL thickness (m)")

# Add other plot details
# plt.title("Forward Model for 'Earlier Ice'")
plt.xlabel("Inphase (x10^4 ppm)")
plt.ylabel("Quadrature (x10^4 ppm)")
plt.grid(True)

# Import data from the CSV file
csv_file_path = '/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/EarlierIce_FilteredDatasets/261024_E_filtered.csv'
csv_data = pd.read_csv(csv_file_path)

# Extract the "Inphase" and "Quadrature" columns
inphase = csv_data['Inphase']
quadrature = csv_data['Quadrature']

# Scale the data points
scaled_inphase = (2.21 * inphase) + 1.60
scaled_quadrature = (1.11 * quadrature) + 0.08

# # Create a DataFrame with the scaled values
# scaled_data = pd.DataFrame({
#     'scaled_inphase': scaled_inphase,
#     'scaled_quadrature': scaled_quadrature
# })

# # Save the DataFrame to an Excel file
# output_file_path = r'C:\Users\esk22\Downloads\Scaled.xlsx'
# scaled_data.to_excel(output_file_path, index=False)

# print(f"Scaled data saved to {output_file_path}")

# Plot the scaled data points in orange
plt.scatter(scaled_inphase, scaled_quadrature, color='darkorange', label='Scaled', marker='o', s=20, zorder=4)

# Plot the original data points in dark green
plt.scatter(inphase, quadrature, color='darkgreen', label='Original', marker='o', s=20, zorder=3)

# Add an arrow pointing right between the two groups of data points
arrow_x = (inphase.mean() + scaled_inphase.mean()) / 2  # Midpoint x-coordinate
arrow_y = (quadrature.mean() + scaled_quadrature.mean()) / 2  # Midpoint y-coordinate
plt.annotate(
   '', 
   xy=(scaled_inphase.mean(), scaled_quadrature.mean()),  # Arrowhead at scaled data
   xytext=(inphase.mean(), quadrature.mean()),  # Arrow tail at original data
   arrowprops=dict(facecolor='black', arrowstyle='->', lw=2),  # Arrow style
   zorder=5
)

# # Calculate the combined range for both groups of data
# # x_min = min(inphase.min(), scaled_inphase.min()) - 0.1
# # x_max = max(inphase.max(), scaled_inphase.max()) + 0.1
# # y_min = min(quadrature.min(), scaled_quadrature.min()) - 0.1
# # y_max = max(quadrature.max(), scaled_quadrature.max()) + 0.1

# # # Set axis limits to zoom in on the two groups of data
# # plt.xlim(x_min, x_max)
# # plt.ylim(y_min, y_max)

# # # Set axis limits to zoom in on the scaled data points
# # plt.xlim(scaled_inphase.min() - 0.1, scaled_inphase.max() + 0.1)
# # plt.ylim(scaled_quadrature.min() - 0.1, scaled_quadrature.max() + 0.1)

# Add a legend
plt.legend()

# # Save the plot as an image file
# # plt.savefig('plot.png')

# Show the plot
plt.show()       # deleted so tables can print in the terminal 
#plt.pause(5)  # Display the plot for 5 seconds



# Calculating the Ice and Snow values, first with 4 smallest distances, then inverse distance weighting, then loop through all sheets to get full transect converted. 

import pandas as pd
import numpy as np

# File paths
scaled_file_path = 'C:/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/REPEAT_scaled_Earlier/Scaled_261024_repeat.xlsx'
results_file_path = 'C:/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/REPEAT_scaled_Earlier/Results/HCPResults_261024_incasedoubled.xlsx'

# Load the scaled data
scaled_data = pd.read_excel(scaled_file_path)

# Create a list to store the results
results_list_2 = []

# Loop through each scaled_inphase and scaled_quadrature value
for index, row in scaled_data.iterrows():
    scaled_inphase = row['scaled_inphase']
    scaled_quadrature = row['scaled_quadrature']

    # Create an empty DataFrame to store results from all sheets
    all_distances = pd.DataFrame()

    # Loop through each sheet in the Excel file
    for sheet_name in sheet_names:
        # Load the data for the current sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        df['I'] = df['I'] / 10000
        df['Q'] = df['Q'] / 10000

        df['Distance'] = np.sqrt((df['I'] - scaled_inphase)**2 + (df['Q'] - scaled_quadrature)**2)
        df['Sheet Name'] = sheet_name
        all_distances = pd.concat([all_distances, df], ignore_index=True)

    closest_rows = all_distances.nsmallest(4, 'Distance')
    closest_rows['Weight'] = 1 / closest_rows['Distance']
    closest_rows['Consolidated Ice x Weight'] = closest_rows['Consolidated Ice'] * closest_rows['Weight']
    ice_snow_result = closest_rows['Consolidated Ice x Weight'].sum() / closest_rows['Weight'].sum()
    closest_rows['SIPL x Weight'] = closest_rows['SIPL'] * closest_rows['Weight']
    sipl_result = closest_rows['SIPL x Weight'].sum() / closest_rows['Weight'].sum()

    # --- Add a progress print for debugging ---
    if index % 10 == 0:
        print(f"Processed {index+1}/{len(scaled_data)} rows. Example result: CI={ice_snow_result:.3f}, SIPL={sipl_result:.3f}")

    results_list_2.append({
        'scaled_inphase': scaled_inphase,
        'scaled_quadrature': scaled_quadrature,
        'Consolidated Ice Result': ice_snow_result,
        'SIPL Result': sipl_result
    })

# Convert the results list to a DataFrame
results = pd.DataFrame(results_list_2)

# Save the results to an Excel file
results.to_excel(results_file_path, index=False)

print(f"Results saved to {results_file_path}")



# Plotting Snow + Ice and SIPL thickness results against latitude

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File path for the results Excel file
# results_file_path = r'C:/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/REPEAT_scaled_Earlier/Results/HCPResults_261024.xlsx'
results_file_path = r'C:/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/REPEAT_scaled_Earlier/Results/OutlierRemoved_HCP/HCP_Results_NoOutliers_261024.xlsx'

# Load the results data from the Excel file
results = pd.read_excel(results_file_path)

# Ensure the results DataFrame is loaded
# Plot "Consolidated Ice Result" and "SIPL Result"
plt.figure(figsize=(10, 6))


# ## Uncomment if you want to plot against distance instead of latitude

# # Choose a reference latitude (e.g., the minimum latitude)
# reference_lat = results['latitude'].min()

# # Convert latitude to distance in kilometers from the reference point
# results['distance_km'] = (results['latitude'] - reference_lat) * 111.32

# # For point measurements as well
# if all(col in results.columns for col in ['P_lat', 'P_CI', 'P_SIPL']):
#     point_data = results[['P_lat', 'P_CI', 'P_SIPL']].dropna()
#     point_data['distance_km'] = (point_data['P_lat'] - reference_lat) * 111.32

# # Now plot using distance_km instead of latitude
# plt.plot(results['distance_km'], -results['Ice + Snow Result'], label='CI Thickness', color="#8EDDFC")
# plt.plot(results['distance_km'], -results['SIPL Result'], label='SIPL Thickness', color='darkorange')

# if all(col in results.columns for col in ['P_lat', 'P_CI', 'P_SIPL']):
#     plt.scatter(point_data['distance_km'], -point_data['P_CI'], color="#0974CC", marker='D', s=60, label='CI Point Measurement', zorder=3)
#     plt.scatter(point_data['distance_km'], -point_data['P_SIPL'], color="#C90C02", marker='D', s=60, label='SIPL Point Measurement', zorder=3)




# Plotting ice thicknesses vs latitude

plt.plot(results['latitude'], results['Consolidated Ice Result'], label='CI Thickness', color="#8EDDFC") # marker='o'
plt.plot(results['latitude'], results['SIPL Result'], label='SIPL Thickness', color='darkorange') # marker='x'

# Plot point measurements
if all(col in results.columns for col in ['P_lat', 'P_CI', 'P_SIPL']):
    point_data = results[['P_lat', 'P_CI', 'P_SIPL']].dropna()
    plt.scatter(point_data['P_lat'], point_data['P_CI'], color="#0974CC", marker='D', s=60, label='CI Point Measurement', zorder=3)
    plt.scatter(point_data['P_lat'], point_data['P_SIPL'], color="#C90C02", marker='D', s=60, label='SIPL Point Measurement', zorder=3)

# Get min and max latitude
min_lat = results['latitude'].min()
max_lat = results['latitude'].max()
# Create 8 evenly spaced latitude values
xticks = np.linspace(min_lat, max_lat, 6)
xticks_rounded = np.round(xticks, 3)
plt.xticks(xticks_rounded, xticks_rounded.astype(str), rotation=0, ha='center')


# # Part of plotting ice thickness results against the index of results 
# plt.plot(
#     results['longitude'], -results['Ice + Snow Result'],
#     label='CI Thickness', color='blue', alpha=0.2
# )
# plt.plot(
#     results['longitude'], -results['SIPL Result'],
#     label='SIPL Thickness', color='darkorange', alpha=0.2
# )

# # --- Add lines of best fit (fully opaque) ---
# # For Ice + Snow Result
# coeffs_ice = np.polyfit(results['longitude'], -results['Ice + Snow Result'], 1)
# fit_ice = np.poly1d(coeffs_ice)
# plt.plot(
#     results['longitude'], fit_ice(results['longitude']),
#     color='navy', linestyle='--', linewidth=2, label='CI Thickness Trend'
# )

# # For SIPL Result
# coeffs_sipl = np.polyfit(results['longitude'], -results['SIPL Result'], 1)
# fit_sipl = np.poly1d(coeffs_sipl)
# plt.plot(
#     results['longitude'], fit_sipl(results['longitude']),
#     color='darkorange', linestyle='--', linewidth=2, label='SIPL Thickness Trend'
# )
# # --- End lines of best fit ---



# # # Plot faded original data - with the mean CI and SIPL thicknesses for each longitde value given
# # plt.plot(
# #     results['longitude'], -results['Ice + Snow Result'],
# #     label='CI Thickness', color='blue', alpha=0.5
# # )
# # plt.plot(
# #     results['longitude'], -results['SIPL Result'],
# #     label='SIPL Thickness', color='darkorange', alpha=0.5
# # )

# # # --- Compute and plot average at each longitude ---
# # grouped = results.groupby('longitude').agg({
# #     'Consolidated Ice Result': 'mean',
# #     'SIPL Result': 'mean'
# # }).reset_index()

# # plt.plot(
# #     grouped['longitude'], -grouped['Consolidated Ice Result'],
# #     color='navy', linestyle='--', linewidth=2, label='CI Thickness Mean'
# # )
# # plt.plot(
# #     grouped['longitude'], -grouped['SIPL Result'],
# #     color='darkorange', linestyle='--', linewidth=2, label='SIPL Thickness Mean'
# # )
# # # --- End average lines ---


# Add labels, title, and legend
plt.ylim(6,0)
plt.xlabel('Latitude', fontsize=14)
plt.ylabel('Thickness (m)', fontsize=14)
plt.legend(fontsize=14, loc='upper right')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.grid(True)
plt.show()


# # Plot "Ice + Snow Result"
# plt.plot(results.index, -results['Ice + Snow Result'], label='Consolidated ice', color="#8EDDFC") # marker='o'

# # Plot "SIPL Result"
# plt.plot(results.index, -results['SIPL Result'], label='SIPL', color='darkorange') #marker='x'

# # Add labels, title, and legend
# plt.xlabel('Index (Number of Values)', fontsize=14)
# plt.ylabel('Thickness (m)', fontsize=14)
# plt.legend(fontsize=14)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.grid(True)
# # plt.show()




# Removing outliers from the results 

def filter_outliers(series):
    mask = [True]  # Always keep the first value
    for i in range(1, len(series) - 1):
        prev = series.iloc[i - 1]
        curr = series.iloc[i]
        next_ = series.iloc[i + 1]
        # Avoid division by zero
        prev_diff = abs(curr - prev) / abs(prev) if prev != 0 else 0
        next_diff = abs(curr - next_) / abs(next_) if next_ != 0 else 0
        if prev_diff > 0.1 and next_diff > 0.1:
            mask.append(False)
        else:
            mask.append(True)
    mask.append(True)  # Always keep the last value
    return pd.Series(mask, index=series.index)

# Get masks for both series
ice_mask = filter_outliers(results['Consolidated Ice Result'])
sipl_mask = filter_outliers(results['SIPL Result'])

# Filtered data
filtered_ice = results['Consolidated Ice Result'][ice_mask]
filtered_sipl = results['SIPL Result'][sipl_mask]

# Outlier data
outlier_ice = results['Consolidated Ice Result'][~ice_mask]
outlier_sipl = results['SIPL Result'][~sipl_mask]

# Print outlier indices and values
print("Outliers in Consolidated Ice Result:")
print(outlier_ice)
print("Outliers in SIPL Result:")
print(outlier_sipl)

# Use the intersection of valid indices for both series
valid_idx = filtered_ice.index.intersection(filtered_sipl.index)
filtered_ice = filtered_ice.loc[valid_idx]
filtered_sipl = filtered_sipl.loc[valid_idx]

plt.figure(figsize=(10, 6))

# Plot filtered results
plt.plot(valid_idx, -filtered_ice, label='CI', color="#8EDDFC", zorder=1)
plt.plot(valid_idx, -filtered_sipl, label='SIPL', color='darkorange', zorder=1)

# Plot outliers as red "x"
plt.scatter(outlier_ice.index, -outlier_ice, color="#0974CC", marker='x', label='CI Outliers', zorder=3)
plt.scatter(outlier_sipl.index, -outlier_sipl, color="#C90C02", marker='x', label='SIPL Outliers', zorder=3)

plt.xlabel('Index', fontsize=14)
plt.ylabel('Thickness (m)', fontsize=14)
plt.legend(fontsize=12, loc='upper right')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(-6,0)
plt.grid(True)
plt.show()

# Combine filtered results and coordinates into a DataFrame
filtered_df = results.loc[valid_idx, ['Consolidated Ice Result', 'SIPL Result', 'longitude', 'latitude']]

# Export to Excel
output_path = r'C:/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/REPEAT_scaled_Earlier/Results/OutlierRemoved_HCP/HCP_Results_NoOutliers.xlsx'
filtered_df.to_excel(output_path, index=False)

print(f"Filtered results exported to {output_path}")
