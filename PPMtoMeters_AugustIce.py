# August Ice I and Q measurement data to Ice + Snow and SIPL thickness measurements
# I.e. finding the closest point on the forward model 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

print("Script is running...")

# Load the Excel file
file_path = '/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/AugustIce1000_Trendlines.xlsx'
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
plt.title("Forward Model for 'August Ice - 1000mS/m bulk cond'")
plt.xlabel("Inphase (x10^4 ppm)")
plt.ylabel("Quadrature (x10^4 ppm)")
plt.grid(True)

# Define the additional data points, these are the measured values from EM31 data
additional_x = [2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 1.53, 1.51, 1.54, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05, 2.05]  
additional_y = [5.17, 6.15, 6.24, 5.20, 5.10, 5.07, 5.79, 5.69, 10.30, 3.76, 3.72, 3.79, 4.79, 5.22, 5.65, 6.19, 6.05, 6.43, 5.39, 7.95] 

# Plot the additional, EM31 data points with red "x" marks
plt.scatter(additional_x, additional_y, color='red', marker='x', s=50, zorder=4)

# Define the new set of data points, these are the expected values from the forward model 
new_x = [3.83, 4.67, 4.58, 4.01, 4.24, 3.99, 4.04, 4.36, 6.79, 2.69, 2.83, 3.00, 3.86, 3.97, 4.51, 4.69, 4.32, 4.62, 3.94, 5.10]  # Expected x (I) values from the forward model
new_y = [3.94, 5.02, 4.88, 4.27, 4.48, 4.24, 4.39, 4.67, 8.24, 2.92, 3.00, 3.03, 4.04, 4.21, 4.78, 5.01, 4.51, 4.94, 4.04, 5.57] 

# Plot the new set of data points with yellow marks
plt.scatter(new_x, new_y, color='yellow', marker='o', s=50, zorder=5)

# Define the additional data values for "Ice + Snow thickness" and "SIPL thickness"
ice_snow_thickness = [1.89, 1.67, 1.71, 1.75, 1.75, 1.76, 1.68, 1.69, 1.15, 2.12, 2.1, 2.16, 1.83, 1.77, 1.74, 1.72, 1.83, 1.69, 1.90, 1.62]
sipl_thickness = [0.86, 0.57, 0.57, 1.04, 0.77, 0.97, 1.12, 0.8, 0.28, 1.96, 1.83, 1.44, 0.97, 0.99, 0.64, 0.53, 0.6, 0.57, 0.78, 0.42]

# Link the points with lines
for i in range(min(len(additional_x), len(new_x))):
    plt.plot([additional_x[i], new_x[i]], [additional_y[i], new_y[i]], color='orange', linestyle='-', linewidth=1, zorder=4)

    # Annotate each line segment with "Ice + Snow thickness" and "SIPL thickness"
for i in range(min(len(additional_x), len(new_x))):
    # Calculate the midpoint of the line segment
    midpoint_x = (additional_x[i] + new_x[i]) / 2
    midpoint_y = (additional_y[i] + new_y[i]) / 2
    
    # Ice + Snow thickness in blue, positioned slightly above the line
    plt.text(midpoint_x, midpoint_y + 0.3, f"{ice_snow_thickness[i]:.1f}", color='blue', fontsize=8, ha='center')
    
    # SIPL thickness in black, positioned slightly below the line
    plt.text(midpoint_x, midpoint_y - 0.3, f"{sipl_thickness[i]:.1f}", color='black', fontsize=8, ha='center')


# # Import data from the CSV file
# csv_file_path = '/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/AugustIce_FilteredDatasets/251024transectWR_stationarycamp_A.csv'
# csv_data = pd.read_csv(csv_file_path)

# # Extract the "Inphase" and "Quadrature" columns
# inphase = csv_data['Inphase']
# quadrature = csv_data['Quadrature']

# # Scale the data points
# scaled_inphase = inphase
# scaled_quadrature = (0.74 * quadrature) + 0.29

# Convert additional_y to a NumPy array
additional_y = np.array(additional_y)

# Scale the data points
scaled_inphase = additional_x  # Assuming additional_x is already in the correct format
scaled_quadrature = (0.74 * additional_y) + 0.29


# # Create a DataFrame with the scaled values
# scaled_data = pd.DataFrame({
#     'scaled_inphase': scaled_inphase,
#     'scaled_quadrature': scaled_quadrature
# })

# # Save the DataFrame to an Excel file
# output_file_path = r'C:\Users\esk22\Downloads\Scaled_251024transectWR_stationarycamp_A.xlsx'
# scaled_data.to_excel(output_file_path, index=False)

# print(f"Scaled data saved to {output_file_path}")

# Plot the scaled data points in orange
plt.scatter(scaled_inphase, scaled_quadrature, color='darkviolet', label='Scaled point measurements', marker='o', s=20, zorder=4)

# Plot the original data points in dark green
plt.scatter(additional_x, additional_y, color='darkgreen', label='Original point measurements', marker='o', s=20, zorder=3)

# # Add an arrow pointing right between the two groups of data points
# arrow_x = (additional_x.mean() + scaled_inphase.mean()) / 2  # Midpoint x-coordinate
# arrow_y = (additional_y.mean() + scaled_quadrature.mean()) / 2  # Midpoint y-coordinate
# plt.annotate(
#    '', 
#    xy=(scaled_inphase.mean(), scaled_quadrature.mean()),  # Arrowhead at scaled data
#    xytext=(additional_x.mean(), additional_y.mean()),  # Arrow tail at original data
#    arrowprops=dict(facecolor='black', arrowstyle='->', lw=2),  # Arrow style
#    zorder=5
# )

# Add legend entries for the text annotations
blue_text_legend = Line2D([], [], color='blue', marker='_', linestyle='None', label='Ice and Snow thickness (m)')
black_text_legend = Line2D([], [], color='black', marker='_', linestyle='None', label='SIPL thickness (m)')

# Add legend entries for measured and expected points
measured_legend = Line2D([], [], color='red', marker='x', linestyle='None', label='Measured')
expected_legend = Line2D([], [], color='yellow', marker='o', linestyle='None', label='Expected')
scaled_legend = Line2D([], [], color='darkviolet', marker='o', linestyle='None', label='Scaled')

# Add a legend
plt.legend(handles=[measured_legend, expected_legend, scaled_legend, blue_text_legend, black_text_legend], loc='upper left') 

# # Set the x and y axis limits to zoom in on the data points - if i want to hihglight different between measured and expected
# plt.xlim(min(min(additional_x), min(new_x)) - 0.5, max(max(additional_x), max(new_x)) + 0.5)
# plt.ylim(min(min(additional_y), min(new_y)) - 0.5, max(max(additional_y), max(new_y)) + 0.5)

# # Calculate the combined range for both groups of data
# x_min = min(inphase.min(), scaled_inphase.min()) - 0.1
# x_max = max(inphase.max(), scaled_inphase.max()) + 0.1
# y_min = min(quadrature.min(), scaled_quadrature.min()) - 0.1
# y_max = max(quadrature.max(), scaled_quadrature.max()) + 0.1

# # Set axis limits to zoom in on the two groups of data
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)

# # Set axis limits to zoom in on the scaled data points
# plt.xlim(scaled_inphase.min() - 0.1, scaled_inphase.max() + 0.1)
# plt.ylim(scaled_quadrature.min() - 0.1, scaled_quadrature.max() + 0.1)


# Save the plot as an image file
plt.savefig('plot.png')

# Show the plot
plt.show()




# # Calculating the Ice and Snow values, first with 4 smallest distances, then inverse distance weighting, then loop through all sheets to get full transect converted. 

# import pandas as pd
# import numpy as np

# # File paths
# scaled_file_path = 'C:/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/ResultsSpreadsheets/Scaled_041124.xlsx'
# results_file_path = 'C:/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/ResultsSpreadsheets/Results_041124.xlsx'

# # Load the scaled data
# scaled_data = pd.read_excel(scaled_file_path)

# # Create a list to store the results
# results_list = []

# # Loop through each scaled_inphase and scaled_quadrature value
# for index, row in scaled_data.iterrows():
#     scaled_inphase = row['scaled_inphase']
#     scaled_quadrature = row['scaled_quadrature']

#     # Create an empty DataFrame to store results from all sheets
#     all_distances = pd.DataFrame()

#     # Loop through each sheet in the Excel file
#     for sheet_name in sheet_names:
#         # Load the data for the current sheet
#         df = pd.read_excel(file_path, sheet_name=sheet_name)

#         # Calculate the distance for each row
#         df['Distance'] = np.sqrt((df['I'] - scaled_inphase)**2 + (df['Q'] - scaled_quadrature)**2)

#         # Add a column to track the sheet name
#         df['Sheet Name'] = sheet_name

#         # Append the results to the combined DataFrame
#         all_distances = pd.concat([all_distances, df], ignore_index=True)

#     # Find the four rows with the smallest distances across all sheets
#     closest_rows = all_distances.nsmallest(4, 'Distance')

#     # Add a new column for weights
#     closest_rows['Weight'] = 1 / closest_rows['Distance']

#     # Add a new column for "Value x Weight" for "Ice + Snow"
#     closest_rows['Ice + Snow x Weight'] = closest_rows['Ice + Snow'] * closest_rows['Weight']

#     # Calculate the weighted average for "Ice + Snow"
#     ice_snow_result = closest_rows['Ice + Snow x Weight'].sum() / closest_rows['Weight'].sum()

#     # Add a new column for "Value x Weight" for "SIPL"
#     closest_rows['SIPL x Weight'] = closest_rows['SIPL'] * closest_rows['Weight']

#     # Calculate the weighted average for "SIPL"
#     sipl_result = closest_rows['SIPL x Weight'].sum() / closest_rows['Weight'].sum()

#     # Append the results to the results list
#     results_list.append({
#         'scaled_inphase': scaled_inphase,
#         'scaled_quadrature': scaled_quadrature,
#         'Ice + Snow Result': ice_snow_result,
#         'SIPL Result': sipl_result
#     })

# # Convert the results list to a DataFrame
# results = pd.DataFrame(results_list)

# # Save the results to an Excel file
# results.to_excel(results_file_path, index=False)

# print(f"Results saved to {results_file_path}")


# # Plotting Snow + Ice and SIPL thickness results against longitude

# import pandas as pd
# import matplotlib.pyplot as plt

# # File path for the results Excel file
# results_file_path = r'C:/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/ResultsSpreadsheets/Results_311024.xlsx'

# # Load the results data from the Excel file
# results = pd.read_excel(results_file_path)

# # Ensure the results DataFrame is loaded
# # Plot "Ice + Snow Result" and "SIPL Result"
# plt.figure(figsize=(10, 6))

# # Plot "Ice + Snow Result" with longitude on the x-axis
# plt.plot(results['longitude'], results['Ice + Snow Result'], label='Ice + Snow Thickness', color='blue') # marker='o'
 
# # Plot "SIPL Result" with longitude on the x-axis
# plt.plot(results['longitude'], results['SIPL Result'], label='SIPL Thickness', color='darkorange') # marker='x'

# # Add labels, title, and legend
# plt.xlabel('Longitude')
# plt.ylabel('Thickness (m)')
# plt.title('"Ice Thicknesses on Earlier Ice 31/10/24"')
# plt.legend()

# # # Plot "Ice + Snow Result"
# # plt.plot(results.index, results['Ice + Snow Result'], label='Ice + Snow Result', color='blue') # marker='o'

# # # Plot "SIPL Result"
# # plt.plot(results.index, results['SIPL Result'], label='SIPL Result', color='orange') #marker='x'

# # # Add labels, title, and legend
# # plt.xlabel('Index (Number of Values)')
# # plt.ylabel('Result')
# # plt.title('Ice Thicknesses on Earlier Ice 04/11/24')
# # plt.legend()

# # Add grid for better readability
# plt.grid(True)

# # Show the plot
# plt.show()