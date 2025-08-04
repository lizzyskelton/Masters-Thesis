import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

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

# Define a color map for consolidated ice shading (red to purple)
ice_cmap = LinearSegmentedColormap.from_list("red_to_purple", ["red", "purple"])
# Gather all consolidated ice values for normalization
consolidated_ice_values = []
for sheet_name in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    if "Consolidated Ice" in df.columns:
        consolidated_ice_values.append(df["Consolidated Ice"].iloc[0] / 10)
consolidated_ice_values = np.array(consolidated_ice_values)
ice_min, ice_max = consolidated_ice_values.min(), consolidated_ice_values.max()
plt.figure(figsize=(10, 6))

# Find the indices of the min and max consolidated ice values
min_idx = np.argmin(consolidated_ice_values)
max_idx = np.argmax(consolidated_ice_values)

for idx, sheet_name in enumerate(sheet_names):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # Divide the values by 10,000
    df[x_column] = df[x_column] / 10000
    df[y_column] = df[y_column] / 10000
    # Get the label from the "Consolidated Ice" column
    if "Consolidated Ice" in df.columns:
        ice_val = df["Consolidated Ice"].iloc[0]
        label = f"{ice_val:.1f}"  # Only one decimal place
    else:
        label = sheet_name
    # Plot the curve in black
    plt.plot(df[x_column], df[y_column], linestyle='-', color='black', linewidth=0.5, zorder=1, label=label)
    plt.scatter(df[x_column], df[y_column], c=df[color_column], cmap=cmap, norm=norm, marker='o', s=20, zorder=2)
    df_sipl_0 = df[df[color_column] == 0]
    plt.plot(df_sipl_0[x_column], df_sipl_0[y_column], linestyle='-', color='red', linewidth=2, zorder=3)
    # Add label beside every 5th curve, and always for min/max
    # if (idx % 5 == 0 or idx == min_idx or idx == max_idx) and "Consolidated Ice" in df.columns:
    #     plt.text(df[x_column].iloc[-1] - 0.1, df[y_column].iloc[-1], f"{ice_val:.1f}",
    #              color='black', fontsize=7, va='center', ha='right', fontweight='bold')

# # Collect SIPL=0 points from each sheet, then plotting them in red
# sipl0_x = []
# sipl0_y = []
# for sheet_name in sheet_names:
#     df = pd.read_excel(file_path, sheet_name=sheet_name)
#     df[x_column] = df[x_column] / 10000
#     df[y_column] = df[y_column] / 10000
#     df_sipl_0 = df[df[color_column] == 0]
#     if not df_sipl_0.empty:
#         # Take the first (or only) point where SIPL=0
#         sipl0_x.append(df_sipl_0[x_column].iloc[0])
#         sipl0_y.append(df_sipl_0[y_column].iloc[0])

# plt.plot(sipl0_x, sipl0_y, color='red', linewidth=0.5, marker='o', label='no SIPL', zorder=10)

# Set the title and labels 
plt.colorbar(label="SIPL thickness (m)")
plt.xlabel("Inphase (x10^4 ppm)")
plt.ylabel("Quadrature (x10^4 ppm)")
plt.grid(True)
leg = plt.legend(title="CI Thickness (m)", fontsize=7, ncol=3)
leg.get_title().set_fontweight('bold')
leg.get_title().set_ha('left')  # Left-align the legend title
plt.tight_layout()

# Define the additional data points, these are the measured values from EM31 data, an average of all data points in a 2m raduis
additional_x = [0.45, 0.76, 0.95, 1.20, 1.22, 1.17, -0.01, 0.00, 0.00, 0.95, 0.99, -0.23, 0.48, 0.48]  
additional_y = [1.92, 2.63, 2.79, 3.18, 3.21, 3.07, 0.93, 0.93, 0.92, 2.76, 2.76, 0.89, 2.61, 2.54] 

# Plot the additional data points with red "x" marks
plt.scatter(additional_x, additional_y, color='red', marker='x', s=50, zorder=4, label='Measured')

# Define the new set of data points, these are the expected values from the forward model 
new_x = [1.25, 1.67, 1.80, 2.39, 2.42, 2.21, 0.70, 0.71, 0.74, 2.14, 2.11, 0.68, 1.71, 1.89]  # Measured x (I) values from the EM31 data
new_y = [1.27, 1.77, 1.87, 2.27, 2.31, 2.12, 0.59, 0.61, 0.60, 2.20, 2.15, 0.59, 1.96, 2.00]  # Measured y (Q) values from the EM31 data 

# Plot the new set of data points with yellow marks
plt.scatter(new_x, new_y, color='yellow', marker='o', s=50, zorder=5, label='Expected')

# Link the points with lines
for i in range(min(len(additional_x), len(new_x))):
    plt.plot([additional_x[i], new_x[i]], [additional_y[i], new_y[i]], color='orange', linestyle='-', linewidth=1, zorder=4)

# Define the additional data values for "Ice + Snow thickness" and "SIPL thickness"
ice_snow_thickness = [3.33, 2.71, 2.63, 2.45, 2.42, 2.52, 5, 4.92, 4.95, 2.39, 2.43, 5.04, 2.52, 2.51]
sipl_thickness = [3.43, 2.94, 2.57, 1.55, 1.6, 1.77, 3.97, 3.98, 3.53, 2.22, 2.22, 4.2, 3.23, 2.55]

    # Annotate each line segment with "Ice + Snow thickness" and "SIPL thickness"
for i in range(min(len(additional_x), len(new_x))):
    # Calculate the midpoint of the line segment
    midpoint_x = (additional_x[i] + new_x[i]) / 2
    midpoint_y = (additional_y[i] + new_y[i]) / 2
    
    # Ice + Snow thickness in blue, positioned slightly above the line
    plt.text(midpoint_x, midpoint_y + 0.1, f"{ice_snow_thickness[i]:.1f}", color='blue', fontsize=8, ha='center')
    
    # SIPL thickness in black, positioned slightly below the line
    plt.text(midpoint_x, midpoint_y - 0.2, f"{sipl_thickness[i]:.1f}", color='black', fontsize=8, ha='center')

# Add legend entries for the text annotations
blue_text_legend = Line2D([], [], color='blue', marker='_', linestyle='None', label='CI thickness (m)')
black_text_legend = Line2D([], [], color='black', marker='_', linestyle='None', label='SIPL thickness (m)')

# Add legend entries for measured and expected points
measured_legend = Line2D([], [], color='red', marker='x', linestyle='None', label='Measured')
expected_legend = Line2D([], [], color='yellow', marker='o', linestyle='None', label='Expected')

# Add a legend
plt.legend(handles=[measured_legend, expected_legend, blue_text_legend, black_text_legend], loc='upper left')  # Adjust the location as needed

# # Define the scaled data points using the given equations
# scaled_x = [(1.25 * x) + 0.85 for x in additional_x]
# scaled_y = [(0.76 * y) - 0.10 for y in additional_y]

# # Plot the scaled data points with pink dots
# plt.scatter(scaled_x, scaled_y, color='pink', marker='o', s=50, zorder=5, label='Scaled')

# # Link the scaled points with the measured/additional points
# for i in range(len(additional_x)):
#     plt.plot([additional_x[i], scaled_x[i]], [additional_y[i], scaled_y[i]], color='purple', linestyle='-', linewidth=1, zorder=4)

# link_legend = Line2D([], [], color='purple', linestyle='-', linewidth=1, label='Measured → Scaled')
orange_link_legend = Line2D([], [], color='orange', linestyle='-', linewidth=1, label='Measured → Expected')

# Add a legend entry for the scaled points
scaled_legend = Line2D([], [], color='pink', marker='o', linestyle='None', label='Scaled')

# Update the legend to include the scaled points
plt.legend(handles=[measured_legend, expected_legend, orange_link_legend, blue_text_legend, black_text_legend], loc='upper left')  # Adjust the location as needed

# # Set the x and y axis limits to zoom in on the data points - if i want to hihglight different between measured and expected
# plt.xlim(min(min(additional_x), min(new_x)) - 0.5, max(max(additional_x), max(new_x)) + 0.5)
# plt.ylim(min(min(additional_y), min(new_y)) - 0.5, max(max(additional_y), max(new_y)) + 0.5)

#Save the plot as an image file
plt.savefig('plot.png')

# Show the plot
plt.show()       # deleted so tables can print in the terminal 
#plt.pause(5)  # Display the plot for 5 seconds

# # Create a table of the data points
# data_table = {
#     'Additional X': additional_x,
#     'Additional Y': additional_y,
#     'New X': new_x[:len(additional_x)],
#     'New Y': new_y[:len(additional_y)],
#     'Scaled X': scaled_x,
#     'Scaled Y': scaled_y
# }

# # Convert the data into a pandas DataFrame for better formatting
# data_df = pd.DataFrame(data_table)

# # Print the table
# print(data_df)















import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

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

# Define a color map for consolidated ice shading (red to purple)
ice_cmap = LinearSegmentedColormap.from_list("red_to_purple", ["red", "purple"])
# Gather all consolidated ice values for normalization
consolidated_ice_values = []
for sheet_name in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    if "Consolidated Ice" in df.columns:
        consolidated_ice_values.append(df["Consolidated Ice"].iloc[0] / 10)
consolidated_ice_values = np.array(consolidated_ice_values)
ice_min, ice_max = consolidated_ice_values.min(), consolidated_ice_values.max()
plt.figure(figsize=(10, 6))

# Find the indices of the min and max consolidated ice values
min_idx = np.argmin(consolidated_ice_values)
max_idx = np.argmax(consolidated_ice_values)

for idx, sheet_name in enumerate(sheet_names):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # Divide the values by 10,000
    df[x_column] = df[x_column] / 10000
    df[y_column] = df[y_column] / 10000
    # Get the label from the "Consolidated Ice" column
    if "Consolidated Ice" in df.columns:
        ice_val = df["Consolidated Ice"].iloc[0]
        label = f"{ice_val:.1f}"  # Only one decimal place
    else:
        label = sheet_name
    # Plot the curve in black
    plt.plot(df[x_column], df[y_column], linestyle='-', color='black', linewidth=0.5, zorder=1, label=label)
    plt.scatter(df[x_column], df[y_column], c=df[color_column], cmap=cmap, norm=norm, marker='o', s=20, zorder=2)
    df_sipl_0 = df[df[color_column] == 0]
    plt.plot(df_sipl_0[x_column], df_sipl_0[y_column], linestyle='-', color='red', linewidth=2, zorder=3)
    # # Add label beside every 5th curve, and always for min/max
    # if (idx % 5 == 0 or idx == min_idx or idx == max_idx) and "Consolidated Ice" in df.columns:
    #     plt.text(df[x_column].iloc[-1] - 0.1, df[y_column].iloc[-1], f"{ice_val:.1f}",
    #              color='black', fontsize=7, va='center', ha='right', fontweight='bold')

# # Collect SIPL=0 points from each sheet, then plotting them in red
# sipl0_x = []
# sipl0_y = []
# for sheet_name in sheet_names:
#     df = pd.read_excel(file_path, sheet_name=sheet_name)
#     df[x_column] = df[x_column] / 10000
#     df[y_column] = df[y_column] / 10000
#     df_sipl_0 = df[df[color_column] == 0]
#     if not df_sipl_0.empty:
#         # Take the first (or only) point where SIPL=0
#         sipl0_x.append(df_sipl_0[x_column].iloc[0])
#         sipl0_y.append(df_sipl_0[y_column].iloc[0])

# plt.plot(sipl0_x, sipl0_y, color='red', linewidth=0.5, marker='o', label='no SIPL', zorder=10)


# Set the title and labels 
plt.colorbar(label="SIPL thickness (m)")
plt.xlabel("Inphase (x10^4 ppm)")
plt.ylabel("Quadrature (x10^4 ppm)")
plt.grid(True)
leg = plt.legend(title="CI Thickness (m)", fontsize=7, ncol=3)
leg.get_title().set_fontweight('bold')
leg.get_title().set_ha('left')  # Left-align the legend title
plt.tight_layout()

# Define the measured data points, these are the measured values from EM31 data, an average of all data points in a 2m raduis
measured_x = [0.45, 0.76, 0.95, 1.20, 1.22, 1.17, -0.01, 0.00, 0.00, 0.95, 0.99, -0.23, 0.48, 0.48]  
measured_y = [1.92, 2.63, 2.79, 3.18, 3.21, 3.07, 0.93, 0.93, 0.92, 2.76, 2.76, 0.89, 2.61, 2.54] 

# Plot the measured data points with red "x" marks
plt.scatter(measured_x, measured_y, color='red', marker='x', s=50, zorder=4, label='Measured')

# Define the new set of data points, these are the expected values from the forward model 
new_x = [2.32, 3.06, 3.30, 4.29, 4.36, 4.00, 1.32, 1.35, 1.40, 3.87, 3.81, 1.29, 3.14, 3.44]  
new_y = [2.14, 2.85, 3.00, 3.52, 3.57, 3.32, 1.07, 1.11, 1.08, 3.42, 3.35, 1.06, 3.12, 3.16]  

# # Plot the expected data points with yellow marks
plt.scatter(new_x, new_y, color="#FFFB1F", marker='o', s=50, zorder=5, label='Expected')

# Link the points with lines
for i in range(min(len(measured_x), len(new_x))):
    plt.plot([measured_x[i], new_x[i]], [measured_y[i], new_y[i]], color='orange', linestyle='-', linewidth=1, zorder=4)

# Define the drill-hole data values for "Ice + Snow thickness" and "SIPL thickness"
ice_snow_thickness = [3.33, 2.71, 2.63, 2.45, 2.42, 2.52, 5, 4.92, 4.95, 2.39, 2.43, 5.04, 2.52, 2.51]
sipl_thickness = [3.43, 2.94, 2.57, 1.55, 1.6, 1.77, 3.97, 3.98, 3.53, 2.22, 2.22, 4.2, 3.23, 2.55]

    # Annotate each line segment with "Ice + Snow thickness" and "SIPL thickness"
for i in range(min(len(measured_x), len(new_x))):
    # Calculate the midpoint of the line segment
    midpoint_x = (measured_x[i] + new_x[i]) / 2
    midpoint_y = (measured_y[i] + new_y[i]) / 2

    # Ice + Snow thickness in blue, positioned slightly above the line
    plt.text(midpoint_x, midpoint_y + 0.1, f"{ice_snow_thickness[i]:.1f}", color='blue', fontsize=8, ha='center')
    
    # SIPL thickness in black, positioned slightly below the line
    plt.text(midpoint_x, midpoint_y - 0.2, f"{sipl_thickness[i]:.1f}", color='black', fontsize=8, ha='center')

# Add legend entries for the text annotations
blue_text_legend = Line2D([], [], color='blue', marker='_', linestyle='None', label='CI thickness (m)')
black_text_legend = Line2D([], [], color='black', marker='_', linestyle='None', label='SIPL thickness (m)')

# Add legend entries for measured and expected points
measured_legend = Line2D([], [], color='red', marker='x', linestyle='None', label='Measured')
expected_legend = Line2D([], [], color="#FFFB1F", marker='o', linestyle='None', label='Expected')
orange_link_legend = Line2D([], [], color='orange', linestyle='-', linewidth=1, label='Measured → Expected')

# # Add a legend
# plt.legend(handles=[measured_legend, expected_legend, orange_link_legend, blue_text_legend, black_text_legend], loc='upper left')  # Adjust the location as needed

# # Define the scaled data points using the given equations
scaled_x = [(2.21 * x) + 1.60 for x in measured_x]
scaled_y = [(1.11 * y) + 0.08 for y in measured_y]

# Plot the scaled data points with pink dots
plt.scatter(scaled_x, scaled_y, color='pink', marker='o', s=50, zorder=5, label='Scaled')

# Link the scaled points with the measured points
for i in range(len(measured_x)):
    plt.plot([measured_x[i], scaled_x[i]], [measured_y[i], scaled_y[i]], color='purple', linestyle='-', linewidth=1, zorder=4)

purple_link_legend = Line2D([], [], color='purple', linestyle='-', linewidth=1, label='Measured → Scaled')

# Add a legend entry for the scaled points
scaled_legend = Line2D([], [], color='pink', marker='o', linestyle='None', label='Scaled')

# Update the legend to include the scaled points
plt.legend(handles=[measured_legend, expected_legend, scaled_legend, orange_link_legend, purple_link_legend, blue_text_legend, black_text_legend], loc='upper left')  # Adjust the location as needed

# # Set the x and y axis limits to zoom in on the data points - if i want to hihglight different between measured and expected
# plt.xlim(min(min(measured_x), min(new_x)) - 0.5, max(max(measured_x), max(new_x)) + 0.5)
# plt.ylim(min(min(measured_y), min(new_y)) - 0.5, max(max(measured_y), max(new_y)) + 0.5)

#Save the plot as an image file
plt.savefig('plot.png')

# Show the plot
plt.show()       # deleted so tables can print in the terminal 
#plt.pause(5)  # Display the plot for 5 seconds

# Create a table of the data points
data_table = {
    'Measured X': measured_x,
    'Measured Y': measured_y,
    'New X': new_x[:len(measured_x)],
    'New Y': new_y[:len(measured_y)],
    'Scaled X': scaled_x,
    'Scaled Y': scaled_y
}

# Convert the data into a pandas DataFrame for better formatting
data_df = pd.DataFrame(data_table)

# Print the table
print(data_df)
