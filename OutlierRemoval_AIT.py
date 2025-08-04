## Filtering Apparent Ice Thickness values on August ice
# Removing values more than 10% different to a value either side 
# Eliabeth Skelton 04/07/2025

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = r'C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\Repeat_HCP_AITResults\SCALED_results_HCP\AIT_HCP_scaled_ArrayB.xlsx'
df = pd.read_excel(file_path)

# Filtering function (same as your CI/SIPL method)
def filter_outliers(series):
    mask = [True]  # Always keep the first value
    for i in range(1, len(series) - 1):
        prev = series.iloc[i - 1]
        curr = series.iloc[i]
        next_ = series.iloc[i + 1]
        prev_diff = abs(curr - prev) / abs(prev) if prev != 0 else 0
        next_diff = abs(curr - next_) / abs(next_) if next_ != 0 else 0
        if prev_diff > 0.1 and next_diff > 0.1:
            mask.append(False)
        else:
            mask.append(True)
    mask.append(True)  # Always keep the last value
    return pd.Series(mask, index=series.index)

# Apply filter to Apparent Ice Thickness (IDW)
ait_mask = filter_outliers(df['Apparent Ice Thickness (IDW)'])
filtered_ait = df['Apparent Ice Thickness (IDW)'][ait_mask]
outlier_ait = df['Apparent Ice Thickness (IDW)'][~ait_mask]

# Print outlier indices and values
print("Outliers in Apparent Ice Thickness (IDW):")
print(outlier_ait)

# Plot filtered and outlier results
plt.figure(figsize=(10, 6))
plt.plot(filtered_ait.index, -filtered_ait, label='Apparent Ice Thickness (filtered)', color="#B088FF", zorder=1)
plt.scatter(outlier_ait.index, -outlier_ait, color="#FF4136", marker='x', label='AIT Outliers', zorder=3)
plt.xlabel('Index (Number of Values)', fontsize=14)
plt.ylabel('Thickness (m)', fontsize=14)
plt.legend(fontsize=12, loc='upper right')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(-6, 0)
plt.grid(True)
plt.tight_layout()
plt.show()

# Export filtered results with coordinates
filtered_df = df.loc[filtered_ait.index, ['Apparent Ice Thickness (IDW)', 'longitude', 'latitude']]
output_path = r'C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\Repeat_HCP_AITResults\SCALED_results_HCP\NoOutliers_Scaled_HCP_AITResults\NoOutliers_Results.xlsx'
filtered_df.to_excel(output_path, index=False)
print(f"Filtered results exported to {output_path}")