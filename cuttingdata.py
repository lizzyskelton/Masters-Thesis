import pandas as pd

# File path
csv_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\Repeat_HCP_AITResults\SCALED_results_HCP\NoOutliers_Scaled_HCP_AITResults\readyforQGIS\NoOutliers_Results_ArrayB.csv"

# The two points (longitude, latitude)
point1 = (166.4385, -77.8539)
point2 = (166.4531, -77.8545)

# Load data
df = pd.read_csv(csv_path)

# Find the index of the two points (using a small tolerance for floating point comparison)
tol = 1e-4
idx1 = df[(abs(df['longitude'] - point1[0]) < tol) & (abs(df['latitude'] - point1[1]) < tol)].index[0]
idx2 = df[(abs(df['longitude'] - point2[0]) < tol) & (abs(df['latitude'] - point2[1]) < tol)].index[0]

# Get the slice between the two indices (inclusive)
start, end = sorted([idx1, idx2])
df_between = df.loc[start:end].reset_index(drop=True)

# Save to a new CSV
output_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ApparentThickness_Aug\Repeat_HCP_AITResults\SCALED_results_HCP\NoOutliers_Scaled_HCP_AITResults\readyforQGIS\ArrayB_AIT_BetweenPoints_HCP.csv"
df_between.to_csv(output_path, index=False)

print(f"Filtered data saved to {output_path}")


# ## Separating March and May data for plotting in QGIS

# import pandas as pd

# # File path
# # csv_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\ResultsSpreadsheets\EarlierIce_NoOutliers\CommaDelim_forQGIS\EarlierIce_Results_NoOutliers.csv"
# csv_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\REPEAT_scaled_Earlier\Results\OutlierRemoved_HCP\commadelimforQGIS\Combined_HCP_Results_NoOutliers.csv"

# # The two pairs of points (longitude, latitude)
# pair1_point1 = (166.5973, -77.9079)
# pair1_point2 = (166.6085, -77.9071)
# pair2_point1 = (166.6309, -77.9055)
# pair2_point2 = (166.6633, -77.9025)

# # Load data
# df = pd.read_csv(csv_path)

# tol = 1e-4

# def find_index(df, point, tol):
#     matches = df[(abs(df['longitude'] - point[0]) < tol) & (abs(df['latitude'] - point[1]) < tol)]
#     if not matches.empty:
#         return matches.index[0]
#     # If not found, print the closest row
#     df['dist'] = ((df['longitude'] - point[0])**2 + (df['latitude'] - point[1])**2)**0.5
#     closest = df.nsmallest(1, 'dist')
#     print(f"Point {point} not found. Closest row:")
#     print(closest[['longitude', 'latitude', 'dist']])
#     return None

# # Find indices for both pairs
# idx1a = find_index(df, pair1_point1, tol)
# idx1b = find_index(df, pair1_point2, tol)
# idx2a = find_index(df, pair2_point1, tol)
# idx2b = find_index(df, pair2_point2, tol)

# if None in [idx1a, idx1b, idx2a, idx2b]:
#     print("One or more points not found. Please check the coordinates.")
# else:
#     # Get slices for both pairs (inclusive)
#     s1, e1 = sorted([idx1a, idx1b])
#     s2, e2 = sorted([idx2a, idx2b])
#     multi_year_df = pd.concat([df.loc[s1:e1], df.loc[s2:e2]]).reset_index(drop=True)
#     multi_year_df['label'] = 'Multi-year'

#     # Mark all indices included in multi_year_df
#     multi_year_indices = set(df.loc[s1:e1].index).union(df.loc[s2:e2].index)
#     march_may_df = df.loc[~df.index.isin(multi_year_indices)].copy().reset_index(drop=True)
#     march_may_df['label'] = 'March_May'

#     # Save both datasets
#     multi_year_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\REPEAT_scaled_Earlier\Results\OutlierRemoved_HCP\EarlierIce_Results_NoOutliers_MultiYear.csv"
#     march_may_path = r"C:\Users\esk22\Downloads\Lizzy_Analysis\BackUp_April_2024\REPEAT_scaled_Earlier\Results\OutlierRemoved_HCP\EarlierIce_Results_NoOutliers_MarchMay.csv"
#     multi_year_df.to_csv(multi_year_path, index=False)
#     march_may_df.to_csv(march_may_path, index=False)

#     print(f"Multi-year data saved to {multi_year_path}")
#     print(f"March_May data saved to {march_may_path}")