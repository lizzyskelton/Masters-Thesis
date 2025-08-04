# Filtering .csv files to only include August Ice measurements

import geopandas as gpd
import pandas as pd

# File paths
csv_file_path = '/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/AllEM31dataascsv/311024.csv'
shapefile_path = '/Users/esk22/Downloads/QGIS/AugustIce/AugustIce.shp'

# Load the CSV file as a DataFrame
csv_data = pd.read_csv(csv_file_path)

# Ensure the CSV file has latitude and longitude columns (adjust column names if necessary)
latitude_column = 'latitude'  # Replace with the actual column name for latitude
longitude_column = 'longitude'  # Replace with the actual column name for longitude

# Convert the CSV data to a GeoDataFrame
csv_gdf = gpd.GeoDataFrame(
    csv_data,
    geometry=gpd.points_from_xy(csv_data[longitude_column], csv_data[latitude_column]),
    crs="EPSG:4326"  # Assuming WGS84 coordinate system; adjust if needed
)

# Load the shapefile as a GeoDataFrame
polygon_gdf = gpd.read_file(shapefile_path)

# Ensure both GeoDataFrames use the same CRS
csv_gdf = csv_gdf.to_crs(polygon_gdf.crs)

# Perform a spatial join to filter points within the polygon
points_within_polygon = gpd.sjoin(csv_gdf, polygon_gdf, how="inner", predicate="within")

# Drop the geometry column if you want a regular DataFrame
filtered_data = points_within_polygon.drop(columns='geometry')

# Print or save the filtered data
filtered_data.to_csv('/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/AugustIce_FilteredDatasets/FilteredData.csv', index=False)

