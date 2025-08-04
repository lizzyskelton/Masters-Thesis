# Filtering .xlsx files to only include Earlier Ice measurements

import geopandas as gpd
import pandas as pd

# File paths
excel_file_path = 'C:/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/ResultsSpreadsheets/Results_041124.xlsx'
shapefile_path = 'C:/Users/esk22/OneDrive - University of Canterbury/Masters Project Lizzy/QGIS/EarlierIce/EarlierIce.shp'

# Load the Excel file as a DataFrame
excel_data = pd.read_excel(excel_file_path)

latitude_column = 'latitude'  
longitude_column = 'longitude' 

# Convert the Excel data to a GeoDataFrame
excel_gdf = gpd.GeoDataFrame(
    excel_data,
    geometry=gpd.points_from_xy(excel_data[longitude_column], excel_data[latitude_column]),
    crs="EPSG:4326"  # Assuming WGS84 coordinate system; adjust if needed
)

# Load the shapefile as a GeoDataFrame
polygon_gdf = gpd.read_file(shapefile_path)

# Ensure both GeoDataFrames use the same CRS
excel_gdf = excel_gdf.to_crs(polygon_gdf.crs)

# Perform a spatial join to filter points within the polygon
points_within_polygon = gpd.sjoin(excel_gdf, polygon_gdf, how="inner", predicate="within")

# Drop the geometry column if you want a regular DataFrame
filtered_data = points_within_polygon.drop(columns='geometry')

# Save the filtered data to an Excel file
filtered_data.to_excel('C:/Users/esk22/Downloads/Lizzy_Analysis/BackUp_April_2024/ResultsSpreadsheets/Earlier Ice Results/FilteredData.xlsx', index=False)