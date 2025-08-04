import pandas as pd
from geopy.distance import geodesic

def find_closest_coordinates(csv_file, target_lat, target_lon, lat_col='latitude', lon_col='longitude'):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Ensure the columns exist
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"CSV must contain '{lat_col}' and '{lon_col}' columns.")
    
    # Convert coordinates to numeric values (handle potential non-numeric data)
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=[lat_col, lon_col])  # Remove rows with invalid coordinates
    
    # Compute distances and find the closest match
    target_point = (target_lat, target_lon)
    df['distance'] = df.apply(lambda row: geodesic(target_point, (row[lat_col], row[lon_col])).meters, axis=1)
    closest_row = df.loc[df['distance'].idxmin()]
    
    return closest_row

if __name__ == "__main__":
    csv_path = input("Enter the path to the CSV file: ")
    target_lat = float(input("Enter the target latitude: "))
    target_lon = float(input("Enter the target longitude: "))
    
    try:
        closest_match = find_closest_coordinates(csv_path, target_lat, target_lon)
        print("Closest match found:")
        print(closest_match)
    except Exception as e:
        print(f"Error: {e}")
    
