## Script to combine multiple Excel files into one

import pandas as pd

# List of Excel file paths to combine
file_paths = [

    r"C:\Users\.......xlsx",
    r"C:\Users\.......xlsx",
    r"C:\Users\.......xlsx",
]

# Read and concatenate all sheets, keeping only the first header row
dfs = [pd.read_excel(fp) for fp in file_paths]
combined_df = pd.concat(dfs, ignore_index=True)

# Remove duplicate header rows if present (sometimes happens if headers are repeated)
if combined_df.columns[0] == combined_df.iloc[0, 0]:
    combined_df = combined_df.iloc[1:]

# Save the combined DataFrame
output_path = r"C:\Users\.......xlsx"
combined_df.to_excel(output_path, index=False)

print(f"Combined data saved to {output_path}")