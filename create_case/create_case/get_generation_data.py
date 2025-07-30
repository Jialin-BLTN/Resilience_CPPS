import os
import zipfile
import pandas as pd

"""
The function is used to extract all solar power data from raw files, and combine them as one file named "output_solar.csv"
"""

# Define directory paths
base_dir = 'zip'

# Initialize an empty dataframe to store combined slices
combined_df = pd.DataFrame()

# Iterate over files in the directory
for filename in os.listdir(base_dir):
    if filename.endswith(".zip"):
        # Full path to zip file
        zip_path = os.path.join(base_dir, filename)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)

            # For each file in the extracted ZIP
            for file_in_zip in zip_ref.namelist():
                if file_in_zip.startswith("60d_SCED_Gen_Resource") and file_in_zip.endswith(".csv"):
                    csv_path = os.path.join(base_dir, file_in_zip)
                    # Read the CSV into a dataframe
                    df = pd.read_csv(csv_path)

                    # Get rows where "Resource Type" is "PVGR"
                    slice_df = df[df["Resource Type"] == "PVGR"]

                    # Append the slice to the master dataframe
                    #combined_df = combined_df.append(slice_df)
                    combined_df = pd.concat([combined_df, slice_df], ignore_index=True)

# Reset index for the combined dataframe
combined_df.reset_index(drop=True, inplace=True)

# Save the combined dataframe if necessary
combined_df.to_csv(os.path.join(base_dir, 'output_solar.csv'), index=False)