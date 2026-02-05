import pandas as pd
import numpy as np
import os
import os.path as osp
import argparse

def extract_npy_filenames_to_csv(args):
    """
    Extract filenames (without extension) of all .npy files in the specified directory
    and write the results to a designated CSV file with a single 'genome_id' column
    """
    # Define input and output file paths
    input_dir = args.input_dir  # Directory containing target .npy files
    output_csv = "./scripts/data/genome_unique_ids.csv"  # Path for the output CSV file
    
    # Check if the input directory exists
    if not osp.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist!")
        return
    
    # Create output directory if it does not exist
    output_dir = osp.dirname(output_csv)
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Collect filenames (without .npy extension) of all valid .npy files
    genome_ids = []
    for filename in os.listdir(input_dir):
        # Process only .npy files, exclude subdirectories and other file types
        if filename.endswith(".csv") and osp.isfile(osp.join(input_dir, filename)):
            # Extract filename by removing the last 4 characters (.npy extension)
            genome_id = filename[:-4]
            genome_ids.append(genome_id)
    
    # Check if any .npy files were found in the input directory
    if not genome_ids:
        print(f"Warning:  {input_dir}!")
        return
    
    # Sort the collected genome IDs (optional, retain for ordered output)
    genome_ids.sort()
    
    # Write the collected genome IDs to the CSV file
    try:
        with open(output_csv, "w", encoding="utf-8") as f:
            # Write CSV header
            f.write("genome_id\n")
            # Write each genome ID as a separate line
            for gid in genome_ids:
                f.write(f"{gid}\n")
        print(f"Success! Written {len(genome_ids)} genome_ids to {output_csv}")
    except Exception as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./results')
    args = parser.parse_args()
    extract_npy_filenames_to_csv(args)