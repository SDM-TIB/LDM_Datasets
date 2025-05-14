#!/usr/bin/env python3
"""
Script to combine split CSV files back into single files.
"""

import os
import pandas as pd
import argparse
import glob

def combine_csv_files(split_dir, output_file):
    """
    Combine multiple CSV files into a single file.
    
    Args:
        split_dir (str): Directory containing the split CSV files
        output_file (str): Path to save the combined file
    """
    print(f"Processing files in {split_dir}...")
    
    # Get all part files, sorted numerically
    part_files = sorted(glob.glob(os.path.join(split_dir, "*_part*.csv")),
                        key=lambda x: int(os.path.basename(x).split("_part")[1].split(".")[0]))
    
    if not part_files:
        print(f"No part files found in {split_dir}")
        return
    
    print(f"Found {len(part_files)} part files")
    
    # Initialize an empty DataFrame to store the combined data
    combined_df = pd.DataFrame()
    
    # Read and combine each part file
    for part_file in part_files:
        print(f"  Reading {os.path.basename(part_file)}")
        df = pd.read_csv(part_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the combined data
    combined_df.to_csv(output_file, index=False)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Saved combined file: {output_file} ({file_size_mb:.2f} MB, {len(combined_df)} rows)")

def main():
    parser = argparse.ArgumentParser(description='Combine split CSV files back into single files')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Base directory containing the split directories')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save the combined files')
    args = parser.parse_args()
    
    # Get all split directories
    split_dirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    
    for split_dir in split_dirs:
        input_path = os.path.join(args.input_dir, split_dir)
        output_file = os.path.join(args.output_dir, f"{split_dir}.csv")
        combine_csv_files(input_path, output_file)
    
    print("Done!")

if __name__ == "__main__":
    main()