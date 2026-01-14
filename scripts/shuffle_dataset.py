#!/usr/bin/env python3
"""
Script to randomly shuffle the records in Indian_guy_dataset_final.csv 
while maintaining sequential job_applicant_id starting from 1
"""

import pandas as pd
import numpy as np
import os

def shuffle_dataset():
    # File path
    dataset_file = "dataset/indian_guy_cs_filtered.csv"
    
    print("Reading dataset...")
    df = pd.read_csv(dataset_file)
    print(f"Dataset contains {len(df)} rows")
    
    # Create backup of original file
    backup_file = dataset_file + ".pre_shuffle_backup"
    if os.path.exists(dataset_file):
        print(f"Creating backup: {backup_file}")
        df.to_csv(backup_file, index=False)
    
    # Randomly shuffle the rows (excluding the header)
    print("Randomly shuffling records...")
    shuffled_df = df.sample(frac=1, random_state=None).reset_index(drop=True)
    
    # Reset job_applicant_id to be sequential starting from 1
    shuffled_df['job_applicant_id'] = range(1, len(shuffled_df) + 1)
    
    print(f"Shuffled dataset: {len(shuffled_df)} rows")
    print(f"job_applicant_id range: {shuffled_df['job_applicant_id'].min()} to {shuffled_df['job_applicant_id'].max()}")
    
    # Save the shuffled dataset
    print(f"Saving shuffled dataset to {dataset_file}...")
    shuffled_df.to_csv(dataset_file, index=False)
    
    print("Shuffle completed successfully!")
    print(f"Dataset now contains {len(shuffled_df)} randomly ordered records with job_applicant_id from 1 to {len(shuffled_df)}")

if __name__ == "__main__":
    shuffle_dataset()
