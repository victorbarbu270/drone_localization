import pandas as pd
import numpy as np
import os
from scipy import signal

# Define the correct mapping of files to motion states
FILE_TO_STATE = {
    'MOTION0.CSV.xls': 0,  # STATIONARY_FLAT
    'MOTION1.CSV.xls': 1,  # STATIONARY_VERTICAL
    'MOTION2.CSV.xls': 2,  # MOVING_UP
    'MOTION3.CSV.xls': 3,  # MOVING_DOWN
    'MOTION4.CSV.xls': 4,  # HOVERING
    'MOTION5.CSV.xls': 5,  # MOVING_FORWARD
    'MOTION6.CSV.xls': 6,  # MOVING_BACKWARD
    'MOTION7.CSV.xls': 7,  # MOVING_LEFT
    'MOTION8.CSV.xls': 8,  # MOVING_RIGHT
    'MOTION9.CSV.xls': 9,  # ROTATING_CW
    'MOTION10.CSV.xls': 10  # ROTATING_CCW
}

# Motion state labels (matching original data collection)
MOTION_STATES = {
    0: "STATIONARY_FLAT",
    1: "STATIONARY_VERTICAL",
    2: "MOVING_UP",
    3: "MOVING_DOWN",
    4: "HOVERING",
    5: "MOVING_FORWARD",
    6: "MOVING_BACKWARD",
    7: "MOVING_LEFT",
    8: "MOVING_RIGHT",
    9: "ROTATING_CW",
    10: "ROTATING_CCW"
}

def process_file(filename, state):
    """Process a single motion data file."""
    try:
        # Read the CSV file with the correct format
        df = pd.read_csv(filename)
        
        # Apply low-pass filter to accelerometer data
        nyquist = 25  # Assuming 50Hz sampling rate
        cutoff = 5    # 5Hz cutoff frequency
        order = 4     # Filter order
        
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low')
        
        # Apply filter to each axis
        df['accel_x_filtered'] = signal.filtfilt(b, a, df['accel_x'])
        df['accel_y_filtered'] = signal.filtfilt(b, a, df['accel_y'])
        df['accel_z_filtered'] = signal.filtfilt(b, a, df['accel_z'])
        
        # Override the state with our mapping
        df['state'] = state
        
        print(f"Processed {filename}: {len(df)} samples with state {state} ({MOTION_STATES[state]})")
        return df
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None

def main():
    # Create output directory if it doesn't exist
    output_dir = 'processed_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each motion file
    all_data = []
    total_samples = 0
    
    print("\nProcessing motion files...")
    for filename, state in sorted(FILE_TO_STATE.items()):
        print(f"Looking for file: {filename}")
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found!")
            continue
            
        print(f"Processing {filename}...")
        df = process_file(filename, state)
        if df is not None:
            print(f"Successfully processed {filename} with {len(df)} samples")
            all_data.append(df)
            total_samples += len(df)
        else:
            print(f"Failed to process {filename}")
    
    if not all_data:
        print("No valid data was processed!")
        return
    
    # Combine all data
    print("\nCombining datasets...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total number of samples: {len(combined_df)}\n")
    print("Samples per motion state:")
    for state in sorted(combined_df['state'].unique()):
        count = len(combined_df[combined_df['state'] == state])
        print(f"State {state} ({MOTION_STATES[state]}): {count} samples")
    
    # Save combined dataset
    output_path = os.path.join(output_dir, 'combined_dataset.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved combined dataset to {output_path}")

if __name__ == "__main__":
    main() 