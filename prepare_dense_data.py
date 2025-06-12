#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Parameters for sequence creation
SEQUENCE_LENGTH = 50  # 50 timestamps per sequence (about 1 second of data at 50Hz)
STEP_SIZE = 5        # Reduced step size for more sequences with overlap
TEST_SIZE = 0.2      # 20% for validation
RANDOM_SEED = 42     # For reproducibility
N_FEATURES = 3       # X, Y, Z accelerometer
FLATTENED_SIZE = SEQUENCE_LENGTH * N_FEATURES

def create_flattened_sequences(df, sequence_length, step_size):
    """Create flattened sequences for Dense model using sliding windows."""
    sequences = []
    labels = []
    
    # Group by motion state to ensure sequences don't cross different motions
    for _, group in df.groupby('state'):
        # Get the filtered accelerometer data
        data = group[['accel_x_filtered', 'accel_y_filtered', 'accel_z_filtered']].values
        
        # Calculate number of sequences for this group
        n_sequences = (len(data) - sequence_length) // step_size + 1
        
        # Create sequences using sliding windows
        for i in range(n_sequences):
            start_idx = i * step_size
            end_idx = start_idx + sequence_length
            
            if end_idx <= len(data):
                # Flatten the sequence immediately
                sequence = data[start_idx:end_idx].flatten()  # Shape: (150,)
                label = group.iloc[start_idx]['state']
                sequences.append(sequence)
                labels.append(label)
    
    return np.array(sequences), np.array(labels)

def standardize_data(X_train, X_val):
    """Standardize the data using StandardScaler."""
    # Data is already 2D, no need to reshape
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler parameters
    os.makedirs('models', exist_ok=True)
    with open('models/scaler_params.h', 'w') as f:
        f.write('#ifndef SCALER_PARAMS_H\n')
        f.write('#define SCALER_PARAMS_H\n\n')
        f.write('// Scaler parameters for standardization\n')
        f.write('const float SCALER_MEAN[] = {')
        f.write(', '.join([f'{x:.6f}f' for x in scaler.mean_]))
        f.write('};\n\n')
        f.write('const float SCALER_SCALE[] = {')
        f.write(', '.join([f'{x:.6f}f' for x in scaler.scale_]))
        f.write('};\n\n')
        f.write('#endif // SCALER_PARAMS_H\n')
    
    # Print standardization parameters
    print("\nStandardization Parameters:")
    print("Mean values:")
    for i in range(0, FLATTENED_SIZE, N_FEATURES):
        print(f"  Timestamp {i//N_FEATURES}:")
        print(f"    X: {scaler.mean_[i]:.6f}")
        print(f"    Y: {scaler.mean_[i+1]:.6f}")
        print(f"    Z: {scaler.mean_[i+2]:.6f}")
    
    # Verify standardization
    means = np.mean(X_train_scaled, axis=0)
    stds = np.std(X_train_scaled, axis=0)
    
    print("\nVerification - Scaled Data Statistics:")
    print("Mean range: [{:.6f}, {:.6f}] (should be close to 0)".format(
        np.min(means), np.max(means)))
    print("Std range: [{:.6f}, {:.6f}] (should be close to 1)".format(
        np.min(stds), np.max(stds)))
    
    return X_train_scaled, X_val_scaled

def prepare_data_for_dense():
    """Prepare the processed data for Dense model training."""
    # Load the processed dataset
    data_path = os.path.join('processed_data', 'combined_dataset.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find processed data at {data_path}")
    
    df = pd.read_csv(data_path)
    print("Loaded dataset with shape:", df.shape)
    
    # Ensure state column is integer
    df['state'] = df['state'].astype(np.int32)
    
    # Create sequences
    print("\nCreating sequences...")
    X, y = create_flattened_sequences(df, SEQUENCE_LENGTH, STEP_SIZE)
    print(f"Created {len(X)} sequences of shape {X.shape}")
    
    # Print original class distribution
    print("\nOriginal samples per class:")
    for state in sorted(df['state'].unique()):
        count = len(df[df['state'] == state])
        print(f"State {state}: {count} samples")
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    print("\nData split summary:")
    print(f"Training sequences: {len(X_train)}")
    print(f"Validation sequences: {len(X_val)}")
    
    # Standardize the sequences
    print("\nStandardizing sequences...")
    X_train, X_val = standardize_data(X_train, X_val)
    
    # Convert to appropriate types for TensorFlow Lite compatibility
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    
    # Create output directory
    output_dir = 'dense_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the prepared data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    
    # Save data properties for model configuration
    properties = {
        'sequence_length': SEQUENCE_LENGTH,
        'n_features': N_FEATURES,
        'flattened_size': FLATTENED_SIZE,
        'n_classes': len(np.unique(y_train)),
        'class_labels': sorted(df['state'].unique().tolist())
    }
    
    # Save properties as text file
    with open(os.path.join(output_dir, 'data_properties.txt'), 'w') as f:
        for key, value in properties.items():
            f.write(f"{key}: {value}\n")
    
    print("\nData properties:")
    for key, value in properties.items():
        print(f"{key}: {value}")
    
    print(f"\nPrepared data has been saved to {output_dir}/")
    
    # Print class distribution
    print("\nClass distribution in training set:")
    for label in sorted(np.unique(y_train)):
        count = np.sum(y_train == label)
        percentage = count / len(y_train) * 100
        print(f"State {label}: {count} sequences ({percentage:.1f}%)")
    
    return properties

if __name__ == "__main__":
    prepare_data_for_dense() 