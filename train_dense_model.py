#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def load_training_data():
    """Load and preprocess the training data."""
    data_dir = 'dense_data'  # Changed from lstm_data to dense_data
    
    # Load the prepared data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    
    # Data is already flattened, no need for reshape
    return X_train, X_val, y_train, y_val

def create_model(input_size, n_classes):
    """Create a Dense model with adjusted dropout for better generalization."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_size,)),
        Dropout(0.2),  # Reduced dropout for better feature preservation
        Dense(32, activation='relu'),
        Dropout(0.2),  # Reduced dropout
        Dense(n_classes, activation='softmax')
    ])
    
    # Use a custom learning rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0  # Gradient clipping to prevent extreme updates
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """Plot training history metrics."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def convert_to_tflite(model, input_size):
    """Convert model to TFLite format."""
    # Save model parameters
    with open('models/model_params.h', 'w') as f:
        f.write('#ifndef MODEL_PARAMS_H\n')
        f.write('#define MODEL_PARAMS_H\n\n')
        f.write(f'#define SEQUENCE_LENGTH 50\n')
        f.write(f'#define N_FEATURES 3\n')
        f.write(f'#define N_CLASSES {model.output_shape[-1]}\n')
        f.write(f'#define DENSE1_UNITS 64\n')
        f.write(f'#define DENSE2_UNITS 32\n')
        f.write(f'#define FLATTENED_SIZE {input_size}\n\n')
        f.write('#endif // MODEL_PARAMS_H\n')
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    converter.target_spec.supported_types = [tf.float32]
    converter._experimental_disable_per_channel = True
    converter.allow_custom_ops = False
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open('models/motion_classifier.tflite', 'wb') as f:
        f.write(tflite_model)
    
    # Convert to C array
    os.system('python convert_to_arduino.py')

def compute_class_weights(y_train):
    """Compute class weights with extra emphasis on vertical motions."""
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    
    # Base weights
    class_weights = {i: total_samples / (len(class_counts) * count) 
                    for i, count in enumerate(class_counts)}
    
    # Give extra weight to UP/DOWN motions (states 2 and 3)
    class_weights[2] *= 1.2  # UP
    class_weights[3] *= 1.2  # DOWN
    
    return class_weights

def main():
    # Load the training data
    print("Loading training data...")
    X_train, X_val, y_train, y_val = load_training_data()
    
    print("\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Create and train model
    print("\nCreating and training model...")
    model = create_model(X_train.shape[1], len(np.unique(y_train)))
    model.summary()
    
    # Compute class weights
    class_weights = compute_class_weights(y_train)
    print("\nClass weights:")
    for class_idx, weight in class_weights.items():
        print(f"Class {class_idx}: {weight:.2f}")
    
    # Train the model with adjusted parameters
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,  # Increased epochs
        batch_size=16,  # Reduced batch size for better generalization
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=30,  # Increased patience
                restore_best_weights=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=15,  # Increased patience
                min_lr=0.00001,
                verbose=1
            )
        ],
        class_weight=class_weights,
        verbose=1
    )
    
    # Save the model
    model.save('models/motion_classifier.keras')
    print("\nModel saved to models/motion_classifier.keras")
    
    # Plot training history
    plot_training_history(history)
    print("Training history plot saved as 'training_history.png'")
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate confusion matrix
    y_pred = np.argmax(model.predict(X_val), axis=1)
    plot_confusion_matrix(y_val, y_pred)
    print("Confusion matrix plot saved as 'confusion_matrix.png'")
    
    # Convert to TFLite
    print("\nConverting to TFLite format...")
    convert_to_tflite(model, X_train.shape[1])

if __name__ == "__main__":
    main() 