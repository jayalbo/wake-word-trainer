from keras.models import Sequential
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score
import keras.backend as K
from tensorflow.keras.utils import to_categorical

def balanced_accuracy_metric(y_true, y_pred):
    y_true_argmax = K.argmax(y_true, axis=-1)
    y_pred_argmax = K.argmax(y_pred, axis=-1)
    
    # Convert to float32 to avoid integer division
    y_true_argmax = K.cast(y_true_argmax, 'float32')
    y_pred_argmax = K.cast(y_pred_argmax, 'float32')
    
    # Calculate per-class accuracy
    pos_acc = K.mean(K.cast(K.equal(y_pred_argmax, y_true_argmax) * K.equal(y_true_argmax, 1), 'float32'))
    neg_acc = K.mean(K.cast(K.equal(y_pred_argmax, y_true_argmax) * K.equal(y_true_argmax, 0), 'float32'))
    
    return (pos_acc + neg_acc) / 2

def load_audio_data(data_dir, target_sr=16000, duration=1.0):
    X = []
    y = []
    label_encoder = LabelEncoder()
    
    # First, collect all labels
    labels = []
    for label in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, label)):
            labels.append(label)
    
    label_encoder.fit(labels)
    print(f"Classes: {label_encoder.classes_}")
    
    # Process each class
    for label in labels:
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
            
        for filename in os.listdir(class_dir):
            if not filename.endswith('.wav'):
                continue
                
            file_path = os.path.join(class_dir, filename)
            try:
                # Load audio with consistent duration
                audio, sr = librosa.load(file_path, sr=target_sr, duration=duration)
                
                # Pad or truncate to ensure consistent length
                target_length = int(target_sr * duration)
                if len(audio) < target_length:
                    audio = np.pad(audio, (0, target_length - len(audio)))
                else:
                    audio = audio[:target_length]
                
                # Extract features with fixed parameters
                n_mels = 40  # Reduced from 64
                n_mfcc = 13
                hop_length = 512
                
                # Calculate number of frames
                n_frames = 1 + (target_length - 2048) // hop_length
                
                # Extract MFCC
                mfcc = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=n_mfcc)
                
                # Extract mel spectrogram
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=target_sr, n_mels=n_mels)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Ensure consistent shapes
                if mfcc.shape[1] < n_frames:
                    mfcc = np.pad(mfcc, ((0, 0), (0, n_frames - mfcc.shape[1])))
                else:
                    mfcc = mfcc[:, :n_frames]
                    
                if mel_spec_db.shape[1] < n_frames:
                    mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, n_frames - mel_spec_db.shape[1])))
                else:
                    mel_spec_db = mel_spec_db[:, :n_frames]
                
                # Combine features
                features = np.concatenate([mfcc, mel_spec_db], axis=0)
                
                # Normalize features
                features = (features - np.mean(features)) / (np.std(features) + 1e-6)
                
                # Reshape for CNN input (add channel dimension)
                features = np.expand_dims(features, axis=-1)
                
                X.append(features)
                y.append(label)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
    
    if not X:
        raise ValueError("No valid audio files were processed")
    
    X = np.array(X)
    y = label_encoder.transform(y)
    y = to_categorical(y)
    
    print(f"Data shape: {X.shape}")
    print(f"Number of samples per class:")
    for i, label in enumerate(label_encoder.classes_):
        count = np.sum(y[:, i])
        print(f"{label}: {count}")
    
    return X, y, label_encoder

def augment_data(X, y):
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        # Original data
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # Only augment minority class (wake)
        if np.argmax(y[i]) == 1:  # wake class
            # Add Gaussian noise
            for _ in range(3):  # Create 3 noisy versions
                noise_level = random.uniform(0.001, 0.01)
                noise = np.random.normal(0, noise_level, X[i].shape)
                augmented_X.append(X[i] + noise)
                augmented_y.append(y[i])
            
            # Time shift
            for _ in range(2):  # Create 2 shifted versions
                shift = np.random.randint(-3, 4)
                shifted = np.roll(X[i], shift, axis=1)
                augmented_X.append(shifted)
                augmented_y.append(y[i])
            
            # Frequency masking
            freq_mask = X[i].copy()
            mask_size = random.randint(2, 4)
            mask_start = random.randint(0, X[i].shape[0] - mask_size)
            freq_mask[mask_start:mask_start + mask_size, :, :] = 0
            augmented_X.append(freq_mask)
            augmented_y.append(y[i])
            
            # Time masking
            time_mask = X[i].copy()
            mask_size = random.randint(5, 10)
            mask_start = random.randint(0, X[i].shape[1] - mask_size)
            time_mask[:, mask_start:mask_start + mask_size, :] = 0
            augmented_X.append(time_mask)
            augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)

def create_model(input_shape, num_classes):
    model = Sequential([
        # Input layer
        Input(shape=input_shape),
        
        # First Conv Block - reduced complexity
        Conv2D(16, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),
        
        # Second Conv Block
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.15),
        
        # Third Conv Block
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        # Dense layers
        Flatten(),
        Dense(64),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.1),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(data_dir, model_path='wake_word_model.keras'):
    # Load and preprocess data
    X, y, label_encoder = load_audio_data(data_dir)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate class weights - less aggressive weighting
    n_samples = len(y_train)
    class_counts = np.sum(y_train, axis=0)
    total = np.sum(class_counts)
    class_weights = {i: (total / (2 * count)) for i, count in enumerate(class_counts)}
    
    print("Class weights:", class_weights)
    
    # Create and compile model
    model = create_model(input_shape=X.shape[1:], num_classes=len(label_encoder.classes_))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Reduced learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # More gentle reduction
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        batch_size=16,  # Reduced batch size
        epochs=100,  # Increased epochs
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save the final model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Print final metrics
    print("\nFinal training metrics:")
    for metric in history.history:
        print(f"{metric}: {history.history[metric][-1]}")
    
    return model, history, label_encoder

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)
    
    # Set paths
    data_dir = "dataset"
    model_path = "wake_word_model.keras"
    
    # Train the model
    print("\nTraining model...")
    model, history, label_encoder = train_model(data_dir, model_path)
    
    print("\nTraining completed!")
    print(f"Model saved to {model_path}")
    
    # Print final metrics
    print("\nFinal training metrics:")
    for metric in history.history:
        print(f"{metric}: {history.history[metric][-1]}") 