from keras.models import Sequential
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_audio_data(data_dir, max_frames=100):
    features = []
    labels = []
    label_encoder = LabelEncoder()
    
    # Get all subdirectories (each subdirectory is a class)
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        print(f"Loading data from {class_name}...")
        
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.wav'):
                file_path = os.path.join(class_dir, file_name)
                try:
                    # Load and preprocess audio
                    y, sr = librosa.load(file_path, sr=None)
                    
                    # Extract MFCCs
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    
                    # Pad or truncate to max_frames
                    if mfcc.shape[1] < max_frames:
                        pad_width = ((0, 0), (0, max_frames - mfcc.shape[1]))
                        mfcc = np.pad(mfcc, pad_width, mode='constant')
                    else:
                        mfcc = mfcc[:, :max_frames]
                    
                    # Normalize
                    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
                    
                    features.append(mfcc)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    # Convert labels to numbers
    labels = label_encoder.fit_transform(labels)
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Reshape for CNN input (samples, height, width, channels)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # Convert labels to one-hot encoding
    y = np.eye(len(classes))[y]
    
    return X, y, label_encoder

def augment_data(X, y):
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        # Original data
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # Add some noise
        noise = np.random.normal(0, 0.01, X[i].shape)
        augmented_X.append(X[i] + noise)
        augmented_y.append(y[i])
        
        # Time shift
        shift = np.random.randint(-2, 3)
        shifted = np.roll(X[i], shift, axis=1)
        augmented_X.append(shifted)
        augmented_y.append(y[i])
    
    return np.array(augmented_X), np.array(augmented_y)

def create_model(input_shape, num_classes):
    model = Sequential([
        # Input layer
        Input(shape=input_shape),
        
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Flatten and dense layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile with better learning rate and optimizer
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_model(X_train, y_train, X_val, y_val, model_path):
    # Create and train the model
    model = create_model(input_shape=(X_train.shape[1], X_train.shape[2], 1),
                        num_classes=len(np.unique(np.argmax(y_train, axis=1))))
    
    # Augment training data
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    
    # Early stopping and model checkpoint
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train with augmented data
    history = model.fit(
        X_train_aug, y_train_aug,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=100,
        callbacks=[early_stopping, checkpoint]
    )
    
    return model, history

if __name__ == "__main__":
    # Set paths
    data_dir = "dataset"
    model_path = "wake_word_model.keras"
    
    # Load and preprocess data
    print("Loading audio data...")
    X, y, label_encoder = load_audio_data(data_dir)
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Train the model
    print("Training model...")
    model, history = train_model(X_train, y_train, X_val, y_val, model_path)
    
    print("Training completed!")
    print(f"Model saved to {model_path}") 