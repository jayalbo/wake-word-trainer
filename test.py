import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def extract_features(audio_path, target_sr=16000, duration=1.0):
    try:
        # Load audio with consistent duration
        audio, sr = librosa.load(audio_path, sr=target_sr, duration=duration)
        
        # Pad or truncate to ensure consistent length
        target_length = int(target_sr * duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Extract features with fixed parameters
        n_mels = 40
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
        
        # Reshape for CNN input (add batch and channel dimensions)
        features = np.expand_dims(np.expand_dims(features, axis=0), axis=-1)
        
        return features
        
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    print("\nLoading model...")
    model = tf.keras.models.load_model('wake_word_model.keras')
    
    print("\nModel Summary:")
    print("=" * 50)
    model.summary()
    print("\nModel Output Shape:", model.output_shape)
    print("Model Output Layer:", model.layers[-1].name)
    print("=" * 50)
    
    # Test directories
    wake_dir = os.path.join("dataset", "wake")
    not_wake_dir = os.path.join("dataset", "not_wake")
    
    print("\nChecking source directories:")
    print(f"Wake directory exists: {os.path.exists(wake_dir)}")
    print(f"Not wake directory exists: {os.path.exists(not_wake_dir)}")
    print("-" * 50)
    
    # Process all files
    predictions = []
    true_labels = []
    confidences = []
    total_files = 0
    
    # Process wake files
    for filename in os.listdir(wake_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(wake_dir, filename)
            features = extract_features(file_path)
            
            if features is not None:
                prediction = model.predict(features, verbose=0, batch_size=1)
                pred_class = np.argmax(prediction)
                confidence = prediction[0][pred_class] * 100
                
                predictions.append(pred_class)
                true_labels.append(1)  # 1 for wake
                confidences.append(confidence)
                total_files += 1
                
                print(f"File: {filename}")
                print(f"Source: wake")
                print(f"Prediction: {'wake' if pred_class == 1 else 'not_wake'}")
                print(f"Confidence: {confidence:.2f}%")
                print(f"Raw Prediction: {prediction[0]}")
                print("-" * 50)
    
    # Process not_wake files
    for filename in os.listdir(not_wake_dir):
        if filename.endswith('.wav'):
            file_path = os.path.join(not_wake_dir, filename)
            features = extract_features(file_path)
            
            if features is not None:
                prediction = model.predict(features, verbose=0, batch_size=1)
                pred_class = np.argmax(prediction)
                confidence = prediction[0][pred_class] * 100
                
                predictions.append(pred_class)
                true_labels.append(0)  # 0 for not_wake
                confidences.append(confidence)
                total_files += 1
                
                print(f"File: {filename}")
                print(f"Source: not_wake")
                print(f"Prediction: {'wake' if pred_class == 1 else 'not_wake'}")
                print(f"Confidence: {confidence:.2f}%")
                print(f"Raw Prediction: {prediction[0]}")
                print("-" * 50)
    
    # Calculate metrics
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    confidences = np.array(confidences)
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(cm, classes=['not_wake', 'wake'])
    
    # Print summary
    print("\nTest Results Summary:")
    print("=" * 50)
    print(f"Total files tested: {total_files}")
    print(f"Files predicted as wake: {np.sum(predictions == 1)} ({np.mean(predictions == 1)*100:.1f}%)")
    print(f"Files predicted as not_wake: {np.sum(predictions == 0)} ({np.mean(predictions == 0)*100:.1f}%)")
    print(f"\nAverage confidence: {np.mean(confidences):.2f}%")
    print(f"Minimum confidence: {np.min(confidences):.2f}%")
    print(f"Maximum confidence: {np.max(confidences):.2f}%")
    
    # Print confusion matrix details
    print("\nConfusion Matrix Details:")
    print(f"True Wake (correctly predicted wake): {cm[1,1]}")
    print(f"True Not Wake (correctly predicted not wake): {cm[0,0]}")
    print(f"False Wake (incorrectly predicted wake): {cm[0,1]}")
    print(f"False Not Wake (incorrectly predicted not wake): {cm[1,0]}")
    
    # Calculate accuracy metrics
    accuracy = np.mean(predictions == true_labels)
    wake_accuracy = np.mean(predictions[true_labels == 1] == 1)
    not_wake_accuracy = np.mean(predictions[true_labels == 0] == 0)
    
    print("\nAccuracy Metrics:")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Wake Accuracy: {wake_accuracy*100:.2f}%")
    print(f"Not Wake Accuracy: {not_wake_accuracy*100:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    main() 