import librosa
import numpy as np
from keras.models import load_model
import os

def extract_features(audio_path, max_frames=100):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
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
    
    # Reshape for model input
    features = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)
    
    return features

def predict_audio(model_path, audio_path):
    # Load the model
    model = load_model(model_path)
    
    # Extract features
    features = extract_features(audio_path)
    
    # Make prediction
    prediction = model.predict(features, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    # Map class index to label
    class_labels = ['not_wake', 'wake']
    predicted_label = class_labels[predicted_class]
    
    return predicted_label, confidence

if __name__ == "__main__":
    model_path = "wake_word_model.keras"
    
    # Test directory containing audio files to test
    test_dir = "test_samples"
    
    if not os.path.exists(test_dir):
        print(f"Please create a '{test_dir}' directory and add some .wav files to test")
        exit(1)
    
    print("Testing audio files...")
    for file_name in os.listdir(test_dir):
        if file_name.endswith('.wav'):
            audio_path = os.path.join(test_dir, file_name)
            predicted_label, confidence = predict_audio(model_path, audio_path)
            print(f"\nFile: {file_name}")
            print(f"Predicted: {predicted_label}")
            print(f"Confidence: {confidence:.2%}") 