# Voice Recognition System

A machine learning-based voice recognition system that can detect wake words and other voice commands. The system uses a Convolutional Neural Network (CNN) to process audio features and make predictions.

## Features

- Real-time audio processing
- MFCC feature extraction
- Deep CNN model architecture
- Data augmentation for better training
- High accuracy (>99% on test samples)
- Easy to use training and testing scripts

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Librosa
- NumPy
- scikit-learn

Install the requirements:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── train.py           # Training script
├── test.py           # Testing script
├── requirements.txt  # Python dependencies
├── dataset/          # Training data directory
│   ├── wake/        # Wake word samples
│   └── not_wake/    # Non-wake word samples
└── test_samples/     # Test audio samples
```

## Setup

1. Create the required directories:

```bash
mkdir -p dataset/wake dataset/not_wake test_samples
```

2. Add your audio samples:
   - Place wake word samples in `dataset/wake/`
   - Place non-wake word samples in `dataset/not_wake/`
   - All samples should be in WAV format

## Training

To train the model:

```bash
python train.py
```

The training script will:

1. Load and preprocess audio samples
2. Extract MFCC features
3. Train the CNN model
4. Save the trained model as `wake_word_model.keras`

## Testing

To test the model with new audio samples:

1. Place test audio files in the `test_samples/` directory
2. Run the test script:

```bash
python test.py
```

The script will:

- Process each audio file
- Make predictions
- Show confidence scores

## Model Architecture

The model uses a CNN architecture with:

- Multiple convolutional layers
- Batch normalization
- Dropout for regularization
- Data augmentation
- Early stopping to prevent overfitting

## Best Practices

1. **Audio Quality**:

   - Use consistent recording conditions
   - Minimize background noise
   - Maintain consistent volume levels

2. **Training Data**:

   - Collect at least 50 samples per class
   - Include various voices and accents
   - Ensure samples are properly labeled

3. **Testing**:
   - Test with different voices
   - Test in various environments
   - Verify confidence scores

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Train custom wake word detection models with TensorFlow. Includes audio processing, feature extraction, and CNN model training tools.
# wake-word-trainer
