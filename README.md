# Wake Word Trainer

A toolkit for training wake word detection models using TensorFlow. Processes voice samples, extracts MFCC features, and trains a CNN model for accurate wake word recognition. Includes tools for data preprocessing, model training, and testing.

## Features

- Real-time audio processing
- MFCC feature extraction
- Deep CNN model architecture
- Data augmentation for better training
- High accuracy (>99% on test samples)
- Easy to use training and testing scripts
- Export to both Keras and TensorFlow.js formats

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Librosa
- NumPy
- scikit-learn
- TensorFlow.js (for model export)

Install the requirements:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── train.py           # Training script
├── test.py           # Testing script
├── export_model.py   # Model export script
├── requirements.txt  # Python dependencies
├── dataset/          # Training data directory
│   ├── wake/        # Wake word samples
│   └── not_wake/    # Non-wake word samples
├── test_samples/     # Test audio samples
└── output/          # Exported models
    ├── keras_model.keras
    ├── tfjs_model/
    └── index.html
```

## Complete Workflow

1. **Setup**:

   ```bash
   mkdir -p dataset/wake dataset/not_wake test_samples
   ```

2. **Add Training Data**:

   - Place wake word samples in `dataset/wake/`
   - Place non-wake word samples in `dataset/not_wake/`
   - All samples should be in WAV format

3. **Train the Model**:

   ```bash
   python train.py
   ```

   This will:

   - Load and preprocess audio samples
   - Extract MFCC features
   - Train the CNN model
   - Save the trained model as `wake_word_model.keras`

4. **Test the Model**:

   ```bash
   python test.py
   ```

   This will:

   - Process test audio files
   - Make predictions
   - Show confidence scores

5. **Export the Model**:
   ```bash
   python export_model.py
   ```
   This will:
   - Create an `output` directory
   - Save the Keras model
   - Convert to TensorFlow.js format
   - Create a demo HTML page

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
