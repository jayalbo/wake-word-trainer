# Wake Word Detection Model

A deep learning model for detecting wake words in audio samples using MFCC and mel spectrogram features.

## Model Performance

The model achieves the following performance metrics:

- Overall Accuracy: 89.19%
- Wake Word Detection (Recall): 100.00%
- Non-Wake Word Detection (Specificity): 86.48%
- Average Confidence: 85.10%

### Confusion Matrix

- True Wake (correctly identified wake words): 135
- True Not Wake (correctly identified non-wake words): 467
- False Wake (incorrectly classified as wake): 73
- False Not Wake (missed wake words): 0

## Features

- Uses both MFCC (13 coefficients) and mel spectrogram (40 bands) features
- Consistent audio duration handling (1 second)
- Feature normalization and standardization
- Data augmentation for minority class
- Balanced class weights
- Early stopping and learning rate reduction
- Model checkpointing

## Model Architecture

```
Input Layer
├── Conv2D (16 filters, 3x3)
│   ├── BatchNormalization
│   ├── LeakyReLU
│   └── MaxPooling2D
├── Conv2D (32 filters, 3x3)
│   ├── BatchNormalization
│   ├── LeakyReLU
│   └── MaxPooling2D
├── Conv2D (64 filters, 3x3)
│   ├── BatchNormalization
│   ├── LeakyReLU
│   └── MaxPooling2D
├── Flatten
├── Dense (64 units)
│   ├── BatchNormalization
│   ├── LeakyReLU
│   └── Dropout
└── Output Layer (2 units, softmax)
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- librosa
- numpy
- scikit-learn
- seaborn
- matplotlib

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model:

```bash
python train.py
```

The training script will:

- Load and preprocess audio data
- Extract features (MFCC and mel spectrogram)
- Train the model with data augmentation
- Save the best model to `wake_word_model.keras`

### Testing

To evaluate the model:

```bash
python test.py
```

The test script will:

- Load the trained model
- Process test audio files
- Generate a confusion matrix
- Display detailed performance metrics

## Dataset Structure

```
dataset/
├── wake/
│   └── [wake word audio files]
└── not_wake/
    └── [non-wake word audio files]
```

## Model Parameters

- Sample rate: 16000 Hz
- Audio duration: 1.0 seconds
- MFCC features: 13 coefficients
- Mel spectrogram: 40 bands
- Hop length: 512 samples
- Batch size: 16
- Learning rate: 0.0005
- Early stopping patience: 15
- Class weights: Balanced based on class distribution

## License

MIT License
