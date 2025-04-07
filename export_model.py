import os
import shutil
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np

def export_models():
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Source model path
    source_model = 'wake_word_model.keras'
    
    if not os.path.exists(source_model):
        print(f"Error: Source model {source_model} not found!")
        return
    
    # Load the model
    model = tf.keras.models.load_model(source_model)
    
    # Get the input shape from the model
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    
    # Create a new model with explicit input shape
    inputs = tf.keras.Input(shape=input_shape[1:])
    outputs = model(inputs)
    new_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Copy the original Keras model
    shutil.copy(source_model, 'output/keras_model.keras')
    
    # Convert to TensorFlow.js format with explicit input shape
    tfjs.converters.save_keras_model(new_model, 'output/tfjs_model')
    
    # Create a simple HTML demo file
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Wake Word Detector Demo</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.0.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands"></script>
    <script src="https://cdn.jsdelivr.net/npm/ml-matrix@6.10.4"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .controls {
            margin: 20px 0;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            background: #007bff;
            color: white;
            cursor: pointer;
            font-size: 16px;
        }
        button:disabled {
            background: #ccc;
        }
        button.recording {
            background: #dc3545;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            font-size: 18px;
        }
        .confidence {
            font-weight: bold;
        }
        .wake {
            background: #d4edda;
            color: #155724;
        }
        .not-wake {
            background: #f8d7da;
            color: #721c24;
        }
        #status {
            margin-top: 10px;
            color: #666;
        }
        .visualizer {
            width: 100%;
            height: 100px;
            background: #000;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Wake Word Detector Demo</h1>
        <div class="controls">
            <button id="recordButton">Start Recording</button>
            <div id="status">Model loading...</div>
        </div>
        <canvas id="visualizer" class="visualizer"></canvas>
        <div id="result"></div>
    </div>

    <script>
        let model;
        let audioContext;
        let analyser;
        let microphone;
        let isRecording = false;
        const recordButton = document.getElementById('recordButton');
        const resultDiv = document.getElementById('result');
        const statusDiv = document.getElementById('status');
        const visualizer = document.getElementById('visualizer');
        const ctx = visualizer.getContext('2d');

        // MFCC parameters
        const SAMPLE_RATE = 16000;
        const FRAME_LENGTH = 400;  // 25ms at 16kHz
        const FRAME_STEP = 160;    // 10ms at 16kHz
        const NUM_MFCC = 13;
        const NUM_FRAMES = 100;    // 1 second of audio

        // Load the model
        async function loadModel() {
            try {
                model = await tf.loadLayersModel('tfjs_model/model.json');
                statusDiv.textContent = 'Model loaded! Click "Start Recording" to begin.';
                recordButton.disabled = false;
            } catch (error) {
                statusDiv.textContent = 'Error loading model: ' + error.message;
                console.error(error);
            }
        }

        // Initialize audio context and analyzer
        async function setupAudio() {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: SAMPLE_RATE
            });
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 2048;
            
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: SAMPLE_RATE,
                        channelCount: 1
                    }
                });
                microphone = audioContext.createMediaStreamSource(stream);
                microphone.connect(analyser);
            } catch (error) {
                statusDiv.textContent = 'Error accessing microphone: ' + error.message;
            }
        }

        // Compute MFCC features
        function computeMFCC(audioData) {
            // Convert to mono if stereo
            const monoData = audioData.length === 2 ? 
                audioData[0].map((x, i) => (x + audioData[1][i]) / 2) : 
                audioData[0];

            // Apply pre-emphasis
            const preEmphasis = 0.97;
            const emphasized = new Float32Array(monoData.length);
            emphasized[0] = monoData[0];
            for (let i = 1; i < monoData.length; i++) {
                emphasized[i] = monoData[i] - preEmphasis * monoData[i - 1];
            }

            // Frame the signal
            const frames = [];
            for (let i = 0; i < emphasized.length - FRAME_LENGTH; i += FRAME_STEP) {
                frames.push(emphasized.slice(i, i + FRAME_LENGTH));
            }

            // Apply Hamming window
            const window = frames.map(frame => {
                return frame.map((x, i) => {
                    return x * (0.54 - 0.46 * Math.cos(2 * Math.PI * i / (FRAME_LENGTH - 1)));
                });
            });

            // Compute FFT
            const fft = new FFT(FRAME_LENGTH);
            const magnitudes = window.map(frame => {
                const fftResult = fft.forward(frame);
                return fftResult.magnitude;
            });

            // Compute mel filterbank
            const melBasis = createMelFilterbank(SAMPLE_RATE, FRAME_LENGTH, NUM_MFCC);
            const melFeatures = magnitudes.map(mag => {
                return melBasis.map(filter => {
                    return filter.reduce((sum, val, i) => sum + val * mag[i], 0);
                });
            });

            // Compute DCT
            const mfcc = melFeatures.map(features => {
                return computeDCT(features, NUM_MFCC);
            });

            // Pad or truncate to NUM_FRAMES
            if (mfcc.length > NUM_FRAMES) {
                mfcc.length = NUM_FRAMES;
            } else while (mfcc.length < NUM_FRAMES) {
                mfcc.push(new Array(NUM_MFCC).fill(0));
            }

            // Normalize
            const mfccArray = mfcc.flat();
            const mean = mfccArray.reduce((a, b) => a + b) / mfccArray.length;
            const std = Math.sqrt(mfccArray.reduce((a, b) => a + Math.pow(b - mean, 2)) / mfccArray.length);
            const normalized = mfccArray.map(x => (x - mean) / std);

            return normalized;
        }

        // Process audio data
        function processAudio(audioData) {
            // Compute MFCC features
            const features = computeMFCC(audioData);
            
            // Create tensor and reshape
            let tensor = tf.tensor(features);
            tensor = tensor.reshape([1, NUM_FRAMES, NUM_MFCC, 1]);
            
            // Make prediction
            const prediction = model.predict(tensor);
            const probabilities = prediction.dataSync();
            
            // Clean up
            tensor.dispose();
            prediction.dispose();
            
            return probabilities;
        }

        // Visualize audio
        function visualize() {
            if (!isRecording) return;
            
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            
            analyser.getByteTimeDomainData(dataArray);
            
            ctx.fillStyle = 'rgb(0, 0, 0)';
            ctx.fillRect(0, 0, visualizer.width, visualizer.height);
            
            ctx.lineWidth = 2;
            ctx.strokeStyle = 'rgb(0, 255, 0)';
            ctx.beginPath();
            
            const sliceWidth = visualizer.width * 1.0 / bufferLength;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * visualizer.height / 2;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
                
                x += sliceWidth;
            }
            
            ctx.lineTo(visualizer.width, visualizer.height / 2);
            ctx.stroke();
            
            requestAnimationFrame(visualize);
        }

        // Handle recording
        recordButton.addEventListener('click', async () => {
            if (!isRecording) {
                if (!audioContext) {
                    await setupAudio();
                }
                
                isRecording = true;
                recordButton.textContent = 'Stop Recording';
                recordButton.classList.add('recording');
                resultDiv.textContent = '';
                visualize();
                
                // Start recording and processing
                const bufferSize = 4096;
                const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
                
                processor.onaudioprocess = (e) => {
                    const inputData = e.inputBuffer.getChannelData(0);
                    const probabilities = processAudio([inputData]);
                    
                    const isWake = probabilities[0] > 0.5;
                    const confidence = (isWake ? probabilities[0] : probabilities[1]) * 100;
                    
                    resultDiv.className = isWake ? 'wake' : 'not-wake';
                    resultDiv.innerHTML = `Predicted: <strong>${isWake ? 'wake' : 'not wake'}</strong><br>
                                         Confidence: <span class="confidence">${confidence.toFixed(2)}%</span>`;
                };
                
                microphone.connect(processor);
                processor.connect(audioContext.destination);
                
            } else {
                isRecording = false;
                recordButton.textContent = 'Start Recording';
                recordButton.classList.remove('recording');
                resultDiv.textContent = '';
            }
        });

        // Load model when page loads
        loadModel();
    </script>
</body>
</html>
    """
    
    with open('output/index.html', 'w') as f:
        f.write(html_content)
    
    print("Models exported successfully!")
    print("1. Keras model saved as: output/keras_model.keras")
    print("2. TensorFlow.js model saved in: output/tfjs_model/")
    print("3. Demo page created at: output/index.html")
    print("\nTo test the model:")
    print("1. Start a local server in the output directory")
    print("2. Open index.html in your browser")
    print("3. Allow microphone access when prompted")
    print("4. Click 'Start Recording' to begin testing")

if __name__ == "__main__":
    export_models() 