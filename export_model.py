import os
import shutil
import tensorflow as tf
import tensorflowjs as tfjs

def export_models():
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Source model path
    source_model = "wake_word_model.keras"
    
    if not os.path.exists(source_model):
        print(f"Error: {source_model} not found. Please train the model first.")
        return
    
    # Copy the original Keras model to output
    keras_output = os.path.join(output_dir, "keras_model.keras")
    shutil.copy2(source_model, keras_output)
    print(f"Copied Keras model to {keras_output}")
    
    # Convert to TensorFlow.js format
    tfjs_output_dir = os.path.join(output_dir, "tfjs_model")
    if os.path.exists(tfjs_output_dir):
        shutil.rmtree(tfjs_output_dir)
    
    # Convert the model
    tfjs.converters.save_keras_model(
        tf.keras.models.load_model(source_model),
        tfjs_output_dir
    )
    print(f"Converted model to TensorFlow.js format in {tfjs_output_dir}")
    
    # Create a simple HTML file to demonstrate usage
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Wake Word Detection Demo</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands"></script>
</head>
<body>
    <h1>Wake Word Detection Demo</h1>
    <p>This is a placeholder for the wake word detection implementation.</p>
    <p>The model files are in the tfjs_model directory.</p>
</body>
</html>"""
    
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html_content)
    print("Created demo HTML file")
    
    print("\nExport completed successfully!")
    print(f"Output directory: {output_dir}")
    print("\nContents:")
    print("- keras_model.keras (Original Keras model)")
    print("- tfjs_model/ (TensorFlow.js model files)")
    print("- index.html (Demo page)")

if __name__ == "__main__":
    export_models() 