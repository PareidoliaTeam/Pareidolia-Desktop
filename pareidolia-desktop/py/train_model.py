"""
Created by Aleaxngelo Orozco Gutierrez on 2-10-2026

Currently is supposed to use file paths to get image training data and export a working model
However, until a virtual envoirment is made with the nseecery packages, it currently returns an error.
Future work will allow the python function to execute successfully and allow flexible model creation options
"""
import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# Model constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
# THIS WILL NEED CHANGING depending on how many labels the user will want to make, so it will not always remain 2 as it appears now
NUM_CLASSES = 2  # Binary classification: thing present or not present
LEARNING_RATE = 0.001

def create_cnn_model():
    """Creates a CNN model for image classification."""
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
        
        # Conv Block 1
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Conv Block 2
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Conv Block 3
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        layers.Dropout(0.2),
        
        # Flatten
        layers.Flatten(),
        
        # Dense Layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_images_from_folders(positives_path, negatives_path):
    """Load images from separate positive and negative folders."""
    images = []
    labels = []
    
    # Load negative images (label 0)
    if os.path.exists(negatives_path):
        for img_file in os.listdir(negatives_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(negatives_path, img_file)
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                images.append(img)
                labels.append(0)  # Negative class
    
    # Load positive images (label 1)
    if os.path.exists(positives_path):
        for img_file in os.listdir(positives_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(positives_path, img_file)
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                images.append(img)
                labels.append(1)  # Positive class
    
    if len(images) == 0:
        return None, None
    
    images = np.array(images, dtype='float32') / 255.0
    labels = keras.utils.to_categorical(labels, NUM_CLASSES)
    
    return images, labels

def preprocess_frame(frame):
    """Preprocess a frame for prediction."""
    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def convert_model_to_tflite(model, X_train, model_folder):
    """
    Converts a trained Keras model to TensorFlow Lite format with quantization.
    Saves both the original .keras model and the converted .tflite model.
    
    @param model: The trained Keras model
    @param X_train: Training data for representative dataset (for quantization)
    @param model_folder: Folder path where models will be saved
    """
    try:
        # Ensure model folder exists
        os.makedirs(model_folder, exist_ok=True)
        
        # Save the original Keras model
        keras_model_path = os.path.join(model_folder, 'model.keras')
        model.save(keras_model_path)
        print(f"Keras model saved to: {keras_model_path}")
        
        # Create inference model (remove data_augmentation layer if it exists)
        inference_layers = [l for l in model.layers if l.name != "data_augmentation"]
        inference_model = tf.keras.Sequential(inference_layers)
        
        # Build the inference model with sample data
        _ = inference_model(tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], tf.float32))
        
        # Define representative dataset for quantization
        def representative_dataset():
            for i in range(min(200, len(X_train))):
                x = X_train[i:i+1].astype(np.float32)
                # Data should already be normalized (0..1) from preprocessing
                yield [x]
        
        # Convert to TFLite with quantization
        converter = tf.lite.TFLiteConverter.from_keras_model(inference_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the TFLite model
        tflite_model_path = os.path.join(model_folder, 'model.tflite')
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to: {tflite_model_path}")
        
        # Print model details
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        in0 = interpreter.get_input_details()[0]
        out0 = interpreter.get_output_details()[0]
        
        print("\n=== TFLite MODEL INPUT ===")
        print(f"name: {in0['name']}")
        print(f"shape: {in0['shape']}")
        print(f"dtype: {in0['dtype']}")
        print(f"quantization (scale, zero_point): {in0['quantization']}")
        
        print("\n=== TFLite MODEL OUTPUT ===")
        print(f"name: {out0['name']}")
        print(f"shape: {out0['shape']}")
        print(f"dtype: {out0['dtype']}")
        print(f"quantization (scale, zero_point): {out0['quantization']}")
        
        return {
            'success': True,
            'keras_model': keras_model_path,
            'tflite_model': tflite_model_path
        }
        
    except Exception as e:
        print(f"Error converting model to TFLite: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

# in the future, multiple paths will be nesecery. a list of paths may be the implementation later on.
# functions are declared above and used here
if __name__ == "__main__":
    # Check for required arguments
    if len(sys.argv) < 5:
        print("Error: Missing required arguments")
        print("Usage: python train_model.py <positives_path> <negatives_path> <model_path> <epochs>")
        sys.exit(1)
    
    # Get command line arguments
    positives_path = sys.argv[1]
    negatives_path = sys.argv[2]
    model_path = sys.argv[3]
    epochs = int(sys.argv[4])
    
    print(f"Loading positive images from: {positives_path}")
    print(f"Loading negative images from: {negatives_path}")
    print(f"Model will be saved to: {model_path}")
    print(f"Training for {epochs} epochs")
    
    # Load and prepare images
    X_train, y_train = load_images_from_folders(positives_path, negatives_path)
    
    if X_train is None or len(X_train) == 0:
        print("Error: No images found or failed to load images")
        sys.exit(1)
    
    print(f"Loaded {len(X_train)} images")
    
    # Create the model
    model = create_cnn_model()
    print("Model created successfully")
    
    # Train the model
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Get final metrics
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    # Get the model folder from model_path
    model_folder = os.path.dirname(model_path)
    
    # Convert model to TFLite and save both formats
    print("Converting model to TFLite format...")
    conversion_result = convert_model_to_tflite(model, X_train, model_folder)
    
    if not conversion_result['success']:
        print(f"Warning: TFLite conversion failed: {conversion_result['error']}")
    else:
        print("Model conversion completed successfully")
    
    # Print final metrics in a format easy to parse for the UI
    print(f"FINAL_LOSS:{final_loss}")
    print(f"FINAL_ACCURACY:{final_accuracy}")
    print(f"FINAL_VAL_LOSS:{final_val_loss}")
    print(f"FINAL_VAL_ACCURACY:{final_val_accuracy}")

