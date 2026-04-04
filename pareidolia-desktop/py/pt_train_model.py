"""
Created by Aleaxngelo Orozco Gutierrez on 2-10-2026

Currently is supposed to use file paths to get image training data and export a working model
However, until a virtual envoirment is made with the nseecery packages, it currently returns an error.
Future work will allow the python function to execute successfully and allow flexible model creation options

--- CONSOLE TESTING ---
Usage:
    python train_model.py '<labels_json>' <model_output_path> <epochs>

Arguments:
    labels_json       - JSON string mapping label names to arrays of folder paths.
                        Each folder should contain .jpg / .jpeg / .png image files.
    model_output_path - Full path where model.keras (and model.tflite) will be saved.
                        The parent directory will be created if it does not exist.
    epochs            - Integer number of training epochs.

Example (macOS / Linux) — replace the paths with your own folders:
    python train_model.py '{"Apple": ["/Users/you/images/apples"], "Orange": ["/Users/you/images/oranges"]}' /Users/you/models/fruit/model.keras 10

    python train_model.py '{"Sunflowers": ["/Users/alexangeloorozco/Documents/PareidoliaApp/datasets/Flowers/positives"], "Not Sunflowers": ["/Users/alexangeloorozco/Documents/PareidoliaApp/datasets/Flowers/negatives", "/Users/alexangeloorozco/Documents/PareidoliaApp/datasets/Orange/positives"]}' /Users/alexangeloorozco/Documents/PareidoliaApp/models/Round/models 3

Example (Windows) — use double-quotes around the JSON and escape inner quotes:
    python train_model.py "{\"Apple\": [\"C:/images/apples\"], \"Orange\": [\"C:/images/oranges\"]}" C:/models/fruit/model.keras 10

Multiple folders per label are supported:
    python train_model.py '{"Apple": ["/path/batch1", "/path/batch2"], "Orange": ["/path/oranges"]}' /Users/you/models/fruit/model.keras 20
-----------------------
"""
import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import timm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import math
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from .pt_model import RepVGGClassifier
from .image_data_module import ImageDataModule
from .epoch_history_printer import EpochHistoryPrinter

# Model constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
# NUM_CLASSES is now determined dynamically from the labels JSON at runtime
LEARNING_RATE = 0.001

def create_cnn_model(num_classes):
    """Creates a CNN model for image classification.
    
    @param num_classes: Number of output classes, determined from labels JSON
    """
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
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_images_from_json(labels_json):
    """Load images from a JSON mapping of label names to arrays of folder paths.
    
    @param labels_json: JSON string (or dict) mapping label names to lists of folder paths.
                        Example: {"Apple": ["/path/to/folder1"], "Orange": ["/path/a", "/path/b"]}
    @returns: (images, labels, num_classes, label_names)
              images      - float32 numpy array normalized to [0, 1]
              labels      - one-hot encoded label array
              num_classes - number of unique labels found
              label_names - ordered list of label names (index matches one-hot position)
    """
    import json

    if isinstance(labels_json, str):
        labels_dict = json.loads(labels_json)
    else:
        labels_dict = labels_json

    label_names = list(labels_dict.keys())
    num_classes = len(label_names)

    images = []
    labels = []

    for label_index, label_name in enumerate(label_names):
        folder_paths = labels_dict[label_name]
        for folder_path in folder_paths:
            if not os.path.exists(folder_path):
                print(f"Warning: folder not found, skipping: {folder_path}")
                continue
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, img_file)

                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    images.append(img)
                    labels.append(label_index)

    if len(images) == 0:
        return None, None, 0, []

    images = np.array(images, dtype='float32') / 255.0
    labels = keras.utils.to_categorical(labels, num_classes)

    return images, labels, num_classes, label_names

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

# Functions are declared above and used here
if __name__ == "__main__":
    # Check for required arguments
    # Usage: python train_model.py <labels_json> <model_path> <epochs>
    #   labels_json - JSON string mapping label names to arrays of folder paths
    #                 e.g. '{"Apple": ["/path/to/apples"], "Orange": ["/path/a", "/path/b"]}'
    if len(sys.argv) < 5:
        print("Error: Missing required arguments")
        print('Usage: python train_model.py <labels_json> <model_path> <epochs> <pretrained_model_name>')
        sys.exit(1)
    
    # Get command line arguments
    labels_json_str = sys.argv[1]
    model_folder = sys.argv[2]
    epochs = int(sys.argv[3])
    pretrained_model_name = sys.argv[4]
    
    print(f"Model will be saved to folder: {model_folder}")
    print(f"Training for {epochs} epochs")

    pl.seed_everything(42, workers=True)

    data_module = ImageDataModule(
        data_dir="./data",
        batch_size=64,
        img_size=224,
        num_workers=0,
        val_split=0.2,
        seed=42,
        cifar10=False,
        labels_json=labels_json_str,
    )
    
    # Load and prepare images from the JSON label map
    model = RepVGGClassifier(
        model_name=pretrained_model_name,
        num_classes=10,    # CIFAR-10
        lr=3e-4,
        hidden_dim=512,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="repvgg-a2-cifar10-{epoch:02d}-{val_acc:.4f}",
    )

    early_stop_cb = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=5,
    )

    lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")

    csv_logger = CSVLogger("logs", name="repvgg_a2_cifar10")

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor_cb, RichProgressBar(refresh_rate=1), EpochHistoryPrinter()],
        deterministic=True,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        logger=csv_logger,
    )
    model = model.to(trainer.strategy.root_device)
    
    # Get final metrics
    # final_loss = history.history['loss'][-1]
    # final_accuracy = history.history['accuracy'][-1]
    # final_val_loss = history.history['val_loss'][-1]
    # final_val_accuracy = history.history['val_accuracy'][-1]
    
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

