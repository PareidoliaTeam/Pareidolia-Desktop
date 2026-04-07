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
import json
import sys
import os
import shutil
import numpy as np
import cv2
import timm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import math
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from pt_model import RepVGGClassifier
from image_data_module import ImageDataModule
from epoch_history_printer import EpochHistoryPrinter
from pathlib import Path
import glob
import subprocess

# Model constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
# NUM_CLASSES is now determined dynamically from the labels JSON at runtime
LEARNING_RATE = 0.001
CONVERTER_VENV_NAME = "converter-venv-macos-tf219"
CONVERTER_REQUIREMENTS = [
    "tensorflow==2.19.1",
    "tf-keras==2.19.0",
    "onnx==1.16.2",
    "onnx2tf==1.28.8",
    "onnx-graphsurgeon==0.5.8",
    "sng4onnx==2.0.1",
    "ai-edge-litert==2.1.3",
    "psutil==7.2.2",
    "onnxruntime==1.24.4",
    "onnxsim==0.4.36",
]


def run_logged_subprocess(command, env=None):
    print("[Conversion] Running:", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    result.check_returncode()


def get_converter_python_path():
    converter_override = os.environ.get("PAREIDOLIA_CONVERTER_PYTHON")
    if converter_override:
        return converter_override

    current_python = Path(sys.executable).resolve()
    if current_python.parent.name == "bin":
        pareidolia_root = current_python.parent.parent.parent
    else:
        pareidolia_root = Path.home() / "Documents" / "PareidoliaApp"

    converter_venv = pareidolia_root / CONVERTER_VENV_NAME
    if os.name == "nt":
        return str(converter_venv / "Scripts" / "python.exe")
    return str(converter_venv / "bin" / "python")


def is_converter_ready(converter_python):
    if not os.path.exists(converter_python):
        return False

    result = subprocess.run(
        [converter_python, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False

    installed = {}
    for line in result.stdout.splitlines():
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        normalized = name.strip().lower().replace("_", "-")
        installed[normalized] = version.strip()

    for requirement in CONVERTER_REQUIREMENTS:
        req_name, req_version = requirement.split("==", 1)
        normalized_req = req_name.strip().lower().replace("_", "-")
        if installed.get(normalized_req) != req_version:
            return False

    return True


def ensure_converter_environment():
    converter_python = get_converter_python_path()
    if is_converter_ready(converter_python):
        print(f"[Conversion] Using existing converter python: {converter_python}")
        return converter_python

    converter_venv_dir = Path(converter_python).parent.parent
    print(f"[Conversion] Creating converter venv at: {converter_venv_dir}")
    run_logged_subprocess([sys.executable, "-m", "venv", str(converter_venv_dir)])

    run_logged_subprocess([converter_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run_logged_subprocess([converter_python, "-m", "pip", "install", *CONVERTER_REQUIREMENTS])

    if not is_converter_ready(converter_python):
        raise RuntimeError("Converter environment setup completed, but onnx2tf is still unavailable.")

    print(f"[Conversion] Converter ready: {converter_python}")
    return converter_python


def import_tensorflow_keras():
    """Lazy import TensorFlow/Keras to avoid macOS import-order crashes."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    return tf, keras, layers, models

def create_cnn_model(num_classes):
    """Creates a CNN model for image classification.
    
    @param num_classes: Number of output classes, determined from labels JSON
    """
    _, keras, layers, models = import_tensorflow_keras()

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

    _, keras, _, _ = import_tensorflow_keras()

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
    tf, keras, _, _ = import_tensorflow_keras()

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

def convert_pt_to_onnx(model, model_folder):
    try:
        os.makedirs(model_folder, exist_ok=True)

        dummy_input = torch.randn(1, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
        
        model.to_onnx(
            file_path=os.path.join(model_folder, "model.onnx"),
            input_sample=dummy_input,
            export_params=True,
            opset_version=17,
            dynamo=False,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
        print(f"Model exported to ONNX format at {os.path.join(model_folder, 'model.onnx')}")

        return {
            'success': True,
            'onnx_model': os.path.join(model_folder, "model.onnx")
            # 'tflite_model': tflite_model_path
        }
    
    except Exception as e:
        print(f"Error exporting model to ONNX: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
    
def verify_onnx_integrity(onnx_model_path):
    import onnx

    try:
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model integrity check passed.")
        return True
    except Exception as e:
        print(f"ONNX model integrity check failed: {str(e)}")
        return False
    
def compute_mean_std_welford_fast(image_paths, image_size=256, crop_size=224):
    """Vectorized Welford — updates per image batch of pixels, not per pixel."""
    tf, _, _, _ = import_tensorflow_keras()

    n     = 0
    mean  = np.zeros(3, dtype=np.float64)
    M2    = np.zeros(3, dtype=np.float64)

    for path in image_paths:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [image_size, image_size])
        img = tf.image.resize_with_crop_or_pad(img, crop_size, crop_size)
        img = tf.cast(img, tf.float32) / 255.0

        pixels = tf.reshape(img, [-1, 3]).numpy()    # [50176, 3]
        batch_n = len(pixels)

        # parallel Welford merge (Chan's parallel algorithm)
        batch_mean = pixels.mean(axis=0)
        batch_M2   = pixels.var(axis=0) * batch_n

        delta  = batch_mean - mean # delta between batch mean and current mean
        total  = n + batch_n       # total count of pixels after adding this batch

        mean  += delta * (batch_n / total) # adjust the running mean based on the current batch

        # squared difference between means measures how far apart the means are from each other
        # variance is based on the squared differences from the mean
        # 
        M2    += batch_M2 + delta**2 * (n * batch_n / total) # running total of squared differences from the mean, adjusted for the new batch
        n      = total # update total count of pixels

    std = np.sqrt(M2 / n) # variance is M2/n, std is sqrt of variance; this is std of the entire dataset after processing all batches

    return mean.astype(np.float32), std.astype(np.float32)

def representative_data_gen():
    tf, _, _, _ = import_tensorflow_keras()

    image_paths = glob.glob("calibration_images/cifar10/*.JPEG")
    mean, std = compute_mean_std_welford_fast(image_paths)
    print("Calculated mean:", mean)
    print("Calculated std:", std)
    assert len(image_paths) >= 100, f"Only {len(image_paths)} images found"

    for path in image_paths[:128]:
        img = tf.io.read_file(path)                            # read raw file bytes from disk
        img = tf.image.decode_jpeg(img, channels=3)            # decode to uint8 [H, W, 3], values 0-255
        img = tf.image.resize(img, [256, 256])                 # resize to 256x256, becomes float32
        img = tf.image.resize_with_crop_or_pad(img, 224, 224)  # center crop to 224x224 (crop_pct=0.875)
        img = tf.cast(img, tf.float32) / 255.0                 # normalize [0, 255] → [0, 1]
        img = (img - mean) / std                               # apply ImageNet mean/std, range ≈ [-2, +2]
        img = tf.expand_dims(img, axis=0)                      # add batch dim [224,224,3] → [1,224,224,3]
        yield [img]                                            # yield one sample to the converter

def convert_onnx_to_tf(onnx_model_path, tf_model_path, converter_python):
    try:
        if os.path.exists(tf_model_path):
            shutil.rmtree(tf_model_path)

        os.makedirs(os.path.dirname(tf_model_path), exist_ok=True)
        print("[Conversion] ONNX model path:", onnx_model_path)
        print("[Conversion] onnx2tf output folder:", tf_model_path)

        converter_bin_dir = str(Path(converter_python).parent)
        env = os.environ.copy()
        env["PATH"] = converter_bin_dir + os.pathsep + env.get("PATH", "")

        onnx2tf_command = [
            converter_python,
            "-m",
            "onnx2tf",
            "-i",
            onnx_model_path,
            "-o",
            tf_model_path,
            "-v",
            "warn",
        ]
        # macOS-specific stability workaround: skip onnx optimizer subprocess
        if sys.platform == "darwin":
            onnx2tf_command.append("-nuo")

        run_logged_subprocess(onnx2tf_command, env=env)

        print(f"ONNX model converted to TensorFlow format at: {tf_model_path}")
        return {
            'success': True,
            'tf_model': tf_model_path
        }
    except Exception as e:
        print(f"Error converting ONNX to TensorFlow: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


def collect_generated_tflite(tf_model_path, onnx_model_path):
    base_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
    preferred_files = [
        os.path.join(tf_model_path, f"{base_name}_float32.tflite"),
        os.path.join(tf_model_path, f"{base_name}_float16.tflite"),
        os.path.join(tf_model_path, f"{base_name}.tflite"),
    ]
    for candidate in preferred_files:
        if os.path.exists(candidate):
            return candidate

    fallback = sorted(glob.glob(os.path.join(tf_model_path, "*.tflite")))
    return fallback[0] if fallback else None


def finalize_tflite_output(tf_model_path, model_folder, onnx_model_path):
    try:
        generated_tflite = collect_generated_tflite(tf_model_path, onnx_model_path)
        if generated_tflite is None:
            raise FileNotFoundError(f"No .tflite file generated in {tf_model_path}")

        final_tflite_path = os.path.join(model_folder, "model.tflite")
        shutil.copy2(generated_tflite, final_tflite_path)
        print(f"Copied TFLite model to: {final_tflite_path}")
        return {
            'success': True,
            'tflite_model': final_tflite_path
        }
    except Exception as e:
        print(f"Error finalizing TFLite output: {str(e)}")
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
    
    print(json.dumps(labels_json_str, indent=2))

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

    trainer.fit(model, datamodule=data_module)

    model.eval()

    print("Converting model to ONNX format...")
    onnx_conversion_result = convert_pt_to_onnx(model, model_folder)
    if not onnx_conversion_result['success']:
        print(f"Error: ONNX conversion failed: {onnx_conversion_result['error']}")
        sys.exit(1)
    print("ONNX model conversion completed successfully")
    
    onnx_model_path = onnx_conversion_result.get('onnx_model')
    # print(f"ONNX model path: {onnx_model_path}")
    if onnx_model_path and os.path.exists(onnx_model_path):
        if verify_onnx_integrity(onnx_model_path):
            print("ONNX model is valid and ready for TFLite conversion.")
        else:
            print("ONNX model integrity check failed.")
            sys.exit(1)

    if sys.platform == "darwin":
        print("Preparing converter environment (macOS)...")
        try:
            converter_python = ensure_converter_environment()
        except Exception as e:
            print(f"Error setting up converter environment: {str(e)}")
            sys.exit(1)
    else:
        print("Using current Python environment for conversion (non-macOS).")
        converter_python = sys.executable

    print("Converting ONNX model to TF format...")
    tf_conversion_result = convert_onnx_to_tf(
        onnx_model_path,
        os.path.join(model_folder, "tf_out"),
        converter_python
    )
    if not tf_conversion_result['success']:
        print(f"Error: ONNX to TF conversion failed: {tf_conversion_result['error']}")
        sys.exit(1)
    print("ONNX to TF conversion completed successfully")
    
    print("Finalizing TFLite model output...")
    tf_model_path = tf_conversion_result.get('tf_model')
    tflite_conversion_result = finalize_tflite_output(tf_model_path, model_folder, onnx_model_path)
    if not tflite_conversion_result['success']:
        print(f"Error: TFLite output finalization failed: {tflite_conversion_result['error']}")
        sys.exit(1)
    print("TFLite model conversion completed successfully")
    
    tflite_model_path = tflite_conversion_result.get('tflite_model')
    print(f"TFLite model path: {tflite_model_path}")
    # Get final metrics
    # final_loss = history.history['loss'][-1]
    # final_accuracy = history.history['accuracy'][-1]
    # final_val_loss = history.history['val_loss'][-1]
    # final_val_accuracy = history.history['val_accuracy'][-1]
    
    # Convert model to TFLite and save both formats
    # print("Converting model to TFLite format...")
    # conversion_result = convert_model_to_tflite(model, X_train, model_folder)
    
    # if not conversion_result['success']:
    #     print(f"Warning: TFLite conversion failed: {conversion_result['error']}")
    # else:
    #     print("Model conversion completed successfully")
    
    # Print final metrics in a format easy to parse for the UI
    # print(f"FINAL_LOSS:{final_loss}")
    # print(f"FINAL_ACCURACY:{final_accuracy}")
    # print(f"FINAL_VAL_LOSS:{final_val_loss}")
    # print(f"FINAL_VAL_ACCURACY:{final_val_accuracy}")
