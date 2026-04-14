"""
Author: Armando Vega
Date Created: 8 April 2026

Last Modified By: Armando Vega
Date Last Modified: 8 April 2026

Description: 

--- CONSOLE TESTING ---
Usage:
    python pt_train_model.py '<labels_json>' <model_output_path> <epochs> '<pretrained_model_name>'

Arguments:

    labels_json       - JSON string mapping label names to arrays of folder paths.
                        Each folder should contain .jpg / .jpeg / .png image files.

    model_output_path - Full path where model.onnx, tf_out, (and model.tflite) will be saved.
                        The parent directory will be created if it does not exist.

    epochs            - Integer number of training epochs.

    pretrained_model_name - Name of the pre-trained model to use as backbone (e.g. "repvgg_a2", "resnet18", etc.)


Example (macOS / Linux) — replace the paths with your own folders:
    python pt_train_model.py '{"Apple": ["/Users/you/images/apples"], "Orange": ["/Users/you/images/oranges"]}' /Users/you/models/fruit/model.onnx 10 "repvgg_a2"

    python pt_train_model.py '{"Sunflowers": ["/Users/alexangeloorozco/Documents/PareidoliaApp/datasets/Flowers/positives"], "Not Sunflowers": ["/Users/alexangeloorozco/Documents/PareidoliaApp/datasets/Flowers/negatives", "/Users/alexangeloorozco/Documents/PareidoliaApp/datasets/Orange/positives"]}' /Users/alexangeloorozco/Documents/PareidoliaApp/models/Round/models 3 "repvgg_a2"

Example (Windows) — use double-quotes around the JSON and escape inner quotes:
    python pt_train_model.py  "{\"Apple\": [\"C:/images/apples\"], \"Orange\": [\"C:/images/oranges\"]}" C:/models/fruit/model.onnx 10 "repvgg_a2"

Multiple folders per label are supported:
    python pt_train_model.py '{"Apple": ["/path/batch1", "/path/batch2"], "Orange": ["/path/oranges"]}' /Users/you/models/fruit/model.onnx 20 "repvgg_a2"
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
    """
    Runs a converter-related subprocess command, prints stdout/stderr so
    conversion logs are visible, and raises if the command fails.
    """
    print("[Conversion] Running:", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    result.check_returncode()

def get_converter_python_path():
    """
    Resolves the Python path for the converter environment.
    Uses PAREIDOLIA_CONVERTER_PYTHON when set; otherwise builds the default
    platform-specific venv path from the inferred Pareidolia root.
    """
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
    """
    Checks whether the converter environment is ready by confirming the Python
    executable exists and that all pinned converter requirements are installed
    with exact versions.
    """
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
    """
    Ensures a valid converter virtual environment exists.
    Reuses an existing ready environment, otherwise creates the venv, installs
    pinned converter dependencies, validates readiness, and returns its Python.
    """
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
    
def rep_test(loader):
    tf, _, _, _ = import_tensorflow_keras()
    # mean, std = compute_mean_std_welford_fast_test(loader)
    # print("Calculated mean:", mean)
    # print("Calculated std:", std)
    MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    for idx, (image, _) in enumerate(loader):
        # image is [batch, 3, 224, 224] from PyTorch DataLoader
        # Process each image in the batch
        for i in range(image.shape[0]):
            img = image[i]  # [3, 224, 224]
            img = tf.convert_to_tensor(img.permute(1, 2, 0).numpy())  # → [224, 224, 3]
            img = tf.image.resize(img, [256, 256])                     # resize to 256x256
            img = tf.image.resize_with_crop_or_pad(img, 224, 224)     # center crop to 224x224
            img = tf.cast(img, tf.float32) 
            # / 255.0                    # normalize [0, 255] → [0, 1]
            img = (img - MEAN) / STD                                  # apply ImageNet mean/std
            img = tf.expand_dims(img, axis=0)                         # add batch dim [1, 224, 224, 3]
            yield [img]

def compute_mean_std_welford_fast_test(loader, image_size=256, crop_size=224):
    """Vectorized Welford — updates per image batch of pixels, not per pixel."""
    tf, _, _, _ = import_tensorflow_keras()

    n     = 0
    mean  = np.zeros(3, dtype=np.float64)
    M2    = np.zeros(3, dtype=np.float64)

    for i, (image, _) in enumerate(loader):
        img = tf.convert_to_tensor(image[0].permute(1, 2, 0).numpy())  # → [H, W, C]
        img = tf.image.resize(img, [image_size, image_size], method=tf.image.ResizeMethod.BICUBIC)
        img = tf.image.resize_with_crop_or_pad(img, crop_size, crop_size)
        img = tf.cast(img, tf.float32) 
        # / 255.0

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

def tf_to_tflite(tf_model_path, tflite_model_path, loader):
    tf, _, _, _ = import_tensorflow_keras()

    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.representative_dataset = representative_data_gen
        converter.representative_dataset = lambda: rep_test(loader)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32
        tflite_model = converter.convert()

        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)

        print(f"TFLite model saved to: {tflite_model_path}")
        return {
            'success': True,
            'tflite_model': tflite_model_path
        }
    except Exception as e:
        print(f"Error converting TF to TFLite: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
class RepVGGWithPreprocess(pl.LightningModule):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = x / 255.0
        x = (x - self.mean) / self.std
        logits = self.model(x)
        return torch.softmax(logits, dim=1)

# Functions are declared above and used here
if __name__ == "__main__":
    # Check for required arguments
    # Usage: python pt_train_model.py <labels_json> <model_path> <epochs> <pretrained_model_name>
    #   labels_json - JSON string mapping label names to arrays of folder paths
    #                 e.g. '{"Apple": ["/path/to/apples"], "Orange": ["/path/a", "/path/b"]}'
    if len(sys.argv) < 5:
        print("Error: Missing required arguments")
        print('Usage: python pt_train_model.py <labels_json> <model_path> <epochs> <pretrained_model_name>')
        sys.exit(1)
    
    # Get command line arguments
    labels_json_str = sys.argv[1]
    model_folder = sys.argv[2]
    epochs = int(sys.argv[3])
    pretrained_model_name = sys.argv[4]

    labels_json = json.loads(labels_json_str)
    
    print(json.dumps(labels_json_str, indent=2))

    print(f"Model will be saved to folder: {model_folder}")
    print(f"Training for {epochs} epochs")

    pl.seed_everything(77, workers=True)

    data_module = ImageDataModule(
        data_dir="./data",
        batch_size=64,
        img_size=224,
        num_workers=0,
        val_split=0.2,
        seed=77,
        cifar10=False,
        labels_json=labels_json_str,
    )
    
    # Load and prepare images from the JSON label map
    model = RepVGGClassifier(
        model_name=pretrained_model_name,
        num_classes=len(labels_json),    # CIFAR-10
        lr=3e-4,
        hidden_dim=512,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="repvgg-{epoch:02d}-{val_acc:.4f}",
    )

    early_stop_cb = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=5,
    )

    lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")

    csv_logger = CSVLogger("logs", name="repvgg")

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
    # Patch forward for export
    # original_forward = model.forward
    # model.forward = lambda x: torch.softmax(original_forward(x), dim=1)
    # mean, std = compute_mean_std_welford_fast_test(data_module.representative_dataloader())
    wrapped_model = RepVGGWithPreprocess(model)
    wrapped_model.to(trainer.strategy.root_device)


    print("Converting model to ONNX format...")
    onnx_conversion_result = convert_pt_to_onnx(wrapped_model, model_folder)
    if not onnx_conversion_result['success']:
        print(f"Error: ONNX conversion failed: {onnx_conversion_result['error']}")
        sys.exit(1)
    print("ONNX model conversion completed successfully")
    
    onnx_model_path = onnx_conversion_result.get('onnx_model')
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
    tf_to_tflite_result = tf_to_tflite(tf_model_path, os.path.join(model_folder, "model.tflite"), data_module.representative_dataloader())
    if not tf_to_tflite_result['success']:
        print(f"Error: TF to TFLite conversion failed: {tf_to_tflite_result['error']}")
        sys.exit(1)
    # tflite_conversion_result = finalize_tflite_output(tf_model_path, model_folder, onnx_model_path)
    # if not tflite_conversion_result['success']:
    #     print(f"Error: TFLite output finalization failed: {tflite_conversion_result['error']}")
    #     sys.exit(1)
    print("TFLite model conversion completed successfully")

    print("Number of classes: ", len(labels_json))
    
    # tflite_model_path = tflite_conversion_result.get('tflite_model')
    # print(f"TFLite model path: {tflite_model_path}")
