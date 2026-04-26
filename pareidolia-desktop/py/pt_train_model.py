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
import copy
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
import tempfile

# Model constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
EXPORT_IMG_HEIGHT = 256
EXPORT_IMG_WIDTH = 256
IMG_CHANNELS = 3
REPRESENTATIVE_SAMPLE_COUNT = 128
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


def delete_existing_checkpoints(model_folder):
    """
    Removes any existing Lightning checkpoint files so each training run starts
    with a clean checkpoint directory and retains only the new best checkpoint.
    """
    for ckpt_path in Path(model_folder).glob("*.ckpt"):
        try:
            ckpt_path.unlink()
            print(f"Deleted old checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"Warning: could not delete {ckpt_path}: {e}")


def run_logged_subprocess(command, env=None, cwd=None):
    """
    Runs a converter-related subprocess command, prints stdout/stderr so
    conversion logs are visible, and raises if the command fails.
    """
    print("[Conversion] Running:", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, env=env, cwd=cwd)
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


def ensure_onnx2tf_test_image_data(working_dir):
    """
    onnx2tf attempts to load a local dummy image npy from the current working
    directory before falling back to a network download. Create a valid local
    file so conversion remains deterministic in restricted environments.
    """
    os.makedirs(working_dir, exist_ok=True)
    sample_path = os.path.join(
        working_dir,
        "calibration_image_sample_data_20x128x128x3_float32.npy",
    )
    if os.path.exists(sample_path):
        return sample_path

    sample_images = np.random.default_rng(42).random(
        (20, 128, 128, 3),
        dtype=np.float32,
    )
    np.save(sample_path, sample_images)
    print(f"[Conversion] Wrote local onnx2tf test data: {sample_path}")
    return sample_path


def wrap_tf_saved_model_with_preprocess(base_saved_model_path, wrapped_saved_model_path, converter_python=None):
    """
    Wraps the converted TensorFlow SavedModel with TF-native resize and center
    crop preprocessing. This avoids asking onnx2tf to translate the crop path
    from ONNX while still producing a final model that owns the full
    preprocessing contract.
    """
    if sys.platform == "darwin":
        worker_script = str(Path(__file__).resolve().with_name("tf_wrap_saved_model_worker.py"))
        env = os.environ.copy()
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
        env.setdefault("TF_NUM_INTEROP_THREADS", "1")
        env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        run_logged_subprocess(
            [
                converter_python or sys.executable,
                worker_script,
                "--base-saved-model",
                base_saved_model_path,
                "--output",
                wrapped_saved_model_path,
            ],
            env=env,
        )
        return wrapped_saved_model_path

    tf, _, _, _ = import_tensorflow_keras()

    if os.path.exists(wrapped_saved_model_path):
        shutil.rmtree(wrapped_saved_model_path)

    base_model = tf.saved_model.load(base_saved_model_path)
    serving_fn = base_model.signatures["serving_default"]
    input_key = next(iter(serving_fn.structured_input_signature[1].keys()))
    output_key = next(iter(serving_fn.structured_outputs.keys()))

    class WrappedModule(tf.Module):
        @tf.function(
            input_signature=[
                tf.TensorSpec(
                    shape=[1, None, None, IMG_CHANNELS],
                    dtype=tf.float32,
                    name="input",
                )
            ]
        )
        def serving_default(self, x):
            x = tf.image.resize(
                x,
                [EXPORT_IMG_HEIGHT, EXPORT_IMG_WIDTH],
                method=tf.image.ResizeMethod.BILINEAR,
            )
            x = tf.image.resize_with_crop_or_pad(x, IMG_HEIGHT, IMG_WIDTH)
            outputs = serving_fn(**{input_key: x})
            return {"output": outputs[output_key]}

    wrapped_model = WrappedModule()
    tf.saved_model.save(
        wrapped_model,
        wrapped_saved_model_path,
        signatures={"serving_default": wrapped_model.serving_default},
    )
    print(f"[Conversion] Wrapped TensorFlow model saved to: {wrapped_saved_model_path}")
    return wrapped_saved_model_path

def convert_pt_to_onnx(model, model_folder):
    try:
        os.makedirs(model_folder, exist_ok=True)

        # Export on CPU to avoid MPS/ONNX tracing device issues on macOS.
        model = copy.deepcopy(model).to("cpu")
        model.eval()

        model_device = next(model.parameters()).device
        dummy_input = torch.randn(
            1,
            IMG_CHANNELS,
            IMG_HEIGHT,
            IMG_WIDTH,
            device=model_device,
        )
        
        model.to_onnx(
            file_path=os.path.join(model_folder, "model.onnx"),
            input_sample=dummy_input,
            export_params=True,
            opset_version=17,
            dynamo=False,
            input_names=['input'],
            output_names=['output'],
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
    """
    Deprecated: this function was an initial test for building representative samples directly as a generator, 
    but was found to cause TF/PyTorch runtime lock contention issues on macOS when both frameworks are used in 
    the same process. The current implementation materializes the representative samples into a numpy array in a 
    separate function, and on macOS the TFLite conversion is done in a separate subprocess to avoid these issues.
    """
    tf, _, _, _ = import_tensorflow_keras()

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

def build_representative_samples(loader, max_samples=REPRESENTATIVE_SAMPLE_COUNT):
    """
    Materializes representative calibration samples as NHWC float32 arrays in
    the exported model's input domain.

    The representative loader already applies the same spatial resize used by
    training/evaluation, so calibration should consume raw 0..255 pixels here
    and let the exported model wrapper handle normalization.
    """
    samples = []

    for image_batch, _ in loader:
        batch_size = image_batch.shape[0]
        for i in range(batch_size):
            img = image_batch[i].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
            samples.append(img)

            if len(samples) >= max_samples:
                return np.stack(samples, axis=0).astype(np.float32)

    if not samples:
        raise RuntimeError("Representative dataset is empty; cannot run TFLite quantization.")

    return np.stack(samples, axis=0).astype(np.float32)

def compute_mean_std_welford_fast_test(loader, image_size=256, crop_size=224):
    """
    Deprecated: this function was the initial welford's algorithm implementation for calculating mean and std for the representative dataset,
    but will be used in the future for a user if they build their own model with their own dataset.

    Vectorized Welford — updates per image batch of pixels, not per pixel. Used a dataloader instead of file paths. Creates threading issues on MacOS.
    """
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
    """
    Deprecated: this function was the initial welford's algorithm implementation for calculating mean and std for the representative dataset,
    but will be used in the future for a user if they build their own model with their own dataset.

    Vectorized Welford — updates per image batch of pixels, not per pixel.
    """
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
        conversion_workdir = os.path.dirname(tf_model_path)
        ensure_onnx2tf_test_image_data(conversion_workdir)
        print("[Conversion] ONNX model path:", onnx_model_path)
        print("[Conversion] onnx2tf output folder:", tf_model_path)

        converter_bin_dir = str(Path(converter_python).parent)
        env = os.environ.copy()
        env["PATH"] = converter_bin_dir + os.pathsep + env.get("PATH", "")

        onnx2tf_base_command = [
            converter_python,
            "-m",
            "onnx2tf",
            "-i",
            onnx_model_path,
            "-o",
            tf_model_path,
            "-v",
            "warn",
            "-ois",
            f"input:1,{IMG_CHANNELS},{IMG_HEIGHT},{IMG_WIDTH}",
        ]

        onnx2tf_command = list(onnx2tf_base_command)

        # macOS-specific stability workaround: skip onnx optimizer subprocess
        if sys.platform == "darwin":
            onnx2tf_command.append("-nuo")

        try:
            run_logged_subprocess(onnx2tf_command, env=env, cwd=conversion_workdir)
        except Exception:
            auto_json_path = os.path.join(tf_model_path, "model_auto.json")
            if not os.path.exists(auto_json_path):
                raise

            print(f"[Conversion] Retrying with auto-generated parameter replacement: {auto_json_path}")
            retry_command = list(onnx2tf_base_command) + ["-prf", auto_json_path]
            if sys.platform == "darwin":
                retry_command.append("-nuo")
            run_logged_subprocess(retry_command, env=env, cwd=conversion_workdir)

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

def tf_to_tflite(tf_model_path, tflite_model_path, representative_samples, converter_python):
    """
    Checks for the current OS and creates a subprocess to run the TF to TFLite conversion with the representative dataset, 
    passing the representative samples via a temporary file on disk for macOS to avoid TF/PyTorch runtime lock contention issues. 
    
    On non-macOS platforms, it runs the conversion directly in-process since these issues are not present.
    """
    representative_path = None
    try:
        os.makedirs(os.path.dirname(tflite_model_path), exist_ok=True)

        if sys.platform == "darwin":
            # macOS-only isolation to avoid TF/PyTorch runtime lock contention.
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".npy",
                prefix="representative_",
                dir=os.path.dirname(tflite_model_path),
                delete=False,
            ) as temp_file:
                representative_path = temp_file.name

            np.save(representative_path, representative_samples)

            worker_script = str(Path(__file__).resolve().with_name("tflite_convert_worker.py"))
            env = os.environ.copy()
            env.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
            env.setdefault("TF_NUM_INTEROP_THREADS", "1")
            env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
            env.setdefault("OMP_NUM_THREADS", "1")

            run_logged_subprocess(
                [
                    converter_python,
                    worker_script,
                    "--saved-model",
                    tf_model_path,
                    "--output",
                    tflite_model_path,
                    "--rep-data",
                    representative_path,
                ],
                env=env,
            )
        else: # Windows route and fallback for non-macOS
            tf, _, _, _ = import_tensorflow_keras()

            def representative_dataset():
                for sample in representative_samples:
                    yield [np.expand_dims(sample.astype(np.float32), axis=0)]

            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
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
    finally:
        if representative_path and os.path.exists(representative_path):
            try:
                os.remove(representative_path)
            except OSError:
                print(f"Could not delete temporary representative data file: {representative_path}")

class RepVGGWithPreprocess(pl.LightningModule):
    """
    Model wrapper that applies the exported model's normalization before
    forwarding to the base model.

    The exported model expects RGB images that have already been center-cropped
    to 224x224 and are provided as raw 0..255 pixel values in NCHW layout.
    """
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
        num_classes=len(labels_json),
        lr=3e-4,
        hidden_dim=512,
    )

    delete_existing_checkpoints(model_folder)

    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath=model_folder,
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

    best_model_path = checkpoint_cb.best_model_path
    if best_model_path:
        print(f"Loading best checkpoint for export: {best_model_path}")
        model = RepVGGClassifier.load_from_checkpoint(best_model_path)
    else:
        print("No best checkpoint found; exporting the current in-memory model.")

    model.eval()

    # wrap model to allow for preproccessing in onnx export
    wrapped_model = RepVGGWithPreprocess(model)

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

    tf_inner_model_path = os.path.join(model_folder, "tf_out_inner")
    tf_wrapped_model_path = os.path.join(model_folder, "tf_out")

    print("Converting ONNX model to TF format...")
    tf_conversion_result = convert_onnx_to_tf(
        onnx_model_path,
        tf_inner_model_path,
        converter_python
    )
    if not tf_conversion_result['success']:
        print(f"Error: ONNX to TF conversion failed: {tf_conversion_result['error']}")
        sys.exit(1)
    print("ONNX to TF conversion completed successfully")

    print("Wrapping TensorFlow model with resize and center crop preprocessing...")
    try:
        tf_model_path = wrap_tf_saved_model_with_preprocess(
            tf_conversion_result.get('tf_model'),
            tf_wrapped_model_path,
            converter_python,
        )
    except Exception as e:
        print(f"Error wrapping TF model with preprocessing: {str(e)}")
        sys.exit(1)
    
    print("Finalizing TFLite model output...")
    representative_loader = data_module.representative_dataloader()
    representative_samples = build_representative_samples(representative_loader)

    tf_to_tflite_result = tf_to_tflite(
        tf_model_path,
        os.path.join(model_folder, "model.tflite"),
        representative_samples,
        converter_python
    )
    if not tf_to_tflite_result['success']:
        print(f"Error: TF to TFLite conversion failed: {tf_to_tflite_result['error']}")
        sys.exit(1)

    print("TFLite model conversion completed successfully")

    print("Number of classes: ", len(labels_json))
