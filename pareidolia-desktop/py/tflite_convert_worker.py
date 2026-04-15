import argparse
import os
import sys
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
if sys.platform == "darwin":
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import tensorflow as tf


def representative_dataset(samples):
    for sample in samples:
        yield [np.expand_dims(sample.astype(np.float32), axis=0)]


def main():
    parser = argparse.ArgumentParser(description="Convert TF SavedModel to TFLite with quantization.")
    parser.add_argument("--saved-model", required=True, help="Path to TensorFlow SavedModel directory.")
    parser.add_argument("--output", required=True, help="Path to output .tflite file.")
    parser.add_argument("--rep-data", required=True, help="Path to representative .npy samples (NHWC float32).")
    args = parser.parse_args()

    samples = np.load(args.rep_data).astype(np.float32)
    if samples.ndim != 4:
        raise ValueError(f"Representative data must be rank-4 NHWC, got shape {samples.shape}")

    if sys.platform == "darwin":
        try:
            tf.config.threading.set_inter_op_parallelism_threads(int(os.environ.get("TF_NUM_INTEROP_THREADS", "1")))
            tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("TF_NUM_INTRAOP_THREADS", "1")))
        except Exception:
            pass

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(samples)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite model written: {args.output}")


if __name__ == "__main__":
    main()
