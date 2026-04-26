import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
if sys.platform == "darwin":
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import tensorflow as tf


IMG_HEIGHT = 224
IMG_WIDTH = 224
EXPORT_IMG_HEIGHT = 256
EXPORT_IMG_WIDTH = 256
IMG_CHANNELS = 3


def main():
    parser = argparse.ArgumentParser(description="Wrap a TF SavedModel with resize and center crop preprocessing.")
    parser.add_argument("--base-saved-model", required=True, help="Path to the converted base SavedModel directory.")
    parser.add_argument("--output", required=True, help="Path to the wrapped SavedModel directory.")
    args = parser.parse_args()

    if sys.platform == "darwin":
        try:
            tf.config.threading.set_inter_op_parallelism_threads(int(os.environ.get("TF_NUM_INTEROP_THREADS", "1")))
            tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("TF_NUM_INTRAOP_THREADS", "1")))
        except Exception:
            pass

    if os.path.exists(args.output):
        import shutil
        shutil.rmtree(args.output)

    base_model = tf.saved_model.load(args.base_saved_model)
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
        args.output,
        signatures={"serving_default": wrapped_model.serving_default},
    )

    print(f"Wrapped TF SavedModel written: {args.output}")


if __name__ == "__main__":
    main()
