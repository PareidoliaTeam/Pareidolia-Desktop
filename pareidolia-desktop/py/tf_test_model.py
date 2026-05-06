import sys
import json
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from train_model import load_images_from_json, MODEL_TYPE_PRETRAINED, MODEL_TYPE_SCRATCH


def load_tf_model(model_path):
    """Load TensorFlow models, with a fallback for older MobileNetV2 Lambda saves."""
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
            custom_objects={
                "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
            },
        )

def evaluate(model_path, labels_json_str, project_type=MODEL_TYPE_SCRATCH):
    try:
        # Check if file path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError("Invalid File Path")

        # load keras file
        model = load_tf_model(model_path)
        model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # recreate test split
        model_type = MODEL_TYPE_PRETRAINED if project_type == MODEL_TYPE_PRETRAINED else MODEL_TYPE_SCRATCH
        X_all, y_all, NUM_CLASSES, label_names = load_images_from_json(labels_json_str, model_type)
        if X_all is None or len(X_all) == 0:
            raise ValueError("No images found or failed to load images")

        output_shape = model.output_shape
        model_classes = None
        if isinstance(output_shape, (tuple, list)) and output_shape:
            model_classes = output_shape[-1]

        if model_classes is not None and int(model_classes) != int(NUM_CLASSES):
            raise ValueError(
                "Model class count mismatch: "
                f"the loaded model outputs {int(model_classes)} classes, "
                f"but the current dataset defines {int(NUM_CLASSES)} labels. "
                "Retrain the model with the current label set, or restore the label set "
                "that was used when this model was created."
            )

        X_train, x_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

        # Evaluate model and return results
        results = model.evaluate(x_test, y_test, verbose=0)

        print(json.dumps({
            "success": True,
            "accuracy": float(results[1]),
            "loss": float(results[0]),
            "total_images": len(x_test),
            "model_classes": int(model_classes) if model_classes is not None else None,
            "dataset_classes": int(NUM_CLASSES)
        }))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__  == "__main__":
    project_type = sys.argv[3] if len(sys.argv) > 3 else MODEL_TYPE_SCRATCH
    evaluate(sys.argv[1], sys.argv[2], project_type)
