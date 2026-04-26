import sys
import json
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from train_model import load_images_from_json

def evaluate(model_path, labels_json_str):
    try:
        # Check if file path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError("Invalid File Path")

        # load keras file
        model = tf.keras.models.load_model(model_path)

        # recreate test split
        X_all, y_all, NUM_CLASSES, label_names = load_images_from_json(labels_json_str);
        if X_all is None or len(X_all) == 0:
            print("Error: No images found or failed to load images")
            sys.exit(1)
        X_train, x_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=42)

        # Evaluate model and return results
        results = model.evaluate(x_test, y_test, verbose=0)

        print(json.dumps({
            "success": True,
            "accuracy": float(results[1]),
            "loss": float(results[0]),
            "total_images": len(x_test)
        }))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__  == "__main__":
    evaluate(sys.argv[1], sys.argv[2])