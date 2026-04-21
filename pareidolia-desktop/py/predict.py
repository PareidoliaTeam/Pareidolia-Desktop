import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

def predict(model_path, img_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Load model
        model = tf.keras.models.load_model(model_path)

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = model.predict(img_array, verbose=0)
        best_class = np.argmax(preds[0])
        confidence = float(preds[0][best_class])

        result = {
            "success": True,
            "label": f"Class {best_class}",
            "confidence": confidence
        }
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"success": False, "error": "Missing arguments"}))
    else:
        predict(sys.argv[1], sys.argv[2])