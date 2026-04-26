import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import torch
from torchvision import transforms
from PIL import Image
from pt_model import RepVGGClassifier
import os

def predict(model_path, img_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Pytorch model
        if model_path.endswith('.ckpt'):
            # Prep model for evaluation
            model = RepVGGClassifier.load_from_checkpoint(model_path)
            model.eval()
            model.freeze();

            # Preprocess pipeline
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Grabs the image and applys preprocessing
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0)

            # Prediction process
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, 1)

                # Create results json
                result = {
                    "success": True,
                    "label": f"Class {int(pred.item())}",
                    "confidence": float(conf.item())
                }
                print(json.dumps(result))

        # Tensorflow model
        elif model_path.endswith('.keras'):
            # Load model
            model = tf.keras.models.load_model(model_path)

            # Prepares the image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Run prediction
            preds = model.predict(img_array, verbose=0)
            best_class = np.argmax(preds[0])
            confidence = float(preds[0][best_class])

            # Create results json
            result = {
                "success": True,
                "label": f"Class {best_class}",
                "confidence": confidence
            }
            print(json.dumps(result))

        # Should not happen but a safety
        else:
            print(json.dumps({"success": False, "error": "Unrecognized Type"}))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"success": False, "error": "Missing arguments"}))
    else:
        predict(sys.argv[1], sys.argv[2])