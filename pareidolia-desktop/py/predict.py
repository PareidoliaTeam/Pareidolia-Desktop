import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import torch
from torchvision import transforms
from PIL import Image
from pt_model import RepVGGClassifier, ScratchCNNClassifier
from train_model import MODEL_TYPE_PRETRAINED, MODEL_TYPE_SCRATCH, normalize_images_for_model
from pt_train_model import compute_mean_std_welford_from_loader
from image_data_module import ImageDataModule
import os
import contextlib

def load_pytorch_metadata(model_path):
    metadata_path = os.path.join(os.path.dirname(model_path), "model-metadata.pytorch.json")
    if not os.path.exists(metadata_path):
        return {}

    with open(metadata_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def get_label_names(labels_json_str, metadata=None):
    metadata = metadata or {}
    metadata_labels = metadata.get("label_names")
    if isinstance(metadata_labels, list) and metadata_labels:
        return [str(label) for label in metadata_labels]

    if not labels_json_str:
        return []

    labels = json.loads(labels_json_str)
    if isinstance(labels, dict):
        return list(labels.keys())
    if isinstance(labels, list):
        return [str(label) for label in labels]
    return []

def label_for_class(class_index, label_names):
    if 0 <= class_index < len(label_names):
        return label_names[class_index]
    return f"Class {class_index}"

def top_predictions(probabilities, label_names, limit=3):
    ranked_indices = np.argsort(probabilities)[::-1][:limit]
    return [
        {
            "label": label_for_class(int(class_index), label_names),
            "class_index": int(class_index),
            "confidence": float(probabilities[class_index])
        }
        for class_index in ranked_indices
    ]

def infer_pt_project_type_from_checkpoint(checkpoint_path, fallback=MODEL_TYPE_SCRATCH):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    keys = state_dict.keys()

    if any(key.startswith("backbone.") or key.startswith("head.") for key in keys):
        return MODEL_TYPE_PRETRAINED

    if any(key.startswith("model.") for key in keys):
        return MODEL_TYPE_SCRATCH

    return MODEL_TYPE_PRETRAINED if fallback == MODEL_TYPE_PRETRAINED else MODEL_TYPE_SCRATCH

def load_tf_model(model_path):
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

def predict(model_path, img_path, labels_json_str=None, project_type=MODEL_TYPE_SCRATCH):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Pytorch model
        if model_path.endswith('.ckpt'):
            metadata = load_pytorch_metadata(model_path)
            label_names = get_label_names(labels_json_str, metadata)
            project_type = metadata.get("project_type") or infer_pt_project_type_from_checkpoint(model_path, project_type)
            # Prep model for evaluation
            model_class = RepVGGClassifier if project_type == MODEL_TYPE_PRETRAINED else ScratchCNNClassifier
            model = model_class.load_from_checkpoint(model_path, map_location="cpu")
            model.eval()
            model.freeze()

            normalization_mean = metadata.get("normalization_mean") or [0.485, 0.456, 0.406]
            normalization_std = metadata.get("normalization_std") or [0.229, 0.224, 0.225]
            if project_type == MODEL_TYPE_SCRATCH and labels_json_str:
                with contextlib.redirect_stdout(sys.stderr):
                    data_module = ImageDataModule(
                        labels_json=labels_json_str,
                        batch_size=32,
                        img_size=224,
                        seed=42
                    )
                    stats_loader = data_module.normalization_stats_dataloader()
                    normalization_mean, normalization_std = compute_mean_std_welford_from_loader(stats_loader)

            # Preprocess pipeline
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=normalization_mean, std=normalization_std),
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
                class_index = int(pred.item())
                probabilities = probs[0].cpu().numpy()
                result = {
                    "success": True,
                    "label": label_for_class(class_index, label_names),
                    "class_index": class_index,
                    "confidence": float(conf.item()),
                    "top_predictions": top_predictions(probabilities, label_names)
                }
                print(json.dumps(result))

        # Tensorflow model
        elif model_path.endswith('.keras'):
            label_names = get_label_names(labels_json_str)
            # Load model
            model = load_tf_model(model_path)

            # Prepares the image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            model_type = MODEL_TYPE_PRETRAINED if project_type == MODEL_TYPE_PRETRAINED else MODEL_TYPE_SCRATCH
            img_array = normalize_images_for_model([img_array], model_type)[0]
            img_array = np.expand_dims(img_array, axis=0)

            # Run prediction
            preds = model.predict(img_array, verbose=0)
            best_class = np.argmax(preds[0])
            confidence = float(preds[0][best_class])

            # Create results json
            result = {
                "success": True,
                "label": label_for_class(int(best_class), label_names),
                "class_index": int(best_class),
                "confidence": confidence,
                "top_predictions": top_predictions(preds[0], label_names)
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
        labels_json_str = sys.argv[3] if len(sys.argv) > 3 else None
        project_type = sys.argv[4] if len(sys.argv) > 4 else MODEL_TYPE_SCRATCH
        predict(sys.argv[1], sys.argv[2], labels_json_str, project_type)
