import sys
import json
import torch
import pytorch_lightning as pl
from image_data_module import ImageDataModule
from pt_model import MobileNetClassifier, ScratchCNNClassifier
from pt_train_model import compute_mean_std_welford_from_loader, MODEL_TYPE_PRETRAINED, MODEL_TYPE_SCRATCH


def infer_project_type_from_checkpoint(checkpoint_path, fallback=MODEL_TYPE_PRETRAINED):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    keys = state_dict.keys()

    if any(key.startswith("backbone.") or key.startswith("head.") for key in keys):
        return MODEL_TYPE_PRETRAINED

    if any(key.startswith("model.") for key in keys):
        return MODEL_TYPE_SCRATCH

    return MODEL_TYPE_PRETRAINED if fallback == MODEL_TYPE_PRETRAINED else MODEL_TYPE_SCRATCH


def get_model_class_count(model):
    """Infer the number of output classes from a loaded Lightning model."""
    hparams = getattr(model, "hparams", None)
    if hparams is not None:
        if hasattr(hparams, "num_classes"):
            return int(hparams.num_classes)
        if isinstance(hparams, dict) and "num_classes" in hparams:
            return int(hparams["num_classes"])

    model_classes = None
    for module in model.modules():
        out_features = getattr(module, "out_features", None)
        if out_features is not None:
            model_classes = int(out_features)

    return model_classes


def evaluate(checkpoint_path, labels_json_string, project_type=MODEL_TYPE_PRETRAINED):
    try:
        # Initialize data module
        data_module = ImageDataModule(
            labels_json=labels_json_string,
            batch_size=32,
            img_size=224,
            seed=42
        )

        checkpoint_project_type = infer_project_type_from_checkpoint(checkpoint_path, project_type)
        model_class = MobileNetClassifier if checkpoint_project_type == MODEL_TYPE_PRETRAINED else ScratchCNNClassifier

        if checkpoint_project_type == MODEL_TYPE_SCRATCH:
            stats_loader = data_module.normalization_stats_dataloader()
            normalization_mean, normalization_std = compute_mean_std_welford_from_loader(stats_loader)
            data_module.set_normalization(normalization_mean, normalization_std)

        # Setup test stage
        data_module.setup(stage="test")

        # load model
        model = model_class.load_from_checkpoint(checkpoint_path, map_location="cpu")
        model_classes = get_model_class_count(model)
        dataset_classes = getattr(data_module, "num_classes", None)

        if model_classes is not None and dataset_classes is not None and int(model_classes) != int(dataset_classes):
            raise ValueError(
                "Model class count mismatch: "
                f"the loaded model outputs {int(model_classes)} classes, "
                f"but the current dataset defines {int(dataset_classes)} labels. "
                "Retrain the model with the current label set, or restore the label set "
                "that was used when this model was created."
            )

        trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)

        # Test abd get results
        results = trainer.test(model, datamodule=data_module, verbose=False)
        metrics = results[0]

        print(json.dumps({
            "success": True,
            "accuracy": float(metrics.get("test_acc", 0)),
            "loss": float(metrics.get("test_loss", 0)),
            "total_images": len(data_module.test_ds),
            "model_classes": int(model_classes) if model_classes is not None else None,
            "dataset_classes": int(dataset_classes) if dataset_classes is not None else None
        }))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__  == "__main__":
    project_type = sys.argv[3] if len(sys.argv) > 3 else MODEL_TYPE_PRETRAINED
    evaluate(sys.argv[1], sys.argv[2], project_type)


