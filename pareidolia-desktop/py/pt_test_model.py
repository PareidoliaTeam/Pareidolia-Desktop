import sys
import json
import pytorch_lightning as pl
from image_data_module import ImageDataModule
from pt_model import RepVGGClassifier, ScratchCNNClassifier
from pt_train_model import compute_mean_std_welford_from_loader, MODEL_TYPE_PRETRAINED, MODEL_TYPE_SCRATCH

def evaluate(checkpoint_path, labels_json_string, project_type=MODEL_TYPE_PRETRAINED):
    try:
        # Initialize data module
        data_module = ImageDataModule(
            labels_json=labels_json_string,
            batch_size=32,
            img_size=224,
            seed=42
        )

        model_class = RepVGGClassifier if project_type == MODEL_TYPE_PRETRAINED else ScratchCNNClassifier

        if project_type == MODEL_TYPE_SCRATCH:
            stats_loader = data_module.normalization_stats_dataloader()
            normalization_mean, normalization_std = compute_mean_std_welford_from_loader(stats_loader)
            data_module.set_normalization(normalization_mean, normalization_std)

        # Setup test stage
        data_module.setup(stage="test")

        # load model
        model = model_class.load_from_checkpoint(checkpoint_path)
        trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)

        # Test abd get results
        results = trainer.test(model, datamodule=data_module, verbose=False)
        metrics = results[0]

        print(json.dumps({
            "success": True,
            "accuracy": float(metrics.get("test_acc", 0)),
            "loss": float(metrics.get("test_loss", 0)),
            "total_images": len(data_module.test_ds)
        }))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__  == "__main__":
    project_type = sys.argv[3] if len(sys.argv) > 3 else MODEL_TYPE_PRETRAINED
    evaluate(sys.argv[1], sys.argv[2], project_type)


