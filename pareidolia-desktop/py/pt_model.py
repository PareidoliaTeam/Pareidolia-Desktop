# Cell 1: Model
import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl
from torchmetrics import Accuracy

class RepVGGClassifier(pl.LightningModule):
    """
    Author: Armando Vega
    Date Created: 8 April 2026

    Last Modified By: Armando Vega
    Date Last Modified: 8 April 2026

    A PyTorch Lightning module that imports a pre-trained model as the backbone and a custom head for classification. The base trained model is RepVGG-A2 but
    can be swapped out for any of the models that timm offers. 
    """
    def __init__(self, model_name="repvgg_a2", num_classes=10, lr=1e-3, hidden_dim=512):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        n_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = self.train_acc(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = self.val_acc(logits, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = self.test_acc(logits, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {"preds": preds, "probs": probs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
# Cell 1: Model
# class MobileNetV4Classifier(pl.LightningModule):
#     """
#     Author: Armando Vega
#     Date Created: 13 April 2026
    
#     A PyTorch Lightning module utilizing MobileNetV4 as a backbone.
#     Optimized for TFLite conversion and mobile deployment.
#     """
#     def __init__(self, model_name="mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k", 
#                  num_classes=10, lr=1e-3, hidden_dim=512):
#         super().__init__()
#         self.save_hyperparameters()

#         # Load the backbone
#         self.backbone = timm.create_model(
#             model_name,
#             pretrained=True,
#             num_classes=0,       # Remove the original ImageNet head
#             global_pool="avg",   # Use Global Average Pooling
#         )

#         # Dynamic feature detection to avoid "mat1 and mat2" shape errors
#         # For mobilenetv4_conv_medium, this will be 1280
#         n_features = self.backbone.num_features
        
#         self.head = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(n_features, hidden_dim), 
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hidden_dim, num_classes),
#         )

#         # Metrics
#         self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
#         self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
#         self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

#     def forward(self, x):
#         # MobileNetV4 expects [Batch, 3, 384, 384]
#         features = self.backbone(x)
#         return self.head(features)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = nn.functional.cross_entropy(logits, y)
        
#         self.train_acc(logits, y)
#         self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
#         self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         loss = nn.functional.cross_entropy(logits, y)
        
#         self.val_acc(logits, y)
#         self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
#         self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
#         return loss

#     def configure_optimizers(self):
#         # Adam is a solid choice for MobileNet fine-tuning
#         return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)