# Cell 1: Model
import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl
from torchmetrics import Accuracy


def _activation_from_name(name):
    activation_name = str(name or "relu").lower()
    activations = {
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "leakyrelu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "none": None,
        "linear": None,
    }
    if activation_name not in activations:
        raise ValueError(f"Unsupported activation for PyTorch scratch model: {name}")

    activation_cls = activations[activation_name]
    return activation_cls() if activation_cls is not None else None


def _as_int(parameters, key, default):
    return int(parameters.get(key, default))


def _as_float(parameters, key, default):
    return float(parameters.get(key, default))


def _same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        return tuple(k // 2 for k in kernel_size)
    return kernel_size // 2


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


class ScratchCNNClassifier(pl.LightningModule):
    """
    A PyTorch Lightning module that builds a CNN from the same layer JSON shape
    used by the TensorFlow scratch model builder.
    """
    def __init__(self, layers_json=None, num_classes=10, lr=1e-3, input_channels=3):
        super().__init__()
        self.save_hyperparameters()

        self.model = self._build_model(layers_json or [], num_classes, input_channels)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def _build_model(self, layers_json, num_classes, input_channels):
        modules = []
        in_channels = input_channels
        has_flattened = False
        last_linear_features = None

        for layer_data in layers_json:
            layer_type = layer_data.get("type")
            parameters = layer_data.get("parameters", {}) or {}

            if layer_type in ("RandomFlip", "RandomRotation", "RandomZoom", "RandomContrast"):
                continue

            if layer_type == "Conv2D":
                filters = _as_int(parameters, "units", parameters.get("filters", 32))
                kernel_size = _as_int(parameters, "kernel_size", 3)
                stride = _as_int(parameters, "stride", 1)
                padding_param = parameters.get("padding", "same")
                padding = _same_padding(kernel_size) if padding_param == "same" else int(padding_param)

                modules.append(nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding))
                activation = _activation_from_name(parameters.get("activation", "relu"))
                if activation is not None:
                    modules.append(activation)

                in_channels = filters
                has_flattened = False
                last_linear_features = None
            elif layer_type == "MaxPooling2D":
                pool_size = _as_int(parameters, "pool_size", 2)
                modules.append(nn.MaxPool2d(kernel_size=pool_size))
                has_flattened = False

            elif layer_type == "AveragePooling2D":
                pool_size = _as_int(parameters, "pool_size", 2)
                modules.append(nn.AvgPool2d(kernel_size=pool_size))
                has_flattened = False

            elif layer_type == "GlobalAveragePooling2D":
                modules.append(nn.AdaptiveAvgPool2d((1, 1)))
                modules.append(nn.Flatten())
                has_flattened = True
                last_linear_features = in_channels

            elif layer_type == "Flatten":
                modules.append(nn.Flatten())
                has_flattened = True
                last_linear_features = None
                
            elif layer_type == "Dense":
                if not has_flattened:
                    modules.append(nn.Flatten())
                    has_flattened = True

                units = _as_int(parameters, "units", 128)
                if last_linear_features is None:
                    modules.append(nn.LazyLinear(units))
                else:
                    modules.append(nn.Linear(last_linear_features, units))

                activation = _activation_from_name(parameters.get("activation", "relu"))
                if activation is not None:
                    modules.append(activation)

                last_linear_features = units
            elif layer_type == "Dropout":
                modules.append(nn.Dropout(_as_float(parameters, "rate", 0.2)))
            else:
                raise ValueError(f"Unsupported PyTorch scratch layer type: {layer_type}")

        if not has_flattened:
            modules.append(nn.Flatten())

        if last_linear_features is None:
            modules.append(nn.LazyLinear(num_classes))
        else:
            modules.append(nn.Linear(last_linear_features, num_classes))

        return nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

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
