# Cell 2: DataModule
import json
import os
import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets
from .numpy_image_dataset import NumpyImageDataset

class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir=None,
        batch_size=32,
        img_size=224,
        val_split=0.2,
        num_workers=4,
        seed=42,
        cifar10=False,
        labels_json=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.val_split = val_split
        self.num_workers = 0 if os.name == "nt" else num_workers  # Windows workaround for num_workers > 0
        self.seed = seed
        self.cifar10 = cifar10
        self.labels_json = labels_json
        self._json_dataset_cache = None

        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # def calculate_mean_std(self, images):
        

    def load_images_from_json(self, labels_json):
        """Load images from a JSON mapping of label names to arrays of folder paths."""
        if isinstance(labels_json, str):
            labels_json = labels_json.strip()
            if labels_json.startswith("{"):
                labels_dict = json.loads(labels_json)
            elif os.path.exists(labels_json):
                with open(labels_json, "r", encoding="utf-8") as f:
                    labels_dict = json.load(f)
            else:
                raise FileNotFoundError(f"labels_json path not found: {labels_json}")
        else:
            labels_dict = labels_json

        label_names = list(labels_dict.keys())
        num_classes = len(label_names)

        images = []
        labels = []

        for label_index, label_name in enumerate(label_names):
            folder_paths = labels_dict[label_name]
            for folder_path in folder_paths:
                if not os.path.exists(folder_path):
                    print(f"Warning: folder not found, skipping: {folder_path}")
                    continue

                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(folder_path, img_file)
                        img = cv2.imread(img_path)

                        if img is None:
                            continue

                        img = cv2.resize(img, (self.img_size, self.img_size))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        images.append(img)
                        labels.append(label_index)

        if len(images) == 0:
            return None, None, 0, []

        images = np.array(images, dtype=np.float32) / 255.0
        labels = np.array(labels, dtype=np.int64)

        return images, labels, num_classes, label_names

    def _get_json_dataset(self):
        if self._json_dataset_cache is None:
            images, labels, num_classes, label_names = self.load_images_from_json(self.labels_json)
            if images is None:
                raise ValueError("No images were loaded from labels_json.")
            self._json_dataset_cache = (images, labels, num_classes, label_names)
        return self._json_dataset_cache

    def prepare_data(self):
        if self.cifar10:
            datasets.CIFAR10(root=self.data_dir, train=True, download=True)
            datasets.CIFAR10(root=self.data_dir, train=False, download=True)
        elif self.labels_json is not None:
            self._get_json_dataset()
        elif not self.data_dir or not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} not found!")

    def setup(self, stage=None):
        if self.labels_json is not None:
            images, labels, num_classes, label_names = self._get_json_dataset()
            self.num_classes = num_classes
            self.class_names = label_names

            if stage in ("fit", None):
                train_base = NumpyImageDataset(images, labels, transform=self.train_transform)
                val_base = NumpyImageDataset(images, labels, transform=self.eval_transform)
                n_total = len(train_base)
                n_val = int(self.val_split * n_total)
                n_train = n_total - n_val
                generator = torch.Generator().manual_seed(self.seed)
                train_subset, val_subset = random_split(range(n_total), [n_train, n_val], generator=generator)
                self.train_ds = torch.utils.data.Subset(train_base, train_subset.indices)
                self.val_ds = torch.utils.data.Subset(val_base, val_subset.indices)

            if stage in ("test", None):
                self.test_ds = NumpyImageDataset(images, labels, transform=self.eval_transform)

            if stage in ("predict", None):
                self.predict_ds = NumpyImageDataset(images, labels, transform=self.eval_transform)

            return

        if stage in ("fit", None):
            if not self.cifar10:
                train_base = datasets.ImageFolder(self.data_dir, transform=self.train_transform)
                val_base = datasets.ImageFolder(self.data_dir, transform=self.eval_transform)
            else:
                train_base = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.train_transform, download=False)
                val_base = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.eval_transform, download=False)

            n_total = len(train_base)
            n_val = int(self.val_split * n_total)
            n_train = n_total - n_val
            generator = torch.Generator().manual_seed(self.seed)

            train_subset, val_subset = random_split(range(n_total), [n_train, n_val], generator=generator)
            self.train_ds = torch.utils.data.Subset(train_base, train_subset.indices)
            self.val_ds = torch.utils.data.Subset(val_base, val_subset.indices)

            self.num_classes = len(train_base.classes)
            self.class_names = train_base.classes

        if stage in ("test", None):
            if self.cifar10:
                self.test_ds = datasets.CIFAR10(root=self.data_dir, train=False, transform=self.eval_transform, download=False)
            else:
                self.test_ds = datasets.ImageFolder(self.data_dir, transform=self.eval_transform)

        if stage in ("predict", None):
            if self.cifar10:
                self.predict_ds = datasets.CIFAR10(root=self.data_dir, train=False, transform=self.eval_transform, download=False)
            else:
                self.predict_ds = datasets.ImageFolder(self.data_dir, transform=self.eval_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )