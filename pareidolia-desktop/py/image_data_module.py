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
from numpy_image_dataset import NumpyImageDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class NumpyImageDataset(Dataset):
    """
    Author: Armando Vega
    Date Created: 7 April 2026

    Last Modified By: Armando Vega
    Date Last Modified: 7 April 2026

    A dataset class that wraps an indexable collection of images and labels,
    applying optional transformations. Individual images should be HWC numpy
    arrays. Both uint8 0..255 images and float images are supported.
    """

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                scale = 255.0 if image.max() <= 1.0 else 1.0
                image = np.clip(image * scale, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        image = Image.fromarray(image)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

class ImageDataModule(pl.LightningDataModule):
    """
    Author: Armando Vega
    Date Created: 7 April 2026

    Last Modified By: Armando Vega
    Date Last Modified: 7 April 2026

    Pytorch Lightning DataModule for loading and preparing image datasets for training, validation, testing, and prediction.
    Supports loading from a directory structure or from a JSON mapping of label names to folder paths looking like: 
    {
        "cat": ["path/to/cat_images_folder"],
        "dog": ["path/to/dog_images_folder", "path/to/more_dog_images_folder"]
    }

    The DataModule handles splitting datasets into training, validation etc. based on specified parameters and applies generalized tranformations for each.
    CIFAR-10 dataset loading is also supported as a special case for debugging or quick experimentation. A directory for data batches can be specified if 
    loading data is the preferred method. 
    """
    def __init__(
        self,
        data_dir=None, # the directory to load data from; ignored if labels_json is provided or cifar10=True
        batch_size=32, # the batch size to use for all dataloaders
        img_size=224,  # final crop size presented to the backbone after preprocessing
        val_split=0.2, # the fraction of the dataset to use for validation when loading from a directory or JSON; ignored if cifar10=True since CIFAR-10 has a predefined train/test split
        num_workers=4, # the number of worker processes to use for data loading; set to 0 for Windows to avoid issues with multiprocessing
        seed=42,       # reproducibility seed
        cifar10=False, 
        labels_json=None, # this is a string btw
        test_split=0.1,   # the fraction of the dataset to use for testing when loading from a directory or JSON; ignored if cifar10=True since CIFAR-10 has a predefined train/test split
        normalization_mean=None,
        normalization_std=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = 0 if os.name == "nt" else num_workers  # Windows workaround for num_workers > 0; creates issues otherwise
        self.seed = seed
        self.cifar10 = cifar10
        self.labels_json = labels_json
        self._json_dataset_cache = None
        self._split_indices_cache = None
        self.resize_size = 256
        self.normalization_mean = list(normalization_mean or IMAGENET_MEAN)
        self.normalization_std = list(normalization_std or IMAGENET_STD)

        if not (0 <= self.val_split < 1):
            raise ValueError(f"val_split must be in [0, 1), got {self.val_split}")
        if not (0 <= self.test_split < 1):
            raise ValueError(f"test_split must be in [0, 1), got {self.test_split}")
        if self.val_split + self.test_split >= 1:
            raise ValueError(
                f"val_split + test_split must be < 1, got {self.val_split + self.test_split}"
            )

        self._build_transforms()

    def _build_transforms(self):
        normalize = transforms.Normalize(mean=self.normalization_mean, std=self.normalization_std)

        self.train_transform = transforms.Compose([ # training preprocessing: resize up, augment, then crop to model size
            transforms.Resize((self.resize_size, self.resize_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            normalize,
        ])

        self.eval_transform = transforms.Compose([ # evaluation preprocessing matches exported model preprocessing
            transforms.Resize((self.resize_size, self.resize_size)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            normalize,
        ])

        self.rep_transform = transforms.Compose([ # representative data should match the wrapped TF/TFLite model's raw input domain
            transforms.Resize((self.resize_size, self.resize_size)),
            transforms.PILToTensor(),
        ])

        self.stats_transform = transforms.Compose([
            transforms.Resize((self.resize_size, self.resize_size)),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
        ])

    def set_normalization(self, mean, std):
        self.normalization_mean = [float(value) for value in mean]
        self.normalization_std = [max(float(value), 1e-6) for value in std]
        self._build_transforms()

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

                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        images.append(img)
                        labels.append(label_index)

        if len(images) == 0:
            return None, None, 0, []

        labels = np.array(labels, dtype=np.int64)

        print(f"Loaded {len(images)} images across {num_classes} classes from JSON dataset.")

        return images, labels, num_classes, label_names

    def _get_json_dataset(self):
        if self._json_dataset_cache is None:
            images, labels, num_classes, label_names = self.load_images_from_json(self.labels_json)
            if images is None:
                raise ValueError("No images were loaded from labels_json.")
            self._json_dataset_cache = (images, labels, num_classes, label_names)
        return self._json_dataset_cache

    def _get_split_indices(self, n_total):
        if self._split_indices_cache is not None and self._split_indices_cache["n_total"] == n_total:
            return (
                self._split_indices_cache["train_indices"],
                self._split_indices_cache["val_indices"],
                self._split_indices_cache["test_indices"],
            )

        n_val = int(self.val_split * n_total)
        n_test = int(self.test_split * n_total)
        n_train = n_total - n_val - n_test
        if n_train <= 0:
            raise ValueError(
                f"Dataset too small for val_split={self.val_split} and test_split={self.test_split} (n_total={n_total})."
            )

        generator = torch.Generator().manual_seed(self.seed)
        all_indices = torch.randperm(n_total, generator=generator).tolist()

        print("All indices: ", all_indices)  # Print the first 10 shuffled indices for debugging

        train_indices = all_indices[:n_train]
        val_indices = all_indices[n_train:n_train + n_val]
        test_indices = all_indices[n_train + n_val:]

        self._split_indices_cache = {
            "n_total": n_total,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
        }
        return train_indices, val_indices, test_indices

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

            n_total = len(images)
            train_indices, val_indices, test_indices = self._get_split_indices(n_total)
            eval_base = NumpyImageDataset(images, labels, transform=self.eval_transform)

            rep_base = NumpyImageDataset(images, labels, transform=self.rep_transform)
            self.rep_ds = torch.utils.data.Subset(rep_base, train_indices)

            if stage in ("fit", None):
                train_base = NumpyImageDataset(images, labels, transform=self.train_transform)
                self.train_ds = torch.utils.data.Subset(train_base, train_indices)
                self.val_ds = torch.utils.data.Subset(eval_base, val_indices)

            if stage in ("test", None):
                self.test_ds = torch.utils.data.Subset(eval_base, test_indices)

            if stage in ("predict", None):
                self.predict_ds = NumpyImageDataset(images, labels, transform=self.eval_transform)

            return

        if self.cifar10:
            if stage in ("fit", None):
                train_base = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.train_transform, download=False)
                val_base = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.eval_transform, download=False)

                n_total = len(train_base)
                n_val = int(self.val_split * n_total)
                n_train = n_total - n_val

                generator = torch.Generator().manual_seed(self.seed)
                all_indices = torch.randperm(n_total, generator=generator).tolist()
                train_indices = all_indices[:n_train]
                val_indices = all_indices[n_train:]

                self.train_ds = torch.utils.data.Subset(train_base, train_indices)
                self.val_ds = torch.utils.data.Subset(val_base, val_indices)
                self.num_classes = len(train_base.classes)
                self.class_names = train_base.classes

            if stage in ("test", None):
                self.test_ds = datasets.CIFAR10(root=self.data_dir, train=False, transform=self.eval_transform, download=False)
        else:
            eval_base = datasets.ImageFolder(self.data_dir, transform=self.eval_transform)
            n_total = len(eval_base)
            train_indices, val_indices, test_indices = self._get_split_indices(n_total)

            if stage in ("fit", None):
                train_base = datasets.ImageFolder(self.data_dir, transform=self.train_transform)

                self.train_ds = torch.utils.data.Subset(train_base, train_indices)
                self.val_ds = torch.utils.data.Subset(eval_base, val_indices)

                self.num_classes = len(eval_base.classes)
                self.class_names = eval_base.classes

            if stage in ("test", None):
                self.test_ds = torch.utils.data.Subset(eval_base, test_indices)

        if stage in ("predict", None):
            if self.cifar10:
                self.predict_ds = datasets.CIFAR10(root=self.data_dir, train=False, transform=self.eval_transform, download=False)
            else:
                self.predict_ds = datasets.ImageFolder(self.data_dir, transform=self.eval_transform)

    # DataLoader objects for training, validation, testing, and prediction. Uses the respective datasets and applies appropriate shuffling and multiprocessing settings.
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
    
    def representative_dataloader(self):
        return DataLoader(
            self.rep_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def normalization_stats_dataloader(self):
        if self.labels_json is not None:
            images, labels, _, _ = self._get_json_dataset()
            train_indices, _, _ = self._get_split_indices(len(images))
            stats_base = NumpyImageDataset(images, labels, transform=self.stats_transform)
            stats_ds = torch.utils.data.Subset(stats_base, train_indices)
        elif self.cifar10:
            stats_base = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.stats_transform, download=False)
            n_total = len(stats_base)
            n_val = int(self.val_split * n_total)
            n_train = n_total - n_val
            generator = torch.Generator().manual_seed(self.seed)
            train_indices = torch.randperm(n_total, generator=generator).tolist()[:n_train]
            stats_ds = torch.utils.data.Subset(stats_base, train_indices)
        else:
            stats_base = datasets.ImageFolder(self.data_dir, transform=self.stats_transform)
            train_indices, _, _ = self._get_split_indices(len(stats_base))
            stats_ds = torch.utils.data.Subset(stats_base, train_indices)

        return DataLoader(
            stats_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
