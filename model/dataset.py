import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    from torchvision import transforms

class PneumoniaDataset(Dataset):
    """
    Custom PyTorch Dataset for Chest X-Ray Pneumonia binary classification.
    Supports Albumentations (CLAHE, RandomRotation, HorizontalFlip, Resize, Normalize)
    with pure PyTorch torchvision fallback.
    """
    def __init__(self, root_dir: str, split: str = "train", transform=None, is_train: bool = True):
        self.split_dir = Path(root_dir) / split
        self.transform = transform
        self.is_train = is_train
        self.image_paths = []
        self.labels = []

        # Load NORMAL (label 0) and PNEUMONIA (label 1)
        for label_name, label_idx in [("NORMAL", 0), ("PNEUMONIA", 1)]:
            class_dir = self.split_dir / label_name
            if class_dir.exists():
                for ext in ["*.jpeg", "*.jpg", "*.png", "*.JPEG", "*.JPG", "*.PNG"]:
                    for img_path in class_dir.glob(ext):
                        # Filter out corrupted or hidden macOS metadata files
                        if not img_path.name.startswith("._"):
                            self.image_paths.append(str(img_path))
                            self.labels.append(label_idx)

        # Default transforms if none provided
        if self.transform is None:
            self.transform = self._get_default_transforms(is_train)

    def _get_default_transforms(self, is_train: bool):
        if HAS_ALBUMENTATIONS:
            if is_train:
                return A.Compose([
                    A.Resize(224, 224),
                    A.RandomRotate90(p=0.2),
                    A.HorizontalFlip(p=0.3),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
            else:
                return A.Compose([
                    A.Resize(224, 224),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ])
        else:
            if is_train:
                return transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                return transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        img_np = np.array(image)

        if HAS_ALBUMENTATIONS and isinstance(self.transform, A.Compose):
            augmented = self.transform(image=img_np)
            tensor_img = augmented["image"]
        else:
            tensor_img = self.transform(image)

        return tensor_img, torch.tensor(label, dtype=torch.float32)
