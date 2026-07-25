"""
EfficientNet-B0 Pneumonia Detection Training Script
Fine-tunes EfficientNet-B0 on Chest X-Ray dataset with weighted loss and early stopping.
"""
import os
import time
import json
import logging
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models

from dataset import PneumoniaDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience: int = 3, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Freeze initial feature extraction layers, fine-tune last blocks
    for name, param in model.features.named_parameters():
        if not name.startswith("7") and not name.startswith("6"):
            param.requires_grad = False

    # Replace binary classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=1),
    )
    return model

def train():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    data_dir = config["dataset"]["data_dir"]
    full_train_dataset = PneumoniaDataset(data_dir, split="train", is_train=True)
    test_dataset = PneumoniaDataset(data_dir, split="test", is_train=False)

    # Train / Val split (85% train, 15% validation for reliable metric calculation)
    train_size = int(0.85 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_ds, val_ds = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["dataset"]["batch_size"], shuffle=True,
        num_workers=config["dataset"]["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["dataset"]["batch_size"], shuffle=False,
        num_workers=config["dataset"]["num_workers"], pin_memory=True
    )

    model = build_model().to(device)

    # CHOICE JUSTIFICATION FOR CLASS IMBALANCE:
    # We choose Weighted Loss (BCEWithLogitsLoss with pos_weight) over a Weighted Sampler.
    # Weighted loss directly scales the loss contribution of the minority class without
    # duplicating samples in mini-batches, avoiding over-fitting to repeated augmented copies.
    pos_weight_val = torch.tensor([config["training"]["pos_weight"]], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    early_stopping = EarlyStopping(patience=config["training"]["patience"])
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / config["training"]["checkpoint_name"]

    epochs = config["training"]["epochs"]
    log.info("Starting training for %d epochs...", epochs)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_running_loss / len(val_ds)
        val_acc = correct / total if total > 0 else 0.0
        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        log.info(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Checkpoint if best model so far
        is_best = early_stopping(val_loss)
        if is_best:
            log.info("Saving best model checkpoint to %s", checkpoint_path)
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_version": "1.0.0",
                "training_date": time.strftime("%Y-%m-%d"),
                "metrics": {
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            }, checkpoint_path)

        if early_stopping.early_stop:
            log.info("Early stopping triggered at epoch %d", epoch)
            break

    log.info("Training complete.")

if __name__ == "__main__":
    train()
