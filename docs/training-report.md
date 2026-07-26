# Training Report — EfficientNet-B0 Pneumonia Detection

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | EfficientNet-B0 (torchvision, ImageNet pretrained) |
| Classifier Head | Dropout(0.2) → Linear(1280, 1) |
| Fine-tuned Layers | Last 2 feature blocks (blocks 6, 7) + classifier |
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=1) |
| Loss Function | BCEWithLogitsLoss (pos_weight=0.346) |
| Early Stopping | Patience=3, delta=1e-4 on val_loss |
| Epochs Trained | 10 (with early stopping) |
| Batch Size | 32 |

## Test Set Evaluation Results (N=624)

| Metric | Score | Clinical Target | Status |
|--------|-------|-----------------|--------|
| **Recall (Pneumonia)** | **98.72%** | ≥96% | ✅ Exceeds |
| **ROC-AUC** | **0.9515** | ≥0.95 | ✅ Meets |
| **F1-Score** | **0.9006** | ≥0.88 | ✅ Exceeds |
| **Accuracy** | **86.38%** | ≥85% | ✅ Meets |
| **Precision (Pneumonia)** | **82.80%** | ≥80% | ✅ Meets |

## Confusion Matrix

|  | Predicted Normal | Predicted Pneumonia |
|--|-----------------|---------------------|
| **Actual Normal** (234) | 154 (TN) | 80 (FP) |
| **Actual Pneumonia** (390) | 5 (FN) | 385 (TP) |

## Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NORMAL | 0.97 | 0.66 | 0.78 | 234 |
| PNEUMONIA | 0.83 | 0.99 | 0.90 | 390 |
| **Weighted Avg** | **0.88** | **0.86** | **0.86** | **624** |

## Clinical Trade-Off Analysis

In emergency triage and chest radiography screening, **false negatives are life-threatening** (sending a patient with active pneumonia home untreated). By weighting the loss function toward positive sensitivity, PulmoSight achieves an **outstanding 98.72% recall** — missing only 5 out of 390 real pneumonia scans.

The 80 false positives are acceptable in a clinical screening context: they prompt follow-up radiological review rather than diagnostic omission, which is the safer failure mode.

## Training History

Training converged within ~10 epochs with the ReduceLROnPlateau scheduler reducing the learning rate once after epoch 3.

## Reproducibility

- **Random Seed**: 42 (for train/val split)
- **Framework**: PyTorch 2.x, torchvision EfficientNet_B0_Weights.IMAGENET1K_V1
- **Hardware**: Google Colab T4 GPU
- **Training Time**: ~15 minutes on T4 GPU
