# Dataset Documentation

## Source

**Kaggle: Chest X-Ray Images (Pneumonia)** by Paul Mooney  
Original publication: Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning", *Cell*, 2018.

Dataset URL: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Overview

- **Total Images**: 5,856 pediatric chest radiographies (JPEG format)
- **Task**: Binary classification — `NORMAL` vs `PNEUMONIA`
- **Class Distribution**: ~27% Normal, ~73% Pneumonia (3:1 imbalance)

## Directory Layout

```text
pulmosight/
  data/
    chest_xray/
      train/
        NORMAL/      (1,341 images)
        PNEUMONIA/   (3,875 images)
      val/
        NORMAL/      (8 images)
        PNEUMONIA/   (8 images)
      test/
        NORMAL/      (234 images)
        PNEUMONIA/   (390 images)
```

## Preprocessing Pipeline

| Step | Training | Evaluation |
|------|----------|------------|
| Resize | 224×224 | 224×224 |
| CLAHE | p=0.4, clip_limit=2.0 | — |
| Random Rotation | ±90° (p=0.2) | — |
| Horizontal Flip | p=0.3 | — |
| Normalize | ImageNet (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]) | Same |

Implementation uses **Albumentations** with a **torchvision** fallback when Albumentations is not installed.

## Class Imbalance Handling

**Choice: Weighted Binary Cross-Entropy Loss** (`BCEWithLogitsLoss` with `pos_weight`)

```
pos_weight = N_normal / N_pneumonia = 1341 / 3875 ≈ 0.346
```

### Justification over Weighted Random Sampler
Weighted loss directly scales gradient contributions without duplicating minority samples in mini-batches. This prevents the model from overfitting to repeated augmented copies of normal scans while still prioritizing recall on the pneumonia class.

## Known Limitations

1. **Pediatric bias**: Dataset is entirely pediatric chest X-rays — model performance on adult radiographs is unvalidated.
2. **Single-institution**: All images sourced from one medical center (Guangzhou Women and Children's Medical Center).
3. **Small validation set**: Original val split contains only 16 images; we use an 85/15 train/val split instead.
4. **JPEG compression**: Some images may have compression artifacts affecting fine-grained feature extraction.
