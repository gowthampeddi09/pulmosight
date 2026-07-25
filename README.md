# PulmoSight — Advanced AI Medical Intelligence Platform for Chest X-Ray Pneumonia Detection

PulmoSight is a production-grade, full-stack medical AI web application for Chest X-Ray Pneumonia detection. It combines deep learning computer vision (EfficientNet-B0), visual explainability (Grad-CAM), multi-provider LLM clinical report generation (Google Gemini 2.0 Flash with Groq fallback), and automated PDF report generation into a secure, responsive web platform.

---

## 📋 Table of Contents

- [Executive Summary](#executive-summary)
- [System Architecture & Data Flow](#system-architecture--data-flow)
- [Dataset & Preprocessing Pipeline](#dataset--preprocessing-pipeline)
- [Model Architecture & Model Selection](#model-architecture--model-selection)
- [Class Imbalance & Loss Formulation](#class-imbalance--loss-formulation)
- [Explainability Engine (Grad-CAM & Region Mapping)](#explainability-engine-grad-cam--region-mapping)
- [LLM Integration & Prompt Engineering](#llm-integration--prompt-engineering)
- [Hospital-Style PDF Generation](#hospital-style-pdf-generation)
- [Authentication & History Management](#authentication--history-management)
- [API Reference](#api-reference)
- [Model Training Guide (Local & Google Colab GPU)](#model-training-guide-local--google-colab-gpu)
- [Docker & Local Setup Instructions](#docker--local-setup-instructions)
- [Testing Strategy](#testing-strategy)
- [Ethical & Clinical Disclaimer](#ethical--clinical-disclaimer)

---

## 🏥 Executive Summary

Pneumonia is a leading cause of morbidity worldwide. Early detection via chest radiography is critical, but manual interpretation is subject to radiologist availability and inter-observer variability. 

PulmoSight provides an end-to-end decision support tool designed to:
1. **Classify** Chest X-rays as `PNEUMONIA` or `NORMAL` with high sensitivity (Recall $\ge 96\%$).
2. **Explain** model decisions using Grad-CAM heatmaps overlaying specific anatomical lung zones.
3. **Synthesize** structured, clinical-style reports using Google Gemini AI.
4. **Export** hospital-grade PDF documentation for clinical records.

---

## 🏗 System Architecture & Data Flow

PulmoSight enforces a strict layered architecture (`Routes -> Services -> Inference / LLM / DB Repository`):

```
+-----------------------------------------------------------------------------------+
|                                  Next.js Frontend                                 |
|                         (React 18 / TypeScript / Tailwind)                        |
+------------------------------------------+----------------------------------------+
                                           | HTTP / REST (JWT Bearer Auth)
                                           v
+-----------------------------------------------------------------------------------+
|                                 FastAPI Backend                                   |
|                                                                                   |
|  +--------------------+    +---------------------+    +------------------------+  |
|  |   API Controllers  | -> |    Service Layer    | -> |   PyTorch EfficientNet |  |
|  | (Thin validation)  |    |  (Orchestration)    |    |   Inference Engine     |  |
|  +--------------------+    +----------+----------+    +------------------------+  |
|                                       |                                           |
|                                       |---------> +----------------------------+  |
|                                       |           |  Grad-CAM Heatmap Engine   |  |
|                                       |           +----------------------------+  |
|                                       |                                           |
|                                       |---------> +----------------------------+  |
|                                       |           | Google Gemini / Groq LLM   |  |
|                                       |           +----------------------------+  |
|                                       |                                           |
|                                       +---------> +----------------------------+  |
|                                                   | ReportLab PDF Streaming    |  |
|                                                   +----------------------------+  |
+-------------------------------------------+---------------------------------------+
                                            |
                                            v
                                 +---------------------+
                                 | PostgreSQL Database |
                                 |  (Users & History)  |
                                 +---------------------+
```

---

## 📊 Dataset & Preprocessing Pipeline

### Dataset Overview
- **Source**: Kaggle — *Chest X-Ray Images (Pneumonia)* by Paul Mooney (Kermany et al., *Cell 2018*).
- **Total Images**: 5,856 pediatric chest radiographies (JPEG format).
- **Classes**: `NORMAL` (1,575 images, ~27%) vs `PNEUMONIA` (4,273 images, ~73%).

### Directory Layout
Place the dataset in the root directory under `data/chest_xray/`:
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

### Preprocessing & Data Augmentation
During training, images undergo spatial and intensity transformations via Albumentations:
1. **Resize**: Rescaled to $224 \times 224$ pixels.
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances soft-tissue pulmonary contrast without amplifying noise.
3. **Random Rotation & Flip**: Slight rotational variance ($\pm 10^\circ$) and horizontal flipping ($p=0.3$).
4. **ImageNet Normalization**: Channel-wise mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

---

## 🧠 Model Architecture & Model Selection

### Architecture Choice: EfficientNet-B0
- **Parameter Efficiency**: 5.3M parameters (compared to ResNet-50's 25.6M), enabling low-latency inference on CPU ($\sim 30\text{ms}$).
- **Compound Scaling**: EfficientNet balances network depth, width, and image resolution using a fixed compound coefficient ($\phi$).
- **Classifier Head**:
  ```python
  model.classifier = nn.Sequential(
      nn.Dropout(p=0.2, inplace=True),
      nn.Linear(in_features=1280, out_features=1),  # Binary classification logit
  )
  ```

---

## ⚖️ Class Imbalance & Loss Formulation

The dataset exhibits a 3:1 imbalance ratio favoring Pneumonia.

### Choice: Weighted Binary Cross-Entropy Loss
We use `BCEWithLogitsLoss(pos_weight=...)`:
$$\text{pos\_weight} = \frac{N_{\text{normal}}}{N_{\text{pneumonia}}} = \frac{1341}{3875} \approx 0.346$$

#### Justification over Weighted Random Sampler
Weighted loss directly scales gradient contributions without duplicating minority samples in mini-batches. This prevents the model from overfitting to repeated augmented copies of normal scans.

---

## 🔍 Explainability Engine (Grad-CAM & Region Mapping)

To make AI decisions interpretable:
1. **Grad-CAM Computation**: Hooks into the final convolutional block (`features[-1]`) of EfficientNet-B0 to extract gradients and activations.
2. **Heatmap Overlay**: Blends normalized activation maps ($[0, 1]$) with original X-rays using a jet colormap without requiring external heavy dependencies like Matplotlib.
3. **Anatomical Spatial Analysis**: Maps the heatmap to a $3 \times 3$ grid corresponding to lung quadrants (apex, mid-zone, base, mediastinum) to feed structured textual descriptions into the LLM prompt.

---

## 🤖 LLM Integration & Prompt Engineering

PulmoSight uses **Google Gemini 2.0 Flash** (`google-generativeai` SDK) as the primary LLM provider, with **Groq Llama 3.1 70B** as an automatic secondary fallback.

### Structured Prompt Design
Instead of unconstrained "explain this image" prompts, the prompt enforces strict formatting:
- Input inputs: Prediction Label, Confidence %, Model Version, Grad-CAM Anatomical Observation, and Patient Age/Gender/Symptoms.
- Enforced sections: `Clinical Summary`, `Possible Differential Considerations`, `Recommended Next Steps`, `Limitations of AI Analysis`, `Urgency Level`, and `Disclaimer`.

---

## 📄 Hospital-Style PDF Generation

Generated using `reportlab` (streaming PDF output directly to the browser):
- **Layout**: Clean header with institution branding, patient metadata table, diagnostic badge, embedded Grad-CAM heatmap, formatted LLM report sections, and legal disclaimer.

---

## 🔒 Authentication & History Management

- **JWT Auth**: Password hashing with `passlib` (bcrypt), 15-minute access tokens, and 7-day refresh tokens.
- **Paginated History**: `GET /api/v1/history` supports pagination, search filtering by filename/label, date range queries, and sorting.
- **Cascade Deletion**: Deleting a record purges associated images and heatmaps from storage.

---

## 🔌 API Reference

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| `GET` | `/api/v1/health` | Service & DB connectivity health check | No |
| `GET` | `/api/v1/model-info` | Model architecture & performance metrics | No |
| `POST` | `/api/v1/auth/register` | Register new user account | No |
| `POST` | `/api/v1/auth/login` | Authenticate & obtain JWT tokens | No |
| `POST` | `/api/v1/auth/refresh` | Refresh access token | No |
| `GET` | `/api/v1/auth/me` | Current user profile | Yes |
| `POST` | `/api/v1/predict` | Upload Chest X-Ray for inference & Grad-CAM | Yes |
| `POST` | `/api/v1/generate-report` | Generate LLM clinical report | Yes |
| `GET` | `/api/v1/history` | Paginated prediction history | Yes |
| `GET` | `/api/v1/prediction/{id}` | Full prediction details | Yes |
| `DELETE` | `/api/v1/prediction/{id}` | Delete prediction & files | Yes |
| `GET` | `/api/v1/prediction/{id}/pdf` | Stream PDF clinical report | Yes |

---

## 💻 Model Training Guide (Local & Google Colab GPU)

### Option 1: Google Colab (Recommended for Fast GPU Training)
1. Open Google Colab in your browser or via the VS Code / Antigravity extension.
2. Open `model/train_pulmosight.ipynb`.
3. Select **GPU (T4)** from Runtime -> Change runtime type.
4. Execute cells sequentially. Weights auto-save to `weights/best_model.pth`.

### Option 2: Local Python Execution
```bash
# 1. Install dependencies
pip install -r backend/requirements.txt albumentations scikit-learn pyyaml

# 2. Train model
python model/train.py

# 3. Evaluate model on test set
python model/evaluate.py
```

---

## 🐳 Docker & Local Setup Instructions

### 1. Environment Configuration
Ensure `.env` exists in the root directory with your keys:
```env
DATABASE_URL=postgresql+asyncpg://pulmosight:pulmosight_secret@postgres:5432/pulmosight
SECRET_KEY=f7a9c2e1d4b8a3f6e9c1d5b7a2f4e8c3d6b9a1f5e7c2d4b8a3f6e9c1d5b7a2
GOOGLE_API_KEY=your_google_ai_studio_key
GROQ_API_KEY=your_groq_api_key
```

### 2. Launch with Docker Compose
```bash
docker-compose up --build
```

### 3. Access Services
- **Web UI**: [http://localhost:3000](http://localhost:3000)
- **FastAPI OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧪 Testing Strategy

Run the backend pytest suite inside Docker:
```bash
docker-compose exec backend pytest tests/ -v
```

Tests cover:
- Health check endpoints
- User registration, duplicate email handling, login validation
- File validation (corrupted files, invalid extensions, low resolution)
- Prediction pipeline & history retrieval
- Record deletion & cleanup

---

## ⚠️ Ethical & Clinical Disclaimer

> **IMPORTANT**: PulmoSight is an AI decision-support system built for **educational, portfolio demonstration, and research purposes only**. It is NOT a certified medical device (FDA/CE) and must never be used as the sole basis for clinical diagnosis, treatment decisions, or patient triage. All predictions and generated reports must be validated by a licensed physician or radiologist.
