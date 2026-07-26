# PulmoSight — Advanced Medical AI Intelligence Platform

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
- [Folder Structure](#folder-structure)
- [Model Training Guide (Local & Google Colab GPU)](#model-training-guide-local--google-colab-gpu)
- [Docker & Local Setup Instructions](#docker--local-setup-instructions)
- [Testing Strategy](#testing-strategy)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [Ethical & Clinical Disclaimer](#ethical--clinical-disclaimer)

---

## 🏥 Executive Summary

Pneumonia is a leading cause of morbidity worldwide. Early detection via chest radiography is critical, but manual interpretation is subject to radiologist availability and inter-observer variability.

PulmoSight provides an end-to-end decision support tool designed to:
1. **Classify** Chest X-rays as `PNEUMONIA` or `NORMAL` with high sensitivity (Recall ≥ 98%).
2. **Explain** model decisions using Grad-CAM heatmaps overlaying specific anatomical lung zones.
3. **Synthesize** structured, clinical-style reports using Google Gemini AI with Groq fallback.
4. **Export** hospital-grade PDF documentation for clinical records.

---

## 🏗 System Architecture & Data Flow

PulmoSight enforces a strict layered architecture (`Routes → Services → Inference / LLM / DB Repository`):

```
┌───────────────────────────────────────────────────────────────────────────┐
│                           Next.js Frontend                               │
│                    (React 18 / TypeScript / Tailwind CSS)                │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │ HTTP / REST (JWT Bearer Auth)
                                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            FastAPI Backend                               │
│                                                                          │
│  ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────────┐ │
│  │  API Controllers │ →  │  Service Layer  │ →  │ PyTorch EfficientNet │ │
│  │ (Thin validation)│    │ (Orchestration) │    │  Inference Engine    │ │
│  └──────────────────┘    └────────┬────────┘    └──────────────────────┘ │
│                                   │                                      │
│                                   ├─→ Grad-CAM Heatmap Engine            │
│                                   ├─→ Google Gemini / Groq LLM Client    │
│                                   └─→ ReportLab PDF Generator            │
│                                                                          │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
                          ┌────────────────────┐
                          │ PostgreSQL Database │
                          │  (Users & History)  │
                          └────────────────────┘
```

Full architecture documentation: [`docs/architecture.md`](docs/architecture.md)

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
1. **Resize**: Rescaled to 224×224 pixels.
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances soft-tissue pulmonary contrast without amplifying noise.
3. **Random Rotation & Flip**: Slight rotational variance (±10°) and horizontal flipping (p=0.3).
4. **ImageNet Normalization**: Channel-wise mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

Full dataset documentation: [`docs/dataset.md`](docs/dataset.md)

---

## 🧠 Model Architecture & Model Selection

### Architecture Choice: EfficientNet-B0
- **Parameter Efficiency**: 5.3M parameters (compared to ResNet-50's 25.6M), enabling low-latency inference on CPU (~30ms).
- **Compound Scaling**: EfficientNet balances network depth, width, and image resolution using a fixed compound coefficient (φ).
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
```
pos_weight = N_normal / N_pneumonia = 1341 / 3875 ≈ 0.346
```

#### Justification over Weighted Random Sampler
Weighted loss directly scales gradient contributions without duplicating minority samples in mini-batches. This prevents the model from overfitting to repeated augmented copies of normal scans.

---

## 📈 Empirical Test Performance & Evaluation

Evaluating the fine-tuned EfficientNet-B0 model on the unseen Kaggle test dataset (N=624 images: 234 Normal, 390 Pneumonia):

| Metric | Score | Clinical Target | Assessment |
| :--- | :---: | :---: | :--- |
| **Recall / Sensitivity (Pneumonia)** | **98.72%** | ≥96.0% | **Exceeds Target** |
| **ROC-AUC** | **0.9515** | ≥0.950 | **Meets Target** |
| **F1-Score** | **0.9006** | ≥0.880 | **Exceeds Target** |
| **Overall Accuracy** | **86.38%** | ≥85.0% | **Meets Target** |
| **Precision (Pneumonia)** | **82.80%** | ≥80.0% | **Meets Target** |

### Confusion Matrix
|  | Predicted Normal | Predicted Pneumonia |
|--|-----------------|---------------------|
| **Actual Normal** (234) | 154 (TN) | 80 (FP) |
| **Actual Pneumonia** (390) | 5 (FN) | 385 (TP) |

### Clinical Trade-Off Analysis
In emergency triage, **false negatives are life-threatening** (sending a pneumonia patient home untreated). By weighting the loss function toward positive sensitivity, PulmoSight achieves **98.72% recall** (missing only 5 out of 390 real pneumonia scans). The 80 false positives prompt follow-up review rather than diagnostic omission.

Full training report: [`docs/training-report.md`](docs/training-report.md)

---

## 🔍 Explainability Engine (Grad-CAM & Region Mapping)

1. **Custom Grad-CAM Implementation**: Hooks into the final convolutional block (`features[-1]`) of EfficientNet-B0 to extract gradients and activations. No external Grad-CAM library required.
2. **Heatmap Overlay**: Blends normalized activation maps with original X-rays using a jet colormap — pure NumPy/PIL implementation without Matplotlib.
3. **Anatomical Spatial Analysis**: Maps the heatmap to a 3×3 grid corresponding to lung quadrants (apex, mid-zone, base, mediastinum) to feed structured textual descriptions into the LLM prompt.

---

## 🤖 LLM Integration & Prompt Engineering

PulmoSight uses **Google Gemini 2.0 Flash** (`google-generativeai` SDK) as the primary LLM provider, with **Groq Llama 3.1 70B** as an automatic secondary fallback. Both are **free-tier APIs** — no paid subscription required.

### Dual-Provider Fallback Chain
1. **Primary**: Google Gemini 2.0 Flash (fast, free, 30s timeout)
2. **Fallback**: Groq Llama 3.1 70B (free tier, 30s timeout)
3. **Canned fallback**: If both fail, return a structured template report

### Structured Prompt Design
Instead of unconstrained "explain this image" prompts, the prompt enforces strict formatting:
- **Inputs**: Prediction Label, Confidence %, Model Version, Grad-CAM Anatomical Observation, Patient Age/Gender/Symptoms.
- **Enforced sections**: `Clinical Summary`, `Possible Differential Considerations`, `Recommended Next Steps`, `Limitations of AI Analysis`, `Urgency Level`, and `Disclaimer`.

---

## 📄 Hospital-Style PDF Generation

Generated using `reportlab` (streaming PDF output directly to the browser):
- **Layout**: Clean header with institution branding, patient metadata table, diagnostic badge, embedded Grad-CAM heatmap, formatted LLM report sections, and legal disclaimer.
- **Choice over weasyprint**: reportlab is pure Python — no system-level dependencies (libcairo, pango), making Docker builds simpler and faster.

---

## 🔒 Authentication & History Management

- **JWT Auth**: Password hashing with `passlib` (bcrypt), 15-minute access tokens, and 7-day refresh tokens.
- **Paginated History**: `GET /api/v1/history` supports pagination, search filtering by filename/label, and sorting.
- **Cascade Deletion**: Deleting a record purges associated images and heatmaps from storage.

---

## 🔌 API Reference

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `GET` | `/api/v1/health` | Service & DB health check | No |
| `GET` | `/api/v1/model-info` | Model architecture & metrics | No |
| `GET` | `/api/v1/metrics` | Runtime request metrics | No |
| `POST` | `/api/v1/auth/register` | Register new user | No |
| `POST` | `/api/v1/auth/login` | Authenticate & get JWT | No |
| `POST` | `/api/v1/auth/refresh` | Refresh access token | No |
| `GET` | `/api/v1/auth/me` | Current user profile | Yes |
| `POST` | `/api/v1/predict` | Upload X-Ray for inference | Yes |
| `POST` | `/api/v1/generate-report` | Generate LLM clinical report | Yes |
| `GET` | `/api/v1/history` | Paginated prediction history | Yes |
| `GET` | `/api/v1/prediction/{id}` | Full prediction details | Yes |
| `DELETE` | `/api/v1/prediction/{id}` | Delete prediction & files | Yes |
| `GET` | `/api/v1/prediction/{id}/heatmap` | Grad-CAM heatmap image | Yes |
| `GET` | `/api/v1/prediction/{id}/image` | Original uploaded image | Yes |
| `GET` | `/api/v1/prediction/{id}/pdf` | Stream PDF report | Yes |

Interactive OpenAPI docs: `http://localhost:8000/docs`

Full API reference: [`docs/api-reference.md`](docs/api-reference.md)

---

## 📁 Folder Structure

```text
pulmosight/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # Thin route controllers
│   │   ├── auth/            # JWT security & dependencies
│   │   ├── database/        # SQLAlchemy async session
│   │   ├── inference/       # Model loading & prediction
│   │   ├── llm/             # Gemini/Groq client & prompts
│   │   ├── models/          # SQLAlchemy ORM models
│   │   ├── schemas/         # Pydantic v2 schemas
│   │   ├── services/        # Business logic orchestration
│   │   ├── utils/           # Validation, file handling, logging
│   │   ├── xai/             # Grad-CAM, overlay, observation
│   │   ├── config.py        # Pydantic settings
│   │   └── main.py          # FastAPI application entry
│   ├── alembic/             # Database migrations
│   ├── tests/               # pytest suite (8+ tests)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── app/                 # Next.js App Router pages
│   ├── lib/                 # API client (Axios)
│   ├── Dockerfile
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── package.json
├── model/
│   ├── train.py             # EfficientNet-B0 training script
│   ├── dataset.py           # Custom PyTorch Dataset
│   ├── evaluate.py          # Test evaluation & metrics
│   └── config.yaml          # Training hyperparameters
├── weights/                 # Trained model checkpoint (gitignored)
├── docs/
│   ├── architecture.md
│   ├── api-reference.md
│   ├── dataset.md
│   ├── training-report.md
│   └── training_metrics.json
├── docker-compose.yml
├── .env.example
├── .gitignore
├── SPEC.md
└── README.md
```

---

## 💻 Model Training Guide (Local & Google Colab GPU)

### Option 1: Google Colab (Recommended for Fast GPU Training)
1. Open Google Colab in your browser.
2. Open `model/train_pulmosight.ipynb`.
3. Select **GPU (T4)** from Runtime → Change runtime type.
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
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

Required variables:
```env
DATABASE_URL=postgresql+asyncpg://pulmosight:pulmosight_secret@postgres:5432/pulmosight
SECRET_KEY=<random-64-char-hex-string>
GOOGLE_API_KEY=<your-google-ai-studio-key>
GROQ_API_KEY=<your-groq-api-key>
```

### 2. Launch with Docker Compose
```bash
docker-compose up --build
```

### 3. Access Services
- **Web UI**: [http://localhost:3000](http://localhost:3000)
- **FastAPI OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## 🧪 Testing Strategy

Run the backend pytest suite:
```bash
cd backend
pytest tests/ -v
```

Or inside Docker:
```bash
docker-compose exec backend pytest tests/ -v
```

Tests cover:
- Health check endpoint
- User registration & duplicate email handling
- Login validation & credential rejection
- File validation (corrupted files, invalid extensions)
- Prediction pipeline end-to-end
- Prediction detail retrieval
- Record deletion & 404 confirmation after deletion

---

## 🚧 Known Limitations

1. **Pediatric dataset bias**: Model trained exclusively on pediatric chest X-rays — adult radiograph performance is unvalidated.
2. **Single-institution source**: All training images from one medical center, limiting generalizability.
3. **Binary classification only**: Does not distinguish between bacterial and viral pneumonia subtypes.
4. **No DICOM support**: Accepts JPEG/PNG only, not native DICOM format.
5. **Free-tier LLM rate limits**: Google Gemini and Groq free tiers have rate limits; high-volume usage may trigger fallback reports.

---

## 🔮 Future Improvements

1. **Multi-class classification**: Distinguish bacterial vs viral pneumonia and other pathologies (TB, lung nodules).
2. **DICOM ingestion**: Native DICOM parsing with proper windowing and metadata extraction.
3. **ONNX export**: Convert model to ONNX for faster cross-platform inference.
4. **WebSocket real-time updates**: Push prediction status updates to the frontend in real time.
5. **Role-based access control (RBAC)**: Differentiate between radiologists, clinicians, and administrators.
6. **Audit logging**: Track all prediction and report generation events for regulatory compliance.
7. **Multi-language reports**: Generate clinical reports in multiple languages.

---

## ⚠️ Ethical & Clinical Disclaimer

> **IMPORTANT**: PulmoSight is an AI decision-support system built for **educational, portfolio demonstration, and research purposes only**. It is NOT a certified medical device (FDA/CE) and must never be used as the sole basis for clinical diagnosis, treatment decisions, or patient triage. All predictions and generated reports must be validated by a licensed physician or radiologist.

> **Patient Data Notice**: The optional patient age, gender, and symptoms fields are provided solely as LLM prompt context to improve report relevance. They are NOT real Protected Health Information (PHI) and should not be used to store actual patient records. PulmoSight does not claim HIPAA compliance.
