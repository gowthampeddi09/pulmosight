ROLE
You are a senior full-stack AI engineer. Build a production-grade, portfolio-quality
medical AI web application from scratch for a technical hiring assignment. Code
quality, architecture, and correctness matter more than speed. Write code the way
an experienced engineer would: varied, purposeful, commented only where it matters,
with real error handling — not uniform, templated, over-commented AI-style boilerplate.

PROJECT
Advanced AI Medical Intelligence Platform — Chest X-Ray Pneumonia Detection.
Upload a chest X-ray -> Pneumonia/Normal prediction with confidence -> Grad-CAM
heatmap -> LLM-generated structured clinical-style report -> downloadable PDF ->
browsable prediction history.

DISEASE & TASK
- Binary classification: Pneumonia vs Normal, Kaggle "Chest X-Ray Images (Pneumonia)"
  dataset (Paul Mooney). Tell me exactly where to place the downloaded dataset;
  do not try to download it yourself.
- Optimize primarily for recall/sensitivity on the Pneumonia class (a missed real
  case is the costlier clinical error), while reporting precision/recall/F1/ROC-AUC/
  confusion matrix in full.
- Handle the dataset's known class imbalance with weighted loss or a weighted
  sampler — pick one and justify the choice in a code comment.

TECH STACK
- Backend: FastAPI, Python 3.11, SQLAlchemy ORM, PostgreSQL, JWT auth, Pydantic v2
- DL: PyTorch, EfficientNet-B0 (timm or torchvision weights, transfer learning)
- XAI: Grad-CAM (implement or use pytorch-grad-cam) on the final conv block
- LLM: Google Gemini 2.0 Flash (primary, free tier) via `google-generativeai` SDK +
  Groq Llama 3.1 70B (fallback, free tier) — structured prompt, not "explain this"
- Frontend: Next.js (App Router), TypeScript, Tailwind CSS
- Deployment: Docker + docker-compose (frontend, backend, postgres)
- Testing: pytest, 8-12 real tests covering API endpoints and service-layer logic

FOLDER STRUCTURE
pulmosight/
  backend/
    app/
      api/          (route definitions only — thin controllers)
      services/      (business logic, orchestrates inference/llm/db)
      models/        (SQLAlchemy ORM models)
      schemas/       (Pydantic request/response schemas)
      database/      (session, engine, alembic migrations)
      utils/         (image validation, logging config, file handling)
      inference/     (model loading + prediction logic)
      xai/           (grad-cam implementation)
      llm/           (prompt construction + Gemini/Groq API client)
    tests/
    Dockerfile
    requirements.txt
  frontend/  (Next.js app + Dockerfile)
  model/
    train.py
    dataset.py
    evaluate.py
    config.yaml
  weights/   (trained checkpoint — gitignored, documented in README)
  docs/
    architecture.md
    api-reference.md
    dataset.md
    training-report.md
  docker-compose.yml
  .env.example
  README.md

BACKEND ARCHITECTURE (strict layering — do not collapse this)
Routes -> Service Layer -> Inference Engine / LLM Client / DB Repository
- Route handlers contain no business logic — only validation, calling the service
  layer, and shaping the HTTP response.
- The service layer orchestrates: validate image -> run inference -> run Grad-CAM
  -> call LLM -> persist to DB -> return result.
- Never import the DB session directly into a route handler; use FastAPI Depends.

API ENDPOINTS
- POST /api/v1/predict — image upload -> prediction, confidence, processing_time,
  model_version, heatmap URL
- POST /api/v1/generate-report — prediction_id (+ optional patient_age, gender,
  symptoms) -> LLM-generated structured report
- GET /api/v1/history — paginated, search/sort/filter by label and date
- GET /api/v1/prediction/{id} — full record including heatmap and report
- DELETE /api/v1/prediction/{id}
- GET /api/v1/health — DB connectivity + model-loaded status
- GET /api/v1/model-info — architecture, version, training date, metrics summary
- GET /api/v1/metrics — basic runtime metrics (request count, avg latency)
All endpoints return proper status codes (400/404/422/500) with a consistent JSON
error shape: {"error": {"code": ..., "message": ...}}.

DATABASE SCHEMA (predictions table)
id (UUID, PK), filename, prediction (enum PNEUMONIA/NORMAL), confidence (float),
model_version (string), processing_time_ms (float), heatmap_path (string),
report_text (text, nullable), patient_age/gender/symptoms (nullable, LLM context
only — not real PHI, note clearly in README), created_at (timestamp).
Use Alembic for migrations, not create_all() in production code paths.

MODEL TRAINING (model/train.py)
- EfficientNet-B0, ImageNet-pretrained, fine-tune last blocks + new head
- Albumentations: RandomRotation, small-probability horizontal flip, CLAHE,
  Resize(224x224), Normalize (ImageNet stats)
- EarlyStopping on validation loss/recall, ModelCheckpoint for best weights
- Log every run's hyperparameters, final metrics, training curves (PNG in docs/)
- Auto-save accuracy/precision/recall/F1/ROC-AUC/confusion matrix/classification
  report to docs/training-report.md at the end of training

EXPLAINABILITY (app/xai/)
- Grad-CAM on the last conv block of EfficientNet-B0; save heatmap overlay, return
  its path/URL from /predict
- Also generate a short rule-based (non-LLM) textual observation of which lung
  region shows highest activation — feeds the LLM prompt as structured input

LLM INTEGRATION (app/llm/)
- Google Gemini 2.0 Flash via `google-generativeai` SDK (primary, free tier)
- Groq Llama 3.1 70B via `groq` SDK (secondary fallback, free tier)
- Never hardcode API keys — load from GOOGLE_API_KEY / GROQ_API_KEY env vars
- Structured prompt: prediction label, confidence, model version, Grad-CAM textual
  observation, optional patient age/gender/symptoms
- LLM must return: Clinical Summary, Possible Differential Considerations,
  Recommended Next Steps, Limitations of AI Analysis, Urgency Level, and a clear
  Disclaimer (not a diagnostic tool, requires physician review)
- Validate the LLM response before storing; handle timeouts/rate limits with a
  graceful fallback message, never crash the request

PDF REPORT GENERATION
- Hospital-style layout: patient info (if any), prediction + confidence, embedded
  heatmap, LLM report sections, disclaimer, timestamp, model version
- Use reportlab or weasyprint — pick one, justify briefly in a code comment

FRONTEND (Next.js)
- Upload page: drag-and-drop/file picker with client-side validation before upload
- Result view: original image -> prediction badge -> confidence gauge (visual) ->
  Grad-CAM heatmap toggle -> "Generate Report" -> LLM report -> "Download PDF"
- History page: search, sort by date/label, delete, click-through to detail
- Dark mode toggle
- Loading and error states on every async action — no silent failures

VALIDATION & ROBUSTNESS
- Reject non-image files, corrupted images, and low-resolution images with a
  specific 400 reason
- Pydantic validates all request bodies including patient fields
- Python `logging` module throughout (INFO/WARNING/ERROR) — no bare print()

TESTING
8-12 pytest tests: successful prediction flow, invalid file type rejection,
corrupted image rejection, history pagination, delete (incl. 404 on missing id),
LLM service failure fallback, and at least one direct test on the Grad-CAM/
inference service layer.

DEPLOYMENT
Dockerfile for backend, Dockerfile for frontend, docker-compose.yml wiring
backend+frontend+postgres with env vars and healthchecks. .env.example listing
every required variable with a placeholder — never commit a real .env.

DOCUMENTATION (docs/ + README.md)
Overview, architecture diagram (ASCII/mermaid fine), folder structure, setup
(local + Docker), API reference summary (or Swagger link at /docs), dataset
description/source, training summary with final metrics table, explainability
approach in plain language, known limitations, ethical/clinical disclaimer,
and a Future Improvements section.

WHAT NOT TO DO
- Don't put everything in one main.py/app.py, don't skip the service layer.
- Don't fabricate metrics — train on real data, report real numbers, explain
  them honestly even if recall isn't perfect.
- Don't claim this is a diagnostic device anywhere — frame it as decision-support/
  portfolio demonstration only.
- No print() logging, no hardcoded secrets, no skipped validation "for a demo."

DELIVERABLE CHECKLIST
Before calling this done, confirm: model trained on real data with real metrics
in docs/, all 7+ endpoints working end-to-end through the frontend, Grad-CAM
heatmaps genuinely vary per image, LLM report generation works with graceful
fallback, PDF downloads correctly, docker-compose brings up the full stack from
a clean clone, README complete, pytest suite passes.

Start by scaffolding the folder structure and a working /health endpoint
end-to-end through Docker before writing any model or frontend code, so the
deployment path is validated early rather than at the end.