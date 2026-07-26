# PulmoSight API Reference

Base URL: `http://localhost:8000/api/v1`

Interactive OpenAPI docs available at: `http://localhost:8000/docs`

## Authentication

All protected endpoints require a JWT Bearer token in the `Authorization` header:
```
Authorization: Bearer <access_token>
```

---

## System Endpoints

### `GET /health`
**Auth**: None  
**Description**: Check database connectivity and model initialization state.

**Response** `200`:
```json
{
  "status": "healthy",
  "database": "connected",
  "model_loaded": true
}
```

### `GET /model-info`
**Auth**: None  
**Description**: Retrieve metadata about the loaded PyTorch model.

**Response** `200`:
```json
{
  "architecture": "EfficientNet-B0",
  "version": "1.0.0",
  "training_date": "2026-07-25",
  "input_size": "224x224",
  "num_classes": 2,
  "labels": ["NORMAL", "PNEUMONIA"],
  "model_loaded": true,
  "metrics": { "accuracy": 0.8638, "recall_pneumonia": 0.9872 }
}
```

### `GET /metrics`
**Auth**: None  
**Description**: Basic runtime metrics.

**Response** `200`:
```json
{
  "request_count": 42,
  "average_latency_ms": 312.5,
  "prediction_count": 38,
  "report_count": 12
}
```

---

## Auth Endpoints

### `POST /auth/register`
**Auth**: None

**Body**:
```json
{
  "email": "doctor@hospital.com",
  "password": "SecurePassword123!",
  "full_name": "Dr. Sarah Smith"
}
```

**Response** `201`:
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

### `POST /auth/login`
**Auth**: None

**Body**:
```json
{
  "email": "doctor@hospital.com",
  "password": "SecurePassword123!"
}
```

**Response** `200`: Same as register response.

### `POST /auth/refresh`
**Auth**: None

**Body**:
```json
{ "refresh_token": "eyJ..." }
```

### `GET /auth/me`
**Auth**: Required

**Response** `200`:
```json
{
  "id": "uuid",
  "email": "doctor@hospital.com",
  "full_name": "Dr. Sarah Smith",
  "is_active": true,
  "created_at": "2026-07-25T10:30:00Z"
}
```

---

## Prediction Endpoints

### `POST /predict`
**Auth**: Required  
**Content-Type**: `multipart/form-data`

**Body**: `file` — JPEG or PNG chest X-ray image (max 10 MB, min 100x100px)

**Response** `200`:
```json
{
  "id": "uuid",
  "filename": "xray_abc123.jpg",
  "prediction": "PNEUMONIA",
  "confidence": 0.9234,
  "model_version": "1.0.0",
  "processing_time_ms": 42.3,
  "heatmap_url": "/api/v1/prediction/{id}/heatmap",
  "gradcam_observation": "Primary finding: strong activation in the lower-left...",
  "created_at": "2026-07-25T10:30:00Z"
}
```

**Error Responses**: `400` (INVALID_FILE_TYPE, CORRUPTED_IMAGE, LOW_RESOLUTION, FILE_TOO_LARGE)

### `POST /generate-report`
**Auth**: Required

**Body**:
```json
{
  "prediction_id": "uuid",
  "patient_age": 45,
  "patient_gender": "Male",
  "patient_symptoms": "persistent cough, fever"
}
```

**Response** `200`:
```json
{
  "prediction_id": "uuid",
  "report_text": "## Clinical Summary\n...",
  "generated_by": "gemini"
}
```

### `GET /history`
**Auth**: Required

**Query Parameters**: `page`, `per_page`, `label`, `search`, `sort_by`, `sort_order`

**Response** `200`:
```json
{
  "items": [ ...PredictionResponse objects... ],
  "total": 42,
  "page": 1,
  "per_page": 10,
  "total_pages": 5
}
```

### `GET /prediction/{id}`
**Auth**: Required  
**Response** `200`: Full prediction details including report_text, patient context.

### `DELETE /prediction/{id}`
**Auth**: Required  
**Response** `204`: No content. Deletes record and associated files.

### `GET /prediction/{id}/heatmap`
**Auth**: Required  
**Response** `200`: PNG image of Grad-CAM heatmap overlay.

### `GET /prediction/{id}/image`
**Auth**: Required  
**Response** `200`: Original uploaded X-ray image.

### `GET /prediction/{id}/pdf`
**Auth**: Required  
**Response** `200`: Streamed PDF clinical report (`application/pdf`).

---

## Error Format

All errors return a consistent JSON shape:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description"
  }
}
```

Common error codes: `INVALID_FILE_TYPE`, `CORRUPTED_IMAGE`, `LOW_RESOLUTION`, `FILE_TOO_LARGE`, `INVALID_ID`, `NOT_FOUND`, `INVALID_CREDENTIALS`, `EMAIL_EXISTS`, `INVALID_TOKEN`, `VALIDATION_ERROR`, `INTERNAL_ERROR`.
