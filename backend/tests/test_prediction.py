import pytest
import io

@pytest.mark.asyncio
async def test_health_check(client):
    res = await client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "database" in data
    assert "model_loaded" in data


@pytest.mark.asyncio
async def test_model_info(client):
    res = await client.get("/api/v1/model-info")
    assert res.status_code == 200
    data = res.json()
    assert data["architecture"] == "EfficientNet-B0"
    assert "version" in data
    assert "labels" in data
    assert data["num_classes"] == 2


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    res = await client.get("/api/v1/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "request_count" in data
    assert "average_latency_ms" in data
    assert "prediction_count" in data


@pytest.mark.asyncio
async def test_invalid_file_upload(client):
    # Register and obtain token
    reg = await client.post("/api/v1/auth/register", json={
        "email": "tech@lab.com", "password": "PassWord123!", "full_name": "Tech"
    })
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Upload invalid file type (txt file)
    files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    res = await client.post("/api/v1/predict", files=files, headers=headers)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_FILE_TYPE"


@pytest.mark.asyncio
async def test_corrupted_image_rejection(client):
    """Upload a file that has an image extension/content-type but contains garbage bytes."""
    reg = await client.post("/api/v1/auth/register", json={
        "email": "corrupt@lab.com", "password": "PassWord123!", "full_name": "Test"
    })
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Random garbage bytes with a .jpg extension
    files = {"file": ("corrupted.jpg", io.BytesIO(b"\x00\x01\x02\x03BADDATA"), "image/jpeg")}
    res = await client.post("/api/v1/predict", files=files, headers=headers)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "CORRUPTED_IMAGE"


@pytest.mark.asyncio
async def test_prediction_flow(client, dummy_image_bytes):
    # Register user
    reg = await client.post("/api/v1/auth/register", json={
        "email": "radiologist@lab.com", "password": "PassWord123!", "full_name": "Radiologist"
    })
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Upload valid JPEG X-ray
    files = {"file": ("xray.jpg", io.BytesIO(dummy_image_bytes), "image/jpeg")}
    res = await client.post("/api/v1/predict", files=files, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] in ["NORMAL", "PNEUMONIA"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "id" in data
    assert "processing_time_ms" in data
    assert "model_version" in data

    pred_id = data["id"]

    # Retrieve Detail
    detail_res = await client.get(f"/api/v1/prediction/{pred_id}", headers=headers)
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == pred_id

    # History should contain the prediction
    history_res = await client.get("/api/v1/history", headers=headers)
    assert history_res.status_code == 200
    assert history_res.json()["total"] >= 1

    # Test Delete
    del_res = await client.delete(f"/api/v1/prediction/{pred_id}", headers=headers)
    assert del_res.status_code == 204

    # Confirm 404 after deletion
    detail_res_after = await client.get(f"/api/v1/prediction/{pred_id}", headers=headers)
    assert detail_res_after.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_prediction(client):
    """Deleting a non-existent UUID should return 404."""
    reg = await client.post("/api/v1/auth/register", json={
        "email": "delete_test@lab.com", "password": "PassWord123!", "full_name": "Test"
    })
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    fake_id = "00000000-0000-0000-0000-000000000000"
    res = await client.delete(f"/api/v1/prediction/{fake_id}", headers=headers)
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_unauthenticated_predict_rejected(client):
    """Predict endpoint should reject requests without auth token."""
    files = {"file": ("test.jpg", io.BytesIO(b"fake"), "image/jpeg")}
    res = await client.post("/api/v1/predict", files=files)
    assert res.status_code == 403  # HTTPBearer returns 403 when no token


@pytest.mark.asyncio
async def test_invalid_uuid_returns_400(client):
    """Invalid UUID format should return 400, not 500."""
    reg = await client.post("/api/v1/auth/register", json={
        "email": "uuid_test@lab.com", "password": "PassWord123!", "full_name": "Test"
    })
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    res = await client.get("/api/v1/prediction/not-a-uuid", headers=headers)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_ID"
