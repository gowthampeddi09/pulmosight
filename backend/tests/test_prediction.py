import pytest
import io

@pytest.mark.asyncio
async def test_health_check(client):
    res = await client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "database" in data


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

    pred_id = data["id"]

    # Retrieve Detail
    detail_res = await client.get(f"/api/v1/prediction/{pred_id}", headers=headers)
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == pred_id

    # Test Delete
    del_res = await client.delete(f"/api/v1/prediction/{pred_id}", headers=headers)
    assert del_res.status_code == 204

    # Confirm 404 after deletion
    detail_res_after = await client.get(f"/api/v1/prediction/{pred_id}", headers=headers)
    assert detail_res_after.status_code == 404
