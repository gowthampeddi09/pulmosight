import pytest

@pytest.mark.asyncio
async def test_register_and_login(client):
    # Test User Registration
    reg_payload = {
        "email": "doctor@hospital.com",
        "password": "SecurePassword123!",
        "full_name": "Dr. Sarah Smith"
    }
    res = await client.post("/api/v1/auth/register", json=reg_payload)
    assert res.status_code == 201
    data = res.json()
    assert "access_token" in data
    assert "refresh_token" in data

    # Test Duplicate Registration Prevention
    res_dup = await client.post("/api/v1/auth/register", json=reg_payload)
    assert res_dup.status_code == 409
    assert res_dup.json()["error"]["code"] == "EMAIL_EXISTS"

    # Test Login
    login_payload = {
        "email": "doctor@hospital.com",
        "password": "SecurePassword123!"
    }
    res_login = await client.post("/api/v1/auth/login", json=login_payload)
    assert res_login.status_code == 200
    assert "access_token" in res_login.json()

    # Test Protected /me endpoint
    token = res_login.json()["access_token"]
    res_me = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res_me.status_code == 200
    assert res_me.json()["email"] == "doctor@hospital.com"


@pytest.mark.asyncio
async def test_invalid_login(client):
    login_payload = {
        "email": "nonexistent@hospital.com",
        "password": "WrongPassword"
    }
    res = await client.post("/api/v1/auth/login", json=login_payload)
    assert res.status_code == 401
    assert res.json()["error"]["code"] == "INVALID_CREDENTIALS"
