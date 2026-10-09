import pytest
from fastapi.testclient import TestClient
from main import app
from database import Base, engine

Base.metadata.create_all(bind=engine)
client = TestClient(app)

def test_inference_no_file():
    response = client.post("/api/inference")
    assert response.status_code == 422

def test_get_transactions():
    response = client.get("/api/transactions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
