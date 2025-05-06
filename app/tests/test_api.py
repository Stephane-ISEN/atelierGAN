import sys
import os
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from main import app

client = TestClient(app)

def test_generate_image():
    response = client.get("/generate_image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 1000  # Vérifie que l'image n'est pas vide