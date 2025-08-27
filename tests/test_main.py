import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get('/')
    assert response.status_code == 200
    assert 'ASL Auto-QC and CBF Estimator' in response.json().get('detail', '')


def test_process_data():
    payload = {'sample_id': 1, 'value': 2.0}
    response = client.post('/api/process', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data['sample_id'] == 1
    assert data['result'] == 4.0
