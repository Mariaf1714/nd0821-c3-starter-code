import pytest
from fastapi.testclient import TestClient
from starter.main import app


client = TestClient(app)


@pytest.fixture()
def client():
    """Client for API testing"""
    client = TestClient(app)
    return client


def test_welcome(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome!"}


def test_positive_inference(client):
    item = {
        'age': 28,
        'workclass': 'Private',
        'fnlgt': 215646,
        'education': 'HS-grad',
        'education-num': 8,
        'marital-status': 'Divorced',
        'occupation': 'Handlers-cleaners',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 20,
        'native-country': 'United-States',
        'salary': '<=50K'
    }
    r = client.post("/predict", json=item)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_negative_inference(client):
    item = {
        'age': 52,
        'workclass': 'Self-emp-not-inc',
        'fnlgt': 209642,
        'education': 'Masters',
        'education-num': 13,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 45,
        'native-country': 'United-States',
        'salary': '>50K'
    }
    r = client.post("/predict", json=item)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}
