import pytest
import numpy as np
from fastapi.testclient import TestClient
from starter.main import app
from starter.starter.ml.model import train_model


@pytest.fixture()
def trainedModel():

    X = np.array([[1, 1]])
    y = np.array([1])
    model = train_model(X, y)

    return model, X, y


client = TestClient(app)


@pytest.fixture()
def client():
    """Client for API testing"""
    client = TestClient(app)
    return client