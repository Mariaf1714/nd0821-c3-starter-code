import pytest
import pandas as pd
from .ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def data():
    """ Simple function to generate some fake Pandas data."""
    df = pd.DataFrame(
        {
            "age": [29, 50, 35, 60],
            "capital_gain": [0, 0, 0, 0],
            "hours_per_week": [20, 40, 40, 40],
            "salary": [0, 1, 0, 1]
        }
    )

    X_train = df.iloc[:2, :3]
    y_train = df["salary"][:2]
    X_test = df.iloc[2:, :3]
    y_test = df["salary"][2:]


    return X_train, y_train, X_test, y_test


def test_train_model(data):
    """
    Tests the function train_model.

    Inputs
    ------
    data : fake data consisting of X_train, y_train, X_test and y_test
    """
    X_train, y_train, _, _ = data
    model = train_model(X_train, y_train)

    # test whether model is fitted
    assert not len(dir(model)) == len(dir(type(model)()))


def test_inference(data):
    """ Tests the function inference.

    Inputs
    ------
    data : pytest fixture
        fake data consisting of X_train, y_train, X_test and y_test
    train_model : function
        Function for training model
    """

    X_train, y_train, X_test, _ = data
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert len(preds) == 2


def test_compute_model_metrics(data):
    """ Tests the function compute_model_metrics.

    Inputs
    ------
    data : pytest fixture
        fake data consisting of X_train, y_train, X_test and y_test
    train_model : function
        Function for training model
    inference : function
        Function for predicting values
    """
    X_train, y_train, X_test, y_test = data
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert 0.0 <= precision <=  1.0
    assert 0.0 <= recall <=  1.0
    assert 0.0 <= fbeta <=  1.0









