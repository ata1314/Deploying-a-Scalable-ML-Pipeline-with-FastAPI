import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)


@pytest.fixture(scope="session")
def data():
    """Load and return the census dataset."""
    df = pd.read_csv("data/census.csv")
    df.columns = df.columns.str.strip()
    df = df.apply(
        lambda x: x.str.strip() if x.dtype == "object" else x
    )
    return df


@pytest.fixture(scope="session")
def trained_model(data):
    """Train and return a model with encoder and lb."""
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    train, _ = train_test_split(
        data, test_size=0.2, random_state=42
    )
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X_train, y_train)
    return model, encoder, lb


def test_train_model(trained_model):
    """
    Test that train_model returns a model that has predict
    and fit methods, confirming it is a valid sklearn model.
    """
    model, _, _ = trained_model
    assert hasattr(model, "predict")
    assert hasattr(model, "fit")


def test_inference(trained_model, data):
    """
    Test that inference returns predictions with the correct
    shape and type (numpy array).
    """
    model, encoder, lb = trained_model
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    _, test = train_test_split(
        data, test_size=0.2, random_state=42
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_test)
    assert preds.shape[0] == X_test.shape[0]
    assert isinstance(preds, np.ndarray)


def test_compute_model_metrics():
    """
    Test that compute_model_metrics returns the expected
    precision, recall, and fbeta values for known inputs.
    """
    y = np.array([1, 0, 1, 1, 0])
    preds = np.array([1, 0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
    assert precision == 1.0
    assert recall == pytest.approx(2 / 3)
