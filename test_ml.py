import pytest
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics
from ml.data import process_data

# Load sample data for testing
df = pd.read_csv(os.path.join("data", "census.csv"))
train, test = df[:3000], df[3000:3500]  # small subsets for fast testing
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

# Process the data
X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)

def test_train_model_returns_correct_type():
    """Test if the train_model function returns a RandomForestClassifier."""
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_compute_model_metrics_output():
    """Test compute_model_metrics returns 3 float values."""
    model = train_model(X_train, y_train)
    preds = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

def test_processed_data_shape():
    """Test if processed data returns matching samples between X and y."""
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
