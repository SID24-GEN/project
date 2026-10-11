import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.train import (
    FEATURES,
    N_CLUSTERS,
    create_model,
    find_best_k,
    load_data,
    prepare_data,
    train_model
)

from ml.predict import (
    predict_segment,
    validate_customer_data
)


# ---------------------------------------
# Test 1: dataset
# ---------------------------------------

def test_load_data():

    df = load_data()

    assert len(df) == 200

    for column in FEATURES:
        assert column in df.columns

    assert not df[FEATURES].isnull().any().any()


# ---------------------------------------
# Test 2: prepare_data()
# ---------------------------------------

def test_prepare_data():

    X = prepare_data(load_data())

    assert list(X.columns) == FEATURES


# ---------------------------------------
# Test 3: create_model()
# ---------------------------------------

def test_create_model():

    model = create_model()

    assert isinstance(model, Pipeline)

    assert isinstance(model.named_steps["scaler"], StandardScaler)

    assert isinstance(model.named_steps["kmeans"], KMeans)

    assert model.named_steps["kmeans"].n_clusters == N_CLUSTERS


# ---------------------------------------
# Test 4: training
# ---------------------------------------

def test_train_model():

    model, X, labels, profiles = train_model()

    assert len(labels) == len(X)

    assert len(set(labels)) == N_CLUSTERS

    assert len(profiles) == N_CLUSTERS

    assert sum(p["size"] for p in profiles.values()) == len(X)


# ---------------------------------------
# Test 5: high-income, low-spending group exists
# ---------------------------------------

def test_target_segment_exists():

    _, _, _, profiles = train_model()

    names = [p["name"] for p in profiles.values()]

    assert "Untapped High Earners" in names


# ---------------------------------------
# Test 6: choosing k
# ---------------------------------------

def test_find_best_k():

    X = prepare_data(load_data())

    results = find_best_k(X, 2, 6)

    assert list(results["k"]) == [2, 3, 4, 5, 6]

    # inertia must fall as k grows
    assert results["inertia"].is_monotonic_decreasing


# ---------------------------------------
# Test 7: predict_segment()
# ---------------------------------------

def test_predict_segment():

    model, _, _, profiles = train_model()

    artifact = {"model": model, "profiles": profiles}

    result = predict_segment(artifact, 40, 90, 15)

    assert result["name"] == "Untapped High Earners"

    assert result["cluster"] in profiles


# ---------------------------------------
# Test 8: input validation
# ---------------------------------------

def test_validate_customer_data():

    assert validate_customer_data(30, 60, 50) == []

    assert len(validate_customer_data(10, 60, 50)) == 1

    assert len(validate_customer_data(30, -5, 150)) == 2
