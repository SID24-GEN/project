import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from ml.train import prepare_data, create_model
from ml.predict import predict_price


# Load original dataset
df = pd.read_csv("data/house_data.csv")


# ---------------------------------------
# Test 1: prepare_data()
# ---------------------------------------

X, y = prepare_data(df)

expected_columns = [
	"Area_sqft",
	"Bedrooms",
	"Bathrooms",
	"Age"
]

assert list(X.columns) == expected_columns
assert y.name == "Price"

print("Test 1 passed: prepare_data()")


# ---------------------------------------
# Test 2: create_model()
# ---------------------------------------

model = create_model()

assert isinstance(model, Pipeline)

assert isinstance(
	model.named_steps["scaler"],
	StandardScaler
)

assert isinstance(
	model.named_steps["regressor"],
	LinearRegression
)

print("Test 2 passed: create_model()")


# ---------------------------------------
# Test 3: Train model
# ---------------------------------------

model.fit(X, y)

predictions = model.predict(X)

assert len(predictions) == len(y)

assert np.all(np.isfinite(predictions))

print("Test 3 passed: model training and prediction")


# ---------------------------------------
# Test 4: predict_price()
# ---------------------------------------

price = predict_price(
	model,
	area=1350,
	bedrooms=3,
	bathrooms=2,
	age=6
)

assert np.isscalar(price)

assert np.isfinite(price)

print("Test 4 passed: predict_price()")


print("\nAll model tests passed.")
