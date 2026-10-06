import os
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------
# Paths
# ---------------------------------------

DATA_PATH = "data/house_data.csv"
MODEL_PATH = "models/house_model.pkl"


# ---------------------------------------
# Load Dataset
# ---------------------------------------

def load_data():

	df = pd.read_csv(DATA_PATH)

	return df


# ---------------------------------------
# Prepare Features and Target
# ---------------------------------------

def prepare_data(df):

	features = [
		"Area_sqft",
		"Bedrooms",
		"Bathrooms",
		"Age"
	]

	target = "Price"

	X = df[features]
	y = df[target]

	return X, y


# ---------------------------------------
# Create ML Pipeline
# ---------------------------------------

def create_model():

	model = Pipeline([
		("scaler", StandardScaler()),
		("regressor", LinearRegression())
	])

	return model


# ---------------------------------------
# Train Model
# ---------------------------------------

def train_model():

	df = load_data()

	X, y = prepare_data(df)

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42
	)

	model = create_model()

	model.fit(X_train, y_train)

	return model, X_test, y_test


# ---------------------------------------
# Save Model
# ---------------------------------------

def save_model(model):
	
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
	
    with open(MODEL_PATH, "wb") as file:
		
        pickle.dump(model, file)


# ---------------------------------------
# Main
# ---------------------------------------

if __name__ == "__main__":

	model, X_test, y_test = train_model()

	save_model(model)

	print("Training completed successfully.")
