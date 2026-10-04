import pickle

from sklearn.metrics import (
	mean_absolute_error,
	mean_squared_error,
	r2_score
)

from ml.train import train_model


MODEL_PATH = "models/house_model.pkl"


# ---------------------------------------
# Load Saved Model
# ---------------------------------------

def load_model():

	with open(MODEL_PATH, "rb") as file:
		model = pickle.load(file)

	return model


# ---------------------------------------
# Evaluate Model
# ---------------------------------------

def evaluate_model(model, X_test, y_test):

	predictions = model.predict(X_test)

	mae = mean_absolute_error(
		y_test,
		predictions
	)

	mse = mean_squared_error(
		y_test,
		predictions
	)

	rmse = mse ** 0.5

	r2 = r2_score(
		y_test,
		predictions
	)

	return {
		"MAE": mae,
		"MSE": mse,
		"RMSE": rmse,
		"R2": r2
	}


# ---------------------------------------
# Main
# ---------------------------------------

if __name__ == "__main__":

	model, X_test, y_test = train_model()

	results = evaluate_model(
		model,
		X_test,
		y_test
	)

	print("\nModel Evaluation")
	print("----------------")

	for metric, value in results.items():

		print(f"{metric}: {value:.4f}")
