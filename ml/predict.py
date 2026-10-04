import pickle


MODEL_PATH = "models/house_model.pkl"


# ---------------------------------------
# Load Model
# ---------------------------------------

def load_model():

	with open(MODEL_PATH, "rb") as file:

		model = pickle.load(file)

	return model


# ---------------------------------------
# Predict House Price
# ---------------------------------------

def predict_price(
	model,
	area,
	bedrooms,
	bathrooms,
	age
):

	input_data = [[
		area,
		bedrooms,
		bathrooms,
		age
	]]

	prediction = model.predict(input_data)

	return prediction[0]


# ---------------------------------------
# Main
# ---------------------------------------

if __name__ == "__main__":

	model = load_model()

	price = predict_price(
		model,
		area=1350,
		bedrooms=3,
		bathrooms=2,
		age=6
	)

	print(
		f"Predicted Price: ₹{price:,.2f}"
	)
