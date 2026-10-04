from flask import (
	Flask,
	render_template,
	request
)

from ml.predict import (
	load_model,
	predict_price
)

from utils.validation import (
	validate_house_data
)


# ---------------------------------------
# Flask App
# ---------------------------------------

app = Flask(__name__)


# ---------------------------------------
# Load Model Once
# ---------------------------------------

model = load_model()


# ---------------------------------------
# Home
# ---------------------------------------

@app.route("/")
def home():

	return render_template(
		"index.html"
	)


# ---------------------------------------
# Prediction
# ---------------------------------------

@app.route(
	"/predict",
	methods=["POST"]
)
def predict():

	try:

		area = float(
			request.form["area"]
		)

		bedrooms = int(
			request.form["bedrooms"]
		)

		bathrooms = int(
			request.form["bathrooms"]
		)

		age = float(
			request.form["age"]
		)


		# Validate input

		errors = validate_house_data(
			area,
			bedrooms,
			bathrooms,
			age
		)


		if errors:

			return render_template(
				"index.html",
				errors=errors
			)


		# Predict

		price = predict_price(
			model,
			area,
			bedrooms,
			bathrooms,
			age
		)


		return render_template(
			"index.html",
			prediction=price
		)


	except ValueError:

		return render_template(
			"index.html",
			errors=[
				"Please enter valid numbers."
			]
		)


# ---------------------------------------
# Start Server
# ---------------------------------------

if __name__ == "__main__":

	app.run(
		debug=True
	)
