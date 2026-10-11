from flask import (
    Flask,
    render_template,
    request
)

from ml.predict import (
    load_model,
    predict_segment,
    validate_customer_data
)


# ---------------------------------------
# Flask App
# ---------------------------------------

app = Flask(__name__)


# ---------------------------------------
# Load Model Once
# ---------------------------------------

artifact = load_model()

profiles = sorted(
    artifact["profiles"].values(),
    key=lambda p: -p["avg_spending"]
)


# ---------------------------------------
# Home
# ---------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        profiles=profiles
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

        age = float(request.form["age"])

        income = float(request.form["income"])

        spending = float(request.form["spending"])


        # Validate input

        errors = validate_customer_data(
            age,
            income,
            spending
        )

        if errors:

            return render_template(
                "index.html",
                profiles=profiles,
                errors=errors
            )


        # Predict

        segment = predict_segment(
            artifact,
            age,
            income,
            spending
        )

        return render_template(
            "index.html",
            profiles=profiles,
            segment=segment
        )

    except ValueError:

        return render_template(
            "index.html",
            profiles=profiles,
            errors=["Please enter valid numbers."]
        )


# ---------------------------------------
# Start Server
# ---------------------------------------

if __name__ == "__main__":

    app.run(debug=True)
