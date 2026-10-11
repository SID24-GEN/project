import pickle

from ml.train import MODEL_PATH


# ---------------------------------------
# Load Model
# ---------------------------------------

def load_model():

    with open(MODEL_PATH, "rb") as file:

        artifact = pickle.load(file)

    return artifact


# ---------------------------------------
# Validate Customer Input
# ---------------------------------------

def validate_customer_data(age, income, spending):

    errors = []

    if not 18 <= age <= 100:
        errors.append("Age must be between 18 and 100.")

    if not 0 <= income <= 500:
        errors.append(
            "Annual income must be between 0 and 500 (in thousands)."
        )

    if not 1 <= spending <= 100:
        errors.append(
            "Spending score must be between 1 and 100."
        )

    return errors


# ---------------------------------------
# Predict Customer Segment
# ---------------------------------------

def predict_segment(artifact, age, income, spending):

    import pandas as pd

    input_data = pd.DataFrame(
        [[age, income, spending]],
        columns=[
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    cluster = int(artifact["model"].predict(input_data)[0])

    profile = artifact["profiles"][cluster]

    return {
        "cluster": cluster,
        **profile
    }


# ---------------------------------------
# Main
# ---------------------------------------

if __name__ == "__main__":

    artifact = load_model()

    result = predict_segment(
        artifact,
        age=40,
        income=90,
        spending=15
    )

    print(f"Segment  : {result['name']}")
    print(f"Strategy : {result['strategy']}")
