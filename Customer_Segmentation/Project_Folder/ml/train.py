import os
import pickle

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------
# Paths
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "mall_customers.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "segmentation_model.pkl")


# ---------------------------------------
# Settings
# ---------------------------------------

FEATURES = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

N_CLUSTERS = 5
RANDOM_STATE = 42


# ---------------------------------------
# Load Dataset
# ---------------------------------------

def load_data():

    df = pd.read_csv(DATA_PATH)

    return df


# ---------------------------------------
# Prepare Features
# ---------------------------------------

def prepare_data(df):

    X = df[FEATURES].copy()

    return X


# ---------------------------------------
# Create ML Pipeline
# ---------------------------------------

def create_model(n_clusters=N_CLUSTERS):

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=RANDOM_STATE
        ))
    ])

    return model


# ---------------------------------------
# Choose Number of Clusters
# (elbow method + silhouette score)
# ---------------------------------------

def find_best_k(X, k_min=2, k_max=10):

    scaled = StandardScaler().fit_transform(X)

    results = []

    for k in range(k_min, k_max + 1):

        kmeans = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=RANDOM_STATE
        ).fit(scaled)

        results.append({
            "k": k,
            "inertia": kmeans.inertia_,
            "silhouette": silhouette_score(
                scaled,
                kmeans.labels_
            )
        })

    return pd.DataFrame(results)


# ---------------------------------------
# Name Each Segment
# ---------------------------------------

def name_segment(income, spending):

    high_income = income >= 70
    low_income = income <= 40

    high_spend = spending >= 60
    low_spend = spending <= 40

    if high_income and high_spend:
        return "Premium Spenders"

    if high_income and low_spend:
        return "Untapped High Earners"

    if low_income and high_spend:
        return "Budget Enthusiasts"

    if low_income and low_spend:
        return "Cautious Savers"

    return "Everyday Shoppers"


SEGMENT_STRATEGY = {

    "Premium Spenders":
        "Loyal, high-value customers. Reward with VIP "
        "programs and early access to new collections.",

    "Untapped High Earners":
        "High income but low spending. The best target for "
        "promotions, personalised offers and premium launches.",

    "Budget Enthusiasts":
        "Spend a lot relative to income. Offer instalment "
        "plans, bundles and seasonal discounts.",

    "Cautious Savers":
        "Low income, low spending. Use value deals and "
        "coupons to keep them engaged at low cost.",

    "Everyday Shoppers":
        "Average income and spending. Keep them with "
        "steady loyalty points and regular offers."
}


# ---------------------------------------
# Build Segment Profiles
# ---------------------------------------

def build_profiles(df, labels):

    data = df.copy()
    data["Cluster"] = labels

    profiles = {}

    for cluster, group in data.groupby("Cluster"):

        income = group["Annual Income (k$)"].mean()
        spending = group["Spending Score (1-100)"].mean()

        name = name_segment(income, spending)

        profiles[int(cluster)] = {
            "name": name,
            "size": int(len(group)),
            "avg_age": round(group["Age"].mean(), 1),
            "avg_income": round(income, 1),
            "avg_spending": round(spending, 1),
            "strategy": SEGMENT_STRATEGY[name]
        }

    return profiles


# ---------------------------------------
# Train Model
# ---------------------------------------

def train_model():

    df = load_data()

    X = prepare_data(df)

    model = create_model()

    model.fit(X)

    labels = model.named_steps["kmeans"].labels_

    profiles = build_profiles(df, labels)

    return model, X, labels, profiles


# ---------------------------------------
# Save Model
# ---------------------------------------

def save_model(model, profiles):

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    artifact = {
        "model": model,
        "profiles": profiles
    }

    with open(MODEL_PATH, "wb") as file:

        pickle.dump(artifact, file)


# ---------------------------------------
# Main
# ---------------------------------------

if __name__ == "__main__":

    model, X, labels, profiles = train_model()

    save_model(model, profiles)

    print("Training completed successfully.")
    print(f"Customers clustered: {len(X)}")
    print(f"Segments found: {len(profiles)}")
