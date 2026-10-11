import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score
)

from ml.train import (
    BASE_DIR,
    N_CLUSTERS,
    find_best_k,
    train_model
)


PLOT_DIR = os.path.join(BASE_DIR, "static", "plots")


# ---------------------------------------
# Evaluate Model
# ---------------------------------------

def evaluate_model(model, X, labels):

    scaled = model.named_steps["scaler"].transform(X)

    kmeans = model.named_steps["kmeans"]

    return {
        "Clusters": kmeans.n_clusters,
        "Inertia": kmeans.inertia_,
        "Silhouette": silhouette_score(scaled, labels),
        "Davies-Bouldin": davies_bouldin_score(scaled, labels),
        "Calinski-Harabasz": calinski_harabasz_score(scaled, labels)
    }


# ---------------------------------------
# Plots (Seaborn)
# ---------------------------------------

def save_plots(X, labels, profiles, k_results):

    os.makedirs(PLOT_DIR, exist_ok=True)

    sns.set_theme(style="whitegrid")

    palette = sns.color_palette("deep", N_CLUSTERS)


    # Elbow method

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.lineplot(
        data=k_results, x="k", y="inertia",
        marker="o", ax=ax
    )

    ax.axvline(N_CLUSTERS, color="red", linestyle="--")
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "elbow.png"), dpi=130)
    plt.close(fig)


    # Silhouette score

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.lineplot(
        data=k_results, x="k", y="silhouette",
        marker="o", ax=ax
    )

    ax.axvline(N_CLUSTERS, color="red", linestyle="--")
    ax.set_title("Silhouette Score by k")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "silhouette.png"), dpi=130)
    plt.close(fig)


    # Segments: income vs spending

    plot_df = X.copy()
    plot_df["Segment"] = [profiles[int(l)]["name"] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.scatterplot(
        data=plot_df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Segment",
        palette=palette,
        s=55,
        ax=ax
    )

    ax.set_title("Customer Segments")
    ax.legend(title="Segment", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "segments.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------
# Main
# ---------------------------------------

if __name__ == "__main__":

    model, X, labels, profiles = train_model()

    k_results = find_best_k(X)

    results = evaluate_model(model, X, labels)

    save_plots(X, labels, profiles, k_results)

    print("\nChoosing k")
    print("----------")
    print(k_results.round(3).to_string(index=False))

    print("\nModel Evaluation")
    print("----------------")

    for metric, value in results.items():

        print(f"{metric}: {value:.4f}")

    print("\nSegments")
    print("--------")

    for cluster, p in sorted(profiles.items()):

        print(
            f"{p['name']:<22} n={p['size']:<3} "
            f"age={p['avg_age']:<5} "
            f"income={p['avg_income']:<6} "
            f"spending={p['avg_spending']}"
        )

    print(f"\nPlots saved to: {PLOT_DIR}")
