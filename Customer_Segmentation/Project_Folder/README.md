# Mall Customer Segmentation

A machine learning web application that groups mall customers into segments using K-Means clustering. The project uses Python, Pandas, Scikit-learn, Seaborn, and Flask to train, evaluate, and serve a clustering model through a simple web interface.

## Project Overview

**Problem:** group 200 mall customers by age, annual income and spending score to support targeted marketing.

**Approach:** K-Means clustering on standardised features. The number of clusters was chosen with the elbow method and the silhouette score.

**Result:** 5 customer segments, including a high-income, low-spending group ("Untapped High Earners") that is a clear target for promotions.

```text
Customer Dataset
        ↓
Data Preparation
        ↓
Choose k (Elbow + Silhouette)
        ↓
K-Means Training
        ↓
Model Evaluation + Seaborn Plots
        ↓
Saved ML Model
        ↓
Flask Web Application
        ↓
Customer Segment Prediction
```

## Features

* Load and prepare customer data
* Choose the number of clusters with the elbow method and silhouette score
* Train a K-Means clustering pipeline (StandardScaler + KMeans)
* Evaluate cluster quality (silhouette, Davies-Bouldin, Calinski-Harabasz)
* Generate Seaborn plots (elbow, silhouette, customer segments)
* Name each segment and suggest a marketing strategy
* Predict a customer's segment through a Flask web application
* Simple HTML/CSS user interface
* Automated tests for the model and input validation

## Project Structure

```text
Project Folder/
│
├── app.py                        # Flask application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
├── data/
│   └── mall_customers.csv        # Dataset (200 customers)
│
├── ml/
│   ├── __init__.py
│   ├── train.py                  # Model training + choosing k
│   ├── predict.py                # Prediction + input validation
│   └── evaluate.py               # Model evaluation + plots
│
├── models/
│   └── segmentation_model.pkl    # Trained model
│
├── templates/
│   └── index.html                # Web interface
│
├── static/
│   ├── style.css                 # Web styling
│   └── plots/                    # Generated Seaborn plots
│
└── tests/
    └── test_model.py             # Model tests
```

## Technologies Used

* **Python**
* **Pandas** for data manipulation
* **NumPy** for numerical operations
* **Scikit-learn** for K-Means and metrics
* **Seaborn / Matplotlib** for visualisation
* **Flask** for the web application
* **HTML/CSS** for the frontend
* **Pytest** for testing
* **Pickle** for model persistence

## Dataset

`data/mall_customers.csv` has 200 rows with these columns: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`.

The file included here is a **synthetic dataset** generated with a fixed random seed to match the layout of the well-known Mall Customers dataset. To use the real data, replace the CSV with the same column names and re-run `python ml/train.py` and `python ml/evaluate.py`. The segment results will then reflect your real data.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SID24-GEN/project.git
cd project/Project_Folder
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python ml/train.py
```

The trained model will be saved inside:

```text
models/
```

## Evaluate the Model

```bash
python ml/evaluate.py
```

This prints the cluster metrics and saves the plots to `static/plots/`.

## Run the Application

```bash
python app.py
```

Then open your browser and visit:

```text
http://127.0.0.1:5000
```

## Testing

```bash
pytest
```

## Machine Learning Workflow

1. Load the customer dataset.
2. Select the features: Age, Annual Income, Spending Score.
3. Standardise the features so each has equal influence.
4. Try k = 2 to 10 and record the inertia (elbow method) and silhouette score.
5. Train K-Means with k = 5.
6. Evaluate the clusters and plot them with Seaborn.
7. Name each segment from its average income and spending.
8. Save the model and segment profiles.
9. Load them in the Flask application.
10. Accept customer details and return the matching segment and strategy.

## Results

Results from running `python ml/evaluate.py` on the included dataset:

| Segment | Customers | Avg age | Avg income (k$) | Avg spending |
|---|---|---|---|---|
| Untapped High Earners | 35 | 41.4 | 88.4 | 16.8 |
| Premium Spenders | 40 | 33.3 | 87.1 | 81.7 |
| Budget Enthusiasts | 36 | 27.4 | 34.4 | 67.0 |
| Everyday Shoppers | 66 | 45.3 | 54.1 | 49.6 |
| Cautious Savers | 23 | 43.0 | 24.3 | 18.6 |

* Silhouette score at k = 5: **0.480**, Davies-Bouldin: **0.789**
* The elbow in the inertia curve is at k = 5. The silhouette score is almost the same at k = 5 (0.480) and k = 6 (0.486), so k = 5 was kept because it is simpler and easier to act on.
* **Untapped High Earners** (high income, low spending) is the group to target with promotions and personalised offers.

## Future Improvements

* Use the real Mall Customers dataset.
* Add Gender and other features to the clustering.
* Compare K-Means with DBSCAN and hierarchical clustering.
* Show the customer's position on the segment plot in the web interface.
* Deploy the application online.

## Author

**Siddhesh Narewadikar**

Interested in Artificial Intelligence, Machine Learning, Python development, and automation.

## License

This project is intended for educational and portfolio purposes.
