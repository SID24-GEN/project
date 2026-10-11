# House Price Predictor

A small machine learning web app that estimates the price of a house from its size, number of rooms and age. You type in four details, and the app shows a predicted price.

Built with **Python, Pandas, Scikit-learn and Flask**.

---

## How it works

```text
House data (CSV)
      ↓
Prepare features and target
      ↓
Train a Linear Regression model
      ↓
Evaluate the model (MAE, RMSE, R²)
      ↓
Save the trained model
      ↓
Flask web app loads the model
      ↓
You enter house details → input is checked → price is predicted
```

**Inputs**

| Input | Meaning | Allowed values |
|---|---|---|
| Area | House size in sqft | 100 to 10,000 |
| Bedrooms | Number of bedrooms | whole number, 1 to 10 |
| Bathrooms | Number of bathrooms | whole number, 1 to 10 |
| Age | Age of the house in years | 0 to 100 |

**Output:** the predicted price in rupees (₹).

If an input is missing or outside these limits, the app shows a clear error message instead of a prediction.

---

## Project structure

```text
project/
├── README.md                          ← you are here
└── house_price_predictor/
    └── Project_Folder/
        ├── app.py                     # Flask web app
        ├── requirements.txt           # Python packages needed
        ├── data/
        │   └── house_data.csv         # Dataset (100 houses)
        ├── ml/
        │   ├── train.py               # Prepares data and trains the model
        │   ├── evaluate.py            # Measures how accurate the model is
        │   └── predict.py             # Loads the model and predicts a price
        ├── models/
        │   └── house_model.pkl        # Saved trained model
        ├── templates/
        │   └── index.html             # Web page
        ├── static/
        │   └── style.css              # Page styling
        └── tests/
            ├── test_model.py          # Tests for the model code
            └── test_validation.py     # Input checks and their tests
```

---

## Getting started

You need **Python 3.9 or newer**.

### 1. Download the project

```bash
git clone https://github.com/SID24-GEN/project.git
cd project/house_price_predictor
```

### 2. Create a virtual environment (recommended)

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the packages

```bash
pip install -r Project_Folder/requirements.txt
```

> Run all the commands below from the `house_price_predictor` folder.

---

## Using the project

### Run the web app

```bash
python Project_Folder/app.py
```

Open your browser and go to **http://127.0.0.1:5000**, enter the house details and click the predict button.

### Train the model again (optional)

A trained model is already included. Run this only if you change the data or the training code:

```bash
python Project_Folder/ml/train.py
```

### Check how accurate the model is (optional)

macOS / Linux:

```bash
PYTHONPATH=Project_Folder python Project_Folder/ml/evaluate.py
```

Windows (Command Prompt):

```bash
set PYTHONPATH=Project_Folder
python Project_Folder/ml/evaluate.py
```

### Run the tests

```bash
pytest
```

---

## The machine learning part

1. **Load** the data from `house_data.csv`.
2. **Choose features:** Area, Bedrooms, Bathrooms and Age. The target is Price.
3. **Split** the data: 80% for training and 20% for testing.
4. **Scale** the features so they are on the same scale, then fit a **Linear Regression** model (both steps run together in one Scikit-learn pipeline).
5. **Evaluate** on the 20% the model has not seen.
6. **Save** the model to `models/house_model.pkl`.
7. The **Flask app** loads that file once when it starts and uses it for every prediction.

### Results

Measured on the 20% test set:

| Metric | Value | What it means |
|---|---|---|
| R² | 0.99 | The model explains about 99% of the price differences in the test data |
| MAE | ₹4.4 lakh (about 439,600) | Average size of the error |
| RMSE | ₹5.2 lakh (about 520,300) | Error measure that punishes big mistakes more |

---

## Limitations

* **Small dataset.** The model learns from only 100 houses, so the high R² should not be read as proof it will work well on real-world listings.
* **Narrow range.** The training houses are between about 700 and 2,900 sqft, with 1 to 6 bedrooms and ages up to 30 years. Houses far outside this range will pass the input checks but their predictions will be unreliable.
* **Few features.** Location, floor, amenities and market conditions are not used, and these matter a lot in real prices.
* **Simple model.** Linear Regression assumes a straight-line relationship between the inputs and the price.

---

## Ideas for improvement

* Train on a larger, real dataset
* Add more features such as location
* Compare with Random Forest or Gradient Boosting
* Show a price range instead of a single number
* Deploy the app online

---

## Author

**Siddhesh Narewadikar**
[LinkedIn](https://www.linkedin.com/in/siddhesh-narewadikar-782a82315/) · [GitHub](https://github.com/SID24-GEN)

---

## License

This project is for learning and portfolio purposes.
