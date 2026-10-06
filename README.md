# House Price Predictor

A machine learning web application that predicts house prices based on property-related features. The project uses Python, Pandas, NumPy, Scikit-learn, and Flask to train, evaluate, and serve a regression model through a simple web interface.

## Project Overview

The House Price Predictor follows a basic machine learning workflow:

```text
House Price Dataset
        ↓
Data Preparation
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Saved ML Model
        ↓
Flask Web Application
        ↓
House Price Prediction
```

## Features

* Load and process house price data
* Validate input data
* Train a machine learning regression model
* Evaluate model performance
* Save the trained model for later predictions
* Predict house prices through a Flask web application
* Simple HTML/CSS user interface
* Automated tests for model and input validation

## Project Structure

```text
Project Folder/
│
├── app.py                    # Flask application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── data/
│   └── house_data.csv        # Dataset
│
├── ml/
│   ├── __init__.py
│   ├── train.py              # Model training
│   ├── predict.py            # Prediction logic
│   └── evaluate.py           # Model evaluation
│
├── models/
│   └── house_model.pkl       # Trained model
│
│
├── templates/
│   └── index.html            # Web interface
│
├── static/
│   └── style.css             # Web styling
│
└── tests/
    └── test_model.py     # Model tests
```

## Technologies Used

* **Python**
* **Pandas** for data manipulation
* **NumPy** for numerical operations
* **Scikit-learn** for machine learning
* **Flask** for the web application
* **HTML/CSS** for the frontend
* **Pytest** for testing
* **Joblib/Pickle** for model persistence

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/SID24-GEN/project.git
cd house_price_predictor
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

To train the machine learning model:

```bash
python ml/train.py
```

The trained model will be saved inside:

```text
models/
```

## Run the Application

Start the Flask application:

```bash
python app.py
```

Then open your browser and visit:

```text
http://127.0.0.1:5000
```

## Testing

Run the automated tests using:

```bash
pytest
```

## Machine Learning Workflow

The project follows these main steps:

1. Load the house price dataset.
2. Validate and prepare the data.
3. Separate input features and target values.
4. Split the dataset into training and testing sets.
5. Train the regression model.
6. Evaluate the model using appropriate regression metrics.
7. Save the trained model.
8. Load the model in the Flask application.
9. Accept user input through the web interface.
10. Generate the predicted house price.

## Future Improvements

* Add more real-world house features.
* Improve model performance through feature engineering.
* Compare multiple regression algorithms.
* Add data visualization and exploratory data analysis.
* Add model performance metrics to the web interface.
* Deploy the application online.
* Add better input validation and error handling.

## Author

**Siddhesh Narewadikar**

Interested in Artificial Intelligence, Machine Learning, Python development, and automation.

## License

This project is intended for educational and portfolio purposes.
