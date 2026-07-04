# TCS Stock Price Prediction Using Machine Learning

## Project Overview

This project predicts the closing price of TCS (Tata Consultancy Services) stock using Machine Learning.

The project uses historical stock market data to perform feature engineering, train machine learning models, evaluate their performance, and predict future stock prices.

Two regression models are implemented and compared:

- Linear Regression
- Random Forest Regressor

The trained Random Forest model is saved for future predictions using the Pickle library.

---

## Objectives

- Load and preprocess historical TCS stock data.
- Perform feature engineering using technical indicators.
- Train Machine Learning regression models.
- Compare model performance.
- Predict stock closing prices.
- Save the trained model for future use.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Pickle

---

## Dataset

Dataset Used:

- `TCS.csv`

The dataset contains historical stock information including:

- Date
- Open Price
- High Price
- Low Price
- Close Price
- Volume

---

## Data Preprocessing

The following preprocessing steps were performed:

- Loaded the dataset using Pandas.
- Displayed dataset columns.
- Selected the required columns.
- Converted the Date column into DateTime format.
- Removed rows containing missing values.

---

## Feature Engineering

The following new features were created to improve prediction accuracy:

### 1. Moving Average (MA_10)

Average closing price of the previous 10 trading days.

### 2. Moving Average (MA_20)

Average closing price of the previous 20 trading days.

### 3. Daily Return

Percentage change in closing price compared to the previous trading day.

### 4. Volatility

Measures the fluctuation of stock prices using the rolling standard deviation of daily returns.

---

## Feature Scaling

The input features were standardized using **StandardScaler** before training the models.

---

## Machine Learning Models

Two regression models were trained and compared:

### Linear Regression

A simple regression model that predicts stock prices using a linear relationship between the input features and the target variable.

### Random Forest Regressor

An ensemble learning algorithm that builds multiple decision trees and averages their predictions to improve accuracy.

---

## Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- R² Score

The evaluation metrics help compare the prediction performance of both models.

---

## Model Saving

The trained Random Forest model is saved as:

```
model.pkl
```

This allows the trained model to be reused without retraining.

---

## Prediction

The project also demonstrates how to predict the closing price of TCS stock using sample input values.

---

## Visualization

### Actual vs Predicted Stock Price

The project generates a line graph comparing:

- Actual Closing Price
- Predicted Closing Price

This visualization helps evaluate how closely the model predictions match the actual stock prices.

---

## Project Structure

```
TCS-Stock-Price-Prediction/
│
├── README.md
├── requirements.txt
├── TCS.csv
├── stock_prediction.py
├── screenshots/
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AKHIL-MARADANA/TCS-Stock-Price-Prediction.git
```

### 2. Move into the project folder

```bash
cd TCS-Stock-Price-Prediction
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python stock_prediction.py
```

---

## Requirements

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

The project uses:

- pandas
- numpy
- matplotlib
- scikit-learn

---

## Author

**Akhil Maradana**

GitHub: https://github.com/AKHIL-MARADANA