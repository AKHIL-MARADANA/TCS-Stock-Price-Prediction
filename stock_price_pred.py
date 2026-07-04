# =========================================================
# STOCK PRICE PREDICTION USING MACHINE LEARNING
# DATASET : TCS.csv
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split #used to divide the data into training and test 
from sklearn.preprocessing import StandardScaler #
from sklearn.linear_model import LinearRegression #
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import pickle

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("TCS.csv")

# Display first 5 rows

print(df.head())

# =========================================================
# DISPLAY COLUMN NAMES
# =========================================================

print("\nColumns in Dataset:\n")
print(df.columns)

# =========================================================
# SELECT IMPORTANT COLUMNS
# =========================================================

df = df[['Date',
         'Open',
         'High',
         'Low',
         'Close',
         'Volume']]

# =========================================================
# CONVERT DATE COLUMN
#actually pd reads the csv file as text.. dtype is string
# =========================================================

df['Date'] = pd.to_datetime(df['Date'])
#now pandas converted those strings into dates.

# =========================================================
# FEATURE ENGINEERING
# =========================================================

#MOVING AVG_N : It is the avg of previous N closing prices

# 10-Day Moving Average

df['MA_10'] = df['Close'].rolling(window=10).mean()

# 20-Day Moving Average

df['MA_20'] = df['Close'].rolling(window=20).mean()

# Daily Return : percent change from previous day i.e, (todays close-yesterdays close)/yesterdays close

df['Daily_Return'] = df['Close'].pct_change()  #How much did the price change relative to yesterday?

# Volatility  : Measures how much stock price fluctuates.

df['Volatility'] = df['Daily_Return'].rolling(window=10).std()

#Small standard deviation → returns don't change much → stock is stable.
#Large standard deviation → returns fluctuate a lot → stock is volatile.

# =========================================================
# REMOVE NULL VALUES
# =========================================================

df.dropna(inplace=True)  #For first 9 rows: MA_10 not available , For first 19 rows : MA_20 not available

# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df[['Open',
        'High',
        'Low',
        'Volume',
        'MA_10',
        'MA_20',
        'Daily_Return',
        'Volatility']]

y = df['Close']

# =========================================================
# FEATURE SCALING z=(x-mue)/sigma
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X) 

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# MODEL 1 : LINEAR REGRESSION
# =========================================================

lr_model = LinearRegression() #creates an empty lr model: y=b0+b1x1+b2x2+...+b8x8

lr_model.fit(X_train, y_train) #It chooses coefficients that minimize prediction error on the training data.
                               #the error means the difference between the actual Close price and the predicted Close price.
                               
lr_predictions = lr_model.predict(X_test) #For every row in X_test, the model substitutes the feature values into the learned equation.

# =========================================================
# MODEL 2 : RANDOM FOREST REGRESSOR
# =========================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)#creates 100 decision trees.

rf_model.fit(X_train, y_train) 

rf_predictions = rf_model.predict(X_test) 
#The tree tests many candidate conditions such as Open < 3550 or Volume < 2200000 and selects the one that reduces the prediction error the most."
#error is the sum of sqaures of(output-avg output)
#The sample goes through all 100 trees.
#Random Forest then averages all predictions.

# =========================================================
# MODEL EVALUATION
# =========================================================

print("\n===================================")
print("LINEAR REGRESSION RESULTS")
print("===================================")

print("Mean Absolute Error :",
      mean_absolute_error(y_test, lr_predictions))  #∑|actual-predicted|/n

print("R2 Score :",
      r2_score(y_test, lr_predictions)) #R² measures how much of the variation in stock prices is explained by the model.
                                        #R² = 1-∑(actual-predicted)^2/∑(actual-mean)^2
print("\n===================================")
print("RANDOM FOREST RESULTS")
print("===================================")

print("Mean Absolute Error :",
      mean_absolute_error(y_test, rf_predictions))

print("R2 Score :",
      r2_score(y_test, rf_predictions))

# =========================================================
# SAVE MODEL
# =========================================================

pickle.dump(rf_model, open('model.pkl', 'wb'))

print("\nModel Saved Successfully")

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(12,6))

plt.plot(
    y_test.values[:100],
    label='Actual Price'
)

plt.plot(
    rf_predictions[:100],
    label='Predicted Price'
)

plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('TCS Stock Price Prediction')

plt.legend()

plt.show()

# =========================================================
# EXAMPLE PREDICTION
# =========================================================

sample_input = np.array([[
    3500,      # Open Price
    3550,      # High Price
    3480,      # Low Price
    2500000,   # Volume
    3495,      # MA_10
    3470,      # MA_20
    0.01,      # Daily Return
    0.02       # Volatility
]])

# Scale input data

sample_input_scaled = scaler.transform(sample_input)

# Predict stock price

prediction = rf_model.predict(sample_input_scaled)

print("\nPredicted Closing Price :")
print(prediction[0])