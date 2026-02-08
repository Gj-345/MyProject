import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

"""
Linear Regression on Weather Dataset
------------------------------------
This script demonstrates linear regression using a real-world weather
dataset. It includes data loading, train-test split, model training,
prediction, and regression performance evaluation.
"""

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
dataset = pd.read_csv(
    "https://raw.githubusercontent.com/kongruksamza/MachineLearning/"
    "refs/heads/master/Linear%20Regression/Weather.csv"
)

# Feature and target
X = dataset["MinTemp"].values.reshape(-1, 1)
y = dataset["MaxTemp"].values.reshape(-1, 1)

# --------------------------------------------------
# 2. Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# --------------------------------------------------
# 3. Train Linear Regression model
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# 4. Prediction
# --------------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------------
# 5. Compare actual vs predicted values
# --------------------------------------------------
results = pd.DataFrame({
    "Actual": y_test.flatten(),
    "Predicted": y_pred.flatten()
})

# --------------------------------------------------
# 6. Model evaluation
# --------------------------------------------------
print("MAE :", metrics.mean_absolute_error(y_test, y_pred))
print("MSE :", metrics.mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R²  :", metrics.r2_score(y_test, y_pred))
