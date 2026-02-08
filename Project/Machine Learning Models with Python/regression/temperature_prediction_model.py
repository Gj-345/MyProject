import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

"""
Linear Regression on Weather Dataset
------------------------------------
This script applies linear regression to a real-world weather dataset
to predict maximum temperature based on minimum temperature.
"""

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
dataset = pd.read_csv(
    "https://raw.githubusercontent.com/kongruksamza/MachineLearning/"
    "refs/heads/master/Linear%20Regression/Weather.csv"
)

# Feature (X) and target (y)
X = dataset["MinTemp"].values.reshape(-1, 1)
y = dataset["MaxTemp"].values.reshape(-1, 1)

# --------------------------------------------------
# 2. Train-test split (80% train, 20% test)
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
# 6. Visualization
# --------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("MinTemp")
plt.ylabel("MaxTemp")
plt.title("Linear Regression: MinTemp vs MaxTemp")
plt.legend()
plt.show()
