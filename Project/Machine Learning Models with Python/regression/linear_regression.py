import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""
Linear Regression Example
-------------------------
This script demonstrates a basic linear regression workflow using Python
and scikit-learn, including data generation, model training, prediction,
and visualization.
"""

# --------------------------------------------------
# 1. Generate synthetic dataset
# --------------------------------------------------
rng = np.random
X = rng.rand(50) * 10
y = 2 * X + rng.randn(50)

# Reshape input data to match scikit-learn requirements
X_reshaped = X.reshape(-1, 1)

# --------------------------------------------------
# 2. Train Linear Regression model
# --------------------------------------------------
model = LinearRegression()
model.fit(X_reshaped, y)

# Optional: Model evaluation and parameters
# Uncomment to inspect model performance
# print("R^2 Score:", model.score(X_reshaped, y))
# print("Coefficient:", model.coef_)
# print("Intercept:", model.intercept_)

# --------------------------------------------------
# 3. Prediction on new data
# --------------------------------------------------
X_test = np.linspace(-1, 11)
X_test_reshaped = X_test.reshape(-1, 1)
y_pred = model.predict(X_test_reshaped)

# --------------------------------------------------
# 4. Visualization
# --------------------------------------------------
plt.scatter(X, y, label="Training Data")
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Model")
plt.legend()
plt.show()
