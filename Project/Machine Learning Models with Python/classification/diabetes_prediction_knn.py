import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

"""
K-Nearest Neighbors (KNN) for Diabetes Prediction
-------------------------------------------------
This script applies KNN classification to predict diabetes outcomes
using a real-world dataset. Model performance is evaluated using
classification report and confusion matrix.
"""

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
df = pd.read_csv("diabetes.csv")

# Features (X) and target (y)
X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

# --------------------------------------------------
# 2. Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0
)

# --------------------------------------------------
# 3. (Optional) Compare different K values
# --------------------------------------------------
"""
k_neighbors = np.arange(1, 9)
train_score = np.empty(len(k_neighbors))
test_score = np.empty(len(k_neighbors))

for i, k in enumerate(k_neighbors):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    train_score[i] = model.score(X_train, y_train)
    test_score[i] = model.score(X_test, y_test)

plt.title("Compare K values")
plt.plot(k_neighbors, test_score, label="Test Score")
plt.plot(k_neighbors, train_score, label="Train Score")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Score")
plt.legend()
plt.show()
"""

# --------------------------------------------------
# 4. Train final KNN model
# --------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)

# --------------------------------------------------
# 5. Prediction and evaluation
# --------------------------------------------------
y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))

# Confusion matrix (cross-tabulation)
conf_matrix = pd.crosstab(
    y_test,
    y_pred,
    rownames=["Actual"],
    colnames=["Predicted"],
    margins=True
)

print(conf_matrix)
