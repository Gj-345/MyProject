from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

"""
Gaussian Naive Bayes Classification
----------------------------------
This script demonstrates Gaussian Naive Bayes classification
using the Iris dataset, including data loading, train-test split,
model training, prediction, and evaluation.
"""

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
iris = load_iris()
X = iris["data"]
y = iris["target"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# --------------------------------------------------
# 2. Train Gaussian Naive Bayes model
# --------------------------------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# --------------------------------------------------
# 3. Prediction
# --------------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------------
# 4. Model evaluation
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy)
