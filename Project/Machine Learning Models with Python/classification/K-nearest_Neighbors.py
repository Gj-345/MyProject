from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

"""
K-Nearest Neighbors (KNN) Classification
---------------------------------------
This script demonstrates a basic KNN classification workflow
using the Iris dataset, including data splitting, model training,
prediction, and evaluation.
"""

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
iris = load_iris()
X = iris["data"]
y = iris["target"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0
)

# --------------------------------------------------
# 2. Create and train KNN model
# --------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# --------------------------------------------------
# 3. Prediction
# --------------------------------------------------
y_pred = knn.predict(X_test)

# --------------------------------------------------
# 4. Model evaluation
# --------------------------------------------------
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris["target_names"]
    )
)

print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
