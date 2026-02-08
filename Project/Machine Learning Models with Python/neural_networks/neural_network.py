import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

"""
Neural Network Classification (MLP) on MNIST
--------------------------------------------
This script demonstrates handwritten digit classification
using a Multi-Layer Perceptron (MLP) on the MNIST dataset.
It includes data loading, preprocessing, visualization,
model training, prediction, and evaluation.
"""

# --------------------------------------------------
# 1. Load and prepare dataset
# --------------------------------------------------
mnist_raw = loadmat("mnist-original.mat")

X = mnist_raw["data"].T          # shape: (70000, 784)
y = mnist_raw["label"][0]        # shape: (70000,)

# Shuffle dataset
shuffle_idx = np.random.permutation(len(X))
X, y = X[shuffle_idx], y[shuffle_idx]

# Train-test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# --------------------------------------------------
# 2. Visualize training samples
# --------------------------------------------------
fig, axes = plt.subplots(10, 10, figsize=(8, 8))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].reshape(28, 28),
              cmap="binary",
              interpolation="nearest")
    ax.text(0.05, 0.85, str(int(y_train[i])),
            transform=ax.transAxes,
            color="black")
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

# --------------------------------------------------
# 3. Train Neural Network (MLP)
# --------------------------------------------------
model = MLPClassifier()
model.fit(X_train, y_train)

# --------------------------------------------------
# 4. Prediction
# --------------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------------
# 5. Visualize prediction results
# --------------------------------------------------
fig, axes = plt.subplots(10, 10, figsize=(8, 8))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28),
              cmap="binary",
              interpolation="nearest")

    ax.text(0.05, 0.85, str(int(y_test[i])),
            transform=ax.transAxes)

    ax.text(0.55, 0.85, str(int(y_pred[i])),
            transform=ax.transAxes,
            color="green" if y_test[i] == y_pred[i] else "red")

    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

# --------------------------------------------------
# 6. Model evaluation
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy)
