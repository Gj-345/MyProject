from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.io import loadmat
import matplotlib.pyplot as plt

"""
PCA on MNIST (Image Reconstruction)
----------------------------------
This script demonstrates how Principal Component Analysis (PCA)
can be used to reduce dimensionality of image data and reconstruct
handwritten digit images with fewer features.
"""

# --------------------------------------------------
# 1. Load MNIST dataset
# --------------------------------------------------
mnist_raw = loadmat("mnist-original.mat")

X = mnist_raw["data"].T          # shape: (70000, 784)
y = mnist_raw["label"][0]        # shape: (70000,)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0
)

print("Before PCA:", X_train.shape)

# --------------------------------------------------
# 2. Apply PCA
# --------------------------------------------------
# Keep 95% of variance
pca = PCA(0.95)
X_train_reduced = pca.fit_transform(X_train)

# Reconstruct image from reduced representation
X_reconstructed = pca.inverse_transform(X_train_reduced)

print("After PCA:", X_train_reduced.shape)

# --------------------------------------------------
# 3. Visualization: Original vs PCA reconstructed image
# --------------------------------------------------
plt.figure(figsize=(8, 4))

# Original image (784 features)
plt.subplot(1, 2, 1)
plt.imshow(X_train[0].reshape(28, 28),
           cmap="gray",
           interpolation="nearest")
plt.xlabel("784 Features")
plt.title("Original Image")

# PCA reconstructed image
plt.subplot(1, 2, 2)
plt.imshow(X_reconstructed[0].reshape(28, 28),
           cmap="gray",
           interpolation="nearest")
plt.xlabel(f"{X_train_reduced.shape[1]} Features")
plt.title("PCA Reconstructed Image")

plt.tight_layout()
plt.show()
