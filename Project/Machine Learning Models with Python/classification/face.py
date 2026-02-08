"""
Face Recognition with PCA + SVM (GridSearchCV)
- Dataset: LFW people (sklearn.datasets.fetch_lfw_people)
- Pipeline: PCA -> SVC
- Hyperparameter tuning: GridSearchCV
- Evaluation: accuracy + confusion matrix heatmap
"""

import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix


# -----------------------------
# 1) Load dataset
# -----------------------------
faces = fetch_lfw_people(min_faces_per_person=60)

X = faces.data
y = faces.target
target_names = faces.target_names

print("Classes:", len(target_names))
print("X shape:", X.shape)
print("y shape:", y.shape)


# -----------------------------
# 2) Preview sample images
# -----------------------------
fig, axes = plt.subplots(3, 5, figsize=(8, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.images[i], cmap="bone")
    ax.set(xticks=[], yticks=[])
    ax.set_ylabel(target_names[y[i]].split()[-1], color="black")
plt.tight_layout()
plt.show()


# -----------------------------
# 3) Build model: PCA + SVC
# -----------------------------
pca = PCA(n_components=150, svd_solver="randomized", whiten=True)
svc = SVC(kernel="rbf", class_weight="balanced")
model = make_pipeline(pca, svc)


# -----------------------------
# 4) Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# -----------------------------
# 5) Hyperparameter tuning (GridSearchCV)
# -----------------------------
param_grid = {
    "svc__C": [1, 5, 10, 50],
    "svc__gamma": [0.001, 0.005, 0.01, 0.05],
}

grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

print("\nBest parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)

best_model = grid.best_estimator_


# -----------------------------
# 6) Predict & evaluate
# -----------------------------
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print(f"\nTest Accuracy: {acc:.2f}%")


# -----------------------------
# 7) Show predictions on sample test images
# -----------------------------
fig, axes = plt.subplots(4, 6, figsize=(10, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(62, 47), cmap="bone")
    ax.set(xticks=[], yticks=[])

    true_name = target_names[y_test[i]].split()[-1]
    pred_name = target_names[y_pred[i]].split()[-1]

    color = "green" if y_pred[i] == y_test[i] else "red"
    ax.set_ylabel(pred_name, color=color)
    ax.set_title(true_name, fontsize=9)

plt.tight_layout()
plt.show()


# -----------------------------
# 8) Confusion matrix (heatmap)
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

sb.heatmap(
    cm,
    annot=True,
    fmt="d",
    square=True,
    cbar=False,
    xticklabels=target_names,
    yticklabels=target_names,
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

