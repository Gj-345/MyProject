import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

"""
Gaussian Naive Bayes on Tabular Dataset
---------------------------------------
This script demonstrates how to preprocess categorical data using
label encoding and apply Gaussian Naive Bayes for classification.
"""

# --------------------------------------------------
# 1. Data preprocessing
# --------------------------------------------------
def clean_data(dataset):
    """
    Encode categorical columns using LabelEncoder.
    """
    for column in dataset.columns:
        if dataset[column].dtype == object:
            encoder = LabelEncoder()
            dataset[column] = encoder.fit_transform(dataset[column])
    return dataset


def split_feature_target(dataset, target_col):
    """
    Split dataset into features (X) and target labels (y).
    """
    X = dataset.drop(target_col, axis=1)
    y = dataset[target_col].copy()
    return X, y


# --------------------------------------------------
# 2. Load and prepare dataset
# --------------------------------------------------
dataset = pd.read_csv("adult.csv")
dataset = clean_data(dataset)

# Train-test split
train_set, test_set = train_test_split(
    dataset, test_size=0.2, random_state=0
)

X_train, y_train = split_feature_target(train_set, "income")
X_test, y_test = split_feature_target(test_set, "income")

# --------------------------------------------------
# 3. Train Gaussian Naive Bayes model
# --------------------------------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# --------------------------------------------------
# 4. Prediction and evaluation
# --------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy:", accuracy)
