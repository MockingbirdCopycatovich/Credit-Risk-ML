import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from data_loader import load_application_train


def preprocess(df):
    # Create a copy to avoid modifying original DataFrame
    df = df.copy()

    # Extract target variable
    y = df["TARGET"]

    # Remove target and ID column from features
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])

    # Fill missing values with constant
    X = X.fillna(-999)

    # Encode categorical features
    for col in X.select_dtypes(include=["object", "string"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return X, y


def train():
    # Load dataset
    df = load_application_train("data/raw")

    # Preprocess data
    X, y = preprocess(df)

    # Initialize Logistic Regression model
    model = LogisticRegression(max_iter=1000)

    # Create Stratified 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = []

    # Perform cross-validation
    for train_idx, val_idx in skf.split(X, y):

        # Split data into train and validation sets
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train model
        model.fit(X_train, y_train)

        # Get predicted probabilities for class 1
        preds = model.predict_proba(X_val)[:, 1]

        # Calculate ROC-AUC
        score = roc_auc_score(y_val, preds)
        scores.append(score)

    # Print mean and std of CV scores
    print(f"CV ROC-AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


if __name__ == "__main__":
    train()