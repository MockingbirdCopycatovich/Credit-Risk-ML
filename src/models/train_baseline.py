from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.data_loader import load_application_train
from src.preprocessing import clean_data

def train():
    """
    Train baseline Logistic Regression model
    using 5-fold cross-validation and evaluate ROC-AUC.
    """
    df = load_application_train("data/raw")
    df = clean_data(df)

    # Load and clean the dataset
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "string"]).columns


    # Preprocessing pipelines
    # Numerical pipeline:
    # - Fill missing values with median
    # - Scale features for better convergence
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline:
    # - Fill missing values with "Unknown"
    # - Apply One-Hot Encoding
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine numerical and categorical preprocessing
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    # Define baseline model
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))
    ])

    # Model evaluation
        # Perform 5-fold cross-validation
        # Evaluate using ROC-AUC metric
    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    print("ROC-AUC per fold:", scores)
    print("Mean ROC-AUC:", scores.mean())

if __name__ == "__main__":
    train()
