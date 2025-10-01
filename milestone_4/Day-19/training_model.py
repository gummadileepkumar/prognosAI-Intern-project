# train_model.py
# Train a RandomForest on the Iris dataset and save the model + dataset.

import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Load Iris data as a DataFrame
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    # Split into train and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build pipeline: scale features -> RandomForest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Evaluate quick test accuracy for verification
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"Test accuracy: {test_acc:.3f}")

    # Save the trained pipeline and metadata
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": X.columns.tolist(),
            "target_names": iris.target_names.tolist()
        },
        "iris_pipeline.joblib"
    )

    # Save full dataset for the Streamlit app
    df_full = X.copy()
    df_full["target"] = y
    df_full.to_csv("iris_dataset.csv", index=False)
    print("Saved iris_pipeline.joblib and iris_dataset.csv")

if __name__ == "__main__":
    main()
