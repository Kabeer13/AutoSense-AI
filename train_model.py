
import pandas as pd
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

# Create output folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load dataset
df = pd.read_csv("data/EngineFaultDB_Final.csv")
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=["Fault"])
y = df["Fault"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Model 1: PCA + LDA
pca_lda = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=11)),
    ("lda", LinearDiscriminantAnalysis())
])

# Model 2: Random Forest
random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

models = {
    "PCA + LDA": Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=11)),
        ("lda", LinearDiscriminantAnalysis())
    ]),

    "LDA Only": Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis())
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
}

results = {}

for name, model in models.items():

    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    print(f"\n{name} RESULTS")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))

    print("\nClassification Report:")
    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
        output_dict=True
    )

    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    results[name] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report
    }

    # Save trained model
   
    # Save trained model
    filename = {
        "PCA + LDA": "pca_lda.joblib",
        "LDA Only": "lda_only.joblib",
        "Random Forest": "random_forest.joblib"
    }[name]

    joblib.dump(model, f"models/{filename}")
    print("Model saved:", f"models/{filename}")

    

# Save evaluation results
with open("results/model_metrics.json", "w") as file:
    json.dump(results, file, indent=4)

print("\nALL MODELS TRAINED SUCCESSFULLY!")
print("Evaluation results saved in results/model_metrics.json")
