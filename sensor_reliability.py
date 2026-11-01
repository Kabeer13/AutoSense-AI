
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Create results folder
os.makedirs("results", exist_ok=True)

# Load dataset
df = pd.read_csv("data/EngineFaultDB_Final.csv")
df = df.drop_duplicates().reset_index(drop=True)

X = df.drop(columns=["Fault"])
y = df["Fault"]

# Recreate the exact same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load trained models
models = {
    "PCA + LDA": joblib.load("models/pca_lda.joblib"),
    "LDA Only": joblib.load("models/lda_only.joblib"),
    "Random Forest": joblib.load("models/random_forest.joblib")
}

# Training medians and standard deviations
training_medians = X_train.median()
training_std = X_train.std()

# Store results
results = []

def evaluate(model_name, model, test_data, condition, severity):
    predictions = model.predict(test_data)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    results.append({
        "Model": model_name,
        "Condition": condition,
        "Severity": severity,
        "Accuracy": accuracy,
        "Macro_F1": macro_f1
    })

    print(
        f"{model_name} | {condition} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro F1: {macro_f1:.4f}"
    )


# 1. CLEAN DATA
print("\n========== CLEAN DATA ==========")

for name, model in models.items():
    evaluate(name, model, X_test.copy(), "Clean", 0)


# 2. SENSOR NOISE EXPERIMENT
print("\n========== SENSOR NOISE ==========")

noise_levels = [0.05, 0.10, 0.20]

for level in noise_levels:

    rng = np.random.default_rng(42)

    noisy_data = X_test.copy()

    noise = rng.normal(
        loc=0,
        scale=training_std.values * level,
        size=noisy_data.shape
    )

    noisy_data = noisy_data + noise

    for name, model in models.items():
        evaluate(
            name,
            model,
            noisy_data,
            "Gaussian Noise",
            level
        )


# 3. MISSING SENSOR EXPERIMENT
print("\n========== MISSING SENSOR ==========")

for sensor in X.columns:

    missing_data = X_test.copy()

    # Simulate one unavailable sensor
    missing_data[sensor] = training_medians[sensor]

    for name, model in models.items():
        evaluate(
            name,
            model,
            missing_data,
            f"Missing: {sensor}",
            1
        )


# 4. SAVE RESULTS
results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/sensor_reliability_results.csv",
    index=False
)

print("\n===================================")
print("SENSOR RELIABILITY EXPERIMENT DONE!")
print("Results saved to:")
print("results/sensor_reliability_results.csv")
print("===================================")

# Display summary
print("\nRESULT SUMMARY:")
print(
    results_df.groupby(
        ["Model", "Condition"]
    )[["Accuracy", "Macro_F1"]].mean().round(4)
)
