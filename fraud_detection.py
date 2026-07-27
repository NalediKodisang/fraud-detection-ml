"""
=========================================================
 AI Fraud Detection System Using Random Forest
 Author : Naledi Kodisang
=========================================================

Description:
This project detects fraudulent financial transactions
using Machine Learning (Random Forest Classifier).

Dataset:
transactions.csv

Features:
- Amount
- Location
- Time

Target:
- Fraud
    0 = Legitimate
    1 = Fraudulent
=========================================================
"""

# =====================================================
# Import Libraries
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =====================================================
# Load Dataset
# =====================================================

def load_data():

    try:

        data = pd.read_csv("data/transactions.csv")

        print("=" * 60)
        print("Dataset loaded successfully.")
        print("=" * 60)

        return data

    except FileNotFoundError:

        print("ERROR: Dataset not found.")
        print("Place transactions.csv inside the data folder.")
        exit()


# =====================================================
# Explore Dataset
# =====================================================

def explore_data(data):

    print("\nFIRST 5 ROWS")
    print("-" * 60)
    print(data.head())

    print("\nDATASET INFORMATION")
    print("-" * 60)
    data.info()

    print("\nSTATISTICAL SUMMARY")
    print("-" * 60)
    print(data.describe())

    print("\nMISSING VALUES")
    print("-" * 60)
    print(data.isnull().sum())

    print("\nFRAUD DISTRIBUTION")
    print("-" * 60)
    print(data["Fraud"].value_counts())


# =====================================================
# Visualize Fraud Distribution
# =====================================================

def plot_fraud_distribution(data):

    plt.figure(figsize=(6,4))

    data["Fraud"].value_counts().plot(
        kind="bar"
    )

    plt.title("Fraud Distribution")

    plt.xlabel("Fraud")

    plt.ylabel("Number of Transactions")

    plt.xticks(
        [0,1],
        ["Legitimate","Fraud"]
    )

    plt.tight_layout()

    plt.savefig("images/fraud_distribution.png")

    plt.show()


# =====================================================
# Clean Dataset
# =====================================================

def clean_data(data):

    data = data.drop_duplicates()

    data = data.fillna(0)

    return data


# =====================================================
# Prepare Features
# =====================================================

def prepare_data(data):

    X = data[
        [
            "Amount",
            "Location",
            "Time"
        ]
    ]

    y = data["Fraud"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )

    return X_train, X_test, y_train, y_test

# =====================================================
# Train Random Forest Model
# =====================================================

def train_model(X_train, y_train):

    print("\nTraining Random Forest Model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Model trained successfully!")

    return model


# =====================================================
# Evaluate Model
# =====================================================

def evaluate_model(model, X_test, y_test, feature_names):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.2%}")

    print("\nClassification Report")
    print("-" * 60)

    print(classification_report(y_test, predictions))

    # ------------------------------------------
    # Confusion Matrix
    # ------------------------------------------

    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Fraud"]
    )

    display.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig("images/confusion_matrix.png")

    plt.show()

    # ------------------------------------------
    # Feature Importance
    # ------------------------------------------

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance")
    print("-" * 60)

    print(importance)

    plt.figure(figsize=(8,5))

    plt.bar(
        importance["Feature"],
        importance["Importance"]
    )

    plt.title("Feature Importance")

    plt.xlabel("Features")

    plt.ylabel("Importance Score")

    plt.tight_layout()

    plt.savefig("images/feature_importance.png")

    plt.show()


# =====================================================
# Predict New Transaction
# =====================================================

def predict_transaction(model):

    print("\n" + "=" * 60)
    print("NEW TRANSACTION PREDICTION")
    print("=" * 60)

    amount = float(input("Enter Transaction Amount: R"))

    location = int(input("Enter Location (1-5): "))

    time = int(input("Enter Time (0-23): "))

    new_transaction = pd.DataFrame(
        [[amount, location, time]],
        columns=[
            "Amount",
            "Location",
            "Time"
        ]
    )

    prediction = model.predict(new_transaction)

    probability = model.predict_proba(new_transaction)

    print("\nPrediction Results")
    print("-" * 60)

    if prediction[0] == 1:

        print("⚠ Fraudulent Transaction Detected")

    else:

        print("✓ Legitimate Transaction")

    print(f"\nLegitimate Probability : {probability[0][0]:.2%}")

    print(f"Fraud Probability      : {probability[0][1]:.2%}")
# =====================================================
# Main Program
# =====================================================

def main():

    print("\n" + "=" * 60)
    print("AI FRAUD DETECTION SYSTEM")
    print("=" * 60)

    data = load_data()

    explore_data(data)

    plot_fraud_distribution(data)

    data = clean_data(data)

    X_train, X_test, y_train, y_test = prepare_data(data)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test, feature_names=["Amount", "Location", "Time"])

    while True:

        choice = input("\nPredict a new transaction? (y/n): ").strip().lower()

        if choice == "y":
            predict_transaction(model)
        else:
            break

    print("\nProgram finished. Check the images/ folder for saved charts.")


if __name__ == "__main__":
    main()