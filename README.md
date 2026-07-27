 AI Fraud Detection System

A machine learning project that detects fraudulent financial transactions using a Random Forest Classifier.

## Overview
This project trains a supervised model on transaction data (amount, location, time) to classify transactions as legitimate or fraudulent, and includes visualizations of the results.

## Features
- Data cleaning and exploration
- Fraud distribution visualization
- Random Forest model training
- Accuracy, precision, recall, F1-score reporting
- Confusion matrix and feature importance charts
- Interactive prediction on new transactions

## Project Structure

fraud-detection/
├── data/
│ └── transactions.csv
├── images/
│ ├── fraud_distribution.png
│ ├── confusion_matrix.png
│ └── feature_importance.png
├── fraud_detection.py
├── requirements.txt
└── README.md


## Setup
```bash
pip install -r requirements.txt
python fraud_detection.py
```

## Dataset
Place a `transactions.csv` file inside the `data/` folder with the columns: `Amount`, `Location`, `Time`, `Fraud` (0 = legitimate, 1 = fraudulent).

## Author
Naledi Kodisang
