# K-Nearest Neighbors Classification Using Snowflake Data Warehouse

## Overview
This project implements a K-Nearest Neighbors (KNN) classification model using data fetched directly from a Snowflake data warehouse.  
The objective is to perform multi-class classification by preprocessing numerical features, tuning the number of neighbors, and evaluating the model using multiple performance metrics.

This project demonstrates database connectivity, data preprocessing, model training, hyperparameter testing, and model persistence.

---

## Technologies Used
- Python
- Pandas
- Snowflake Connector for Python
- Scikit-learn
- K-Nearest Neighbors (KNN)

---

## Dataset
- Source: Snowflake Data Warehouse
- Database: `CLASSIFICATION`
- Schema: `PUBLIC`
- Table: `KNN`
- Target column: **C8**
- Columns **C9** and **C10** are dropped during preprocessing
- Dataset contains numerical features suitable for distance-based algorithms

---

## Project Workflow

1. Connect to Snowflake and retrieve data
2. Perform basic data exploration
3. Remove unnecessary columns
4. Apply feature scaling using StandardScaler
5. Split dataset into training and testing sets
6. Train KNN models with different values of `k` (3 to 10)
7. Evaluate each model using classification metrics
8. Save the trained model and scaler

---

## Data Preprocessing
- Numerical features are standardized to improve distance calculations
- Target column (**C8**) is excluded from scaling
- Stratified train-test split is used to maintain class distribution

---

## Model Used
**K-Nearest Neighbors (KNN)**

KNN is a distance-based algorithm that classifies data points based on the majority class of their nearest neighbors. Feature scaling is essential for accurate distance computation.

---

## Evaluation Metrics
The model is evaluated using:
- Accuracy Score
- Precision (Macro)
- Recall (Macro)
- ROC AUC Score (One-vs-Rest)
- Confusion Matrix
- Classification Report

These metrics help evaluate performance across all classes.

---

## Model Persistence
- Trained KNN model is saved as `model.joblib`
- Scaler object is saved as `scaling.joblib`
- Saved models can be reused for inference without retraining

---

## How to Run the Project

1. Install required libraries:
   ```bash
   pip install pandas scikit-learn snowflake-connector-python joblib
