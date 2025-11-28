# Income Classification MLOps Pipeline 💰📊🚀

**An end-to-end Machine Learning project that predicts whether an individual's income exceeds $50K/yr based on census data. Features a full pipeline from raw data to SQL normalization, MLflow tracking, and Dockerized deployment.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![Docker](https://img.shields.io/badge/Deployment-Docker-blue)

---

## 📖 Project Overview

This project goes beyond simple model training by implementing a robust **MLOps workflow**. It ingests raw census data, normalizes it into a **3NF SQLite Database**, performs advanced feature engineering, tracks experiments using **MLflow (via DagsHub)**, and serves the best model via a **FastAPI** backend and **Streamlit** frontend.

---

## 🏗️ Architecture & Workflow

The project pipeline consists of five major stages:

1.  **Data Engineering (SQL)**:
    * Ingested raw `income_evaluation.csv`.
    * Cleaned and normalized data into **3rd Normal Form (3NF)** using **SQLite**.
    * Tables created: `Personal_Details`, `Employment_Details`, `Education_Details`, `Financial_Details`, `Location_Details`.
    * Data is rejoined via SQL queries for analysis.

2.  **Exploratory Data Analysis (EDA)**:
    * Automated reporting using **ydata-profiling**.
    * Analysis of class imbalance, correlations, and missing values.

3.  **Model Training & Experimentation**:
    * **Tracking**: All runs logged to **MLflow** (hosted on DagsHub).
    * **Models Compared**: Logistic Regression, Ridge Classifier, Random Forest, XGBoost.
    * **Feature Engineering**: Created interaction terms (e.g., `Age_to_hours_ratio`, `Age_squared`).
    * **Feature Selection**: Applied Variance Threshold, Correlation Threshold, and Feature Importance.

4.  **Deployment**:
    * **Backend**: FastAPI service to serve predictions.
    * **Frontend**: Streamlit app for user interaction.
    * **Containerization**: Dockerized services for consistent deployment.

---

## 📂 Repository Structure

```text
income-classification-ml/
├── Income_Classification/
│   ├── app/
│   │   ├── fastapi/
│   │   │   ├── main.py              # API endpoint for inference
│   │   │   └── random_forest_model.joblib # Serialized model
│   │   └── streamlit/
│   │       └── income.py            # User Interface
│   ├── Classification.ipynb         # Main notebook: Data -> SQL -> MLflow
│   ├── Dockerfile                   # Docker build for the app
│   ├── Dockerfile-fastapi           # Docker build for API
│   └── requirements.txt             # Python dependencies
├── ML_Project_Plan.docx             # Project planning document
└── README.md                        # Documentation
