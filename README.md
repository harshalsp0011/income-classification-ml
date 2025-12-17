# Income Classification MLOps Pipeline 💰📊🚀

**An end-to-end Machine Learning project that predicts whether an individual's income exceeds $50K/yr based on census data. Features a full pipeline from raw data to SQL normalization, MLflow tracking, and Dockerized deployment.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue) ![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green) ![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red) ![Docker](https://img.shields.io/badge/Deployment-Docker-blue)

---

## 📖 Project Overview

This project goes beyond simple model training by implementing a robust **MLOps workflow**. It ingests raw census data, normalizes it into a **3NF SQLite Database**, performs advanced feature engineering, tracks experiments using **MLflow (via DagsHub)**, and serves the best model via a **FastAPI** backend and **Streamlit** frontend.

---

## 📁 Dataset Information

**Dataset**: Adult Census Income Evaluation Dataset  
**Source**: UCI Machine Learning Repository - [Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

### Download Instructions:
1. Visit the [UCI Adult Dataset page](https://archive.ics.uci.edu/ml/datasets/adult)
2. Download `adult.data` (training set, 32,561 instances)
3. Rename the file to `income_evaluation.csv` 
4. Place it in the `Income_Classification/` folder or update the path in notebook

### Required Files:
- **Original Dataset**: `income_evaluation.csv` 
  - Downloaded from UCI as `adult.data` and renamed
  - Current expected path in notebook: `/content/income_evaluation.csv`
  - **Update this path in Cell 2** before running: `file_path = 'your/path/to/income_evaluation.csv'`
  
- **Cleaned Dataset**: `cleaned_income_evaluation.csv` (automatically generated after running data cleaning cells)

### Dataset Features (15 columns):
The dataset contains these exact columns:
- `age`, `workclass`, `fnlwgt`, `education`, `education-num`, `marital-status`
- `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`
- `hours-per-week`, `native-country`, `income` (<=50K or >50K)

### Sample Data:
First row: `39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K`

---

## 🏗️ Architecture & Workflow

The project pipeline consists of five major stages:

1.  **Data Engineering (SQL)**:
    * Ingested raw `income_evaluation.csv` from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/adult).
    * Cleaned and normalized data into **3rd Normal Form (3NF)** using **SQLite**.
    * Tables created: `Personal_Details`, `Employment_Details`, `Education_Details`, `Financial_Details`, `Location_Details`.
    * Data is rejoined via SQL queries for analysis.
    * **File Path Configuration**: Update `file_path = '/content/income_evaluation.csv'` in notebook Cell 2 to your local CSV location.

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
│   ├── income_evaluation.csv        # [REQUIRED] Raw dataset (download from UCI)
│   ├── cleaned_income_evaluation.csv # Generated after data cleaning
│   ├── income_evaluation.db         # SQLite 3NF database (auto-generated)
│   ├── Dockerfile                   # Docker build for the app
│   ├── Dockerfile-fastapi           # Docker build for API
│   └── requirements.txt             # Python dependencies
├── ML_Project_Plan.docx             # Project planning document
└── README.md                        # Documentation



## 📊 Model Performance
Experiments were tracked using MLflow. Below are the results from the top-performing models on the test set:

Model	               Feature Selection	         F1-Score (CV Mean)
XGBoost	            Variance Threshold	      0.8087
XGBoost	            Correlation Threshold	   0.8072
Random Forest	      Feature Importance	      0.7802
Ridge Classifier	   None	                     0.7490
Logistic Regression	None	                     0.6309


##Champion Model: XGBClassifier with Variance Threshold feature selection was chosen for deployment due to its superior stability and F1-score.
