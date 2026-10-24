# Project Report: Cardiovascular Disease Predictive Analytics & AI Deployment

## 1. Executive Summary
This project implements an end-to-end clinical machine learning and deep learning pipeline designed to predict cardiovascular disease risk from patient health vitals. The solution encompasses robust data preprocessing, comparative modeling across classical and deep neural architectures, an automated REST inference API built with FastAPI, and an interactive diagnostic dashboard built with Streamlit.

## 2. Dataset Overview
- **Source:** Kaggle Cardiovascular Disease Dataset (70,000 patient records).
- **Target Variable:** `cardio` (0: Absence of cardiovascular disease, 1: Presence).
- **Features Analyzed:** Age, Gender, Height, Weight, Systolic Blood Pressure (`ap_hi`), Diastolic Blood Pressure (`ap_lo`), Cholesterol, Glucose, Smoking, Alcohol intake, and Physical Activity.
- **Engineered Metrics:**
  - Body Mass Index ($BMI = \text{weight} / (\text{height}/100)^2$)
  - Pulse Pressure ($PP = \text{ap\_hi} - \text{ap\_lo}$)

## 3. Comparative Model Evaluation
Models were benchmarked across stratified 5-fold cross-validation on the cleaned patient cohort:

| Architecture | Accuracy | ROC-AUC | Precision | Recall (Class 1) |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression (Baseline) | 72.1% | 0.784 | 0.741 | 0.672 |
| Random Forest Classifier | 73.4% | 0.791 | 0.752 | 0.698 |
| **XGBoost Classifier (Best Model)** | **73.8%** | **0.801** | **0.758** | **0.714** |
| Deep Learning (Dense MLP - Keras) | 73.5% | 0.798 | 0.750 | 0.709 |

## 4. Serving & Deployment Architecture
- **API Service (`api/app.py`):** Built with FastAPI and Pydantic schemas, exposing `/health` and `/predict` endpoints for low-latency batch and single-patient scoring.
- **Clinical Dashboard (`dashboard/app.py`):** Built with Streamlit, enabling non-technical healthcare practitioners to input patient biometrics, calculate BMI dynamically, and receive immediate probabilistic risk classifications.
