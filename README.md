# 🫀 Cardiovascular Disease Predictive Analytics & AI Deployment
### Cognevance Technologies — AI & Machine Learning (Level 3 Project)

---

## 📌 Project Overview
Cardiovascular diseases (CVDs) are the leading cause of global mortality. This project delivers an end-to-end, production-grade Machine Learning and Deep Learning system designed to accurately classify and predict cardiovascular disease risk from patient biometric indicators.

The pipeline integrates automated data preprocessing, clinical feature engineering, multi-model benchmarking (Classical vs. Deep Learning), low-latency REST API deployment via **FastAPI**, and an interactive diagnostic web portal powered by **Streamlit**.

---

## 🏗️ Repository Architecture

```text
cognevance_CardioDiseasePrediction/
│
├── api/
│   └── app.py                  # High-performance FastAPI serving endpoints (/predict, /health)
│
├── dashboard/
│   └── app.py                  # Interactive Streamlit clinical interface & analytics portal
│
├── src/
│   └── train_dl.py             # Deep Neural Network (MLP) architecture using TensorFlow/Keras
│
├── outputs/
│   └── visualizations/         # Confusion matrices, ROC-AUC curves, and EDA distribution plots
│
├── report/
│   └── project_report.md       # Comprehensive technical project documentation & audit
│
├── cardio_train (1).csv        # Primary patient cohort dataset (70,000 clinical records)
├── main.py                     # Core data preprocessing and classical ML benchmark pipeline
├── requirements.txt            # System environment dependencies
└── README.md                   # Project documentation and reproduction guide
