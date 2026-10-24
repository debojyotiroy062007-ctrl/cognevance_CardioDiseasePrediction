```markdown
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

```

---

## 📊 Dataset & Feature Engineering

The underlying model is trained on a clinical cohort of **70,000 anonymized patient records** evaluating 11 core physiological features:

* **Demographics:** Age (standardized from days to years), Gender.
* **Objective Measurements:** Height, Weight, Systolic Blood Pressure (`ap_hi`), Diastolic Blood Pressure (`ap_lo`).
* **Examination Markers:** Cholesterol level (1: normal, 2: above normal, 3: well above normal), Glucose level.
* **Subjective Lifestyle Inputs:** Smoking status, Alcohol consumption, Physical activity.
* **Engineered Clinical Indicators:**
* **Body Mass Index ($BMI$):** $\text{BMI} = \frac{\text{weight (kg)}}{(\text{height (m)})^2}$
* **Pulse Pressure ($PP$):** $\text{ap\_hi} - \text{ap\_lo}$ (identifies arterial stiffness and hypertension severity)



---

## 📈 Model Performance & Benchmarking

To ensure clinical validity and minimize false-negative misclassifications, multiple model paradigms were evaluated under stratified 5-fold cross-validation:

| Algorithm / Architecture | Accuracy | ROC-AUC | Precision | Recall (Class 1) |
| --- | --- | --- | --- | --- |
| Logistic Regression (Baseline) | 72.1% | 0.784 | 0.741 | 0.672 |
| Random Forest Classifier | 73.4% | 0.791 | 0.752 | 0.698 |
| **XGBoost Classifier (Best Classical ML)** | **73.8%** | **0.801** | **0.758** | **0.714** |
| Deep Neural Network (Dense MLP) | 73.5% | 0.798 | 0.750 | 0.709 |

### Key Findings

* **Optimal Thresholding:** Blood pressure (`ap_hi` and `ap_lo`) accompanied by calculated $BMI$ demonstrated the highest feature importance across both tree-based models and neural gradients.
* **Deep Learning Implementation:** A Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout regularization matched tree-based performance without exhibiting divergence or overfitting.

---

## 🚀 Setup & Execution Guide

### 1. Clone the Repository & Set Up Environment

```bash
git clone [https://github.com/debojyotiroy062007-ctrl/cognevance_CardioDiseasePrediction.git](https://github.com/debojyotiroy062007-ctrl/cognevance_CardioDiseasePrediction.git)
cd cognevance_CardioDiseasePrediction

# Initialize virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

```

### 2. Train Models

* **Classical ML Pipeline:**
```bash
python main.py

```


* **Deep Learning Neural Pipeline:**
```bash
python src/train_dl.py

```



---

## 🌐 Deployment & Serving

### 1. Run the FastAPI Service

Launch the inference server for programmatic scoring:

```bash
uvicorn api.app:app --reload --port 8000

```

* **Live Interactive Swagger Docs:** `http://localhost:8000/docs`
* **Health Check Route:** `GET http://localhost:8000/health`
* **Prediction Route:** `POST http://localhost:8000/predict`

**Sample JSON Request Payload:**

```json
{
  "age_years": 54.0,
  "gender": 2,
  "height": 172.0,
  "weight": 80.0,
  "ap_hi": 135,
  "ap_lo": 85,
  "cholesterol": 2,
  "gluc": 1,
  "smoke": 0,
  "alco": 0,
  "active": 1
}

```

### 2. Launch the Streamlit Diagnostic Dashboard

Launch the interactive web portal for live clinical evaluations:

```bash
streamlit run dashboard/app.py

```

* **Local Web Interface:** `http://localhost:8501`

---

## 🛠️ Technology Stack

* **Core Machine Learning:** Python, NumPy, Pandas, Scikit-learn, XGBoost
* **Deep Learning:** TensorFlow / Keras (Multi-Layer Perceptrons)
* **API Framework:** FastAPI, Pydantic, Uvicorn
* **User Interface:** Streamlit
* **Visualizations:** Matplotlib, Seaborn

```

```
