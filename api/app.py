from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Cardiovascular Disease Risk Prediction API", version="1.0.0")

class PatientData(BaseModel):
    age_years: float
    gender: int  # 1: female, 2: male
    height: float
    weight: float
    ap_hi: int   # Systolic blood pressure
    ap_lo: int   # Diastolic blood pressure
    cholesterol: int  # 1: normal, 2: above normal, 3: well above normal
    gluc: int         # 1: normal, 2: above normal, 3: well above normal
    smoke: int        # 0: no, 1: yes
    alco: int         # 0: no, 1: yes
    active: int       # 0: no, 1: yes

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Cardiovascular Disease Predictor"}

@app.post("/predict")
def predict_cardio_risk(patient: PatientData):
    # Clinical risk heuristic / Model inference wrapper
    bmi = patient.weight / ((patient.height / 100) ** 2)
    risk_score = 0.1
    if patient.ap_hi >= 140 or patient.ap_lo >= 90:
        risk_score += 0.4
    if patient.cholesterol > 1:
        risk_score += 0.2
    if patient.age_years > 50:
        risk_score += 0.15
    if bmi > 25:
        risk_score += 0.1
    
    probability = min(max(risk_score, 0.05), 0.95)
    prediction = 1 if probability >= 0.5 else 0
    risk_category = "High Risk" if prediction == 1 else "Low Risk"

    return {
        "prediction": prediction,
        "cardiovascular_risk": risk_category,
        "probability": round(probability, 4),
        "calculated_bmi": round(bmi, 2)
    }
