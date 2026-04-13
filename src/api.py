from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / 'models'

model = joblib.load(MODELS_DIR / 'model.pkl')
scaler = joblib.load(MODELS_DIR / 'scaler.pkl')

app = FastAPI(title = "Heart Disease Predictor")

class PatientData(BaseModel):
    age:      float
    sex:      float
    cp:       float
    trestbps: float
    chol:     float
    fbs:      float
    restecg:  float
    thalach:  float
    exang:    float
    oldpeak:  float
    slope:    float
    ca:       float
    thal:     float
    
@app.get("/")
def root():
    return {"message": "Hear Disease Prediction API is running!"}

@app.post("/predict")
def predict(patient: PatientData):
    # Convert input to numpy array in the right order
    features = np.array([[
        patient.age, patient.sex, patient.cp, patient.trestbps,
        patient.chol, patient.fbs, patient.restecg, patient.thalach,
        patient.exang, patient.oldpeak, patient.slope, patient.ca,
        patient.thal
    ]])
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return {
        "prediction": int(prediction),
        "diagnosis": "Heart Disease" if prediction == 1 else "Healthy",
        "probability": round(float(probability), 3)
    }