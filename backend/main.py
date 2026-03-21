from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="Loan Default Prediction API",
    description="Predicts whether a customer is likely to default on a loan.",
    version="1.0.0"
)

# Load the trained model
model = joblib.load("model.pkl")

# Define the input schema
class LoanApplication(BaseModel):
    acct_age: float
    limit: float
    outs: float
    loan_tenure: float
    instalamt: float
    kyc_scr: float
    criff_33: float
    credit_history_length: float
    overspend_ratio: float
    max_consec_overspend: float
    outbal_slope: float
    slope_mtd: float

# Health check endpoint
@app.get("/")
def root():
    return {"status": "Loan Default Prediction API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(application: LoanApplication):
    data = np.array([[
        application.acct_age,
        application.limit,
        application.outs,
        application.loan_tenure,
        application.instalamt,
        application.kyc_scr,
        application.criff_33,
        application.credit_history_length,
        application.overspend_ratio,
        application.max_consec_overspend,
        application.outbal_slope,
        application.slope_mtd
    ]])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "prediction": int(prediction),
        "label": "Defaulter" if prediction == 1 else "Non-Defaulter",
        "default_probability": round(float(probability), 4)
    }