import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel



app = FastAPI()
model = joblib.load("loan_catboost.pkl")




class Form(BaseModel):
    hit_number: int
    visit_number: int
    utm_medium: str
    device_category: str
    device_brand: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str
    hour: int
    minute: int
    second: int
    visit_year: int
    visit_month: int
    visit_day: int
    visit_dayofweek: int
    visit_is_weekend: int
    hit_year: int
    hit_month: int
    hit_day: int
    hit_dayofweek: int
    hit_is_weekend: int
    brand: str
    model: str

class Prediction(BaseModel):
    Result: int
    Probability: float

@app.get("/status")
def status():
    return {"status": "I'm OK"}

@app.get("/version")
def version():
    return {"model_type": type(model).__name__}

@app.post("/predict", response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame([form.dict()])

    df = df[model.feature_names_]

    y_pred = model.predict(df)
    y_proba = model.predict_proba(df)[:, 1]

    return {
        "Result": int(y_pred[0]),
        "Probability": float(y_proba[0])
    }
