# # app.py

# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib

# # Load model + vectorizer
# model = joblib.load("sentiment_model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# # Initialize FastAPI
# app = FastAPI(title="Twitter Sentiment API")

# # Input schema
# class TextIn(BaseModel):
#     text: str

# @app.post("/predict")
# def predict_sentiment(data: TextIn):
#     # Transform input
#     X = vectorizer.transform([data.text])
#     prediction = model.predict(X)[0]
#     return {"sentiment": prediction}
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained model + vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = FastAPI(title="Twitter Sentiment API")

# Single text input schema
class TextIn(BaseModel):
    text: str

# Endpoint for single text prediction
@app.post("/predict")
def predict_sentiment(data: TextIn):
    X = vectorizer.transform([data.text])
    prediction = model.predict(X)[0]
    return {"sentiment": prediction}

# Endpoint for batch prediction via CSV
@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if "text" not in df.columns:
        return {"error": "CSV must have a 'text' column"}
    
    X = vectorizer.transform(df["text"].astype(str))
    preds = model.predict(X)
    df["sentiment"] = preds
    return df.to_dict(orient="records")
