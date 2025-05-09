from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from preprocess_montant import preprocess_montant
from refactoring_preprocess import preprocess_freq
from sklearn.pipeline import Pipeline

# Initialisation de FastAPI
app = FastAPI()

# Chargement des modèles et des preprocess
model_montant = joblib.load('model_montant.pkl')  # Modèle XGBoost pour prédiction du montant
preprocess_montant = joblib.load('preprocessing_montant.pkl')  # Prétraitement pour montant
xgb_regressor = joblib.load('xgb_regressor.pkl')  # Modèle XGBoost pour fréquence
full_model_pipeline = joblib.load('full_model_pipeline.pkl')  # Pipeline complet pour fréquence

# Modèle de données pour la requête
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float

# Health check
@app.get("/health")
def health_check():
    return {"status": "OK"}

# Route Predict Montant
@app.post("/predict_montant")
def predict_montant(request: PredictionRequest):
    # Prétraitement du montant
    preprocessed_data = preprocess_montant([request.feature1, request.feature2])
    # Prédiction avec le modèle du montant
    montant_prediction = model_montant.predict([preprocessed_data])
    return {"predicted_montant": montant_prediction[0]}

# Route Predict Freq
@app.post("/predict_freq")
def predict_freq(request: PredictionRequest):
    # Prétraitement pour la fréquence
    preprocessed_data_freq = preprocess_freq([request.feature1, request.feature2])
    # Prédiction avec le modèle de fréquence
    freq_prediction = xgb_regressor.predict([preprocessed_data_freq])
    return {"predicted_freq": freq_prediction[0]}

# Route Predict Global
@app.post("/predict_global")
def predict_global(request: PredictionRequest):
    # Prétraitement des données
    preprocessed_data_montant = preprocess_montant([request.feature1, request.feature2])
    preprocessed_data_freq = preprocess_freq([request.feature1, request.feature2])

    # Prédiction pour le montant
    montant_prediction = model_montant.predict([preprocessed_data_montant])[0]

    # Prédiction pour la fréquence
    freq_prediction = xgb_regressor.predict([preprocessed_data_freq])[0]

    # Calcul du produit des deux prédictions
    global_prediction = montant_prediction * freq_prediction
    return {"predicted_global": global_prediction}
