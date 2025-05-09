from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from preprocess_montant import preprocess_montant
from refactoring_preprocess import preprocess_freq
import pandas as pd
from sklearn.pipeline import Pipeline

# Initialisation de FastAPI
app = FastAPI()

# Chargement des modèles et des preprocess
model_montant = joblib.load('model_montant.pkl')  # Modèle XGBoost pour prédiction du montant
preprocess_montant = preprocess_montant()  # Prétraitement pour montant
xgb_regressor = joblib.load('xgb_regressor_model.pkl')  # Modèle XGBoost pour fréquence
full_model_pipeline = joblib.load('full_model_pipeline.pkl')  # Pipeline complet pour fréquence
# Chargement du pipeline de prétraitement pour la fréquence

# Modèle de données pour la requête
class Item(BaseModel):
    ID: int
    # Ajoutez ici d'autres champs selon votre modèle de données
    # Par exemple :
    # column1: str
    # column2: float
    # Assurez-vous que les noms des champs correspondent à ceux de votre DataFrame
class ItemMontant(BaseModel):
    ID: int
    # Ajoutez ici d'autres champs selon votre modèle de données
    # Par exemple :
    # column1: str
    # column2: float
    # Assurez-vous que les noms des champs correspondent à ceux de votre DataFrame
class ItemFreq(BaseModel):
    ID: int
    # Ajoutez ici d'autres champs selon votre modèle de données
    # Par exemple :
    # column1: str
    # column2: float
    # Assurez-vous que les noms des champs correspondent à ceux de votre DataFrame

# Route pour prédire le montant
@app.post("/predict_montant")
async def predict_montant(item: ItemMontant):
    # Convertir l'objet en DataFrame
    data = pd.DataFrame([item.dict()])

    # Prétraitement des données
    data_preprocessed = preprocess_montant.transform(data)

    # Prédiction
    prediction = model_montant.predict(data_preprocessed)

    return {"prediction": prediction.tolist()}

# Route pour prédire la fréquence
@app.post("/predict_freq")
async def predict_freq(item: ItemFreq):
    # Convertir l'objet en DataFrame
    data = pd.DataFrame([item.dict()])

    # Prétraitement des données
    data_preprocessed = preprocess_freq.transform(data)

    # Prédiction
    prediction = xgb_regressor.predict(data_preprocessed)

    return {"prediction": prediction.tolist()}

# Route pour prédire le montant avec le pipeline complet
@app.post("/predict_global")
async def predict_global(item: Item):
    # Convertir l'objet en DataFrame
    data = pd.DataFrame([item.dict()])

    # Prétraitement des données
    data_preprocessed = preprocess_freq.transform(data)

    # Prédiction
    prediction = full_model_pipeline.predict(data_preprocessed)

    return {"prediction": prediction.tolist()}

# Route de santé
@app.get("/")
async def root():
    return {"message": "API de prédiction en cours d'exécution. Utilisez /docs pour accéder à la documentation."}

# Pour exécuter l'application, utilisez la commande suivante dans le terminal :
# uvicorn main:app --reload
# Assurez-vous d'avoir installé FastAPI et Uvicorn :
# pip install fastapi uvicorn
