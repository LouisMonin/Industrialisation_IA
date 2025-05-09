from fastapi import FastAPI
from pydantic import BaseModel
from model_to_pkl import ColumnSelector, MissingValueFiller, ManualCountEncoder, ColumnDropper, ScalerWrapper
from refactoring_preprocess import OrdinalEncoderByTargetFrequency, CategoricalEncoderByTargetFrequency, NumericConverter
import joblib
import pandas as pd

# Initialisation de FastAPI
app = FastAPI()

# Chargement des modèles et des preprocess
model_montant = joblib.load('model_montant.pkl')  # Modèle XGBoost pour prédiction du montant
preprocess_montant_pipeline = joblib.load('preprocessing_montant.pkl')  # Pipeline de prétraitement pour montant
full_model_pipeline = joblib.load('full_model_pipeline.pkl')  # Pipeline complet pour la fréquence

# Modèle de données pour la requête
class ItemMontant(BaseModel):
    ID: int
    # Ajoutez ici d'autres champs selon votre modèle de données
    # Par exemple :
    # column1: str
    # column2: float

class ItemFreq(BaseModel):
    ID: int
    # Ajoutez ici d'autres champs selon votre modèle de données
    # Par exemple :
    # column1: str
    # column2: float

# Route pour prédire le montant
@app.post("/predict_montant")
async def predict_montant(item: ItemMontant):
    # Convertir l'objet en DataFrame
    data = pd.DataFrame([item.dict()])

    # Prétraitement des données avec le pipeline
    data_preprocessed = preprocess_montant_pipeline.transform(data)

    # Prédiction avec le modèle de montant
    prediction = model_montant.predict(data_preprocessed)

    return {"prediction": prediction.tolist()}

# Route pour prédire la fréquence
@app.post("/predict_freq")
async def predict_freq(item: ItemFreq):
    # Convertir l'objet en DataFrame
    data = pd.DataFrame([item.dict()])

    # Prétraitement des données avec le pipeline de fréquence
    data_preprocessed = full_model_pipeline.transform(data)

    # Prédiction avec le modèle complet pour la fréquence
    prediction = full_model_pipeline.predict(data_preprocessed)

    return {"prediction": prediction.tolist()}

# Route pour prédire l'output final avec le pipeline complet
@app.post("/predict_global")
async def predict_global(item: ItemMontant):
    # Convertir l'objet en DataFrame
    data = pd.DataFrame([item.dict()])

    # Prétraitement des données avec le pipeline complet pour fréquence
    data_preprocessed = full_model_pipeline.transform(data)

    # Prédiction avec le modèle complet
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
