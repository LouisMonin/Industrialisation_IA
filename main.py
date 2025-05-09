"""FastAPI application for prediction service.
This application provides endpoints for health check and prediction of
montant, frequency, and a global prediction based on the provided features.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

app = FastAPI()

# Modèle de données pour la prédiction
class PredictionRequest(BaseModel):
    """
    Encode les colonnes catégorielles en fonction de la fréquence de la variable cible.
    Paramètres :
        - target_col : str, nom de la colonne cible (par défaut 'FREQ')
        - id_col : str, nom de la colonne d'identification (par défaut 'ID')
    """
    feature1: float
    feature2: float

# Route Health
@app.get("/health")
def health_check():
    """
    Vérifie la santé de l'application.
    Renvoie un message de statut.
    """
    return {"status": "OK"}

# Route Predict Montant
@app.post("/predict_montant")
def predict_montant(request: PredictionRequest):
    """
    Prédiction du montant basé sur les caractéristiques fournies.
    Paramètres :
        - request : PredictionRequest, contenant les caractéristiques pour la prédiction
    """
    # Remplacez par votre logique de prédiction
    montant = request.feature1 * 1.5 + request.feature2 * 0.5  # Exemple de calcul
    return {"predicted_montant": montant}

# Route Predict Freq
@app.post("/predict_freq")
def predict_freq(request: PredictionRequest):
    """
    Prédiction de la fréquence basée sur les caractéristiques fournies.
    Paramètres :
        - request : PredictionRequest, contenant les caractéristiques pour la prédiction
    """
    # Remplacez par votre logique de prédiction
    freq = request.feature1 * 2 + request.feature2 * 0.3  # Exemple de calcul
    return {"predicted_freq": freq}

# Route Predict Global
@app.post("/predict_global")
def predict_global(request: PredictionRequest):
    """
    Prédiction globale basée sur les caractéristiques fournies.
    Paramètres :
        - request : PredictionRequest, contenant les caractéristiques pour la prédiction
    """
    montant = request.feature1 * 1.5 + request.feature2 * 0.5  # Exemple de calcul
    freq = request.feature1 * 2 + request.feature2 * 0.3  # Exemple de calcul
    global_prediction = montant * freq  # Produit de montant et fréquence
    return {"predicted_global": global_prediction}
