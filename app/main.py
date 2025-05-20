"""FastAPI application for predicting insurance claims amount and frequency."""
import webbrowser
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from app.configmontant import ORDINAL_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

# Initialisation de FastAPI
app = FastAPI(
    title="API de Prédiction CM",
    description="API pour prédire le montant et la fréquence des sinistres",
    version="1.0",
)

# Chargement des modèles et des preprocess
model_frequence = joblib.load('./app/model_frequence.pkl')
preprocess_freq_pipeline = joblib.load('./app/preprocessing_frequence.pkl')
model_montant = joblib.load('./app/model_montant.pkl')
preprocess_montant_pipeline = joblib.load('./app/preprocessing_montant.pkl')

# Modèles de données pour la requête
class Item(BaseModel):
    """Modèle de données pour la requête de prédiction."""
    ID: int
    ordinal_columns: list = ORDINAL_COLUMNS
    categorical_columns: list = CATEGORICAL_COLUMNS
    numeric_columns: list = NUMERIC_COLUMNS


# Route de santé
@app.get("/health")
async def health():
    """Vérifie la santé de l'API."""
    return {"status": "healthy"}

# Route pour prédire le montant
@app.post("/predict_montant")
async def predict_montant(item: Item):
    """Prédit le montant d'un sinistre."""
    data = pd.DataFrame([item.dict()])
    data_preprocessed = preprocess_montant_pipeline.transform(data)
    prediction = model_montant.predict(data_preprocessed)
    return {"prediction": prediction.tolist()}

# Route pour prédire la fréquence
@app.post("/predict_freq")
async def predict_freq(item: Item):
    """Prédit la fréquence d'un sinistre."""
    data = pd.DataFrame([item.dict()])
    data_preprocessed = preprocess_freq_pipeline.transform(data)
    prediction = model_frequence.predict(data_preprocessed)
    return {"prediction": prediction.tolist()}

# Route pour prédire le montant et la fréquence (produit des deux)
@app.post("/predict_global")
#Produit de prediction de montant et de fréquence
async def predict_global(item: Item):
    """Prédit la franchise associée à un sinistre."""
    data = pd.DataFrame([item.dict()])
    data_preprocessed = preprocess_montant_pipeline.transform(data)
    montant_prediction = model_montant.predict(data_preprocessed)

    # Prétraitement pour la fréquence
    data_freq = pd.DataFrame({"ID": item.ID})
    data_freq_preprocessed = preprocess_freq_pipeline.transform(data_freq)
    freq_prediction = model_frequence.predict(data_freq_preprocessed)

    # Produit des deux prédictions
    global_prediction = montant_prediction * freq_prediction
    return {"prediction": global_prediction.tolist()}

# Fonction pour ouvrir Swagger UI dans le navigateur par défaut
def open_browser():
    """Ouvre Swagger UI dans le navigateur par défaut."""
    webbrowser.open("http://127.0.0.1:8000/docs", new=2)

# Démarrer le serveur FastAPI et ouvrir Swagger UI automatiquement
if __name__ == "__main__":
    import threading

    # Démarrer le serveur dans un thread séparé
    threading.Thread(target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000)).start()

    # Ouvrir Swagger UI dans le navigateur par défaut
    open_browser()