import webbrowser
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialisation de FastAPI
app = FastAPI(
    title="API de Prédiction CM",
    description="API pour prédire le montant et la fréquence des sinistres, avec des modèles XGBoost.",
    version="1.0",
)

# Chargement des modèles et des preprocess
model_montant = joblib.load('model_montant.pkl')  # Modèle XGBoost pour prédiction du montant
preprocess_montant_pipeline = joblib.load('preprocessing_montant.pkl')  # Pipeline de prétraitement pour montant
full_model_pipeline = joblib.load('full_model_pipeline.pkl')  # Pipeline complet pour la fréquence

# Modèles de données pour la requête
class ItemMontant(BaseModel):
    ID: int
    BDTOPO_BAT_MAX_HAUTEUR: float
    HAUTEUR_MAX: float

class ItemFreq(BaseModel):
    ID: int

# Route de santé
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Route pour prédire le montant
@app.post("/predict_montant")
async def predict_montant(item: ItemMontant):
    data = pd.DataFrame([item.dict()])
    data_preprocessed = preprocess_montant_pipeline.transform(data)
    prediction = model_montant.predict(data_preprocessed)
    return {"prediction": prediction.tolist()}

# Route pour prédire la fréquence
@app.post("/predict_freq")
async def predict_freq(item: ItemFreq):
    data = pd.DataFrame([item.dict()])
    data_preprocessed = full_model_pipeline.transform(data)
    prediction = full_model_pipeline.predict(data_preprocessed)
    return {"prediction": prediction.tolist()}

# Route pour prédire le montant et la fréquence (produit des deux)
@app.post("/predict_global")
#Produit de prediction de montant et de fréquence
async def predict_global(item: ItemMontant):
    data = pd.DataFrame([item.dict()])
    data_preprocessed = preprocess_montant_pipeline.transform(data)
    montant_prediction = model_montant.predict(data_preprocessed)

    # Prétraitement pour la fréquence
    data_freq = pd.DataFrame({"ID": item.ID})
    data_freq_preprocessed = full_model_pipeline.transform(data_freq)
    freq_prediction = full_model_pipeline.predict(data_freq_preprocessed)

    # Produit des deux prédictions
    global_prediction = montant_prediction * freq_prediction
    return {"prediction": global_prediction.tolist()}

# Fonction pour ouvrir Swagger UI dans le navigateur par défaut
def open_browser():
    webbrowser.open("http://127.0.0.1:8000/docs", new=2)

# Démarrer le serveur FastAPI et ouvrir Swagger UI automatiquement
if __name__ == "__main__":
    import threading

    # Démarrer le serveur dans un thread séparé
    threading.Thread(target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000)).start()

    # Ouvrir Swagger UI dans le navigateur par défaut
    open_browser()
