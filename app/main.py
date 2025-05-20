"""FastAPI application for predicting insurance claims amount and frequency."""
import webbrowser
import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from configmontant import ORDINAL_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from typing import Optional

# Initialisation de FastAPI
app = FastAPI(
    title="API de Prédiction CM",
    description="API pour prédire le montant et la fréquence des sinistres",
    version="1.0",
)

# Chargement des modèles et des preprocess
model_frequence = joblib.load('model_frequence.pkl')
preprocess_freq_pipeline = joblib.load('preprocessing_frequence.pkl')
model_montant = joblib.load('model_montant.pkl')
preprocess_montant_pipeline = joblib.load('preprocessing_montant.pkl')

# Modèles de données pour la requête
class Item(BaseModel):
    """Modèle de données pour la requête de prédiction."""
    ID: int
    DEROG13: Optional[int] = None
    DEROG14: Optional[int] = None
    DEROG16: Optional[int] = None
    ANCIENNETE: Optional[int] = None
    CARACT2: Optional[int] = None
    DUREE_REQANEUF: Optional[int] = None
    CARACT5: Optional[int] = None
    TYPBAT2: Optional[int] = None
    DEROG1: Optional[int] = None
    DEROG6: Optional[int] = None
    DEROG7: Optional[int] = None
    DEROG9: Optional[int] = None
    DEROG10: Optional[int] = None
    DEROG11: Optional[int] = None
    DEROG15: Optional[int] = None
    CA1: Optional[int] = None
    CA2: Optional[int] = None
    CA3: Optional[int] = None
    KAPITAL1: Optional[int] = None
    KAPITAL2: Optional[int] = None
    KAPITAL3: Optional[int] = None
    KAPITAL4: Optional[int] = None
    KAPITAL5: Optional[int] = None
    KAPITAL6: Optional[int] = None
    KAPITAL7: Optional[int] = None
    KAPITAL8: Optional[int] = None
    KAPITAL9: Optional[int] = None
    KAPITAL10: Optional[int] = None
    KAPITAL11: Optional[int] = None
    KAPITAL12: Optional[int] = None
    KAPITAL13: Optional[int] = None
    KAPITAL14: Optional[int] = None
    KAPITAL15: Optional[int] = None
    KAPITAL16: Optional[int] = None
    KAPITAL17: Optional[int] = None
    KAPITAL18: Optional[int] = None
    KAPITAL19: Optional[int] = None
    KAPITAL20: Optional[int] = None
    KAPITAL21: Optional[int] = None
    KAPITAL22: Optional[int] = None
    KAPITAL23: Optional[int] = None
    KAPITAL24: Optional[int] = None
    KAPITAL25: Optional[int] = None
    KAPITAL26: Optional[int] = None
    KAPITAL27: Optional[int] = None
    KAPITAL28: Optional[int] = None
    KAPITAL29: Optional[int] = None
    KAPITAL30: Optional[int] = None
    KAPITAL31: Optional[int] = None
    KAPITAL32: Optional[int] = None
    KAPITAL33: Optional[int] = None
    KAPITAL36: Optional[int] = None
    KAPITAL38: Optional[int] = None
    KAPITAL39: Optional[int] = None
    SURFACE1: Optional[int] = None
    SURFACE2: Optional[int] = None
    SURFACE3: Optional[int] = None
    SURFACE5: Optional[int] = None
    SURFACE7: Optional[int] = None
    SURFACE8: Optional[int] = None
    SURFACE9: Optional[int] = None
    SURFACE10: Optional[int] = None
    SURFACE11: Optional[int] = None
    SURFACE12: Optional[int] = None
    SURFACE13: Optional[int] = None
    SURFACE14: Optional[int] = None
    SURFACE15: Optional[int] = None
    SURFACE16: Optional[int] = None
    SURFACE17: Optional[int] = None
    SURFACE18: Optional[int] = None
    SURFACE19: Optional[int] = None
    SURFACE20: Optional[int] = None
    SURFACE21: Optional[int] = None
    NBBAT1: Optional[int] = None
    NBBAT2: Optional[int] = None
    NBBAT3: Optional[int] = None
    NBBAT4: Optional[int] = None
    NBBAT5: Optional[int] = None
    NBBAT6: Optional[int] = None
    NBBAT7: Optional[int] = None
    NBBAT8: Optional[int] = None
    NBBAT9: Optional[int] = None
    NBBAT10: Optional[int] = None
    NBBAT11: Optional[int] = None
    NBBAT13: Optional[int] = None
    NBBAT14: Optional[int] = None
    TAILLE3: Optional[int] = None
    TAILLE4: Optional[int] = None
    NBSINCONJ: Optional[int] = None
    NBSINSTRT: Optional[int] = None
    RISK1: Optional[int] = None
    RISK2: Optional[int] = None
    RISK3: Optional[int] = None
    RISK4: Optional[int] = None
    RISK5: Optional[int] = None
    RISK7: Optional[int] = None
    EQUIPEMENT1: Optional[int] = None
    EQUIPEMENT3: Optional[int] = None
    EQUIPEMENT4: Optional[int] = None
    EQUIPEMENT6: Optional[int] = None
    EQUIPEMENT7: Optional[int] = None
    ZONE_VENT: Optional[int] = None



    ACTIVIT2: Optional[int] = None
    VOCATION: Optional[int] = None
    ADOSS: Optional[int] = None
    CARACT1: Optional[int] = None
    CARACT3: Optional[int] = None
    INDEM1: Optional[int] = None
    TYPBAT1: Optional[int] = None
    INDEM2: Optional[int] = None
    FRCH1: Optional[int] = None
    FRCH2: Optional[int] = None
    DEROG2: Optional[int] = None
    DEROG3: Optional[int] = None
    DEROG4: Optional[int] = None
    DEROG5: Optional[int] = None
    DEROG8: Optional[int] = None
    DEROG12: Optional[int] = None
    KAPITAL34: Optional[int] = None
    KAPITAL35: Optional[int] = None
    KAPITAL37: Optional[int] = None
    KAPITAL40: Optional[int] = None
    KAPITAL41: Optional[int] = None
    KAPITAL42: Optional[int] = None
    KAPITAL43: Optional[int] = None
    RISK6: Optional[int] = None
    RISK8: Optional[int] = None
    RISK9: Optional[int] = None
    RISK10: Optional[int] = None
    RISK11: Optional[int] = None
    RISK12: Optional[int] = None
    RISK13: Optional[int] = None
    EQUIPEMENT2: Optional[int] = None
    EQUIPEMENT5: Optional[int] = None
    ESPINSEE: Optional[int] = None
    AN_EXERC: Optional[int] = None
    ZONE: Optional[int] = None
    TYPERS: Optional[int] = None




    NB_CASERNES: Optional[float] = None
    BDTOPO_BAT_MAX_HAUTEUR: Optional[float] = None
    HAUTEUR_MAX: Optional[float] = None
    HAUTEUR: Optional[float] = None
    BDTOPO_BAT_MAX_HAUTEUR_MAX: Optional[float] = None
    MEN_SURF: Optional[float] = None
    IND_SNV: Optional[float] = None
    IND_INC: Optional[float] = None
    IND_Y9: Optional[float] = None
    IND_0_Y1: Optional[float] = None
    IND: Optional[float] = None
    LOG_SOC: Optional[float] = None
    LOG_INC: Optional[float] = None
    LOG_APA3: Optional[float] = None
    LOG_AVA1: Optional[float] = None
    MEN_MAIS: Optional[float] = None
    MEN_COLL: Optional[float] = None
    MEN_FMP: Optional[float] = None
    MEN_PROP: Optional[float] = None
    MEN_PAUV: Optional[float] = None
    MEN: Optional[float] = None
    COEFASS: Optional[float] = None
    DISTANCE_111: Optional[float] = None
    DISTANCE_112: Optional[float] = None
    DISTANCE_121: Optional[float] = None
    DISTANCE_122: Optional[float] = None
    DISTANCE_123: Optional[float] = None
    DISTANCE_124: Optional[float] = None
    DISTANCE_131: Optional[float] = None
    DISTANCE_132: Optional[float] = None
    DISTANCE_133: Optional[float] = None
    DISTANCE_141: Optional[float] = None
    DISTANCE_142: Optional[float] = None
    DISTANCE_211: Optional[float] = None
    DISTANCE_212: Optional[float] = None
    DISTANCE_213: Optional[float] = None
    DISTANCE_221: Optional[float] = None
    DISTANCE_222: Optional[float] = None
    DISTANCE_223: Optional[float] = None
    DISTANCE_231: Optional[float] = None
    DISTANCE_242: Optional[float] = None
    DISTANCE_243: Optional[float] = None
    DISTANCE_244: Optional[float] = None
    DISTANCE_311: Optional[float] = None
    DISTANCE_312: Optional[float] = None
    DISTANCE_313: Optional[float] = None
    DISTANCE_321: Optional[float] = None
    DISTANCE_322: Optional[float] = None
    DISTANCE_323: Optional[float] = None
    DISTANCE_324: Optional[float] = None
    DISTANCE_331: Optional[float] = None
    DISTANCE_332: Optional[float] = None
    DISTANCE_333: Optional[float] = None
    DISTANCE_334: Optional[float] = None
    DISTANCE_335: Optional[float] = None
    DISTANCE_411: Optional[float] = None
    DISTANCE_412: Optional[float] = None
    DISTANCE_421: Optional[float] = None
    DISTANCE_422: Optional[float] = None
    DISTANCE_423: Optional[float] = None
    DISTANCE_511: Optional[float] = None
    DISTANCE_512: Optional[float] = None
    DISTANCE_521: Optional[float] = None
    DISTANCE_522: Optional[float] = None
    DISTANCE_523: Optional[float] = None
    PROPORTION_11: Optional[float] = None
    PROPORTION_12: Optional[float] = None
    PROPORTION_13: Optional[float] = None
    PROPORTION_14: Optional[float] = None
    PROPORTION_21: Optional[float] = None
    PROPORTION_22: Optional[float] = None
    PROPORTION_23: Optional[float] = None
    PROPORTION_24: Optional[float] = None
    PROPORTION_31: Optional[float] = None
    PROPORTION_32: Optional[float] = None
    PROPORTION_33: Optional[float] = None
    PROPORTION_41: Optional[float] = None
    PROPORTION_42: Optional[float] = None
    PROPORTION_51: Optional[float] = None
    PROPORTION_52: Optional[float] = None
    MEN_1IND: Optional[float] = None
    MEN_5IND: Optional[float] = None
    LOG_A1_A2: Optional[float] = None
    LOG_A2_A3: Optional[float] = None
    IND_Y1_Y2: Optional[float] = None
    IND_Y2_Y3: Optional[float] = None
    IND_Y3_Y4: Optional[float] = None
    IND_Y4_Y5: Optional[float] = None
    IND_Y5_Y6: Optional[float] = None
    IND_Y6_Y7: Optional[float] = None
    IND_Y7_Y8: Optional[float] = None
    IND_Y8_Y9: Optional[float] = None
    DISTANCE_1: Optional[float] = None
    DISTANCE_2: Optional[float] = None
    ALTITUDE_1: Optional[float] = None
    ALTITUDE_2: Optional[float] = None
    ALTITUDE_3: Optional[float] = None
    ALTITUDE_4: Optional[float] = None
    ALTITUDE_5: Optional[float] = None
    NBJTX25_MM_A: Optional[float] = None
    NBJTX25_MMAX_A: Optional[float] = None
    NBJTX25_MSOM_A: Optional[float] = None
    NBJTX0_MM_A: Optional[float] = None
    NBJTX0_MMAX_A: Optional[float] = None
    NBJTX0_MSOM_A: Optional[float] = None
    NBJTXI27_MM_A: Optional[float] = None
    NBJTXI27_MMAX_A: Optional[float] = None
    NBJTXI27_MSOM_A: Optional[float] = None
    NBJTXS32_MM_A: Optional[float] = None
    NBJTXS32_MMAX_A: Optional[float] = None
    NBJTXS32_MSOM_A: Optional[float] = None
    NBJTXI20_MM_A: Optional[float] = None
    NBJTXI20_MMAX_A: Optional[float] = None
    NBJTXI20_MSOM_A: Optional[float] = None
    NBJTX30_MM_A: Optional[float] = None
    NBJTX30_MMAX_A: Optional[float] = None
    NBJTX30_MSOM_A: Optional[float] = None
    NBJTX35_MM_A: Optional[float] = None
    NBJTX35_MMAX_A: Optional[float] = None
    NBJTX35_MSOM_A: Optional[float] = None
    NBJTN10_MM_A: Optional[float] = None
    NBJTN10_MMAX_A: Optional[float] = None
    NBJTN10_MSOM_A: Optional[float] = None
    NBJTNI10_MM_A: Optional[float] = None
    NBJTNI10_MMAX_A: Optional[float] = None
    NBJTNI10_MSOM_A: Optional[float] = None
    NBJTN5_MM_A: Optional[float] = None
    NBJTN5_MMAX_A: Optional[float] = None
    NBJTN5_MSOM_A: Optional[float] = None
    NBJTNS25_MM_A: Optional[float] = None
    NBJTNS25_MMAX_A: Optional[float] = None
    NBJTNS25_MSOM_A: Optional[float] = None
    NBJTNI15_MM_A: Optional[float] = None
    NBJTNI15_MMAX_A: Optional[float] = None
    NBJTNI15_MSOM_A: Optional[float] = None
    NBJTNI20_MM_A: Optional[float] = None
    NBJTNI20_MMAX_A: Optional[float] = None
    NBJTNI20_MSOM_A: Optional[float] = None
    NBJTNS20_MM_A: Optional[float] = None
    NBJTNS20_MMAX_A: Optional[float] = None
    NBJTNS20_MSOM_A: Optional[float] = None
    NBJTMS24_MM_A: Optional[float] = None
    NBJTMS24_MMAX_A: Optional[float] = None
    NBJTMS24_MSOM_A: Optional[float] = None
    TAMPLIAB_VOR_MM_A: Optional[float] = None
    TAMPLIAB_VOR_MMAX_A: Optional[float] = None
    TAMPLIM_VOR_MM_A: Optional[float] = None
    TAMPLIM_VOR_MMAX_A: Optional[float] = None
    TM_VOR_MM_A: Optional[float] = None
    TM_VOR_MMAX_A: Optional[float] = None
    TMM_VOR_MM_A: Optional[float] = None
    TMM_VOR_MMAX_A: Optional[float] = None
    TMMAX_VOR_MM_A: Optional[float] = None
    TMMAX_VOR_MMAX_A: Optional[float] = None
    TMMIN_VOR_MM_A: Optional[float] = None
    TMMIN_VOR_MMAX_A: Optional[float] = None
    TN_VOR_MM_A: Optional[float] = None
    TN_VOR_MMAX_A: Optional[float] = None
    TNAB_VOR_MM_A: Optional[float] = None
    TNAB_VOR_MMAX_A: Optional[float] = None
    TNMAX_VOR_MM_A: Optional[float] = None
    TNMAX_VOR_MMAX_A: Optional[float] = None
    TX_VOR_MM_A: Optional[float] = None
    TX_VOR_MMAX_A: Optional[float] = None
    TXAB_VOR_MM_A: Optional[float] = None
    TXAB_VOR_MMAX_A: Optional[float] = None
    TXMIN_VOR_MM_A: Optional[float] = None
    TXMIN_VOR_MMAX_A: Optional[float] = None
    NBJFF10_MM_A: Optional[float] = None
    NBJFF10_MMAX_A: Optional[float] = None
    NBJFF10_MSOM_A: Optional[float] = None
    NBJFF16_MM_A: Optional[float] = None
    NBJFF16_MMAX_A: Optional[float] = None
    NBJFF16_MSOM_A: Optional[float] = None
    NBJFF28_MM_A: Optional[float] = None
    NBJFF28_MMAX_A: Optional[float] = None
    NBJFF28_MSOM_A: Optional[float] = None
    NBJFXI3S10_MM_A: Optional[float] = None
    NBJFXI3S10_MMAX_A: Optional[float] = None
    NBJFXI3S10_MSOM_A: Optional[float] = None
    NBJFXI3S16_MM_A: Optional[float] = None
    NBJFXI3S16_MMAX_A: Optional[float] = None
    NBJFXI3S16_MSOM_A: Optional[float] = None
    NBJFXI3S28_MM_A: Optional[float] = None
    NBJFXI3S28_MMAX_A: Optional[float] = None
    NBJFXI3S28_MSOM_A: Optional[float] = None
    NBJFXY8_MM_A: Optional[float] = None
    NBJFXY8_MMAX_A: Optional[float] = None
    NBJFXY8_MSOM_A: Optional[float] = None
    NBJFXY10_MM_A: Optional[float] = None
    NBJFXY10_MMAX_A: Optional[float] = None
    NBJFXY10_MSOM_A: Optional[float] = None
    NBJFXY15_MM_A: Optional[float] = None
    NBJFXY15_MMAX_A: Optional[float] = None
    NBJFXY15_MSOM_A: Optional[float] = None
    FFM_VOR_MM_A: Optional[float] = None
    FFM_VOR_MMAX_A: Optional[float] = None
    FXI3SAB_VOR_MM_A: Optional[float] = None
    FXI3SAB_VOR_MMAX_A: Optional[float] = None
    FXIAB_VOR_MM_A: Optional[float] = None
    FXIAB_VOR_MMAX_A: Optional[float] = None
    FXYAB_VOR_MM_A: Optional[float] = None
    FXYAB_VOR_MMAX_A: Optional[float] = None
    FFM_VOR_COM_MM_A_Y: Optional[float] = None
    FFM_VOR_COM_MMAX_A_Y: Optional[float] = None
    FXI3SAB_VOR_COM_MM_A_Y: Optional[float] = None
    FXI3SAB_VOR_COM_MMAX_A_Y: Optional[float] = None
    FXIAB_VOR_COM_MM_A_Y: Optional[float] = None
    FXIAB_VOR_COM_MMAX_A_Y: Optional[float] = None
    FXYAB_VOR_COM_MM_A_Y: Optional[float] = None
    FXYAB_VOR_COM_MMAX_A_Y: Optional[float] = None
    FFM_VOR_MM_Y: Optional[float] = None
    FFM_VOR_MMAX_Y: Optional[float] = None
    FXI3SAB_VOR_MM_Y: Optional[float] = None
    FXI3SAB_VOR_MMAX_Y: Optional[float] = None
    FXIAB_VOR_MM_Y: Optional[float] = None
    FXIAB_VOR_MMAX_Y: Optional[float] = None
    FXYAB_VOR_MM_Y: Optional[float] = None
    FXYAB_VOR_MMAX_Y: Optional[float] = None


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
    threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000)).start()

    # Ouvrir Swagger UI dans le navigateur par défaut
    #open_browser()