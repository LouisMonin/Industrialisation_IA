"""
Ce script crée un pipeline de prétraitement pour les données d'entraînement.
Il encode les colonnes ordinales et catégorielles en fonction de la fréquence de la variable cible
et convertit les colonnes numériques en types appropriés.
Il sauvegarde ensuite le pipeline dans un fichier .pkl pour une utilisation ultérieure.
"""

# Importation des bibliothèques nécessaires
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from preprocessing_refactoring_montant import OrdinalEncoderByTargetFrequency, CategoricalEncoderByTargetFrequency, NumericConverter


# === 1. Charger les données ===
X = pd.read_csv("/Users/sabine/Desktop/CYTech/S3/Data_science/ProjetFinal/03.Données/train_input.csv", low_memory=False)
y = pd.read_csv("/Users/sabine/Desktop/CYTech/S3/Data_science/ProjetFinal/03.Données/train_output.csv", low_memory=False)

# === 2. Créer les objets de prétraitement ===
ordinal_encoder = OrdinalEncoderByTargetFrequency(target_col='FREQ', id_col='ID')
categorical_encoder = CategoricalEncoderByTargetFrequency(target_col='FREQ', id_col='ID')
numeric_converter = NumericConverter()

# === 3. Créer un pipeline ===
preprocessing_pipeline = Pipeline([
    ("ordinal_encoder", ordinal_encoder),
    ("categorical_encoder", categorical_encoder),
    ("numeric_converter", numeric_converter)
])

# === 4. Entraîner le pipeline ===
preprocessing_pipeline.fit(X, y)

# === 5. Sauvegarder dans un fichier .pkl ===
with open("preprocessing_montant.pkl", "wb") as f:
    pickle.dump(preprocessing_pipeline, f)

print("✅ Pipeline de prétraitement sauvegardé dans preprocessing_montant.pkl")
