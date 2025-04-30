# Importer les bibliothèques nécessaires
import pandas as pd
from config import ORDINAL_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS



# Encode les colonnes ordinales en fonction de la fréquence de la variable cible  
def encode_ordinal_by_target_frequency(X, y):

    # X : pd.DataFrame, DataFrame contenant les variables explicatives
    # y : pd.DataFrame, DataFrame contenant la variable cible

    # Sortie :  pd.DataFrame, DataFrame X avec les colonnes ordinales encodées
    
    target_name = 'FREQ'

    X_encoded = X.copy()

    # Fusionner X et y selon la colonne 'ID'
    merged = pd.merge(X_encoded, y, left_on='ID', right_on='ID')

    for col in ORDINAL_COLUMNS:
        # Calculer la fréquence de la cible par valeur de la colonne
        freq_target = merged.groupby(col)[target_name].sum()
        
        # Ordonner les valeurs par fréquence croissante
        sorted_values = freq_target.sort_values().index
        
        # Dictionnaire d'encodage
        encoding_dict = {value: idx for idx, value in enumerate(sorted_values)}
        
        # Appliquer l'encodage
        X_encoded[col] = X_encoded[col].map(encoding_dict)

    return X_encoded


# Encode les colonnes catégorielles en fonction de la fréquence de la variable cible
def encode_categorical_by_target_frequency(X, y):

    # X : pd.DataFrame, DataFrame contenant les variables explicatives
    # y : pd.DataFrame, DataFrame contenant la variable cible

    # Sortie :  pd.DataFrame, DataFrame X avec les colonnes catégorielles encodées
    
    target_name = 'FREQ'

    X_encoded = X.copy()
    
    # Fusionner X et y selon la colonne 'ID'
    merged = pd.merge(X_encoded, y, left_on='ID', right_on='ID')
    
    for col in CATEGORICAL_COLUMNS:
        # Calculer la fréquence de la cible par valeur de la colonne
        freq_target = merged.groupby(col)[target_name].sum()
        
        # Ordonner les valeurs par fréquence croissante
        sorted_values = freq_target.sort_values().index
        
        # Dictionnaire d'encodage
        encoding_dict = {value: idx for idx, value in enumerate(sorted_values)}
        
        # Appliquer l'encodage
        X_encoded[col] = X_encoded[col].map(encoding_dict)

    return X_encoded



# Convertir les colonnes numériques
def convert_to_numeric(X):
    
    # X : pd.DataFrame, DataFrame contenant les variables explicatives
    # Sortie :  pd.DataFrame, DataFrame X avec les colonnes numériques converties
    
    X[NUMERIC_COLUMNS] = X[NUMERIC_COLUMNS].apply(pd.to_numeric, errors='coerce')

    return X


