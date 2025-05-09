"""
Ce module contient des fonctions pour le prétraitement des données, notamment :
- L'encodage des colonnes ordinales et catégorielles en fonction de la fréquence
de la variable cible.
- La conversion des colonnes numériques en types numériques.
"""

import pandas as pd
from config_montant import ORDINAL_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS



def encode_ordinal_by_target_frequency(x_data, y_data):
    """
    Encode les colonnes ordinales en fonction de la fréquence de la variable cible.

    Paramètres :
        - X : pd.DataFrame, DataFrame contenant les variables explicatives
      (doit contenir une colonne 'ID')
    - y : pd.DataFrame, DataFrame contenant la variable cible avec la colonne 'ID' et 'FREQ'

    Retour :
    - X_encoded : pd.DataFrame, DataFrame X avec les colonnes ordinales encodées
    """

    target_name = 'FREQ'
    x_encoded = x_data.copy()

    # Fusionner X et y sur la colonne 'ID'
    merged = pd.merge(x_encoded, y_data, on='ID')

    for col in ORDINAL_COLUMNS:
        if col in x_encoded.columns:
            # Calculer la fréquence de la cible par valeur de la colonne
            freq_target = merged.groupby(col)[target_name].sum()

            # Ordonner les valeurs par fréquence croissante
            sorted_values = freq_target.sort_values().index

            # Dictionnaire d'encodage
            encoding_dict = {value: idx for idx, value in enumerate(sorted_values)}

            # Appliquer l'encodage
            x_encoded[col] = x_encoded[col].map(encoding_dict)

    return x_encoded



def encode_categorical_by_target_frequency(x_data, y_data):
    """
    Encode les colonnes catégorielles en fonction de la fréquence de la variable cible.

    Paramètres :
    - X : pd.DataFrame, DataFrame contenant les variables explicatives
    (doit contenir une colonne 'ID')
    - y : pd.DataFrame, DataFrame contenant la variable cible avec la colonne 'ID' et 'FREQ'

    Retour :
    - X_encoded : pd.DataFrame, DataFrame X avec les colonnes catégorielles encodées
    """

    target_name = 'FREQ'

    x_encoded = x_data.copy()

    # Fusionner X et y selon la colonne 'ID'
    merged = pd.merge(x_encoded, y_data, left_on='ID', right_on='ID')

    for col in CATEGORICAL_COLUMNS:
        if col in x_encoded.columns:
            # Calculer la fréquence de la cible par valeur de la colonne
            freq_target = merged.groupby(col)[target_name].sum()

            # Ordonner les valeurs par fréquence croissante
            sorted_values = freq_target.sort_values().index

            # Dictionnaire d'encodage
            encoding_dict = {value: idx for idx, value in enumerate(sorted_values)}

            # Appliquer l'encodage
            x_encoded[col] = x_encoded[col].map(encoding_dict)

    return x_encoded



def convert_to_numeric(x_data):
    """
    Convertit les colonnes numériques spécifiées en types numériques.

    Paramètres :
    - x_data : pd.DataFrame, DataFrame contenant les variables explicatives.

    Retour :
    - pd.DataFrame : DataFrame x_data avec les colonnes numériques converties en types numériques.
      Les valeurs non convertibles sont remplacées par NaN.
    """
    cols_to_convert = [col for col in NUMERIC_COLUMNS if col in x_data.columns]

    # Appliquer la conversion uniquement sur les colonnes présentes
    x_data[cols_to_convert] = x_data[cols_to_convert].apply(pd.to_numeric, errors='coerce')


    return x_data
