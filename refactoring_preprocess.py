"""
Refactoring du code de prétraitement pour le rendre plus modulaire et réutilisable.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from config import ORDINAL_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


class OrdinalEncoderByTargetFrequency(BaseEstimator, TransformerMixin):
    """
    Encode les colonnes ordinales en fonction de la fréquence de la variable cible.
    Paramètres :
        - target_col : str, nom de la colonne cible (par défaut 'FREQ')
        - id_col : str, nom de la colonne d'identification (par défaut 'ID')
    """

    def __init__(self, target_col='FREQ', id_col='ID'):
        """
        Initialise l'encodeur avec les colonnes cibles et d'identification.
        """
        self.target_col = target_col
        self.id_col = id_col
        self.encoding_maps = {}

    def fit(self, x_data, y_data):
        """
        Ajuste l'encodeur sur les données d'entrée et la cible.
        """
        df = x_data.copy()
        df[self.target_col] = y_data[self.target_col].values
        df[self.id_col] = y_data[self.id_col].values

        for col in ORDINAL_COLUMNS:
            if col in df.columns:
                freq = df.groupby(col)[self.target_col].sum()
                sorted_values = freq.sort_values().index
                self.encoding_maps[col] = {val: i for i, val in enumerate(sorted_values)}
        return self

    def transform(self, x_data):
        """
        Transforme les données d'entrée en appliquant l'encodage.
        """
        x_data_encoded = x_data.copy()
        for col, mapping in self.encoding_maps.items():
            if col in x_data_encoded.columns:
                x_data_encoded[col] = x_data_encoded[col].map(mapping)
        return x_data_encoded


class CategoricalEncoderByTargetFrequency(BaseEstimator, TransformerMixin):
    """
    Encode les colonnes catégorielles en fonction de la fréquence de la variable cible.
    Paramètres :
        - target_col : str, nom de la colonne cible (par défaut 'FREQ')
        - id_col : str, nom de la colonne d'identification (par défaut 'ID')
    """

    def __init__(self, target_col='FREQ', id_col='ID'):
        """
        Initialise l'encodeur avec les colonnes cibles et d'identification.
        """
        self.target_col = target_col
        self.id_col = id_col
        self.encoding_maps = {}

    def fit(self, x_data, y):
        """
        Ajuste l'encodeur sur les données d'entrée et la cible.
        """
        df = x_data.copy()
        df[self.target_col] = y[self.target_col].values
        df[self.id_col] = y[self.id_col].values

        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                freq = df.groupby(col)[self.target_col].sum()
                sorted_values = freq.sort_values().index
                self.encoding_maps[col] = {val: i for i, val in enumerate(sorted_values)}
        return self

    def transform(self, x_data):
        """
        Transforme les données d'entrée en appliquant l'encodage.
        """
        x_data_encoded = x_data.copy()
        for col, mapping in self.encoding_maps.items():
            if col in x_data_encoded.columns:
                x_data_encoded[col] = x_data_encoded[col].map(mapping)
        return x_data_encoded


class NumericConverter(BaseEstimator, TransformerMixin):
    """
    Convertit les colonnes numériques en types numériques.
    """
    def __init__(self):
        self.columns = list(NUMERIC_COLUMNS)

    def fit(self, x_data):
        """
        Ajuste le convertisseur sur les données d'entrée.
        """
        self.columns = [col for col in self.columns if col in x_data.columns]
        return self

    def transform(self, x_data):
        """
        Transforme les données d'entrée en appliquant la conversion numérique.
        """
        x_data_converted = x_data.copy()
        x_data_converted[self.columns] = x_data_converted[self.columns].apply(
            lambda col: pd.to_numeric(col, errors='coerce')
        )
        return x_data_converted
