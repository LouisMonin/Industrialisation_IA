import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from config import ORDINAL_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


class OrdinalEncoderByTargetFrequency(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='FREQ', id_col='ID'):
        self.target_col = target_col
        self.id_col = id_col
        self.encoding_maps = {}

    def fit(self, X, y):
        df = X.copy()
        df[self.target_col] = y[self.target_col].values
        df[self.id_col] = y[self.id_col].values

        for col in ORDINAL_COLUMNS:
            if col in df.columns:
                freq = df.groupby(col)[self.target_col].sum()
                sorted_values = freq.sort_values().index
                self.encoding_maps[col] = {val: i for i, val in enumerate(sorted_values)}
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col, mapping in self.encoding_maps.items():
            if col in X_encoded.columns:
                X_encoded[col] = X_encoded[col].map(mapping)
        return X_encoded


class CategoricalEncoderByTargetFrequency(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='FREQ', id_col='ID'):
        self.target_col = target_col
        self.id_col = id_col
        self.encoding_maps = {}

    def fit(self, X, y):
        df = X.copy()
        df[self.target_col] = y[self.target_col].values
        df[self.id_col] = y[self.id_col].values

        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                freq = df.groupby(col)[self.target_col].sum()
                sorted_values = freq.sort_values().index
                self.encoding_maps[col] = {val: i for i, val in enumerate(sorted_values)}
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col, mapping in self.encoding_maps.items():
            if col in X_encoded.columns:
                X_encoded[col] = X_encoded[col].map(mapping)
        return X_encoded


class NumericConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = [col for col in NUMERIC_COLUMNS]

    def fit(self, X, y=None):
        self.columns = [col for col in self.columns if col in X.columns]
        return self

    def transform(self, X):
        X_converted = X.copy()
        X_converted[self.columns] = X_converted[self.columns].apply(pd.to_numeric, errors='coerce')
        return X_converted
