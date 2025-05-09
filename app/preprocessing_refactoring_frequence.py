import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Garde uniquement les colonnes sélectionnées.
    """
    def __init__(self, selected_columns):
        self.selected_columns = selected_columns

    def fit(self, X, y=None): # pylint: disable=unused-argument
        return self

    def transform(self, X):
        return X[self.selected_columns].copy()

class MissingValueFiller(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=None, cat_cols=None):
        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def fit(self, X, y=None): # pylint: disable=unused-argument
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.num_cols:
            for col in self.num_cols:
                if col in X_copy.columns:
                    X_copy[col] = X_copy[col].fillna(0)
        if self.cat_cols:
            for col in self.cat_cols:
                if col in X_copy.columns:
                    X_copy[col] = X_copy[col].fillna("Inconnu")
        return X_copy

class ManualCountEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        self.count_maps = {}

    def fit(self, X, y=None): # pylint: disable=unused-argument
        for col in self.cat_cols:
            counts = X[col].value_counts()
            self.count_maps[col] = counts.to_dict()
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.cat_cols:
            X_copy[col] = X_copy[col].map(self.count_maps[col]).fillna(0)
        return X_copy

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=None, missing_thresh=0.4, var_thresh=0.01, corr_thresh=0.95):
        self.num_cols = num_cols
        self.missing_thresh = missing_thresh
        self.var_thresh = var_thresh
        self.corr_thresh = corr_thresh
        self.columns_to_drop_ = []

    def fit(self, X, y=None): # pylint: disable=unused-argument
        X_copy = X.copy()
        drop_cols = []

        for col in X_copy.columns:
            if col in self.num_cols:
                missing_ratio = (X_copy[col] == 0).sum() / len(X_copy)
            else:
                missing_ratio = (X_copy[col] == "Inconnu").sum() / len(X_copy)
            if missing_ratio > self.missing_thresh:
                drop_cols.append(col)

        var_series = X_copy.var(numeric_only=True)
        low_var_cols = var_series[var_series < self.var_thresh].index.tolist()
        drop_cols += low_var_cols

        corr_matrix = X_copy.select_dtypes(include=["number"]).corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [col for col in upper_tri.columns if any(upper_tri[col] > self.corr_thresh)]
        drop_cols += high_corr_cols

        self.columns_to_drop_ = list(set(drop_cols))
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')

class ScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=None):
        self.num_cols = num_cols
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        if self.num_cols:
            self.scaler.fit(X[self.num_cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.num_cols:
            X_copy[self.num_cols] = self.scaler.transform(X_copy[self.num_cols])
        return X_copy
