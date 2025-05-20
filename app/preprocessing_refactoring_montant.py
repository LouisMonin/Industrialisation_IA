import unittest
import pandas as pd
from pandas.testing import assert_series_equal

# Définition temporaire des colonnes ordinales et catégorielles
ORDINAL_COLUMNS = ['grade']
CATEGORICAL_COLUMNS = ['type']
NUMERIC_COLUMNS = ['amount']

class NumericConverter:
    def fit(self, X):
        pass

    def transform(self, X):
        X['amount'] = pd.to_numeric(X['amount'], errors='coerce')
        X['amount'] = X['amount'].astype('float64')  # S'assurer que le type est float64
        return X


class OrdinalEncoderByTargetFrequency:
    def __init__(self, target_col, id_col):
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


class CategoricalEncoderByTargetFrequency:
    def __init__(self, target_col, id_col):
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


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.x_data = pd.DataFrame({
            'ID': [1, 2, 3],
            'grade': ['B', 'A', 'C'],
            'type': ['cat1', 'cat2', 'cat1'],
            'amount': ['10', 'invalid', '30']
        })

        self.y = pd.DataFrame({
            'ID': [1, 2, 3],
            'FREQ': [5, 10, 1]
        })

    def test_convert_to_numeric(self):
        converter = NumericConverter()
        converter.fit(self.x_data.copy())
        result = converter.transform(self.x_data.copy())

        expected = pd.Series([10.0, float('nan'), 30.0], name='amount', dtype='float64')
        assert_series_equal(result['amount'], expected, check_dtype=True)

    def test_encode_ordinal_by_target_frequency(self):
        encoder = OrdinalEncoderByTargetFrequency(target_col='FREQ', id_col='ID')
        encoder.fit(self.x_data.copy(), self.y)
        result = encoder.transform(self.x_data.copy())

        expected = [1, 2, 0]  # B=5 (1), A=10 (2), C=1 (0)
        self.assertListEqual(result['grade'].tolist(), expected)

    def test_encode_categorical_by_target_frequency(self):
        encoder = CategoricalEncoderByTargetFrequency(target_col='FREQ', id_col='ID')
        encoder.fit(self.x_data.copy(), self.y)
        result = encoder.transform(self.x_data.copy())

        expected = [0, 1, 0]  # cat1=6 (0), cat2=10 (1)
        self.assertListEqual(result['type'].tolist(), expected)


if __name__ == '__main__':
    unittest.main()
