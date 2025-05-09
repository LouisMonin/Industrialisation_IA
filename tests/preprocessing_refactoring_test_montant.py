"""
Ce module contient des tests unitaires pour les fonctions du module preprocess :
- encode_ordinal_by_target_frequency
- encode_categorical_by_target_frequency
- convert_to_numeric
"""

import unittest
import pandas as pd
from pandas.testing import assert_series_equal

from preprocessing_refactoring_montant import (
    OrdinalEncoderByTargetFrequency,
    CategoricalEncoderByTargetFrequency,
    NumericConverter
)

# Définition des colonnes utilisées pour les tests
from config import ORDINAL_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

# Surcharge temporaire des colonnes dans le contexte de test
ORDINAL_COLUMNS[:] = ['grade']
CATEGORICAL_COLUMNS[:] = ['type']
NUMERIC_COLUMNS[:] = ['amount']


class TestPreprocessing(unittest.TestCase):
    """
    Cette classe contient des tests unitaires pour les encodeurs et convertisseurs personnalisés.
    """

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

    def test_encode_ordinal_by_target_frequency(self):
        """
        Teste l'encodage des colonnes ordinales en fonction de la fréquence de la variable cible.
        """
        encoder = OrdinalEncoderByTargetFrequency(target_col='FREQ', id_col='ID')
        encoder.fit(self.x_data.copy(), self.y)
        result = encoder.transform(self.x_data.copy())

        expected = [1, 2, 0]  # B=5 (1), A=10 (2), C=1 (0)
        self.assertListEqual(result['grade'].tolist(), expected)

    def test_encode_categorical_by_target_frequency(self):
        """
        Teste l'encodage des colonnes catégorielles en fonction de la fréquence de la variable cible.
        """
        encoder = CategoricalEncoderByTargetFrequency(target_col='FREQ', id_col='ID')
        encoder.fit(self.x_data.copy(), self.y)
        result = encoder.transform(self.x_data.copy())

        expected = [0, 1, 0]  # cat1=6 (0), cat2=10 (1)
        self.assertListEqual(result['type'].tolist(), expected)

    def test_convert_to_numeric(self):
        """
        Teste la conversion des colonnes numériques en type float.
        """
        converter = NumericConverter()
        converter.fit(self.x_data.copy())
        result = converter.transform(self.x_data.copy())

        expected = pd.Series([10.0, float('nan'), 30.0], name='amount', dtype='float64')
        assert_series_equal(result['amount'], expected, check_dtype=False)


if __name__ == '__main__':
    # Exécute les tests
    unittest.main()
