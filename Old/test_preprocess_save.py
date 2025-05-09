"""
Ce module contient des tests unitaires pour les fonctions du module preprocess :
- encode_ordinal_by_target_frequency
- encode_categorical_by_target_frequency
- convert_to_numeric
"""

import unittest
import pandas as pd
from pandas.testing import assert_series_equal
import Old.preprocess as preprocess

# Importation des colonnes de configuration
from config_montant import ORDINAL_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

# Surcharge les colonnes utilisées pour les tests
preprocess.ORDINAL_COLUMNS = ['grade']
preprocess.CATEGORICAL_COLUMNS = ['type']
preprocess.NUMERIC_COLUMNS = ['amount']


class TestPreprocessing(unittest.TestCase):
    """
    Cette classe contient des tests unitaires pour les fonctions du module preprocess :
    - encode_ordinal_by_target_frequency
    - encode_categorical_by_target_frequency
    - convert_to_numeric
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
        result = preprocess.encode_ordinal_by_target_frequency(self.x_data.copy(), self.y)
        # FREQ par grade : A=10, B=5, C=1 ⇒ trié : C (0), B (1), A (2)
        expected = [1, 2, 0]  # B, A, C
        self.assertListEqual(result['grade'].tolist(), expected)

    def test_encode_categorical_by_target_frequency(self):
        """
        Teste l'encodage des colonnes catégorielles en fonction de la fréquence de la variable cible
        """
        result = preprocess.encode_categorical_by_target_frequency(self.x_data.copy(), self.y)
        # FREQ par type : cat1 = 5 + 1 = 6, cat2 = 10 ⇒ trié : cat1 (0), cat2 (1)
        expected = [0, 1, 0]
        self.assertListEqual(result['type'].tolist(), expected)

    def test_convert_to_numeric(self):
        """
        Teste la conversion des colonnes numériques en types numériques.
        """
        result = preprocess.convert_to_numeric(self.x_data.copy())
        expected = pd.Series([10.0, float('nan'), 30.0], name='amount', dtype='float64')
        assert_series_equal(result['amount'], expected, check_dtype=False)


if __name__ == '__main__':
    # Exécute les tests
    unittest.main()
