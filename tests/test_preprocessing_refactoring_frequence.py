import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

from app.preprocessing_refactoring_frequence import (
    ColumnSelector,
    MissingValueFiller,
    ManualCountEncoder
)

class TestPreprocessingFrequence(unittest.TestCase):
    """
    Cette classe contient des tests unitaires pour les transformateurs de prétraitement de fréquences.
    """

    def setUp(self):
        self.x_data = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'category': ['cat1', 'cat2', 'cat1', 'cat3'],
            'amount': [100, 200, None, 400],
            'value': [10, 20, 10, None],
            'other': [1, 1, 1, 1]
        })

        self.x_data_large_missing = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'category': ['cat1', 'cat2', 'cat1', None],
            'amount': [100, 200, None, 400],
            'value': [None, 20, 10, None],
            'other': [1, 1, 1, 1]
        })

    def test_column_selector(self):
        """
        Teste la sélection des colonnes dans le DataFrame.
        """
        selector = ColumnSelector(selected_columns=['ID', 'category', 'amount'])
        result = selector.transform(self.x_data)

        expected = self.x_data[['ID', 'category', 'amount']]
        assert_frame_equal(result, expected)

    def test_missing_value_filler(self):
        """
        Teste le remplissage des valeurs manquantes pour les colonnes numériques et catégorielles.
        """
        filler = MissingValueFiller(num_cols=['amount', 'value'], cat_cols=['category'])
        result = filler.transform(self.x_data_large_missing)

        # Convertir 'amount' et 'value' à int64 après remplissage
        result['amount'] = result['amount'].astype('int64')
        result['value'] = result['value'].astype('int64')  # Ajout de cette ligne pour convertir 'value'

        expected = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'category': ['cat1', 'cat2', 'cat1', 'Inconnu'],
            'amount': [100, 200, 0, 400],
            'value': [0, 20, 10, 0],
            'other': [1, 1, 1, 1]
        })

        assert_frame_equal(result, expected)

    def test_manual_count_encoder(self):
        """
        Teste l'encodage des colonnes catégorielles par la fréquence des valeurs.
        """
        encoder = ManualCountEncoder(cat_cols=['category'])
        encoder.fit(self.x_data)
        result = encoder.transform(self.x_data)

        expected = pd.DataFrame({
            'ID': [1, 2, 3, 4],
            'category': [2, 1, 2, 1],
            'amount': [100, 200, None, 400],
            'value': [10, 20, 10, None],
            'other': [1, 1, 1, 1]
        })

        assert_frame_equal(result, expected)

if __name__ == '__main__':
    # Exécute les tests
    unittest.main()
