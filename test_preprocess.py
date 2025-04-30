import unittest
import pandas as pd
from pandas.testing import assert_series_equal
import Preprocess

# Surcharge les colonnes utilisées pour les tests
Preprocess.ORDINAL_COLUMNS = ['grade']
Preprocess.CATEGORICAL_COLUMNS = ['type']
Preprocess.NUMERIC_COLUMNS = ['amount']


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.X = pd.DataFrame({
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
        result = Preprocess.encode_ordinal_by_target_frequency(self.X.copy(), self.y)
        # FREQ par grade : A=10, B=5, C=1 ⇒ trié : C (0), B (1), A (2)
        expected = [1, 2, 0]  # B, A, C
        self.assertListEqual(result['grade'].tolist(), expected)

    def test_encode_categorical_by_target_frequency(self):
        result = Preprocess.encode_categorical_by_target_frequency(self.X.copy(), self.y)
        # FREQ par type : cat1 = 5 + 1 = 6, cat2 = 10 ⇒ trié : cat1 (0), cat2 (1)
        expected = [0, 1, 0]
        self.assertListEqual(result['type'].tolist(), expected)

    def test_convert_to_numeric(self):
        result = Preprocess.convert_to_numeric(self.X.copy())
        expected = pd.Series([10.0, float('nan'), 30.0], name='amount', dtype='float64')
        assert_series_equal(result['amount'], expected, check_dtype=False)


if __name__ == '__main__':
    unittest.main()
