# preprocessing_pipeline_fréquence.py

import joblib
from sklearn.pipeline import Pipeline
from preprocessing_refactoring_frequence import ColumnSelector, MissingValueFiller, ManualCountEncoder, ColumnDropper, ScalerWrapper
from config_frequence import NUMERICAL_COLUMNS, CATEGORIAL_COLUMNS

ALL_COLUMNS = NUMERICAL_COLUMNS + CATEGORIAL_COLUMNS

# Chargement du modèle entraîné
model = joblib.load("model_frequence.pkl")

# Construction du pipeline complet
pipeline = Pipeline(steps=[
    ("select", ColumnSelector(selected_columns=ALL_COLUMNS)),
    ("missing", MissingValueFiller(num_cols=NUMERICAL_COLUMNS, cat_cols=CATEGORIAL_COLUMNS)),
    ("encoding", ManualCountEncoder(cat_cols=CATEGORIAL_COLUMNS)),
    ("drop", ColumnDropper(num_cols=NUMERICAL_COLUMNS)),
    ("scaling", ScalerWrapper(num_cols=NUMERICAL_COLUMNS)),
    ("model", model)
])

# Sauvegarde finale du pipeline
joblib.dump(pipeline, "preprocessing_frequence.pkl")
print("✅ Pipeline complet sauvegardé sous preprocessing_frequence.pkl")
