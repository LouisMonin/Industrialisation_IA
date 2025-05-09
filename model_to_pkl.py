import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ——————————————————————————————————————
# Étapes de prétraitement custom
# ——————————————————————————————————————

class MissingValueFiller(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=None, cat_cols=None):
        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
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

    def fit(self, X, y=None):
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

    def fit(self, X, y=None):
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

# ——————————————————————————————————————
# Colonnes issues de ton fichier config
# ——————————————————————————————————————

CATEGORIAL_COLUMNS = ["ACTIVIT2", "VOCATION", "CARACT1", "CARACT3", "CARACT4", "TYPBAT1", "INDEM2", "FRCH1", "FRCH2",
    "DEROG12", "DEROG13", "DEROG14", "DEROG16", "TAILLE1", "TAILLE2", "COEFASS", "RISK6", "RISK8",
    "RISK9", "RISK10", "RISK11", "RISK12", "RISK13", "EQUIPEMENT2", "EQUIPEMENT5", "DISTANCE_111",
    "DISTANCE_112", "DISTANCE_121", "DISTANCE_122", "DISTANCE_123", "DISTANCE_124", "DISTANCE_131",
    "DISTANCE_132", "DISTANCE_133", "DISTANCE_141", "DISTANCE_142", "DISTANCE_211", "DISTANCE_212",
    "DISTANCE_213", "DISTANCE_221", "DISTANCE_222", "DISTANCE_223", "DISTANCE_231", "DISTANCE_242",
    "DISTANCE_243", "DISTANCE_244", "DISTANCE_311", "DISTANCE_312", "DISTANCE_313", "DISTANCE_321",
    "DISTANCE_322", "DISTANCE_323", "DISTANCE_324", "DISTANCE_331", "DISTANCE_332", "DISTANCE_333",
    "DISTANCE_334", "DISTANCE_335", "DISTANCE_411", "DISTANCE_412", "DISTANCE_421", "DISTANCE_422",
    "DISTANCE_423", "DISTANCE_511", "DISTANCE_512", "DISTANCE_521", "DISTANCE_522", "DISTANCE_523",
    "PROPORTION_11", "PROPORTION_12", "PROPORTION_13", "PROPORTION_14", "PROPORTION_21", "PROPORTION_22",
    "PROPORTION_23", "PROPORTION_24", "PROPORTION_31", "PROPORTION_32", "PROPORTION_33", "PROPORTION_41",
    "PROPORTION_42", "PROPORTION_51", "PROPORTION_52", "MEN", "MEN_PAUV", "MEN_1IND", "MEN_5IND",
    "MEN_PROP", "MEN_FMP", "MEN_COLL", "MEN_MAIS", "LOG_AVA1", "LOG_A1_A2", "LOG_A2_A3", "LOG_APA3",
    "LOG_INC", "LOG_SOC", "IND", "IND_0_Y1", "IND_Y1_Y2", "IND_Y2_Y3", "IND_Y3_Y4", "IND_Y4_Y5",
    "IND_Y5_Y6", "IND_Y6_Y7", "IND_Y7_Y8", "IND_Y8_Y9", "IND_Y9", "IND_INC", "IND_SNV", "MEN_SURF",
    "DISTANCE_1", "DISTANCE_2", "ALTITUDE_1", "ALTITUDE_2", "ALTITUDE_3", "ALTITUDE_4", "ALTITUDE_5",
    "BDTOPO_BAT_MAX_HAUTEUR_MAX", "HAUTEUR", "HAUTEUR_MAX", "BDTOPO_BAT_MAX_HAUTEUR", "NB_CASERNES",
    "NBJTX25_MM_A", "NBJTX25_MMAX_A", "NBJTX25_MSOM_A", "NBJTX0_MM_A", "NBJTX0_MMAX_A", "NBJTX0_MSOM_A",
    "NBJTXI27_MM_A", "NBJTXI27_MMAX_A", "NBJTXI27_MSOM_A", "NBJTXS32_MM_A", "NBJTXS32_MMAX_A",
    "NBJTXS32_MSOM_A", "NBJTXI20_MM_A", "NBJTXI20_MMAX_A", "NBJTXI20_MSOM_A", "NBJTX30_MM_A",
    "NBJTX30_MMAX_A", "NBJTX30_MSOM_A", "NBJTX35_MM_A", "NBJTX35_MMAX_A", "NBJTX35_MSOM_A",
    "NBJTN10_MM_A", "NBJTN10_MMAX_A", "NBJTN10_MSOM_A", "NBJTNI10_MM_A", "NBJTNI10_MMAX_A",
    "NBJTNI10_MSOM_A", "NBJTN5_MM_A", "NBJTN5_MMAX_A", "NBJTN5_MSOM_A", "NBJTNS25_MM_A", "NBJTNS25_MMAX_A",
    "NBJTNS25_MSOM_A", "NBJTNI15_MM_A", "NBJTNI15_MMAX_A", "NBJTNI15_MSOM_A", "NBJTNI20_MM_A",
    "NBJTNI20_MMAX_A", "NBJTNI20_MSOM_A", "NBJTNS20_MM_A", "NBJTNS20_MMAX_A", "NBJTNS20_MSOM_A",
    "NBJTMS24_MM_A", "NBJTMS24_MMAX_A", "NBJTMS24_MSOM_A", "TAMPLIAB_VOR_MM_A", "TAMPLIAB_VOR_MMAX_A",
    "TAMPLIM_VOR_MM_A", "TAMPLIM_VOR_MMAX_A", "TM_VOR_MM_A", "TM_VOR_MMAX_A", "TMM_VOR_MM_A",
    "TMM_VOR_MMAX_A", "TMMAX_VOR_MM_A", "TMMAX_VOR_MMAX_A", "TMMIN_VOR_MM_A", "TMMIN_VOR_MMAX_A",
    "TN_VOR_MM_A", "TN_VOR_MMAX_A", "TNAB_VOR_MM_A", "TNAB_VOR_MMAX_A", "TNMAX_VOR_MM_A",
    "TNMAX_VOR_MMAX_A", "TX_VOR_MM_A", "TX_VOR_MMAX_A", "TXAB_VOR_MM_A", "TXAB_VOR_MMAX_A",
    "TXMIN_VOR_MM_A", "TXMIN_VOR_MMAX_A", "NBJFF10_MM_A", "NBJFF10_MMAX_A", "NBJFF10_MSOM_A",
    "NBJFF16_MM_A", "NBJFF16_MMAX_A", "NBJFF16_MSOM_A", "NBJFF28_MM_A", "NBJFF28_MMAX_A",
    "NBJFF28_MSOM_A", "NBJFXI3S10_MM_A", "NBJFXI3S10_MMAX_A", "NBJFXI3S10_MSOM_A", "NBJFXI3S16_MM_A",
    "NBJFXI3S16_MMAX_A", "NBJFXI3S16_MSOM_A", "NBJFXI3S28_MM_A", "NBJFXI3S28_MMAX_A",
    "NBJFXI3S28_MSOM_A", "NBJFXY8_MM_A", "NBJFXY8_MMAX_A", "NBJFXY8_MSOM_A", "NBJFXY10_MM_A",
    "NBJFXY10_MMAX_A", "NBJFXY10_MSOM_A", "NBJFXY15_MM_A", "NBJFXY15_MMAX_A", "NBJFXY15_MSOM_A",
    "FFM_VOR_MM_A", "FFM_VOR_MMAX_A", "FXI3SAB_VOR_MM_A", "FXI3SAB_VOR_MMAX_A", "FXIAB_VOR_MM_A",
    "FXIAB_VOR_MMAX_A", "FXYAB_VOR_MM_A", "FXYAB_VOR_MMAX_A", "FFM_VOR_COM_MM_A_Y", "FFM_VOR_COM_MMAX_A_Y",
    "FXI3SAB_VOR_COM_MM_A_Y", "FXI3SAB_VOR_COM_MMAX_A_Y", "NBJRR50_MM_A", "NBJRR50_MMAX_A",
    "NBJRR50_MSOM_A", "NBJRR1_MM_A", "NBJRR1_MMAX_A", "NBJRR1_MSOM_A", "NBJRR5_MM_A", "NBJRR5_MMAX_A",
    "NBJRR5_MSOM_A", "NBJRR10_MM_A", "NBJRR10_MMAX_A", "NBJRR10_MSOM_A", "NBJRR30_MM_A",
    "NBJRR30_MMAX_A", "NBJRR30_MSOM_A", "NBJRR100_MM_A", "NBJRR100_MMAX_A", "NBJRR100_MSOM_A",
    "RR_VOR_MM_A", "RR_VOR_MMAX_A", "RRAB_VOR_MM_A", "RRAB_VOR_MMAX_A", "ESPINSEE" ]  # Mets ici la liste complète

NUMERICAL_COLUMNS = [ "ID",
  "TYPERS",
  "ANCIENNETE",
  "DUREE_REQANEUF",
  "TYPBAT2",
  "KAPITAL12",
  "KAPITAL25",
  "KAPITAL32",
  "SURFACE1",
  "SURFACE4",
  "SURFACE10",
  "NBBAT1",
  "RISK1",
  "RISK7",
  "EQUIPEMENT4",
  "EQUIPEMENT6",
  "ZONE_VENT",
  "ANNEE_ASSURANCE",
  "AN_EXERC",
  "ZONE",
  "surface_totale",
  "capital_total",
  "surface_par_batiment",
  "capital_par_surface",
  "capital_moyen_par_batiment" ]   # Mets ici la liste complète

# ——————————————————————————————————————
# Chargement du modèle entraîné
# ——————————————————————————————————————

model = joblib.load("xgb_regressor_model.pkl")

# ——————————————————————————————————————
# Construction du pipeline complet
# ——————————————————————————————————————

pipeline = Pipeline(steps=[
    ("missing", MissingValueFiller(num_cols=NUMERICAL_COLUMNS, cat_cols=CATEGORIAL_COLUMNS)),
    ("encoding", ManualCountEncoder(cat_cols=CATEGORIAL_COLUMNS)),
    ("drop", ColumnDropper(num_cols=NUMERICAL_COLUMNS)),
    ("scaling", ScalerWrapper(num_cols=NUMERICAL_COLUMNS)),
    ("model", model)
])

# ——————————————————————————————————————
# Sauvegarde finale du pipeline
# ——————————————————————————————————————

joblib.dump(pipeline, "full_model_pipeline.pkl")
print("✅ Pipeline complet sauvegardé sous full_model_pipeline.pkl")
