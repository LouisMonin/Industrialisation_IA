# Importer les bibliothèques nécessaires
import pandas as pd
pd.set_option('display.max_columns', 375)
import numpy as np
from category_encoders import CountEncoder
from sklearn.metrics import mean_squared_error
from IPython.display import display


# Charger les données
print("Chargement des données...")
X_train = pd.read_csv("/Users/sabine/Desktop/CYTech/S3/Data_science/ProjetFinal/03.Données/train_input.csv", sep=',')
y_train = pd.read_csv("/Users/sabine/Desktop/CYTech/S3/Data_science/ProjetFinal/03.Données/train_output.csv", sep=',')  # Convertir en Series si nécessaire

# On affiche le nombre de lignes et de colonnes
print("Nombre de lignes et de colonnes dans X_train :", X_train.shape)
print("Nombre de lignes et de colonnes dans y_train :", y_train.shape)

# Afficher un aperçu des données pour vérifier leur chargement
print("Aperçu de X_train :")
display(X_train.head())
print("Aperçu de y_train :")
display(y_train.head())

# Avant cela, on organise les données en ordre croissant selon la colonne 'id'
X_train = X_train.sort_values(by='ID')
y_train = y_train.sort_values(by='ID')
# On réinitialise les index
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# Répartition entre données d'entraînement et de validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Afficher le nombre de lignes et de colonnes après la séparation
print("Nombre de lignes et de colonnes après séparation :")
print("X_train :", X_train.shape)
print("X_val :", X_val.shape)
print("y_train :", y_train.shape)
print("y_val :", y_val.shape)

# Afficher un aperçu des données d'entraînement et de validation
print("Aperçu de X_train :")
display(X_train.head())
print("Aperçu de X_val :")
display(X_val.head())
print("Aperçu de y_train :")
display(y_train.head())
print("Aperçu de y_val :")
display(y_val.head())

# Sélectionner uniquement les colonnes ordinales
ordinal_columns = ['NB_CASERNES', 'BDTOPO_BAT_MAX_HAUTEUR', 'HAUTEUR_MAX', 'HAUTEUR', 'BDTOPO_BAT_MAX_HAUTEUR_MAX', 'MEN_SURF', 'IND_SNV', 'IND_INC', 'IND_Y9', 'IND_0_Y1', 'IND', 'LOG_SOC', 'LOG_INC', 'LOG_APA3', 'LOG_AVA1', 'MEN_MAIS', 'MEN_COLL', 'MEN_FMP', 'MEN_PROP', 'MEN_PAUV', 'MEN', 'COEFASS', 'DISTANCE_111', 'DISTANCE_112', 'DISTANCE_121', 'DISTANCE_122', 'DISTANCE_123', 'DISTANCE_124', 'DISTANCE_131', 'DISTANCE_132', 'DISTANCE_133', 'DISTANCE_141', 'DISTANCE_142', 'DISTANCE_211', 'DISTANCE_212', 'DISTANCE_213', 'DISTANCE_221', 'DISTANCE_222', 'DISTANCE_223', 'DISTANCE_231', 'DISTANCE_242', 'DISTANCE_243', 'DISTANCE_244', 'DISTANCE_311', 'DISTANCE_312', 'DISTANCE_313', 'DISTANCE_321', 'DISTANCE_322', 'DISTANCE_323', 'DISTANCE_324', 'DISTANCE_331', 'DISTANCE_332', 'DISTANCE_333', 'DISTANCE_334', 'DISTANCE_335', 'DISTANCE_411', 'DISTANCE_412', 'DISTANCE_421', 'DISTANCE_422', 'DISTANCE_423', 'DISTANCE_511', 'DISTANCE_512', 'DISTANCE_521', 'DISTANCE_522', 'DISTANCE_523', 'PROPORTION_11', 'PROPORTION_12', 'PROPORTION_13', 'PROPORTION_14', 'PROPORTION_21', 'PROPORTION_22', 'PROPORTION_23', 'PROPORTION_24', 'PROPORTION_31', 'PROPORTION_32', 'PROPORTION_33', 'PROPORTION_41', 'PROPORTION_42', 'PROPORTION_51', 'PROPORTION_52', 'MEN_1IND', 'MEN_5IND', 'LOG_A1_A2', 'LOG_A2_A3', 'IND_Y1_Y2', 'IND_Y2_Y3', 'IND_Y3_Y4', 'IND_Y4_Y5', 'IND_Y5_Y6', 'IND_Y6_Y7', 'IND_Y7_Y8', 'IND_Y8_Y9', 'DISTANCE_1', 'DISTANCE_2', 'ALTITUDE_1', 'ALTITUDE_2', 'ALTITUDE_3', 'ALTITUDE_4', 'ALTITUDE_5', 'NBJTX25_MM_A', 'NBJTX25_MMAX_A', 'NBJTX25_MSOM_A', 'NBJTX0_MM_A', 'NBJTX0_MMAX_A', 'NBJTX0_MSOM_A', 'NBJTXI27_MM_A', 'NBJTXI27_MMAX_A', 'NBJTXI27_MSOM_A', 'NBJTXS32_MM_A', 'NBJTXS32_MMAX_A', 'NBJTXS32_MSOM_A', 'NBJTXI20_MM_A', 'NBJTXI20_MMAX_A', 'NBJTXI20_MSOM_A', 'NBJTX30_MM_A', 'NBJTX30_MMAX_A', 'NBJTX30_MSOM_A', 'NBJTX35_MM_A', 'NBJTX35_MMAX_A', 'NBJTX35_MSOM_A', 'NBJTN10_MM_A', 'NBJTN10_MMAX_A', 'NBJTN10_MSOM_A', 'NBJTNI10_MM_A', 'NBJTNI10_MMAX_A', 'NBJTNI10_MSOM_A', 'NBJTN5_MM_A', 'NBJTN5_MMAX_A', 'NBJTN5_MSOM_A', 'NBJTNS25_MM_A', 'NBJTNS25_MMAX_A', 'NBJTNS25_MSOM_A', 'NBJTNI15_MM_A', 'NBJTNI15_MMAX_A', 'NBJTNI15_MSOM_A', 'NBJTNI20_MM_A', 'NBJTNI20_MMAX_A', 'NBJTNI20_MSOM_A', 'NBJTNS20_MM_A', 'NBJTNS20_MMAX_A', 'NBJTNS20_MSOM_A', 'NBJTMS24_MM_A', 'NBJTMS24_MMAX_A', 'NBJTMS24_MSOM_A', 'TAMPLIAB_VOR_MM_A', 'TAMPLIAB_VOR_MMAX_A', 'TAMPLIM_VOR_MM_A', 'TAMPLIM_VOR_MMAX_A', 'TM_VOR_MM_A', 'TM_VOR_MMAX_A', 'TMM_VOR_MM_A', 'TMM_VOR_MMAX_A', 'TMMAX_VOR_MM_A', 'TMMAX_VOR_MMAX_A', 'TMMIN_VOR_MM_A', 'TMMIN_VOR_MMAX_A', 'TN_VOR_MM_A', 'TN_VOR_MMAX_A', 'TNAB_VOR_MM_A', 'TNAB_VOR_MMAX_A', 'TNMAX_VOR_MM_A', 'TNMAX_VOR_MMAX_A', 'TX_VOR_MM_A', 'TX_VOR_MMAX_A', 'TXAB_VOR_MM_A', 'TXAB_VOR_MMAX_A', 'TXMIN_VOR_MM_A', 'TXMIN_VOR_MMAX_A', 'NBJFF10_MM_A', 'NBJFF10_MMAX_A', 'NBJFF10_MSOM_A', 'NBJFF16_MM_A', 'NBJFF16_MMAX_A', 'NBJFF16_MSOM_A', 'NBJFF28_MM_A', 'NBJFF28_MMAX_A', 'NBJFF28_MSOM_A', 'NBJFXI3S10_MM_A', 'NBJFXI3S10_MMAX_A', 'NBJFXI3S10_MSOM_A', 'NBJFXI3S16_MM_A', 'NBJFXI3S16_MMAX_A', 'NBJFXI3S16_MSOM_A', 'NBJFXI3S28_MM_A', 'NBJFXI3S28_MMAX_A', 'NBJFXI3S28_MSOM_A', 'NBJFXY8_MM_A', 'NBJFXY8_MMAX_A', 'NBJFXY8_MSOM_A', 'NBJFXY10_MM_A', 'NBJFXY10_MMAX_A', 'NBJFXY10_MSOM_A', 'NBJFXY15_MM_A', 'NBJFXY15_MMAX_A', 'NBJFXY15_MSOM_A', 'FFM_VOR_MM_A', 'FFM_VOR_MMAX_A', 'FXI3SAB_VOR_MM_A', 'FXI3SAB_VOR_MMAX_A', 'FXIAB_VOR_MM_A', 'FXIAB_VOR_MMAX_A', 'FXYAB_VOR_MM_A', 'FXYAB_VOR_MMAX_A', 'FFM_VOR_COM_MM_A_Y', 'FFM_VOR_COM_MMAX_A_Y', 'FXI3SAB_VOR_COM_MM_A_Y', 'FXI3SAB_VOR_COM_MMAX_A_Y', 'NBJRR50_MM_A', 'NBJRR50_MMAX_A', 'NBJRR50_MSOM_A', 'NBJRR1_MM_A', 'NBJRR1_MMAX_A', 'NBJRR1_MSOM_A', 'NBJRR5_MM_A', 'NBJRR5_MMAX_A', 'NBJRR5_MSOM_A', 'NBJRR10_MM_A', 'NBJRR10_MMAX_A', 'NBJRR10_MSOM_A', 'NBJRR30_MM_A', 'NBJRR30_MMAX_A', 'NBJRR30_MSOM_A', 'NBJRR100_MM_A', 'NBJRR100_MMAX_A', 'NBJRR100_MSOM_A', 'RR_VOR_MM_A', 'RR_VOR_MMAX_A', 'RRAB_VOR_MM_A', 'RRAB_VOR_MMAX_A', 'TAILLE1', 'TAILLE2', 'CARACT4', 'SURFACE4', 'SURFACE6']


merged = pd.merge(X_train, y_train, left_index=True, right_index=True)

# Affichage avant encodage
print("Aperçu de X_train avant encodage :")
display(X_train[ordinal_columns].head())
print("Aperçu de X_val avant encodage :")
display(X_val[ordinal_columns].head())

# Pour chaque colonne ordinale
for col in ordinal_columns :
    
    # Calculer la fréquence des sinistres pour chaque valeur de la colonne ordinale
    freq_sinistre = merged.groupby(col)['FREQ'].sum()
    
    # Trier les valeurs de la colonne ordinale par la fréquence des sinistres
    sorted_values = freq_sinistre.sort_values().index
    
    # Créer un dictionnaire d'encodage basé sur l'ordre des valeurs triées
    encoding_dict = {value: idx for idx, value in enumerate(sorted_values)}
    
    # Appliquer l'encodage à la colonne dans X_train et X_val 
    X_train[col] = X_train[col].map(encoding_dict)
    X_val[col] = X_val[col].map(encoding_dict)

# Afficher les résultats pour vérifier en mettant en avant les colonnes ordinales
print("Aperçu de X_train avant encodage des ordinales :")
display(X_train[ordinal_columns].head())
print("Aperçu de X_val avant encodage des ordinales :")
display(X_val[ordinal_columns].head())


# Liste des colonnes catégorielles
categorical_columns = ['ACTIVIT2', 'VOCATION', 'ADOSS', 'CARACT1', 'CARACT3', 'INDEM1', 'TYPBAT1', 'INDEM2', 'FRCH1', 'FRCH2', 'DEROG2', 'DEROG3', 'DEROG4', 'DEROG5', 'DEROG8', 'DEROG12', 'KAPITAL34', 'KAPITAL35', 'KAPITAL37', 'KAPITAL40', 'KAPITAL41', 'KAPITAL42', 'KAPITAL43', 'RISK6', 'RISK8', 'RISK9', 'RISK10', 'RISK11', 'RISK12', 'RISK13', 'EQUIPEMENT2', 'EQUIPEMENT5', 'ESPINSEE', 'AN_EXERC', 'ZONE', 'TYPERS']

# Affichage des colonnes catégorielles avant encodage
print("Aperçu de X_train avant encodage des catégorielles :")
display(X_train[categorical_columns].head())
print("Aperçu de X_val avant encodage des catégorielles :")
display(X_val[categorical_columns].head())

# Pour chaque colonne catégorielle
for col in categorical_columns:
    
    # Calculer la fréquence des sinistres pour chaque valeur de la colonne
    freq_sinistre = merged.groupby(col)['FREQ'].sum()
    
    # Trier les valeurs de la colonne par fréquence des sinistres
    sorted_values = freq_sinistre.sort_values().index
    
    # Créer un dictionnaire d'encodage basé sur l'ordre des valeurs triées
    encoding_dict = {value: idx for idx, value in enumerate(sorted_values)}
    
    # Appliquer l'encodage à la colonne dans X_train et X_val
    X_train[col] = X_train[col].map(encoding_dict)
    X_val[col] = X_val[col].map(encoding_dict)

# Afficher les résultats pour vérifier en mettant en avant les colonnes catégorielles
print("Aperçu de X_train après encodage des catégorielles :")
display(X_train[categorical_columns].head())
print("Aperçu de X_val après encodage des catégorielles :")
display(X_val[categorical_columns].head())


# Liste des colonnes numériques
numeric_columns = ['DEROG13', 'DEROG14', 'DEROG16','ANCIENNETE', 'CARACT2', 'DUREE_REQANEUF', 'CARACT5', 'TYPBAT2', 'DEROG1', 'DEROG6', 'DEROG7', 'DEROG9', 'DEROG10', 'DEROG11', 'DEROG15', 'CA1', 'CA2', 'CA3', 'KAPITAL1', 'KAPITAL2', 'KAPITAL3', 'KAPITAL4', 'KAPITAL5', 'KAPITAL6', 'KAPITAL7', 'KAPITAL8', 'KAPITAL9', 'KAPITAL10', 'KAPITAL11', 'KAPITAL12', 'KAPITAL13', 'KAPITAL14', 'KAPITAL15', 'KAPITAL16', 'KAPITAL17', 'KAPITAL18', 'KAPITAL19', 'KAPITAL20', 'KAPITAL21', 'KAPITAL22', 'KAPITAL23', 'KAPITAL24', 'KAPITAL25', 'KAPITAL26', 'KAPITAL27', 'KAPITAL28', 'KAPITAL29', 'KAPITAL30', 'KAPITAL31', 'KAPITAL32', 'KAPITAL33', 'KAPITAL36', 'KAPITAL38', 'KAPITAL39', 'SURFACE1', 'SURFACE2', 'SURFACE3', 'SURFACE5', 'SURFACE7', 'SURFACE8', 'SURFACE9', 'SURFACE10', 'SURFACE11', 'SURFACE12', 'SURFACE13', 'SURFACE14', 'SURFACE15', 'SURFACE16', 'SURFACE17', 'SURFACE18', 'SURFACE19', 'SURFACE20', 'SURFACE21', 'NBBAT1', 'NBBAT2', 'NBBAT3', 'NBBAT4', 'NBBAT5', 'NBBAT6', 'NBBAT7', 'NBBAT8', 'NBBAT9', 'NBBAT10', 'NBBAT11', 'NBBAT13', 'NBBAT14', 'TAILLE3', 'TAILLE4', 'NBSINCONJ', 'NBSINSTRT', 'RISK1', 'RISK2', 'RISK3', 'RISK4', 'RISK5', 'RISK7', 'EQUIPEMENT1', 'EQUIPEMENT3', 'EQUIPEMENT4', 'EQUIPEMENT6', 'EQUIPEMENT7', 'ZONE_VENT']

# Convertir les colonnes en numérique 
X_train[numeric_columns] = X_train[numeric_columns].apply(pd.to_numeric, errors='coerce')
X_val[numeric_columns] = X_val[numeric_columns].apply(pd.to_numeric, errors='coerce')


# On créé X_train_filtered et y_train_filtered qui contiennent les lignes où FREQ est différent de 0

Id_freq_diff_0 = y_train[y_train['FREQ'] != 0]['ID']
X_train_filtered = X_train[X_train['ID'].isin(Id_freq_diff_0)]
y_train_filtered = y_train[y_train['FREQ'] != 0]

# On affiche le nombre de lignes et de colonnes
print("Nombre de lignes et de colonnes dans X_train_filtered :", X_train_filtered.shape)
print("Nombre de lignes et de colonnes dans y_train_filtered :", y_train_filtered.shape)

# On affiche un aperçu des données pour vérifier leur chargement
print("Aperçu de X_train_filtered :")
display(X_train_filtered.head())
print("Aperçu de y_train_filtered :")
display(y_train_filtered.head())

# On enlève certaines colonnes qui ne sont pas nécessaires dans X_train_filtered et X_val

columns_delete = [
    "NBJTX25_MM_A", "NBJTX25_MMAX_A", "NBJTX25_MSOM_A", "NBJTX0_MM_A", "NBJTX0_MMAX_A", "NBJTX0_MSOM_A",
    "NBJTXI27_MM_A", "NBJTXI27_MMAX_A", "NBJTXI27_MSOM_A", "NBJTXS32_MM_A", "NBJTXS32_MMAX_A", "NBJTXS32_MSOM_A",
    "NBJTXI20_MM_A", "NBJTXI20_MMAX_A", "NBJTXI20_MSOM_A", "NBJTX30_MM_A", "NBJTX30_MMAX_A", "NBJTX30_MSOM_A",
    "NBJTX35_MM_A", "NBJTX35_MMAX_A", "NBJTX35_MSOM_A", "NBJTN10_MM_A", "NBJTN10_MMAX_A", "NBJTN10_MSOM_A",
    "NBJTNI10_MM_A", "NBJTNI10_MMAX_A", "NBJTNI10_MSOM_A", "NBJTN5_MM_A", "NBJTN5_MMAX_A", "NBJTN5_MSOM_A",
    "NBJTNS25_MM_A", "NBJTNS25_MMAX_A", "NBJTNS25_MSOM_A", "NBJTNI15_MM_A", "NBJTNI15_MMAX_A", "NBJTNI15_MSOM_A",
    "NBJTNI20_MM_A", "NBJTNI20_MMAX_A", "NBJTNI20_MSOM_A", "NBJTNS20_MM_A", "NBJTNS20_MMAX_A", "NBJTNS20_MSOM_A",
    "NBJTMS24_MM_A", "NBJTMS24_MMAX_A", "NBJTMS24_MSOM_A", "TAMPLIAB_VOR_MM_A", "TAMPLIAB_VOR_MMAX_A",
    "TAMPLIM_VOR_MM_A", "TAMPLIM_VOR_MMAX_A", "TM_VOR_MM_A", "TM_VOR_MMAX_A", "TMM_VOR_MM_A", "TMM_VOR_MMAX_A",
    "TMMAX_VOR_MM_A", "TMMAX_VOR_MMAX_A", "TMMIN_VOR_MM_A", "TMMIN_VOR_MMAX_A", "TN_VOR_MM_A", "TN_VOR_MMAX_A",
    "TNAB_VOR_MM_A", "TNAB_VOR_MMAX_A", "TNMAX_VOR_MM_A", "TNMAX_VOR_MMAX_A", "TX_VOR_MM_A", "TX_VOR_MMAX_A",
    "TXAB_VOR_MM_A", "TXAB_VOR_MMAX_A", "TXMIN_VOR_MM_A", "TXMIN_VOR_MMAX_A", "NBJFF10_MM_A", "NBJFF10_MMAX_A",
    "NBJFF10_MSOM_A", "NBJFF16_MM_A", "NBJFF16_MMAX_A", "NBJFF16_MSOM_A", "NBJFF28_MM_A", "NBJFF28_MMAX_A",
    "NBJFF28_MSOM_A", "NBJFXI3S10_MMAX_A", "NBJFXI3S10_MSOM_A", "NBJFXI3S16_MM_A", "NBJFXI3S16_MMAX_A",
    "NBJFXI3S16_MSOM_A", "NBJFXI3S28_MM_A", "NBJFXI3S28_MMAX_A", "NBJFXI3S28_MSOM_A", "NBJFXY8_MM_A",
    "NBJFXY8_MMAX_A", "NBJFXY8_MSOM_A", "NBJFXY10_MM_A", "NBJFXY10_MMAX_A", "NBJFXY10_MSOM_A", "NBJFXY15_MM_A",
    "NBJFXY15_MMAX_A", "NBJFXY15_MSOM_A", "FFM_VOR_MM_A", "FFM_VOR_MMAX_A", "FXI3SAB_VOR_MM_A", "FXI3SAB_VOR_MMAX_A",
    "FXIAB_VOR_MM_A", "FXIAB_VOR_MMAX_A", "FXYAB_VOR_MM_A", "FXYAB_VOR_MMAX_A", "FFM_vor_com_MM_A_y",
    "FFM_vor_com_MMAX_A_y", "FXI3SAB_vor_com_MM_A_y", "FXI3SAB_vor_com_MMAX_A_y", "NBJRR50_MM_A", "NBJRR50_MMAX_A",
    "NBJRR50_MSOM_A", "NBJRR1_MM_A", "NBJRR1_MMAX_A", "NBJRR1_MSOM_A", "NBJRR5_MM_A", "NBJRR5_MMAX_A",
    "NBJRR5_MSOM_A", "NBJRR10_MM_A", "NBJRR10_MMAX_A", "NBJRR10_MSOM_A", "NBJRR30_MM_A", "NBJRR30_MMAX_A",
    "NBJRR30_MSOM_A", "NBJRR100_MM_A", "NBJRR100_MMAX_A", "NBJRR100_MSOM_A", "RR_VOR_MM_A", "RR_VOR_MMAX_A",
    "RRAB_VOR_MM_A", "RRAB_VOR_MMAX_A"
]

columns_delete2 = [
    "distance_111", "distance_112", "distance_121", "distance_122", "distance_123", "distance_124",
    "distance_131", "distance_132", "distance_133", "distance_141", "distance_142", "distance_211",
    "distance_212", "distance_213", "distance_221", "distance_222", "distance_223", "distance_231",
    "distance_242", "distance_243", "distance_244", "distance_311", "distance_312", "distance_313",
    "distance_321", "distance_322", "distance_323", "distance_324", "distance_331", "distance_332",
    "distance_333", "distance_334", "distance_335", "distance_411", "distance_412", "distance_421",
    "distance_422", "distance_423", "distance_511", "distance_512", "distance_521", "distance_522",
    "distance_523", "proportion_11", "proportion_12", "proportion_13", "proportion_14", "Proportion_21",
    "proportion_22", "proportion_23", "proportion_24", "proportion_31", "proportion_32", "proportion_33",
    "proportion_41", "proportion_42", "proportion_51", "proportion_52"
]

# Convertir en majuscules
columns_delete2 = [col.upper() for col in columns_delete2]



# Filtrer les colonnes à supprimer qui existent réellement
columns_to_delete_in_train = [col for col in columns_delete if col in X_train_filtered.columns]
columns_to_delete_in_val = [col for col in columns_delete if col in X_val.columns]

columns_to_delete2_in_train = [col for col in columns_delete2 if col in X_train_filtered.columns]
columns_to_delete2_in_val = [col for col in columns_delete2 if col in X_val.columns]

# Afficher le nombre de colonnes avant la suppression
print("Nombre de colonnes dans X_train_filtered avant suppression :", X_train_filtered.shape[1])
print("Nombre de colonnes dans X_val avant suppression :", X_val.shape[1])

# Supprimer les colonnes existantes
X_train_filtered = X_train_filtered.drop(columns=columns_to_delete_in_train + columns_to_delete2_in_train, errors='ignore')
X_val = X_val.drop(columns=columns_to_delete_in_val + columns_to_delete2_in_val, errors='ignore')

# Afficher le nombre de colonnes restantes après la suppression
print("Nombre de colonnes restantes dans X_train_filtered :", X_train_filtered.shape[1])
print("Nombre de colonnes restantes dans X_val :", X_val.shape[1])
