# Importer les bibliothèques nécessaires
import pandas as pd



# Encode les colonnes ordinales en fonction de la fréquence de la variable cible  
def encode_ordinal_by_target_frequency(X, y):

    # X : pd.DataFrame, DataFrame contenant les variables explicatives
    # y : pd.DataFrame, DataFrame contenant la variable cible

    # Sortie :  pd.DataFrame, DataFrame X avec les colonnes ordinales encodées
    
    target_name = 'FREQ'

    ordinal_columns = ['NB_CASERNES', 'BDTOPO_BAT_MAX_HAUTEUR', 'HAUTEUR_MAX', 'HAUTEUR', 'BDTOPO_BAT_MAX_HAUTEUR_MAX', 'MEN_SURF', 'IND_SNV', 'IND_INC', 'IND_Y9', 'IND_0_Y1', 'IND', 'LOG_SOC', 'LOG_INC', 'LOG_APA3', 'LOG_AVA1', 'MEN_MAIS', 'MEN_COLL', 'MEN_FMP', 'MEN_PROP', 'MEN_PAUV', 'MEN', 'COEFASS', 'DISTANCE_111', 'DISTANCE_112', 'DISTANCE_121', 'DISTANCE_122', 'DISTANCE_123', 'DISTANCE_124', 'DISTANCE_131', 'DISTANCE_132', 'DISTANCE_133', 'DISTANCE_141', 'DISTANCE_142', 'DISTANCE_211', 'DISTANCE_212', 'DISTANCE_213', 'DISTANCE_221', 'DISTANCE_222', 'DISTANCE_223', 'DISTANCE_231', 'DISTANCE_242', 'DISTANCE_243', 'DISTANCE_244', 'DISTANCE_311', 'DISTANCE_312', 'DISTANCE_313', 'DISTANCE_321', 'DISTANCE_322', 'DISTANCE_323', 'DISTANCE_324', 'DISTANCE_331', 'DISTANCE_332', 'DISTANCE_333', 'DISTANCE_334', 'DISTANCE_335', 'DISTANCE_411', 'DISTANCE_412', 'DISTANCE_421', 'DISTANCE_422', 'DISTANCE_423', 'DISTANCE_511', 'DISTANCE_512', 'DISTANCE_521', 'DISTANCE_522', 'DISTANCE_523', 'PROPORTION_11', 'PROPORTION_12', 'PROPORTION_13', 'PROPORTION_14', 'PROPORTION_21', 'PROPORTION_22', 'PROPORTION_23', 'PROPORTION_24', 'PROPORTION_31', 'PROPORTION_32', 'PROPORTION_33', 'PROPORTION_41', 'PROPORTION_42', 'PROPORTION_51', 'PROPORTION_52', 'MEN_1IND', 'MEN_5IND', 'LOG_A1_A2', 'LOG_A2_A3', 'IND_Y1_Y2', 'IND_Y2_Y3', 'IND_Y3_Y4', 'IND_Y4_Y5', 'IND_Y5_Y6', 'IND_Y6_Y7', 'IND_Y7_Y8', 'IND_Y8_Y9', 'DISTANCE_1', 'DISTANCE_2', 'ALTITUDE_1', 'ALTITUDE_2', 'ALTITUDE_3', 'ALTITUDE_4', 'ALTITUDE_5', 'NBJTX25_MM_A', 'NBJTX25_MMAX_A', 'NBJTX25_MSOM_A', 'NBJTX0_MM_A', 'NBJTX0_MMAX_A', 'NBJTX0_MSOM_A', 'NBJTXI27_MM_A', 'NBJTXI27_MMAX_A', 'NBJTXI27_MSOM_A', 'NBJTXS32_MM_A', 'NBJTXS32_MMAX_A', 'NBJTXS32_MSOM_A', 'NBJTXI20_MM_A', 'NBJTXI20_MMAX_A', 'NBJTXI20_MSOM_A', 'NBJTX30_MM_A', 'NBJTX30_MMAX_A', 'NBJTX30_MSOM_A', 'NBJTX35_MM_A', 'NBJTX35_MMAX_A', 'NBJTX35_MSOM_A', 'NBJTN10_MM_A', 'NBJTN10_MMAX_A', 'NBJTN10_MSOM_A', 'NBJTNI10_MM_A', 'NBJTNI10_MMAX_A', 'NBJTNI10_MSOM_A', 'NBJTN5_MM_A', 'NBJTN5_MMAX_A', 'NBJTN5_MSOM_A', 'NBJTNS25_MM_A', 'NBJTNS25_MMAX_A', 'NBJTNS25_MSOM_A', 'NBJTNI15_MM_A', 'NBJTNI15_MMAX_A', 'NBJTNI15_MSOM_A', 'NBJTNI20_MM_A', 'NBJTNI20_MMAX_A', 'NBJTNI20_MSOM_A', 'NBJTNS20_MM_A', 'NBJTNS20_MMAX_A', 'NBJTNS20_MSOM_A', 'NBJTMS24_MM_A', 'NBJTMS24_MMAX_A', 'NBJTMS24_MSOM_A', 'TAMPLIAB_VOR_MM_A', 'TAMPLIAB_VOR_MMAX_A', 'TAMPLIM_VOR_MM_A', 'TAMPLIM_VOR_MMAX_A', 'TM_VOR_MM_A', 'TM_VOR_MMAX_A', 'TMM_VOR_MM_A', 'TMM_VOR_MMAX_A', 'TMMAX_VOR_MM_A', 'TMMAX_VOR_MMAX_A', 'TMMIN_VOR_MM_A', 'TMMIN_VOR_MMAX_A', 'TN_VOR_MM_A', 'TN_VOR_MMAX_A', 'TNAB_VOR_MM_A', 'TNAB_VOR_MMAX_A', 'TNMAX_VOR_MM_A', 'TNMAX_VOR_MMAX_A', 'TX_VOR_MM_A', 'TX_VOR_MMAX_A', 'TXAB_VOR_MM_A', 'TXAB_VOR_MMAX_A', 'TXMIN_VOR_MM_A', 'TXMIN_VOR_MMAX_A', 'NBJFF10_MM_A', 'NBJFF10_MMAX_A', 'NBJFF10_MSOM_A', 'NBJFF16_MM_A', 'NBJFF16_MMAX_A', 'NBJFF16_MSOM_A', 'NBJFF28_MM_A', 'NBJFF28_MMAX_A', 'NBJFF28_MSOM_A', 'NBJFXI3S10_MM_A', 'NBJFXI3S10_MMAX_A', 'NBJFXI3S10_MSOM_A', 'NBJFXI3S16_MM_A', 'NBJFXI3S16_MMAX_A', 'NBJFXI3S16_MSOM_A', 'NBJFXI3S28_MM_A', 'NBJFXI3S28_MMAX_A', 'NBJFXI3S28_MSOM_A', 'NBJFXY8_MM_A', 'NBJFXY8_MMAX_A', 'NBJFXY8_MSOM_A', 'NBJFXY10_MM_A', 'NBJFXY10_MMAX_A', 'NBJFXY10_MSOM_A', 'NBJFXY15_MM_A', 'NBJFXY15_MMAX_A', 'NBJFXY15_MSOM_A', 'FFM_VOR_MM_A', 'FFM_VOR_MMAX_A', 'FXI3SAB_VOR_MM_A', 'FXI3SAB_VOR_MMAX_A', 'FXIAB_VOR_MM_A', 'FXIAB_VOR_MMAX_A', 'FXYAB_VOR_MM_A', 'FXYAB_VOR_MMAX_A', 'FFM_VOR_COM_MM_A_Y', 'FFM_VOR_COM_MMAX_A_Y', 'FXI3SAB_VOR_COM_MM_A_Y', 'FXI3SAB_VOR_COM_MMAX_A_Y', 'NBJRR50_MM_A', 'NBJRR50_MMAX_A', 'NBJRR50_MSOM_A', 'NBJRR1_MM_A', 'NBJRR1_MMAX_A', 'NBJRR1_MSOM_A', 'NBJRR5_MM_A', 'NBJRR5_MMAX_A', 'NBJRR5_MSOM_A', 'NBJRR10_MM_A', 'NBJRR10_MMAX_A', 'NBJRR10_MSOM_A', 'NBJRR30_MM_A', 'NBJRR30_MMAX_A', 'NBJRR30_MSOM_A', 'NBJRR100_MM_A', 'NBJRR100_MMAX_A', 'NBJRR100_MSOM_A', 'RR_VOR_MM_A', 'RR_VOR_MMAX_A', 'RRAB_VOR_MM_A', 'RRAB_VOR_MMAX_A', 'TAILLE1', 'TAILLE2', 'CARACT4', 'SURFACE4', 'SURFACE6']

    X_encoded = X.copy()

    # Fusionner X et y selon la colonne 'ID'
    merged = pd.merge(X_encoded, y, left_on='ID', right_on='ID')

    for col in ordinal_columns:
        # Calculer la fréquence de la cible par valeur de la colonne
        freq_target = merged.groupby(col)[target_name].sum()
        
        # Ordonner les valeurs par fréquence croissante
        sorted_values = freq_target.sort_values().index
        
        # Dictionnaire d'encodage
        encoding_dict = {value: idx for idx, value in enumerate(sorted_values)}
        
        # Appliquer l'encodage
        X_encoded[col] = X_encoded[col].map(encoding_dict)

    return X_encoded



# Encode les colonnes catégorielles en fonction de la fréquence de la variable cible
def encode_categorical_by_target_frequency(X, y):

    # X : pd.DataFrame, DataFrame contenant les variables explicatives
    # y : pd.DataFrame, DataFrame contenant la variable cible

    # Sortie :  pd.DataFrame, DataFrame X avec les colonnes catégorielles encodées
    
    target_name = 'FREQ'

    categorical_columns = ['ACTIVIT2', 'VOCATION', 'ADOSS', 'CARACT1', 'CARACT3', 'INDEM1', 'TYPBAT1', 'INDEM2', 'FRCH1', 'FRCH2', 'DEROG2', 'DEROG3', 'DEROG4', 'DEROG5', 'DEROG8', 'DEROG12', 'KAPITAL34', 'KAPITAL35', 'KAPITAL37', 'KAPITAL40', 'KAPITAL41', 'KAPITAL42', 'KAPITAL43', 'RISK6', 'RISK8', 'RISK9', 'RISK10', 'RISK11', 'RISK12', 'RISK13', 'EQUIPEMENT2', 'EQUIPEMENT5', 'ESPINSEE', 'AN_EXERC', 'ZONE', 'TYPERS']

    X_encoded = X.copy()
    
    # Fusionner X et y selon la colonne 'ID'
    merged = pd.merge(X_encoded, y, left_on='ID', right_on='ID')
    
    for col in categorical_columns:
        # Calculer la fréquence de la cible par valeur de la colonne
        freq_target = merged.groupby(col)[target_name].sum()
        
        # Ordonner les valeurs par fréquence croissante
        sorted_values = freq_target.sort_values().index
        
        # Dictionnaire d'encodage
        encoding_dict = {value: idx for idx, value in enumerate(sorted_values)}
        
        # Appliquer l'encodage
        X_encoded[col] = X_encoded[col].map(encoding_dict)

    return X_encoded



# Convertir les colonnes numériques
def convert_to_numeric(X):
    
    # X : pd.DataFrame, DataFrame contenant les variables explicatives
    # Sortie :  pd.DataFrame, DataFrame X avec les colonnes numériques converties
    
    numeric_columns = ['DEROG13', 'DEROG14', 'DEROG16','ANCIENNETE', 'CARACT2', 'DUREE_REQANEUF', 'CARACT5', 'TYPBAT2', 'DEROG1', 'DEROG6', 'DEROG7', 'DEROG9', 'DEROG10', 'DEROG11', 'DEROG15', 'CA1', 'CA2', 'CA3', 'KAPITAL1', 'KAPITAL2', 'KAPITAL3', 'KAPITAL4', 'KAPITAL5', 'KAPITAL6', 'KAPITAL7', 'KAPITAL8', 'KAPITAL9', 'KAPITAL10', 'KAPITAL11', 'KAPITAL12', 'KAPITAL13', 'KAPITAL14', 'KAPITAL15', 'KAPITAL16', 'KAPITAL17', 'KAPITAL18', 'KAPITAL19', 'KAPITAL20', 'KAPITAL21', 'KAPITAL22', 'KAPITAL23', 'KAPITAL24', 'KAPITAL25', 'KAPITAL26', 'KAPITAL27', 'KAPITAL28', 'KAPITAL29', 'KAPITAL30', 'KAPITAL31', 'KAPITAL32', 'KAPITAL33', 'KAPITAL36', 'KAPITAL38', 'KAPITAL39', 'SURFACE1', 'SURFACE2', 'SURFACE3', 'SURFACE5', 'SURFACE7', 'SURFACE8', 'SURFACE9', 'SURFACE10', 'SURFACE11', 'SURFACE12', 'SURFACE13', 'SURFACE14', 'SURFACE15', 'SURFACE16', 'SURFACE17', 'SURFACE18', 'SURFACE19', 'SURFACE20', 'SURFACE21', 'NBBAT1', 'NBBAT2', 'NBBAT3', 'NBBAT4', 'NBBAT5', 'NBBAT6', 'NBBAT7', 'NBBAT8', 'NBBAT9', 'NBBAT10', 'NBBAT11', 'NBBAT13', 'NBBAT14', 'TAILLE3', 'TAILLE4', 'NBSINCONJ', 'NBSINSTRT', 'RISK1', 'RISK2', 'RISK3', 'RISK4', 'RISK5', 'RISK7', 'EQUIPEMENT1', 'EQUIPEMENT3', 'EQUIPEMENT4', 'EQUIPEMENT6', 'EQUIPEMENT7', 'ZONE_VENT']

    X[numeric_columns] = X[numeric_columns].apply(pd.to_numeric, errors='coerce')

    return X


