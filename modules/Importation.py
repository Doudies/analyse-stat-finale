import pandas as pd
import numpy as np
import io

def lire_fichier(uploaded_file, sep=";"):
    """
    Lit un fichier Excel (.xlsx, .xls) ou CSV avec gestion automatique d'encodage
    """
    filename = uploaded_file.name.lower()
    
    if filename.endswith(('.xlsx', '.xls')):
        # Lecture Excel
        try:
            df = pd.read_excel(uploaded_file)
            print(f"✅ Fichier Excel chargé : {uploaded_file.name}")
            return df
        except Exception as e:
            raise ValueError(f"Erreur lecture Excel : {str(e)}")
    
    elif filename.endswith('.csv'):
        # Lecture CSV avec gestion d'encodage
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16', 'windows-1252']
        
        # Convertir l'objet upload en bytes pour le réutiliser
        content = uploaded_file.getvalue()
        
        for encoding in encodings:
            try:
                # Convertir bytes en string avec l'encodage testé
                content_str = content.decode(encoding)
                # Lire depuis le string
                df = pd.read_csv(io.StringIO(content_str), sep=sep)
                print(f"✅ CSV chargé avec encodage : {encoding}")
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                # Autre erreur (séparateur, etc.)
                continue
        
        # Si aucun encodage ne fonctionne
        raise ValueError(
            f"Impossible de décoder le fichier CSV {uploaded_file.name}. "
            f"Essayez de le sauvegarder en UTF-8 ou Excel."
        )
    
    else:
        raise ValueError(f"Format non supporté : {uploaded_file.name}. Utilisez .xlsx, .xls ou .csv")

def nettoyer_donnees(df):
    """Nettoie les données (version basique)"""
    # Supprime les doublons
    df = df.drop_duplicates()
    
    # Supprime les colonnes vides
    df = df.dropna(axis=1, how='all')
    
    # Supprime les lignes avec trop de NaN
    threshold = len(df.columns) * 0.5  # 50% de valeurs manquantes
    df = df.dropna(thresh=threshold)
    
    return df

def preprocess_pour_series_temporelles(df, colonne_date):
    """Prépare les données pour l'analyse de séries temporelles"""
    # Convertir en datetime
    df[colonne_date] = pd.to_datetime(df[colonne_date], errors='coerce')
    
    # Supprimer les dates invalides
    df = df.dropna(subset=[colonne_date])
    
    # Trier par date
    df = df.sort_values(colonne_date)
    
    # Définir comme index
    df = df.set_index(colonne_date)
    
    return df