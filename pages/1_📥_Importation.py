import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

try:
    from Importation import lire_fichier, nettoyer_donnees, preprocess_pour_series_temporelles
    IMPORT_OK = True
except ImportError as e:
    IMPORT_OK = False
    st.error(f"Erreur d'import : {str(e)}")

st.title("📥 Importation des Données")

uploaded_file = st.file_uploader(
    "**Téléchargez votre fichier**",
    type=["xlsx", "xls", "csv"],
    help="Formats acceptés : Excel (.xlsx, .xls) ou CSV"
)

if uploaded_file is not None:
    try:
        # Utilise la fonction améliorée
        df = lire_fichier(uploaded_file)
        
        # Sauvegarde dans la session
        st.session_state['df_original'] = df
        st.session_state['file_name'] = uploaded_file.name
        
        # Affichage
        st.success(f"✅ Fichier '{uploaded_file.name}' chargé avec succès !")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Lignes", df.shape[0])
        with col2:
            st.metric("📈 Colonnes", df.shape[1])
        
        # Aperçu
        with st.expander("🔍 **Aperçu des données**", expanded=True):
            st.dataframe(df.head(10))
        
        # Nettoyage optionnel
        if st.checkbox("Appliquer le nettoyage automatique"):
            df_clean = nettoyer_donnees(df.copy())
            st.session_state['df_clean'] = df_clean
            st.success("✅ Données nettoyées")
            st.dataframe(df_clean.head())
        
        # Sélection colonne date
        st.subheader("📅 Configuration série temporelle")
        colonnes = df.columns.tolist()
        colonne_date = st.selectbox(
            "Sélectionnez la colonne de date/heure (optionnel)",
            ['Aucune'] + colonnes,
            help="Nécessaire pour les analyses de séries temporelles"
        )
        
        if colonne_date != 'Aucune':
            try:
                df_time = preprocess_pour_series_temporelles(df.copy(), colonne_date)
                st.session_state['df_time'] = df_time
                st.session_state['date_column'] = colonne_date
                st.success(f"✅ Série temporelle configurée sur '{colonne_date}'")
            except Exception as e:
                st.warning(f"⚠️ Impossible de configurer la série temporelle : {str(e)}")
        
        # Navigation
        st.markdown("---")
        st.success("✅ **Données prêtes pour l'analyse !**")
        st.page_link("pages/2_📊_Analyse_Exploratoire.py", 
                    label="➡️ Passer à l'analyse exploratoire", 
                    icon="📊")
        
    except Exception as e:
        st.error(f"❌ Erreur : {str(e)}")
        st.info("💡 **Conseils :**")
        st.write("1. Pour les CSV : assurez-vous qu'il utilise UTF-8 ou Latin1")
        st.write("2. Pour les Excel : vérifiez que le fichier n'est pas corrompu")
        st.write("3. Essayez de convertir votre CSV en Excel (.xlsx)")
else:
    st.info("👆 **Veuillez télécharger un fichier Excel (.xlsx, .xls) ou CSV**")
