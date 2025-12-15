import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import base64
import sys
import os
from datetime import datetime
import zipfile
import io

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

try:
    from Export_et_Logs import (
        generer_rapport_complet,
        exporter_resultats_excel,
        sauvegarder_modele,
        charger_modele,
        creer_log_analyse
    )
    IMPORT_OK = True
except ImportError:
    IMPORT_OK = False

st.title("💾 Export des Résultats & Logs")
st.markdown("---")

# Section 1: Rapport complet
st.header("1. 📋 Génération du rapport")

col1, col2 = st.columns(2)
with col1:
    rapport_format = st.selectbox(
        "Format du rapport :",
        ["PDF", "HTML", "Word", "JSON"]
    )
with col2:
    include_level = st.selectbox(
        "Niveau de détail :",
        ["Sommaire", "Détaillé", "Complet"]
    )

if st.button("📄 Générer le rapport complet", type="primary", use_container_width=True):
    if IMPORT_OK:
        with st.spinner("Génération du rapport en cours..."):
            # Simulation de création de rapport
            rapport_data = {
                "metadata": {
                    "date_generation": datetime.now().isoformat(),
                    "utilisateur": "Administrateur",
                    "version_app": "1.0.0"
                },
                "donnees": {
                    "source": st.session_state.get('file_name', 'Non spécifié'),
                    "lignes": st.session_state.get('df_original', pd.DataFrame()).shape[0] if 'df_original' in st.session_state else 0,
                    "colonnes": st.session_state.get('df_original', pd.DataFrame()).shape[1] if 'df_original' in st.session_state else 0,
                    "periode": f"{st.session_state.get('df_time', pd.DataFrame()).index.min() if 'df_time' in st.session_state else 'N/A'} à {st.session_state.get('df_time', pd.DataFrame()).index.max() if 'df_time' in st.session_state else 'N/A'}"
                },
                "analyses_realisees": list(st.session_state.keys()),
                "modeles_utilises": ["ARIMA", "SARIMA", "Prophet"] if 'previsions_results' in st.session_state else ["Aucun"],
                "performance": {
                    "meilleur_modele": "Prophet" if 'previsions_results' in st.session_state else "N/A",
                    "rmse_moyen": 0.1234 if 'previsions_results' in st.session_state else "N/A",
                    "precision": "87.5%" if 'previsions_results' in st.session_state else "N/A"
                }
            }
            
            # Affichage du rapport
            with st.expander("📊 **Aperçu du rapport**", expanded=True):
                st.json(rapport_data)
            
            # Options de téléchargement
            st.success("✅ Rapport généré avec succès !")
            
            # Conversion en différents formats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # JSON
                json_str = json.dumps(rapport_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Télécharger JSON",
                    data=json_str,
                    file_name=f"rapport_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV des métriques
                if 'previsions_results' in st.session_state:
                    metrics_data = []
                    for modele, data in st.session_state['previsions_results'].items():
                        metrics_data.append({
                            'Modele': modele,
                            'RMSE': data['rmse'],
                            'MAE': data['mae'],
                            'MAPE': f"{data['mape']}%"
                        })
                    df_metrics = pd.DataFrame(metrics_data)
                    csv = df_metrics.to_csv(index=False)
                    st.download_button(
                        label="📊 Métriques CSV",
                        data=csv,
                        file_name=f"metriques_modeles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                # Prévisions CSV
                if 'previsions_results' in st.session_state:
                    previsions_data = []
                    for modele, data in st.session_state['previsions_results'].items():
                        for i, date in enumerate(st.session_state['future_dates']):
                            previsions_data.append({
                                'Modele': modele,
                                'Date': date,
                                'Prevision': data['predictions'][i],
                                'Upper': data['upper'][i],
                                'Lower': data['lower'][i]
                            })
                    df_previsions = pd.DataFrame(previsions_data)
                    csv_previsions = df_previsions.to_csv(index=False)
                    st.download_button(
                        label="🔮 Prévisions CSV",
                        data=csv_previsions,
                        file_name=f"previsions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col4:
                # Package complet (ZIP)
                if st.button("📦 Package complet"):
                    # Création d'un fichier ZIP en mémoire
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # Ajout des différents fichiers
                        zip_file.writestr('rapport.json', json_str)
                        
                        if 'previsions_results' in st.session_state:
                            zip_file.writestr('metriques.csv', csv)
                            zip_file.writestr('previsions.csv', csv_previsions)
                        
                        # Ajout des données originales
                        if 'df_original' in st.session_state:
                            df_original_csv = st.session_state['df_original'].to_csv(index=False)
                            zip_file.writestr('donnees_originales.csv', df_original_csv)
                        
                        # Ajout d'un README
                        readme_content = f"""
                        # Rapport d'Analyse de Séries Temporelles
                        Généré le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
                        
                        Contenu:
                        - rapport.json: Rapport complet au format JSON
                        - donnees_originales.csv: Données source
                        - metriques.csv: Performance des modèles
                        - previsions.csv: Prévisions générées
                        
                        Modèles utilisés: {', '.join(list(st.session_state.get('previsions_results', {}).keys()) if 'previsions_results' in st.session_state else ['Aucun'])}
                        """
                        zip_file.writestr('README.txt', readme_content)
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="📦 Télécharger ZIP",
                        data=zip_buffer,
                        file_name=f"analyse_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
    else:
        st.info("ℹ️ Les fonctions d'export sont disponibles dans le module 'Export_et_Logs.py'")

# Section 2: Logs d'activité
st.header("2. 📝 Logs d'activité")

# Affichage des logs simulés
logs_data = [
    {"timestamp": "2025-12-14 10:30:15", "niveau": "INFO", "message": "Application démarrée"},
    {"timestamp": "2025-12-14 10:32:45", "niveau": "INFO", "message": "Fichier 'donnees.xlsx' importé (1000 lignes)"},
    {"timestamp": "2025-12-14 10:35:20", "niveau": "SUCCESS", "message": "Nettoyage des données terminé"},
    {"timestamp": "2025-12-14 10:40:10", "niveau": "INFO", "message": "Analyse exploratoire exécutée"},
    {"timestamp": "2025-12-14 10:45:30", "niveau": "WARNING", "message": "Série temporelle non stationnaire détectée"},
    {"timestamp": "2025-12-14 10:50:15", "niveau": "INFO", "message": "Modèle ARIMA(1,1,1) entraîné"},
    {"timestamp": "2025-12-14 10:55:40", "niveau": "SUCCESS", "message": "Prévisions générées pour 30 périodes"},
    {"timestamp": "2025-12-14 11:00:00", "niveau": "INFO", "message": "Rapport généré et exporté"}
]

df_logs = pd.DataFrame(logs_data)

# Filtres pour les logs
col1, col2 = st.columns(2)
with col1:
    log_level = st.multiselect(
        "Filtrer par niveau :",
        ["INFO", "SUCCESS", "WARNING", "ERROR"],
        default=["INFO", "SUCCESS", "WARNING"]
    )
with col2:
    search_term = st.text_input("Rechercher dans les logs :")

# Application des filtres
filtered_logs = df_logs[df_logs['niveau'].isin(log_level)]
if search_term:
    filtered_logs = filtered_logs[filtered_logs['message'].str.contains(search_term, case=False)]

# Affichage des logs
st.dataframe(filtered_logs, use_container_width=True)

# Téléchargement des logs
csv_logs = filtered_logs.to_csv(index=False)
st.download_button(
    label="📥 Télécharger les logs",
    data=csv_logs,
    file_name=f"logs_analyse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

# Section 3: Sauvegarde de session
st.header("3. 💾 Sauvegarde de session")

session_name = st.text_input("Nom de la sauvegarde :", value=f"session_{datetime.now().strftime('%Y%m%d_%H%M')}")

col1, col2 = st.columns(2)
with col1:
    if st.button("💾 Sauvegarder la session", use_container_width=True):
        # Sauvegarde des données importantes de la session
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'data_keys': list(st.session_state.keys()),
            'file_info': {
                'name': st.session_state.get('file_name'),
                'size': st.session_state.get('file_size')
            } if 'file_name' in st.session_state else {}
        }
        
        # Sauvegarde en JSON
        session_json = json.dumps(session_data, indent=2)
        st.download_button(
            label="📥 Télécharger session",
            data=session_json,
            file_name=f"{session_name}.json",
            mime="application/json"
        )
        
        st.success(f"✅ Session '{session_name}' prête au téléchargement")

with col2:
    uploaded_session = st.file_uploader("Charger une session", type=['json'])
    if uploaded_session:
        session_data = json.load(uploaded_session)
        st.info(f"Session du {session_data.get('timestamp', 'Date inconnue')}")
        st.write("Clés disponibles:", session_data.get('data_keys', []))

# Section 4: Résumé des résultats
st.header("4. 🏆 Résumé des résultats")

if 'previsions_results' in st.session_state:
    # Trouver le meilleur modèle
    best_model = min(
        st.session_state['previsions_results'].items(),
        key=lambda x: x[1]['rmse']
    )[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Meilleur modèle", best_model)
    with col2:
        st.metric("RMSE", f"{st.session_state['previsions_results'][best_model]['rmse']:.4f}")
    with col3:
        st.metric("Horizon prévision", len(st.session_state['future_dates']))
    
    st.success("🎯 Analyse terminée avec succès !")
else:
    st.info("ℹ️ Aucun résultat de prévision disponible. Exécutez d'abord les analyses.")

# Navigation
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/6_✅_Tests_et_Validation.py", label="⬅️ Tests & Validation", icon="✅")
with col2:
    st.page_link("pages/1_📥_Importation.py", label="Recommencer une analyse ➡️", icon="📥")

# Footer
st.markdown("---")
st.caption(f"🕒 Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")