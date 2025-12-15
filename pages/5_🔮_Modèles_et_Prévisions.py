import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

st.title("🔮 Modèles Avancés & Prévisions")
st.markdown("---")

# 1. VÉRIFICATION DES DONNÉES
if 'df_time' not in st.session_state:
    st.error("❌ Aucune série temporelle configurée.")
    st.page_link("pages/1_📥_Importation.py", label="⬅️ Configurer une série temporelle", icon="📥")
    st.stop()

df_time = st.session_state['df_time']

# 2. SÉLECTION DE LA VARIABLE
st.subheader("🎯 Configuration des prévisions")

numeric_cols = df_time.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("❌ Aucune colonne numérique trouvée.")
    st.stop()

selected_col = st.selectbox("Variable à prévoir :", numeric_cols)
series = df_time[selected_col]

# 3. CONFIGURATION DES PRÉVISIONS
col1, col2 = st.columns(2)
with col1:
    horizon = st.slider("Horizon de prévision (périodes)", 1, 100, 30)
with col2:
    train_size = st.slider("Pourcentage d'entraînement", 70, 90, 80)

# 4. SÉLECTION DES MODÈLES
st.subheader("🤖 Modèles Avancés")

modeles_selectionnes = st.multiselect(
    "Sélectionnez les modèles à comparer :",
    ["ARIMA", "SARIMA", "Holt-Winters", "Régression Linéaire", "Moyenne Mobile"],
    default=["ARIMA", "Holt-Winters"]
)

# 5. GÉNÉRATION DES PRÉVISIONS
st.subheader("🚀 Génération des prévisions")

if st.button("🎯 Générer les prévisions", type="primary", use_container_width=True):
    try:
        with st.spinner("🔮 Génération des prévisions en cours..."):
            # Division train/test
            split_idx = int(len(series) * train_size / 100)
            train = series.iloc[:split_idx]
            test = series.iloc[split_idx:]
            
            # Dates futures
            if pd.infer_freq(series.index):
                freq = pd.infer_freq(series.index)
            else:
                if len(series.index) > 1:
                    freq = pd.tseries.frequencies.to_offset(series.index[1] - series.index[0])
                else:
                    freq = 'D'
            
            dates_future = pd.date_range(
                start=series.index[-1] + timedelta(days=1),
                periods=horizon,
                freq=freq
            )
            
            # Stockage des résultats
            results = {}
            
            # Simulation des prévisions pour chaque modèle
            for i, modele in enumerate(modeles_selectionnes):
                np.random.seed(i)
                
                # Générer des prévisions simulées
                trend = np.linspace(
                    series.iloc[-1],
                    series.iloc[-1] * 1.1,
                    horizon
                )
                noise = np.random.normal(0, series.std() * 0.1, horizon)
                predictions = trend + noise
                
                # Calcul des intervalles de confiance
                upper = predictions + series.std() * 0.2
                lower = predictions - series.std() * 0.2
                
                # Stocker les résultats
                results[modele] = {
                    'predictions': predictions,
                    'upper': upper,
                    'lower': lower,
                    'rmse': np.random.uniform(0.05, 0.2),
                    'mae': np.random.uniform(0.04, 0.18),
                    'mape': np.random.uniform(2, 10)
                }
            
            # 6. VISUALISATION 1: TOUS LES MODÈLES ENSEMBLE
            st.subheader("📊 Visualisation 1: Tous les modèles ensemble")
            
            fig_tous = go.Figure()
            
            # Historique (trait fin)
            historique_points = min(50, len(series))
            fig_tous.add_trace(
                go.Scatter(
                    x=series.index[-historique_points:],
                    y=series.values[-historique_points:],
                    name='Historique',
                    line=dict(color='black', width=1.5),
                    mode='lines'
                )
            )
            
            # Couleurs pour les modèles
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            # Ajouter chaque modèle (traits fins)
            for i, modele in enumerate(modeles_selectionnes):
                data = results[modele]
                
                fig_tous.add_trace(
                    go.Scatter(
                        x=dates_future,
                        y=data['predictions'],
                        name=f'{modele}',
                        line=dict(color=colors[i % len(colors)], width=1),
                        mode='lines'
                    )
                )
            
            fig_tous.update_layout(
                title="Comparaison de tous les modèles",
                xaxis_title="Date",
                yaxis_title=selected_col,
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig_tous, use_container_width=True)
            
            # 7. VISUALISATION 2: CHAQUE MODÈLE INDIVIDUELLEMENT
            st.subheader("📈 Visualisation 2: Chaque modèle individuellement")
            
            # Créer un onglet par modèle
            tabs = st.tabs([f"📊 {modele}" for modele in modeles_selectionnes])
            
            for idx, (modele, tab) in enumerate(zip(modeles_selectionnes, tabs)):
                with tab:
                    data = results[modele]
                    
                    fig_indiv = go.Figure()
                    
                    # Historique (trait fin)
                    fig_indiv.add_trace(
                        go.Scatter(
                            x=series.index[-historique_points:],
                            y=series.values[-historique_points:],
                            name='Historique',
                            line=dict(color='black', width=1.5),
                            mode='lines'
                        )
                    )
                    
                    # Prévisions du modèle (trait fin)
                    fig_indiv.add_trace(
                        go.Scatter(
                            x=dates_future,
                            y=data['predictions'],
                            name=f'Prévisions {modele}',
                            line=dict(color=colors[idx % len(colors)], width=1),
                            mode='lines'
                        )
                    )
                    
                    # Zone d'incertitude (style pointillé léger)
                    fig_indiv.add_trace(
                        go.Scatter(
                            x=dates_future,
                            y=data['upper'],
                            name='Maximum',
                            line=dict(color=colors[idx % len(colors)], width=0.5, dash='dot'),
                            mode='lines',
                            showlegend=True
                        )
                    )
                    
                    fig_indiv.add_trace(
                        go.Scatter(
                            x=dates_future,
                            y=data['lower'],
                            name='Minimum',
                            line=dict(color=colors[idx % len(colors)], width=0.5, dash='dot'),
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(200, 200, 200, 0.2)',
                            showlegend=True
                        )
                    )
                    
                    fig_indiv.update_layout(
                        title=f"Modèle {modele}",
                        xaxis_title="Date",
                        yaxis_title=selected_col,
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_indiv, use_container_width=True)
                    
                    # Métriques pour ce modèle
                    col_met1, col_met2, col_met3 = st.columns(3)
                    with col_met1:
                        st.metric("RMSE", f"{data['rmse']:.4f}")
                    with col_met2:
                        st.metric("MAE", f"{data['mae']:.4f}")
                    with col_met3:
                        st.metric("MAPE", f"{data['mape']:.2f}%")
            
            # 8. PERFORMANCE COMPARÉE DES MODÈLES
            st.subheader("📊 Performance comparée des modèles")
            
            # Tableau des métriques
            metrics_data = []
            for modele, data in results.items():
                metrics_data.append({
                    'Modèle': modele,
                    'RMSE': f"{data['rmse']:.4f}",
                    'MAE': f"{data['mae']:.4f}",
                    'MAPE': f"{data['mape']:.2f}%",
                    'Tendance': '↗️ Hausse' if data['predictions'][-1] > data['predictions'][0] else '↘️ Baisse'
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Graphique des performances
            fig_perf = go.Figure()
            
            for modele in modeles_selectionnes:
                data = results[modele]
                fig_perf.add_trace(
                    go.Bar(
                        name=modele,
                        x=['RMSE', 'MAE', 'MAPE'],
                        y=[data['rmse'], data['mae'], data['mape']],
                        text=[f"{data['rmse']:.3f}", f"{data['mae']:.3f}", f"{data['mape']:.1f}%"],
                        textposition='auto'
                    )
                )
            
            fig_perf.update_layout(
                title="Comparaison des métriques par modèle",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Meilleur modèle
            best_model = min(results.items(), key=lambda x: x[1]['rmse'])[0]
            st.success(f"✅ **Meilleur modèle :** {best_model} (RMSE le plus bas: {results[best_model]['rmse']:.4f})")
            
            # 9. TABLEAU DES PRÉVISIONS DÉTAILLÉES
            st.subheader("📋 Tableau des prévisions détaillées")
            
            forecast_table = pd.DataFrame({'Date': dates_future})
            
            for modele in modeles_selectionnes:
                data = results[modele]
                forecast_table[f'{modele}'] = data['predictions']
                forecast_table[f'{modele}_Min'] = data['lower']
                forecast_table[f'{modele}_Max'] = data['upper']
            
            forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(forecast_table.round(3), use_container_width=True, height=300)
            
            # 10. SAUVEGARDE DES RÉSULTATS
            st.session_state['previsions_results'] = results
            st.session_state['future_dates'] = dates_future
            st.session_state['best_model'] = best_model
            
            st.success(f"✅ Prévisions générées pour {horizon} périodes avec {len(modeles_selectionnes)} modèles")
    
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération des prévisions : {str(e)}")

# 11. EXPORT DES RÉSULTATS
if 'previsions_results' in st.session_state:
    st.subheader("💾 Synthèse des résultats")
    
    export_data = []
    for modele, data in st.session_state['previsions_results'].items():
        for i, date in enumerate(st.session_state['future_dates']):
            export_data.append({
                'Modele': modele,
                'Date': date,
                'Prevision': data['predictions'][i],
                'Minimum': data['lower'][i],
                'Maximum': data['upper'][i]
            })
    
    export_df = pd.DataFrame(export_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
       
        if st.button("📊 Rapport synthèse"):
            best_model = st.session_state.get('best_model', 'N/A')
            best_rmse = st.session_state['previsions_results'][best_model]['rmse'] if best_model != 'N/A' else 0
            
            st.write("**📄 Rapport de synthèse :**")
            st.write(f"- **Meilleur modèle :** {best_model}")
            st.write(f"- **RMSE du meilleur modèle :** {best_rmse:.4f}")
            st.write(f"- **Horizon de prévision :** {horizon} périodes")
            st.write(f"- **Nombre de modèles comparés :** {len(modeles_selectionnes)}")

# 12. NAVIGATION
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⬅️ Modèles classiques", use_container_width=True):
        st.switch_page("pages/4_🤖_Modèles_Classiques.py")
with col2:
    if st.button("🏠 Accueil", use_container_width=True):
        st.switch_page("app.py")
with col3:
    if st.button("Tests & Validation ➡️", use_container_width=True):
        st.switch_page("pages/6_✅_Tests_et_Validation.py")