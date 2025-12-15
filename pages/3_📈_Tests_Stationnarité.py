import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

st.title("📈 Tests de Stationnarité & Décomposition")
st.markdown("---")

# 1. VÉRIFICATION DES DONNÉES
if 'df_time' not in st.session_state:
    st.error("❌ Aucune série temporelle configurée.")
    st.info("Veuillez d'abord sélectionner une colonne de date dans la page Importation.")
    st.page_link("pages/1_📥_Importation.py", label="⬅️ Aller à l'importation", icon="📥")
    st.stop()

df_time = st.session_state['df_time']

# 2. SÉLECTION DE LA VARIABLE
st.subheader("🎯 Sélection de la variable")

numeric_cols = df_time.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("❌ Aucune colonne numérique trouvée.")
    st.stop()

selected_col = st.selectbox("Sélectionnez la variable à analyser :", numeric_cols)
series = df_time[selected_col]

# 3. VISUALISATION DE LA SÉRIE
st.subheader("📊 Série temporelle")

fig_series = go.Figure()
fig_series.add_trace(go.Scatter(
    x=series.index,
    y=series.values,
    mode='lines',
    name='Série originale',
    line=dict(color='blue', width=2)
))

fig_series.update_layout(
    title=f"Série temporelle - {selected_col}",
    xaxis_title="Date",
    yaxis_title=selected_col,
    height=400
)
st.plotly_chart(fig_series, use_container_width=True)

# 4. TESTS DE STATIONNARITÉ
st.subheader("🔬 Tests de Stationnarité")

col1, col2 = st.columns(2)

# Test ADF
with col1:
    st.markdown("#### 📊 Test ADF (Augmented Dickey-Fuller)")
    if st.button("Exécuter test ADF", key="adf_btn"):
        try:
            result = adfuller(series.dropna())
            
            st.write(f"**Statistique ADF :** {result[0]:.4f}")
            st.write(f"**p-value :** {result[1]:.4f}")
            
            # Affichage des valeurs critiques
            st.write("**Valeurs critiques :**")
            for key, value in result[4].items():
                st.write(f"- {key} : {value:.4f}")
            
            if result[1] < 0.05:
                st.success("✅ **Série STATIONNAIRE** (p-value < 0.05)")
                st.write("La série ne présente pas de racine unitaire.")
            else:
                st.warning("⚠️ **Série NON STATIONNAIRE** (p-value ≥ 0.05)")
                st.write("Une différenciation peut être nécessaire.")
                
        except Exception as e:
            st.error(f"Erreur : {str(e)}")

# Test KPSS
with col2:
    st.markdown("#### 📊 Test KPSS")
    if st.button("Exécuter test KPSS", key="kpss_btn"):
        try:
            result = kpss(series.dropna(), regression='c')
            
            st.write(f"**Statistique KPSS :** {result[0]:.4f}")
            st.write(f"**p-value :** {result[1]:.4f}")
            
            # Affichage des valeurs critiques
            st.write("**Valeurs critiques :**")
            for key, value in result[3].items():
                st.write(f"- {key} : {value:.4f}")
            
            if result[1] > 0.05:
                st.success("✅ **Série STATIONNAIRE** (KPSS p-value > 0.05)")
            else:
                st.warning("⚠️ **Série NON STATIONNAIRE** (KPSS p-value ≤ 0.05)")
                
        except Exception as e:
            st.error(f"Erreur : {str(e)}")

# 5. DÉTECTION DE LA NATURE DE LA SAISONNALITÉ
st.subheader("🌊 Nature de la Saisonnalité")

# Sélection de la période
period = st.number_input("Période saisonnière supposée", min_value=2, max_value=52, value=12)

if st.button("🔍 Analyser la nature de la saisonnalité", type="primary"):
    try:
        # Détection automatique de la nature (additive vs multiplicative)
        series_clean = series.dropna()
        
        if len(series_clean) < period * 2:
            st.warning(f"⚠️ Pas assez de données pour analyser une période de {period}. Minimum requis: {period * 2}")
        else:
            # Calcul des statistiques par période
            n_periods = len(series_clean) // period
            
            # Calcul des moyennes et écarts-types par saison
            seasonal_stats = []
            for i in range(period):
                values = []
                for j in range(n_periods):
                    idx = i + j * period
                    if idx < len(series_clean):
                        val = series_clean.iloc[idx]
                        if not np.isnan(val):
                            values.append(val)
                
                if values:
                    seasonal_stats.append({
                        'saison': i+1,
                        'moyenne': np.mean(values),
                        'ecart_type': np.std(values) if len(values) > 1 else 0,
                        'coeff_variation': (np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
                    })
            
            if seasonal_stats:
                # Créer un DataFrame pour l'affichage
                stats_df = pd.DataFrame(seasonal_stats)
                
                st.write("**📊 Statistiques par saison :**")
                st.dataframe(stats_df.round(4), use_container_width=True)
                
                # Graphique des moyennes saisonnières
                fig_season = go.Figure()
                fig_season.add_trace(go.Bar(
                    x=stats_df['saison'],
                    y=stats_df['moyenne'],
                    name='Moyenne',
                    marker_color='blue'
                ))
                fig_season.add_trace(go.Scatter(
                    x=stats_df['saison'],
                    y=stats_df['ecart_type'],
                    name='Écart-type',
                    mode='lines+markers',
                    line=dict(color='red', width=2),
                    yaxis='y2'
                ))
                
                fig_season.update_layout(
                    title="Moyennes et écarts-types par saison",
                    xaxis_title="Saison",
                    yaxis_title="Moyenne",
                    yaxis2=dict(
                        title="Écart-type",
                        overlaying='y',
                        side='right'
                    ),
                    height=400
                )
                st.plotly_chart(fig_season, use_container_width=True)
                
                # Calcul du coefficient de variation moyen
                avg_cv = stats_df['coeff_variation'].mean()
                
                # Décision sur la nature
                if avg_cv < 0.1:
                    nature = "**ADDITIVE**"
                    interpretation = "Les variations saisonnières sont constantes dans le temps"
                    st.success(f"✅ Nature de la saisonnalité : {nature}")
                    st.write(f"**Coefficient de variation moyen :** {avg_cv:.4f} (< 0.1)")
                    st.write(f"**Interprétation :** {interpretation}")
                else:
                    nature = "**MULTIPLICATIVE**"
                    interpretation = "Les variations saisonnières sont proportionnelles au niveau de la série"
                    st.warning(f"⚠️ Nature de la saisonnalité : {nature}")
                    st.write(f"**Coefficient de variation moyen :** {avg_cv:.4f} (≥ 0.1)")
                    st.write(f"**Interprétation :** {interpretation}")
                
                # Stocker la nature détectée
                st.session_state['seasonal_nature'] = nature
                st.session_state['seasonal_period'] = period
                st.session_state['seasonal_stats'] = stats_df
                
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {str(e)}")

# 6. DÉCOMPOSITION SELON LA NATURE DÉTECTÉE
st.subheader("📉 Décomposition de la série")

if 'seasonal_nature' in st.session_state:
    st.write(f"**Nature détectée :** {st.session_state['seasonal_nature']}")
    st.write(f"**Période utilisée :** {st.session_state['seasonal_period']}")
    
    model_type = "additive" if "ADDITIVE" in st.session_state['seasonal_nature'] else "multiplicative"
    
    if st.button("🔧 Décomposer selon la nature détectée"):
        try:
            # Décomposition
            decomposition = seasonal_decompose(series.dropna(), model=model_type, period=st.session_state['seasonal_period'])
            
            # Création du graphique
            fig_decomp = make_subplots(
                rows=4, cols=1,
                subplot_titles=("Série Observée", "Tendance", "Saisonnalité", "Résidus"),
                vertical_spacing=0.08
            )
            
            # Série observée
            fig_decomp.add_trace(
                go.Scatter(x=decomposition.observed.index, y=decomposition.observed, 
                          mode='lines', name='Observé', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Tendance
            fig_decomp.add_trace(
                go.Scatter(x=decomposition.trend.index, y=decomposition.trend, 
                          mode='lines', name='Tendance', line=dict(color='red', width=2)),
                row=2, col=1
            )
            
            # Saisonnalité
            fig_decomp.add_trace(
                go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, 
                          mode='lines', name='Saisonnalité', line=dict(color='green')),
                row=3, col=1
            )
            
            # Résidus
            fig_decomp.add_trace(
                go.Scatter(x=decomposition.resid.index, y=decomposition.resid, 
                          mode='lines', name='Résidus', line=dict(color='orange')),
                row=4, col=1
            )
            
            fig_decomp.update_layout(height=800, showlegend=False, 
                                    title_text=f"Décomposition {model_type} (période={st.session_state['seasonal_period']})")
            st.plotly_chart(fig_decomp, use_container_width=True)
            
            # Statistiques des résidus
            st.subheader("📊 Statistiques des résidus")
            resid_clean = decomposition.resid.dropna()
            
            if len(resid_clean) > 0:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Moyenne", f"{resid_clean.mean():.4f}")
                with col2:
                    st.metric("Écart-type", f"{resid_clean.std():.4f}")
                with col3:
                    st.metric("Skewness", f"{stats.skew(resid_clean):.4f}")
                with col4:
                    st.metric("Kurtosis", f"{stats.kurtosis(resid_clean):.4f}")
                
        except Exception as e:
            st.error(f"Erreur lors de la décomposition : {str(e)}")
else:
    st.info("ℹ️ Analysez d'abord la nature de la saisonnalité ci-dessus.")

# 7. ACF ET PACF
st.subheader("📉 Fonctions d'autocorrélation")

if st.button("📈 Calculer ACF et PACF"):
    try:
        # Créer les graphiques ACF et PACF
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF
        plot_acf(series.dropna(), ax=ax1, lags=min(40, len(series)//2))
        ax1.set_title("Fonction d'Autocorrélation (ACF)")
        ax1.grid(True, alpha=0.3)
        
        # PACF
        plot_pacf(series.dropna(), ax=ax2, lags=min(40, len(series)//2))
        ax2.set_title("Fonction d'Autocorrélation Partielle (PACF)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Détection de période saisonnière via ACF
        st.subheader("🔍 Détection de période saisonnière")
        
        from statsmodels.tsa.stattools import acf
        acf_values = acf(series.dropna(), nlags=min(40, len(series)//2))
        
        # Chercher les pics significatifs
        significant_lags = []
        for i in range(1, len(acf_values)):
            if abs(acf_values[i]) > 0.3:  # Seuil de significativité
                significant_lags.append(i)
        
        if significant_lags:
            # Trouver la période dominante
            periods = []
            for i in range(len(significant_lags)-1):
                diff = significant_lags[i+1] - significant_lags[i]
                if 2 <= diff <= 24:  # Périodes raisonnables
                    periods.append(diff)
            
            if periods:
                from collections import Counter
                period_counts = Counter(periods)
                most_common_period = period_counts.most_common(1)[0][0]
                
                st.success(f"📅 **Période saisonnière détectée :** {most_common_period}")
                st.write(f"**Pics significatifs aux lags :** {significant_lags}")
            else:
                st.info("ℹ️ Aucune période saisonnière claire détectée")
        else:
            st.info("ℹ️ Pas d'autocorrélation significative détectée")
        
    except Exception as e:
        st.error(f"Erreur : {str(e)}")

# 8. NAVIGATION
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⬅️ Analyse exploratoire", use_container_width=True):
        st.switch_page("pages/2_📊_Analyse_Exploratoire.py")
with col2:
    if st.button("🏠 Accueil", use_container_width=True):
        st.switch_page("app.py")
with col3:
    if st.button("Modèles classiques ➡️", use_container_width=True):
        st.switch_page("pages/4_🤖_Modèles_Classiques.py")