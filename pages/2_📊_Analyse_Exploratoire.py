import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import sys
import os
try:
    import plotly.express as px
except ImportError:
    st.warning("Installation de plotly en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "scipy"])
    import plotly.express as px
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

st.title("📊 Analyse Exploratoire")
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
    st.error("❌ Aucune colonne numérique trouvée dans les données.")
    st.stop()

# Sélectionner la colonne de valeur
value_col = st.selectbox(
    "Sélectionnez la colonne à analyser :",
    numeric_cols
)

series = df_time[value_col]

# 3. SKEWNESS ET KURTOSIS SEULS EN PREMIER (comme les tests)
st.subheader("📐 Skewness & Kurtosis - Mesures de Forme")

# Ligne unique pour Skewness et Kurtosis
col_skew, col_kurt = st.columns(2)

with col_skew:
    # Calcul Skewness
    skewness = series.skew()
    
    # Container pour mise en forme
    with st.container():
        st.markdown("### 📐 **SKEWNESS**")
        st.markdown(f"<h1 style='text-align: center; color: {'green' if abs(skewness) < 0.5 else 'orange'};'>{skewness:.4f}</h1>", 
                   unsafe_allow_html=True)
        
        # Interprétation
        if abs(skewness) < 0.5:
            st.success("✅ **Distribution symétrique**")
            st.caption("|Skewness| < 0.5 → Bonne symétrie")
        elif skewness > 0:
            st.warning("↗️ **Asymétrie positive**")
            st.caption("Skewness > 0 → Queue à droite")
        else:
            st.warning("↙️ **Asymétrie négative**")
            st.caption("Skewness < 0 → Queue à gauche")

with col_kurt:
    # Calcul Kurtosis
    kurtosis = series.kurtosis()
    kurtosis_fisher = kurtosis - 3
    
    # Container pour mise en forme
    with st.container():
        st.markdown("### 📏 **KURTOSIS**")
        st.markdown(f"<h1 style='text-align: center; color: {'green' if abs(kurtosis_fisher) < 0.5 else 'red'};'>{kurtosis:.4f}</h1>", 
                   unsafe_allow_html=True)
        
        # Interprétation
        if abs(kurtosis_fisher) < 0.5:
            st.success("✅ **Distribution normale**")
            st.caption("Kurtosis ≈ 3 → Aplatissement normal")
        elif kurtosis_fisher > 0:
            st.error("📈 **Leptokurtique**")
            st.caption("Kurtosis > 3 → Distribution pointue")
        else:
            st.info("📉 **Platikurtique**")
            st.caption("Kurtosis < 3 → Distribution aplatie")

st.markdown("---")

# 4. AFFICHER LES DONNÉES (le reste de ton analyse détaillée)
st.subheader("📋 Données")

# Préparer les données pour l'affichage
df_display = df_time[[value_col]].copy()
df_display.index.name = 'date'

st.write(f"**Aperçu (10 premières lignes) :**")
st.dataframe(df_display.head(10), use_container_width=True)

st.write(f"**Dimensions :** {df_time.shape[0]} lignes × {df_time.shape[1]} colonnes")

st.markdown("---")

# 5. GRAPHIQUE DE LA SÉRIE TEMPORELLE
st.subheader("📈 Série temporelle")

fig = px.line(
    x=df_time.index,
    y=df_time[value_col],
    title=f"Évolution de {value_col} dans le temps",
    labels={'x': 'Date', 'y': value_col}
)

fig.update_xaxes(tickformat="%Y-%m-%d", tickangle=45)
fig.update_layout(height=500)

st.plotly_chart(fig, use_container_width=True)

# 6. STATISTIQUES DESCRIPTIVES
st.subheader("📊 Statistiques descriptives")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Moyenne", f"{df_time[value_col].mean():.2f}")
with col2:
    st.metric("Médiane", f"{df_time[value_col].median():.2f}")
with col3:
    st.metric("Minimum", f"{df_time[value_col].min():.2f}")
with col4:
    st.metric("Maximum", f"{df_time[value_col].max():.2f}")

# Plus de statistiques
with st.expander("📈 Statistiques détaillées"):
    stats_data = {
        'Statistique': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Variance', 'CV%'],
        'Valeur': [
            df_time[value_col].count(),
            f"{df_time[value_col].mean():.4f}",
            f"{df_time[value_col].std():.4f}",
            f"{df_time[value_col].min():.4f}",
            f"{df_time[value_col].quantile(0.25):.4f}",
            f"{df_time[value_col].quantile(0.5):.4f}",
            f"{df_time[value_col].quantile(0.75):.4f}",
            f"{df_time[value_col].max():.4f}",
            f"{df_time[value_col].var():.4f}",
            f"{(df_time[value_col].std() / df_time[value_col].mean() * 100) if df_time[value_col].mean() != 0 else 0:.2f}%"
        ]
    }
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

# 7. HISTOGRAMME
st.subheader("📊 Distribution")

fig_hist = px.histogram(
    df_time,
    x=value_col,
    nbins=30,
    title=f"Distribution de {value_col}",
    marginal="box"
)

fig_hist.update_layout(height=400)
st.plotly_chart(fig_hist, use_container_width=True)

# 8. BOX PLOT
st.subheader("📦 Box Plot")

fig_box = px.box(
    df_time,
    y=value_col,
    title=f"Box Plot de {value_col}"
)

fig_box.update_layout(height=300)
st.plotly_chart(fig_box, use_container_width=True)

# 9. NUAGE DE POINTS AVEC LAG
st.subheader("📈 Nuage de points avec Lag")

max_lag = min(12, len(df_time) // 2)
lag = st.slider("Décalage (lag)", 1, max_lag, 1, key="lag_slider")

if len(df_time) > lag:
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=df_time[value_col].iloc[:-lag],
        y=df_time[value_col].iloc[lag:],
        mode='markers',
        marker=dict(size=8, opacity=0.6),
        name=f'Lag {lag}'
    ))
    fig_scatter.update_layout(
        title=f"Nuage de points (lag={lag})",
        xaxis_title=f"{value_col}(t)",
        yaxis_title=f"{value_col}(t+{lag})",
        height=400
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Calculer la corrélation avec lag
    correlation = np.corrcoef(df_time[value_col].iloc[:-lag], df_time[value_col].iloc[lag:])[0, 1]
    st.write(f"**Corrélation avec lag {lag} :** {correlation:.3f}")

# 10. MATRICE DE CORRÉLATION (si plusieurs variables)
if len(numeric_cols) > 1:
    st.subheader("🔄 Matrice de Corrélation")
    
    selected_corr_cols = st.multiselect(
        "Sélectionnez les colonnes pour la corrélation :",
        numeric_cols,
        default=[value_col] + [c for c in numeric_cols if c != value_col][:min(4, len(numeric_cols)-1)]
    )
    
    if len(selected_corr_cols) > 1:
        corr_matrix = df_time[selected_corr_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            hoverinfo='text'
        ))
        
        fig_corr.update_layout(
            title="Matrice de corrélation",
            width=600,
            height=600
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Tableau de corrélation
        with st.expander("📋 Tableau de corrélation"):
            st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), 
                        use_container_width=True)

# 11. TESTS STATISTIQUES (optionnels)
st.subheader("🧪 Tests Statistiques")

if st.button("🔬 Exécuter les tests statistiques"):
    with st.spinner("Calcul des tests en cours..."):
        try:
            # Test de normalité (Shapiro-Wilk)
            from scipy.stats import shapiro
            shapiro_stat, shapiro_p = shapiro(series.dropna())
            
            # Test de stationnarité (ADF)
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(series.dropna())
            
            col_test1, col_test2 = st.columns(2)
            
            with col_test1:
                st.write("**Test de Shapiro-Wilk (normalité) :**")
                st.write(f"- Statistique : {shapiro_stat:.4f}")
                st.write(f"- p-value : {shapiro_p:.4f}")
                
                if shapiro_p > 0.05:
                    st.success("✅ Distribution normale (p > 0.05)")
                else:
                    st.warning("⚠️ Distribution non normale (p ≤ 0.05)")
            
            with col_test2:
                st.write("**Test ADF (stationnarité) :**")
                st.write(f"- Statistique ADF : {adf_result[0]:.4f}")
                st.write(f"- p-value : {adf_result[1]:.4f}")
                
                if adf_result[1] < 0.05:
                    st.success("✅ Série stationnaire (p < 0.05)")
                else:
                    st.warning("⚠️ Série non stationnaire (p ≥ 0.05)")
        
        except Exception as e:
            st.error(f"Erreur lors des tests : {str(e)}")

# 12. NAVIGATION
st.markdown("---")
col_nav1, col_nav2 = st.columns(2)
with col_nav1:
    if st.button("⬅️ Retour à l'importation", use_container_width=True):
        st.switch_page("pages/1_📥_Importation.py")
with col_nav2:
    if st.button("Tests de stationnarité ➡️", use_container_width=True):
        st.switch_page("pages/3_📈_Tests_Stationnarité.py")