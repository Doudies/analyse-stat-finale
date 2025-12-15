import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from scipy import stats

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

st.title("🤖 Modèles Classiques de Séries Temporelles")
st.markdown("---")

# 1. VÉRIFICATION DES DONNÉES
if 'df_time' not in st.session_state:
    st.error("❌ Aucune série temporelle configurée.")
    st.page_link("pages/1_📥_Importation.py", label="⬅️ Configurer une série temporelle", icon="📥")
    st.stop()

df_time = st.session_state['df_time']

# 2. SÉLECTION DE LA VARIABLE
st.subheader("🎯 Sélection de la variable")

numeric_cols = df_time.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("❌ Aucune colonne numérique trouvée.")
    st.stop()

selected_col = st.selectbox("Sélectionnez la variable à modéliser :", numeric_cols)
series = df_time[selected_col]

# 3. SKEWNESS ET KURTOSIS DE LA SÉRIE
st.subheader("📊 Mesures de Forme de la Série")

col1, col2 = st.columns(2)

with col1:
    skewness = series.skew()
    st.metric("📐 **Skewness**", f"{skewness:.4f}")
    
    if abs(skewness) < 0.5:
        st.success("✅ Symétrique")
    elif skewness > 0:
        st.warning("↗️ Queue droite")
    else:
        st.warning("↙️ Queue gauche")

with col2:
    kurtosis = series.kurtosis()
    st.metric("📏 **Kurtosis**", f"{kurtosis:.4f}")
    
    kurtosis_fisher = kurtosis - 3
    if abs(kurtosis_fisher) < 0.5:
        st.success("✅ Normal")
    elif kurtosis_fisher > 0:
        st.error("📈 Pointue")
    else:
        st.info("📉 Aplatie")

st.markdown("---")

# 4. MOYENNES MOBILES CENTRÉES
st.subheader("📊 Moyennes Mobiles Centrées")

K = len(series)
st.write(f"**Nombre de données (K) :** {K}")

# Calcul selon K pair/impair
if K % 2 == 0:  # K pair
    window_size = 4
    method = "K pair → Fenêtre de 4"
else:  # K impair
    window_size = 3
    method = "K impair → Fenêtre de 3"

st.write(f"**Méthode :** {method}")

# Calcul des moyennes mobiles centrées
df_ma = pd.DataFrame({
    'Date': series.index,
    selected_col: series.values
})

# Initialiser avec NaN
ma_values = np.full(len(series), np.nan)

if window_size == 4:  # K pair
    for i in range(2, len(series) - 2):
        ma_values[i] = (
            (1/8) * series.iloc[i-2] +
            (1/4) * series.iloc[i-1] +
            (1/4) * series.iloc[i] +
            (1/4) * series.iloc[i+1] +
            (1/8) * series.iloc[i+2]
        )
else:  # window_size == 3 (K impair)
    for i in range(1, len(series) - 1):
        ma_values[i] = (
            series.iloc[i-1] +
            series.iloc[i] +
            series.iloc[i+1]
        ) / 3

df_ma['Moyenne_Mobile_Centree'] = ma_values

# Afficher le tableau
st.write("**Tableau des moyennes mobiles centrées :**")

df_display = df_ma.copy()
if pd.api.types.is_datetime64_any_dtype(df_display['Date']):
    df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d')

st.dataframe(df_display.round(3), use_container_width=True, height=300)

st.markdown("---")

# 5. RÉGRESSION LINÉAIRE - CORRIGÉE
st.subheader("📈 Régression Linéaire")

if st.button("🔧 Calculer la régression linéaire", type="primary"):
    try:
        # VÉRIFIER qu'on a assez de données
        if len(series) < 2:
            st.error("❌ Pas assez de données pour la régression (minimum 2 points)")
            st.stop()
        
        # Préparation des données
        X = np.arange(len(series)).reshape(-1, 1)  # Variable temps: [0, 1, 2, ...]
        y = series.values
        
        # VÉRIFIER qu'il n'y a pas de NaN
        if np.any(np.isnan(y)):
            st.warning("⚠️ Données contenant des NaN. Nettoyage en cours...")
            mask = ~np.isnan(y)
            X = X[mask]
            y = y[mask]
        
        if len(y) < 2:
            st.error("❌ Pas assez de données valides après nettoyage")
            st.stop()
        
        # Importer les modules nécessaires
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        except ImportError:
            st.error("❌ scikit-learn n'est pas installé. Installez-le avec: pip install scikit-learn")
            st.stop()
        
        # Régression linéaire
        model = LinearRegression()
        model.fit(X, y)
        
        # Prédictions
        y_pred = model.predict(X)
        
        # Calcul des métriques
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # Calcul des résidus
        residuals = y - y_pred
        
        # Statistiques des résidus
        if len(residuals) > 0:
            resid_mean = residuals.mean()
            resid_std = residuals.std()
            resid_skew = pd.Series(residuals).skew()
            resid_kurt = pd.Series(residuals).kurtosis()
        else:
            resid_mean = resid_std = resid_skew = resid_kurt = np.nan
        
        # Créer le tableau des résultats
        st.write("**📋 Résultats de la régression linéaire :**")
        
        results_table = pd.DataFrame({
            'Paramètre': [
                'Coefficient (pente)', 
                'Intercept', 
                'R²',
                'MAE',
                'RMSE'
            ],
            'Valeur': [
                f"{model.coef_[0]:.6f}",
                f"{model.intercept_:.6f}",
                f"{r2:.6f}",
                f"{mae:.6f}",
                f"{rmse:.6f}"
            ],
            'Interprétation': [
                f"Changement par unité de temps",
                f"Valeur initiale (t=0)",
                f"{r2*100:.1f}% de variance expliquée",
                "Erreur absolue moyenne",
                "Racine de l'erreur quadratique moyenne"
            ]
        })
        
        st.dataframe(results_table, use_container_width=True)
        
        # Tableau des statistiques des résidus
        if len(residuals) > 0:
            st.write("**📊 Statistiques des résidus :**")
            
            residuals_table = pd.DataFrame({
                'Statistique': ['Moyenne', 'Écart-type', 'Skewness', 'Kurtosis'],
                'Valeur': [
                    f"{resid_mean:.6f}",
                    f"{resid_std:.6f}",
                    f"{resid_skew:.6f}",
                    f"{resid_kurt:.6f}"
                ],
                'Valeur idéale': ['0', 'Minimale', '0', '0']
            })
            
            st.dataframe(residuals_table, use_container_width=True)
        
        # Équation
        st.write(f"**📝 Équation du modèle :**")
        st.code(f"ŷ(t) = {model.coef_[0]:.6f} × t + {model.intercept_:.6f}")
        
        # Tableau des prédictions (premières 10)
        st.write("**🔍 Prédictions (10 premières valeurs) :**")
        
        predictions_table = pd.DataFrame({
            't': range(1, min(11, len(series)+1)),
            'Date': series.index[:10] if len(series) >= 10 else series.index,
            'Valeur réelle': y[:10] if len(y) >= 10 else y,
            'Prédiction': y_pred[:10] if len(y_pred) >= 10 else y_pred,
            'Résidu': residuals[:10] if len(residuals) >= 10 else residuals
        })
        
        # Formater les dates
        if pd.api.types.is_datetime64_any_dtype(predictions_table['Date']):
            predictions_table['Date'] = predictions_table['Date'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(predictions_table.round(3), use_container_width=True)
        
        # Test de normalité des résidus
        if len(residuals) >= 3:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                st.write("**🧪 Test de Shapiro-Wilk (normalité des résidus) :**")
                st.write(f"- Statistique : {shapiro_stat:.4f}")
                st.write(f"- p-value : {shapiro_p:.4f}")
                if shapiro_p > 0.05:
                    st.success("✅ Résidus normalement distribués (p > 0.05)")
                else:
                    st.warning("⚠️ Résidus non normalement distribués (p ≤ 0.05)")
            except:
                st.info("ℹ️ Test de Shapiro-Wilk non disponible (trop de données)")
        
        # Stocker les résultats
        st.session_state['regression_results'] = {
            'equation': f"ŷ(t) = {model.coef_[0]:.6f} × t + {model.intercept_:.6f}",
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'predictions': y_pred,
            'residuals': residuals
        }
        
        st.success("✅ Régression linéaire calculée avec succès !")
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la régression : {str(e)}")
        st.info("Vérifiez que vos données ne contiennent pas de valeurs manquantes ou infinies.")

# 6. EXPORT
st.subheader("💾 Export des résultats")

if st.button("📥 Exporter tous les résultats"):
    try:
        # Préparer les données
        export_data = pd.DataFrame({
            'Date': series.index,
            'Valeur': series.values,
            'Moyenne_Mobile_Centree': ma_values
        })
        
        # Ajouter les prédictions de régression si disponibles
        if 'regression_results' in st.session_state:
            # Recalculer pour s'assurer de la correspondance
            X = np.arange(len(series)).reshape(-1, 1)
            coef = model.coef_[0]
            intercept = model.intercept_
            export_data['Prediction_Regression'] = coef * X.flatten() + intercept
            export_data['Residu_Regression'] = series.values - export_data['Prediction_Regression'].values
        
        # Formater les dates
        if pd.api.types.is_datetime64_any_dtype(export_data['Date']):
            export_data['Date'] = export_data['Date'].dt.strftime('%Y-%m-%d')
        
        # Convertir en CSV
        csv = export_data.to_csv(index=False)
        
        # Bouton de téléchargement
        st.download_button(
            label="📥 Télécharger CSV",
            data=csv,
            file_name=f"modeles_classiques_{selected_col}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Erreur lors de l'export : {str(e)}")

# 7. NAVIGATION
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⬅️ Tests de stationnarité", use_container_width=True):
        st.switch_page("pages/3_📈_Tests_Stationnarité.py")
with col2:
    if st.button("🏠 Accueil", use_container_width=True):
        st.switch_page("app.py")
with col3:
    if st.button("Prévisions avancées ➡️", use_container_width=True):
        st.switch_page("pages/5_🔮_Modèles_et_Prévisions.py")