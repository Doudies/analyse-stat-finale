import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
st.markdown("""
<style>
    .stProgress .st-bo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)
st.set_page_config(page_title="Modèles Classiques", page_icon="📐")
def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# =========================================================
# 📌 Chargement du dataset (robuste)
# =========================================================
possible_keys = ["df", "df_raw", "df_loaded", "dataframe"]
df = next((st.session_state.get(k) for k in possible_keys if k in st.session_state), None)

if df is None:
    st.error("❌ Aucune série chargée. Veuillez importer les données d’abord.")
    st.stop()

# =========================================================
# 📌 Conversion de la colonne Date → datetime
# =========================================================
date_col = df.columns[0]
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

df = df.dropna(subset=[date_col]).reset_index(drop=True)
dates = df[date_col]

# =========================================================
# 📌 Conversion valeurs : '96,1' → 96.1
# =========================================================
val_col = df.columns[1]
df[val_col] = (
    df[val_col]
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)

series = df[val_col]

# =========================================================
# TITRE
# =========================================================
st.title("📐 Modèles Classiques")
st.write("Cette section applique deux modèles simples : **Moyenne Mobile** et **Régression Linéaire**.")

# =========================================================
# Choix du modèle
# =========================================================
modele = st.selectbox(
    "Choisissez un modèle :",
    ["Moyenne Mobile", "Régression Linéaire"],
)

st.markdown("---")

# =========================================================
# 1️⃣ MOYENNE MOBILE
# =========================================================
if modele == "Moyenne Mobile":
    st.header("📎 Moyenne Mobile (MA) ")

    k = st.number_input("Choisissez la fenêtre (k)", min_value=2, max_value=20, value=3)

    # ---------------------------------------------------------
    # 📌 Fonction correcte de Moyenne Mobile (cours USTHB)
    # ---------------------------------------------------------
    def moyenne_mobile_exacte(series, k):
        n = len(series)
        mm = [None] * n
        m = k // 2

        if k % 2 == 1:
            # ---- k impair : MA centrée ----
            for t in range(m, n - m):
                mm[t] = np.mean(series[t-m : t+m+1])

        else:
            # ---- k pair : centrage double ----
            mm_inter = [None] * n
            for t in range(k-1, n):
                mm_inter[t] = np.mean(series[t-k+1 : t+1])

            for t in range(k, n):
                if mm_inter[t] is not None and mm_inter[t-1] is not None:
                    mm[t-1] = (mm_inter[t] + mm_inter[t-1]) / 2

        return mm

    if st.button("🧮 Calculer la Moyenne Mobile"):
        try:
            mm_values = moyenne_mobile_exacte(series.values, k)
            df_mm = df.copy()
            df_mm[f"MM({k})"] = mm_values

            # ======================
            # 🧾 TABLEAU EXACT
            # ======================
            st.subheader(f"📋 Tableau de la Moyenne Mobile (k = {k})")
            st.dataframe(df_mm[[date_col, f"MM({k})"]])

            # ======================
            # Résidus + indicateurs
            # ======================
            mm_series = pd.Series(mm_values)
            resid = series - mm_series
            resid = resid.dropna()

            mse = np.mean(resid**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(resid))

            st.subheader("📊 Indicateurs de performance")
            st.write(f"**MSE :** {mse:.4f}")
            st.write(f"**RMSE :** {rmse:.4f}")
            st.write(f"**MAE :** {mae:.4f}")
            mm_series = pd.Series(mm_values, index=series.index)

            # Zone valide uniquement
            mask = (~mm_series.isna()) & (series != 0)

            mape_value = np.mean(
               np.abs((series[mask] - mm_series[mask]) / series[mask])
            ) * 100

            st.write(f"**MAPE :** {mape_value:.4f} %")

            st.session_state["ma_results"] = df_mm[[date_col, f"MM({k})"]]

            # Résidus
            st.subheader("📉 Résidus")
            fig_res, ax_res = plt.subplots(figsize=(8, 3))
            ax_res.plot(resid.values, marker="o")
            ax_res.axhline(0, color="red", linestyle="--")
            st.pyplot(fig_res)

            # ACF
            st.subheader("📈 ACF des résidus")
            fig_acf, ax_acf = plt.subplots(figsize=(8, 3))
            plot_acf(resid, ax=ax_acf)
            st.pyplot(fig_acf)

        except Exception as e:
            st.error(f"Erreur : {e}")

# =========================================================
# 2️⃣ RÉGRESSION LINÉAIRE
# =========================================================
if modele == "Régression Linéaire":
    st.header("📎 Régression Linéaire (Tendance)")

    h = st.number_input("Nombre de périodes à prévoir :", 1, 24, 4)

    if st.button("📉 Calculer la Régression Linéaire"):
        try:
            # Encodage du temps
            t = np.arange(len(series))
            coeffs = np.polyfit(t, series, 1)
            trend = coeffs[0] * t + coeffs[1]

            # Prévisions
            t_future = np.arange(len(series), len(series) + h)
            forecast = coeffs[0] * t_future + coeffs[1]
            future_dates = pd.date_range(
               start=dates.iloc[-1],
               periods=h + 1,
               freq="MS"
            )[1:]

            # Résidus
            resid = series - trend

            mse = np.mean(resid**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(resid))
            # MAPE
            try:
                mape_val = mape(series, trend)
            except:
                mape_val = None

            # ===== Tableau des prévisions =====
            st.subheader("📋 Prévisions linéaires")
            df_forecast = pd.DataFrame({
                "Date": future_dates,
                "Prévision": forecast
            })
            st.dataframe(df_forecast)
            st.session_state["linreg_results"] = df_forecast

            # ===== Metrics =====
            st.subheader("📊 Indicateurs de performance")
            st.write(f"**MSE :** {mse:.4f}")
            st.write(f"**RMSE :** {rmse:.4f}")
            st.write(f"**MAE :** {mae:.4f}")
            try:
                mape_val = mape(series, trend)
                st.write(f"**MAPE :** {mape_val:.4f} %")
            except:
                st.warning("MAPE non calculable (valeurs manquantes).")


            # ===== Résidus =====
            st.subheader("📉 Résidus")
            fig_res, ax_res = plt.subplots(figsize=(8, 3))
            ax_res.plot(resid, marker="o")
            ax_res.axhline(0, color="red", linestyle="--")
            st.pyplot(fig_res)

            # ===== ACF =====
            st.subheader("📈 ACF des résidus")
            fig_acf, ax_acf = plt.subplots(figsize=(8, 3))
            plot_acf(resid, ax=ax_acf)
            st.pyplot(fig_acf)


        except Exception as e:
            st.error(f"Erreur : {e}")
