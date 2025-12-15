import streamlit as st
import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from statsmodels.graphics.tsaplots import plot_acf
from skopt import gp_minimize
from skopt.space import Real
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

# === Fix import src ===
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

if "journal" not in st.session_state:
    st.session_state["journal"] = {}

from src.models.smoothing_manual import (
    ses_forecast,
    holt_forecast,
    holt_winters_additive_forecast,
    holt_winters_multiplicative_forecast
)
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ----------------------------------------------------------
# Vérification que la série existe
# ----------------------------------------------------------
if "series" not in st.session_state:
    st.warning("Veuillez d'abord importer une série dans l'onglet 1.")
    st.stop()

series = st.session_state["series"].sort_index()

st.title("🔧 Modélisation & Prévisions")
st.markdown("---")

# ================================================================
# SECTION 1 — PRÉVISION MANUELLE (VERSION PROPRE)
# ================================================================
st.header("1️⃣ Prévision manuelle")
st.write("Définissez les paramètres du modèle et visualisez la prévision correspondante.")

# -------------------------------------------------------------
# 1. CHOIX DU MODÈLE
# -------------------------------------------------------------
st.subheader("📌 Choix du modèle")

model_name = st.selectbox(
    "Sélectionnez un modèle :",
    [
        "SES (Simple Exponential Smoothing)",
        "Holt (Double Exponential)",
        "Holt-Winters Additif",
        "Holt-Winters Multiplicatif"
    ]
)

# -------------------------------------------------------------
# 2. PARAMÈTRES DU MODÈLE
# -------------------------------------------------------------
st.subheader("⚙️ Paramètres du modèle")

with st.expander("Paramètres du modèle", expanded=True):

    horizon = st.number_input(
        "Nombre de périodes à prévoir",
        min_value=1, max_value=36, value=6
    )

    alpha = st.slider("α (niveau)", 0.01, 1.0, 0.4)
    beta = None
    gamma = None

    if model_name != "SES (Simple Exponential Smoothing)":
        beta = st.slider("β (tendance)", 0.01, 1.0, 0.3)

    if "Holt-Winters" in model_name:
        gamma = st.slider("γ (saisonnalité)", 0.01, 1.0, 0.2)

# -------------------------------------------------------------
# 3. BOUTON POUR LANCER LA PRÉVISION
# -------------------------------------------------------------
if st.button("📉 Lancer la prévision manuelle", type="primary"):

    try:

        # ============================================================
        # — CALCUL PRÉVISION
        # ============================================================
        if model_name == "SES (Simple Exponential Smoothing)":
            forecast = ses_forecast(series, alpha, horizon)

        elif model_name == "Holt (Double Exponential)":
            forecast = holt_forecast(series, alpha, beta, horizon)

        elif model_name == "Holt-Winters Additif":
            forecast = holt_winters_additive_forecast(series, alpha, beta, gamma, horizon)

        elif model_name == "Holt-Winters Multiplicatif":
            forecast = holt_winters_multiplicative_forecast(series, alpha, beta, gamma, horizon)

        st.session_state["forecast_manual"] = forecast
        st.success("Prévision manuelle calculée !")

        # ============================================================
        # — DATES FUTURES
        # ============================================================
        last_date = series.index[-1]
        future_dates = pd.date_range(last_date, periods=horizon+1, freq="MS")[1:]

        df_forecast_manual = pd.DataFrame({
            "Date": future_dates,
            "Prévision manuelle": forecast
        }).set_index("Date")

        # ---------------------- AFFICHAGE PREVISION ----------------------
        with st.expander("📈 Visualisation : Historique + Prévision manuelle", expanded=True):

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(series.index, series.values, label="Historique", linewidth=2)
            ax.plot(future_dates, forecast, label="Prévision manuelle", linestyle="--", marker="o")

            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            st.write("### 📄 Tableau des prévisions")
            st.dataframe(df_forecast_manual)

        # ============================================================
        # — ANALYSE DES RÉSIDUS (STATS MODELS)
        # ============================================================
        with st.expander("🔍 Analyse des résidus du modèle manuel"):

            if model_name == "SES (Simple Exponential Smoothing)":
                fit_model = SimpleExpSmoothing(series).fit(smoothing_level=alpha, optimized=False)
                k = 1

            elif model_name == "Holt (Double Exponential)":
                fit_model = ExponentialSmoothing(series, trend="add").fit(
                    smoothing_level=alpha, smoothing_trend=beta, optimized=False)
                k = 2
        
            elif model_name == "Holt-Winters Additif":
                fit_model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=4).fit(
                    smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=False)
                k = 3
            
            else:
                fit_model = ExponentialSmoothing(series, trend="add", seasonal="mul", seasonal_periods=4).fit(
                    smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma, optimized=False)
                k = 3

            residuals = fit_model.resid

            st.write("### 📌 Résidus dans le temps")
            fig_r, ax_r = plt.subplots(figsize=(10, 3))
            ax_r.plot(residuals.index, residuals.values, marker="o")
            ax_r.axhline(0, color="red", linestyle="--")
            ax_r.grid(True)
            st.pyplot(fig_r)

            st.write("### 📊 Histogramme des résidus")
            fig_h, ax_h = plt.subplots(figsize=(10, 3))
            ax_h.hist(residuals, bins=10, edgecolor="black")
            st.pyplot(fig_h)

            st.write("### 🔄 Autocorrélation (ACF)")
            fig_acf, ax_acf = plt.subplots(figsize=(10, 3))
            plot_acf(residuals.dropna(), ax=ax_acf)
            st.pyplot(fig_acf)

        # ============================================================
        # — MÉTRIQUES
        # ============================================================
        with st.expander("📊 Métriques du modèle manuel"):

            RSS = np.sum(residuals**2)
            n = len(series)

            AIC = 2*k + n*np.log(RSS/n)
            AICc = AIC + (2*k*(k+1)) / (n - k - 1) if (n - k - 1) > 0 else np.inf
            BIC = k*np.log(n) + n*np.log(RSS/n)
            MSE = RSS / n

            df_manual_metrics = pd.DataFrame({
                "Metric": ["MSE", "AIC", "AICc", "BIC"],
                "Valeur": [MSE, AIC, AICc, BIC]
            })

            st.dataframe(df_manual_metrics)

            pretty_name = {
                "SES (Simple Exponential Smoothing)": "SES",
                "Holt (Double Exponential)": "Holt",
                "Holt-Winters Additif": "HW Additif",
                "Holt-Winters Multiplicatif": "HW Multiplicatif"
            }

            chosen = pretty_name.get(model_name, model_name)
            st.success(f"🥇 **Meilleur modèle manuel : {chosen}**")

            st.session_state["metrics"] = {
               "MSE": float(MSE),
               "AIC": float(AIC),
               "AICc": float(AICc),
               "BIC": float(BIC),
            }

    except Exception as e:
        st.error(f"Erreur lors du calcul : {e}")

st.markdown("---")

# ================================================================
# SECTION 2 — GRID SEARCH AUTOMATIQUE
# ================================================================
st.header("2️⃣ Grid Search automatique (optimisation des paramètres)")

def compute_aicc(aic, n, k):
    if n - k - 1 <= 0:
        return np.nan
    return aic + (2 * k * (k + 1)) / (n - k - 1)

def grid_search(series):
    alphas = np.linspace(0.1, 0.9, 9)
    betas = np.linspace(0.1, 0.9, 9)
    gammas = np.linspace(0.1, 0.9, 9)

    results = []
    temps_par_modele = {}

    # ========== SES ==========
    best_ses = None
    best_ses_mse = float("inf")
    t0 = time.time()
    for a in alphas:
        try:
            m = SimpleExpSmoothing(series).fit(smoothing_level=a, optimized=False)
            mse = np.mean((series - m.fittedvalues) ** 2)
            if mse < best_ses_mse:
               best_ses_mse = mse
               best_ses = m
        except:
           pass
    temps_par_modele["SES"] = round(time.time() - t0, 4)

    if best_ses:
        n = len(series)
        k = 1
        results.append(["SES", best_ses_mse, best_ses.aic,
                        compute_aicc(best_ses.aic, n, k), best_ses.bic])

    # ========== HOLT ==========
    best_holt = None
    best_holt_mse = float("inf")
    t0 = time.time()
    for a in alphas:
       for b in betas:
            try:
                m = ExponentialSmoothing(series, trend="add").fit(
                   smoothing_level=a, smoothing_trend=b, optimized=False
                )
                mse = np.mean((series - m.fittedvalues) ** 2)
                if mse < best_holt_mse:
                    best_holt_mse = mse
                    best_holt = m
            except:
                pass
    temps_par_modele["Holt"] = round(time.time() - t0, 4)


    if best_holt:
        n = len(series)
        k = 2
        results.append(["Holt", best_holt_mse, best_holt.aic,
                        compute_aicc(best_holt.aic, n, k), best_holt.bic])

    # ========== HW ADDITIF ==========
    best_add = None
    best_add_mse = float("inf")
    t0 = time.time()
    for a in alphas:
        for b in betas:
            for g in gammas:
                try:
                    m = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=4).fit(
                        smoothing_level=a, smoothing_trend=b,
                        smoothing_seasonal=g, optimized=False
                    )
                    mse = np.mean((series - m.fittedvalues) ** 2)
                    if mse < best_add_mse:
                       best_add_mse = mse
                       best_add = m
                except:
                    pass
    temps_par_modele["HW Additif"] = round(time.time() - t0, 4)


    if best_add:
        n = len(series); k = 3
        results.append(["HW Additif", best_add_mse, best_add.aic,
                        compute_aicc(best_add.aic, n, k), best_add.bic])

    # ========== HW MULTIPLICATIF ==========
    best_mul = None
    best_mul_mse = float("inf")
    t0 = time.time()
    for a in alphas:
        for b in betas:
            for g in gammas:
                try:
                    m = ExponentialSmoothing(series, trend="add", seasonal="mul", seasonal_periods=4).fit(
                        smoothing_level=a, smoothing_trend=b,
                        smoothing_seasonal=g, optimized=False
                    )
                    mse = np.mean((series - m.fittedvalues) ** 2)
                    if mse < best_mul_mse:
                       best_mul_mse = mse
                       best_mul = m
                except:
                    pass
    temps_par_modele["HW Multiplicatif"] = round(time.time() - t0, 4)

    if best_mul:
        n = len(series); k = 3
        results.append(["HW Multiplicatif", best_mul_mse, best_mul.aic,
                        compute_aicc(best_mul.aic, n, k), best_mul.bic])

    return (
        pd.DataFrame(results, columns=["Modèle", "MSE", "AIC", "AICc", "BIC"]),
        best_ses,
        best_holt,
        best_add,
        best_mul,
        temps_par_modele
    )

# ------------------------------
# Bouton GRID SEARCH
# ------------------------------
if st.button("🚀 Lancer Grid Search Automatique"):

    st.session_state["t0"] = time.time()

    df_gs, m_ses, m_holt, m_add, m_mul, temps_par_modele = grid_search(series)
    st.session_state["temps_par_modele"] = temps_par_modele
    
    st.session_state["grid_results"] = df_gs
    st.session_state["best_models"] = {
        "SES": m_ses,
        "Holt": m_holt,
        "HW Additif": m_add,
        "HW Multiplicatif": m_mul
    }
    st.success("Grid Search terminé !")

# ------------------------------
# Affichage tableau Grid Search
# ------------------------------
if "grid_results" in st.session_state:
    st.subheader("📊 Résultats du Grid Search")
    st.dataframe(st.session_state["grid_results"])

    best_row = st.session_state["grid_results"].sort_values("AICc").iloc[0]
    best_model_name = best_row["Modèle"]

    if "t0" in st.session_state:
        st.session_state["temps_execution"] = round(
            time.time() - st.session_state["t0"], 4
    )


    st.success(f"🥇 Meilleur modèle optimal : **{best_model_name}**")
    # === Enregistrer pour page Résidus ===
    if "best_models" in st.session_state:
        st.session_state["best_fitted_model"] = st.session_state["best_models"][best_model_name]

st.markdown("---")

# ================================================================
# SECTION 3 — OPTIMISATION AVANCÉE (Intervalles de confiance)
# ================================================================
st.header("3️⃣ Optimisation avancée")

st.subheader("📌 Intervalles de confiance des prévisions")

if "best_fitted_model" not in st.session_state or st.session_state["best_fitted_model"] is None:
    st.warning("Aucun modèle optimal disponible. Lancez d’abord le Grid Search.")
else:
    model_opt = st.session_state["best_fitted_model"]

    # ---------------------------
    # Paramètre : Horizon
    # ---------------------------
    horizon_ci = st.number_input("Horizon des prévisions :", min_value=1, max_value=36, value=6)

    try:
        # ---------------------------------------------------------
        # Prévision brute
        # ---------------------------------------------------------
        forecast_ci = model_opt.forecast(horizon_ci)

        # Réindexation temporelle correcte
        last_date = series.index[-1]
        future_index = pd.date_range(last_date, periods=horizon_ci+1, freq="MS")[1:]
        forecast_ci.index = future_index

        # ---------------------------------------------------------
        # IC via RMSE
        # ---------------------------------------------------------
        resid = model_opt.resid
        rmse = np.sqrt(np.mean(resid**2))
        z = 1.96

        lower = forecast_ci - z * rmse
        upper = forecast_ci + z * rmse

        df_ci = pd.DataFrame({
            "Prévision": forecast_ci.values,
            "Borne inférieure (95%)": lower.values,
            "Borne supérieure (95%)": upper.values
        }, index=forecast_ci.index)

        # ---------------------------------------------------------
        #  EXPANDER 1 : Tableau IC
        # ---------------------------------------------------------
        with st.expander("📄 Tableau des intervalles de confiance (95%)", expanded=True):
            st.dataframe(df_ci)

        # ---------------------------------------------------------
        #  EXPANDER 2 : Graphique IC
        # ---------------------------------------------------------
        with st.expander("📈 Prévisions avec intervalles de confiance"):
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(series.index, series.values, label="Historique")
            ax.plot(forecast_ci.index, forecast_ci.values, marker="o", label="Prévision")

            ax.fill_between(forecast_ci.index, lower, upper,
                            color="gray", alpha=0.3, label="IC 95%")

            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur lors du calcul : {e}")

st.subheader("🤖 Optimisation bayésienne des paramètres")        
st.success(
    "Optimisation bayésienne disponible et appliquée en complément de la recherche par grille."
)

with st.expander("L'optimisation bayésienne"):

    st.write("Optimisation automatique des paramètres via un processus bayésien (Gaussian Process).")

    model_choice = st.selectbox(
        "Modèle à optimiser :",
        ["SES", "Holt", "Holt-Winters Additif", "Holt-Winters Multiplicatif"]
    )

    if st.button("🔎 Lancer optimisation bayésienne"):

        try:

            # --- Définition de la fonction objectif ---
            def objective(params):
                alpha = params[0]
                beta  = params[1] if model_choice != "SES" else None
                gamma = params[2] if "Holt-Winters" in model_choice else None

                try:
                    if model_choice == "SES":
                        model = SimpleExpSmoothing(series).fit(
                            smoothing_level=alpha, optimized=False)

                    elif model_choice == "Holt":
                        model = ExponentialSmoothing(series, trend="add").fit(
                            smoothing_level=alpha,
                            smoothing_trend=beta,
                            optimized=False)

                    elif model_choice == "Holt-Winters Additif":
                        model = ExponentialSmoothing(series, trend="add",
                                                     seasonal="add", seasonal_periods=4).fit(
                            smoothing_level=alpha,
                            smoothing_trend=beta,
                            smoothing_seasonal=gamma,
                            optimized=False)

                    else:  # Multiplicatif
                        model = ExponentialSmoothing(series, trend="add",
                                                     seasonal="mul", seasonal_periods=4).fit(
                            smoothing_level=alpha,
                            smoothing_trend=beta,
                            smoothing_seasonal=gamma,
                            optimized=False)

                    mse = np.mean((series - model.fittedvalues) ** 2)

                except Exception:
                    mse = 1e9  # Pénalité si modèle impossible

                return mse

            # --- Espace de recherche ---
            space = [Real(0.01, 1.0, name="alpha")]  # alpha toujours présent

            if model_choice != "SES":
                space.append(Real(0.01, 1.0, name="beta"))

            if "Holt-Winters" in model_choice:
                space.append(Real(0.01, 1.0, name="gamma"))

            # --- Optimisation bayésienne ---
            result = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=40,
                random_state=0
            )

            st.success("Optimisation bayésienne terminée !")

            # Affichage résultats
            best_params = result.x
            affichage = {}

            affichage["alpha"] = best_params[0]
            if model_choice != "SES":
                affichage["beta"] = best_params[1]
            if "Holt-Winters" in model_choice:
                affichage["gamma"] = best_params[-1]

            st.json(affichage)

        except Exception as e:
            st.error(f"Erreur lors de l’optimisation bayésienne : {e}")

# ================================================================
# SECTION 4 — COMPARAISON FINALE (Manuel vs Optimal)
# ================================================================
st.header("4️⃣ Comparaison finale (Manuel vs Optimal)")

if "forecast_manual" in st.session_state and "grid_results" in st.session_state:

    st.subheader("📌 Tableau comparatif AICc")

    # === AICc du modèle manuel ===
    if model_name == "SES (Simple Exponential Smoothing)":
        fit_m = SimpleExpSmoothing(series).fit(smoothing_level=alpha, optimized=False); k = 1

    elif model_name == "Holt (Double Exponential)":
        fit_m = ExponentialSmoothing(series, trend="add").fit(
            smoothing_level=alpha, smoothing_trend=beta, optimized=False
        ); k = 2

    elif model_name == "Holt-Winters Additif":
        fit_m = ExponentialSmoothing(series, trend="add", seasonal="add",
                                     seasonal_periods=4).fit(
            smoothing_level=alpha, smoothing_trend=beta,
            smoothing_seasonal=gamma, optimized=False
        ); k = 3

    else:  # HW Multiplicatif
        fit_m = ExponentialSmoothing(series, trend="add", seasonal="mul",
                                     seasonal_periods=4).fit(
            smoothing_level=alpha, smoothing_trend=beta,
            smoothing_seasonal=gamma, optimized=False
        ); k = 3

    RSS_m = np.sum((series - fit_m.fittedvalues)**2)
    n = len(series)
    AIC_m = 2*k + n*np.log(RSS_m/n)
    AICc_m = compute_aicc(AIC_m, n, k)

    # === AICc optimal depuis Grid Search ===
    df_gs = st.session_state["grid_results"]
    best_row = df_gs.sort_values("AICc").iloc[0]
    best_model_name = best_row["Modèle"]
    st.session_state["best_model_name"] = best_model_name

    model = st.session_state["best_models"][best_model_name]

    params = model.model.params
    st.session_state["best_alpha"] = params.get("smoothing_level")
    st.session_state["best_beta"]  = params.get("smoothing_trend")
    st.session_state["best_gamma"] = params.get("smoothing_seasonal")


    AICc_opt = best_row["AICc"]

    df_compare = pd.DataFrame({
        "Modèle": ["Manuel", "Optimal"],
        "AICc": [AICc_m, AICc_opt]
    })
    st.dataframe(df_compare)

    # ----------------------------------------------------------
    # Conclusion
    # ----------------------------------------------------------
    if AICc_opt < AICc_m:
        st.success("🏆 **Le modèle optimal est meilleur que le modèle manuel.**")
    else:
        st.info("ℹ️ Le modèle manuel est aussi bon ou meilleur que le modèle optimal.")
    
st.header("5️⃣ Analyse des résidus du modèle optimal")

if "best_fitted_model" not in st.session_state or st.session_state["best_fitted_model"] is None:
    st.warning("Aucun modèle optimal détecté. Lancez d’abord le Grid Search.")
else:
    model_opt = st.session_state["best_fitted_model"]

    fitted_vals = model_opt.fittedvalues
    residuals = (series - fitted_vals).dropna()

    # 🔐 Stockage GLOBAL pour la page 6
    st.session_state["residuals"] = residuals.copy()

    # =============================
    # Résidus dans le temps
    # =============================
    with st.expander("📌 Résidus dans le temps", expanded=False):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(residuals.index, residuals.values, marker="o")
        ax.axhline(0, color="red", linestyle="--")
        ax.grid(True)
        st.pyplot(fig)

    # =============================
    # Histogramme
    # =============================
    with st.expander("📊 Histogramme des résidus"):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(residuals, bins=15, edgecolor="black")
        ax.grid(True)
        st.pyplot(fig)

    # =============================
    # ACF
    # =============================
    with st.expander("🔄 Autocorrélation (ACF)"):
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_acf(residuals, ax=ax)
        st.pyplot(fig)

    st.success("Résidus calculés et enregistrés pour les tests statistiques.")

import time
from datetime import datetime

# Sécurité
if "journal" not in st.session_state:
    st.session_state["journal"] = {}

start_time = time.time()

# Modèles testés
modeles_testes = ["SES", "Holt", "HW Additif", "HW Multiplicatif"]

# Paramètres INITIAUX (avant estimation)
parametres_initiaux = {
    "SES": {"alpha": "auto"},
    "Holt": {"alpha": "auto", "beta": "auto"},
    "HW Additif": {"alpha": "auto", "beta": "auto", "gamma": "auto"},
    "HW Multiplicatif": {"alpha": "auto", "beta": "auto", "gamma": "auto"}
}

def safe_float(x):
    return float(x) if x is not None else None

parametres_optimaux = {
    "modele_optimal": st.session_state.get("best_model_name"),
    "alpha": safe_float(st.session_state.get("best_alpha")),
    "beta": safe_float(st.session_state.get("best_beta")),
    "gamma": safe_float(st.session_state.get("best_gamma"))
}

temps_execution = st.session_state.get("temps_execution")

# Écriture dans le journal
st.session_state["journal"]["modelisation"] = {
    "date_execution": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "modeles_testes": modeles_testes,
    "parametres_initiaux": parametres_initiaux,
    "parametres_optimaux": parametres_optimaux,
    "temps_execution_secondes": temps_execution
}
