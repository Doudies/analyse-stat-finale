import streamlit as st
import pandas as pd
import sys, os
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

from src.exploration.analysis import describe_series, plot_series

st.title("📊 Analyse Exploratoire de la Série Temporelle")

# Vérifier qu'une série a été chargée
if "series" not in st.session_state:
    st.warning("Veuillez d'abord importer une série dans l'onglet **1. Importation**.")
    st.stop()

series = st.session_state["series"]

# ---------------------
# 1. Statistiques
# ---------------------
st.subheader("📌 Statistiques Descriptives")
stats = describe_series(series)

df_stats = pd.DataFrame.from_dict(stats, orient="index", columns=["Valeur"])
st.table(df_stats)

# ---------------------
# 2. Visualisation
# ---------------------
st.subheader("📉 Visualisation de la Série")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(series.index, series.values, label="Série")
ax.grid(True)
plt.tight_layout()

st.pyplot(fig)

# ---------------------
# 3. Skewness / Kurtosis
# ---------------------
st.subheader("📐 Shape de la Distribution")

skew_val = float(series.skew())
kurt_val = float(series.kurt())

col1, col2 = st.columns(2)
col1.metric("Skewness (Asymétrie)", f"{skew_val:.3f}")
col2.metric("Kurtosis (Aplatissement)", f"{kurt_val:.3f}")
