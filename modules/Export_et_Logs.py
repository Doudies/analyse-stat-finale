import streamlit as st
import pandas as pd
import json
from datetime import datetime
import zipfile
import io
import uuid

st.title("📤 Export & Logs")
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
# =========================
# Sécurité
# =========================
if "series" not in st.session_state:
    st.warning("Aucune série disponible.")
    st.stop()

# Initialisation du journal si absent
if "journal" not in st.session_state:
    st.session_state["journal"] = {}
    
st.session_state["journal"].setdefault(
    "session",
    {
        "session_id": str(uuid.uuid4()),
        "date_debut": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
)
# =========================
# Export Moyenne Mobile
# =========================
st.header("📊 Export Moyenne Mobile")

if "ma_results" in st.session_state:
    st.download_button(
        label="⬇️ Télécharger la moyenne mobile (CSV)",
        data=st.session_state["ma_results"].to_csv(index=False, sep=";").encode("utf-8"),
        file_name="moyenne_mobile.csv",
        mime="text/csv"
    )
else:
    st.info("Aucun tableau de moyenne mobile à exporter.")

# =========================
# EXPORT RÉGRESSION LINÉAIRE
# =========================
st.header("📊 Export Régression Linéaire")

if "linreg_results" in st.session_state:
    st.download_button(
        label="⬇️ Télécharger les prévisions de la régression linéaire (CSV)",
        data=st.session_state["linreg_results"].to_csv(
            index=False, sep=";"
        ).encode("utf-8"),
        file_name="regression_lineaire_previsions.csv",
        mime="text/csv"
    )
else:
    st.info("Aucun résultat de régression linéaire à exporter.")

# =========================
# EXPORT DES PRÉVISIONS 
# =========================
st.header("📁 Export des prévisions")

if "forecast_manual" not in st.session_state:
    st.warning("Aucune prévision disponible à exporter.")
    st.stop()

horizon = len(st.session_state["forecast_manual"])
last_date = st.session_state["series"].index[-1]

future_dates = pd.date_range(
    last_date,
    periods=horizon + 1,
    freq="MS"
)[1:]

export_data = pd.DataFrame({
    "Horizon": range(1, len(st.session_state["forecast_manual"]) + 1),
    "Prévision": st.session_state["forecast_manual"]
})

st.download_button(
    label="⬇️ Télécharger les prévisions (CSV)",
    data=export_data.to_csv(index=False, sep=";").encode("utf-8"),
    file_name="resultats_prevision.csv",
    mime="text/csv"
)

# =========================
# EXPORT GRID SEARCH
# =========================
st.header("📊 Export Grid Search")

if "grid_results" in st.session_state:
    st.download_button(
        label="⬇️ Télécharger les résultats du Grid Search (CSV)",
        data=st.session_state["grid_results"].to_csv(
            index=False, sep=";"
        ).encode("utf-8"),
        file_name="grid_search_resultats.csv",
        mime="text/csv"
    )
else:
    st.info("Aucun résultat de Grid Search à exporter.")

# =========================
# JOURNAL DE PRÉVISION
# =========================

st.session_state["journal"]["serie_temporelle"] = (
    st.session_state.get("serie_description")
)

st.session_state["journal"]["prevision"] = {
    "previsions": "ponctuelles + IC 95%",
    "metriques": st.session_state.get("metrics"),
    "exports": {
        "csv": "resultats_prevision.csv",
        "json": "journal_execution.json"
    },
    "visualisations": [
        "historique + prevision",
        "intervalles de confiance",
        "residus"
    ]
}
# =========================
# TEMPS D'EXÉCUTION PAR MODÈLE
# =========================
st.header("⏱ Temps d'exécution par modèle")

if "temps_par_modele" in st.session_state:
    st.table(
        pd.DataFrame(
            list(st.session_state["temps_par_modele"].items()),
            columns=["Modèle", "Temps (secondes)"]
        )
    )
    st.session_state["journal"]["temps_execution_par_modele"] = (
    st.session_state["temps_par_modele"]
)
else:
    st.info("Aucun temps d'exécution par modèle disponible.")

st.session_state["journal"]["optimisation"] = {
    "critere": "AICc",
    "modele_retenu": st.session_state.get("best_model_name"),
    "autres_modeles": [
        m for m in st.session_state.get("modeles_testes", [])
        if m != st.session_state.get("best_model_name")
    ]
}

# =========================
# LOGS
# =========================
st.header("🧾 Journal d'exécution")

journal = st.session_state["journal"]
journal["date_export"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

st.json(journal)

# ---- BOUTON JSON ----
st.download_button(
    label="⬇️ Télécharger le journal d'exécution (JSON)",
    data=json.dumps(journal, indent=4),
    file_name="journal_execution.json",
    mime="application/json"
)
# =========================
# EXPORT ARCHIVE ZIP
# =========================
st.header("📦 Export global (ZIP)")

zip_buffer = io.BytesIO()

with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:

    # Journal JSON
    if "journal" in st.session_state:
        zipf.writestr(
            "journal_execution.json",
            json.dumps(st.session_state["journal"], indent=4)
        )

    # Prévisions
    if "forecast_manual" in st.session_state:
        df_prev = pd.DataFrame({
            "Horizon": range(1, len(st.session_state["forecast_manual"]) + 1),
            "Prévision": st.session_state["forecast_manual"]
        })
        zipf.writestr(
            "resultats_prevision.csv",
            df_prev.to_csv(index=False, sep=";")
        )

    # Grid Search
    if "grid_results" in st.session_state:
        zipf.writestr(
            "grid_search_resultats.csv",
            st.session_state["grid_results"].to_csv(index=False, sep=";")
        )

    # Moyenne mobile
    if "ma_results" in st.session_state:
        zipf.writestr(
            "moyenne_mobile.csv",
            st.session_state["ma_results"].to_csv(index=False, sep=";")
        )

    # Régression linéaire
    if "linreg_results" in st.session_state:
        zipf.writestr(
            "regression_lineaire_previsions.csv",
            st.session_state["linreg_results"].to_csv(index=False, sep=";")
        )

zip_buffer.seek(0)

st.download_button(
    label="⬇️ Télécharger l’archive complète (ZIP)",
    data=zip_buffer,
    file_name="resultats_complets.zip",
    mime="application/zip"
)
