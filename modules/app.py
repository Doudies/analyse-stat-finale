import streamlit as st
import os

# Configuration de la page
st.set_page_config(
    page_title="Analyse Statistique Avancée",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'interface
def load_css():
    st.markdown("""
    <style>
        /* Thème principal */
        :root {
            --primary: #6C63FF;
            --secondary: #FF6584;
            --dark: #0F172A;
            --light: #F8FAFC;
        }
        
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        }
        
        /* Cards */
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin: 1rem 0;
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }
        
        /* Boutons */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 50px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: rgba(15, 23, 42, 0.9);
        }
        
        /* File upload */
        .uploadedFile {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        /* Titres */
        h1 {
            background: linear-gradient(135deg, #667eea 0%, #f5576c 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        
        /* Métriques */
        .stMetric {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Charger le CSS
load_css()

# Header avec animation
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1 style='font-size: 3.5rem; margin-bottom: 1rem;'>📊 DataAnalyzer Pro</h1>
        <p style='font-size: 1.2rem; opacity: 0.9;'>
            Analyse statistique avancée • Régression • Tests • Prévisions
        </p>
    </div>
    """, unsafe_allow_html=True)

# Introduction avec features
st.markdown("---")

# Features en cards
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown("""
        <div class='card'>
            <h3>📈 Régression Linéaire</h3>
            <p>Analyse complète avec métriques et visualisations interactives</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown("""
        <div class='card'>
            <h3>🧪 Tests Statistiques</h3>
            <p>Test t de Student, ANOVA, tests de stationnarité ADF/KPSS</p>
        </div>
        """, unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown("""
        <div class='card'>
            <h3>🔮 Prévisions</h3>
            <p>Modèles Holt-Winters, lissage exponentiel, intervalles de confiance</p>
        </div>
        """, unsafe_allow_html=True)

# Instructions d'utilisation
st.markdown("""
## 🚀 Comment utiliser l'application

1. **Importation** (page 1) : Chargez votre fichier CSV/Excel
2. **Analyse exploratoire** (page 2) : Visualisez vos données
3. **Tests de stationnarité** (page 3) : Vérifiez les propriétés de la série
4. **Modèles classiques** (page 4) : Moyenne mobile et régression
5. **Modélisation** (page 5) : Modèles avancés et prévisions
6. **Validation** (page 6) : Tests et validation croisée

---

## 📁 Formats supportés

- CSV (séparateur point-virgule ou virgule)
- Excel (.xlsx, .xls)
- Données copiées directement
""")

# Footer
st.markdown("""
---
<div style='text-align: center; padding: 2rem;'>
    <p style='opacity: 0.7;'>
        Projet Master ROMARIN • Application d'Analyse Statistique
    </p>
</div>
""", unsafe_allow_html=True)