import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from scipy import stats

def test_adfuller(series, autolag='AIC'):
    """
    Test de Dickey-Fuller augmenté (ADF)
    Retourne un dictionnaire avec les résultats
    """
    result = adfuller(series.dropna(), autolag=autolag)
    
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Number of Lags Used': result[2],
        'Number of Observations': result[3],
        'Critical Values': result[4],
        'Stationnaire': result[1] < 0.05
    }

def test_kpss(series, regression='c'):
    """
    Test KPSS pour la stationnarité
    """
    result = kpss(series.dropna(), regression=regression)
    
    return {
        'KPSS Statistic': result[0],
        'p-value': result[1],
        'Number of Lags': result[2],
        'Critical Values': result[3],
        'Stationnaire': result[1] > 0.05
    }

def calculer_acf_pacf(series, nlags=40):
    """
    Calcule les fonctions ACF et PACF
    Retourne deux listes : acf_values, pacf_values
    """
    # ACF simple
    n = len(series)
    series_clean = series.dropna()
    mean = series_clean.mean()
    var = series_clean.var()
    
    acf_values = []
    for lag in range(min(nlags, n-1)):
        numerator = ((series_clean.iloc[lag:] - mean) * (series_clean.iloc[:n-lag] - mean)).sum()
        denominator = (n - lag) * var if var != 0 else 1
        acf_values.append(numerator / denominator)
    
    # PACF via statsmodels
    from statsmodels.tsa.stattools import pacf
    pacf_values = pacf(series_clean, nlags=nlags)
    
    return acf_values, pacf_values

def decomposer_serie(series, period=12, model='additive'):
    """
    Décomposition d'une série temporelle
    Retourne un dictionnaire avec les composantes
    """
    result = seasonal_decompose(series.dropna(), model=model, period=period, extrapolate_trend='freq')
    
    return {
        'observed': result.observed,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'resid': result.resid,
        'model': model
    }

def decomposition_additive(series, period):
    """
    Décomposition additive (pour compatibilité)
    Retourne: trend, seasonal, resid
    """
    decomposition = decomposer_serie(series, period=period, model='additive')
    return decomposition['trend'], decomposition['seasonal'], decomposition['resid']

def decomposition_multiplicative(series, period):
    """
    Décomposition multiplicative
    Retourne: trend, seasonal, resid
    """
    decomposition = decomposer_serie(series, period=period, model='multiplicative')
    return decomposition['trend'], decomposition['seasonal'], decomposition['resid']

def test_additive_vs_multiplicative(series, period):
    """
    Test pour déterminer si la saisonnalité est additive ou multiplicative
    """
    # Calcul des moyennes et écarts-types par période
    n_periods = len(series) // period
    seasonal_data = []
    
    for i in range(period):
        values = []
        for j in range(n_periods):
            idx = i + j * period
            if idx < len(series):
                val = series.iloc[idx]
                if not np.isnan(val):
                    values.append(val)
        if values:  # seulement si on a des valeurs
            seasonal_data.append(values)
    
    if not seasonal_data:
        return {
            'moyennes': [],
            'ecarts_type': [],
            'a': 0,
            'b': 0,
            'nature': "Indéterminé",
            'coeff_variation_moyen': 0
        }
    
    moyennes = [np.mean(vals) for vals in seasonal_data]
    ecarts_type = [np.std(vals) for vals in seasonal_data]
    
    # Calcul des coefficients de variation
    coeff_var = []
    for mean, std in zip(moyennes, ecarts_type):
        if mean != 0 and not np.isnan(mean) and not np.isnan(std):
            coeff_var.append(std / mean)
    
    if not coeff_var:
        avg_coeff_var = 0
    else:
        avg_coeff_var = np.nanmean(coeff_var)
    
    # Règle de décision
    if avg_coeff_var < 0.1 or np.isnan(avg_coeff_var):
        nature = "Additive"
        a = 0
        b = avg_coeff_var if not np.isnan(avg_coeff_var) else 0
    else:
        nature = "Multiplicative"
        a = avg_coeff_var
        b = 0
    
    return {
        'moyennes': moyennes,
        'ecarts_type': ecarts_type,
        'a': a,
        'b': b,
        'nature': nature,
        'coeff_variation_moyen': avg_coeff_var
    }

def calculer_statistiques_descriptives(series):
    """
    Calcule les statistiques descriptives d'une série
    """
    series_clean = series.dropna()
    
    if len(series_clean) == 0:
        return {}
    
    # Statistiques de base
    stats_dict = {
        'count': len(series_clean),
        'mean': series_clean.mean(),
        'std': series_clean.std(),
        'min': series_clean.min(),
        '25%': series_clean.quantile(0.25),
        '50%': series_clean.quantile(0.50),
        '75%': series_clean.quantile(0.75),
        'max': series_clean.max(),
        'range': series_clean.max() - series_clean.min(),
        'variance': series_clean.var(),
        'skewness': series_clean.skew(),
        'kurtosis': series_clean.kurtosis(),
        'cv': (series_clean.std() / series_clean.mean() * 100) if series_clean.mean() != 0 else 0
    }
    
    # Tests de normalité
    try:
        if len(series_clean) >= 3 and len(series_clean) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(series_clean)
            stats_dict['shapiro_stat'] = shapiro_stat
            stats_dict['shapiro_p'] = shapiro_p
            stats_dict['normal'] = shapiro_p > 0.05
        else:
            stats_dict['shapiro_stat'] = None
            stats_dict['shapiro_p'] = None
            stats_dict['normal'] = None
    except:
        stats_dict['shapiro_stat'] = None
        stats_dict['shapiro_p'] = None
        stats_dict['normal'] = None
    
    return stats_dict

def detecter_saisonnalite(series, max_period=24):
    """
    Détecte automatiquement la période saisonnière
    """
    series_clean = series.dropna()
    n = len(series_clean)
    
    if n < 10:
        return {
            'period_detectee': None,
            'peaks_autocorr': [],
            'autocorr_values': []
        }
    
    # Méthode 1: Autocorrélation
    autocorr = []
    mean = series_clean.mean()
    var = series_clean.var()
    
    for lag in range(1, min(max_period*2, n//2)):
        numerator = ((series_clean.iloc[lag:] - mean) * (series_clean.iloc[:n-lag] - mean)).sum()
        denominator = (n - lag) * var if var != 0 else 1
        autocorr.append(numerator / denominator)
    
    # Chercher les pics dans l'autocorrélation
    autocorr_array = np.array(autocorr)
    peaks = []
    for i in range(1, len(autocorr_array)-1):
        if (autocorr_array[i] > autocorr_array[i-1] and 
            autocorr_array[i] > autocorr_array[i+1] and 
            autocorr_array[i] > 0.3):
            peaks.append(i+1)  # +1 car on commence à lag=1
    
    # Méthode 2: Prendre la période avec la plus forte autocorrélation
    best_period = None
    if peaks:
        # Prendre le pic avec la plus forte autocorrélation
        peak_values = [autocorr_array[p-1] for p in peaks]
        best_idx = np.argmax(peak_values)
        best_period = peaks[best_idx]
        
        # Vérifier si c'est un multiple d'une période plus petite
        for p in [2, 3, 4, 6, 7, 12, 24]:
            if best_period % p == 0 and best_period > p:
                best_period = p
                break
    
    return {
        'period_detectee': best_period,
        'peaks_autocorr': peaks,
        'autocorr_values': autocorr_array.tolist()
    }

def plot_acf_pacf(series, lags=40):
    """
    Génère les graphiques ACF et PACF
    Retourne une figure matplotlib
    """
    series_clean = series.dropna()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # ACF
    plot_acf(series_clean, lags=min(lags, len(series_clean)//2), ax=ax1)
    ax1.set_title('Autocorrelation Function (ACF)')
    ax1.grid(True, alpha=0.3)
    
    # PACF
    plot_pacf(series_clean, lags=min(lags, len(series_clean)//2), ax=ax2)
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def generer_rapport_stationnarite(series):
    """
    Génère un rapport complet de stationnarité
    """
    # Tests
    adf_result = test_adfuller(series)
    kpss_result = test_kpss(series)
    
    # Détection saisonnalité
    saisonnalite = detecter_saisonnalite(series)
    
    # Statistiques
    stats = calculer_statistiques_descriptives(series)
    
    # Conclusion
    is_adf_stationary = adf_result['p-value'] < 0.05
    is_kpss_stationary = kpss_result['p-value'] > 0.05
    
    if is_adf_stationary and is_kpss_stationary:
        conclusion_stationnarite = "Série stationnaire"
        recommandation = "Pas de différenciation nécessaire"
    elif not is_adf_stationary and not is_kpss_stationary:
        conclusion_stationnarite = "Série non stationnaire"
        recommandation = "Différenciation recommandée (d=1)"
    else:
        conclusion_stationnarite = "Résultats contradictoires"
        recommandation = "Différenciation prudente recommandée"
    
    return {
        'adf_test': adf_result,
        'kpss_test': kpss_result,
        'saisonnalite': saisonnalite,
        'statistiques': stats,
        'conclusion_stationnarite': conclusion_stationnarite,
        'recommandation': recommandation,
        'periode_saisonniere': saisonnalite['period_detectee']
    }

# Exemple d'utilisation
if __name__ == "__main__":
    print("=" * 60)
    print("Module Tests_Stationnartié.py")
    print("=" * 60)
    print("\nFonctions disponibles :")
    print("1.  test_adfuller(series) - Test ADF")
    print("2.  test_kpss(series) - Test KPSS")
    print("3.  calculer_acf_pacf(series) - Calcule ACF et PACF")
    print("4.  decomposer_serie(series, period) - Décomposition")
    print("5.  decomposition_additive(series, period)")
    print("6.  decomposition_multiplicative(series, period)")
    print("7.  test_additive_vs_multiplicative(series, period)")
    print("8.  calculer_statistiques_descriptives(series)")
    print("9.  detecter_saisonnalite(series)")
    print("10. plot_acf_pacf(series) - Graphiques ACF/PACF")
    print("11. generer_rapport_stationnarite(series) - Rapport complet")
    print("\n" + "=" * 60)
    
    # Créer des données de test
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    test_series = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
    
    print("\nTest avec données aléatoires :")
    print(f"- Longueur : {len(test_series)}")
    print(f"- Moyenne : {test_series.mean():.2f}")
    print(f"- Écart-type : {test_series.std():.2f}")
    
    # Test ADF
    adf = test_adfuller(test_series)
    print(f"\nTest ADF :")
    print(f"- p-value : {adf['p-value']:.4f}")
    print(f"- Stationnaire : {adf['Stationnaire']}")