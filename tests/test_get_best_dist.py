import pandas as pd
import pytest
from unittest.mock import MagicMock
from hidroModelSelect.distCompare import HidroModelSelector

@pytest.fixture
def mock_selector():
    """Crea una instancia de HidroModelSelector mockeando el dataframe resultante."""
    # Se pasan datos ficticios, ya que no vamos a realizar un ajuste matemático aquí
    selector = HidroModelSelector([10.0, 20.0, 30.0])
    selector.get_ranking_dataframe = MagicMock()
    return selector

def test_criterio1_ninguna_valida(mock_selector):
    """Caso 1: Ninguna distribución cumple ks_pv >= 0.05. Retorna la de mayor ks_pv."""
    df = pd.DataFrame({
        'ks_pv': [0.01, 0.04, 0.02],
        'ad_c': [1.0, 1.2, 0.9],
        'aic': [100, 102, 105],
        'aicc': [100.5, 102.5, 105.5],
        'bic': [101, 103, 106],
        'params': [[1, 2], [1, 2, 3], [1, 2]], # Dist2 con 3 parámetros
        'k_params': [2, 3, 2]
    }, index=['Dist1', 'Dist2', 'Dist3'])
    
    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 30
    
    best = mock_selector.get_best_dist()
    assert len(best) == 1
    assert best.index[0] == 'Dist2'  # Mayor ks_pv (0.04)
    assert best['transp'].iloc[0] == 'pv_max'

def test_criterio1_solo_una_valida(mock_selector):
    """Caso 2: Solo una distribución cumple ks_pv >= 0.05."""
    df = pd.DataFrame({
        'ks_pv': [0.01, 0.06, 0.02],
        'ad_c': [1.0, 1.2, 0.9],
        'aic': [100, 102, 105],
        'aicc': [100.5, 102.5, 105.5],
        'bic': [101, 103, 106],
        'params': [[1, 2, 3], [1, 2], [1, 2]], # Dist1 con 3 parámetros
        'k_params': [3, 2, 2]
    }, index=['Dist1', 'Dist2', 'Dist3'])
    
    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 30
    
    best = mock_selector.get_best_dist()
    assert len(best) == 1
    assert best.index[0] == 'Dist2'
    assert best['transp'].iloc[0] == 'pv_H0'

def test_criterio2_ninguna_cumple_adc(mock_selector):
    """Caso 3: Varias cumplen ks_pv >= 0.05, pero ninguna cumple el valor crítico de ad_c."""
    df = pd.DataFrame({
        'ks_pv': [0.06, 0.07, 0.08],
        'ad_c': [0.85, 1.2, 0.9],
        'aic': [100, 102, 105],
        'aicc': [100.5, 102.5, 105.5],
        'bic': [101, 103, 106],
        'params': [[1, 2], [1, 2], [1, 2, 3]], # Dist3 con 3 parámetros
        'k_params': [2, 2, 3]
    }, index=['Dist1', 'Dist2', 'Dist3'])
    
    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 30
    
    best = mock_selector.get_best_dist()
    assert len(best) == 1
    assert best.index[0] == 'Dist1'  # Menor ad_c global de las válidas (0.85)
    assert best['transp'].iloc[0] == 'ad_cMax'

def test_criterio2_solo_una_cumple_adc(mock_selector):
    """Caso 4: Varias cumplen ks_pv >= 0.05, pero solo una cumple ad_c <= 0.752."""
    df = pd.DataFrame({
        'ks_pv': [0.06, 0.07, 0.08],
        'ad_c': [0.7, 1.2, 0.9],
        'aic': [100, 102, 105],
        'aicc': [100.5, 102.5, 105.5],
        'bic': [101, 103, 106],
        'params': [[1, 2], [1, 2], [1, 2]],
        'k_params': [2, 2, 2]
    }, index=['Dist1', 'Dist2', 'Dist3'])
    
    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 30
    
    best = mock_selector.get_best_dist()
    assert len(best) == 1
    assert best.index[0] == 'Dist1'
    assert best['transp'].iloc[0] == 'ad_cH0'

def test_criterio3_n_mayor_40(mock_selector):
    """Caso 5: Selecciona con BIC solicitado explícitamente."""
    df = pd.DataFrame({
        'ks_pv': [0.06, 0.07, 0.08],
        'ad_c': [0.7, 0.6, 0.5],
        'aic': [100, 105, 110],
        'aicc': [100.5, 105.5, 110.5],
        'bic': [101, 102, 104],  # Mínimo BIC es 101. Dist1 y Dist2 están en rango <= +2.0
        'params': [[1, 2], [1, 2], [1, 2]],
        'k_params': [2, 2, 2]
    }, index=['Dist1', 'Dist2', 'Dist3'])
    
    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 80
    
    best = mock_selector.get_best_dist(criterion="bic")
    assert len(best) == 1
    # Entre Dist1 y Dist2, Dist2 tiene menor ad_c (0.6 vs 0.7)
    assert best.index[0] == 'Dist2'

def test_criterio3_n_menor_igual_40(mock_selector):
    """Caso 6: Selecciona con AICc como criterio predeterminado."""
    df = pd.DataFrame({
        'ks_pv': [0.06, 0.07, 0.08],
        'ad_c': [0.7, 0.6, 0.5],
        'aic': [100, 101.5, 104], # Mínimo AIC es 100. Dist1 y Dist2 están en rango <= +2.0
        'aicc': [100.5, 102.0, 104.5], # Mínimo AICc es 100.5. Dist1 y Dist2 en rango <= 2.0
        'bic': [101, 105, 110],
        'params': [[1, 2], [1, 2], [1, 2]],
        'k_params': [2, 2, 2]
    }, index=['Dist1', 'Dist2', 'Dist3'])
    
    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 30
    
    best = mock_selector.get_best_dist()
    assert len(best) == 1
    # Entre Dist1 y Dist2, Dist2 tiene menor ad_c (0.6 vs 0.7)
    assert best.index[0] == 'Dist2'

def test_criterios_no_numericos_rechazan_seleccion(mock_selector):
    """Rechaza la selección cuando ningún criterio es numérico y finito."""
    df = pd.DataFrame({
        'ks_pv': [0.06, 0.07, 0.08],
        'ad_c': [0.7, 0.6, 0.5],
        # Usamos cadenas de texto intencionadamente para causar un TypeError y forzar la Excepción
        'aic': ['x', 'y', 'z'], 
        'aicc': ['x', 'y', 'z'],
        'bic': ['x', 'y', 'z'],
        'params': [[1, 2], [1, 2], [1, 2]],
        'k_params': [2, 2, 2]
    }, index=['Dist1', 'Dist2', 'Dist3'])
    
    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 30
    
    with pytest.raises(RuntimeError, match="criterio"):
        mock_selector.get_best_dist()

def test_criterio3_mixed_params(mock_selector):
    """Utiliza AICc común con candidatos de distinto número de parámetros."""
    df = pd.DataFrame({
        'ks_pv': [0.10, 0.15, 0.20],
        'ad_c': [0.5, 0.4, 0.6],
        'aic': [100, 105, 110],
        'aicc': [100.5, 105.5, 110.5],
        'bic': [108, 102, 104],
        'params': [[1, 2], [1, 2, 3], [1, 2]], # Dist2 tiene 3 parámetros
        'k_params': [2, 3, 2]
    }, index=['Dist1', 'Dist2', 'Dist3'])

    mock_selector.get_ranking_dataframe.return_value = df

    # Con n = 90:
    mock_selector.n = 90

    # AICc común: [100.5, 105.5, 110.5].
    # Solo Dist1 queda dentro del margen de 2 puntos.
    best = mock_selector.get_best_dist()

    assert len(best) == 1
    assert best.index[0] == "Dist1"
    assert best["metri"].iloc[0] == pytest.approx(100.5)
    assert best["transp"].iloc[0] == "optima_ci"


def test_sin_columna_k_params(mock_selector):
    """Rama else línea 400: DataFrame sin columna k_params, fallback a len(params)."""
    df = pd.DataFrame({
        'ks_pv': [0.06, 0.07, 0.08],
        'ad_c': [0.7, 0.6, 0.5],
        'aic': [100, 101.5, 104],
        'aicc': [100.5, 102.0, 104.5],
        'bic': [101, 105, 110],
        'params': [[1, 2], [1, 2, 3], [1, 2]],
    }, index=['Dist1', 'Dist2', 'Dist3'])

    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 30

    best = mock_selector.get_best_dist()
    assert len(best) == 1
    assert best.index[0] == 'Dist2'
    assert best['transp'].iloc[0] == 'optima_ci'


def test_bic_todas_n_mayor_40(mock_selector):
    """Caso: El uso de BIC se solicita expresamente mediante `criterion="bic"`."""
    df = pd.DataFrame({
        'ks_pv': [0.06, 0.07, 0.08],
        'ad_c': [0.7, 0.6, 0.5],
        'aic': [100, 101.5, 104],
        'aicc': [100.5, 102.0, 104.5],
        'bic': [101, 105, 110],
        'params': [[1, 2], [1, 2], [1, 2]],
        'k_params': [2, 2, 2]
    }, index=['Dist1', 'Dist2', 'Dist3'])

    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 80

    best = mock_selector.get_best_dist(criterion="bic")
    assert len(best) == 1
    # BIC: [101, 105, 110], min=101. Óptimas: solo Dist1 (101 <= 103).
    # Como Dist1 es la única que pasa Criterio 3 -> optima_ci
    assert best.index[0] == 'Dist1'
    assert best['transp'].iloc[0] == 'optima_ci'
