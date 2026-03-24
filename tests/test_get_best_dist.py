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
        'bic': [101, 103, 106]
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
        'bic': [101, 103, 106]
    }, index=['Dist1', 'Dist2', 'Dist3'])
    
    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 30
    
    best = mock_selector.get_best_dist()
    assert len(best) == 1
    assert best.index[0] == 'Dist2'
    assert best['transp'].iloc[0] == 'pv_H0'

def test_criterio2_ninguna_cumple_adc(mock_selector):
    """Caso 3: Varias cumplen ks_pv >= 0.05, pero ninguna cumple ad_c <= 0.752."""
    df = pd.DataFrame({
        'ks_pv': [0.06, 0.07, 0.08],
        'ad_c': [0.8, 1.2, 0.9],
        'aic': [100, 102, 105],
        'bic': [101, 103, 106]
    }, index=['Dist1', 'Dist2', 'Dist3'])
    
    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 30
    
    best = mock_selector.get_best_dist()
    assert len(best) == 1
    assert best.index[0] == 'Dist1'  # Menor ad_c global de las válidas (0.8)
    assert best['transp'].iloc[0] == 'ad_cMax'

def test_criterio2_solo_una_cumple_adc(mock_selector):
    """Caso 4: Varias cumplen ks_pv >= 0.05, pero solo una cumple ad_c <= 0.752."""
    df = pd.DataFrame({
        'ks_pv': [0.06, 0.07, 0.08],
        'ad_c': [0.7, 1.2, 0.9],
        'aic': [100, 102, 105],
        'bic': [101, 103, 106]
    }, index=['Dist1', 'Dist2', 'Dist3'])
    
    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 30
    
    best = mock_selector.get_best_dist()
    assert len(best) == 1
    assert best.index[0] == 'Dist1'
    assert best['transp'].iloc[0] == 'ad_cH0'

def test_criterio3_n_mayor_40(mock_selector):
    """Caso 5: Varias cumplen ambos filtros, muestra es > 40. Usa BIC."""
    df = pd.DataFrame({
        'ks_pv': [0.06, 0.07, 0.08],
        'ad_c': [0.7, 0.6, 0.5],
        'aic': [100, 105, 110],
        'bic': [101, 102, 104]  # Mínimo BIC es 101. Dist1 y Dist2 están en rango <= +2.0
    }, index=['Dist1', 'Dist2', 'Dist3'])
    
    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 50
    
    best = mock_selector.get_best_dist()
    assert len(best) == 1
    # Entre Dist1 y Dist2, Dist2 tiene menor ad_c (0.6 vs 0.7)
    assert best.index[0] == 'Dist2'

def test_criterio3_n_menor_igual_40(mock_selector):
    """Caso 6: Varias cumplen ambos filtros, muestra es <= 40. Usa AIC."""
    df = pd.DataFrame({
        'ks_pv': [0.06, 0.07, 0.08],
        'ad_c': [0.7, 0.6, 0.5],
        'aic': [100, 101.5, 104], # Mínimo AIC es 100. Dist1 y Dist2 están en rango <= +2.0
        'bic': [101, 105, 110]
    }, index=['Dist1', 'Dist2', 'Dist3'])
    
    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 30
    
    best = mock_selector.get_best_dist()
    assert len(best) == 1
    # Entre Dist1 y Dist2, Dist2 tiene menor ad_c (0.6 vs 0.7)
    assert best.index[0] == 'Dist2'

def test_excepcion_fallback(mock_selector):
    """Caso 7: Falla la lógica por un KeyError/TypeError, se hace fallback a la de menor ad_c."""
    df = pd.DataFrame({
        'ks_pv': [0.06, 0.07, 0.08],
        'ad_c': [0.7, 0.6, 0.5],
        # Usamos cadenas de texto intencionadamente para causar un TypeError y forzar la Excepción
        'aic': ['x', 'y', 'z'], 
        'bic': ['x', 'y', 'z']
    }, index=['Dist1', 'Dist2', 'Dist3'])
    
    mock_selector.get_ranking_dataframe.return_value = df
    mock_selector.n = 30
    
    best = mock_selector.get_best_dist()
    assert len(best) == 1
    # Al fallar, debería devolver la de menor ad_c entre todas las válidas (Dist3, con 0.5)
    assert best.index[0] == 'Dist3'
