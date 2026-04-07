import numpy as np
import pandas as pd
import scipy.stats as st
import pytest
from hidroModelSelect.distCompare import HidroModelSelector

@pytest.fixture
def sample_data():
    """Provee un conjunto de datos de ejemplo (ej. precipitaciones máximas anuales)."""
    return np.array([45.2, 56.3, 34.1, 78.5, 65.0, 52.1, 48.9, 61.2, 55.4, 41.2])

def test_initialization(sample_data):
    """Verifica que el selector se inicializa correctamente y ordena los datos."""
    selector = HidroModelSelector(sample_data)
    assert len(selector.obs_sort) == len(sample_data)
    assert np.all(np.diff(selector.obs_sort) >= 0)  # Verifica que estén ordenados de menor a mayor
    assert selector.n == len(sample_data)
    assert selector.results == {}

def test_fit_distribution(sample_data):
    """Verifica el ajuste de una distribución individual y el cálculo de sus estadísticos."""
    selector = HidroModelSelector(sample_data)
    selector.fit_distribution('Normal', st.norm)
    
    assert 'Normal' in selector.results
    res = selector.results['Normal']
    
    # Comprobar que los estadísticos clave están presentes
    expected_keys = ['aic', 'aicc', 'bic', 'a2', 'adc', 'ks', 'ks_pv', 'params', 'k_params', 'd_aicc']
    for key in expected_keys:
        assert key in res
        
    # Verificar que las métricas principales no son nulas
    assert not np.isnan(res['aic'])
    assert not np.isnan(res['a2'])
    assert len(res['params']) > 0

def test_ranking_dataframe(sample_data):
    """Verifica que el ranking se genera correctamente y está ordenado por AICc."""
    selector = HidroModelSelector(sample_data)
    selector.fit_distribution('Normal', st.norm)
    selector.fit_distribution('Gumbel', st.gumbel_r)
    
    df = selector.get_ranking_dataframe()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.iloc[0]['aicc'] <= df.iloc[1]['aicc']  # Debe estar ordenado de menor a mayor AICc
    
    # Probar el método de la mejor distribución
    best_df = selector.get_best_dist()
    assert not best_df.empty
    assert len(best_df) == 1  # Debe retornar solo la fila ganadora
    
    # Verificar la nueva métrica de trazabilidad
    assert 'transp' in best_df.columns
    assert best_df.iloc[0]['transp'] in ['pv_max', 'pv_H0', 'ad_cMax', 'ad_cH0', 'optima_ci', 'optima_ci_fallback']
    
    # Verificar que las no ganadoras también conservan la trazabilidad
    for model, res in selector.results.items():
        assert 'transp' in res
        assert res['transp'] != ''