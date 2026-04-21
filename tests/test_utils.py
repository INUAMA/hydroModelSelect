import numpy as np
import pytest
from hidroModelSelect.distCompare import HidroModelSelector

def test_pbias_general():
    """Prueba un caso general con omisión y comisión."""
    obs = np.array([10, 20, 30, 40])
    sim = np.array([5, 25, 30, 50]) # Subestima, Sobreestima, Perfecto, Sobreestima
    
    # diff = [5, -5, 0, -10]
    # sum(obs) = 100
    # sum(diff) = -10 -> pbias_total = -10%
    # sum(omission) = 5 -> pbias_omision = 5%
    # sum(commission) = -15 -> pbias_comision = -15%
    
    total, omision, comision = HidroModelSelector.pbias_desglosado(obs, sim)
    
    assert total == pytest.approx(-10.0)
    assert omision == pytest.approx(5.0)
    assert comision == pytest.approx(-15.0)

def test_pbias_perfect_model():
    """Prueba un modelo perfecto donde obs y sim son idénticos."""
    obs = np.array([10, 20, 30])
    sim = np.array([10, 20, 30])
    total, omision, comision = HidroModelSelector.pbias_desglosado(obs, sim)
    assert total == 0.0
    assert omision == 0.0
    assert comision == 0.0

def test_pbias_zero_observation():
    """Prueba el caso extremo donde la suma de observaciones es cero."""
    obs = np.array([0, 0, 0])
    sim = np.array([1, 2, 3])
    total, omision, comision = HidroModelSelector.pbias_desglosado(obs, sim)
    assert np.isnan(total)
    assert np.isnan(omision)
    assert np.isnan(comision)