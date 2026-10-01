from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import genextreme

from hidroModelSelect import HidroModelSelector


@pytest.mark.parametrize("c", [0.2, -0.2, 0.0])
def test_gev_envia_a_adc_la_forma_sin_invertir_signo(c):
    datos = genextreme.ppf(np.linspace(0.05, 0.95, 50), c)
    selector = HidroModelSelector(datos)

    with patch.object(
        genextreme, "fit", return_value=(c, 0.0, 1.0)
    ):
        with patch.object(
            selector, "_calc_adc", wraps=selector._calc_adc
        ) as adc:
            selector.fit_distribution("GEV", genextreme)

    assert selector.fit_errors == {}
    adc.assert_called_once()
    assert adc.call_args.args[1] == "GEV"
    assert adc.call_args.args[2] == pytest.approx(c)

def test_adc_gev_coincide_con_referencia_numerica():
    c = 0.2
    datos = genextreme.ppf(np.linspace(0.05, 0.95, 50), c)
    selector = HidroModelSelector(datos)

    with patch.object(
        genextreme, "fit", return_value=(c, 0.0, 1.0)
    ):
        with patch.object(
            selector, "_calculate_anderson_stat", return_value=0.5
        ):
            selector.fit_distribution("GEV", genextreme)

    assert selector.fit_errors == {}

    # Referencia calculada por separado con aritmética decimal:
    # n=50, theta3=0.2, A²=0.5, h0=0.851.
    assert selector.results["GEV"]["adc"] == pytest.approx(
        0.307666211882245877,
        rel=1e-12,
    )
    assert selector.results["GEV"]["params"] == (
        c, 0.0, 1.0
    )
