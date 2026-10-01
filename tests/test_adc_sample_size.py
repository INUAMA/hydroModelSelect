from unittest.mock import patch
import pytest
import numpy as np
from scipy.stats import norm, genextreme, pearson3, gumbel_r, lognorm

from hidroModelSelect import HidroModelSelector


def test_normal_con_nueve_datos_conserva_ajuste_sin_adc():
    datos = norm.ppf(np.linspace(0.05, 0.95, 9))
    selector = HidroModelSelector(datos)

    with patch.object(
        selector, "_calc_adc", wraps=selector._calc_adc,
    ) as calcular_adc:
        selector.fit_distribution("Normal", norm)

    assert selector.fit_errors == {}
    assert "Normal" in selector.results
    calcular_adc.assert_not_called()

    resultado = selector.results["Normal"]
    assert np.isnan(resultado["adc"])
    assert "tamaño muestral" in resultado["adc_reason"]

    for campo in ("aic", "aicc", "bic", "a2", "ad_c"):
        assert np.isfinite(resultado[campo])

@pytest.mark.parametrize(
    "nombre, distribucion, parametros",
    [
        pytest.param(
            "GEV", genextreme, (0.2, 0.0, 1.0),
            id="gev",
        ),
        pytest.param(
            "Pearson3", pearson3, (1.0, 0.0, 1.0),
            id="pearson3",
        ),
    ],
)
def test_familia_tres_parametros_con_19_datos_no_calcula_adc(
    nombre, distribucion, parametros,
):
    datos = distribucion.ppf(
        np.linspace(0.05, 0.95, 19), *parametros,
    )
    selector = HidroModelSelector(datos)

    with patch.object(
        distribucion, "fit", return_value=parametros,
    ):
        with patch.object(
            selector, "_calc_adc", wraps=selector._calc_adc,
        ) as calcular_adc:
            selector.fit_distribution(nombre, distribucion)

    assert selector.fit_errors == {}
    assert nombre in selector.results
    calcular_adc.assert_not_called()

    resultado = selector.results[nombre]
    assert resultado["params"] == parametros
    assert resultado["k_params"] == 3
    assert np.isnan(resultado["adc"])
    assert "tamaño muestral" in resultado["adc_reason"]

    for campo in ("aic", "aicc", "bic", "a2", "ad_c"):
        assert np.isfinite(resultado[campo])

@pytest.mark.parametrize(
    "nombre, distribucion, parametros, n",
    [
        pytest.param(
            "Normal", norm, (0.0, 1.0), 10, id="normal",
        ),
        pytest.param(
            "Gumbel", gumbel_r, (0.0, 1.0), 10, id="gumbel",
        ),
        pytest.param(
            "Lognormal", lognorm, (0.5, 0.0, 2.0), 10,
            id="lognormal",
        ),
        pytest.param(
            "GEV", genextreme, (0.2, 0.0, 1.0), 20, id="gev",
        ),
        pytest.param(
            "Pearson3", pearson3, (1.0, 0.0, 1.0), 20,
            id="pearson3",
        ),
    ],
)
def test_adc_disponible_en_tamano_minimo(
    nombre, distribucion, parametros, n,
):
    datos = distribucion.ppf(
        np.linspace(0.05, 0.95, n), *parametros,
    )
    selector = HidroModelSelector(datos)

    with patch.object(
        distribucion, "fit", return_value=parametros,
    ):
        with patch.object(
            selector, "_calc_adc", wraps=selector._calc_adc,
        ) as calcular_adc:
            selector.fit_distribution(nombre, distribucion)

    assert selector.fit_errors == {}
    calcular_adc.assert_called_once()

    resultado = selector.results[nombre]
    assert np.isfinite(resultado["adc"])
    assert resultado["adc_reason"] is None
