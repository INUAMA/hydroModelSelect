from unittest.mock import patch

import pytest

from hidroModelSelect import HidroModelSelector
from sqrt_etmax import sqrt_etmax
from scipy.stats import norm, uniform


def test_ad_conserva_contribuciones_de_colas_extremas():
    """El cálculo conserva las contribuciones de ambas colas."""
    selector = HidroModelSelector(
        [-12.0, -2.0, -1.0, 0.0, 1.0, 2.0, 12.0]
    )

    # Controlamos el ajuste para evaluar una Normal estándar.
    # Sus funciones de probabilidad conservan su implementación real.
    with patch.object(norm, "fit", return_value=(0.0, 1.0)):
        selector.fit_distribution("Normal", norm)

    assert selector.fit_errors == {}
    assert selector.results["Normal"]["a2"] == pytest.approx(
        22.321511572974426,
        rel=1e-12,
    )

def test_ad_conserva_resultado_con_probabilidades_ordinarias():
    selector = HidroModelSelector(
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    )

    with patch.object(norm, "fit", return_value=(0.0, 1.0)):
        selector.fit_distribution("Normal", norm)

    assert selector.fit_errors == {}
    assert selector.results["Normal"]["a2"] == pytest.approx(
        0.6753511234537477,
        rel=1e-12,
    )


def test_ad_personalizado_conserva_cola_extrema():
    datos = [0.0, 0.5, 1.0, 4.0, 8.0, 10000.0]
    selector = HidroModelSelector(datos)

    # Comprobamos que el caso incluye una CDF redondeada a uno.
    assert sqrt_etmax.cdf(
        datos[-1], 2.0, scale=1.0 / 0.7
    ) == 1.0

    with patch.object(
        sqrt_etmax,
        "fit_custom",
        return_value=(2.0, 0.7),
    ):
        selector.fit_distribution(
            "SQRT-ETmax",
            sqrt_etmax,
            is_custom=True,
        )

    assert selector.fit_errors == {}
    assert selector.results["SQRT-ETmax"]["a2"] == pytest.approx(
        13.32199020609099,
        rel=1e-12,
    )

@pytest.mark.parametrize(
    "datos",
    [
        pytest.param(
            [0.0, 0.25, 0.5, 0.75],
            id="cdf_realmente_cero",
        ),
        pytest.param(
            [0.25, 0.5, 0.75, 1.0],
            id="sf_realmente_cero",
        ),
    ],
)
def test_ad_no_oculta_probabilidades_realmente_nulas(datos):
    selector = HidroModelSelector(datos)

    # Uniforme sobre [0, 1]: probabilidades exactas en los extremos.
    estadistico = selector._calculate_anderson_stat(
        uniform.logcdf(datos),
        uniform.logsf(datos),
    )
    assert estadistico == float("inf")

    with patch.object(uniform, "fit", return_value=(0.0, 1.0)):
        selector.fit_distribution("Uniforme", uniform)

    assert "Uniforme" not in selector.results

    error = selector.fit_errors["Uniforme"]
    assert error["error_type"] == "ValueError"
    assert "estadísticos" in error["message"]
