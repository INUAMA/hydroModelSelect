from unittest.mock import patch
import numpy as np
import pytest
from scipy.stats import (
    genextreme, gumbel_r, lognorm, norm, pearson3, expon,
)

from hidroModelSelect import HidroModelSelector
from sqrt_etmax import sqrt_etmax

def test_renombrar_lognormal_conserva_ajuste():
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)

    selector.fit_distribution("Lognormal", lognorm)
    selector.fit_distribution("Lognormal_prueba", lognorm)

    assert selector.fit_errors == {}
    assert set(selector.results) == {
        "Lognormal",
        "Lognormal_prueba",
    }

    referencia = selector.results["Lognormal"]
    renombrado = selector.results["Lognormal_prueba"]

    assert referencia["k_params"] == 2
    assert renombrado["k_params"] == referencia["k_params"]
    assert renombrado["params"][1] == 0.0
    assert renombrado["params"] == pytest.approx(
        referencia["params"]
    )

    for campo in (
        "aic", "aicc", "bic", "a2", "ad_c", "ks", "ks_pv",
    ):
        assert renombrado[campo] == pytest.approx(
            referencia[campo]
        ), campo

@pytest.mark.parametrize(
    "nombre",
    ["SQRT-ETmax", "SQRT_ETmax", "SQRT_prueba"],
)
def test_sqrt_generica_fija_loc_por_identidad(nombre):
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)

    # Simulamos el ajuste para comprobar las opciones que recibe.
    with patch.object(
        sqrt_etmax,
        "fit",
        return_value=(2.0, 0.0, 1.0),
    ) as ajuste:
        selector.fit_distribution(
            nombre, sqrt_etmax, is_custom=False
        )

    assert ajuste.call_args.kwargs.get("floc") == 0
    assert selector.fit_errors == {}
    assert selector.results[nombre]["k_params"] == 2


@pytest.mark.parametrize(
    "nombre",
    ["SQRT-ETmax", "SQRT_ETmax"],
)
def test_etiqueta_sqrt_no_fija_loc_de_normal(nombre):
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)

    with patch.object(norm, "fit", wraps=norm.fit) as ajuste:
        selector.fit_distribution(nombre, norm)

    assert "floc" not in ajuste.call_args.kwargs
    assert selector.fit_errors == {}
    assert selector.results[nombre]["k_params"] == 2

@pytest.mark.parametrize(
    "nombre, distribucion",
    [
        ("Lognormal", lognorm),
        ("Lognormal_prueba", lognorm),
        ("SQRT-ETmax", sqrt_etmax),
        ("SQRT_prueba", sqrt_etmax),
    ],
    ids=[
        "lognormal",
        "lognormal_renombrada",
        "sqrt",
        "sqrt_renombrada",
    ],
)
def test_floc_explicito_prevalece(nombre, distribucion):
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)
    parametros = (2.0, 0.25, 1.0)

    with patch.object(
        distribucion,
        "fit",
        return_value=parametros,
    ) as ajuste:
        selector.fit_distribution(
            nombre, distribucion, floc=0.25
        )

    assert ajuste.call_args.kwargs["floc"] == 0.25
    assert selector.fit_errors == {}
    assert selector.results[nombre]["params"] == parametros
    assert selector.results[nombre]["k_params"] == 2

@pytest.mark.parametrize(
    "nombre, distribucion, parametros, tipo_laio",
    [
        ("Gumbel", gumbel_r, (0.0, 2.0), "EV1"),
        ("Normal", norm, (2.0, 3.0), "NORM"),
        ("Log_Normal", lognorm, (0.8, 0.0, 2.0), "NORM"),
        ("GEV", genextreme, (0.1, 0.0, 2.0), "GEV"),
        ("Pearson3", pearson3, (1.0, 2.0, 3.0), "GAM"),
    ],
    ids=["gumbel", "normal", "lognormal", "gev", "pearson3"],
)
def test_laio_no_depende_de_etiqueta(
    nombre, distribucion, parametros, tipo_laio,
):
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)
    llamadas = []

    for etiqueta in (nombre, f"{nombre}_prueba"):
        with patch.object(
            distribucion, "fit", return_value=parametros
        ), patch.object(
            selector, "_calc_adc", return_value=0.25
        ) as correccion:
            selector.fit_distribution(etiqueta, distribucion)

        assert selector.fit_errors == {}
        assert etiqueta in selector.results
        correccion.assert_called_once()

        assert correccion.call_args.args[1] == tipo_laio
        llamadas.append(correccion.call_args)

        assert selector.results[etiqueta]["adc"] == pytest.approx(
            0.25
        )

    assert llamadas[0] == llamadas[1]

@pytest.mark.parametrize(
    "nombre",
    ["Gumbel", "Normal", "Log_Normal", "GEV", "Pearson3"],
)
def test_etiqueta_no_activa_laio_en_familia_no_soportada(nombre):
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)

    with patch.object(
        expon, "fit", return_value=(0.0, 2.0)
    ), patch.object(
        selector, "_calc_adc", return_value=0.25
    ) as correccion:
        selector.fit_distribution(nombre, expon)

    assert selector.fit_errors == {}
    assert nombre in selector.results
    correccion.assert_not_called()
    assert np.isnan(selector.results[nombre]["adc"])
