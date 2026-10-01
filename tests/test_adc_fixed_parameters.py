from unittest.mock import patch
import pytest
import numpy as np

from scipy.stats import norm, gumbel_r, genextreme, pearson3, lognorm
from hidroModelSelect import HidroModelSelector


def test_normal_con_loc_fija_conserva_ajuste_sin_adc():
    datos = norm.ppf(np.linspace(0.05, 0.95, 50))
    selector = HidroModelSelector(datos)

    with patch.object(
        selector, "_calc_adc", wraps=selector._calc_adc,
    ) as calcular_adc:
        selector.fit_distribution("Normal", norm, floc=0.0)

    assert selector.fit_errors == {}
    assert "Normal" in selector.results
    calcular_adc.assert_not_called()

    resultado = selector.results["Normal"]
    assert resultado["params"][-2] == 0.0
    assert resultado["k_params"] == 1
    assert np.isnan(resultado["adc"])
    assert "fijados" in resultado["adc_reason"]

    for campo in ("aic", "aicc", "bic", "a2", "ad_c"):
        assert np.isfinite(resultado[campo])

def test_normal_con_escala_fija_conserva_ajuste_sin_adc():
    datos = norm.ppf(np.linspace(0.05, 0.95, 50))
    selector = HidroModelSelector(datos)

    with patch.object(
        selector, "_calc_adc", wraps=selector._calc_adc,
    ) as calcular_adc:
        selector.fit_distribution("Normal", norm, fscale=1.0)

    assert selector.fit_errors == {}
    calcular_adc.assert_not_called()

    resultado = selector.results["Normal"]
    assert resultado["params"][-1] == 1.0
    assert resultado["k_params"] == 1
    assert np.isnan(resultado["adc"])
    assert "fijados" in resultado["adc_reason"]
    assert np.isfinite(resultado["aicc"])


@pytest.mark.parametrize(
    "opciones",
    [
        pytest.param({}, id="sin_restricciones"),
        pytest.param({"floc": None}, id="loc_libre"),
        pytest.param({"fscale": None}, id="escala_libre"),
    ],
)
def test_normal_con_parametros_libres_conserva_adc(opciones):
    datos = norm.ppf(np.linspace(0.05, 0.95, 50))
    selector = HidroModelSelector(datos)

    with patch.object(
        selector, "_calc_adc", wraps=selector._calc_adc,
    ) as calcular_adc:
        selector.fit_distribution("Normal", norm, **opciones)

    assert selector.fit_errors == {}
    calcular_adc.assert_called_once()

    resultado = selector.results["Normal"]
    assert resultado["k_params"] == 2
    assert np.isfinite(resultado["adc"])
    assert resultado["adc_reason"] is None

@pytest.mark.parametrize(
    "nombre, distribucion, parametros, opciones",
    [
        pytest.param(
            "Gumbel", gumbel_r, (0.0, 1.0),
            {"floc": 0.0}, id="gumbel_loc_fija",
        ),
        pytest.param(
            "GEV", genextreme, (0.2, 0.0, 1.0),
            {"f0": 0.2}, id="gev_forma_fija",
        ),
        pytest.param(
            "Pearson3", pearson3, (1.0, 0.0, 1.0),
            {"f0": 1.0}, id="pearson_forma_fija",
        ),
    ],
)
def test_restricciones_adicionales_conservan_ajuste_sin_adc(
    nombre, distribucion, parametros, opciones,
):
    datos = distribucion.ppf(
        np.linspace(0.05, 0.95, 50), *parametros,
    )
    selector = HidroModelSelector(datos)

    with patch.object(
        distribucion, "fit", return_value=parametros,
    ):
        with patch.object(
            selector, "_calc_adc", wraps=selector._calc_adc,
        ) as calcular_adc:
            selector.fit_distribution(
                nombre, distribucion, **opciones,
            )

    assert selector.fit_errors == {}
    assert nombre in selector.results
    calcular_adc.assert_not_called()

    resultado = selector.results[nombre]
    assert resultado["params"] == parametros
    assert resultado["k_params"] == len(parametros) - 1
    assert np.isnan(resultado["adc"])
    assert "fijados" in resultado["adc_reason"]
    assert np.isfinite(resultado["aicc"])

@pytest.mark.parametrize(
    "opciones",
    [
        pytest.param({"f0": 0.5}, id="forma_fija"),
        pytest.param({"fscale": 2.0}, id="escala_fija"),
    ],
)
def test_lognormal_con_restriccion_adicional_no_calcula_adc(
    opciones,
):
    parametros = (0.5, 0.0, 2.0)
    datos = lognorm.ppf(
        np.linspace(0.05, 0.95, 50), *parametros,
    )
    selector = HidroModelSelector(datos)

    with patch.object(
        lognorm, "fit", return_value=parametros,
    ) as ajustar:
        with patch.object(
            selector, "_calc_adc", wraps=selector._calc_adc,
        ) as calcular_adc:
            selector.fit_distribution(
                "Lognormal", lognorm, **opciones,
            )

    assert selector.fit_errors == {}
    assert ajustar.call_args.kwargs["floc"] == 0
    calcular_adc.assert_not_called()

    resultado = selector.results["Lognormal"]
    assert resultado["params"] == parametros
    assert resultado["k_params"] == 1
    assert np.isnan(resultado["adc"])
    assert "fijados" in resultado["adc_reason"]
    assert np.isfinite(resultado["aicc"])

@pytest.mark.parametrize(
    "opciones",
    [
        pytest.param({}, id="loc_cero_predeterminada"),
        pytest.param({"floc": 0.0}, id="loc_cero_explicita"),
        pytest.param(
            {"floc": 0.0, "f0": None, "fscale": None},
            id="forma_y_escala_libres",
        ),
    ],
)
def test_lognormal_dos_parametros_conserva_adc(opciones):
    parametros = (0.5, 0.0, 2.0)
    datos = lognorm.ppf(
        np.linspace(0.05, 0.95, 50), *parametros,
    )
    selector = HidroModelSelector(datos)

    with patch.object(
        lognorm, "fit", return_value=parametros,
    ):
        with patch.object(
            selector, "_calc_adc", wraps=selector._calc_adc,
        ) as calcular_adc:
            selector.fit_distribution(
                "Lognormal", lognorm, **opciones,
            )

    assert selector.fit_errors == {}
    calcular_adc.assert_called_once()
    assert calcular_adc.call_args.args[1] == "NORM"

    resultado = selector.results["Lognormal"]
    assert resultado["k_params"] == 2
    assert np.isfinite(resultado["adc"])
    assert resultado["adc_reason"] is None

def test_lognormal_con_loc_libre_conserva_ajuste_sin_adc():
    parametros = (0.5, 0.0, 2.0)
    datos = lognorm.ppf(
        np.linspace(0.05, 0.95, 50), *parametros,
    )
    selector = HidroModelSelector(datos)

    with patch.object(
        lognorm, "fit", return_value=parametros,
    ) as ajustar:
        with patch.object(
            selector, "_calc_adc", wraps=selector._calc_adc,
        ) as calcular_adc:
            selector.fit_distribution(
                "Lognormal", lognorm, floc=None,
            )

    assert selector.fit_errors == {}
    assert ajustar.call_args.kwargs["floc"] is None
    calcular_adc.assert_not_called()

    resultado = selector.results["Lognormal"]
    assert resultado["params"] == parametros
    assert resultado["k_params"] == 3
    assert np.isnan(resultado["adc"])
    assert "localización" in resultado["adc_reason"]
    assert np.isfinite(resultado["aicc"])
