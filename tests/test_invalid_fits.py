import pytest
from unittest.mock import patch

import numpy as np
from scipy.stats import norm

from hidroModelSelect import HidroModelSelector
from sqrt_etmax import sqrt_etmax

@pytest.mark.parametrize(
    "parametros, mensaje",
    [
        ((np.nan, 1.0), "finitos"),
        ((np.inf, 1.0), "finitos"),
        ((-np.inf, 1.0), "finitos"),
        ((0.0, np.nan), "finitos"),
        ((0.0, np.inf), "finitos"),
        ((0.0, -np.inf), "finitos"),
        ((0.0, 0.0), "escala"),
        ((0.0, -1.0), "escala"),
    ],
    ids=[
        "loc_nan",
        "loc_inf",
        "loc_menos_inf",
        "scale_nan",
        "scale_inf",
        "scale_menos_inf",
        "scale_cero",
        "scale_negativa",
    ],
)
def test_ajuste_rechaza_parametros_invalidos(parametros, mensaje):
    datos = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0])
    selector = HidroModelSelector(datos)

    with patch.object(norm, "fit", return_value=parametros):
        selector.fit_distribution("Normal", norm)

    assert "Normal" not in selector.results
    assert selector.get_ranking_dataframe().empty

    error = selector.fit_errors["Normal"]
    assert error["error_type"] == "ValueError"
    assert mensaje in error["message"]

@pytest.mark.parametrize(
    "nombre, distribucion, personalizada, metodo",
    [
        ("Normal", norm, False, "logpdf"),
        ("SQRT-ETmax", sqrt_etmax, True, "log_likelihood"),
    ],
    ids=["generica", "personalizada"],
)
@pytest.mark.parametrize(
    "valor",
    [np.nan, np.inf, -np.inf],
    ids=["nan", "inf", "menos_inf"],
)
def test_ajuste_rechaza_verosimilitud_no_finita(
    nombre, distribucion, personalizada, metodo, valor,
):
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)

    # logpdf devuelve un valor por observación;
    # log_likelihood devuelve el total de la muestra.
    resultado_simulado = (
        valor if personalizada else np.full(datos.shape, valor)
    )

    with patch.object(
        distribucion,
        metodo,
        return_value=resultado_simulado,
    ):
        selector.fit_distribution(
            nombre,
            distribucion,
            is_custom=personalizada,
        )

    assert nombre not in selector.results
    assert selector.get_ranking_dataframe().empty

    error = selector.fit_errors[nombre]
    assert error["error_type"] == "ValueError"
    assert "verosimilitud" in error["message"]

@pytest.mark.parametrize(
    "nombre, distribucion, personalizada",
    [
        ("Normal", norm, False),
        ("SQRT-ETmax", sqrt_etmax, True),
    ],
    ids=["generica", "personalizada"],
)
@pytest.mark.parametrize(
    "valor",
    [np.nan, np.inf, -np.inf, -0.1, 1.1],
    ids=["nan", "inf", "menos_inf", "menor_cero", "mayor_uno"],
)
def test_ajuste_rechaza_cdf_invalida(
    nombre, distribucion, personalizada, valor,
):
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)

    cdf_simulada = np.linspace(0.1, 0.9, datos.size)
    cdf_simulada[2] = valor

    with patch.object(
        distribucion,
        "cdf",
        return_value=cdf_simulada,
    ):
        selector.fit_distribution(
            nombre,
            distribucion,
            is_custom=personalizada,
        )

    assert nombre not in selector.results
    assert selector.get_ranking_dataframe().empty

    error = selector.fit_errors[nombre]
    assert error["error_type"] == "ValueError"
    assert "CDF" in error["message"]

@pytest.mark.parametrize("criterio", ["aic", "aicc", "bic"])
@pytest.mark.parametrize(
    "valor",
    [np.nan, np.inf],
    ids=["nan", "inf"],
)
def test_ajuste_rechaza_criterio_no_finito(criterio, valor):
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)

    criterios = {
        "aic": 30.0,
        "aicc": 34.0,
        "bic": 29.0,
    }
    criterios[criterio] = valor

    with patch.object(
        selector,
        "_calculate_aic_bic",
        return_value=(
            criterios["aic"],
            criterios["aicc"],
            criterios["bic"],
        ),
    ):
        selector.fit_distribution("Normal", norm)

    assert "Normal" not in selector.results
    assert selector.get_ranking_dataframe().empty

    error = selector.fit_errors["Normal"]
    assert error["error_type"] == "ValueError"
    assert "criterios de información" in error["message"]

@pytest.mark.parametrize(
    "con_otro_candidato",
    [False, True],
    ids=["solo_aicc_no_definido", "con_aicc_finito"],
)
def test_aicc_no_definido_conserva_ajuste_y_delta_no_disponible(
    con_otro_candidato,
):
    datos = np.array([1.0, 2.0, 4.0])
    selector = HidroModelSelector(datos)

    if con_otro_candidato:
        selector.fit_distribution(
            "Normal_loc_fijo", norm, floc=0
        )

    selector.fit_distribution("Normal", norm)

    assert "Normal" in selector.results
    assert selector.fit_errors == {}

    resultado = selector.results["Normal"]
    assert resultado["k_params"] == 2
    assert np.isfinite(resultado["aic"])
    assert np.isfinite(resultado["bic"])
    assert np.isposinf(resultado["aicc"])
    assert np.isnan(resultado["d_aicc"])

    if con_otro_candidato:
        otro = selector.results["Normal_loc_fijo"]
        assert otro["k_params"] == 1
        assert np.isfinite(otro["aicc"])
        assert otro["d_aicc"] == pytest.approx(0.0)

@pytest.mark.parametrize(
    "metodo, resultado_simulado",
    [
        ("_calculate_anderson_stat", np.nan),
        ("_ad_corr", np.inf),
        ("_calc_adc", np.nan),
        ("kstest", (np.nan, 0.5)),
        ("kstest", (0.2, np.inf)),
    ],
    ids=[
        "a2_nan",
        "ad_corregido_inf",
        "adc_nan",
        "ks_nan",
        "pvalor_ks_inf",
    ],
)
def test_ajuste_rechaza_estadisticos_no_finitos(
    metodo, resultado_simulado,
):
    # Tamaño suficiente para alcanzar el cálculo ADC que verifica la prueba.
    datos = np.linspace(0.5, 12.0, 50)
    selector = HidroModelSelector(datos)

    if metodo == "kstest":
        parche = patch(
            "hidroModelSelect.distCompare.kstest",
            return_value=resultado_simulado,
        )
    else:
        parche = patch.object(
            selector,
            metodo,
            return_value=resultado_simulado,
        )

    with parche:
        selector.fit_distribution("Normal", norm)

    assert "Normal" not in selector.results
    assert selector.get_ranking_dataframe().empty

    error = selector.fit_errors["Normal"]
    assert error["error_type"] == "ValueError"
    assert "estadísticos" in error["message"]

@pytest.mark.parametrize(
    "metodo, estimador, clave",
    [
        ("mle", "fit_custom", "SQRT-ETmax"),
        ("lmoments", "fit_lmoments", "SQRT-ETmax_Lmom"),
    ],
    ids=["mle", "lmoments"],
)
def test_parametros_personalizados_invalidos_se_rechazan_antes_de_cdf(
    metodo, estimador, clave,
):
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)

    with patch.object(
        sqrt_etmax, estimador, return_value=(np.nan, 0.7)
    ), patch.object(
        sqrt_etmax, "cdf", wraps=sqrt_etmax.cdf
    ) as cdf:
        selector.fit_distribution(
            "SQRT-ETmax",
            sqrt_etmax,
            is_custom=True,
            custom_type=metodo,
        )

    cdf.assert_not_called()
    assert clave not in selector.results

    error = selector.fit_errors[clave]
    assert error["error_type"] == "ValueError"
    assert "parámetros" in error["message"]
