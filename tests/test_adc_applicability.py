import numpy as np
import pytest
from unittest.mock import patch
from scipy.stats import genextreme, norm, pearson3
from hidroModelSelect import HidroModelSelector


def test_gamma_conserva_forma_original_en_correccion_muestral():
    selector = HidroModelSelector(np.arange(1.0, 51.0))

    resultado = selector._get_laio_coeffs("GAM", 1.0)

    # Laio (2004):
    # tabla 3: coeficientes asintóticos evaluados en forma=2;
    # tabla 5: corrección muestral con la forma original=1.
    esperado = (
        0.1593003925091835,
        0.2292543,
        1.189713746627988,
    )

    assert resultado == pytest.approx(esperado, rel=1e-12)

@pytest.mark.parametrize(
    "familia, forma",
    [
        pytest.param("GEV", np.nan, id="gev_nan"),
        pytest.param("GEV", np.inf, id="gev_inf"),
        pytest.param("GEV", -np.inf, id="gev_menos_inf"),
        pytest.param("GAM", np.nan, id="gam_nan"),
        pytest.param("GAM", np.inf, id="gam_inf"),
        pytest.param("GAM", -np.inf, id="gam_menos_inf"),
        pytest.param("GAM", 0.0, id="gam_cero"),
        pytest.param("GAM", -1.0, id="gam_negativa"),
    ],
)
def test_coeficientes_rechazan_forma_invalida(familia, forma):
    selector = HidroModelSelector(np.arange(1.0, 51.0))

    with pytest.raises(ValueError, match="forma"):
        selector._get_laio_coeffs(familia, forma)

def test_gev_rechaza_forma_inferior_al_dominio_tabulado():
    selector = HidroModelSelector(np.arange(1.0, 51.0))

    with pytest.raises(ValueError, match="dominio"):
        selector._get_laio_coeffs("GEV", -1.2)

def test_adc_fuera_de_dominio_conserva_ajuste_gev():
    forma = -1.2
    parametros = (forma, 0.0, 1.0)
    datos = genextreme.ppf(
        np.linspace(0.05, 0.95, 50),
        *parametros,
    )
    selector = HidroModelSelector(datos)

    with patch.object(
        genextreme, "fit", return_value=parametros,
    ):
        selector.fit_distribution("GEV", genextreme)

    assert selector.fit_errors == {}
    assert "GEV" in selector.results

    resultado = selector.results["GEV"]
    assert resultado["params"] == parametros
    assert np.isnan(resultado["adc"])
    assert "dominio" in resultado["adc_reason"]

    for campo in ("aic", "aicc", "bic", "a2", "ad_c"):
        assert np.isfinite(resultado[campo])

    ranking = selector.get_ranking_dataframe()
    assert "GEV" in ranking.index

def test_ajuste_mm_se_conserva_sin_aplicar_adc():
    datos = norm.ppf(np.linspace(0.05, 0.95, 50))
    selector = HidroModelSelector(datos)

    with patch.object(norm, "fit", return_value=(0.0, 1.0)):
        with patch.object(
            selector, "_calc_adc", wraps=selector._calc_adc,
        ) as calcular_adc:
            selector.fit_distribution(
                "Normal", norm, method="MM",
            )

    assert selector.fit_errors == {}
    calcular_adc.assert_not_called()

    resultado = selector.results["Normal"]
    assert resultado["fit_method"] == "mm"
    assert np.isnan(resultado["adc"])
    assert "estimador" in resultado["adc_reason"]

    for campo in ("aic", "aicc", "bic", "a2", "ad_c"):
        assert np.isfinite(resultado[campo])

@pytest.mark.parametrize(
    "nombre, distribucion, parametros",
    [
        pytest.param(
            "GEV", genextreme, (0.8, 0.0, 1.0),
            id="gev_no_regular",
        ),
        pytest.param(
            "Pearson3", pearson3, (2.0, 0.0, 1.0),
            id="gamma_forma_uno",
        ),
    ],
)
def test_mle_no_regular_conserva_ajuste_sin_adc(
    nombre, distribucion, parametros,
):
    # Pearson III: forma GAM = (2 / skew)**2 = 1.
    datos = distribucion.ppf(
        np.linspace(0.05, 0.95, 50),
        *parametros,
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
    calcular_adc.assert_not_called()

    resultado = selector.results[nombre]
    assert resultado["params"] == parametros
    assert resultado["fit_method"] == "mle"
    assert np.isnan(resultado["adc"])
    assert "estimador" in resultado["adc_reason"]
    assert np.isfinite(resultado["a2"])
    assert np.isfinite(resultado["aicc"])
