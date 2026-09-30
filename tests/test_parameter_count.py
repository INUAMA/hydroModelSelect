import pytest
from scipy.stats import norm

from hidroModelSelect import HidroModelSelector
from scipy.stats import lognorm, norm
import numpy as np

@pytest.mark.parametrize(
    "opciones",
    [
        pytest.param({"floc": None}, id="floc_none"),
        pytest.param({"fscale": None}, id="fscale_none"),
        pytest.param(
            {"floc": None, "fscale": None},
            id="ambos_none",
        ),
    ],
)
def test_restricciones_none_conservan_ajuste_y_criterios(opciones):
    selector = HidroModelSelector(
        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    )

    selector.fit_distribution("Sin_restricciones", norm)
    selector.fit_distribution("Con_none", norm, **opciones)

    assert selector.fit_errors == {}

    referencia = selector.results["Sin_restricciones"]
    con_none = selector.results["Con_none"]

    assert con_none["params"] == pytest.approx(
        referencia["params"]
    )
    assert referencia["k_params"] == 2
    assert con_none["k_params"] == 2

    for criterio in ("aic", "aicc", "bic"):
        assert con_none[criterio] == pytest.approx(
            referencia[criterio]
        )

@pytest.mark.parametrize(
    "opciones",
    [
        pytest.param({"floc": 0}, id="localizacion_cero"),
        pytest.param(
            {"floc": 0, "fscale": None},
            id="localizacion_cero_escala_libre",
        ),
        pytest.param({"fscale": 10.0}, id="escala_fija"),
        pytest.param(
            {"floc": None, "fscale": 10.0},
            id="localizacion_libre_escala_fija",
        ),
    ],
)
def test_restriccion_efectiva_descuenta_un_parametro(opciones):
    datos = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    parametros_referencia = norm.fit(datos, **opciones)

    selector = HidroModelSelector(datos)
    selector.fit_distribution("Normal", norm, **opciones)

    assert selector.fit_errors == {}

    resultado = selector.results["Normal"]
    assert resultado["k_params"] == 1
    assert resultado["params"] == pytest.approx(
        parametros_referencia
    )

@pytest.mark.parametrize("alias", ["f0", "fs", "fix_s"])
def test_forma_fija_respeta_alias_y_escala_libre(alias):
    datos = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    parametros_referencia = lognorm.fit(
        datos,
        f0=0.8,
        floc=0,
    )

    selector = HidroModelSelector(datos)
    opciones = {
        alias: 0.8,
        "fscale": None,
    }
    selector.fit_distribution(
        "Lognormal",
        lognorm,
        **opciones,
    )

    assert selector.fit_errors == {}

    resultado = selector.results["Lognormal"]
    assert resultado["k_params"] == 1
    assert resultado["params"][0] == pytest.approx(0.8)
    assert resultado["params"][1] == 0
    assert resultado["params"] == pytest.approx(
        parametros_referencia
    )

def test_floc_none_conserva_aicc_no_disponible():
    selector = HidroModelSelector([1.0, 2.0, 4.0])
    selector.fit_distribution("Normal", norm, floc=None)

    assert selector.fit_errors == {}

    resultado = selector.results["Normal"]
    assert resultado["k_params"] == 2
    assert np.isposinf(resultado["aicc"])
    assert np.isnan(resultado["d_aicc"])

    with pytest.raises(RuntimeError, match="criterio"):
        selector.get_best_dist(criterion="aicc")

    assert "Normal" in selector.results
    assert selector.fit_errors == {}
