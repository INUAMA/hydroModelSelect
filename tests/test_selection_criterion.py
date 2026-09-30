import numpy as np
import pytest
from scipy.stats import norm
from copy import deepcopy

from hidroModelSelect import HidroModelSelector


@pytest.fixture
def selector_criterios():
    selector = HidroModelSelector(np.arange(1.0, 91.0))

    casos = [
        ("Dos_parametros", 2, -50.0, 0.4),
        ("Tres_parametros", 3, -50.5, 0.3),
    ]

    for nombre, k, log_lik, ad_c in casos:
        aic, aicc, bic = selector._calculate_aic_bic(log_lik, k)
        selector.results[nombre] = {
            "params": tuple([1.0] * k),
            "k_params": k,
            "aic": aic,
            "aicc": aicc,
            "bic": bic,
            "ad_c": ad_c,
            "ks_pv": 0.2,
        }

    return selector


def test_seleccion_predeterminada_usa_aicc_para_todos(
    selector_criterios,
):
    elegido = selector_criterios.get_best_dist()

    assert elegido.index.tolist() == ["Dos_parametros"]
    assert elegido.iloc[0]["metri"] == pytest.approx(
        selector_criterios.results["Dos_parametros"]["aicc"]
    )

@pytest.mark.parametrize("criterion", ["aic", "aicc", "bic"])
def test_criterio_explicito_utiliza_columna_solicitada(
    selector_criterios, criterion,
):
    elegido = selector_criterios.get_best_dist(
        criterion=criterion,
    )

    assert elegido.index.tolist() == ["Dos_parametros"]
    assert elegido.iloc[0]["metri"] == pytest.approx(
        selector_criterios.results["Dos_parametros"][criterion]
    )

@pytest.mark.parametrize(
    "criterion",
    [
        pytest.param("no_existe", id="desconocido"),
        pytest.param("", id="vacio"),
        pytest.param(None, id="none"),
        pytest.param(42, id="numero"),
        pytest.param(["aicc"], id="lista"),
        pytest.param(np.array(["aicc"]), id="array"),
    ],
)
def test_criterio_invalido_produce_valueerror(
    selector_criterios, criterion,
):
    with pytest.raises(ValueError, match="criterion"):
        selector_criterios.get_best_dist(criterion=criterion)

def test_aicc_no_disponible_impide_seleccion_sin_borrar_ajuste():
    selector = HidroModelSelector([1.0, 2.0, 4.0])
    selector.fit_distribution("Normal", norm)

    assert selector.fit_errors == {}
    assert np.isposinf(selector.results["Normal"]["aicc"])

    with pytest.raises(RuntimeError, match="criterio"):
        selector.get_best_dist(criterion="aicc")

    assert "Normal" in selector.results
    assert selector.fit_errors == {}
    assert selector.results["Normal"]["criterion"] == "aicc"
    assert selector.results["Normal"]["transp"] == (
        "Criterio no disponible"
    )

def test_aicc_excluye_y_registra_candidato_no_disponible():
    selector = HidroModelSelector([1.0, 2.0, 4.0])
    selector.fit_distribution("Normal", norm)
    selector.fit_distribution("Normal_loc_fijo", norm, floc=0)

    assert selector.fit_errors == {}
    assert np.isposinf(selector.results["Normal"]["aicc"])
    assert np.isfinite(
        selector.results["Normal_loc_fijo"]["aicc"]
    )

    elegido = selector.get_best_dist(criterion="aicc")

    assert elegido.index.tolist() == ["Normal_loc_fijo"]
    assert set(selector.results) == {
        "Normal",
        "Normal_loc_fijo",
    }
    assert selector.results["Normal"]["transp"] == (
        "Criterio no disponible"
    )
    assert selector.fit_errors == {}

@pytest.mark.parametrize("criterion", ["aic", "aicc", "bic"])
def test_criterio_explicito_utiliza_columna_solicitada(
    selector_criterios, criterion,
):
    elegido = selector_criterios.get_best_dist(
        criterion=criterion,
    )

    assert elegido.index.tolist() == ["Dos_parametros"]
    assert elegido.iloc[0]["metri"] == pytest.approx(
        selector_criterios.results["Dos_parametros"][criterion]
    )
    assert elegido.iloc[0]["criterion"] == criterion

    for resultado in selector_criterios.results.values():
        assert resultado["criterion"] == criterion

def test_seleccion_conserva_valores_de_los_ajustes(
    selector_criterios,
):
    originales = deepcopy(selector_criterios.results)

    selector_criterios.get_best_dist(criterion="aicc")

    assert set(selector_criterios.results) == set(originales)

    for nombre, ajuste_original in originales.items():
        for campo, valor_original in ajuste_original.items():
            assert (
                selector_criterios.results[nombre][campo]
                == valor_original
            ), (nombre, campo)

def test_cambiar_criterio_actualiza_elegibilidad_y_trazabilidad():
    selector = HidroModelSelector([1.0, 2.0, 4.0])
    selector.fit_distribution("Normal", norm)
    selector.fit_distribution("Normal_loc_fijo", norm, floc=0)

    assert selector.fit_errors == {}

    for criterion in ("aicc", "bic", "aicc"):
        elegido = selector.get_best_dist(criterion=criterion)
        nombre_elegido = elegido.index[0]

        assert elegido.iloc[0]["criterion"] == criterion
        assert elegido.iloc[0]["metri"] == pytest.approx(
            selector.results[nombre_elegido][criterion]
        )

        for resultado in selector.results.values():
            assert resultado["criterion"] == criterion

        if criterion == "aicc":
            assert nombre_elegido == "Normal_loc_fijo"
            assert selector.results["Normal"]["transp"] == (
                "Criterio no disponible"
            )
        else:
            assert selector.results["Normal"]["transp"] != (
                "Criterio no disponible"
            )
