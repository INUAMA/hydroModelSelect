import numpy as np
import pytest

from hidroModelSelect import HidroModelSelector


def test_alternativa_por_ks_registra_estado_fallback():
    selector = HidroModelSelector(np.arange(1.0, 53.0))
    selector.results = {
        "A": {
            "aic": 100.0, "aicc": 100.3, "bic": 104.0,
            "ks_pv": 0.01, "ad_c": 2.0,
        },
        "B": {
            "aic": 102.0, "aicc": 102.3, "bic": 106.0,
            "ks_pv": 0.02, "ad_c": 3.0,
        },
    }

    elegido = selector.get_best_dist()

    assert elegido.index.tolist() == ["B"]
    assert elegido.iloc[0]["transp"] == "pv_max"
    assert elegido.iloc[0]["selection_status"] == "fallback"
    assert selector.results["B"]["selection_status"] == "fallback"
    assert len(selector.results) == 2

@pytest.mark.parametrize(
    "ad_a, etiqueta, estado",
    [
        pytest.param(
            2.0, "ad_cMax", "fallback",
            id="ninguno_cumple_ad",
        ),
        pytest.param(
            0.3, "ad_cH0", "passes_current_checks",
            id="uno_cumple_ambos",
        ),
    ],
)
def test_estado_distingue_cumplimiento_de_ad(
    ad_a, etiqueta, estado,
):
    selector = HidroModelSelector(np.arange(1.0, 53.0))
    selector.results = {
        "A": {
            "aic": 100.0, "aicc": 100.3, "bic": 104.0,
            "ks_pv": 0.20, "ad_c": ad_a,
        },
        "B": {
            "aic": 102.0, "aicc": 102.3, "bic": 106.0,
            "ks_pv": 0.30, "ad_c": 3.0,
        },
    }

    elegido = selector.get_best_dist()

    assert elegido.index.tolist() == ["A"]
    assert elegido.iloc[0]["transp"] == etiqueta
    assert elegido.iloc[0]["selection_status"] == estado
    assert selector.results["A"]["transp"] == etiqueta
    assert selector.results["A"]["selection_status"] == estado
    assert selector.results["B"]["selection_status"] is None
    assert len(selector.results) == 2

@pytest.mark.parametrize(
    "pvalores",
    [
        pytest.param((0.01, 0.02), id="ninguno_supera_ks"),
        pytest.param((0.20, 0.30), id="ninguno_supera_ad"),
    ],
)
def test_modo_estricto_rechaza_alternativas_y_conserva_ajustes(
    pvalores,
):
    selector = HidroModelSelector(np.arange(1.0, 53.0))
    selector.results = {
        "A": {
            "aic": 100.0, "aicc": 100.3, "bic": 104.0,
            "ks_pv": pvalores[0], "ad_c": 2.0,
        },
        "B": {
            "aic": 102.0, "aicc": 102.3, "bic": 106.0,
            "ks_pv": pvalores[1], "ad_c": 3.0,
        },
    }
    estadisticas_antes = {
        nombre: resultado.copy()
        for nombre, resultado in selector.results.items()
    }

    with pytest.raises(RuntimeError, match="ningún candidato"):
        selector.get_best_dist(require_pass=True)

    assert set(selector.results) == {"A", "B"}
    assert selector.fit_errors == {}

    for nombre, estadisticas in estadisticas_antes.items():
        for campo, valor in estadisticas.items():
            assert selector.results[nombre][campo] == valor

def test_modo_estricto_conserva_ganador_que_cumple_controles():
    selector = HidroModelSelector(np.arange(1.0, 53.0))
    selector.results = {
        "A": {
            "aic": 100.0, "aicc": 100.3, "bic": 104.0,
            "ks_pv": 0.20, "ad_c": 0.3,
        },
        "B": {
            "aic": 101.0, "aicc": 101.3, "bic": 105.0,
            "ks_pv": 0.30, "ad_c": 0.4,
        },
    }

    habitual = selector.get_best_dist()
    estricto = selector.get_best_dist(require_pass=True)

    assert habitual.index.tolist() == ["A"]
    assert estricto.index.tolist() == ["A"]
    assert estricto.iloc[0]["transp"] == "optima_ci"
    assert (
        estricto.iloc[0]["selection_status"]
        == "passes_current_checks"
    )
    assert (
        selector.results["A"]["selection_status"]
        == "passes_current_checks"
    )
    assert selector.results["B"]["selection_status"] is None
    assert selector.fit_errors == {}

def test_cambiar_criterio_limpia_estado_del_ganador_excluido():
    selector = HidroModelSelector(np.arange(1.0, 53.0))
    selector.results = {
        "A": {
            "aic": 100.0, "aicc": np.inf, "bic": 104.0,
            "ks_pv": 0.20, "ad_c": 0.3,
        },
        "B": {
            "aic": 110.0, "aicc": 110.3, "bic": 114.0,
            "ks_pv": 0.30, "ad_c": 0.4,
        },
    }

    primero = selector.get_best_dist(criterion="aic")
    assert primero.index.tolist() == ["A"]
    assert (
        selector.results["A"]["selection_status"]
        == "passes_current_checks"
    )

    segundo = selector.get_best_dist(criterion="aicc")

    assert segundo.index.tolist() == ["B"]
    assert selector.results["A"]["selection_status"] is None
    assert (
        selector.results["A"]["transp"]
        == "Criterio no disponible"
    )
    assert (
        selector.results["B"]["selection_status"]
        == "passes_current_checks"
    )
    assert np.isinf(selector.results["A"]["aicc"])
    assert len(selector.results) == 2

def test_modo_estricto_limpia_seleccion_alternativa_anterior():
    selector = HidroModelSelector(np.arange(1.0, 53.0))
    selector.results = {
        "A": {
            "aic": 100.0, "aicc": 100.3, "bic": 104.0,
            "ks_pv": 0.01, "ad_c": 2.0,
        },
        "B": {
            "aic": 102.0, "aicc": 102.3, "bic": 106.0,
            "ks_pv": 0.02, "ad_c": 3.0,
        },
    }

    elegido = selector.get_best_dist()
    assert elegido.index.tolist() == ["B"]
    assert selector.results["B"]["selection_status"] == "fallback"

    with pytest.raises(RuntimeError, match="ningún candidato"):
        selector.get_best_dist(require_pass=True)

    assert set(selector.results) == {"A", "B"}
    assert all(
        resultado["selection_status"] is None
        for resultado in selector.results.values()
    )
    assert selector.fit_errors == {}

    recuperado = selector.get_best_dist()
    assert recuperado.index.tolist() == ["B"]
    assert recuperado.iloc[0]["selection_status"] == "fallback"

@pytest.mark.parametrize(
    "valor",
    [
        pytest.param("False", id="texto"),
        pytest.param(None, id="none"),
        pytest.param(1, id="entero"),
        pytest.param([False], id="lista"),
    ],
)
def test_require_pass_invalido_conserva_estado(valor):
    selector = HidroModelSelector(np.arange(1.0, 53.0))
    selector.results = {
        "A": {
            "aic": 100.0, "aicc": 100.3, "bic": 104.0,
            "ks_pv": 0.20, "ad_c": 0.3,
        },
    }
    selector.get_best_dist()
    estado_anterior = {
        nombre: resultado.copy()
        for nombre, resultado in selector.results.items()
    }

    with pytest.raises(ValueError, match="require_pass"):
        selector.get_best_dist(require_pass=valor)

    assert selector.results == estado_anterior
    assert selector.fit_errors == {}
