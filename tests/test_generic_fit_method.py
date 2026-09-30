from scipy.stats import norm

from hidroModelSelect import HidroModelSelector

from unittest.mock import patch
import pytest
import numpy as np
from copy import deepcopy

def test_ajuste_generico_registra_mle_por_defecto():
    selector = HidroModelSelector(
        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    )

    with patch.object(norm, "fit", wraps=norm.fit) as ajuste:
        selector.fit_distribution("Normal", norm)

    assert selector.fit_errors == {}
    ajuste.assert_called_once()
    assert ajuste.call_args.kwargs["method"] == "mle"
    assert selector.results["Normal"]["fit_method"] == "mle"

@pytest.mark.parametrize(
    "method, esperado",
    [
        ("MLE", "mle"),
        ("mle", "mle"),
        ("MM", "mm"),
        ("mm", "mm"),
    ],
)
def test_metodo_generico_se_envia_y_registra(method, esperado):
    selector = HidroModelSelector(
        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    )

    with patch.object(norm, "fit", wraps=norm.fit) as ajuste:
        selector.fit_distribution(
            "Normal",
            norm,
            method=method,
            floc=0,
        )

    assert selector.fit_errors == {}
    ajuste.assert_called_once()

    opciones = ajuste.call_args.kwargs
    assert opciones["method"].lower() == esperado
    assert opciones["floc"] == 0

    resultado = selector.results["Normal"]
    assert resultado["fit_method"] == esperado
    assert resultado["params"][-2] == 0

    ranking = selector.get_ranking_dataframe()
    assert ranking.loc["Normal", "fit_method"] == esperado

@pytest.mark.parametrize(
    "method",
    [
        pytest.param("no_existe", id="desconocido"),
        pytest.param("", id="vacio"),
        pytest.param(None, id="none"),
        pytest.param(42, id="numero"),
        pytest.param(["mle"], id="lista"),
        pytest.param(np.array(["mle"]), id="array"),
    ],
)
def test_metodo_generico_invalido_se_rechaza_antes_del_ajuste(
    method,
):
    selector = HidroModelSelector(
        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    )

    with patch.object(norm, "fit", wraps=norm.fit) as ajuste:
        with pytest.raises(ValueError, match="method"):
            selector.fit_distribution(
                "Normal",
                norm,
                method=method,
            )

        ajuste.assert_not_called()

    assert selector.results == {}
    assert selector.fit_errors == {}

def test_metodo_invalido_conserva_ajuste_anterior():
    selector = HidroModelSelector(
        [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    )
    selector.fit_distribution("Normal", norm)

    assert selector.fit_errors == {}
    assert "Normal" in selector.results
    resultados_antes = deepcopy(selector.results)

    with patch.object(norm, "fit", wraps=norm.fit) as ajuste:
        with pytest.raises(ValueError, match="method"):
            selector.fit_distribution(
                "Normal",
                norm,
                method="no_existe",
            )

        ajuste.assert_not_called()

    assert selector.results == resultados_antes
    assert selector.fit_errors == {}
