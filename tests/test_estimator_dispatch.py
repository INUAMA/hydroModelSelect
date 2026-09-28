from unittest.mock import patch

import numpy as np
import pytest

from hidroModelSelect import HidroModelSelector
from sqrt_etmax import sqrt_etmax


@pytest.mark.parametrize(
    "opciones, llamadas_mle, llamadas_lmom, metodo_esperado",
    [
        pytest.param({}, 1, 0, "mle", id="predeterminado"),
        pytest.param(
            {"custom_type": "mle"}, 1, 0, "mle", id="mle"
        ),
        pytest.param(
            {"custom_type": "mel"}, 1, 0, "mle", id="alias_mel"
        ),
        pytest.param(
            {"custom_type": "lmoments"}, 0, 1, "lmoments",
            id="lmoments",
        ),
    ],
)
def test_metodo_invoca_y_registra_estimador(
    opciones, llamadas_mle, llamadas_lmom, metodo_esperado
):
    datos = np.array([.5, 1., 2., 4., 8., 12.])
    selector = HidroModelSelector(datos)

    with patch.object(
        sqrt_etmax, "fit_custom", wraps=sqrt_etmax.fit_custom
    ) as mle, patch.object(
        sqrt_etmax, "fit_lmoments", wraps=sqrt_etmax.fit_lmoments
    ) as lmom:
        selector.fit_distribution(
            "SQRT-ETmax",
            sqrt_etmax,
            is_custom=True,
            **opciones,
        )

    assert mle.call_count == llamadas_mle
    assert lmom.call_count == llamadas_lmom

    assert len(selector.results) == 1
    resultado = next(iter(selector.results.values()))
    assert resultado["fit_method"] == metodo_esperado

@pytest.mark.parametrize(
    "metodo",
    [
        pytest.param("no_existe", id="desconocido"),
        pytest.param("", id="vacio"),
        pytest.param(None, id="none"),
        pytest.param(1, id="numero"),
        pytest.param(["mle"], id="lista"),
        pytest.param(np.array(["mle"]), id="array"),
    ],
)
def test_metodo_invalido_se_rechaza_antes_del_ajuste(metodo):
    datos = np.array([.5, 1., 2., 4., 8., 12.])
    selector = HidroModelSelector(datos)

    with patch.object(
        sqrt_etmax, "fit_custom", return_value=(2.0, 0.7)
    ) as mle, patch.object(
        sqrt_etmax, "fit_lmoments", return_value=(2.0, 0.7)
    ) as lmom:
        with pytest.raises(ValueError, match="custom_type"):
            selector.fit_distribution(
                "SQRT-ETmax",
                sqrt_etmax,
                is_custom=True,
                custom_type=metodo,
            )

        mle.assert_not_called()
        lmom.assert_not_called()

    assert selector.results == {}