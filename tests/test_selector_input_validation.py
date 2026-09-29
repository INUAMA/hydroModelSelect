import pytest
import numpy as np

from hidroModelSelect import HidroModelSelector


@pytest.mark.parametrize(
    "datos, mensaje",
    [
        pytest.param(
            [],
            "al menos una observación",
            id="vacia",
        ),
        pytest.param(
            2.0,
            "unidimensional",
            id="escalar",
        ),
        pytest.param(
            [[1.0, 2.0], [3.0, 4.0]],
            "unidimensional",
            id="bidimensional",
        ),
    ],
)
def test_selector_rechaza_dimension_o_tamano_invalidos(
    datos, mensaje,
):
    with pytest.raises(ValueError, match=mensaje):
        HidroModelSelector(datos)

@pytest.mark.parametrize(
    "valor",
    [
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="infinito_positivo"),
        pytest.param(-np.inf, id="infinito_negativo"),
    ],
)
def test_selector_rechaza_observaciones_no_finitas(valor):
    with pytest.raises(ValueError, match="finitos"):
        HidroModelSelector([1.0, valor, 4.0])

@pytest.mark.parametrize(
    "mascara",
    [
        pytest.param(
            [False, True, False],
            id="mascara_parcial",
        ),
        pytest.param(
            [True, True, True],
            id="mascara_total",
        ),
    ],
)
def test_selector_rechaza_observaciones_enmascaradas(mascara):
    datos = np.ma.array(
        [1.0, 2.0, 4.0],
        mask=mascara,
    )

    with pytest.raises(ValueError, match="enmascaradas"):
        HidroModelSelector(datos)

@pytest.mark.parametrize(
    "datos",
    [
        pytest.param(
            [1.0, 2.0 + 1.0j, 4.0],
            id="lista_compleja",
        ),
        pytest.param(
            np.array([1.0, 2.0 + 1.0j, 4.0]),
            id="array_complejo",
        ),
        pytest.param(
            np.array([1.0 + 0.0j, 2.0 + 0.0j, 4.0 + 0.0j]),
            id="complejos_sin_parte_imaginaria",
        ),
        pytest.param(
            np.array(
                [1.0, np.complex128(2.0 + 1.0j), 4.0],
                dtype=object,
            ),
            id="complejo_dentro_de_object",
        ),
    ],
)
def test_selector_rechaza_observaciones_complejas(datos):
    with pytest.raises(ValueError, match="complejos"):
        HidroModelSelector(datos)

@pytest.mark.parametrize(
    "datos",
    [
        pytest.param(
            [1.0, "no_es_un_numero", 4.0],
            id="texto",
        ),
        pytest.param(
            [1.0, object(), 4.0],
            id="objeto",
        ),
        pytest.param(
            [1, 10**1000, 4],
            id="entero_no_representable",
        ),
        pytest.param(
            [[1.0, 2.0], [3.0]],
            id="estructura_irregular",
        ),
    ],
)
def test_selector_rechaza_entradas_no_convertibles(datos):
    with pytest.raises(ValueError, match="numéricos reales"):
        HidroModelSelector(datos)

@pytest.mark.parametrize(
    "datos",
    [
        pytest.param(
            [4.0, -1.0, 0.0, 2.0],
            id="lista",
        ),
        pytest.param(
            (4.0, -1.0, 0.0, 2.0),
            id="tupla",
        ),
        pytest.param(
            np.array([4, -1, 0, 2], dtype=np.int64),
            id="array_enteros",
        ),
        pytest.param(
            np.array([4.0, -1.0, 0.0, 2.0]),
            id="array_flotantes",
        ),
        pytest.param(
            np.ma.array(
                [4.0, -1.0, 0.0, 2.0],
                mask=False,
            ),
            id="mascara_sin_observaciones_ocultas",
        ),
    ],
)
def test_selector_normaliza_entradas_validas(datos):
    selector = HidroModelSelector(datos)

    assert selector.n == 4
    assert selector.obs.dtype.kind == "f"

    np.testing.assert_array_equal(
        selector.obs,
        [4.0, -1.0, 0.0, 2.0],
    )
    np.testing.assert_array_equal(
        selector.obs_sort,
        [-1.0, 0.0, 2.0, 4.0],
    )

def test_cambiar_entrada_no_modifica_selector():
    datos = np.array([4.0, 1.0, 2.0])
    selector = HidroModelSelector(datos)

    # Crear el selector no debe modificar la entrada.
    np.testing.assert_array_equal(datos, [4.0, 1.0, 2.0])

    datos[:] = 99.0

    np.testing.assert_array_equal(
        selector.obs, [4.0, 1.0, 2.0]
    )
    np.testing.assert_array_equal(
        selector.obs_sort, [1.0, 2.0, 4.0]
    )


def test_cambiar_obs_no_modifica_entrada_ni_copia_ordenada():
    datos = np.array([4.0, 1.0, 2.0])
    selector = HidroModelSelector(datos)

    selector.obs[0] = 99.0

    np.testing.assert_array_equal(
        datos, [4.0, 1.0, 2.0]
    )
    np.testing.assert_array_equal(
        selector.obs_sort, [1.0, 2.0, 4.0]
    )
