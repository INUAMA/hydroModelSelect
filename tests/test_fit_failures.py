import pytest
from unittest.mock import patch

import numpy as np
from scipy.stats import norm, gumbel_r
from sqrt_etmax import sqrt_etmax
from hidroModelSelect import HidroModelSelector


def test_reajuste_fallido_elimina_resultado_anterior():
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)

    selector.fit_distribution("Normal", norm)
    assert "Normal" in selector.results

    with patch.object(
        norm,
        "fit",
        side_effect=RuntimeError("Fallo simulado del ajuste"),
    ):
        selector.fit_distribution("Normal", norm)

    assert "Normal" not in selector.results

@pytest.mark.parametrize(
    "ajuste_previo",
    [False, True],
    ids=["primer_intento", "reajuste"],
)
def test_ajuste_fallido_registra_causa(ajuste_previo):
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)

    if ajuste_previo:
        selector.fit_distribution("Normal", norm)

    with patch.object(
        norm,
        "fit",
        side_effect=RuntimeError("Fallo simulado del ajuste"),
    ):
        selector.fit_distribution("Normal", norm)

    assert "Normal" not in selector.results
    assert selector.fit_errors["Normal"] == {
        "error_type": "RuntimeError",
        "message": "Fallo simulado del ajuste",
    }

def test_reintento_exitoso_reemplaza_resultado_y_limpia_error():
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)

    selector.fit_distribution("Normal", norm)
    anterior = selector.results["Normal"]

    with patch.object(
        norm,
        "fit",
        side_effect=RuntimeError("Fallo simulado del ajuste"),
    ):
        selector.fit_distribution("Normal", norm)

    assert "Normal" not in selector.results
    assert "Normal" in selector.fit_errors

    # Fuera del bloque with, norm.fit vuelve a funcionar normalmente.
    selector.fit_distribution("Normal", norm)

    assert "Normal" in selector.results
    assert selector.results["Normal"] is not anterior
    assert "Normal" not in selector.fit_errors

def test_fallo_actualiza_deltas_y_conserva_otro_candidato():
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)
    distribuciones = {"Normal": norm, "Gumbel": gumbel_r}

    for nombre, distribucion in distribuciones.items():
        selector.fit_distribution(nombre, distribucion)

    retirado, restante = sorted(
        selector.results,
        key=lambda nombre: selector.results[nombre]["aicc"],
    )

    esperado = selector.results[restante].copy()
    assert esperado["d_aicc"] > 0
    esperado["d_aicc"] = 0.0

    with patch.object(
        distribuciones[retirado],
        "fit",
        side_effect=RuntimeError("Fallo simulado del ajuste"),
    ):
        selector.fit_distribution(
            retirado, distribuciones[retirado]
        )

    assert set(selector.results) == {restante}
    assert selector.results[restante]["d_aicc"] == pytest.approx(0.0)
    assert selector.results[restante] == esperado

@pytest.mark.parametrize(
    "fallo_previo",
    [False, True],
    ids=["sin_intentos", "todos_fallidos"],
)
def test_sin_ajustes_devuelve_ranking_vacio_y_rechaza_seleccion(
    fallo_previo,
):
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)

    if fallo_previo:
        with patch.object(
            norm,
            "fit",
            side_effect=RuntimeError("Fallo simulado del ajuste"),
        ):
            selector.fit_distribution("Normal", norm)

    assert selector.get_ranking_dataframe().empty

    with pytest.raises(RuntimeError, match="No hay ajustes"):
        selector.get_best_dist()

@pytest.mark.parametrize(
    "metodo, estimador, clave",
    [
        ("mle", "fit_custom", "SQRT-ETmax"),
        ("lmoments", "fit_lmoments", "SQRT-ETmax_Lmom"),
    ],
    ids=["mle", "lmoments"],
)
def test_ajuste_personalizado_gestiona_fallo_y_reintento(
    metodo, estimador, clave,
):
    datos = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
    selector = HidroModelSelector(datos)
    opciones = {"is_custom": True, "custom_type": metodo}

    selector.fit_distribution("SQRT-ETmax", sqrt_etmax, **opciones)
    assert set(selector.results) == {clave}
    anterior = selector.results[clave]

    with patch.object(
        sqrt_etmax,
        estimador,
        side_effect=RuntimeError("Fallo simulado del ajuste"),
    ):
        selector.fit_distribution("SQRT-ETmax", sqrt_etmax, **opciones)

    assert selector.results == {}
    assert selector.fit_errors == {
        clave: {
            "error_type": "RuntimeError",
            "message": "Fallo simulado del ajuste",
        },
    }

    selector.fit_distribution("SQRT-ETmax", sqrt_etmax, **opciones)

    assert set(selector.results) == {clave}
    assert selector.results[clave] is not anterior
    assert selector.fit_errors == {}
